#ifndef FILE_AMG_PC_VERTEX_IMPL_HPP
#define FILE_AMG_PC_VERTEX_IMPL_HPP

namespace amg
{
  /** Options **/

  template<class FACTORY>
  class VertexAMGPC<FACTORY> :: Options : public FACTORY::Options,
					  public VertexAMGPCOptions
  {
  public:
    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix) override
    {
      FACTORY::Options::SetFromFlags(flags, prefix);
      VertexAMGPCOptions::SetFromFlags(fes, flags, prefix);
    }
  }; // VertexAMGPC::Options


  template<class FACTORY, class HTVD, class HTED>
  class ElmatVAMG<FACTORY, HTVD, HTED> :: Options : public VertexAMGPC<FACTORY>::Options
  {
  };

  /** END Options **/


  /** VertexAMGPC **/

  template<class FACTORY>
  VertexAMGPC<FACTORY> :: VertexAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : BaseAMGPC(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY>
  VertexAMGPC<FACTORY> :: VertexAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : BaseAMGPC(blf, flags, name, opts)
  {
    ;
  } // VertexAMGPC(..)


  template<class FACTORY>
  VertexAMGPC<FACTORY> :: ~VertexAMGPC ()
  {
    ;
  } // ~VertexAMGPC


  template<class FACTORY>
  void VertexAMGPC<FACTORY> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    BaseAMGPC::InitLevel(freedofs);

    auto & O(static_cast<Options&>(*options));

    if (freedofs == nullptr) // postpone to FinalizeLevel
      { return; }

    if (O.spec_ss == VertexAMGPCOptions::SPECIAL_SUBSET::SPECSS_FREE) {
      cout << IM(3) << "taking subset for coarsening from freedofs" << endl;
      O.ss_select = finest_freedofs;
      //cout << " freedofs (for coarsening) set " << options->ss_select->NumSet() << " of " << options->ss_select->Size() << endl;
      //cout << *options->ss_select << endl;
    }

  } // VertexAMGPC<FACTORY>::InitLevel


  template<class FACTORY>
  shared_ptr<BaseAMGPC::Options> VertexAMGPC<FACTORY> :: NewOpts ()
  {
    return make_shared<Options>();
  }


  template<class FACTORY>
  void VertexAMGPC<FACTORY> :: SetOptionsFromFlags (BaseAMGPC::Options& _O, const Flags & flags, string prefix)
  {
    Options* myO = dynamic_cast<Options*>(&_O);
    if (myO == nullptr)
      { throw Exception("Invalid Opts!"); }
    Options & O(*myO);

    O.SetFromFlags(bfa->GetFESpace(), flags, prefix);
  } // VertexAMGPC<FACTORY> :: SetOptionsFromFlags


  template<class FACTORY>
  shared_ptr<EQCHierarchy> VertexAMGPC<FACTORY> :: BuildEQCH ()
  {
    auto & O = static_cast<Options&>(*options);

    /** Build inital EQC-Hierarchy **/
    auto fpd = finest_mat->GetParallelDofs();
    size_t maxset = 0;
    switch (O.subset) {
    case(Options::DOF_SUBSET::RANGE_SUBSET): { maxset = O.ss_ranges.Last()[1]; break; }
    case(Options::DOF_SUBSET::SELECTED_SUBSET): {
      auto sz = O.ss_select->Size();
      for (auto k : Range(sz))
	if (O.ss_select->Test(--sz))
	  { maxset = sz+1; break; }
      break;
    } }
    maxset = min2(maxset, fpd->GetNDofLocal());

    shared_ptr<EQCHierarchy> eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

    return eqc_h;
  } // VertexAMGPC::BuildEQCH


  template<class FACTORY>
  shared_ptr<TopologicMesh> VertexAMGPC<FACTORY> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh"); RegionTimer rt(t);

    auto eqc_h = BuildEQCH();

    SetUpMaps();

    return BuildAlgMesh(BuildTopMesh(eqc_h));
  } // VertexAMGPC::BuildInitialMesh


  template<class FACTORY>
  void VertexAMGPC<FACTORY> :: SetUpMaps ()
  {
    static Timer t("SetUpMaps"); RegionTimer rt(t);

    auto & O = static_cast<Options&>(*options);

    const size_t ndof = bfa->GetFESpace()->GetNDof();
    size_t n_verts = -1;

    switch(O.subset) {
    case(Options::DOF_SUBSET::RANGE_SUBSET): {

      size_t in_ss = 0;
      for (auto range : O.ss_ranges)
	{ in_ss += range[1] - range[0]; }

      if (O.dof_ordering == Options::DOF_ORDERING::VARIABLE_ORDERING)
	{ throw Exception("not implemented (but easy)"); }
      else if (O.dof_ordering == Options::DOF_ORDERING::REGULAR_ORDERING) {
	const size_t dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	n_verts = in_ss / dpv;

	d2v_array.SetSize(ndof); d2v_array = -1;

	auto n_block_types = O.block_s.Size();

	// cout << in_ss << " " << dpv << " " << ndof << " " << n_verts << endl;
	
	if (dpv == 1) { // range subset , regular order, 1 dof per V
	  v2d_array.SetSize(n_verts);
	  int c = 0;
	  for (auto range : O.ss_ranges) {
	    for (auto dof : Range(range[0], range[1])) {
	      d2v_array[dof] = c;
	      v2d_array[c++] = dof;
	    }
	  }
	}
	else if (n_block_types == 1) { // range subset, regular order, N dofs per V in a single block
	  use_v2d_tab = true;
	  v2d_table = Table<int>(n_verts, dpv);
	  auto v2da = v2d_table.AsArray();
	  int c = 0;
	  for (auto range : O.ss_ranges) {
	    for (auto dof : Range(range[0], range[1])) {
	      d2v_array[dof] = c / dpv;
	      v2da[c++] = dof;
	    }
	  }
	}
	else { // range subset , regular order, N dofs per V in multiple blocks
	  use_v2d_tab = true;
	  v2d_table = Table<int>(n_verts, dpv);
	  const int num_block_types = O.block_s.Size();
	  int block_type = 0; // we currently mapping DOFs in O.block_s[block_type]-blocks
	  int cnt_block = 0; // how many of those blocks have we gone through
	  int block_s = O.block_s[block_type];
	  int bos = 0;
	  for (auto range_num : Range(O.ss_ranges)) {
	    INT<2,size_t> range = O.ss_ranges[range_num];
	    while ( (range[1] > range[0]) && (block_type < num_block_types) ) {
	      int blocks_in_range = (range[1] - range[0]) / block_s; // how many blocks can I fit in here ?
	      int need_blocks = n_verts - cnt_block; // how many blocks of current size I still need.
	      auto map_blocks = min2(blocks_in_range, need_blocks);
	      for (auto l : Range(map_blocks)) {
		for (auto j : Range(block_s)) {
		  d2v_array[range[0]] = cnt_block;
		  v2d_table[cnt_block][bos+j] = range[0]++;
		}
		cnt_block++;
	      }
	      if (cnt_block == n_verts) {
		bos += block_s;
		block_type++;
		cnt_block = 0;
		if (block_type < O.block_s.Size())
		  { block_s = O.block_s[block_type]; }
	      }
	    }
	  }
	} // range, regular, N dofs, multiple blocks
      } // REGULAR_ORDERING
      break;
    } // RANGE_SUBSET
    case(Options::DOF_SUBSET::SELECTED_SUBSET): {

      // cout << " ss_sel " << O.ss_select << endl;

      const auto & subset = *O.ss_select;
      size_t in_ss = subset.NumSet();

      if (O.dof_ordering == Options::DOF_ORDERING::VARIABLE_ORDERING)
	{ throw Exception("not implemented (but easy)"); }
      else if (O.dof_ordering == Options::DOF_ORDERING::REGULAR_ORDERING) {
	const size_t dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	n_verts = in_ss / dpv;

	d2v_array.SetSize(ndof); d2v_array = -1;

	auto n_block_types = O.block_s.Size();

	if (dpv == 1) { // select subset, regular order, 1 dof per V
	  v2d_array.SetSize(n_verts);
	  auto& subset = *O.ss_select;
	  for (int j = 0, k = 0; k < n_verts; j++) {
	    // cout << j << " " << k << " " << n_verts << " ss " << subset.Test(j) << endl;
	    if (subset.Test(j)) {
	      auto d = j; auto svnr = k++;
	      d2v_array[d] = svnr;
	      v2d_array[svnr] = d;
	    }
	  }
	}
	else { // select subset, regular order, N dofs per V
	  use_v2d_tab = true;
	  v2d_table = Table<int>(n_verts, dpv);
	  int block_type = 0; // we currently mapping DOFs in O.block_s[block_type]-blocks
	  int cnt_block = 0; // how many of those blocks have we gone through
	  int block_s = O.block_s[block_type];
	  int j = 0, col_os = 0;
	  const auto blockss = O.block_s.Size();
	  for (auto k : Range(subset.Size())) {
	    if (subset.Test(k)) {
	      d2v_array[k] = cnt_block;
	      v2d_table[cnt_block][col_os + j++] = k;
	      if (j == block_s) {
		j = 0;
		cnt_block++;
	      }
	      if (cnt_block == n_verts) {
		block_type++;
		cnt_block = 0;
		col_os += block_s;
		if (block_type + 1 < blockss)
		  { block_s = O.block_s[block_type]; }
	      }
	    }
	  }
 	} // select subset, reg. order, N dofs per V
      } // REGULAR_ORDERING
      break;
    } // SELECTED_SUBSET
    } // switch(O.subset)
    
    if (O.store_v_nodes) {
      auto fes = bfa->GetFESpace();
      size_t numset = 0;
      O.v_nodes.SetSize(n_verts);
      for (NODE_TYPE NT : { NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL } ) {
	if (numset < n_verts) {
	  Array<int> dnums;
	  for (auto k : Range(ma->GetNNodes(NT))) {
	    NodeId id(NT, k);
	    fes->GetDofNrs(id, dnums);
	    for (auto dof : dnums) {
	      auto top_vnum = d2v_array[dof];
	      if (top_vnum != -1) {
		O.v_nodes[top_vnum] = id;
		numset++;
		break;
	      }
	    }
	  }
	  // cout << " after NT " << NT << ", set " << numset << " of " << n_verts << endl;
	}
      }
    }

    // cout << " (unsorted) d2v_array: " << endl; prow2(d2v_array); cout << endl << endl;
    // if (use_v2d_tab) {
    //  cout << " (unsorted) v2d_table: " << endl << v2d_table << endl << endl;
    // }
    // else {
    //  cout << " (unsorted) v2d_array: " << endl; prow2(v2d_array); cout << endl << endl;
    // }

  } // VertexAMGPC::SetUpMaps


  template<class FACTORY>
  shared_ptr<BaseAMGFactory> VertexAMGPC<FACTORY> :: BuildFactory ()
  {
    auto opts = dynamic_pointer_cast<Options>(options);
    return make_shared<FACTORY>(opts);
  } // VertexAMGPC::BuildFactory


  template<class FACTORY>
  shared_ptr<BaseDOFMapStep> VertexAMGPC<FACTORY> :: BuildEmbedding (BaseAMGFactory::AMGLevel & level)
  {
    shared_ptr<TopologicMesh> mesh = level.cap->mesh;

    static Timer t("BuildEmbedding"); RegionTimer rt(t);
    auto & O(static_cast<Options&>(*options));

    shared_ptr<BaseMatrix> f_loc_mat;
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { f_loc_mat = parmat->GetMatrix(); }
    else
      { f_loc_mat = finest_mat; }

    shared_ptr<BaseDOFMapStep> E = nullptr;

    constexpr int max_switch = min(MAX_SYS_DIM, FACTORY::BS);
    Switch<max_switch> // 0-based
      (GetEntryDim(f_loc_mat.get())-1, [&, this] (auto BSM) {
	constexpr int BS = BSM + 1;
	if constexpr( (BS == 1) || (BS == 2) || (BS == 3) || (BS == 6) )
	  { E = this->BuildEmbedding_impl<BS>(mesh); }
	else
	  { return; }
      } );

    return E;
  } // VertexAMGPC::BuildEmbedding


  template<class FACTORY> template<int BSA>
  shared_ptr<BaseDOFMapStep> VertexAMGPC<FACTORY> :: BuildEmbedding_impl (shared_ptr<TopologicMesh> mesh)
  {
    /**
       Embedding  = E_S * E_D * P
       E_S      ... from 0..ndof to subset                              // entries: N x N
       E_D      ... disp to disp-rot emb or compound-to multidim, etc   // entries: N x dofpv
       P        ... permutation matrix from re-sorting vertex numbers   // entries: dofpv x dofpv
    **/

    constexpr int BS = FACTORY::BS;
    typedef stripped_spm_tm<Mat<BSA, BSA, double>> T_E_S;
    typedef stripped_spm_tm<Mat<BSA, BS, double>> T_E_D;
    typedef typename FACTORY::TSPM_TM T_P;

    shared_ptr<T_E_S> E_S;
    shared_ptr<T_E_D> E_D;
    shared_ptr<T_P> P;

    
    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();
    shared_ptr<BaseMatrix> f_loc_mat;
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { f_loc_mat = parmat->GetMatrix(); }
    else
      { f_loc_mat = finest_mat; }

    /** Subset **/
    E_S = BuildES<BSA>();

    /** DOFs **/
    size_t subset_count = (E_S == nullptr) ? fpds->GetNDofLocal() : E_S->Width();
    E_D = BuildED<BSA>(subset_count, mesh);

    /** Permutation **/
    if ( fpds->GetCommunicator().Size() > 2) {
      auto & vsort = node_sort[NT_VERTEX];
      P = BuildPermutationMatrix<typename T_P::TENTRY>(vsort);
    }

    /** Multiply mats **/
    auto a_is_b_times_c = [](auto & a, auto & b, auto & c) {
      if (b != nullptr) {
    	if (c != nullptr) { a = MatMultAB(*b, *c); }
    	else { a = b; } }
      else { a = c; }
    };
    shared_ptr<T_E_D> E, EDP;

    // auto prt = [&](auto x, auto name) {
    //   cout << name << " " << x << endl;
    //   if (x)
    // 	{ print_tm_spmat(cout, *x); cout << endl << endl; }
    // };
    // prt(E_S, "E_S");
    // prt(E_D, "E_D");
    // prt(P, "P");

    if constexpr(BSA == BS) {
    	if constexpr (BSA == 1) {
    	    a_is_b_times_c(E, E_S, P);
    	  }
    	else {
    	  if (E_D == nullptr) {
    	    a_is_b_times_c(E, E_S, P);
    	  }
    	  else {
    	    a_is_b_times_c(EDP, E_D, P);
    	    a_is_b_times_c(E, E_S, EDP);
    	  }
    	}
      }
    else {
      if (E_D == nullptr)
	{ throw Exception("E_D must not be nullptr here!!"); }
      if (P != nullptr)
	{ EDP = MatMultAB(*E_D, *P); }
      else
	{ EDP = E_D; }
      if (E_S != nullptr)
	{ E = MatMultAB(*E_S, *EDP); }
      else
	{ E = EDP; }
    }

    // prt(E, "E");

    /** DOF-Map  **/
    shared_ptr<BaseDOFMapStep> emb_step = nullptr;
    
    shared_ptr<ParallelDofs> mpds = nullptr;
    int have_embed = fpds->GetCommunicator().AllReduce((E == nullptr) ? 0 : 1, MPI_SUM);
    if (have_embed) {
      mpds = factory->BuildParallelDofs(mesh);
      if (E == nullptr) { // probably just for rank 0! (ParallelDofs - constructor has to be called by every member of the communicator!)
	Array<int> perow(fpds->GetNDofLocal()); perow = 1;
	E = make_shared<T_E_D>(perow, mpds->GetNDofLocal());
	for (auto k : Range(perow.Size())) {
	  E->GetRowIndices(k)[0] = k;
	  SetIdentity(E->GetRowValues(k)[0]);
	}
      }
      emb_step = make_shared<ProlMap<T_E_D>>(E, fpds, mpds);
    }

    return emb_step;
  } // VertexAMGPC::BuildEmbedding_impl


  template<class FACTORY> template<int BSA>
  shared_ptr<stripped_spm_tm<Mat<BSA, BSA, double>>> VertexAMGPC<FACTORY> :: BuildES ()
  {
    const auto & O(static_cast<Options&>(*options));

    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();

    typedef stripped_spm_tm<Mat<BSA, BSA, double>> TS;
    shared_ptr<TS> E_S = nullptr;
    if (O.subset == Options::DOF_SUBSET::RANGE_SUBSET) {
      INT<2, size_t> notin_ss = {0, fpds->GetNDofLocal() };
      for (auto pair : O.ss_ranges) {
	if (notin_ss[0] == pair[0])
	  { notin_ss[0] = pair[1]; }
	else if (notin_ss[1] == pair[1])
	  { notin_ss[1] = pair[0]; }
      }
      int is_triv = ( (notin_ss[1] - notin_ss[0]) == 0 ) ? 1 : 0;
      fpds->GetCommunicator().AllReduce(is_triv, MPI_SUM);
      if (is_triv == 0) {
	Array<int> perow(fpds->GetNDofLocal()); perow = 0;
	int cnt_cols = 0;
	for (auto pair : O.ss_ranges)
	  if (pair[1] > pair[0])
	    { perow.Range(pair[0], pair[1]) = 1; cnt_cols += pair[1] - pair[0]; }
	E_S = make_shared<TS>(perow, cnt_cols); cnt_cols = 0;
	for (auto pair : O.ss_ranges)
	  for (auto c : Range(pair[0], pair[1])) {
	    SetIdentity(E_S->GetRowValues(c)[0]);
	    E_S->GetRowIndices(c)[0] = cnt_cols++;
	  }
      }
    }
    else if (O.subset == Options::DOF_SUBSET::SELECTED_SUBSET) {
      if (O.ss_select == nullptr)
	{ throw Exception("SELECTED_SUBSET, but no ss_select!"); }
      const auto & SS(*O.ss_select);
      int is_triv = (SS.NumSet() == SS.Size()) ? 1 : 0;
      fpds->GetCommunicator().AllReduce(is_triv, MPI_SUM);
      if (is_triv == 0) {
	int cnt_cols = SS.NumSet();
	Array<int> perow(fpds->GetNDofLocal());
	for (auto k : Range(perow))
	  { perow[k] = SS.Test(k) ? 1 : 0; }
	E_S = make_shared<TS>(perow, cnt_cols); cnt_cols = 0;
	for (auto k : Range(fpds->GetNDofLocal())) {
	  if (SS.Test(k)) {
	    SetIdentity(E_S->GetRowValues(k)[0]);
	    E_S->GetRowIndices(k)[0] = cnt_cols++;
	  }
	}
      }
    }

    return E_S;
  } // VertexAMGPC::BuildES


  template<class FACTORY>
  shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    Options & O = static_cast<Options&>(*options);

    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(Options::TOPO::MESH_TOPO):  { top_mesh = BTM_Mesh(eqc_h); break; }
    case(Options::TOPO::ALG_TOPO):   { top_mesh = BTM_Alg(eqc_h); break; }
    case(Options::TOPO::ELMAT_TOPO): { throw Exception("cannot do topology from elmats!"); break; }
    default: { throw Exception("invalid topology type!"); break; }
    }

    return top_mesh;
  } // VertexAMGPC::BuildTopMesh


  template<class FACTORY>
  shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    throw Exception("I don't think this is functional...");

    // const auto &O(*options);
    // node_sort.SetSize(4);
    // shared_ptr<BlockTM> top_mesh;
    // switch (O.v_pos) {
    // case(BAO::VERTEX_POS): {
    //   top_mesh = MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
    // 				  false, node_sort[2], false, node_sort[3]);
    //   break;
    // }
    // case(BAO::GIVEN_POS): {
    //   throw Exception("Cannot combine custom vertices with topology from mesh");
    //   break;
    // }
    // default: { throw Exception("kinda unexpected case"); break; }
    // }
    // // TODO: re-map d2v/v2d
    // // this only works for the simplest case anyways...
    // auto fvs = make_shared<BitArray>(top_mesh->template GetNN<NT_VERTEX>()); fvs->Clear();
    // auto & vsort = node_sort[NT_VERTEX];
    // for (auto k : Range(top_mesh->template GetNN<NT_VERTEX>()))
    //   if (finest_freedofs->Test(k))
    // 	{ fvs->SetBit(vsort[k]); }
    // free_verts = fvs;
    // return top_mesh;

    return nullptr;
  } // VertexAMGPC::BTM_Mesh


  template<class FACTORY>
  shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BTM_Alg (shared_ptr<EQCHierarchy> eqc_h)
  {
    static Timer t("BTM_Alg"); RegionTimer rt(t);

    auto & O = static_cast<Options&>(*options);
    node_sort.SetSize(4);

    auto top_mesh = make_shared<BlockTM>(eqc_h);

    size_t n_verts = 0, n_edges = 0;

    auto & vert_sort = node_sort[NT_VERTEX];

    auto fpd = finest_mat->GetParallelDofs();

    /** Vertices **/
    auto set_vs = [&](auto nv, auto v2d) {
      n_verts = nv;
      vert_sort.SetSize(nv);
      top_mesh->SetVs (nv, [&](auto vnr) LAMBDA_INLINE { return fpd->GetDistantProcs(v2d(vnr)); },
		       [&vert_sort](auto i, auto j) LAMBDA_INLINE { vert_sort[i] = j; });
      free_verts = make_shared<BitArray>(nv);
      if (finest_freedofs != nullptr) {
	// cout << " finest_freedofs 1: " << finest_freedofs << endl;
	// if (finest_freedofs)
	  // { prow2(*finest_freedofs); cout << endl; }
	free_verts->Clear();
	for (auto k : Range(nv)) {
	  // cout << k << " dof " << v2d(k) << " sort " << vert_sort[k] << " free " << finest_freedofs->Test(v2d(k)) << endl;
	  if (finest_freedofs->Test(v2d(k)))
	    { free_verts->SetBit(vert_sort[k]); }
	}
      }
      else
	{ free_verts->Set(); }
      // cout << "diri verts: " << endl;
      // for (auto k : Range(free_verts->Size()))
      // 	if (!free_verts->Test(k)) { cout << k << " " << endl; }
      // cout << endl;
      // cout << "diri DOFs: " << endl;
      // for (auto k : Range(finest_freedofs->Size()))
      // 	if (!finest_freedofs->Test(k)) { cout << k << " " << endl; }
      // cout << endl;
      // cout << " VSORT: " << endl; prow2(vert_sort); cout << endl;
    };
    
    if (use_v2d_tab) {
      set_vs(v2d_table.Size(), [&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; });
    }
    else {
      set_vs(v2d_array.Size(), [&](auto i) LAMBDA_INLINE { return v2d_array[i]; });
    }

    /** Edges **/ 
    auto create_edges = [&](auto v2d, auto d2v) LAMBDA_INLINE {
      auto traverse_graph = [&](const auto& g, auto fun) LAMBDA_INLINE { // vertex->dof,  // dof-> vertex
	for (auto k : Range(n_verts)) {
	  int row = v2d(k); // for find_in_sorted_array
	  auto ri = g.GetRowIndices(row);
	  auto pos = find_in_sorted_array(row, ri); // no duplicates
	  if (pos+1 < ri.Size()) {
	    for (auto col : ri.Part(pos+1)) {
	      auto j = d2v(col);
	      // cout << " row col " << row << " " << col << ", vi vj " << k << " " << j << endl;
	      if (j != -1) {
		// cout << "dofs " << row << " " << col << " are (orig) vertices " << k << " " << j << ", sorted " << vert_sort[k] << " " << vert_sort[j] << endl;
		fun(vert_sort[k],vert_sort[j]);
	      }
	      // else {
		// cout << "dont use " << row << " " << j << " with dof " << col << endl;
	      // }
	    }
	  }
	}
      }; // traverse_graph
      auto bspm = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat);
      if (!bspm) { bspm = dynamic_pointer_cast<BaseSparseMatrix>( dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix()); }
      if (!bspm) { throw Exception("could not get BaseSparseMatrix out of finest_mat!!"); }
      n_edges = 0;
      traverse_graph(*bspm, [&](auto vk, auto vj) LAMBDA_INLINE { n_edges++; });
      Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);
      n_edges = 0;
      traverse_graph(*bspm, [&](auto vk, auto vj) LAMBDA_INLINE{
	    if (vk < vj) { epairs[n_edges++] = {vk, vj}; }
	    else { epairs[n_edges++] = {vj, vk}; }
	});
      // cout << " edge pair list: " << endl;
      // prow2(epairs); cout << endl;
      top_mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) LAMBDA_INLINE { return epairs[num]; }, // (already v-sorted)
				   [](auto node_num, auto id) LAMBDA_INLINE { /* dont care about edge-sort! */ });
      auto tme = top_mesh->GetNodes<NT_EDGE>();
      // cout << " tmesh edges: " << endl << tme << endl;
      // cout << "final n_edges: " << top_mesh->GetNN<NT_EDGE>() << endl;
    }; // create_edges

    // auto create_edges = [&](auto v2d, auto d2v) LAMBDA_INLINE {
    if (use_v2d_tab) {
      // create_edges([&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; },
      // 		   [&](auto i) LAMBDA_INLINE { return d2v_array[i]; } );
      create_edges([&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; },
		   [&](auto i) LAMBDA_INLINE { // I dont like it ...
		     auto v = d2v_array[i];
		     if ( (v != -1) && (v2d_table[v][0] == i) )
		       { return v; }
		     return -1;
		   });
    }
    else {
      create_edges([&](auto i) LAMBDA_INLINE { return v2d_array[i]; },
		   [&](auto i) LAMBDA_INLINE { return d2v_array[i]; } );
    }

    // update v2d/d2v with vert_sort
    if (use_v2d_tab) {
      Array<int> cnt(n_verts); cnt = 0;
      for (auto k : Range(n_verts)) {
	auto vk = vert_sort[k];
	for (auto d : v2d_table[k])
	  { d2v_array[d] = vk; }
      }
      for (auto k : Range(d2v_array)) {
	auto vnr = d2v_array[k];
	if  (vnr != -1)
	  { v2d_table[vnr][cnt[vnr]++] = k; }
      }
    }
    else {
      for (auto k : Range(n_verts)) {
	auto d = v2d_array[k];
	d2v_array[d] = vert_sort[k];
      }
      for (auto k : Range(d2v_array))
	if  (d2v_array[k] != -1)
	  { v2d_array[d2v_array[k]] = k; }
    }
    
    //cout << " (sorted) d2v_array: " << endl; prow2(d2v_array); cout << endl << endl;
    //if (use_v2d_tab) {
      //cout << " (sorted) v2d_table: " << endl << v2d_table << endl << endl;
    //}
    //else {
    // cout << " (sorted) v2d_array: " << endl; prow2(v2d_array); cout << endl << endl;
    //}
    
    return top_mesh;
  } // VertexAMGPC::BTM_Alg




  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> VertexAMGPC<FACTORY> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    Options & O = static_cast<Options&>(*options);

    shared_ptr<TMESH> alg_mesh;

    switch(O.energy) {
    case(Options::TRIV_ENERGY): { alg_mesh = BuildAlgMesh_TRIV(top_mesh); break; }
    case(Options::ALG_ENERGY): { alg_mesh = BuildAlgMesh_ALG(top_mesh); break; }
    case(Options::ELMAT_ENERGY): { throw Exception("Cannot do elmat energy!"); }
    default: { throw Exception("Invalid Energy!"); break; }
    }

    return alg_mesh;
  } // VertexAMGPC::BuildAlgMesh


  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> VertexAMGPC<FACTORY> :: BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh)
  {
    static Timer t("BuildAlgMesh_ALG"); RegionTimer rt(t);

    shared_ptr<typename FACTORY::TMESH> alg_mesh;

    shared_ptr<BaseMatrix> f_loc_mat;
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { f_loc_mat = parmat->GetMatrix(); }
    else
      { f_loc_mat = finest_mat; }
    auto spmat = dynamic_pointer_cast<BaseSparseMatrix>(f_loc_mat);

    if (use_v2d_tab) {
      alg_mesh = BuildAlgMesh_ALG_blk(top_mesh, spmat,
				      [&](auto d) LAMBDA_INLINE { return d2v_array[d]; },
				      [&](auto v) LAMBDA_INLINE { return v2d_table[v]; } );
    }
    else {
      alg_mesh = BuildAlgMesh_ALG_scal(top_mesh, spmat,
				       [&](auto d) LAMBDA_INLINE { return d2v_array[d]; },
				       [&](auto v) LAMBDA_INLINE { return v2d_array[v]; } );
    }

    return alg_mesh;
  } // EmbedVAMG::BuildAlgMesh_ALG


  template<class FACTORY>
  void VertexAMGPC<FACTORY> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    BaseAMGPC::InitFinestLevel(finest_level);
    /** Explicitely assemble matrix associated with the finest mesh. **/
    finest_level.cap->free_nodes = free_verts;
  }


  template<class FACTORY>
  Table<int> VertexAMGPC<FACTORY> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    if (amg_level.crs_map == nullptr) {
      throw Exception("Crs Map not saved!!");
      return move(Table<int>());
    }

    auto & O(static_cast<Options&>(*options));

    // if (amg_level.level == 0 && O.smooth_lo_only) {
    //   throw Exception("Asked for BGS blocks for HO part!!");
    //   return move(Table<int>());
    // }

    int NCV = amg_level.crs_map->GetMappedNN<NT_VERTEX>();
    int n_blocks = NCV;
    if (amg_level.disc_map != nullptr)
      { n_blocks += amg_level.disc_map->GetNDroppedNodes<NT_VERTEX>(); }
    TableCreator<int> cblocks(n_blocks);
    auto it_blocks = [&](auto NV, auto map_v) LAMBDA_INLINE {
      for (auto k : Range(NV)) {
	auto cv = map_v(k);
	if (cv != -1)
	  { cblocks.Add(cv, k); }
      }
    };
    auto it_blocks_2 = [&](auto NV, auto map_v) LAMBDA_INLINE {
      if (use_v2d_tab) {
	for (auto k : Range(NV)) {
	  auto cv = map_v(k);
	  if (cv != -1)
	    for (auto dof : v2d_table[k])
	      { cblocks.Add(cv, dof); }
	}
      }
      else {
	for (auto k : Range(NV)) {
	  auto cv = map_v(k);
	  if (cv != -1)
	    { cblocks.Add(cv, v2d_array[k]); }
	}
      }
    };
    auto vmap = amg_level.crs_map->GetMap<NT_VERTEX>();
    for (; !cblocks.Done(); cblocks++) {
      if (amg_level.disc_map == nullptr) {
	auto map_v = [&](auto v) -> int LAMBDA_INLINE { return vmap[v]; };
	if (amg_level.level == 0)
	  { it_blocks_2(vmap.Size(), map_v); }
	else
	  { it_blocks(vmap.Size(), map_v); }
	}
      else {
	const auto & drop = *amg_level.disc_map->GetDroppedNodes<NT_VERTEX>();
	auto drop_map = amg_level.disc_map->GetMap<NT_VERTEX>();
	auto map_v =  [&](auto v) -> int LAMBDA_INLINE {
	  auto midv = drop_map[v]; // have to consider drop OR DIRI !!
	  return (midv == -1) ? midv : vmap[midv];
	};
	if (amg_level.level == 0) {
	  it_blocks_2(drop.Size(), map_v);
	  int c = NCV;
	  if (use_v2d_tab) {
	    for (auto k : Range(drop.Size())) {
	      if (drop.Test(k)) {
		for (auto dof : v2d_table[k])
		  { cblocks.Add(c, dof); }
		c++;
	      }
	    }
	  }
	  else {
	    for (auto k : Range(drop.Size())) {
	      if (drop.Test(k))
		{ cblocks.Add(c++, v2d_array[k]); }
	    }
	  }
	}
	else {
	  it_blocks(drop.Size(), map_v);
	  int c = NCV;
	  for (auto k : Range(drop.Size())) {
	    if (drop.Test(k))
	      { cblocks.Add(c++, k); }
	  }
	}
      }
    }

    auto blocks = cblocks.MoveTable();

    return move(blocks);
  } // VertexAMGPC<FACTORY>::GetGSBlocks


  /** END VertexAMGPC **/


  /** ElmatVAMG **/


  template<class FACTORY, class HTVD, class HTED>
  ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (const PDE & apde, const Flags & aflags, const string aname)
    : VertexAMGPC<FACTORY>(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY, class HTVD, class HTED>
  ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : VertexAMGPC<FACTORY>(blf, flags, name, opts), ht_vertex(nullptr), ht_edge(nullptr)
  {
  } // ElmatVAMG(..)


  template<class FACTORY, class HTVD, class HTED>
  ElmatVAMG<FACTORY, HTVD, HTED> :: ~ElmatVAMG ()
  {
    if (ht_vertex != nullptr)
      { delete ht_vertex; }
    if (ht_edge != nullptr)
      { delete ht_edge; }
  } // ~ElmatVAMG


  template<class FACTORY, class HTVD, class HTED>
  shared_ptr<BlockTM> ElmatVAMG<FACTORY, HTVD, HTED> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    Options & O = static_cast<Options&>(*options);

    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(Options::TOPO::MESH_TOPO):  { top_mesh = BTM_Mesh(eqc_h); break; }
    case(Options::TOPO::ALG_TOPO):   { top_mesh = BTM_Alg(eqc_h); break; }
    case(Options::TOPO::ELMAT_TOPO): { top_mesh = BTM_Elmat(eqc_h); break; }
    default: { throw Exception("invalid topology type!"); break; }
    }

    return top_mesh;
  } // ElmatVAMG::BuildTopMesh


  template<class FACTORY, class HTVD, class HTED>
  shared_ptr<BlockTM> ElmatVAMG<FACTORY, HTVD, HTED> :: BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h)
  {
    throw Exception("topo from elmat TODO");
    return nullptr;
  } // ElmatVAMG::BTM_Elmat

  /** END ElmatVAMG **/

} // namespace amg

#endif
