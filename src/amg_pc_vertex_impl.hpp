#ifndef FILE_AMG_PC_VERTEX_IMPL_HPP
#define FILE_AMG_PC_VERTEX_IMPL_HPP

namespace amg
{
  /** Options **/

  class VertexAMGPCOptions
  {
  public:
    /** Which subset of DOFs to perform the coarsening on **/
    enum DOF_SUBSET : char { RANGE_SUBSET = 0,        // use Union { [ranges[i][0], ranges[i][1]) }
			     SELECTED_SUBSET = 1 };   // given by bitarray
    DOF_SUBSET subset = RANGE_SUBSET;
    Array<INT<2, size_t>> ss_ranges; // ranges must be non-overlapping and incresing
    /** special subsets **/
    enum SPECIAL_SUBSET : char { SPECSS_NONE = 0,
				 SPECSS_FREE = 1,            // take free-dofs as subset
				 SPECSS_NODALP2 = 2 };       // 0..nv, and then first DOF of each edge
    SPECIAL_SUBSET spec_ss = SPECSS_NONE;
    shared_ptr<BitArray> ss_select;
    
    /** How the DOFs in the subset are mapped to vertices **/
    enum DOF_ORDERING : char { REGULAR_ORDERING = 0,
			       VARIABLE_ORDERING = 1 };
    /**	REGULAR: sum(block_s) DOFs per "vertex", defined by block_s and ss_ranges/ss_select
	   e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
	   each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks) 
	subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13], but DOF 13 not in subet
	   
	VARIABLE: PLACEHOLDER !! || DOFs for vertex k: v_blocks[k] || ignores subset
    **/
    DOF_ORDERING dof_ordering = REGULAR_ORDERING;
    Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
    Table<int> v_blocks;

    /** AMG-Vertex <-> Mesh-Node Identification **/
    bool store_v_nodes = false;
    bool has_node_dofs[4] = { false, false, false, false };
    Array<NodeId> v_nodes;

    /** How do we define the topology ? **/
    enum TOPO : char { ALG_TOPO = 0,        // by entries of the finest level sparse matrix
		       MESH_TOPO = 1,       // via the mesh
		       ELMAT_TOPO = 2 };    // via element matrices
    TOPO topo = ALG_TOPO;

    /** How do we compute vertex positions (if we need them) (outdated..) **/
    enum POSITION : char { VERTEX_POS = 0,    // take from mesh vertex-positions
			   GIVEN_POS = 1 };   // supplied from outside
    POSITION v_pos = VERTEX_POS;
    FlatArray<Vec<3>> v_pos_array;

    /** How do we compute the replacement matrix **/
    enum ENERGY : char { TRIV_ENERGY = 0,     // uniform weights
			 ALG_ENERGY = 1,      // from the sparse matrix
			 ELMAT_ENERGY = 2 };  // from element matrices
    ENERGY energy = ALG_ENERGY;

  public:
    
    VertexAMGPCOptions () { ; }

    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix);
  }; // class VertexAMGPCOptions


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


  void VertexAMGPCOptions :: SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix)
  {

    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals, auto default_val) {
      string val = flags.GetStringFlag(prefix + key, "");
      bool found = false;
      for (auto k : Range(vals)) {
	if (val == vals[k]) {
	  found = true;
	  opt = decltype(opt)(k);
	  break;
	}
      }
      if (!found)
	{ opt = default_val; }
    };

    auto ma = fes->GetMeshAccess();
    auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

    set_enum_opt(subset, "on_dofs", {"range", "select"}, RANGE_SUBSET);

    switch (subset) {
    case (RANGE_SUBSET) : {
      auto &low = flags.GetNumListFlag(pfit("lower"));
      if (low.Size()) { // multiple ranges given by user
	auto &up = flags.GetNumListFlag(pfit("upper"));
	ss_ranges.SetSize(low.Size());
	for (auto k : Range(low.Size()))
	  { ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }
	cout << IM(3) << "subset for coarsening defined by user range(s)" << endl;
	cout << IM(5) << ss_ranges << endl;
	break;
      }
      size_t lowi = flags.GetNumFlag(pfit("lower"), -1);
      size_t upi = flags.GetNumFlag(pfit("upper"), -1);
      if ( (lowi != size_t(-1)) && (upi != size_t(-1)) ) { // single range given by user
	ss_ranges.SetSize(1);
	ss_ranges[0] = { lowi, upi };
	cout << IM(3) << "subset for coarsening defined by (single) user range" << endl;
	cout << IM(5) << ss_ranges << endl;
	break;
      }
      auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
      if (flags.GetDefineFlagX(pfit("lo")).IsFalse()) { // e.g nodalp2 (!)
	if (fes->GetMeshAccess()->GetDimension() == 2)
	  if (!flags.GetDefineFlagX(pfit("force_nolo")).IsTrue())
	    { throw Exception("lo = False probably does not make sense in 2D! (set force_nolo to True to override this!)"); }
	has_node_dofs[NT_EDGE] = true;
	ss_ranges.SetSize(1);
	ss_ranges[0][0] = 0;
	ss_ranges[0][1] = fes->GetNDof();
	cout << IM(3) << "subset for coarsening is ALL DOFs!" << endl;
      }
      else { // per default, use only low-order DOFs for coarsening
	auto get_lo_nd = [](auto & fes) LAMBDA_INLINE {
	  if (auto lofes = fes->LowOrderFESpacePtr()) // some spaces do not have a lo-space!
	    { return lofes->GetNDof(); }
	  else
	    { return fes->GetNDof(); }
	};
	std::function<void(shared_ptr<FESpace>, size_t)> set_lo_ranges =
	  [&](auto afes, auto offset) -> void LAMBDA_INLINE {
	  if (auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(afes)) {
	    size_t n_spaces = comp_fes->GetNSpaces();
	    size_t sub_os = offset;
	    for (auto space_nr : Range(n_spaces)) {
	      auto space = (*comp_fes)[space_nr];
	      set_lo_ranges(space, sub_os);
	      sub_os += space->GetNDof();
	    }
	  }
	  else if (auto reo_fes = dynamic_pointer_cast<ReorderedFESpace>(afes)) {
	    // presumably, all vertex-DOFs are low order, and these are still the first ones, so this should be fine
	    auto base_space = reo_fes->GetBaseSpace();
	    set_lo_ranges(base_space, offset);
	  }
	  else {
	    INT<2, size_t> r = { offset, offset + get_lo_nd(afes) };
	    ss_ranges.Append(r);
	    // ss_ranges.Append( { offset, offset + get_lo_nd(afes) } ); // for some reason does not work ??
	  }
	};
	ss_ranges.SetSize(0);
	set_lo_ranges(fes, 0);
	cout << IM(3) << "subset for coarsening defined by low-order range(s)" << endl;
	for (auto r : ss_ranges)
	  { cout << IM(5) << r[0] << " " << r[1] << endl; }
      }
      break;
    }
    case (SELECTED_SUBSET) : {
      set_enum_opt(spec_ss, "subset", {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__",
	    "free", "nodalp2"}, SPECSS_NONE);
      cout << IM(3) << "subset for coarsening defined by bitarray" << endl;
      // NONE - set somewhere else. FREE - set in initlevel 
      if (spec_ss == SPECSS_NODALP2) {
	if (ma->GetDimension() == 2) {
	  cout << IM(4) << "In 2D nodalp2 does nothing, using default lo base functions!" << endl;
	  subset = RANGE_SUBSET;
	  ss_ranges.SetSize(1);
	  ss_ranges[0][0] = 0;
	  if (auto lospace = fes->LowOrderFESpacePtr()) // e.g compound has no LO space
	    { ss_ranges[0][1] = lospace->GetNDof(); }
	  else
	    { ss_ranges[0][1] = fes->GetNDof(); }
	}
	else {
	  cout << IM(3) << "taking nodalp2 subset for coarsening" << endl;
	  /** 
	      Okay, we have to be careful here. We use infomation from block_s as a heuristic:
	      - We assume that the first sum(block_s) dofs of each vertex are the right ones
	      - We assume that we can split the DOFs for each edge into sum(block_s) parts and take the first
	      one each. Compatible with reordered compound space, because the order of DOFs WITHIN AN EDGE stays the same
	      example: an edge has 15 DOFs, block_s = [2,1]. then we split the DOFs into
	      [0..4], [5..10], [10..14] and take DOfs [0, 5, 10].
	      This works for:
	      - Vector-H1
	      - [Reordered Vector-H1, Reordered Vector-H1]
	      - Reordered([Vec-H1, Vec-H1])
	      - Reordered([H1, H1, ...])
	      - Ofc H1(dim=..)
	      (compound of multidim does not work anyways)
	  **/
	  has_node_dofs[NT_EDGE] = true;
	  ss_select = make_shared<BitArray>(fes->GetNDof());
	  ss_select->Clear();
	  if ( (block_s.Size() == 1) && (block_s[0] == 1) ) { // this is probably correct
	    for (auto k : Range(ma->GetNV()))
	      { ss_select->SetBit(k); }
	    Array<DofId> dns;
	    for (auto k : Range(ma->GetNEdges())) {
	      // fes->GetDofNrs(NodeId(NT_EDGE, k), dns);
	      fes->GetEdgeDofNrs(k, dns);
	      if (dns.Size())
		{ ss_select->SetBit(dns[0]); }
	    }
	  }
	  else { // an educated guess
	    const size_t dpv = std::accumulate(block_s.begin(), block_s.end(), 0);
	    Array<int> dnums;
	    auto jumped_set = [&] () LAMBDA_INLINE {
	      const int jump = dnums.Size() / dpv;
	      for (auto k : Range(dpv))
		{ ss_select->SetBit(dnums[k*jump]); }
	    };
	    for (auto k : Range(ma->GetNV())) {
	      fes->GetDofNrs(NodeId(NT_VERTEX, k), dnums);
	      jumped_set(); // probably sets all
	    }
	    for (auto k : Range(ma->GetNEdges())) {
	      // fes->GetEdgeDofNrs(k, dnums);
	      fes->GetDofNrs(NodeId(NT_EDGE, k), dnums);
	      jumped_set();
	    }
	    //cout << " best guess numset " << ss_select->NumSet() << endl;
	    //cout << *ss_select << endl;
	  }
	}
	cout << IM(3) << "nodalp2 set: " << ss_select->NumSet() << " of " << ss_select->Size() << endl;
      }
      break;
    }
    default: { throw Exception("Not implemented"); break; }
    }

    set_enum_opt(topo, "edges", {"alg", "mesh", "elmat"}, ALG_TOPO);
    set_enum_opt(v_pos, "vpos", {"vertex", "given"}, VERTEX_POS);
    set_enum_opt(energy, "energy", {"triv", "alg", "elmat"}, ALG_ENERGY);

  } // VertexAMGPCOptions::SetOptionsFromFlags


  /** END Options **/


  /** VertexAMGPC **/

  template<class FACTORY>
  VertexAMGPC<FACTORY> :: VertexAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : BaseAMGPC(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY>
  VertexAMGPC<FACTORY> :: VertexAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : BaseAMGPC(blf, flags, name, nullptr)
  {
    options = (opts == nullptr) ? MakeOptionsFromFlags(flags) : opts;
  } // VertexAMGPC(..)


  template<class FACTORY>
  VertexAMGPC<FACTORY> :: ~VertexAMGPC ()
  {
    ;
  } // ~VertexAMGPC


  template<class FACTORY>
  void VertexAMGPC<FACTORY> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    auto & O(static_cast<Options&>(*options));

    if (freedofs == nullptr) // postpone to FinalizeLevel
      { return; }

    if (bfa->UsesEliminateInternal() || O.smooth_lo_only) {
      auto fes = bfa->GetFESpace();
      auto lofes = fes->LowOrderFESpacePtr();
      finest_freedofs = make_shared<BitArray>(*freedofs);
      auto& ofd(*finest_freedofs);
      if (bfa->UsesEliminateInternal() ) { // clear freedofs on eliminated DOFs
	auto rmax = (O.smooth_lo_only && (lofes != nullptr) ) ? lofes->GetNDof() : freedofs->Size();
	for (auto k : Range(rmax))
	  if (ofd.Test(k)) {
	    COUPLING_TYPE ct = fes->GetDofCouplingType(k);
	    if ((ct & CONDENSABLE_DOF) != 0)
	      ofd.Clear(k);
	  }
      }
      if (O.smooth_lo_only && (lofes != nullptr) ) { // clear freedofs on all high-order DOFs
	for (auto k : Range(lofes->GetNDof(), freedofs->Size()))
	  { ofd.Clear(k); }
      }
    }
    else
      { finest_freedofs = freedofs; }

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

    //cout << " (unsorted) d2v_array: " << endl; prow2(d2v_array); cout << endl << endl;
    //if (use_v2d_tab) {
    //  cout << " (unsorted) v2d_table: " << endl << v2d_table << endl << endl;
    //}
    //else {
    //  cout << " (unsorted) v2d_array: " << endl; prow2(v2d_array); cout << endl << endl;
    //}

  } // VertexAMGPC::SetUpMaps


  template<class FACTORY>
  shared_ptr<BaseAMGFactory> VertexAMGPC<FACTORY> :: BuildFactory ()
  {
    return make_shared<FACTORY>(options);
  } // VertexAMGPC::BuildFactory


  template<class FACTORY>
  shared_ptr<BaseDOFMapStep> VertexAMGPC<FACTORY> :: BuildEmbedding (shared_ptr<TopologicMesh> mesh)
  {
    static Timer t("BuildEmbedding"); RegionTimer rt(t);
    return nullptr;
  } // VertexAMGPC::BuildEmbedding


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
       	// if (!free_verts->Test(k)) { cout << k << " " << endl; }
      // cout << endl;
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
    finest_level.free_nodes = free_verts;
  }


  template<class FACTORY>
  Table<int>&& VertexAMGPC<FACTORY> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    if (amg_level.crs_map == nullptr) {
      throw Exception("Crs Map not saved!!");
      return move(Table<int>());
    }

    int NCV = amg_level.crs_map->GetMappedNN<NT_VERTEX>();
    int n_blocks = NCV;
    if (amg_level.disc_map != nullptr)
      { n_blocks += amg_level.disc_map->GetNDroppedNodes<NT_VERTEX>(); }
    TableCreator<int> cblocks(n_blocks);
    auto it_blocks = [&](auto NV, auto map_v) {
      for (auto k : Range(NV)) {
	auto cv = map_v(k);
	if (cv != -1)
	  { cblocks.Add(cv, k); }
      }
    };
    auto vmap = amg_level.crs_map->GetMap<NT_VERTEX>();
    for (; !cblocks.Done(); cblocks++) {
      if (amg_level.disc_map == nullptr)
	{ calc_blocks(vmap.Size(), [&](auto v)->int { return vmap[v]; }); }
      else {
	const auto & drop = *amg_level.disc_map->GetDroppedNodes<NT_VERTEX>();
	auto drop_map = amg_level.disc_map->GetMap<NT_VERTEX>();
	calc_blocks(drop.Size(), [&](auto v)->int {
	    auto midv = drop_map[v]; // have to consider drop OR DIRI !!
	    return (midv == -1) ? midv : vmap[midv];
	  });
	int c = NCV;
	for (auto k : Range(drop.Size())) {
	  if (drop.Test(k))
	    { cblocks.Add(c++, k); }
	}
      }
    }
    return cblocks.MoveTable();
  } // VertexAMGPC<FACTORY>::GetGSBlocks


  /** END VertexAMGPC **/


  /** ElmatVAMG **/


  template<class FACTORY, class HTVD, class HTED>
  ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (const PDE & apde, const Flags & aflags, const string aname)
    : VertexAMGPC<FACTORY>(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY, class HTVD, class HTED>
  ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : VertexAMGPC<FACTORY>(blf, flags, name, nullptr)
  {
    options = (opts == nullptr) ? this->MakeOptionsFromFlags(flags) : opts;
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


  /** END ElmatVAMG **/

} // namespace amg

#endif
