#ifndef FILE_AMGPC_IMPL_HPP
#define FILE_AMGPC_IMPL_HPP

namespace amg
{

  /** Options **/

  // single base class so we only have the enums once
  struct BaseEmbedAMGOptions
  {
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
			       VARIBALE_ORDERING = 1 };
    /**	REGULAR: sum(block_s) DOFs per "vertex", determined by ss_select and block_s
	   e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
	   each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks) 
	   
	VARIABLE: DOFs for vertex k: v_blocks[k] (not conistently implemented)
	subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13], but DOF 13 not in subet
    **/
    DOF_ORDERING dof_ordering = REGULAR_ORDERING;
    Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
    Table<int> v_blocks;

    /** How do we define the topology ? **/
    enum TOPO : char { ALG_TOPO = 0,        // by en entry in the finest level sparse matrix (restricted to subset)
		       MESH_TOPO = 1,       // via the mesh
		       ELMAT_TOPO = 2 };    // via element matrices
    TOPO topo = ALG_TOPO;

    /** How do we compute vertex positions (if we need them) **/
    enum POSITION : char { VERTEX_POS = 0,    // take from mesh vertex-positions
			   GIVEN_POS = 1 };   // supplied from outside
    POSITION v_pos = VERTEX_POS;
    FlatArray<Vec<3>> v_pos_array;

    /** How do we compute the replacement matrix **/
    enum ENERGY : char { TRIV_ENERGY = 0,     // uniform weights
			 ALG_ENERGY = 1,      // from the sparse matrix
			 ELMAT_ENERGY = 2 };  // from element matrices
    ENERGY energy = ALG_ENERGY;

    /** What we do on the coarsest level **/
    enum CLEVEL : char { INV_CLEV = 0,        // invert coarsest level
			 SMOOTH_CLEV = 1,    // smooth coarsest level
			 NO_CLEV = 2 };
    CLEVEL clev = INV_CLEV;
    INVERSETYPE cinv_type = MASTERINVERSE;
    INVERSETYPE cinv_type_loc = SPARSECHOLESKY;
    size_t clev_nsteps = 1;                   // if smoothing, how many steps do we do?

    /** smoother versions **/
    enum SM_VER : char { VER1 = 0,    // old version
			 VER2 = 1,    // newer (maybe a bit faster than ver3 without overlap)
			 VER3 = 2 };  // newest, optional overlap
    SM_VER sm_ver = VER3;
    bool mpi_overlap = true;          // overlap communication/computation (only VER3)
    bool mpi_thread = false;          // do MPI-comm in seperate thread (only VER3)
  }; // struct BaseEmbedAMGOptions


  template<class FACTORY>
  struct EmbedVAMG<FACTORY>::Options : public FACTORY::Options,
				       public BaseEmbedAMGOptions
  {
    bool keep_vp = false;
    bool mat_ready = false;
    bool sync = false;

    /** Smoothers **/
    bool old_smoothers = false;
    bool smooth_symmetric = false;

    bool do_test = false;
    bool smooth_lo_only = false;

    bool mpi_overlap = true;
    bool mpi_thread = false;
  };


  /** EmbedVAMG **/


  template<class FACTORY>
  shared_ptr<typename EmbedVAMG<FACTORY>::Options> EmbedVAMG<FACTORY> :: MakeOptionsFromFlags (const Flags & flags, string prefix)
  {
    auto opts = make_shared<Options>();
    auto& O(*opts);

    SetDefaultOptions(O);

    SetOptionsFromFlags(O, flags, prefix);

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };
    
    set_bool(O.sync, "sync");
    set_bool(O.old_smoothers, "oldsm");
    set_bool(O.smooth_symmetric, "symsm");
    set_bool(O.do_test, "do_test");
    set_bool(O.smooth_lo_only, "smooth_lo_only");
    set_bool(O.mpi_overlap, "sm_mpi_overlap");
    set_bool(O.mpi_thread, "sm_mpi_thread");

    set_bool(opts->mpi_overlap, "sm_mpi_overlap");
    set_bool(opts->mpi_thread, "sm_mpi_thread");

    return opts;
  }
  

  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: SetDefaultOptions (Options& O)
  { ; }


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: SetOptionsFromFlags (Options & O, const Flags & flags, string prefix)
  {

    FACTORY::SetOptionsFromFlags(O, flags, prefix);

    auto pfit = [prefix] (string x) { return prefix + x; };

    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals, auto default_val) {
      string val = flags.GetStringFlag(pfit(key), "");
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

    typedef BaseEmbedAMGOptions BAO;

    set_enum_opt(O.subset, "on_dofs", {"range", "select"}, BAO::RANGE_SUBSET);

    switch (O.subset) {
    case (BAO::RANGE_SUBSET) : {
      auto &low = flags.GetNumListFlag(pfit("lower"));
      if (low.Size()) { // defined on multiple ranges
	auto &up = flags.GetNumListFlag(pfit("upper"));
	O.ss_ranges.SetSize(low.Size());
	for (auto k : Range(low.Size()))
	  { O.ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }
	cout << IM(3) << "subset for coarsening defined by ranges, first range is [" << low[0] << ", " << up[0] << ")" << endl;
      }
      else { // a single range
	size_t lowi = flags.GetNumFlag(pfit("lower"), 0);
	size_t upi = flags.GetNumFlag(pfit("upper"), bfa->GetFESpace()->GetNDof());
	// coarsen low order part, except if we are explicitely told not to
	if ( (lowi == 0) && (upi == bfa->GetFESpace()->GetNDof()) &&
	     (!flags.GetDefineFlagX(pfit("lo")).IsFalse()) )
	  if (auto lospace = bfa->GetFESpace()->LowOrderFESpacePtr()) // e.g compound has no LO space
	    { lowi = 0; upi = lospace->GetNDof(); }
	O.ss_ranges.SetSize(1);
	O.ss_ranges[0] = { lowi, upi };
	cout << IM(3) << "subset for coarsening defined by range [" << lowi << ", " << upi << ")" << endl;
      }
      break;
    }
    case (BAO::SELECTED_SUBSET) : {
      set_enum_opt(O.spec_ss, "subset", {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__",
	    "free", "nodalp2"}, BAO::SPECSS_NONE);
      cout << IM(3) << "subset for coarsening defined by bitarray" << endl;
      // NONE - set somewhere else. FREE - set in initlevel 
      if (O.spec_ss == BAO::SPECSS_NODALP2) {
	cout << IM(3) << "taking nodalp2 subset for coarsening" << endl;
	auto fes = bfa->GetFESpace();
	O.ss_select = make_shared<BitArray>(fes->GetNDof());
	O.ss_select->Clear();
	for (auto k : Range(ma->GetNV()))
	  { O.ss_select->Set(k); }
	Array<DofId> dns;
	for (auto k : Range(ma->GetNEdges())) {
	  fes->GetDofNrs(NodeId(NT_EDGE, k), dns);
	  if (dns.Size())
	    { O.ss_select->Set(dns[0]); }
	}
	cout << IM(3) << "nodalp2 set: " << O.ss_select->NumSet() << " of " << O.ss_select->Size() << endl;
      }
      break;
    }
    default: { throw Exception("Not implemented"); break; }
    }
      
    set_enum_opt(O.dof_ordering, "dof_order", {"regular", "variable"}, BAO::REGULAR_ORDERING);

    set_enum_opt(O.topo, "edges", {"alg", "mesh", "elmat"}, BAO::ALG_TOPO);

    set_enum_opt(O.v_pos, "vpos", {"vertex", "given"}, BAO::VERTEX_POS);

    set_enum_opt(O.energy, "energy", {"triv", "alg", "elmat"}, BAO::ALG_ENERGY);

    set_enum_opt(O.clev, "clev", {"inv", "sm", "none"}, BAO::INV_CLEV);

    ModifyOptions(O, flags, prefix);

  } // EmbedVAMG::MakeOptionsFromFlags


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: ModifyOptions (Options & O, const Flags & flags, string prefix)
  { ; }


  template<class FACTORY>
  EmbedVAMG<FACTORY> :: EmbedVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name)
    : Preconditioner(blf, flags, name), bfa(blf)
  {
    options = MakeOptionsFromFlags (flags);
  } // EmbedVAMG::EmbedVAMG


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: InitLevel (shared_ptr<BitArray> freedofs)
  {

    if (freedofs == nullptr) // postpone to FinalizeLevel
      { return; }

    if (bfa->UsesEliminateInternal() || options->smooth_lo_only) {
      auto fes = bfa->GetFESpace();
      auto lofes = fes->LowOrderFESpacePtr();
      finest_freedofs = make_shared<BitArray>(*freedofs);
      auto& ofd(*finest_freedofs);
      if (bfa->UsesEliminateInternal() ) { // clear freedofs on eliminated DOFs
	auto rmax = (options->smooth_lo_only && (lofes != nullptr) ) ? lofes->GetNDof() : freedofs->Size();
	for (auto k : Range(rmax))
	  if (ofd.Test(k)) {
	    COUPLING_TYPE ct = fes->GetDofCouplingType(k);
	    if ((ct & CONDENSABLE_DOF) != 0)
	      ofd.Clear(k);
	  }
      }
      if (options->smooth_lo_only && (lofes != nullptr) ) { // clear freedofs on all high-order DOFs
	for (auto k : Range(lofes->GetNDof(), freedofs->Size()))
	  { ofd.Clear(k); }
      }
    }
    else
      { finest_freedofs = freedofs; }

    if (options->spec_ss == BaseEmbedAMGOptions::SPECSS_FREE) {
      cout << IM(3) << "taking subset for coarsening from freedofs" << endl;
      options->ss_select = finest_freedofs;
    }
  } // EmbedVAMG::InitLevel


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: FinalizeLevel (const BaseMatrix * mat)
  {

    if (mat != nullptr)
      { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { finest_mat = bfa->GetMatrixPtr(); }

    Finalize();

  } // EmbedVAMG::FinalizeLevel


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: Finalize ()
  {

    if (options->sync)
      {
	if (auto pds = finest_mat->GetParallelDofs()) {
	  static Timer t(string("Sync1")); RegionTimer rt(t);
	  pds->GetCommunicator().Barrier();
	}
      }

    if (finest_freedofs == nullptr)
      { finest_freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()); }
    
    /** Set dummy-ParallelDofs **/
    shared_ptr<BaseMatrix> fine_spm = finest_mat;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
      { fine_spm = pmat->GetMatrix(); }
    else {
      Array<int> perow (fine_spm->Height() ); perow = 0;
      Table<int> dps (perow);
      NgsMPI_Comm c(MPI_COMM_WORLD);
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : AMG_ME_COMM;
      fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(dps), GetEntryDim(fine_spm.get()), false));
    }

    auto mesh = BuildInitialMesh();

    factory = BuildFactory(mesh);

    /** Set up Smoothers **/
    BuildAMGMat();
    
  } // EmbedVAMG::Finalize


  template<class FACTORY>
  shared_ptr<FACTORY> EmbedVAMG<FACTORY> :: BuildFactory (shared_ptr<TMESH> mesh)
  {
    auto emb_step = BuildEmbedding(mesh);
    auto f = make_shared<FACTORY>(mesh, options, emb_step);
    f->free_verts = free_verts; free_verts = nullptr;
    return f;
  } // EmbedVAMG::BuildFactory


  template<class FACTORY>
  shared_ptr<BaseSparseMatrix> EmbedVAMG<FACTORY> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat,
								       shared_ptr<ParallelDofs> & pardofs) const
  { return mat; }


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: BuildAMGMat ()
  {
    static Timer t("BuildAMGMat"); RegionTimer rt(t);
    /** Build coarse level matrices and grid-transfer **/
    
    if (finest_mat == nullptr)
      { throw Exception("Do not have a finest level matrix!"); }

    shared_ptr<BaseSparseMatrix> fspm = nullptr;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { fspm = dynamic_pointer_cast<BaseSparseMatrix>(pmat->GetMatrix()); }
    else
      { fspm = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat); }

    if (fspm == nullptr)
      { throw Exception("Could not cast finest mat!"); }
    
    Array<shared_ptr<BaseSparseMatrix>> mats ({ fspm });
    auto dof_map = make_shared<DOFMap>();
    factory->SetupLevels(mats, dof_map);

    // Set up smoothers

    static Timer tsync("Sync2");
    static Timer tsm("SetupSmoothers");
    if (options->sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }
    tsm.Start();

    Array<shared_ptr<BaseSmoother>> smoothers(mats.Size()-1);
    for (auto k : Range(size_t(0), mats.Size()-1)) {
      smoothers[k] = BuildSmoother (mats[k], dof_map->GetParDofs(k), (k==0) ? finest_freedofs : nullptr);
      smoothers[k]->Finalize(); // do i even need this anymore ?
    }

    if (options->sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }
    tsm.Stop();

    amg_mat = make_shared<AMGMatrix> (dof_map, smoothers);


    // Coarsest level setup

    if (mats.Last() != nullptr) { // we might drop out because of redistribution at some point
    
      if (options->clev == BaseEmbedAMGOptions::INV_CLEV) {

	static Timer t("CoarseInv"); RegionTimer rt(t);

	shared_ptr<BaseMatrix> coarse_inv;

	auto cpds = dof_map->GetMappedParDofs();
	auto comm = cpds->GetCommunicator();
	auto cspm = mats.Last();

	cspm = RegularizeMatrix(cspm, cpds);

	if constexpr(MAX_SYS_DIM < mat_traits<typename FACTORY::TM>::HEIGHT) {
	    throw Exception(string("MAX_SYS_DIM = ") + to_string(MAX_SYS_DIM) + string(", need at least ") +
			    to_string(mat_traits<typename FACTORY::TM>::HEIGHT) + string(" for coarsest level exact Inverse!"));
	  }

	if (comm.Size() > 2) { // parallel inverse
	  auto parmat = make_shared<ParallelMatrix> (cspm, cpds, cpds, C2D);
	  parmat->SetInverseType(options->cinv_type);
	  coarse_inv = parmat->InverseMatrix();
	}
	else if ( (comm.Size() == 1) ||
		  ( (comm.Size() == 2) && (comm.Rank() == 1) ) ) { // local inverse
	  cspm->SetInverseType(options->cinv_type_loc);
	  auto cinv = cspm->InverseMatrix();
	  if (comm.Size() > 1)
	    { coarse_inv = make_shared<ParallelMatrix> (cinv, cpds, cpds, C2C); } // dummy parmat
	  else
	    { coarse_inv = cinv; }
	}
	else if (comm.Rank() == 0) { // some dummy matrix
	  Array<int> perow(0);
	  auto cinv = make_shared<SparseMatrix<double>>(perow);
	  if (comm.Size() > 1)
	    { coarse_inv = make_shared<ParallelMatrix> (cinv, cpds, cpds, C2C); } // dummy parmat
	  else
	    { coarse_inv = cinv; }
	}

	amg_mat->SetCoarseInv(coarse_inv);
      }

    } // mats.Last() != nullptr


    if (options->do_test)
      {
	printmessage_importance = 1;
	netgen::printmessage_importance = 1;
	Test();
      }

  } // EmbedVAMG::BuildAMGMat


  template<class FACTORY>
  shared_ptr<BlockTM> EmbedVAMG<FACTORY> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(BAO::MESH_TOPO): { top_mesh = BTM_Mesh(eqc_h); break; }
    case(BAO::ALG_TOPO): { top_mesh = BTM_Alg(eqc_h); break; }
    case(BAO::ELMAT_TOPO): { throw Exception("cannot do topology from elmats!"); break; }
    default: { throw Exception("invalid topology type!"); break; }
    }

    return top_mesh;
  }; //  EmbedVAMG::BuildTopMesh


  template<class FACTORY>
  shared_ptr<BlockTM> EmbedVAMG<FACTORY> :: BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    static Timer t("BTM_Mesh"); RegionTimer rt(t);

    node_sort.SetSize(4);

    typedef BaseEmbedAMGOptions BAO;

    const auto &O(*options);

    shared_ptr<BlockTM> top_mesh;

    switch (O.v_pos) {
    case(BAO::VERTEX_POS): {
      top_mesh = MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
				  false, node_sort[2], false, node_sort[3]);
      break;
    }
    case(BAO::GIVEN_POS): {
      throw Exception("Cannot combine custom vertices with topology from mesh");
      break;
    }
    default: { throw Exception("kinda unexpected case"); break; }
    }

    // this only works for the simplest case anyways...
    auto fvs = make_shared<BitArray>(top_mesh->template GetNN<NT_VERTEX>()); fvs->Clear();
    auto & vsort = node_sort[NT_VERTEX];
    for (auto k : Range(top_mesh->template GetNN<NT_VERTEX>()))
      if (finest_freedofs->Test(k))
	{ fvs->Set(vsort[k]); }
    free_verts = fvs;

    return top_mesh;
  } // EmbedVAMG::BTM_Mesh
    

  template<class FACTORY>
  shared_ptr<BlockTM> EmbedVAMG<FACTORY> :: BTM_Alg (shared_ptr<EQCHierarchy> eqc_h)
  {
    static Timer t("BTM_Alg"); RegionTimer rt(t);

    node_sort.SetSize(4);

    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    auto top_mesh = make_shared<BlockTM>(eqc_h);

    size_t n_verts = 0, n_edges = 0;

    auto & vert_sort = node_sort[NT_VERTEX];

    auto fpd = finest_mat->GetParallelDofs();

    // this only works for the simplest case anyways...

    // vertices
    auto set_vs = [&](auto nv, auto v2d) {
      vert_sort.SetSize(nv);
      top_mesh->SetVs (nv, [&](auto vnr) -> FlatArray<int> LAMBDA_INLINE { return fpd->GetDistantProcs(v2d(vnr)); },
		       [&vert_sort](auto i, auto j){ vert_sort[i] = j; });
      free_verts = make_shared<BitArray>(nv); free_verts->Clear();
      for (auto k : Range(nv)) {
	if (finest_freedofs->Test(v2d(k)))
	  { free_verts->Set(vert_sort[k]); }
	// else
	//   { free_verts->Clear(vert_sort[k]); }
      }
    };

    // edges 
    auto create_edges = [&](auto v2d, auto d2v) LAMBDA_INLINE {
      auto traverse_graph = [&](const auto& g, auto fun) LAMBDA_INLINE { // vertex->dof,  // dof-> vertex
	for (auto k : Range(n_verts)) {
	  int row = v2d(k); // for find_in_sorted_array
	  auto ri = g.GetRowIndices(row);
	  auto pos = find_in_sorted_array(row, ri); // no duplicates
	  if (pos+1 < ri.Size()) {
	    for (auto col : ri.Part(pos+1)) {
	      auto j = d2v(col);
	      if (j != -1) {
		// cout << "dofs " << row << " " << col << " are vertices " << k << " " << j << endl;
		fun(vert_sort[k],vert_sort[j]);
	      }
	      // else {
	      // 	cout << "dont use " << row << " " << j << " with dof " << col << endl;
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
	top_mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) LAMBDA_INLINE { return epairs[num]; }, // (already v-sorted)
				     [](auto node_num, auto id) { /* dont care about edge-sort! */ });
      // cout << "final n_edges: " << top_mesh->GetNN<NT_EDGE>() << endl;
      }; // create_edges

    if (O.dof_ordering == BAO::REGULAR_ORDERING) {
      const auto fes_bs = fpd->GetEntrySize();
      int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
      const auto bs0 = O.block_s[0]; // is this not kind of redundant ?
      if (O.subset == BAO::RANGE_SUBSET) {
	auto r0 = O.ss_ranges[0]; const auto maxd = r0[1];
	const int stride = bs0/fes_bs; // probably 1
	int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	n_verts = (r0[1] - r0[0]) / stride;
	auto d2v = [&](auto d) -> int LAMBDA_INLINE { return ( (d % stride == 0) && (d < r0[1]) && (r0[0] <= d) ) ? d/stride : -1; };
	auto v2d = [&](auto v) LAMBDA_INLINE { return r0[0] + v * stride; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
      }
      else { // SELECTED, subset by bitarray (is this tested??)
	size_t maxset = 0;
	auto sz = O.ss_select->Size();
	for (auto k : Range(sz))
	  if (O.ss_select->Test(--sz))
	    { maxset = sz+1; break; }
	// cout << "maxset " << maxset << endl;
	n_verts = O.ss_select->NumSet() * fes_bs / dpv;
	// cout << "n_verts " << n_verts << endl;
	Array<int> first_dof(n_verts);
	Array<int> compress(maxset); compress = -1;
	for (size_t k = 0, j = 0; k < n_verts; j += bs0 / fes_bs)
	  if (O.ss_select->Test(j))
	    { first_dof[k] = j; compress[j] = k++; }
	// cout << "first_dof  "; prow(first_dof); cout << endl << endl;
	// cout << "compress  "; prow(compress); cout << endl << endl;
	auto d2v = [&](auto d) -> int LAMBDA_INLINE { return (d+1 > maxset) ? -1 : compress[d]; };
	auto v2d = [&](auto v) LAMBDA_INLINE { return first_dof[v]; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
      }
    }
    else { // VARIABLE, subset given via table anyways (is this even tested??)
      auto& vblocks = O.v_blocks;
      n_verts = vblocks.Size();
      auto v2d = [&](auto v) { return vblocks[v][0]; };
      Array<int> compress(vblocks[n_verts-1][0]); compress = -1;
      for (auto k : Range(compress.Size())) { compress[v2d(k)] = k; }
      auto d2v = [&](auto d) -> int { return (d+1 > compress.Size()) ? -1 : compress[d]; };
      set_vs (n_verts, v2d);
      create_edges ( v2d , d2v );
    }

    cout << IM(3) << "AMG performed on " << top_mesh->GetNNGlobal<NT_VERTEX>() << " vertices, ndof local is: " << fpd->GetNDofGlobal() << endl;
    cout << IM(3) << "AMG performed on " << top_mesh->GetNNGlobal<NT_EDGE>() << " edges" << endl;
    if (top_mesh->GetEQCHierarchy()->GetCommunicator().Size() == 1) // this is only loc info
      { cout << IM(3) << "free dofs " << finest_freedofs->NumSet() << " ndof local is: " << fpd->GetNDofLocal() << endl; }

    return top_mesh;
  } // EmbedVAMG :: BTM_Alg


  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> EmbedVAMG<FACTORY> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh"); RegionTimer rt(t);

    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    /** Build inital EQC-Hierarchy **/
    shared_ptr<EQCHierarchy> eqc_h;
    auto fpd = finest_mat->GetParallelDofs();
    size_t maxset = 0;
    switch (O.subset) {
    case(BAO::RANGE_SUBSET): { maxset = O.ss_ranges.Last()[1]; break; }
    case(BAO::SELECTED_SUBSET): {
      auto sz = O.ss_select->Size();
      for (auto k : Range(sz))
	if (O.ss_select->Test(--sz))
	  { maxset = sz+1; break; }
      break;
    } }

    eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

    auto top_mesh = BuildTopMesh(eqc_h);

    /** vertex positions, if we need them **/
    if (O.keep_vp) {
      node_pos.SetSize(1);
      auto & vsort = node_sort[0]; // also kinda hard coded for vertex-vertex pos, and no additional permutation
      auto & vpos(node_pos[NT_VERTEX]); vpos.SetSize(top_mesh->template GetNN<NT_VERTEX>());
      for (auto k : Range(vpos.Size()))
	ma->GetPoint(k,vpos[vsort[k]]);
    }
    
    /** Convert FreeDofs to FreeVerts (should probably do this above where I have v2d mapping1)**/

    return BuildAlgMesh(top_mesh);
  }


  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> EmbedVAMG<FACTORY> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    shared_ptr<TMESH> alg_mesh;

    switch(O.energy) {
    case(BAO::TRIV_ENERGY): { alg_mesh = BuildAlgMesh_TRIV(top_mesh); break; }
    case(BAO::ALG_ENERGY): { alg_mesh = BuildAlgMesh_ALG(top_mesh); break; }
    case(BAO::ELMAT_ENERGY): { throw Exception("Cannot do elmat energy!"); }
    default: { throw Exception("Invalid Energy!"); break; }
    }

    return alg_mesh;
  } // EmbedVAMG::BuildAlgMesh


  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> EmbedVAMG<FACTORY> :: BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh)
  {
    static Timer t("BuildAlgMesh_ALG"); RegionTimer rt(t);

    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    shared_ptr<TMESH> alg_mesh;

    switch(O.dof_ordering) {
    case(BAO::VARIBALE_ORDERING): {
      throw Exception("not implemented (but easy)");
      break;
    }
    case(BAO::REGULAR_ORDERING): {
      shared_ptr<BaseSparseMatrix> spmat;
      if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
	{ spmat = dynamic_pointer_cast<BaseSparseMatrix>(parmat->GetMatrix()); }
      else
	{ spmat = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat); }

      const int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
      const auto& block_sizes = O.block_s;
      const int nblocks = block_sizes.Size();

      if (dpv == 1) { // most of the time - tone DOF per vertex (can be multidim DOF)
	auto n_verts = top_mesh->GetNN<NT_VERTEX>();
	auto& vsort = node_sort[NT_VERTEX];
	Array<int> d2v_array(bfa->GetFESpace()->GetNDof());
	Array<int> v2d_array(n_verts);

	if (O.subset == BAO::RANGE_SUBSET) {
	  const auto start = O.ss_ranges[0][0];
	  for (auto k : Range(n_verts)) {
	    auto svnr = vsort[k];
	    auto d = start + k;
	    d2v_array[d] = svnr;
	    v2d_array[svnr] = d;
	  }
	}
	else { // BAO::SELECTED_SUBSET
	  auto& subset = *O.ss_select;
	  for (int j = 0, k = 0; k < n_verts; j++) {
	    if (subset.Test(j)) {
	      auto d = j; auto svnr = vsort[k++];
	      d2v_array[d] = svnr;
	      v2d_array[svnr] = d;
	    }
	  }
	}

	auto d2v = [&](auto d) -> int LAMBDA_INLINE { return d2v_array[d]; };
	auto v2d = [&](auto v) -> int LAMBDA_INLINE { return v2d_array[v]; };
	alg_mesh = BuildAlgMesh_ALG_scal(top_mesh, spmat, d2v, v2d);
	break;
      }
      else { // probably only compound spaces
	throw Exception("Compound FESpaces todo - but should be easy!! ");
	break;
      }
    } // case(BAO::REGULAR_ORDERING)
    default: { throw Exception("Invalid DOF_ORDERING!"); break; }
    } // switch(O.dof_ordering)

    return alg_mesh;
  } // EmbedVAMG::BuildAlgMesh_ALG


  template<class FACTORY> shared_ptr<BaseDOFMapStep> EmbedVAMG<FACTORY> :: BuildEmbedding (shared_ptr<TMESH> mesh)
  {
    static Timer t("BuildEmbedding"); RegionTimer rt(t);

    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    /** Basically just dispatch to templated method **/

    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();

    shared_ptr<BaseMatrix> f_loc_mat;
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { f_loc_mat = parmat->GetMatrix(); }
    else
      { f_loc_mat = finest_mat; }

    if (auto spm_tm = dynamic_pointer_cast<SparseMatrix<double>> (f_loc_mat))
      { return BuildEmbedding_impl<1>(mesh); }
#ifdef ELASTICITY
    else if (auto spm_tm = dynamic_pointer_cast<stripped_spm_tm<Mat<2,2,double>>> (f_loc_mat))
      { return BuildEmbedding_impl<2>(mesh); }
    else if (auto spm_tm = dynamic_pointer_cast<stripped_spm_tm<Mat<3,3,double>>> (f_loc_mat))
      { return BuildEmbedding_impl<3>(mesh); }
    else if (auto spm_tm = dynamic_pointer_cast<stripped_spm_tm<Mat<6,6,double>>> (f_loc_mat))
      { return BuildEmbedding_impl<6>(mesh); }
#endif
    else
      { throw Exception(string("strange mat, type = ") + typeid(*f_loc_mat).name()); }

    return nullptr;
  } // EmbedVAMG::BuildEmbedding


  template<class FACTORY> template<int N>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<FACTORY> :: BuildEmbedding_impl (shared_ptr<typename FACTORY::TMESH> mesh)
  {
    /**
       Embedding  = E_SS * E_DOF * P
       E_S      ... from 0..ndof to subset                                             // N x N
       E_D      ... disp to disp-rot emb or compound-to multidim, (TODO:rot-ordering)  // N x dofpv
       P        ... permutation matrix from re-sorting vertex numbers                  // dofpv x dofpv
    **/
    
    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();

    shared_ptr<BaseMatrix> f_loc_mat;
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { f_loc_mat = parmat->GetMatrix(); }
    else
      { f_loc_mat = finest_mat; }

    constexpr int M = mat_traits<typename FACTORY::TSPM_TM::TENTRY>::HEIGHT;
    typedef stripped_spm_tm<typename strip_mat<Mat<N, N, double>>::type> T_E_S;
    typedef stripped_spm_tm<typename strip_mat<Mat<N, M, double>>::type> T_E_D;
    typedef typename FACTORY::TSPM_TM T_P;

    /**
       E_S can be nullptr when subset is all DOFs, which happens when:
         - fes is low order
    	 - nodalp2 and fes is order 2
     **/
    shared_ptr<T_E_S> E_S = BuildES<N>();

    size_t subset_count = (E_S == nullptr) ? fpds->GetNDofLocal() : E_S->Width();

    /**
       E_D can be nullptr when N==M and:
          - fes has same multidim as AMG (N == M), which have the same "meaning"
    	    as the AMG ones and no sorting within multdim-DOFs is needed (or probably always when N == M == 1)
     **/
    shared_ptr<T_E_D> E_D = BuildED<N>(subset_count, mesh);


    /**
       P is nullptr when either sequential or NP==2, in which case rank 1 has everything 
       need no sorting in those cases!
     **/
    shared_ptr<T_P> P = nullptr;
    if ( fpds->GetCommunicator().Size() > 2) {
      auto & vsort = node_sort[NT_VERTEX];
      P = BuildPermutationMatrix<typename T_P::TENTRY>(vsort);
    }

    auto a_is_b_times_c = [](auto & a, auto & b, auto & c) {
      if (b != nullptr) {
    	if (c != nullptr)
    	  { a = MatMultAB(*b, *c); }
    	else
    	  { a = b; }
      }
      else
    	{ a = c; }
    };

    shared_ptr<T_E_D> E, EDP;

    if constexpr(N == M) {
    	if constexpr (N == 1) {
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
      assert(E_D != nullptr); // E_D cannot be nullptr, we must have incorrectly called this method
      if (P != nullptr)
	{ EDP = MatMultAB(*E_D, *P); }
      else
	{ EDP = E_D; }
      if (E_S != nullptr)
	{ E = MatMultAB(*E_S, *EDP); }
      else
	{ E = EDP; }
    }

    shared_ptr<BaseDOFMapStep> emb_step = nullptr;

    if (E != nullptr)
      { emb_step = make_shared<ProlMap<T_E_D>>(E, fpds, nullptr); }

    return emb_step;
  } // EmbedVAMG::EmbedBuildEmbedding_impl


  template<class FACTORY> template<int N>
  shared_ptr<stripped_spm_tm<typename strip_mat<Mat<N, N, double>>::type>> EmbedVAMG<FACTORY> :: BuildES ()
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);
    
    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();

    typedef stripped_spm_tm<typename strip_mat<Mat<N, N, double>>::type> TS;
    shared_ptr<TS> E_S = nullptr;
    if (O.subset == BAO::RANGE_SUBSET) {
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
	int cnt_rows = 0;
	for (auto pair : O.ss_ranges)
	  { cnt_rows += pair[1] - pair[0]; }
	Array<int> perow(cnt_rows); perow = 1;
	E_S = make_shared<TS>(perow, fpds->GetNDofLocal()); cnt_rows = 0;
	for (auto pair : O.ss_ranges)
	  for (auto c : Range(pair[0], pair[1])) {
	    SetIdentity(E_S->GetRowValues(cnt_rows)[0]);
	    E_S->GetRowIndices(cnt_rows++)[0] = c;
	  }
      }
    }
    else if (O.subset == BAO::SELECTED_SUBSET) {
      if (O.ss_select == nullptr)
	{ throw Exception("SELECTED_SUBSET, but no ss_select!"); }
      const auto & SS(*O.ss_select);
      int is_triv = (SS.NumSet() == SS.Size()) ? 1 : 0;
      fpds->GetCommunicator().AllReduce(is_triv, MPI_SUM);
      if (is_triv == 0) {
	int cnt_rows = SS.NumSet();
	Array<int> perow(cnt_rows); perow = 1;
	E_S = make_shared<TS>(perow, fpds->GetNDofLocal()); cnt_rows = 0;
	for (auto k : Range(fpds->GetNDofLocal())) {
	  if (SS.Test(k)) {
	    SetIdentity(E_S->GetRowValues(cnt_rows)[0]);
	    E_S->GetRowIndices(cnt_rows++)[0] = k;
	  }
	}
      }
    }

    return E_S;
  }



  /** EmbedWithElmats **/

  template<class FACTORY, class HTVD, class HTED>
  EmbedWithElmats<FACTORY, HTVD, HTED> :: EmbedWithElmats (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : EmbedVAMG<FACTORY>(bfa, aflags, name), ht_vertex(nullptr), ht_edge(nullptr)
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    if (O.energy == BAO::ELMAT_ENERGY) {
      shared_ptr<FESpace> lofes = bfa->GetFESpace();
      if (auto V = lofes->LowOrderFESpacePtr())
	{ lofes = V; }
      size_t NV = lofes->GetNDof(); // TODO: this overestimates for compound spaces
      ht_vertex = new HashTable<int, HTVD>(NV);
      ht_edge = new HashTable<INT<2,int>, HTED>(8*NV);
    }
  }


  template<class FACTORY, class HTVD, class HTED>
  EmbedWithElmats<FACTORY, HTVD, HTED> ::  ~EmbedWithElmats ()
  {
    if (ht_vertex != nullptr) delete ht_vertex;
    if (ht_edge   != nullptr) delete ht_edge;
  }


  template<class FACTORY, class HTVD, class HTED>
  shared_ptr<BlockTM> EmbedWithElmats<FACTORY, HTVD, HTED> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(BAO::MESH_TOPO): { top_mesh = this->BTM_Mesh(eqc_h); break; }
    case(BAO::ALG_TOPO): { top_mesh = this->BTM_Alg(eqc_h); break; }
    case(BAO::ELMAT_TOPO): { top_mesh = BTM_Elmat(eqc_h); break; }
    }

    return top_mesh;
  }; //  EmbedVAMG::BuildTopMesh


  template<class FACTORY, class HTVD, class HTED>
  shared_ptr<BlockTM> EmbedWithElmats<FACTORY, HTVD, HTED> ::  BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h)
  {
    throw Exception("topo from elmat TODO"); return nullptr;
  }


  template<class FACTORY, class HTVD, class HTED>
  shared_ptr<typename EmbedWithElmats<FACTORY, HTVD, HTED>::TMESH> EmbedWithElmats<FACTORY, HTVD, HTED> ::  BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    shared_ptr<TMESH> alg_mesh;

    switch(O.energy) {
    case(BAO::TRIV_ENERGY): { alg_mesh = this->BuildAlgMesh_TRIV(top_mesh); break; }
    case(BAO::ALG_ENERGY): { alg_mesh = this->BuildAlgMesh_ALG(top_mesh); break; }
    case(BAO::ELMAT_ENERGY): { alg_mesh = BuildAlgMesh_ELMAT(top_mesh); break; }
    default: { throw Exception("Invalid Energy!"); break; }
    }

    return alg_mesh;
  } // EmbedVAMG::BuildAlgMesh


} // namespace amg

#endif //  FILE_AMGPC_IMPL_HPP
