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
    bool mat_ready = false;
    bool sync = false;

    /** Smoothers **/
    bool old_smoothers = false;
    bool smooth_symmetric = false;

    bool do_test = false;
    bool smooth_lo_only = false;
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

    
    auto fes = bfa->GetFESpace();

    set_enum_opt(O.dof_ordering, "dof_order", {"regular", "variable"}, BAO::REGULAR_ORDERING);

    auto &bs = flags.GetNumListFlag(pfit("dof_blocks"));

    if (bs.Size()) {
      O.block_s.SetSize(bs.Size());
      for (auto k : Range(O.block_s))
	{ O.block_s[k] = int(bs[k]); }
    }


    set_enum_opt(O.subset, "on_dofs", {"range", "select"}, BAO::RANGE_SUBSET);

    switch (O.subset) {
    case (BAO::RANGE_SUBSET) : {
      auto &low = flags.GetNumListFlag(pfit("lower"));
      if (low.Size()) { // multiple ranges given by user
	auto &up = flags.GetNumListFlag(pfit("upper"));
	O.ss_ranges.SetSize(low.Size());
	for (auto k : Range(low.Size()))
	  { O.ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }
	cout << IM(3) << "subset for coarsening defined by user range(s)" << endl;
	cout << IM(5) << O.ss_ranges << endl;
	break;
      }
      size_t lowi = flags.GetNumFlag(pfit("lower"), -1);
      size_t upi = flags.GetNumFlag(pfit("upper"), -1);
      if ( (lowi != size_t(-1)) && (upi != size_t(-1)) ) { // single range given by user
	O.ss_ranges.SetSize(1);
	O.ss_ranges[0] = { lowi, upi };
	cout << IM(3) << "subset for coarsening defined by (single) user range" << endl;
	cout << IM(5) << O.ss_ranges << endl;
	break;
      }
      auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
      if (flags.GetDefineFlagX(pfit("lo")).IsFalse()) { // e.g nodalp2 (!)
	if (fes->GetMeshAccess()->GetDimension() == 2)
	  if (!flags.GetDefineFlagX(pfit("force_nolo")).IsTrue())
	    { throw Exception("lo = False probably does not make sense in 2D! (set force_nolo to True to override this!)"); }
	O.has_node_dofs[NT_EDGE] = true;
	O.ss_ranges.SetSize(1);
	O.ss_ranges[0][0] = 0;
	O.ss_ranges[0][1] = fes->GetNDof();
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
	    O.ss_ranges.Append(r);
	    // O.ss_ranges.Append( { offset, offset + get_lo_nd(afes) } ); // for some reason does not work ??
	  }
	};
	O.ss_ranges.SetSize(0);
	set_lo_ranges(fes, 0);
	cout << IM(3) << "subset for coarsening defined by low-order range(s)" << endl;
	for (auto r : O.ss_ranges)
	  cout << IM(5) << r[0] << " " << r[1] << endl;
      }
      break;
    }
    case (BAO::SELECTED_SUBSET) : {
      set_enum_opt(O.spec_ss, "subset", {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__",
	    "free", "nodalp2"}, BAO::SPECSS_NONE);
      cout << IM(3) << "subset for coarsening defined by bitarray" << endl;
      // NONE - set somewhere else. FREE - set in initlevel 
      if (O.spec_ss == BAO::SPECSS_NODALP2) {
	if (ma->GetDimension() == 2) {
	  cout << IM(4) << "In 2D nodalp2 does nothing, using default lo base functions!" << endl;
	  O.subset = BAO::RANGE_SUBSET;
	  O.ss_ranges.SetSize(1);
	  O.ss_ranges[0][0] = 0;
	  if (auto lospace = fes->LowOrderFESpacePtr()) // e.g compound has no LO space
	    { O.ss_ranges[0][1] = lospace->GetNDof(); }
	  else
	    { O.ss_ranges[0][1] = fes->GetNDof(); }
	}
	else {
	  cout << IM(3) << "taking nodalp2 subset for coarsening" << endl;
	  /** 
	      Okay, we have to be careful here. We use infomation from O.block_s as a heuristic:
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
	  O.has_node_dofs[NT_EDGE] = true;
	  O.ss_select = make_shared<BitArray>(fes->GetNDof());
	  O.ss_select->Clear();
	  if ( (O.block_s.Size() == 1) && (O.block_s[0] == 1) ) { // this is probably correct
	    for (auto k : Range(ma->GetNV()))
	    { O.ss_select->SetBit(k); }
	    Array<DofId> dns;
	    for (auto k : Range(ma->GetNEdges())) {
	      // fes->GetDofNrs(NodeId(NT_EDGE, k), dns);
	      fes->GetEdgeDofNrs(k, dns);
	      if (dns.Size())
		{ O.ss_select->SetBit(dns[0]); }
	    }
	  }
	  else { // an educated guess
	    const size_t dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	    Array<int> dnums;
	    auto jumped_set = [&]() {
	      const int jump = dnums.Size() / dpv;
	      // int os = 0;
	      // for (auto k : Range(O.block_s.Size())) {
	      // 	for (auto j : Range(O.block_s[k]))
	      // 	  { O.ss_select->SetBit(dnums[os+j]); }
	      // 	os += O.block_s[k] * jump;
	      // }
	      for (auto k : Range(dpv))
		{ O.ss_select->SetBit(dnums[k*jump]); }
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
	    //cout << " best guess numset " << O.ss_select->NumSet() << endl;
	    //cout << *O.ss_select << endl;
	  }
	}
	cout << IM(3) << "nodalp2 set: " << O.ss_select->NumSet() << " of " << O.ss_select->Size() << endl;
      }
      break;
    }
    default: { throw Exception("Not implemented"); break; }
    }
      
    set_enum_opt(O.topo, "edges", {"alg", "mesh", "elmat"}, BAO::ALG_TOPO);

    set_enum_opt(O.v_pos, "vpos", {"vertex", "given"}, BAO::VERTEX_POS);

    set_enum_opt(O.energy, "energy", {"triv", "alg", "elmat"}, BAO::ALG_ENERGY);

    set_enum_opt(O.clev, "clev", {"inv", "sm", "none"}, BAO::INV_CLEV);

    auto set_opt_kv = [&](auto & opt, string name, Array<string> keys, auto vals) {
      string flag_opt = flags.GetStringFlag(pfit(name), "");
      for (auto k : Range(keys))
	if (flag_opt == keys[k]) {
	  opt = vals[k];
	  return;
	}
    };

    set_opt_kv(O.cinv_type, "cinv_type", { "masterinverse", "mumps" }, Array<INVERSETYPE>({ MASTERINVERSE, MUMPS }));

    set_opt_kv(O.cinv_type_loc, "cinv_type_loc", { "pardiso", "pardisospd", "sparsecholesky", "superlu", "superlu_dist", "mumps", "umfpack" },
	       Array<INVERSETYPE>({ PARDISO, PARDISOSPD, SPARSECHOLESKY, SUPERLU, SUPERLU_DIST, MUMPS, UMFPACK }));

    auto num_sm_ver = flags.GetNumFlag(pfit("sm_ver"), 3);
    if (num_sm_ver == 1)
      { O.sm_ver = BAO::VER1; }
    else if (num_sm_ver == 2)
      { O.sm_ver = BAO::VER2; }
    else
      { O.sm_ver = BAO::VER3; }

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
      //cout << " freedofs (for coarsening) set " << options->ss_select->NumSet() << " of " << options->ss_select->Size() << endl;
      //cout << *options->ss_select << endl;
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
      //cout << " sm " << k << " " << mats.Size() << endl;
      //cout << *mats[k] << endl;
      smoothers[k] = BuildSmoother (mats[k], dof_map->GetParDofs(k), (k==0) ? finest_freedofs : nullptr);
      smoothers[k]->Finalize(); // do i even need this anymore ?
    }

    if (options->sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }
    tsm.Stop();

    amg_mat = make_shared<AMGMatrix> (dof_map, smoothers);

    //cout << " now coarse level " << endl;
    // Coarsest level setup

    if (mats.Last() != nullptr) { // we might drop out because of redistribution at some point
    
      if (options->clev == BaseEmbedAMGOptions::INV_CLEV) {

	static Timer t("CoarseInv"); RegionTimer rt(t);

	shared_ptr<BaseMatrix> coarse_inv;

	auto cpds = dof_map->GetMappedParDofs();
	auto comm = cpds->GetCommunicator();
	auto cspm = mats.Last();

	cspm = RegularizeMatrix(cspm, cpds);

	// cout << "cspm: " << endl;
	// print_tm_spmat(cout, static_cast<SparseMatrix<typename FACTORY::TM>&>(*cspm));
	// cout << endl;

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

    throw Exception("I don't think this is functional...");

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

    // TODO: re-map d2v/v2d

    // this only works for the simplest case anyways...
    auto fvs = make_shared<BitArray>(top_mesh->template GetNN<NT_VERTEX>()); fvs->Clear();
    auto & vsort = node_sort[NT_VERTEX];
    for (auto k : Range(top_mesh->template GetNN<NT_VERTEX>()))
      if (finest_freedofs->Test(k))
	{ fvs->SetBit(vsort[k]); }
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


    // cout << " btm alg, fds " << endl << *finest_freedofs << endl;

    // cout << " make mesh from mat: " << endl;
    // shared_ptr<BaseMatrix> fspm;
    // if (auto pm = dynamic_cast<ParallelMatrix*>(finest_mat.get()))
    //   fspm = pm->GetMatrix();
    // else
    //   fspm = finest_mat;
    // auto msm = dynamic_pointer_cast<SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>>>(fspm);
    // cout << " msm " << msm << endl;
    // // auto msm = dynamic_pointer_cast<SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>>>(finest_mat);
    // print_tm_spmat(cout, *msm); cout << endl;
    
    // vertices
    auto set_vs = [&](auto nv, auto v2d) {
      n_verts = nv;
      vert_sort.SetSize(nv);
      top_mesh->SetVs (nv, [&](auto vnr) LAMBDA_INLINE { return fpd->GetDistantProcs(v2d(vnr)); },
		       [&vert_sort](auto i, auto j){ vert_sort[i] = j; });
      free_verts = make_shared<BitArray>(nv);
      if (finest_freedofs != nullptr) {
	free_verts->Clear();
	for (auto k : Range(nv)) {
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
    };
    
    if (use_v2d_tab) {
      set_vs(v2d_table.Size(), [&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; });
    }
    else {
      set_vs(v2d_array.Size(), [&](auto i) LAMBDA_INLINE { return v2d_array[i]; });
    }

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
  } // EmbedVAMG :: BTM_Alg


  template<class FACTORY>
  void EmbedVAMG<FACTORY> :: SetUpMaps ()
  {
    static Timer t("SetUpMaps"); RegionTimer rt(t);

    typedef BaseEmbedAMGOptions BAO;
    auto & O(*options);

    const size_t ndof = bfa->GetFESpace()->GetNDof();
    size_t n_verts = -1;

    switch(O.subset) {
    case(BAO::RANGE_SUBSET): {

      size_t in_ss = 0;
      for (auto range : O.ss_ranges)
	{ in_ss += range[1] - range[0]; }

      if (O.dof_ordering == BAO::VARIABLE_ORDERING)
	{ throw Exception("not implemented (but easy)"); }
      else if (O.dof_ordering == BAO::REGULAR_ORDERING) {
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
    case(BAO::SELECTED_SUBSET): {

      // cout << " ss_sel " << O.ss_select << endl;

      const auto & subset = *O.ss_select;
      size_t in_ss = subset.NumSet();

      if (O.dof_ordering == BAO::VARIABLE_ORDERING)
	{ throw Exception("not implemented (but easy)"); }
      else if (O.dof_ordering == BAO::REGULAR_ORDERING) {
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
	  cout << " after NT " << NT << ", set " << numset << " of " << n_verts << endl;
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

  } // EmbedVAMG::SetUpMaps


  template<class FACTORY>
  shared_ptr<typename FACTORY::TMESH> EmbedVAMG<FACTORY> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh"); RegionTimer rt(t);

    SetUpMaps();

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
    // cout << "E_S: " << endl;
    // if (E_S) { cout << E_S->Height() << " x " << E_S->Width() << endl; cout << *E_S << endl; }
    // else cout << " NO E_S!!" << endl;

    size_t subset_count = (E_S == nullptr) ? fpds->GetNDofLocal() : E_S->Width();

    /**
       E_D can be nullptr when N == M and:
          - fes has same multidim as AMG (N == M), which have the same "meaning"
    	    as the AMG ones and no sorting within multdim-DOFs is needed (or probably always when N == M == 1)
     **/
    shared_ptr<T_E_D> E_D = BuildED<N>(subset_count, mesh);
    // cout << "E_D: " << endl; if (E_D) cout << *E_D << endl; cout << endl;
      

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

    // auto prt = [](auto name, auto x) {
    //   cout << name << " ";
    //   if (x == nullptr)
    // 	{ cout << "nullptr!" << endl; }
    //   else
    // 	{ cout << x->Height() << " x " << x->Width() << endl << "--" << endl << *x << endl << "---" << endl; }
    // };
    // prt("E_S", E_S);
    // prt("E_D", E_D);
    // prt("P", P);
    // prt("E", E);

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
    else if (O.subset == BAO::SELECTED_SUBSET) {
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
