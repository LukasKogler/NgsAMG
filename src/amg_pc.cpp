#define FILE_AMG_PC_CPP

#include "amg_pc.hpp"

/** Need all smoother headers here, not just BaseSmoother! **/
#include "amg_smoother2.hpp"
#include "amg_smoother3.hpp"
#include "amg_blocksmoother.hpp"

/** Implementing VertexAMGPCOptions SetFromFlags here is easiest **/
#include "amg_pc_vertex.hpp"

namespace amg
{

  /** Options **/

  void BaseAMGPC::Options :: SetFromFlags (const Flags & flags, string prefix)
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

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };

    auto set_opt_kv = [&](auto & opt, string name, Array<string> keys, auto vals) {
      string flag_opt = flags.GetStringFlag(prefix + name, "");
      for (auto k : Range(keys))
	if (flag_opt == keys[k]) {
	  opt = vals[k];
	  return;
	}
    };

    set_enum_opt(clev, "clev", {"inv", "sm", "none"}, Options::CLEVEL::INV_CLEV);

    set_opt_kv(cinv_type, "cinv_type", { "masterinverse", "mumps" }, Array<INVERSETYPE>({ MASTERINVERSE, MUMPS }));

    set_opt_kv(cinv_type_loc, "cinv_type_loc", { "pardiso", "pardisospd", "sparsecholesky", "superlu", "superlu_dist", "mumps", "umfpack" },
	       Array<INVERSETYPE>({ PARDISO, PARDISOSPD, SPARSECHOLESKY, SUPERLU, SUPERLU_DIST, MUMPS, UMFPACK }));

    set_enum_opt(sm_type, "sm_type", {"gs", "bgs"}, Options::SM_TYPE::GS);

    gs_ver = Options::GS_VER(max(0, min(3, int(flags.GetNumFlag(prefix + "gs_ver", 3)))));

    set_bool(sm_mpi_overlap, "sm_mpi_overlap");
    set_bool(sm_mpi_thread, "sm_mpi_thread");

    set_bool(sync, "sync");
    set_bool(do_test, "do_test");
    set_bool(smooth_lo_only, "smooth_lo_only");

  } // Options::SetFromFlags

  /** END Options**/


  /** BaseAMGPC **/

  BaseAMGPC :: BaseAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : Preconditioner(&apde, aflags, aname)
  { throw Exception("PDE constructor not implemented!"); }
    

  BaseAMGPC :: BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : Preconditioner(blf, flags, name), options(opts), bfa(blf)
  {
    ;
  } // BaseAMGPC(..)


  BaseAMGPC :: ~BaseAMGPC ()
  {
    ;
  } // ~BaseAMGPC


  void BaseAMGPC :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    if (options == nullptr) // should never happen
      { options = MakeOptionsFromFlags(flags); }
  } // BaseAMGPC::InitLevel


  void BaseAMGPC :: FinalizeLevel (const BaseMatrix * mat)
  {
    if (mat != nullptr)
      { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { finest_mat = bfa->GetMatrixPtr(); }

    Finalize();
  } // BaseAMGPC::FinalizeLevel


  const BaseMatrix & BaseAMGPC :: GetAMatrix () const 
  {
    if (finest_mat == nullptr)
      { throw Exception("BaseAMGPC - finest mat not ready!"); }
    return *finest_mat;
  } // BaseAMGPC


  const BaseMatrix & BaseAMGPC :: GetMatrix () const 
  {
    if (amg_mat == nullptr)
      { throw Exception("BaseAMGPC - amg_mat not ready!"); }
    return *amg_mat;
  } // BaseAMGPC::GetMatrix


  shared_ptr<BaseMatrix> BaseAMGPC :: GetMatrixPtr () 
  {
    if (amg_mat == nullptr)
      { throw Exception("BaseAMGPC - amg_mat not ready!"); }
    return amg_mat;
  } // BaseAMGPC::GetMatrixPtr


  void BaseAMGPC :: Mult (const BaseVector & b, BaseVector & x) const 
  {
    GetMatrix().Mult(b, x);
  } // BaseAMGPC::Mult


  void BaseAMGPC :: MultTrans (const BaseVector & b, BaseVector & x) const 
  {
    GetMatrix().MultTrans(b, x);
  } // BaseAMGPC::MultTrans


  void BaseAMGPC :: MultAdd (double s, const BaseVector & b, BaseVector & x) const 
  {
    GetMatrix().MultAdd(s, b, x);
  } // BaseAMGPC::MultAdd


  void BaseAMGPC :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const 
  {
    GetMatrix().MultTransAdd(s, b, x);
  } // BaseAMGPC::MultTransAdd


  shared_ptr<BaseAMGPC::Options> BaseAMGPC :: MakeOptionsFromFlags (const Flags & flags, string prefix)
  {
    auto opts = NewOpts();
    SetDefaultOptions(*opts);
    SetOptionsFromFlags(*opts, flags, prefix);
    ModifyOptions(*opts, flags, prefix);
    return opts;
  } // BaseAMGPC::MakeOptionsFromFlags


  void BaseAMGPC :: SetDefaultOptions (Options& O)
  {
    ;
  } // BaseAMGPC::SetDefaultOptions


  void BaseAMGPC :: SetOptionsFromFlags (Options& O, const Flags & flags, string prefix)
  {
    O.SetFromFlags(flags, prefix);
  } //BaseAMGPC::SetOptionsFromFlags


  void BaseAMGPC :: ModifyOptions (Options & O, const Flags & flags, string prefix)
  {
    ;
  } // BaseAMGPC::ModifyOptions


  void BaseAMGPC :: Finalize ()
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

    factory = BuildFactory();

    BuildAMGMat();
  } // BaseAMGPC::Finalize


  void BaseAMGPC :: BuildAMGMat ()
  {
    static Timer t("BuildAMGMat"); RegionTimer rt(t);

    auto dof_map = make_shared<DOFMap>();

    Array<BaseAMGFactory::AMGLevel> amg_levels(1); InitFinestLevel(amg_levels[0]);

    factory->SetUpLevels(amg_levels, dof_map);

    Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);
    for (auto k : Range(amg_levels))
      { smoothers[k] = BuildSmoother(amg_levels[k]); }

  } // BaseAMGPC::BuildAMGMAt


  void BaseAMGPC :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    auto finest_mesh = BuildInitialMesh();

    finest_level.level = 0;
    finest_level.mesh = finest_mesh; // TODO: get out of factory??
    finest_level.eqc_h = finest_level.mesh->GetEQCHierarchy();
    finest_level.pardofs = finest_mat->GetParallelDofs();
    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix();
    finest_level.mat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat)
      : dynamic_pointer_cast<BaseSparseMatrix>(fpm);
    finest_level.embed_map = BuildEmbedding(finest_mesh);
  } // BaseAMGPC::InitFinestLevel


  shared_ptr<BaseSmoother> BaseAMGPC :: BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level)
  {
    auto & O (*options);
    
    shared_ptr<BaseSmoother> smoother = nullptr;

    switch(O.sm_type) {
    case(Options::SM_TYPE::GS)  : { smoother = BuildGSSmoother(amg_level.mat, amg_level.pardofs, amg_level.eqc_h, GetFreeDofs(amg_level)); break; }
    case(Options::SM_TYPE::BGS) : { smoother = BuildBGSSmoother(amg_level.mat, amg_level.pardofs, amg_level.eqc_h, GetGSBlocks(amg_level)); break; }
    default : { throw Exception("Invalid Smoother type!"); break; }
    }

    return smoother;
  } // BaseAMGPC::BuildSmoother


  shared_ptr<BaseSmoother> BaseAMGPC :: BuildGSSmoother (shared_ptr<BaseSparseMatrix> spm, shared_ptr<ParallelDofs> pardofs,
							 shared_ptr<EQCHierarchy> eqc_h, shared_ptr<BitArray> freedofs)
  {
    if (spm == nullptr)
      { throw Exception("BuildGSSmoother needs a mat!"); }
    if (pardofs == nullptr)
      { throw Exception("BuildGSSmoother needs pardofs!"); }

    auto & O (*options);

    shared_ptr<BaseSmoother> smoother = nullptr;

    Switch<MAX_SYS_DIM>
      (GetEntryDim(spm.get()), [&] (auto BS)
       {
	 if constexpr ( (BS == 0) || (BS == 4) || (BS == 5)
#ifndef ELASTICITY
			|| (BS == 6)
#endif
			) {
	   throw Exception("Smoother for that dim is not compiled!!");
	   return;
	 }
	 else {
	   switch(O.gs_ver) {
	   case(Options::GS_VER::VER1) : {
	     auto spmm = dynamic_pointer_cast<stripped_spm<Mat<BS, BS, double>>>(spm);
	     smoother = make_shared<HybridGSS<BS>>(spmm, pardofs, freedofs);
	     break;
	   }
	   case(Options::GS_VER::VER2) : {
	     auto parmat = make_shared<ParallelMatrix>(spm, pardofs, pardofs, C2D);
	     smoother = make_shared<HybridGSS2<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, freedofs);
	     break;
	   }
	   case(Options::GS_VER::VER3) : {
	     if (eqc_h == nullptr)
	       { throw Exception("BuildGSSmoother needs eqc_h!"); }
	     auto parmat = make_shared<ParallelMatrix>(spm, pardofs, pardofs, C2D);
	     smoother = make_shared<HybridGSS3<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, eqc_h, freedofs, O.sm_mpi_overlap, O.sm_mpi_thread);
	     break;
	   }
	   default: { throw Exception("Invalid Smoother type!!"); break; }
	   }
	 }
       });

    return smoother;
  } // BaseAMGPC::BuildBGSSmoother


  shared_ptr<BitArray> BaseAMGPC :: GetFreeDofs (const BaseAMGFactory::AMGLevel & amg_level)
  {
    if (amg_level.level == 0)
      { return finest_freedofs; }
    else
      { return amg_level.free_nodes; }
  } // BaseAMGPC::GetFreeDofs


  shared_ptr<BaseSmoother> BaseAMGPC :: BuildBGSSmoother (shared_ptr<BaseSparseMatrix> spm, shared_ptr<ParallelDofs> pardofs,
							  shared_ptr<EQCHierarchy> eqc_h, Table<int> && blocks)
  {
    if (spm == nullptr)
      { throw Exception("BuildBGSSmoother needs a mat!"); }
    if (pardofs == nullptr)
      { throw Exception("BuildBGSSmoother needs pardofs!"); }
    if (eqc_h == nullptr)
      { throw Exception("BuildBGSSmoother needs eqc_h!"); }

    auto & O (*options);
    shared_ptr<BaseSmoother> smoother = nullptr;

    Switch<MAX_SYS_DIM>
      (GetEntryDim(spm.get()), [&] (auto BS)
       {
	 if constexpr( (BS == 0) || (BS == 4) || (BS == 5)
#ifndef ELASTICITY
		       || (BS == 6)
#endif
		       ) {
	   throw Exception("Smoother for that dim is not compiled!!");
	   return;
	 }
	 else {
	   smoother = make_shared<HybridBS<typename strip_mat<Mat<BS, BS, double>>::type>> (spm, eqc_h, move(blocks), O.sm_mpi_overlap, O.sm_mpi_thread);
	 }
       });

    return smoother;
  } // BaseAMGPC::BuildBGSSmoother


  Table<int>&& BaseAMGPC :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    throw Exception("BaseAMGPC::GetGSBlocks not overloaded!");
    return move(Table<int>());
  } // BaseAMGPC::GetGSBlocks

  /** END BaseAMGPC **/


  /** VertexAMGPCOptions **/

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

    BaseAMGPC::Options::SetFromFlags(flags, prefix);

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

  /** END VertexAMGPCOptions **/

} // namespace amg
