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

    auto set_opt_sv = [&](auto & opt, string flag_opt, FlatArray<string> keys, auto & vals) {
      for (auto k : Range(keys))
	if (flag_opt == keys[k]) {
	  opt = vals[k];
	  return;
	}
    };
    
    auto set_opt_kv = [&](auto & opt, string name, Array<string> keys, auto vals) {
      string flag_opt = flags.GetStringFlag(prefix + name, "");
      set_opt_sv(opt, flag_opt, keys, vals);
    };

    set_enum_opt(clev, "clev", {"inv", "sm", "none"}, Options::CLEVEL::INV_CLEV);

    set_opt_kv(cinv_type, "cinv_type", { "masterinverse", "mumps" }, Array<INVERSETYPE>({ MASTERINVERSE, MUMPS }));

    set_opt_kv(cinv_type_loc, "cinv_type_loc", { "pardiso", "pardisospd", "sparsecholesky", "superlu", "superlu_dist", "mumps", "umfpack" },
	       Array<INVERSETYPE>({ PARDISO, PARDISOSPD, SPARSECHOLESKY, SUPERLU, SUPERLU_DIST, MUMPS, UMFPACK }));

    Array<string> sm_names ( { "gs", "bgs" } );
    Array<Options::SM_TYPE> sm_types ( { Options::SM_TYPE::GS, Options::SM_TYPE::BGS } );
    set_enum_opt(sm_type, "sm_type", { "gs", "bgs" }, Options::SM_TYPE::GS);
    auto & spec_sms = flags.GetStringListFlag(prefix + "spec_sm_types");
    spec_sm_types.SetSize(spec_sms.Size());
    for (auto k : Range(spec_sms.Size()))
      { set_opt_sv(spec_sm_types[k], spec_sms[k], sm_names, sm_types); }

    gs_ver = Options::GS_VER(max(0, min(3, int(flags.GetNumFlag(prefix + "gs_ver", 3) - 1))));

    set_bool(sm_symm, "sm_symm");
    set_bool(sm_mpi_overlap, "sm_mpi_overlap");
    set_bool(sm_mpi_thread, "sm_mpi_thread");
    set_bool(sm_shm, "sm_shm");
    set_bool(sm_sl2, "sm_sl2");

    set_bool(sync, "sync");
    set_bool(do_test, "do_test");
    set_bool(smooth_lo_only, "smooth_lo_only");
    set_bool(regularize_cmats, "regularize_cmats");

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

    const auto & O(*options);

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

  } // BaseAMGPC::InitLevel


  void BaseAMGPC :: FinalizeLevel (const BaseMatrix * mat)
  {
    if (mat != nullptr)
      { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { finest_mat = bfa->GetMatrixPtr(); }

    Finalize();
  } // BaseAMGPC::FinalizeLevel


  shared_ptr<AMGMatrix> BaseAMGPC :: GetAMGMatrix () const
  {
    return amg_mat;
  } // BaseAMGPC::GetAMGMatrix


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
    O.sm_shm = !bfa->GetFESpace()->IsParallel();
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
    auto & O(*options);

    static Timer t("BuildAMGMat"); RegionTimer rt(t);
    static Timer tsync("BuildAMGMat::sync");

    auto dof_map = make_shared<DOFMap>();

    Array<BaseAMGFactory::AMGLevel> amg_levels(1); InitFinestLevel(amg_levels[0]);

    if (options->sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }

    factory->SetUpLevels(amg_levels, dof_map);

    /** Smoothers **/
    Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);
    for (int k = 0; k < amg_levels.Size() - 1; k++) {
      if ( (k > 0) && O.regularize_cmats) // Regularize coarse level matrices
	{ RegularizeMatrix(amg_levels[k].mat, amg_levels[k].pardofs); }
      smoothers[k] = BuildSmoother(amg_levels[k]);
    }

    amg_mat = make_shared<AMGMatrix> (dof_map, smoothers);

    /** Coarsest level inverse **/
    if (O.sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }
    

    if ( (amg_levels.Size() > 1) && (amg_levels.Last().mat != nullptr) ) { // otherwise, dropped out
      switch(O.clev) {
      case(Options::CLEVEL::INV_CLEV) : {
	static Timer t("CoarseInv"); RegionTimer rt(t);

	auto & c_lev = amg_levels.Last();
	auto cpds = dof_map->GetMappedParDofs();
	auto comm = cpds->GetCommunicator();
	auto cspm = c_lev.mat;

	// auto cspmtm = dynamic_pointer_cast<SparseMatrixTM<Mat<3,3,double>>>(cspm);
	// cout << " Coarse mat: " << endl;
	// print_tm_spmat(cout, *cspmtm);

	if (O.regularize_cmats)
	  {  RegularizeMatrix(cspm, cpds); }

	shared_ptr<BaseMatrix> coarse_inv = nullptr;
      
	if (GetEntryDim(cspm.get()) > MAX_SYS_DIM) // when would this ever happen??
	  { throw Exception("Cannot inv coarse level, MAX_SYS_DIM insufficient!"); }

	if (comm.Size() > 2) {
	  auto parmat = make_shared<ParallelMatrix> (cspm, cpds, cpds, C2D);
	  parmat->SetInverseType(O.cinv_type);
	  coarse_inv = parmat->InverseMatrix();
	}
	else if ( (comm.Size() == 1) ||
		  ( (comm.Size() == 2) && (comm.Rank() == 1) ) ) { // local inverse
	  cspm->SetInverseType(O.cinv_type_loc);
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
	break;
      }
      default : { break; }
      }
    }

    if (options->do_test)
      {
	printmessage_importance = 1;
	netgen::printmessage_importance = 1;
	Test();
      }

  } // BaseAMGPC::BuildAMGMAt


  void BaseAMGPC :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    auto finest_mesh = BuildInitialMesh();

    // cout << "init mesh: " << endl << finest_mesh << endl;
    // if (finest_mesh != nullptr)
      // cout << *finest_mesh << endl;
    // else
      // { throw Exception("HAVE NOT BUILT FINEST MESH CORRECTLY!!!!"); }

    finest_level.level = 0;
    finest_level.mesh = finest_mesh; // TODO: get out of factory??
    finest_level.eqc_h = finest_level.mesh->GetEQCHierarchy();
    finest_level.pardofs = finest_mat->GetParallelDofs();
    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
    finest_level.mat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat)
      : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());
    finest_level.embed_map = BuildEmbedding(finest_mesh);
  } // BaseAMGPC::InitFinestLevel


  shared_ptr<BaseSmoother> BaseAMGPC :: BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level)
  {
    auto & O (*options);
    
    shared_ptr<BaseSmoother> smoother = nullptr;

    // cout << " smoother, mat " << amg_level.mat->Height() << " x " << amg_level.mat->Width() << endl;
    // cout << " pds " << amg_level.pardofs->GetNDofLocal() << endl;

    Options::SM_TYPE sm_type = O.sm_type;

    if (O.spec_sm_types.Size() > amg_level.level)
      { sm_type = O.spec_sm_types[amg_level.level]; }

    switch(sm_type) {
    case(Options::SM_TYPE::GS)  : { smoother = BuildGSSmoother(amg_level.mat, amg_level.pardofs, amg_level.eqc_h, GetFreeDofs(amg_level)); break; }
    case(Options::SM_TYPE::BGS) : { smoother = BuildBGSSmoother(amg_level.mat, amg_level.pardofs, amg_level.eqc_h, move(GetGSBlocks(amg_level))); break; }
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

    Switch<MAX_SYS_DIM> // 
      (GetEntryDim(spm.get())-1, [&] (auto BSM)
       {
	 constexpr int BS = BSM + 1;
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
	     auto hgsm = make_shared<HybridGSS<BS>>(spmm, pardofs, freedofs);
	     hgsm->Finalize();
	     hgsm->SetSymmetric(O.sm_symm);
	     smoother = hgsm;
	     break;
	   }
	   case(Options::GS_VER::VER2) : {
	     auto parmat = make_shared<ParallelMatrix>(spm, pardofs, pardofs, C2D);
	     auto hgsm = make_shared<HybridGSS2<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, freedofs);
	     hgsm->Finalize();
	     hgsm->SetSymmetric(O.sm_symm);
	     smoother = hgsm;
	     break;
	   }
	   case(Options::GS_VER::VER3) : {
	     if (eqc_h == nullptr)
	       { throw Exception("BuildGSSmoother needs eqc_h!"); break; }
	     auto parmat = make_shared<ParallelMatrix>(spm, pardofs, pardofs, C2D);
	     auto hgsm = make_shared<HybridGSS3<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, eqc_h, freedofs, O.sm_mpi_overlap, O.sm_mpi_thread);
	     hgsm->Finalize();
	     hgsm->SetSymmetric(O.sm_symm);
	     smoother = hgsm;
	     break;
	   }
	   default: { throw Exception("Invalid GS version!!"); break; }
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
							  shared_ptr<EQCHierarchy> eqc_h, Table<int> && _blocks)
  {
    if (spm == nullptr)
      { throw Exception("BuildBGSSmoother needs a mat!"); }
    if (pardofs == nullptr)
      { throw Exception("BuildBGSSmoother needs pardofs!"); }
    if (eqc_h == nullptr)
      { throw Exception("BuildBGSSmoother needs eqc_h!"); }

    auto blocks = move(_blocks);
    // cout << " BGSS w. blocks " << blocks << endl;

    auto & O (*options);
    shared_ptr<BaseSmoother> smoother = nullptr;

    Switch<MAX_SYS_DIM> // 0-based
      (GetEntryDim(spm.get())-1, [&] (auto BSM)
       {
	 constexpr int BS = BSM + 1;
	 if constexpr( (BS == 4) || (BS == 5)
#ifndef ELASTICITY
		       || (BS == 6)
#endif
		       ) {
	   throw Exception("Smoother for that dim is not compiled!!");
	   return;
	 }
	 else {
	   auto parmat = make_shared<ParallelMatrix>(spm, pardofs);
	   auto bsm = make_shared<HybridBS<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, eqc_h, move(blocks),
											    O.sm_mpi_overlap, O.sm_mpi_thread,
											    O.sm_shm, O.sm_sl2);
	   bsm->Finalize();
	   bsm->SetSymmetric(O.sm_symm);
	   smoother = bsm;
	 }
       });

    return smoother;
  } // BaseAMGPC::BuildBGSSmoother


  Table<int> BaseAMGPC :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    throw Exception("BaseAMGPC::GetGSBlocks not overloaded!");
    return move(Table<int>());
  } // BaseAMGPC::GetGSBlocks


  void BaseAMGPC :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
  {
    ;
  }

  /** END BaseAMGPC **/


  /** VertexAMGPCOptions **/

  void VertexAMGPCOptions :: SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix)
  {

    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals) {
      string val = flags.GetStringFlag(prefix + key, "");
      for (auto k : Range(vals)) {
	if (val == vals[k]) {
	  opt = decltype(opt)(k);
	  break;
	}
      }
    };

    auto ma = fes->GetMeshAccess();
    auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

    BaseAMGPC::Options::SetFromFlags(flags, prefix);

    set_enum_opt(subset, "on_dofs", {"range", "select"});

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
	std::function<void(shared_ptr<FESpace>, size_t, Array<INT<2,size_t>> &)> set_lo_ranges =
	  [&](auto afes, auto offset, auto & ranges) -> void LAMBDA_INLINE {
	  if (auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(afes)) {
	    size_t n_spaces = comp_fes->GetNSpaces();
	    size_t sub_os = offset;
	    for (auto space_nr : Range(n_spaces)) {
	      auto space = (*comp_fes)[space_nr];
	      set_lo_ranges(space, sub_os, ranges);
	      sub_os += space->GetNDof();
	    }
	  }
	  else if (auto reo_fes = dynamic_pointer_cast<ReorderedFESpace>(afes)) {
	    // presumably, all vertex-DOFs are low order, and these are still the first ones, so this should be fine
	    auto base_space = reo_fes->GetBaseSpace();
	    Array<INT<2,size_t>> oranges; // original ranges - not taking reorder into account
	    set_lo_ranges(base_space, 0, oranges);
	    size_t orange_sum = 0;
	    for (auto & r : oranges)
	      { orange_sum += (r[1] - r[0]); }
	    INT<2, size_t> r = { offset, offset + orange_sum };
	    ranges.Append(r);
	    // set_lo_ranges(base_space, offset, ranges);
	  }
	  else {
	    INT<2, size_t> r = { offset, offset + get_lo_nd(afes) };
	    ranges.Append(r);
	    // ss_ranges.Append( { offset, offset + get_lo_nd(afes) } ); // for some reason does not work ??
	  }
	};
	ss_ranges.SetSize(0);
	set_lo_ranges(fes, 0, ss_ranges);
	cout << IM(3) << "subset for coarsening defined by low-order range(s)" << endl;
	for (auto r : ss_ranges)
	  { cout << IM(5) << r[0] << " " << r[1] << endl; }
      }
      break;
    }
    case (SELECTED_SUBSET) : {
      set_enum_opt(spec_ss, "subset", {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__",
	    "free", "nodalp2"});
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

    set_enum_opt(topo, "edges", { "alg", "mesh", "elmat" });
    set_enum_opt(v_pos, "vpos", { "vertex", "given" } );
    set_enum_opt(energy, "energy", { "triv", "alg", "elmat" });

  } // VertexAMGPCOptions::SetOptionsFromFlags

  /** END VertexAMGPCOptions **/

} // namespace amg
