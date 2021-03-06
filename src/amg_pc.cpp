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
  class AMGSmoother2 : public BaseSmoother
  {
  protected:
    int start_level;
    shared_ptr<AMGMatrix> amg_mat;
  public:

    AMGSmoother2 (shared_ptr<AMGMatrix> _amg_mat, int _start_level = 0)
      : BaseSmoother(_amg_mat->GetSmoother(_start_level)->GetAMatrix()),
	start_level(_start_level), amg_mat(_amg_mat)
    { ; }

    ~AMGSmoother2 () { ; }

    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero = false) const override
    {
      amg_mat->SmoothVFromLevel(start_level, x, b, res, res_updated, update_res, x_zero);
    }

    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = true, bool x_zero = false) const override
    {
      // x = 0.0;
      // amg_mat->SmoothVFromLevel(start_level, x, b, res, res_updated, update_res, x_zero);
    }

    virtual int VHeight () const override { return amg_mat->GetSmoother(start_level)->GetAMatrix()->VHeight(); }
    virtual int VWidth () const override  { return amg_mat->GetSmoother(start_level)->GetAMatrix()->VWidth(); }
    // virtual AutoVector CreateVector () const override { return sm->CreateVector(); };
    virtual AutoVector CreateColVector () const override { return amg_mat->GetSmoother(start_level)->CreateColVector(); };
    virtual AutoVector CreateRowVector () const override { return amg_mat->GetSmoother(start_level)->CreateColVector(); };

  }; // class AMGSmoother2

  class SmootherBM : public BaseMatrix
  {
  protected:
    shared_ptr<BaseSmoother> sm;
    AutoVector res;
    bool sym;
  public:
    // SmootherBM(shared_ptr<BaseSmoother> _sm, bool _sym = true) : sm(_sm), sym(_sym) { res.AssignPointer(sm->CreateColVector()); }
    SmootherBM(shared_ptr<BaseSmoother> _sm, bool _sym = true)
      : sm(_sm), res(move(_sm->CreateColVector())), sym(_sym)
    { ; }

    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    {
      BaseVector & r2 (const_cast<BaseVector&>(*res));
      x = 0.0;
      x.Cumulate();
      b.Distribute();
      r2 = b;
      // updated, update, zero
      if (sym) {
	sm->Smooth(x, b, r2, true, true, true);
	x.Distribute();
	sm->SmoothBack(x, b, r2, true, false, false);
	x.Cumulate();
      }
      else
	{ sm->Smooth(x, b, r2, true, false, true); }
    }
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override { Mult(b, x); }

    virtual int VHeight () const override { return sm->VHeight(); }
    virtual int VWidth () const override  { return sm->VWidth(); }
    // virtual AutoVector CreateVector () const override { return sm->CreateVector(); };
    virtual AutoVector CreateColVector () const override { return sm->CreateColVector(); };
    virtual AutoVector CreateRowVector () const override { return sm->CreateColVector(); };
  };

  void DoTest (BaseMatrix &mat, BaseMatrix &pc, NgMPI_Comm & gcomm) {
    auto i1 = printmessage_importance;
    auto i2 = netgen::printmessage_importance;
    printmessage_importance = 1;
    netgen::printmessage_importance = 1;
    EigenSystem eigen(mat, pc); // need parallel mat
    eigen.SetPrecision(1e-12);
    eigen.SetMaxSteps(10000);
    int ok = eigen.Calc();
    if (ok == 0) {
      double minev = 0.0;
      for (int k = 1; k <= eigen.NumEigenValues(); k++)
	if (eigen.EigenValue(k) > 1e-10)
	  { minev = eigen.EigenValue(k); break; }
      if (gcomm.Rank() == 0) {
	cout << IM(1) << " Min Eigenvalue : " << minev << endl; 
	cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
	cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/minev << endl;
      }
    }
    printmessage_importance = i1;
    netgen::printmessage_importance = i2;
  }

  /** Options **/

  void BaseAMGPC::Options :: SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix)
  {

    auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

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

    SetEnumOpt(flags, mg_cycle, pfit("mg_cycle"), {"V", "W", "BS"}, { V_CYCLE, W_CYCLE, BS_CYCLE }, Options::MG_CYCLE::V_CYCLE);
    SetEnumOpt(flags, clev, pfit("clev"), {"inv", "sm", "none"}, { INV_CLEV, SMOOTH_CLEV, NO_CLEV }, Options::CLEVEL::INV_CLEV);
    SetEnumOpt(flags, cinv_type, pfit("cinv_type"), { "masterinverse", "mumps" }, Array<INVERSETYPE>({ MASTERINVERSE, MUMPS }));
    SetEnumOpt(flags, cinv_type_loc, pfit("cinv_type_loc"), { "pardiso", "pardisospd", "sparsecholesky", "superlu", "superlu_dist", "mumps", "umfpack" },
	       Array<INVERSETYPE>({ PARDISO, PARDISOSPD, SPARSECHOLESKY, SUPERLU, SUPERLU_DIST, MUMPS, UMFPACK }));

    sm_type.SetFromFlagsEnum(flags, prefix+"sm_type", { "gs", "bgs" }, { GS, BGS });
    // Array<string> sm_names ( { "gs", "bgs" } );
    // Array<Options::SM_TYPE> sm_types ( { Options::SM_TYPE::GS, Options::SM_TYPE::BGS } );
    // SetEnumOpt(flags, sm_type, pfit("sm_type"), { "gs", "bgs" }, Options::SM_TYPE::GS);
    // auto & spec_sms = flags.GetStringListFlag(prefix + "spec_sm_types");
    // spec_sm_types.SetSize(spec_sms.Size());
    // for (auto k : Range(spec_sms.Size()))
    //   { set_opt_sv(spec_sm_types[k], spec_sms[k], sm_names, sm_types); }

    gs_ver = Options::GS_VER(max(0, min(3, int(flags.GetNumFlag(prefix + "gs_ver", 3) - 1))));

    set_bool(sm_symm, "sm_symm");
    sm_steps = flags.GetNumFlag(prefix + "sm_steps", 1);
    set_bool(sm_mpi_overlap, "sm_mpi_overlap");
    set_bool(sm_mpi_thread, "sm_mpi_thread");
    set_bool(sm_shm, "sm_shm");
    set_bool(sm_sl2, "sm_sl2");

    set_bool(sync, "sync");
    set_bool(do_test, "do_test");
    set_bool(test_levels, "test_levels");
    set_bool(test_smoothers, "test_smoothers");
    set_bool(smooth_lo_only, "smooth_lo_only");
    set_bool(regularize_cmats, "regularize_cmats");
    set_bool(force_ass_flmat, "faflm");

    SetEnumOpt(flags, energy, pfit("energy"), { "triv", "alg", "elmat" }, { TRIV_ENERGY, ALG_ENERGY, ELMAT_ENERGY }, Options::ENERGY::ALG_ENERGY);

    SetEnumOpt(flags, log_level_pc, pfit("log_level_pc"), {"none", "basic", "normal", "extra"}, { NONE, BASIC, NORMAL, EXTRA }, Options::LOG_LEVEL_PC::NONE);
    set_bool(print_log_pc, "print_log_pc");
    log_file_pc = flags.GetStringFlag(prefix + string("log_file_pc"), "");

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
    O.SetFromFlags(bfa->GetFESpace(), flags, prefix);
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
      NgMPI_Comm c(MPI_COMM_WORLD, false);
      Array<int> me({ c.Rank() });
      NgMPI_Comm mecomm = (c.Size() == 1) ? c : c.SubCommunicator(me);
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

    Array<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels(1);
    amg_levels[0] = make_shared<BaseAMGFactory::AMGLevel>();
    InitFinestLevel(*amg_levels[0]);

    if (options->sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }

    factory->SetUpLevels(amg_levels, dof_map);

    /** Smoothers **/
    Array<shared_ptr<BaseSmoother>> smoothers = BuildSmoothers(amg_levels, dof_map);

    amg_mat = make_shared<AMGMatrix> (dof_map, smoothers);
    amg_mat->SetSTK(O.sm_steps);
    amg_mat->SetVWB(O.mg_cycle);

    auto gcomm = dof_map->GetParDofs()->GetCommunicator();

    if ( O.test_levels ) {
      for (int k = 1; k < amg_levels.Size() - 1; k++) {
	if (gcomm.Rank() == 0)
	  { cout << " test AMG-smoother from level " << k << endl; }
	shared_ptr<BaseMatrix> smwrap, smt;
	if (k == 0) { smt = finest_mat; smwrap = amg_mat; }
	else {
	  smwrap = make_shared<SmootherBM>(make_shared<AMGSmoother2>(amg_mat, k), false);
	  shared_ptr<BaseSparseMatrix> spmat;
	  if (k == 0) { /** This makes a difference with force_ass_flmat **/
	    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
	    spmat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat) : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());
	  }
	  else
	    { spmat = amg_levels[k]->cap->mat; }
	  smt = spmat;
	  if (amg_levels[k]->cap->pardofs)
	    { smt = make_shared<ParallelMatrix>(smt, amg_levels[k]->cap->pardofs, amg_levels[k]->cap->pardofs, C2D); }
	}
	DoTest(*smt, *smwrap, gcomm);
	if (gcomm.Rank() == 0)
	  { cout << " done testing AMG-smoother from level " << k << endl << endl; }
      }
    }

    /** Coarsest level inverse **/
    if (O.sync)
      { RegionTimer rt(tsync); dof_map->GetParDofs(0)->GetCommunicator().Barrier(); }
    

    if ( (amg_levels.Size() > 1) && (amg_levels.Last()->cap->mat != nullptr) ) { // otherwise, dropped out
      switch(O.clev) {
      case(Options::CLEVEL::INV_CLEV) : {
	static Timer t("CoarseInv"); RegionTimer rt(t);

	auto & c_lev = amg_levels.Last();
	auto cpds = dof_map->GetMappedParDofs();
	auto comm = cpds->GetCommunicator();
	auto cspm = c_lev->cap->mat;

	// auto cspmtm = dynamic_pointer_cast<SparseMatrixTM<Mat<6,6,double>>>(cspm);
	// cout << " Coarse mat: " << endl;
	// print_tm_spmat(cout, *cspmtm);

	if (O.regularize_cmats)
	  { RegularizeMatrix(cspm, cpds); }

	shared_ptr<BaseMatrix> coarse_inv = nullptr, coarse_mat = nullptr;
      
	if (GetEntryDim(cspm.get()) > MAX_SYS_DIM) // when would this ever happen??
	  { throw Exception("Cannot inv coarse level, MAX_SYS_DIM insufficient!"); }

	auto gcomm = dof_map->GetParDofs()->GetCommunicator();

	Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);

	if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
	  { cout << " invert coarsest level matrix " << endl; }

	if (comm.Size() > 2) {
	  auto parmat = make_shared<ParallelMatrix> (cspm, cpds, cpds, C2D);
	  parmat->SetInverseType(O.cinv_type);
	  coarse_inv = parmat->InverseMatrix();
	  coarse_mat = parmat;
	}
	else if ( (comm.Size() == 1) ||
		  ( (comm.Size() == 2) && (comm.Rank() == 1) ) ) { // local inverse
	  cspm->SetInverseType(O.cinv_type_loc);
	  auto cinv = cspm->InverseMatrix();
	  if (comm.Size() > 1)
	    { coarse_inv = make_shared<ParallelMatrix> (cinv, cpds, cpds, C2C); } // dummy parmat
	  else
	    { coarse_inv = cinv; }
	  coarse_mat = cspm;
	}
	else if (comm.Rank() == 0) { // some dummy matrix
	  Array<int> perow(0);
	  auto cinv = make_shared<SparseMatrix<double>>(perow);
	  if (comm.Size() > 1)
	    { coarse_inv = make_shared<ParallelMatrix> (cinv, cpds, cpds, C2C); } // dummy parmat
	  else
	    { coarse_inv = cinv; }
	  coarse_mat = cspm;
	}
	amg_mat->SetCoarseInv(coarse_inv, coarse_mat);

	if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
	  { cout << " coarsest level matrix inverted" << endl << endl; }

	break;
      }
      default : { break; }
      }
    }

    if ( O.test_levels ) {
      for (int k = 1; k < amg_levels.Size() - 1; k++) {
	shared_ptr<BaseMatrix> smwrap, smt;
	if (k == 0) { smt = finest_mat; smwrap = amg_mat; }
	else {
	  smwrap = make_shared<SmootherBM>(make_shared<AMGSmoother2>(amg_mat, k), false);
	  shared_ptr<BaseSparseMatrix> spmat;
	  if (k == 0) { /** This makes a difference with force_ass_flmat **/
	    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
	    spmat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat) : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());
	  }
	  else
	    { spmat = amg_levels[k]->cap->mat; }
	  smt = spmat;
	  if (amg_levels[k]->cap->pardofs)
	    { smt = make_shared<ParallelMatrix>(smt, amg_levels[k]->cap->pardofs, amg_levels[k]->cap->pardofs, C2D); }
	}
	if (gcomm.Rank() == 0)
	  { cout << " test AMG-smoother (+INV) from level " << k << endl; }
	DoTest(*smt, *smwrap, gcomm);
      }
    }

    if (options->do_test)
      {
	printmessage_importance = 1;
	netgen::printmessage_importance = 1;
	Test();
      }

  } // BaseAMGPC::BuildAMGMAt





  Array<shared_ptr<BaseSmoother>> BaseAMGPC :: BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels,
							       shared_ptr<DOFMap> dof_map)
  {
    auto & O(*options);

    auto gcomm = dof_map->GetParDofs()->GetCommunicator();

    Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);

    if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
      { cout << " set up smoothers " << endl; }

    for (int k = 0; k < amg_levels.Size() - 1; k++) {
      if ( (k > 0) && O.regularize_cmats) { // Regularize coarse level matrices
	if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NORMAL)
	  { cout << "  regularize matrix on level " << k << endl; }
	// if (k <= 1)
	// RegularizeMatrix(amg_levels[k]->cap->mat, amg_levels[k]->cap->pardofs);
      }
      if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NORMAL)
	{ cout << "  set up smoother on level " << k << endl; }
      smoothers[k] = BuildSmoother(*amg_levels[k]);
    }

    if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
      { cout << " smoothers built" << endl; }

    if ( O.test_smoothers ) {
      for (int k = 0; k < amg_levels.Size() - 1; k++) {
	shared_ptr<ParallelDofs> pardofs = (k == 0) ? finest_mat->GetParallelDofs() : amg_levels[k]->cap->pardofs;
	shared_ptr<BaseSparseMatrix> spmat;
	if (k == 0) { /** This makes a difference with force_ass_flmat **/
	  auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
	  spmat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat) : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());
	}
	else
	  { spmat = amg_levels[k]->cap->mat; }
	shared_ptr<BaseMatrix> smt = spmat;
	if (pardofs)
	  { smt = make_shared<ParallelMatrix>(smt, pardofs, pardofs, C2D); }
	auto smwrap = make_shared<SmootherBM>(smoothers[k]);
	if (gcomm.Rank() == 0)
	  { cout << " test smoother on level " << k << endl; }
	DoTest(*smt, *smwrap, gcomm);
      }
    }

    return smoothers;
  } // BaseAMGPC :: BuildSmoothers


  void BaseAMGPC :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {

    finest_level.level = 0;
    finest_level.cap = factory->AllocCap();

    finest_level.cap->mesh = BuildInitialMesh(); // TODO: get out of factory??

    finest_level.cap->eqc_h = finest_level.cap->mesh->GetEQCHierarchy();

    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
    finest_level.cap->mat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat)
      : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());

    finest_level.embed_map = BuildEmbedding(finest_level);

    if (finest_level.embed_map == nullptr)
      { finest_level.cap->pardofs = finest_mat->GetParallelDofs(); }
    else {
      /** Explicitely assemble matrix associated with the finest mesh. **/
      if (options->force_ass_flmat) {
	finest_level.cap->mat = finest_level.embed_map->AssembleMatrix(finest_level.cap->mat);
	finest_level.embed_done = true;
      }
      /** Either way, pardofs associated with the mesh are the mapped pardofs of the embed step **/
      finest_level.cap->pardofs = finest_level.embed_map->GetMappedParDofs();
    }
  } // BaseAMGPC::InitFinestLevel


  shared_ptr<BaseSmoother> BaseAMGPC :: BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level)
  {
    auto & O (*options);
    
    shared_ptr<BaseSmoother> smoother = nullptr;

    // cout << " smoother, mat " << amg_level.cap->mat->Height() << " x " << amg_level.cap->mat->Width() << endl;
    // cout << " pds " << amg_level.cap->pardofs->GetNDofLocal() << endl;

    Options::SM_TYPE sm_type = O.sm_type.GetOpt(amg_level.level);
    // if (O.spec_sm_types.Size() > amg_level.level)
    //   { sm_type = O.spec_sm_types[amg_level.level]; }

    shared_ptr<ParallelDofs> pardofs = (amg_level.level == 0) ? finest_mat->GetParallelDofs() : amg_level.cap->pardofs;
    shared_ptr<BaseSparseMatrix> spmat;
    if (amg_level.level == 0) { /** This makes a difference with force_ass_flmat **/
      auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
      spmat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat) : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());
    }
    else
      { spmat = amg_level.cap->mat; }

    /** bandaid fix. can happen if crsening is stuck and then we redistribute! **/
    if ( sm_type == Options::SM_TYPE::BGS ) {
      int no_cmp = (amg_level.crs_map == nullptr) ? 1 : 0;
      pardofs->GetCommunicator().AllReduce(no_cmp, MPI_SUM);
      if (no_cmp != 0)
	{ sm_type = Options::SM_TYPE::GS; }
    }

    switch(sm_type) {
    case(Options::SM_TYPE::GS)  : { smoother = BuildGSSmoother(spmat, pardofs, amg_level.cap->eqc_h, GetFreeDofs(amg_level)); break; }
    case(Options::SM_TYPE::BGS) : { smoother = BuildBGSSmoother(spmat, pardofs, amg_level.cap->eqc_h, move(GetGSBlocks(amg_level))); break; }
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
	     auto hgsm = make_shared<HybridGSS3<typename strip_mat<Mat<BS, BS, double>>::type>> (parmat, eqc_h, freedofs,
												 O.regularize_cmats, O.sm_mpi_overlap, O.sm_mpi_thread);
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
      { return amg_level.cap->free_nodes; }
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

    auto ma = fes->GetMeshAccess();
    auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

    BaseAMGPC::Options::SetFromFlags(fes, flags, prefix);

    SetEnumOpt(flags, subset, pfit("on_dofs"), {"range", "select"}, { RANGE_SUBSET, SELECTED_SUBSET });

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
      SetEnumOpt(flags, spec_ss, pfit("subset"), {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__",
	    "free", "nodalp2"}, { SPECSS_NONE, SPECSS_FREE, SPECSS_NODALP2 });
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

    SetEnumOpt(flags, topo, pfit("edges"), { "alg", "mesh", "elmat" }, { ALG_TOPO, MESH_TOPO, ELMAT_TOPO });
    SetEnumOpt(flags, v_pos, pfit("vpos"), { "vertex", "given" }, { VERTEX_POS, GIVEN_POS } );
  } // VertexAMGPCOptions::SetOptionsFromFlags

  /** END VertexAMGPCOptions **/

} // namespace amg
