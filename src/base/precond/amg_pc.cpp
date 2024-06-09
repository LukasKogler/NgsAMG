
#include "utils_sparseLA.hpp"
#include <base.hpp>
#include <utils.hpp>
#include <utils_io.hpp>

#define FILE_AMG_PC_CPP

#include "amg_pc.hpp"

#include <universal_dofs.hpp>

/** Need all smoother headers here, not just BaseSmoother! **/
#include <gssmoother.hpp>
#include <block_gssmoother.hpp>
#include <dyn_block.hpp>
#include <dyn_block_smoother.hpp>


/** Implementing VertexAMGPCOptions SetFromFlags here is easiest **/
#include "amg_pc_vertex.hpp"

namespace amg
{

extern template class HybridDISmoother<double>;
extern template class HybridDISmoother<Mat<2,2,double>>;
extern template class HybridDISmoother<Mat<3,3,double>>;
extern template class HybridDISmoother<Mat<6,6,double>>;

AutoVector CreatePDVector(shared_ptr<ParallelDofs> pardofs)
{
  return CreateSuitableSPVector(pardofs->GetNDofLocal(), pardofs->GetEntrySize(), pardofs, DISTRIBUTED);
}

bool CheckBAConsistency(string name, shared_ptr<BitArray> ba, shared_ptr<ParallelDofs> pds)
{
  if (!pds)
    { return true; }

  cout << " CHECK BA " << name << endl;

  auto v = CreatePDVector(pds);
  v.Distribute();
  auto fv = v.FVDouble();
  int es = pds->GetEntrySize();

  for (auto k : Range(ba->Size()))
    { fv(es*k) = ba->Test(k) ? 1 : 0; }
  v.Cumulate();

  bool is_ok = true;
  for (auto k : Range(ba->Size())) {
    int sum = fv(es*k);
    if (ba->Test(k)) {
      int corsum = 1 + pds->GetDistantProcs(k).Size();
      if (sum != corsum) {
        cout << " dof " << k << ", shared with "; prow(pds->GetDistantProcs(k)); cout << " SET, but sum = " << sum << " / " << corsum << endl;
        is_ok = false;
      }
    } else {
      if (sum > 0) {
        cout << " dof " << k << ", shared with "; prow(pds->GetDistantProcs(k)); cout << " NST, but sum = " << sum << " / " << 0 << endl;
        is_ok = false;
      }
    }
  }

  return is_ok;
}

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
  shared_ptr<BaseVector> res;
  bool sym;
public:
  // SmootherBM(shared_ptr<BaseSmoother> _sm, bool _sym = true) : sm(_sm), sym(_sym) { res.AssignPointer(sm->CreateColVector()); }
  SmootherBM(shared_ptr<BaseSmoother> _sm, bool _sym = true)
    : sm(_sm), sym(_sym)
  {
    res = _sm->CreateColVector();
  }

  ~SmootherBM()
  {
    cout << " SM BM DESTR" << endl;
    res = nullptr;
    cout << " SM BM DESTR II" << endl;
  }

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
    // sm->Smooth(x, b, r2, false, false, false);
    sm->SmoothBack(x, b, r2, false, false, false);
    }
    else
      { sm->Smooth(x, b, r2, false, false, false); }
  }
  virtual void MultTrans (const BaseVector & b, BaseVector & x) const override { Mult(b, x); }

  virtual int VHeight () const override { return sm->VHeight(); }
  virtual int VWidth () const override  { return sm->VWidth(); }
  // virtual AutoVector CreateVector () const override { return sm->CreateVector(); };
  virtual AutoVector CreateColVector () const override { return sm->CreateColVector(); };
  virtual AutoVector CreateRowVector () const override { return sm->CreateColVector(); };
};

void DoTestLAPACK (BaseMatrix &mat, BaseMatrix &pc, BitArray* free, NgMPI_Comm & gcomm, string message) {
  auto i1 = printmessage_importance;
  auto i2 = netgen::printmessage_importance;
  printmessage_importance = 1;
  netgen::printmessage_importance = 1;
  if ( (gcomm.Rank() == 0) && (message.size() > 0) )
    { cout << IM(1) << message << endl; }

  if (gcomm.Rank() == 0) {
    Matrix<Complex> dpc = MakeDense<Complex> (pc, free);
    Matrix<Complex> asy(dpc.Height());
    asy = dpc - Trans(dpc);
    ofstream out ("dpc_" + to_string(mat.Height()) + ".out");
    out << dpc << endl;
    ofstream asy_out ("asy_" + to_string(mat.Height()) + ".out");
    for (auto k : Range(asy.Height()))
      for (auto j : Range(asy.Height()))
        if ( abs(asy(k,j)) <= 1e-12 * sqrt(abs(dpc(k,k) * dpc(j,j))) )
          { asy(k,j) = 0.0; }
        else
          { asy_out << k << " " << j << " diff " << asy(k,j) << " (k,j) " << dpc(k,j) << ", (j,k) " << dpc(j,k) << endl; }
  }
  ProductMatrix pc_mat(pc, mat);
  Matrix<Complex> dmat = MakeDense<Complex>( pc_mat, free);

  // Matrix<double> ddmat = MakeDense<double>( pc_mat, free);
  // cout << " d dense mat " << endl << ddmat << endl;
  // Matrix<double> asy(ddmat.Height());
  // asy = ddmat - Trans(ddmat);
  // cout << " asy part " << endl << asy << endl;

  int n_elim = dmat.Height();
  Matrix<Complex> mat2(n_elim), ev(n_elim);
  mat2 = Complex(0.0);
  for (int i = 0; i < n_elim; i++)
    { mat2(i,i) = 1.0; }

  cout << "call lapack" << endl;
  Vector<Complex> lami(n_elim);
  LaEigNSSolve (n_elim, &dmat(0,0), &mat2(0,0), &lami(0), 1, &ev(0,0), 0, 'B');
  if (gcomm.Rank() == 0) {
    ofstream out ("eigenvalues.out");
    for (int i = 0; i < n_elim; i++)
      { out << lami(i).real() << " " << lami(i).imag() << "\n"; }
    cout << IM(1) << " Min Eigenvalue : " << lami(0) << endl;
    cout << IM(1) << " Max Eigenvalue : " << lami(n_elim - 1) << endl;
    cout << IM(1) << " Condition   " << lami(n_elim - 1)/lami(0) << endl;
  }

  printmessage_importance = i1;
  netgen::printmessage_importance = i2;
}

void
DoTest (BaseMatrix const &mat, BaseMatrix const &pc, NgMPI_Comm gcomm, string message)
{
  static Timer t("EVTest");
  RegionTimer rt(t);

  auto i1 = printmessage_importance;
  auto i2 = netgen::printmessage_importance;
  printmessage_importance = 1;
  netgen::printmessage_importance = 1;

  if ( (gcomm.Rank() == 0) && (message.size() > 0) )
    { cout << IM(1) << message << endl; }

  EigenSystem eigen(mat, pc); // need parallel mat
  eigen.SetPrecision(1e-12);
  // eigen.SetMaxSteps(10000);

  int ok = eigen.Calc();

  if (ok == 0) {
    double minev = 0.0; int nzero = 0;
    for (int k = 1; k <= eigen.NumEigenValues(); k++)
      if (eigen.EigenValue(k) > 5e-5)
        { minev = eigen.EigenValue(k); nzero = k-1; break; }
    // cout << " all evals " << endl;
    // for (int k = 1; k <= eigen.NumEigenValues(); k++)
    //   cout << eigen.EigenValue(k) << " ";
    // cout << endl;
    if (gcomm.Rank() == 0) {
      if (nzero > 0)
        { cout << " Detected " << nzero << " zero EigenValues " << endl; }
      cout << " Min Eigenvalue : " << minev << endl;
      cout << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl;
      cout << " Condition   " << eigen.MaxEigenValue()/minev << endl;
    }
  }
  else if (gcomm.Rank() == 0)
    { cout << " EigenSystem Calc failed " << endl; }

  printmessage_importance = i1;
  netgen::printmessage_importance = i2;
}

void
DoTest (BaseMatrix const &mat, BaseMatrix const &pc, std::string const &message = "")
{
  DoTest(mat, pc, MatToUniversalDofs(mat).GetCommunicator(), message);
}

void TestSmoother (shared_ptr<BaseMatrix> mt, shared_ptr<BaseSmoother> sm, NgMPI_Comm & gcomm, string message)
{
  SmootherBM sm_wrap(sm);
  auto v = mt->CreateColVector();
  DoTest(*mt, sm_wrap, gcomm, message);
}

void TestSmoother (shared_ptr<BaseSmoother> sm, NgMPI_Comm & gcomm, string message)
{
  TestSmoother(sm->GetAMatrix(), sm, gcomm, message);
}

void TestSmootherLAPACK (shared_ptr<BaseMatrix> mt, shared_ptr<BaseSmoother> sm, shared_ptr<BitArray> free, NgMPI_Comm & gcomm, string message)
{
  SmootherBM sm_wrap(sm);
  DoTestLAPACK(*mt, sm_wrap, free.get(), gcomm, message);
}

void TestAMGMat (shared_ptr<BaseMatrix> mt, shared_ptr<AMGMatrix> amg_mat, int start_lev, NgMPI_Comm & gcomm, string message)
{
  SmootherBM sm_wrap(make_shared<AMGSmoother2>(amg_mat, start_lev), false);

  DoTest(*mt, sm_wrap, gcomm, message);
}

void TestAMGMatLAPACK (shared_ptr<BaseMatrix> mt, shared_ptr<AMGMatrix> amg_mat, int start_lev, shared_ptr<BitArray> free, NgMPI_Comm & gcomm, string message)
{
  SmootherBM sm_wrap(make_shared<AMGSmoother2>(amg_mat, start_lev), false);
  DoTestLAPACK(*mt, sm_wrap, free.get(), gcomm, message);
}

void Test2LevelConstant (int const level,
                         shared_ptr<BaseMatrix> fMat,
                         shared_ptr<BaseMatrix> cMat,
                         NgMPI_Comm & gcomm,
                         shared_ptr<AMGMatrix> amg_mat)
{
  static Timer t("Test2LevelConstant"); RegionTimer rt(t);

  cout << " Test2LevelConstant " << level << endl;
  cout << " Test2LevelConstant " << fMat << " " << cMat << " " << amg_mat << endl;

  cMat->SetInverseType(SPARSECHOLESKY);

  auto cInv = cMat->InverseMatrix();

  std::string msg = "2-Level AMG-Test " + std::to_string(level) + " -> " + std::to_string(level + 1);

  // cout << " Test 2-level constant levels " << level << " -> " << level + 1 << endl;

  auto smallMap = make_shared<DOFMap>();
  smallMap->AddStep(amg_mat->GetMap()->GetStep(level));

  Array<shared_ptr<BaseSmoother>> singleSmoother(1);
  singleSmoother[0] = amg_mat->GetSmoother(level);

  auto small_amg_mat = make_shared<AMGMatrix>(smallMap, singleSmoother);
  small_amg_mat->SetCoarseInv(cInv, cMat);

  TestAMGMat(fMat, small_amg_mat, 0, gcomm, msg);
}

/** Options **/

void BaseAMGPC::Options :: SetFromFlags (shared_ptr<FESpace> fes, shared_ptr<BaseMatrix> finest_mat, const Flags & flags, string prefix)
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

  sm_type.SetFromFlagsEnum(flags,
                           prefix + "sm_type",
                           { "gs", "bgs", "jacobi", "hiptmair", "amg_smoother", "dyn_block_gs" },
                           { GS, BGS, JACOBI, HIPTMAIR, AMGSM, DYNBGS });

  // Array<string> sm_names ( { "gs", "bgs" } );
  // Array<Options::SM_TYPE> sm_types ( { Options::SM_TYPE::GS, Options::SM_TYPE::BGS } );
  // SetEnumOpt(flags, sm_type, pfit("sm_type"), { "gs", "bgs" }, Options::SM_TYPE::GS);
  // auto & spec_sms = flags.GetStringListFlag(prefix + "spec_sm_types");
  // spec_sm_types.SetSize(spec_sms.Size());
  // for (auto k : Range(spec_sms.Size()))
  //   { set_opt_sv(spec_sm_types[k], spec_sms[k], sm_names, sm_types); }

  sm_symm.SetFromFlags(flags, pfit("sm_symm"));
  sm_symm_loc.SetFromFlags(flags, pfit("sm_symm_loc"));
  sm_steps.SetFromFlags(flags, pfit("sm_steps"));
  sm_steps_loc.SetFromFlags(flags, pfit("sm_steps_loc"));
  set_bool(sm_NG_MPI_overlap, "sm_NG_MPI_overlap");
  set_bool(sm_NG_MPI_thread, "sm_NG_MPI_thread");
  set_bool(sm_shm, "sm_shm");
  set_bool(sm_sl2, "sm_sl2");

  set_bool(sync, "sync");
  set_bool(do_test, "do_test");
  set_bool(test_levels, "test_levels");
  test_2level.SetFromFlags(flags, pfit("test_2level"));
  set_bool(test_smoothers, "test_smoothers");
  set_bool(smooth_lo_only, "smooth_lo_only");
  set_bool(regularize_cmats, "regularize_cmats");
  set_bool(force_ass_flmat, "faflm");

  set_bool(smooth_after_emb, "smooth_after_emb");

  SetEnumOpt(flags, energy, pfit("energy"), { "triv", "alg", "elmat" }, { TRIV_ENERGY, ALG_ENERGY, ELMAT_ENERGY }, Options::ENERGY::ALG_ENERGY);

  SetEnumOpt(flags, log_level_pc, pfit("log_level_pc"), {"none", "basic", "normal", "extra", "debug"}, { NONE, BASIC, NORMAL, EXTRA, DBG }, Options::LOG_LEVEL_PC::NONE);
  set_bool(print_log_pc, "print_log_pc");
  log_file_pc = flags.GetStringFlag(prefix + string("log_file_pc"), "");

} // Options::SetFromFlags

/** END Options**/


/** BaseAMGPC **/

BaseAMGPC :: BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
  : Preconditioner(blf, flags, name)
  , options(opts)
  , bfa(blf)
  , strict_alg_mode(false)
{
  assert(blf != nullptr); // Need to supply a BLF!
} // BaseAMGPC(..)

BaseAMGPC :: BaseAMGPC (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts)
  : Preconditioner(shared_ptr<BilinearForm>(nullptr), const_cast<Flags&>(flags).SetFlag("not_register_for_auto_update", true), name)
  , options(opts)
  , bfa(nullptr)
  , strict_alg_mode(true)
{
  assert(A != nullptr); // Need to supply a matrix!
  // In normal mode initialize, we are guessing options that are not set explicitly
  // based on the blf/fespace/mesh. In strict algebraic mode, we need to do that based
  // only on the matrix where possible. So, we need to set it already in the constructor!
  finest_mat = A;
} // BaseAMGPC(..)


BaseAMGPC :: ~BaseAMGPC ()
{
  ;
} // ~BaseAMGPC


void BaseAMGPC :: InitLevel (shared_ptr<BitArray> freedofs)
{
  // make sure options are created
  InitializeOptions();

  // if (inStrictAlgMode())
  // {
  //   assert(finest_mat != nullptr); // matrix not set in InitLevel in strict algebraic mode!
  // }

  const auto & O(*options);

  if ( (!inStrictAlgMode()) && (bfa->UsesEliminateInternal() || O.smooth_lo_only) ) {
    auto fes = bfa->GetFESpace();
    auto lofes = fes->LowOrderFESpacePtr();
    finest_freedofs = make_shared<BitArray>(*freedofs);
    auto& ofd(*finest_freedofs);
    if (bfa->UsesEliminateInternal() ) { // clear freedofs on eliminated DOFs
      auto rmax = (O.smooth_lo_only && (lofes != nullptr) ) ? lofes->GetNDof() : freedofs->Size();
      for (auto k : Range(rmax)) {
        if (ofd.Test(k)) {
          COUPLING_TYPE ct = fes->GetDofCouplingType(k);
          if ((ct & CONDENSABLE_DOF) != 0)
            { ofd.Clear(k); }
        }
      }
    }
    if (O.smooth_lo_only && (lofes != nullptr) ) {
      // clear freedofs on all high-order DOFs
      for (auto k : Range(lofes->GetNDof(), freedofs->Size()))
        { ofd.Clear(k); }
    }
  }
  else
    { finest_freedofs = freedofs; }

  // CheckBAConsistency("finest_freedofs", freedofs, bfa->GetFESpace()->GetParallelDofs());

} // BaseAMGPC::InitLevel


void BaseAMGPC :: FinalizeLevel (const BaseMatrix * mat)
{
  shared_ptr<BaseMatrix> sp(const_cast<BaseMatrix*>(mat), NOOP_Deleter);

  this->FinalizeLevel(sp);
}

void BaseAMGPC :: FinalizeLevel (shared_ptr<BaseMatrix> mat)
{
  if ( mat != nullptr )
    { finest_mat = mat; }

  if ( ( finest_mat == nullptr ) && ( !inStrictAlgMode() ) )
  { finest_mat = bfa->GetMatrixPtr(); }

  if (finest_mat == nullptr)
  {
    throw Exception("BaseAMGPC::FinalizeLevel - BLF not assembled or no matrix given!");
  }

  Finalize();

  // NgMPI_Comm dc(GetAMGMatrix()->GetSmoother(0)->GetAMatrix()->GetParallelDofs()->GetCommunicator());
  // TestSmoother(GetAMGMatrix()->GetSmoother(0)->GetAMatrix(), const_pointer_cast<BaseSmoother>(GetAMGMatrix()->GetSmoother(0)),
  // 		 dc, string("\n extra FL test smoother with own mat"));

  // TestSmoother(finest_mat, const_pointer_cast<BaseSmoother>(GetAMGMatrix()->GetSmoother(0)),
  // 		 dc, string("\n extra FL test smoother with finest_mat"));
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


void BaseAMGPC :: InitializeOptions ()
{
  if (options == nullptr)
  {
    options = MakeOptionsFromFlags(GetFlags());
  }
}

shared_ptr<BaseAMGPC::Options> BaseAMGPC :: MakeOptionsFromFlags (const Flags & flags, string prefix)
{
  auto opts = NewOpts();
  auto & O(static_cast<Options&>(*opts));
  SetDefaultOptions(*opts);
  SetOptionsFromFlags(*opts, flags, prefix);
  ModifyOptions(*opts, flags, prefix);
  return opts;
} // BaseAMGPC::MakeOptionsFromFlags


void BaseAMGPC :: SetDefaultOptions (Options& O)
{
  if (inStrictAlgMode()) {
    // in strict algebraic mode, pure MPI is probably the more
    // common case than serial/hybrid, so turn it off by default
    O.sm_shm = false;
  }
  else {
    O.sm_shm = !bfa->GetFESpace()->IsParallel();
  }
} // BaseAMGPC::SetDefaultOptions


void BaseAMGPC :: SetOptionsFromFlags (Options& O, const Flags & flags, string prefix)
{
  O.SetFromFlags(inStrictAlgMode() ? nullptr : bfa->GetFESpace(), finest_mat, flags, prefix);
} //BaseAMGPC::SetOptionsFromFlags


void BaseAMGPC :: ModifyOptions (Options & O, const Flags & flags, string prefix)
{
  ;
} // BaseAMGPC::ModifyOptions


void BaseAMGPC :: Finalize ()
{

  if (options->sync) {
    if (auto pds = finest_mat->GetParallelDofs()) {
      static Timer t(string("Sync1")); RegionTimer rt(t);
      pds->GetCommunicator().Barrier();
    }
  }

  if ( ( !inStrictAlgMode() ) && ( finest_freedofs == nullptr ) )
    { finest_freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()); }

  /** Set dummy-ParallelDofs **/
  // shared_ptr<BaseMatrix> fine_spm = finest_mat;
  // if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
  //   { fine_spm = pmat->GetMatrix(); }
  // else {
  //   Array<int> perow (fine_spm->Height() ); perow = 0;
  //   Table<int> dps (perow);
  //   NgMPI_Comm c(NG_MPI_COMM_WORLD, false);
  //   Array<int> me({ c.Rank() });
  //   NgMPI_Comm mecomm = (c.Size() == 1) ? c : c.SubCommunicator(me);
  //   fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , std::move(dps), GetEntryDim(fine_spm.get()), false));
  // }

  BuildAMGMat();
} // BaseAMGPC::Finalize


void BaseAMGPC :: BuildAMGMat ()
{
  static Timer t    ("BaseAMGPC::BuildAMGMat");
  static Timer tSync("BaseAMGPC::BuildAMGMat - sync");

  RegionTimer rt(t);

  auto & O(*options);

  Array<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels(1);
  amg_levels[0] = make_shared<BaseAMGFactory::AMGLevel>();
  InitFinestLevel(*amg_levels[0]);

  auto gUDofs = amg_levels[0]->cap->uDofs;
  auto gcomm = gUDofs.GetCommunicator();

  auto syncUp = [&]()
  {
    if ( gUDofs.IsParallel() && O.sync)
    {
      RegionTimer rt(tSync);
      gcomm.Barrier();
    }
  };

  syncUp();

  auto dof_map = make_shared<DOFMap>();

  GetBaseFactory().SetUpLevels(amg_levels, dof_map);

  /** Smoothers **/
  Array<shared_ptr<BaseSmoother>> smoothers = BuildSmoothers(amg_levels, dof_map);

  if ( options->smooth_after_emb )
  {
    // TODO: improve logging for smooth_after_emb

    // add an extra dof-step in front, the embedding!
    dof_map->PrependStep(this->cachedEmbMap);
    // TODO: do something better here, fix up GS-blocks while I am at it

    auto smoother = BuildGSSmoother(finest_mat, finest_freedofs);
    bool symm = O.sm_symm.GetOpt(0);
    int nsteps = O.sm_steps.GetOpt(0);
    if ( symm || (nsteps > 1) )
      { smoother = make_shared<ProxySmoother>(smoother, nsteps, symm); }

    smoothers.Insert(0, smoother);

    // discard everything, just do 1 level
    // dof_map = make_shared<DOFMap>();
    // dof_map->AddStep(this->cachedEmbMap);
    // smoothers.SetSize0();
    // smoothers.Append(smoother);
    // amg_levels.SetSize(1);

    this->cachedEmbMap = nullptr;
  }

  dof_map->Finalize();

  amg_mat = make_shared<AMGMatrix> (dof_map, smoothers);
  amg_mat->SetVWB(O.mg_cycle);

  if ( O.test_levels )
  {
    for (int k = 0; k < amg_levels.Size() - 1; k++)
    {
      /** This makes a difference with force_ass_flmat **/
      shared_ptr<BaseMatrix> smt = (k == 0) ? finest_mat
                                            : WrapParallelMatrix(amg_levels[k]->cap->mat,
                                                                 amg_levels[k]->cap->uDofs,
                                                                 amg_levels[k]->cap->uDofs,
                                                                 PARALLEL_OP::C2D);
      TestAMGMat(smt, amg_mat, k, gcomm, string(" \n Test AMG-smoother (excl. inv) from level " + to_string(k)) );
    }
  }

  /** Coarsest level inverse **/
  syncUp();

  if ( (amg_levels.Size() > 1 || options->smooth_after_emb) && (amg_levels.Last()->cap->mat != nullptr))
  { // otherwise, dropped out
    switch(O.clev)
    {
      case(Options::CLEVEL::INV_CLEV):
      {
        auto [cMat, cInv] = CoarseLevelInv(*amg_levels.Last());

        cout << " invert cmat w. dim " << cMat->Height() << endl; 

        amg_mat->SetCoarseInv(cInv, cMat);

        if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::BASIC)
          { cout << " coarsest level matrix inverted" << endl << endl; }

        // if (coarse_inv) {
        //   cout << endl << endl << "COARSE INV : " << endl;
        //   cout << *coarse_inv << endl << endl;
        // }

        break;
      }
      default : { break; }
    }
  }

  if ( O.test_levels )
  {
    for (int k = amg_levels.Size() - 2; k >= 0 ; k--)
    {
      shared_ptr<BaseMatrix> smt = (k == 0) ? finest_mat
                                            : WrapParallelMatrix(amg_levels[k]->cap->mat,
                                                                 amg_levels[k]->cap->uDofs,
                                                                 amg_levels[k]->cap->uDofs,
                                                                 PARALLEL_OP::C2D);
      TestAMGMat(smt, amg_mat, k, gcomm, string("\n Test AMG-smoother (incl. inv) from level " + to_string(k)) );
    }
  }

  for (int k = 0; k < amg_levels.Size() - 1; k++)
  {
    if ( O.test_2level.GetOpt(k) )
    {
      shared_ptr<BaseMatrix> smt = (k == 0) ? finest_mat : amg_levels[k]->cap->mat;
      smt = WrapParallelMatrix(smt, amg_levels[k]->cap->uDofs, amg_levels[k]->cap->uDofs, PARALLEL_OP::C2D);

      auto csparse = my_dynamic_pointer_cast<BaseSparseMatrix>(amg_levels[k+1]->cap->mat, "BuildAMGMat - c sparse");

      shared_ptr<BaseMatrix> cmat = WrapParallelMatrix(csparse, amg_levels[k+1]->cap->uDofs, amg_levels[k+1]->cap->uDofs, PARALLEL_OP::C2D);

      auto pds = cmat->GetParallelDofs();

      if (O.regularize_cmats)
        { RegularizeMatrix(csparse, pds); }

      Test2LevelConstant(k, smt, cmat, gcomm, amg_mat);
    }
  }

  if (options->do_test)
  {
    // Preconditioner::Test prints to cout on all ranks
    DoTest(GetAMatrix(), GetMatrix(), "Test AMG");
  }

  if (O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
  {
    auto occs = amg_mat->GetOC();

    if (gcomm.Rank() == 0)
    {
      cout << endl << " actual OC ~ " << occs[0] << endl;
      cout << "   components: "; prow2(occs.Part(1)); cout << endl << endl;
    }
  }
} // BaseAMGPC::BuildAMGMAt


std::tuple<std::shared_ptr<SparseMat<3, 3>>,
           UniversalDofs,
           std::shared_ptr<BitArray>>
ConvertToBSThree(SparseMat<6, 6> const &A,
                 UniversalDofs const &origUD,
                 shared_ptr<BitArray> origF)
{
  int h = A.Height();
  int H = 2 * h;

  Array<int> perow(H);
  for (auto k : Range(h))
  {
    auto const nCols = A.GetRowIndices(k).Size();
    perow[2*k]   = 2 * nCols;
    perow[2*k+1] = 2 * nCols;
  }

  auto newA = make_shared<SparseMat<3,3>>(perow, H);

  for (auto k : Range(h))
  {
    auto oldRIs = A.GetRowIndices(k);
    auto ri0 = newA->GetRowIndices(2*k);
    auto ri1 = newA->GetRowIndices(2*k+1);

    for (auto j : Range(oldRIs))
    {
      ri0[2*j]   = 2*oldRIs[j];
      ri0[2*j+1] = 2*oldRIs[j]+1;
      ri1[2*j]   = 2*oldRIs[j];
      ri1[2*j+1] = 2*oldRIs[j]+1;
    }

    auto oldRVs = A.GetRowValues(k);
    auto rv0 = newA->GetRowValues(2*k);
    auto rv1 = newA->GetRowValues(2*k+1);

    for (auto j : Range(oldRIs))
    {
      Iterate<3>([&](auto ii) {
        Iterate<3>([&](auto jj)
        {
          rv0[2*j]   (ii, jj) = oldRVs[j](ii    , jj);
          rv0[2*j+1] (ii, jj) = oldRVs[j](ii    , 3 + jj);
          rv1[2*j]   (ii, jj) = oldRVs[j](3 + ii, jj);
          rv1[2*j+1] (ii, jj) = oldRVs[j](3 + ii, 3 + jj);
        });
      });
    }
  }

  // cout << " OLD MAT: " << endl;
  // print_tm_spmat(cout, A);
  // cout << " NEW MAT: " << endl;
  // print_tm_spmat(cout, *newA);

  shared_ptr<ParallelDofs> newPDs = nullptr;

  if ( origUD.GetParallelDofs() )
  {
    auto oldPDs = origUD.GetParallelDofs();

    TableCreator<int> createNewDPs(H);

    for(; !createNewDPs.Done(); createNewDPs++)
    {
      for (auto k : Range(h))
      {
        auto dps = oldPDs->GetDistantProcs(k);
        createNewDPs.Add(2*k, dps);
        createNewDPs.Add(2*k+1, dps);
      }
    }

    newPDs =  make_shared<ParallelDofs>(oldPDs->GetCommunicator(),
                                        createNewDPs.MoveTable(),
                                        3,
                                        false);
  }

  UniversalDofs newUD(newPDs, H, 3);

  shared_ptr<BitArray> newF = nullptr;

  if (origF)
  {
    newF = make_shared<BitArray>(2 * h);
    newF->Clear();

    for (auto k : Range(h))
    {
      if ( origF->Test(k) )
      {
        newF->SetBit(2*k);
        newF->SetBit(2*k+1);
      }
    }
  }

  return std::make_tuple(newA, newUD, newF);
}


std::tuple<std::shared_ptr<BaseMatrix>, shared_ptr<BaseMatrix>>
BaseAMGPC::
CoarseLevelInv(BaseAMGFactory::AMGLevel const &coarseLevel)
{
  auto & O(*options);

  static Timer t("BaseAMGPC::CoarseLevelInv");
  RegionTimer rt(t);

  auto uDofs = coarseLevel.cap->uDofs;
  auto gComm = uDofs.GetCommunicator();

  auto cspm = my_dynamic_pointer_cast<BaseSparseMatrix>(coarseLevel.cap->mat, "CoarseLevelInv - c sparse");

  auto coarseMat = WrapParallelMatrix(cspm, uDofs, uDofs, PARALLEL_OP::C2D);

  shared_ptr<BitArray> usedFree = coarseLevel.cap->free_nodes;

  if (O.regularize_cmats)
    { RegularizeMatrix(cspm, uDofs.GetParallelDofs()); }

  DispatchSquareMatrix(*cspm, [&](auto const &cA, auto CBS)
  {
    constexpr int BS = CBS;

    if ( BS > MAX_SYS_DIM )
    {
      if constexpr(BS == 6)
      {
        // support for 3d elasticity with NGSolve compiled with MAX_SYS_DIM=3,
        //   convert from block-size 6 to a block-size 3 matrix
        auto [A, ud, f] = ConvertToBSThree(cA, uDofs, usedFree);
        cspm     = A;
        uDofs    = ud;
        usedFree = f;
      }
      else
      {
        throw Exception("No inverse available for block-size = " + std::to_string(BS));
      }
    }
  });

  if ( cspm == nullptr )
  {
    return std::make_tuple(nullptr, nullptr);
  }

  // cout << " coarseLevel.cap->free_nodes = " << coarseLevel.cap->free_nodes << endl;
  // if ( coarseLevel.cap->free_nodes )
  // {
  //   cout << " coarseLevel.cap->free_nodes->Size() = " << coarseLevel.cap->free_nodes->Size() << endl;
  // }

  auto coarseMatForInv = WrapParallelMatrix(cspm, uDofs, uDofs, PARALLEL_OP::C2D);

  if (gComm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::BASIC)
    { cout << " invert coarsest level matrix " << endl; }

  shared_ptr<BaseMatrix> coarseInv = nullptr;

  if (uDofs.IsTrulyParallel())
  {
    // propper parallel inverse
    coarseMat->SetInverseType(O.cinv_type);
    coarseInv = coarseMatForInv->InverseMatrix(usedFree);
  }
  else {
    // local inverse
    if ( (gComm.Size() == 1) ||
         ( (gComm.Size() == 2) && (gComm.Rank() == 1) ) ) { // local inverse
      cspm->SetInverseType(O.cinv_type_loc);
      coarseInv = cspm->InverseMatrix(usedFree);
    }
    else if (gComm.Rank() == 0) { // some dummy matrix
      Array<int> perow(0);
      coarseInv = make_shared<SparseMatrix<double>>(perow);
    }
    coarseInv = WrapParallelMatrix(coarseInv, uDofs, uDofs, PARALLEL_OP::D2C);
  }

  // cout << " cinv done = " << coarseInv << endl;
  // std::ofstream of("ngs_amg_cmat_inv.out"); of << *coarseInv << endl;

  return std::make_tuple(coarseMat, coarseInv);
} // BaseAMGPC::CoarseLevelInv



Array<shared_ptr<BaseSmoother>> BaseAMGPC :: BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels,
                    shared_ptr<DOFMap> dof_map)
{
  auto & O(*options);

  // auto gcomm = dof_map->GetParDofs()->GetCommunicator();
  auto gcomm = dof_map->GetUDofs().GetCommunicator();

  Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);

  if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::BASIC)
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

  if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::BASIC)
    { cout << " smoothers built" << endl; }

  // if ( O.log_level_pc == Options::LOG_LEVEL_PC::DBG )
  // {
  //   for (int k = 0; k < amg_levels.Size() - 1; k++) {
  //     std::ofstream ofs("smoother_r_" + std::to_string(amg_levels[k]->cap->uDofs.GetCommunicator().Rank()) +
  //                               "_l_" + std::to_string(k) + ".out");
  //     smoothers[k]->PrintTo(ofs);
  //   }
  // }

  if ( O.test_smoothers ) {
    for (int k = 0; k < amg_levels.Size() - 1; k++) {
      // cout << " smt : " << endl << *smt << endl;
      // DoTest(*smt, *smoothers[k], gcomm, string("\n test (add) smoother on level " + to_string(k)));
      TestSmoother(smoothers[k], gcomm, string("\n test smoother on level " + to_string(k)));
      // TestSmoother(smoothers[k]->GetAMatrix(), smoothers[k], gcomm, string("\n test smoother with own mat on level " + to_string(k)));
    }
  }

  return smoothers;
} // BaseAMGPC :: BuildSmoothers


void BaseAMGPC :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
{

  finest_level.level = 0;
  finest_level.cap = GetBaseFactory().AllocCap();

  finest_level.cap->mesh = BuildInitialMesh(); // TODO: get out of factory??

  finest_level.cap->eqc_h = finest_level.cap->mesh->GetEQCHierarchy();

  // auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat);
  // finest_level.cap->mat = (fpm == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(finest_mat)
  //   : dynamic_pointer_cast<BaseSparseMatrix>(fpm->GetMatrix());

  auto fineMat = my_dynamic_pointer_cast<BaseSparseMatrix>(GetLocalMat(finest_mat),
                   "BaseAMGPC::InitFinestLevel");

  auto embMap = BuildEmbedding(finest_level);

  if ( embMap == nullptr )
  {
    // trivial case - no embedding !
    finest_level.cap->mat   = fineMat;
    finest_level.embed_map  = nullptr;
    finest_level.cap->uDofs = MatToUniversalDofs(*finest_mat, DOF_SPACE::ROWS);
  }
  else if ( options->smooth_after_emb ) // smooth twice on level 0, once before, once after embed
  {
    finest_level.cap->mat   = embMap->AssembleMatrix(fineMat);
    finest_level.embed_map  = nullptr;
    finest_level.cap->uDofs = embMap->GetMappedUDofs();

    this->cachedEmbMap = embMap;
  }
  else // smooth before embedding, concatenates embedding with first dof-coarse-map
  {
    finest_level.cap->mat  = fineMat;
    finest_level.embed_map = embMap;

    /** Explicitely assemble matrix associated with the finest mesh. **/
    if (options->force_ass_flmat) // TODO: is this still used anywhere?
    {
      finest_level.cap->mat = embMap->AssembleMatrix(finest_level.cap->mat);
      finest_level.embed_done = true;
    }
    /** Either way, pardofs associated with the mesh are the mapped pardofs of the embed step **/
    finest_level.cap->uDofs = embMap->GetMappedUDofs();
  }
} // BaseAMGPC::InitFinestLevel


shared_ptr<BaseSmoother>
BaseAMGPC::
BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level)
{
  static Timer t("BuildSmoother"); RegionTimer rt(t);

  auto & O (*options);

  shared_ptr<BaseSmoother> smoother = nullptr;

  // cout << " BaseAMGPC::smoother, level " << amg_level.level << ", mat " << amg_level.cap->mat->Height() << " x " << amg_level.cap->mat->Width() << endl;
  // cout << " pds " << amg_level.cap->pardofs->GetNDofLocal() << endl;

  Options::SM_TYPE sm_type = SelectSmoother(amg_level);
  // if (O.spec_sm_types.Size() > amg_level.level)
  //   { sm_type = O.spec_sm_types[amg_level.level]; }

  shared_ptr<BaseMatrix> mat = (amg_level.level == 0 && !options->smooth_after_emb) ? finest_mat
                                                      : WrapParallelMatrix(amg_level.cap->mat,
                                                                           amg_level.cap->uDofs,
                                                                           amg_level.cap->uDofs,
                                                                           PARALLEL_OP::C2D);

  switch(sm_type) {
    case(Options::SM_TYPE::GS):     { smoother = BuildGSSmoother(mat, GetFreeDofs(amg_level)); break; }
    case(Options::SM_TYPE::BGS):    { smoother = BuildBGSSmoother(mat, std::move(GetGSBlocks(amg_level))); break; }
    case(Options::SM_TYPE::JACOBI): { smoother = BuildJacobiSmoother(mat, GetFreeDofs(amg_level)); break; }
    case(Options::SM_TYPE::DYNBGS): { smoother = BuildJacobiSmoother(mat, GetFreeDofs(amg_level)); break; }
    default:                        { throw Exception("Invalid Smoother type!"); break; }
  }

  bool symm = O.sm_symm.GetOpt(amg_level.level);
  int nsteps = O.sm_steps.GetOpt(amg_level.level);
  if ( symm || (nsteps > 1) )
    { smoother = make_shared<ProxySmoother>(smoother, nsteps, symm); }

  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG )
  {
    std::string fName = "amg_smoother_rk_" + std::to_string(amg_level.cap->uDofs.GetCommunicator().Rank()) +
                                    "_l_" + std::to_string(amg_level.level) + ".out";
    std::ofstream ofs(fName);
    ofs << *smoother << endl;
  }

  return smoother;
} // BaseAMGPC::BuildSmoother


shared_ptr<BaseSmoother>
BaseAMGPC::
BuildGSSmoother (shared_ptr<BaseMatrix> A,
                 shared_ptr<BitArray>   freedofs)
{
  auto const &O(*options);

  shared_ptr<BaseSmoother> smoother = nullptr;

  auto [rRowUD, colUD, locA, opType] = UnwrapParallelMatrix(A);

  // structured binding reference capturing
  UniversalDofs const &rowUD = rRowUD;

  DispatchSquareMatrix(locA, [&](auto spA, auto BS)
  {
    if constexpr(!isSmootherSupported<BS>())
    {
      throw Exception("Smoother for that dim is not compiled!!");
    }
    else
    {
      using BSTM = typename strip_mat<Mat<BS, BS, double>>::type;

      auto parDOFs = rowUD.GetParallelDofs();

      if (parDOFs != nullptr)
      {
        smoother = make_shared<HybridGSSmoother<BSTM>>
          (A, freedofs, O.regularize_cmats, O.sm_NG_MPI_overlap, O.sm_NG_MPI_thread);
      }
      else
      {
        shared_ptr<SparseMatrix<BSTM>> ptr = spA;
        smoother = make_shared<GSS3<BSTM>>(ptr, freedofs, O.regularize_cmats);
      }
    }
  });

  smoother->Finalize();

  return smoother;
} // BaseAMGPC::BuildGSSmoother


shared_ptr<BaseSmoother>
BaseAMGPC::
BuildBGSSmoother (shared_ptr<BaseMatrix> A,
                  Table<int> && _blocks)
{
  auto const &O(*options);

  shared_ptr<BaseSmoother> smoother = nullptr;

  auto [rRowUD, colUD, locA, opType] = UnwrapParallelMatrix(A);

  // structured binding reference capturing
  UniversalDofs const &rowUD = rRowUD;

  auto blocks = std::move(_blocks);

  // cout << " BGSS w. blocks " << blocks.Size() << " blocks for " << A->Height() << " DOFs " << endl;

  DispatchSquareMatrix(locA, [&](auto spA, auto BS)
  {
    if constexpr(!isSmootherSupported<BS>())
    {
      throw Exception("Smoother for that dim is not compiled!!");
    }
    else
    {
      using BSTM = typename strip_mat<Mat<BS, BS, double>>::type;

      // turn off optimization for non-overlapping blocks for now
      bool blocks_no_overlap = false;

      bool use_bs2 = true;

      if (rowUD.IsParallel())
      {
        bool smooth_symm_loc = false;
        int nsteps_loc = 1;

        smoother = make_shared<HybridBS<BSTM>>
            (A, std::move(blocks), O.regularize_cmats, O.sm_NG_MPI_overlap, O.sm_NG_MPI_thread,
             O.sm_shm, O.sm_sl2, use_bs2, blocks_no_overlap, smooth_symm_loc, nsteps_loc);
      }
      else
      {
        // Note: I am not sure Bsmoother2 works for elasticity that can be singular
        // SM-parallel for BSmoother2 was never implemented!
        // bool const shm = false; // O.sm_shm
        if ( use_bs2 )
        {
          smoother = make_shared<BSmoother2<BSTM>>
              (spA, std::move(blocks), O.sm_shm, O.sm_sl2, O.regularize_cmats, blocks_no_overlap);
        }
        else
        {
          smoother = make_shared<BSmoother<BSTM>>
              (spA, std::move(blocks), O.sm_shm, O.sm_sl2, O.regularize_cmats);
        }
      }
      return;
    }
  });

  smoother->Finalize();

  return smoother;
} // BaseAMGPC::BuildBGSSmoother


shared_ptr<BaseSmoother>
BaseAMGPC::
BuildJacobiSmoother (shared_ptr<BaseMatrix> A,
                     shared_ptr<BitArray>   freedofs)
{
  auto const &O(*options);

  shared_ptr<BaseSmoother> smoother = nullptr;

  auto [rowUD, colUD, locA, opType] = UnwrapParallelMatrix(A);

  if (rowUD.IsParallel())
  {
    throw Exception("BuildJacobiSmoother parallel TODO correctly!!");
  }

  DispatchSquareMatrix(locA, [&](auto spA, auto BS)
  {
#if MAX_SYS_DIM < 6 // missing DiagonalMatrix in NGSolve
    if constexpr(BS == 6)
    {
      throw Exception("Jacobi for block-size 6 not supported - increase NGSolve MAX_SYS_DIM to 6!!");
    }
    else
    {
#endif
    if constexpr(!isSmootherSupported<BS>())
    {
      throw Exception("Smoother for that dim is not compiled!!");
    }
    else
    {
      using BSTM = typename strip_mat<Mat<BS, BS, double>>::type;

      smoother = make_shared<JacobiSmoother<BSTM>>(spA, freedofs);
    }
#if MAX_SYS_DIM < 6 // missing DiagonalMatrix in NGSolve
    }
#endif
  });

  smoother->Finalize();

  return smoother;
} // BaseAMGPC::BuildJacobiSmoother


shared_ptr<BaseSmoother>
BaseAMGPC::
BuildDynamicBlockGSSmoother (shared_ptr<BaseMatrix> A,
                             shared_ptr<BitArray> freedofs)
{
  shared_ptr<BaseSmoother> sm = nullptr;

  if (A == nullptr)
  {
    throw Exception("BuildDynamicBlockGSSmoother - no matrix!");
  }

  if (MatToUniversalDofs(*A).IsParallel())
  {
    int    const numLocSteps  = true;
    bool   const commInThread = true;
    bool   const overlapComm  = true;
    double const uRel         = 1.0;

    sm = make_shared<HybridDynBlockSmoother<double>>(A,
                                                     freedofs,
                                                     numLocSteps,
                                                     commInThread,
                                                     overlapComm,
                                                     uRel);
  }
  else
  {
    auto spA = my_dynamic_pointer_cast<SparseMatrix<double>>(A, "BaseAMGPC::BuildDynamicBlockGSSmoother - matrix");

    auto dynSPA = make_shared<DynBlockSparseMatrix<double>>(*spA);

    sm = make_shared<DynBlockSmoother<double>>(dynSPA, freedofs);
  }

  return sm;
} // BaseAMGPC::BuildDynamicBlockGSSmoother


shared_ptr<BitArray> BaseAMGPC :: GetFreeDofs (const BaseAMGFactory::AMGLevel & amg_level)
{
  if (amg_level.level == 0) {
    // if (finest_freedofs) {
// cout << " DIRI FFDS: " << endl;
// for (auto k :Range(finest_freedofs->Size()))
  // if (!finest_freedofs->Test(k))
    // cout << k << " ";
// cout << endl;
    // }
  }
  if (amg_level.level == 0 && !options->smooth_after_emb)
    { return finest_freedofs; }
  else
    { return amg_level.cap->free_nodes; }
} // BaseAMGPC::GetFreeDofs


Table<int> BaseAMGPC :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
{
  throw Exception("BaseAMGPC::GetGSBlocks not overloaded!");
  return std::move(Table<int>());
} // BaseAMGPC::GetGSBlocks


void BaseAMGPC :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
{
  ;
}

/** END BaseAMGPC **/


/** VertexAMGPCOptions **/

void VertexAMGPCOptions :: SetFromFlags (shared_ptr<FESpace> fes, shared_ptr<BaseMatrix> finest_mat, const Flags & flags, string prefix)
{
  bool strict_alg_mode = (fes == nullptr);

  shared_ptr<MeshAccess> ma = strict_alg_mode ? nullptr : fes->GetMeshAccess();

  auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

  BaseAMGPC::Options::SetFromFlags(fes, finest_mat, flags, prefix);

  SetEnumOpt(flags, subset, pfit("on_dofs"), {"range", "select"}, { RANGE_SUBSET, SELECTED_SUBSET });

  switch (subset) {
    case (RANGE_SUBSET) : {

      auto &low = flags.GetNumListFlag(pfit("lower"));
      auto &up = flags.GetNumListFlag(pfit("upper"));

      size_t lowi = flags.GetNumFlag(pfit("lower"), -1);
      size_t upi = flags.GetNumFlag(pfit("upper"), -1);

      // multiple ranges explicitly given
      if (low.Size()) { // multiple ranges given by user
        if (low.Size() != up.Size())
        {
          throw Exception("Given Lower/Upper ranges do not map!");
        }

        ss_ranges.SetSize(low.Size());
        for (auto k : Range(low.Size()))
          { ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }

        if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
        {
          cout << IM(3) << "subset for coarsening defined by user range(s)" << endl;
          cout << IM(5) << ss_ranges << endl;
        }
        break;
      }

      // single range explicitly given
      if ( (lowi != size_t(-1)) && (upi != size_t(-1)) ) {
        ss_ranges.SetSize(1);
        ss_ranges[0] = { lowi, upi };
        if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
        {
          cout << IM(3) << "subset for coarsening defined by (single) user range" << endl;
          cout << IM(5) << ss_ranges << endl;
        }
        break;
      }

      // range not specified
      if (strict_alg_mode) // take everything
      {
        ss_ranges.SetSize(1);
        ss_ranges[0][0] = 0;
        // In strict algebraic mode, the finest matrix is alreadt set in the constructor!
        ss_ranges[0][1] = finest_mat->Height();
        if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
          { cout << IM(3) << "subset for coarsening is ALL DOFs!" << endl; }
      }
      else { // make a best guess based on the blf/fespace
        auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
        if (flags.GetDefineFlagX(pfit("lo")).IsFalse()) {
          // high-order AMG, subset is ALL dofs of FES
          // I think the only current use case is a 3d H1 space with order 2 and nodalp2
          //   (Note: for order>2 3d H1 with nodalp2, we use SPECSS_NODAL_P2)
          if (fes->GetMeshAccess()->GetDimension() == 2)
            if (!flags.GetDefineFlagX(pfit("force_nolo")).IsTrue())
              { throw Exception("lo = False probably does not make sense in 2D! (set force_nolo to True to override this!)"); }
          has_node_dofs[NT_EDGE] = true;
          ss_ranges.SetSize(1);
          ss_ranges[0][0] = 0;
          ss_ranges[0][1] = fes->GetNDof();
          if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
            { cout << IM(3) << "subset for coarsening is ALL DOFs!" << endl; }
        }
        else {
          // coarsen on the low-order DOFs of the FESpace
          auto get_lo_nd = [](auto & fes) LAMBDA_INLINE {
            if (auto lofes = fes->LowOrderFESpacePtr()) // some spaces do not have a lo-space!
              { return lofes->GetNDof(); }
            else
              { return fes->GetNDof(); }
          };
          std::function<void(shared_ptr<FESpace>, size_t, Array<IVec<2,size_t>> &)> set_lo_ranges =
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
              Array<IVec<2,size_t>> oranges; // original ranges - not taking reorder into account
              set_lo_ranges(base_space, 0, oranges);
              size_t orange_sum = 0;
              for (auto & r : oranges)
                { orange_sum += (r[1] - r[0]); }
              IVec<2, size_t> r = { offset, offset + orange_sum };
              ranges.Append(r);
              // set_lo_ranges(base_space, offset, ranges);
            }
            else {
              IVec<2, size_t> r = { offset, offset + get_lo_nd(afes) };
              ranges.Append(r);
              // ss_ranges.Append( { offset, offset + get_lo_nd(afes) } ); // for some reason does not work ??
            }
          };
          ss_ranges.SetSize(0);
          set_lo_ranges(fes, 0, ss_ranges);

          if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
          {
            cout << IM(3) << "subset for coarsening defined by low-order range(s)" << endl;
            for (auto r : ss_ranges)
              { cout << IM(5) << r[0] << " " << r[1] << endl; }
          }
        }
      } // make a best guess based on the blf/fespace
      break;
    } // RANGE_SUBSET
    case (SELECTED_SUBSET) : {
      SetEnumOpt(flags, spec_ss, pfit("subset"), {"__DO_NOT_SET_THIS_FROM_FLAGS_PLEASE_I_DO_NOT_THINK_THAT_IS_A_GOOD_IDEA__", "free", "nodalp2"},
                                                  { SPECSS_NONE, SPECSS_FREE, SPECSS_NODALP2 });
      if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
        { cout << IM(3) << "subset for coarsening defined by bitarray" << endl; }
      // NONE - set somewhere else. FREE - set in initlevel
      switch(spec_ss) {
        case(SPECSS_NONE): {
          // nothing to do - this must come from somewhere else
          break;
        }
        case(SPECSS_FREE): {
          // nothing to do - the subset will come from InitLevel
          break;
        }
        case(SPECSS_NODALP2): {
          if (strict_alg_mode) {
            throw Exception("nodalp2 subset not available in strict algebraic mode!");
            break;
          }
          if (ma->GetDimension() == 2) {
            if (log_level_pc > Options::LOG_LEVEL_PC::NORMAL)
              { cout << IM(4) << "In 2D nodalp2 does nothing, using default lo base functions!" << endl; }
            subset = RANGE_SUBSET;
            ss_ranges.SetSize(1);
            ss_ranges[0][0] = 0;
            if (auto lospace = fes->LowOrderFESpacePtr()) // e.g compound has no LO space
              { ss_ranges[0][1] = lospace->GetNDof(); }
            else
              { ss_ranges[0][1] = fes->GetNDof(); }
          }
          else {
            if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
              { cout << IM(3) << "taking nodalp2 subset for coarsening" << endl; }
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
          if (log_level_pc > Options::LOG_LEVEL_PC::BASIC)
            { cout << IM(3) << "nodalp2 set: " << ss_select->NumSet() << " of " << ss_select->Size() << endl; }
        }

      }
      break;
    }
  }

  SetEnumOpt(flags, topo, pfit("topology"), { "alg", "mesh", "elmat" }, { ALG_TOPO, MESH_TOPO, ELMAT_TOPO });
  SetEnumOpt(flags, v_pos, pfit("vpos"), { "vertex", "given" }, { VERTEX_POS, GIVEN_POS } );

  SetEnumOpt(flags, dof_ordering, pfit("dof_ordering"), {"regular", "p2Emb"}, { REGULAR_ORDERING, P2_ORDERING});

  calc_elmat_evs = flags.GetDefineFlagX(pfit("calc_elmat_evs")).IsTrue();
  aux_elmat_version = max(0.0, min(2.0, flags.GetNumFlag(pfit("aux_elmat_version"), 0.0)));

} // VertexAMGPCOptions::SetOptionsFromFlags

/** END VertexAMGPCOptions **/

} // namespace amg
