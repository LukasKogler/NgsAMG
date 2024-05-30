// #define FILE_SM_CPP

#include <utils_sparseLA.hpp>

#include "base_smoother.hpp"

namespace amg {

/** BaseSmoother **/

void BaseSmoother :: Mult (const BaseVector & b, BaseVector & x) const
{
  x = 0.0;
  MultAdd(1.0, b, x);
} // BaseSmoother :: Mult

void BaseSmoother :: MultTrans (const BaseVector & b, BaseVector & x) const
{
  x = 0.0;
  MultAdd(1.0, b, x);
} // BaseSmoother :: MultTrans

void BaseSmoother :: MultTransAdd (double scal, const BaseVector & b, BaseVector & x) const
{
  MultAdd(scal, b, x);
} // BaseSmoother :: MultTrans

void BaseSmoother :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  throw Exception("BaseSmoother :: MultAdd not overloaded!");
}

/** END BaseSmoother **/


/** ProxySmoother **/

size_t ProxySmoother :: GetNOps () const
{
  return nsteps * (symm ? 2 : 1) * sm->GetNOps();
} // ProxySmoother::GetNOps


size_t ProxySmoother :: GetANZE () const
{
  return sm->GetANZE();
} // ProxySmoother::GetANZE

/** END ProxySmoother **/


/** RichardsonSmoother **/

RichardsonSmoother :: RichardsonSmoother (shared_ptr<BaseMatrix> _mat, shared_ptr<BaseMatrix> _prec, double _omega)
  : BaseSmoother(_mat)
  , prec(_prec)
  , omega(_omega)
{ ; }


void RichardsonSmoother :: Smooth (BaseVector  &x, const BaseVector &b,
                                   BaseVector  &res, bool res_updated,
                                   bool update_res, bool x_zero) const
{
  if (!res_updated && x_zero)
    { x += omega * (*prec) * b; }
  else {
    if (!res_updated)
      { res = b - (*sysmat) * x; }
    x += omega * (*prec) * res;
  }
  if (update_res)
    { res = b - (*sysmat) * x; }
} // RichardsonSmoother::Smooth


void RichardsonSmoother :: SmoothBack (BaseVector  &x, const BaseVector &b,
                                       BaseVector &res, bool res_updated,
                                       bool update_res, bool x_zero) const
{
  Smooth(x, b, res, res_updated, update_res, x_zero);
} // RichardsonSmoother::SmoothBack

/** END RichardsonSmoother **/

/** JacobiSmoother **/

template<class TM>
JacobiSmoother<TM>::JacobiSmoother (shared_ptr<BaseMatrix> _mat,
                                shared_ptr<BitArray> _freedofs,
                                double _omega)
  : RichardsonSmoother(_mat,  nullptr, _omega)
  , diagInv(make_shared<DiagonalMatrix<TM>>(_mat->Height()))
  , freedofs(_freedofs)
{
  this->prec = diagInv;

  auto A = GetAMatrix();
  auto sparseA = GetLocalTMM<TM>(A);
  const auto & cSA(*sparseA);
  for (auto k : Range(A->Height()))
  {
    if ( (_freedofs == nullptr) || _freedofs->Test(k) )
    {
      TM d = cSA(k,k);
      CalcInverse(d);
      (*diagInv)(k) = d; // diagonal matrix (i) returns the TM-reference
    }
    else
    {
      (*diagInv)(k) = 0.0; // diagonal matrix (i) returns the TM-reference
    }
  }
}

/** END JacobiSmoother **/

/** HiptMairSmoother **/

HiptMairSmoother::
HiptMairSmoother (shared_ptr<BaseSmoother> _smpot, shared_ptr<BaseSmoother> _smrange,
                  shared_ptr<BaseMatrix> _Apot, shared_ptr<BaseMatrix> _Arange,
                  shared_ptr<BaseMatrix> _D, shared_ptr<BaseMatrix> _DT,
                  shared_ptr<BaseMatrix> _ARD, shared_ptr<BaseMatrix> _DTAR,
                  bool _potFirst)
  : BaseSmoother(_smrange->GetAMatrix())
  , potFirst(_potFirst)
  , smpot(_smpot)
  , smrange(_smrange)
  , Apot(_Apot)
  , D(_D)
  , DT(_DT)
  , AR_D(_ARD)
  , DT_AR( ( _DTAR == nullptr ) && ( _ARD != nullptr ) ? TransposeAGeneric(_ARD) : _DTAR )
{
  solpot = smpot->CreateColVector();
  respot = smpot->CreateColVector();
  rhspot = smpot->CreateColVector();
} // HiptMairSmoother(..)


HiptMairSmoother::
HiptMairSmoother (shared_ptr<BaseSmoother> _smpot, shared_ptr<BaseSmoother> _smrange,
                  shared_ptr<BaseMatrix> _Apot, shared_ptr<BaseMatrix> _Arange,
                  shared_ptr<BaseMatrix> _D, shared_ptr<BaseMatrix> _DT,
                  bool _adOpt, bool _potFirst)
  : HiptMairSmoother(_smpot, _smrange,
                     _Apot, _Arange,
                     _D, _DT,
                     _adOpt ? MatMultABGeneric(_Arange, _D) : nullptr, nullptr,
                     _potFirst )
{
  solpot = smpot->CreateColVector();
  respot = smpot->CreateColVector();
  rhspot = smpot->CreateColVector();
} // HiptMairSmoother(..)


bool
HiptMairSmoother::
CalcPotRhs (BaseVector &x, const BaseVector &b, BaseVector  &res, BaseVector &cRhs,
            bool res_updated, bool x_zero, bool update_res) const
{
  static Timer t("HiptMairSmoother::CalcPotRhs");
  RegionTimer rt(t);

  if (!res_updated)
  {
    if ( update_res || (DT_AR == nullptr) )
    {
      this->CalcResiduum(x, b, res, x_zero);
      res.Distribute();
      cRhs.SetParallelStatus(DISTRIBUTED);
      DT->Mult(res, cRhs);
      return true;
    }
    else
    {
      b.Distribute();
      cRhs.SetParallelStatus(DISTRIBUTED);
      DT->Mult(b, cRhs);
      DT_AR->MultAdd(-1.0, x, cRhs);
      return false;
    }
  }
  else
  {
    res.Distribute();
    cRhs.SetParallelStatus(DISTRIBUTED);
    DT->Mult(res, cRhs);
    return true;
  }
} // HiptMairSmoother::CalcPotRhs


bool
HiptMairSmoother::
ApplyPotUpdate (BaseVector &rangeX, BaseVector &potUpdate,
                const BaseVector &b, BaseVector  &res,
                bool res_updated, bool update_res) const
{
  static Timer t("HiptMairSmoother::ApplyPotUpdate");
  RegionTimer rt(t);

  potUpdate.Cumulate();
  rangeX.Cumulate();

  D->MultAdd(1.0, potUpdate, rangeX);

  if (update_res)
  {
    if ( (AR_D == nullptr) || (!res_updated) )
    {
      this->CalcResiduum(rangeX, b, res, false);
    }
    else // A@D stored in a seperate matrix should be faster!
    { // res was up-to-date before pot-smoothing!
      res.Distribute();
      res -= (*AR_D) * potUpdate;
    }
  }

  return update_res;
} // HiptMairSmoother::ApplyPotUpdate


void HiptMairSmoother :: Smooth (BaseVector &x, const BaseVector &b, BaseVector  &res,
          bool res_updated, bool update_res, bool x_zero) const
{
  static Timer t("HiptMairSmoother::Smooth");

  static Timer tr("HiptMairSmoother::Smooth - range");
  static Timer tp("HiptMairSmoother::Smooth - pot");
  // cout << " smrange " << typeid(*smrange).name() << endl;
  // cout << " smpot " << typeid(*smpot).name() << endl;
  // cout << " FW b " << endl << b << endl;
  // cout << " FW res " << endl << res << endl;
  // cout << " FW x " << endl << x << endl;
  // cout << " HMS FW " << endl;

  // cout << "HiptMairSmoother::Smooth, #rows " << GetAMatrix()->Height() << ", res " << res_updated << "->" << update_res << ", x " << x_zero << endl;

  RegionTimer rt(t);

  bool resUp = res_updated;
  bool xZero = x_zero;

  if (!potFirst)
  {
    RegionTimer rt(tr);

    smrange->Smooth(x, b, res, resUp, resUp, xZero);

    xZero = false;
  }

  resUp = CalcPotRhs(x, b, res, *rhspot, resUp, xZero, resUp);
  *solpot = 0;

  {
    RegionTimer rt(tp);
    smpot->Smooth(*solpot, *rhspot, *respot, false, false, true);
  }

  resUp = ApplyPotUpdate(x, *solpot, b, res, resUp, potFirst ? resUp : update_res);

  if (potFirst)
  {
    RegionTimer rt(tr);

    smrange->Smooth(x, b, res, resUp, update_res, false);
  }
}


void HiptMairSmoother :: SmoothBack (BaseVector &x, const BaseVector &b, BaseVector &res,
              bool res_updated, bool update_res, bool x_zero) const
{
  static Timer t("HiptMairSmoother::SmoothBack");
  static Timer tr("HiptMairSmoother::SmoothBack - range");
  static Timer tp("HiptMairSmoother::SmoothBack - pot");

  RegionTimer rt(t);

  // cout << "HiptMairSmoother::SmoothBack, #rows " << GetAMatrix()->Height() << ", res " << res_updated << "->" << update_res << ", x " << x_zero << endl;

  bool resUp = res_updated;
  bool xZero = x_zero;
  
  if (potFirst)
  {
    RegionTimer rt(tr);

    smrange->SmoothBack(x, b, res, resUp, resUp, x_zero);
  }

  resUp = CalcPotRhs(x, b, res, *rhspot, resUp, xZero, resUp);
  *solpot = 0;

  {
    RegionTimer rt(tp);
    smpot->SmoothBack(*solpot, *rhspot, *respot, false, false, true);
  }

  resUp = ApplyPotUpdate(x, *solpot, b, res, resUp, potFirst ? update_res : resUp);  

  if (!potFirst)
  {
    RegionTimer rt(tr);

    smrange->SmoothBack(x, b, res, resUp, update_res, false);
  }
}


void HiptMairSmoother :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  throw Exception("HiptMairSmoother::MultAdd not implemented (should be easy)");
} // HiptMairSmoother::MultAdd


void HiptMairSmoother :: PrintTo (ostream & os, string prefix) const
{
  std::string spaces(prefix.size(), ' ');

  os << prefix << "HiptMairSmoother,  dim range = " << GetAMatrix()->Height() << ", dim pot = " << Apot->Height() << endl;
  os << spaces << "  dim D  = " << D->Height() << " x " << D->Width() << endl;
  os << spaces << "  dim DT = " << DT->Height() << " x " << DT->Width() << endl;
  os << " D-mat:  " << endl << *D << endl;
  os << " DT-mat: " << endl << *DT << endl;
  smpot->PrintTo(os, "  potential space SM ");
  smrange->PrintTo(os, "  range space SM ");
} // HiptMairSmoother::PrintTo


BaseMatrix* local_mat (BaseMatrix * amat)
{
  auto parmat = dynamic_cast<ParallelMatrix*>(amat);
  if (parmat)
    { return parmat->GetMatrix().get(); }
  else
    { return amat; }
}

size_t HiptMairSmoother :: GetNOps () const
{
  // if (AR_D)
  //   { cout << " HPT OPS " << smpot->GetNOps() << " " << smrange->GetNOps() << " " << GetScalNZE(local_mat(D.get())) << " " << GetScalNZE(AR_D.get()) << endl; }
  // else
  //   { cout << " HPT OPS " << smpot->GetNOps() << " " << smrange->GetNOps() << " " << " 2 * " << GetScalNZE(local_mat(D.get())) << " " << smrange->GetANZE(); }
  if (AR_D)
    { return smpot->GetNOps() + smrange->GetNOps() + 2 * GetScalNZE(AR_D.get()); }
  else
    { return smpot->GetNOps() + smrange->GetNOps() + 2 * GetScalNZE(D.get()) + smrange->GetANZE(); }
} // HiptMairSmoother::GetNOps


size_t HiptMairSmoother :: GetANZE () const
{
  return smrange->GetANZE();
} // HiptMairSmoother :: GetANZE

/** END HiptMairSmoother **/


template class JacobiSmoother<double>;
template class JacobiSmoother<Mat<2,2,double>>;
template class JacobiSmoother<Mat<3,3,double>>;
#ifdef ELASTICITY
template class JacobiSmoother<Mat<6,6,double>>;
#endif

} // namespace amg