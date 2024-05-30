#ifndef FILE_BASE_SMOOTHER_HPP
#define FILE_BASE_SMOOTHER_HPP

#include <base.hpp>

#include <utils_sparseLA.hpp>

namespace amg
{


enum SMOOTHING_DIRECTION : char
{
  FORWARD  = 0,
  BACKWARD = 1
};

INLINE ostream& operator << (ostream &os, SMOOTHING_DIRECTION const &dir)
{
  switch(dir)
  {
    case(FORWARD)  : { os << "FORWARD";  break; }
    case(BACKWARD) : { os << "BACKWARD"; break; }
  }
  return os;
}


template<SMOOTHING_DIRECTION DIR>
constexpr SMOOTHING_DIRECTION REVERSE()
{
  if constexpr(DIR == FORWARD)
  {
    return BACKWARD;
  }
  else
  {
    return FORWARD;
  }
}

/** BaseSmoother: Base class for all smoothers to be used with any AMG preconditioner **/
class BaseSmoother : public BaseMatrix
{
protected:
  shared_ptr<BaseMatrix> sysmat;

  void SetSysMat (shared_ptr<BaseMatrix> _sysmat) { sysmat = _sysmat; }

public:
  BaseSmoother (shared_ptr<BaseMatrix> _sysmat, shared_ptr<ParallelDofs> par_dofs)
    : BaseMatrix(par_dofs)
    , sysmat(_sysmat)
  { ; }

  BaseSmoother (shared_ptr<BaseMatrix> _sysmat)
    : BaseMatrix(_sysmat->GetParallelDofs())
    , sysmat(_sysmat)
  { ; }

  BaseSmoother (shared_ptr<ParallelDofs> par_dofs)
    : BaseMatrix(par_dofs)
    , sysmat(nullptr)
  { ; }

  virtual ~BaseSmoother() = default;

  /**
     res_updated: is residuum up to date??
      update_res:  if true, residuum will be up to date on return
      x_zero:      if true, assumes x is zero on input (can be used for optimization)
    **/
  virtual void Smooth (BaseVector &x, const BaseVector &b,
                       BaseVector &res, bool res_updated = false,
                       bool update_res = true, bool x_zero = false) const = 0;
  virtual void SmoothBack (BaseVector &x, const BaseVector &b,
                           BaseVector &res, bool res_updated = false,
                           bool update_res = true, bool x_zero = false) const = 0;
  virtual void SmoothSymm (BaseVector &x, const BaseVector &b,
                           BaseVector &res, bool res_updated = false,
                           bool update_res = true, bool x_zero = false) const
  {
    Smooth    (x, b, res, res_updated, update_res, x_zero);
    SmoothBack(x, b, res, update_res, update_res, false);
  }

  virtual void SmoothK (int k, BaseVector &x, const BaseVector &b,
          BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const
  {
    Smooth(x, b, res, res_updated, update_res, x_zero);
    for (auto j : Range(k-1))
      { Smooth(x, b, res, update_res, update_res, false); }
  }

  virtual void SmoothBackK (int k, BaseVector &x, const BaseVector &b,
          BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const
  {
    SmoothBack(x, b, res, res_updated, update_res, x_zero);
    for (auto j : Range(k-1))
      { SmoothBack(x, b, res, update_res, update_res, false); }
  }

  virtual void SmoothSymmK (int k, BaseVector &x, const BaseVector &b,
          BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const
  {
    SmoothSymm(x, b, res, res_updated, update_res, x_zero);
    for (auto j : Range(k-1))
      { SmoothSymm(x, b, res, update_res, update_res, false); }
  }

  virtual Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }
  // virtual Array<MemoryUsage> GetMemoryUsage() const override = 0;

  virtual void Finalize() { ; }

  // return the underlying matrix (if exists)
  virtual shared_ptr<BaseMatrix> GetAMatrix() const { return sysmat; }

  virtual AutoVector CreateRowVector () const override { return GetAMatrix()->CreateRowVector(); };
  virtual AutoVector CreateColVector () const override { return GetAMatrix()->CreateColVector(); };
  virtual int VHeight () const override { return sysmat->Height(); }
  virtual int VWidth () const override { return sysmat->Width(); }

  virtual void Mult (const BaseVector & b, BaseVector & x) const override;
  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
  virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
  virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;

  virtual void CalcResiduum(BaseVector const &x,
                            BaseVector const &b,
                            BaseVector       &res,
                            bool       const &xZero) const
  {
    res = b;
    if (!xZero)
    {
      res -= (*GetAMatrix()) * x;
    }
  }

  // #ops for one "Smooth" operation
  virtual size_t GetNOps () const
  { return GetScalNZE(GetAMatrix().get()); }

  // NZE of system matrix (no NZE in basematrix...)
  virtual size_t GetANZE () const
  { return GetScalNZE(GetAMatrix().get()); }

  virtual void PrintTo (ostream & os, string prefix = "") const
  {
    os << prefix << "BaseSmoother::PrintTo not overloaded!" << endl;
  }
}; // class BaseSmoother

INLINE ostream & operator << (ostream &os, BaseSmoother & sm)
{
  sm.PrintTo(os, "");
  return os;
}

/** END BaseSmoother **/


/** ProxySmoother: acts like nsteps (symmetric or not)steps of another smoother **/

class ProxySmoother : public BaseSmoother
{
protected:
  shared_ptr<BaseSmoother> sm;
  int nsteps;
  bool symm;
public:
  ProxySmoother (shared_ptr<BaseSmoother> _sm, int _nsteps, bool _symm)
    : BaseSmoother(_sm->GetAMatrix()), sm(_sm), nsteps(_nsteps), symm(_symm)
  { ; }
  ~ProxySmoother () = default;
  virtual void Smooth (BaseVector &x, const BaseVector &b,
          BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const override
  {
    if (symm)
      { sm->SmoothSymmK(nsteps, x, b, res, res_updated, update_res, x_zero); }
    else
      { sm->SmoothK(nsteps, x, b, res, res_updated, update_res, x_zero); }
  }
  virtual void SmoothBack (BaseVector &x, const BaseVector &b,
              BaseVector &res, bool res_updated = false,
              bool update_res = true, bool x_zero = false) const override
  {
    if (symm)
      { sm->SmoothSymmK(nsteps, x, b, res, res_updated, update_res, x_zero); }
    else
      { sm->SmoothBackK(nsteps, x, b, res, res_updated, update_res, x_zero); }
  }
  virtual Array<MemoryUsage> GetMemoryUsage() const override
  { return sm->GetMemoryUsage(); }
  virtual void Finalize () override { sm->Finalize(); }
  virtual shared_ptr<BaseMatrix> GetAMatrix () const override { return sm->GetAMatrix(); }
  shared_ptr<BaseSmoother> GetSmoother () const { return sm; }
  // virtual shared_ptr<BaseMatrix> GetMatrix () const = 0;
  virtual AutoVector CreateRowVector () const override
  { return sm->GetAMatrix()->CreateRowVector(); };
  virtual AutoVector CreateColVector () const override
  { return sm->GetAMatrix()->CreateColVector(); };
  virtual int VHeight () const override
  { return sm->Height(); }
  virtual int VWidth () const override
  { return sm->Width(); }
  virtual void Mult (const BaseVector & b, BaseVector & x) const override
  { sm->Mult(b, x); }
  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override
  { sm->MultAdd(s, b, x); }
  virtual void MultTrans (const BaseVector & b, BaseVector & x) const override
  { sm->MultTrans(b, x); }
  virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override
  { sm->MultTransAdd(s, b, x); }
  virtual void PrintTo (ostream & os, string prefix = "") const override
  {
    os << prefix << "ProxySmoother, symm = " << symm << ", nsteps = " << nsteps << endl;
    os << prefix << "  inner smoother = " << endl;
    sm->PrintTo(os, prefix + "   ");
  }

  virtual size_t GetNOps () const override;
  virtual size_t GetANZE () const override;
}; // class ProxySmoother

template<class S>
INLINE shared_ptr<const S> unwrap_smoother (shared_ptr<const BaseSmoother> sm)
{
  shared_ptr<const S> sp = dynamic_pointer_cast<const S>(sm);

  if ( (sp == nullptr) && (sm != nullptr) )
  {
    if (auto proxy = dynamic_pointer_cast<ProxySmoother const>(sm))
    {
      return dynamic_pointer_cast<S>(proxy->GetSmoother());
    }
  }
  return sp;
}

/** END ProxySmoother **/


/** RichardsonSmoother **/

class RichardsonSmoother : public BaseSmoother
{
public:
  RichardsonSmoother (shared_ptr<BaseMatrix> _mat, shared_ptr<BaseMatrix> _prec, double _omega = 1.0);
  virtual ~RichardsonSmoother () = default;
  virtual void Smooth (BaseVector &x, const BaseVector &b,
          BaseVector &res, bool res_updated,
          bool update_res, bool x_zero) const override;
  virtual void SmoothBack (BaseVector &x, const BaseVector &b,
              BaseVector &res, bool res_updated,
              bool update_res, bool x_zero) const override;
  virtual int VHeight () const override { return sysmat->Height(); }
  virtual int VWidth () const override { return sysmat->Width(); }
  void setOmega (double _omega) { omega = _omega; }
protected:
  shared_ptr<BaseMatrix> prec;
  double omega;
}; // class RichardsonSmoother

/** END RichardsonSmoother **/

/** JacobiSmoother **/

template<class TMAT>
class JacobiSmoother : public RichardsonSmoother
{
public:
  using TM = TMAT;
  JacobiSmoother (shared_ptr<BaseMatrix> _mat, shared_ptr<BitArray> _freedofs, double _omega = 0.9);
  virtual ~JacobiSmoother () = default;
protected:
  shared_ptr<DiagonalMatrix<TMAT>> diagInv;
  shared_ptr<BitArray> freedofs;
};

/** END JacobiSmoother **/

  /** HiptMairSmoother **/

class HiptMairSmoother : public BaseSmoother
{
protected:

  /** sm .. smoother in pre-image of G, smrange .. smoother in image of G **/
  bool                     potFirst;
  shared_ptr<BaseSmoother> smpot, smrange;
  shared_ptr<BaseMatrix> Apot, D, DT, AR_D, DT_AR;
  shared_ptr<BaseVector> solpot, respot, rhspot;

  bool
  CalcPotRhs (BaseVector &x, const BaseVector &b, BaseVector  &res, BaseVector &cRhs,
              bool res_updated, bool x_zero, bool update_res) const;

  bool
  ApplyPotUpdate (BaseVector &rangeX, BaseVector &potUpdate,
                  const BaseVector &b, BaseVector  &res,
                  bool res_updated, bool update_res) const;
                 
public:

  HiptMairSmoother (shared_ptr<BaseSmoother> _smpot, shared_ptr<BaseSmoother> _smrange,
                    shared_ptr<BaseMatrix> _Apot, shared_ptr<BaseMatrix> _Arange,
                    shared_ptr<BaseMatrix> _D, shared_ptr<BaseMatrix> _DT,
                    shared_ptr<BaseMatrix> _ARD, shared_ptr<BaseMatrix> _DTAR,
                    bool _potFirst = true);

  HiptMairSmoother (shared_ptr<BaseSmoother> _smpot, shared_ptr<BaseSmoother> _smrange,
        shared_ptr<BaseMatrix> _Apot, shared_ptr<BaseMatrix> _Arange,
        shared_ptr<BaseMatrix> _D, shared_ptr<BaseMatrix> _DT,
        bool _adOpt = false, bool _potFirst = true);

  virtual ~HiptMairSmoother () = default;

  virtual void Smooth (BaseVector &x, const BaseVector &b,
      BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const override;
  virtual void SmoothBack (BaseVector &x, const BaseVector &b,
              BaseVector &res, bool res_updated = false,
              bool update_res = true, bool x_zero = false) const override;

  shared_ptr<BaseSmoother> GetSMPot () { return smpot; }
  shared_ptr<BaseSmoother> GetSMRange () { return smrange; }
  shared_ptr<BaseMatrix> GetAPot () { return Apot; }
  shared_ptr<BaseMatrix> GetD () { return D; }
  shared_ptr<BaseMatrix> GetDT () { return DT; }

  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

  virtual void PrintTo (ostream & os, string prefix = "") const override;

  virtual size_t GetNOps () const override;

  virtual size_t GetANZE () const override;

}; // class HiptMairSmoother

/** END HiptMairSmoother **/

} // namespace amg

#endif // FILE_BASE_SMOOTHER_HPP
