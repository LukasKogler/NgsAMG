#ifndef FILE_AMG_SMOOTHER_HPP
#define FILE_AMG_SMOOTHER_HPP

#include "amg.hpp"

namespace amg {

  /** Base class for all smoothers to be used with any AMG preconditioner **/
  class BaseSmoother : public BaseMatrix
  {
  protected:
    shared_ptr<BaseMatrix> sysmat;

    void SetSysMat (shared_ptr<BaseMatrix> _sysmat) { sysmat = _sysmat; }

  public:
    BaseSmoother (shared_ptr<BaseMatrix> _sysmat, shared_ptr<ParallelDofs> par_dofs)
      : BaseMatrix(par_dofs), sysmat(_sysmat)
    { ; }

    BaseSmoother (shared_ptr<BaseMatrix> _sysmat)
      : BaseSmoother(_sysmat, _sysmat->GetParallelDofs())
    { ; }
    
    BaseSmoother (shared_ptr<ParallelDofs> par_dofs)
      : BaseSmoother(nullptr, par_dofs)
    { ; }

    virtual ~BaseSmoother(){}

    /**
       res_updated: is residuum up to date??
       update_res:  if true, updates the residuum while smoothing
       x_zero:      if true, assumes x is zero on input (can be used for optimization)
     **/
    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero = false) const = 0;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = true, bool x_zero = false) const = 0;

    virtual void SmoothK (int k, BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero = false) const
    {
      Smooth(x, b, res, res_updated, update_res, x_zero);
      for (auto j : Range(k-1))
	{ Smooth(x, b, res, update_res, update_res, false); }
    }

    virtual void SmoothBackK (int k, BaseVector  &x, const BaseVector &b,
			      BaseVector &res, bool res_updated = false,
			      bool update_res = true, bool x_zero = false) const
    {
      SmoothBack(x, b, res, res_updated, update_res, x_zero);
      for (auto j : Range(k-1))
	{ SmoothBack(x, b, res, update_res, update_res, false); }
    }

    virtual Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }
    // virtual Array<MemoryUsage> GetMemoryUsage() const override = 0;

    virtual void Finalize() { ; }

    virtual shared_ptr<BaseMatrix> GetAMatrix() const
    { return sysmat; }

    // return the underlying matrix
    // virtual shared_ptr<BaseMatrix> GetMatrix () const = 0;

    virtual AutoVector CreateRowVector () const override { return GetAMatrix()->CreateRowVector(); };
    virtual AutoVector CreateColVector () const override { return GetAMatrix()->CreateColVector(); };

    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
  };
  

  /** HiptMairSmoother **/

  class HiptMairSmoother : public BaseSmoother
  {
  protected:

    /** sm .. smoother in pre-image of G, smrange .. smoother in image of G **/
    shared_ptr<BaseSmoother> smpot, smrange;
    shared_ptr<BaseMatrix> Apot, Arange, D, DT;
    shared_ptr<BaseVector> solpot, respot, rhspot;

  public:

    HiptMairSmoother (shared_ptr<BaseSmoother> _smpot, shared_ptr<BaseSmoother> _smrange,
		      shared_ptr<BaseMatrix> _Apot, shared_ptr<BaseMatrix> _Arange,
		      shared_ptr<BaseMatrix> _D, shared_ptr<BaseMatrix> _DT)
      : BaseSmoother(_Arange), smpot(_smpot), smrange(_smrange), Apot(_Apot), Arange(_Arange), D(_D), DT(_DT)
    {
      solpot = smpot->CreateColVector();
      respot = smpot->CreateColVector();
      rhspot = smpot->CreateColVector();
    }

    ~HiptMairSmoother () { ; }

    virtual void Smooth (BaseVector  &x, const BaseVector &b,
			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero = false) const override;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = true, bool x_zero = false) const override;

    virtual int VHeight () const override { return Arange->Height(); }
    virtual int VWidth () const override { return Arange->Width(); }
    virtual AutoVector CreateVector () const override { return Arange->CreateColVector(); };
    virtual AutoVector CreateRowVector () const override { return Arange->CreateRowVector(); };
    virtual AutoVector CreateColVector () const override { return Arange->CreateColVector(); };

    virtual shared_ptr<BaseMatrix> GetAMatrix() const override { return Arange; }

    shared_ptr<BaseSmoother> GetSMPot () { return smpot; }
    shared_ptr<BaseSmoother> GetSMRange () { return smrange; }
    shared_ptr<BaseMatrix> GetAPot () { return Apot; }
    shared_ptr<BaseMatrix> GetARange () { return Arange; }
    shared_ptr<BaseMatrix> GetD () { return D; }
    shared_ptr<BaseMatrix> GetDT () { return DT; }

    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

  }; // class HiptMairSmoother

  /** END HiptMairSmoother **/


  /** 
      HybridGSS means (block-) Gauss-Seidel, applied for each subdomain-diagonal block
      of the matrix seperately.
      (The absolute value of) Off-diagonal entries get added to the diagonal blocks to
      prevent overshooting
  **/
  template<int BS>
  class HybridGSS : public BaseSmoother
  {
  public:
    using TSCAL = double;
    using TM = typename strip_mat<Mat<BS,BS,TSCAL>> :: type;
    using TV = typename strip_vec<Vec<BS,TSCAL>> :: type;
    using TSPMAT = stripped_spm<TM,TV,TV>;
    using TSPMAT_TM = stripped_spm_tm<TM>;
    HybridGSS ( const shared_ptr<TSPMAT> & amat,
		const shared_ptr<ParallelDofs> & apds,
		const shared_ptr<BitArray> & atake_dofs);
    ~HybridGSS();
    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = true,
    			 bool update_res = true, bool x_zero = true) const override;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = false, bool x_zero = false) const override;
    virtual Array<MemoryUsage> GetMemoryUsage() const override;
    void SetSymmetric (bool sym) { symmetric = sym; }

    // virtual shared_ptr<BaseMatrix> GetMatrix () const override { return spmat; }

    virtual int VHeight () const override { return A.Height(); }
    virtual int VWidth () const override { return A.Width(); }
    virtual AutoVector CreateVector () const override { return A.CreateVector(); };
    virtual AutoVector CreateRowVector () const override { return A.CreateRowVector(); };
    virtual AutoVector CreateColVector () const override { return A.CreateColVector(); };

  protected:

    bool symmetric = false;

    string name;
    shared_ptr<BitArray> free_dofs;
    shared_ptr<ParallelDofs> parallel_dofs;
    BitArray mf_exd, mf_dofs;
    size_t H;
    NgsAMG_Comm comm;

    shared_ptr<TSPMAT> spmat = nullptr;
    const TSPMAT& A;
    // additional diag-part of A
    TSPMAT* addA = nullptr;
    // part of the offdiag-part of A; "C_DL"; ndof X nex_nonmaster
    TSPMAT* CLD = nullptr; 
    mutable Array<TV> buffer; // [ex_p0, ex_p1, .....]
    // space to store "old" x-values. needed when we smooth without an updated residual
    mutable Array<TV> x_buffer;
    // space to store A_DDhat x_Dhat
    mutable Array<TV> ax_buffer;
    Array<int> buf_os; // buffer[buf_os[k]..buf_os[k+1]) for ex-proc k!
    mutable Array<int> buf_cnt;
    int nexp;
    int nexp_smaller; mutable Array<MPI_Request> rr_gather;
    int nexp_larger; mutable Array<MPI_Request> rr_scatter;
    mutable Array<MPI_Request> rsds;
    Array<TM> diag;

    void SetUpMat ();
    void CalcDiag ();

    void gather_vec (const BaseVector & vec) const;
    void scatter_vec (const BaseVector & vec) const;
    
    /**
       type:
         0 - FW
         1 - BW
         2 - FW/BW
         3 - BW/FW
     **/
    void smoothfull (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
		     bool res_updated = false, bool update_res = true, bool x_zero = false) const;
        
  public:
    virtual void Finalize () override { CalcDiag(); }
    
  }; // end class HybridSmoother

  /** 
      Regularizes the diagonal blocks.
      diag_block(RMIN..RMAX, RMIN..RMAX) is regularized
      diag_block(0..RMIN, 0..RMIN) is not touched
      Useful for elasticity-AMG performed on a displacement-only formulation.
  **/
  template<int BS, int RMIN, int RMAX>
  class StabHGSS : public HybridGSS<BS>
  {
  public:
    using TSPMAT = typename HybridGSS<BS>::TSPMAT;
    StabHGSS ( const shared_ptr<TSPMAT> & amat,
	       const shared_ptr<ParallelDofs> & apds,
	       const shared_ptr<BitArray> & atake_dofs)
      : HybridGSS<BS>(amat, apds, atake_dofs)
    { name = string("StabHGS<")+to_string(BS)+","+to_string(RMIN)+","+to_string(RMAX)+string(">"); }
  protected:
    using HybridGSS<BS>::spmat, HybridGSS<BS>::parallel_dofs,
      HybridGSS<BS>::H, HybridGSS<BS>::diag, HybridGSS<BS>::free_dofs,
      HybridGSS<BS>::name;
    virtual void Finalize () override { CalcRegDiag(); }
    void CalcRegDiag ();
  };
    
} // end namespace amg

#endif
