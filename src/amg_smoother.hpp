#ifndef FILE_AMGSM
#define FILE_AMGSM

namespace amg {

  /** Base class for all smoothers to be used with any AMG preconditioner **/
  class BaseSmoother : public BaseMatrix
  {
  public:
    BaseSmoother (){}
    BaseSmoother (const shared_ptr<ParallelDofs> & par_dofs)
      : BaseMatrix(par_dofs) {}
    virtual ~BaseSmoother(){}
    /**
       res_updated: is residuum up to date??
       update_res:  if true, updates the residuum while smoothing
       x_zero:      if true, assumes x is zero on input (can be used for optimization)
     **/
    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero =false) const = 0;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = true, bool x_zero =false) const = 0;
    virtual string SType() const { return "base"; }
    // virtual void Mult (const BaseVector & x, BaseVector & y) const = 0;
    // virtual void MultTrans (const BaseVector & x, BaseVector & y) const = 0;
  };
  
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
    using TSPMAT = typename strip_spmat<TM,TV,TV> :: type;
    HybridGSS ( const shared_ptr<const TSPMAT> & amat,
		const shared_ptr<ParallelDofs> & apds,
		const shared_ptr<const BitArray> & atake_dofs);
    ~HybridGSS();
    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = true,
    			 bool update_res = true, bool x_zero = true) const override
    {
      // smoothfull(3, x, b, res, res_updated, update_res, x_zero);
      smoothfull(0, x, b, res, res_updated, update_res, x_zero);
    }
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = false, bool x_zero = false) const override
    {
      smoothfull(1, x, b, res, res_updated, update_res, x_zero);
      // smoothfull(3, x, b, res, res_updated, update_res, x_zero);
    }
    virtual string SType() const override { return name; }
  protected:
    string name;
    shared_ptr<const BitArray> free_dofs;
    shared_ptr<ParallelDofs> parallel_dofs;
    BitArray mf_exd, mf_dofs;
    size_t H;
    NgsAMG_Comm comm;

    shared_ptr<const TSPMAT> spmat = nullptr;
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
        
  }; // end class HybridSmoother


#ifndef FILE_AMGSM_CPP
  extern template class HybridGSS<1>;
  extern template class HybridGSS<2>;
#endif
  // template<> class HybridGSS<3>;
  // template<> class HybridGSS<6>;
  
} // end namespace amg

#endif
