#ifndef FILE_AMGSM2
#define FILE_AMGSM2

namespace amg
{

  /**
     Splits a ParallelMatrix with a SparseMatrix inside in this way: (M .. master, S .. slave)
        
         A         =       M       +         S
     
     A_MM  A_MS          m    0           0    A_MS
                  ->               +   
     A_SM  A_SS          0    0          A_SM   S_SS

     m = A_MM + Adist_SS

     The MM-block of M is an actual diagonal block of the global matrix!
     
     S_SS has entries ij where master(i)!=master(j)
  **/
  template<class TM>
  class HybridMatrix : public BaseMatrix
  {
  protected:
    bool dummy = false;
    shared_ptr<SparseMatrix<TM>> M; // the master-master block
    shared_ptr<SparseMatrix<TM>> sp_S; // master-slave and slave-master blocks
    shared_ptr<BaseMatrix> S; // master-slave and slave-master blocks
    shared_ptr<ParallelDofs> pardofs;
    
  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    // using BS = typename mat_traits<TM>::HEIGHT;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    // using TV = strip_vec<Vec<BS,TSCAL>> :: type;
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    HybridMatrix (shared_ptr<ParallelMatrix> parmat);

    HybridMatrix (shared_ptr<SparseMatrixTM<TM>> _M); // dummy constructor for actually local mats!

    INLINE shared_ptr<SparseMatrix<TM>> GetM () { return M; }
    INLINE shared_ptr<BaseMatrix> GetS () { return S; }
    INLINE shared_ptr<SparseMatrix<TM>> GetSPS () { return sp_S; }

    void gather_vec (const BaseVector & vec) const;
    void scatter_vec (const BaseVector & vec) const;

    virtual bool IsComplex() const override { return is_same<double, TSCAL>::value ? false : true; }

    virtual int VHeight () const override { return M->Height(); }
    virtual int VWidth () const override { return M->Width(); }
    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const override;

  protected:
    void SetUpMats (SparseMatrixTM<TM> & A);

    // for gather/scatter OPs
    NgsAMG_Comm comm;
    int nexp;
    int nexp_smaller; mutable Array<MPI_Request> rr_gather;
    int nexp_larger; mutable Array<MPI_Request> rr_scatter;
  }; // class HybridSmoother


  template<class TM>
  class HybridGSS2 : public BaseSmoother
  {
  public:
    HybridGSS2 (shared_ptr<BaseMatrix> _A, shared_ptr<BitArray> _subset);

    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = true, bool x_zero = false) const override;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = true, bool x_zero = false) const override;

    virtual Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }

    using TV = typename strip_vec<Vec<mat_traits<TM>::HEIGHT,typename mat_traits<TM>::TSCAL>> :: type;

    void SetSymmetric (bool sym) { smooth_symmetric = sym; }

    virtual Array<TM> CalcAdditionalDiag ();

  protected:
    shared_ptr<BaseMatrix> origA;
    shared_ptr<HybridMatrix<TM>> A;
    shared_ptr<JacobiPrecond<TM>> jac;
    shared_ptr<ParallelDofs> pardofs;
    NgsAMG_Comm comm;
    bool smooth_symmetric = false;

    /**
       type:
       0 - FW
       1 - BW
       2 - FW/BW
       3 - BW/FW
    **/
    void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
			 bool res_updated = false, bool update_res = true, bool x_zero = false) const;
			       
  };

} // namespace amg

#endif
