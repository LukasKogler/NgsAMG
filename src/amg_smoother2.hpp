#ifndef FILE_AMGSM2
#define FILE_AMGSM2

namespace amg
{

  /** 
      Sequential Gauss-Seidel. Can update residual during computation.
      If add_diag is given, adds uses (D + add_D)^-1 instead if D^-1
  **/
  template<class TM>
  class GSS2
  {
  protected:
    size_t H;
    shared_ptr<SparseMatrix<TM>> spmat;
    shared_ptr<BitArray> freedofs;
    Array<TM> dinv;

  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    // using BS = mat_traits<TM>::HEIGHT;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    GSS2 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset = nullptr,
	   FlatArray<TM> add_diag = FlatArray<TM>(0, nullptr));

  protected:
    virtual void SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const;
    virtual void SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const;

  public:
    // smooth with RHS
    virtual void Smooth (BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(x, b, false); }
    virtual void SmoothBack (BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(x, b, true); }

    // smooth with residual and keep it up do date
    virtual void SmoothRES (BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(x, res, false); }
    virtual void SmoothBackRES (BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(x, res, true); }

    shared_ptr<BitArray> GetFreeDofs () const { return freedofs; }

  protected:
    void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
			 bool res_updated = false, bool update_res = true, bool x_zero = false) const;

  };




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
    mutable bool scatter_done = true;
    shared_ptr<SparseMatrix<TM>> M; // the master-master block
    shared_ptr<BaseMatrix> S; // master-slave and slave-master blocks
    shared_ptr<ParallelDofs> pardofs;
    
  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    // using BS = typename mat_traits<TM>::HEIGHT;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    // using TV = strip_vec<Vec<BS,TSCAL>> :: type;
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    HybridMatrix (shared_ptr<BaseMatrix> mat);

    shared_ptr<SparseMatrix<TM>> GetM () { return M; }
    shared_ptr<BaseMatrix> GetS () { return S; }

    void gather_vec (const BaseVector & vec) const;
    void scatter_vec (const BaseVector & vec) const;
    void finish_scatter () const;
    
    virtual bool IsComplex() const override { return is_same<double, TSCAL>::value ? false : true; }

    virtual int VHeight () const override { return M->Height(); }
    virtual int VWidth () const override { return M->Width(); }
    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const override;

  protected:
    void SetUpMats (shared_ptr<SparseMatrix<TM>> A);

    // for gather/scatter OPs
    int nexp;
    int nexp_smaller; mutable Array<MPI_Request> rr_gather;
    int nexp_larger; mutable Array<MPI_Request> rr_scatter;
  }; // class HybridSmoother


  /**
     Any kind of hybrid Smoother, based on local smoothers for diagonal blocks of the global matrix.
   **/
  template<class TM>
  class HybridSmoother : public BaseSmoother
  {
  public:

    HybridSmoother (shared_ptr<BaseMatrix> _A, bool _csr = false);


    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = false, bool x_zero = false) const override;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = false, bool x_zero = false) const override;

    virtual Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }

    shared_ptr<BaseVector> Sx; // to stash C * x_old
    bool smooth_symmetric = false;
    bool can_smooth_res = false; // temporary hack for block smoothers (they cant update res)

    void SetSymmetric (bool sym) { smooth_symmetric = sym; }

    virtual int VHeight () const override { return A->Height(); }
    virtual int VWidth () const override { return A->Width(); }

    // virtual shared_ptr<BaseMatrix> GetMatrix () const override { return A; }

  protected:

    // type: 0 - FW / 1 - BW / 2 - FW/BW / 3 - BW/FW
    virtual void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
				 bool res_updated, bool update_res, bool x_zero) const;

    // apply local smoothing operation with right hand side rhs
    virtual void SmoothLocal (BaseVector &x, const BaseVector &b) const = 0;
    virtual void SmoothBackLocal (BaseVector &x, const BaseVector &b) const = 0;

    // apply local smoothing, assuming that res is the residuum. update the residuum.
    virtual void SmoothRESLocal (BaseVector &x, BaseVector &res) const
    { throw Exception("SmoothRESLocal not implemented"); }
    virtual void SmoothBackRESLocal (BaseVector &x, BaseVector &res) const
    { throw Exception("SmoothRESLocal not implemented"); }

    shared_ptr<HybridMatrix<TM>> A;

    Array<TM> CalcAdditionalDiag ();
  };


  /** l1-smoother from "Multigrid Smoothers for Ultraparallel Computing" **/
  template<class TM>
  class HybridGSS2 : public HybridSmoother<TM>
  {
  public:
    HybridGSS2 (shared_ptr<BaseMatrix> _A, shared_ptr<BitArray> _subset);

    using TV = typename strip_vec<Vec<mat_traits<TM>::HEIGHT,typename mat_traits<TM>::TSCAL>> :: type;


  protected:
    using HybridSmoother<TM>::A;

    shared_ptr<GSS2<TM>> jac;

    virtual void SmoothLocal (BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothBackLocal (BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothRESLocal (BaseVector &x, BaseVector &res) const override;
    virtual void SmoothBackRESLocal (BaseVector &x, BaseVector &res) const override;
  };


  template<class TM>
  class HybridBlockSmoother : public HybridSmoother<TM>
  {
  public:
    HybridBlockSmoother (shared_ptr<BaseMatrix> _A, shared_ptr<Table<int>> _blocktable);

  protected:
    using HybridSmoother<TM>::A;

    shared_ptr<BlockJacobiPrecond<TM, typename HybridMatrix<TM>::TV, typename HybridMatrix<TM>::TV>> jac;

    virtual void SmoothLocal (BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothBackLocal (BaseVector &x, const BaseVector &b) const override;

    // virtual void Smooth (BaseVector  &x, const BaseVector &b,
    // 			 BaseVector  &res, bool res_updated = false,
    // 			 bool update_res = false, bool x_zero = false) const override;
    // virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    // 			     BaseVector &res, bool res_updated = false,
    // 			     bool update_res = false, bool x_zero = false) const override;
  };


} // namespace amg

#endif
