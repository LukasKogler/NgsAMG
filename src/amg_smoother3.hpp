#ifndef FILE_AMGSM3
#define FILE_AMGSM3

#include <condition_variable>

#include "amg_smoother.hpp"

namespace amg
{

  class BackgroundMPIThread;

  /** Sequential Gauss-Seidel. Can update residual during computation. **/
  template<class TM>
  class GSS3 : public BaseSmoother
  {
  protected:
    size_t H;
    shared_ptr<SparseMatrix<TM>> spmat;
    shared_ptr<BitArray> freedofs;
    Array<TM> dinv;
    size_t first_free, next_free;

  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    // using BS = mat_traits<TM>::HEIGHT;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    GSS3 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset = nullptr);

    /** take elements form repl_dinv as diagonal inverses instead of computing them from mat  **/
    GSS3 (shared_ptr<SparseMatrix<TM>> mat, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset = nullptr);

  protected:
    void SetUp (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset);

    virtual void SmoothRESInternal (size_t first, size_t next, BaseVector &x, BaseVector &res, bool backwards) const;
    virtual void SmoothRHSInternal (size_t first, size_t next, BaseVector &x, const BaseVector &b, bool backwards) const;

  public:
    // smooth with RHS
    virtual void Smooth (size_t first, size_t next, BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(first, next, x, b, false); }
    virtual void SmoothBack (size_t first, size_t next, BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(first, next, x, b, true); }

    // smooth with residual and keep it up do date
    virtual void SmoothRES (size_t first, size_t next, BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(first, next, x, res, false); }
    virtual void SmoothBackRES (size_t first, size_t next, BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(first, next, x, res, true); }

    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated,
    			 bool update_res, bool x_zero) const override
    {
      if (res_updated)
	{ SmoothRESInternal(size_t(0), H, x, res, false); }
      else
	{ SmoothRHSInternal(size_t(0), H, x, b, false); }
    } // GSS3::Smooth

    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated,
    			     bool update_res, bool x_zero) const override
    {
      if (res_updated)
	{ SmoothRESInternal(size_t(0), H, x, res, true); }
      else
	{ SmoothRHSInternal(size_t(0), H, x, b, true); }
    } // GSS3::SmoothBack

    shared_ptr<BitArray> GetFreeDofs () const { return freedofs; }

    virtual void CalcDiags ();

    virtual int VHeight () const override { return H; }
    virtual int VWidth () const override { return H; }
    virtual AutoVector CreateVector () const override
    { return make_unique<VVector<typename strip_vec<Vec<BS(), double>>::type>>(H); };
    virtual AutoVector CreateRowVector () const override { return CreateVector(); }
    virtual AutoVector CreateColVector () const override { return CreateVector(); }

  protected:
    void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
			 bool res_updated = false, bool update_res = true, bool x_zero = false) const;

  }; // class GSS3


  /** 
      Sequential Gauss-Seidel. Can update residual during computation.
      If add_diag is given, adds uses (D + add_D)^-1 instead if D^-1
      Compresses rows/cols from orig. sparse mat that it needs.
      Only use this for "small" blocks
  **/
  template<class TM>
  class GSS4
  {
  protected:
    Array<int> xdofs/*, resdofs*/;        // dofnrs we upadte, dofnrs we need (so also all neibs)
    shared_ptr<SparseMatrix<TM>> cA;  // compressed A
    Array<TM> dinv;

  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    GSS4 (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset = nullptr);

    GSS4 (shared_ptr<SparseMatrix<TM>> A, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset = nullptr);

  protected:

    void SetUp (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset);

    void CalcDiags ();

    template<class TLAM> INLINE void iterate_rows (TLAM lam, bool bw) const;
      
    virtual void SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const;
    virtual void SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const;

  public:
    // smooth with RHS
    INLINE void Smooth (BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(x, b, false); }
    INLINE void SmoothBack (BaseVector &x, const BaseVector &b) const
    { SmoothRHSInternal(x, b, true); }

    // smooth with residual and keep it up do date
    INLINE void SmoothRES (BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(x, res, false); }
    INLINE void SmoothBackRES (BaseVector &x, BaseVector &res) const
    { SmoothRESInternal(x, res, true); }

  }; // class GSS4


  /**
     Adds a third "valid" parallel status for parallel vectors:
       "CONCENTRATED":  DISTRIBUTED, but for each DOF a designated proc has the full value
                        and the others have zeros.
	     !!! in this context, the master of a DOF is not necessarily the lowest proc that shares it
	         (but it is always ONE of them) !!!
     Relevant transformations are:
        - DISTRIBUTED -> CONCENTRATED
        - CONCENTRATED -> CUMULATED 
   **/
  template<class TSCAL>
  class DCCMap
  {
  public:

    DCCMap (shared_ptr<EQCHierarchy> eqc_h, shared_ptr<ParallelDofs> _pardofs);

    ~DCCMap ();

    shared_ptr<BitArray> GetMasterDOFs () { return m_dofs; }

    /** buffer G vals, start M recv / G send, and zeros out G vec vals (-> reserve M/G buffers)**/
    void StartDIS2CO (BaseVector & vec);

    /** wait for M recv to finish, add M buf vals to vec, free M buffer  **/
    void ApplyDIS2CO (BaseVector & vec);

    /** wait for G send to finish, free G buffer, (if shortcut, free requests instead of waiting) */
    void FinishDIS2CO (bool shortcut);

    /** buffer M vals, start M send / G recv (-> reserve M/G buffers) **/
    void StartCO2CU (BaseVector & vec);
    
    /** wait for G recv to finish, replace G values, free G buffer **/
    void ApplyCO2CU (BaseVector & vec);

    /*** wait for M send to finish, free M buffer, (if shortcut, free requests instead of waiting) */
    void FinishCO2CU (bool shortcut);

    FlatArray<int> GetMDOFs (int kp) { return m_ex_dofs[kp]; }
    FlatArray<int> GetGDOFs (int kp) { return g_ex_dofs[kp]; }

    /** used for DIS2CO and CO2CU **/
    void WaitM ();
    void WaitG ();
    void WaitD2C ();

    /** used for DIS2CO **/
    void BufferG (BaseVector & vec);
    void ApplyM (BaseVector & vec);

    /** used for CO2CU **/
    void BufferM (BaseVector & vec);
    void ApplyG (BaseVector & vec);

  protected:

    /** Call in constructor to allocate MPI requests and buffers **/
    void AllocMPIStuff ();

    /** Overload and Call in constructor, decide who is master of which DOFs (constructs m_dofs, m_ex_dofs, g_ex_dofs) **/
    virtual void CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h) = 0;

    shared_ptr<ParallelDofs> pardofs;

    int block_size;

    shared_ptr<BitArray> m_dofs;

    Array<MPI_Request> m_reqs;
    Array<MPI_Request> m_send, m_recv;
    Table<int> m_ex_dofs;      // master ex-DOFs for each dist-proc (we are master of these)
    Table<TSCAL> m_buffer;    // buffer for master-DOF vals for each dist-proc

    Array<MPI_Request> g_reqs;
    Array<MPI_Request> g_send, g_recv;
    Table<int> g_ex_dofs;      // ghost ex-DOFs  for each dist-proc (they are master of these)
    Table<TSCAL> g_buffer;    // buffer for ghost-DOF vals  for each dist-proc

  }; // class DCCMap




  /**
     Splits eqch EQC into evenly sized chunks, one chunk goes to each proc.
     There is a minimum chunk size. If there are more procs than chunks for an EQC,
     randomly select ranks to assign them to.
  **/
  // template<class TSCAL>
  // class ChunkedDCCMap : public DCCMap<TSCAL>
  // {

  // };

  /** master of each DOF not necessarily lowest rank **/
  template<class TSCAL>
  class ChunkedDCCMap : public DCCMap<TSCAL>
  {
  public:
    ChunkedDCCMap (shared_ptr<EQCHierarchy> eqc_h, shared_ptr<ParallelDofs> _pardofs,
		   int _MIN_CHUNK_SIZE = 50);

  protected:

    using DCCMap<TSCAL>::pardofs;
    using DCCMap<TSCAL>::m_dofs;
    using DCCMap<TSCAL>::m_ex_dofs;
    using DCCMap<TSCAL>::g_ex_dofs;

    const int MIN_CHUNK_SIZE;

    virtual void CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h) override;
  };


  /**
     Splits a ParallelMatrix with a SparseMatrix inside in this way: (M .. master, G .. ghost)
        
         A         =       M       +         G
     
     A_MM  A_MG          M_MM  0           0    A_MG
                  ->               +   
     A_GM  A_GG           0    0          A_GM  G_GG

     m = A_MM + Adist_GG

     The MM-block of M is an actual diagonal block of the global matrix!
     
     G_GG has entries ij where master(i)!=master(j)
  **/
  template<class TM>
  class HybridMatrix2 : public BaseMatrix
  {
  public:
    using TSCAL = typename mat_traits<TM>::TSCAL;
    static constexpr int BS () { return mat_traits<TM>::HEIGHT; }
    using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

    HybridMatrix2 (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map);

    shared_ptr<SparseMatrix<TM>> GetM () { return M; }
    shared_ptr<SparseMatrix<TM>> GetG () { return G; }
    shared_ptr<DCCMap<TSCAL>> GetMap () { return dcc_map; }

    INLINE bool HasG () { return g_zero; } // local G can still be nullptr (e.g rank 0)

    /** BaseMatrix Overloads **/
    virtual bool IsComplex() const override { return is_same<double, TSCAL>::value ? false : true; }
    virtual int VHeight () const override { return M->Height(); }
    virtual int VWidth () const override { return M->Width(); }
    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
    virtual void Mult (const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
    virtual void MultTrans (const BaseVector & x, BaseVector & y) const override;

    virtual AutoVector CreateVector () const override;
    virtual AutoVector CreateRowVector () const override;
    virtual AutoVector CreateColVector () const override;

  protected:
    bool dummy;

    shared_ptr<ParallelDofs> pardofs;

    bool g_zero;
    shared_ptr<SparseMatrix<TM>> M, G;

    shared_ptr<DCCMap<TSCAL>> dcc_map;

    void SetUpMats (shared_ptr<SparseMatrix<TM>> A);
  }; // HybridMatrix2


  /**
     Any kind of hybrid Smoother, based on local smoothers for diagonal blocks of the global matrix.
   **/
  template<class TM>
  class HybridSmoother2 : public BaseSmoother
  {
  public:

    HybridSmoother2 (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h,
		     bool _overlap = false, bool _in_thread = false);

    virtual ~HybridSmoother2 ();

    virtual void Smooth (BaseVector  &x, const BaseVector &b,
    			 BaseVector  &res, bool res_updated = false,
    			 bool update_res = false, bool x_zero = false) const override;
    virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated = false,
    			     bool update_res = false, bool x_zero = false) const override;

    virtual Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }

    shared_ptr<BaseVector> Gx; // to stash M * x_old
    bool smooth_symmetric = false;

    void SetSymmetric (bool sym) { smooth_symmetric = sym; }

    virtual int VHeight () const override { return A->Height(); }
    virtual int VWidth () const override { return A->Width(); }

    virtual AutoVector CreateVector () const override { return A->CreateVector(); };
    virtual AutoVector CreateRowVector () const override { return A->CreateRowVector(); };
    virtual AutoVector CreateColVector () const override { return A->CreateColVector(); };

    // virtual shared_ptr<BaseMatrix> GetMatrix () const override { return A; }

  protected:
    bool overlap = false;
    bool in_thread = false;
    shared_ptr<BackgroundMPIThread> mpi_thread;

    // type: 0 - FW / 1 - BW / 2 - FW/BW / 3 - BW/FW
    virtual void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
				 bool res_updated, bool update_res, bool x_zero) const;

    /**
       Stages:
          0 ... smooth local part 1       (hide distributed -> concentrated)
          1 ... smooth exchange part
          2 ... smooth local part 2       (hide concentrated -> cumulated)
     **/

    // apply local smoothing operation with right hand side rhs
    virtual void SmoothLocal (int stage, BaseVector &x, const BaseVector &b) const = 0;
    virtual void SmoothBackLocal (int stage, BaseVector &x, const BaseVector &b) const = 0;
    // apply local smoothing, assuming that res is the residuum. update the residuum.
    virtual void SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const = 0;
    virtual void SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const = 0;

    shared_ptr<HybridMatrix2<TM>> A;
    shared_ptr<BaseMatrix> origA;

    virtual Array<TM> CalcModDiag (shared_ptr<BitArray> free);
  };


  template<class TM>
  class HybridGSS3 : public HybridSmoother2<TM>
  {
  public:
    using TV = typename strip_vec<Vec<mat_traits<TM>::HEIGHT,typename mat_traits<TM>::TSCAL>> :: type;

    HybridGSS3 (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h, shared_ptr<BitArray> _subset,
		bool _overlap, bool _in_thread);

    virtual void Finalize () override;

  protected:

    using HybridSmoother2<TM>::A;

    shared_ptr<BitArray> subset;

    size_t split_ind;

    shared_ptr<GSS3<TM>> jac_loc, jac_exo;
    shared_ptr<GSS4<TM>> jac_ex;

    virtual void SmoothLocal (int stage, BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothBackLocal (int stage, BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const override;
    virtual void SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const override;
  }; // HybridGSS3


  template<class TM, int RMIN, int RMAX>
  class RegHybridGSS3 : public HybridGSS3<TM>
  {
    using HybridGSS3<TM>::A;
  public:
    RegHybridGSS3 (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h, shared_ptr<BitArray> _subset,
		   bool _overlap, bool _in_thread);
  protected:
    virtual Array<TM> CalcModDiag (shared_ptr<BitArray> free) override;
  };

} // namespace amg

#endif
