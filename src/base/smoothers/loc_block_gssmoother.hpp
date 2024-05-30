#ifndef FILE_AMG_BLOCKSMOOTHER_LOC_HPP
#define FILE_AMG_BLOCKSMOOTHER_LOC_HPP

#include "base_smoother.hpp"
// #include "block_gssmoother.hpp"

namespace amg
{

  template<class TM>
  class BSmoother2 : public BaseSmoother
  {
  public:
    static_assert(ngbla::Height<TM>() == ngbla::Width<TM>(), "BSmoother2 cannot do non-square entries (i think)!");

    using TSCAL = typename mat_traits<TM>::TSCAL;
    using TV = typename strip_vec<Vec<ngbla::Height<TM>(),TSCAL>>::type;

    class BSBlock
    {
    public:
      bool LU = false, md = false;
      FlatArray<int> dofnrs;
      FlatArray<int> firsti; // double length if L and U stored seperately!
      FlatArray<int> cols;
      FlatMatrix<TM> diag, diag_inv;
      FlatArray<TM> vals, mdadd;
      // FlatArray<int> unique_rows; // for forward operator (alloc full array)

      BSBlock() {}

      ~BSBlock() = default;

      INLINE tuple<int, int> GetAllocSize (const SparseMatrixTM<TM> & A, FlatArray<int> block_dofs, bool _LU, bool _md);
      INLINE void Alloc (int*& ptr_i, TM*& ptr_v);
      INLINE void Alloc (LocalHeap & lh);
      INLINE void SetFromSPMat (const SparseMatrixTM<TM> & A, FlatArray<int> dofs, LocalHeap & lh, bool pinv,
				FlatArray<TM> md);
      INLINE void SetLUFromSPMat (int block_nr, FlatArray<int> d2blk, // L and U stored seperately
				  const SparseMatrixTM<TM> & A, FlatArray<int> dofs, LocalHeap & lh, bool pinv,
				  FlatArray<TM> md);
      INLINE void Prefetch () const;

      INLINE void PrintTo (ostream & os, string prefix = "") const;

      INLINE void RichardsonUpdate (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
				    FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const;
      INLINE void RichardsonUpdate_FW_zig (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
					   FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const;
      INLINE void RichardsonUpdate_BW_zig (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
					   FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const;
      INLINE void RichardsonUpdate_FW_zig_saveU (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
						 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> u) const;
      INLINE void RichardsonUpdate_FW_zig_savebL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
						  FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bl) const;
      INLINE void RichardsonUpdate_BW_zig_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
						 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> l) const;
      INLINE void RichardsonUpdate_saveU (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
					  FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> u) const;
      INLINE void RichardsonUpdate_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
					  FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> l) const;
      INLINE void RichardsonUpdate_savebL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
					   FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bl) const;
      INLINE void RichardsonUpdate_subtractU_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
						    FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const;
      INLINE void RichardsonUpdate_RES (double omega, FlatVector<TV> smallupdate, FlatVector<TV> bigsol,
					FlatVector<TV> smallres, FlatVector<TV> bigres) const;
      INLINE void MultAdd (double scal, FlatVector<TV> xsmall, FlatVector<TV> xlarge,
			   FlatVector<TV> ysmall, FlatVector<TV> ylarge) const;
      INLINE void Mult (FlatVector<TV> xsmall, FlatVector<TV> xlarge,
			FlatVector<TV> ysmall, FlatVector<TV> ylarge) const;
      INLINE void MultAdd_mat (double scal, FlatVector<TV> xsmall, FlatVector<TV> xlarge,
			       FlatVector<TV> ysmall, FlatVector<TV> ylarge) const;
      INLINE void Mult_mat (FlatVector<TV> xsmall, FlatVector<TV> xlarge,
			    FlatVector<TV> ysmall, FlatVector<TV> ylarge) const;
    }; // class BSBlock

  protected:
    using BaseSmoother::sysmat; // TODO: get rid of this too..
    Array<int> buffer_inds, allgroups;
    Array<TM> buffer_vals;
    Array<BSBlock> blocks;
    size_t maxbs, height;
    size_t n_groups, n_blocks_tot;
    Array<size_t> n_blocks_group, fi_blocks;

    // Table<int> block_colors;
    // Array<SharedLoop2> loops;
    // Array<Partitioning> color_balance;

    bool pinv = false;
    bool parallel = true;    // use shared memory parallelization
    bool use_sl2 = false;    // use SharedLoop2
    bool blocks_no = false;  // guaranteed no overlap in blocks (optimizations for zero initial guess, symmetric smooting)
    LocalHeap buffer_lh;

    // working vectors
    mutable AutoVector myresb, update;

  public:

    BSmoother2 (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks,
		bool _parallel = true, bool _use_sl2 = false, bool _pinv = false, bool _blocks_no = false,
		FlatArray<TM> md = FlatArray<TM>(0, nullptr));

    BSmoother2 (shared_ptr<SparseMatrix<TM>> _spmat,  FlatArray<Table<int>> _blocks,
		bool _parallel = true, bool _use_sl2 = false, bool _pinv = false, bool _blocks_no = false,
		FlatArray<TM> md = FlatArray<TM>(0, nullptr));

    virtual ~BSmoother2() = default;

    virtual void Smooth (BaseVector &x, const BaseVector &b,
    			 BaseVector &res, bool res_updated,
    			 bool update_res, bool x_zero) const override;
    virtual void SmoothK (int steps, BaseVector &x, const BaseVector &b,
			  BaseVector &res, bool res_updated = false,
			  bool update_res = true, bool x_zero = false) const override;
    virtual void SmoothSymmK (int steps, BaseVector &x, const BaseVector &b,
			      BaseVector &res, bool res_updated = false,
			      bool update_res = true, bool x_zero = false) const override;
    virtual void SmoothBack (BaseVector &x, const BaseVector &b,
    			     BaseVector &res, bool res_updated,
    			     bool update_res, bool x_zero) const override;
    virtual void SmoothBackK (int steps, BaseVector &x, const BaseVector &b,
			      BaseVector &res, bool res_updated = false,
			      bool update_res = true, bool x_zero = false) const override;
    virtual void SmoothGroups (FlatArray<int> groups, int steps, BaseVector &x, const BaseVector &b,
			       BaseVector &res, bool res_updated,
			       bool update_res, bool x_zero, bool reverse, bool symm) const;

    /** Calling these is "allowed" mut not exactly encouraged **/

    /** Standard smooting, also for overlapping blocks **/
    INLINE void SmoothWO (FlatArray<int> groups, BaseVector & x, const BaseVector & b,
			  BaseVector &res, int steps, bool res_updated,
			  bool update_res, bool x_zero, bool reverse, bool symm) const;
    INLINE void SmoothSimple    (FlatArray<int> groups, BaseVector & x, const BaseVector & b, int steps, bool reverse, bool symm) const;
    INLINE void SmoothRESSimple (FlatArray<int> groups, BaseVector & x, BaseVector & res, int steps, bool reverse, bool symm) const;

    /** Smoothing with optimizations for non-overlapping blocks **/
    INLINE void SmoothNO (FlatArray<int> groups, BaseVector &x, const BaseVector &b,
			  BaseVector &res, int steps, bool res_updated,
			  bool update_res, bool x_zero, bool reverse, bool symm) const;
    INLINE void Smooth                 (FlatArray<int> groups, BaseVector & x, const BaseVector & b,
					int steps, bool zig, bool reverse) const;
    INLINE void Smooth_CR              (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					int steps, bool zig, bool reverse) const;
    INLINE void Smooth_SYMM            (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					int steps, bool zig) const;
    INLINE void Smooth_SYMM_CR         (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					int steps, bool zig) const;
    INLINE void Smooth_savebL          (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					bool zig) const;
    INLINE void SmoothBack_usebL       (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res) const;
    INLINE void SmoothBack_usebL_saveL (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res) const;

    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

    void Mult_Sysmat (const BaseVector & x, BaseVector & y) const;
    void MultAdd_Sysmat (double s, const BaseVector & x, BaseVector & y) const;

    virtual shared_ptr<BaseMatrix> GetAMatrix () const override;
    virtual int VHeight () const override { return height; }
    virtual int VWidth () const override { return height; }
    virtual AutoVector CreateRowVector () const override
    { return make_unique<VVector<TV>> (height); };
    virtual AutoVector CreateColVector () const override
    { return make_unique<VVector<TV>> (height); };

    virtual void PrintTo (ostream & os, string prefix = "") const override;

  protected:
    void SetUp (shared_ptr<SparseMatrix<TM>> _spmat, FlatArray<Table<int>> _block_array, FlatArray<TM> md);

  private:
    template<class TLAM> INLINE void IterateBlocks (FlatArray<int> groups, bool reverse, TLAM lam) const;
  }; // class BSmoother2


  template<class TM>
  class BSmoother2SysMat : public BaseMatrix
  {
  protected:
    shared_ptr<BSmoother2<TM>> bsm;
  public:

    BSmoother2SysMat (shared_ptr<BSmoother2<TM>> _bsm)
      : bsm(_bsm)
    { ; }

    virtual ~BSmoother2SysMat() = default;

    virtual AutoVector CreateRowVector () const override { return bsm->CreateRowVector(); };
    virtual AutoVector CreateColVector () const override { return bsm->CreateColVector(); };
    virtual int VHeight () const override { return bsm->Height(); }
    virtual int VWidth () const override { return bsm->Width(); }

    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    { bsm->Mult_Sysmat(b, x); }
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override
    { bsm->MultAdd_Sysmat(s, b, x); }
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override
    { bsm->Mult_Sysmat(b, x); }
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override
    { bsm->MultAdd_Sysmat(s, b, x); }

  }; // class BSmoother2SysMat


#ifndef FILE_LOC_BGSS_CPP
    extern template class BSmoother2<double>;
    extern template class BSmoother2<Mat<2,2,double>>;
    extern template class BSmoother2<Mat<3,3,double>>;
  #ifdef ELASTICITY
    extern template class BSmoother2<Mat<6,6,double>>;
  #endif // ELASTICITY
#endif // FILE_LOC_BGSS_CPP

} // namespace amg

#endif // FILE_AMG_BLOCKSMOOTHER_LOC_HPP
