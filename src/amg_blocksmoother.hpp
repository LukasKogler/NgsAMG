#ifndef FILE_AMG_BS_HPP
#define FILE_AMG_BS_HPP

namespace amg
{

  /**
     Local Block-GS. Mostly copied from NGSolve, but added capability to smooth
     with residual updates.
   **/
  template<class TM>
  class BSmoother
  {
  public:
    static_assert(mat_traits<TM>::HEIGHT == mat_traits<TM>::WIDTH, "BSmoother cannot do non-square entries (i think)!");
    
    using TSCAL = typename mat_traits<TM>::TSCAL;
    using TV = typename strip_vec<Vec<mat_traits<TM>::HEIGHT,TSCAL>>::type;

  protected:
    shared_ptr<SparseMatrix<TM>> spmat;
    Table<int> blocks;
    Array<TSCAL> buffer; /** buffer for diagonal inverse mats **/
    Array<FlatMatrix<TM>> dinv; /** diag inverse mats **/
    Table<int> block_colors;
    size_t maxbs;
    // Array<SharedLoop2> loops;
  public:

    BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks, FlatArray<TM> md = FlatArray<TM>(0, nullptr));

    /** perform "steps" steps of FW/BW Block-Gauss-Seidel sweeps **/
    void Smooth     (BaseVector & x, const BaseVector & b, int steps = 1) const;
    void SmoothBack (BaseVector & x, const BaseVector & b, int steps = 1) const;
    /** perform "steps" steps of FW/BW Block-Gauss-Seidel sweeps, keeping res = b - A * x up to date **/
    void SmoothRES     (BaseVector & x, BaseVector & res, int steps = 1) const;
    void SmoothBackRES (BaseVector & x, BaseVector & res, int steps = 1) const;

  private:
    template<class TLAM> INLINE void Smooth_impl    (BaseVector & x, const BaseVector & b, TLAM get_col, int steps = 1) const;
    template<class TLAM> INLINE void SmoothRES_impl (BaseVector & x, BaseVector & res, TLAM get_col, int steps = 1) const;
  }; // class BSmoother


  /**
     Hybrid Block-Smoother. Requires that blocks do not cross subdomain boundaries.
     From any blocks that touch a subdomain boundary, all DOFs the calling proc is not
     master of are removed. If the resulting block is empty, it is removed entirely.
   **/
  template<class TM>
  class HybridBS : public HybridSmoother2<TM>
  {
  protected:
    using HybridSmoother2<TM>::A;

    // loc1, ex, loc2
    Array<shared_ptr<BSmoother<TM>>> loc_smoothers;

  public:
    HybridBS (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h, Table<int> && blocks,
	       bool _overlap, bool _in_thread);
  protected:

    /** Filter blocks:
	    i) remove all non-master dofs from all blocks and remove all now empty blocks
	   ii) remove blocks i am not master of
	  iii) partition blocks into 3 stages: local1, ex, local2
    **/
    Array<Table<int>> FilterBlocks (Table<int> && blocks);

    /** Inherited from BaseSmoother **/
    virtual Array<MemoryUsage> GetMemoryUsage() const override;
    
    /** Inherited from HybridSmoother2 **/
    virtual void SmoothLocal (int stage, BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothBackLocal (int stage, BaseVector &x, const BaseVector &b) const override;
    virtual void SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const override;
    virtual void SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const override;
  }; // class HybridBS

} // namespace amg

#endif
