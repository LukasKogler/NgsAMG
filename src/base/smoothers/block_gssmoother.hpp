#ifndef FILE_AMG_BS_HPP
#define FILE_AMG_BS_HPP

#include "base_smoother.hpp"
#include "gssmoother.hpp"
#include "hybrid_smoother.hpp"
#include "loc_block_gssmoother.hpp"

namespace amg
{

/**
 *  Local Block-GS. Mostly copied from NGSolve, but added capability to smooth
 *  with residual updates.
 */
template<class TM>
class BSmoother : public BaseSmoother
{
public:
  static_assert(ngbla::Height<TM>() == ngbla::Width<TM>(), "BSmoother cannot do non-square entries (i think)!");

  using TSCAL = typename mat_traits<TM>::TSCAL;
  using TV = typename strip_vec<Vec<ngbla::Height<TM>(),TSCAL>>::type;

protected:
  shared_ptr<SparseMatrix<TM>> spmat;
  Table<int> blocks;
  Array<TM> buffer; /** buffer for diagonal inverse mats **/
  Array<FlatMatrix<TM>> dinv; /** diag inverse mats **/
  Table<int> block_colors;
  size_t maxbs;

  Array<SharedLoop2> loops;
  Array<Partitioning> color_balance;

  bool parallel = true, use_sl2 = false, pinv = false;

public:

  BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks,
        bool _parallel = true, bool _use_sl2 = false, bool _pinv = false,
        FlatArray<TM> md = FlatArray<TM>(0, nullptr));

  virtual ~BSmoother() = default;

  /** perform "steps" steps of FW/BW Block-Gauss-Seidel sweeps **/
  void SmoothInternal     (BaseVector &x, BaseVector const &b, int steps = 1) const;
  void SmoothBackInternal (BaseVector &x, BaseVector const &b, int steps = 1) const;
  /** perform "steps" steps of FW/BW Block-Gauss-Seidel sweeps, keeping res = b - A * x up to date **/
  void SmoothRESInternal     (BaseVector &x, BaseVector &res, int steps = 1) const;
  void SmoothBackRESInternal (BaseVector &x, BaseVector &res, int steps = 1) const;

  void
  Smooth (BaseVector &x, BaseVector const &b, BaseVector &res,
          bool res_updated, bool update_res, bool x_zero) const override;
  void
  SmoothBack (BaseVector &x, BaseVector const &b, BaseVector &res,
              bool res_updated, bool update_res, bool x_zero) const override;

  virtual int VHeight () const override { return spmat->Height(); }
  virtual int VWidth () const override { return spmat->Height(); }
  virtual AutoVector CreateVector () const override
  { return make_unique<VVector<TV>>(spmat->Height()); }
  virtual AutoVector CreateRowVector () const override { return CreateVector(); }
  virtual AutoVector CreateColVector () const override { return CreateVector(); }

  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

private:
  template<class TLAM> INLINE void IterateBlocks (bool reverse, TLAM lam) const;
  INLINE void Smooth_impl    (BaseVector & x, const BaseVector & b, int steps, bool reverse) const;
  INLINE void SmoothRES_impl (BaseVector & x, BaseVector & res, int steps, bool reverse) const;
}; // class BSmoother


/**
  * Hybrid Block-Smoother. Requires that blocks do not cross subdomain boundaries.
  * From any blocks that touch a subdomain boundary, all DOFs the calling proc is not
  * master of are removed. If the resulting block is empty, it is removed entirely.
  */
template<class TM>
class HybridBS : public HybridSmoother<TM>
{
public:
  using TSCAL = typename mat_traits<TM>::TSCAL;

  HybridBS (shared_ptr<BaseMatrix> _A,
            Table<int> && blocks,
            bool _pinv = false,
            bool _overlap = true,
            bool _in_thread = false,
            bool _parallel = true,
            bool _sl2 = false,
            bool _bs2 = true,
            bool _no_blocks = false,
            bool _symm_loc = false,
            int _nsteps_loc = 1);

  virtual ~HybridBS() = default;

  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

protected:

  /** Filter blocks:
    i) remove all non-master dofs from all blocks and remove all now empty blocks
    ii) remove blocks i am not master of
  iii) partition blocks into 3 stages: local1, ex, local2
  **/
  Array<Table<int>> FilterBlocks (Table<int> && blocks);

  /** Inherited from BaseSmoother **/
  virtual Array<MemoryUsage> GetMemoryUsage() const override;

protected:
  using SMOOTH_STAGE = typename HybridBaseSmoother<TSCAL>::SMOOTH_STAGE;

  void
  SmoothStageRHS (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b,
                  BaseVector                &res,
                  bool                const &x_zero) const override;

  void
  SmoothStageRes (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b,
                  BaseVector                &res,
                  bool                const &x_zero) const override;
                
private:
  // loc1, ex, loc2
  bool bs2;
  bool no_blocks;
  bool symm_loc;
  Array<shared_ptr<BSmoother<TM>>> loc_smoothers;
  shared_ptr<BSmoother2<TM>> loc_smoother_2;
}; // class HybridBS

} // namespace amg

#endif
