#ifndef FILE_AMG_DYN_BLOCK_SMOOTHER_HPP
#define FILE_AMG_DYN_BLOCK_SMOOTHER_HPP

#include <base.hpp>
#include <dyn_block.hpp>

#include "base_smoother.hpp"
#include "hybrid_smoother.hpp"

namespace amg
{

template<class TSCAL> class HybridDynBlockSmoother;

template<class TSCAL>
class DynBlockSmoother : public BaseSmoother
{
  friend class HybridDynBlockSmoother<TSCAL>;

public:

  DynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>        A,
                   shared_ptr<BitArray>                           freeDofs = nullptr,
                   double                                  const &uRel = 1.0);

  DynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>        A,
                   FlatArray<TSCAL>                               modDiag,
                   shared_ptr<BitArray>                           freeDofs = nullptr,
                   double                                  const &uRel = 1.0);

  virtual ~DynBlockSmoother() = default;

  void
  Smooth (BaseVector &x, BaseVector const &b,
          BaseVector &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const override;

  void
  SmoothBack (BaseVector &x, BaseVector const &b,
              BaseVector &res, bool res_updated = false,
              bool update_res = true, bool x_zero = false) const override;

protected:

  void SetUp(BitArray const *freeDofs,
             FlatArray<TSCAL> modDiag = FlatArray<TSCAL>(0, nullptr));

  INLINE FlatMatrix<TSCAL>
  getDiagBlock (unsigned const &kBlock) const
  {
    return FlatMatrix<TSCAL>(_scalBlockSize[kBlock], _scalBlockSize[kBlock], _data.Data() + _dataOff[kBlock]);
  } // getDiagBlock

  template<SMOOTHING_DIRECTION DIR>
  INLINE void
  SmoothImpl (BaseVector &x, BaseVector const &b,
              BaseVector &res, bool res_updated = false,
              bool update_res = true, bool x_zero = false) const;

  template<SMOOTHING_DIRECTION DIR>
  INLINE void
  SmoothRHS(BaseVector &x, BaseVector const &b) const;

  template<SMOOTHING_DIRECTION DIR>
  INLINE void
  SmoothRes(BaseVector &x, BaseVector &res) const;

  template<SMOOTHING_DIRECTION DIR, class TLAM>
  INLINE void
  IterateBlocks(TLAM lam) const;

  template<SMOOTHING_DIRECTION DIR, class TLAM>
  INLINE void
  IterateBlocks(FlatArray<unsigned> blockNums, TLAM lam) const;

  INLINE void
  updateBlockRHS (unsigned          const &kBlock,
                  FlatVector<TSCAL>       &x,
                  FlatVector<TSCAL> const &b) const;

  template<SMOOTHING_DIRECTION DIR>
  INLINE void
  updateBlocksRHS (FlatArray<unsigned>      blockNums,
                   FlatVector<TSCAL>       &x,
                   FlatVector<TSCAL> const &b) const;


  INLINE void
  updateBlockRes (unsigned          const &kBlock,
                  FlatVector<TSCAL>       &x,
                  FlatVector<TSCAL>       &res) const;

  template<SMOOTHING_DIRECTION DIR>
  INLINE void
  updateBlocksRes (FlatArray<unsigned>      blockNums,
                   FlatVector<TSCAL>       &x,
                   FlatVector<TSCAL>       &b) const;


  DynBlockSparseMatrix<TSCAL> const &getDynA() const { return *_A; }

  shared_ptr<DynBlockSparseMatrix<TSCAL>> _A;

  double _omega;

  shared_ptr<BitArray> _freeBlocks;

  Array<unsigned> _scalBlockSize;
  Array<TSCAL>    _data;
  Array<unsigned> _dataOff;

  Vector<TSCAL>   _rowBuffer0;
  Vector<TSCAL>   _rowBuffer1;
  Vector<TSCAL>   _colBuffer0;
  Vector<TSCAL>   _colBuffer1;
}; // class DynBlockSmoother


/**
 * Base class for hybrid smoothers for dynamic block-matrices.
 * It can compute modified diagonals that have enough extra weight
 * to make the parallel hybrid smoother convergent if the local
 * smoothers are.
 */
template<class TSCAL>
class HybridDynBlockSmoother : public HybridBaseSmoother<TSCAL>
{
public:
  HybridDynBlockSmoother(shared_ptr<DynamicBlockHybridMatrix<TSCAL>>  A,
                         shared_ptr<BitArray>    freeDofs = nullptr,
                         int                     numLocSteps = 1,
                         bool                    commInThread = true,
                         bool                    overlapComm = true,
                         double                  uRel = 1.0);

  HybridDynBlockSmoother(shared_ptr<BaseMatrix>  A,
                         shared_ptr<BitArray>    freeDofs = nullptr,
                         int                     numLocSteps = 1,
                         bool                    commInThread = true,
                         bool                    overlapComm = true,
                         double                  uRel = 1.0);

protected:
  using SMOOTH_STAGE = typename HybridBaseSmoother<TSCAL>::SMOOTH_STAGE;

  Array<TSCAL> CalcScalModDiag (shared_ptr<BitArray> scalFree);

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

  DynamicBlockHybridMatrix<TSCAL> const &GetDynBlockHybA() const { return *_hybDynBlockA; }

  unsigned        _splitInd;
  Array<unsigned> _locBlockNums;
  Array<unsigned> _exBlockNums;

  INLINE FlatArray<unsigned> GetBlocksForStage(SMOOTH_STAGE const &stage) const;


  shared_ptr<DynamicBlockHybridMatrix<TSCAL>> _hybDynBlockA;
  shared_ptr<DynBlockSmoother<TSCAL>>         _locSmoother;
}; // class HybridDynBlockSmoother

#ifndef FILE_DYN_BLOCK_SMOOTHER_CPP
extern template class DynBlockSmoother<double>;
extern template class HybridDynBlockSmoother<double>;
#endif // FILE_DYN_BLOCK_SMOOTHER_CPP

} // namespace amg

#endif // FILE_AMG_DYN_BLOCK_SMOOTHER_HPP