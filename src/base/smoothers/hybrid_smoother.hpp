#ifndef FIL_HYBRID_SMOOTHER_HPP
#define FIL_HYBRID_SMOOTHER_HPP


#include "hybrid_base_smoother.hpp"

namespace amg
{

/**
 * Base-class for hybrid smoothers for regular sparse matrices.
 * It can compute modified diagonals that have enough extra weight
 * to make the parallel hybrid smoother convergent if the local
 * smoothers are convergent.
 */
template<class TM>
class HybridSmoother : public HybridBaseSmoother<typename mat_traits<TM>::TSCAL>
{
public:
  using TSCAL = typename mat_traits<TM>::TSCAL;

  HybridSmoother (shared_ptr<HybridMatrix<TM>> _A,
                  int  _numLocSteps  = 1,
                  bool _commInThread = true,
                  bool _overlapComm  = true);

  HybridSmoother (shared_ptr<BaseMatrix> _A,
                  int  _numLocSteps  = 1,
                  bool _commInThread = true,
                  bool _overlapComm  = true);

  virtual ~HybridSmoother () = default;

  Array<MemoryUsage> GetMemoryUsage() const override { return Array<MemoryUsage>(); }

  void PrintTo (ostream & os, string prefix = "") const override;

  size_t GetNOps () const override;
  size_t GetANZE () const override;

protected:

  Array<TM> CalcModDiag (shared_ptr<BitArray> free);

  HybridMatrix<TM>       &GetHybSparseA ()       { return *_hybridSpA; }
  HybridMatrix<TM> const &GetHybSparseA () const { return *_hybridSpA; }

private:
  shared_ptr<HybridMatrix<TM>> _hybridSpA;
}; // class HybridSmoother

/** END HybridSmoother **/


/**
 * Hybrid Smoother for regular sparse-matrices, local "smoothers"
 * given by direct inverse. Used mostly for debugging.
 */
template<class TM>
class HybridDISmoother : public HybridSmoother<TM>
{
protected:
  shared_ptr<BaseMatrix> loc_inv;

public:
  using TSCAL = typename HybridSmoother<TM>::TSCAL;

  HybridDISmoother (shared_ptr<BaseMatrix> A,
                    shared_ptr<BitArray> freedofs,
                    bool overlap = false,
                    bool NG_MPI_thread = false);


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

}; // class HybridDISmoother

/** END HybridDISmoother **/

#ifndef FILE_HYBRID_SMOOTHER_CPP
extern template class HybridDISmoother<double>;
extern template class HybridDISmoother<Mat<2,2,double>>;
extern template class HybridDISmoother<Mat<3,3,double>>;
#ifdef ELASTICITY
extern template class HybridDISmoother<Mat<6,6,double>>;
#endif // ELASTICITY
#endif // FILE_HYBRID_SMOOTHER_CPP

} // namespace amg

#endif // FIL_HYBRID_SMOOTHER_HPP
