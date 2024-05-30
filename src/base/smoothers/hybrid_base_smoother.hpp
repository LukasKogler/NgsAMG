#ifndef FILE_AMG_HYBRID_BASE_SMOOTHER_HPP
#define FILE_AMG_HYBRID_BASE_SMOOTHER_HPP

#include <base.hpp>
#include <hybrid_matrix.hpp>

#include "base_smoother.hpp"

namespace amg
{

class BackgroundMPIThread;

/**
 * HybridBaseSmoother is any smoother that can be split into three sub-smoothers,
 * two local ones working on internal rows that do not neighbour any exchange-row
 * and one working on any exchange row or neighbor of one.
 * The local smoothes are used to hide the MPI-communication for the exchange smooth.
 */
template<class TSCAL>
class HybridBaseSmoother : public BaseSmoother
{
public:
  HybridBaseSmoother (shared_ptr<HybridBaseMatrix<TSCAL>> A,
                      int numLocSteps,
                      bool commInThread,
                      bool overlapComm = true);

  virtual ~HybridBaseSmoother () = default;

  void
  Smooth (BaseVector       &x,
          BaseVector const &b,
          BaseVector       &res,
          bool res_updated = false,
          bool update_res = false,
          bool x_zero = false) const override;

  void
  SmoothBack (BaseVector       &x,
              BaseVector const &b,
              BaseVector       &res,
              bool res_updated = false,
              bool update_res = false,
              bool x_zero = false) const override;

  void
  PrintTo (ostream & os, string prefix = "") const override;

protected:
 HybridBaseMatrix<TSCAL> const &GetHybridA() const;

  /**
      Local smooting stages:
    **/
  enum SMOOTH_STAGE : char
  {
    LOC_PART_1 = 0, // smooth local part 1 (hides distributed -> concentrated)
    EX_PART    = 1, // smooth exchange
    LOC_PART_2 = 2  // smooth local part 2 (hides concentrated -> cumulated)
  };

  virtual
  void
  SmoothStageRHS (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b,
                  BaseVector                &res, // can be used as work-vector
                  bool                const &x_zero) const = 0;

  virtual
  void
  SmoothStageRes (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b, // can sometimes be used for optimization
                  BaseVector                &res,
                  bool                const &x_zero) const = 0;


private:
  shared_ptr<HybridBaseMatrix<TSCAL>> _hybridA;

  int  _numLocSteps;
  bool _commInThread;
  bool _overlapComm;

  shared_ptr<BaseVector> _stashedGx; // to stash M * x_old
  shared_ptr<BackgroundMPIThread> NG_MPI_thread;

private:
  template<SMOOTHING_DIRECTION DIR>
  void
  SmoothImpl (BaseVector       &x,
              BaseVector const &b,
              BaseVector       &res,
              bool res_updated,
              bool update_res,
              bool x_zero) const;

  template<SMOOTHING_DIRECTION DIR>
  void
  SmoothImplRHS (BaseVector       &x,
                 BaseVector const &b,
                 BaseVector       &res, // can use this as work-vector
                 bool x_zero) const;

  template<SMOOTHING_DIRECTION DIR>
  void
  SmoothImplRES (BaseVector       &x,
                 BaseVector const &b,
                 BaseVector       &res,
                 bool x_zero) const;

  template<SMOOTHING_DIRECTION DIR, bool RES>
  void
  CallStageKernelsImpl(BaseVector       &x,
                       BaseVector const &b,
                       BaseVector       &res,
                       bool       const &need_d2c,
                       bool       const &x_zero) const;

  void StartDIS2CO  (BaseVector &vec) const;
  void FinishDIS2CO (BaseVector &vec) const;
  void StartCO2CU   (BaseVector &vec) const;
  void FinishCO2CU  (BaseVector &vec) const;
}; // class HybridBaseSmoother


template<class TSCAL>
INLINE ostream& operator << (ostream &os, typename HybridBaseSmoother<TSCAL>::SMOOTH_STAGE const &stage)
{
  switch(stage)
  {
    case(HybridBaseSmoother<TSCAL>::SMOOTH_STAGE::LOC_PART_1): { os << "LOC_PART_1"; break; }
    case(HybridBaseSmoother<TSCAL>::SMOOTH_STAGE::EX_PART):    { os << "EX_PART";    break; }
    case(HybridBaseSmoother<TSCAL>::SMOOTH_STAGE::LOC_PART_2): { os << "LOC_PART_2"; break; }
  }
  return os;
}

} // namespace amg

#endif // FILE_AMG_HYBRID_BASE_SMOOTHER_HPP