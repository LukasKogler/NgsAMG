
#include <condition_variable>

#include "hybrid_base_smoother.hpp"

namespace amg
{

/** BackgroundMPIThread **/

class BackgroundMPIThread
{
protected:
  unique_ptr<std::thread> t;
  mutex glob_mut;
  bool thread_ready = true, thread_done = false, end_thread = false;
  std::condition_variable cv;
  function<void(void)> thread_exec_fun = [](){};

public:
  BackgroundMPIThread ()
  {
#ifdef USE_TAU
    TAU_PROFILE_SET_NODE(NgMPI_Comm(MPI_COMM_WORLD).Rank());
#endif
    t = make_unique<std::thread>([this](){ this->thread_fpf(); });
  } // BackgroundMPIThread (..)

  ~BackgroundMPIThread () { thread_ready = true; end_thread = true; cv.notify_one(); t->join(); }

  void thread_fpf () {
#ifdef USE_TAU
    TAU_REGISTER_THREAD();
    TAU_PROFILE_SET_NODE(NgMPI_Comm(MPI_COMM_WORLD).Rank());
#endif
    while( !end_thread ) {
// cout << "--- ulock" << endl;
std::unique_lock<std::mutex> lk(glob_mut);
// cout << "--- wait" << endl;
cv.wait(lk, [&](){ return thread_ready; });
// cout << "--- woke up,  " << thread_ready << " " << thread_done << " " << end_thread << endl;
if (!end_thread && !thread_done) {
  thread_exec_fun();
  /** if thread_exec_fun is a lambda, it can keep alive objects unnecessarily! **/
  thread_exec_fun = [](){}; thread_done = true; thread_ready = false;
  cv.notify_one();
}
    }
    // cout << "--- am done!" << endl;
  }


  void StartInThread (function<void(void)> _thread_exec_fun)
  {
    std::lock_guard<std::mutex> lk(glob_mut);
    thread_done = false; thread_ready = true;
    thread_exec_fun = _thread_exec_fun;
    cv.notify_one();
  } // BackgroundMPIThread::StartInThread

  void WaitForThread ()
  {
    std::unique_lock<std::mutex> lk(glob_mut);
    cv.wait(lk, [&]{ return thread_done; });
  } // BackgroundMPIThread::WaitForThread

}; // class BackgroundMPIThread


shared_ptr<BackgroundMPIThread> GetGlobMPIThread () {
  static shared_ptr<BackgroundMPIThread> glob_gt;
  if (glob_gt == nullptr)
    { glob_gt = make_shared<BackgroundMPIThread>(); }
  return glob_gt;
}

/** END BackgroundMPIThread **/



/** HybridBaseSmoother **/


template<class TSCAL>
HybridBaseSmoother<TSCAL>::
HybridBaseSmoother (shared_ptr<HybridBaseMatrix<TSCAL>> A,
                    int numLocSteps,
                    bool commInThread,
                    bool overlapComm)
  : BaseSmoother(A)
  , _hybridA(A)
  , _numLocSteps(numLocSteps)
  , _commInThread(commInThread)
  , _overlapComm(overlapComm)
{
  _stashedGx = GetHybridA().CreateVector();

  mpi_thread = _commInThread ? GetGlobMPIThread() : nullptr;
} // HybridBaseSmoother (..)


template<class TSCAL>
HybridBaseMatrix<TSCAL> const &
HybridBaseSmoother<TSCAL>::
GetHybridA() const
{
  return *_hybridA;
} // HybridBaseSmoother::GetHybridA


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
StartDIS2CO (BaseVector &vec) const
{
  if ( _overlapComm && _commInThread )
  {
    mpi_thread->StartInThread([&]()
    {
      auto &dCCMap = GetHybridA().GetDCCMap();

      dCCMap.StartDIS2CO(vec);
      dCCMap.WaitD2C();
    });
  }
  else
  {
    auto &dCCMap = GetHybridA().GetDCCMap();

    dCCMap.StartDIS2CO(vec);

    if (!_overlapComm)
      { dCCMap.WaitD2C(); }
  }
} // HybridBaseSmoother::StartDIS2CO


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
FinishDIS2CO (BaseVector &vec) const
{
  auto &dCCMap = GetHybridA().GetDCCMap();

  // wait for comm. to finish (buffer vals have arrived)
  if ( _overlapComm )
  {
    if ( _commInThread )
    {
      mpi_thread->WaitForThread();
    }
    else
    {
      dCCMap.WaitD2C();
    }
  }

  dCCMap.ApplyM(vec); // vec += buffer
} // HybridBaseSmoother::FinishDIS2CO


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
StartCO2CU (BaseVector &vec) const
{
  if ( _overlapComm && _commInThread )
  {
    auto &dCCMap = GetHybridA().GetDCCMap();

    mpi_thread->StartInThread([&] () {
      dCCMap.StartCO2CU(vec);
      dCCMap.ApplyCO2CU(vec);
      dCCMap.FinishCO2CU(); // probably dont take a shortcut ??
    });
  }
  else
  {
    auto &dCCMap = GetHybridA().GetDCCMap();

    dCCMap.StartCO2CU(vec);

    if ( !_overlapComm )
    {
      dCCMap.ApplyCO2CU(vec);
      dCCMap.FinishCO2CU(); // probably dont take a shortcut ??
    }
  }
} // HybridBaseSmoother::StartCO2CU


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
FinishCO2CU (BaseVector &vec) const
{
  if ( _overlapComm )
  {
    if ( _commInThread )
    {
      mpi_thread->WaitForThread();
    }
    else
    {
      auto &dCCMap = GetHybridA().GetDCCMap();

      dCCMap.ApplyCO2CU(vec);
      dCCMap.FinishCO2CU(); // probably dont take a shortcut ??
    }
  }
} // HybridBaseSmoother::FinishCO2CU


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
Smooth (BaseVector       &x,
        BaseVector const &b,
        BaseVector       &res,
        bool res_updated,
        bool update_res,
        bool x_zero) const
{
  SmoothImpl<FORWARD>(x, b, res, res_updated, update_res, x_zero);
} // HybridBaseSmoother::SmoothBack


template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
SmoothBack (BaseVector       &x,
            BaseVector const &b,
            BaseVector       &res,
            bool res_updated,
            bool update_res,
            bool x_zero) const
{
  SmoothImpl<BACKWARD>(x, b, res, res_updated, update_res, x_zero);
} // HybridBaseSmoother::SmoothBack


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
void
HybridBaseSmoother<TSCAL>::
SmoothImpl (BaseVector       &x,
            BaseVector const &b,
            BaseVector       &res,
            bool res_updated,
            bool update_res,
            bool x_zero) const
{
  /**
    *  Cost in increasing order:
    *      I   RES update
    *      II  smooth with RHS
    *      III smooth with RES
    *      IV  smooth with RHS + res-update
    */
  if (update_res)
  {
    if (!res_updated)
    {
      if (!x_zero)
      {
        // [II + I] < [I + III]
        SmoothImplRHS<DIR>(x, b, res, x_zero);

        res = b;
        res -= (*GetAMatrix()) * x;
      }
      else
      {
        // III < IV
        res = b;
        SmoothImplRES<DIR>(x, b, res, x_zero);
      }
    }
    else
    {
      // III < IV
      SmoothImplRES<DIR>(x, b, res, x_zero);
    }
  }
  else
  {
    SmoothImplRHS<DIR>(x, b, res, x_zero);
  }
} // DynBlockSmoother::SmoothImpl


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
void
HybridBaseSmoother<TSCAL>::
SmoothImplRES (BaseVector       &x,
               BaseVector const &b,
               BaseVector       &res,
               bool x_zero) const
{
  constexpr SMOOTH_STAGE P1 = ( DIR == FORWARD ) ? LOC_PART_1 : LOC_PART_2;
  constexpr SMOOTH_STAGE P2 = ( DIR == FORWARD ) ? LOC_PART_2 : LOC_PART_1;

  /**
   * The local smoothers only see "M", so if we start out with
   *     res = b - A xold = b - M xold - G xold
   * Afterwards, we have
   *     res = b - M xnew - G xold
   * Therefore, we stash G xold and in the end update the residuum with
   *     res += G xold - G xnew
   *
   */

  auto const &hybA = GetHybridA();

  bool const stashGx = (!x_zero) && hybA.HasGGlobal();

  auto &gx = *_stashedGx;

  // most of the time status is correct here and this is a noop
  x.Cumulate();
  // res.Distribute(); // no! can use it if cumulated (so basically never)

  gx.SetParallelStatus(DISTRIBUTED);

  auto &locX   = *x.GetLocalVector();
  auto &locRes = *res.GetLocalVector();
  auto &locGx  = *gx.GetLocalVector();

  if ( stashGx )
  {
    gx.SetParallelStatus(DISTRIBUTED);

    if (hybA.HasGLocal())
    {
      hybA.GetG()->Mult(locX, locGx);
    }
    else
    {
      locGx = 0.0;
    }
  }

  CallStageKernelsImpl<DIR, true>(x, b, res, x_zero, res.GetParallelStatus() == DISTRIBUTED);
  
  // bool const need_d2c = res.GetParallelStatus() == DISTRIBUTED;

  // bool xZero = x_zero;

  // if (need_d2c)
  // {
  //   StartDIS2CO(res);
  // }

  // SmoothStageRes(P1, DIR, locX, locRes, xZero);

  // if (need_d2c)
  // {
  //   FinishDIS2CO(res);
  // }

  // SmoothStageRes(EX_PART, DIR, locX, locRes, xZero);

  // for (int s = 1; s < _numLocSteps; s++)
  // {
  //   // these smoothes continue to use "old" values of x for non-master DOFs which is what we want

  //   SmoothStageRes(P2, DIR, locX, locRes, xZero);

  //   xZero = false; // cannot use x ZIG any more from this point on

  //   SmoothStageRes(EX_PART, DIR, locX, locRes, xZero);

  //   SmoothStageRes(P1, DIR, locX, locRes, xZero);
  // }

  // /**
  //  * This is strictly speaking not true, x has "new" values on master and "old" values on other
  //  * ranks, true "DISTRIBUTED" would be "new" on master and ZERO elsewhere. This does not matter
  //  * here only because we are using CO2CU which overwrites the "old" values instead of adding to
  //  * them.
  //  */
  // x.SetParallelStatus(DISTRIBUTED);

  // StartCO2CU(x);

  // SmoothStageRes(P2, DIR, locX, locRes, xZero);

  // FinishCO2CU(x);

  if ( hybA.HasGGlobal() )
  {
    res.SetParallelStatus(DISTRIBUTED); // should be noop

    if ( stashGx ) // stashed G*x_old
    {
      locRes += locGx; // gx is distributed!
    }

    if ( hybA.HasGLocal() )
    {
      hybA.GetG()->MultAdd(-1, locX, locRes); // x cumulated, res distributed
    }
  }
} // DynBlockSmoother::SmoothImplRes


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
void
HybridBaseSmoother<TSCAL>::
SmoothImplRHS (BaseVector       &x,
               BaseVector const &actualB,
               BaseVector       &res, // can use this as work-vector
               bool x_zero) const
{
  constexpr SMOOTH_STAGE P1 = ( DIR == FORWARD ) ? LOC_PART_1 : LOC_PART_2;
  constexpr SMOOTH_STAGE P2 = ( DIR == FORWARD ) ? LOC_PART_2 : LOC_PART_1;

  // The local smoothers only see "M", so we feed in "b - Gx" as RHS
  auto const &hybA = GetHybridA();

  bool const use_bmGX = (!x_zero) && hybA.HasGGlobal();

  auto &bMGx = *_stashedGx;
  auto &locBMGx  = *bMGx.GetLocalVector();

  // most of the time status is correct here and this is a noop
  x.Cumulate();

  if (use_bmGX)
  {
    actualB.Distribute();
    bMGx.SetParallelStatus(DISTRIBUTED);
    bMGx = actualB;

    if (hybA.HasGLocal())
    {
      hybA.GetG()->MultAdd(-1.0, *x.GetLocalVector(), locBMGx);
    }
  }

  BaseVector const &usedB = use_bmGX ? bMGx : actualB;

  // this is kinda annoying, but the DCC communication uses non-const references
  CallStageKernelsImpl<DIR, false>(x, usedB, res, x_zero, usedB.GetParallelStatus() == DISTRIBUTED);

  // auto       &locX = *x.GetLocalVector();
  // auto const &locB = *b.GetLocalVector();

  // bool const need_d2c = actualB.GetParallelStatus() == DISTRIBUTED;

  // bool xZero = x_zero;

  // if (need_d2c)
  // {
  //   StartDIS2CO(ncB);
  // }

  // SmoothStageRHS(P1, DIR, locX, locB, xZero);

  // if (need_d2c)
  // {
  //   FinishDIS2CO(ncB);
  // }

  // SmoothStageRHS(EX_PART, DIR, locX, locB, xZero);

  // for (int s = 1; s < _numLocSteps; s++)
  // {
  //   // these smoothes continue to use "old" values of x for non-master DOFs which is what we want

  //   SmoothStageRHS(P2, DIR, locX, locB, xZero);

  //   xZero = false; // cannot use x ZIG any more from this point on

  //   SmoothStageRHS(EX_PART, DIR, locX, locB, xZero);

  //   SmoothStageRHS(P1, DIR, locX, locB, xZero);
  // }

  // /**
  //  * This is strictly speaking not true, x has "new" values on master and "old" values on other
  //  * ranks, true "DISTRIBUTED" would be "new" on master and ZERO elsewhere. This does not matter
  //  * here only because we are using CO2CU which overwrites the "old" values instead of adding to
  //  * them.
  //  */
  // x.SetParallelStatus(DISTRIBUTED);

  // StartCO2CU(x);

  // SmoothStageRHS(P2, DIR, locX, locB, xZero);

  // FinishCO2CU(x);
} // HybridBaseSmoother::SmoothImplRHS


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR, bool RES>
void
HybridBaseSmoother<TSCAL>::
CallStageKernelsImpl(BaseVector       &x,
                     BaseVector const &b,
                     BaseVector       &res,
                     bool       const &x_zero,
                     bool       const &need_d2c) const
{
  constexpr SMOOTH_STAGE P1 = ( DIR == FORWARD ) ? LOC_PART_1 : LOC_PART_2;
  constexpr SMOOTH_STAGE P2 = ( DIR == FORWARD ) ? LOC_PART_2 : LOC_PART_1;

  BaseVector &rb = RES ? res : const_cast<BaseVector&>(b);

  // cout << " CallStageKernelsImpl, DIR = " << DIR << ", RES = " << RES
  //      << ", x_zero = " << x_zero << ", need_d2c = " << need_d2c << endl;

  auto       &locX  = *x.GetLocalVector();
  auto const &locB  = *b.GetLocalVector();
  auto       &locR  = *res.GetLocalVector();

  bool xZero = x_zero;

  auto callSM = [&](auto stage)
  {
    if constexpr( RES )
    {
      SmoothStageRes(stage, DIR, locX, locB, locR, xZero);
    }
    else
    {
      SmoothStageRHS(stage, DIR, locX, locB, locR, xZero);
    }
  };

  if (need_d2c)
  {
    StartDIS2CO(rb);
  }

  callSM(P1);

  if (need_d2c)
  {
    FinishDIS2CO(rb);
  }

  callSM(EX_PART);

  for (int s = 1; s < _numLocSteps; s++)
  {
    // these smoothes continue to use "old" values of x for non-master DOFs which is what we want

    callSM(P2);

    xZero = false; // cannot use x ZIG any more from this point on

    callSM(EX_PART);

    callSM(P1);
  }

  /**
   * This is strictly speaking not true, x has "new" values on master and "old" values on other
   * ranks, true "DISTRIBUTED" would be "new" on master and ZERO elsewhere. This does not matter
   * here only because we are using CO2CU which overwrites the "old" values instead of adding to
   * them.
   */
  x.SetParallelStatus(DISTRIBUTED);

  StartCO2CU(x);

  callSM(P2);

  FinishCO2CU(x);
}

template<class TSCAL>
void
HybridBaseSmoother<TSCAL>::
PrintTo (ostream & os, string prefix) const
{
  std::string spaces(prefix.size(), ' ');

  os << prefix << "HybridBaseSmoother " << endl;
  os << spaces << "   A dims: " << GetHybridA().Height() << " x " << GetHybridA().Width() << endl;
  os << spaces << "   overlap MPI-communication: " << _overlapComm << endl;
  os << spaces << "   using MPI-thread: " << _commInThread << endl;
  os << spaces << "   #local steps:     " << _numLocSteps << endl;
} // HybridSmoother::PrintTo


/** END HybridBaseSmoother **/


template class HybridBaseSmoother<double>;

} // namespace amg