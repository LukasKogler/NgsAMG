#ifndef FILE_AMG_HYBRID_SMOOTHER_UTILS_HPP
#define FILE_AMG_HYBRID_SMOOTHER_UTILS_HPP

#include <dcc_map.hpp>
#include <base.hpp>

namespace amg
{


template<class TM, class TIT_D, class TIT_OD>
INLINE void
CalcHybridSmootherRDGItGeneric(size_t                                 const &N,
                               DCCMap<typename mat_traits<TM>::TSCAL> const &dCCMap,
                               Array<TM>                                    &modDiag,
                               BitArray                               const *freeDOFs,
                               TIT_D                                         iterateD,
                               TIT_OD                                        iterateOD)
{
  static Timer t("CalcHybridSmootherRDGItGeneric");
  RegionTimer rt(t);

  constexpr int BS = Height<TM>();

  auto const &parDOFs = *dCCMap.GetParallelDofs();

  modDiag.SetSize0();

  /** No need for adding to the diagonal **/
  if ( ( parDOFs.GetDistantProcs().Size() == 0 ) )
    { return; }

  typedef typename mat_traits<TM>::TV_COL TV;

  Array<TM> origDiag(N);

  // have to zero this since there is no guarantee that all rows are
  // touched by iterateD (e.h. hyrbid-block only goes through M-rows)
  origDiag = 0.0;

  iterateD([&](auto const &k, auto const &diag) LAMBDA_INLINE
  {
    origDiag[k] = diag;
  });
  // for (auto k : Range(N))
  //   { origDiag[k] = getDiag(k); }

  MyAllReduceDofData (parDOFs, origDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  Array<TV> scalSqrt(N * BS);

  for (auto k : Range(N))
  {
    auto &sqrti = scalSqrt[k];

    if constexpr( is_same<TM,double>::value )
    {
      sqrti = sqrt(origDiag[k]);
    }
    else
    {
      Iterate<BS>([&](auto l) LAMBDA_INLINE { sqrti[l] = sqrt(origDiag[k](l,l)); });
    }
  }

  // auto &addDiag = origDiag;
  Array<TV> addDiag(N);
  addDiag = 0;

  iterateOD([&](auto k, auto iterateOffDiag) LAMBDA_INLINE
  {
    if ( freeDOFs && !freeDOFs->Test(k))
      { return; }

    auto &sqrtK = scalSqrt[k];

    iterateOffDiag([&](auto const &col, auto const &val)
    {
      if constexpr( is_same<TM,double>::value )
      {
        addDiag[k] += fabs(val) / ( sqrtK * scalSqrt[col] );
          // add_diag[k] += fabs(rvs[j]);
          // add_diag[k] += fabs(rvs[j]) / ( diag[ris[j]] );
      }
      else
      {
        auto &sqrtJ = scalSqrt[col];

        Iterate<BS>([&](auto l) LAMBDA_INLINE
        {
          Iterate<BS>([&](auto m) LAMBDA_INLINE
          {
            addDiag[k](l.value) += fabs(val(l.value,m.value)) / ( sqrtK[l.value] * sqrtJ[m.value] );
          });
        });
      }
    });
  });

  MyAllReduceDofData (parDOFs, addDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  modDiag.SetSize(N);

  auto const isMaster = *dCCMap.GetMasterDOFs();

  // const double theta = 0.25;
  const double theta = 0.1;

  for (auto k : Range(N))
  {
    if ( !isMaster.Test(k) || ( freeDOFs && !freeDOFs->Test(k) ) )
    {
      modDiag[k] = 0;
      continue;
    }

    auto const &d  = origDiag[k];
    auto const &ad = addDiag[k];
    auto       &md = modDiag[k];

    if constexpr( is_same<TM, double>::value )
    {
      md = max(1.0, 0.51 * (1.0 + ad)) * d;
      // md = d;
    }
    else
    {
      double maxfac = 1.0;

      // Iterate<TMH>([&](auto i) { maxfac = max(maxfac, 0.5 * ( 1 + theta + ad(i.value) ) ); });
      Iterate<BS>([&](auto i) LAMBDA_INLINE
      {
        maxfac = max(maxfac, 0.51 * ( 1.0 + ad(i.value) ) );
      });

      md = maxfac * d;

      // md = d;
    }
  }
} // CalcHybridSmootherRDGItGeneric


template<class TM, class TSCAL, class TGET_D, class TIT_OD>
INLINE void
CalcHybridSmootherRDG(size_t        const &N,
                      DCCMap<TSCAL> const &dCCMap,
                      Array<TM>           &modDiag,
                      BitArray      const *freeDOFs,
                      TGET_D               getDiag,
                      TIT_OD               iterateOffDiag)
{

  auto iterateD = [&](auto lam) LAMBDA_INLINE
  {
    for (auto k : Range(N))
    {
      lam(k, getDiag(k));
    }
  };

  auto iterateOD = [&](auto lam) LAMBDA_INLINE
  {
    for (auto k : Range(N))
    {
      lam(k, [&](auto kernel) LAMBDA_INLINE { iterateOffDiag(k, kernel); });
    }
  };

  CalcHybridSmootherRDGItGeneric(N,
                                 dCCMap,
                                 modDiag,
                                 freeDOFs,
                                 iterateD,
                                 iterateOD);
} // CalcHybridSmootherRDG



template<class TM, class TSCAL, class TGET_D, class TIT_OD>
INLINE void
CalcHybridSmootherRDGV1(size_t        const &N,
                        DCCMap<TSCAL> const &dCCMap,
                        Array<TM>           &modDiag,
                        BitArray      const *freeDOFs,
                        TGET_D               getDiag,
                        TIT_OD               iterateOffDiag)
{
  static Timer t("CalcHybridSmootherRDG");
  RegionTimer rt(t);

  constexpr int BS = Height<TM>();

  auto const &parDOFs = *dCCMap.GetParallelDofs();

  modDiag.SetSize0();

  /** No need for adding to the diagonal **/
  if ( ( parDOFs.GetDistantProcs().Size() == 0 ) )
    { return; }

  typedef typename mat_traits<TM>::TV_COL TV;

  Array<TM> origDiag(N);

  for (auto k : Range(N))
    { origDiag[k] = getDiag(k); }

  MyAllReduceDofData (parDOFs, origDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  Array<TV> scalSqrt(N * BS);

  for (auto k : Range(N))
  {
    auto &sqrti = scalSqrt[k];

    if constexpr( is_same<TM,double>::value )
    {
      sqrti = sqrt(origDiag[k]);
    }
    else
    {
      Iterate<BS>([&](auto l) { sqrti[l] = sqrt(origDiag[k](l,l)); });
    }
  }

  // auto &addDiag = origDiag;
  Array<TV> addDiag(N);
  addDiag = 0;

  for (auto k : Range(N))
  {
    if ( freeDOFs && !freeDOFs->Test(k))
      { continue; }

    auto &sqrtK = scalSqrt[k];

    iterateOffDiag(k, [&](auto const &col, auto const &val)
    {
      if constexpr( is_same<TM,double>::value )
      {
        addDiag[k] += fabs(val) / ( sqrtK * scalSqrt[col] );
          // add_diag[k] += fabs(rvs[j]);
          // add_diag[k] += fabs(rvs[j]) / ( diag[ris[j]] );
      }
      else
      {
        auto &sqrtJ = scalSqrt[col];

        Iterate<BS>([&](auto l)
        {
          Iterate<BS>([&](auto m)
          {
            addDiag[k](l.value) += fabs(val(l.value,m.value)) / ( sqrtK[l.value] * sqrtJ[m.value] );
          });
        });
      }
    });
  }

  MyAllReduceDofData (parDOFs, addDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  modDiag.SetSize(N);

  auto const isMaster = *dCCMap.GetMasterDOFs();

  // const double theta = 0.25;
  const double theta = 0.1;

  for (auto k : Range(N))
  {
    if ( !isMaster.Test(k) || ( freeDOFs && !freeDOFs->Test(k) ) )
    {
      modDiag[k] = 0;
      continue;
    }

    auto const &d  = origDiag[k];
    auto const &ad = addDiag[k];
    auto       &md = modDiag[k];

    if constexpr( is_same<TM, double>::value )
    {
      md = max(1.0, 0.51 * (1.0 + ad)) * d;
      // md = d;
    }
    else
    {
      double maxfac = 1.0;

      // Iterate<TMH>([&](auto i) { maxfac = max(maxfac, 0.5 * ( 1 + theta + ad(i.value) ) ); });
      Iterate<BS>([&](auto i)
      {
        maxfac = max(maxfac, 0.51 * ( 1.0 + ad(i.value) ) );
      });

      md = maxfac * d;

      // md = d;
    }
  }
} // CalcHybridSmootherRDG

} // namespace amg

#endif // FILE_AMG_HYBRID_SMOOTHER_UTILS_HPP