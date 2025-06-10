#define FILE_HYBRID_SMOOTHER_CPP

#include <utils_arrays_tables.hpp>
#include <utils_io.hpp>
#include <reducetable.hpp>

#include "hybrid_smoother.hpp"
#include "hybrid_smoother_utils.hpp"

namespace amg
{



/** HybridSmoother **/


template<class TM>
HybridSmoother<TM>::
HybridSmoother (shared_ptr<HybridMatrix<TM>> _A,
                int  _numLocSteps,
                bool _commInThread,
                bool _overlapComm)
  : HybridBaseSmoother<TSCAL>(_A,
                              _numLocSteps,
                              _commInThread,
                              _overlapComm)
  , _hybridSpA(_A)
{
} // HybridSmoother (..)


template<class TM>
HybridSmoother<TM>::
HybridSmoother (shared_ptr<BaseMatrix> _A,
                int  _numLocSteps,
                bool _commInThread,
                bool _overlapComm)
  : HybridSmoother<TM>(make_shared<HybridMatrix<TM>>(_A),
                       _numLocSteps,
                       _commInThread,
                       _overlapComm)
{
} // HybridSmoother (..)

template<class TM>
INLINE
void
add_od_to_v (const TM & mat, typename mat_traits<TM>::TV_COL & vec)
{
  Iterate<ngbla::Height<TM>()>([&](auto i) LAMBDA_INLINE
  {
    Iterate<ngbla::Height<TM>()>([&](auto j) LAMBDA_INLINE
    {
      vec(i.value) += fabs(mat(i.value, j.value));
    });
  });
}

template<>
INLINE
void
add_od_to_v<double> (const double & v, double & w)
{
  w += fabs(v);
}



template<class TM>
Array<TM>
HybridSmoother<TM>::
CalcModDiag (shared_ptr<BitArray> free)
{
  auto &A = GetHybSparseA();

  auto const &M = *A.GetSpM();
  auto const &G = *A.GetSpG();

  Array<TM> mod_diag;

  CalcHybridSmootherRDG(A.Height(),
                        A.GetDCCMap(),
                        mod_diag,
                        free.get(),
                        [&](auto k) LAMBDA_INLINE -> TM const& { return M(k,k); },
                        [&](auto row, auto lam) LAMBDA_INLINE
                        {
                          auto ris = G.GetRowIndices(row);
                          auto rvs = G.GetRowValues(row);

                          for (auto j : Range(ris))
                          {
                            lam(ris[j], rvs[j]);
                          }
                        });

  return mod_diag;

//   auto spG = A.GetSpG();
//   if (spG == nullptr)
//     { return mod_diag; }

//   auto pardofs = A.GetParallelDofs();

//   if ( ( pardofs == nullptr ) || ( pardofs->GetDistantProcs().Size() == 0 ) )
//     { return mod_diag; }

//   auto const &M = *A.GetSpM();

//   typedef typename mat_traits<TM>::TV_COL TV;

//   constexpr auto TMH = ngbla::Height<TM>();

//   Array<TM> diag(A.Height());

//   for (auto k : Range(M.Height()))
//     { diag[k] = M(k,k); }

//   MyAllReduceDofData (*pardofs, diag,
//     [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

//   Array<TV> add_diag(A.Height()); add_diag = 0;

//   for (auto k : Range(A.Height()))
//   {
//     if (free && !free->Test(k))
//       { continue; }

//     auto ris = spG->GetRowIndices(k);
//     auto rvs = spG->GetRowValues(k);
//     double sqrti[TMH];

//     if constexpr( is_same<TM,double>::value )
//       { sqrti[0] = sqrt(diag[k]); }
//     else
//       { Iterate<TMH>([&](auto l) { sqrti[l] = sqrt(diag[k](l,l)); }); }

//     for (auto j : Range(rvs))	{
//       if constexpr(is_same<TM,double>::value)
//       {
//         auto sqrti = sqrt(diag[k]);
//         if (!free || free->Test(ris[j]))
//           add_diag[k] += fabs(rvs[j]) / ( sqrti * sqrt(diag[ris[j]]));
//           // add_diag[k] += fabs(rvs[j]);
//           // add_diag[k] += fabs(rvs[j]) / ( diag[ris[j]] );
//       }
//       else
//       {
//         Iterate<TMH>([&](auto l)
//         {
//           Iterate<TMH>([&](auto m)
//           {
//             add_diag[k](l.value) += fabs(rvs[j](l.value,m.value)) / (sqrti[l.value] * sqrt(diag[ris[j]](m.value,m.value)));
//           });
//         });
//       }
//     }
//   }

//   MyAllReduceDofData (*pardofs, add_diag,
//     [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

//   mod_diag.SetSize(A.Height());

//   auto ismaster = *A.GetDCCMap().GetMasterDOFs();

//   // const double theta = 0.25;
//   const double theta = 0.1;

//   for (auto k : Range(A.Height()))
//   {
//     if ( !ismaster.Test(k) || (free && !free->Test(k)) )
//       { mod_diag[k] = 0; continue; }

//     auto ad = add_diag[k];
//     auto d = diag[k];
//     auto & md = mod_diag[k];
// // { md = max( 1.0, 1.0 * ( 1 + theta + ad ) ) * d; }
//     double maxfac = 1.0;

//     if constexpr( is_same<TM, double>::value )
//     {
//       md = max(1.0, 0.51 * (1.0 + ad)) * d;
//       // md = d;
//     }
//     else
//     {
//       // Iterate<TMH>([&](auto i) { maxfac = max(maxfac, 0.5 * ( 1 + theta + ad(i.value) ) ); });
//       Iterate<TMH>([&](auto i) { maxfac = max(maxfac, 0.51 * ( 1.0 + ad(i.value) ) ); });
//       md = maxfac * d;
//       // md = d;
//     }
//   }

//   return mod_diag;
} // HybridSmoother::CalcModDiag


template<class TM>
void
HybridSmoother<TM>::
PrintTo (ostream & os, string prefix) const
{
  HybridBaseSmoother<TSCAL>::PrintTo(os, prefix);

  os << prefix << "  HybridSmoother, BS = " << ngbla::Height<TM>() << endl;
} // HybridSmoother::PrintTo


template<class TM>
size_t
HybridSmoother<TM>::
GetNOps () const
{
  return GetHybSparseA().NZE() * GetHybSparseA().EntrySize();
} // HybridSmoother::GetNOps


template<class TM>
size_t
HybridSmoother<TM>::
GetANZE () const
{
  return GetHybSparseA().NZE() * GetHybSparseA().EntrySize();
} // HybridSmoother::GetNOps

/** END HybridSmoother **/


/** HybridDISmoother **/

template<class TM>
HybridDISmoother<TM>::
HybridDISmoother (shared_ptr<BaseMatrix> _A,
                  shared_ptr<BitArray> _freedofs,
                  bool overlap,
                  bool NG_MPI_thread)
  : HybridSmoother<TM>(_A, overlap, NG_MPI_thread, 1)
{
  Array<TM> mod_diag = this->CalcModDiag(nullptr);

  auto &A = this->GetHybSparseA();

  if (A.Height())
  {
    const auto &spM = *A.GetSpM();

    shared_ptr<BitArray> freedofs = A.GetDCCMap().GetMasterDOFs();

    if (_freedofs)
    {
      BitArray fd = freedofs->And(*_freedofs);
      BitArray* fd2 = new BitArray(std::move(fd));
      freedofs = shared_ptr<BitArray>(fd2);
    }

    A.GetM()->SetInverseType(SPARSECHOLESKY);

    for (auto k : Range(mod_diag))
    {
      TM tmp = mod_diag[k];
      mod_diag[k] = spM(k,k);
      auto & dg = const_cast<TM&>(spM(k,k));
      dg = tmp;
    }

    loc_inv = A.GetM()->InverseMatrix(freedofs);

    for (auto k : Range(mod_diag))
    {
      auto & dg = const_cast<TM&>(spM(k,k));
      dg = mod_diag[k];
    }
  }
  else // dummy for rank 0!
  {
    loc_inv = A.GetSpM();
  }
} // HybridDISmoother(..)


template<class TM>
void
HybridDISmoother<TM>::
SmoothStageRHS (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  if (stage == SMOOTH_STAGE::LOC_PART_1)
  {
    x = (*loc_inv) * b;
  }
} // HybridDISmoother::SmoothStageRHS


template<class TM>
void
HybridDISmoother<TM>::
SmoothStageRes (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  if (stage == SMOOTH_STAGE::LOC_PART_1)
  {
    x += (*loc_inv) * res;
  }
} // HybridDISmoother::SmoothStageRes


template class HybridSmoother<double>;
template class HybridSmoother<Mat<2,2,double>>;
template class HybridSmoother<Mat<3,3,double>>;
#ifdef ELASTICITY
template class HybridSmoother<Mat<6,6,double>>;
#endif


template class HybridDISmoother<double>;
template class HybridDISmoother<Mat<2,2,double>>;
template class HybridDISmoother<Mat<3,3,double>>;
#ifdef ELASTICITY
template class HybridDISmoother<Mat<6,6,double>>;
#endif

/** END HybridDISmoother **/

} // namespace amg