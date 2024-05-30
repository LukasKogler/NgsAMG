#ifndef FILE_AMG_SPMSTUFF_HPP
#define FILE_AMG_SPMSTUFF_HPP

#include <base.hpp>
#include <utils.hpp>

/**
 * Instantiation of SparseMatrix types not instantiated in NGSolve (non-square entries).
 * Sparse Matrix-Matrix multiplication.
 * Sparse Matrix Transpose not instantiated in NGSolve
*/
namespace amg
{

// Strip Mat<1,1,double> and Vec<1,double> -> double
template<class T> struct strip_mat { typedef T type; };
template<> struct strip_mat<Mat<1,1,double>> { typedef double type; };
template<> struct strip_mat<Mat<1,1,float>>  { typedef float  type; };

template<class T> struct strip_vec { typedef T type; };
template<> struct strip_vec<Vec<1, double>> { typedef double type; };
template<> struct strip_vec<Vec<1, float>>  { typedef float  type; };

// BS -> stripped Mat/Vec
template<int BSR, int BSC, class TSCAL=double>
using StripTM = typename strip_mat<Mat<BSR, BSC, TSCAL>>::type;

template<int BS, class TSCAL=double>
using StripVec = typename strip_vec<Vec<BS, TSCAL>>::type;

// Strip SparseMatrix<Mat<...>,..> -> SparseMatrix<double>
// template<class TM> using stripped_spm_tm =  SparseMatrixTM<typename strip_mat<TM>::type>;

template<class TM> struct stripped_spm_tm_struct                    { typedef SparseMatrixTM<TM>     type; };
template<>         struct stripped_spm_tm_struct<Mat<1, 1, double>> { typedef SparseMatrixTM<double> type; };
template<class TM>  using stripped_spm_tm = typename stripped_spm_tm_struct<TM>::type;
template<int H, int W> using SparseMatTM = stripped_spm_tm<StripTM<H, W>>;

template<class TM> struct stripped_spm_struct                    { typedef SparseMatrix<TM>     type; };
template<>         struct stripped_spm_struct<Mat<1, 1, double>> { typedef SparseMatrix<double> type; };
template<class TM>  using stripped_spm = typename stripped_spm_struct<TM>::type;
template<int H, int W> using SparseMat = stripped_spm<StripTM<H, W>>;

template<int H, int W>
INLINE SparseMatTM<H, W> const&
TMRef(SparseMatTM<H, W> const &A)
{
  return A;
}

//  Matrix Transpose
template<class TM>
using trans_mat = typename strip_mat<Mat<Width<TM>(), Height<TM>(), typename mat_traits<TM>::TSCAL>>::type;
template<class TM>
using trans_spm_tm = SparseMatrixTM<trans_mat<typename TM::TENTRY>>;
template<class TM>
using trans_spm = SparseMatrix<trans_mat<typename TM::TENTRY>>;

template<int H, int W>
shared_ptr<SparseMat<W, H>>
TransposeSPMImpl (SparseMatTM<H, W> const &mat);


// Matrix Multiplication
template<class TSA, class TSB> struct mult_scal { typedef void type; };
template<> struct mult_scal<double, double> { typedef double type; };
template<class TMA, class TMB>
using mult_mat = typename strip_mat<Mat<Height<TMA>(), Width<TMB>(),
          typename mult_scal<typename mat_traits<TMA>::TSCAL, typename mat_traits<TMB>::TSCAL>::type>>::type;
template<class TMA, class TMB>
using mult_spm_tm = stripped_spm_tm<mult_mat<typename TMA::TENTRY, typename TMB::TENTRY>>;

template<int A, int B, int C>
shared_ptr<SparseMat<A, C>>
MatMultABImpl (SparseMatTM<A, B> const &matAB, SparseMatTM<B, C> const &matBC);

template <int A, int B, int C>
void
MatMultABUpdateValsImpl (SparseMatTM<A, B> const &mata,
                         SparseMatTM<B, C> const &matb,
                         SparseMatTM<A, C>       &prod);



template<class TM> struct TM_OF_SPM { typedef typename std::remove_reference<typename std::result_of<TM(int, int)>::type >::type type; };
template<> struct TM_OF_SPM<SparseMatrix<double>> { typedef double type; };

template<class TSPM> constexpr int EntryHeight () { return Height<typename TM_OF_SPM<TSPM>::type>(); }
template<class TSPM> constexpr int EntryWidth ()  { return Width<typename TM_OF_SPM<TSPM>::type>(); }

INLINE Timer<TTracing, TTiming>& timer_hack_restrictspm2 () { static Timer t("RestrictMatrix2"); return t; }

template <int H, int W>
INLINE shared_ptr<SparseMat<W, W>>
RestrictMatrix (SparseMatTM<W, H> const &PT,
                SparseMatTM<H, H> const &A,
                SparseMatTM<H, W> const P)
{
  RegionTimer rt(timer_hack_restrictspm2());
  // I think (PTA)*P is faster than PT * (AP):
  //     PTA: few  rows, many cols
  //     AP : many rows, few  cols
  // TODO: benchmark !!
  // auto AP   = MatMultABImpl<H, H, W>(A, P);
  // auto PTAP = MatMultABImpl<W, H, W>(PT, *AP);
  auto PTA  = MatMultABImpl<W, H, H>(PT, A);
  auto PTAP = MatMultABImpl<W, H, W>(*PTA, P);
  return PTAP;
}

INLINE Timer<TTracing, TTiming>& timer_hack_restrictspm1 () { static Timer t("RestrictMatrix1"); return t; }
template <int H, int W>
INLINE shared_ptr<SparseMat<W, W>>
RestrictMatrixTM (SparseMatTM<H, H> const &A,
                  SparseMatTM<H, W> const P)
{
  RegionTimer rt(timer_hack_restrictspm1());
  auto PT = TransposeSPMImpl(P);
  return RestrictMatrixTM<H, W>(*PT, A, P);
}

} // namespace amg

/**
 * NGSolve only has NxN, 1xN and Nx1 up to N=MAX_SYS_DIM.
 * The rest of the needed sparse matrices are compiled into the AMG library.
 */

#ifdef FILE_UTILS_SPARSEMM_CPP
#define SPARSE_MM_EXTERN
#else
#define SPARSE_MM_EXTERN extern
#endif

namespace ngla
{

template<int H, int W> struct IsMatrixCompiledTrait { static constexpr bool value = false; };
template<> struct IsMatrixCompiledTrait<1, 1> { static constexpr bool value = true; };

#define InstIMCTrait(N, M)\
  template<> struct IsMatrixCompiledTrait<N, M> { static constexpr bool value = true; }; \

#define InstIMCTrait2(N, M)\
  InstIMCTrait(N, M); \
  InstIMCTrait(M, N); \

#define InstSPMS(N,M)				  \
  SPARSE_MM_EXTERN template class SparseMatrixTM<Mat<N,M,double>>; \
  SPARSE_MM_EXTERN template class SparseMatrix<Mat<N,M,double>>; \
  InstIMCTrait(N, M);\

#define InstSPM2(N,M)\
  InstSPMS(N, M); \
  InstSPMS(M, N);\

  // this does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
  // EXTERN template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;

#if MAX_SYS_DIM < 2
  InstSPM2(1,2);
  InstSPMS(2,2);
#else
  InstIMCTrait2(1, 2);
  InstIMCTrait(2, 2);
#endif

#if MAX_SYS_DIM < 3
  InstSPM2(1,3);
  InstSPM2(2,3);
  InstSPMS(3,3);
#else
  InstIMCTrait2(1,3);
  InstIMCTrait(3,3);
  InstSPM2(2,3);
#endif

#ifdef ELASTICITY
#if MAX_SYS_DIM < 6
  InstSPM2(1,6);
  InstSPMS(6,6);
#else
  InstIMCTrait2(1,6);
  InstIMCTrait(6,6);
  InstSPM2(3, 6);
#endif
#endif

// some extra instantiations to fix missing symbols with ELASTICITY/STOKES,
// I am not sure why they are even there
InstSPMS(2,6);
InstSPMS(6,2);
InstSPMS(5,6);
InstSPMS(6,5);

#undef InstIMCTrait
#undef InstSparseMMTrait
#undef InstIMCTrait2
} // namespace ngla


/** Sparse-Matrix transpose */

namespace amg
{

#define InstTransMat(H,W)						\
  SPARSE_MM_EXTERN template shared_ptr<SparseMat<W, H>>	\
  TransposeSPMImpl<H,W>(SparseMatTM<H, W> const &A); \


  /** [A \times B] Transpose **/
  InstTransMat(1,1);
  InstTransMat(1,2);
  InstTransMat(2,1);
  InstTransMat(2,2);
  InstTransMat(3,3);
  InstTransMat(1,3);
  InstTransMat(3,1);
  InstTransMat(2,3);
#ifdef ELASTICITY
  InstTransMat(1,6);
  InstTransMat(3,6);
  InstTransMat(6,6);
#endif //ELASTICITY

// some extra instantiations to fix missing symbols with ELASTICITY/STOKES,
// I am not sure why they are even there
InstTransMat(1,4);
InstTransMat(4,1);
InstTransMat(1,5);
InstTransMat(5,1);
InstTransMat(3,2);
InstTransMat(6,2);
InstTransMat(6,5);
InstTransMat(5,6);


#undef InstTransMat
}


/** Sparse-Matrix multiplication */

namespace amg
{

template<int H, int N, int W> struct IsSparseMMCompiledTrait { static constexpr bool value = false; };

#define InstSparseMMTrait(H, N, W)\
  template<> struct IsSparseMMCompiledTrait<H, N, W> { static constexpr bool value = true; }; \

#define InstMultMatUpdate(A, B, C) \
  SPARSE_MM_EXTERN template void \
  MatMultABUpdateValsImpl<A,B,C> (SparseMatTM<A, B> const &mata, SparseMatTM<B, C> const &matb, SparseMatTM<A, C> &prod); \

#define InstMultMat(A,B,C)						\
  SPARSE_MM_EXTERN template shared_ptr<SparseMat<A,C>>			\
  MatMultABImpl<A,B,C> (SparseMatTM<A, B> const &matAB, SparseMatTM<B, C> const &matBC); \
  InstMultMatUpdate(A,B,C); \
  InstSparseMMTrait(A,B,C); \

#define InstEmbedMults(N,M) /* embedding NxN to MxM */	\
  InstMultMat(N,M,M); /* conctenate prols */		\
  InstMultMat(N,N,M); /* A * P */			\
  InstMultMat(M,N,M); /* PT * [A*P] */

  /** [A \times B] * [B \times C] **/
  InstMultMat(1,1,1);
  InstMultMat(2,2,2);
  InstMultMat(3,3,3);
  InstEmbedMults(1,2);
  InstEmbedMults(2,1);
  InstEmbedMults(1,3);
  InstEmbedMults(3,1);
  InstEmbedMults(2,3);
#ifdef ELASTICITY
  InstMultMat(6,6,6);
  InstEmbedMults(1,6);
  InstEmbedMults(2,6);
  InstEmbedMults(3,6);
#endif // ELASTICITY

// some extra instantiations to fix missing symbols with ELASTICITY/STOKES,
// I am not sure why they are even there
InstMultMat(1,4,1);
InstMultMat(1,5,1);
InstMultMat(1,6,1);
InstMultMat(2,3,2);
InstMultMat(2,6,2);
InstMultMat(3,2,1);
InstMultMat(3,2,2);
InstMultMat(3,3,2);
InstMultMat(3,6,3);
InstMultMat(4,1,1);
InstMultMat(4,4,1);
InstMultMat(4,4,4);
InstMultMat(5,1,1);
InstMultMat(5,5,1);
InstMultMat(5,5,5);
InstMultMat(5,5,6);
InstMultMat(5,6,5);
InstMultMat(5,6,6);
InstMultMat(6,1,1);
InstMultMat(6,2,1);
InstMultMat(6,2,2);
InstMultMat(6,3,3);
InstMultMat(6,6,1);
InstMultMat(6,6,2);
InstMultMat(6,6,3);
InstMultMat(6,6,5);
InstMultMat(6,5,5);
InstMultMat(6,5,6);
InstEmbedMults(1,4);
InstEmbedMults(1,5);

#undef InstSparseMMTrait
#undef InstMultMat
#undef InstEmbedMults

template<int H, int W>
constexpr bool IsMatrixCompiled()
{
  return IsMatrixCompiledTrait<H, W>::value;
}

template<int H, int N, int W>
constexpr bool IsSparseMMCompiled()
{
  return IsSparseMMCompiledTrait<H, N, W>::value;
}

} // namespace amg


namespace amg
{

template<class TA>
INLINE auto
TransposeSPM (TA const &A)
{
  static constexpr int H = Height<typename TA::TENTRY>();
  static constexpr int W = Width<typename TA::TENTRY>();

  return TransposeSPMImpl<H,W>(TMRef<H,W>(A));
} // TransposeSPM


template<class TA, class TB>
INLINE auto
MatMultAB (TA const &A, TB const &B)
{
  static constexpr int HA = Height<typename TA::TENTRY>();
  static constexpr int WA = Width<typename TA::TENTRY>();

  static constexpr int HB = Height<typename TB::TENTRY>();
  static constexpr int WB = Width<typename TB::TENTRY>();

  static_assert(WA == HB,
                "ILLEGAL MatMultAB specialization!" );

  static_assert(IsSparseMMCompiled<HA, WA, WB>(),
                "Uncompiled MatMultAB specialization!" );

  return MatMultABImpl<HA, WA, WB>(TMRef<HA, WA>(A), TMRef<HB, WB>(B));
} // MatMultAB


template<class TA, class TB, class TC>
void
MatMultABUpdateVals (TA const &A, TB const &B, TC &C)
{
  static constexpr int HA = Height<typename TA::TENTRY>();
  static constexpr int WA = Width<typename TA::TENTRY>();

  static constexpr int HB = Height<typename TB::TENTRY>();
  static constexpr int WB = Width<typename TB::TENTRY>();

  static_assert(WA == HB,
                "ILLEGAL MatMultABUpdateVals specialization!" );

  static_assert(IsSparseMMCompiled<HA, WA, WB>(),
                "Uncompiled MatMultABUpdateVals specialization!" );

  MatMultABUpdateValsImpl<HA, WA, WB>(TMRef<HA, WA>(A), TMRef<HB, WB>(B), C);
} // MatMultABUpdateVals

} // namespace amg
#endif
