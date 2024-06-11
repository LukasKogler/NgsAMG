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


// this way of instantiating templates has some issue on apple (and perhaps with clang too)
// #ifdef FILE_UTILS_SPARSEMM_CPP
//   #define SPARSE_MM_EXTERN
// #else
//   #define SPARSE_MM_EXTERN extern
// #endif

#ifndef FILE_UTILS_SPARSEMM_CPP

// template instantiations
 
namespace ngla
{

/**
 * NGSolve instantiates NxN, 1xN and Nx1 up to N=MAX_SYS_DIM.
 * The rest of the needed sparse matrices are compiled into the AMG library.
 */

template<int H, int W> struct IsMatrixCompiledTrait { static constexpr bool value = false; };
template<> struct IsMatrixCompiledTrait<1, 1> { static constexpr bool value = true; };

#define InstIMCTrait(N, M) \
  template<> struct IsMatrixCompiledTrait<N, M> { static constexpr bool value = true; }; \

#define InstIMCTrait2(N, M) \
  InstIMCTrait(N, M); \
  InstIMCTrait(M, N); \

#define InstSPMS(N,M) \
  extern template class SparseMatrixTM<Mat<N,M,double>>; \
  extern template class SparseMatrix<Mat<N,M,double>>; \
  InstIMCTrait(N, M); \

#define InstSPM2(N,M)\
  InstSPMS(N, M); \
  InstSPMS(M, N); \


// does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
// extern template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;

#if MAX_SYS_DIM < 2
  InstSPM2(1,2);
  InstSPMS(2,2);
#else
  InstIMCTrait2(1,2);
  InstIMCTrait(2,2);
#endif

#if MAX_SYS_DIM < 3
  InstSPM2(1,3);
  InstSPM2(2,3);
  InstSPMS(3,3);
#else
  InstSPM2(2,3);
  InstIMCTrait2(1,3);
  InstIMCTrait(3,3);
#endif

#ifdef ELASTICITY

// 1x4, 4x4, 1x5, 5x5 would be compiled into NGSolve with MAX_SYS_DIM large enough
// so to keep it simple instantiate them too for smaller MAX_SYS_DIM

#if MAX_SYS_DIM < 4
  InstSPM2(1,4);
  InstSPMS(4,4);
#else
  InstIMCTrait2(1,4);
  InstIMCTrait(4,4);
#endif

#if MAX_SYS_DIM < 5
  InstSPM2(1,5);
  InstSPMS(5,5);
#else
  InstIMCTrait2(1,5);
  InstIMCTrait(5,5);
#endif

#if MAX_SYS_DIM < 6
  InstSPM2(1,6);
  InstSPMS(6,6);
#else
  InstIMCTrait2(1,6);
  InstIMCTrait(6,6);
#endif

InstSPM2(3, 6);

#endif // ELASTICITY

#undef InstSPM2
#undef InstSPMS
#undef InstIMCTrait
#undef InstIMCTrait2

template<int H, int W>
constexpr bool IsMatrixCompiled()
{
  return IsMatrixCompiledTrait<H, W>::value;
}

} // namespace ngla


/** Sparse-Matrix transpose */

namespace amg
{
  template<int H, int W> struct IsTransMatCompiledTrait { static constexpr bool value = false; };

#define InstTransMat(H,W) \
  extern template shared_ptr<SparseMat<W, H>>	\
  TransposeSPMImpl<H,W>(SparseMatTM<H, W> const &A); \
  template<> struct IsTransMatCompiledTrait<H, W> { static constexpr bool value = true; }; \

#define InstTransMat2(N,M) \
  InstTransMat(M,N); \
  InstTransMat(N,M);


InstTransMat(1,1);

InstTransMat2(1,2);
InstTransMat(2,2);

InstTransMat2(1,3);
InstTransMat2(2,3);
InstTransMat(3,3);

#ifdef ELASTICITY
InstTransMat2(1,4);
InstTransMat(4,4);

InstTransMat2(1,5);
InstTransMat(5,5);

InstTransMat2(1,6);
InstTransMat2(3,6);
InstTransMat(6,6);
#endif //ELASTICITY



template<int H, int W>
static constexpr bool
IsTransMatCompiled()
{
  return IsTransMatCompiledTrait<H,W>::value;
}

#undef InstTransMat
#undef InstTransMat2
}

/** Sparse-Matrix multiplication */

namespace amg
{

template<int H, int N, int W> struct IsSparseMMCompiledTrait { static constexpr bool value = false; };

#define InstSparseMMTrait(H, N, W)\
  template<> struct IsSparseMMCompiledTrait<H, N, W> { static constexpr bool value = true; }; \

#define InstMultMat(A,B,C)						\
  extern template shared_ptr<SparseMat<A,C>>			\
  MatMultABImpl<A,B,C> (SparseMatTM<A, B> const &matAB, SparseMatTM<B, C> const &matBC); \
  InstMultMatUpdate(A,B,C); \
  InstSparseMMTrait(A,B,C); \

#define InstMultMatUpdate(A, B, C) \
  extern template void \
  MatMultABUpdateValsImpl<A,B,C> (SparseMatTM<A, B> const &mata, SparseMatTM<B, C> const &matb, SparseMatTM<A, C> &prod); \

#define InstProlMults(N,M) /* embedding NxN to MxM */	\
  InstMultMat(N,M,M); /* conctenate prols */		\
  InstMultMat(N,N,M); /* A * P */			\
  InstMultMat(M,N,M); /* PT * [A*P] */

#define InstProlMults2(N,M) \
  InstProlMults(N,M); \
  InstProlMults(M,N); \

/** [A \times B] * [B \times C] **/
InstMultMat(1,1,1);

InstMultMat(2,2,2);
InstProlMults2(1,2);

InstProlMults2(1,3);
InstProlMults2(2,3);
InstMultMat(3,3,3);

#ifdef ELASTICITY
InstProlMults2(1,4);
InstMultMat(4,4,4);

InstProlMults2(1,5);
InstMultMat(5,5,5);

InstProlMults2(1,6);
InstProlMults2(3,6);
InstMultMat(6,6,6);
#endif // ELASTICITY

#undef InstProlMults2
#undef InstProlMults
#undef InstMultMatUpdate
#undef InstMultMat
#undef InstSparseMMTrait

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

  static_assert(IsMatrixCompiled<W, H>(),
                "TransposeSPM not compiled!");

  static_assert(IsTransMatCompiled<H, W>(),
                "Uncompiled TransposeSPM specialization!" );

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

#endif // FILE_UTILS_SPARSEMM_CPP
#endif // FILE_AMG_SPMSTUFF_HPP
