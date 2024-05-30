#ifndef FILE_UTILS_DENSELA_HPP
#define FILE_UTILS_DENSELA_HPP

#include <base.hpp>
#include <utils.hpp>


#include <ngblas.hpp>

namespace ngbla
{

// ngbla::MatrixView<ngbla::Mat<2>, ngbla::RowMajor, ngbla::undefined_size, ngbla::undefined_size, long unsigned int>,
// ngbla::FlatVector<const ngbla::Vec<2, double> >,
// ngbla::VectorView<ngbla::Vec<2, double>, long unsigned int, std::integral_constant<int, 1> >)

  template <typename TM, class TA, class TB, typename FUNC>
  void T_NgGEMV (double s, BareSliceMatrix<TM,RowMajor> a, FlatVector<TA> x, FlatVector<TB> y, FUNC func) NETGEN_NOEXCEPT
  {
    for (size_t i = 0; i < y.Size(); i++)
      {
        TB sum = 0;
        for (size_t j = 0; j < x.Size(); j++)
          sum += a(i,j) * x(j);
        func(y(i), s*sum);
      }
  }
  template <typename TM, class TA, class TB, typename FUNC>
  void T_NgGEMV (double s, BareSliceMatrix<TM,ColMajor> a, FlatVector<TA> x, FlatVector<TB> y, FUNC func) NETGEN_NOEXCEPT
  {
    for (size_t i = 0; i < y.Size(); i++)
      {
        TB sum = 0;
        for (size_t j = 0; j < x.Size(); j++)
          sum += a(i,j) * x(j);
        func(y(i), s*sum);
      }
  }
  template <typename TM, class TA, class TB, typename FUNC>
  void T_NgGEMV (double s, BareSliceMatrix<TM,RowMajor> a, FlatVector<TA const> x, FlatVector<TB> y, FUNC func) NETGEN_NOEXCEPT
  {
    for (size_t i = 0; i < y.Size(); i++)
      {
        TB sum = 0;
        for (size_t j = 0; j < x.Size(); j++)
          sum += a(i,j) * x(j);
        func(y(i), s*sum);
      }
  }
  template <typename TM, class TA, class TB, typename FUNC>
  void T_NgGEMV (double s, BareSliceMatrix<TM,ColMajor> a, FlatVector<TA const> x, FlatVector<TB> y, FUNC func) NETGEN_NOEXCEPT
  {
    for (size_t i = 0; i < y.Size(); i++)
      {
        TB sum = 0;
        for (size_t j = 0; j < x.Size(); j++)
          sum += a(i,j) * x(j);
        func(y(i), s*sum);
      }
  }

  template <bool ADD, ORDERING ord, int N, int M, class TSA, class TSB, class TSC, class TSD>
  void NgGEMV (double s, MatrixView<Mat<N, M, double>,ord, TSA, TSB> a, FlatVector<Vec<M, double> const> x, VectorView<Vec<N, double>, TSC, TSD> y) NETGEN_NOEXCEPT
  {
    if constexpr(ADD)
    {
      T_NgGEMV (s, a, x, y, [](auto & y, auto sum) { y+=sum; });
    }
    else
    {
      T_NgGEMV (s, a, x, y, [](auto & y, auto sum) { y=sum; });
    }
  }

  template <bool ADD, ORDERING ord, int N, int M, class TSA, class TSB, class TSC, class TSD>
  void NgGEMV (double s, MatrixView<Mat<N, M, double>,ord, TSA, TSB> a, FlatVector<Vec<M, double>> x, VectorView<Vec<N, double>, TSC, TSD> y) NETGEN_NOEXCEPT
  {
    if constexpr(ADD)
    {
      T_NgGEMV (s, a, x, y, [](auto & y, auto sum) { y+=sum; });
    }
    else
    {
      T_NgGEMV (s, a, x, y, [](auto & y, auto sum) { y=sum; });
    }
  }
} // namespae ngbla

namespace amg
{

template<class T>
constexpr T RelZeroTol()
{
  if constexpr(is_same<T, double>::value)
  {
    return 1e-12;
  }
  else
  {
    return 1e-7;
  }
}

template<class T>
constexpr T AbsZeroTol()
{
  if constexpr(is_same<T, double>::value)
  {
    return 1e-20;
  }
  else
  {
    return 1e-14;
  }
}

template<class T>
constexpr T RelSmallTol()
{
  if constexpr(is_same<T, double>::value)
  {
    return 1e-10;
  }
  else
  {
    return 1e-5;
  }
}


template<class TSCAL>
INLINE void
LapackEigenValuesSymmetricLH (LocalHeap &lh,
                              ngbla::FlatMatrix<TSCAL> a,
                              ngbla::FlatVector<TSCAL> lami,
                              ngbla::FlatMatrix<TSCAL> evecs = ngbla::FlatMatrix<TSCAL>(0,0))
{
  static Timer t("LapackEigenValuesSymmetricLH");
  RegionTimer rt(t);

  char jobz, uplo = 'U';
  integer n = a.Height();
  integer lwork=(n+2)*n+1;

  // double* work = new double[lwork];
  FlatArray<TSCAL> workAr(lwork, lh);
  TSCAL *work = workAr.Data();
  integer info;

  TSCAL *matA;

  if ( evecs.Height() )
  {
    // eigenvectors are calculated
    evecs = a;
    jobz = 'V';
    matA = &evecs(0,0);
  }
  else
  {
    // only eigenvalues are calculated, matrix a is destroyed!!
    jobz = 'N';
    matA = &a(0,0);
  }

  if constexpr(std::is_same<TSCAL, double>::value)
  {
    dsyev_(&jobz, &uplo , &n , matA, &n, &lami(0), work, &lwork, &info);
  }
  else
  {
    ssyev_(&jobz, &uplo , &n , matA, &n, &lami(0), work, &lwork, &info);
  }

  if (n <= 0)
  {
    std::cout << " LapackEigenValuesSymmetricLH with n = " << n << "!" << endl;
  }

  // if (info)
  //   std::cerr << "LapackEigenValuesSymmetric, info = " << info << std::endl;

  if (info)
  {
    std::cout << "LapackEigenValuesSymmetric, info = " << info << std::endl;
  }
}

// a.Rows(..).Cols(..) += alpha * x
template<class TA, class TIND, int BS>
INLINE void
addTM(TA &a, TIND const &offi, TIND const &offj, double const & alpha, Mat<BS, BS, double> const &x)
{
  Iterate<BS>([&](auto ii) {
    Iterate<BS>([&](auto jj) {
      a(offi + ii.value, offj + jj.value) += alpha * x(ii.value, jj.value);
    });
  });
}

template<class TA, class TIND>
INLINE void
addTM(TA &a, TIND const &offi, TIND const &offj, double const & alpha, double const &x)
{
  a(offi, offj) += alpha * x;
}

// a.Rows(..).Cols(..) = alpha * x
template<class TA, class TIND, int BS>
INLINE void
setFromTM(TA &a, TIND const &offi, TIND const &offj, double const & alpha, Mat<BS, BS, double> const &x)
{
  Iterate<BS>([&](auto ii) {
    Iterate<BS>([&](auto jj) {
      a(offi + ii.value, offj + jj.value) = alpha * x(ii.value, jj.value);
    });
  });
}

template<class TA, class TIND>
INLINE void
setFromTM(TA &a, TIND const &offi, TIND const &offj, double const & alpha, double const &x)
{
  a(offi, offj) = alpha * x;
}

// a += alpha * x.Rows(..).Cols(..)
template<class TA, class TIND, int BS>
INLINE void
addToTM(Mat<BS, BS, double> &a, double const & alpha, TA const &x, TIND const &offi, TIND const &offj)
{
  Iterate<BS>([&](auto ii) {
    Iterate<BS>([&](auto jj) {
      a(ii.value, jj.value) += alpha * x(offi + ii.value, offj + jj.value);
    });
  });
}

template<class TA, class TIND>
INLINE void
addToTM(double &a, double const & alpha, TA const &x, TIND const &offi, TIND const &offj)
{
  a += alpha * x(offi, offj);
}

// a = alpha * x.Rows(..).Cols(..)
template<class TA, class TIND, int BS>
INLINE void
setTM(Mat<BS, BS, double> &a, double const & alpha, TA const &x, TIND const &offi, TIND const &offj)
{
  Iterate<BS>([&](auto ii) {
    Iterate<BS>([&](auto jj) {
      a(ii.value, jj.value) = alpha * x(offi + ii.value, offj + jj.value);
    });
  });
}

template<class TA, class TIND>
INLINE void
setTM(double &a, double const & alpha, TA const &x, TIND const &offi, TIND const &offj)
{
  a = alpha * x(offi, offj);
}

INLINE void TimedLapackEigenValuesSymmetric (ngbla::FlatMatrix<double> a, ngbla::FlatVector<double> lami,
                                             ngbla::FlatMatrix<double> evecs)
{
  static Timer t("LapackEigenValuesSymmetric"); RegionTimer rt(t);
  LapackEigenValuesSymmetric (a, lami, evecs);
}

// find lambdas such that Ax = lambda B x
INLINE std::tuple<double, double, double> DenseEquivTestAAA(ngbla::FlatMatrix<double> A, ngbla::FlatMatrix<double> B, ngcore::LocalHeap & lh, bool print = false)
{
  HeapReset hr(lh);

  int const N = A.Height();

  FlatMatrix<double> evecsA(N, N, lh);
  FlatVector<double> evalsA(N, lh);
  TimedLapackEigenValuesSymmetric(A, evalsA, evecsA);

  double eps = 1e-12 * evalsA(N-1);

  // scale eigenvectors with 1/sqrt(evals(k))
  FlatMatrix<double> sqAi(N, N, lh); sqAi = 0.0;
  int count_zero = 0;
  for (auto k : Range(N)) {
    if (evalsA(k) > eps)
    {
      double fac = 1.0 / sqrt(evalsA(k));
      // sqAi += fac * evecsA.Rows(k, k+1) * Trans(evecsA.Rows(k, k+1));
      for (auto l : Range(N))
        for (auto m : Range(N))
          { sqAi(l,m) += fac * evecsA(k, l) * evecsA(k, m); }
    }
    else
    {
      count_zero++;
    }
  }

  // sqrt(Ainv) * B * sqrt(Binv)
  FlatMatrix<double> AiBAi(N, N, lh); AiBAi = sqAi * B * sqAi;

  FlatMatrix<double> evecsABA(N, N, lh);
  FlatVector<double> evalsABA(N, lh);
  TimedLapackEigenValuesSymmetric(AiBAi, evalsABA, evecsABA);

  double min_nz = evalsABA(count_zero);

  double kappa = (min_nz == 0.0) ? std::numeric_limits<double>::infinity() : evalsABA(N-1) / min_nz;

  if (print)
  {
    std::cout << " DenseEqivTest, A kernel dim = " << count_zero
              << ", bounds: " << min_nz << "*A <= B <= " << evalsABA(N-1) << "*A, kappa = "
              << kappa << std::endl;

    std::cout << " min. evec of Ai_B_Ai w. lam = " << evalsABA(count_zero) << " = " << evecsABA.Row(count_zero) << endl;
    std::cout << " max. evec of Ai_B_Ai w. lam = " << evalsABA(N-1)        << " = " << evecsABA.Row(N-1)        << endl;
  }

  return std::make_tuple(min_nz, evalsABA(N-1), kappa);
}

template<int N>
INLINE double MEV (FlatMatrix<double> L, FlatMatrix<double> R)
{
  static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
  HeapReset hr(lh);

  FlatMatrix<double> evecsL(N, N, lh);
  FlatVector<double> evalsL(N, lh);
  TimedLapackEigenValuesSymmetric(L, evalsL, evecsL);

  double eps = 1e-12 * evalsL(N-1);

  // evals[first_nz:] are not zero
  int first_nz = N-1;
  for (auto k : Range(N))
    if (evalsL(k) > eps)
{ first_nz = min2(first_nz, k); }

  int M = N-first_nz; // dim of non-kernel space
  Matrix<double> projR(M,M), temp(M,N);

  if (M == N) {
    temp = evecsL * R;
    projR = temp * Trans(evecsL);
  }
  else {
    temp = evecsL.Rows(first_nz, N) * R;
    projR = temp * Trans(evecsL).Cols(first_nz, N);
  }

  FlatVector<double> inv_diags(M, lh);
  for (auto k : Range(M))
    { inv_diags[k] = 1.0 / sqrt(evalsL(first_nz+k)); }

  for (auto k : Range(M)) // TODO: scal evecs instead
    for (auto j : Range(M))
{ projR(k,j) *= inv_diags[k] * inv_diags[j]; }

  // FlatMatrix<double> evecsPR(M, M, lh);
  // FlatVector<double> evalsPR(M, lh);
  // TimedLapackEigenValuesSymmetric(projR, evalsPR, evecsPR);

  FlatMatrix<double> evecsPR(0, 0, lh);
  FlatVector<double> evalsPR(M, lh);
  TimedLapackEigenValuesSymmetric(projR, evalsPR, evecsPR);

  return evalsPR(0);
} // MEV

template<int IMIN, int H, int JMIN, int W, class TMAT>
class FlatMat : public MatExpr<FlatMat<IMIN, H, JMIN, W, TMAT>>
{
public:
  using TSCAL = typename mat_traits<TMAT>::TSCAL;
  using TELEM = typename mat_traits<TMAT>::TELEM;

  TMAT & basemat;

  FlatMat () = delete;
  FlatMat (const FlatMat & m) = default;

  static constexpr bool IsLinear() { return false; }
  // enum { IS_LINEAR = 0 };

  auto View() const { return *this; }
  auto ViewRW() { return *this; }
  tuple<size_t,size_t> Shape() const { return { H, W }; }

  INLINE FlatMat (const TMAT & m)
    : basemat(const_cast<TMAT&>(m))
  { ; }

  template<typename TB>
  INLINE FlatMat & operator= (const Expr<TB> & m)
  {
    Iterate<H>([&](auto i) {
  Iterate<W>([&](auto j) {
      basemat(IMIN + i.value, JMIN + j.value) = m.Spec()(i, j);
    });
});
    return *this;
  }

  template<typename TB>
  INLINE FlatMat & operator+= (const Expr<TB> & m)
  {
    Iterate<H>([&](auto i) {
        Iterate<W>([&](auto j) {
            basemat(IMIN + i.value, JMIN + j.value) += m.Spec()(i, j);
          });
      });
    return *this;
  }

  INLINE FlatMat & operator= (TSCAL s)
  {
    Iterate<H>([&](auto i) {
  Iterate<W>([&](auto j) {
      basemat(IMIN + i.value, JMIN + j.value) = s;
    });
});
    return *this;
  }

  INLINE TELEM & operator() (size_t i, size_t j) { return basemat(IMIN + i, JMIN + j); }
  INLINE TELEM & operator() (size_t i) { return (*this)(i/H, i%H); }

  INLINE const TELEM & operator() (size_t i, size_t j) const { return basemat(IMIN + i, JMIN + j); }
  INLINE const TELEM & operator() (size_t i) const { return (*this)(i/H, i%H); }

  static constexpr size_t Height () { return H; }
  static constexpr size_t Width  () { return W; }
}; // class FlatMat

/** Small workaround so we do not always need to specify "TMAT" **/
template<int IMIN, int H, int JMIN, int W, class TMAT>
FlatMat<IMIN, H, JMIN, W, TMAT> MakeFlatMat (const TMAT & m)
{ return FlatMat<IMIN,H,JMIN,W,TMAT>(m); }



template<class TSCAL>
INLINE TSCAL
calc_trace (FlatMatrix<TSCAL> x)
{
  TSCAL sum = 0;

  for (auto k : Range(x.Height()))
  {
    sum += x(k, k);
  }

  return sum;
} // calc_trace

template<class T, int W, class TSCAL>
INLINE T
calc_trace (FlatMatrixFixWidth<W,TSCAL> x)
{
  T sum = 0;
  Iterate<W>([&](auto k) { sum += x(k, k); });
  return sum;
} // calc_trace

template<int W, class TSCAL>
INLINE TSCAL
calc_trace (FlatMatrixFixWidth<W,TSCAL> x)
{
  return calc_trace<TSCAL>(x);
}

template<class T, int H, int W, class C>
INLINE T
calc_trace (Mat<H,W,C> const &x)
{
  T sum = 0;
  Iterate<H>([&](auto k) { sum += x(k, k); });
  return sum;
} // calc_trace

template<int A, int B, class TSCAL>
INLINE TSCAL
calc_trace (Mat<A,B,TSCAL> const &x)
{
  return calc_trace<TSCAL>(x);
}

INLINE double
calc_trace (double x)
{
  return x;
}

INLINE float
calc_trace (float x)
{
  return x;
}

template<class TSCAL, int N>
INLINE TSCAL
CalcAvgTrace (Mat<N, N, TSCAL> const &x)
{
  TSCAL trace = 0;

  Iterate<N>([&](auto i)
  {
    trace += x(i, i);
  });

  if constexpr(N > 1)
  {
    return trace / N;
  }
  else
  {
    return trace;
  }
}

template<class TSCAL>
INLINE TSCAL
CalcAvgTrace (FlatMatrix<TSCAL> x)
{
  TSCAL trace = 0;

  for (auto i : Range(x.Height()))
  {
    trace += x(i, i);
  }

  return trace / x.Height();
}

template<class TOUT, int N, class TSCAL,
         class TENABLE = std::enable_if_t<!std::is_same_v<TOUT, TSCAL>>>
INLINE TOUT
CalcAvgTrace (Mat<N, N, TSCAL> const &x)
{
  return TOUT(CalcAvgTrace(x));
}

template<class TSCAL, class TENABLE = std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE TSCAL
CalcAvgTrace(TSCAL const &x)
{
  return TSCAL(x);
}

template<int A, int B, class C> INLINE double fabsum (const Mat<A,B,C> & x)
{
  double sum = 0;
  for (auto k : Range(A))
      for (auto j : Range(B))
        { sum += fabs(x(k,j)); }
  return sum;
} // fabsum

INLINE double calc_trace (FlatMatrix<double> x)
{
  double sum = 0;
  for (auto k : Range(x.Height()))
      { sum += x(k,k); }
  return sum;
} // calc_trace

template<class TM>
INLINE void mat_to_scal (int n, FlatMatrix<TM> matb, FlatMatrix<double> mats)
{
  constexpr int H = Height<TM>(), W = Width<TM>();
  int row = 0, col = 0, kH = 0, jW = 0;
  for (auto k : Range(n)) {
    kH = k*H;
    for (auto j : Range(n)) {
      jW = j*W;
      Iterate<H>([&](auto kk) {
          Iterate<W>([&](auto jj) {
            mats(kH+kk, jW+jj) = matb(k, j)(kk,jj);
          });
      });
    }
  }
} // mat_to_scal

template<class TM>
INLINE void scal_to_mat (int n, FlatMatrix<double> mats, FlatMatrix<TM> matb)
{
  constexpr int H = Height<TM>(), W = Width<TM>();
  int row = 0, col = 0, kH = 0, jW = 0;
  for (auto k : Range(n)) {
    kH = k*H;
    for (auto j : Range(n)) {
      jW = j*W;
      Iterate<H>([&](auto kk) {
          Iterate<W>([&](auto jj) {
            matb(k, j)(kk,jj) = mats(kH+kk, jW+jj);
          });
      });
    }
  }
} // mat_to_scal


template<int N>
INLINE Mat<N,N,double> TripleProd (const Mat<N,N,double> & A, const Mat<N,N,double> & B, const Mat<N,N,double> & C)
{
  static Mat<N,N,double> X;
  X = B * C;
  return A * X;
}
INLINE double TripleProd (double A, double B, double C)
{ return A * B * C; }

template<int N>
INLINE void AddTripleProd (double fac, Mat<N,N,double> & out, const Mat<N,N,double> & A, const Mat<N,N,double> & B, const Mat<N,N,double> & C)
{
  static Mat<N,N,double> X;
  X = B * C;
  out += fac * A * X;
}
INLINE void AddTripleProd (double fac, double out, double A, double B, double C)
{ out += fac * A * B * C; }


template<int N>
INLINE Mat<N,N,double> AT_B_A (const Mat<N,N,double> & A, const Mat<N,N,double> & B)
{
  static Mat<N,N,double> X;
  X = B * A;
  return Trans(A) * X;
}
INLINE double AT_B_A (double A, double B)
{ return A * B * A; }

template<int N>
INLINE void Add_AT_B_A (double fac, Mat<N,N,double> & out, const Mat<N,N,double> & A, const Mat<N,N,double> & B)
{
  static Mat<N,N,double> X;
  X = B * A;
  out += Trans(A) * X;
}

INLINE void Add_AT_B_A (double fac, double & out, double A, double B)
{ out += fac * A * B * A; }

template<class TSCAL>
bool CheckForSSPD (FlatMatrix<TSCAL> A, LocalHeap & lh);

extern template bool CheckForSSPD<float>  (FlatMatrix<float>  A, LocalHeap &lh);
extern template bool CheckForSSPD<double> (FlatMatrix<double> A, LocalHeap &lh);

template<class TSCAL>
bool CheckForSPD (FlatMatrix<TSCAL> A, LocalHeap & lh);

extern template bool CheckForSPD<float>  (FlatMatrix<float>  A, LocalHeap &lh);
extern template bool CheckForSPD<double> (FlatMatrix<double> A, LocalHeap &lh);

template<class TSCAL>
void CalcPseudoInverseNew (FlatMatrix<TSCAL> mat, LocalHeap & lh);

extern template void CalcPseudoInverseNew<float>  (FlatMatrix<float>  mat, LocalHeap &lh);
extern template void CalcPseudoInverseNew<double> (FlatMatrix<double> mat, LocalHeap &lh);

template<int N, class TSCAL>
void CalcPseudoInverseNew (FlatMatrix<Mat<N, N, TSCAL>> mat, LocalHeap & lh)
{
  auto const H = mat.Height();
  auto const W = mat.Width();

  FlatMatrix<TSCAL> B(mat.Height() * N, mat.Width() * N, lh);

  for (auto K : Range(H))
  {
    auto const KOff = K * N;
    for (auto J : Range(W))
    {
      auto const JOff = J * N;
      Iterate<N>([&](auto const &k){
        Iterate<N>([&](auto const &j) {
          B(KOff + k.value, JOff + j.value) = mat(K, J)(k.value, j.value);
        });
      });
    }
  }

  CalcPseudoInverseNew(B, lh);

  for (auto K : Range(H))
  {
    auto const KOff = K * N;
    for (auto J : Range(W))
    {
      auto const JOff = J * N;
      Iterate<N>([&](auto const &k){
        Iterate<N>([&](auto const &j) {
          mat(K, J)(k.value, j.value) = B(KOff + k.value, JOff + j.value);
        });
      });
    }
  }
}

INLINE void CalcPseudoInverseNew (double &mat, LocalHeap &lh) { mat = 1.0/mat; }
INLINE void CalcPseudoInverseNew (float  &mat, LocalHeap &lh) { mat = 1.0/mat; }

INLINE void CalcPseudoInverseNew (double &mat) { mat = 1.0/mat; }
INLINE void CalcPseudoInverseNew (float  &mat) { mat = 1.0/mat; }

template<int N, class TSCAL>
INLINE void
CalcPseudoInverseNew (Mat<N, N, TSCAL> & mat, LocalHeap & lh)
{
  FlatMatrix<TSCAL> mat2(N, N, lh);
  mat2 = mat;
  CalcPseudoInverseNew(mat2, lh);
  mat = mat2;
} // CalcPseudoInverseNew


template<class TSCAL>
INLINE void
CalcPseudoInverseFM (FlatMatrix<TSCAL> & M, LocalHeap & lh)
{
  HeapReset hr(lh);
  const int N = M.Height();
  FlatMatrix<TSCAL> evecs(N, N, lh);
  FlatVector<TSCAL> evals(N, lh);
  TimedLapackEigenValuesSymmetric(M, evals, evecs);
  TSCAL tol = 0; for (auto v : evals) tol += v;
  tol = 1e-12 * tol; tol = max2(tol, 1e-15);
  int DK = 0; // dim kernel
  for (auto & v : evals) {
    if (v > tol)
      { v = 1/sqrt(v); }
    else {
      DK++;
      v = 0;
    }
  }
  int NS = N-DK;
  for (auto i : Range(N))
    for (auto j : Range(N))
      evecs(i,j) *= evals(i);
  if (DK > 0)
    { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
  else
    { M = Trans(evecs) * evecs; }
} // CalcPseudoInverseFM

INLINE double MIN_EV_HARM2 (double A, double B, double R) { return 0.5 * R * (A+B)/(A*B); }
template<int N>
INLINE double MIN_EV_HARM2 (const Mat<N,N,double> & A, const Mat<N,N,double> & B, const Mat<N,N,double> & aR)
{
  /**
     \alpha^{-1} (0.5 * A^{\dagger} + 0.5 * B^{\dagger})^{\dagger} \leq R
      SVD of LHS:
        (0.5 * A^{\dagger} + 0.5 * B^{\dagger})^{\dagger} = Q Sigma QT

      Then
        \alpha^{-1} Q Sigma QT \leq R
      We only care about non-kernel of LHS! So Q now non-kernel cols of Q.
        \alpha^{-1} QTQ Sigma QTQ \leq QTRQ
        \alpha^{-1} Sigma_small \leq QTRQ
    **/
  static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
  HeapReset hr(lh);

  /** A(A+B)^{-1}B = (A^{-1}+B^{-1})^{-1}**/
  FlatMatrix<double> L(N, N, lh);
  L = A + B;
  CalcPseudoInverseFM(L, lh); // Is CPI<3,3,6> valid in 3d ??
  FlatMatrix<double> SB(N, N, lh);
  SB = L * B;
  L = 2 * A * SB;

  /** Q, Sigma^{\dagger} **/
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  TimedLapackEigenValuesSymmetric(L, evals, evecs);
  double tol = 0;
  for (auto v : evals)
    { tol += fabs(v); }
  tol = max2(1e-12 * tol, 1e-15);
  int M = 0;
  for (auto k : Range(N))
    if (evals(k) > tol)
      { M++; }
  if (M == 0)
    { return 0; }
  const int NS = M; M = N - NS;
  for (auto k : Range(NS))
    { evals(M+k) = 1.0/sqrt(evals(M+k)); }
  auto Q = evecs.Rows(N-NS, N); // non-kernel space of LHS (condition is always fulfilled on kernel of LHS)

  /** QT R Q **/
  FlatMatrix<double> RQ(N, NS, lh), QTRQ(NS, NS, lh);
  RQ = aR * Trans(Q);
  QTRQ = Q * RQ;
  for (auto k : Range(NS))
    for (auto j : Range(NS))
      { QTRQ(k, j) *= evals(M+k)*evals(M+j); }

  FlatMatrix<double> evecs_small(NS, NS, lh);
  FlatVector<double> evals_small(NS, lh);
  TimedLapackEigenValuesSymmetric(QTRQ, evals_small, evecs_small);
  return evals_small(0);
}


template<int N, class T> INLINE void CalcPseudoInverse (T & m)
{
  // static Timer t("CalcPseudoInverse"); RegionTimer rt(t);
  // static Timer tl("CalcPseudoInverse - Lapck");

  static Matrix<double> M(N,N), evecs(N,N);
  static Vector<double> evals(N);
  M = m;
  // cout << "pseudo inv M: " << endl << M << endl;
  // tl.Start();
  TimedLapackEigenValuesSymmetric(M, evals, evecs);
  // tl.Stop();
  // cout << "pseudo inv evals: "; prow(evals); cout << endl;
  double tol = 0; for (auto v : evals) tol += v;
  tol = 1e-12 * tol; tol = max2(tol, 1e-15);
  // cout << "tol: " << tol << endl;
  for (auto & v : evals)
    v = (v > tol) ? 1/sqrt(v) : 0;
  // cout << "rescaled evals: "; prow(evals); cout << endl;
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      evecs(i.value,j.value) *= evals(i.value);
    });
  });
  // cout << "rescaled evecs: " << endl << evecs << endl;
  m = Trans(evecs) * evecs;
}


INLINE int
CalcPseudoInverseWithTolNonZeroBlock (FlatMatrix<double>  M,
                                      LocalHeap          &lh,
                                      double relTolR = 1e-12,
                                      double relTolZ = 1e-12)
{
  static Timer t("CalcPseudoInverseWithTolNonZeroBlock");
  RegionTimer rt(t);

  // TODO: extract non-zero block, call CPI only on that!

  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetricLH(lh, M, evals, evecs);

  // cout << " evals "; prow(evals); cout << endl;

  double avgEV = 0;
  for (auto v : evals)
    { avgEV += v; }
  avgEV = avgEV / N;

  double const tolR = max2(relTolR * avgEV, 1e-20);
  double const tolZ = max2(relTolZ * avgEV, 1e-20);

  int DK = 0; // dim kernel
  for (auto & v : evals)
  {
    if (v > tolZ)
    {
      double const effV = max(tolR, v);
      v = 1/sqrt(effV);
    }
    else
    {
      DK++;
      v = 0;
    }
  }
  // cout << " scaled evals "; prow(evals); cout << endl;
  int NS = N-DK;
  // cout << " N DK NS " << N << " " << DK << " " << NS << endl;

  for (auto i : Range(N))
    for (auto j : Range(N))
      { evecs(i,j) *= evals(i); }

  if (DK > 0)
    { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
  else
    { M = Trans(evecs) * evecs; }

  return NS;
  // cout << " done CPI FB for " << endl << M << endl;
} // CalcPseudoInverseWithTolNonZeroBlock

INLINE int
CalcPseudoInverseWithTol (FlatMatrix<double> & M,
                          LocalHeap & lh,
                          double relTolR = 1e-12,
                          double relTolZ = 1e-12)
{
  static Timer t("CalcPseudoInverseWithTol");
  RegionTimer rt(t);

  auto const N = M.Height();

  double avgEV = 0;
  for (auto k : Range(N))
    { avgEV += M(k,k); }
  avgEV = avgEV / N;

  // double const tolR = relTolR * avgEV;
  double const tolZ = relTolZ * avgEV;

  FlatArray<int> extNZeroRows(N, lh);

  int n = 0; // dim kernel
  for (auto k : Range(N)) {
    if (M(k,k) > tolZ)
    {
      extNZeroRows[n++] = k;
    }
  }

  if ( n == 0 )
  {
    return 0;
  }
  else if (n < N)
  {
    // cout << " w. tol = " << tol << " reduced inv " << N << " -> " << n << endl;
    // if (n == 0)
    // {
    //   cout << " BIG MAT WAS: " << endl << M << endl;
    // }

    FlatMatrix<double> smallM(n, n, lh);

    FlatArray<int> nZeroRows = extNZeroRows.Range(0, n);

    smallM = M.Rows(nZeroRows).Cols(nZeroRows);

    int rk = CalcPseudoInverseWithTolNonZeroBlock(smallM, lh, relTolR, relTolZ);

    auto allRows = makeSequence(N, lh);

    iterate_AC(allRows, nZeroRows, [&](ABC_KIND const &whereRow, auto const &row, auto const &smallRow)
    {
      if (whereRow == INTERSECTION)
      {
        iterate_AC(allRows, nZeroRows, [&](ABC_KIND const &whereCol, auto const &col, auto const &smallCol)
        {
          if ( whereCol == INTERSECTION )
          {
            M(row, col) = smallM(smallRow, smallCol);
          }
          else
          {
            M(row, col) = 0.0;
          }
        });
      }
      else
      {
        for (auto col : Range(N))
        {
          M(row, col) = 0.0;
        }
      }
    });

    return rk;
  }
  else
  {
    return CalcPseudoInverseWithTolNonZeroBlock(M, lh, relTolR, relTolZ);
  }

} // CalcPseudoInverseWithTol


template<int N>
INLINE void
ToFlat (FlatMatrix<Mat<N, N, double>> A, FlatMatrix<double> B)
{
  auto const H = A.Height();
  auto const W = A.Width();

  for (auto K : Range(H))
  {
    auto const KOff = K * N;
    for (auto J : Range(W))
    {
      auto const JOff = J * N;
      Iterate<N>([&](auto const &k){
        Iterate<N>([&](auto const &j) {
          B(KOff + k.value, JOff + j.value) = A(K, J)(k.value, j.value);
        });
      });
    }
  }
} // ToFlat


template<int N>
INLINE void
ToTM (FlatMatrix<double> A, FlatMatrix<Mat<N, N, double>> B)
{
  auto const H = B.Height();
  auto const W = B.Width();

  for (auto K : Range(H))
  {
    auto const KOff = K * N;
    for (auto J : Range(W))
    {
      auto const JOff = J * N;
      Iterate<N>([&](auto const &k){
        Iterate<N>([&](auto const &j) {
          B(K, J)(k.value, j.value) = A(KOff + k.value, JOff + j.value);
        });
      });
    }
  }
} // ToTM


template<int N>
INLINE void
ToFlat (Mat<N, N, double> const &A, FlatMatrix<double> B)
{
  Iterate<N>([&](auto const &k){
    Iterate<N>([&](auto const &j) {
      B(k.value, j.value) = A(k.value, j.value);
    });
  });
} // ToFlat


template<int N>
INLINE void
ToTM (FlatMatrix<double> A, Mat<N, N, double> &B)
{
  Iterate<N>([&](auto const &k) {
    Iterate<N>([&](auto const &j) {
      B(k.value, j.value) = A(k.value, j.value);
    });
  });
} // ToTM


template<int N>
INLINE void
CalcPseudoInverseWithTol (FlatMatrix<Mat<N, N, double>> mat, LocalHeap & lh, double relTolR = 12-12, double relTolZ = 12-12)
{
  auto const H = mat.Height();
  auto const W = mat.Width();

  FlatMatrix<double> B(mat.Height() * N, mat.Width() * N, lh);

  ToFlat(mat, B);

  CalcPseudoInverseWithTol(B, lh, relTolR, relTolZ);

  ToTM(B, mat);
} // CalcPseudoInverseWithTol

template<int N>
INLINE double MIN_EV_FG_ONE_SIDE2 (const Mat<N,N,double> & A, const Mat<N,N,double> & B,
            const Mat<N,N,double> & R)
{
  static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
  HeapReset hr(lh);

  FlatMatrix<double> evecs(N,N,lh);
  FlatVector<double> evals(N, lh);
  TimedLapackEigenValuesSymmetric(B, evals, evecs);

  double eps2 = 0; for (auto v : evals) { eps2 += v; } eps2 = 1e-12 * eps2;
  int first_nz = N-1;
  for (auto k : Range(N))
    if (evals(k) > eps2)
      { first_nz = min2(first_nz, k); }
  int M = first_nz; // probably 1 or zero most of the time

  static Mat<N,N,double> TMP, I_MINUS_STUFF, L;

  double mev = -1; // case M==N should return 0, not 1 I think [other one will return 1...]
  if(M == 0) // no B kernel
    { mev =  MEV<N>(A, R); }
  else if (M < N) {
    FlatMatrix<double> VtAV(M, M, lh);
    auto VK = evecs.Rows(0, M); // kernel-evecs ov B
    VtAV = VK * A * Trans(VK);
    Switch<N>(M, [&](auto ceM) {
      CalcPseudoInverse<ceM>(VtAV);
    });
    TMP = -1 * Trans(VK) * VtAV * VK;
    I_MINUS_STUFF = TMP * A;
    Iterate<N>([&](auto i) { I_MINUS_STUFF(i.value, i.value) += 1.0; });
    L = A * I_MINUS_STUFF;
    mev = MEV<N>(L, R);
  }
  else
    { mev = 0; }

  return mev;
}

INLINE double MIN_EV_FG2 (double A, double B, double R) { return R / sqrt(A*B); }

template<int N>
INLINE double MIN_EV_FG2 (const Mat<N,N,double> & A, const Mat<N,N,double> & B,
        const Mat<N,N,double> & R, bool prtms = false)
{
  static Timer t("EV_FG"); RegionTimer rt(t);

  double mev_a = fabs(MIN_EV_FG_ONE_SIDE2<N>(A, B, R)); // -eps

  double mev_b = fabs(MIN_EV_FG_ONE_SIDE2<N>(B, A, R)); // -eps

  return sqrt(mev_a * mev_b);
}

template<class T>
INLINE void CalcPseudoInverse_impl (FlatMatrix<double> & M, T & out, LocalHeap & lh)
{
  // static Timer t("CalcPseudoInverse_impl"); RegionTimer rt(t);
  // static Timer tl("CalcPseudoInverse - Lapck");
  const int N = M.Height();
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  TimedLapackEigenValuesSymmetric(M, evals, evecs);
  double tol = 0; for (auto v : evals) tol += v;
  tol = 1e-12 * tol; tol = max2(tol, 1e-15);
  int DK = 0; // dim kernel
  for (auto & v : evals) {
    if (v > tol)
      { v = 1/sqrt(v); }
    else {
      DK++;
      v = 0;
    }
  }
  int NS = N-DK;
  for (auto i : Range(N))
    for (auto j : Range(N))
      evecs(i,j) *= evals(i);
  if (DK > 0)
    { out = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
  else
    { out = Trans(evecs) * evecs; }
}


template<int N>
INLINE void CalcPseudoInverse (Mat<N, N, double> & M, LocalHeap & lh)
{
  // static Timer t("CalcPseudoInverse expr");
  HeapReset hr(lh);
  FlatMatrix<double> Mf(N,N,lh);
  Mf = M;
  CalcPseudoInverse_impl(Mf, M, lh);
}

template<> INLINE void CalcPseudoInverse<1> (Mat<1, 1, double> & M, LocalHeap & lh)
{
  CalcPseudoInverseNew(M(0,0), lh);
}

template<class TM>
INLINE void CalcHarmonicExt(FlatMatrix<TM> A,
                            FlatArray<int> relim,
                            FlatArray<int> rstay,
                            FlatMatrix<TM> hex,
                            LocalHeap &lh,
                            bool usePinv = false)
{
  FlatMatrix<TM> Aii(relim.Size(), relim.Size(), lh);
  Aii = A.Rows(relim).Cols(relim);
  if (usePinv)
    { CalcPseudoInverseNew(Aii); }
  else
    { CalcInverse(Aii); }
  hex = - Aii * A.Rows(relim).Cols(rstay);
}

template<int IMIN, int N, int NN> INLINE void RegTM (Mat<NN,NN,double> & m, double maxadd = -1)
{
  // static Timer t(string("RegTM<") + to_string(IMIN) + string(",") + to_string(3) + string(",") + to_string(6) + string(">")); RegionTimer rt(t);
  static_assert( (IMIN + N <= NN) , "ILLEGAL RegTM!!");
  static Matrix<double> M(N,N), evecs(N,N);
  static Vector<double> evals(N);
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      M(i.value, j.value) = m(IMIN+i.value, IMIN+j.value);
    });
  });
  TimedLapackEigenValuesSymmetric(M, evals, evecs);
  const double eps = max2(1e-15, 1e-12 * evals(N-1));
  double min_nzev = 0; int nzero = 0;
  for (auto k : Range(N))
    if (evals(k) > eps)
      { min_nzev = evals(k); break; }
    else
      { nzero++; }
  if (maxadd >= 0)
    { min_nzev = min(maxadd, min_nzev); }
  if (nzero < N) {
    for (auto l : Range(nzero)) {
      Iterate<N>([&](auto i) {
          Iterate<N>([&](auto j) {
            m(IMIN+i.value, IMIN+j.value) += min_nzev * evecs(l, i.value) * evecs(l, j.value);
          });
      });
    }
  }
  else {
    SetIdentity(m);
    if (maxadd >= 0)
      { m *= maxadd; }
  }
} // RegTM

template<int BS, class TSCAL, class TGETETR, class TLAM>
INLINE
void
CallOnNonZeroDiagonalBlock (int const &n, TGETETR mat, LocalHeap &lh, TLAM lam, FlatMatrix<double> flatMat = FlatMatrix<double>())
{
  // mat = expand(lam(compress(mat)))

  HeapReset hr(lh);

  constexpr double REL_EPS = RelZeroTol<TSCAL>();
  constexpr double ABS_EPS = AbsZeroTol<TSCAL>();

  int const nScal = n * BS;

  auto getEtr = [&](auto i, auto ii, auto j, auto jj) -> TSCAL
  {
    if constexpr(BS == 1)
    {
      return mat(i, j);
    }
    else
    {
      return mat(i, j)(ii, jj);
    }
  };

  auto setEtr = [&](auto i, auto ii, auto j, auto jj, TSCAL const &val) -> void
  {
    if constexpr(BS == 1)
    {
      mat(i, j) = val;
    }
    else
    {
      mat(i, j)(ii, jj) = val;
    }
  };

  double maxd = 0.0;
  for (auto k : Range(n))
  {
    Iterate<BS>([&](auto kk)
    {
      maxd=max(maxd, getEtr(k, kk, k, kk));
    });
  }

  double const thresh = max(ABS_EPS, REL_EPS * maxd);

  FlatArray<int> nZR(nScal, lh);
  FlatArray<int> nNZR(nScal, lh);
  FlatArray<int> compress( nScal, lh);

  int nZ = 0;
  int nNZ = 0;

  for (auto k : Range(n))
  {
    Iterate<BS>([&](auto kk)
    {
      int const scalRow = BS * k + kk.value;

      if ( getEtr(k, kk, k, kk) > thresh )
      {
        compress[scalRow] = nNZ;
        nNZR[nNZ++] = scalRow;
      }
      else
      {
        compress[scalRow] = -1;
        nZR[nZ++] = scalRow;
      }
    });
  }

  if constexpr(BS == 1)
  {
    if ( ( flatMat.Height() == nScal ) && ( nNZ == nScal ) )
    {
      lam(flatMat);
      return;
    }
  }

  if ( nNZ == 0 )
  {
    // I want identical zero in this case and not 1e-320 entries
    for (auto k : Range(n))
    {
      for (auto j : Range(n))
      {
        mat(k, j) = 0.0;
      }
    }
    return;
  }

  nZR.Assign(nZR.Range(0, nZ));
  nNZR.Assign(nNZR.Range(0, nNZ));

  FlatMatrix<TSCAL> smallA(nNZ, nNZ, lh);

  for (auto k : Range(n))
  {
    Iterate<BS>([&](auto kk)
    {
      int const scalRow  = k*BS + kk;
      int const smallRow = compress[scalRow];

      if ( smallRow != -1 )
      {
        for (auto j : Range(n))
        {
          Iterate<BS>([&](auto jj)
          {
            int const scalCol  = j*BS + jj;
            int const smallCol = compress[scalCol];

            if ( smallCol != -1 )
            {
              smallA(smallRow, smallCol) = getEtr(k, kk, j, jj);
            }
          });
        }
      }
    });
  }

  lam(smallA);

  for (auto k : Range(n))
  {
    Iterate<BS>([&](auto kk)
    {
      int const scalRow  = k*BS + kk;
      int const smallRow = compress[scalRow];

      if ( smallRow != -1 )
      {
        for (auto j : Range(n))
        {
          Iterate<BS>([&](auto jj)
          {
            int const scalCol  = j*BS + jj;
            int const smallCol = compress[scalCol];

            if ( smallCol != -1 )
            {
              setEtr(k, kk, j, jj, smallA(smallRow, smallCol));
            }
            else
            {
              setEtr(k, kk, j, jj, 0.0);
            }
          });
        }
      }
      else
      {
        for (auto j : Range(n))
        {
          Iterate<BS>([&](auto jj)
          {
            setEtr(k, kk, j, jj, 0.0);
          });
        }
      }
    });
  }

}


template<class TM, class TLAM>
INLINE
void
CallOnNonZeroDiagonalBlock (FlatMatrix<TM> mat, LocalHeap &lh, TLAM lam)
{
  FlatMatrix<TScal<TM>> flatMat;

  if constexpr(is_same_v<TM, TScal<TM>>)
  {
    flatMat.Assign(mat);
  }

  CallOnNonZeroDiagonalBlock<Height<TM>(), TScal<TM>>(mat.Height(),
                                           [&](auto i, auto j) -> TM& { return mat(i, j); },
                                           lh,
                                           lam,
                                           flatMat);
}

template<int N, class TSCAL, class TLAM>
INLINE
void
CallOnNonZeroDiagonalBlock (Mat<N, N, TSCAL> &mat, LocalHeap &lh, TLAM lam)
{
  FlatMatrix<TSCAL> flatMat(mat);

  CallOnNonZeroDiagonalBlock<1, TSCAL>(N,
                                [&](auto i, auto j) -> TSCAL& { return mat(i, j); },
                                lh,
                                lam,
                                flatMat);
}

template<class TLAM, class TSCAL, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE
void
CallOnNonZeroDiagonalBlock (TSCAL &mat, LocalHeap &lh, TLAM lam)
{
  lam(0, mat);
}

template<class TSCAL>
bool TryDirectInverse_Lapack (FlatMatrix<TSCAL> A, LocalHeap &lh);

extern template bool TryDirectInverse_Lapack<float>  (FlatMatrix<float>  A, LocalHeap &lh);
extern template bool TryDirectInverse_Lapack<double> (FlatMatrix<double> A, LocalHeap &lh);

template<class TSCAL>
bool TryDirectInverse_simple (FlatMatrix<TSCAL> A, LocalHeap &lh);

extern template bool TryDirectInverse_simple<float>  (FlatMatrix<float>  A, LocalHeap &lh);
extern template bool TryDirectInverse_simple<double> (FlatMatrix<double> A, LocalHeap &lh);

template<class TSCAL>
INLINE bool
TryDirectInverse (FlatMatrix<TSCAL> A, LocalHeap & lh)
{
  // static Timer t("TryDirectInverse"); RegionTimer rt(t);
  if (A.Height() >= 50)
    { return TryDirectInverse_Lapack(A, lh); }
  else
    { return TryDirectInverse_simple (A, lh); }
} // TryDirectInverse


/** Fallback inverse for singular matrices - via SVD **/
template<class TSCAL, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE void
CalcPseudoInverseWithTolNonZeroBlock (FlatMatrix<TSCAL>  M,
                                      LocalHeap         &lh,
                                      TSCAL const        relTol = RelZeroTol<TSCAL>())
{
  // TODO: extract non-zero block, call CPI only on that!

  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<TSCAL> evecs(N, N, lh);
  FlatVector<TSCAL> evals(N, lh);
  LapackEigenValuesSymmetricLH(lh, M, evals, evecs);

  // cout << " evals "; prow(evals); cout << endl;

  TSCAL tol = 0;
  for (auto v : evals)
    { tol += v; }
  tol = relTol * tol / N;
  tol = max2(tol, AbsZeroTol<TSCAL>());

  int DK = 0; // dim kernel
  for (auto & v : evals) {
    if (v > tol)
      { v = 1/sqrt(v); }
    else {
      DK++;
      v = 0;
    }
  }
  // cout << " scaled evals "; prow(evals); cout << endl;
  int NS = N-DK;
  // cout << " N DK NS " << N << " " << DK << " " << NS << endl;

  for (auto i : Range(N))
    for (auto j : Range(N))
      { evecs(i,j) *= evals(i); }

  if (DK > 0)
    { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
  else
    { M = Trans(evecs) * evecs; }
  // cout << " done CPI FB for " << endl << M << endl;
} // CalcPseudoInverseWithTol


template<class TSCAL, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE void
CalcPseudoInverseTryNormal (FlatMatrix<TSCAL>        scalarMat,
                            LocalHeap               &lh,
                            TSCAL             const  relTol = RelZeroTol<TSCAL>())
{
  if (!TryDirectInverse(scalarMat, lh))
  {
    CalcPseudoInverseWithTolNonZeroBlock(scalarMat, lh, relTol);
  }
}

template<class TM, class ENABLE=std::enable_if_t<!is_scalar_type<TM>::value>>
INLINE void
CalcPseudoInverseTryNormal (FlatMatrix<TM>        mat,
                            LocalHeap            &lh,
                            TScal<TM>      const  relTol = RelZeroTol<TScal<TM>>())
{
  CallOnNonZeroDiagonalBlock(mat, lh, [&](FlatMatrix<TScal<TM>> scalarMat)
  {
    if (!TryDirectInverse(scalarMat, lh))
    {
      CalcPseudoInverseWithTolNonZeroBlock(scalarMat, lh, relTol);
    }
  });
}

template<int N, class TSCAL>
INLINE void
CalcPseudoInverseTryNormal (Mat<N, N, TSCAL>       &mat,
                            LocalHeap              &lh,
                            TSCAL            const  relTol = RelZeroTol<TSCAL>())
{
  CallOnNonZeroDiagonalBlock(mat, lh, [&](FlatMatrix<TSCAL> scalarMat)
  {
    if (!TryDirectInverse(scalarMat, lh))
    {
      CalcPseudoInverseWithTolNonZeroBlock(scalarMat, lh, relTol);
    }
  });
}

template<class TSCAL, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE void
CalcPseudoInverseTryNormal (TSCAL &mat, LocalHeap &lh, TSCAL const relTol = RelZeroTol<TSCAL>())
{
  mat = ( std::fabs(mat) > AbsZeroTol<TSCAL>() ) ? 1.0 / mat : 0.0;
}



INLINE void
printEvals (FlatMatrix<double> & M, LocalHeap & lh)
{
  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(M, evals, evecs);

  cout << " evals "; prow(evals); cout << endl;

} // printEvals


template<int N>
INLINE void
printEvals (FlatMatrix<Mat<N, N, double>> mat, LocalHeap & lh)
{
  auto const H = mat.Height();
  auto const W = mat.Width();

  FlatMatrix<double> B(mat.Height() * N, mat.Width() * N, lh);

  ToFlat(mat, B);

  printEvals(B, lh);
} // printEvals


template<int N>
INLINE void
printEvals (Mat<N, N, double> mat, LocalHeap & lh)
{
  FlatMatrix<double> B(N, N, lh);

  Iterate<N>([&](auto const &k){
    Iterate<N>([&](auto const &j) {
      B(k.value, j.value) = mat(k.value, j.value);
    });
  });

  printEvals(B, lh);
} // printEvals

INLINE void
printEvals (double const &x, LocalHeap & lh)
{
  ;
} // printEvals
} // namespace amg

#endif // FILE_UTILSE_DENSELA_HPP
