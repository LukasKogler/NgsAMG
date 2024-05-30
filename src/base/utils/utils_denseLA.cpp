#include <base.hpp>
#include <core/profiler.hpp>
#include <ng_lapack.hpp>

#include "utils_denseLA.hpp"

namespace amg
{

template<class TSCAL>
bool
CheckForSPD (FlatMatrix<TSCAL> A, LocalHeap &lh)
{
  static Timer t("CheckForSPD"); RegionTimer rt(t);
  /** we can do this with dpotrf (cholesky for positive definite) **/
  integer n = A.Height();
  // {
  //   cout << " A 1 " << endl << A << endl;
  //   FlatMatrix<double> evecs(n,n,lh);
  //   FlatVector<double> evals(n,lh);
  //   LapackEigenValuesSymmetric(A, evals, evecs);
  //   cout << " evals = " << evals << endl;
  //   // cout << " evecs = " << endl << evecs;
  //   FlatMatrix<double> dgm(n,n,lh); dgm = 0;
  //   dgm(0,0) = -evals(0);
  //   for (auto k : Range(integer(1), n))
  // 	{ dgm(k,k) = evals(k); }
  //   A = Trans(evecs) * dgm * evecs;
  // }
  // {
  //   cout << " A 2 " << endl << A << endl;
  //   FlatMatrix<double> evecs(n,n,lh);
  //   FlatVector<double> evals(n,lh);
  //   LapackEigenValuesSymmetric(A, evals, evecs);
  //   cout << " evals = " << evals << endl;
  //   // cout << " evecs = " << endl << evecs;
  // }
  if (n == 0)
    { return true; }
  /** a trivial case ... **/
  for (auto k : Range(n))
    if (A(k,k) < 0)
      { return false; }
  integer info = 0;
  char uplo = 'U';
  if constexpr(std::is_same_v<TSCAL, double>)
  {
    dpotrf_ (&uplo, &n, &A(0,0), &n, &info);
  }
  else
  {
    spotrf_ (&uplo, &n, &A(0,0), &n, &info);
  }
  // if (info != 0)
    // { cout << " check for spd, info = " << info << endl; }
  return info == 0;
} // CheckForSPD

template bool CheckForSPD<float> (FlatMatrix<float>  A, LocalHeap &lh);
template bool CheckForSPD<double>(FlatMatrix<double> A, LocalHeap &lh);


template<class TSCAL>
bool CheckForSSPD (FlatMatrix<TSCAL> A, LocalHeap &lh)
{
  // static Timer t("CheckForSSPD"); RegionTimer rt(t);

  // std::cout << " CheckForSSPD, A = " << endl << A << endl;

  /** can do this with dpstrf (cholesky for semi-positive definite) **/
  integer n = A.Height();
  if (n == 0)
    { return true; }

  /** a trivial case ... **/
  TSCAL maxD = abs(A(0, 0));

  for (auto k : Range(integer(1), n))
  {
    maxD = max(maxD, abs(A(k, k)));
  }

  TSCAL tol = RelZeroTol<TSCAL>() * maxD;
  // TSCAL tol = RelZeroTol<TSCAL>() * CalcAvgTrace(A);

  // cout << " tol = " << tol << endl;

  for (auto k : Range(n))
  {
    // if (A(k,k) < -tol)
    //   { cout << " A " << k << " = " << A(k,k) << " -> !" << endl; }
    if (A(k,k) < -tol)
      { return false; }
  }

  integer info = 0;
  char uplo = 'U';

  /** need to save one copy for SSPD check **/
  FlatMatrix<TSCAL> A2(n, n, lh);
  A2 = A;

  FlatArray<integer> P(n, lh);
  integer rank;
  FlatArray<TSCAL> W(2*n, lh);

  /**
   * See lapack working note 161, section 7:
   *   THIS DOES NOT CHECK for positivce definiteness.
   *   The algorithm simply stops after "rank" steps.
   *   This can be either due to NEGATIVE OR ZERO evals,
   *   therefore successfull completion does NOT imply SSPD!
   */
  if constexpr(is_same<double, TSCAL>::value)
  {
    dpstrf_ (&uplo, &n, &A(0,0), &n, P.Data(), &rank, &tol, W.Data(), &info);
  }
  else
  {
    spstrf_ (&uplo, &n, &A(0,0), &n, P.Data(), &rank, &tol, W.Data(), &info);
  }

  // cout << "   -> info = " << info << endl;
  // cout << "   -> rank = " << rank << endl;

  if (info == 0)
    { return true; }
  else if (rank == 0) // zero matrix is SSPD...
    { return true; }
  else if (info < 0) // something went horribly wrong - should probably throw an exception...
    { return false; }


  /**
   * Check residuum ||A - P UT U PT|| to see if we have an inverse
   *  If we have one, the matrix MUST be SSPD, if it does not, it HAS
   *  to have a negative EV!
   */

  integer n2 = rank;
  for (auto k : Range(n))
    { P[k] = P[k]-1; }
  for (auto k : Range(n2-1))
    for (auto j : Range(k+1, n2))
    { A(k, j) = 0; }

  // cout << " zeroed A in : " << endl << A << endl;

  double eps = RelSmallTol<TSCAL>() * calc_trace(A2)/n;

  // cout << " eps " << endl << eps << endl;

  A2.Rows(P).Cols(P) -= A.Cols(0, n2) * Trans(A.Cols(0, n2));

  // cout << " diff: " << endl << A2 << endl;

  TSCAL diffnorm;

  char norm = 'm'; // max norm
  if constexpr(is_same<double, TSCAL>::value)
  {
    diffnorm = dlange_(&norm, &n, &n, &A2(0,0), &n, NULL);
  }
  else
  {
    diffnorm = slange_(&norm, &n, &n, &A2(0,0), &n, NULL);
  }

  // cout << " diffnorm: " << diffnorm << endl;
  // cout << " diffnorm rel: " << diffnorm/eps << endl;

  return diffnorm < eps;
} // CheckForSSPD

template bool CheckForSSPD<float> (FlatMatrix<float>  A, LocalHeap &lh);
template bool CheckForSSPD<double>(FlatMatrix<double> A, LocalHeap &lh);

/** Fallback inverse for singular matrices - via SVD **/
template<class TSCAL>
INLINE void
CalcPseudoInverseFB (FlatMatrix<TSCAL> M, LocalHeap &lh)
{
  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<TSCAL> evecs(N, N, lh);
  FlatVector<TSCAL> evals(N, lh);
  LapackEigenValuesSymmetric(M, evals, evecs);
  // cout << " evals "; prow(evals); cout << endl;
  TSCAL tol = 0; for (auto v : evals) tol += v;
  tol = RelZeroTol<TSCAL>() * tol / N; tol = max2(tol, 1e-15);

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
      evecs(i,j) *= evals(i);

  if (DK > 0)
    { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
  else
    { M = Trans(evecs) * evecs; }
  // cout << " done CPI FB for " << endl << M << endl;
} // CalcPseudoInverseFB

/** Copied from ngsolve/basiclinalg/ng_lapack.hpp, then modified if info is != 0 **/
template<class TSCAL>
INLINE void
CPI_TN_Lapack (FlatMatrix<TSCAL> A, LocalHeap & lh)
{
  integer n = A.Width();

  if (n == 0)
    { return; }

  FlatMatrix<TSCAL> a(n, n, lh); a = A;

  integer lda = a.Dist();
  integer info;
  char uplo = 'U';

  if constexpr(std::is_same_v<TSCAL, double>)
  {
    dpotrf_ (&uplo, &n, &a(0,0), &lda, &info);
  }
  else
  {
    spotrf_ (&uplo, &n, &a(0,0), &lda, &info);
  }

  if (info != 0)
    { CalcPseudoInverseFB(A, lh); return; }

  if constexpr(std::is_same_v<TSCAL, double>)
  {
    dpotri_ (&uplo, &n, &a(0,0), &lda, &info);
  }
  else
  {
    spotri_ (&uplo, &n, &a(0,0), &lda, &info);
  }

  if (info != 0)
    { CalcPseudoInverseFB(A, lh); return; }

  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++)
      { A(j,i) = a(i,j); A(i,j) = a(i,j); }
} // CPI_TN_Lapack

/** Copied from ngsolve/basiclinalg/calcinerse.cpp, then modified if mat singular **/
template<class TSCAL>
INLINE void
CPI_TN (FlatMatrix<TSCAL> A, LocalHeap &lh)
{
  int n = A.Height();
  if (n == 0)
    { return; }
  TSCAL eps = 0;
  for (int j = 0; j < n; j++)
    { eps = max(eps, A(j,j)); }
  eps = RelZeroTol<TSCAL>() * eps;

  FlatMatrix<TSCAL> inv(n, n, lh); inv = A;

  ngstd::ArrayMem<int,100> p(n);   // pivot-permutation

  for (int j = 0; j < n; j++)
    { p[j] = j; }

  bool isok = true;

  for (int j = 0; j < n; j++)
  {
    // pivot search
    TSCAL maxval = abs(inv(j,j));
    int r = j;
    for (int i = j+1; i < n; i++)
      if (abs (inv(j, i)) > maxval)
        {
          r = i;
          maxval = abs (inv(j, i));
        }

    TSCAL rest = 0.0;
    for (int i = j+1; i < n; i++)
      rest += abs(inv(r, i));

    // if ( (maxval < 1e-20*rest) || (rest < eps) )
      // { isok = false; break; }

    if ( maxval < max( AbsZeroTol<TSCAL>() * rest, eps) )
      { isok = false; break; }

    // exchange rows
    if (r > j)
      {
        for (int k = 0; k < n; k++)
          swap (inv(k, j), inv(k, r));
        swap (p[j], p[r]);
      }

    // transformation
    TSCAL hr;
    CalcInverse (inv(j,j), hr);

    for (int i = 0; i < n; i++)
    {
      TSCAL h = hr * inv(j, i);
      inv(j, i) = h;
    }

    inv(j,j) = hr;

    for (int k = 0; k < n; k++)
      if (k != j)
        {
          TSCAL help = inv(n*k+j);
          TSCAL h = help * hr;
          for (int i = 0; i < n; i++)
          {
            TSCAL h = help * inv(n*j+i);
            inv(n*k+i) -= h;
          }
          inv(k,j) = -h;
        }
  }

  if (isok)
  {
    // row exchange
    VectorMem<100,TSCAL> hv(n);
    for (int i = 0; i < n; i++)
    {
      for (int k = 0; k < n; k++) hv(p[k]) = inv(k, i);
      for (int k = 0; k < n; k++) inv(k, i) = hv(k);
    }
    A = inv;
  }
  else
    { CalcPseudoInverseFB(A, lh); }
} // CPI_TN

// not needed anymore ?!
// template<class TSCAL, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
// void CalcPseudoInverseTryNormal (FlatMatrix<TSCAL> A, LocalHeap &lh)
// {
//   // static Timer t("CalcPseudoInverseTryNormal"); RegionTimer rt(t);
//   if (A.Height() >= 50)
//     { CPI_TN_Lapack(A, lh); }
//   else
//     { CPI_TN (A, lh); }
// } // CalcPseudoInverseTryNormal


template<class TSCAL>
void
CalcPseudoInverseNew (FlatMatrix<TSCAL> mat, LocalHeap &lh)
{
  int N = mat.Height(), M = 0;

  TSCAL maxd = 0;
  for (auto k : Range(N))
    { maxd = max2(maxd, mat(k,k)); }

  TSCAL eps = RelZeroTol<TSCAL>() * maxd;

  for (auto k : Range(N))
    if (mat(k,k) > eps)
      { M++; }
  
  if (M == N)
  {
    CalcPseudoInverseTryNormal(mat, lh);
  }
  else if (M > 0)
  {
    FlatMatrix<TSCAL> small_mat(M, M, lh);
    FlatArray<int>    nzeros(M, lh);
    M = 0;
    for (auto k : Range(N))
    {
      if (mat(k,k) > eps)
        { nzeros[M++] = k; }
    }
    small_mat = mat.Rows(nzeros).Cols(nzeros);
    // cout << " CPO on reduces mat " << endl << small_mat << endl;
    CalcPseudoInverseTryNormal(small_mat, lh);
    mat.Rows(nzeros).Cols(nzeros) = small_mat;
  }
} // CalcPseudoInverseNew

template void CalcPseudoInverseNew<float>  (FlatMatrix<float>  mat, LocalHeap &lh);
template void CalcPseudoInverseNew<double> (FlatMatrix<double> mat, LocalHeap &lh);


/** Copied from ngsolve/basiclinalg/ng_lapack.hpp, then modified if info is != 0 **/
template<class TSCAL>
bool
TryDirectInverse_Lapack (FlatMatrix<TSCAL> A, LocalHeap & lh)
{
  integer n = A.Width();

  if (n == 0)
    { return true; }

  FlatMatrix<TSCAL> a(n, n, lh); a = A;
  integer lda = a.Dist();
  integer info;
  char uplo = 'U';

  if constexpr(std::is_same_v<TSCAL, double>)
  {
    dpotrf_ (&uplo, &n, &a(0,0), &lda, &info);
  }
  else
  {
    spotrf_ (&uplo, &n, &a(0,0), &lda, &info);
  }

  if (info != 0)
    { return false; }

  if constexpr(std::is_same_v<TSCAL, double>)
  {
    dpotri_ (&uplo, &n, &a(0,0), &lda, &info);
  }
  else
  {
    spotri_ (&uplo, &n, &a(0,0), &lda, &info);
  }

  if (info != 0)
    { return false; }

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j <= i; j++)
      { A(j,i) = a(i,j); A(i,j) = a(i,j); }
  }
  return true;
} // TryDirectInverse_Lapack

template bool TryDirectInverse_Lapack<float>  (FlatMatrix<float>  A, LocalHeap &lh);
template bool TryDirectInverse_Lapack<double> (FlatMatrix<double> A, LocalHeap &lh);

/** Copied from ngsolve/basiclinalg/calcinerse.cpp, then modified if mat singular **/
template<class TSCAL>
bool
TryDirectInverse_simple (FlatMatrix<TSCAL> A, LocalHeap & lh)
{
  int n = A.Height();
  if (n == 0)
    { return false; }

  TSCAL eps = 0;
  for (int j = 0; j < n; j++)
    { eps = max(eps, A(j,j)); }
  eps = RelZeroTol<TSCAL>() * eps;

  FlatMatrix<TSCAL> inv(n, n, lh); inv = A;
  ngstd::ArrayMem<int,100> p(n);   // pivot-permutation

  for (int j = 0; j < n; j++)
    { p[j] = j; }

  bool isok = true;

  for (int j = 0; j < n; j++)
  {
    // pivot search
    TSCAL maxval = abs(inv(j,j));
    int r = j;
    for (int i = j+1; i < n; i++)
    {
      if (abs (inv(j, i)) > maxval)
      {
        r = i;
        maxval = abs (inv(j, i));
      }
    }

    TSCAL rest = 0.0;
    for (int i = j+1; i < n; i++)
      rest += abs(inv(r, i));

    // if ( (maxval < 1e-20*rest) || (rest < eps) )
      // { isok = false; break; }

    if ( maxval < max( AbsZeroTol<TSCAL>() * rest, eps) )
      { isok = false; break; }

    // exchange rows
    if (r > j)
    {
      for (int k = 0; k < n; k++)
      {
        swap (inv(k, j), inv(k, r));
      }

      swap (p[j], p[r]);
    }

    // transformation
    TSCAL hr;
    CalcInverse (inv(j,j), hr);

    for (int i = 0; i < n; i++)
    {
      TSCAL h = hr * inv(j, i);
      inv(j, i) = h;
    }
    inv(j,j) = hr;

    for (int k = 0; k < n; k++)
    {
      if (k != j)
      {
        TSCAL help = inv(n*k+j);
        TSCAL h = help * hr;
        for (int i = 0; i < n; i++)
          {
            TSCAL h = help * inv(n*j+i);
            inv(n*k+i) -= h;
          }
        inv(k,j) = -h;
      }
    }
  }

  if (isok)
  {
    // row exchange
    VectorMem<100,TSCAL> hv(n);
    for (int i = 0; i < n; i++)
    {
      for (int k = 0; k < n; k++) hv(p[k]) = inv(k, i);
      for (int k = 0; k < n; k++) inv(k, i) = hv(k);
    }
    A = inv;
  }

  return isok;
} // TryDirectInverse_simple

template bool TryDirectInverse_simple<float>  (FlatMatrix<float>  A, LocalHeap &lh);
template bool TryDirectInverse_simple<double> (FlatMatrix<double> A, LocalHeap &lh);

} // namespace amg
