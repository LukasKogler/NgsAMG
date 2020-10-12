#define FILE_AMG_BLA_CPP

#include "amg.hpp"
#include "ng_lapack.hpp"

namespace amg
{

  bool CheckForSPD (FlatMatrix<double> A, LocalHeap & lh)
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
    dpotrf_ (&uplo, &n, &A(0,0), &n, &info);
    if (info != 0)
      { cout << " check for spd, info = " << info << endl; }
    return info == 0;
  } // CheckForSPD


  bool CheckForSSPD (FlatMatrix<double> A, LocalHeap & lh)
  {
    static Timer t("CheckForSPD"); RegionTimer rt(t);
    /** can do this with dpstrf (cholesky for semi-positive definite) **/
    integer n = A.Height();
    if (n == 0)
      { return true; }
    /** a trivial case ... **/
    for (auto k : Range(n))
      if (A(k,k) < 0)
	{ return false; }
    integer info = 0;
    char uplo = 'U';
    FlatVector<integer> P(n, lh);
    integer rank;
    double tol = 1e-12 * calc_trace(A)/n;
    FlatArray<double> W(2*n, lh);
    dpstrf_ (&uplo, &n, &A(0,0), &n, &P(0), &rank, &tol, &W[0], &info);
    return info == 0;
  } // CheckForSSPD

} // namespace amg
