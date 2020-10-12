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
    // throw Exception("THIS DOES NOT WORK!!");
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
    FlatMatrix<double> A2(n, n, lh); A2 = A; /** used to check for SSPD **/
    FlatArray<integer> P(n, lh);
    integer rank;
    double tol = 1e-12 * calc_trace(A)/n;
    FlatArray<double> W(2*n, lh);
    /** See lapack working note 161, section 7: THIS DOES NOT CHECK for positivce definiteness. The algorithm simply stops after
	"rank" steps. This can be either due to NEGATIVE OR ZERO evals. We cannot directly use this!! **/
    dpstrf_ (&uplo, &n, &A(0,0), &n, P.Data(), &rank, &tol, W.Data(), &info);
    if (info == 0)
      { return true; }
    else if (rank == 0) // zero matrix is SSPD...
      { return true; }
    else if (info < 0) // something went horribly wrong - should probably throw an exception...
      { return false; }
    // else
      // { cout << " check for sspd, info = " << info << ", rank = " << rank << " N = " << n << endl; }

    /** Check residuum ||A - P UT U PT|| to see if we have an inverse - then we have to be SSPD, or
	we dont - then we must be indefinite! **/
    integer n2 = rank;
    for (auto k : Range(n))
      { P[k] = P[k]-1; }
    for (auto k : Range(n2-1))
      for (auto j : Range(k+1, n2))
    	{ A(k, j) = 0; }

    double eps = 1e-10 * calc_trace(A2)/n;
    A2.Rows(P).Cols(P) -= A.Cols(0, n2) * Trans(A.Cols(0, n2));
    char norm = 'm'; // max norm
    double diffnorm = dlange_(&norm, &n, &n, &A2(0,0), &n, NULL);
    cout << " diffnorm: " << diffnorm << endl;
    cout << " diffnorm rel: " << diffnorm/eps << endl;

    return diffnorm < eps;
  } // CheckForSSPD

} // namespace amg
