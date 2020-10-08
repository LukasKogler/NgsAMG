#define FILE_AMG_BLA_CPP

#include "amg.hpp"
#include "ng_lapack.hpp"

namespace amg
{

  bool CheckForSPD (FlatMatrix<double> A, LocalHeap & lh)
  {
    /** we can do this with dpotrf (cholesky for positive definite) **/
    integer n = A.Height();
    if (n == 0)
      { return true; }
    integer info;
    char uplo = 'U';
    dpotrf_ (&uplo, &n, &A(0,0), &n, &info);
    return info == 0;
  } // CheckForSPD


  bool CheckForSSPD (FlatMatrix<double> A, LocalHeap & lh)
  {
    /** can do this with dpstrf (cholesky for semi-positive definite) **/
    integer n = A.Height();
    if (n == 0)
      { return true; }
    integer info;
    char uplo = 'U';
    FlatVector<integer> P(n, lh);
    integer rank;
    double tol = 1e-12 * calc_trace(A)/n;
    FlatArray<double> W(2*n, lh);
    dpstrf_ (&uplo, &n, &A(0,0), &n, &P(0), &rank, &tol, &W[0], &info);
    return info == 0;
  } // CheckForSSPD

} // namespace amg
