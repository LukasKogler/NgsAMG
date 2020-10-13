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
    // if (info != 0)
      // { cout << " check for spd, info = " << info << endl; }
    return info == 0;
  } // CheckForSPD


  bool CheckForSSPD (FlatMatrix<double> A, LocalHeap & lh)
  {
    // throw Exception("THIS DOES NOT WORK!!");
    static Timer t("CheckForSSPD"); RegionTimer rt(t);
    /** can do this with dpstrf (cholesky for semi-positive definite) **/
    integer n = A.Height();
    if (n == 0)
      { return true; }
    /** a trivial case ... **/
    double tol = 1e-12 * calc_trace(A)/n;
    for (auto k : Range(n))
      if (A(k,k) < -tol)
	{ return false; }
    integer info = 0;
    char uplo = 'U';
    FlatMatrix<double> A2(n, n, lh); A2 = A; /** used to check for SSPD **/
    FlatArray<integer> P(n, lh);
    integer rank;
    FlatArray<double> W(2*n, lh);
    /** See lapack working note 161, section 7: THIS DOES NOT CHECK for positivce definiteness. The algorithm simply stops after
	"rank" steps. This can be either due to NEGATIVE OR ZERO evals. We cannot directly use this!! **/
    // cout << " A in : " << endl << A << endl;
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
    // cout << " zeroed A in : " << endl << A << endl;

    double eps = 1e-10 * calc_trace(A2)/n;
    A2.Rows(P).Cols(P) -= A.Cols(0, n2) * Trans(A.Cols(0, n2));
    char norm = 'm'; // max norm
    double diffnorm = dlange_(&norm, &n, &n, &A2(0,0), &n, NULL);
    // cout << " diff: " << endl << A2 << endl;
    // cout << " diffnorm: " << diffnorm << endl;
    // cout << " diffnorm rel: " << diffnorm/eps << endl;

    return diffnorm < eps;
  } // CheckForSSPD


  /** Fallback inverse for singular matrices - via SVD **/
  INLINE void CalcPseudoInverseFB (FlatMatrix<double> & M, LocalHeap & lh)
  {
    // cout << " CPI FB for " << endl << M << endl;
    // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
    const int N = M.Height();
    FlatMatrix<double> evecs(N, N, lh);
    FlatVector<double> evals(N, lh);
    LapackEigenValuesSymmetric(M, evals, evecs);
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
      { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
    else
      { M = Trans(evecs) * evecs; }
    // cout << " done CPI FB for " << endl << M << endl;
  }

  /** Copied from ngsolve/basiclinalg/ng_lapack.hpp, then modified if info is != 0 **/
  INLINE void CPI_TN_Lapack (FlatMatrix<double> A, LocalHeap & lh)
  {
    integer n = A.Width();
    if (n == 0)
      { return; }
    FlatMatrix<double> a(n, n, lh); a = A;
    integer lda = a.Dist();
    integer info;
    char uplo = 'U';
    dpotrf_ (&uplo, &n, &a(0,0), &lda, &info);
    if (info != 0)
      { CalcPseudoInverseFB(A, lh); return; }
    dpotri_ (&uplo, &n, &a(0,0), &lda, &info);
    if (info != 0)
      { CalcPseudoInverseFB(A, lh); return; }
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++)
	{ A(j,i) = a(i,j); A(i,j) = a(i,j); }
  }

  /** Copied from ngsolve/basiclinalg/calcinerse.cpp, then modified if mat singular **/
  INLINE void CPI_TN (FlatMatrix<double> A, LocalHeap & lh)
  {
    int n = A.Height();
    if (n == 0)
      { return; }
    double eps = 0;
    for (int j = 0; j < n; j++)
      { eps = max(eps, A(j,j)); }
    eps = 1e-14 * eps;
    FlatMatrix<double> inv(n, n, lh); inv = A;
    ngstd::ArrayMem<int,100> p(n);   // pivot-permutation
    for (int j = 0; j < n; j++)
      { p[j] = j; }
    bool isok = true;
    for (int j = 0; j < n; j++)
      {
	// pivot search
	double maxval = abs(inv(j,j));
	int r = j;
	for (int i = j+1; i < n; i++)
	  if (abs (inv(j, i)) > maxval)
	    {
	      r = i;
	      maxval = abs (inv(j, i));
	    }
        double rest = 0.0;
        for (int i = j+1; i < n; i++)
          rest += abs(inv(r, i));
	// if ( (maxval < 1e-20*rest) || (rest < eps) )
	  // { isok = false; break; }
	if ( maxval < max(1e-20*rest, eps) )
	  { isok = false; break; }
	// exchange rows
	if (r > j)
	  {
	    for (int k = 0; k < n; k++)
	      swap (inv(k, j), inv(k, r));
	    swap (p[j], p[r]);
	  }
	// transformation
	double hr;
	CalcInverse (inv(j,j), hr);
	for (int i = 0; i < n; i++)
	  {
	    double h = hr * inv(j, i);
	    inv(j, i) = h;
	  }
	inv(j,j) = hr;
	for (int k = 0; k < n; k++)
	  if (k != j)
	    {
	      double help = inv(n*k+j);
	      double h = help * hr;   
	      for (int i = 0; i < n; i++)
		{
		  double h = help * inv(n*j+i); 
		  inv(n*k+i) -= h;
		}
	      inv(k,j) = -h;
	    }
      }
    if (isok) {
      // row exchange
      VectorMem<100,double> hv(n);
      for (int i = 0; i < n; i++)
	{
	  for (int k = 0; k < n; k++) hv(p[k]) = inv(k, i);
	  for (int k = 0; k < n; k++) inv(k, i) = hv(k);
	}
      A = inv;
    }
    else
      { CalcPseudoInverseFB(A, lh); }
  }

  void CalcPseudoInverseTryNormal (FlatMatrix<double> A, LocalHeap & lh)
  {
    // static Timer t("CalcPseudoInverseTryNormal"); RegionTimer rt(t);
    if (A.Height() >= 50)
      { CPI_TN_Lapack(A, lh); }
    else
      { CPI_TN (A, lh); }
  }

  void CalcPseudoInverseNew (FlatMatrix<double> mat, LocalHeap & lh)
  {
    int N = mat.Height(), M = 0;
    double maxd = 0;
    for (auto k : Range(N))
      { maxd = max2(maxd, mat(k,k)); }
    double eps = 1e-8 * maxd;
    for (auto k : Range(N))
      if (mat(k,k) > eps)
	{ M++; }
    if (M == N)
      { CalcPseudoInverseTryNormal(mat, lh); }
    else if (M > 0) {
      FlatMatrix<double> small_mat(M, M, lh);
      FlatArray<int> nzeros(M, lh);
      M = 0;
      for (auto k : Range(N))
	if (mat(k,k) > eps)
	  { nzeros[M++] = k; }
      small_mat = mat.Rows(nzeros).Cols(nzeros);
      // cout << " CPO on reduces mat " << endl << small_mat << endl;
      CalcPseudoInverseTryNormal(small_mat, lh);
      mat.Rows(nzeros).Cols(nzeros) = small_mat;
    }
  }

} // namespace amg
