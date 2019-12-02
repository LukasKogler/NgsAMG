#ifndef FILE_AMG_BLA_HPP
#define FILE_AMG_BLA_HPP

namespace amg
{

  
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

    FlatVector inv_diags(M, lh);
    for (auto k : Range(M))
      { inv_diags[k] = 1.0 / sqrt(evalsL(first_nz+k)); }

    for (auto k : Range(M)) // TODO: scal evecs instead
      for (auto j : Range(M))
	{ projR(k,j) *= inv_diags[k] * inv_diags[j]; }

    FlatMatrix<double> evecsPR(M, M, lh);
    FlatVector<double> evalsPR(M, lh);
    TimedLapackEigenValuesSymmetric(projR, evalsPR, evecsPR);

    return evalsPR(0);
  } // MEV


  /** ~ fake geometric mean ~ **/
  INLINE double MIN_EV_FG (double A, double B, double Qij, double Qji, double R) { return R / sqrt(A*B); }
  template<int N>
  INLINE double MIN_EV_FG_ONE_SIDE (const Mat<N,N,double> & A, const Mat<N,N,double> & B,
				     const Mat<N,N,double> & Qij, const Mat<N,N,double> & Qji,
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
    
    if (M == 0)
      { return MEV<N>(A, R); }

    auto VK = evecs.Rows(0,M); // kernel-evecs
    
    FlatMatrix<double> QjiVK(N, M, lh), AK (M,M,lh);
    QjiVK = Qji * Trans(VK);
    AK = Trans(QjiVK) * A * QjiVK;
    Iterate<N>([&](auto i) {
	if (M == i.value + 1)
	  { CalcPseudoInverse<i.value + 1>(AK); }
      });
    static Mat<N,N,double> TMP, I_MINUS_STUFF, L;
    TMP = -1 * QjiVK * AK * Trans(QjiVK);

    I_MINUS_STUFF = TMP * A;
    Iterate<N>([&](auto i) { I_MINUS_STUFF(i.value, i.value) += 1.0; });

    L = A * I_MINUS_STUFF;

    return MEV<N>(L, R);
  }

  template<int N>
  INLINE double MIN_EV_FG (const Mat<N,N,double> & A, const Mat<N,N,double> & B,
			    const Mat<N,N,double> & Qij, const Mat<N,N,double> & Qji,
			    const Mat<N,N,double> & R, bool prtms = false)
  {
    static Timer t("EV_FG"); RegionTimer rt(t);

    double mev_a = fabs(MIN_EV_FG_ONE_SIDE(A, B, Qij, Qji, R)); // -eps

    double mev_b = fabs(MIN_EV_FG_ONE_SIDE(B, A, Qji, Qij, R)); // -eps
    
    return sqrt(mev_a * mev_b);
  }


  INLINE double MIN_EV_HARM (double A, double B, double R) { return 0.5 * R * (A+B)/(A*B); }
  template<int N>
  INLINE double MIN_EV_HARM (const Mat<N,N,double> & A, const Mat<N,N,double> & B, const Mat<N,N,double> & aR)
  {
    /**
       Compute inf <Rx,x> / <A(A+B)^(-1)Bx,x> (<= 1!):
         - decompose L
	 - project R to ortho(ker(L))
	     [ we can ignore any x in ker(L), if its in ker(R), we have 1, else "inf", both not interesting ]
	 - return minimum EV of L^(-1/2) R L^(-1/2) [projected to ortho(ker(L))]
     **/
    static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
    HeapReset hr(lh);

    static Mat<N,N,double> SUM;

    SUM = A + B;
    CalcPseudoInverse<N>(SUM); // Is CPI<3,3,6> valid in 3d ??

    FlatMatrix<double> L(N, N, lh), R(N, N, lh);
    L = A * SUM * B;
    R = aR;

    return 0.5 * MEV<N>(L,R);
  } // MIN_EV_HARM

} // namespace amg

#endif
