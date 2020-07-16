#ifndef FILE_AMG_BLA_HPP
#define FILE_AMG_BLA_HPP

namespace amg
{
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
    const int NS = N-M;
    for (auto k : Range(NS))
      { evals(M+k) = 1.0/sqrt(evals(M+k)); }
    auto Q = evals.Rows(N-M, N); // non-kernel space of LHS (condition is always fulfilled on kernel of LHS)

    /** QT R Q **/
    FlatMatrix<double> RQ(NS, NS, lh), QTRQ(NS, NS, lh);
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


  template<int IMIN, int H, int JMIN, int W, class TMAT>
  class FlatMat : public MatExpr<FlatMat<IMIN, H, JMIN, W, TMAT>>
  {
  public:
    using TSCAL = typename mat_traits<TMAT>::TSCAL;
    using TELEM = typename mat_traits<TMAT>::TELEM;

    TMAT & basemat;

    FlatMat () = delete;
    FlatMat (const FlatMat & m) = default;

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

    INLINE constexpr size_t Height () const { return H; }
    INLINE constexpr size_t Width () const { return W; }
  }; // class FlatMat

  /** Small workaround so we do not always need to specify "TMAT" **/
  template<int IMIN, int H, int JMIN, int W, class TMAT>
  FlatMat<IMIN, H, JMIN, W, TMAT> MakeFlatMat (const TMAT & m)
  { return FlatMat<IMIN,H,JMIN,W,TMAT>(m); }


} // namespace amg

#endif
