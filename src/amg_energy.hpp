#ifndef FILE_AMG_ENERGY_HPP
#define FILE_AMG_ENERGY_HPP

namespace amg
{

  template<int ADIM, class T_V_DATA, class T_E_DATA>
  class H1Energy
  {
  public:
    using TVD = T_V_DATA;
    using TED = T_E_DATA;
    // static constexpr int DPV () { return DIM; }
    static constexpr int DIM = ADIM;
    static constexpr int DPV = ADIM;
    static constexpr bool NEED_ROBUST = false;
    // static constexpr bool STM = true; // edge-mat is scal times identity !
    typedef typename strip_mat<Mat<DIM,DIM,double>>::type TM;

    static INLINE double GetApproxWeight (const TED & ed) { return ed; }
    static INLINE double GetApproxVWeight (const TVD & vd) { return vd; }
    static INLINE const TM & GetEMatrix (const TED & ed) {
      if constexpr (ADIM == 1) { return ed; }
      else {
	static TM m;
	SetScalIdentity(ed, m);
	return m;
      }
    }
    static INLINE const TM & GetVMatrix (const TVD & vd) {
      if constexpr (ADIM == 1) { return vd; }
      else {
	static TM m;
	SetScalIdentity(vd/DPV, m);
	return m;
      }
    }

    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db);

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
    static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & em, const TVD & vdi, const TVD & vdj);

    static INLINE void QtMQ (const TM & Qij, const TM & M)
    { ; }

    static INLINE void AddQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
    { A += val * M; }

    static INLINE TM HMean (const TM & a, const TM & b)
    {
      if constexpr(is_same<TM, double>::value)
	{ return (2.0*a*b)/(a+b); }
      else
	{ return (2.0*a(0,0)*b(0,0))/(a(0,0)+b(0,0)); }
    }

    static INLINE TM GMean (const TM & a, const TM & b) {
      if constexpr(is_same<TM, double>::value)
	{ return sqrt(a*b); }
      else
	{ return sqrt(a(0,0)*b(0,0)); }
    }

    static INLINE TM AMean (const TM & a, const TM & b) {
      if constexpr(is_same<TM, double>::value)
	{ return (a+b)/2.0; }
      else
	{ return (a(0,0)+b(0,0))/2.0; }
    }

  };


#ifdef ELASTICITY

  template<int ADIM, class T_V_DATA, class T_E_DATA>
  class EpsEpsEnergy
  {
  public:
    using TVD = T_V_DATA;
    using TED = T_E_DATA;
    static constexpr int DIM = ADIM;
#ifdef ELASTICITY_ROBUST_ECW
    static constexpr bool NEED_ROBUST = true;
#else
    static constexpr bool NEED_ROBUST = false;
#endif
    static constexpr int dofpv () { return (DIM == 2) ? 3 : 6; }
    static constexpr int disppv () { return DIM; }
    static constexpr int rotpv () { return (DIM == 2) ? 1 : 3; }

    static constexpr int DISPPV = disppv();
    static constexpr int ROTPV = rotpv();
    static constexpr int DPV = dofpv();
    typedef Mat<DPV, DPV, double> TM;

    static INLINE double GetApproxWeight (const TED & ed) { return calc_trace(ed) / DPV; }
    static INLINE double GetApproxVWeight (const TVD & vd) { return vd.wt; }
    static INLINE const TM & GetEMatrix (const TED & ed) { return ed; }
    static INLINE const TM & GetVMatrix (const TVD & vd) {
      if constexpr (ADIM == 1) { return vd.wt; }
      else { // TODO: this should really be a proper TM matrix eventually ... 
	static TM m;
	SetScalIdentity(vd.wt/DPV, m);
	return m;
      }
    }

    static INLINE void CalcQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void ModQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db);

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
    static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & ed, const TVD & vdi, const TVD & vdj);

    static INLINE void QtMQ (const TM & cQij, TM & M)
    {
      // static Mat<disppv(), rotpv(), double> X;
      // static Mat<disppv(), disppv(), double> M11;
      // static Mat<disppv(), rotpv(), double> M11X;
      // static Mat<rotpv(), rotpv(), double> M22X;
      // GetTMBlock<0, disppv()>(X, Qij);
      // GetTMBlock<0, 0>(M11, M);
      // M11X = M11 * X;
      // AddTMBlock<0, disppv()>(M, M11X);
      // AddTMBlock<disppv(), 0>(M, Trans(M11X));
      // M22X = M22 * X;
      // AddTMBlock<disppv(), disppv()>(M, Trans(X) * M22X);

      static Mat<disppv(), rotpv(), double> M11X;
      static Mat<rotpv(), rotpv(), double> M22X;
      TM & Qij = const_cast<TM&>(cQij); // oh no ...
      auto X = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
      auto M11 = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      auto M12 = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      auto M21 = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      auto M22 = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
      M11X = M11 * X;
      M22X = M22 * X;
      M12 += M11X;
      M21 += Trans(M11X);
      M22 += Trans(X) * M22X;

      // static TM MQ;
      // MQ = M * Qij;
      // M = Trans(Qij) * MQ;
    }

    static INLINE void AddQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
    {
      // static Mat<disppv(), rotpv(), double> X;
      // static Mat<disppv(), rotpv(), double> M12;
      // static Mat<disppv(), disppv(), double> M11;
      // static Mat<disppv(), rotpv(), double> M11X;
      // static Mat<rotpv(), rotpv(), double> M22X;
      // GetTMBlock<0, disppv()>(M12, M);
      // GetTMBlock<0, disppv()>(X, Qij);
      // GetTMBlock<0, 0>(M11, M);
      // M11X = M11 * X;
      // M22X = M22 * X;
      // AddTMBlock<0, 0>(val, A, M11);
      // M12 += M11X
      // AddTMBlock<0, disppv()>(val, A, M12);
      // AddTMBlock<disppv(), 0>(val, A, Trans(M12));
      // AddTMBlock<disppv(), disppv()>(val, A, M22 + Trans(X) * M22X);

      static Mat<disppv(), rotpv(), double> M11X;
      static Mat<rotpv(), rotpv(), double> M22X;
      auto & Qij = const_cast<TM&>(Qij); // oh no ...
      auto X = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
      auto M11 = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      auto M12 = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      auto M21 = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      auto M22 = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
      M11X = M12 + M11 * X;
      M22X = M22 * X;
      MakeFlatMat<0, DISPPV, 0, DISPPV>(A) += val * M11;
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(A) += M12 + M11X;
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(A) += M21 + Trans(M11X);
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(A) += M22 + Trans(X) * M22X;

      // static TM MQ;
      // MQ = M * Qij;
      // M = Trans(Qij) * MQ;
      // A += val * (M + Trans(Qij) * MQ);
    }


    /** Fake harmonic mean - do not use this for stable coarsening ! **/
    static INLINE TM HMean (const TM & A, const TM & B)
    {
      double tra = calc_trace(A), trb = calc_trace(B);
      double tr = (2.0*tra*trb)/(tra+trb);
      return tr * 0.5 * (A/tra + B/trb);
    }


    /** Fake geometric mean - do not use this for stable coarsening ! **/
    static INLINE TM GMean (const TM & A, const TM & B)
    {
      double tra = calc_trace(A), trb = calc_trace(B);
      double tr = sqrt(tra*trb);
      return tr * 0.5 * (A/tra + B/trb);
    }

    /** Actually the real algebraic mean **/
    static INLINE TM AMean (const TM & A, const TM & B)
    {
      return 0.5 * (A + B);
    }

  };

#endif

} // namespac amg

#endif
