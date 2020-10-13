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
    static INLINE TVD CalcMPDataWW (const TVD & da, const TVD & db); // with weights

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
    static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & em, const TVD & vdi, const TVD & vdj);

    static INLINE void QtMQ (const TM & Qij, const TM & M)
    { ; }

    static INLINE void AddQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
    { A += val * M; }

    static INLINE void SetQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
    { A = val * M; }

    static INLINE void CalcMQ (double scal, const TM & Q, TM M, TM & out)
    { out = scal * M; }
    static INLINE void AddMQ (double scal, const TM & Q, TM M, TM & out)
    { out += scal * M; }
    static INLINE void CalcQTM (double scal, const TM & Q, TM M, TM & out)
    { out = scal * M; }
    static INLINE void AddQTM (double scal, const TM & Q, TM M, TM & out)
    { out += scal * M; }

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
    static INLINE double GetApproxVWeight (const TVD & vd) { return calc_trace(vd.wt) / DPV; }
    static INLINE const TM & GetEMatrix (const TED & ed) { return ed; }
    static INLINE const TM & GetVMatrix (const TVD & vd) { return vd.wt; }

    static INLINE void CalcQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void ModQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db);
    static INLINE TVD CalcMPDataWW (const TVD & da, const TVD & db); // with weights

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
    static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & ed, const TVD & vdi, const TVD & vdj);

    static INLINE void QtMQ (const TM & Qij, TM & M)
    {
      /** I       A  B    I Q
	  QT I    BT D      I **/
      static Mat<disppv(), rotpv(), double> AQ;
      static Mat<rotpv(), rotpv(), double> M22X;
      auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
      auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
      AQ = A * Q;
      D += Trans(Q) * (AQ + B) + BT * Q;
      B += AQ;
      BT = Trans(B);
    }

    static INLINE void AddQtMQ (double val, TM & aA, const TM & Qij, const TM & M)
    {
      /** I       A  B    I Q
	  QT I    BT D      I **/
      static Mat<disppv(), rotpv(), double> AQpB;
      auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
      auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
      AQpB = A * Q + B;
      MakeFlatMat<0, DISPPV, 0, DISPPV>(aA) += val * A;
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(aA) += val * AQpB;
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(aA) += val * Trans(AQpB);
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(aA) += val * (Trans(Q) * AQpB + BT * Q + D);
    }


    static INLINE void SetQtMQ (double val, TM & aA, const TM & Qij, const TM & M)
    {
      /** I       A  B    I Q
	  QT I    BT D      I **/
      static Mat<disppv(), rotpv(), double> AQpB;
      auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
      auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
      AQpB = A * Q + B;
      MakeFlatMat<0, DISPPV, 0, DISPPV>(aA) = val * A;
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(aA) = val * AQpB;
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(aA) = val * Trans(AQpB);
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(aA) = val * (Trans(Q) * AQpB + BT * Q + D);
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

    static INLINE void CalcMQ (double scal, const TM & Q, const TM & M, TM & out)
    {
      /** A  B   I Q  =  A   AQ+B
	  BT C   0 I  =  BT BTQ+C **/
      static Mat<DISPPV, ROTPV, double> AQ;
      static Mat<ROTPV, ROTPV, double> BTQ;
      BTQ = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
      AQ = MakeFlatMat<0, DISPPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
      MakeFlatMat<0, DISPPV, 0, DISPPV>(out) = scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) = scal * ( MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M) + AQ );
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) = scal * MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) = scal * (MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M) + BTQ);
    }

    static INLINE void AddMQ (double scal, const TM & Q, const TM & M, TM & out)
    {
      /** A  B   I Q  =  A   AQ+B
	  BT C   0 I  =  BT BTQ+C **/
      static Mat<DISPPV, ROTPV, double> AQ;
      static Mat<ROTPV, ROTPV, double> BTQ;
      BTQ = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
      AQ = MakeFlatMat<0, DISPPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
      MakeFlatMat<0, DISPPV, 0, DISPPV>(out) += scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) += scal * ( MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M) + AQ );
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) += scal * MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) += scal * (MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M) + BTQ);
    }

    static INLINE void CalcQTM (double scal, const TM & Q, const TM & M, TM & out)
    {
      /** I  0   A  B   =    A      B
	  QT I   BT C   =  QTA+BT QTB+C **/
      static Mat<DISPPV, ROTPV, double> QTA;
      static Mat<ROTPV, ROTPV, double> QTB;
      QTA = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      QTB = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      MakeFlatMat<0, DISPPV, 0, DISPPV>(out) = scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) = scal * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) = scal * ( MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) + QTA );
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) = scal * (MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M) + QTB);
    }

    static INLINE void AddQTM (double scal, const TM & Q, const TM & M, TM & out)
    {
      /** I  0   A  B   =    A      B
	  QT I   BT C   =  QTA+BT QTB+C **/
      static Mat<DISPPV, ROTPV, double> QTA;
      static Mat<ROTPV, ROTPV, double> QTB;
      QTA = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      QTB = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      MakeFlatMat<0, DISPPV, 0, DISPPV>(out) += scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
      MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) += scal * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
      MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) += scal * ( MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) + QTA );
      MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) += scal * (MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M) + QTB);
    }
};

#endif

} // namespac amg

#endif
