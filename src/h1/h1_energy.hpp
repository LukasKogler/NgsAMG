#ifndef FILE_H1_ENERGY_HPP
#define FILE_H1_ENERGY_HPP

#include <utils_sparseLA.hpp>
#include <utils_sparseMM.hpp>

namespace amg
{

template<int ADIM, class T_V_DATA, class T_E_DATA>
class H1Energy
{
public:
  using TVD = T_V_DATA;
  using TED = T_E_DATA;
  static constexpr int DIM = ADIM;
  static constexpr int DPV = ADIM;
  static constexpr bool NEED_ROBUST = false;
  using TM = StripTM<DIM,DIM>;

  class TQ
  {
  public:

    TQ(){}

    ~TQ() = default;

    template<class TMU> INLINE void MQ(TMU &E) const {}

    template<class TSCAL, class TMU>
    INLINE void
    MQ(TSCAL const &scal, TMU const &E, TMU &EQ) const
    {
      EQ = scal * E;
    }

    template<class TSCAL, class TMU>
    INLINE TMU
    GetMQ(TSCAL const &scal, TMU const &E) const
    {
      TMU EQ = scal * E;
      return EQ;
    }

    template<class TMU> INLINE void QTM(TMU &E) const {}

    template<class TSCAL, class TMU>
    INLINE void
    QTM(TSCAL const &scal, TMU const &E, TMU &QTE) const
    {
      QTE = scal * E;
    }

    template<class TSCAL, class TMU>
    INLINE TMU
    GetQTM(TSCAL const &scal, TMU const &E) const
    {
      TMU QTE = scal * E;
      return QTE;
    }

    template<class TMU> INLINE void QTMQ(TMU &E) const {}

    template<class TSCAL, class TMU>
    INLINE void
    QTMQ(TSCAL const &scal, TM const &E, TM &QT_E_Q) const
    {
      QT_E_Q = scal * E;
      QTMQ(QT_E_Q);
    }

    template<class TSCAL, class TMU>
    INLINE TMU
    GetQTMQ(TSCAL const &scal, TMU const &E) const
    {
      TMU QT_E_Q = scal * E;
      return QT_E_Q;
    }

    template<class TSCAL, class TMU>
    INLINE void
    QMQT(TMU &E) const {}

    template<class TSCAL, class TMU>
    INLINE void
    QMQT(TSCAL const &scal, TMU const &E, TMU &Q_E_QT) const
    {
      Q_E_QT = scal * E;
    }

    template<class TSCAL, class TMU>
    INLINE TMU
    GetQMQT(TSCAL const &scal, TMU const &E) const
    {
      TMU Q_E_QT = scal * E;
      return Q_E_QT;
    }

    template<class TSCAL>
    INLINE void
    SetQ(TSCAL &E) const
    {
      E = TSCAL(1.0);
    }

    template<class TSCAL>
    INLINE void
    SetQ(double const &alpha, TSCAL &E) const
    {
      E = TSCAL(alpha);
    }
  };

  template<class TSCAL = double>
  static INLINE TSCAL
  GetApproxWeight (TED const &ed)
  {
    return ed;
  }

  template<class TSCAL = double>
  static INLINE TSCAL
  GetApproxVWeight (TVD const &vd)
  {
    // temporary ugly workaround due to switching the H1 data to IVec<2,double>
    TSCAL wt;

    if constexpr( is_same<TVD, double>::value )
      { wt = vd; }
    else
      { wt = vd[0]; }

    return wt;
  }


  static INLINE void
  SetEMatrix (TED &ed, TM const &m)
  {
    if constexpr(std::is_same_v<TM, double>)
    {
      ed = m;
    }
    else
    {
      ed = m(0,0);
    }
  }

  template<class TMU>
  static INLINE void
  SetEMatrix (TED & ed, TMU const &m)
  {
    // for dim>1, TM is Mat<DIM,DIM>, but TED is just double
    if constexpr(std::is_same<TMU, float>::value ||
                 std::is_same<TMU, double>::value)
    {
      ed = m;
    }
    else
    {
      ed = m(0,0);
    }
  }

  template<class TMU>
  static INLINE void
  SetVMatrix (TVD & vd, TMU const &m)
  {
    // for dim>1, TM is Mat<DIM,DIM>, but TVD is just IVec<2,double> (second entry unused ATM)
    if constexpr(std::is_same<TMU, float>::value ||
                 std::is_same<TMU, double>::value)
    {
      // this is dumb and should be cleaned up...
      if constexpr(std::is_same<TVD, float>::value ||
                   std::is_same<TVD, double>::value)
      {
        vd = m;
      }
      else
      {
        vd[0] = m;
      }
    }
    else
    {
      if constexpr(std::is_same<TVD, float>::value ||
                   std::is_same<TVD, double>::value)
      {
        vd[0] = m(0,0);
      }
      else
      {
        vd = m(0,0);
      }
    }
  }

  static INLINE std::tuple<TQ, TQ> GetQijQji(TVD const &vdi, TVD const &vdj) { return std::make_tuple(TQ(), TQ()); }
  static INLINE TQ                 GetQij   (TVD const &vdi, TVD const &vdj) { return TQ(); }
  static INLINE TQ                 GetQiToj (TVD const &vdi, TVD const &vdj) { return TQ(); }

  template<class TSCAL = double>
  static INLINE StripTM<DPV, DPV, TSCAL>
  GetEMatrix (TED const &ed)
  {
    StripTM<DPV, DPV, TSCAL> mat;
    SetScalIdentity(ed, mat);
    return mat;
  }
  
  template<class TSCAL = double>
  static INLINE StripTM<DPV, DPV, TSCAL>
  GetVMatrix (TVD const &vd)
  {
    StripTM<DPV, DPV, TSCAL> mat;
    SetScalIdentity(GetApproxVWeight<TSCAL>(vd), mat);
    return mat;
  }

  static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
  static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
  static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh, double scale = 1.0);
  static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh, double scale = 1.0);
  static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE void CalcInvQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE void CalcK (const TVD& di, const TVD& dj, FlatVector<TM> K);
  static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE TVD CalcMPData (const TVD & da, const TVD & db);
  static INLINE TVD CalcMPDataWW (const TVD & da, const TVD & db); // with weights

  static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
  static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & em, const TVD & vdi, const TVD & vdj);

  template<bool BOTH_ROWS, class TLAM>
  static INLINE void
  CalcRMBlockImpl (const TED & ed,
                   const TVD & vdi,
                   const TVD & vdj,
                   TLAM lam)
  {
    auto const &E = GetEMatrix(ed);

    lam(0, 0,  E);
    lam(0, 1, -E);

    if constexpr(BOTH_ROWS)
    {
      lam(1, 0, -E);
      lam(1, 1,  E);
    }
  }

  template<class TLAM>
  static INLINE void
  CalcRMBlock(const TED & ed,
              const TVD & vdi,
              const TVD & vdj,
              TLAM lam)
  {
    CalcRMBlockImpl<true>(ed, vdi, vdj, lam);
  }

  template<class TLAM>
  static INLINE void
  CalcRMBlockRow(const TED & ed,
                 const TVD & vdi,
                 const TVD & vdj,
                 TLAM lam)
  {
    CalcRMBlockImpl<false>(ed, vdi, vdj, lam);
  }

  static INLINE void QtMQ (const TM & Qij, const TM & M)
  { ; }

  static INLINE void AddQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
  { A += val * M; }

  static INLINE void SetQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
  { A = val * M; }

  static INLINE TM CalcQtMQ (const TM & Qij, const TM & M)
  { return M; }

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
    // return (2.0 * a * b) / (a + b);
    if constexpr(is_same<TM, double>::value)
      { return (2.0 * a * b) / (a + b); }
    else
      { TM hMean; SetScalIdentity((2.0 * a(0,0) * b(0,0)) / (a(0,0) + b(0,0)), hMean); return hMean; }
  }

  static INLINE TM GMean (const TM & a, const TM & b)
  {
    // return sqrt(a * b);
    if constexpr(is_same<TM, double>::value)
      { return sqrt(a * b); }
    else
      { TM gMean; SetScalIdentity(sqrt(a(0,0) * b(0,0)), gMean); return gMean; }
  }

  static INLINE TM AMean (const TM & a, const TM & b)
  {
    // return 0.5 * (a + b);
      if constexpr(is_same<TM, double>::value)
      { return 0.5 * (a + b); }
    else
      { TM aMean; SetScalIdentity(0.5 * (a(0,0) + b(0,0)), aMean); return aMean; }
  }
};

} // namespac amg

#endif // FILE_H1_ENERGY_HPP
