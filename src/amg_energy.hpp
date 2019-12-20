#ifndef FILE_AMG_ENERGY_HPP
#define FILE_AMG_ENERGY_HPP

namespace amg
{

  template<int DIM, class T_V_DATA, class T_E_DATA>
  class H1Energy
  {
  public:
    using TVD = T_V_DATA;
    using TED = T_E_DATA;
    static constexpr int DPV () { return DIM; }
    typedef typename strip_mat<Mat<DIM,DIM,double>>::type TM;

    static INLINE void CalcQ  (const Vec<3> & t, TM & Q);
    static INLINE void ModQ  (const Vec<3> & t, TM & Q);
    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db);
  };


#ifdef ELASTICITY

  template<int DIM, class T_V_DATA, class T_E_DATA>
  class EpsEpsEnergy
  {
  public:
    using TVD = T_V_DATA;
    using TED = T_E_DATA;
    static constexpr int DPV () { return (DIM == 2) ? 3 : 6; }
    typedef typename Mat<DPV(DIM), DPV(DIM), double> TM;

    static INLINE void CalcQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void ModQ  (const Vec<DIM> & t, TM & Q);
    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh);
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db);
  };

#endif

} // namespac amg

#endif
