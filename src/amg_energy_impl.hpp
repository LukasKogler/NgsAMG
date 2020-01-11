#ifndef FILEAMG_ENERGY_IMPL_HPP
#define FILEAMG_ENERGY_IMPL_HPP

namespace amg
{

  /** H1Energy **/

  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void CalcQ (const Vec<3> & t, TM & Q)
  {
    SetIdentity(Q);
  } // H1Energy::CalcQ


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void ModQ (const Vec<3> & t, TM & Q)
  {
    ;
  } // H1Energy::ModQ


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void CalcQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    CalcQ(Qij);
  } // H1Energy::CalcQij


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void ModQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    ModQ(Qij);
  } // H1Energy::ModQij


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    CalcQ(QHh);
  } // H1Energy::CalcQHh


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    ModQ(QHh);
  } // H1Energy::ModQHh


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void CalcQs (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    CalcQs(Qij);
    CalcQs(Qji);
  } // H1Energy::CalcQs


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA> :: void ModQs (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    ModQ(Qij);
    ModQ(Qji);
  } // H1Energy::ModQs


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE H1Energy<DIM, T_V_DATA, T_E_DATA>::TVD H1Energy<DIM, T_V_DATA, T_E_DATA> :: CalcMPData (const TVD & da, const TVD & db)
  {
    return TVD(0);
  } // H1Energy::CalcMPData


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void H1Energy :: CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
  {
    mat(0,0) = 0; Iterate<DIM>([&](auto i) LAMBDA_INLINE { mat(0,0)(i.value, i.value) = calc_trace(ed); });
    mat(1,1) = mat(0,0);
    mat(1,0) = -mat(0,0);
    mat(0,1) = -mat(0,0);
  } // H1Energy::CalcRMBlock

  /** END H1Energy **/


#ifdef ELASTICITY

  /** EpsEpsEnergy **/

  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: CalcQ  (const Vec<DIM> & t, TM & Q)
  {
    Q = 0;
    Iterate<DIM>([&] (auto i) LAMBDA_INLINE { Q(i.value, i.value) = 1.0; } );
    if constexpr(DIM == 2) {
	Q(0,2) = -t(1);
	Q(1,2) =  t(0);
      }
    else {
      // Q(1,5) = - (Q(2,4) = t(0));
      // Q(2,3) = - (Q(0,5) = t(1));
      // Q(0,4) = - (Q(1,3) = t(2));
      Q(1,5) = - (Q(2,4) = -t(0));
      Q(2,3) = - (Q(0,5) = -t(1));
      Q(0,4) = - (Q(1,3) = -t(2));
    }
  } // EpsEpsEnergy::CalcQ


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: ModQ  (const Vec<DIM> & t, TM & Q)
  {
    if constexpr(DIM == 2) {
	Q(0,2) = -t(1);
	Q(1,2) =  t(0);
      }
    else {
      // Q(1,5) = - (Q(2,4) = t(0));
      // Q(2,3) = - (Q(0,5) = t(1));
      // Q(0,4) = - (Q(1,3) = t(2));
      Q(1,5) = - (Q(2,4) = -t(0));
      Q(2,3) = - (Q(0,5) = -t(1));
      Q(0,4) = - (Q(1,3) = -t(2));
    }
  } // EpsEpsEnergy::ModQ


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: CalcQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
  } // EpsEpsEnergy::CalcQij


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: ModQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
  } // EpsEpsEnergy::ModQij


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: CalcQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    Vec<DIM> t = dh.pos - dH.pos; // H -> h
    CalcQ(t, QHh);
  } // EpsEpsEnergy::CalcQHh


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    Vec<DIM> t = dh.pos - dH.pos; // H -> h
    ModQ(t, QHh);
  } // EpsEpsEnergy::ModQHh


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
    t *= -1;
    CalcQ(t, Qji);
  } // EpsEpsEnergy::CalcQs


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
    t *= -1;
    ModQ(t, Qji);
  } // EpsEpsEnergy::ModQs


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA>::TVD EpsEpsEnergy<DIM, T_V_DATA, T_E_DATA> :: CalcMPData (const TVD & da, const TVD & db)
  {
    T_V_DATA o; o.pos = 0.5 * (da.pos + db.pos);
    return move(o);
  } // EpsEpsEnergy::CalcMPData


  template<int DIM, class T_V_DATA, class T_E_DATA>
  INLINE void H1Energy :: EpsEpsEnergy (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
  {
    static TM Qij, Qji, QiM, QjM;
    CalcQs( vdi, vdj, Qij, Qji);
    QiM = Trans(Qij) * M;
    QjM = Trans(Qji) * M;
    mat(0,0) =  QiM * Qij;
    mat(0,1) = -QiM * Qji;
    mat(1,0) = -QjM * Qij;
    mat(1,1) =  QjM * Qji;
  } // EpsEpsEnergy::CalcRMBlock

  /** END EpsEpsEnergy **/

#endif
} // namespace amg

#endif
