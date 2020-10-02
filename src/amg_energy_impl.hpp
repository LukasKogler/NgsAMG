#ifndef FILEAMG_ENERGY_IMPL_HPP
#define FILEAMG_ENERGY_IMPL_HPP

namespace amg
{

  /** H1Energy **/


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: CalcQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    SetIdentity(Qij);
  } // H1Energy::CalcQij


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: ModQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    ;
  } // H1Energy::ModQij


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: CalcQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    SetIdentity(QHh);
  } // H1Energy::CalcQHh


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    ;
  } // H1Energy::ModQHh


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: CalcQs (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    SetIdentity(Qij);
    SetIdentity(Qji);
  } // H1Energy::CalcQs


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: ModQs (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    ;
  } // H1Energy::ModQs


  template<int DIM, class TVD, class TED>
  INLINE typename H1Energy<DIM, TVD, TED>::TVD H1Energy<DIM, TVD, TED> :: CalcMPData (const TVD & da, const TVD & db)
  {
    return TVD(0);
  } // H1Energy::CalcMPData


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
  {
    SetScalIdentity(calc_trace(ed), mat(0,0)); 
    // if (DIM == 1)
    //   { mat(0,0) = ed; }
    // else
    //   {
    // 	mat(0,0) = 0;
    // 	Iterate<DIM>([&](auto i) LAMBDA_INLINE { mat(0,0)(i.value, i.value) = calc_trace(ed); });
    //   }
    mat(1,1) = mat(0,0);
    mat(1,0) = -mat(0,0);
    mat(0,1) = -mat(0,0);
  } // H1Energy::CalcRMBlock


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: CalcRMBlock2 (FlatMatrix<TM> mat, const TM & em, const TVD & vdi, const TVD & vdj)
  {
    CalcRMBlock(mat, em(0,0), vdi, vdj);
  } // H1Energy::CalcRMBlock2

  /** END H1Energy **/


#ifdef ELASTICITY

  /** EpsEpsEnergy **/

  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcQ  (const Vec<DIM> & t, TM & Q)
  {
    Q = 0;
    Iterate<DPV>([&] (auto i) LAMBDA_INLINE { Q(i.value, i.value) = 1.0; } );
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


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: ModQ  (const Vec<DIM> & t, TM & Q)
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


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
  } // EpsEpsEnergy::CalcQij


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: ModQij (const TVD & di, const TVD & dj, TM & Qij)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
  } // EpsEpsEnergy::ModQij


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    Vec<DIM> t = dh.pos - dH.pos; // H -> h
    CalcQ(t, QHh);
  } // EpsEpsEnergy::CalcQHh


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
  {
    Vec<DIM> t = dh.pos - dH.pos; // H -> h
    ModQ(t, QHh);
  } // EpsEpsEnergy::ModQHh


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
    t *= -1;
    CalcQ(t, Qji);
  } // EpsEpsEnergy::CalcQs


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    Vec<DIM> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
    t *= -1;
    ModQ(t, Qji);
  } // EpsEpsEnergy::ModQs


  template<int DIM, class TVD, class TED>
  INLINE typename EpsEpsEnergy<DIM, TVD, TED>::TVD EpsEpsEnergy<DIM, TVD, TED> :: CalcMPData (const TVD & da, const TVD & db)
  {
    TVD o(0); o.pos = 0.5 * (da.pos + db.pos);
    return move(o);
  } // EpsEpsEnergy::CalcMPData


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
  {
    static TM Qij, Qji, QiM, QjM;
    CalcQs( vdi, vdj, Qij, Qji);
    QiM = Trans(Qij) * ed;
    QjM = Trans(Qji) * ed;
    mat(0,0) =  QiM * Qij;
    mat(0,1) = -QiM * Qji;
    mat(1,0) = -QjM * Qij;
    mat(1,1) =  QjM * Qji;
  } // EpsEpsEnergy::CalcRMBlock


  template<int DIM, class TVD, class TED>
  INLINE void EpsEpsEnergy<DIM, TVD, TED> :: CalcRMBlock2 (FlatMatrix<TM> mat, const TM & ed, const TVD & vdi, const TVD & vdj)
  {
    CalcRMBlock(mat, ed, vdi, vdj);
  } // EpsEpsEnergy::CalcRMBlock2

  /** END EpsEpsEnergy **/

#endif
} // namespace amg

#endif
