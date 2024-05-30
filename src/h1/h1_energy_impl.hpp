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
  INLINE void H1Energy<DIM, TVD, TED> :: CalcQHh (const TVD & dH, const TVD & dh, TM & QHh, double scale)
  {
    SetIdentity(QHh);
  } // H1Energy::CalcQHh


  template<int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED> :: ModQHh (const TVD & dH, const TVD & dh, TM & QHh, double scale)
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
  INLINE void H1Energy<DIM, TVD, TED> :: CalcInvQs (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
  {
    SetIdentity(Qij);
    SetIdentity(Qji);
  } // H1Energy::CalcInvQs
  
  
  template <int DIM, class TVD, class TED>
  INLINE void H1Energy<DIM, TVD, TED>::CalcK(const TVD& di,
                                             const TVD& dj,
                                             FlatVector<TM> K)
  {
    SetIdentity(K(0));
    SetScalIdentity(-1.0, K(1));
  } // H1Energy::CalcK


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
  INLINE typename H1Energy<DIM, TVD, TED>::TVD H1Energy<DIM, TVD, TED> :: CalcMPDataWW (const TVD & da, const TVD & db)
  {
    return da + db;
  } // EpsEpsEnergy::CalcMPData


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

} // namespace amg

#endif
