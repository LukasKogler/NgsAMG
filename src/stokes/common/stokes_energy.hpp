#ifndef FILE_STOKES_ENERGY_HPP
#define FILE_STOKES_ENERGY_HPP

#include <utils_denseLA.hpp>

namespace amg
{

/** StokesEnergy **/

template<class AENERGY, class ATVD, class ATED>
class StokesEnergy
{
public:

  /** A wrapper around a normal energy **/

  using ENERGY = AENERGY;
  using TVD = ATVD;
  using TED = ATED;

  static constexpr int DIM = ATVD::DIM; // hdiv-stokes uses H1-energy<1,..>, so take the dim from v-data
  static constexpr int DPV = ENERGY::DPV;
  static constexpr bool NEED_ROBUST = ENERGY::NEED_ROBUST;

  // workaround for QtQM functions is SPW-agg that were a workaround
  // around some strange gcc bug that caused it to crash when those
  // were static methods in elasticity energy!
  static constexpr int DISPPV = DPV;
  static constexpr int ROTPV  = 0;

  // entry-type
  using TM = typename ENERGY::TM;

  template<class TSCAL = double>
  static INLINE TSCAL GetApproxWeight (const TED & ed) {
    TSCAL wi = ENERGY::template GetApproxWeight<TSCAL>(ed.edi);
    TSCAL wj = ENERGY::template GetApproxWeight<TSCAL>(ed.edj);
    // return ( (wi>0) && (wj>0) ) ? (wi + wj) : 0;
    // return ( (wi > 1e-12) && (wj > 1e-12) ) ? (2 * wi * wj) / (wi + wj) : 0;
    return ( (wi > 1e-12) && (wj > 1e-12) ) ? L2Norm(ed.flow) : 0;
  }

  template<class TSCAL = double>
  static INLINE TSCAL GetApproxVWeight (const TVD & vd) {
    return ENERGY::template GetApproxVWeight<TSCAL>(vd.vd);
  }

  static INLINE TM GetEMatrix (const TED & ed) {
    static TM emi, emj;
    emi = ENERGY::GetEMatrix(ed.edi);
    double tri = calc_trace(emi) / DPV;
    emi /= tri;
    emj = ENERGY::GetEMatrix(ed.edj);
    double trj = calc_trace(emj) / DPV;
    emj /= trj;
    double f = ( (tri > 0) && (trj > 0) ) ? ( 2 * tri * trj / (tri + trj) ) : 0;
    return f * (emi + emj);
  }

  static INLINE TM GetVMatrix (const TVD & svd)
  { return svd.IsReal() ? ENERGY::GetVMatrix(svd.vd) : TM(0.0); }

  template<class TMU>
  static INLINE void
  SetVMatrix (TVD & svd, TMU const &m)
  {
    ENERGY::SetVMatrix(svd.vd, m);
  }
  
  static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij)
    { ENERGY::CalcQij(di.vd, dj.vd, Qij); }

  static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij)
    { ENERGY::ModQij(di.vd, dj.vd, Qij); }

  static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh, double scale = 1.0)
    { ENERGY::CalcQHh(dH.vd, dh.vd, QHh, scale); }

  static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
    { ENERGY::CalcQHh(dH.vd, dh.vd, QHh, 1.0); }

  static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
    { ENERGY::CalcQs(di.vd, dj.vd, Qij, Qji); }

  static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
    { ENERGY::ModQs(di.vd, dj.vd, Qij, Qji); }

  static INLINE TVD CalcMPData (const TVD & da, const TVD & db)
  {
    TVD dc;
    dc.vd = ENERGY::CalcMPData(da.vd, db.vd);
    dc.vol = 0.0; // ... idk what us appropriate ...
    return dc;
  }

  static INLINE TVD CalcMPDataWW (const TVD & da, const TVD & db)
  {
    TVD dc;
    dc.vd = ENERGY::CalcMPDataWW(da.vd, db.vd);
    // we use this in SPW agg for tentative coarse points
    // we PROBABLY dont NEED "correct" volume, but it won't hurt
    dc.vol = da.vol + db.vol;
    return dc;
  }

  static INLINE void QtMQ(const TM & Qij, const TM & M)
  { ENERGY::QtMQ(Qij, M); }

  static INLINE void AddQtMQ(double val, TM & A, const TM & Qij, const TM & M)
  { ENERGY::AddQtMQ(val, A, Qij, M); }

  static INLINE void SetQtMQ (double val, TM & A, const TM & _Qij, const TM & M)
  { ENERGY::SetQtMQ(val, A, _Qij, M); }

  static INLINE void CalcMQ (double scal, const TM & Q, TM M, TM & out)
  { ENERGY::CalcMQ(scal, Q, M, out); }

  static INLINE void AddMQ (double scal, const TM & Q, TM M, TM & out)
  { ENERGY::AddMQ(scal, Q, M, out); }

  static INLINE void CalcQTM (double scal, const TM & Q, TM M, TM & out)
  { ENERGY::CalcQTM(scal, Q, M, out); }

  static INLINE void AddQTM (double scal, const TM & Q, TM M, TM & out)
  { ENERGY::AddQTM(scal, Q, M, out); }

  // I dont think I need this in Agglomerator !
  // static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
  // { ENERGY::CalcRMBlock(mat, GetEMatrix(ed), vdi, vdj); }

  static INLINE double GetApproxWtDualEdge (const TED & eij, bool revij,
              const TED & eik, bool revik)
  {
    const typename ENERGY::TM EM_ij = revij ? ENERGY::GetEMatrix(eij.edj) : ENERGY::GetEMatrix(eij.edi);
    const typename ENERGY::TM EM_ik = revik ? ENERGY::GetEMatrix(eik.edj) : ENERGY::GetEMatrix(eik.edi);
    // double aw1 = ENERGY::GetApproxWeight(EM_ij), aw2 = ENERGY::GetApproxWeight(EM_ik);
    double aw1 = fabs(calc_trace(EM_ij)), aw2 = fabs(calc_trace(EM_ik));
    return (2 * aw1 * aw2) / (aw1 + aw2);
  }

  static INLINE void CalcRMBlock (FlatMatrix<TM> mat,
          const TVD & vi, const TVD & vj, const TVD & vk,
          const TED & eij, bool revij,
          const TED & eik, bool revik)
  {
    // 

    static typename TVD::TVD vij, vik;
    static typename ENERGY::TM EM, Qij_ik, Qik_ij;

    /** facet mid points **/
    vij = ENERGY::CalcMPData(vi.vd, vj.vd);
    vik = ENERGY::CalcMPData(vi.vd, vk.vd);

    /** half-edge mats **/
    const typename ENERGY::TM EM_ij = revij ? ENERGY::GetEMatrix(eij.edj) : ENERGY::GetEMatrix(eij.edi);
    const typename ENERGY::TM EM_ik = revik ? ENERGY::GetEMatrix(eik.edj) : ENERGY::GetEMatrix(eik.edi);

    /** trafo half-edge mats to edge-MP **/
    ENERGY::CalcQs(vij, vik, Qij_ik, Qik_ij);
    ENERGY::QtMQ(Qij_ik, EM_ij);
    ENERGY::QtMQ(Qik_ij, EM_ik);

    /** (fake-) harmonic mean (should 0.5 times harmonic mean) **/
    EM = ENERGY::HMean(EM_ij, EM_ik);

    /** Calc contrib **/
    ENERGY::CalcRMBlock2(mat, EM, vij, vik);
  }

}; // class StokesEnergy

/** END StokesEnergy **/

} // namespace amg

#endif // FILE_STOKES_ENERGY_HPP
