#ifndef FILE_ELAST_ENERGY_IMPL
#define FILE_ELAST_ENERGY_IMPL

#include "elasticity_energy.hpp"

namespace amg {

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcQ(const Vec<DIM>& t, TM& Q,
                                               double si, double sj)
{
  Q = 0;
  const double rscal = si / sj;
  Iterate<DISPPV>([&](auto i) LAMBDA_INLINE { Q(i.value, i.value) = 1.0; });
  Iterate<ROTPV>([&](auto i) LAMBDA_INLINE {
    Q(DISPPV + i.value, DISPPV + i.value) = rscal;
  });
  if constexpr (DIM == 2) {
    Q(0, 2) = -si * t(1);
    Q(1, 2) = si * t(0);
  } else {
    // Q(1,5) = - (Q(2,4) = t(0));
    // Q(2,3) = - (Q(0,5) = t(1));
    // Q(0,4) = - (Q(1,3) = t(2));
    Q(1, 5) = -(Q(2, 4) = -si * t(0));
    Q(2, 3) = -(Q(0, 5) = -si * t(1));
    Q(0, 4) = -(Q(1, 3) = -si * t(2));
  }
}  // EpsEpsEnergy::CalcQ

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::ModQ(const Vec<DIM>& t, TM& Q,
                                              double si, double sj)
{
  const double rscal = si / sj;
  Iterate<ROTPV>([&](auto i) LAMBDA_INLINE {
    Q(DISPPV + i.value, DISPPV + i.value) = rscal;
  });
  if constexpr (DIM == 2) {
    Q(0, 2) = -si * t(1);
    Q(1, 2) = si * t(0);
  } else {
    // Q(1,5) = - (Q(2,4) = t(0));
    // Q(2,3) = - (Q(0,5) = t(1));
    // Q(0,4) = - (Q(1,3) = t(2));
    Q(1, 5) = -(Q(2, 4) = -si * t(0));
    Q(2, 3) = -(Q(0, 5) = -si * t(1));
    Q(0, 4) = -(Q(1, 3) = -si * t(2));
  }
}  // EpsEpsEnergy::ModQ

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcQij(const TVD& di, const TVD& dj,
                                                 TM& Qij)
{
  Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  CalcQ(t, Qij, di.rot_scaling, sqrt(di.rot_scaling * dj.rot_scaling));
}  // EpsEpsEnergy::CalcQij

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::ModQij(const TVD& di, const TVD& dj,
                                                TM& Qij)
{
  Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  ModQ(t, Qij, di.rot_scaling, sqrt(di.rot_scaling * dj.rot_scaling));
}  // EpsEpsEnergy::ModQij

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcQHh(const TVD& dH, const TVD& dh,
                                                 TM& QHh, double glob_scale)
{
  Vec<DIM> t = glob_scale * (dh.pos - dH.pos);  // H -> h
  CalcQ(t, QHh, dH.rot_scaling, dh.rot_scaling);
}  // EpsEpsEnergy::CalcQHh

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::ModQHh(const TVD& dH, const TVD& dh,
                                                TM& QHh, double glob_scale)
{
  Vec<DIM> t = glob_scale * (dh.pos - dH.pos);  // H -> h
  ModQ(t, QHh, dH.rot_scaling, dh.rot_scaling);
}  // EpsEpsEnergy::ModQHh

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcQs(const TVD& di, const TVD& dj,
                                                TM& Qij, TM& Qji)
{
  Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  const double mid_rot_scaling = sqrt(di.rot_scaling * dj.rot_scaling);
  CalcQ(t, Qij, di.rot_scaling, mid_rot_scaling);
  t *= -1;
  CalcQ(t, Qji, dj.rot_scaling, mid_rot_scaling);
}  // EpsEpsEnergy::CalcQs

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcInvQs(const TVD& di, const TVD& dj,
                                                TM& Qij, TM& Qji)
{
  Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  const double mid_rot_scaling = sqrt(di.rot_scaling * dj.rot_scaling);
  CalcQ(t, Qji, mid_rot_scaling, dj.rot_scaling); // Qji^{-1} ~ mid -> j 
  t *= -1;
  CalcQ(t, Qij, mid_rot_scaling, di.rot_scaling); // Qij^{-1} ~ mid -> i
}  // EpsEpsEnergy::CalcQs

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcK(const TVD& di,
                                               const TVD& dj,
                                               FlatVector<TM> K)
{
  /**
   * r-scale:
   *   i    .. alpha
   *   j    .. gamma
   *  {i,j} .. beta = sqrt(alpha * gamma)    
   * S .. 0.5 * skew(t^{i->j})
   * Q =
   *  |  I   -alpha S    |
   *  |  0  beta/alpha I |
   *  | -I    beta S     | 
   *  |  0  beta/gamma I |
   * K =
   *  |       I               0       |
   *  | -alpha^{-1} S    alpha/beta I |
   *  |      -I               0       | 
   *  | -gamma^{-1} S   -gamma/beta I |
   * Such that Q^T K = 0
   * 
   * Q^T Q = K^T K =
   *  | 2I              0                 |
   *  | 0   (alpha/gamma + gamma/alpha) I |
   * 
   */
  const double &alpha = di.rot_scaling;
  const double &gamma = dj.rot_scaling;
  const double  beta = sqrt(alpha * gamma);

  K = 0.0;

  Iterate<DIM>([&](auto k) {
    K(0)(k.value,       k.value)       =  1.0;
    K(0)(k.value + DIM, k.value + DIM) =  alpha / beta;
    K(1)(k.value,       k.value)       = -1.0;
    K(1)(k.value + DIM, k.value + DIM) = -gamma / beta;
  });

  Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  
  const double  ainv = 1.0/alpha;
  const double  ginv = 1.0/gamma;

  if constexpr (DIM == 2) {
    K(0)(DIM    , 1) = -ainv * t(1);
    K(0)(DIM + 1, 0) =  ainv * t(0);
    K(1)(DIM    , 1) = -ginv * t(1);
    K(1)(DIM + 1, 0) =  ginv * t(0);
  } else {
    K(0)(DIM + 1, 2) = -(K(0)(DIM + 2, 1) = -ainv * t(0));
    K(0)(DIM + 2, 0) = -(K(0)(DIM + 0, 2) = -ainv * t(1));
    K(0)(DIM + 0, 1) = -(K(0)(DIM + 1, 0) = -ainv * t(2));
    K(1)(DIM + 1, 2) = -(K(1)(DIM + 2, 1) = -ginv * t(0));
    K(1)(DIM + 2, 0) = -(K(1)(DIM + 0, 2) = -ginv * t(1));
    K(1)(DIM + 0, 1) = -(K(1)(DIM + 1, 0) = -ginv * t(2));
  }
}  // EpsEpsEnergy::CalcK

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::ModQs(const TVD& di, const TVD& dj,
                                               TM& Qij, TM& Qji)
{
  // Vec<DIM> t = 0.5 * (dj.pos - di.pos);  // i -> j
  // const double mid_rot_scaling = sqrt(di.rot_scaling * dj.rot_scaling);
  // ModQ(t, Qij, di.rot_scaling, mid_rot_scaling);
  // t *= -1;
  // ModQ(t, Qji, dj.rot_scaling, mid_rot_scaling);

  /**
   * mid-point is choosen such that Qji = Qij^{-1}
   * This works out to
   *      p_mid = pi + lambda * (pj - pi)
   * with
   *      lambda = scale_j / (scale_i + scale_j)
   * In other words,
   *    p_mid = scale_i / (scale_i + scale_j) * pi + scale_j * (scale_i + scale_j) * pj
   *          = (1 - lambda) * pi + lambda * pj
   */

  double const lambda = dj.rot_scaling / ( di.rot_scaling + di.rot_scaling );
  const double mid_rot_scaling = sqrt(di.rot_scaling * dj.rot_scaling);

  Vec<DIM> tij = ( dj.pos - di.pos );

  Vec<DIM> tim = lambda * tij;  // i -> j
  ModQ(tim, Qij, di.rot_scaling, mid_rot_scaling);

  Vec<DIM> tjm = (lambda - 1) * tij;  // i -> j
  ModQ(tjm, Qji, dj.rot_scaling, mid_rot_scaling);
}  // EpsEpsEnergy::ModQs

template <int DIM, class TVD, class TED>
INLINE typename EpsEpsEnergy<DIM, TVD, TED>::TVD
EpsEpsEnergy<DIM, TVD, TED>::CalcMPData(const TVD& da, const TVD& db)
{
  // TVD o(0);
  // o.pos = 0.5 * (da.pos + db.pos);
  // o.rot_scaling = sqrt(da.rot_scaling * db.rot_scaling);
  // return std::move(o);

  /**
   * mid-point is choosen such that Qji = Qij^{-1}
   * This works out to
   *      p_mid = pi + lambda * (pj - pi)
   * with
   *      lambda = scale_j / (scale_i + scale_j)
   * In other words,
   *    p_mid = scale_i / (scale_i + scale_j) * pi + scale_j * (scale_i + scale_j) * pj
   *          = (1 - lambda) * pi + lambda * pj
   */

  TVD o(0);
  double const lambda = db.rot_scaling / ( da.rot_scaling + db.rot_scaling );

  o.pos         = (1 - lambda) * da.pos + lambda * db.pos;
  o.rot_scaling = sqrt(da.rot_scaling * db.rot_scaling);

  return o;
}  // EpsEpsEnergy::CalcMPData

template <int DIM, class TVD, class TED>
INLINE typename EpsEpsEnergy<DIM, TVD, TED>::TVD
EpsEpsEnergy<DIM, TVD, TED>::CalcMPDataWW(const TVD& da, const TVD& db)
{
  TVD o = CalcMPData(da, db);
  // o.wt = da.wt + db.wt;
  static TM Qij, Qji;
  CalcQs(da, db, Qij, Qji);
  SetQtMQ(1.0, o.wt, Qji, da.wt);
  AddQtMQ(1.0, o.wt, Qij, db.wt);
  return std::move(o);
}  // EpsEpsEnergy::CalcMPDataWW

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcRMBlock(FlatMatrix<TM> mat,
                                                     const TED& ed,
                                                     const TVD& vdi,
                                                     const TVD& vdj)
{
  // static TM Qij, Qji, QiM, QjM;
  // CalcQs(vdi, vdj, Qij, Qji);
  // QiM = Trans(Qij) * ed;
  // QjM = Trans(Qji) * ed;
  // mat(0, 0) = QiM * Qij;
  // mat(0, 1) = -QiM * Qji;
  // mat(1, 0) = -QjM * Qij;
  // mat(1, 1) = QjM * Qji;

  CalcRMBlock(ed, vdi, vdj, [&](auto i, auto j, auto const &val) { mat(i, j) = val; });
}  // EpsEpsEnergy::CalcRMBlock

template <int DIM, class TVD, class TED>
INLINE void EpsEpsEnergy<DIM, TVD, TED>::CalcRMBlock2(FlatMatrix<TM> mat,
                                                      const TM& ed,
                                                      const TVD& vdi,
                                                      const TVD& vdj)
{
  CalcRMBlock(mat, ed, vdi, vdj);
}  // EpsEpsEnergy::CalcRMBlock2

}  // namespace amg

#endif  // FILE_ELAST_ENERGY_IMPL