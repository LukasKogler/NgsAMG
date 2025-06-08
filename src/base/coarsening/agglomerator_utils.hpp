#ifndef FILE_AGGLOMEARTION_UTILS_HPP
#define FILE_AGGLOMEARTION_UTILS_HPP

#include <base.hpp>

#include <utils.hpp>
#include <utils_denseLA.hpp>
#include <utils_arrays_tables.hpp>
#include <utils_numeric_types.hpp>
#include <utils_io.hpp>


namespace amg
{
/** General Utility **/

/**
 *  Returns the min. generalized ev of Ex = \lambda D x, i.e.
 *  the largest lambda such that
 *    (Ex,x) / (Dx, x) > \lambda    \forall x
 */
template<class TSCAL>
INLINE TSCAL
MinGEV (FlatMatrix<TSCAL>        E,
        FlatMatrix<TSCAL>        D,
        LocalHeap               &lh,
        TSCAL             const &relTolZ = RelZeroTol<TSCAL>())
{
  static Timer t("MinGEV");
  RegionTimer rt(t);

  int const N = D.Height();

  FlatMatrix<TSCAL> evecsD(N, N, lh);
  FlatVector<TSCAL> lamsD(N, lh);

  LapackEigenValuesSymmetricLH(lh, D, lamsD, evecsD);

  TSCAL const eps = relTolZ * lamsD(N-1);

  int firstNZ = N;
  for (auto k : Range(N))
  {
    if ( lamsD(k) > eps )
      { firstNZ = k; break; }
  }

  // indices [first_nz, N) are the non-zero evals

  for (auto k : Range(firstNZ, N))
  {
    TSCAL sqInv = 1.0 / sqrt(double(lamsD(k)));

    evecsD.Row(k) *= sqInv;
  }

  // this is actaully completely unnecessary...
  // if constexpr(false) // NOTE: untested
  // {
  //   //  E projected to the "zero" evecs must be zero!
  //   int const numZero = firstNZ;

  //   FlatMatrix<double> Eproj(N, numZero, lh);

  //   // vecs are in rows
  //   auto const zeroVecsT = evecsD.Rows(0, numZero);

  //   Eproj = E * Trans(zeroVecsT);

  //   // only check diagonal entries
  //   for (auto l : Range(numZero))
  //   {
  //     // <Ev,v> must be zero
  //     TSCAL Evv = InnerPoduct(evecsD.Row(l), Eproj.Col(l));

  //     // v has L2-norm 1, so Evv = <Ev,v>/<v,v>
  //     if ( Evv > relTolZ )
  //     {
  //       // this is a vector such that Ev!=0 and Dv=0 -> Ev > lambda Dv for all lambda
  //       return TSCAL(0);
  //     }
  //   }
  // }

  int const numNZero = N - firstNZ;

  if ( numNZero == 0 )
  {
    return TSCAL(1);
  }

  FlatMatrix<TSCAL> EP(N, numNZero, lh);
  FlatMatrix<TSCAL> PEP(numNZero, numNZero, lh);

  EP = E * Trans(evecsD.Rows(firstNZ, N));
  PEP = evecsD.Rows(firstNZ, N) * EP;

  FlatVector<TSCAL> lamsPEP(numNZero, lh);
  LapackEigenValuesSymmetricLH(lh, D, lamsPEP);

  return lamsPEP(0);
} // MinGEV


template<class TSCAL, int N>
INLINE TSCAL
MinGEV (Mat<N, N, TSCAL> const &E,
        Mat<N, N, TSCAL> const &D,
        LocalHeap              &lh,
        TSCAL            const &relTolZ = RelZeroTol<TSCAL>())
{
  TSCAL avgEV = 0;

  for (auto k : Range(N))
    { avgEV += D(k,k); }

  avgEV = avgEV / N;

  // double const tolR = relTolR * avgEV;
  TSCAL const tolZ = relTolZ * avgEV;

  FlatArray<int> extNZeroRows(N, lh);
  int cntNZ = 0;

  for (auto k : Range(N))
  {
    bool const nZeroD = D(k,k) > tolZ;

    if ( nZeroD )
    {
      bool const nZeroE = E(k,k) > tolZ;

      if ( nZeroE )
      {
        extNZeroRows[cntNZ++] = k;
      }
      else
      {
        // found a vector such that Dx!=0 and Ex=0 -> E >= \lambda D only for \lambda=0!
        return TSCAL(0);
      }
    }
    // else if (nZeroE)
    // {
    //   // found a vector such that Dx=0 and Ex!=0 -> no problem, but skip row
    // }

  }

  if ( cntNZ == 0 )
  {
    return 1;
  }
  else if ( cntNZ < N )
  {
    FlatMatrix<TSCAL> smallE(cntNZ, cntNZ, lh);
    FlatMatrix<TSCAL> smallD(cntNZ, cntNZ, lh);

    // FlatArray<int> nZeroRows = extNZeroRows.Range(0, cntNZ);
    // smallE = E.Rows(nZeroRows).Cols(nZeroRows);
    // smallD = D.Rows(nZeroRows).Cols(nZeroRows);

    for (auto l : Range(cntNZ))
    {
      auto const row = extNZeroRows[l];

      for (auto ll : Range(cntNZ))
      {
        auto const col = extNZeroRows[ll];

        smallE(l, ll) = E(row, col);
        smallD(l, ll) = D(row, col);
      }
    }

    return MinGEV(smallE, smallD, lh, relTolZ);
  }
  else
  {
    FlatMatrix<TSCAL> flatE(N, N, const_cast<TSCAL*>(&E(0,0)));
    FlatMatrix<TSCAL> flatD(N, N, const_cast<TSCAL*>(&D(0,0)));

    return MinGEV(flatE, flatD, lh, relTolZ);
  }
} // MinGEV


INLINE float
MinGEV (float const &E, float const &D)
{
  return E / D;
}


INLINE double
MinGEV (double const &E, double const &D)
{
  return E / D;
}


template<AVG_TYPE avgType, class T>
INLINE T
Average (T const &a, T const &b)
{
  if      constexpr( avgType == MIN  ) { return min(a, b); }
  else if constexpr( avgType == GEOM ) { return sqrt(a * b); }
  else if constexpr( avgType == HARM ) { return 2 * (a * b) / (a + b); }
  else if constexpr( avgType == ALG  ) { return (a + b) / 2; }
  else if constexpr( avgType == MAX  ) { return max(a, b); }
  else { return T(-1); }
} // Average


template<class T>
INLINE T
Average (AVG_TYPE const &avgType, T const &a, T const &b)
{
  switch(avgType)
  {
    case(MIN):  { return min(a, b); }
    case(GEOM): { return sqrt(a * b); }
    case(HARM): { return 2 * (a * b) / (a + b); }
    case(ALG):  { return (a + b) / 2; }
    case(MAX):  { return max(a, b); }
    default:    { return -1.0; }
  }
} // Average


template<class T>
INLINE T
HalfHarmonicAvg (T const &a, T const &b)
{
  return ( a * b ) / ( a + b );
}

/** END utility **/


/** Weights **/

template<class ENERGY, class TWEIGHT, class TVD, class TM>
INLINE TWEIGHT
CalcApproxSOC (AVG_TYPE const &avgType,
               TVD      const &vdi,
               TVD      const &vdj,
               TWEIGHT  const &baseWij,
               TM       const &di,
               TM       const &dj,
               bool     const &l2Boost)
{
  TWEIGHT wij = baseWij;

  // could this interfere with the "stab-boost" which leaves some contribs in diags?
  if ( l2Boost )
  {
    wij += HalfHarmonicAvg(ENERGY::GetApproxVWeight(vdi),
                           ENERGY::GetApproxVWeight(vdj));
  }

  TWEIGHT wdi = CalcAvgTrace(di);
  TWEIGHT wdj = CalcAvgTrace(dj);

  return wij / Average(avgType, wdi, wdj);
} // CalcApproxSOC

template<class TECON, class TLAM>
void
IterateAhatBlockEC (TECON          const &eCon,
                    FlatArray<int>        vnums,
                    TLAM                  lam)
{
  for (auto row : Range(vnums))
  {
    auto const vK   = vnums[row];

    auto neibsVK = eCon.GetRowIndices(vK);
    auto eNrs    = eCon.GetRowValues(vK);

    iterate_intersection(neibsVK, vnums, [&](auto const &j, auto const &col)
    {
      auto const vJ = neibsVK[j];

      if (vJ < vK)
      {
        lam(row, vK, col, vJ, int(eNrs[j]));
      }
    });
  }
} // IterateAhatBlock


template<class ENERGY, class TSCAL, class AGG_DATA>
FlatMatrix<TSCAL>
AssembleAhatBlock (AGG_DATA       const &aggData,
                   FlatArray<int>        vNums,
                   LocalHeap            &lh)
{
  typedef typename AGG_DATA::TMU TM;

  constexpr bool ROBUST = Height<TM>() > 1;

  static_assert(Height<TM>() == 1 || Height<TM>() == ENERGY::DPV, "AssembleAharBlock");

  int const nScal = Height<TM>() * vNums.Size();

  FlatMatrix<TSCAL> Ablock(nScal, nScal, lh);
  Ablock = 0;

  IterateAhatBlockEC(
    aggData.GetEdgeCM(),
    vNums,
    [&](auto row, auto vK, auto col, auto vJ, int const &edgeNum)
    {
      // int off[2] = { row * ENERGY::DPV, col * ENERGY::DPV };

      // Note: ENERGY::CalcRMBlock will not work with TWEIGHT=float

      auto const &E = aggData.GetEdgeMatrix(edgeNum);

      if constexpr(ROBUST)
      {
        int off0 = row * ENERGY::DPV;
        int off1 = col * ENERGY::DPV;

        auto [ Qij, Qji ] = ENERGY::GetQijQji(aggData.GetVData(vK), aggData.GetVData(vJ));

        TM Qi_E = Qij.GetQTM(1.0, E);

        addTM(Ablock, off0, off0,  1.0, Qij.GetMQ(1.0, Qi_E));
        addTM(Ablock, off0, off1, -1.0, Qji.GetMQ(1.0, Qi_E));

        TM Qj_E = Qji.GetQTM(1.0, E);

        addTM(Ablock, off1, off0, -1.0, Qij.GetMQ(1.0, Qi_E));
        addTM(Ablock, off1, off1,  1.0, Qji.GetMQ(1.0, Qi_E));
      }
      else
      {
        Ablock(row, row) += E;
        Ablock(row, col) -= E;
        Ablock(col, row) -= E;
        Ablock(col, col) += E;
      }

      // ENERGY::CalcRMBlock(
      //   aggData.GetEdgeMatrix(edgeNum),
      //   aggData.GetVData(vK),
      //   aggData.GetVData(vJ),
      //   [&](auto i, auto j, auto const &val)
      //   {
      //     addTM(Ablock, off[i], off[j], 1.0, val);
      //   }
      // );
    }
  );

  return Ablock;
} // AssembleAhatBlock


template<class ENERGY, class TSCAL, class AGG_DATA>
FlatMatrix<TSCAL>
AssembleAhatBlockScal (AGG_DATA       const &aggData,
                       FlatArray<int>        vNums,
                       LocalHeap            &lh)
{
  FlatMatrix<TSCAL> Ablock(vNums.Size(), vNums.Size(), lh);
  Ablock = 0;

  IterateAhatBlock(
    aggData.GetEdgeCM(),
    vNums,
    [&](auto row, auto vK, auto col, auto vJ, int const &edgeNum)
    {
      int off[2] = { row * ENERGY::TPV, col * ENERGY::TPV };

      TWEIGHT const wt = calc_trace(aggData.GetEdgeData(edgeNum)) / ENERGY::DPV;

      Ablock(row, row) += wt;
      Ablock(row, col) -= wt;
      Ablock(col, row) -= wt;
      Ablock(col, col) += wt;
    }
  );

  return Ablock;
} // AssembleAhatBlockScal


template<class ENERGY, class AGG_DATA>
INLINE bool
AggregateWideStabilityCheck (TWEIGHT const &rho,
                             AGG_DATA const &aggData,
                             FlatArray<int> aggI,
                             FlatArray<int> aggJ,
                             bool    const &blockSmoother,
                             bool    const &useHack,
                             LocalHeap &lh)
{
  typedef typename AGG_DATA::TMU TM;
  typedef          TScal<TM>     TSCAL;
  // typedef TWEIGHT TSCAL;

  constexpr int  BS     = Height<TM>();
  constexpr bool ROBUST = BS > 1;

  int const n = aggI.Size() + aggJ.Size();

  if ( n < 3 )
    { return true; }

  int const N = BS * n;

  // sorted aggregates
  // auto allMems = merge_arrays_lh(aggI, aggJ, lh);

  FlatArray<int> allMems(n, lh);
  allMems.Part(0, aggI.Size())           = aggI;
  allMems.Part(aggI.Size(), aggJ.Size()) = aggJ;
  QuickSort(allMems);

  cout << " AggregateWideStabilityCheck, #mems = " << n << endl;
  cout << "   rho = " << rho << endl;
  cout << "   allMems: "; prow(allMems); cout << endl;

  // sub-assembled A-contributions of all members
  FlatMatrix<TSCAL> A = AssembleAhatBlock<ENERGY, TSCAL>(aggData, allMems, lh);

  // the smoother-block, either diagonal or block-diagonal,
  // always includes outside connections!
  FlatMatrix<TSCAL> M(N, N, lh);

  // start with A, or zero
  if ( blockSmoother )
  {
    M = A;
  }
  else
  {
    M = 0;
  }

  // replace diagonals with full diagonals
  for (auto k : Range(n))
  {
    setFromTM(M, k*BS, k*BS, 1.0, aggData.GetAuxDiag(allMems[k]));
  }

  /**
   * We need
   *   \forall u:   rho * inf_r (M(u-r), (u-r)) <  (Au,u)
   * Where the infimum is taken over all kernel functions (rigid bodies) r \in range(P).
   * That is,
   *   rho M < A on the space orthogonal to the kernel,
   * We can write  M = P (PTMP) PT + PorthoT M PorthoT, and just check
   *  rho M < A + rho P (PTMP) P
   *
   * That is,
   *     A - rho ( M - P (PTMP) P ) >= 0
   *
   * TODO: If M = D, PTMP can be computed much cheaper as \sum_i QmiT Mii Qmi, etc.
   *
   */

  FlatMatrix<TSCAL> P (N, BS, lh);

  // it really does NOT matter where we start
  auto &vd = aggData.GetVData(allMems[0]);

  for (auto k : Range(n))
  {
    if constexpr(ROBUST)
    {
      auto Q = ENERGY::GetQiToj(vd, aggData.GetVData(allMems[k]));
      TM Pk; SetIdentity(Pk); Q.MQ(Pk);
      setFromTM(P, k * BS, 0, 1.0, Pk);
    }
    else
    {
      P(k, 0) = 1.0;
    }
  }


  cout << " P: " << endl << P << endl;

  FlatMatrix<TSCAL> PT_M(BS, N, lh);
  PT_M = Trans(P) * M;

  FlatMatrix<TSCAL> PT_M_P(BS, BS, lh);
  PT_M_P = PT_M * P;

  cout << " PT_M_P: " << endl << PT_M_P << endl;
  CalcPseudoInverseNew(PT_M_P, lh);
  cout << " INV PT_M_P: " << endl << PT_M_P << endl;

  FlatMatrix<double> invPTMP_PT_M(BS, N, lh);
  invPTMP_PT_M = PT_M_P * PT_M;

  cout << " A: " << endl << A << endl;
  cout << " M: " << endl << M << endl;

  M -= Trans(PT_M) * invPTMP_PT_M;

  cout << " M ortho RB: " << endl << M << endl;

  A -= rho * M;

  cout << " A - rho * M: " << endl << A << endl;
  // if ( useHack )
  // {
  //   /*
  //    * Regularizing with RB does not necessarily cover the entire kernel (e.h. vertices on a line in 3d).
  //    * The hack is to just add a eps*identity matrix, do Xpotrf-based SPD-check for that.
  //    * If that succeeds, we know the all evals lam(A - rho M + eps I) > 0,
  //    * i.e. if there is any v such that Av = 0 and Bv = mu v (which is the problematic case)
  //    * it follows that mu < (eps) / (rho + eps).
  //    * That is, choosing eps = rho * TOL / (1-TOL) means that the SPD-check succeeds iff.
  //    *    For all vectors v in the kernel of A that are not in the kernel of B,
  //    *    there holds (Bv,v) < TOL
  //    * That is, we are losing only eigen-vectors of B with small Eigenvalues  with this check
  //    */
  //   TSCAL const eps = 5 * RelZeroTol<TSCAL>() * CalcAvgTrace(A);

  //   for (auto k : Range(N))
  //   {
  //     A(k,k) += eps;
  //   }

  //   return CheckForSPD(A, lh);
  // }
  // else
  {
    // use the Xpstrf-based check
    return CheckForSSPD(A, lh);
  }
} // AggregateWideStabilityCheck


template<int N, class T>
INLINE bool
PairStabilityCheckWInv (TWEIGHT    const &rho,
                    Mat<N,N,T> const &Cinv,
                    Mat<N,N,T> const &E,
                    LocalHeap        &lh)
{
  /**
   * Checks whether
   *     E - rho C >= 0
   * Under the assumption that ker(C) \subseteq ker(E),
   * since we have Cinv and not C here, we instead do the equivalent check
   *     C^{-1} E C^{-1} - rho C^{-1} >= 0
   */
  Mat<N, N, T> CinvE = Cinv * E;

  FlatMatrix<T> M(N, N, lh);

  M = E * CinvE - rho * E;

  return CheckForSSPD(M, lh);
} // PairStabilityCheckWInv


template<int N>
INLINE Mat<N,N,float>
TripleProd (Mat<N,N,float> const &A,
            Mat<N,N,float> const &B,
            Mat<N,N,float> const &C)
{
  Mat<N,N,float> X = B * C;
  return A * X;
}

INLINE float
TripleProd (float A, float B, float C)
{ return A * B * C; }

template<int N, class T>
INLINE bool
PairStabilityCheck (TWEIGHT    const &rho,
                    Mat<N,N,T> const &C,
                    Mat<N,N,T> const &E,
                    LocalHeap        &lh)
{
  /**
   * Checks whether
   *     E - rho C >= 0
   */
  FlatMatrix<T> M(N, N, lh);

  M = E - rho * C;

  return CheckForSSPD(M, lh);
} // PairStabilityCheck

template<class ENERGY, class AGG_DATA, class TVD, class TM>
INLINE void
AddNeibBoost (TM &E,
              int const &vi,
              TVD const &vdi,
              int const &vj,
              TVD const &vdj,
              int const &edgeNum,
              AGG_DATA const &aggData,
              LocalHeap &lh)
{
  static Timer t("AddNeibBoost");
  RegionTimer rt(t);
  // typedef typename AGG_DATA::TMU TM;

  static_assert(std::is_same_v<typename AGG_DATA::TMU, TM>, "AddNeibBoost - HOW??");

  // cout << " AddNeibBoost " << vi << " x " << vj << endl;

  auto neibsI = aggData.GetEdgeCM().GetRowIndices(vi);
  auto neibsJ = aggData.GetEdgeCM().GetRowIndices(vj);

  auto edgeNumsI = aggData.GetEdgeCM().GetRowValues(vi);
  auto edgeNumsJ = aggData.GetEdgeCM().GetRowValues(vj);

  auto const vdMid = ENERGY::CalcMPData(vdi, vdj);

  iterate_intersection(neibsI, neibsJ, [&](auto const &idxI,
                                           auto const &idxJ)
  {
    int const edgeNumI(edgeNumsI[idxI]);
    int const edgeNumJ(edgeNumsJ[idxJ]);

    // cout << " add common neib " << neibsI[idxI] << " = " << neibsJ[idxJ] << endl;
    // cout << "     edge " << vi << " -> " << neibsI[idxI] << " = " << edgeNumI << endl;
    // cout << "     edge " << vj << " -> " << neibsI[idxJ] << " = " << edgeNumJ << endl;

    auto const &vdn = aggData.vData[neibsI[idxI]];

    // cout << " VD i " << vdi << endl;
    // cout << " VD j " << vdj << endl;
    // cout << " VD n " << vdn << endl;

    // cout << " ED i-n " << endl; print_tm(cout, aggData.GetEdgeMatrix(edgeNumI)); cout << endl;
    // cout << " ED j-n " << endl; print_tm(cout, aggData.GetEdgeMatrix(edgeNumJ)); cout << endl;

    TM E_in = ENERGY::GetQij(vdn, vdi).GetQTMQ(1.0, aggData.GetEdgeMatrix(edgeNumI));
    TM E_jn = ENERGY::GetQij(vdn, vdj).GetQTMQ(1.0, aggData.GetEdgeMatrix(edgeNumJ));

    TM Esum = E_in + E_jn;

    // cout << " E_in: " << endl; print_tm(cout, E_in); cout << endl;
    // cout << " E_jn: " << endl; print_tm(cout, E_jn); cout << endl;
    // cout << " Esum: " << endl; print_tm(cout, Esum); cout << endl;

    CalcPseudoInverseNew(Esum, lh);

    // cout << " INV Esum: " << endl; print_tm(cout, Esum); cout << endl;

    TM halfHMean = TripleProd(E_in, Esum, E_jn);

    // cout << " halfHMean: " << endl; print_tm(cout, halfHMean); cout << endl;

    ENERGY::GetQiToj(vdMid, vdn).QTMQ(halfHMean);

    // cout << " E before: " << endl; print_tm(cout, E); cout << endl;
    E += halfHMean; // should be 0.5x I think?
    // cout << " E AFTER: " << endl; print_tm(cout, E); cout << endl;
  });
} // AddNeibBoost


template<class ENERGY, class TVD, class TM>
INLINE void
AddL2Boost (TM &E,
           int const &vi,
           TVD const &vdi,
           int const &vj,
           TVD const &vdj)
{
  TM const Li = ENERGY::GetVMatrix(vdi);
  TM const Lj = ENERGY::GetVMatrix(vdj);

  auto const tri = CalcAvgTrace(Li);
  auto const trj = CalcAvgTrace(Lj);

  auto const H = HalfHarmonicAvg(tri, trj);

  E += H/tri * Li + H/trj * Lj;
} // AddL2Boost


template<class TM>
INLINE TScal<TM>
CalcRobustPairSOCViaInvs (TM         const &Ainv,
                   TM         const &Binv,
                   TM         const &E,
                   LocalHeap        &lh,
                   TScal<TM>  const &zeroEVThresh = RelZeroTol<TScal<TM>>())
{
  /*
   * Smallest eval of Ev = lam Cv, on ortho(ker(C))
   * where C = inv(Ainv + Binv)
   *
   * That is, project A onto ortho(ker(C)), and get smallest EV of
   *    projE v = lambda projC v
   * Where projC v is diagonal, i.e. we can pre-scale E and do simple EV-symm!
   */

  constexpr int N = Height<TM>();

  TM const Cinv = Ainv + Binv;

  FlatMatrix<TScal<TM>> flatCinv(N, N, const_cast<TScal<TM>*>(&Cinv(0,0)));

  FlatVector<TScal<TM>> lamsC(N, lh);
  FlatMatrix<TScal<TM>> evsC(N, N, lh);

  LapackEigenValuesSymmetricLH(lh, flatCinv, lamsC, evsC);

  // really small Cinv evals correspond to HUGE di/dj evals, which are captured
  // by the trace-based scalar SOC so take whatver here I think
  TScal<TM> const lamThresh = 1e-6 * lamsC(N-1);

  int firstK = 0;

  for (auto k : Range(N))
  {
    if (lamsC(k) > lamThresh)
    {
      firstK = k;
      break;
    }
  }

  int Nsmall = N - firstK;

  if ( Nsmall == 0 )
  {
    return 0;
  }

  for (auto k : Range(firstK, N))
  {
    auto const sqrtLam = sqrt(lamsC(k));
    evsC.Row(k) *= sqrtLam;
  }

  FlatMatrix<TScal<TM>> ECinv(Nsmall, Nsmall, lh);
  FlatMatrix<TScal<TM>> CinvECinv(Nsmall, Nsmall, lh);

  FlatMatrix<TScal<TM>> flatE(N, N, const_cast<TScal<TM>*>(&(E(0,0))));

  ECinv     = flatE * Trans(evsC.Rows(firstK, N));
  CinvECinv =  evsC.Rows(firstK, N) * ECinv;

  FlatVector<TScal<TM>> lams(Nsmall, lh);

  LapackEigenValuesSymmetricLH(lh, CinvECinv, lams);

  return abs(lams(0));
} // CalcRobustPairSOCViaInvs


template<class TM>
INLINE TWEIGHT
CalcRobustPairSOC (TM         const &D,
                   TM         const &E,
                   LocalHeap        &lh,
                   TScal<TM>  const &zeroEVThresh = RelZeroTol<TScal<TM>>())
{
  static Timer t("CalcRobustPairSOC");
  RegionTimer rt(t);
  /*
   * Smallest eval of Ev = lam Dv, on ortho(ker(D))
   */

  constexpr int N = Height<TM>();

  FlatMatrix<TScal<TM>> flatD(N, N, const_cast<TScal<TM>*>(&D(0,0)));

  FlatVector<TScal<TM>> lamsD(N, lh);
  FlatMatrix<TScal<TM>> evsD(N, N, lh);

  LapackEigenValuesSymmetricLH(lh, flatD, lamsD, evsD);

  // cout << " lamsD: " << endl;
  // for (auto k : Range(N))
  // {
  //   cout << k << ": " << lamsD(k) << endl;
  // }

  TScal<TM> const lamThresh = zeroEVThresh * lamsD(N-1);

  // cout << " -> lamThresh = " << lamThresh << endl;

  int firstK = 0;

  for (auto k : Range(N))
  {
    if (lamsD(k) > lamThresh)
    {
      firstK = k;
      break;
    }
  }

  int Nsmall = N - firstK;

  if ( Nsmall == 0 )
  {
    return 0;
  }

  for (auto k : Range(firstK, N))
  {
    auto const sqrtILam = 1.0 / sqrt(lamsD(k));

    evsD.Row(k) *= sqrtILam;
  }

  FlatMatrix<TScal<TM>> EDinv(Nsmall, Nsmall, lh);
  FlatMatrix<TScal<TM>> DinvEDinv(Nsmall, Nsmall, lh);

  FlatMatrix<TScal<TM>> flatE(N, N, const_cast<TScal<TM>*>(&(E(0,0))));

  // cout << " Nsmall evecs: " << endl << Trans(evsD.Rows(firstK, N)) << endl;

  EDinv     = flatE * Trans(evsD.Rows(firstK, N));
  DinvEDinv = evsD.Rows(firstK, N) * EDinv;

  FlatVector<TScal<TM>> lams(Nsmall, lh);

  // cout << " DinvEDinv = " << endl << DinvEDinv << endl;

  LapackEigenValuesSymmetricLH(lh, DinvEDinv, lams);

  // cout << " ROB-SOC reduced to " << Nsmall << ", lams = " << endl;
  // for (auto k : Range(Nsmall))
  // {
  //   cout << k << ": " << lams(k) << endl;
  // }

  return std::max(TWEIGHT(0), TWEIGHT(lams(0)));
} // CalcRobustPairSOC



template<class ENERGY, class AGG_DATA, class TM>
INLINE void
PrepRobSOC (TM &E,
            TM &C,
            int const &vi,
            int const &vj,
            int const &edgeNum,
            AGG_DATA  const &aggData,
            SPWConfig const &cfg,
            LocalHeap       &lh)
{
  typedef TScal<TM> TSCAL; 

  // static_assert(std::is_same<TM, typename AGG_DATA::TMU>::value, "CheckRobSOC - type mismatch!");
  static_assert(Height<TM>() == Height<typename AGG_DATA::TMU>(), "CheckRobSOC - type mismatch!");

  auto const &vdi = aggData.GetVData(vi);
  auto const &vdj = aggData.GetVData(vj);

  E = aggData.GetEdgeMatrix(edgeNum);

  if (cfg.neibBoost)
  {
    AddNeibBoost<ENERGY>(E, vi, vdi, vj, vdj, edgeNum, aggData, lh);
  }

  auto [Qij, Qji] = ENERGY::GetQijQji(vdi, vdj);

  TM di = Qji.GetQTMQ(1.0, aggData.GetAuxDiag(vi));
  TM dj = Qij.GetQTMQ(1.0, aggData.GetAuxDiag(vj));

  // A(A+B)^perp B is the only way this works - kernel of that expression
  // is the union of kernels of A and B
  TM dSum = di + dj;

  CalcPseudoInverseNew(dSum, lh);

  C = TripleProd(di, dSum, dj);
}


template<class ENERGY, class AGG_DATA>
INLINE bool
CheckRobSOC (TWEIGHT const &rho,
             int const &vi,
             int const &vj,
             int const &edgeNum,
             AGG_DATA  const &aggData,
             SPWConfig const &cfg,
             LocalHeap       &lh)
{
  typedef typename AGG_DATA::TMU TM;

  static_assert(Height<TM>() == Height<typename AGG_DATA::TMU>(), "CheckRobSOC - type mismatch!");

  TM E, C;
  PrepRobSOC<ENERGY>(E, C, vi, vj, edgeNum, aggData, cfg, lh);

  return PairStabilityCheck(rho, C, E, lh);
} // CheckRobSOC


template<class ENERGY, class AGG_DATA>
INLINE TWEIGHT
CalcRobSOC (int const &vi,
            int const &vj,
            int const &edgeNum,
            AGG_DATA  const &aggData,
            SPWConfig const &cfg,
            LocalHeap       &lh)
{
  typedef typename AGG_DATA::TMU TM;

  static_assert(Height<TM>() == Height<typename AGG_DATA::TMU>(), "CheckRobSOC - type mismatch!");

  TM E, C;
  PrepRobSOC<ENERGY>(E, C, vi, vj, edgeNum, aggData, cfg, lh);

  constexpr TScal<TM> zeroEVThresh = 1e2 * RelZeroTol<TScal<TM>>();

  return CalcRobustPairSOC(C, E, lh, zeroEVThresh);
} // CalcRobSOC


template<class ENERGY, class AGG_DATA>
INLINE TWEIGHT
CalcRobSOCV (int const &vi,
             int const &vj,
             int const &edgeNum,
             AGG_DATA  const &aggData,
             SPWConfig const &cfg,
             LocalHeap       &lh)
{
  typedef typename AGG_DATA::TMU TM;

  static_assert(Height<TM>() == Height<typename AGG_DATA::TMU>(), "CheckRobSOC - type mismatch!");

  cout << " CalcRobSOCV " << vi << " x " << vj << endl;

  TM E, C;
  PrepRobSOC<ENERGY>(E, C, vi, vj, edgeNum, aggData, cfg, lh);

  cout << " EVALS di " << endl;
  printEvals(aggData.auxDiags[vi], lh);
  cout << " EVALS dj " << endl;
  printEvals(aggData.auxDiags[vj], lh);
  cout << " EVALS E " << endl;
  printEvals(E, lh);
  cout << " EVALS C " << endl;
  printEvals(C, lh);

  // cout << " PREP DONE " << endl;
  // cout << " E = " << endl; print_tm(cout, E); cout << endl;
  // cout << " C = " << endl; print_tm(cout, C); cout << endl;

  // RelZeroTol is a bit too strict here, that would identify
  // too many effective kernel evals as non-kernel
  constexpr TScal<TM> zeroEVThresh = 1e2 * RelZeroTol<TScal<TM>>();

  TWEIGHT wt = CalcRobustPairSOC(C, E, lh, zeroEVThresh);

  // cout << " WT = " << wt << endl;

  return wt;
} // CalcRobSOC

INLINE TWEIGHT
CalcScalVertexSOC (TWEIGHT const &edgeTrace,
                   TWEIGHT const &auxDiag)
{
  return edgeTrace / auxDiag;
} // CalcScalVertexSOC


template<class ENERGY, class TM>
INLINE bool
CheckVertexWeight (TWEIGHT   const &relThresh,
                   TWEIGHT   const &vertThresh,
                   TWEIGHT   const &maxTrOD,
                   TM        const &L,
                   TM        const &D,
                   LocalHeap       &lh)
{
  TWEIGHT const lTrace = CalcAvgTrace(L);
  TWEIGHT const dTrace = CalcAvgTrace(D);

  bool isL2Dominant = CalcScalVertexSOC(lTrace, dTrace) > vertThresh;

  if ( isL2Dominant )
  {
    isL2Dominant = lTrace > relThresh * maxTrOD;
  }

  if constexpr(Height<TM>() > 1)
  {
    if ( isL2Dominant )
    {
      isL2Dominant = MinGEV(L, D, lh) > vertThresh;
    }
  }

  return isL2Dominant;
} // CheckVertexWeight


template<class ENERGY, class TVD, class TM>
INLINE TWEIGHT
CalcApproxJoinSOC (TVD      const &vdO,
                   TVD      const &vdj,
                   TWEIGHT  const &baseWOj,
                   TM       const &dO,
                   TM       const &dj,
                   bool     const &l2Boost)
{
  // weight for orphan joining vdj
  TWEIGHT wij = baseWOj;

  if ( l2Boost )
  {
    wij += HalfHarmonicAvg(ENERGY::GetApproxVWeight(vdO),
                           ENERGY::GetApproxVWeight(vdj));
  }

  TWEIGHT wdO = CalcAvgTrace(dO);

  return wij / wdO;
} // CalcApproxJoinSOC


template<class ENERGY, class AGG_DATA>
INLINE bool
CheckRobJoinSOC (TWEIGHT const &rho,
                 int const &vO,
                 int const &vj,
                 int const &edgeNum,
                 AGG_DATA  const &aggData,
                 SPWConfig const &cfg,
                 LocalHeap       &lh)
{
  // typedef typename ENERGY::TM TM;
  typedef typename AGG_DATA::TMU TM;

  // static_assert(std::is_same<TM, typename AGG_DATA::TMU>::value, "CheckRobJoinSOC - type mismatch!");
  static_assert(Height<TM>() == Height<typename AGG_DATA::TMU>(), "CheckRobJoinSOC - type mismatch!");

  auto const &vdO = aggData.GetVData(vO);
  auto const &vdj = aggData.GetVData(vj);

  TM E = aggData.GetEdgeMatrix(edgeNum);

  if (cfg.neibBoost)
  {
    // cout << " CheckRobJoinSOC, pre-boost E = "     << endl; print_tm(cout, E); cout << endl;

    AddNeibBoost<ENERGY>(E, vO, vdO, vj, vdj, edgeNum, aggData, lh);
  }

  /**
   * Instead of the harmonic mean of the diagonals,
   * we compare only to the orphan diagonal here.
   * Ideally, we want to restrict t
   * This can be problematic with rank-deficiency,
   * but we want to prefer joining into aggs with
   * multiple (fine) edges connecting into them
   * anyways, and the neib-boost also helps.
   *
   * TODO: consider doing something differnt,
   *       e.g. a weighted harmonic AVG?
   */
  TM diagO = aggData.GetAuxDiag(vO);

  TM M = E - rho * diagO;

  // cout << " CheckRobJoinSOC, E = "     << endl; print_tm(cout, E); cout << endl;
  // cout << " CheckRobJoinSOC, diagO = " << endl; print_tm(cout, diagO); cout << endl;
  // cout << " CheckRobJoinSOC, M = "     << endl; print_tm(cout, M); cout << endl;

  FlatMatrix<TScal<TM>> flatM(Height<TM>(), Height<TM>(), const_cast<TScal<TM>*>(&M(0, 0)));

  return CheckForSSPD(flatM, lh);
} // CheckRobJoinSOC


template<class ENERGY, class AGG_DATA>
INLINE TWEIGHT
CalcRobJoinSOC (int const &vO,
                int const &vj,
                int const &edgeNum,
                AGG_DATA  const &aggData,
                SPWConfig const &cfg,
                LocalHeap       &lh)
{
  // typedef typename ENERGY::TM TM;
  typedef typename AGG_DATA::TMU TM;

  auto const &vdO = aggData.GetVData(vO);
  auto const &vdj = aggData.GetVData(vj);

  TM E = aggData.GetEdgeMatrix(edgeNum);

  if (cfg.neibBoost)
  {
    AddNeibBoost<ENERGY>(E, vO, vdO, vj, vdj, edgeNum, aggData, lh);
  }

  /**
   * Instead of the harmonic mean of the diagonals,
   * we compare only to the orphan diagonal here.
   * Ideally, we want to restrict t
   * This can be problematic with rank-deficiency,
   * but we want to prefer joining into aggs with
   * multiple (fine) edges connecting into them
   * anyways, and the neib-boost also helps.
   *
   * TODO: consider doing something differnt,
   *       e.g. a weighted harmonic AVG?
   */
  TM diagO = aggData.GetAuxDiag(vO);

  return MinGEV(E, diagO, lh);
} // CalcRobJoinSOC

/** END Weights **/

} // namespace amg

#endif // FILE_AGGLOMEARTION_UTILS_HPP
