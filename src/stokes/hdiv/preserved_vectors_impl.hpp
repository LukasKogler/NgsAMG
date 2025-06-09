#ifndef FILE_PRESERVED_VECTORS_IMPL_HPP
#define FILE_PRESERVED_VECTORS_IMPL_HPP

#include "preserved_vectors.hpp"
#include <ngs_stdcpp_include.hpp>

namespace amg
{


template<class TSCAL>
void
CheckEvalsABC(FlatMatrix<TSCAL> A, LocalHeap &lh)
{
  HeapReset hr(lh);

  int const N = A.Height();

  FlatMatrix<TSCAL> evecs(N, N, lh);
  FlatVector<TSCAL> evals(N, lh);
  LapackEigenValuesSymmetric(A, evals, evecs);

  std::cout << " eval-check of " << N << " x " << N << " matrix, evals = " << std::endl;

  for (auto k : Range(N))
  {
    cout << k << ": " << evals(k) << std::endl;

    if ( abs(evals(k)) < 1e-14 )
    {
      std::cout << "     -> evec = ";
      for (auto j : Range(N))
      {
        cout << evecs(k, j) << " ";
      }
      cout << endl;
    }
  }
}

/**
 * Input:
 *  - V:     n \times k, vectors to be preserved
 *  - nPres: number of vectors to keep without modification, nPres <= k
 *           !! the first "nPres" cols of V must be orthogonal !!
 *           (that makes the computations simpler and for now I know I only need nPres = 1, so whatever...)
 *  - P:     basis functions that preserve V    (output)
 *           P ~ n \ times k, on return, first "m" cols contain values
 *  - W:     coords of preserved vectors wrt. W (output)
 *           W ~ n \ times k, on return, first "m" rows contain values
 *  - lh: working space
 * Output:
 *  Returns "m", the number of linearly independent vectors needed to represent the
 *  columns of V.
 *
 * Computes P, W such that
 *   V = P W
 * such that P ~ n \times m
 *  P = ( P_0 | P_1 ) = ( v_0, v_1, .. v_nPres | P_1)
 * where the columns of P_1 are orthonormal and W ~ m \times k
 *  W = ( W_0 | W_1 ) = ( e_0, e_1, .. e_nPres | W_1 )
 *
 * Note: DOES NOT RESET LOCALHEAP!
 */
INLINE
size_t
computePWV1(FlatMatrix<double>        V,
          size_t             const &nPres,
          FlatMatrix<double>        P,
          FlatMatrix<double>        W,
          LocalHeap                &lh, bool doco=false)
{
  size_t const maxDOFs = V.Width();
  size_t const n = V.Height();
  size_t const k = V.Width();

  // cout << " computePW, preserve " << nPres << " special BFs " << endl;
  // cout << " V: " << endl << V << endl;
  // cout << " P: " << endl << P << endl;
  // cout << " W: " << endl << W << endl;

  /** first "nPres" cols **/
  FlatVector<double> sqPColNorms(nPres, lh);

  for (auto l : Range(nPres))
  {
    P.Col(l) = V.Col(l);
    W.Col(l) = 0.0;
    W(l,l)   = 1.0;
    // pColNorms(l) = L2Norm(P.Col(l));
    sqPColNorms(l) = InnerProduct(P.Col(l), P.Col(l));
  }

  // # of linearly independent vectors
  size_t m = nPres;

  double const tol = 1e-6;

  FlatVector<double> pCol(n, lh);
  for (auto l : Range(nPres, k))
  {
    size_t const nVecsToUse = min(l, m);

    if ( doco )
    cout << " orthogonalize " << l << " against [0, " << nVecsToUse << ") " << endl;

    pCol = V.Col(l);

    double const initialColNorm = sqrt(InnerProduct(pCol, pCol));

    // cout << " pCol = " << pCol << endl;

    // first "nPres" cols of P are not normalized
    for (auto ll : Range(nPres))
    {
      double const ip = InnerProduct(pCol, P.Col(ll));
      double const fac = ip / sqPColNorms(ll);
      pCol -= fac * P.Col(ll);
      W(ll, l) = fac;
      if ( doco )
      cout << " IP " << ll << ", " << l << " =  " << ip << " -> fac = " << fac << endl;
    }

    // cout << " pCol II = " << pCol << endl;

    // subsequent cols of P ARE normalized
    for (auto ll : Range(nPres, nVecsToUse))
    {
      double const fac = InnerProduct(pCol, P.Col(ll));
      pCol -= InnerProduct(pCol, P.Col(ll)) * P.Col(ll);
      W(ll, l) = fac;
      if ( doco )
      cout << " IP " << ll << ", " << l << " =  " << fac << endl;
    }

    // cout << " pCol III = " << pCol << endl;

    for (auto ll : Range(nVecsToUse, maxDOFs))
    {
      W(ll,l) = 0.0;
    }

    double colNorm = sqrt(InnerProduct(pCol, pCol));

    // cout << " colNorm = " << colNorm << endl;
    if ( doco )
    {
      cout << "ortho col " << l << " " << colNorm << " > "
           << tol << " * " << initialColNorm
           << " = " << tol * initialColNorm
           << ", frac = " << colNorm / (tol * initialColNorm) << std::endl; 
    }

    if (colNorm > tol * initialColNorm)
    {
      double const invColNorm = 1.0 / colNorm;
      if(doco)
      cout << " -> USE V-col " << l << " -> P-col " << m << ", normalize " << invColNorm << std::endl;
      W(m,l)   = colNorm;
      P.Col(m) = invColNorm * pCol;
      m++;
    }
  }

  // cout << " -> need " << m << " coarse BFs in first cols of P = " << endl << P << endl;
  // cout << "  coarse PVEC basis: " << endl << W << endl;

  // FlatMatrix<double> PW(V.Height(), V.Width(), lh);
  // PW = P.Cols(0, m) * W.Rows(0, m);
  // cout << "  Pw = " << endl << PW << endl;

  return m;
}

INLINE
size_t
computePW(FlatMatrix<double>        V,
          size_t             const &nPres,
          FlatMatrix<double>        P,
          FlatMatrix<double>        W,
          LocalHeap                &lh)
{
  size_t const maxDOFs = V.Width();
  size_t const n = V.Height();
  size_t const k = V.Width();

  // static constexpr bool doco = false;
  // bool doco = false;

  // cout << " computePW, preserve " << nPres << " special BFs " << endl;
  // cout << " V: " << endl << V << endl;
  // cout << " P: " << endl << P << endl;
  // cout << " W: " << endl << W << endl;

  /** first "nPres" cols **/
  FlatVector<double> sqPColNorms(nPres, lh);

  for (auto l : Range(nPres))
  {
    P.Col(l) = V.Col(l);
    W.Col(l) = 0.0;
    W(l,l)   = 1.0;
    // pColNorms(l) = L2Norm(P.Col(l));
    sqPColNorms(l) = InnerProduct(P.Col(l), P.Col(l));
  }

  // # of linearly independent vectors
  size_t m = nPres;

  double const lowerTol = 1e-8;
  double const upperTol = 1e-3;

  FlatVector<double> pCol(n, lh);
  for (auto l : Range(nPres, k))
  {
    size_t const nVecsToUse = min(l, m);

    // if ( doco )
    //   cout << " orthogonalize " << l << " against [0, " << nVecsToUse << ") " << endl;

    pCol = V.Col(l);

    double const initialColNorm = sqrt(InnerProduct(pCol, pCol));

    // cout << " pCol = " << pCol << endl;

    // first "nPres" cols of P are not normalized
    for (auto ll : Range(nPres))
    {
      double const ip = InnerProduct(pCol, P.Col(ll));
      double const fac = ip / sqPColNorms(ll);
      pCol -= fac * P.Col(ll);
      W(ll, l) = fac;
      // if ( doco )
      // cout << " IP " << ll << ", " << l << " =  " << ip << " -> fac = " << fac << endl;
    }

    // cout << " pCol II = " << pCol << endl;

    // subsequent cols of P ARE normalized
    for (auto ll : Range(nPres, nVecsToUse))
    {
      double const fac = InnerProduct(pCol, P.Col(ll));
      pCol -= InnerProduct(pCol, P.Col(ll)) * P.Col(ll);
      W(ll, l) = fac;
      // if ( doco )
      // cout << " IP " << ll << ", " << l << " =  " << fac << endl;
    }

    // cout << " pCol III = " << pCol << endl;

    for (auto ll : Range(nVecsToUse, maxDOFs))
    {
      W(ll,l) = 0.0;
    }

    double colNorm = sqrt(InnerProduct(pCol, pCol));

    // if (colNorm < upperTol * initialColNorm && colNorm > lowerTol * initialColNorm)
    // {
    //   std::cout << " TURN ON output, undecided!" << std::endl;
    //   doco = true;
    // }

    // cout << " colNorm = " << colNorm << endl;
    // if ( doco )
    // {
    //   cout << "ortho col " << l << " " << colNorm << " > "
    //        << upperTol << " * " << initialColNorm
    //        << " = " << upperTol * initialColNorm
    //        << ", frac = " << colNorm / (upperTol * initialColNorm) << std::endl; 
    // }

    // if we are below lowerTol, we consider the vector to be in the range,
    // if we are above upperTol, we consider it to be linearly independent,
    // in between we re-orthogonalize to be sure 
    if (colNorm > upperTol * initialColNorm)
    {
      double const invColNorm = 1.0 / colNorm;
      // if(doco)
      // cout << " -> USE V-col " << l << " -> P-col " << m << ", normalize " << invColNorm << std::endl;
      W(m,l)   = colNorm;
      P.Col(m) = invColNorm * pCol;
      m++;
    }
    else if ( colNorm > lowerTol * initialColNorm)
    {
      static constexpr int maxTries = 2;

      for (auto numTry : Range(maxTries))
      {
        // re-orthogonalize
        for (auto ll : Range(nPres))
        {
          double const ip = InnerProduct(pCol, P.Col(ll));
          double const fac = ip / sqPColNorms(ll);
          pCol -= fac * P.Col(ll);
          W(ll, l) += fac;
          // if ( doco )
          // cout << "    try " << numTry << " IP " << ll << ", " << l << " =  " << ip << " -> fac = " << fac << endl;
        }

        for (auto ll : Range(nPres, nVecsToUse))
        {
          double const fac = InnerProduct(pCol, P.Col(ll));
          pCol -= InnerProduct(pCol, P.Col(ll)) * P.Col(ll);
          W(ll, l) += fac;
          // if ( doco )
          // cout << " try " << numTry << " IP " << ll << ", " << l << " =  " << fac << endl;
        }

        double colNormNow = sqrt(InnerProduct(pCol, pCol));
        
        // if ( doco )
        // {
        //   std::cout << " after try #" << numTry << " norm = " << colNormNow << ", frac = " << colNormNow / colNorm << ", wrt orig " << colNormNow / initialColNorm << std::endl;
        // }

        if ( colNormNow > upperTol * colNorm) // confirmed linearly independent!
        {
          // std::cout << "    -> confirmed indep, USE VEC!" << std::endl;
          double const invColNorm = 1.0 / colNorm;

          W(m,l)   = colNorm;
          P.Col(m) = invColNorm * pCol;
          m++;
          break;
        }
        else if ( colNormNow < lowerTol * colNorm ) // linearly dependent
        {
          // std::cout << "    -> DEPENDENT AFTER ALL, DISCARD!" << std::endl;
          break;
        }
      }

    }
  }

  // cout << " -> need " << m << " coarse BFs in first cols of P = " << endl << P << endl;
  // cout << "  coarse PVEC basis: " << endl << W << endl;

  // FlatMatrix<double> PW(V.Height(), V.Width(), lh);
  // PW = P.Cols(0, m) * W.Rows(0, m);
  // cout << "  Pw = " << endl << PW << endl;

  return m;
}


INLINE
size_t
computePWZF(FlatVector<double>        fFlow,
            FlatMatrix<double>        V,
            size_t             const &nPres,
            FlatMatrix<double>        P,
            FlatMatrix<double>        W,
            LocalHeap                &lh)
{
  auto numCDOFs = computePW(V, nPres, P, W, lh);

  // bool const doco = true;
  static constexpr bool doco = false;

  if (doco)
  {
    FlatMatrix<double> PW(V.Height(), V.Width(), lh);
    // FlatMatrix<double> DiffPW(V.Height(), V.Width(), lh);
    PW = P.Cols(0, numCDOFs) * W.Rows(0, numCDOFs);
    // DiffPW = PW - V;
    cout << "  V = " << endl << V << endl;
    cout << "  P = " << endl << P << endl;
    cout << "  W = " << endl << W << endl;
    cout << "  Pw = " << endl << PW << endl;
    cout << " fFlow = " ; prow(fFlow); cout << endl;
    // cout << "  DIFF = " << endl << DiffPW << endl;

    cout << " P COL FLOWS: " << endl;
    for (int j = 0; j < numCDOFs; j++)
    {
      cout << j << ": " << InnerProduct(fFlow, P.Col(j)) << endl;
    }

    // FlatMatrix<double> smallP(V.Height(), numCDOFs, lh);
    // smallP = P.Cols(0, numCDOFs);

    // FlatMatrix<double> ptp(numCDOFs, numCDOFs, lh);
    // ptp = Trans(smallP) * smallP;

    // cout << " ptp pre = " << endl << ptp << endl;
    // CheckEvalsABC(ptp, lh);
  }

  double const nFlow = InnerProduct(fFlow, P.Col(0));

  for (int j = nPres; j < numCDOFs; j++)
  {
    double const jFlow = InnerProduct(fFlow, P.Col(j));

    if (doco)
    {
      cout << " col " << j << " -= " << jFlow/nFlow << " * col 0!" << endl;
    }
    double const alpha = jFlow/nFlow;

    P.Col(j) -= alpha * P.Col(0);
    W.Row(0) += alpha * W.Row(j);

    // TODO: should we re-normalize P cols here?
  }

  if (doco)
  {
    cout << " POST P COL FLOWS: " << endl;
    for (int j = 0; j < numCDOFs; j++)
    {
      cout << j << ": " << InnerProduct(fFlow, P.Col(j)) << endl;
    }

    FlatMatrix<double> PW(V.Height(), V.Width(), lh);
    // FlatMatrix<double> DiffPW(V.Height(), V.Width(), lh);
    PW = P.Cols(0, numCDOFs) * W.Rows(0, numCDOFs);
    // DiffPW = PW - V;
    cout << "  P = " << endl << P << endl;
    cout << "  W = " << endl << W << endl;
    cout << "  V = " << endl << V << endl;
    cout << "  Pw = " << endl << PW << endl;
    // cout << "  DIFF = " << endl << DiffPW << endl;

    // FlatMatrix<double> smallP(V.Height(), numCDOFs, lh);
    // smallP = P.Cols(0, numCDOFs);

    // FlatMatrix<double> ptp(numCDOFs, numCDOFs, lh);
    // ptp = Trans(smallP) * smallP;

    // cout << " ptp POST = " << endl << ptp << endl;
    // CheckEvalsABC(ptp, lh);
  }

    return numCDOFs;
} // computePWZF

/** PreservedVectorsMap **/

template<class TMESH>
PreservedVectorsMap<TMESH> ::
PreservedVectorsMap(AgglomerateCoarseMap<TMESH> const &cmap,
                    MeshDOFs                    const &meshDOFs,
                    PreservedVectors            const &preservedVectors)
  : _cmap(cmap)
  , _fMesh(*my_dynamic_pointer_cast<TMESH>(cmap.GetMesh(),       "PreservedVectorsMap - FMESH"))
  , _cMesh(*my_dynamic_pointer_cast<TMESH>(cmap.GetMappedMesh(), "PreservedVectorsMap - CMESH"))
  , _fineMeshDOFs(meshDOFs)
  , _fineVecs(preservedVectors)
{
  _coarseMeshDOFs = make_shared<MeshDOFs>(my_dynamic_pointer_cast<TMESH>(cmap.GetMappedMesh(), "PreservedVectorsMap - CMESH"));
  _coarseMeshDOFs->Initialize();
} // PreservedVectorsMap(..)


template<class TMESH>
int
PreservedVectorsMap<TMESH> ::
computeCFBufferSize (FlatArray<int> cEdgeBufferOffset)
{
  int const nSpecial       = GetFinePreserved().GetNSpecial();
  int const nPreserved     = GetFinePreserved().GetNPreserved();
  int const maxDOFsPerEdge = nSpecial + nPreserved;

  // auto [cDOFedEdges, cDOF2E, cE2DOF] = _cMesh.GetDOFedEdges();
  auto [cDOFedEdges_SB, cDOF2E_SB, cE2DOF_SB] = _cMesh.GetDOFedEdges();
  auto &cDOFedEdges = cDOFedEdges_SB;
  auto &cDOF2E = cDOF2E_SB;
  auto &cE2DOF = cE2DOF_SB;

  /**
   * Space for:
   *   - total # of fine DOFs
   *   - #fine x #coarse prol-block
   *   - #coarse x #preserved pres. vec coordinates
  */
  int const fixedPerEdge = 1 + maxDOFsPerEdge * nPreserved;

  int totalCount = 0;

  auto c2f_edge = _cmap.template GetMapC2F<NT_EDGE>();

  _cMesh.template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &cEdge)
  {
    cEdgeBufferOffset[cEdge.id] = totalCount;

    if (cDOFedEdges->Test(cEdge.id)) // gg-edge
    {
      auto fEdges = c2f_edge[cEdge.id];

      auto const nFine = std::accumulate(fEdges.begin(), fEdges.end(), 0,
                                        [&](auto const &partialSum, auto fEnum) { return partialSum + GetFineMeshDOFs().GetNDOF(fEnum); });

      totalCount += ( fixedPerEdge + nFine * maxDOFsPerEdge );
    }
  }, false); // need to handle sg-edges in EQCs I am not master of

  return totalCount;
} // PreservedVectorsMap::computeCFBufferSize



template<class TMESH>
template<class TLAM>
INLINE void
PreservedVectorsMap<TMESH> ::
computeCFProlBlocks (FlatArray<int>     bufferOffsets,
                     FlatArray<double>  buffer,
                     TLAM               computeSpecialVecs,
                     LocalHeap         &lh)
{
  int const nSpecial       = GetFinePreserved().GetNSpecial();
  int const nPreserved     = GetFinePreserved().GetNPreserved();

  _fMesh.CumulateData();

  auto eData = get<1>(_fMesh.Data())->Data();

  /**
   * Globally, we preserve "nPreserved" vectors, and locally on each edge also the special vectors.
   * Therefore, we have AT MAX that many DOFs on each edge.
   *  (The number of special vecs per edge must be constant)
  */
  int const preservedOnEdge = nSpecial + nPreserved;
  int const maxDOFsPerEdge  = preservedOnEdge;

  // auto [cDOFedEdges, cDOF2E, cE2DOF] = _cMesh.GetDOFedEdges();
  auto [cDOFedEdges_SB, cDOF2E_SB, cE2DOF_SB] = _cMesh.GetDOFedEdges();
  auto &cDOFedEdges = cDOFedEdges_SB;
  auto &cDOF2E = cDOF2E_SB;
  auto &cE2DOF = cE2DOF_SB;

  auto c2f_edge = _cmap.template GetMapC2F<NT_EDGE>();

  // auto edgeMap = _cmap.template GetMap<NT_EDGE>();
  auto vMap = _cmap.template GetMap<NT_VERTEX>();

  _cMesh.template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &cEdge)
  {
    HeapReset hr(lh);

    if (!cDOFedEdges->Test(cEdge.id)) // gg-edge
    {
      return;
    }

    auto fEnrs = c2f_edge[cEdge.id];

    int totalFine = std::accumulate(fEnrs.begin(), fEnrs.end(), 0,
      [&](auto const &partialSum, auto const &fEnr) { return partialSum + GetFineMeshDOFs().GetNDOF(fEnr); });

    FlatMatrix<double> V(totalFine,       preservedOnEdge, lh); // fine vectors to be preserved
    FlatMatrix<double> P(totalFine,       preservedOnEdge, lh);  // prol-block
    FlatMatrix<double> W(preservedOnEdge, preservedOnEdge, lh); // coords of pres. vecs w.r.t coarse basis

    computeSpecialVecs(cEdge, V);

    for (auto l : Range(nPreserved))
    {
      totalFine = 0;

      auto fVec = GetFinePreserved().GetVector(l).FVDouble();

      int const col = nSpecial + l;

      for (auto k : Range(fEnrs))
      {
        auto const fDOFs = GetFineMeshDOFs().GetDOFNrs(fEnrs[k]);
        auto const nFine = fDOFs.Size();
        // V.Rows(totalFine, totalFine + nFine).Cols(col, col + 1) = fVec.Range(fDOFs.First(), fDOFs.Next());
        for (auto ll : Range(GetFineMeshDOFs().GetNDOF(fEnrs[k])))
        {
          V(totalFine + ll, col) = fVec(GetFineMeshDOFs().EdgeToDOF(fEnrs[k], ll));
        }
        totalFine += nFine;
      }
    }

    // bool const doco = (cEdge.id == 1969) || (cEdge.id == 1928) || (cEdge.id == 2590);
    static constexpr bool doco = false;

    if ( doco )
    {
      cout << " computePW for cEdge " << cEdge << ", fEnrs "; prow2(fEnrs); cout << endl;
      cout << " nSpecial/pres = " << nSpecial << " " << nPreserved << " -> preservedOnEdge = " << preservedOnEdge << endl;
      cout << " maxDOFsPerEdge = " << maxDOFsPerEdge << endl;
      cout << " totalFine = " << totalFine << endl;
      cout << " IN V " << endl << V << endl;
    }

    // int const nCoarse = computePW(V, nSpecial, P, W, lh);

    // int const nCoarse = computePW(V, nSpecial, P, W, lh);


    // Note: we need the flow form coarse vertex 0 to coarse vertex 1, so we have to
    //       consider change in orientation of an edge
    FlatVector<double> fFlow(totalFine, lh);
    int c = 0;
    for (auto j : Range(fEnrs))
    {
      auto const    fENr              = fEnrs[j];
      auto const   &eFlow             = eData[fENr].flow;
      auto   const &fEdge             = _fMesh.template GetNode<NT_EDGE>(fENr);
      double const  orientationFactor = ( vMap[fEdge.v[0]] == cEdge.v[0] ) ? 1.0 : -1.0;

      for (auto ll : Range(GetFineMeshDOFs().GetNDOF(fENr)))
      {
        fFlow[c++] = orientationFactor * ( ll == 0 ? eFlow[ll] : 0.0 );
      }
    }

    // if (cEdge.id == 22)
    // {
      // cout << " cEdge = " << cEdge << endl;
      // cout << " fFlow = " << fFlow << endl;
    // }

    int const nCoarse = computePWZF(fFlow, V, nSpecial, P, W, lh);

    if ( doco )
    {
      FlatMatrix<double> usedP(V.Height(), nCoarse, lh);
      FlatMatrix<double> usedW(nCoarse, nPreserved, lh);
      usedP = P.Cols(0, nCoarse);
      usedW = W.Rows(0, nCoarse);
      std::cout << " usedP: " << std::endl << usedP << endl;
      std::cout << " usedW: " << std::endl << usedP << endl;

      FlatMatrix<double> PTP(nCoarse, nCoarse, lh);
      PTP = Trans(usedP) * usedP;
      std::cout << " evals of PTP: " << std::endl;
      CheckEvalsABC(PTP, lh);

      cout << " nCoarse = " << nCoarse << endl;
      cout << " V " << endl << V << endl;
      cout << " P " << endl << P << endl;
      cout << " W " << endl << W << endl;

      FlatMatrix<double> diff(V.Height(), V.Width(), lh);
      diff = V - usedP * usedW;

      cout << " DIFF: " << endl << diff << endl;
    }

    _coarseMeshDOFs->SetNDOF(cEdge.id, nCoarse);

    int bufOffset = bufferOffsets[cEdge.id];

    // stash number of DOFs in all fine edges mapping to this edge
    buffer[bufOffset++] = totalFine;

    // stash prol-block in buffer
    for (auto k : Range(totalFine))
    {
      for (auto j : Range(nCoarse))
        { buffer[bufOffset++] = P(k, j); }
    }

    // stash coarse pres. vector coordinates in buffer (nCoarse x nTotal)
    for (auto j : Range(nPreserved))
    {
      for (auto k : Range(nCoarse)) // first nSpecial cols of "W" are identity
        { buffer[bufOffset++] = W(k, nSpecial + j); }
    }
  }, false ); // TODO: MPI - master only or not?? probably not?
} // PreservedVectorsMap::computeCFProlBlocks


template<class TMESH>
tuple<shared_ptr<MeshDOFs>, shared_ptr<PreservedVectors>>
PreservedVectorsMap<TMESH> ::
Finalize (FlatArray<int>    bufferOffsets,
          FlatArray<double> buffer)
{
  _coarseMeshDOFs->Finalize();

  int const nSpecial   = GetFinePreserved().GetNSpecial();
  int const nPreserved = GetFinePreserved().GetNPreserved();

  // auto [cDOFedEdges, cDOF2E, cE2DOF] = _cMesh.GetDOFedEdges();
  auto [cDOFedEdges_SB, cDOF2E_SB, cE2DOF_SB] = _cMesh.GetDOFedEdges();
  auto &cDOFedEdges = cDOFedEdges_SB;
  auto &cDOF2E = cDOF2E_SB;
  auto &cE2DOF = cE2DOF_SB;

  const auto &cMeshDOFs = *_coarseMeshDOFs;

  Array<shared_ptr<BaseVector>> cVecs(nPreserved);
  for (auto k : Range(nPreserved))
    { cVecs[k] = make_shared<VVector<double>>(cMeshDOFs.GetNDOF()); }

  _cMesh.template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &cEdge)
  {
    if (!cDOFedEdges->Test(cEdge.id)) // gg-edge
    {
      return;
    }

    int const nDOFs      = cMeshDOFs.GetNDOF(cEdge);
    int const nFineTotal = buffer[bufferOffsets[cEdge.id]];

    // #fine DOFs and the prol-block are stored before the coarse vec chunks!
    int buffOffset       = bufferOffsets[cEdge.id] + 1 + nFineTotal * nDOFs;

    for (auto k : Range(nPreserved)) {
      FlatVector<double> stashedData(nDOFs, buffer.Data() + buffOffset);
      buffOffset += nDOFs;
      cVecs[k]->FVDouble().Range(cMeshDOFs.GetDOFNrs(cEdge)) = stashedData;
    }
  }, false); // MPI??

  _coarseVecs = make_shared<PreservedVectors>(nSpecial, std::move(cVecs));

  return make_tuple(_coarseMeshDOFs, _coarseVecs);
} // PreservedVectorsMap::Finalize


template<class TMESH>
INLINE FlatMatrix<double>
PreservedVectorsMap<TMESH> ::
ReadProlBlockFromBuffer(FlatArray<int>            bufferOffsets,
                        FlatArray<double>         buffer,
                        int                const &cENr)
{
  int const startOffset = bufferOffsets[cENr] + 1;
  int const nRows       = buffer[bufferOffsets[cENr]];
  int const nCols       = _coarseMeshDOFs->GetNDOF(cENr);

  return FlatMatrix<double>(nRows, nCols, buffer.Data() + startOffset);
}


/** END PreservedVectorsMap **/

} // namespace amg

#endif // FILE_PRESERVED_VECTORS_IMPL_HPP