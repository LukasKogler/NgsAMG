#ifndef FILE_AMG_GW_PROL_IMPL
#define FILE_AMG_GW_PROL_IMPL

#ifdef WITH_GW_PROL

namespace amg
{

template<int BS>
INLINE double
L2Norm(Mat<BS, BS, double> const &A)
{
  double n = 0;
  Iterate<BS>([&](auto i) {
    Iterate<BS>([&](auto j) {
      n += sqr(A(i, j));
    });
  });
  return sqrt(n);
}

INLINE double
L2Norm(double const &A)
{
  return A;
}

template<class TMESH, class TLAM>
INLINE
void
iterateFNeibs (int fvk, TMESH const &FM, TLAM lam)
{
  auto const &fecon = *FM.GetEdgeCM();

  auto neibs = fecon.GetRowIndices(fvk);
  auto eNums = fecon.GetRowValues(fvk);

  for (auto j : Range(neibs))
  {
    int eNum = eNums[j];
    lam(j, neibs[j], int(eNums[j]));
  }
}

template<class TMESH, class TLAM>
INLINE
void
iterateFNeibs (FlatArray<int> fvs, TMESH const &FM, TLAM lam)
{
  for (auto k : Range(fvs))
  {
    int const fvk = fvs[k];
    iterateFNeibs(fvk, FM, [&](auto j, auto neib, auto eNum) {
      lam(k, fvk, j, neib, eNum);
    });
  }
}

template<class TMESH, class TLAM>
void
IterateAhatBlock (TMESH          const &FM,
                  FlatArray<int>        vnums,
                  TLAM                  lam)
{
  const auto &fecon  = *FM.GetEdgeCM();
  auto        fedges = FM.template GetNodes<NT_EDGE>();
  auto        fvdata = get<0>(FM.Data())->Data();
  auto        fedata = get<1>(FM.Data())->Data();

  for (auto row : Range(vnums))
  {
    auto const vK   = vnums[row];

    auto neibsVK = fecon.GetRowIndices(vK);
    auto eNrs    = fecon.GetRowValues(vK);

    iterate_intersection(neibsVK, vnums, [&](auto const &j, auto const &col)
    {
      auto const vJ = neibsVK[j];

      if (vJ < vK) {
        int  const  fENr  = eNrs[j];
        auto const &fEdge = fedges[fENr];
        int  const  lEdge = (fEdge.v[0] == vK) ? 0 : 1;

        lam(row, vK, col, vJ, fvdata[fEdge.v[lEdge]], fvdata[fEdge.v[1 - lEdge]], fedata[fENr]);
      }
    });
  }
} // IterateAhatBlock


template<class TMESH, class TLAM_COLS, class TLAM_VALS>
void
IterateAhatRows (TMESH          const &FM,
                 FlatArray<int>        rowNums,
                 FlatArray<int>        colNums,
                 TLAM_COLS             map,
                 TLAM_VALS             consume)
{
  const auto &fecon  = *FM.GetEdgeCM();
  auto        fedges = FM.template GetNodes<NT_EDGE>();
  auto        fvdata = get<0>(FM.Data())->Data();
  auto        fedata = get<1>(FM.Data())->Data();

  for (auto rIdxK : Range(rowNums))
  {
    auto const fVk   = rowNums[rIdxK];
    auto const cVk   = map(fVk);
    auto const cIdxK = find_in_sorted_array(cVk, colNums);

    auto neibsVK = fecon.GetRowIndices(fVk);
    auto eNrs    = fecon.GetRowValues(fVk);

    for (auto idxN : Range(neibsVK))
    {
      auto const  fVj   = neibsVK[idxN];
      auto const  cVj   = map(fVj);
      auto const  rIdxJ = find_in_sorted_array(fVj, rowNums);

      // do not go through "interior" edges twice
      if ( (rIdxJ != -1) && ( fVj >= fVk ) )
       { continue; }

      auto const  cIdxJ = find_in_sorted_array(cVj, colNums);

      // ignore "outside" neighbors that do not map to the desired cols
      if (cIdxJ == -1)
        { continue; }

      int  const  fENr  = eNrs[idxN];
      auto const &fEdge = fedges[fENr];
      int const   lEdge = ( fVk == fEdge.v[0] ) ? 0 : 1;

      consume(fVk, rIdxK, cVk, cIdxK,
              fVj, rIdxJ, cVj, cIdxJ,
              fvdata[fEdge.v[lEdge]],
              fvdata[fEdge.v[1 - lEdge]],
              fedata[fENr]);
    }
  }
} // IterateAhatRows


template<class TMESH, class TLAM>
void
IterateFullAhatRows (TMESH          const &FM,
                     FlatArray<int>        rowNums,
                     TLAM                  consume)
{
  const auto &fecon  = *FM.GetEdgeCM();
  auto        fedges = FM.template GetNodes<NT_EDGE>();
  auto        fvdata = get<0>(FM.Data())->Data();
  auto        fedata = get<1>(FM.Data())->Data();

  for (auto rIdxK : Range(rowNums))
  {
    auto const fVk   = rowNums[rIdxK];

    auto neibsVK = fecon.GetRowIndices(fVk);
    auto eNrs    = fecon.GetRowValues(fVk);

    iterate_AC(neibsVK, rowNums, [&](ABC_KIND const &where, int const idxN, int const idxR)
    {
      int  const fVj = neibsVK[idxN];

      if ( ( where != INTERSECTION ) || ( fVk <= fVj ) ) // do not use interior edges twice!
      {
        int  const  fENr  = eNrs[idxN];
        auto const &fEdge = fedges[fENr];
        int  const  lEdge = ( fVk == fEdge.v[0] ) ? 0 : 1;

        consume(rIdxK, fVk, idxR, fVj, fENr,
                fvdata[fEdge.v[lEdge]],
                fvdata[fEdge.v[1 - lEdge]],
                fedata[fENr]);
      }
    });
  }
} // IterateFullAhatRows

template<int BS, class TLAM>
INLINE
void
IterateScalDiagonalBlock(SparseMat<BS, BS>  const &A,
                         FlatArray<int>           rows,
                         TLAM                     lam)
{
  for (auto kg : Range(rows))
  {
    auto fvk = rows[kg];

    auto ris = A.GetRowIndices(fvk);
    auto rvs = A.GetRowValues(fvk);

    int const offi = kg * BS;

    iterate_AC(ris, rows, [&](auto const &where, auto const &idxN, auto const &idxG)
    {
      if ( where == INTERSECTION )
      {
        int const offj = idxG * BS;

        lam(offi, offj, rvs[idxN]);
      }
    });
  }
}

template<int BS>
INLINE
void
GetDiagonalBlock(SparseMat<BS, BS>  const &A,
                 FlatArray<int>           rows,
                 FlatMatrix<double>       D,
                 bool               const zeroFirst = true)
{
  if (zeroFirst)
  {
    D = 0.0;
  }

  IterateScalDiagonalBlock<BS>(A, rows, [&](auto offi, auto offj, auto const &rv)
  {
    setFromTM(D, offi, offj, 1.0, rv);
  });
}


template<class ENERGY, class TMESH, class TM>
void
AssembleAhatBlock (TMESH          const &FM,
                   FlatArray<int>        vnums,
                   FlatMatrix<TM>        Ablocks,
                   LocalHeap            &lh)
{
  HeapReset hr(lh);

  // FlatMatrix<TM> eblock(2, 2, lh);
  // IterateAhatBlock(FM, vnums,
  //   [&](auto row, auto vK,
  //       auto col, auto vJ,
  //       auto const &dataVK,
  //       auto const &dataVJ,
  //       auto const &eData) {

  //       ENERGY::CalcRMBlock(eblock, eData, dataVK, dataVJ);

  //       Ablocks(row, row) += eblock(0, 0);
  //       Ablocks(row, col) += eblock(0, 1);
  //       Ablocks(col, row) += eblock(1, 0);
  //       Ablocks(col, col) += eblock(1, 1);
  // });

  IterateAhatBlock(FM, vnums,
    [&](auto row, auto vK,
        auto col, auto vJ,
        auto const &dataVK,
        auto const &dataVJ,
        auto const &eData)
    {
      int off[2] = { row, col };

      ENERGY::CalcRMBlock(
        eData,
        dataVK,
        dataVJ,
        [&](auto i, auto j, auto const &val)
        {
          Ablocks(off[i], off[j]) += val;
        }
      );
    }
  );
} // AssembleAhatBlock


template<class ENERGY, class TMESH, class TM>
void
AssembleAhatBlocked (TMESH                      const &FM,
                     FlatArray<int>                   vnums,
                     FlatArray<int>                   perm,
                     FlatArray<int>                   offsets,
                     FlatMatrix<FlatMatrix<TM>>       Ablocks,
                     LocalHeap                       &lh)
{
  HeapReset hr(lh);

  // bool const doPrint = (vnums.Size() == 24) && (vnums[4] == 8);

  const auto &fecon  = *FM.GetEdgeCM();
  auto        fedges = FM.template GetNodes<NT_EDGE>();
  auto        fvdata = get<0>(FM.Data())->Data();
  auto        fedata = get<1>(FM.Data())->Data();

  FlatMatrix<TM> eblock(2, 2, lh);

  int pos;

  for (auto k : Range(vnums))
  {
    auto const vK   = vnums[k];
    auto const rowK = perm[k];
    // if (doPrint) cout << "k" << k << "/" << vnums.Size() << " = " << vK << " -> rowK " << rowK << endl;;
    // auto const catK = merge_pos_in_sorted_array(rowK, offsets) - 1; // I think -1?
    int const catK = merge_pos_in_sorted_array(rowK, offsets) - 1; // I think -1?
    // if (doPrint) cout << " -> catK = " << catK << endl;
    auto const locRowK = rowK - offsets[catK];
    // if (doPrint) cout << " locRow " << locRowK << endl;

    auto neibsK    = fecon.GetRowIndices(vK);
    auto eNrs      = fecon.GetRowValues(vK);

    for (auto j : Range(k))
    {
      auto const vJ   = vnums[j];
      if ( ( pos = find_in_sorted_array(vJ, neibsK) ) != -1 )
      {
        auto const rowJ = perm[j];
        // if (doPrint) cout << "j" << j << "/" << k << " = " << vJ << " -> rowJ " << rowJ << endl;;
        // auto const catJ = merge_pos_in_sorted_array(rowJ, offsets) - 1; // I think -1?
        int const catJ = merge_pos_in_sorted_array(rowJ, offsets) - 1; // I think -1?
        // if (doPrint) cout << " -> catJ = " << catJ << endl;
        auto const locRowJ = rowJ - offsets[catJ];
        // if (doPrint) cout << " locRow " << locRowJ << endl;

        auto const  fENr  = eNrs[pos];
        auto const &fEdge = fedges[fENr];
        int  const  lEdge = (fEdge.v[0] == vK) ? 0 : 1;

        int const cat[2]    = { catK,    catJ };
        int const locRow[2] = { locRowK, locRowJ};

        ENERGY::CalcRMBlock(
          fedata[fENr],
          fvdata[fEdge.v[lEdge]],
          fvdata[fEdge.v[1 - lEdge]],
          [&](auto i, auto j, auto const &val)
          {
            Ablocks(cat[i], cat[j])(locRow[i], locRow[j]) += val;
          });

        // ENERGY::CalcRMBlock(eblock, fedata[fENr], fvdata[fEdge.v[lEdge]], fvdata[fEdge.v[1 - lEdge]]);

        // Ablocks(catK,catK)(locRowK, locRowK) += eblock(0, 0);
        // Ablocks(catK,catJ)(locRowK, locRowJ) += eblock(0, 1);
        // Ablocks(catJ,catK)(locRowJ, locRowK) += eblock(1, 0);
        // Ablocks(catJ,catJ)(locRowJ, locRowJ) += eblock(1, 1);
      }
    }
  }
} // AssembleAhatBlocked

template<class ENERGY, class TMESH, class TSPM, class TLAM>
shared_ptr<TSPM>
AssembleAhatSparse (TMESH const &FM,
                    bool  const &includeVertexContribs,
                    TLAM lamAdd,
                    BitArray *exclude = NULL)
{
  int const NV = FM.template GetNN<NT_VERTEX>();

  auto const &econ = *FM.GetEdgeCM();

  auto fvdata = get<0>(FM.Data())->Data();
  auto fedata = get<1>(FM.Data())->Data();

  Array<int> perow(NV);

  for (auto k : Range(NV))
    { perow[k] = 1 + econ.GetRowIndices(k).Size(); }

  auto spAhat = make_shared<TSPM>(perow, NV);

  for (auto k : Range(NV))
  {
    auto ris = spAhat->GetRowIndices(k);
    auto rvs = spAhat->GetRowValues(k);

    auto neibs = econ.GetRowIndices(k);

    int const pos = merge_pos_in_sorted_array(k, neibs);

    if ( pos > 0)
      { ris.Range(0, pos) = neibs.Range(0, pos); }

    ris[pos] = k;

    if ( pos + 1 < ris.Size())
      { ris.Part(pos + 1) = neibs.Part(pos); }

    rvs = 0.0;
  }

  typedef typename strip_mat<typename ENERGY::TM>::type TM;
  Matrix<TM> eblock(2, 2);

  /**
   * Only assemble master-edges, thus with CUMULATED edge-data we get
   * a C2D compatible matrix!
   */
  FM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto &fEdge)
  {
    if ( exclude && ( exclude->Test(fEdge.v[0]) || exclude->Test(fEdge.v[1]) ) )
      { return; }

    auto ris0 = spAhat->GetRowIndices(fEdge.v[0]);
    auto rvs0 = spAhat->GetRowValues(fEdge.v[0]);

    auto col00 = find_in_sorted_array(fEdge.v[0], ris0);
    auto col01 = find_in_sorted_array(fEdge.v[1], ris0);

    auto ris1 = spAhat->GetRowIndices(fEdge.v[1]);
    auto rvs1 = spAhat->GetRowValues(fEdge.v[1]);

    auto col10 = find_in_sorted_array(fEdge.v[0], ris1);
    auto col11 = find_in_sorted_array(fEdge.v[1], ris1);

    ENERGY::CalcRMBlock(eblock, fedata[fEdge.id], fvdata[fEdge.v[0]], fvdata[fEdge.v[1]]);

    lamAdd(rvs0[col00], eblock(0, 0));
    lamAdd(rvs0[col01], eblock(0, 1));
    lamAdd(rvs1[col10], eblock(1, 0));
    lamAdd(rvs1[col11], eblock(1, 1));
  }, true);

  if (includeVertexContribs)
  {
    auto &Ahat = *spAhat;

    FM.template ApplyEQ<NT_VERTEX>([&](auto eqc, const auto &vNr) {
      lamAdd(Ahat(vNr, vNr), ENERGY::GetVMatrix(fvdata[vNr]));
    }, true);
  }

  return spAhat;
} // AssembleAhatSparse

template<class TSET>
INLINE
void
ReSmoothGroupNonExpansiveCalc (FlatMatrix<double>        A_gg_inv,
                               FlatMatrix<double>        A_gC,
                               double             const &omega,
                               LocalHeap                &lh,
                               TSET                      addProlVals,
                               bool doPrint=false)
{
  static Timer t("ReSmoothGroupNonExpansiveCalc");
  RegionTimer rt(t);

  int const nGScal  = A_gg_inv.Height();
  int const nCGScal = A_gC.Width();

  FlatMatrix<double> prolValsScal(nGScal, nCGScal, lh);

  prolValsScal = -omega * A_gg_inv * A_gC;

  if ( doPrint )
  {
    cout << endl;
    cout << endl;
    Iterate<6>([&](auto l)
    {
      for (auto k : Range(nGScal / 6))
      {
        double sum = 0;
        for (auto j : Range(nCGScal / 6))
        {
          if ( l == 0 )
          {
            cout << " UPDATE " << k << "x" << j << " = " << endl;
            cout << prolValsScal.Rows(6*k, 6*(k+1)).Cols(6*j, 6*(j+1)) << endl;
          }
          sum += prolValsScal(k*6+l, j*6+l);
        }
        // sum = abs(1.0 - sum);
        if ( l == 0 )
          cout << "  UPDATE b-row " << k << "." << int(l.value) << " diagSum = " << sum << endl;
      }
    });

    Iterate<6>([&](auto l)
    {
      cout << " COMP " << l << " update: " << endl;
      for (auto k : Range(nGScal / 6))
      {
        cout << k << ": ";
        double sum = 0;
        for (auto j : Range(nCGScal / 6))
        {
          cout << prolValsScal(k*6+l, j*6+l) << " ";
        }
        cout << endl;
      }
    });

  }

  addProlVals(prolValsScal);
}

template<class TSET>
INLINE
void
ReSmoothGroupNonExpansiveCalc (FlatMatrix<double>        A_gg_inv,
                               FlatMatrix<double>        AP_gC,
                               FlatMatrix<double>        AP_gN,
                               FlatMatrix<double>        E_NC,
                               double             const &omega,
                               LocalHeap                &lh,
                               TSET                      addProlVals)
{
  int const nGScal  = A_gg_inv.Height();
  int const nCGScal = E_NC.Width();
  int const nCNScal = E_NC.Height();

  if ( nCNScal > 0 )
  {
    static Timer t("ReSmoothGroupNonExpansiveCalc - prep");
    RegionTimer rt(t);

    FlatMatrix<double> A_gC(nGScal, nCGScal, lh);
    A_gC = AP_gC;
    A_gC += AP_gN * E_NC;
    // cout << " ReSmoothGroupNonExpansiveCalc, A_gC = " << endl << A_gC << endl;
    ReSmoothGroupNonExpansiveCalc(A_gg_inv, A_gC, omega, lh, addProlVals);
  }
  else
  {
    ReSmoothGroupNonExpansiveCalc(A_gg_inv, AP_gC, omega, lh, addProlVals);
  }
}

template<class ENERGY, class TMESH, class TPMAT, class TSMAT, class TLAM, class TSET>
INLINE
void
ReSmoothGroupNonExpansiveV2 (FlatArray<int>        group,
                             FlatArray<int>        vmap,
                             double         const &omega,
                             TMESH          const &FM,
                             TMESH          const &CM,
                             TPMAT          const &CSP, // prol-mat
                             TSMAT                &A,   // sys-mat
                             TPMAT          const &CAP, // A * P
                             TLAM                  getFNeibs,
                             FlatArray<int>        prolCols,
                             TSET                  addProlVals,
                             LocalHeap            &lh,
                             double         const &cInvTolR = 1e-6,
                             double         const &cInvTolZ = 1e-6,
                             bool           const &firstReSmooth = false)// bool const &doPrint = false)
{
  constexpr int BS = Height<typename ENERGY::TM>();
  typedef typename ENERGY::TM TM;

  static Timer t("ReSmoothGroupNonExpansiveV2");
  RegionTimer rt(t);

  // const bool doPrint = group[0] == 47;
  constexpr bool doPrint = false;

  if(doPrint)
    { cout << " ReSmoothGroupNonExpansiveV2, group = "; prow(group); cout << endl; }

  HeapReset hr(lh);

  auto fVData = get<0>(FM.Data())->Data();
  auto cVData = get<0>(CM.Data())->Data();

  auto const nG    = group.Size();
  auto const nCols = prolCols.Size();

  auto const nGScal    = BS * nG;
  auto const nColsScal = BS * nCols;

  auto allF = mergeFlatArrays(1 + group.Size(), lh,
    [&](auto k) -> FlatArray<int> { return (k == 0) ? group : getFNeibs(group[k-1]); });

  auto fNeibs = setMinus(allF, group, lh);

  auto allCols = mergeFlatArrays(allF.Size(), lh,
    [&](auto k) -> FlatArray<int> { return CSP.GetRowIndices(allF[k]); });

  auto cNCols = setMinus(allCols, prolCols, lh);

  int const nC   = allCols.Size();
  int const nCG  = prolCols.Size();
  int const nCN  = cNCols.Size();
  int const nAll = allF.Size();

  int const nCScal   = nC * BS;
  int const nCGScal  = nCG * BS;
  int const nCNScal  = nCN * BS;
  int const nAllScal = nAll * BS;

  if(doPrint)
  {
    cout << " group:  "; prow(group); cout << endl;
    cout << " allF:   "; prow(allF); cout << endl;
    cout << " fNeibsL "; prow(fNeibs); cout << endl;
    cout << " allCols: "; prow(allCols); cout << endl;
    cout << " prolCols: "; prow(prolCols); cout << endl;
    cout << " cNCols: "; prow(cNCols); cout << endl;
    cout << " nC = " << nC << endl;
    cout << " nCG = " << nCG << endl;
    cout << " nCN = " << nCN << endl;
    cout << " nAll = " << nAll << endl;
    cout << " nCScal = " << nCScal << endl;
    cout << " nCGScal = " << nCGScal << endl;
    cout << " nCNScal = " << nCNScal << endl;
    cout << " nAllScal = " << nAllScal << endl;
  }

  FlatMatrix<double> A_gg(nGScal, nGScal, lh);
  GetDiagonalBlock<BS>(A, group, A_gg);

  FlatMatrix<double> A_gC(nGScal, nCGScal, lh);

  A_gC = 0.0;

  for (auto k : Range(group))
  {
    auto fvk = group[k];
    auto cvk = vmap[fvk];
    auto cPos = find_in_sorted_array(cvk, prolCols);

    auto ris = CAP.GetRowIndices(fvk);
    auto rvs = CAP.GetRowValues(fvk);

    int const offK   = k   * BS;
    int const offPos = cPos * BS;

    iterate_AC(ris, prolCols, [&](auto const &where, auto const &idxR, auto const &idxG)
    {
      if ( where == INTERSECTION ) // prol-col
      {
        int const offJ = idxG * BS;
        // setFromTM(A_gC, offK, offJ, 1.0, rvs[idxR]);
        addTM(A_gC, offK, offJ, 1.0, rvs[idxR]);
      }
      else // not a prol-col, add to contrib for CV!
      {
        auto const col = ris[idxR];

        addTM(A_gC, offK, offPos, 1.0,
              ENERGY::GetQiToj(cVData[cvk], cVData[col]).GetMQ(1.0, rvs[idxR]));
      }
    });
  }

  if (doPrint)
  {
    cout << " A_gg: " << endl; StructuredPrint(BS, A_gg); cout << endl;
  }

  auto rk = CalcPseudoInverseWithTol(A_gg, lh);

  if (doPrint)
  {
    cout << " A_gg RK = " << endl << rk << endl;
    cout << " inv A_gg: " << endl; StructuredPrint(BS, A_gg); cout << endl;
  }

  ReSmoothGroupNonExpansiveCalc(A_gg, A_gC, omega, lh, addProlVals, doPrint);
} // ReSmoothGroupNonExpansiveV2



template<class ENERGY, class TMESH>
INLINE
void
calcAuxHarmonicExtensionV1(TMESH const & M,
                           FlatMatrix<double> E,
                           FlatArray<int> all,
                           FlatArray<int> orig,
                           FlatArray<int> dest,
                           LocalHeap &lh,
                           double const &pInvTol = 1e-6,
                           bool doPrint = false)
{
  constexpr int BS = Height<typename ENERGY::TM>();

  int const nC  = all.Size();
  int const nCG = orig.Size();
  int const nCN = dest.Size();

  int const nCScal = nC * BS;
  int const nCGScal = nCG * BS;
  int const nCNScal = nCN * BS;

  FlatMatrix<double> A_NN(nCNScal, nCNScal, lh);
  FlatMatrix<double> A_NG(nCNScal, nCGScal, lh);

  FlatMatrix<typename ENERGY::TM> eBlock(2, 2, lh);

  A_NN = 0.0;
  A_NG = 0.0;

  IterateFullAhatRows(M, dest,
    [&](auto const k, auto const vK, auto const j, auto const vJ, auto const eNr,
        auto const &dataVK, auto const &dataVJ, auto const &eData)
    {
      ENERGY::CalcRMBlock(eBlock, eData, dataVK, dataVJ);

      // cout << vK << " " << vJ << " -> " << k << " " << j << endl;

      if ( j == -1 ) // outside-edge
      {
        int const col = find_in_sorted_array(vJ, orig);

        if (col != -1) // otherwise, an edge to some OTHER outside vertex
        {
          int const offk = k * BS;
          int const offj = col * BS;

          // cout << "    OUT " << offk << " " << offj << endl;

          addTM    (A_NN, offk, offk, 1.0, eBlock(0, 0));
          setFromTM(A_NG, offk, offj, 1.0, eBlock(0, 1));
        }
      }
      else
      {
        int const offk = k * BS;
        int const offj = j * BS;

        // cout << "     IN " << offk << " " << offj << endl;

        addTM    (A_NN, offk, offk, 1.0, eBlock(0, 0));
        setFromTM(A_NN, offk, offj, 1.0, eBlock(0, 1));
        setFromTM(A_NN, offj, offk, 1.0, eBlock(1, 0));
        addTM    (A_NN, offj, offj, 1.0, eBlock(1, 1));
      }
    }
  );

  // if ( doPrint)
  // {
  //   cout << " A_NN = " << endl << A_NN << endl;
  //   cout << " A_NG = " << endl << A_NG << endl;
  // }

  CalcPseudoInverseWithTol(A_NN, lh, pInvTol, pInvTol);

  // if ( doPrint)
  // {
  //   cout << " INV A_NN = " << endl << A_NN << endl;
  // }

  E = -A_NN * A_NG;
  // cout << " E: " << endl << E << endl;
} // calcAuxHarmonicExtensionV1

template<class ENERGY, class TMESH>
INLINE
void
calcAuxHarmonicExtensionV2(TMESH const & M,
                           FlatMatrix<double> E,
                           FlatArray<int> all,
                           FlatArray<int> orig,
                           FlatArray<int> dest,
                           LocalHeap &lh,
                           double const &pInvTolR = 1e-6,
                           double const &pInvTolZ = 1e-6,
                           bool doPrint = false)
{
  constexpr int BS = Height<typename ENERGY::TM>();

  int const nC  = all.Size();
  int const nCG = orig.Size();
  int const nCN = dest.Size();

  int const nCScal = nC * BS;
  int const nCGScal = nCG * BS;
  int const nCNScal = nCN * BS;

  FlatMatrix<double> A_NN(nCN, nCN, lh);
  FlatMatrix<double> A_NG(nCN, nCG, lh);

  A_NN = 0.0;
  A_NG = 0.0;

  IterateFullAhatRows(M, dest,
    [&](auto const k, auto const vK, auto const j, auto const vJ, auto const eNr,
        auto const &dataVK, auto const &dataVJ, auto const &eData)
    {
      double const wt = ENERGY::GetApproxWeight(eData);

      // cout << vK << " " << vJ << " -> " << k << " " << j << endl;

      if ( j == -1 ) // outside-edge
      {
        int const col = find_in_sorted_array(vJ, orig);

        if (col != -1) // otherwise, an edge to some OTHER outside vertex
        {
          int const offk = k * BS;
          int const offj = col * BS;

          // cout << "    OUT " << offk << " " << offj << endl;

          A_NN(k, k)   += wt;
          A_NG(k, col) -= wt;
        }
      }
      else
      {
        int const offk = k * BS;
        int const offj = j * BS;

        // cout << "     IN " << offk << " " << offj << endl;

        A_NN(k, k) += wt;
        A_NN(k, j) -= wt;
        A_NN(j, k) -= wt;
        A_NN(j, j) += wt;
      }
    }
  );

  // if (doPrint)
  // {
  //   cout << " A_NN = " << endl << A_NN << endl;
  //   cout << " A_NG = " << endl << A_NG << endl;
  // }

  CalcInverse(A_NN);

  if constexpr(BS == 1)
  {
    E = -A_NN * A_NG;
  }
  else
  {
    FlatMatrix<double> Escal(nCN, nCG, lh);
    Escal = -A_NN * A_NG;

  if (doPrint)
    cout << " Escal = " << endl << Escal << endl;

    auto vData = get<0>(M.Data())->Data();

    typename ENERGY::TM Q;
    for (auto k : Range(nCN))
    {
      auto const offk = k * BS;
      for (auto j : Range(nCG))
      {
        auto const offj = j * BS;
        ENERGY::CalcQHh(vData[orig[j]], vData[dest[k]], Q, 1.0);
        setFromTM(E, offk, offj, Escal(k, j), Q);
      }
    }
  }
  // if (doPrint)
  // cout << " E: " << endl << E << endl;
} // calcAuxHarmonicExtensionV2

template<class ENERGY, class TMESH>
INLINE
void
calcAuxHarmonicExtensionV3(TMESH const & M,
                           FlatMatrix<double> E,
                           FlatArray<int> all,
                           FlatArray<int> orig,
                           FlatArray<int> dest,
                           LocalHeap &lh,
                           double const &pInvTolR = 1e-6,
                           double const &pInvTolZ = 1e-6,
                           bool doPrint = false)
{
  typedef typename ENERGY::TM TM;
  constexpr int BS = Height<typename ENERGY::TM>();

  int const nC  = all.Size();
  int const nCG = orig.Size();
  int const nCN = dest.Size();

  // constexpr bool doPrint = false;

  if ( doPrint )
  {
    cout << "calcAuxHarmonicExtensionV3, " << endl;
    cout << " all "; prow(all);   cout << endl;
    cout << " orig "; prow(orig); cout << endl;
    cout << " dest "; prow(dest); cout << endl;
  }

  // "frame"
  constexpr int nF = 1;

  TM             A_FF;               //  Q^{F->N, T} A_NN Q^(F->N)
  FlatMatrix<TM> A_FG(nF, nCG, lh);  //  Q^{F->N, T} A_NG

  A_FF = 0.0;
  A_FG = 0.0;

  auto vData = get<0>(M.Data())->Data();
  auto eData = get<1>(M.Data())->Data();

  // single "frame" vertex, placed wherever
  typename ENERGY::TVD vdF = vData[orig[0]];

  IterateFullAhatRows(M, dest,
    [&](auto const k, auto const vK, auto const j, auto const vJ, auto const eNr,
        auto const &dataVK, auto const &dataVJ, auto const &eData)
    {
      if ( doPrint )
      {
        cout << " IterateFullAhatRows, k = " << k << " " << vK << ", j = " << j << " " << vJ << ", eNr = " << eNr << endl;
      }

      // Note: NN-edge contrib is 0 because frame is RB
      if ( j == -1 ) // N-outside edge
      {
        int const col = find_in_sorted_array(vJ, orig);

        if ( col != - 1) // N-G edge, otherwise, an edge to some OTHER outside vertex
        {
          ENERGY::CalcRMBlockRow(
            eData,
            dataVK,
            dataVJ,
            [&](auto i, auto j, auto const &val)
            {
              auto const Q_NF = ENERGY::GetQiToj(dataVK, vdF);

              if ( j == 0 ) // NN -> FF
              {
                A_FF         += Q_NF.GetQTMQ(1.0, val);
              }
              else // NG -> FG
              {
                A_FG(0, col) += Q_NF.GetQTM(1.0, val);
              }
            }
          );
        }
      }
    }
  );

  // u_N = -Q^{F->N} A_FF^{-1} A_FG

  if (doPrint)
  {
    cout << " A_FF = " << endl; print_tm(cout, A_FF); cout << endl;
    cout << " A_FG = " << endl; print_tm_mat(cout, A_FG); cout << endl;
  }

  FlatMatrix<TM> E_F(nF, nCG, lh);

  if (doPrint)
  {
    cout << " INV A_FF = " << endl; print_tm(cout, A_FF); cout << endl;
  }

  // extension orig->dest = pw * E_F
  if constexpr(BS == 1)
  {

    CalcInverse(A_FF);
    // avoid TM * FlatMatrix<TM>
    for (auto j : Range(nCG))
    {
      E_F(0, j) = -A_FF * A_FG(0, j);
    }

    for (auto k : Range(dest.Size()))
    {
      E.Row(k) = E_F;
    }
  }
  else
  {
    double trAFF = 1.0 / CalcAvgTrace(A_FF);

    FlatMatrix<double> Escal(nF, nCG, lh);
    for (auto j : Range(nCG))
    {
      Escal(0, j) = trAFF * CalcAvgTrace(A_FG(0, j));
    }

    TM invAFF = A_FF;
    FlatMatrix<double> flatAFF(BS, BS, &invAFF(0,0));

    if ( doPrint )
    cout << " A_FF: " << endl << flatAFF << endl;

    auto rk = CalcPseudoInverseWithTol(flatAFF, lh, 1e-6, 1e-6);

    if ( doPrint )
    {
      cout << " A_FF rk = " << rk << endl;
      cout << " inv AFF: " << endl << flatAFF << endl;
    }

    if ( rk == BS )
    {
      // E = - A_NN^{-1} * A_NG
      for (auto j : Range(nCG))
      {
        E_F(0, j) = - invAFF * A_FG(0, j);
      }
    }
    else
    {
      cout << " FRAME-EXT DEFICIENCY !! " << endl;

      // E = A_NN^{-1} ( A_NN Escal - A_NG )
      for (auto j : Range(nCG))
      {
        TM Q; ENERGY::CalcQHh(vData[orig[j]], vdF, Q);
        TM A_Escal = trAFF / Escal(0, j) * A_FF * Q;

        // fake-scal ext, corrected by proper frame-ext
        E_F(0, j) = invAFF * ( A_Escal  - A_FG(0, j) );
      }

    }
    if ( doPrint )
    {
      cout << " E_F = " << endl; print_tm_mat(cout, E_F); cout << endl;
    }

    for (auto k : Range(dest.Size()))
    {
      int const offk = k * BS;

      // auto const Q_FN = ENERGY::GetQiToj(vdF, vData[dest[k]]);
      TM Q_FN; ENERGY::CalcQHh(vdF, vData[dest[k]], Q_FN);

      for (auto j : Range(nCG))
      {
        int const offj = j * BS;

        // u_N = Q^{F->N} u_F = Q^{F->N} E_F u_G
        // setFromTM(E, offk, offj, Q_FN.GetQM(1.0, E_F(0, j))); // TODO: implement GetQM
        TM Q_FN_E = Q_FN * E_F(0, j);
        setFromTM(E, offk, offj, 1.0, Q_FN_E);
      }
    }
  }
} // calcAuxHarmonicExtensionV3



template<class ENERGY, class TMESH, class TSPM>
INLINE
void
calcAuxHarmonicExtensionV4(TMESH const & M,
                           TSPM  const & cA,
                           FlatMatrix<double> E,
                           FlatArray<int> all,
                           FlatArray<int> orig,
                           FlatArray<int> dest,
                           LocalHeap &lh,
                           double const &pInvTolR = 1e-9,
                           double const &pInvTolZ = 1e-9,
                           bool doPrint = false)
{
  // frame-extensionm using C-mat
  typedef typename ENERGY::TM TM;
  constexpr int BS = Height<typename ENERGY::TM>();

  int const nC  = all.Size();
  int const nCG = orig.Size();
  int const nCN = dest.Size();

  // bool const doPrint = true;
  // constexpr bool doPrint = false;

  if ( doPrint )
  {
    cout << "calcAuxHarmonicExtensionV4, " << endl;
    cout << " all "; prow(all);   cout << endl;
    cout << " orig "; prow(orig); cout << endl;
    cout << " dest "; prow(dest); cout << endl;
  }

  // "frame"
  constexpr int nF = 1;

  TM             A_FF;               //  Q^{F->N, T} A_NN Q^(F->N)
  FlatMatrix<TM> A_FG(nF, nCG, lh);  //  Q^{F->N, T} A_NG

  A_FF = 0.0;
  A_FG = 0.0;

  auto vData = get<0>(M.Data())->Data();
  auto eData = get<1>(M.Data())->Data();

  // single "frame" vertex, placed wherever
  typename ENERGY::TVD vdF = vData[orig[0]];

  for (auto k : Range(dest))
  {
    auto const row  = dest[k];
    auto const offK = row * BS;

    auto const Q_NF = ENERGY::GetQiToj(vData[row], vdF);

    auto const &vDataK = vData[row];

    auto ris = cA.GetRowIndices(row);
    auto rvs = cA.GetRowValues(row);

    iterate_AC(ris, dest, [&](auto const &where, auto const &idxCols, auto const &idxDst)
    {
      if ( where != INTERSECTION ) // skip dest-dest -> go into frame, zero energy
      {
        int const col = ris[idxCols];

        int const pos = find_in_sorted_array(col, orig);

        if ( pos != -1 )
        {
          auto const &vDataJ = vData[col];

          // cout << " diagCtrb for col " << col << ": " << endl;
          auto [Qij, Qji] = ENERGY::GetQijQji(vDataK, vDataJ);

          // cout << "   val: " << endl; print_tm(cout, rvs[idxCols]); cout << endl;

          // TM E = Qji.GetQTM(-1.0, Qij.GetMQ(1.0, rvs[idxCols]));
          // cout << "     E: " << endl; print_tm(cout, E); cout << endl;

          // A_ij = - QijT E Qji, A_ii = QijT E Qij = -A_ij Qi->j
          TM diagCtrb = ENERGY::GetQiToj(vDataK, vDataJ).GetMQ(-1.0, rvs[idxCols]);

          cout << " diagCtrb for col " << col << ": " << endl;
          print_tm(cout, diagCtrb);
          cout << endl;

          A_FF         += Q_NF.GetQTMQ(1.0, diagCtrb);
          A_FG(0, pos) += Q_NF.GetQTM(1.0, rvs[idxCols]);
        }
      }
    });
  }

  // u_N = -Q^{F->N} A_FF^{-1} A_FG

  if (doPrint)
  {
    cout << " A_FF = " << endl; print_tm(cout, A_FF); cout << endl;
    cout << " A_FG = " << endl; print_tm_mat(cout, A_FG); cout << endl;
  }

  FlatMatrix<TM> E_F(nF, nCG, lh);

  if (doPrint)
  {
    cout << " INV A_FF = " << endl; print_tm(cout, A_FF); cout << endl;
  }

  // extension orig->dest = pw * E_F
  if constexpr(BS == 1)
  {

    CalcInverse(A_FF);
    // avoid TM * FlatMatrix<TM>
    for (auto j : Range(nCG))
    {
      E_F(0, j) = -A_FF * A_FG(0, j);
    }

    for (auto k : Range(dest.Size()))
    {
      E.Row(k) = E_F;
    }
  }
  else
  {
    double trAFF = 1.0 / CalcAvgTrace(A_FF);

    FlatMatrix<double> Escal(nF, nCG, lh);
    for (auto j : Range(nCG))
    {
      Escal(0, j) = trAFF * CalcAvgTrace(A_FG(0, j));
    }

    TM invAFF = A_FF;
    FlatMatrix<double> flatAFF(BS, BS, &invAFF(0,0));
    cout << " A_FF: " << endl << flatAFF << endl;
    auto rk = CalcPseudoInverseWithTol(flatAFF, lh, 1e-12, 1e-12);

    cout << " A_FF rk = " << rk << endl;
    cout << " inv AFF: " << endl << flatAFF << endl;

    if ( rk == BS )
    {
      // E = - A_NN^{-1} * A_NG
      for (auto j : Range(nCG))
      {
        E_F(0, j) = - invAFF * A_FG(0, j);
      }
    }
    else
    {
      cout << " FRAME-EXT DEFICIENCY !! " << endl;

      // E = A_NN^{-1} ( A_NN Escal - A_NG )
      for (auto j : Range(nCG))
      {
        TM Q; ENERGY::CalcQHh(vData[orig[j]], vdF, Q);
        TM A_Escal = trAFF / Escal(0, j) * A_FF * Q;

        // fake-scal ext, corrected by proper frame-ext
        E_F(0, j) = invAFF * ( A_Escal  - A_FG(0, j) );
      }

    }
    cout << " E_F = " << endl; print_tm_mat(cout, E_F); cout << endl;

    for (auto k : Range(dest.Size()))
    {
      int const offk = k * BS;

      // auto const Q_FN = ENERGY::GetQiToj(vdF, vData[dest[k]]);
      TM Q_FN; ENERGY::CalcQHh(vdF, vData[dest[k]], Q_FN);

      for (auto j : Range(nCG))
      {
        int const offj = j * BS;

        // u_N = Q^{F->N} u_F = Q^{F->N} E_F u_G
        // setFromTM(E, offk, offj, Q_FN.GetQM(1.0, E_F(0, j))); // TODO: implement GetQM
        TM Q_FN_E = Q_FN * E_F(0, j);
        setFromTM(E, offk, offj, 1.0, Q_FN_E);
      }
    }
  }
  if (doPrint)
  cout << " E: " << endl << E << endl;
} // calcAuxHarmonicExtensionV4


template<class ENERGY, class TMESH, class TSPM>
INLINE
void
calcAuxHarmonicExtensionV5(TMESH const & FM,
                           TMESH const & CM,
                           TSPM  const & CSP,
                           FlatMatrix<double> E,
                           FlatArray<int> fineRows,
                           FlatArray<int> orig,
                           FlatArray<int> dest,
                           LocalHeap &lh,
                           double const &pInvTolR = 1e-9,
                           double const &pInvTolZ = 1e-9,
                           bool doPrint = false)
{
  /*
   * frame-extensionm using C-mat = PTAP with current prol "CSP"
   *    E = -Q (QT ADD Q)^{\perp} QT ADO
   * where
   *    [ ADD | ADO ] = PD_T Af ( PD | PO )
   * and Af is assembled from FM.
   *
   * That is, with
   *    E_F = A_FF^{\perp} A_FO
   * where E_F is 1x#orig we can compute
   *    E = -Q E_F
   * one row at a time by pre-multiplying with TQs
   */


  typedef typename ENERGY::TM TM;
  constexpr int BS = Height<typename ENERGY::TM>();

  int const nFine = fineRows.Size();
  int const nCG   = orig.Size();
  int const nCN   = dest.Size();

  // bool const doPrint = true;
  // constexpr bool doPrint = false;

  if ( doPrint )
  {
    cout << "calcAuxHarmonicExtensionV5, " << endl;
    cout << " fineRows "; prow(fineRows); cout << endl;
    cout << " orig "; prow(orig); cout << endl;
    cout << " dest "; prow(dest); cout << endl;
  }

  // "frame"
  constexpr int nF = 1;

  auto cVData = get<0>(CM.Data())->Data();

  // single "frame" vertex, placed wherever
  typename ENERGY::TVD vdF = cVData[orig[0]];

  FlatMatrix<TM> P_gO(nFine, nCG, lh);
  FlatMatrix<TM> PQ  (nFine, nF, lh);  // P_fD Q,   nFine \times nFrame

  P_gO = 0;

  for (auto k : Range(fineRows))
  {
    auto const row = fineRows[k];
    auto ris = CSP.GetRowIndices(row);
    auto rvs = CSP.GetRowValues(row);

    TM PQk = 0;

    iterate_intersection(ris, dest, [&](auto idxR, auto idxD)
    {
      PQk += ENERGY::GetQiToj(vdF, cVData[ris[idxR]]).GetMQ(1.0, rvs[idxR]);;
    });
    PQ(k, 0) = PQk;

    iterate_intersection(ris, orig, [&](auto idxR, auto idxO)
    {
      P_gO(k, idxO) = rvs[idxR];
    });
  }

  FlatMatrix<TM> A_gg(nFine, nFine, lh);
  A_gg = 0;
  AssembleAhatBlock<ENERGY>(FM, fineRows, A_gg, lh);

  if (doPrint)
  {
    cout << " PQ = " << endl;   StructuredPrint(PQ); cout << endl;
    cout << " P_gO = " << endl; StructuredPrint(P_gO); cout << endl;
    cout << " A_gg = " << endl; StructuredPrint(A_gg); cout << endl;
  }

  FlatMatrix<TM> PQT_A_gg(1, nFine, lh);
  PQT_A_gg = Trans(PQ) * A_gg;

  TM             A_FF;               //  PQ^T A_ff PQ     nFrame \times nFrame
  {
    FlatMatrix<TM> flatAFF(1, 1, &A_FF);
    flatAFF = PQT_A_gg * PQ;
  }

  // u_N = -Q^{F->N} A_FF^{-1} A_FG

  if (doPrint)
  {
    cout << " A_FF = " << endl; print_tm(cout, A_FF); cout << endl;
  }

  FlatMatrix<TM> E_F(nF, nCG, lh);

  // extension orig->dest = pw * E_F
  if constexpr(BS == 1)
  {
    FlatMatrix<TM> A_FG(nF, nCG, lh);  //  PQ^T A_ff P_fG   nFrame \times nOrig
    A_FG = PQT_A_gg * P_gO;

    CalcInverse(A_FF);
    // avoid TM * FlatMatrix<TM>
    for (auto j : Range(nCG))
    {
      E_F(0, j) = -A_FF * A_FG(0, j);
    }

    for (auto k : Range(dest.Size()))
    {
      E.Row(k) = E_F;
    }
  }
  else
  {
    double trAFF = CalcAvgTrace(A_FF);

    double trInv = trAFF > 1e-16 ? 1.0 / trAFF : 0.0;

    TM invAFF = trInv * A_FF;
    FlatMatrix<double> flatAFF(BS, BS, &invAFF(0,0));

    if (doPrint)
    {
      cout << " evals A_FF: " << endl;
      printEvals(A_FF, lh);
      cout << " evals SCALED A_FF: " << endl;
      printEvals(invAFF, lh);
      cout << " SCALED A_FF: " << endl << flatAFF << endl;
    }

    auto rk = CalcPseudoInverseWithTol(flatAFF, lh, 1e-6, 1e-6);

    if (doPrint)
    {
      cout << " A_FF rk = " << rk << endl;
      cout << " INV A_FF = " << endl; print_tm(cout, invAFF); cout << endl;
      cout << " evals INV A_FF: " << endl;
      printEvals(invAFF, lh);
    }

    invAFF *= trInv;

    if (doPrint)
    {
      cout << " BACK-SCALED INV A_FF = " << endl; print_tm(cout, invAFF); cout << endl;
      TM test = A_FF * invAFF;
      cout << " A_FF * INV A_FF = " << endl; print_tm(cout, test); cout << endl;

      cout << " evals BACK-SCALED INV A_FF: " << endl;
      printEvals(invAFF, lh);
    }

    if ( rk == 0 )
    {
      // first smoothing-iter, CSP == pWProl, P_gO is zero initially!
      // ( With V5 ext, CSP vals are used in GS-like manner though!)
      E = 0;
      return;
    }

    FlatMatrix<TM> A_FG(nF, nCG, lh);  //  PQ^T A_ff P_fG   nFrame \times nOrig
    A_FG = PQT_A_gg * P_gO;

    if (doPrint)
    {
      cout << " A_FG = " << endl; StructuredPrint(A_FG); cout << endl;
    }

    if ( rk == BS )
    {
      // E = - A_NN^{-1} * A_NG
      for (auto j : Range(nCG))
      {
        E_F(0, j) = - invAFF * A_FG(0, j);
      }
    }
    else 
    {
      // TODO: I don't think this actually matters, the kernel-component
      //       should get projected out again anyways?
      // cout << " FRAME-EXT DEFICIENCY (fix w. full rank Escal plus correction)!! " << endl;

      FlatMatrix<double> Escal(nF, nCG, lh);
      double trSum = 0;
      for (auto j : Range(nCG))
      {
        Escal(0, j) = CalcAvgTrace(A_FG(0, j));
        trSum += Escal(0, j);
      }
      Escal /= trSum;
      // cout << " Escal: " << endl << Escal << endl;

      // fake-scal ext, corrected by proper frame-ext
      //   E = Escal - A_NN^{-1} ( A_NN Escal + A_NG )
      for (auto j : Range(nCG))
      {
        // Escal = scalWt * pw
        TM Q; ENERGY::CalcQHh(cVData[orig[j]], vdF, Q);

        // scalWt * A_NN * pw - A_NG
        TM AE_M_A  = ENERGY::GetQiToj(cVData[orig[j]], vdF).GetMQ(Escal(0, j), A_FF);
        AE_M_A += A_FG(0, j);

        E_F(0, j) = Escal(0, j) * Q - invAFF * AE_M_A;
      }
    }

    if ( doPrint )
    {
      cout << " E_F = " << endl; StructuredPrint(E_F); cout << endl;
    }

    for (auto k : Range(dest.Size()))
    {
      int const offk = k * BS;

      // auto const Q_FN = ENERGY::GetQiToj(vdF, vData[dest[k]]);
      TM Q_FN; ENERGY::CalcQHh(vdF, cVData[dest[k]], Q_FN);

      for (auto j : Range(nCG))
      {
        int const offj = j * BS;

        // u_N = Q^{F->N} u_F = Q^{F->N} E_F u_G
        // setFromTM(E, offk, offj, Q_FN.GetQM(1.0, E_F(0, j))); // TODO: implement GetQM
        TM Q_FN_E = Q_FN * E_F(0, j);
        setFromTM(E, offk, offj, 1.0, Q_FN_E);
      }
    }
  }
  if (doPrint)
  {
    cout << " E: " << endl;
    StructuredPrint(BS, E);
    cout << endl;
  }
} // calcAuxHarmonicExtensionV5



INLINE void
checkRowSums (std::string name, FlatMatrix<double> mat, int BS)
{
  int N = min(3, BS);

  int r = mat.Height() / BS;
  int c = mat.Width() / BS;

  cout << " checkRowSums " << name << ", " << r << " x " << c << ": " << endl;

  for (auto k : Range(r))
  {
    for (auto ck : Range(N))
    {
      auto row = k * BS + ck;
      for (auto cj : Range(N))
      {
        double sum = 0;
        double absSum = 0;
        for (auto j : Range(c))
        {
          double v = mat(row, j * BS + cj);
          sum += v;
          absSum += abs(v);
        }
        cout << "   row (" << k << ", " << ck << "), comp " << cj << ": " << sum << "  vs absSum " << absSum << ", rel = " << sum/absSum << endl;
     }
    }
  }
} // checkRowSums



template<class ENERGY, class TMESH, class TPMAT, class TSMAT, class TM>
INLINE
void
ReSmoothRowNonExpansiveV2 (int            const &fvnr,
                           int            const &CV,
                           double         const &omega,
                           TMESH          const &FM,
                           TMESH          const &CM,
                           TPMAT          const &CSP,   // prol-mat
                           TSMAT                *fmat,  // sys-mat
                           FlatArray<int>        fineNeibs,
                           FlatArray<int>        rowCols,
                           FlatVector<TM>        rowVals,
                           Array<int>           &fineNeighborhood,
                           Array<int>           &otherCols,
                           LocalHeap            &lh,
                           bool           const &considerPrint = false)
{
  /**
    *  Re-smooth row fvnr:
    *     I) Find an extension from rowCols to fine neighborhood (using AUX-mat)
    *          (i)   get prol-block fineNeighborhood <- "all" coarse cols
    *          (ii)  set up local coarse matrix for "all" cols:
    *                   - rowCols
    *                   - all other cols appearing in fineNeighborhood are
    *                     represented by A SINGLE VERTEX
    *          (iii) extension rowCols -> (single) otherCol -> fineNeighborhood
    *    II) Solve for fvnr                                      (either AUX or real mat)
    */

  static Timer t("ReSmoothRowNonExpansiveV2");
  RegionTimer rt(t);

  HeapReset hr(lh);

  auto fvdata = get<0>(FM.Data())->Data();
  auto cvdata = get<0>(CM.Data())->Data();

  /**
   *  When we have access to the real system matrix, we use the row from that
   *  in (II), otherwise we fall back to using the row of the auxiliary matrix.
   */
  bool const have_fmat = (fmat != nullptr);

  constexpr bool doPrint = false;
  // bool const doPrint = (fvnr == 47) && ( CV == 58 );

  if (doPrint)
  {
    cout << endl << " IMPROVE " << fvnr << endl;
    cout << " fineNeibs "; prow2(fineNeibs); cout << endl;
  }

  /** prep **/

  int pos;

  // FlatArray<int> fineNeighborhood(fineNeibs.Size() + 1, lh);

  // pos = merge_pos_in_sorted_array(fvnr, fineNeibs);

  // if ( (pos == -1) || ((pos > 0) && (fineNeibs[pos-1] == fvnr)) )
  //   { fineNeighborhood = fineNeibs; }
  // else
  // {
  //   if (pos > 0)
  //     { fineNeighborhood.Range(0, pos) = fineNeibs.Range(0, pos); }

  //   fineNeighborhood[pos] = fvnr;

  //   if (pos < fineNeibs.Size());
  //     { fineNeighborhood.Part(pos + 1) = fineNeibs.Part(pos); }
  // }

  fineNeighborhood.SetSize(fineNeibs.Size());
  fineNeighborhood = fineNeibs;
  insert_into_sorted_array_nodups(fvnr, fineNeighborhood);

  auto const nF = fineNeighborhood.Size();

  int const diagPos = find_in_sorted_array(fvnr, fineNeighborhood);


  otherCols.SetSize0();
  for (auto k : Range(fineNeighborhood))
  {
    auto const vK    = fineNeighborhood[k];
    auto       kCols = CSP.GetRowIndices(vK);

    for (auto l : Range(kCols))
    {
      auto const col = kCols[l];
      if ( ( pos = find_in_sorted_array(col, rowCols)) == -1 )
        { insert_into_sorted_array_nodups(col, otherCols); }
    }
  }

  auto const nRC     = rowCols.Size();
  auto const offsetO = nRC;
  auto const nOC     = min(1ul, otherCols.Size());
  // auto const nOC     = otherCols.Size();
  auto const nC      = nRC + nOC;

  auto coarsePos = [&](auto col) -> tuple<bool, int> {
    if ( ( pos = find_in_sorted_array(col, rowCols) ) != -1 )
      { return make_tuple(false, pos); }
    else
      { return make_tuple(true, nRC); }
      // { return make_tuple(false, nRC + find_in_sorted_array(col, otherCols)); }
  };

  if (doPrint)
  {
    cout << " rowCols "; prow2(rowCols); cout << endl;
    cout << " otherCols "; prow2(otherCols); cout << endl;
  }

  auto getProlBlock = [&](FlatMatrix<TM> Pblock)
  {
    Pblock = 0.0;

    TM QHh;

    for (auto k : Range(fineNeighborhood))
    {
      auto vK = fineNeighborhood[k];

      auto colsK = CSP.GetRowIndices(vK);
      auto valsK = CSP.GetRowValues(vK);

      if (colsK.Size() == 0) // Dirichlet - prol from cvnr to maintin RBs
      {
        auto [isOther, locCol] = coarsePos(CV);
        ENERGY::CalcQHh(cvdata[CV], fvdata[vK], Pblock(k, locCol), 1.0);
        // auto Q = ENERGY::GetQiToj(cvdata[CV], fvdata[vK]);
        // SetIdentity(Pblock(k, locCol));
        // Q.MQ(Pblock(k, locCol));
      }
      else
      {
        for (auto l : Range(colsK))
        {
          auto const col    = colsK[l];

          auto [isOther, locCol] = coarsePos(col);

          if (isOther)
          {
            // base the fictitious neib wherever, just use first col
            ENERGY::CalcQHh(cvdata[rowCols[0]], cvdata[col], QHh, 1.0);
            // need to add here since we can have the fict. col multiple times
            Pblock(k, locCol) += valsK[l] * QHh;
          }
          else
          {
            Pblock(k, locCol) = valsK[l];
          }
        }
      }
    }
  };

  FlatMatrix<TM> AfRow (1,  nF, lh);
  FlatMatrix<TM> P     (nF, nRC, lh);

  if (nOC == 0)
  {
    // no additional cols - can just directly get the prol-block!
    getProlBlock(P);

    if (!have_fmat)
    {
      AfRow = 0;

      FlatMatrix<TM> eBlock  (2, 2, lh);

      IterateAhatBlock(FM, fineNeighborhood, [&](auto k, auto vK,
                                                 auto j, auto vJ,
                                                 auto const &dataVK, auto const &dataVJ,
                                                 auto const &eData)
      {
        if (k == diagPos)
        {
          ENERGY::CalcRMBlock(eBlock, eData, dataVK, dataVJ);
          AfRow(0, diagPos) += eBlock(0, 0);
          AfRow(0, j)       += eBlock(0, 1);
        }
        else if (j == diagPos)
        {
          ENERGY::CalcRMBlock(eBlock, eData, dataVK, dataVJ);
          AfRow(0, diagPos) += eBlock(1, 1);
          AfRow(0, k)       += eBlock(1, 0);
        }
      });
    }
  }
  else
  {
    /** I.(i) get prol-block */

    FlatMatrix<TM> Ploc(nF, nC, lh);

    getProlBlock(Ploc);

    if (doPrint)
    {
      // cout << "Ploc = " << endl; print_tm_mat(cout, Ploc); cout << endl;
      cout << " Ploc = " << endl; StructuredPrint(Ploc); cout << endl;
      if constexpr(Height<TM>() == 6)
      {
        Iterate<6>([&](auto l)
        {
          for (auto k : Range(Ploc.Height()))
          {
            double sum = 0;

            for (auto j : Range(Ploc.Width()))
            {
              sum += Ploc(k, j)(l.value, l.value);
            }
            sum = abs(1.0 - sum);
            cout << " Ploc row " << k << "." << int(l.value) << " diagSum = " << sum << endl;
          }
        });
      }
    }

    /** I.(ii) assemble AC */

    FlatMatrix<TM> AC      (nC, nC, lh);
    // FlatMatrix<TM> eBlock  (2, 2, lh);
    // FlatMatrix<TM> eBlock_P(2, nC, lh);

    {
      HeapReset hr(lh);
      FlatMatrix<TM> Af   (nF,  nF, lh);
      Af = 0.0;
      AssembleAhatBlock<ENERGY>(FM, fineNeighborhood, Af, lh);
      if (doPrint)
      {
        cout << " Af: " << endl;
        StructuredPrint(Af);
        cout << endl;
      }
      FlatMatrix<TM> Af_Ploc(nF, nC, lh);
      Af_Ploc = Af * Ploc;

      // clang compatibility
      // AfRow = Af.Row(diagPos);
      for (auto j : Range(nF))
      {
        AfRow(0, j) = Af(diagPos, j);
      }

      AC = Trans(Ploc) * Af_Ploc;
    }

    if (doPrint)
    {
      cout << " AC = " << endl; StructuredPrint(AC); cout << endl;
    }

    /**
      * I. (iii) extension rowCols -> otherCols -> fineNeighborhood
      *          - rowCols -> otherCols:        solve A_O_O P + A_O_R = 0
      *          - rowCols -> fineNeighborhood: oldProl * (I, ext)
      */

    FlatMatrix<TM> EC   (nOC, nRC, lh);
    FlatMatrix<TM> A_O_O(nOC, nOC, lh);

    A_O_O = AC.Rows(nRC, nC).Cols(nRC, nC);

    FlatMatrix<TM> AOO(nOC, nOC, lh);
    AOO = A_O_O;

    if (doPrint)
    {
      cout << " A_O_O: " << endl; print_tm_mat(cout, A_O_O); cout << endl;
      cout << " EVALS INV A_O_O " << endl;
      printEvals(AOO, lh);
      AOO = A_O_O;
    }

    CalcPseudoInverseWithTol(A_O_O, lh, 1e-6, 1e-6);
    // CalcPseudoInverseNew(A_O_O, lh);
    // EC = -A_O_O * AC.Rows(nRC, nC).Cols(0, nC); // <- this appears to give the WRONG RESULT!

    if (doPrint)
    {
      FlatMatrix<TM> ID(nOC, nOC, lh);
      ID = A_O_O * AOO;
      cout << " A_O_O x inv A_O_O: " << endl; print_tm_mat(cout, ID); cout << endl;
    }


    FlatMatrix<TM> A_O_C(nOC, nRC, lh);
    A_O_C = AC.Rows(nRC, nC).Cols(0, nC);

    EC = -A_O_O * A_O_C;

    if (doPrint)
    {
      cout << " inv A_O_O " << endl; print_tm_mat(cout, A_O_O); cout << endl;
      cout << " A_O_C " << endl; print_tm_mat(cout, A_O_C); cout << endl;
      // cout << "EC = " << endl; print_tm_mat(cout, EC); cout << endl;
    //   cout << " C0 EC = " << endl; PrintComponentTM(0, EC); cout << endl;
      cout << " EC = " << endl; print_tm_mat(cout, EC); cout << endl;
    }

    P = Ploc.Cols(0, nRC);
    P += Ploc.Cols(nRC, nC) * EC;

    if (doPrint)
    {
      // cout << "P = " << endl; print_tm_mat(cout, P); cout << endl;
      cout << " P = " << endl; StructuredPrint(P); cout << endl;

      if constexpr(Height<TM>() == 6)
      {
        Iterate<6>([&](auto l)
        {
          for (auto k : Range(P.Height()))
          {
            double sum = 0;

            for (auto j : Range(P.Width()))
            {
              sum += P(k, j)(l.value, l.value);
            }
            sum = abs(1.0 - sum);
            cout << " P row " << k << "." << int(l.value) << " diagSum = " << sum << endl;
          }
        });
      }
    }
  }


  /** (II): smooth */

  TM fineDiag;
  TM proj;
  if (have_fmat)
  {
    auto ris = fmat->GetRowIndices(fvnr);
    auto rvs = fmat->GetRowValues(fvnr);

    for (auto j : Range(ris))
    {
      // do I need to search here or must ris and fineNeighborhood be the same??
      // cout << "  -> READ " << j << "/" << ris.Size() << ", col " << ris[j] << ", as FN " << fineNeighborhood[j] << endl;
      AfRow(0, j) = rvs[j];
      if (ris[j] == fvnr)
        { fineDiag = rvs[j]; }
    }
  }
  else
  {
    fineDiag = AfRow(0, diagPos);
  }

  TM fDI = fineDiag;
  CalcPseudoInverseNew(fDI, lh);

  if (doPrint)
  {
    if constexpr(Height<TM>() == 6)
    {
      Iterate<6>([&](auto l)
      {
        for (auto k : Range(AfRow.Height()))
        {
          double sum = 0;

          for (auto j : Range(AfRow.Width()))
          {
            sum += AfRow(k, j)(l.value, l.value);
          }
          // sum = abs(1.0 - sum);
          cout << " AF row " << k << "." << int(l.value) << " diagSum = " << sum << endl;
        }
      });
    }
  }

  // SetIdentity(proj);
  // proj -= omega * fDI * fineDiag;

  FlatMatrix<TM> AP(1, nRC, lh);
  AP = AfRow * P;

  if (doPrint)
  {
    if constexpr(Height<TM>() == 6)
    {
      Iterate<6>([&](auto l)
      {
        for (auto k : Range(AP.Height()))
        {
          double sum = 0;

          for (auto j : Range(AP.Width()))
          {
            sum += AP(k, j)(l.value, l.value);
          }
          // sum = abs(1.0 - sum);
          cout << " AP row " << k << "." << int(l.value) << " diagSum = " << sum << endl;
        }
      });
    }
  }

  if (doPrint)
  {
    cout << " finediag: " << endl; StructuredPrint(fineDiag); cout << endl;
    cout << " INV findDiag: " << endl; StructuredPrint(fDI); cout << endl;
    cout << " AfRow: " << endl;StructuredPrint(AfRow); cout << endl;
    cout << " AP: " << endl; StructuredPrint(AP); cout << endl;
  }

  // TM totalUpdate = 0;
  for (auto j : Range(rowCols))
  {
    // old vals are P, new are (P - omega Dinv A P)
    // if (doPrint)
    // {
    //   TM update = omega * fDI * AP(0, j);
    //   TM newVal = rowVals[j] - update;
    //   if (doPrint)
    //   {
    //     cout << " UPDATE " << j << "/" << rowCols.Size() << ", col " << rowCols[j] << endl;
    //     cout << "    OLD " << endl; print_tm(cout, rowVals[j]); cout << endl;
    //     cout << "    -UP " << endl; print_tm(cout, update); cout << endl;
    //     cout << "    NEW " << endl; print_tm(cout, newVal); cout << endl;
    //   }
    //   totalUpdate += update;
    // }
    rowVals[j] -= omega * fDI * AP(0, j);
  }

  // if (doPrint)
  // {
  //   // cout << " TOTAL UPDATE = " << endl; print_tm(cout, totalUpdate); cout << endl;

  //   bool isBad = false;
  //   if constexpr(Height<TM>() == 6)
  //   {
  //       Iterate<6>([&](auto l)
  //       {
  //         if ( fabs(totalUpdate(l, l))  > 1e-5)
  //         {
  //           std::cout << " UPDATE FOR " << fvnr << ", comp " << int(l.value) << " is BAD!" << endl;
  //           isBad = true;
  //         }
  //       });
  //   }
  //   if ( isBad )
  //     cout << " TOTAL UPDATE = " << endl; print_tm(cout, totalUpdate); cout << endl;
  // }

} // ReSmoothRowNonExpansiveV2



template<class ENERGY, class TMESH, class TPMAT, class TSMAT, class TM>
INLINE
void
ReSmoothRowNonExpansive (int            const &fvnr,
                         int            const &CV,
                         double         const &omega,
                         TMESH          const &FM,
                         TMESH          const &CM,
                         TPMAT          const &CSP,   // prol-mat
                         TSMAT                *fmat,  // sys-mat
                         FlatArray<int>        fineNeibs,
                         FlatArray<int>        rowCols,
                         FlatVector<TM>        rowVals,
                         Array<int>           &fineNeighborhood,
                         Array<int>           &otherCols,
                         LocalHeap            &lh,
                         bool           const &considerPrint = false)
{
  ReSmoothRowNonExpansiveV2<ENERGY>(fvnr, CV, omega, FM, CM, CSP, fmat, fineNeibs, rowCols, rowVals, fineNeighborhood, otherCols, lh,
                                    considerPrint);
} // ReSmoothRowNonExpansive

template<class ENERGY, class TMESH>
INLINE
void
calcAuxHarmonicExtension(TMESH const & M,
                         FlatMatrix<double> E,
                         FlatArray<int> all,
                         FlatArray<int> orig,
                         FlatArray<int> dest,
                         LocalHeap &lh,
                         double const &pInvTol = 1e-6)
{
  static Timer t("calcAuxHarmonicExtension");
  RegionTimer rt(t);
  calcAuxHarmonicExtensionV1<ENERGY>(M, E, all, orig, dest, lh, pInvTol);
} // calcAuxHarmonicExtension



// template<class T>
// struct ComparableByProxy
// {
//   INLINE bool operator == (T const &other) const { return static_cast<T const &>(*this).comparisonProxy() == other.T::comparisonProxy(); }
//   INLINE bool operator <  (T const &other) const { return static_cast<T const &>(*this).comparisonProxy() <  other.T::comparisonProxy(); }
//   INLINE bool operator <= (T const &other) const { return static_cast<T const &>(*this).comparisonProxy() <= other.T::comparisonProxy(); }
//   INLINE bool operator >  (T const &other) const { return static_cast<T const &>(*this).comparisonProxy() >  other.T::comparisonProxy(); }
//   INLINE bool operator >= (T const &other) const { return static_cast<T const &>(*this).comparisonProxy() >= other.T::comparisonProxy(); }
// };


// struct WeightedProp : public ComparableByProxy<WeightedProp>
// {
//   int   _prop;
//   float _weight;

//   WeightedProp() = default;
//   WeightedProp(WeightedProp const &other) = default;
//   WeightedProp(int p, float weight = 0) : _prop(p), _weight(weight) { }

//   INLINE int const &comparisonProxy() const { return _prop; }
// }; // struct WeightedProp


// INLINE bool
// insertAddProp(WeightedProp const &newProp, Array<WeightedProp> &props)
// {
//   int pos = merge_pos_in_sorted_array(newProp, props);
//   if ( (pos > 0) && (props[pos-1]._prop == newProp._prop) )
//     {  props[pos-1]._weight += newProp._weight;   return false; }
//   else
//     {  props.Insert(pos, newProp); return true; }
// } // insertAddProp




static INLINE
uint64_t
encodeVertex(BlockTM const &CM,
             int     const &col)
{
  auto const &eqc_h = *CM.GetEQCHierarchy();

  auto [eqV, lnrSB] = CM.template MapENodeToEQLNR<NT_VERTEX>(col);
  uint64_t eqVID = eqc_h.GetEQCID(eqV);
  uint64_t lnr = lnrSB;

  return eqVID | (lnr << 32);
} // encodeCol

static INLINE
int
decodeVertex(BlockTM  const &CM,
             uint64_t const &code)
{
  auto const &eqc_h = *CM.GetEQCHierarchy();

  uint32_t const *p    = reinterpret_cast<uint32_t const*>(&code);
  uint32_t const &eqID = p[0];
  uint32_t const &lnr  = p[1];

  auto eqc = eqc_h.GetEQCOfID(eqID);

  return ( eqc == -1 ) ? -1 : CM.template MapENodeFromEQC<NT_VERTEX>(lnr, eqc);
} // decodeVertex



/**
  * Class implementing the MPI-exchanges and data merging for the
  * communication in group-wise prolongation.
  *
  * One iteration update of the group-wise prolongation is
  *
  *     P -> A_gg^{-1} ( (AP)_{GC} | (AP)_{GN} )  (I \\ E_NC)
  *           that is
  *     P -> A_gg^{-1} ( (AP)_{gC} + (AP)_{gN} E_NC
  *
  * Where g indicates the group, C the group-cols, and N the other
  * cols present in the g-rows of the matrix AP.
  *
  * The initial "P" is the pw-prol, there cna be multiple iterations
  * of prolongation updates.
  *
  * E_NC is an extension from the group-prols to the "neib"-cols,
  * currently we use
  *   E_NC = -cAhat_NN^{-1} cAhat_NC
  * That is, E_NC does NOT CHANGE BETWEEN ITERATIONS.
  *
  * Also constant between iterations is A_gg.
  *
  * The entries of AP do change between iterations!
  *
  * The communication pattern in every iteration is:
  * I.   gather-on-master:
  *        i) initial prol-update:
  *               AP-row:     cols, vals
  *               coarseAhat: rows, cols, vals
  *               A_gg:       rows, vals
  *       ii) subsequent prol-updates:
  *               AP-row:     vals
  * II.  merge received data on master
  * III. scatter prol-updates
  *
  * Data is buffered into double-buffers and exchanged in a single
  * message.
  *
  * After I.i), row- and col-mapping for updating the AP-vals is
  * kept and re-used in later iterations.
  */
template<class TMESH, class ENERGY>
class GroupedSPExchange
{
  using TM = typename ENERGY::TM;

  static constexpr int BS = Height<TM>();
  static constexpr int SIZE_PER_EDGE = 2 +
                                       2 * SIZE_IN_BUFFER<typename ENERGY::TVD>() +
                                           SIZE_IN_BUFFER<typename ENERGY::TED>();

public:
  GroupedSPExchange(TMESH             const &FM,
                    TMESH             const &CM,
                    SparseMat<BS, BS> const &A,
                    FlatTable<int>           groups,
                    FlatTable<int>           groupsPerEQC,
                    int               const &numIterations,
                    double            const &cInvTolR,
                    double            const &cInvTolZ)
  : _fMesh(FM)
  , _cMesh(CM)
  , _eqc_h(*FM.GetEQCHierarchy())
  , _A(A)
  , _cInvTolR(cInvTolR)
  , _cInvTolZ(cInvTolZ)
  , _groups(groups)
  , _groupsPerEQC(groupsPerEQC)
  {
    auto const neqcs = EQCH().GetNEQCS();

    _exBuffers.SetSize(neqcs);
  }

  void
  InitializeIteration(int               const &round,
                      SparseMat<BS, BS> const &CSP,
                      SparseMat<BS, BS> const &AP,
                      LocalHeap               &lh)
  {
    if ( round == 0 )
    {
      Initialize(CSP, AP, lh);
    }
    else if ( round == 1 )
    {
      CompressBuffers(CSP, AP, lh);
    }
  }

  void
  Initialize(SparseMat<BS, BS> const &CSP,
             SparseMat<BS, BS> const &AP,
             LocalHeap               &lh)
  {
    /**
     *  First-time setup of everything
     */
    auto const neqcs = EQCH().GetNEQCS();

    HeapReset hr(lh);

    _exBufferCntA.SetSize(neqcs);
    _exBufferCntB.SetSize(neqcs);

    _exBufferCntA = 1;
    _exBufferCntB = 0;

    IterateExGroups(CSP, [&](auto eqc) { return !EQCH().IsMasterOfEQC(eqc); },
                    [&](auto eqc, auto group, auto cols)
    {
      HeapReset hr(lh);

      auto [cntA, cntB] = Buffering::CountSizes(A(), AP, CM(), group, cols, lh);

      // cout << " CNTs for eq " << eqc << " grp "; prow(group); cout << " = " << cntA << " " << cntB << endl;

      _exBufferCntA[eqc] += cntA;
      _exBufferCntB[eqc] += cntB;
    });

    // cout << " Initialize, _exBufferCntA = "; prow2(_exBufferCntA); cout << endl;
    // cout << " Initialize, _exBufferCntB = "; prow2(_exBufferCntB); cout << endl;

    for (auto k : Range(neqcs))
    {
      _exBuffers[k].SetSize(_exBufferCntA[k] + _exBufferCntB[k]);
    }

    Array<int> exPerEQC(neqcs);
    exPerEQC = 0;

    if ( neqcs > 1 )
    {
      FM().template ApplyEQ<NT_VERTEX>( Range(1ul, neqcs), [&](auto eqc, auto V)
      {
        exPerEQC[eqc] += CSP.GetRowIndices(V).Size();
      }, false); // everyone
    }

    _valBuffers = Table<TM>(exPerEQC);
  }

  void
  CompressBuffers(SparseMat<BS, BS> const &CSP,
                  SparseMat<BS, BS> const &AP,
                  LocalHeap &lh)
  {
    // cout << " OLD rec-inds: " << endl << _recvIndices << endl;
    // cout << " OLD _exBufferCntA: " << endl << _exBufferCntA << endl;
    // cout << " OLD _exBufferCntB: " << endl << _exBufferCntB << endl;
    // cout << " OLD buffer sizes: " << endl;
    // if ( EQCH().GetNEQCS() > 1 )
    // {
    //   for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
    //   {
    //     cout << eqc << ": " << _exBuffers[eqc].Size() << endl;
    //   }
    // }

    /**
     * Modify offsets/buffers to fit gathering of only "B"-data on subsequent
     */
    if ( EQCH().GetNEQCS() > 1 )
    {
      for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
      {
        auto &exBuffer = _exBuffers[eqc];

        if (EQCH().IsMasterOfEQC(eqc))
        {
          HeapReset hr(lh);

          auto dps = EQCH().GetDistantProcs(eqc);

          auto recInds = _recvIndices[eqc];

          FlatArray<int> newOff(recInds.Size(), lh);
          newOff[0] = 0;
          newOff[1] = 1;
          _exBuffers[eqc][0] = 0; // dummy offB for dummy chunk local entry

          /**
          * Contents of the buffer before were:
          *   [ loc:[offB], neib0:[offB, dataA, dataB], neib1:[offB, dataA, dataB]]
          * The loc-offB is just 0, otherwise offB = 1 + len(dataA)
          * Contents of the new buffer are:
          *   [ loc:[offB], neib0:[offB, dataB], neib1:[offB, dataB]]
          */
          for (auto j : Range(dps))
          {
            // one extra index in front for local part
            int first = recInds[j + 1];
            int next  = recInds[j + 2];

            int const offB = exBuffer[first];

            // cout << " eqc " << eqc << ", neib " << j << ", old offB = " << offB << endl;

            int totSize = next - first; // 1 + cntA + cntB

            // cout << " old totSize = " << totSize << endl;

            // keep the one zero as offB in there
            int cntA  = 1;
            int cntB  = totSize - offB;
            // cout << " old cntB " << cntB << endl;

            newOff[j + 2] = newOff[j + 1] + cntA + cntB;
          }

          recInds = newOff;

          // shorten buffer
          exBuffer.SetSize(newOff.Last());
        }
        else
        {
          // shorten buffer !
          int totSize = exBuffer.Size();
          int offB  = exBuffer[0]; // msg is [cntA, dataA, dataB]
          int cntB  = totSize - offB;
          // cout << " send-eqc " << eqc << " totSize " << totSize << " old offB = " << offB << endl;
          // cout << "   -> cntB = " << cntB << endl;
          exBuffer.SetSize(1 + cntB);
        }
      }
    }

    // cout << " NEW rec-inds: " << endl << _recvIndices << endl;
    // cout << " NEW buffer sizes: " << endl;
    // if ( EQCH().GetNEQCS() > 1 )
    // {
    //   for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
    //   {
    //     cout << eqc << ": " << _exBuffers[eqc].Size() << endl;
    //   }
    // }
    // cout << " _exBufferCntB: " << endl << _exBufferCntB << endl;

    // pure testing
    // _exBufferCntB = 1;
    // IterateExGroups(CSP, [&](auto eqc) { return !EQCH().IsMasterOfEQC(eqc); },
    //                 [&](auto eqc, auto group, auto cols)
    // {
    //   HeapReset hr(lh);
    //   auto cntB = Buffering::CountUpdateSize(AP, group, cols, lh);
    //   _exBufferCntB[eqc] += cntB;
    // });

    // cout << " _exBufferCntB RE-COUNTED: " << endl << _exBufferCntB << endl;
  }

  void
  StartGather(int const &iteration,
              SparseMat<BS, BS> const &CSP,
              SparseMat<BS, BS> const &AP,
              LocalHeap               &lh)
  {
    if ( iteration == 0 )
    {
      // sending everything rows,cols,Agg,AP-vals,c-edges
      StartInitialGather(CSP, AP, lh);
    }
    else
    {
      // sending only new AP-vals
      StartUpdateGather(CSP, AP, lh);
    }
  }


  void
  StartInitialGather(SparseMat<BS, BS> const &CSP,
                     SparseMat<BS, BS> const &AP,
                     LocalHeap               &lh)
  {
    /**
     *  Gathering 2 types of data, 2 different blocks A/B, within each buffer
     *      for every eqc: [ offB, offC, blockA: [grp0, grp1, ..], blockB: [grp0, grp1, ..] ]
     *  Meaning of types:
     *      A) gathered ONCE during first prol-smoothing iteration
     *                rows(=grp-mems), cols, edges, A_gg
     *      B) re-gathered every prol-smoothing iteration
     *                AP-vals
     *   Within A, the rows and cols for every dist-proc are needed again in later
     *   iterations to merge the updated AP-vals. In prol-smoothing iteration 1, they are
     *   copied out of the buffer.
     *   The buffer is then re-sized for pure B-exchange
     */

    auto const &eqc_h = EQCH();

    Array<int> offB(eqc_h.GetNEQCS());

    for (auto eqc : Range(eqc_h.GetNEQCS()))
    {
      // first entry in message is the offset for the AP-data (i.e. 1 + #A-data)
      //  i.e.: [B-off, A-data, B-data]
      auto const bOffset = _exBufferCntA[eqc];

      _exBuffers[eqc][0] = bOffset;

      // reset A/B-off to point to beginning of A/B buffer
      _exBufferCntA[eqc] = 1;
      _exBufferCntB[eqc] = bOffset;

      offB[eqc] = bOffset;
    }


    IterateExGroups(CSP, [&](auto eqc) { return !eqc_h.IsMasterOfEQC(eqc); },
                    [&](auto eqc, auto group, auto cols)
    {
      double *bufAPtr = _exBuffers[eqc].Data() + _exBufferCntA[eqc];
      double *bufBPtr = _exBuffers[eqc].Data() + _exBufferCntB[eqc];

      // cout << " PACK grp "; prow(group); cout << ", curr OFF " << _exBufferCntA[eqc] << ", " << _exBufferCntA[eqc] << endl;
      auto [cntA, cntB] = Buffering::PackIntoBufferInit(A(), AP, FM(), CM(), group, cols, lh, bufAPtr, bufBPtr);
      // cout << " PACK DONE grp "; prow(group); cout << " -> inc cnts " << cntA << " " << cntB << endl;

      // CheckRange("buf A", _exBufferCntA[eqc], cntA, IntRange(1, offB[eqc]));
      // CheckRange("buf B", _exBufferCntB[eqc], cntB, IntRange(offB[eqc], _exBuffers[eqc].Size()));

      _exBufferCntA[eqc] += cntA;
      _exBufferCntB[eqc] += cntB;
    });

    // cout << endl << endl;
    // cout << " EX-BUFFERS going INTO GatherEQCData: " << endl;
    // for (auto eqc : Range(EQCH().GetNEQCS()))
    // {
    //   auto dps = EQCH().GetDistantProcs(eqc);
    //   cout << " eqc " << eqc << ", eq-ID " << EQCH().GetEQCID(eqc) << ", dps "; prow(dps); cout << "   ";
    //   cout << "   ex-BS " << _exBuffers[eqc].Size() << endl;
    // }
    // cout << endl << endl;

    // cout << " enter GatherEQCData " << endl;
    auto [reqs, inds] = GatherEQCData(_exBuffers, EQCH());
    // cout << " OUT OF GatherEQCData " << endl;

    // cout << endl << endl;
    // cout << " EX-BUFFERS COMING OUT OF GatherEQCData: " << endl;
    // for (auto eqc : Range(EQCH().GetNEQCS()))
    // {
    //   auto dps = EQCH().GetDistantProcs(eqc);

    //   cout << " eqc " << eqc << ", eq-ID " << EQCH().GetEQCID(eqc) << ", dps "; prow(dps); cout << "   ";
    //   cout << "   ex-BS " << _exBuffers[eqc].Size() << endl;
    // }
    // cout << endl << endl;

    _currReqs    = std::move(reqs);
    _recvIndices = std::move(inds);
  }

  void
  StartUpdateGather(SparseMat<BS, BS> const &CSP,
                    SparseMat<BS, BS> const &AP,
                    LocalHeap               &lh)
  {
    auto const &eqc_h = EQCH();

    for (auto eqc : Range(eqc_h.GetNEQCS()))
    {
        // b-offset is 1, b-data starts right after the one entry for the offset itself
      int const offB = 1;
      _exBuffers[eqc][0] = offB; // offB
      _exBufferCntB[eqc] = offB;
    }

    IterateExGroups(CSP, [&](auto eqc) { return !eqc_h.IsMasterOfEQC(eqc); },
                    [&](auto eqc, auto group, auto cols)
    {
      double *bufBPtr = _exBuffers[eqc].Data() + _exBufferCntB[eqc];

      auto cntB = Buffering::PackIntoBufferUpdate(AP, group, cols, lh, bufBPtr);

      _exBufferCntB[eqc] += cntB;
    });

    // cout << " _exBufferCntB AFTER FILL! " << endl << _exBufferCntB << endl;

    // _recvIndices are modified to fit gathering of only B in CompressBuffers
    auto reqs = GatherEQCData(_exBuffers, _recvIndices, EQCH());

    _currReqs    = std::move(reqs);
  }


  FlatMatrix<double>
  GetAGG(int groupNum)
  {
    auto const nG     = _groups[groupNum].Size();
    auto const nGScal = nG * BS;

    return FlatMatrix<double>(nGScal, nGScal, _A_gg_data.Data() + _A_gg_off[groupNum]);
  }

  template<class TLAMA, class TLAMB>
  INLINE void
  IterateReceivedDataGeneric(SparseMat<BS, BS> const &CSP,
                             LocalHeap &lh,
                             TLAMA lam,
                             TLAMB unpackLam)
  {
    // cout << "IterateReceivedDataGeneric!" << endl;
    if ( EQCH().GetNEQCS() > 1 )
    {
      for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
      {
        if (EQCH().IsMasterOfEQC(eqc))
        {
          auto distProcs = EQCH().GetDistantProcs(eqc);
          auto &eqBuffer  = _exBuffers[eqc];

          // count within received buffer
          HeapReset hr(lh);
          FlatArray<int> offBuffA(distProcs.Size(), lh);
          FlatArray<int> offBuffB(distProcs.Size(), lh);
          Array<int> bOff(distProcs.Size());

          for (auto l : Range(offBuffA))
          {
            // buffer contains loc:[offB, dataA, dataB], neib0:[offB, dataA, dataB], etc..
            //       -> skip the "loc" part
            // offB = 1 + len(dataA)!
            int const first = _recvIndices[eqc][1 + l];
            int offB = eqBuffer[first]; // first entry is OFFSET for b-data
            offBuffA[l] = first + 1;    // A-data starts at second entry!
            offBuffB[l] = first + offB; // B-data starts at B-offset

            bOff[l] = offB;
          }

          auto eqGroups = _groupsPerEQC[eqc];

          for (auto l : Range(eqGroups))
          {
            auto const groupNum = eqGroups[l];
            auto fullGroup = _groups[groupNum];

            FlatArray<int> groupCols = fullGroup.Size() > 0 ? CSP.GetRowIndices(fullGroup[0])
                                                            : FlatArray<int>(0, lh);

            if ( groupCols.Size() < 2 )
            {
              // cout << " SKIP group " << groupNum << " in eq " << eqc << ", cols: "; prow(groupCols); cout << endl;
              continue;
            }
            // cout << " USE group " << groupNum << " in eq " << eqc << ", cols: "; prow(groupCols); cout << endl;

            lam(eqc, groupNum, fullGroup, distProcs, groupCols, [&](auto l)
            {
              auto [counts, tup] = unpackLam(eqBuffer.Data() + offBuffA[l],
                                            eqBuffer.Data() + offBuffB[l]);

              // CheckRange("offA", offBuffA[l], get<0>(counts), IntRange(_recvIndices[eqc][1+l] + 1, _recvIndices[eqc][1+l] + eqBuffer[_recvIndices[eqc][1+l]]));
              // CheckRange("offB", offBuffB[l], get<1>(counts), IntRange(_recvIndices[eqc][1+l] + eqBuffer[_recvIndices[eqc][1+l]], _recvIndices[eqc][2+l]));

              offBuffA[l] += get<0>(counts);
              offBuffB[l] += get<1>(counts);
              return tup;
            });
          }
        }
        // else
        // {
        //   cout << "SKIP eqc " << eqc << endl;
        // }
      }
    }
  };

  template<class TLAM>
  INLINE void
  IterateReceivedUpdateData(int const &iteration,
                            SparseMat<BS, BS> const &CSP,
                            LocalHeap &lh,
                            TLAM lam)
  {

    // cout << " IterateReceivedData! " << endl;
    // cout << " _recvIndices: " << endl << _recvIndices << endl;

    if ( iteration == 0 )
    {
      IterateReceivedDataGeneric(CSP, lh, lam, [&](double *bufA, double *bufB)
         -> std::tuple<std::tuple<int, int>, FlatMatrix<double>>
      {
        auto [ cnts, dGroupCodes, dAllCols, dA_gg, dAP, dEdgeTup ] =
          Buffering::UnpackFromBuffer(bufA, bufB);

        return std::make_tuple(cnts, dAP);
      });
    }
    else
    {
      IterateReceivedDataGeneric(CSP, lh, lam, [&](double *bufA, double *bufB)
         -> std::tuple<std::tuple<int, int>, FlatMatrix<double>>
      {
        return Buffering::UnpackFromBufferUpdate(bufB);
      });
    }
  }

  template<class TLAM>
  INLINE void
  IterateReceivedData(SparseMat<BS, BS> const &CSP,
                      LocalHeap &lh,
                      TLAM lam)
  {
    IterateReceivedDataGeneric(CSP, lh, lam, [&](double *bufA, double *bufB)
    {
      auto tup = Buffering::UnpackFromBuffer(bufA, bufB);
      auto cnts = get<0>(tup);

      return std::make_tuple(cnts, std::move(tup));
    });
  };

  void
  MergeReceivedData(SparseMat<BS, BS> const &AP,
                    SparseMat<BS, BS> const &CSP,
                    LocalHeap &lh)
  {
    auto const &eqc_h = EQCH();
    auto const  neqcs = eqc_h.GetNEQCS();

    int numExMGroups = 0;
    if ( neqcs > 1 )
    {
      for (auto eqc : Range(1ul, neqcs))
      {
        if (eqc_h.IsMasterOfEQC(eqc))
        {
          numExMGroups += _groupsPerEQC[eqc].Size();
        }
      }
    }

    // cout << " MergeReceivedData - Agg-alloc " << endl;
    {
      // allocate space for gg-diagonals
      size_t A_gg_tot = 0;

      // cout << " numExMGroups = "<< numExMGroups << endl;

      _A_gg_off.SetSize(1 + numExMGroups);

      _A_gg_off[0] = 0;

    if ( neqcs > 1 )
    {
        for (auto eqc : Range(1ul, neqcs))
        {
          if (eqc_h.IsMasterOfEQC(eqc))
          {
            for (auto groupNum : _groupsPerEQC[eqc])
            {
              auto const nGScal = _groups[groupNum].Size() * BS;
              auto groupCols = CSP.GetRowIndices(_groups[groupNum][0]);

              // create no A_gg for groups with a single col (e.g. roots)
              auto const numEntries = ( groupCols.Size() < 2 ) ? 0 : nGScal * nGScal;

              // cout << " eq " << eqc << ", G " << groupNum << ": "; prow(_groups[groupNum]);
              // cout << ", currOff = " << A_gg_tot << ", inc by " << numEntries << endl;
              _A_gg_off[groupNum + 1] = numEntries;
              A_gg_tot += numEntries;
            }
          }
        }
    }

      for (auto k : Range(numExMGroups))
      {
        _A_gg_off[k + 1] += _A_gg_off[k];
      }

      _A_gg_data.SetSize(A_gg_tot);
    }

    // cout << " MergeReceivedData - Agg-alloc OK" << endl;

    Array<int> aPCols;
    std::set<uint64_t> colCodes;

    // allocate space for row/col maps
    auto const avgColsPerRow = ( A().Height() > 0 ) ? A().AsVector().Size() / A().Height() : 0;

    // _apColCodes.Initialize(numExGroups, 2 * numExGroups * avgColsPerRow);

    /**
     *  Round 1 - for every group:
     *    merge AP-cols
     *    set E_NC offsets
    **/
    // TableCreator<T> with anything but T=int lacks the Add(row, array) method, it is not
    // templated in TableCreator-header. Therefore, use per-row array + direct table creator
    // TableCreator<uint64_t> ccc(numExMGroups);
    // TableCreator<double> cENC(numExMGroups);
    Array<int> perRowColCodes(numExMGroups);
    Array<int> perRowENC(numExMGroups);
    perRowColCodes = 0;
    perRowENC = 0;

    TableCreator<int> cdPosG(neqcs);
    TableCreator<int> cdPosC(neqcs);

    // cout << " MergeReceivedData ROW/COL merge " << endl;
    int tcRound = 0;
    for (; !cdPosG.Done(); tcRound++, cdPosG++, cdPosC++)
    {
      IterateReceivedData(CSP, lh, [&](auto eqc, auto groupNum, auto fullGroup,
                                       auto distProcs, auto groupCols, auto unpackBuf)
      {
        HeapReset hr(lh);

        int const numGroupCols = groupCols.Size();

        // local AP-cols
        MergeArrays(aPCols, lh, fullGroup.Size(), [&](auto i) { return AP.GetRowIndices(fullGroup[i]); });

        colCodes.clear();
        for (auto k : Range(aPCols))
        {
          colCodes.insert(encodeVertex(CM(), aPCols[k]));
        }

        for (auto j : Range(distProcs))
        {
          auto [ counts, dGroup, dAllCols, dA_gg, dAP, dEdgeTup] = unpackBuf(j);

          for (auto l : Range(dAllCols))
          {
            colCodes.insert(dAllCols[l]);
          }

          cdPosG.Add(eqc, FlatArray<int>(dGroup.Size(), lh));
          cdPosC.Add(eqc, FlatArray<int>(dAllCols.Size(), lh));
        }

        FlatArray<uint64_t> allCols(colCodes.size(), lh);

        int numAllCols = 0;

        for (auto it = colCodes.begin(); it != colCodes.end(); it++)
        {
          allCols[numAllCols++] = *it;
        }

        int numNCols = numAllCols - numGroupCols;

        // FlatMatrix<double> ENC(numNCols, groupCols.Size(), lh);

        // not templated correctly in TableCreator header
        // ccc.Add(groupNum, allCols);
        // cENC.Add(groupNum, ENC);
        if ( tcRound == 0 )
        {
          perRowColCodes[groupNum] = allCols.Size();
          perRowENC[groupNum]      = numNCols * BS * numGroupCols * BS;
        }
        else
        {
          _apColCodes[groupNum] = allCols;
        }
      });

      if ( tcRound == 0 )
      {
        _apColCodes = std::move(Table<uint64_t>(perRowColCodes));
        _E_NC_data  = std::move(Table<double>(perRowENC));
      }
    }
    // cout << " MergeReceivedData ROW/COL merge OK " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " MergeReceivedData ROW/COL merge OK ALL " << endl;

    // cout << " _apColCodes: " << endl << _apColCodes << endl;
    // cout << " perRowENC: " << endl << perRowENC << endl;

    // _apColCodes = ccc.MoveTable();
    _distGPos = cdPosG.MoveTable();
    _distCPos = cdPosC.MoveTable();
    // _E_NC_data = cENC.MoveTable();

    Array<int> offGPos(EQCH().GetNEQCS());
    Array<int> offCPos(EQCH().GetNEQCS());

    offGPos = 0;
    offCPos = 0;

    /**
     *  Round 2 - for every group:
     *    merge Agg contribs
     *    assemble coarseA-row, calc extension (-> E_NC saved to buffer)
     *    stash C/G offsets
     *  Note: Ap-vals are merged just-in-time during re-smoothing of the groups
    **/
    cout << " MergeReceivedData IterateReceivedData Round 2 " << endl;
    IterateReceivedData(CSP, lh, [&](auto eqc, auto groupNum, auto fullGroup,
                                     auto distProcs, auto groupCols, auto unpackBuf)
    {
      HeapReset hr(lh);

      // cout << " MergeReceivedData for group " << groupNum << " in eqc " << eqc << endl;

      auto allColCodes = _apColCodes[groupNum]; // sorted !

      int const numGroupCols = groupCols.Size();
      int const numAllCols   = allColCodes.Size();
      int const numNCols     = numAllCols - numGroupCols;

      // make codes of group-cols (NOT necessarily sorted!)
      auto groupColCodes = CreateFlatArray<uint64_t>(numGroupCols,
                                                     lh,
                                                     [&](auto j) { return encodeVertex(CM(), groupCols[j]); });

      FlatArray<uint64_t> neibColCodes(numNCols, lh); // sorted!
      // local neib-cols
      FlatArray<int>      nc(numAllCols, lh);   // not necessarily sorted
      // FlatArray<uint64_t> ncc(numAllCols, lh);  // sorted!
      FlatArray<uint64_t> nccc(numAllCols, lh);

      int cN    = 0;
      int cLocN = 0;
      int pos, locCol;
      for (auto j : Range(numAllCols))
      {
        auto colCode = allColCodes[j];

        if ( ( pos = groupColCodes.Pos(colCode) ) == -1 ) // not a group-col
        {
          neibColCodes[cN] = colCode;

          if ( ( locCol = decodeVertex(CM(), colCode) ) != -1 ) // we have the col locally
          {
            // cout << " allCol " << j << ", code = " << colCode << " is neib-col #" << cN << " -> ALSO is locCol " << locCol << endl;
            nc[cLocN] = locCol;
            // ncc[cLocN]  = colCode;
            nccc[cLocN] = cN;
            cLocN++;
          }
          cN++;
        }
      }
      auto locNeibCols     = nc.Part(0, cLocN);  // not sorted
      // auto locNeibColCodes = ncc.Part(0, cLocN); // sorted
      auto locNeibColNPos  = nccc.Part(0, cLocN); // position of code(col) in N-cols-codes

      // cout << " fullGroup: "; prow(fullGroup); cout << endl;
      // cout << " groupCols: "; prow(groupCols); cout << endl;

      // cout << " locNeibCols: "; prow(locNeibCols); cout << endl;
      // cout << " locNeibColNPos: "; prow(locNeibColNPos); cout << endl;

      // cout << "allColCodes: "; prow2(allColCodes); cout << endl;
      // cout << "groupColCodes: "; prow2(groupColCodes); cout << endl;
      // cout << "neibColCodes: "; prow2(neibColCodes); cout << endl;

      // A_gg local block
      auto A_gg = GetAGG(groupNum);
      GetDiagonalBlock<BS>(A(), fullGroup, A_gg);

      int const numNColsScal = numNCols * BS;
      int const numGColsScal = numGroupCols * BS;

      FlatMatrix<TM>     eBlock(2, 2, lh);

      // N-set is ordered by CODES, C-set is ordered by loc-cols!
      FlatMatrix<double> cA_NN(numNColsScal, numNColsScal, lh); // ordered by codes
      FlatMatrix<double> cA_NC(numNColsScal, numGColsScal, lh); // ordered by locCols

      FlatMatrix<double> E_NC(numNColsScal, numGColsScal, _E_NC_data[groupNum].Data());

      cA_NN = 0.0;
      cA_NC = 0.0;

      // local coarseAHat contributions
      IterateFullAhatRows(CM(), locNeibCols,
        [&](auto const k, auto const colK, auto const j, auto const vJ, auto const eNr,
            auto const &dataVK, auto const &dataVJ, auto const &eData)
        {
          ENERGY::CalcRMBlock(eBlock, eData, dataVK, dataVJ);

          // position of code of k'th loc-col in N-codes
          auto const posK = locNeibColNPos[k];
          int const offk = posK * BS;

          if ( j == -1 ) // N-? edge
          {
            int const posJ = find_in_sorted_array(vJ, groupCols);

            if ( posJ != -1 ) // N-G edge
            {
              int const offj = posJ * BS;

              // cout << "    ADD NC " << offk << " " << offj << endl;

              addTM    (cA_NN, offk, offk, 1.0, eBlock(0, 0));
              setFromTM(cA_NC, offk, offj, 1.0, eBlock(0, 1));
            }
          }
          else // N-N edge
          {
            uint64_t const codeJ = encodeVertex(CM(), vJ);
            int const posJ = find_in_sorted_array(codeJ, neibColCodes);
            int const offj = posJ * BS;

            // cout << "     ADD NN " << offk << " " << offj << endl;

            addTM    (cA_NN, offk, offk, 1.0, eBlock(0, 0));
            setFromTM(cA_NN, offk, offj, 1.0, eBlock(0, 1));
            setFromTM(cA_NN, offj, offk, 1.0, eBlock(1, 0));
            addTM    (cA_NN, offj, offj, 1.0, eBlock(1, 1));
          }
        }
      );

      // cout << " LOC cA_NN: " << endl << cA_NN << endl;
      // cout << " LOC cA_NC: " << endl << cA_NC << endl;

      for (auto j : Range(distProcs))
      {
        // cout << " neib " << j << " = " << distProcs[j] << endl;

        auto [ counts, dGroupCodes, dAllCols, dA_gg, dAP, dEdgeTup ] = unpackBuf(j);

        // decode group-codes, this MUST work since master has the full group
        auto dGroup = CreateFlatArray<int>(dGroupCodes.Size(), lh,
                                           [&](auto j) { return decodeVertex(FM(), dGroupCodes[j]); });

        // cout << " decoded grp: "; prow(dGroup); cout << endl;

        // dist-group positions
        FlatArray<int> offKs(dGroupCodes.Size(), _distGPos[eqc].Data() + offGPos[eqc]);

        offGPos[eqc] += dGroupCodes.Size();

        iterate_intersection(dGroup, fullGroup, [&](auto idxD, auto idxF)
        {
          // both dGroup AND fullGroup are sorted so this works!
          offKs[idxD] = BS * idxF;
        });

        // dist-col positions
        FlatArray<int> offLs(dAllCols.Size(), _distCPos[eqc].Data() + offCPos[eqc]);

        offCPos[eqc] += dAllCols.Size();

        for (auto l : Range(dAllCols)) // dAllCols are not sorted
        {
          auto const colCode = dAllCols[l];
          int pos = find_in_sorted_array(colCode, neibColCodes);

          // cout << " colCode " << l << " = " << colCode << " @ " << pos << " in neib-cols " << endl;

          if ( pos == -1 ) // not a neib -> must be a group-col -> must be decodeable
          {
            auto const col = decodeVertex(CM(), colCode);
            pos = find_in_sorted_array(col, groupCols);
            // cout << "    decoded into " << col << " @ " << pos << " in g-cols " << endl;

            offLs[l] = 1 + BS * pos; // positive entry -> gC
          }
          else
          {
            offLs[l] = -(1 + BS * pos); // negative entry -> gN
          }
        }

        // merge A_gg contribution
        for (auto k : Range(offKs))
        {
          for (auto j : Range(offKs))
          {
            A_gg.Rows(offKs[k], offKs[k] + BS).Cols(offKs[j], offKs[j] + BS) += dA_gg.Rows(k * BS, (k + 1) * BS).Cols(j * BS, (j + 1) * BS);
          }
        }

        // assemble coarseAHat contributions

        auto [ numEdges, edgePtr ] = dEdgeTup;

        int offEdge = 0;

        for (auto lEdge : Range(numEdges))
        {
          uint64_t *uiPtr = reinterpret_cast<uint64_t*>(edgePtr + offEdge);
          uint64_t const rowCode = uiPtr[0];
          uint64_t const colCode = uiPtr[1];

          offEdge += 2;

          typename ENERGY::TVD dataVK;
          typename ENERGY::TVD dataVJ;
          typename ENERGY::TED eData;

          offEdge += UnpackFromBuffer(dataVK, edgePtr + offEdge);
          offEdge += UnpackFromBuffer(dataVJ, edgePtr + offEdge);
          offEdge += UnpackFromBuffer(eData,  edgePtr + offEdge);

          ENERGY::CalcRMBlock(eBlock, eData, dataVK, dataVJ);

          // cout << "edge " << lEdge << "/" << numEdges << ", codes " << rowCode << " " << colCode << endl;

          auto posK = find_in_sorted_array(rowCode, neibColCodes);
          auto const offK = posK * BS;

          auto posJ = find_in_sorted_array(colCode, neibColCodes);

          if (posJ == -1) // NC
          {
            auto const col = decodeVertex(CM(), colCode);
            posJ = find_in_sorted_array(col, groupCols);

            auto const offJ = posJ * BS;

            // cout << "  decoded col: " << col << endl;
            // cout << "    add NC " << posK << "->" << offK << ", " << posJ << "->" << offJ << endl;

            addTM    (cA_NN, offK, offK, 1.0, eBlock(0,0));
            setFromTM(cA_NC, offK, offJ, 1.0, eBlock(0,1));
          }
          else // NN
          {
            auto const offJ = posJ * BS;

            // cout << "    add NN " << posK << "->" << offK << ", " << posJ << "->" << offJ << endl;

            addTM    (cA_NN, offK, offK, 1.0, eBlock(0,0));
            setFromTM(cA_NN, offK, offJ, 1.0, eBlock(0,1));
            setFromTM(cA_NN, offJ, offK, 1.0, eBlock(1,0));
            addTM    (cA_NN, offJ, offJ, 1.0, eBlock(1,1));
          }
        }

      }

      // cout << " FINAL cA_NN: " << endl << cA_NN << endl;
      // cout << " FINAL cA_NC: " << endl << cA_NC << endl;

      // compute and store E_NC
      if ( numNCols > 0 ) // almost always for ex-groups
      {
        // cout << " go into CalcPseudoInverseWithTol, lh avail " << lh.Available() << ", used = " << lh.UsedSize() << endl;
        CalcPseudoInverseWithTol(cA_NN, lh, _cInvTolR, _cInvTolZ);
        // cout << " inv cA_NN: " << endl << cA_NN << endl;

        E_NC = -cA_NN * cA_NC;
      }

      // cout << " E_NC: " << endl << E_NC << endl;
    });
    // cout << " MergeReceivedData IterateReceivedData OK " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " MergeReceivedData IterateReceivedData OK " << endl;

    // invert A_gg!
    // cout << " MergeReceivedData INV AGG! " << endl;
    if (EQCH().GetNEQCS() > 1)
    {
      for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
      {
        if (EQCH().IsMasterOfEQC(eqc))
        {
          auto eqGroups = _groupsPerEQC[eqc];

          for (auto j : Range(eqGroups))
          {
            auto const groupNum = eqGroups[j];

            auto groupCols = CSP.GetRowIndices(_groups[groupNum][0]);

            if ( groupCols.Size() < 2 )
            {
              continue;
            }

            HeapReset hr(lh);

            auto A_gg = GetAGG(groupNum);

            // cout << " invert A_gg for group " << groupNum << " in eqc " << eqc << endl;
            // cout << "    rows: "; prow(_groups[groupNum]); cout << endl;
            // cout << "    cols: "; prow(groupCols); cout << endl;
            // cout << A_gg << endl;
            CalcPseudoInverseWithTol(A_gg, lh, _cInvTolR, _cInvTolZ);
            // cout << " invert done!" << endl;
            // cout << " inv A_gg: " << endl << A_gg << endl;
          }
        }
      }
    }
    // cout << " MergeReceivedData AGG-INV OK! " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " MergeReceivedData AGG-INV OK! " << endl;
  }

  template<class TLAM>
  void
  ApplyToReceivedData(int               const &iteration,
                      SparseMat<BS, BS> const &AP,
                      SparseMat<BS, BS> const &CSP,
                      LocalHeap               &lh,
                      TLAM                     applyLam)
  {
    // cout << " ApplyToReceivedData " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " ALL ApplyToReceivedData " << endl;

    // wait for gather to complete
    MyMPI_WaitAll(_currReqs);

    // for (auto k : Range(_currReqs))
    // {
    //   cout << " wait " << k << "/" << _currReqs.Size() << endl;
    //   MPI_Wait(&_currReqs[k], MPI_STATUS_IGNORE);
    //   cout << " wait " << k << "/" << _currReqs.Size() << " OK!" << endl;
    // }

    if ( iteration == 0 )
    {
      // The first time, merge in rows, cols, A_gg, coarseA data, this stays constant
      // The AP-data is merged below fresh every time
      MergeReceivedData(AP, CSP, lh);
    }

    Array<int> offGPos(EQCH().GetNEQCS());
    Array<int> offCPos(EQCH().GetNEQCS());

    offGPos = 0;
    offCPos = 0;

    IterateReceivedUpdateData(iteration, CSP, lh,
      [&](auto eqc, auto groupNum, auto fullGroup,
          auto distProcs, auto groupCols, auto unpackBuf)
    {
      HeapReset hr(lh);

      // cout << " merge dist A_gC, A_gN for group " << groupNum << endl;
      // cout << "     group "; prow(fullGroup); cout << endl;
      // cout << "     groupCols "; prow(groupCols); cout << endl;

      auto allColCodes = _apColCodes[groupNum]; // sorted

      int const numGroupCols = groupCols.Size();
      int const numAllCols  = allColCodes.Size();
      int const numNCols    = numAllCols - numGroupCols;

      // make codes of group-cols (NOT necessarily sorted!)
      auto groupColCodes = CreateFlatArray<uint64_t>(numGroupCols,
                                                     lh,
                                                     [&](auto j) { return encodeVertex(CM(), groupCols[j]); });

      FlatArray<uint64_t> neibColCodes(numNCols, lh); // sorted!
      // local neib-cols
      // FlatArray<int>      nc(numAllCols, lh);   // not necessarily sorted
      // FlatArray<uint64_t> ncc(numAllCols, lh);  // sorted!
      // FlatArray<uint64_t> nccc(numAllCols, lh); // sorted!

      int cN    = 0;
      // int cLocN = 0;
      int pos, locCol;
      for (auto j : Range(numAllCols))
      {
        auto colCode = allColCodes[j];

        if ( ( pos = groupColCodes.Pos(colCode) ) == -1 ) // not a group-col
        {
          neibColCodes[cN] = colCode;

          // if ( ( locCol = decodeVertex(CM(), colCode) ) != -1 ) // we have the col locally
          // {
          //   nc[cLocN] = locCol;
          //   // ncc[cLocN]  = colCode;
          //   // nccc[cLocN] = cN;
          //   cLocN++;
          // }

          cN++;
        }
      }
      // auto locNeibCols     = nc.Part(0, cLocN);  // not sorted
      // auto locNeibColCodes = ncc.Part(0, cLocN); // sorted
      // auto locNeibColNPos  = nccc.Part(0, cLocN); // position in all-N-cols

      // cout << "     groupColCodes "; prow(groupColCodes); cout << endl;
      // cout << "     neibColCodes "; prow(neibColCodes); cout << endl;

      auto const numGScal     = fullGroup.Size() * BS;
      auto const numGColsScal = numGroupCols     * BS;
      auto const numNColsScal = numNCols         * BS;

      /**
       * merge AP-vals, stored as
       *   (AP_gC, AP_gN)
       * where:
       *    G is ordered by local  (F-)num
       *    C is ordered by local  (C-)num
       *    N is ordered by global (C-)code
       */
      FlatMatrix<double> AP_gC(numGScal, numGColsScal, lh);
      FlatMatrix<double> AP_gN(numGScal, numNColsScal, lh);

      AP_gC = 0.0;
      AP_gN = 0.0;

      // local A_gC, A_gN contributions
      for (auto k : Range(fullGroup))
      {
        int const offK = k * BS;

        auto const mem = fullGroup[k];

        auto ris = AP.GetRowIndices(mem);
        auto rvs = AP.GetRowValues(mem);

        for (auto j : Range(ris))
        {
          auto col = ris[j];

          if ( ( pos = find_in_sorted_array(col, groupCols) ) != -1 )
          {
            int const offJ = pos * BS;

            addTM(AP_gC, offK, offJ, 1.0, rvs[j]);
          }
          else
          {
            auto const colCode = encodeVertex(CM(), col);
            pos = find_in_sorted_array(colCode, neibColCodes);

            int const offJ = pos * BS;

            addTM(AP_gN, offK, offJ, 1.0, rvs[j]);
          }
        }
      }

      /** merge in dist A_gC, A_gN contributions **/
      // cout << " merge dist A_gC, A_gN for group " << groupNum << endl;

      for (auto j : Range(distProcs))
      {
        auto dAP = unpackBuf(j);

        int const numDistGroup = dAP.Height() / BS;
        int const numDistCols  = dAP.Width() / BS;

        FlatArray<int> offKs(numDistGroup, _distGPos[eqc].Data() + offGPos[eqc]);
        FlatArray<int> offLs(numDistCols,  _distCPos[eqc].Data() + offCPos[eqc]);

        offGPos[eqc] += numDistGroup;
        offCPos[eqc] += numDistCols;

        // cout << " dAP size " << dAP.Height() << " x " << dAP.Width() << endl;
        // cout << " AP_gC size " << AP_gC.Height() << " x " << AP_gC.Width() << endl;
        // cout << " AP_gN size " << AP_gN.Height() << " x " << AP_gN.Width() << endl;

        for (auto k : Range(offKs))
        {
          auto const offK = offKs[k];

          for (auto l : Range(offLs))
          {
            int const offL = abs(offLs[l]) - 1;

            if ( offLs[l] > 0 ) // positve offset -> gC
            {
              // cout << k << " " << l << " add gC " << offK << " " << offLs[l] << "->" << offL << endl;
              AP_gC.Rows(offK, offK + BS).Cols(offL, offL + BS) += dAP.Rows(k * BS, (k + 1) * BS).Cols(l * BS, (l + 1) * BS);
            }
            else // negative offset -> gN
            {
              // cout << k << " " << l << " add gN " << offK << " " << offLs[l] << "->" << offL << endl;
              AP_gN.Rows(offK, offK + BS).Cols(offL, offL + BS) += dAP.Rows(k * BS, (k + 1) * BS).Cols(l * BS, (l + 1) * BS);
            }
            // cout << "   NOW offLs: "; prow(offLs); cout << endl;
          }
        }
      }

      // cout << " merge dist A_gC, A_gN for group " << groupNum << " OK " << endl;

      auto A_gg = GetAGG(groupNum);

      FlatMatrix<double> E_NC(numNColsScal, numGColsScal, _E_NC_data[groupNum].Data());

      // call lambda on merged received data
      // cout << " applyLam for group " << groupNum << " OK " << endl;
      // cout << " A_gg: " << endl << A_gg << endl;
      // cout << " AP_gC: " << endl << AP_gC << endl;
      // cout << " AP_gN: " << endl << AP_gN << endl;
      // cout << " E_NC: " << endl << E_NC << endl;

      applyLam(groupNum, A_gg, AP_gC, AP_gN, E_NC);
      // cout << " applyLam for group " << groupNum << " OK " << endl;
    });
  }

  void
  StartValScatter(SparseMat<BS, BS> const &CSP)
  {
    // fill into exchange-buffers, start exchange
    if ( EQCH().GetNEQCS() > 1 )
    {
      FM().template ApplyEQ2<NT_VERTEX>(Range(1ul, EQCH().GetNEQCS()),
                                        [&](auto eqc, auto nodes)
      {
        auto ex_rv_row = _valBuffers[eqc];

        size_t off = 0;

        for (auto fVNr : nodes)
        {
          auto rvs = CSP.GetRowValues(fVNr);

          for (auto j : Range(rvs))
          {
            ex_rv_row[off++] = rvs[j];
          }
        }
      }, true); // master-only

      auto reqs = EQCH().ScatterEQCData(_valBuffers);

      _currReqs = std::move(reqs);
    }
  }

  void
  ApplyProlUpdate(SparseMat<BS, BS> &CSP)
  {
    auto const neqcs = EQCH().GetNEQCS();

    // cout << " APU wait! " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " APU wait! " << endl;
    // for (auto k : Range(_currReqs))
    // {
    //   cout << " wait " << k << "/" << _currReqs.Size() << endl;
    //   MPI_Wait(&_currReqs[k], MPI_STATUS_IGNORE);
    //   cout << " wait " << k << "/" << _currReqs.Size() << " OK!" << endl;
    // }
    // cout << " APU wait DONE! " << endl;
    // EQCH().GetCommunicator().Barrier();
    // cout << " APU wait DONE! " << endl;

    MyMPI_WaitAll(_currReqs);

    // fill from val-recv-buffers into SP-vals
    if ( neqcs > 1 )
    {
      FM().template ApplyEQ2<NT_VERTEX>(Range(1ul, neqcs), [&](auto eqc, auto nodes)
      {
        if ( !EQCH().IsMasterOfEQC(eqc) )
        {
          auto ex_rv_row = _valBuffers[eqc];

          size_t off = 0;

          for (auto fvnr : nodes)
          {
            auto rvs = CSP.GetRowValues(fvnr);

            for (auto j : Range(rvs))
              { rvs[j] = ex_rv_row[off++]; }
          }
        }
      }, false); // everyone
    }
  }

private:

  template<class TLAM_USE, class TLAM>
  INLINE void
  IterateExGroups (SparseMat<BS, BS> const &CSP,
                   TLAM_USE use_eqc,
                   TLAM lam)
  {
    if ( EQCH().GetNEQCS() > 1 )
    {
      for (auto eqc : Range(1ul, EQCH().GetNEQCS()))
      {
        // cout << "IterateExGroups, eqc " << eqc << ", use = " << use_eqc(eqc) << endl;
        if (use_eqc(eqc))
        {
          auto eqGroups = _groupsPerEQC[eqc];

          // cout << " IterateExGroups, eqc " << eqc << ", grp-nums: "; prow(eqGroups); cout << endl;

          for (auto j : Range(eqGroups))
          {
            auto groupNum = eqGroups[j];
            auto group = _groups[groupNum];

            FlatArray<int> cols = group.Size() > 0 ? CSP.GetRowIndices(group[0])
                                                  : FlatArray<int>(0, nullptr);

            if ( cols.Size() < 2 )
            {
              continue;
            }

            lam(eqc, group, cols);
          }
        }
      }
    }
  }

  class Buffering
  {
    // static constexpr int BS            = GroupedSPExchange<TMESH, ENERGY>::BS;
    // static constexpr int SIZE_PER_EDGE = GroupedSPExchange<TMESH, ENERGY>::SIZE_PER_EDGE;

    /**
    * Message is split into 2 blocks:
    *   Block A:
    *     RC-section:
    *       - #G, #allC
    *       - G-set, i.e. group-mems (coded)
    *       - allCol-set, i.e. group-cols (coded)
    *     MAT-A-section
    *       - A_GG
    *       - edges as: [row-code, col-code, e-data]
    *   Block B:
    *    - AP_vals, as dense block (G x allCols) (so it can be updated!)
    *
    * During later rounds, only block 2 is exchanged
    */

  public:
    INLINE static int
    CountSizeRC (FlatArray<int> group,
                 FlatArray<int> allCols,
                 LocalHeap      &lh)
    {
      int const nG    = group.Size();
      int const nAllC = allCols.Size();

      return 2 + nG + nAllC;
    }

    INLINE static int
    PackIntoBufferRC (BlockTM        const &FM,
                      BlockTM        const &CM,
                      FlatArray<int>        group,
                      FlatArray<int>        allCols,
                      LocalHeap            &lh,
                      uint64_t             *bufA)
    {
      int const nG    = group.Size();
      int const nAllC = allCols.Size();

      bufA[0] = nG;
      bufA[1] = nAllC;

      int off = 2;

      // group
      FlatArray<uint64_t> gCodes(nG, bufA + off);

      for (auto k : Range(gCodes))
      {
        gCodes[k] = encodeVertex(FM, group[k]);
      }
      off += nG;

      // cols
      FlatArray<uint64_t> allCCodes(nAllC, bufA + off);

      for (auto k : Range(allCCodes))
      {
        allCCodes[k] = encodeVertex(CM, allCols[k]);
      }
      off += nAllC;

      return 2 + nG + nAllC;
    }

    INLINE static int
    CountSizeAM (SparseMat<BS, BS> const &A,
                 SparseMat<BS, BS> const &AP,
                 TMESH             const &CM,
                 FlatArray<int>           group,
                 FlatArray<int>           gCols,
                 FlatArray<int>           allCols,
                 LocalHeap               &lh)
    {
      auto const &eqc_h = *CM.GetEQCHierarchy();

      // A_gg, #edges, edge-data
      int const nG     = group.Size();
      int const nGScal = nG * BS;

      // edge-count
      int nEdges = 0;

      auto neibCols = setMinus(allCols, gCols, lh);

      // edge-data
      IterateFullAhatRows(CM, neibCols,
        [&](auto kRow, auto vK, auto jRow, auto vJ, auto cENr,
            auto const &dataVI, auto const &dataVJ, auto const & dataEIJ)
      {
        // skip outside edges
        if ( jRow == -1 )
        {
          int pos = find_in_sorted_array(vJ, gCols);

          if (pos == -1)
            { return; }
        }

        // only write master-edges
        auto [eqI, locI] = CM.template MapENodeToEQLNR<NT_VERTEX>(vK);
        auto [eqJ, locJ] = CM.template MapENodeToEQLNR<NT_VERTEX>(vJ);

        if ( !eqc_h.IsMasterOfEQC(eqc_h.GetCommonEQC(eqI, eqJ)) )
          { return; }

        nEdges++;
      });

      return nGScal * nGScal + 1 + nEdges * SIZE_PER_EDGE;
    }

    INLINE static int
    PackIntoBufferAM (SparseMat<BS, BS> const &A,
                      SparseMat<BS, BS> const &AP,
                      TMESH             const &CM,
                      FlatArray<int>           group,
                      FlatArray<int>           gCols,
                      FlatArray<int>           allCols,
                      LocalHeap               &lh,
                      double                  *bufA)
    {
      auto const &eqc_h = *CM.GetEQCHierarchy();

      // A_gg, #edges, edge-data
      int const nG     = group.Size();
      int const nAllC  = allCols.Size();

      int const nGScal     = nG * BS;
      int const nAllCScal = nAllC * BS;

      int off = 0;

      // A_gg
      FlatMatrix<double> A_gg(nGScal, nGScal, bufA + off);
      GetDiagonalBlock<BS>(A, group, A_gg);
      off += nGScal * nGScal;

      // edge-count
      auto &nEdges = bufA[off++];
      nEdges = 0;
      // int nEdges = 0;


      auto neibCols = setMinus(allCols, gCols, lh);

      // cout << " PackIntoBufferAM " << endl;
      // cout << "   group: "; prow(group); cout << endl;
      // cout << "   allCols: "; prow(allCols); cout << endl;
      // cout << "   neibCols: "; prow(neibCols); cout << endl;

      // edge-data
      IterateFullAhatRows(CM, neibCols,
        [&](auto kRow, auto vK, auto jRow, auto vJ, auto cENr,
            auto const &dataVI, auto const &dataVJ, auto const & dataEIJ)
      {
        // cout << " check " << kRow << ": " << vK << " x " << jRow << ": " << vJ << endl;
        // skip outside edges
        if ( jRow == -1 )
        {
          int pos = find_in_sorted_array(vJ, gCols);

          if (pos == -1)
            { return; }
        }

        // only write master-edges
        auto [eqI, locI] = CM.template MapENodeToEQLNR<NT_VERTEX>(vK);
        auto [eqJ, locJ] = CM.template MapENodeToEQLNR<NT_VERTEX>(vJ);

        // cout << " eqI/J/commonL " << eqI << " " << eqJ << " " << eqc_h.GetCommonEQC(eqI, eqJ) << endl;

        if ( !eqc_h.IsMasterOfEQC(eqc_h.GetCommonEQC(eqI, eqJ)) )
          { return; }

        auto rowCode = encodeVertex(CM, vK);
        auto colCode = encodeVertex(CM, vJ);

        // cout << " include edge " << vK << " - " << vJ << ", codes " << rowCode << " " << colCode << endl;

        *reinterpret_cast<uint64_t*>(bufA + off) = rowCode;
        *reinterpret_cast<uint64_t*>(bufA + off + 1) = colCode;
        off += 2;

        off += PackIntoBuffer(dataVI,  bufA + off);
        off += PackIntoBuffer(dataVJ,  bufA + off);
        off += PackIntoBuffer(dataEIJ, bufA + off);

        nEdges += 1;
      });

      return off;
    }

    INLINE static int
    CountSizeB (SparseMat<BS, BS> const &AP,
                FlatArray<int>           group,
                FlatArray<int>           allCols,
                LocalHeap               &lh)
    {
      int const nG     = group.Size();
      int const nAllC  = allCols.Size();

      int const nGScal     = nG * BS;
      int const nAllCScal = nAllC * BS;

      return 2 + nGScal * nAllCScal;
    }

    INLINE static int
    PackIntoBufferB (SparseMat<BS, BS> const &AP,
                     FlatArray<int>           group,
                     FlatArray<int>           allCols,
                     LocalHeap               &lh,
                     double                  *buf)
    {
      int const nG     = group.Size();
      int const nAllC  = allCols.Size();

      int const nGScal     = nG * BS;
      int const nAllCScal = nAllC * BS;

      // AP-vals
      buf[0] = nGScal;
      buf[1] = nAllCScal;
      FlatMatrix<double> AP_g(nGScal, nAllCScal, buf + 2);

      AP_g = 0;
      for (auto k : Range(group))
      {
        int const offK = k * BS;

        auto const mem = group[k];

        auto rvs = AP.GetRowValues(mem);
        auto ris = AP.GetRowIndices(mem);

        for (auto j : Range(ris))
        {
          auto posJ = find_in_sorted_array(ris[j], allCols);
          int const offJ = posJ * BS;

          setFromTM(AP_g, offK, offJ, 1.0, rvs[j]);
        }
      }

      return 2 + nGScal * nAllCScal;
    }


    INLINE static std::tuple<int, int> // count block A/B
    CountSizes (SparseMat<BS, BS> const &A,
                SparseMat<BS, BS> const &AP,
                TMESH             const &CM,
                FlatArray<int>           group,
                FlatArray<int>           gCols,
                LocalHeap               &lh)
    {
      int const nG = group.Size();
      int const nC = gCols.Size();

      auto allCols = mergeFlatArrays(nG, lh, [&](auto k) { return AP.GetRowIndices(group[k]); });

      int const cntRC = CountSizeRC(group, allCols, lh);      // rows, cols
      int const cntAM = CountSizeAM(A, AP, CM, group, gCols, allCols, lh); // A_gg, coarseAhat
      int const cntB  = CountSizeB(AP, group, allCols, lh);   // AP

      // cout << "CountSizes, g = "; prow(group); cout << " gCols = "; prow(gCols); cout << endl;
      // cout << "   allCols, s=" << allCols.Size() << ": "; prow(allCols); cout << endl;
      // cout << "     -> cntB = " << cntB << endl;

      return std::make_tuple(cntRC + cntAM, cntB);
    }

    INLINE static int
    CountUpdateSize (SparseMat<BS, BS> const &AP,
                     FlatArray<int>           group,
                     FlatArray<int>           gCols,
                     LocalHeap               &lh)
    {
      int const nG = group.Size();
      int const nC = gCols.Size();

      auto allCols = mergeFlatArrays(nG, lh, [&](auto k) { return AP.GetRowIndices(group[k]); });

      int const cntB  = CountSizeB(AP, group, allCols, lh);

      // cout << "CountUpdateSize, g = "; prow(group); cout << " gCols = "; prow(gCols); cout << endl;
      // cout << "   allCols, s=" << allCols.Size() << ": "; prow(allCols); cout << endl;
      // cout << "     -> cntB = " << cntB << endl;

      return cntB;
    }

    INLINE static std::tuple<int, int>
    PackIntoBufferInit (SparseMat<BS, BS> const &A,
                        SparseMat<BS, BS> const &AP,
                        TMESH             const &FM,
                        TMESH             const &CM,
                        FlatArray<int>           group,
                        FlatArray<int>           gCols,
                        LocalHeap               &lh,
                        double                  *bufA,
                        double                  *bufB)
    {
      int const nG = group.Size();
      int const nC = gCols.Size();

      if ( nC == 1 )
      {
        return std::make_tuple(0, 0);
      }

      auto allCols = mergeFlatArrays(nG, lh, [&](auto k) { return AP.GetRowIndices(group[k]); });

      auto bufAAsUI = reinterpret_cast<uint64_t*>(bufA);

      // [ #group, #cols, rows, cols ]
      int const cntRC = PackIntoBufferRC(FM, CM, group, allCols, lh, bufAAsUI);                   // rows, cols

      // [ A_gg, #edges, edge-data ]
      int const cntAM = PackIntoBufferAM(A, AP, CM, group, gCols, allCols, lh, bufA + cntRC); // A_gg, coarseAhat

      // [ AP-vals ]
      int const cntB  = PackIntoBufferB(AP, group, allCols, lh, bufB);                    // AP

      return std::make_tuple(cntRC + cntAM, cntB);
    }

    INLINE static std::tuple<std::tuple<int, int>,     // cnt A/B
                             FlatArray<uint64_t>,      // rows/group-mems
                             FlatArray<uint64_t>,      // cols
                             FlatMatrix<double>,       // A_gg vals
                             FlatMatrix<double>,       // AP vals (dense)
                             std::tuple<int, double*>> // #edges, ptr to edge-data
    UnpackFromBuffer(double                  *bufA,
                     double                  *bufB)
    {
      // cout << " UnpackFromBuffer ! " << endl;

      uint64_t *bufAAsUI = reinterpret_cast<uint64_t*>(bufA);

      // [ #group, #cols ]
      int nG(bufAAsUI[0]);
      int nCols(bufAAsUI[1]);

      // cout << "    nG = " << nG << endl;
      // cout << "    nCols = " << nCols << endl;

      int cntA = 2;

      int const nGScal    = nG    * BS;
      int const nColsScal = nCols * BS;

      // [ group ]
      FlatArray<uint64_t> groupCodes(nG,  bufAAsUI + cntA);
      cntA += nG;

      // cout << " dist-groupCodes: "; prow(groupCodes); cout << endl;

      // [ cols ]
      FlatArray<uint64_t> colCodes(nCols, bufAAsUI + cntA);
      cntA += nCols;

      // cout << " dist-colCodes: "; prow(colCodes); cout << endl;

      // [ A_gg ]
      FlatMatrix<double> A_gg(nGScal, nGScal, bufA + cntA);
      cntA += nGScal * nGScal;

      // [ #edges ]
      // int nEdges(bufAAsUI[cntA++]);
      int nEdges(bufA[cntA++]);

      // cout << " nEdges = " << nEdges << endl;

      // [ edges ]
      double *edgePtr = bufA + cntA;
      cntA += nEdges * SIZE_PER_EDGE;

      FlatMatrix<double> AP_g(nGScal, nColsScal, bufB + 2);

      int cntB = 2 + nGScal * nColsScal;

      return std::make_tuple(std::make_tuple(cntA, cntB),
                             groupCodes,
                             colCodes,
                             A_gg,
                             AP_g,
                             std::make_tuple(nEdges, edgePtr));
    }


    INLINE static int
    PackIntoBufferUpdate (SparseMat<BS, BS> const &AP,
                          FlatArray<int>           group,
                          FlatArray<int>           gCols,
                          LocalHeap               &lh,
                          double                  *bufB)
    {
      int const nG = group.Size();
      int const nC = gCols.Size();

      if ( nC == 1 )
      {
        bufB[0] = 0;
        bufB[1] = 0;
        return 2;
      }

      auto allCols = mergeFlatArrays(nG, lh, [&](auto k) { return AP.GetRowIndices(group[k]); });

      int const cntB = PackIntoBufferB(AP, group, allCols, lh, bufB); // AP

      // cout << "PackIntoBufferUpdate, g = "; prow(group); cout << " gCols = "; prow(gCols); cout << endl;
      // cout << "   allCols, s=" << allCols.Size() << ": "; prow(allCols); cout << endl;
      // cout << "     -> cntB = " << cntB << endl;

      return cntB;
    }

    INLINE static
    std::tuple<std::tuple<int,int>,
               FlatMatrix<double>>       // AP vals (dense)
    UnpackFromBufferUpdate (double *bufB)
    {
      int const nGScal    = bufB[0];
      int const nColsScal = bufB[1];

      FlatMatrix<double> AP_g(nGScal, nColsScal, bufB + 2);

      int const cntB  = 2 + nGScal * nColsScal;

      return std::make_tuple(std::make_tuple(0, cntB), AP_g);
    }
  }; //class Buffering

private:
  INLINE TMESH             const &FM   () const { return _fMesh; }
  INLINE TMESH             const &CM   () const { return _cMesh; }
  INLINE SparseMat<BS, BS> const &A    () const { return _A; }
  INLINE EQCHierarchy      const &EQCH () const { return _eqc_h; }

  TMESH             const &_fMesh;
  TMESH             const &_cMesh;
  SparseMat<BS, BS> const &_A;
  EQCHierarchy      const &_eqc_h;

  double _cInvTolR;
  double _cInvTolZ;

  FlatTable<int> _groups;
  FlatTable<int> _groupsPerEQC;

  // number of total smoothing iterations
  // int _numIters;

  // for gather
  Array<int>             _exBufferCntA; // counting meta-data + A-data
  Array<int>             _exBufferCntB; // counting B-data
  Table<int>             _recvIndices;  // pointing to ranges from dist-procs within gathered bfufers
  Array<Array<double>>   _exBuffers;    // only done once (rows, cols), KEPT

  // positions for merging of AP-vals on later iteration
  Table<int>             _distGPos;
  Table<int>             _distCPos;

  // for val-scatter
  Table<TM> _valBuffers;

  // re-used
  Array<MPI_Request> _currReqs;

  //
  Array<int> _A_gg_off; // group-num -> A_gg space
  Array<double> _A_gg_data;

  // col-codes for merged AP-rows (the AP-vals are not saved)
  Table<uint64_t> _apColCodes;

  // coarseAhat-extension
  Table<double> _E_NC_data;

}; // class GroupedSpExchange



template<class ENERGY, class TMESH, int BS>
Array<int>
VertexAMGFactory<ENERGY, TMESH, BS>::
FindAggRoots (BaseCoarseMap const &cmap,
              TMESH         const &fmesh,
              TMESH         const &cmesh,
              LocalHeap           &lh) const
{
  static Timer t("FindAggRoots");
  RegionTimer rt(t);

  const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!

  auto       aggs  = cmap.template GetMapC2F<NT_VERTEX>();
  auto const nAggs = aggs.Size();

  auto const &fecon = *fmesh.GetEdgeCM();

  auto fedata = get<1>(fmesh.Data())->Data();

  auto vmap = cmap.template GetMap<NT_VERTEX>();

  Array<int> aggRoots(nAggs); // loc number in agg

  cmesh.template ApplyEQ2<NT_VERTEX>([&](auto agg_eq, auto aggNrs)
  {
    // Let the master of every eqc/coarse vertex decide on the center for each agg!
    if (!eqc_h.IsMasterOfEQC(agg_eq))
    {
      for (auto aggnr : aggNrs)
        { aggRoots[aggnr] = -123456789; }
      return;
    }

    for (auto aggnr : aggNrs)
    {
      HeapReset hr(lh);

      auto agg = aggs[aggnr];

      // bool const doPrint = (agg[0] == 3) && (agg[1] == 514);

      FlatArray<double> dist_from_bnd(agg.Size(), lh);
      dist_from_bnd = -1;

      unsigned cnt_set = 0;


      int cnt_round = 0;

      /**
       * isolated and bnd get distance 0
       * Note: This does not consider off-proc neighbors, this is actually
       *       what we want here as then vertices at the boundaries will have
       *       relatively large "distance".
       */
      for (auto k : Range(agg)) {
        auto vk = agg[k];
        auto neibs = fecon.GetRowIndices(vk);
        if (neibs.Size() == 0) // isolated
          { dist_from_bnd[vk] = 1; cnt_round++; }
        else {
          bool isbnd = false; int pos = -1;
          for (auto j : Range(neibs)) {
            if ( (pos = agg.Pos(neibs[j])) == -1 )  // BND
              { isbnd = true; break; }
          }
          if (isbnd)
            { dist_from_bnd[k] = 0; cnt_round++; }
        }
      }
      // if (doPrint)
      // {
      //   cout << " INIT " << cnt_set << " set, dist "; prow2(dist_from_bnd); cout << endl;
      // }

      cnt_set += cnt_round;

      auto flag_neibs = [&](auto k, int flag) {
        auto neibs = fecon.GetRowIndices(agg[k]);
        int cnt_flagged = 0, pos = -1;
        for (auto j : Range(neibs)) {
            // if (doPrint)
            // {
            //   cout << " mem " << k << " = " << agg[k] << ", neib " << j << " = " << neibs[j] << endl;
            // }
          if ( ( (pos = agg.Pos(neibs[j])) != -1 ) && // in-agg neib
               ( dist_from_bnd[pos] == -1        ) )  // not yet flagged
            { dist_from_bnd[pos] = flag; cnt_flagged++;
            // if (doPrint)
            // {
            //   cout << " mem " << k << " = " << agg[k] << ", neib " << j << " = " << neibs[j] << " -> @pos " << pos << ", flagged as " << flag << endl;
            // }
            }
        }
        return cnt_flagged;
      };

      // from those with maximal distance from the boundary, pick the one with the
      FlatArray<double> frac_in_agg(agg.Size(), lh);

      if (cnt_set == 0)
      {
        // AGG has no connections to the ouside, e.g. only 1 AGG containing all verts, nothing matters!
        dist_from_bnd = 0;
        frac_in_agg = 0;
      }
      else {
        int round = 0;
        // must terminate at some point - isolated already handled
        while (cnt_set < agg.Size()) {
          cnt_round = 0;
          for (auto k : Range(agg)) {
            auto vk = agg[k];
            if (dist_from_bnd[k] == round)
              { cnt_round += flag_neibs(k, round + 1); }
          }
          cnt_set += cnt_round;
      // if (doPrint)
      // {
      //   cout << " in round " << round << " set " << cnt_round << " -> total " << cnt_set << " set, dist "; prow2(dist_from_bnd); cout << endl;
      // }
                round++;
          if (round > 2 * agg.Size())
          {
            cout << " agg-nr " << aggnr << ", agg = "; prow2(agg); cout << endl;
            cout << " ROUND = " << round << " vs size " << agg.Size() << endl;
            cout << " dist_fom_bnd: "; prow2(dist_from_bnd); cout << endl;
            throw Exception("WHILE NOT TERMINATED IN FIND-NEIBS!");
            break;
          }
        }

        if (round == 0)
        {
          /**
           * no interior vertices - all have "distance" 0
           * so re-define distance by the fraction of weight in the agglomerate
           */
          for (auto k : Range(agg))
          {
            double sum   = 0.;
            double sumIn = 0.;
            auto neibs = fecon.GetRowIndices(agg[k]);
            auto eNrs  = fecon.GetRowValues(agg[k]);
            for (auto l : Range(neibs))
            {
              auto   const neib     = neibs[l];
              int    const eNr      = eNrs[l];
              double const approxWt = ENERGY::GetApproxWeight(fedata[eNr]);
              auto   const cVNeib   = vmap[neib];

              sum   += approxWt;
              if (cVNeib == aggnr)
                { sumIn += approxWt; }
            }
            // if sum OR sumIn are zero, must be single isolated
            dist_from_bnd[k] = (sum == 0.0) ? 1.0 : sumIn / sum;
          }
        }

        // if (doPrint)
        // {
        //   cout << " FIND ROOT FOR " << aggnr << ": "; prow2(agg); cout << endl;
        //   cout << " dist_from_bnd "; prow2(dist_from_bnd); cout << endl;
        // }
      }



      // Sort members by (proc_set, distance, frac_in_agg) and pick thae largest as center,
      // which is the one with maximal distance among those with maximal proc-set

      FlatArray<int> inds = makeSequence(agg.Size(), lh);

      if (eqc_h.GetDistantProcs(agg_eq).Size())
      {
        FlatArray<int> eqs(agg.Size(), lh);
        eqs[0] = fmesh.template GetEQCOfNode<NT_VERTEX>(agg[0]);
        int feq = eqs[0];
        for (auto k : Range(size_t(1), agg.Size())) {
          eqs[k] = fmesh.template GetEQCOfNode<NT_VERTEX>(agg[k]);
          if (eqc_h.IsLEQ(feq, eqs[k]))
            { feq = eqs[k]; }
        }
        QuickSort(inds, [&](auto i, auto j) {
          if ( eqs[i] != eqs[j] ) // eq_i < eq_jj
            { return eqc_h.IsLEQ(eqs[i], eqs[j]); }
          else
            { return dist_from_bnd[i] < dist_from_bnd[j]; }
        });
      }
      else
      {
        QuickSortI(dist_from_bnd, inds);
      }

      aggRoots[aggnr] = inds.Last();
    }
  }, false); // everyone!

  // "roots" on non-master EQCs cannot be valid because non-master ranks can have only parts if the agg
  // cmesh.template AllreduceNodalData<NT_VERTEX>(aggRoots, [&](auto &in) { return sum_table(in); });

  return aggRoots;
} // VertexAMGFactory::FindAggRoots

template<class T>
class SingleSweepTable
{
public:
  SingleSweepTable(int expectedRows, int expectedEntries)
  : numRows(0)
  , currOff(0)
  , off(expectedRows + 1)
  , data(expectedEntries)
  {
    off.SetSize(1);
    off[0] = currOff;
    data.SetSize0();
  }

  SingleSweepTable()
    : SingleSweepTable(0, 0)
  {}

  ~SingleSweepTable() = default;

  void Initialize(int expectedRows, int expectedEntries)
  {
    off.SetSize(expectedRows + 1);
    off.SetSize(1);
    off[0] = currOff;

    data.SetSize(expectedEntries);
    data.SetSize0();
  }

  int AddRow(FlatArray<T> rowData)
  {
    int rowId = numRows++;

    currOff += rowData.Size();
    off.Append(currOff);
    data.Append(rowData);

    return rowId;
  }

  FlatArray<T> GetRow(int row)
  {
    // return FlatArray<T>(off[row + 1] - off[row], data.Data() + off[row]);
    return data.Range(off[row], off[row + 1]);
  }

  FlatTable<T> GetFlatTable()
  {
    return FlatTable<T>(numRows, off.Data(), data.Data());
  }

private:
  int numRows;
  size_t currOff;
  Array<size_t> off;
  Array<T> data;
};


template<class ENERGY, class TSPM, class TMESH, class TLAM, class TLAM_ROOT>
INLINE std::tuple<std::tuple<int,int,int>,   // #groups ex-master,loc,ex-rec
                  Table<int>,                // group-nrs-per-ex-eqc
                  Array<int>,                // reps
                  SingleSweepTable<int>,     // groups
                  shared_ptr<TSPM>>          // (empty) sprol
CreateGroupedGraph (BaseCoarseMap const &cmap,
                    TMESH         const &FM,
                    TMESH         const &CM,
                    TLAM_ROOT            isRoot,
                    TLAM                 initGroup,
                    LocalHeap           &lh,
                    int           const &MAX_COLS_PER_GROUP,
                    bool          const &doPrint = false)
{
  auto const eqc_h = *FM.GetEQCHierarchy();

  auto const FNV = FM.template GetNN<NT_VERTEX>();
  auto const CNV = CM.template GetNN<NT_VERTEX>();

  auto const neqcs = eqc_h.GetNEQCS();

  auto vmap = cmap.template GetMap<NT_VERTEX>();

  /** preparation - sorting of eqcs, etc. **/

  Array<int> eqLoc;
  Array<int> eqExMaster;
  Array<int> eqExMinion;

  for (auto k : Range(neqcs))
  {
    if (eqc_h.GetDistantProcs(k).Size())
    {
      if (eqc_h.IsMasterOfEQC(k))
      {
        eqExMaster.Append(k);
      }
      else
      {
        eqExMinion.Append(k);
      }
    }
    else
    {
      eqLoc.Append(k);
    }
  }

  /**
    * We are going through the ex-eqcs in reverse hierarchic order, from more to less parallel.
    * Therefore, when we are handling some eqc, we KNOW that ALL hierachically larger vertices are already handled.
    * Since we are only allowing adding members to groups downwards in the hierarchy, this puts the least restrictions
    * from already taken vertices on the ex-groups.
    */
  QuickSort(eqExMaster, [&](auto eqi, auto eqj){ return eqc_h.GetDistantProcs(eqj).Size() < eqc_h.GetDistantProcs(eqi).Size(); });

  // cout << " eqExMaster: "; prow(eqExMaster); cout << endl;

  // Note: initGroup must only pick vertices we are master of!
  BitArray taken(FNV);

  auto resetTaken = [&]()
  {
    taken.Clear();
    for (auto k : Range(FNV))
    {
      if (vmap[k] == -1)
      {
        taken.SetBit(k);
      }
    }
    // mark all non-master vertices as taken so they dont get added to any groups (do we need this??)
    FM.template ApplyEQ2<NT_VERTEX>(eqExMinion, [&](auto eqc, auto nodes)
    {
      if (!eqc_h.IsMasterOfEQC(eqc))
      {
        for (auto v : nodes)
        {
          taken.SetBit(v);
        }
      }
    }, false);
  };

  auto itGroupWise = [&](FlatArray<int> eqcs, auto lam)
  {
    FM.template ApplyEQ2<NT_VERTEX>(eqcs, [&](auto eqc, auto nodes)
    {
      for (auto fvnr : nodes)
      {
        if (!taken.Test(fvnr))
        {
          // cout << " create group from FV " << fvnr << " in eq " << eqc << endl;
          auto [group, cols] = initGroup(eqc, fvnr, taken);
          // cout << " from " << fvnr << " created group "; prow(group); cout << " with cols "; prow(cols); cout << endl;
          int const nCols = cols.Size();
          lam(eqc, fvnr, group, cols);
          for (auto mem : group)
            { taken.SetBit(mem); }
        }
      }
    });
  };


  /** Actually start creating groups **/

  // Array<int> vertexToGroup(FNV);
  // vertexToGroup = -1;

  int nGroups = 0;

  int estGroups = max(FNV, FNV / 2); // max in case FNV=1

  /** create groups in a single go, so no table-creator! **/
  SingleSweepTable<int> ssGroups(estGroups, FNV);
  SingleSweepTable<int> ssCols(estGroups, estGroups * MAX_COLS_PER_GROUP);
  Array<int> groupReps(estGroups);

  Array<int> perowGraph(FNV);

  groupReps.SetSize0();
  perowGraph = 0;

  Array<int> groupCntPerExEQC(neqcs);
  Array<int> groupToExEQC(estGroups);
  groupCntPerExEQC = 0;
  groupToExEQC.SetSize0();

  auto addGroup = [&](int rep, FlatArray<int> group, FlatArray<int> cols) -> int
  {
    int groupId = nGroups++;

    ssGroups.AddRow(group);
    ssCols.AddRow(cols);

    for (auto mem : group)
    {
      perowGraph[mem] = cols.Size();
    }

    groupReps.Append(rep);

    return groupId;
  };


  resetTaken();

  /** go through master-ex vertices, assign to groups **/

  // #groups, then [#mems, mems, #cols, cols],...
  Array<int> bufferCnt(neqcs);
  bufferCnt = 1;

  // cout << " create EX-groups " << endl;
  itGroupWise(eqExMaster, [&](auto eqc, auto rep, auto group, auto cols)
  {
    auto groupId = addGroup(rep, group, cols);

    groupCntPerExEQC[eqc]++;
    groupToExEQC.Append(eqc);

    // sending rows + cols as [eq, lnr] tuples
    bufferCnt[eqc] += 2 + 2 * ( group.Size() + cols.Size() );
  });

  int numExMasterGroups = nGroups;

  // scatter message-sizes
  // cout << " ScatterEQCArray bufferCnt " << endl;
  // eqc_h.GetCommunicator().Barrier();
  // cout << " ScatterEQCArray bufferCnt " << endl;

  eqc_h.ScatterEQCArray(bufferCnt);

  // cout << " bufferCnt REDUCED: " << endl << bufferCnt << endl;

  /** pack ex-group data into buffers **/

  Array<Array<int>> exBuffers(neqcs);
  for (auto eqc : Range(neqcs))
  {
    exBuffers[eqc].SetSize(bufferCnt[eqc]);
    exBuffers[eqc][0] = 0;
    bufferCnt[eqc] = 1; // bufferCnt is used as offset, skipping first entry
  }

  // cout << " pack buffers " << endl;
  // eqc_h.GetCommunicator().Barrier();
  // cout << " pack buffers " << endl;
  for (auto k : Range(numExMasterGroups))
  {
    auto fvRep = groupReps[k];
    auto [eqc, lnr] = FM.template MapENodeToEQLNR<NT_VERTEX>(fvRep);
    auto dps = eqc_h.GetDistantProcs(eqc);

    exBuffers[eqc][0]++;

    auto &exBuffer = exBuffers[eqc];
    auto &cnt      = bufferCnt[eqc];

    // auto group = groupMems.Range(groupMemOff[k], groupMemOff[k+1]);
    auto group = ssGroups.GetRow(k);

    exBuffer[cnt++] = group.Size();

    for (auto j : Range(group))
    {
      auto [eqc, lnr] = FM.template MapENodeToEQLNR<NT_VERTEX>(group[j]);
      exBuffer[cnt++] = eqc_h.GetEQCID(eqc);
      exBuffer[cnt++] = lnr;
    }

    // auto cols  = groupCols.Range(groupColOff[k], groupColOff[k+1]);
    auto cols = ssCols.GetRow(k);

    exBuffer[cnt++] = cols.Size();

    for (auto j : Range(cols))
    {
      auto [eqc, lnr] = CM.template MapENodeToEQLNR<NT_VERTEX>(cols[j]);
      exBuffer[cnt++] = eqc_h.GetEQCID(eqc);
      exBuffer[cnt++] = lnr;
    }
  }

  // cout << " scatter buffers " << endl;
  // eqc_h.GetCommunicator().Barrier();
  // cout << " scatter buffers " << endl;
  // start ex-group scatter
  auto scatterReqs = eqc_h.ScatterEQCData(exBuffers);

  // create groups from (non-taken) local vertices
  // cout << " create LOC groups " << endl;
  itGroupWise(eqLoc, [&](auto eqc, auto rep, auto group, auto cols)
  {
    addGroup(rep, group, cols);
  });

  int numLocGroups = nGroups - numExMasterGroups;

  // finish ex-group scatter
  MyMPI_WaitAll(scatterReqs);

  // for (auto eqc : Range(1ul, eqc_h.GetNEQCS()))
  // {
    // cout << "EX-BUFFER for " << eqc << ", dps "; prow(eqc_h.GetDistantProcs(eqc)); cout << endl;
    // prow3(exBuffers[eqc], cout, "   ", 20);
    // cout << endl;
  // }

  /**
   *  Go through received groups, filter out the local part of group,
   *  add it to the group-"table", and cols to the col-"table"
   */
  for (auto eqc : eqExMinion)
  {
    auto &recBuf = exBuffers[eqc];

    // cout << " fill eqc " << eqc << ", dps = "; prow(eqc_h.GetDistantProcs(eqc)); cout << endl;

    // cout << " recBuf.Size() = " << recBuf.Size() << endl;

    int numGroups = recBuf[0];

    int cnt = 1;

    for (auto lGrp : Range(numGroups))
    {
      int groupSize = recBuf[cnt++];

      // cout << "group w. size " << groupSize << " @ " << cnt - 1 << "/" << recBuf.Size() << endl;

      // over-write parts of buffer!
      FlatArray<int> potentialMems(recBuf.Part(cnt, groupSize));
      int cntG = 0;

      for (auto j : Range(groupSize))
      {
        auto eqcID = recBuf[cnt++];
        auto locNr = recBuf[cnt++];

        // cout << "  map MEM " << eqcID << ", lnr " << locNr << endl;

        auto eqc = eqc_h.GetEQCOfID(eqcID);

        // cout << "   -> EQC " << eqc << endl;

        if (eqc != -1)
        {
          auto vNum = FM.template MapENodeFromEQC<NT_VERTEX>(locNr, eqc);

          // cout << "    -> vNum " << vNum << endl;
          potentialMems[cntG++] = vNum;
        }
        // else
        // {
        //   cout << "    -> SKIP " << endl;
        // }
      }

      auto mems = potentialMems.Range(0, cntG);

      // cout << " mems left: "; prow(mems) ; cout << endl;

      int numCols = recBuf[cnt++];
      // cout << "   #cols = " << numCols << endl;

      // over-write parts of buffer!
      FlatArray<int> potentialCols(recBuf.Part(cnt, numCols));
      int cntC = 0;

      for (auto j : Range(numCols))
      {
        auto eqcID = recBuf[cnt++];
        auto locNr = recBuf[cnt++];

        auto eqc = eqc_h.GetEQCOfID(eqcID);

        // cout << "  map COL " << eqcID << ", lnr " << locNr << " -> EQC " << eqc << endl;

        if (eqc != -1)
        {
          auto vNum = CM.template MapENodeFromEQC<NT_VERTEX>(locNr, eqc);

          // cout << "    -> vNum " << vNum << endl;

          potentialCols[cntC++] = vNum;
        }
      }

      auto cols = potentialCols.Range(0, cntC);

      addGroup(-1, mems, cols);

      groupCntPerExEQC[eqc]++;
      groupToExEQC.Append(eqc);
    }
  }

  int numExRecGroups = nGroups - numExMasterGroups - numLocGroups;

  // cout << " numExRecGroups = " << numExRecGroups << endl;
  // cout << " nGroups = " << nGroups << endl;
  // cout << " numExMasterGroups = " << numExMasterGroups << endl;
  // cout << " numLocGroups = " << numLocGroups << endl;

  Table<int> groupNumsPerExEQC(groupCntPerExEQC);

  groupCntPerExEQC = 0;

  for (auto k : Range(numExMasterGroups))
  {
    auto eqc = groupToExEQC[k];
    // cout << "M-EX grp " << k << " in eq " << eqc << "!" << endl;
    groupNumsPerExEQC[eqc][groupCntPerExEQC[eqc]++] = k;
  }

  for (auto k : Range(numExRecGroups))
  {
    int groupNum = numExMasterGroups + numLocGroups + k;
    auto eqc = groupToExEQC[numExMasterGroups + k];
    // cout << "Ex-REC grp " << k << ", num = " << groupNum << " in eq " << eqc << "!" << endl;
    groupNumsPerExEQC[eqc][groupCntPerExEQC[eqc]++] = groupNum;
  }

  // cout << " ALLOC PROL " << endl;
  // eqc_h.GetCommunicator().Barrier();
  // cout << " ALLOC PROL " << endl;

  // create prol
  auto sprol = make_shared<TSPM>(perowGraph, CM.template GetNN<NT_VERTEX>());

  auto groups = ssGroups.GetFlatTable();
  auto cols   = ssCols.GetFlatTable();

  for (auto k : Range(groups))
  {
    auto groupCols = cols[k];

    for (auto mem : groups[k])
    {
      // I don't think we need to sort cols here
      sprol->GetRowIndices(mem) = groupCols;
    }
  }


  // if (doPrint)
  {
    auto nRows = sprol->Height();
    auto nCols = sprol->AsVector().Size() / Height<typename ENERGY::TM>() / Height<typename ENERGY::TM>();
    auto avgCols = double(nCols) / double(nRows);

    cout << " NEW  graph has " << nCols << " cols for " << nRows << " rows, avg of " << avgCols << " per row " << endl;
    cout << groups.Size() << " groups for " << vmap.Size() << " vertices, avg size = " << double(vmap.Size()) / double(max(1ul, groups.Size())) << endl;

    size_t maxGS = 4;
    for (auto k : Range(groups))
    {
      maxGS = max(maxGS, groups[k].Size());
    }

    int cnt_drops = 0;
    int cnt_roots = 0;
    Array<int> numGroups(maxGS + 1);
    numGroups = 0;
    for (auto k : Range(groups))
    {
      if ( ( groups[k].Size() == 1 ) && ( isRoot(groups[k][0]) ) )
      {
        cnt_roots++;
      }
      else if ( (groups[k].Size() == 1) && (vmap[groups[k][0]] == -1) )
      {
        cnt_drops++;
      }
      else
      {
        numGroups[groups[k].Size()]++;
      }
    }

    std::cout << " # groups per size: " << std::endl;

    cout << "DROP: "  << cnt_drops << endl;
    cout << "ROOTS: " << cnt_roots << endl;
    for (auto k : Range(numGroups))
    {
      if (numGroups[k]>0)
      {
        cout << k << ": " << numGroups[k] << endl;
      }
    }
  }

  auto gCntT = std::make_tuple(numExMasterGroups,
                               numLocGroups,
                               numExRecGroups);

  return std::make_tuple(gCntT, std::move(groupNumsPerExEQC), std::move(groupReps), std::move(ssGroups), sprol);
} // CreateGroupedGraph


template<class ENERGY, class TMESH, class TLAM>
INLINE std::tuple<Array<int>, Table<int>>
GenerateSProlGraph (BaseCoarseMap const &cmap,
                    TMESH         const &FM,
                    TMESH         const &CM,
                    int           const &MAX_PER_ROW,
                    double        const &MIN_PROL_FRAC,
                    double        const &MIN_SUM_FRAC,
                    TLAM                 isRoot,
                    LocalHeap           &lh)
{
  static Timer t("GenerateSProlGraph");
  RegionTimer rt(t);

  // TODO: we should also consider the vertex-weights here I think -
  //       that could let us ignore connections that are not needed for AMG quality
  //       where the vertex weights are large.

  auto const eqc_h = *FM.GetEQCHierarchy();

  auto const FNV = FM.template GetNN<NT_VERTEX>();
  auto const CNV = CM.template GetNN<NT_VERTEX>();

  auto fvdata = get<0>(FM.Data())->Data();
  auto fedata = get<1>(FM.Data())->Data();
  auto cvdata = get<0>(CM.Data())->Data();

  const auto & fecon = *FM.GetEdgeCM();

  auto vmap = cmap.template GetMap<NT_VERTEX>();

  auto createGraph = [&](auto lam) -> std::tuple<Array<int>, Table<int>>
  {
    Array<int> perow(FNV);

    FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
      if (eqc_h.IsMasterOfEQC(eqc))
      {
        for (auto fvnr : nodes)
        {
          perow[fvnr] = lam(eqc, fvnr).Size();
        }
      }
      else
      {
        for (auto fvnr : nodes)
        {
          perow[fvnr] = 0;
        }
      }
    }, false); // everyone!

    FM.template ScatterNodalData<NT_VERTEX>(perow);

    Table<int> graph(perow);

    FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
      for (auto fvnr : nodes)
      {
        graph[fvnr] = lam(eqc, fvnr);
        QuickSort(graph[fvnr]);
      }
    }, true); // master only!

    return std::make_tuple(std::move(perow), std::move(graph));
  };

  Array<IVec<2,double>> trow;
  Array<int> tcv;
  Array<int> cols;

  auto tup = createGraph([&](auto const &eqc, auto const &fvnr) -> FlatArray<int>
  {
    auto CV = vmap[fvnr];

    // bool const doPrint = (fvnr == 47) && (CV == 58);

    cols.SetSize0();

    // vertex drops - no cols
    if ( is_invalid(CV) )
      { return cols; }

    auto fNeibs = fecon.GetRowIndices(fvnr);

    // all others prolongate from AT LEAST the own coarse vertex
    cols.SetSize(1);
    cols[0] = CV;

    /**
     * center of an agglomerate, or vertices with no neibhors in the
     * same agglomerate only prolongate ONLY from that single coarse vertex
     */
    if ( isRoot(fvnr) ||
         std::none_of(fNeibs.begin(), fNeibs.end(), [&](auto fv) -> bool LAMBDA_INLINE { return vmap[fv] == CV; }) )
    {
      return cols;
    }

    auto fENrs = fecon.GetRowValues(fvnr);

    size_t pos;
    double in_wt = 0;
    double max_wt = 0;
    double tot_wt = 0;

    trow.SetSize0();
    tcv.SetSize0();

    for (auto j : Range(fNeibs.Size()))
    {
      auto fNeib = fNeibs[j];
      auto cNeib = vmap[fNeib];
      if ( !is_invalid(cNeib) )
      {
        int eNum         = int(fENrs[j]);
        double approx_wt = ENERGY::GetApproxWeight(fedata[eNum]);

        max_wt  = max(max_wt, approx_wt);
        tot_wt += approx_wt;

        if ( cNeib == CV )
          { in_wt += approx_wt; }
        else {
          if (eqc_h.IsLEQ(eqc, CM.template GetEQCOfNode<NT_VERTEX>(cNeib))) // could prolongate from that coarse neib
          {
            if ( (pos = tcv.Pos(cNeib)) == size_t(-1) ) {
              trow.Append(IVec<2,double>(cNeib, approx_wt));
              tcv.Append(cNeib);
            }
            else
              { trow[pos][1] += approx_wt; }
          }
        }
      }
    }

    QuickSort(trow, [](const auto & a, const auto & b) LAMBDA_INLINE { return a[1] > b[1]; });

    /**
     * We keep adding coarse neighbors, starting with the strongest one, stopping
     * once the weights fall below the given threshold, otherwise until
     *   used-out-wt > MIN_SUM_FRAC * (total-out-wt)
     * or
     *   MIN_SUM_FRAC * in_wt + used-out-wt > MIN_PROL_FRAC * total-wt,
     * that is
     *   used-out-wt > MIN_PROL_FRAC * total-out-wt
     * or we reach the maximum number of columns.
     *
     * That is, we need to penalize the summed up weight of edges in the same agg which
     * are always used!
     */
    double const wthresh = MIN_PROL_FRAC * max_wt;
    double const sthresh = MIN_SUM_FRAC  * tot_wt;

    double cw_sum = MIN_SUM_FRAC * in_wt;

    size_t max_adds = min2(size_t(MAX_PER_ROW - 1),
                           trow.Size());

    for (auto j : Range(max_adds))
    {
      auto [CV, wt] = trow[j];

      if ( wt < wthresh )
        { break; }

      cw_sum += wt;
      cols.Append(CV);

      // if ( cw_sum > sthresh )
      //   { break; }
    }

    return cols;
  });

  return tup;
} // GenerateSProlGraph

template<class ENERGY, class TSPM, class TMESH, class TLAM>
INLINE std::tuple<std::tuple<int,int,int>,   // #groups ex-master,loc,ex-rec
                  Table<int>,                // group-nrs-per-ex-eqc
                  Array<int>,                // reps
                  SingleSweepTable<int>,     // groups
                  shared_ptr<TSPM>>          // (empty) sprol
CreateEmptyGroupWiseSProl (BaseCoarseMap const &cmap,
                           TMESH         const &FM,
                           TMESH         const &CM,
                           int           const &MAX_PER_ROW,
                           double        const &MIN_PROL_FRAC,
                           double        const &MIN_SUM_FRAC,
                           TLAM                 isRoot,
                           LocalHeap           &lh,
                           bool          const &doPrint = false)
{
  static Timer t("CreateEmptyGroupWiseSProl");
  static Timer tg("CreateEmptyGroupWiseSProl - groups");

  RegionTimer rt(t);

   // Note: Dirichlet are not in any group! (TODO: PARALLEL??)

  // TODO: we should also consider the vertex-weights here I think -
  //       that could let us ignore connections that are not needed for AMG quality
  //       where the vertex weights are large.

  auto const eqc_h = *FM.GetEQCHierarchy();

  auto const FNV = FM.template GetNN<NT_VERTEX>();
  auto const CNV = CM.template GetNN<NT_VERTEX>();

  auto fvdata = get<0>(FM.Data())->Data();
  auto fEData = get<1>(FM.Data())->Data();
  auto cvdata = get<0>(CM.Data())->Data();

  const auto & fecon = *FM.GetEdgeCM();

  auto vmap = cmap.template GetMap<NT_VERTEX>();

  // cout << " CreateEmptyGroupWiseSProl, FM = " << endl << FM << endl<< endl<< endl;
  // cout << " CreateEmptyGroupWiseSProl, CM = " << endl << CM << endl<< endl<< endl;

  // set up a first, tentative graph containing only really strong connections,
  // in a second step, we add more columns so we can create better groups, but the
  // ones we get here are forced
  double const MIN_PROL_FRAC_FORCED = MIN_PROL_FRAC; // max(MIN_PROL_FRAC, 0.2); // 0.2;
  double const MIN_SUM_FRAC_FORCED  = MIN_SUM_FRAC; // MIN_SUM_FRAC;  // 0.15;
  double const MAX_PER_ROW_FORCED   = MAX_PER_ROW; // max(2, MAX_PER_ROW - 1);
  // int const MAX_PER_ROW_EXT = MAX_PER_ROW + min(2, MAX_PER_ROW);
  int const MAX_PER_ROW_EXT = MAX_PER_ROW;
  int const MAX_GRP_SIZE    = 8; // TESTING


  cout << " MAX_PER_ROW_FORCED = " << MAX_PER_ROW_FORCED << endl;
  cout << " MIN_PROL_FRAC_FORCED = " << MIN_PROL_FRAC_FORCED << endl;
  cout << " MIN_SUM_FRAC_FORCED = " << MIN_SUM_FRAC_FORCED << endl;

  auto [ initPerRowSB, initGraphSB ]
    = GenerateSProlGraph<ENERGY>(cmap,
                                 FM,
                                 CM,
                                 MAX_PER_ROW_FORCED,
                                 MIN_PROL_FRAC_FORCED,
                                 MIN_SUM_FRAC_FORCED,
                                 isRoot,
                                 lh);

  eqc_h.GetCommunicator().Barrier();

  // lambda-capture of structured binding
  auto &initPerRow = initPerRowSB;
  auto &initGraph = initGraphSB;

  // std::cout << " initGraph: " << endl << initGraph << endl;

  // cout << " have initPerRow, initGraph" << endl;

  if (doPrint)
  {
    auto nRows = initGraph.Size();
    auto nCols = initGraph.AsArray().Size();
    auto avgCols = double(nCols) / double(nRows);

    cout << " init/forced graph has " << nCols << " cols for " << nRows << " rows, avg of " << avgCols << " per row " << endl;
  }


  Array<double> totalWt(FNV);
  totalWt = 0;
  FM.template ApplyEQ<NT_EDGE>([&](auto eqc, auto const &fEdge)
  {
    double const ewt = ENERGY::GetApproxWeight(fEData[fEdge.id]);

    totalWt[fEdge.v[0]] += ewt;
    totalWt[fEdge.v[1]] += ewt;
  }, false); // only vals for master-verts matter

  // group-sized
  constexpr int START_SIZE = 50;

  Array<int> group(START_SIZE);
  Array<double> addWtG(START_SIZE);
  Array<double> currWtG(START_SIZE);

  // prospective-mem sized
  Array<int>    prospMems(START_SIZE);
  Array<int>    numBlocking(START_SIZE);
  Array<double> addWtP(START_SIZE);
  Array<double> currWtP(START_SIZE);
  Array<double> prospWt(START_SIZE);

  // misc
  Array<int> newCols(START_SIZE);
  Array<int> newColEQCs(START_SIZE);
  Array<int> eligibleMems(START_SIZE);
  Array<int> addCols(START_SIZE);
  Array<int> newForcedCols(START_SIZE);

  Array<int> touchedCols(START_SIZE);
  Array<double> touchedColWeights(START_SIZE);

  // lambda for initializing a group starting with vertex "fvnr" in eqc "eqc"
  auto initGroup = [&](auto eqc, auto fvnr, auto const &taken) -> std::tuple<FlatArray<int>, FlatArray<int>>
  {
    auto viableNewMem = [&](auto vk, FlatArray<int> currGroup, int groupEQC, FlatArray<int> groupCols, FlatArray<int> currColEQCs, auto &taken)
    {
      // cout << " viableNewMem " << vk << " in eqc " << eqc << " for group-eqc " << groupEQC << endl;
      // cout << "   eqc_h.IsLEQ(eqc, groupEQC): " << eqc_h.IsLEQ(eqc, groupEQC) << endl;

      bool viable = ( !taken.Test(vk) ) && ( !isRoot(vk) );

      if ( viable )
      {
        viable &= ( find_in_sorted_array(vk, currGroup) == -1 );
      }

      if ( viable && eqc_h.IsTrulyParallel() )
      {
        /**
         * only allow new members that are hierarchically smaller than the initial member
         * due to the order we are going through the vertices, this only eliminates non-comparable ones
         * as all larger ones are already handled
         */
        viable &= eqc_h.IsLEQ(FM.template GetEQCOfNode<NT_VERTEX>(vk), groupEQC);
      }

      if ( viable && eqc_h.IsTrulyParallel() )
      {
        // cout << "    curr group-procs: "; prow(groupDistProcSet); cout << endl;

        /**
        * In order for a new member be viable, current_group \cup {new_mem} must be allowed
        * to prol from current_cols \cup new_mem_init_cols, that is:
        *  i) the new mems must be allowed to prol from current cols
        * ii) all old mems must be allowed to prol from new mem init-cols
        */

        auto newMemEQC = FM.template GetEQCOfNode<NT_VERTEX>(vk);

        // check (i): eqc(new_mem) <= eqc(col) for all current cols
        if (newMemEQC > 0)
        {
          // cout << "check (I) for neib " << vk << " in " << newMemEQC << endl;
          for (auto j : Range(currColEQCs))
          {
            // cout << "   against col-eq " << currColEQCs[j] << endl;
            if (!eqc_h.IsLEQ(newMemEQC, currColEQCs[j]))
            {
              // cout << "     IS BAD!" << endl;
              return false;
            }
          }
          // cout << "   IS OK " << endl;
        }

        // check (ii): groupDistProcSet \subset procs(newCol) \forall newCols
        if ( groupEQC > 0 )
        {
          // cout << "CHECK (ii) for neib " << vk << endl;
          auto vkCols = initGraph[vk];

          // cout << "   initG: "; prow(vkCols); cout << endl;

          auto addCols = setMinus(vkCols, groupCols, lh);

          // cout << "   -> addCols "; prow(addCols); cout << endl;

          // cout << " groupDistProcSet: "; prow(groupDistProcSet); cout << endl;

          for (auto j : Range(addCols))
          {
            auto addCol = addCols[j];
            auto colEQC = CM.template GetEQCOfNode<NT_VERTEX>(addCol);

            // cout << " colEQC = " << colEQC << endl;
            // cout << "    dps colEQC = "; prow(eqc_h.GetDistantProcs(colEQC));

            if (!eqc_h.IsLEQ(groupEQC, colEQC))
            {
              // cout << " IS BAD! " << endl;
              return false;
            }
          }
          // cout << "IS OK! " << endl;
        }
      }

      return viable;
    };

    auto iterateFC2 = [&](int vk,
                        FlatArray<int> cols,
                        auto lam)
    {
      auto neibs = fecon.GetRowIndices(vk);
      auto eNums = fecon.GetRowValues(vk);

      for (auto j : Range(neibs))
      {
        auto neib = neibs[j];
        auto cNeib = vmap[neib];

        int pos = find_in_sorted_array(cNeib, cols);

        if ( pos != -1 )
        {
          lam(neib, pos, cNeib, int(eNums[j]));
        }
      }
    };

    auto iterateFC = [&](FlatArray<int> vNums,
                        FlatArray<int> cols,
                        auto lam)
    {
      // loop over group members
      for (auto kg : Range(vNums))
      {
        auto const vk = vNums[kg];
        iterateFC2(vk, cols, [&](auto neib, auto cPos, auto cNeib, int eNum) { lam(kg, vk, neib, cPos, cNeib, eNum); });
      }
    };

    auto addPMem = [&](auto vk)
    {
      // cout << "   add PROSP Mem " << vk << endl;

      auto [isNewMem, posP] = sorted_insert_unique(vk, prospMems);

      // cout << "   NEW PROSP " << isNewMem << " @ " << posP << endl;

      if ( isNewMem )
      {
        auto forcedCols = initGraph[vk];

        int cntBlocking = setMinus(initGraph[vk], newCols, lh).Size();

        double wt = 0.0;
        iterateFC2(vk, forcedCols, [&](auto neib, auto cPos, auto cNeib, int eNum) {
          wt += ENERGY::GetApproxWeight(fEData[eNum]);
        });

        numBlocking.Insert(posP, cntBlocking);
        currWtP.Insert(posP, wt);
        prospWt.Append(0.0);
        addWtP.Append(0.0);
        // cout << " size numBlocking = " << numBlocking.Size() << endl;
        // cout << " size currWtP = " << currWtP.Size() << endl;
        // cout << " size prospWt = " << prospWt.Size() << endl;
        // cout << " size addWtP = " << addWtP.Size() << endl;
      }
    };

    auto removeProspMemPos = [&](auto vk, auto pos)
    {
      removeEntry(pos, prospMems);
      removeEntry(pos, currWtP);
      removeEntry(pos, prospWt);
      removeEntry(pos, addWtP);
      removeEntry(pos, numBlocking);
    };

    auto removeProspMem = [&](auto vk)
    {
      removeProspMemPos(vk, find_in_sorted_array(vk, prospMems));
    };


    auto addMem = [&](auto vk, auto eqc, auto groupEQC, auto &taken)
    {
      // cout << " addMem " << vk << endl;
      // cout << " -> pos " << pos << endl;
      // insert vk as new member and
      auto [isNewInGroup, posG] = sorted_insert_unique(vk, group);

      // if ( isNewInGroup && ( eqc > 0 ) )
      // {
      //   merge_a_into_b(eqc_h.GetDistantProcs(eqc), groupProcs, lh);
      // }

      // cout << " new  " << isNewInGroup << " -> set @" << posG << endl;

      double wt = 0.0;
      iterateFC2(vk, newCols, [&](auto neib, auto cPos, auto cNeib, int eNum) {
        wt += ENERGY::GetApproxWeight(fEData[eNum]);
      });

      currWtG.Insert(posG, wt);
      addWtG.Append(0.0);

      // remove it from prospective members
      removeProspMem(vk);

      // add neibs of vk as prospective members
      auto neibs = fecon.GetRowIndices(vk);
      for (auto j : Range(neibs))
      {
        auto const neib = neibs[j];
        if ( viableNewMem(neib, group, groupEQC, newCols, newColEQCs, taken) )
        {
          // cout << "ADD pMEM " << neib << endl;
          addPMem(neib);
        }
      }
    };

    auto addEligibleMems = [&](auto vk, FlatArray<int> additionalCols, auto groupEQC, auto &taken)
    {
      HeapReset hr(lh);

      // cout << " addEligibleMems from " << vk << " w. add. cols "; prow(additionalCols); cout << endl;
      // cout << "          select eligible from "; prow(prospMems); cout << endl;
      for (auto j : Range(additionalCols))
      {
        auto const addCol = additionalCols[j];

        auto [isNew, pos] = sorted_insert_unique(addCol, newCols);

        if ( eqc_h.IsTrulyParallel() && isNew )
        {
          auto colEQC = CM.template GetEQCOfNode<NT_VERTEX>(addCol);

          newColEQCs.Insert(pos, colEQC);
        }
      }

      // cout << " add primary new mem " << vk << endl;
      addMem(vk, eqc, groupEQC, taken);

      // cout << " add eligible form left over prosp-mems "; prow(prospMems); cout << endl;

      eligibleMems.SetSize0(); // addMem changes prospMems, so write eligible ones into seperate array

      for (auto k : Range(prospMems))
      {
        auto pMem = prospMems[k];
        if ( pMem != vk )
        {
          auto stillMissing = setMinus(initGraph[pMem], newCols, lh);
          int const nStillMissing = stillMissing.Size();
          if ( nStillMissing == 0 )
          {
            // cout << "  DO add p-mem " << k << " = " << pMem << "!" << endl;
            eligibleMems.Append(pMem);
          }
          else
          {
            // cout << "  DO NOT add p-mem " << k << " = " << pMem << ", # blocking = " << nStillMissing << ": " ; prow(stillMissing); cout << endl;
            numBlocking[k] = nStillMissing;
          }
        }
      }

      // cout << " -> eligible: "; prow(eligibleMems); cout << endl;
      for (auto j : Range(eligibleMems))
      {
        auto pMem = eligibleMems[j];
        auto pMemEQC = FM.template GetEQCOfNode<NT_VERTEX>(pMem);
        // cout << "  ADD p-mem prev index " << j << " = " << pMem << endl;
        addMem(pMem, pMemEQC, groupEQC, taken);
      }
    };

    auto const CV = vmap[fvnr];

    // cout << " initGroup for " << fvnr << " -> " << CV << endl;

    if (CV == -1)
    {
      group.SetSize(1); group[0] = fvnr;
      newCols.SetSize0();
      // cout << " DIRI, group: "; prow(group); cout << endl;
      // cout << "       cols "; prow(newCols); cout << endl;
      return make_tuple(group.Range(0, 1), newCols.Range(0, 0));
    }
    else if ( isRoot(fvnr) ) // do nothing to roots
    {
      newCols.SetSize(1); newCols[0] = CV;
      group.SetSize(1); group[0] = fvnr;
      // cout << " ROOT, group: "; prow(group); cout << endl;
      // cout << "       cols "; prow(newCols); cout << endl;
      return make_tuple(group.Range(0, 1), newCols.Range(0, 1));
    }

    // cout << " -> NON-TRIVIAL case! " << endl;

    group.SetSize0();
    // groupDistProcs.SetSize0();
    newCols.SetSize0();
    newColEQCs.SetSize0();
    numBlocking.SetSize0();
    addWtP.SetSize0();
    currWtP.SetSize0();
    prospWt.SetSize0();
    addWtG.SetSize0();
    currWtG.SetSize0();
    prospMems.SetSize0();

    /*
     * vertices are iterated through in hierarchic order,  we KNOW all hierachically larger vertices
     * are already handled.
     * !! We are only allowing hierarchically comparable (and therefore hierarchically smaller) new members !!
     *         this is a (small) restriction on what would still produce a hierarachic graph, but it makes things easier
     * therefore, the proc-set OF THE GROUP is non-increasing
     *    (the proc-set of the cols can, however, increase!)
     */
    auto const groupEQC = eqc;

    BitArray x;

    // cout << " init group with " << fvnr << ", LH available: " << lh.Available() << endl;
    // cout << "   starting w. forced cols "; prow(newCols); cout << endl;

    // add fvnr plus viable neibs as prospective members
    {
      HeapReset hr(lh);

      addPMem(fvnr);
      auto neibs = fecon.GetRowIndices(fvnr);
      for (auto j : Range(neibs))
      {
        auto const neib = neibs[j];
        if ( viableNewMem(neib, group, groupEQC, newCols, newColEQCs, taken) )
        {
          addPMem(neib);
        }
      }
    }

    // cout << " initial prosp mems: "; prow(prospMems); cout << endl;

    if ( MAX_GRP_SIZE > 1 )
    {
      addEligibleMems(fvnr, initGraph[fvnr], groupEQC, taken);
    }
    else // TESTING
    {
      // group.SetSize(1); group[0] = fvnr;
      addMem(fvnr, groupEQC, groupEQC, taken);
      newCols.SetSize(initGraph[fvnr].Size());
      newCols = initGraph[fvnr];
    }

    // cout << " initial group: "; prow(group); cout << endl;

    bool foundNeibToAdd = true;

    while ( ( newCols.Size() < MAX_PER_ROW_EXT ) &&
            ( group.Size()   < MAX_GRP_SIZE ) &&
            ( prospMems.Size() && foundNeibToAdd ) )
    {
      HeapReset hr(lh);

      // cout << " with current group "; prow(group); cout << " and cols "; prow(newCols); cout << endl;
      // cout << " look for new member among: "; prow(prospMems); cout << endl;

      // changes to prol-fracs if we added these cols; weights connecting eligible added
      // neibs to neibs mapping to a col in group that is not in initGraph of that vertex
      addWtP = 0.0;
      for (auto k : Range(prospMems))
      {
        int vk = prospMems[k];
        auto forcedCols = initGraph[vk];
        // currWt for prospective mems is the weight from the forced cols
        auto usedNotForced = setMinus(newCols, forcedCols, lh);
        iterateFC2(vk, usedNotForced, [&](auto neib, auto cPos, auto cNeib, int eNum)
        {
          if ( find_in_sorted_array(cNeib, initGraph[vk]) == -1 ) // why do this? should be done with setMinus above ?!
          {
            addWtP[k] += ENERGY::GetApproxWeight(fEData[eNum]);
          }
        });
        // actually, we would also have to add the weight for additional cols below,
        // but this should be good enough
      }

      // calc weight for prospective members
      for (auto l : Range(prospMems))
      {
        HeapReset hr(lh);
        auto pMem = prospMems[l];

        // cout << " pMem " << l << " = " << pMem << endl;

        // additional cols we would need to add pMem to group
        auto forcedCols = initGraph[pMem];
        auto neededAddCols = setMinus(forcedCols, newCols, lh);
        int const nBlocking = neededAddCols.Size();

        // cout << "    forcedCols "; prow(forcedCols); cout << endl;
        // cout << "       of those " << neededAddCols.Size() << " add. needed: "; prow(neededAddCols); cout << endl;

        if (nBlocking == 0)
        {
          // can and will be added for free
          prospWt[l] = 0.0;
          continue;
        }

        // # of verts we could add resulting from this
        auto expandedCols = merge_arrays_lh(newCols, neededAddCols, lh);

        // cout << "    expandedCols would be "; prow(expandedCols); cout << endl;
        eligibleMems.SetSize(1); eligibleMems[0] = l;
        int nAdded = 1;
        for (auto ll : Range(prospMems))
        {
          if (l != ll)
          {
            auto memll = prospMems[ll];
            auto forcedCols = initGraph[memll];
            auto neededAddColsForll = setMinus(forcedCols, expandedCols, lh);
            // cout << "    with that, additionally needed for " << memll << " with cols "; prow(forcedCols); cout << ": "; prow(neededAddCols); cout << endl;
            if (neededAddColsForll.Size() == 0)
            {
              eligibleMems.Append(ll);
              nAdded++;
            }
          }
        }

        // cout << " -> this would let us add " << nAdded << endl;

        // changes to prol-fracs if we added these cols; weights connecting group mems to neibs mapping to a col in neededAddCols
        addWtG = 0.0;
        iterateFC(group, neededAddCols, [&](auto k, auto vk, auto neib, auto cPos, auto cNeib, int eNum) {
          addWtG[k] += ENERGY::GetApproxWeight(fEData[eNum]);
        });


        // cout << " curr-wt G "; prow(currWtG); cout << endl;
        // cout << " add-wt G "; prow(addWtG); cout << endl;
        // cout << " curr-wt P "; prow(currWtP); cout << endl;
        // cout << " add-wt P "; prow(addWtP); cout << endl;

        // prol-factor improvement
        double bestPFIG = 0.0;
        double bestPFIP = 0.0;
        double worstPFIG = 1.0;
        double worstPFIP = 1.0;

        for (auto k : Range(group))
        {
          double const oldWt = currWtG[k];
          double const newWt = oldWt + addWtG[k];
          double const pfi = addWtG[k] / oldWt; // = (newWt / oldWt) - 1
          bestPFIG = max(bestPFIG, pfi);
          worstPFIG = min(worstPFIG, pfi);
        }

        for (auto k : Range(nAdded))
        {
          auto const kMem = eligibleMems[k];
          double const oldWt = currWtP[kMem];
          double const newWt = oldWt + addWtP[kMem];
          // bestPFIP = max(bestPFIP, newWt / oldWt);
          double const pfi = addWtP[kMem] / oldWt; // = (newWt / oldWt) - 1
          bestPFIP  = max(bestPFIG, pfi);
          worstPFIP = min(worstPFIP, pfi);
        }

        double const bestPFI  = max(bestPFIG, bestPFIP);
        double const worstPFI = min(worstPFIG, worstPFIP);

        // #freed/#blocking * prol_frac_improvement
        double const neibWt = double(nAdded) / nBlocking * worstPFI;

        prospWt[l] = neibWt;
      }

      // cout << "prowspWt: "; prow2(prospWt); cout << endl;

      auto idx = makeSequence(prospMems.Size(), lh);
      QuickSortI(prospWt, idx, [&](auto wi, auto wj) { return wi > wj; });

      // add enabled members of strongest col
      double const bestWt = prospWt[idx[0]];
      auto newMem = prospMems[idx[0]];

      // cout << " best weight " << bestWt << " from " << newMem << endl;

      auto additionalCols = setMinus(initGraph[newMem], newCols, lh);

      // cout << "   -> adding "; prow(additionalCols); cout << " with that!" << endl;

      if ( ( additionalCols.Size() == 0 ) || // there is at least 1 "free" member that does not need any add. cols!
           ( ( bestWt > 0.0 ) && ( newCols.Size() + additionalCols.Size() <= MAX_PER_ROW_EXT ) ) )
      {
        addEligibleMems(newMem, additionalCols, groupEQC, taken);
      }
      else
      {
        foundNeibToAdd = false;
      }
    }

    // add touched cols until row is full or all prol-fracs are sufficient
    // cout << " intermed cols = "; prow(newCols); cout << endl;

    if (false) // turn this off for now (debugging)
    if (newCols.Size() < MAX_PER_ROW_EXT)
    {
      HeapReset hr(lh);

      touchedCols.SetSize0();

      // union of all dist-procs of all members !!NO!!
      // !! we only allow new members DOWNWARDS IN HIERARCHY !! so do not use this!
      // auto groupDistProcs = MergeArrays(group.Size(), [&](auto k) { return eqc_h.GetDistantProcs(group[k]); }, lh);

      for (auto k : Range(group))
      {
        auto const vk = group[k];
        iterateFNeibs(vk, FM, [&](auto j, auto vj, int eNum)
        {
          auto cvj = vmap[vj];
          if ( ( cvj != -1 ) &&
               ( find_in_sorted_array(cvj, newCols) == -1 ) )
          {
            insert_into_sorted_array_nodups(cvj, touchedCols);
          }
        });
      }

      // cout << " init. touchedCols = "; prow(touchedCols); cout << endl;

      // in parallel, remove non-hierarchic touched cols
      if ( eqc_h.IsTrulyParallel() && ( groupEQC != 0 ) )
      {
        for (int j = 0; j < touchedCols.Size();)
        {
          auto cvj = touchedCols[j];
          auto cveq = CM.template GetEQCOfNode<NT_VERTEX>(cvj);

          // non-hierarchic if there is a proc in group-proc-set that is not in col-procs
          // if ( setMinus(groupDistProcs, eqc_h.GetDistantProcs(cveq), lh).Size() > 0 )

          // cout << " check " << j << "/" << touchedCols.Size() << " = " << cvj << " in eq " << cveq << endl;

          // can only allow new cols UPWARDS in hierarchy (need to be allowerd to prol from)
          if (!eqc_h.IsLEQ(groupEQC, cveq))
          {
            removeEntry(j, touchedCols); // decrement size
          }
          else
          {
            j++; // increment counter
          }
        }
      }

      // cout << " FINAL. touchedCols = "; prow(touchedCols); cout << endl;

      // best prol-frac improvement of any in-group mem!
      touchedColWeights.SetSize(touchedCols.Size());
      touchedColWeights = 0.0;

      auto &addWt = addWtP;
      addWt.SetSize(touchedCols.Size());

      for (auto k : Range(group))
      {
        auto   const vk     = group[k];
        double const currWt = currWtG[k];

        addWt = 0.0;

        iterateFC2(vk, touchedCols,
          [&](auto neib, auto cPos, auto cNeib, auto eNum) {
            addWt[cPos] += ENERGY::GetApproxWeight(fEData[eNum]);
          });

        for (auto j : Range(touchedCols))
        {
          double const pfi = addWt[j] / currWt;

          double const newPF = (currWt + addWt[j]) / totalWt[vk];

          // cout << " mem " << k << "=" << vk << " to tc " << j << " = " << touchedCols[j]
          //      << ", wt " << currWt << " -> " << currWt + addWt[j] << ", PF " << newPF
          //      << ", PFI " << pfi << endl;

          touchedColWeights[j] = max(touchedColWeights[j], pfi);
        }
      }

      // cout << " add " << newCols.Size() << " -> up to max " << MAX_PER_ROW_EXT << endl;
      // cout << " touchedCols: "; prow(touchedCols); cout << endl;
      // cout << " PFI: "; prow(touchedColWeights); cout << endl;

      auto idx = makeSequence(touchedCols.Size(), lh);
      QuickSortI(touchedColWeights, idx, [&](auto wi, auto wj) { return wi > wj; });

      auto const maxAdd = min(touchedCols.Size(), MAX_PER_ROW_EXT - newCols.Size());

      for (int ll = 0; ll < maxAdd; ll++)
      {
        int    const newCol = touchedCols[idx[ll]];
        double const colPFI = touchedColWeights[idx[ll]];
        // cout << " add col " << newCol << " with wt " << colPFI << endl;
        insert_into_sorted_array_nodups(newCol, newCols);
        if (colPFI < MIN_PROL_FRAC)
        {
          break;
        }
      }
    }

    // TODO: should we do something close to e.g. MPI-boundaries here?
    // TODO: since we are adding cols here based on whatever we can still add to the group,
    //       are we ignoring cols from neibs in other groups too much here? I guess probably??


    // not sorted yet! TODO: make sure I only sort once SOMEwhere (creategraph, here, OR in fill-prol)
    // QuickSort(newCols);

    // cout << " create group around " << fvnr << " with init. graph "; prow(newCols); cout << endl;

    // addCols.SetSize(MAX_PER_ROW_EXT);

    // auto fNeibs = fecon.GetRowIndices(fvnr);
    // auto fEnrs  = fecon.GetRowValues(fvnr);

    // // should we try to add neibs of newly added group-mems too?
    // int pos;
    // for (auto j : Range(fNeibs)) // should we go through here in descending edge-weight order ?
    // {
    //   auto const neib = fNeibs[j];

    //   if (!taken.Test(neib))
    //   {
    //     auto neibCols = initGraph[neib];

    //     int nAddCols = 0;

    //     for (auto l : Range(neibCols))
    //     {
    //       auto const neibCol = neibCols[l];
    //       if ( (pos = find_in_sorted_array(neibCol, newCols)) == -1 )
    //       {
    //         if (eqc_h.IsLEQ(eqc, CM.template GetEQCOfNode<NT_VERTEX>(neibCol))) // admissible
    //         {
    //           addCols[nAddCols++] = neibCol;
    //           if (nNewCols + nAddCols == MAX_PER_ROW_EXT)
    //           {
    //             break;
    //           }
    //         }
    //       }
    //     }

    //     if (nNewCols + nAddCols <= MAX_PER_ROW_EXT)
    //     {
    //       // cout << " add neib " << j << "=" << neib << ", adding " << nAddCols << " cols: "; prow(addCols.Range(0, nAddCols)); cout << endl;
    //       insert_into_sorted_array_nodups(neib, group);
    //       for (auto ll : Range(nAddCols))
    //         { insert_into_sorted_array_nodups(addCols[ll], newCols); }
    //       nNewCols += nAddCols;
    //       // cout << " -> group now "; prow(group); cout << ", cols now "; prow(newCols); cout << endl;
    //     }
    //   }
    // }

    // cout << " -> FINAL GRP: "; prow(group); cout << endl;
    // cout << "      with COLS: "; prow(newCols); cout << endl;

    return std::make_tuple(FlatArray<int>(group), FlatArray<int>(newCols));
  };

  return CreateGroupedGraph<ENERGY, TSPM, TMESH>(cmap, FM, CM, isRoot, initGroup, lh, doPrint);
} // CreateEmptyGroupWiseSProl



template<bool IS_GATHER, class T, class TIN>
INLINE
Array<MPI_Request>
StartPairWiseGS (NgsAMG_Comm comm,
                 FlatArray<int> dps,
                 TIN &buffers)
{
  Array<MPI_Request> reqs(dps.Size());

  if (comm.isValid())
  {
    reqs.SetSize(dps.Size());

    auto doSend = [&](auto dp)
    {
      if constexpr(IS_GATHER)
      {
        return dp < comm.Rank();
      }
      else
      {
        return dp > comm.Rank();
      }
    };

    for (auto k : Range(dps))
    {
      if ( doSend(dps[k]) )
      {
        reqs[k] = comm.ISend(buffers[k], dps[k], NG_MPI_TAG_AMG);
      }
      else
      {
        reqs[k] = comm.IRecv(buffers[k], dps[k], NG_MPI_TAG_AMG);
      }
    }
  }

  return reqs;
} // StartPairWiseGS


template<class T, class TIN>
INLINE
Array<MPI_Request>
StartPairWiseGather (NgsAMG_Comm comm,
                      FlatArray<int> dps,
                      TIN &buffers)
{
  return StartPairWiseGS<true, T>(comm, dps, buffers);
} // StartPairWiseGather


template<class T, class TIN>
INLINE
Array<MPI_Request>
StartPairWiseScatter (NgsAMG_Comm comm,
                      FlatArray<int> dps,
                      TIN &buffers)
{
  return StartPairWiseGS<false, T>(comm, dps, buffers);
} // StartPairWiseScatter


template<int BS>
double
SPOmegaEst(shared_ptr<SparseMat<BS, BS>> pA,
           shared_ptr<BitArray> freeDOFs,
           LocalHeap &lh,
           int steps = 10)
{
  auto D = make_shared<DiagonalMatrix<StripTM<BS,BS>>>(pA->Height());

  Matrix<double> d(BS, BS);

  for (auto k : Range(pA->Height()))
  {
    if ( (freeDOFs == nullptr) || freeDOFs->Test(k) )
    {
      HeapReset hr(lh);
      setFromTM(d, 0, 0, 1.0, (*pA)(k,k));
      CalcPseudoInverseWithTol(d, lh, 1e-10, 1e-10);
      setTM((*D)(k), 1.0, d, 0, 0);
    }
    else
    {
      (*D)(k) = 0.0; // diagonal matrix (i) returns the TM-reference
    }
  }

  auto v = pA->CreateVector();
  auto Av = pA->CreateVector();
  auto w = pA->CreateVector();

  double maxEV = 1.0;

  *v = 1.0;
  int mid = pA->Height() / 2;
  if (pA->Height() > 0)
  {
    (*v).FVDouble()[mid] = 2;
  }

  cout << " EV-est " << endl;
  for (auto k : Range(steps))
  {
    pA->Mult(*v, *Av);
    D->Mult(*Av, *w);

    double n = InnerProduct(*v, *w);

    cout << " it " << k << ", est = " << n << ", relErr = " << fabs(n-maxEV)/maxEV << endl;
    maxEV = n;
    double const scale = 1.0 / L2Norm(*w);
    *v = scale * (*w);
  }

  NgMPI_Comm comm(MPI_COMM_WORLD);

  DoTest(*pA, *D, comm, "KB-test");

  return 1.0/maxEV;
}


template<class ENERGY, class TMESH, int BS>
shared_ptr<ProlMap<typename VertexAMGFactory<ENERGY, TMESH, BS>::TM>>
VertexAMGFactory<ENERGY, TMESH, BS>::
GroupWiseSProl (BaseCoarseMap &cmap,
                LevelCapsule  &fcap,
                LevelCapsule  &ccap)
{
  static Timer t("GroupWiseSProl");
  RegionTimer rt(t);

  static Timer tgraph("GroupWiseSProl - graph");
  static Timer tfill("GroupWiseSProl - fill");
  static Timer tcapMM("GroupWiseSProl - CAP-MM");

  Options &O (static_cast<Options&>(*options));

  const int baselevel        = fcap.baselevel;
  const size_t MAX_PER_ROW   = O.sp_max_per_row.GetOpt(baselevel);
  const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(baselevel);
  const double MIN_SUM_FRAC  = 1.0 - sqrt(MIN_PROL_FRAC); // MPF 0.15 -> 0.61
  // const double omega         = O.sp_omega.GetOpt(baselevel);
  // const bool aux_only = O.sp_aux_only.GetOpt(baselevel); // TODO:placeholder
  // const bool aux_only = O.prol_type.GetOpt(baselevel) == Options::PROL_TYPE::AUX_SMOOTHED;
  int const extraSmoothingSteps = O.sp_extra_steps.GetOpt(baselevel);

  int const smoothingSteps = 1 + extraSmoothingSteps;

  // choose actual omega such that (1-omega)**smoothingSteps = (1 - baseOmega)
  double const baseOmega = O.sp_omega.GetOpt(baselevel);
  // double const omega     = 1 - pow(1 - baseOmega, 1./smoothingSteps);
  double const omega     = baseOmega;

  // cout << " baseOmega " << baseOmega << ", #steps " << smoothingSteps << " -> OMEGA = " << omega << endl;

  bool const printProgress       = O.log_level >= Options::LOG_LEVEL::NORMAL;
  bool const printProgressDetail = O.log_level >= Options::LOG_LEVEL::EXTRA;

  // not sure what that was about exactly, I think experimenting/debugging?
  const double rscale = O.rot_scale.GetOpt(fcap.baselevel);

  const TMESH & FM(static_cast<TMESH&>(*cmap.GetMesh()));
  const TMESH & CM(static_cast<TMESH&>(*cmap.GetMappedMesh()));

  FM.CumulateData();
  CM.CumulateData();

  const auto & eqc_h(*FM.GetEQCHierarchy()); // coarse eqch == fine eqch !!
  auto neqcs = eqc_h.GetNEQCS();

  auto vmap = cmap.template GetMap<NT_VERTEX>();

  const auto & fecon = *FM.GetEdgeCM();
  const auto & cecon = *CM.GetEdgeCM();
  auto fEdges = FM.template GetNodes<NT_EDGE>();
  auto fVData = get<0>(FM.Data())->Data();
  auto fEData = get<1>(FM.Data())->Data();
  auto cVData = get<0>(CM.Data())->Data();

  /**
   * Tolerance for A_gg pseudo-inverse, Eigenvalues smaller
   * than this (relative tolerance) are replaced by
   *  "cInvTolR * avgEVal", that is the Eval of the pseudo-inverse
   * is limited to 1/cInvTolR * avgEval, the prolongation
   * is under-corrected in the range of these Eigenvectors
   */
  double const cInvTolR = 5e-2;

  /*
   * Tolerance for A_gg pseudo-inverse, Eigenvalues smaller
   * than this (relative tolerance) are discarded/set to zero.
   * The prolongation is NOT corrected in the range of these
   * Eigenvectors.
   * This is fine
   */
  double const cInvTolZ = 1e-8;

  auto const FNV = FM.template GetNN<NT_VERTEX>();
  auto const CNV = CM.template GetNN<NT_VERTEX>();

  auto aggs = cmap.template GetMapC2F<NT_VERTEX>();

  //
  auto meshPDs = fcap.uDofs.GetParallelDofs();

  /**
   * On the finest level, the "fmat" can be the pre-embedded matrix,
   * e.g. 3-to-6 embedding. The cast would fail and fmat would be nullptr,
   * in that case, we can and should not use it here. Otherwise, when
   * we DO assemble the mesh-canonic matrix, we can use it.
   * We also need to check for finest level parallel reordering.
   */
  shared_ptr<TSPM> pA  = dynamic_pointer_cast<TSPM>(fcap.mat);
  shared_ptr<TSPM> pAP = nullptr;

  // shared_ptr<TSPM> pAhat    = nullptr;
  // shared_ptr<TSPM> pAhatP   = nullptr;
  // shared_ptr<TSPM> pPTAhatP = nullptr;

  shared_ptr<TSPM> pTAP = nullptr;

  // fmat = nullptr;

  bool haveRealFMat = ( pA != nullptr ) && ( ( pA->GetParallelDofs() == nullptr ) || // not parallel
                                             ( pA->GetParallelDofs() == meshPDs ) ); // not reordered

  // haveRealFMat = false;

  if (!haveRealFMat)
  {
    BitArray diriBA(FNV);
    diriBA.Clear();
    for (auto k : Range(FNV))
    {
      if ( vmap[k] == -1 )
      {
        diriBA.Set(k);
      }
    }

    bool const includeVertexContribs = false;
    pA = AssembleAhatSparse<ENERGY, TMESH>(FM, includeVertexContribs, &diriBA);

    // std::ofstream of("ass_ahat.out");
    // print_tm_spmat(of, *pA);
  }

  // if (!haveRealFMat)
  // {
  //   pA = pAhat;
  // }

  auto pWMap = PWProlMap(cmap, fcap, ccap);

  auto const &A = *pA;
  auto const &pWP = *pWMap->GetProl();

  LocalHeap lh(51380224, "Carrot", false); // 49 MB

  if (false) // debugging, does not work in parallel
  {
    auto freeVerts = make_shared<BitArray>(FNV);
    freeVerts->Clear();
    for (auto k : Range(FNV))
    {
      if (vmap[k] == -1)
      {
        freeVerts->SetBit(k);
      }
    }
    freeVerts->Invert();

    cout << " SPOmegaEST A-AUX " << endl;
    SPOmegaEst<BS>(pA, freeVerts, lh, 30);

    if (auto AA = dynamic_pointer_cast<SparseMat<3,3>>(fcap.mat))
    {
      cout << " SPOmegaEST 3x3 A-AUX: " << endl;
      SPOmegaEst<3>(AA, freeVerts, lh, 30);
    }
  }


  Array<int> aggRoots = FindAggRoots(cmap, FM, CM, lh);

  BitArray isRootBA(FNV);
  isRootBA.Clear();

  CM.template ApplyEQ<NT_VERTEX>([&](auto eqc, auto aggNr)
  {
    auto fv = aggs[aggNr][aggRoots[aggNr]];

    isRootBA.SetBit(fv);

    // for elasticity on level 0, set a "secondary" root so the rotations are not
    // smoothed out too much
    if ( false ) // (BS > 0) && (!haveRealFMat) )
    {
      auto neibs = fecon.GetRowIndices(fv);
      auto fENrs = fecon.GetRowValues(fv);

      double maxWt = 0.0;
      int    maxIdx = -1;
      iterate_intersection(aggs[aggNr], neibs, [&](auto idxA, auto idxN)
      {
        int fENr(fENrs[idxN]);

        auto const avgWt = ENERGY::GetApproxWeight(fEData[fENr]);

        if (avgWt > maxWt)
        {
          maxWt = avgWt;
          maxIdx = idxN;
        }
      });

      if (maxIdx != -1)
      {
        cout << " SET secondary root for agg " << aggNr << ": "; prow(aggs[aggNr]); cout << endl;
        cout << "      initial root = " << fv << ", secondary root = " << neibs[maxIdx] << endl;
        isRootBA.SetBit(neibs[maxIdx]);
      }
    }
  }, true); // master only!! agg-roots only make sense on master!

  // std::cout << " FindAggRoots DONE " << std::endl<< " aggRoogs = "; prow3(aggRoots); cout << endl;

  // TODO: make this configurable from options
  auto isRoot = [&](auto vi) {
    // return false;
    return isRootBA.Test(vi);
    // auto cvi = vmap[vi];
    // return (cvi != -1) && (aggs[cvi][aggRoots[cvi]] == vi);
  };


  /** Generate graph of prolongation **/

  tgraph.Start();

  auto [gCSB, gPerExEQSB, gRSB, gTSB, spSB] =
    CreateEmptyGroupWiseSProl<ENERGY, TSPM>(cmap,
                                            FM,
                                            CM,
                                            MAX_PER_ROW,
                                            MIN_PROL_FRAC,
                                            MIN_SUM_FRAC,
                                            isRoot,
                                            lh,
                                            O.log_level > Options::LOG_LEVEL::NORMAL);

  // cannot lambda-capute structured bindings
  auto &numGroups         = gCSB;
  auto &groupNumsPerExEQC = gPerExEQSB;
  auto &groupReps         = gRSB;
  auto &sprol             = spSB;

  auto groups             = gTSB.GetFlatTable();

  const auto &CSP = *sprol;

  for (auto k : Range(FNV))
  {
    auto const CV = vmap[k];

    if ( CV != -1 )
    {
      auto ris = sprol->GetRowIndices(k);
      auto rvs = sprol->GetRowValues(k);

      int const pos = find_in_sorted_array(CV, ris);

      if (pos == -1)
      {
        cout << " ERR - CV NOT FOUND: " << k << " -> " << vmap[k] << " IN "; prow(ris); cout << endl;
      }

      rvs = .0;
      rvs[pos] = pWP.GetRowValues(k)[0];
    }
  }

  tgraph.Stop();

  // cout << " EQCH: " << endl << eqc_h << endl;
  // cout << " GW-GROUPS: " << endl << groups << endl;
  // cout << " groups per ex-eq: " << endl << groupNumsPerExEQC << endl;
  // cout << " GW-GRAPH: " << endl;
  // for (auto k : Range(FNV))
  // {
  //   cout << k << ": "; prow(CSP.GetRowIndices(k)); cout << endl;
  // }
  // cout << endl << endl;

  cout << " GW-GRAPH IN DETAIL: " << endl;
  FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes)
  {
    for (auto k : Range(nodes))
    {
      auto fv = nodes[k];
      auto cv = vmap[fv];

      if (cv != -1)
      {
        auto [eqV, lnr] = CM.template MapENodeToEQLNR<NT_VERTEX>(cv);
        auto cols = CSP.GetRowIndices(fv);

        for (auto j : Range(cols))
        {
          if (cols[j] < 0 || cols[j] >= CM.template GetNN<NT_VERTEX>())
          {
            cout << " FV " << fv << ", " << k << " in " << eqc << " -> " << cv << endl;
            cout << ", " << lnr << " in " << eqV << endl;
            cout << " cols: " << endl;
            cout << " ERROR - OUT OF RANGE! " << cols[j] << " vs " << CM.template GetNN<NT_VERTEX>() << endl;
          }
          else
          {
            auto [ceqV, clnr] = CM.template MapENodeToEQLNR<NT_VERTEX>(cols[j]);
            if (!eqc_h.IsLEQ(eqc, ceqV))
            {
              cout << " FV " << fv << ", " << k << " in " << eqc << " -> " << cv;
              cout << ", " << lnr << " in " << eqV << endl;
              cout << " col " << j << "/" << cols.Size() << " = " << cols[j] << " = " << clnr << " in " << ceqV << endl;;
              cout << "  ERROR - HIERARCHY, " << eqc << " > " << ceqV << "!" << endl;
              cout << "           dps " << eqc << ": "; prow(eqc_h.GetDistantProcs(eqc)); cout << endl;
              cout << "           dps " << ceqV << ": "; prow(eqc_h.GetDistantProcs(ceqV)); cout << endl;
            }
          }
        }
      }
    }
  }, false);
  cout << " GW-GRAPH CHECK DONE!" << endl;

  eqc_h.GetCommunicator().Barrier();
  cout << " GW-GRAPH CHECK DONE!" << endl;

  /**
   *  In every smoothing interation, the update for a group is somethiong like
   *     A_GG^{-1} (AP_GC | AP_GCN) (I | E_NC)
   *  Where:
   *       - G..group, C..cols, N..neib-cols
   *       - AP ist just A_fine * P     (A_fine can be aux-mat or real mat)
   *       - E_NC is an extension from cols of AP-row that are in group-cols
   *         to those that are not. It is of the form
   *              E_NC = coarseAhat_NN^{-1} coarseAhat_NG
   *  A_GG stays constant over smoothing iterations.
   *  E_NC is the coarse aux-mat extension, it stays constant FOR NOW (but that might change)
   *  AP changes between iterations
   *
   *  In order for the prolongation to be hierarchic, every proc that has a part of
   *  a group has ALL of its group-cols, i.e. the cols of the P-matrix.
   *  HOWEVER, not all members have all "N" cols of the AP-row.
   *
   *  So, we do the following:
   *    I. preparation:
   *       I.i)   collect A_GG:
   *                everyone has part of A_GG, collect that on master and cumulate together (EASY)
   *       I.ii)  collect global AP-rows:
   *                everyone writes the AP-rows (rows of a C2D matrix) they have, write cols as (eq,lnr) tuples
   *                master merges these together
   *                master keeps track of member-to-final-col mapping
   *                       ( This is for E_NC application & for later update !!)
   *       I.iii) collect coarseAhat_C-rows:
   *                everyone collects coarse edges+data to assemble into coarseA_NN, coarseA_NC
   *                master assembles, computes E_NC
   *   II. smoothing iterations:
   *         II.i)   smooth master-groups
   *         II.ii)  scatter master-group val-updates
   *         II.iii) smooth local groups
   *         II.iv)  update ex-recv-group val-updates
   *         II.v)   (if further iterations), update AP
   *             II.v.i)  compute AP (normal spmm, C2D x C2C)
   *             II.v.ii) collect AP-rows, everyone writes, master merges and uses stored local col-mapping
   */


  /** I. Preparation **/


  auto isInterestingC = [&](auto k)
  {
    // return ( k == 78 );
    return std::integral_constant<bool, false>();
  };

  auto isInterestingCols = [&](auto cols)
  {
    return std::integral_constant<bool, false>();
    // for (auto j : Range(cols))
    // {
    //   if (isInterestingC(cols[j]))
    //   {
    //     return true;
    //   }
    // }
    // return false;
  };

  auto updateLocalGroup = [&](auto step, auto groupNum, auto const &usedCSP)
  {
    auto group = groups[groupNum];

    auto ris = CSP.GetRowIndices(group[0]);

    // cout << " updateLocalGroup " << groupNum << endl;

    // Dirichlet, root or just nothing to do
    if ( ris.Size() < 2 )
      { return; }

    auto const &AP = *pAP;

    ReSmoothGroupNonExpansiveV2<ENERGY>(group,
                                      vmap,
                                      omega,
                                      FM,
                                      CM,
                                      usedCSP,
                                      A,
                                      AP,
                                      [&](auto fv) -> FlatArray<int>
                                      {
                                        return A.GetRowIndices(fv);
                                      },
                                      ris,
                                      [&](FlatMatrix<double> scalVals)
                                      {
                                        // if (isInterestingCols(CSP.GetRowIndices(group[0])))
                                        // {
                                        //   cout << step << ", update group " << groupNum << ": "; prow(group); cout << endl;
                                        //   cout << "    cols: "; prow(CSP.GetRowIndices(group[0])); cout << endl;

                                        //   FlatMatrix<double> oldVals(scalVals.Height(), scalVals.Width(), lh);
                                        //   for (auto l : Range(group))
                                        //   {
                                        //     auto const mem = group[l];
                                        //     auto       rvs = CSP.GetRowValues(mem);
                                        //     for (auto j : Range(rvs))
                                        //     {
                                        //       setFromTM(oldVals, l*BS, j*BS, 1.0, rvs[j]);
                                        //     }
                                        //   }
                                        //   cout << " oldVals: " << endl;
                                        //   cout << oldVals << endl;
                                        //   // PrintComponent(0, BS, oldVals);
                                        //   cout << endl;
                                        //   cout << " update: " << endl;
                                        //   cout << scalVals << endl;
                                        //   // PrintComponent(0, BS, scalVals);
                                        //   cout << endl;
                                        // }

                                        for (auto l : Range(group))
                                        {
                                          auto const mem = group[l];
                                          auto       rvs = CSP.GetRowValues(mem);
                                          for (auto j : Range(rvs))
                                          {
                                            addToTM(rvs[j], 1.0, scalVals, l * BS, j * BS);
                                          }
                                        }

                                        // if (isInterestingCols(CSP.GetRowIndices(group[0])))
                                        // {
                                        //   cout << step << ", updated group " << groupNum << ": "; prow(group); cout << endl;
                                        //   cout << "    cols: "; prow(CSP.GetRowIndices(group[0])); cout << endl;

                                        //   FlatMatrix<double> oldVals(scalVals.Height(), scalVals.Width(), lh);
                                        //   for (auto l : Range(group))
                                        //   {
                                        //     auto const mem = group[l];
                                        //     auto       rvs = CSP.GetRowValues(mem);
                                        //     for (auto j : Range(rvs))
                                        //     {
                                        //       setFromTM(oldVals, l*BS, j*BS, 1.0, rvs[j]);
                                        //     }
                                        //   }
                                        //   cout << " newVals: " << endl;
                                        //   cout << oldVals;
                                        //   // PrintComponent(0, BS, oldVals);
                                        //   cout << endl;
                                        // }

                                        // if constexpr(Height<TM>() == 6)
                                        // {
                                        //   // if ( fvnr == 2241 || fvnr == 1840)
                                        //   {
                                        //     Iterate<6>([&](auto l)
                                        //     {
                                        //       for (auto k : Range(group))
                                        //       {
                                        //         auto const mem = group[k];
                                        //         auto       rvs = CSP.GetRowValues(mem);
                                        //         double sum = 0;

                                        //         for (auto j : Range(rvs.Size()))
                                        //         {
                                        //           sum += rvs[j](l.value, l.value);
                                        //         }
                                        //         sum = abs(1.0 - sum);
                                        //         if ( ( sum > 1e-4 ) || ( mem == 15094) )
                                        //         cout << "  GW-SP row " << mem << "." << int(l.value) << " diagSum = " << sum << endl;
                                        //       }
                                        //     });
                                        //   }
                                        // }
                                      },
                                      lh,
                                      cInvTolR,
                                      cInvTolZ,
                                      step == 0); // firstReSmooth
  };


  auto updateExGroup = [&](auto groupNum, auto A_gg_inv, auto AP_gC, auto AP_gN, auto E_NC)
  {
    auto group = groups[groupNum];

    auto ris = CSP.GetRowIndices(group[0]);

    // Dirichlet, root or just nothing to do
    if ( ris.Size() < 2 )
      { return; }

    // cout << " updateExGroup " << groupNum << endl;

    auto upVals = [&](FlatMatrix<double> scalVals)
    {
      // cout << " scalVals: " << endl << scalVals << endl;
      for (auto l : Range(group))
      {
        auto const mem = group[l];
        auto       rvs = CSP.GetRowValues(mem);
        for (auto j : Range(rvs))
        {
          addToTM(rvs[j], 1.0, scalVals, l * BS, j * BS);
        }
      }

      // cout << " NEW prol-block: " << endl;
      // for (auto l : Range(group))
      // {
      //   auto const mem = group[l];
      //   auto       rvs = CSP.GetRowValues(mem);
      //   for (auto j : Range(rvs))
      //   {
      //     setFromTM(scalVals, l*BS, j*BS, 1.0, rvs[j]);
      //     // addToTM(rvs[j], 1.0, scalVals, l * BS, j * BS);
      //   }
      // }
      // cout << scalVals << endl;
    };

    ReSmoothGroupNonExpansiveCalc(A_gg_inv, AP_gC, AP_gN, E_NC, omega, lh, upVals);
  };

  GroupedSPExchange<TMESH, ENERGY> groupedExchange(FM,
                                                   CM,
                                                   A,
                                                   groups,
                                                   groupNumsPerExEQC,
                                                   smoothingSteps,
                                                   cInvTolR,
                                                   cInvTolZ);

  // groups: [numGroups[0] ex-master, numGroups[1] loc, numGroups[2] ex-rec]
  auto [numGExM, numGLoc, numGExR] = numGroups;

  auto locGroupRange = Range(numGExM, numGExM + numGLoc);

  // auto locGroups = groupsPerEQC[0];
  int const midLoc = numGExM + locGroupRange.Size() / 2;

  IntRange localRangeA(locGroupRange.First(), midLoc);
  IntRange localRangeB(midLoc,                locGroupRange.Next());

  shared_ptr<TSPM> pPT;


  if ( smoothingSteps == 1 )
  {
    /**
     * We only do a single round of prolongation smoothing, we need AP only once
     * as A * PWP, so do the cheaper spmm product.
     */
    pAP    = MatMultAB(A, pWP);

    pPT = TransposeSPM(pWP);

    // if ( haveRealFMat )
    // {
    //   pAhatP = MatMultAB(*pAhat, pWP);
    // }
    // else
    // {
    //   pAhatP = pAP;
    // }
  }
  else
  {
    /**
     * We will need to do the MatMultAB with CSP anyways for subsequente iterations.
     * The more expensive spmm product with CSP which has extra zero entries
     * gives us the final sparsity pattern that will not change anymore.
     * So, might as well do it right now so we can always use MatMultABUpdateVals later.
     * Extra bonus: AP sparsity pattern stays constant, GroupedSPExchange does not
     * need to re-size the AP-merging buffers, re-compute the merge-indices, etc.
     */
    pAP = MatMultAB(A, CSP);
    pPT = TransposeSPM(CSP);

    // if ( haveRealFMat )
    // {
    //   pAhatP = MatMultAB(*pAhat, CSP);
    // }
    // else
    // {
    //   pAhatP = pAP;
    // }
  }

  // pPTAhatP = MatMultAB(*pPT, *pAhatP);

  // cout << " sprol: " << endl;
  // {
  //   std::ofstream of("PTAP_l " + std::to_string(baselevel) + "_step_0.out");
  //   print_tm_spmat(of, *pPTAhatP);
  // }


  for (auto step : Range(smoothingSteps))
  {
    cout << "SMOOTHING-ITERATION " << step << "/" << smoothingSteps << endl;

    if ( step > 0 )
    {
      MatMultABUpdateVals(A, CSP, *pAP);
      pPT = TransposeSPM(CSP);

      // if ( haveRealFMat )
      // {
      //   MatMultABUpdateVals(*pAhat, CSP, *pAhatP);
      // }
      // MatMultABUpdateVals(*pPT, *pAhatP, *pPTAhatP);
      // {
      //   std::ofstream of("PTAP_l " + std::to_string(baselevel) + "_step_" + std::to_string(step) + ".out");
      //   print_tm_spmat(of, *pPTAhatP);
      // }
    }

    // auto sprolCopy = make_shared<SparseMat<BS,BS>>(*sprol);
    // auto const &oldCSP = *sprolCopy;
    // sprolCopy->AsVector() = CSP.AsVector();

    bool const firstRound = step == 0;

    groupedExchange.InitializeIteration(step, CSP, *pAP, lh);

    // cout << " InitializeIteration " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " InitializeIteration " << endl;

    groupedExchange.StartGather(step, CSP, *pAP, lh);

    // cout << " StartGather " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " StartGather " << endl;

    // update first half of local groups
    for (auto groupNum : localRangeA)
    {
      updateLocalGroup(step, groupNum, CSP);
    }

    // cout << " localRangeA " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " localRangeA " << endl;

    // wait for buffer-gather communication, merge received data, update ex-groups
    groupedExchange.ApplyToReceivedData(step,
                                        *pAP,
                                        CSP,
                                        lh,
                                        updateExGroup);

    // cout << " ApplyToReceivedData " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " ApplyToReceivedData " << endl;

    // scatter updated ex-vals
    groupedExchange.StartValScatter(CSP);

    // cout << " StartValScatter " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " StartValScatter " << endl;

    // update second half of local groups
    for (auto groupNum : localRangeB)
    {
      updateLocalGroup(step, groupNum, CSP);
    }

    // cout << " localRangeB " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " localRangeB " << endl;

    // waits for updated prol-vals communication
    groupedExchange.ApplyProlUpdate(*sprol);

    // cout << " ApplyProlUpdate " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << " ApplyProlUpdate " << endl;

    if ( false ) // debugging
    {
      for (auto k : Range(FNV))
      {
        auto ris = CSP.GetRowIndices(k);
        auto rvs = CSP.GetRowValues(k);

        if ( ris.Size() > 1 )
        {
          cout << " CSP-CHECK row = " << k << endl;
          TM mSum = 0;
          for (auto j : Range(ris))
          {
            auto const J = ris[j];
            TM P_kJ = rvs[j];

            // cout << " P_kJ " << k << " " << J << ": " << endl;
            // print_tm(cout, P_kJ);
            // cout << endl;

            // M_kJ Q^{J->i} = P_iJ
            auto Q_kJ = ENERGY::GetQiToj(fVData[k], cVData[J]);
            TM M_kJ = Q_kJ.GetMQ(1.0, P_kJ);

            // cout << " M_kJ " << k << " " << J << ": " << endl;
            // print_tm(cout, M_kJ);
            // cout << endl;

            // TM MT = Trans(M_kJ);
            // TM diff = M_kJ - MT;

            mSum += M_kJ;

            // auto nMt   = L2Norm(M_kJ);
            // auto nDiff = L2Norm(diff);

            // if ( nDiff > 1e-4 * nMt )
            // {
            //   cout << "  -> ASYMMETRY!" << endl;
            // }
          }

          cout << " mSUM: " << endl; print_tm(cout, mSum); cout << endl;
          TM id; SetIdentity(id);
          TM diff = mSum - id;
          auto nDiff = L2Norm(diff);

          if ( nDiff > 1e-4 * Height<TM>() )
          {
            cout << "  -> M-SUM MISMATCH!" << endl;
          }

        }
      }
    }

    // if ( step > 0 )
    // if (false) // debugging
    // if ( smoothingSteps > 1 ) // something does not work with smoothingSteps==1
    if ( false ) // ptap not correct right now
    {
      std::cout << " GW-SP BF energy check, CNV = " << CNV << "! " << std::endl;
      cout      << "    BS = " << BS << endl;

      MatMultABUpdateVals(A, CSP, *pAP);
      auto PT = TransposeSPM(CSP);
      auto newPTAP = MatMultAB(*PT, *pAP);

      Vec<BS,double> worst = 0;

      for (auto k : Range(CNV))
      {
        auto const oldAkk = (*pTAP)(k,k);
        auto const newAkk = (*newPTAP)(k,k);

        auto rvsLO = pPT->GetRowValues(k);
        auto rvsLN = PT->GetRowValues(k);

        Iterate<BS>([&](auto l)
        {
          double o, n;
          double ol = 0, nl = 0;

          if constexpr( BS > 1 )
          {
            o = oldAkk(l,l);
            n = newAkk(l,l);

            for (auto j : Range(rvsLO))
            {
              Iterate<BS>([&](auto ll)
              {
                ol += sqr(rvsLO[j](l, ll));
                nl += sqr(rvsLN[j](l, ll));
              });
            }
          }
          else
          {
            for (auto j : Range(rvsLO))
            {
              ol += sqr(rvsLO[j]);
              nl += sqr(rvsLN[j]);
            }
            o = oldAkk;
            n = newAkk;
          }
          ol = sqrt(ol);
          nl = sqrt(nl);

          // old l2 < 1e-10 -> probably kernel
          if ( ol < 1e-10 )
          {
            return;
          }

          // double frac = (n * ol) / (o * nl);
          double frac = n / o;

          bool doPrint = ( frac < 0 ) || isInterestingC(k);
          // bool doPrint = true;

          if ( frac > worst(l) )
          {
            worst(l) = frac;
            doPrint = true;
          }

          if ( doPrint )
          {
            cout << endl << "SM-IT " << step << " change of BF (" << k << "." << l << "): "
                 << o << "  -> " << n  << ", frac = " << n/o << endl;
            cout << "SM-IT " << step << " change of BF (" << k << "." << l << "): " << endl;
            cout << "      l2     " << ol << " -> " << nl << ", fracL = " << nl/ol << endl;
            cout << "    energy   "  << o << "  -> " << n  << ", fracA = " << n/o << endl;
            cout << "    rel-enrg "  << o/ol << "  -> " << n/nl  << ", fracF = " << frac << endl;
          }
        });
      }

      pTAP = newPTAP;
      pPT = PT;
    }

  }

  if (false)
  {
    // auto updateProlVals = [&](bool const exchangeCols, auto updateVals)
    // {
    //   resetDone();

    //   auto handleGroupsInEQCs = [&](FlatArray<int> eqcs, bool progress)
    //   {
    //     for (auto eqc : eqcs)
    //     {
    //       // cout << " update prol-vals in eqc " << eqc << endl;

    //       size_t cnt = 0;
    //       int perc = -100;

    //       auto nodes = FM.template GetENodes<NT_VERTEX>(eqc);

    //       // cout << " " << nodes.Size() << " vertices in EQC " << endl;

    //       for (auto fvnr : nodes)
    //       {
    //         // cout << "    check " << fvnr << " -> " << vmap[fvnr] << ", group# = " << toGroup[fvnr] << ", done = " << int(done.Test(fvnr)) << endl;

    //         if (done.Test(fvnr))
    //           { continue; }

    //         HeapReset hr(lh);

    //         auto groupNum = toGroup[fvnr];
    //         auto group    = groups[groupNum];
    //         auto cols     = sprol->GetRowIndices(group[0]);

    //         updateVals(groupNum);

    //         for (auto k : Range(group))
    //         {
    //           auto const fvK = group[k];
    //           done.SetBit(fvK);
    //         }

    //         if (progress)
    //         {
    //           cnt += group.Size();

    //           size_t new_perc = (100 * (cnt)) / nodes.Size();

    //           if (new_perc > perc + 5)
    //           {
    //             perc = new_perc;
    //             if ( perc > 0 )
    //             {
    //               cout << "  smooth eqc " << eqc << " is ~" << perc << "% done." << endl;
    //             }
    //           }
    //         }
    //       }
    //     }
    //   };

    //   // update groups with master-ex members
    //   handleGroupsInEQCs(eqExMaster, printProgressDetail);

    //   // fill into exchange-buffers, start exchange
    //   FM.template ApplyEQ2<NT_VERTEX>(Range(1ul, neqcs), [&](auto eqc, auto nodes)
    //   {
    //     auto ex_rv_row = ex_rvs[eqc];

    //     size_t off = 0;

    //     for (auto fVNr : nodes)
    //     {
    //       // auto ris = CSP.GetRowIndices(fVNr);
    //       auto rvs = CSP.GetRowValues(fVNr);

    //       for (auto j : Range(rvs))
    //       {
    //         ex_rv_row[off++] = rvs[j];
    //       }
    //     }
    //   }, true); // master-only

    //   auto reqs = eqc_h.ScatterEQCData(ex_rvs);

    //   // update local groups
    //   tfill.Start();
    //   handleGroupsInEQCs(eqLoc, printProgressDetail);
    //   tfill.Stop();

    //   // finish exchange
    //   MyMPI_WaitAll(reqs);

    //   // update ex-rows with received vals

    //   // fill in received cols/vals
    //   FM.template ApplyEQ2<NT_VERTEX>(Range(1ul, neqcs), [&](auto eqc, auto nodes)
    //   {
    //     if ( !eqc_h.IsMasterOfEQC(eqc) )
    //     {
    //       auto ex_rv_row = ex_rvs[eqc];

    //       size_t off = 0;

    //       for (auto fvnr : nodes)
    //       {
    //         auto rvs = CSP.GetRowValues(fvnr);

    //         for (auto j : Range(rvs))
    //           { rvs[j] = ex_rv_row[off++]; }
    //       }
    //     }
    //   }, false); // everyone
    // };

    // unsigned const smoothingSteps = 1 + extraSmoothingSteps;


    // // initialize sprol with PWP-vals
    // //  TODO: PARALLEL! col-exchange!
    // for (auto k : Range(FNV))
    // {
    //   auto const CV = vmap[k];

    //   if ( CV != -1 )
    //   {
    //     auto ris = sprol->GetRowIndices(k);
    //     auto rvs = sprol->GetRowValues(k);

    //     int const pos = find_in_sorted_array(CV, ris);

    //     if (pos == -1)
    //     {
    //       cout << " ERR - CV NOT FOUND: " << k << " -> " << vmap[k] << " IN "; prow(ris); cout << endl;
    //     }

    //     rvs = .0;
    //     rvs[pos] = pWP.GetRowValues(k)[0];
    //   }
    // }

    // for (auto step : Range(smoothingSteps))
    // {
    //   if (printProgress)
    //   {
    //     cout << "Group-wise prolongation smoothing, step " << step << "/" << smoothingSteps << endl;
    //   }

    //   if (step == 1) // A * sP is denser than A *pWP
    //   {
    //     // cout << " pAP = MatMultAB " << endl;
    //     pAP = MatMultAB(A, CSP);
    //   }
    //   else if (step > 1) // no further changes to sparsity
    //   {
    //     // cout << " MatMultABUpdateVals " << endl;
    //     MatMultABUpdateVals(A, CSP, *pAP);
    //   }

    //   // cout << " pAP : " << pAP << endl;

    //   auto const &AP = *pAP;

    //   // cout << " AP: " << endl;
    //   // print_tm_spmat(cout, AP); cout << endl << endl;

    //   // cols do not change, we only need to exchange them the first time!
    //   // bool const exchangeCols = step == 0;

    //   // cols do not change, they MUST come out consistently from CreateEmptyGroupWiseSProl!!
    //   constexpr bool exchangeCols = false;

    //   updateProlVals(exchangeCols, [&](auto groupNum)
    //   {
    //     // cout << " smooth group " << groupNum << "/" << groups.Size() << endl;

    //     auto group = groups[groupNum];

    //     // cout << " smooth group " << groupNum << ": "; prow(group); cout << endl;

    //     auto ris = CSP.GetRowIndices(group[0]);

    //     // cout << " ris: "; prow(ris); cout << endl;

    //     // Dirichlet, root or just nothing to do
    //     if (ris.Size() < 2)
    //       { return; }

    //     // cout << " smooth group " << groupNum << ": " << endl; //prow(group); cout << endl;
    //     // for (auto k : Range(group))
    //     // {
    //     //   auto [eq, locnr] = FM.template MapENodeToEQLNR<NT_VERTEX>(group[k]);
    //     //   cout << " mem " << k << " = " << group[k] << ", " << locnr << " in eq " << eq << " = EQID " << eqc_h.GetEQCID(eq) << " dps "; prow(eqc_h.GetDistantProcs(eq)); cout << endl;
    //     // }

    //     // cout << " ris: "; prow(ris); cout << endl;
    //     // for (auto k : Range(ris))
    //     // {
    //     //   auto [eq, locnr] = CM.template MapENodeToEQLNR<NT_VERTEX>(ris[k]);
    //     //   cout << "  RI " << k << " = " << ris[k] << ", " << locnr << " in eq " << eq << " = EQID " << eqc_h.GetEQCID(eq) << " dps "; prow(eqc_h.GetDistantProcs(eq)); cout << endl;
    //     // }


    //     ReSmoothGroupNonExpansive<ENERGY>(group,
    //                                       vmap,
    //                                       omega,
    //                                       FM,
    //                                       CM,
    //                                       CSP,
    //                                       &A,
    //                                       &AP,
    //                                       [&](auto fv) -> FlatArray<int>
    //                                       {
    //                                         return A.GetRowIndices(fv);
    //                                       },
    //                                       ris,
    //                                       [&](FlatMatrix<double> scalVals)
    //                                       {
    //                                         for (auto l : Range(group))
    //                                         {
    //                                           auto const mem = group[l];
    //                                           auto       rvs = CSP.GetRowValues(mem);
    //                                           for (auto j : Range(rvs))
    //                                           {
    //                                             addToTM(rvs[j], 1.0, scalVals, l * BS, j * BS);
    //                                           }
    //                                         }
    //                                       },
    //                                       lh,
    //                                       1e-6,
    //                                       rscale,
    //                                       (step > 0) );

    //     // cout << " smooth group " << groupNum << " OK!" << endl;
    //   });
    // }
  }

  if ( options->log_level == Options::LOG_LEVEL::DBG ) {
    // cout << " sprol: " << endl;
    std::ofstream of("ngs_amg_group_wise_SP_r_" + std::to_string(fcap.uDofs.GetCommunicator().Rank()) +
                                          "_l_" + std::to_string(fcap.baselevel) + ".out");
    print_tm_spmat(of, *sprol);
  }

  // cout << " make prol-map " << endl;
  auto prolMap = make_shared<ProlMap<TM>>(sprol, fcap.uDofs, ccap.uDofs);

  return prolMap;
} // GroupWiseSProl

} // namespace amg

#endif // WITH_GW_PROL
#endif // FILE_AMG_GW_PROL_IMPL