
#include <base.hpp>
#include <dyn_block_impl.hpp>
#include <ng_lapack.hpp>
#include <utils.hpp>
#include <utils_arrays_tables.hpp>

#include "base_smoother.hpp"
#include "hybrid_smoother_utils.hpp"
#include "dyn_block_smoother.hpp"
#include "utils_io.hpp"


namespace amg
{

/** DynBlockSmoother **/

template<class TSCAL>
DynBlockSmoother<TSCAL>::
DynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>        A,
                 shared_ptr<BitArray>                           freeDofs,
                 double                                  const &uRel)
  : BaseSmoother(A)
  , _A(A)
  , _omega(uRel)
{
  SetUp(freeDofs.get());
} // DynBlockSmoother(..)


template<class TSCAL>
DynBlockSmoother<TSCAL>::
DynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>        A,
                 FlatArray<TSCAL>                               modDiag,
                 shared_ptr<BitArray>                           freeDofs,
                 double                                  const &uRel)
  : BaseSmoother(A)
  , _A(A)
  , _omega(uRel)
{
  SetUp(freeDofs.get(), modDiag);
} // DynBlockSmoother(..)


template<class TSCAL>
void
DynBlockSmoother<TSCAL>::
SetUp(BitArray const *freeDofs, FlatArray<TSCAL> modDiag)
{
  // cout << " DynBlockSmoother::SetUp" << endl;
  auto const &A = getDynA();

  auto const numRowBlocks = A.GetNumBlocks();

  // cout << " numRowBlocks = " << numRowBlocks << endl;
  // cout << " freeDofs: " << freeDofs << endl;

  _scalBlockSize.SetSize(numRowBlocks);
  _dataOff.SetSize(numRowBlocks + 1);

  _dataOff[0] = 0;

  if (freeDofs != nullptr)
  {
    _freeBlocks = make_shared<BitArray>(numRowBlocks);
    _freeBlocks->Clear();
  }
  else
  {
    _freeBlocks = nullptr;
  }

  for (auto kBlock : Range(numRowBlocks))
  {
    auto blockRows = A.GetBlockRows(kBlock);
    auto blockCols = A.GetBlockCols(kBlock);

    // cout << " blockRows: "; prow(blockRows); cout << endl;
    auto const numScalRows = std::accumulate(blockRows.begin(),
                                             blockRows.end(),
                                             0u,
                                             [&](auto const &pSum, auto const &idx) { return pSum + A.GetRowBlockSize(idx); });

    // // cout << " blockCols: "; prow(blockCols); cout << endl;
    auto const numScalCols = blockCols.Size() ?
                               std::accumulate(blockCols.begin(),
                                               blockCols.end(),
                                               0u,
                                               [&](auto const &pSum, auto const &idx) { return pSum + A.GetColBlockSize(idx); }) :
                               0u;

    // cout << " block " << kBlock << "/" << numRowBlocks << ": " << numScalRows << " x " << numScalCols << endl;

    if ( numScalCols > 0 )
    {
      _scalBlockSize[kBlock]   = numScalRows;
      _dataOff[kBlock + 1] = _dataOff[kBlock] + numScalRows * numScalRows;
    }
    else
    {
      _scalBlockSize[kBlock] = 0;
      _dataOff[kBlock + 1]   = _dataOff[kBlock];
    }
  }

  // cout << " maxRows = " << maxRows<< endl;
  // cout << " maxCols = " << maxCols<< endl;

  _data.SetSize(_dataOff.Last());

  _rowBuffer0.SetSize(A.GetMaxScalRowsInRowBlock());
  _rowBuffer1.SetSize(A.GetMaxScalRowsInRowBlock());
  _colBuffer0.SetSize(A.GetMaxScalColsInRowBlock());

  Array<unsigned> scalRowBuffer(A.GetMaxScalRowsInRowBlock());
  Array<unsigned> scalColBuffer(A.GetMaxScalColsInRowBlock());
  Array<unsigned> diagPosBuffer(A.GetMaxScalRowsInRowBlock());

  for (auto kBlock : Range(numRowBlocks))
  {
    // cout << " block " << kBlock << "/" << numRowBlocks << ": " << endl;

    // auto blockRows = A.GetBlockRows(kBlock);
    // auto blockCols = A.GetBlockCols(kBlock);

    // auto const numScalRows = A.ItRows(blockRows, [&](auto l, auto scalRow) { scalRowBuffer[l] = scalRow; });
    // auto const numScalCols = A.ItCols(blockCols, [&](auto l, auto scalCol) { scalColBuffer[l] = scalCol; });

    // cout << " block " << kBlock << "/" << numRowBlocks << ": " << numScalRows << " x " << numScalCols << endl;

    if (_dataOff[kBlock + 1] > _dataOff[kBlock]) // avoid the iterate_intersection for empty rows
    {
      auto [scalRows, scalCols, rowBlock] = A.GetBlockRCV(kBlock,
                                                          scalRowBuffer,
                                                          scalColBuffer);

      // cout << " block " << kBlock << "/" << numRowBlocks << ": " << scalRows.Size() << " x " << scalCols.Size() << endl;

      // auto scalRows = scalRowBuffer.Range(0, numScalRows);
      // auto scalCols = scalColBuffer.Range(0, numScalCols);
      auto diagPos  = diagPosBuffer.Range(0, scalRows.Size());

      iterate_intersection(scalRows, scalCols, [&](auto const &idxR, auto const &idxC) {
        diagPos[idxR] = idxC;
      });

      // cout << " scalRows "; prow2(scalRows); cout << endl;
      // cout << " scalCols "; prow2(scalCols); cout << endl;
      // cout << " diagPos: "; prow2(diagPos); cout << endl;

      // FlatMatrix<TSCAL const> rowBlock(numScalRows, numScalCols, A.GetBlockData(kBlock));

      auto diagBlock = getDiagBlock(kBlock);

      for (auto l : Range(scalRows))
      {
        for (auto j : Range(diagPos))
        {
          diagBlock(l, j) = rowBlock(l, diagPos[j]);
        }
        if (modDiag.Size())
        {
          // cout << " USE MOD-DIAG " << scalRows[l] << ": " << diagBlock(l,l) << " -> " << modDiag[scalRows[l]] << endl;
          diagBlock(l, l) = modDiag[scalRows[l]];
        }
      }

      // cout << " diagBlock " << kBlock << ": " << endl << diagBlock << endl;

      if ( freeDofs != nullptr )
      {
        bool anyFree = false;
        bool allFree = true;

        for (auto l : Range(scalRows))
        {
          auto const scalRow = scalRows[l];

          if ( freeDofs->Test(scalRow) )
          {
            anyFree = true;
          }
          else
          {
            allFree = false;

            diagBlock.Row(l) = 0;
            diagBlock.Col(l) = 0;
            diagBlock(l,l)   = 1.0;
          }
        }

        if ( anyFree )
        {
          // cout << " FREE! " << endl;

          _freeBlocks->SetBit(kBlock);
  
          CalcInverse(diagBlock);

          if (!allFree)
          {
            for (auto l : Range(scalRows))
            {
              auto const scalRow = scalRows[l];

              if ( !freeDofs->Test(scalRow) )
              {
                diagBlock(l,l) = 0.0;
              }
            }
          }
        }
        // else
        // {
        //   cout << " NOT FREE! " << endl;
        // }
      }
      else
      {
        CalcInverse(diagBlock);
      }

      // cout << " INV diagBlock: " << endl << diagBlock << endl;
    }
  }

} // DynBlockSmoother::SetUp


template<class TSCAL>
INLINE void
DynBlockSmoother<TSCAL>::
updateBlockRHS (unsigned          const &kBlock,
                FlatVector<TSCAL>       &x,
                FlatVector<TSCAL> const &b) const
{
  // Note: timers cause real slowdown here!
  auto const &A = getDynA();

  auto nScalCols = A.ItCols(A.GetBlockCols(kBlock), [&](auto l, auto scalCol) {
    _colBuffer0(l) = x(scalCol);
  });

  if (nScalCols == 0)
  {
    return;
  }

  // unsigned const nScalRows = A.GetScalNZE(kBlock) / nScalCols;

  auto nScalRows = A.ItRows(A.GetBlockRows(kBlock), [&](auto l, auto scalRow) {
    _rowBuffer0(l) = b(scalRow);
  });

  auto smallR = _rowBuffer0.Range(0, nScalRows);
  auto smallX = _colBuffer0.Range(0, nScalCols);

  FlatMatrix<TSCAL const> rowBlock(nScalRows, nScalCols, A.GetBlockData(kBlock));
  // FlatMatrix<TSCAL> rowBlock(nScalRows, nScalCols, const_cast<TSCAL*>(A.GetBlockData(kBlock)));

  // Lapck is slower for mat times vec
  smallR -= rowBlock * smallX;
  // smallR -= rowBlock * smallX | Lapack;
  // LapackMultAx(rowBlock, smallX, smallR);

  // A.ItRows(A.GetBlockRows(kBlock), [&](auto l, auto scalRow) {
  //   smallR(l) = b(scalRow) - smallR(l);
  // });

  // t1.Stop();
  // t2.Start();

  auto smallUp = _rowBuffer1.Range(0, nScalRows);

  smallUp = getDiagBlock(kBlock) * smallR;

  A.ItRows(A.GetBlockRows(kBlock), [&](auto l, auto scalRow)
  {
    x(scalRow) += _omega * smallUp(l);
  });

  // t2.Stop();
} // DynBlockSmoother::updateBlockRHS


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
INLINE void
DynBlockSmoother<TSCAL>::
updateBlocksRHS (FlatArray<unsigned>      blockNums,
                 FlatVector<TSCAL>       &x,
                 FlatVector<TSCAL> const &b) const
{
  IterateBlocks<DIR>(blockNums, [&](auto const &kBlock) { updateBlockRHS(kBlock, x, b); });
} // DynBlockSmoother::updateBlocksRHS
                   

template<class TSCAL>
INLINE void
DynBlockSmoother<TSCAL>::
updateBlockRes (unsigned          const &kBlock,
                FlatVector<TSCAL> &x,
                FlatVector<TSCAL> &res) const
{
  // Note: timers cause real slowdown here!
  auto const &A = getDynA();

  auto nScalRows = A.ItRows(A.GetBlockRows(kBlock), [&](auto l, auto scalRow) {
    _rowBuffer0(l) = res(scalRow);
  });

  auto smallR  = _rowBuffer0.Range(0, nScalRows);
  auto smallUp = _rowBuffer1.Range(0, nScalRows);

  smallUp = _omega * getDiagBlock(kBlock) * smallR;

  unsigned const nScalCols = A.GetScalNZE(kBlock) / nScalRows;

  FlatMatrix<TSCAL> rowBlock(nScalRows, nScalCols, const_cast<TSCAL*>(A.GetBlockData(kBlock)));

  auto smallResUp = _colBuffer0.Range(0, nScalCols);

  // Lapck is faster for Trans-mat times vec? ACTUALLY I don't think so!
  smallResUp = Trans(rowBlock) * smallUp;
  // LapackMultAtx(rowBlock, smallUp, smallResUp);

  A.ItRows(A.GetBlockRows(kBlock), [&](auto l, auto scalRow) {
    x(scalRow) += smallUp(l);
  });

  A.ItCols(A.GetBlockCols(kBlock), [&](auto l, auto scalCol) {
    res(scalCol) -= smallResUp(l);
  });
} // DynBlockSmoother::updateBlockRes


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
INLINE void
DynBlockSmoother<TSCAL>::
updateBlocksRes (FlatArray<unsigned>      blockNums,
                 FlatVector<TSCAL>       &x,
                 FlatVector<TSCAL>       &res) const
{
  IterateBlocks<DIR>(blockNums, [&](auto const &kBlock) { updateBlockRes(kBlock, x, res); });
} // DynBlockSmoother::updateBlocksRHS
                   

template<class TSCAL>
void
DynBlockSmoother<TSCAL>::
Smooth (BaseVector &x, BaseVector const &b,
        BaseVector &res, bool res_updated,
        bool update_res, bool x_zero) const
{
  SmoothImpl<FORWARD>(x, b, res, res_updated, update_res, x_zero);
} // DynBlockSmoother::SmoothBack


template<class TSCAL>
void
DynBlockSmoother<TSCAL>::
SmoothBack (BaseVector &x, BaseVector const &b,
            BaseVector &res, bool res_updated,
            bool update_res, bool x_zero) const
{
  SmoothImpl<BACKWARD>(x, b, res, res_updated, update_res, x_zero);
} // DynBlockSmoother::SmoothBack


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
INLINE void
DynBlockSmoother<TSCAL>::
SmoothImpl (BaseVector &x, BaseVector const &b,
            BaseVector &res, bool res_updated,
            bool update_res, bool x_zero) const
{
  // static Timer tres("DynBlockSmoother - RES update");

  /**
   *  Cost in increasing order:
   *      I   RES update
   *      II  smooth with RHS
   *      III smooth with RES
   *      IV  smooth with RHS + res-update
   */
  if (update_res)
  {
    if (!res_updated)
    {
      if (!x_zero)
      {
        // [II + I] < [I + III]
        SmoothRHS<DIR>(x, b);

        res = b;
        res -= (*GetAMatrix()) * x;
      }
      else
      {
        // III < IV
        res = b;
        SmoothRes<DIR>(x, res);
      }
    }
    else
    {
      // III < IV
      SmoothRes<DIR>(x, res);
    }
  }
  else
  {
    SmoothRHS<DIR>(x, b);
  }

  // SmoothRHS<DIR>(x, b);

  // if (update_res)
  // {
  //   RegionTimer rt(tres);
  //   res = b;
  //   res -= (*GetAMatrix()) * x;
  // }

} // DynBlockSmoother::SmoothImpl


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR, class TLAM>
INLINE void
DynBlockSmoother<TSCAL>::
IterateBlocks(TLAM lam) const
{
  auto const &A = getDynA();

  size_t const nBlocks = A.GetNumBlocks();

  if constexpr(DIR == FORWARD)
  {
    for (auto kBlock : Range(nBlocks))
    {
      if (_freeBlocks == nullptr || _freeBlocks->Test(kBlock))
      {
        lam(kBlock);
      }
    }
  }
  else
  {
    for (auto k : Range(nBlocks))
    {
      auto const kBlock = nBlocks - 1 - k;

      if (_freeBlocks == nullptr || _freeBlocks->Test(kBlock))
      {
        lam(kBlock);
      }
    }
  }

} // DynBlockSmoother::IterateBlocks


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR, class TLAM>
INLINE void
DynBlockSmoother<TSCAL>::
IterateBlocks(FlatArray<unsigned> blockNums, TLAM lam) const
{
  auto const &A = getDynA();

  if constexpr(DIR == FORWARD)
  {
    for (auto kBlock : blockNums)
    {
      if (_freeBlocks == nullptr || _freeBlocks->Test(kBlock))
      {
        lam(kBlock);
      }
    }
  }
  else
  {
    size_t const nBlocks = blockNums.Size();

    for (auto k : Range(nBlocks))
    {
      auto const kBlock = blockNums[nBlocks - 1 - k];

      if (_freeBlocks == nullptr || _freeBlocks->Test(kBlock))
      {
        lam(kBlock);
      }
    }
  }

} // DynBlockSmoother::IterateBlocks



template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
INLINE void
DynBlockSmoother<TSCAL>::
SmoothRHS(BaseVector &x, BaseVector const &b) const
{
  static Timer t("DynBlockSmoother - SmoothRHS");
  RegionTimer rt(t);

  // Timer t("SRt");
  // t.Start();
  auto fX = x.FV<TSCAL>();
  auto fB = b.FV<TSCAL>();

  IterateBlocks<DIR>([&](auto kBlock) { updateBlockRHS(kBlock, fX, fB); });
  // t.Stop();
  // cout << " SmoothRHS, FW = " << bool(DIR == FORWARD) << ", t = " << t.GetTime() << endl;
} // DynBlockSmoother::SmoothRhs


template<class TSCAL>
template<SMOOTHING_DIRECTION DIR>
INLINE void
DynBlockSmoother<TSCAL>::
SmoothRes(BaseVector &x, BaseVector &res) const
{
  static Timer t("DynBlockSmoother - SmoothRes");
  RegionTimer rt(t);

  // Timer t("SRt");
  // t.Start();

  auto fX   = x.FV<TSCAL>();
  auto fRes = res.FV<TSCAL>();

  IterateBlocks<DIR>([&](auto kBlock) { updateBlockRes(kBlock, fX, fRes); });
  // t.Stop();
  // cout << " SmoothRes, FW = " << bool(DIR == FORWARD) << ", t = " << t.GetTime() << endl;
} // DynBlockSmoother::SmoothRes


/** END DynBlockSmoother **/


/** HybridDynBlockSmoother **/

extern void TestSmoother (shared_ptr<BaseSmoother> sm, NgMPI_Comm & gcomm, string message);

template<class TSCAL>
HybridDynBlockSmoother<TSCAL>::
HybridDynBlockSmoother(shared_ptr<DynamicBlockHybridMatrix<TSCAL>>  A,
                       shared_ptr<BitArray>    freeDofs,
                       int                     numLocSteps,
                       bool                    commInThread,
                       bool                    overlapComm,
                       double                  uRel)
  : HybridBaseSmoother<TSCAL>(A,
                              numLocSteps,
                              commInThread,
                              overlapComm)
  , _hybDynBlockA(A)
{
  auto const &hybM = *A->GetDynSpM();

  auto const N         = hybM.Height();
  auto const numBlocks = hybM.GetNumBlocks();

  // only check first scal-row in the block
  if (auto parDofs = this->GetParallelDofs())
  {
    _locBlockNums.SetSize(numBlocks);
    _exBlockNums.SetSize(numBlocks);

    cout << " parDofs: " << endl << *parDofs << endl;

    int cntLoc = 0;
    int cntEx = 0;
    int cntFree = 0;

    auto mDofs = A->GetDCCMap().GetMasterDOFs();

    for (auto kBlock : Range(numBlocks))
    {
      auto blockRows = hybM.GetBlockRows(kBlock);

      bool anyMFree = false;
      bool anyMEx   = false;

      hybM.template ItRows(hybM.GetBlockRows(kBlock), [&](auto l, auto scalRow)
      {
        if ( ( freeDofs == nullptr ) || freeDofs->Test(scalRow) )
        {
          if ( mDofs->Test(scalRow))
          {
            anyMFree = true;

            if ( parDofs->GetDistantProcs(scalRow).Size() )
            {
              anyMEx = true;
            }
          }
        }
      });

      if ( anyMEx )
      {
        cout << " BLOCK " << kBlock << " -> EX " << cntEx << endl;
        _exBlockNums[cntEx++] = kBlock;
      }
      else if ( anyMFree )
      {
        cout << " BLOCK " << kBlock << " -> LOC " << cntLoc << endl;
        _locBlockNums[cntLoc++] = kBlock;
      }
      else
      {
        cout << " BLOCK " << kBlock << "  -> DIRI/G! " << endl;
      }
    }

    _locBlockNums.SetSize(cntLoc);
    _exBlockNums.SetSize(cntEx);

    _splitInd = cntLoc / 2;

    // masterAndFree is needed - there can be blocks with mixed,
    // free/DIRI and local/ex rows!
    auto masterAndFree = make_shared<BitArray>(*mDofs);
    if (freeDofs)
    {
      masterAndFree->And(*freeDofs);
    }

    // cout << " M = " << A->GetDynSpM() << endl;
    // if (A->GetDynSpM()->Height() < 200)
    // A->GetDynSpM()->PrintTo(cout);

    // cout << " G = " << A->GetDynSpG() << endl;
    // if ( A->GetDynSpG() != nullptr )
    //   if (A->GetDynSpM()->Height() < 200)
    //     A->GetDynSpG()->PrintTo(cout);

    cout << " HYBRID-DYN block-splitting: " << numBlocks << " blocks, " << endl;
    cout << "    " << _locBlockNums.Size() << " local -> split @ " << _splitInd << endl;
    cout << "    " << _exBlockNums.Size()  << " ex! " << endl;
    cout << "    total free = " << cntFree << endl;

    auto modDiag = CalcScalModDiag(freeDofs);

    // _locSmoother = make_shared<DynBlockSmoother<TSCAL>>(A->GetDynSpM(), masterAndFree, uRel);
    // if (_locSmoother->GetAMatrix()->Height() > 0)
    // {
    //   NgMPI_Comm dummyComm;
    //   cout << " TEST _locSmoother - NO MD" << endl;
    //   TestSmoother(_locSmoother, dummyComm, "HYBRID-DYN smoother, _locSmoother");
    //   cout << " TEST _locSmoother - NO MD" << endl;
    // }


    _locSmoother = make_shared<DynBlockSmoother<TSCAL>>(A->GetDynSpM(), modDiag, masterAndFree, uRel);

    cout << " _locSmoother->GetAMatrix()->Height() = " << _locSmoother->GetAMatrix()->Height() << endl;
    cout << " M->Height() = " << A->GetDynSpM()->Height() << endl;

    if (cntLoc > 0)
    {
      NgMPI_Comm dummyComm;
      cout << " TEST _locSmoother" << endl;
      TestSmoother(_locSmoother, dummyComm, "HYBRID-DYN smoother, _locSmoother");
      cout << " TEST _locSmoother" << endl;
    }
  }
  else
  {
    _locSmoother = make_shared<DynBlockSmoother<TSCAL>>(A->GetDynSpM(), freeDofs, uRel);

    // all local
    _splitInd = numBlocks / 2;

    _locBlockNums.SetSize(numBlocks);

    for (auto k : Range(_locBlockNums))
    {
      _locBlockNums[k] = k;
    }
  }

} // HybridDynBlockSmoother(..)


template<class TSCAL>
HybridDynBlockSmoother<TSCAL>::
HybridDynBlockSmoother(shared_ptr<BaseMatrix>  A,
                       shared_ptr<BitArray>    freeDofs,
                       int                     numLocSteps,
                       bool                    commInThread,
                       bool                    overlapComm,
                       double                  uRel)
  : HybridDynBlockSmoother<TSCAL>(make_shared<DynamicBlockHybridMatrix<TSCAL>>(A),
                                 freeDofs,
                                 numLocSteps,
                                 commInThread,
                                 overlapComm,
                                 uRel)
{
} // HybridDynBlockSmoother(..)


template<class TM, class TIT_D, class TIT_OD>
INLINE void
CalcHybridSmootherRDGItGenericV2(size_t                                 const &N,
                               DCCMap<typename mat_traits<TM>::TSCAL> const &dCCMap,
                               Array<TM>                                    &modDiag,
                               BitArray                               const *freeDOFs,
                               TIT_D                                         iterateD,
                               TIT_OD                                        iterateOD)
{
  static Timer t("CalcHybridSmootherRDGItGeneric");
  RegionTimer rt(t);

  constexpr int BS = Height<TM>();

  auto const &parDOFs = *dCCMap.GetParallelDofs();

  auto const isMaster = *dCCMap.GetMasterDOFs();

  modDiag.SetSize0();

  /** No need for adding to the diagonal **/
  if ( ( parDOFs.GetDistantProcs().Size() == 0 ) )
    { return; }

  typedef typename mat_traits<TM>::TV_COL TV;

  Array<TM> origDiag(N);

  // have to zero this since there is no guarantee that all rows are
  // touched by iterateD (e.h. hyrbid-block only goes through M-rows)
  origDiag = 0.0;

  iterateD([&](auto const &k, auto const &diag) LAMBDA_INLINE
  {
    origDiag[k] = diag;
  });
  // for (auto k : Range(N))
  //   { origDiag[k] = getDiag(k); }

  MyAllReduceDofData (parDOFs, origDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  // cout << " reduced origDiag: " << endl << origDiag << endl;

  Array<TV> scalSqrt(N * BS);

  for (auto k : Range(N))
  {
    auto &sqrti = scalSqrt[k];

    if constexpr( is_same<TM,double>::value )
    {
      sqrti = sqrt(origDiag[k]);
    }
    else
    {
      Iterate<BS>([&](auto l) LAMBDA_INLINE { sqrti[l] = sqrt(origDiag[k](l,l)); });
    }
  }

  // cout << " scalSqrt: " << endl << scalSqrt << endl;

  // auto &addDiag = origDiag;
  Array<TV> addDiag(N);
  addDiag = 0;

  iterateOD([&](auto k, auto iterateOffDiag) LAMBDA_INLINE
  {
    if ( freeDOFs && !freeDOFs->Test(k))
      { return; }

    auto &sqrtK = scalSqrt[k];

    // cout << " calc AD row " << k << endl;

    iterateOffDiag([&](auto const &col, auto const &val)
    {
      if constexpr( is_same<TM,double>::value )
      {
        addDiag[k] += fabs(val) / ( sqrtK * scalSqrt[col] );
          // add_diag[k] += fabs(rvs[j]);
          // add_diag[k] += fabs(rvs[j]) / ( diag[ris[j]] );
      }
      else
      {
        auto &sqrtJ = scalSqrt[col];

        Iterate<BS>([&](auto l) LAMBDA_INLINE
        {
          Iterate<BS>([&](auto m) LAMBDA_INLINE
          {
            addDiag[k](l.value) += fabs(val(l.value,m.value)) / ( sqrtK[l.value] * sqrtJ[m.value] );
          });
        });
      }
    });

    // cout << "     AD row " << k << " = " << addDiag[k] << endl;
  });

  MyAllReduceDofData (parDOFs, addDiag,
    [](auto & a, const auto & b) LAMBDA_INLINE { a += b; });

  cout << " addDiag after CUMU: " << endl << addDiag << endl;

  modDiag.SetSize(N);

  // const double theta = 0.25;
  const double theta = 0.1;

  for (auto k : Range(N))
  {
    cout << " calc MOD-DG " << k << ", master = " << isMaster.Test(k)
         << ", free = " << bool( freeDOFs == nullptr || !freeDOFs->Test(k) ) << endl;
    
    if ( !isMaster.Test(k) || ( freeDOFs && !freeDOFs->Test(k) ) )
    {
      modDiag[k] = 0;
      continue;
    }

    auto const &d  = origDiag[k];
    auto const &ad = addDiag[k];
    auto       &md = modDiag[k];

    cout << "    d = " << d << ", ad = " << ad << endl;

    if constexpr( is_same<TM, double>::value )
    {
      md = max(1.0, 0.51 * (1.0 + ad)) * d;
      cout << " -> FAC = " << max(1.0, 0.51 * (1.0 + ad)) << " -> MD = " << md << endl;
      // md = d;
    }
    else
    {
      double maxfac = 1.0;

      // Iterate<TMH>([&](auto i) { maxfac = max(maxfac, 0.5 * ( 1 + theta + ad(i.value) ) ); });
      Iterate<BS>([&](auto i) LAMBDA_INLINE
      {
        maxfac = max(maxfac, 0.51 * ( 1.0 + ad(i.value) ) );
      });

      md = maxfac * d;

      // md = d;
    }
  }
} // CalcHybridSmootherRDGItGenericV2



template<class TSCAL>
Array<TSCAL>
HybridDynBlockSmoother<TSCAL>::
CalcScalModDiag (shared_ptr<BitArray> scalFree)
{
  auto const &A = this->GetDynBlockHybA();

  auto const &M = *A.GetDynSpM();
  auto ptrG = A.GetDynSpG();


  Array<TSCAL> modDiag;

  auto maxR = M.GetMaxScalRowsInRowBlock();

  auto maxC = M.GetMaxScalColsInRowBlock();

  if ( ptrG != nullptr )
  {
    maxR = max(maxR, ptrG->GetMaxScalRowsInRowBlock());
    maxC = max(maxC, ptrG->GetMaxScalColsInRowBlock());
  }

  Array<unsigned> scalRowBuffer(maxR);
  Array<unsigned> scalColBuffer(maxC);
  Array<unsigned> diagPosBuffer(maxR);

  auto iterateD = [&](auto lam)
  {
    auto const &numBlocks = M.GetNumBlocks();

    for (auto kBlock : Range(numBlocks))
    {
      auto [ sb_scalRows, sb_scalCols, sb_denseBlock ] = M.GetBlockRCV(kBlock, scalRowBuffer, scalColBuffer);

      // capture of structured binding
      auto const &scalRows   = sb_scalRows;
      auto const &scalCols   = sb_scalCols;
      auto const &denseBlock = sb_denseBlock;

      auto const nScalRows = scalRows.Size();
      auto const nScalCols = scalCols.Size();

      if ( nScalCols > 0 ) // avoid the iterate_intersection for empty rows
      {
        auto scalRows = scalRowBuffer.Range(0, nScalRows);
        auto scalCols = scalColBuffer.Range(0, nScalCols);
        auto diagPos  = diagPosBuffer.Range(0, nScalRows);

        iterate_intersection(scalRows, scalCols, [&](auto const &idxR, auto const &idxC) {
          diagPos[idxR] = idxC;
        });

        for (auto l : Range(scalRows))
        {
          // TSCAL dg = denseBlock(l, diagPos[l]);
          // cout << " DIAG row " << scalRows[l] << ": " << dg << endl;
          lam(scalRows[l], denseBlock(l, diagPos[l]));
        }
      }
    }
  };

  auto iterateOD = [&](auto lam)
  {
    if (ptrG == nullptr)
    {
      return;
    }

    auto const &G = *ptrG;

    auto const &numBlocks = G.GetNumBlocks();

    for (auto kBlock : Range(numBlocks))
    {
      auto [ sb_scalRows, sb_scalCols, sb_denseBlock ] = G.GetBlockRCV(kBlock, scalRowBuffer, scalColBuffer);

      // capture of structured binding
      auto const &scalRows   = sb_scalRows;
      auto const &scalCols   = sb_scalCols;
      auto const &denseBlock = sb_denseBlock;

      auto const nScalRows = scalRows.Size();
      auto const nScalCols = scalCols.Size();

      if ( nScalCols > 0 ) // avoid the iterate_intersection for empty rows
      {
        for (auto lR : Range(scalRows))
        {
          auto const scalRow = scalRows[lR];

          lam(scalRow, [&](auto colValKernel)
          {
            for (auto lC : Range(scalCols))
            {
              auto const scalCol = scalCols[lC];

              colValKernel(scalCol, denseBlock(lR, lC));
            }
          });

        }
      }
    }
  };

  CalcHybridSmootherRDGItGeneric<TSCAL>(A.Height(),
                                        A.GetDCCMap(),
                                        modDiag,
                                        scalFree.get(),
                                        iterateD,
                                        iterateOD);

  return modDiag;
} // HybridDynBlockSmoother::CalcScalModDiag


template<class TSCAL>
void
HybridDynBlockSmoother<TSCAL>::
SmoothStageRHS (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  static Timer t("HybridDynBlockSmoother::SmoothStageRHS");
  RegionTimer rt(t);

  auto fX = x.FV<TSCAL>();
  auto fB = b.FV<TSCAL>();

  auto const &locSmoother = *_locSmoother;

  if (direction == FORWARD)
  {
    _locSmoother->template updateBlocksRHS<FORWARD>(GetBlocksForStage(stage), fX, fB);
  }
  else
  {
    _locSmoother->template updateBlocksRHS<BACKWARD>(GetBlocksForStage(stage), fX, fB);
  }
} // HybridDynBlockSmoother::SmoothStageRHS


template<class TSCAL>
void
HybridDynBlockSmoother<TSCAL>::
SmoothStageRes (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  static Timer t("HybridDynBlockSmoother::SmoothStageRes");
  RegionTimer rt(t);

  auto fX   = x.FV<TSCAL>();
  auto fRes = res.FV<TSCAL>();

  if (direction == FORWARD)
  {
    _locSmoother->template updateBlocksRes<FORWARD>(GetBlocksForStage(stage), fX, fRes);
  }
  else
  {
    _locSmoother->template updateBlocksRes<BACKWARD>(GetBlocksForStage(stage), fX, fRes);
  }

} // HybridDynBlockSmoother::SmoothStageRes


template<class TSCAL>
INLINE FlatArray<unsigned>
HybridDynBlockSmoother<TSCAL>::
GetBlocksForStage (SMOOTH_STAGE const &stage) const
{
  switch(stage)
  {
    case(SMOOTH_STAGE::LOC_PART_1): { return _locBlockNums.Range(0, _splitInd); }
    case(SMOOTH_STAGE::EX_PART):    { return _exBlockNums; }
    case(SMOOTH_STAGE::LOC_PART_2): { return _locBlockNums.Range(_splitInd, _locBlockNums.Size()); }
  }
} // HybridDynBlockSmoother::GetBlocksForStage


/** END HybridDynBlockSmoother **/


template class DynBlockSmoother<double>;
template class HybridDynBlockSmoother<double>;

} // namespace amg
