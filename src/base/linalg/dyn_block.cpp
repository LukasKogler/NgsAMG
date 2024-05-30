
#include "dyn_block.hpp"
#include "dyn_block_impl.hpp"

#include "utils_sparseLA.hpp"
#include <cstddef>

#include <utils_arrays_tables.hpp>

namespace amg
{

INLINE bool
lexiLess (FlatArray<int> a,
          FlatArray<int> b)
{
  auto const nA = a.Size();
  auto const nB = b.Size();

  if (nA == nB)
  {
    for (auto l : Range(nA))
    {
      if (a[l] < b[l])
      {
        return true;
      }
      else if (a[l] > b[l])
      {
        return false;
      }
    }
    return false;
  }
  else
  {
    return nA < nB;
  }
} // lexiLess


/** DynVectorBlocking **/

template<class TOFF, class TIND>
DynVectorBlocking<TOFF, TIND>::
DynVectorBlocking()
  : _size(0)
  , _scalSize(0)
{

} // DynVectorBlocking(..)


template<class TOFF, class TIND>
DynVectorBlocking<TOFF, TIND>::
DynVectorBlocking (TOFF        const  &numBlocks,
                   TOFF        const  &numScals,
                   Array<TIND>       &&offSets)
  : _size(numBlocks)
  , _scalSize(numScals)
  , _offsetData(make_shared<Array<TIND>>())
{
  *_offsetData = std::move(offSets);
  _offsets.Assign(_offsetData->Range(0, _size + 1));
} // DynVectorBlocking(..)


template<class TOFF, class TIND>
DynVectorBlocking<TOFF, TIND>::
DynVectorBlocking(DynVectorBlocking<TOFF, TIND> const &other)
  : _size(other._size)
  , _scalSize(other._scalSize)
  , _offsetData(other._offsetData)
{
  _offsets.Assign(_offsetData->Range(0, _size + 1));
} // DynVectorBlocking(..)


template<class TOFF, class TIND>
void
DynVectorBlocking<TOFF, TIND>::
operator= (DynVectorBlocking<TOFF, TIND> const &other)
{
  _size       = other._size;
  _scalSize   = other._scalSize;
  _offsetData = other._offsetData;

  _offsets.Assign(_offsetData->Range(0, _size + 1));
} // DynVectorBlocking::operator=


template<class TOFF, class TIND>
Array<MemoryUsage>
DynVectorBlocking<TOFF, TIND>::
GetMemoryUsage() const
{
  size_t nB = _offsetData->Size() * sizeof(TIND);

  return { { {"DynVectorBlocking"}, nB, 1} };
} // DynVectorBlocking::GetMemoryUsage

/** END DynVectorBlocking **/


/** DynBlockSparseMatrix **/

template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix (SparseMatrix<TSCAL>           const &A,
                      DynVectorBlocking<TOFF, TIND>       *colBlocking)
  : BaseMatrix()
  , _rowBlocking()
  , _colBlocking()
  , _symmetric(true)
{
  // find consecutive chunks of rows with identical cols
  TOFF const nRowsScal = A.Height();

  Array<TIND> rowOffsets(nRowsScal + 1);
  // _rowVecBS.SetSize(_nRowsScal);

  TIND cntBlockRows = 0;
  TIND cInBlock = 0;
  rowOffsets[0]  = 0;

  if (nRowsScal > 0)
  {
    cntBlockRows = 1;
    cInBlock = 1;

    for (auto k : Range(1ul, nRowsScal))
    {
      if (A.GetRowIndices(k - 1) !=
          A.GetRowIndices(k))
      {
        rowOffsets[cntBlockRows++]   = k;
        cInBlock = 0;
      }
      cInBlock++;
    }
    rowOffsets[cntBlockRows]     = nRowsScal;
  }

  _rowBlocking = DynVectorBlocking<TOFF, TIND>(cntBlockRows,
                                               nRowsScal,
                                               std::move(rowOffsets));

  if (colBlocking == nullptr)
  {
    _colBlocking = DynVectorBlocking<TOFF, TIND>(_rowBlocking);
  }
  else
  {
    // I hate it...
    _colBlocking = *colBlocking;
    _symmetric = false;
  }

  Finalize(A);
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix (SparseMatrix<TSCAL>           const  &A,
                      DynVectorBlocking<TOFF, TIND>       &&rowBlocking,
                      DynVectorBlocking<TOFF, TIND>       &&colBlocking,
                      bool                          const  &symmetric)
  : BaseMatrix()
  , _rowBlocking(std::move(rowBlocking))
  , _colBlocking(std::move(colBlocking))
  , _symmetric(symmetric)
{
  Finalize(A);
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix (SparseMatrix<TSCAL>           const &A,
                      DynVectorBlocking<TOFF, TIND> const &rowBlocking,
                      DynVectorBlocking<TOFF, TIND> const &colBlocking,
                      bool                          const &symmetric)
  : BaseMatrix()
  , _rowBlocking(rowBlocking)
  , _colBlocking(colBlocking)
  , _symmetric(symmetric)
{
  Finalize(A);
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix (SparseMatrix<TSCAL>            const  &A,
                      DynVectorBlocking<TOFF, TIND>        &&rowBlocking)
  : BaseMatrix()
  , _rowBlocking(std::move(rowBlocking))
  , _colBlocking()
  , _symmetric(true)
{
  _colBlocking = _rowBlocking;

  Finalize(A);
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix (SparseMatrix<TSCAL>           const &A,
                      DynVectorBlocking<TOFF, TIND> const &rowBlocking)
  : BaseMatrix()
  , _rowBlocking(rowBlocking)
  , _colBlocking()
  , _symmetric(true)
{
  _colBlocking = _rowBlocking;

  Finalize(A);
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
DynBlockSparseMatrix(DynVectorBlocking<TOFF, TIND>       &&rowBlocking,
                     DynVectorBlocking<TOFF, TIND>       &&colBlocking,
                     bool                          const  &symmetric,
                     FlatArray<TIND>                       rowBlockRowCnts,
                     FlatArray<TIND>                       rowBlockColCnts)
  : BaseMatrix()
  , _rowBlocking(rowBlocking)
  , _colBlocking(colBlocking)
  , _symmetric(symmetric)
{
  _numRowBlocks = rowBlockRowCnts.Size();

  _rowOffsets.SetSize(_numRowBlocks + 1);
  _colOffsets.SetSize(_numRowBlocks + 1);
  _dataOffsets.SetSize(_numRowBlocks + 1);

  _rowOffsets[0]  = 0;
  _colOffsets[0]  = 0;
  _dataOffsets[0] = 0;

  size_t nR = 0;
  size_t nC = 0;
  size_t nV = 0;

  _maxScalRowsInRow = 0;
  _maxScalColsInRow = 0;

  for (auto k : Range(_numRowBlocks))
  {
    _rowOffsets[k + 1]  = _rowOffsets[k]  + rowBlockRowCnts[k];
    _colOffsets[k + 1]  = _colOffsets[k]  + rowBlockColCnts[k];    
    _dataOffsets[k + 1] = _dataOffsets[k] + rowBlockRowCnts[k] * rowBlockColCnts[k];    

    nR += rowBlockRowCnts[k];
    nC += rowBlockColCnts[k];
    nV += rowBlockRowCnts[k] * rowBlockColCnts[k];

    _maxScalRowsInRow = max(_maxScalRowsInRow, rowBlockRowCnts[k]);
    _maxScalColsInRow = max(_maxScalColsInRow, rowBlockColCnts[k]);
  }

  _rows.SetSize(nR);
  _cols.SetSize(nC);
  _data.SetSize(nV);

  _numScalNZE = nV;

  _rowBuffer.SetSize(max(_maxScalRowsInRow, 64u));
  _colBuffer.SetSize(max(_maxScalColsInRow, 64u));
} // DynBlockSparseMatrix(..)


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
Finalize (SparseMatrix<TSCAL> const &A)
{
  static Timer t("DynBlockSparseMatrix - Finalize");
  RegionTimer rt(t);

  Timer tTemp("Finalize tmp timer");
  tTemp.Start();

  /** TODO: remove empty block(s) **/

  // column scal->block mapping is needed quite heavily here
  Array<TIND> scal2Block(GetNumScalCols());

  for (auto blockCol : Range(GetNumBlockCols()))
  {
    ItCols(blockCol, [&](auto l, auto scalCol)
    {
      scal2Block[scalCol] = blockCol;
    });
  }

  // sort block-rows lexicographically by block-graph

  auto const nRows     = GetNumBlockRows();
  auto const nRowsScal = GetNumScalRows();

  _rows.SetSize(nRows);

  for (auto k : Range(nRows))
  {
    _rows[k] = k;
  }

  QuickSort(_rows, [&](auto i, auto j)
  {
    return lexiLess(A.GetRowIndices(_rowBlocking.GetRepScalRow(i)),
                    A.GetRowIndices(_rowBlocking.GetRepScalRow(j)));
  });

  // cout << " sorted rows : " << endl << _rows << endl;;

  TOFF maxColSize = 256;

  if ( nRowsScal )
  {
    maxColSize += A.GetRowIndices(0).Size();
  }

  if ( nRows > 0 )
  {
    for (auto k : Range(TOFF(1), nRows))
    {
      auto const bRowKM = _rows[k-1];
      auto const bRowK  = _rows[k];

      auto const scalRowKM = _rowBlocking.GetRepScalRow(bRowKM);
      auto const scalRowK  = _rowBlocking.GetRepScalRow(bRowK);

      if ( A.GetRowIndices(scalRowKM) !=
          A.GetRowIndices(scalRowK) )
      {
        maxColSize += A.GetRowIndices(scalRowK).Size();
      }
    }
  }

  // block-row chunking
  _colOffsets.SetSize(nRows + 1);
  _rowOffsets.SetSize(nRows + 1);
  _dataOffsets.SetSize(nRows + 1);
  // _numScalRows.SetSize(_nRows);
  // _numScalCols.SetSize(_nRows);

  _colOffsets[0] = 0;
  _rowOffsets[0] = 0;
  _dataOffsets[0] = 0;

  auto ARowOffsets = A.GetFirstArray();
  FlatVector<TSCAL> AData = A.AsVector().FVDouble();

  _numScalNZE = AData.Size();

  _data.SetSize(_numScalNZE);
  _cols.SetSize(maxColSize); // probably over-estimation, but only need 1 loop this way

  TIND cntRowBlocks     = 0;
  _maxScalColsInRow = 0;
  _maxScalRowsInRow = 0;
  TOFF dataOff          = 0;
  TOFF colOff           = 0;

  Array<int> idx(200);

  auto getColsVals = [&](auto const &kBlock, auto const &rFirst, auto const &rNext)
  {
    auto blockRows = _rows.Range(_rowOffsets[kBlock], _rowOffsets[kBlock + 1]);

    // QuickSort of _rows above is not "stable", the blockRows can be unsorted
    QuickSort(blockRows);

    // cout << " blockRows: "; prow(blockRows); cout << endl;

    auto const firstBlockRow = blockRows[0];
    auto const firstScalRow  = _rowBlocking.GetRepScalRow(firstBlockRow);

    // auto scalCols = A.GetRowIndices(scalRow);
    auto scalCols = A.GetRowIndices(firstScalRow);

    // cout << " scalCols, s = " << scalCols.Size() << ": " << endl;
    // for (auto j : Range(scalCols))
    // {
    //   cout << j << ": " << scalCols[j] << " -> " << scal2Block[scalCols[j]] << endl;
    // }


    // cout << " scalRows: " << endl;
    TIND const nScalRows = ItRows(blockRows, [&](auto l, auto scalRow)
    {
      // cout << l << ": " << scalRow << " -> " << scal2Block[scalRow] << endl;
      auto rvs = A.GetRowValues(scalRow);

      for (auto ll : Range(rvs))
      {
        _data[dataOff++] = rvs[ll];
      }
    });
    // cout << endl;


    if (scalCols.Size())
    {
      TIND col = scal2Block[scalCols[0]];

      _cols[colOff++] = col;

      for (auto j : Range(1ul, scalCols.Size()))
      {
        TIND const colJ = scal2Block[scalCols[j]];

        if (col != colJ)
        {
          col = colJ;
          _cols[colOff++] = col;
        }
      }
    }

    // _flatRows[kBlock].AssignMemory(nScalRows, scalCols.Size(), dataPtr);
    _dataOffsets[kBlock + 1] = dataOff;
    _colOffsets[kBlock + 1] = colOff;

    // cout << " blockCols: "; prow(_cols.Range(_colOffsets[kBlock], _colOffsets[kBlock+1])); cout << endl;

    // _numScalRows[kBlock] = nScalRows;
    _maxScalRowsInRow = max(_maxScalRowsInRow, nScalRows);

    // _numScalCols[kBlock] = scalCols.Size();
    _maxScalColsInRow = max(_maxScalColsInRow, TIND(scalCols.Size()));

    // FlatMatrix<TSCAL> rowVals(nScalRows, scalCols.Size(), _data.Data() + _dataOffsets[kBlock]);
    // cout << " rowBlock: " << endl << rowVals << endl;
  };


  // cout << " block-row blocking! " << endl;
  if ( nRows > 0 )
  {
    for (auto k : Range(TOFF(1), nRows))
    {
      auto const bRowKM = _rows[k-1];
      auto const bRowK  = _rows[k];

      auto const scalRowKM = _rowBlocking.GetRepScalRow(bRowKM);
      auto const scalRowK  = _rowBlocking.GetRepScalRow(bRowK);

      if ( A.GetRowIndices(scalRowKM) !=
          A.GetRowIndices(scalRowK) )
      {
        // cout << "  RIS DIFF " << k-1 << " " << k << endl;
        // cout << "    scal ris blockRow " << k-1 << " -> scal rows [" << scalRowKM << " " << scalRowK   << "): "; prow(A.GetRowIndices(scalRowKM)); cout << endl;
        // cout << "    scal ris blockRow " << k   << " -> scal rows [" << scalRowK   << " " << _rowVecOffsets[k+1] << "): "; prow(A.GetRowIndices(scalRowK));     cout << endl;

        // row-block [prev_offset, k) is row-block "cntRowBlocks" a complete row-block
        auto const rowBFirst = _rowOffsets[cntRowBlocks];
        auto const rowBNext  = (_rowOffsets[cntRowBlocks + 1]   ) = k;

        getColsVals(cntRowBlocks, rowBFirst, rowBNext);

        cntRowBlocks++;
      }
    }
    _rowOffsets[cntRowBlocks + 1] = nRows;
  }



  if (nRowsScal)
  {
    getColsVals(cntRowBlocks, _rowOffsets[cntRowBlocks], nRows);
    cntRowBlocks++;
  }

  _numRowBlocks = cntRowBlocks;

  _colOffsets.SetSize(cntRowBlocks + 1);
  _rowOffsets.SetSize(cntRowBlocks + 1);
  // _flatRows.SetSize(cntRowBlocks);
  _rowBuffer.SetSize(_maxScalRowsInRow);
  _colBuffer.SetSize(_maxScalColsInRow);

  _cols.SetSize(_colOffsets.Last());

  // cout << " count (block-)row blocks: " << cntRowBlocks << endl;
  // for (auto k : Range(cntRowBlocks))
  // {
  //   cout << " blockRow-Block " << k << "/" << cntRowBlocks << ", s = " << _rowOffsets[k+1] - _rowOffsets[k] << ", off [" << _rowOffsets[k] << "," << _rowOffsets[k+1] << ")" << endl;
  // }

  tTemp.Stop();

  cout << " Convertex to DynBlockSparseMatrix " << endl;
  cout << "   compression vec-rows : " << nRowsScal << " -> " << nRows << endl;
  cout << "   compression vec-cols : " << GetNumScalCols() << " -> " << GetNumBlockCols() << endl;
  cout << "   compression mat-rows : " << nRowsScal << " -> " << _numRowBlocks << endl;
  cout << "   compression RIS      : " << A.AsVector().Size() << " -> " << _colOffsets[_numRowBlocks] << endl;
  cout << "   time to convert      : " << tTemp.GetTime() << endl;
} // DynBlockSparseMatrix::Finalize


// copied here from header and TIND added to template params,
// move to header with the next proper rebuild!
template<class TIND, class FUNC>
INLINE void MergeArrays12 (FlatArray<TIND*> ptrs,
        FlatArray<TIND> sizes,
        // FlatArray<int> minvals,
        FUNC f)
{
  STACK_ARRAY(TIND, minvals, sizes.Size());
  TIND nactive = 0;
  for (auto i : sizes.Range())
    if (sizes[i])
      {
        nactive++;
        minvals[i] = *ptrs[i];
      }
    else
      minvals[i] = numeric_limits<TIND>::max();
  while (nactive)
    {
      TIND minval = minvals[0];
      for (size_t i = 1; i < sizes.Size(); i++)
        minval = min2(minval, minvals[i]);
      f(minval);
      for (TIND i : sizes.Range())
        if (minvals[i] == minval)
          {
            ptrs[i]++;
            sizes[i]--;
            if (sizes[i] == 0)
              {
                nactive--;
                minvals[i] = numeric_limits<TIND>::max();
              }
            else
              minvals[i] = *ptrs[i];
          }
    }
} // MergeArrays1

/** Taken from ngsolve/linalg/sparsematrix.cpp (was not in any ngsolve-header) **/
template <class TIND, typename FUNC>
INLINE void MergeArrays2 (FlatArray<TIND*> ptrs,
                          FlatArray<TIND> sizes,
                          FUNC f)
{
  if (ptrs.Size() <= 16)
    {
      MergeArrays12 (ptrs, sizes, f);
      return;
    }
  struct valsrc { TIND val, src; };
  struct trange { TIND idx, val;  };
  TIND nactive = 0;
  TIND nrange = 0;
  ArrayMem<valsrc,1024> minvals(sizes.Size());
  ArrayMem<trange,1024> ranges(sizes.Size()+1);
  constexpr TIND nhash = 1024; // power of 2
  TIND hashes[nhash];
  for (TIND i = 0; i < nhash; i++)
    hashes[i] = -1;
  for (auto i : sizes.Range())
    while (sizes[i])
      {
        auto val = *ptrs[i];
        sizes[i]--;
        ptrs[i]++;
        if (hashes[val&(nhash-1)] == val) continue;
        minvals[nactive].val = val;
        hashes[val&(nhash-1)] = val;
        minvals[nactive].src = i;
        nactive++;
        break;
      }
  while (nactive > 0)
    {
      int lower = 0;
      if (nrange > 0) lower = ranges[nrange-1].idx;
      while (true)
        {
          int firstval = minvals[lower].val;
          int otherval = firstval;
          for (int i = lower+1; i < nactive; i++)
            {
              if (minvals[i].val != firstval)
                {
                  otherval = minvals[i].val;
                  break;
                }
            }
          if (firstval == otherval)
            { // all values in last range are the same -> presorting commplete
              if (nrange == 0)
                {
                  ranges[0].idx = 0;
                  ranges[0].val = firstval;
                  nrange = 1;
                }
              break;
            }
          TIND midval = (firstval+otherval)/2;
          TIND l = lower, r = nactive-1;
          while (l <= r)
            {
              while (minvals[l].val > midval) l++;
              while (minvals[r].val <= midval) r--;
              if (l < r)
                Swap (minvals[l++], minvals[r--]);
            }
          ranges[nrange].idx = l;
          ranges[nrange].val = midval;
          nrange++;
          lower = l;
        }
      nrange--;
      TIND last = ranges[nrange].idx;
      f(minvals[last].val);
      // insert new values
      FlatArray<valsrc> tmp(nactive-last, &minvals[last]);
      nactive = last;
      for (valsrc vs : tmp)
        while (sizes[vs.src])
          {
            vs.val = *ptrs[vs.src];
            sizes[vs.src]--;
            ptrs[vs.src]++;
            // take next value if already in queue
            if (hashes[vs.val&(nhash-1)] == vs.val) continue;
            TIND prevpos = nactive;
            for (int i = nrange-1; i >= 0; i--)
              {
                if (vs.val <= ranges[i].val)
                  break;
                TIND pos = ranges[i].idx;
                minvals[prevpos] = minvals[pos];
                prevpos = pos;
                ranges[i].idx++;
              }
            minvals[prevpos] = vs;
            hashes[vs.val&(nhash-1)] = vs.val;
            nactive++;
            break;
          }
    }
} // MergeArrays2

template<class TSCAL>
shared_ptr<DynBlockSparseMatrix<TSCAL>>
MatMultAB (DynBlockSparseMatrix<TSCAL> const &A,
           DynBlockSparseMatrix<TSCAL> const &B)
{
  throw Exception("MatMultAB for dyn-sparse is untested!");

  using TIND = typename DynBlockSparseMatrix<TSCAL>::TIND;

  /** TUNTESTED, (so probably broken) **/
  auto const &blockingL = A.GetRowBlocking();
  auto const &blockingM = A.GetColBlocking();
  auto const &blockingR = B.GetColBlocking();

  auto const numRowBlocks = A.GetNumBlocks();

  Array<int> mid2BBlock(blockingM.Size());
  Array<int> mid2BBlockIdx(blockingM.Size());
  Array<int> mid2BBlockRoff(blockingM.Size());
  mid2BBlock = -1;
  mid2BBlockIdx = -1;
  mid2BBlockRoff = -1;

  for(auto kBlock : Range(B.GetNumBlocks()))
  {
    auto rows = B.GetBlockRows(kBlock);

    int cnt = 0;
    for (auto j : Range(rows))
    {
      auto const row = rows[j];

      mid2BBlock[row]     = kBlock;
      mid2BBlockIdx[row]  = j;
      mid2BBlockRoff[row] = cnt;

      cnt += blockingM.GetScalSize(row);
    }
  }

  Array<TIND> rowBlockRowCnts(A.GetNumBlocks());
  Array<TIND> rowBlockColCnts(A.GetNumBlocks());

  rowBlockRowCnts = 0;
  rowBlockColCnts = 0;

  // count cols
  ParallelForRange(numRowBlocks, [&] (IntRange r)
  {
    Array<TIND*> ptrs;
    Array<TIND> sizes;

    for (int kBlock : r)
    {
      auto mata_ci = A.GetBlockCols(kBlock);

      ptrs.SetSize(mata_ci.Size());
      sizes.SetSize(mata_ci.Size());

      for (int j : Range(mata_ci))
      {
        auto bBlock = mid2BBlock[mata_ci[j]];
        if (bBlock != -1)
        {
          auto bCols = B.GetBlockCols(bBlock);
          ptrs[j]  = bCols.Data();
          sizes[j] = bCols.Size();
        }
        else
        {
          ptrs[j] = nullptr;
          sizes[j] = 0;
        }
      }
      int cnti = 0;
      MergeArrays2(ptrs, sizes, [&cnti] (int col) { cnti++; } );

      rowBlockColCnts[kBlock] = cnti;

      rowBlockRowCnts[kBlock] = A.GetBlockRows(kBlock).Size();
    }
  }, TasksPerThread(10));

  // auto prod = make_shared<SparseMat<A, C>>(cnt, matb.Width());
  // prod->AsVector() = 0.0;

  auto prod = make_shared<DynBlockSparseMatrix<TSCAL>>(DynVectorBlocking<>(blockingL),
                                                       DynVectorBlocking<>(blockingR),
                                                       false,
                                                       rowBlockRowCnts,
                                                       rowBlockColCnts);

  // fill cols
  ParallelForRange(numRowBlocks, [&] (IntRange r)
  {
    Array<TIND*> ptrs;
    Array<TIND> sizes;

    for (int kBlock : r)
    {
      auto mata_ci = A.GetBlockCols(kBlock);

      ptrs.SetSize(mata_ci.Size());
      sizes.SetSize(mata_ci.Size());

      for (int j : Range(mata_ci))
      {
        auto bBlock = mid2BBlock[mata_ci[j]];
        if (bBlock != -1)
        {
          auto bCols = B.GetBlockCols(bBlock);
          ptrs[j]  = bCols.Data();
          sizes[j] = bCols.Size();
        }
        else
        {
          ptrs[j] = nullptr;
          sizes[j] = 0;
        }
      }

      auto prodCols = prod->GetBlockCols(kBlock);

      auto *ptr = prodCols.Data();

      MergeArrays2(ptrs, sizes, [&ptr] (int col)
                  {
                    *ptr = col;
                    ptr++;
                  } );

      prod->GetBlockRows(kBlock) = A.GetBlockRows(kBlock);
    }
  }, TasksPerThread(10));


  // fill vals

  TIND maxBS = max(A.GetMaxScalRowsInRowBlock(),
                     A.GetMaxScalColsInRowBlock());
  maxBS = max(maxBS, max(B.GetMaxScalRowsInRowBlock(),
                         B.GetMaxScalColsInRowBlock()));
  maxBS = max(maxBS, max(prod->GetMaxScalRowsInRowBlock(),
                         prod->GetMaxScalColsInRowBlock()));

  Array<TIND> scalBuffer0(maxBS);
  Array<TIND> scalBuffer1(maxBS);
  Array<TIND> scalBuffer2(maxBS);
  Array<TIND> scalBuffer3(maxBS);
  Array<TIND> scalBuffer4(maxBS);
  Array<TIND> scalBuffer5(maxBS);

  ParallelForRange(numRowBlocks, [&] (IntRange r)
  {
    struct thash { int idx; int pos; };

    size_t maxci = 0;
    for (auto kBlock : r)
      { maxci = max2(maxci, size_t (prod->GetBlockCols(kBlock).Size())); }

    size_t nhash = 2048;
    while (nhash < 2*maxci)
      { nhash *= 2; }

    ArrayMem<thash,2048> hash(nhash);

    Array<TIND> offC(maxBS);

    size_t nhashm1 = nhash-1;

    for (auto kBlock : r)
    {
      auto [ scalRowsC, scalColsC, valsC ] = prod->GetBlockRCV(kBlock, scalBuffer0, scalBuffer1);
      auto [ scalRowsA, scalColsA, valsA ] = A.GetBlockRCV(kBlock, scalBuffer2, scalBuffer3);

      auto colsC = prod->GetBlockCols(kBlock);
      valsC = 0;

      int cntC = 0;

      for (int k = 0; k < colsC.Size(); k++)
      {
        size_t hashval = size_t(colsC[k]) & nhashm1; // % nhash;
        hash[hashval].pos = k;
        hash[hashval].idx = colsC[k];
        offC[k] = cntC;
        cntC += blockingR.GetScalSize(colsC[k]);
      }

      auto colsA = A.GetBlockCols(kBlock);

      int cntScalA = 0;
      for (int j : Range(colsA))
      {
        auto aCol = colsA[j];
        int numScalCols = blockingM.GetScalSize(aCol);
        auto valA = valsA.Cols(cntScalA, cntScalA + numScalCols);

        auto const bBlock = mid2BBlock[aCol];

        if (bBlock != -1)
        {
          auto [ scalRowB, scalColsB, valsB ] = B.GetBlockRCV(bBlock, scalBuffer4, scalBuffer5);

          auto const bRow   = mid2BBlockIdx[aCol];
          auto const bROff  = mid2BBlockRoff[aCol];
          auto const numScalRowsB = blockingM.GetScalSize(bRow);

          auto colsB = B.GetBlockCols(bBlock);

          int cntScalB = 0;
          for (int k = 0; k < colsB.Size(); k++)
          {
            auto bCol = colsB[k];
            auto numScalColsB = blockingR.GetScalSize(bCol);
            auto valB = valsB.Rows(bROff, bROff + numScalRowsB)
                             .Cols(cntScalB, cntScalB + numScalColsB);
            
            unsigned hashval = unsigned(bCol) & nhashm1; // % nhash;
            
            unsigned pos = (hash[hashval].idx == bCol) ? hash[hashval].pos
                                                       : find_in_sorted_array(bCol, colsC);



            auto valC = valsC.Cols(offC[pos], offC[pos] + numScalColsB);

            valC += valA * valB;

            cntScalB += numScalColsB;
          }
        }

        cntScalA += numScalCols;
      }
    }
  }, TasksPerThread(10));

  return prod;
}

template
shared_ptr<DynBlockSparseMatrix<double>>
MatMultAB (DynBlockSparseMatrix<double> const &A,
           DynBlockSparseMatrix<double> const &B);


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
PrintTo(ostream &os) const
{
  os << " DynBlockSparseMatrix, #row-blocks = " << GetNumBlocks() << std::endl;

  os << "   row-blocking: " << GetNumScalRows() << " -> " << GetNumBlockRows() << endl;
  for (auto k : Range(_rowBlocking.Size()))
  {
    os << "    " << k << " -> "; prow(_rowBlocking.GetScalNums(k)); os << endl;
  }
  os << endl;

  os << "   col-blocking: " << GetNumScalCols() << " -> " << GetNumBlockCols() << endl;
  for (auto k : Range(_colBlocking.Size()))
  {
    os << "    " << k << " -> "; prow(_colBlocking.GetScalNums(k)); os << endl;
  }
  os << endl;

  Array<TIND> buffer0(GetMaxScalRowsInRowBlock() * 2);
  Array<TIND> buffer1(GetMaxScalColsInRowBlock() * 2);
  std::string off = "  ";
  for (auto kBlock : Range(GetNumBlocks()))
  {
    os << off << "block " << kBlock << "/" << GetNumBlocks() << ": " << endl;
    os << off << " block-rows: "; prow(GetBlockRows(kBlock)); os << endl;
    os << off << " block-cols: "; prow(GetBlockCols(kBlock)); os << endl;

    auto [scalRows, scalCols, blk] = GetBlockRCV(kBlock, buffer0, buffer1);

    os << off << "  scal rows: "; prow(scalRows, os); os << endl;
    os << off << "  scal cols: "; prow(scalCols, os); os << endl;
    os << "block, " << scalRows.Size() << " x " << scalCols.Size() << ": " << endl;
    if (scalRows.Size() * scalCols.Size() > 0)
      { os << blk << endl; }
  }
  os << endl;
} // DynBlockSparseMatrix::PrintTo


template<class TSCAL, class TOFF, class TIND>
int
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
VHeight() const
{
  return _rowBlocking.ScalSize();
}


template<class TSCAL, class TOFF, class TIND>
int
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
VWidth()  const
{
  return _colBlocking.ScalSize();
}


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
Mult(const BaseVector &x, BaseVector &y) const
{
  static Timer t("DynBlockSparseMatrix::Mult");
  RegionTimer rt(t);

  auto fVX = x.FV<TSCAL>();
  auto fVY = y.FV<TSCAL>();

  MultiplyInternal(fVX, fVY, [&](auto &y, auto const &x) { y = x; });
} // DynBlockSparseMatrix::Mult


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultTrans (const BaseVector &x, BaseVector &y) const
{
  throw Exception("Called DynBlockSparseMatrix::MultTrans");
} // DynBlockSparseMatrix::MultTrans


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultAdd (double  s, const BaseVector &x, BaseVector &y) const
{
  static Timer t("DynBlockSparseMatrix::MultAdd");
  RegionTimer rt(t);

  auto fVX = x.FV<TSCAL>();
  auto fVY = y.FV<TSCAL>();

  MultiplyInternal(fVX, fVY, [&](auto &y, auto const &x) { y += s * x; });
} // DynBlockSparseMatrix::MultAdd


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultAdd (Complex s, const BaseVector &x, BaseVector &y) const
{
  if constexpr(is_same<TSCAL, Complex>::value)
  {
    auto fVX = x.FV<TSCAL>();
    auto fVY = y.FV<TSCAL>();

    MultiplyInternal(fVX, fVY, [&](auto &y, auto const &x) { y += s * x; });
  }
  else
  {
    throw Exception("Called DynBlockSparseMatrix::MultAdd complex on real matrix");
  }
} // DynBlockSparseMatrix::MultAdd


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultTransAdd (double  s, const BaseVector &x, BaseVector &y) const
{
  throw Exception("Called DynBlockSparseMatrix::MultTransAdd (double)");
} // DynBlockSparseMatrix::MultTransAdd


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultTransAdd (Complex s, const BaseVector &x, BaseVector &y) const
{
  throw Exception("Called DynBlockSparseMatrix::MultTransAdd (Complex)");
} // DynBlockSparseMatrix::MultTransAdd


template<class TSCAL, class TOFF, class TIND>
void
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
MultConjTransAdd (Complex s, const BaseVector &x, BaseVector &y) const
{
  throw Exception("Called DynBlockSparseMatrix::MultConjTransAdd");
} // DynBlockSparseMatrix::MultConjTransAdd


template<class TSCAL, class TOFF, class TIND>
AutoVector
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
CreateRowVector () const
{
  return make_unique<VVector<TSCAL>>(GetNumScalCols());
} // DynBlockSparseMatrix::CreateRowVector


template<class TSCAL, class TOFF, class TIND>
AutoVector
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
CreateColVector () const
{
  return make_unique<VVector<TSCAL>>(GetNumScalRows());
} // DynBlockSparseMatrix::CreateColVector


template<class TSCAL, class TOFF, class TIND>
template<class TLAM>
INLINE void
DynBlockSparseMatrix<TSCAL, TOFF, TIND>::
MultiplyBlock(TIND              const &kBlock,
              FlatVector<TSCAL> const &x,
              FlatVector<TSCAL>       &y,
              TLAM                     lamOut) const
{
  // cout << " MultiplyBlock " << kBlock << endl;
  // FlatMatrix<TSCAL> const rowVals = _flatRows[kBlock];

  // TOFF const numScalRows = rowVals.Height();
  // TOFF const numScalCols = rowVals.Width();

  // cout << " numScalRows = " << numScalRows << endl;
  // cout << " numScalCols = " << numScalCols << endl;

  // input
  TOFF            const &colFirst = _colOffsets[kBlock];
  TOFF            const &colNext  = _colOffsets[kBlock + 1];
  FlatArray<TIND>        cols     = _cols.Range(colFirst, colNext);

  // cout << " blockCols: "; prow(cols); cout << endl;

  // cout << " get col-vals " << endl;
  // get col-vals
  // cout << " scal-cols " << endl;
  auto const nScalCols = ItCols(cols, [&](auto l, auto scalCol) { _colBuffer(l) = x(scalCol); });

  // output
  TOFF            const &rowFirst = _rowOffsets[kBlock];
  TOFF            const &rowNext  = _rowOffsets[kBlock + 1];
  FlatArray<TIND>        rows     = _rows.Range(rowFirst, rowNext);

  if (nScalCols > 0)
  {
    auto const dataFirst = _dataOffsets[kBlock];
    auto const dataNext  = _dataOffsets[kBlock + 1];
    auto const nZE       = dataNext - dataFirst;
    auto const nScalRows = nZE / nScalCols;

    auto smallX = _colBuffer.Range(0, nScalCols);
    auto smallY = _rowBuffer.Range(0, nScalRows);

    FlatMatrix<TSCAL> rowVals(nScalRows, nScalCols, _data.Data() + dataFirst);

    // cout << " dense mat-vec " << endl;
    // dense mat-vector
    smallY = rowVals * smallX;

    // set row-vals
    ItRows(rows, [&](auto l, auto scalRow) {
      // cout << l << " -> " << scalRow << endl;
      lamOut(y(scalRow), smallY(l));
      // cout << "    OK " << endl;
    });
  }
  else
  {
    ItRows(rows, [&](auto l, auto scalRow) {
      // cout << l << " -> " << scalRow << endl;
      lamOut(y(scalRow), TSCAL(0));
      // cout << "    OK " << endl;
    });
  }

  // cout << " MultiplyBlock " << kBlock <<  " OK " << endl;
} // DynBlockSparseMatrix::MultiplyBlock


template<class T>
INLINE size_t
SizeInBytes(Array<T> const &a)
{
  return a.AllocSize() * sizeof(T);
} // SizeInBytes

template<class T>
INLINE size_t
SizeInBytes(Vector<T> const &a)
{
  return a.Size() * sizeof(T);
} // SizeInBytes

template<class TSCAL, class TOFF, class TIND>
Array<MemoryUsage>
DynBlockSparseMatrix<TSCAL,TOFF,TIND>::
GetMemoryUsage () const
{
  cout << " DynBlockSparseMatrix::GetMemoryUsage" << std::endl;

  size_t baseMU = SizeInBytes(_data);

  size_t rCO = 0;
  rCO += _rowBlocking.GetMemoryUsage()[0].NBytes();

  if (!_symmetric)
  {
    rCO += _colBlocking.GetMemoryUsage()[0].NBytes();
  }

  cout << "  r/c off = " << double(rCO) / baseMU << endl;

  size_t rCB = 0;
  rCB += SizeInBytes(_colBuffer);
  rCB += SizeInBytes(_rowBuffer);

  cout << "  r/c buf = " << double(rCB) / baseMU << endl;

  size_t mR =  0;
  mR += SizeInBytes(_rowOffsets);
  mR += SizeInBytes(_rows);
  cout << "     rows = " << double(mR) / baseMU << endl;

  size_t mC =  0;
  mC += SizeInBytes(_colOffsets);
  mC += SizeInBytes(_cols);
  cout << "     cols = " << double(mC) / baseMU << endl;

  size_t mD =  0;
  mD += SizeInBytes(_data);
  cout << "     vals = " << double(mD) / baseMU << endl;

  size_t mFM =  0;
  mFM += SizeInBytes(_dataOffsets);
  cout << "    fmats = " << double(mFM) / baseMU << endl;


  cout << "     cols reduction possible = " << double(_cols.Size()) / _data.Size()  << ", achieved " << double(_cols.AllocSize()) / _data.Size() << endl;

  size_t nB = rCO + rCB + mR + mC + mD + mFM;

  cout << "   total = " << nB << ", in MB = " << nB/1024.0/1024 << endl;

  baseMU = SizeInBytes(_data) + SizeInBytes(_cols);

  cout << "       rel. to baseMU = " << double(nB) / baseMU << endl;

  return { {"DynBlockSparseMatrix", nB, 12 } };
} // DynBlockSparseMatrix::GetMemoryUsage


shared_ptr<DynBlockSparseMatrix<double>>
ConvertToDynSPM(BaseMatrix const &A)
{
  SparseMatrix<double> const *spPtr = my_dynamic_cast<SparseMatrix<double> const>(GetLocalMat(&A), "ConvertToDynSPM");

  return make_shared<DynBlockSparseMatrix<double>>(*spPtr);
} // ConvertToDynSPM

/** END DynBlockSparseMatrix **/


template class DynVectorBlocking<>;
template class DynBlockSparseMatrix<double>;

} // namespace amg