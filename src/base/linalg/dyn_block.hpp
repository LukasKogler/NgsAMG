#ifndef FILE_AMG_DYN_BLOCK_HPP
#define FILE_AMG_DYN_BLOCK_HPP

#include <base.hpp>
#include <sparsematrix.hpp>
#include <universal_dofs.hpp>

namespace amg
{

template<class TSCAL> class HybridDynBlockSmoother;

template<class ATOFF = size_t, class ATIND = unsigned>
class DynVectorBlocking
{
public:
  using TOFF = ATOFF;
  using TIND = ATIND;
  
  DynVectorBlocking ();

  DynVectorBlocking (TOFF        const  &numBlocks,
                     TOFF        const  &numScals, 
                     Array<TIND>       &&offSets);

  DynVectorBlocking (DynVectorBlocking<TOFF, TIND> const &other);

  void operator= (DynVectorBlocking<TOFF, TIND> const &other);

  ~DynVectorBlocking () = default;

  INLINE ATOFF Size ()     const { return _size; };
  INLINE ATOFF ScalSize () const { return _scalSize; };

  INLINE IntRange GetScalNums (TIND const &blockNum) const
  {
    return IntRange(_offsets[blockNum], _offsets[blockNum + 1]);
  }

  INLINE TIND GetScalSize (TIND const &blockNum) const
  {
    return _offsets[blockNum + 1] - _offsets[blockNum];
  }

  INLINE TIND GetRepScalRow(TIND const &blockNum) const
  {
    return _offsets[blockNum];
  }
 
  FlatArray<TIND> GetOffsets ()    { return _offsets; }

public:
  template<class TLAM>
  INLINE int
  ItScalNums (FlatArray<TIND> blockNums,
              TLAM            lam) const
  {
    int c = 0;

    for (auto j : Range(blockNums))
    {
      auto const blockNum = blockNums[j];

      auto const scalFirst = _offsets[blockNum];
      auto const scalNext  = _offsets[blockNum + 1];

      auto const nScal = scalNext - scalFirst;

      for (auto l : Range(nScal))
      {
        lam(c + l, scalFirst + l);
      }
      c += nScal;
    }

    return c;
  } // ItScalNums


  template<class TLAM>
  INLINE int
  ItScalNums (TIND            const &blockNum,
              TLAM                   lam) const
  {
    auto const scalFirst = _offsets[blockNum];
    auto const scalNext  = _offsets[blockNum + 1];

    auto const nScal = scalNext - scalFirst;

    for (auto l : Range(nScal))
    {
      lam(l, scalFirst + l);
    }

    return nScal;
  } // ItScalNums
  
  Array<MemoryUsage> GetMemoryUsage() const;
private:
  TOFF _size;
  TOFF _scalSize;

  shared_ptr<Array<TIND>> _offsetData;
  FlatArray<TIND> _offsets;
}; // class DynVectorBlocking


template<class ATSCAL, class ATOFF = size_t, class ATIND = unsigned>
class DynBlockSparseMatrix : public BaseMatrix
{
public:
  using TSCAL = ATSCAL;
  using TOFF  = ATOFF;
  using TIND  = ATIND;

  friend class HybridDynBlockSmoother<TSCAL>;

  DynBlockSparseMatrix(SparseMatrix<TSCAL>           const &A,
                       DynVectorBlocking<TOFF, TIND>       *colBlocking = nullptr);

  DynBlockSparseMatrix(SparseMatrix<TSCAL>           const  &A,
                       DynVectorBlocking<TOFF, TIND>       &&rowBlocking,
                       DynVectorBlocking<TOFF, TIND>       &&colBlocking,
                       bool                          const  &symmetric);

  DynBlockSparseMatrix(SparseMatrix<TSCAL>           const &A,
                       DynVectorBlocking<TOFF, TIND> const &rowBlocking,
                       DynVectorBlocking<TOFF, TIND> const &colBlocking,
                       bool                          const &symmetric);

  DynBlockSparseMatrix(SparseMatrix<TSCAL>           const  &A,
                       DynVectorBlocking<TOFF, TIND>       &&rowBlocking);

  DynBlockSparseMatrix(SparseMatrix<TSCAL>           const &A,
                       DynVectorBlocking<TOFF, TIND> const &rowBlocking);

  DynBlockSparseMatrix(DynVectorBlocking<TOFF, TIND>       &&rowBlocking,
                       DynVectorBlocking<TOFF, TIND>       &&colBlocking,
                       bool                          const  &symmetric,
                       FlatArray<TIND>                       rowBlockRowCnts,
                       FlatArray<TIND>                       rowBlockColCnts);

  virtual ~DynBlockSparseMatrix() = default;

  int VHeight() const override;
  int VWidth()  const override;

  void Mult             (const BaseVector &x, BaseVector &y)            const override;
  void MultTrans        (const BaseVector &x, BaseVector &y)            const override;
  void MultAdd          (double  s, const BaseVector &x, BaseVector &y) const override;
  void MultAdd          (Complex s, const BaseVector &x, BaseVector &y) const override;
  void MultTransAdd     (double  s, const BaseVector &x, BaseVector &y) const override;
  void MultTransAdd     (Complex s, const BaseVector &x, BaseVector &y) const override;
  void MultConjTransAdd (Complex s, const BaseVector &x, BaseVector &y) const override;

  AutoVector CreateRowVector () const override;
  AutoVector CreateColVector () const override;

  Array<MemoryUsage> GetMemoryUsage () const override;

  TOFF   GetNumBlockRows () const { return _rowBlocking.Size(); }
  TOFF   GetNumScalRows ()  const { return _rowBlocking.ScalSize(); }
  TOFF   GetNumBlockCols () const { return _colBlocking.Size(); }
  TOFF   GetNumScalCols ()  const { return _colBlocking.ScalSize(); }
  TOFF   GetNumBlocks ()    const { return _numRowBlocks; }
  size_t GetNZE ()          const { return _data.Size(); }

  TIND   GetMaxScalRowsInRowBlock() const { return _maxScalRowsInRow; }
  TIND   GetMaxScalColsInRowBlock() const { return _maxScalColsInRow; }

  DynVectorBlocking<TOFF, TIND> const& GetRowBlocking () const { return _rowBlocking; }
  DynVectorBlocking<TOFF, TIND> const& GetColBlocking () const { return _colBlocking; }

  void PrintTo(ostream &os) const;

  INLINE IntRange GetScalRows (TIND rowNum) const
  {
    return _rowBlocking.GetScalNums(rowNum);
  }

  INLINE TIND GetRowBlockSize (TIND rowNum) const
  {
    return _rowBlocking.GetScalSize(rowNum);
  }

  INLINE IntRange GetScalCols (TIND colNum) const
  {
    return _colBlocking.GetScalNums(colNum);
  }

  INLINE TIND GetColBlockSize (TIND colNum) const
  {
    return _colBlocking.GetScalSize(colNum);
  }

  INLINE FlatArray<TIND> GetBlockRows (TIND kBlock) const
  {
    return _rows.Range(_rowOffsets[kBlock], _rowOffsets[kBlock+1]);
  }

  INLINE FlatArray<TIND> GetBlockCols (TIND kBlock) const
  {
    return _cols.Range(_colOffsets[kBlock], _colOffsets[kBlock+1]);
  }

  INLINE TOFF GetScalNZE (TIND kBlock) const
  {
    return _dataOffsets[kBlock + 1] - _dataOffsets[kBlock];
  }

  INLINE TSCAL       *GetBlockData (TIND kBlock)
  {
    return _data.Data() + _dataOffsets[kBlock];
  }

  INLINE TSCAL const *GetBlockData (TIND kBlock) const
  {
    return _data.Data() + _dataOffsets[kBlock];
  }

public:
  template<class TLAM>
  INLINE int
  ItCols(FlatArray<TIND> blockCols,
         TLAM            lam) const
  {
    return _colBlocking.ItScalNums(blockCols, lam);
  } // ItCols

  template<class TLAM>
  INLINE int
  ItCols(TIND const &blockCol,
         TLAM        lam) const
  {
    return _colBlocking.ItScalNums(blockCol, lam);
  } // ItCols

  template<class TLAM>
  INLINE int
  ItRows(FlatArray<TIND> blockRows,
         TLAM            lam) const
  {
    return _rowBlocking.ItScalNums(blockRows, lam);
  } // ItRows

  template<class TLAM>
  INLINE int
  ItRows(TIND const &blockRow,
         TLAM        lam) const
  {
    return _rowBlocking.ItScalNums(blockRow, lam);
  } // ItRows

  INLINE
  std::tuple<FlatArray<TIND>,
             FlatArray<TIND>,
             FlatMatrix<TSCAL>>
  GetBlockRCV(TIND            const &kBlock,
              FlatArray<TIND>        scalRowBuffer,
              FlatArray<TIND>        scalColBuffer);

  INLINE
  std::tuple<FlatArray<TIND>,
             FlatArray<TIND>,
             FlatMatrix<TSCAL const>>
  GetBlockRCV(TIND            const &kBlock,
              FlatArray<TIND>        scalRowBuffer,
              FlatArray<TIND>        scalColBuffer) const;

protected:
  template<class TLAM>
  INLINE void
  MultiplyBlock(TIND              const &kBlock,
                FlatVector<TSCAL> const &x,
                FlatVector<TSCAL>       &y,
                TLAM                     lamOut) const;

  template<class TLAM>
  INLINE void
  MultiplyInternal(FlatVector<TSCAL> const &x,
                   FlatVector<TSCAL>       &y,
                   TLAM                     lamOut) const
  {
    for (auto kRow : Range(_numRowBlocks))
    {
      MultiplyBlock(kRow, x, y, lamOut);
    }
  }


private:
  void Finalize(SparseMatrix<TSCAL> const &A);

  // input/row-vector
  DynVectorBlocking<TOFF, TIND> _rowBlocking;

  // output/col-vector
  DynVectorBlocking<TOFF, TIND> _colBlocking;

  // data
  bool                     _symmetric;
  TOFF                     _numRowBlocks;
  TOFF                     _numScalNZE;
  Array<TOFF>              _rowOffsets;
  Array<TIND>              _rows;
  Array<TOFF>              _colOffsets;
  Array<TIND>              _cols;
  Array<TOFF>              _dataOffsets;
  Array<TSCAL>             _data;

  TIND _maxScalColsInRow;
  Vector<TSCAL>   _colBuffer;

  TIND _maxScalRowsInRow;
  Vector<TSCAL>    _rowBuffer;
}; // class DynBlockSparseMatrix


shared_ptr<DynBlockSparseMatrix<double>>
ConvertToDynSPM(BaseMatrix const &A);

template<class TSCAL>
shared_ptr<DynBlockSparseMatrix<TSCAL>>
MatMultAB (DynBlockSparseMatrix<TSCAL> const &A,
           DynBlockSparseMatrix<TSCAL> const &B);

extern template class DynBlockSparseMatrix<double>;

} // namespace amg

#endif // FILE_AMG_DYN_BLOCK_HPP