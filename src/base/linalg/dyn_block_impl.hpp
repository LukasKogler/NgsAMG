#ifndef FILE_AMG_DYN_BLOCK_IMPL_HPP
#define FILE_AMG_DYN_BLOCK_IMPL_HPP

#include "dyn_block.hpp"

namespace amg
{

template<class TSCAL, class TOFF, class TIND>
INLINE
std::tuple<FlatArray<TIND>,
           FlatArray<TIND>,
           FlatMatrix<TSCAL>>
DynBlockSparseMatrix<TSCAL, TOFF, TIND>::
GetBlockRCV(TIND            const &kBlock,
            FlatArray<TIND>        scalRowBuffer,
            FlatArray<TIND>        scalColBuffer)
{
  auto blockRows = GetBlockRows(kBlock);
  auto blockCols = GetBlockCols(kBlock);

  auto const numScalRows = ItRows(blockRows, [&](auto l, auto scalRow) { scalRowBuffer[l] = scalRow; });

  auto const numScalCols = ItCols(blockCols, [&](auto l, auto scalCol) { scalColBuffer[l] = scalCol; });

  return std::make_tuple(scalRowBuffer.Range(0, numScalRows),
                         scalColBuffer.Range(0, numScalCols),
                         FlatMatrix<TSCAL> (numScalRows, numScalCols, GetBlockData(kBlock)));
} // DynBlockSparseMatrix::GetBlockRCV


template<class TSCAL, class TOFF, class TIND>
INLINE
std::tuple<FlatArray<TIND>,
           FlatArray<TIND>,
           FlatMatrix<TSCAL const>>
DynBlockSparseMatrix<TSCAL, TOFF, TIND>::
GetBlockRCV(TIND            const &kBlock,
            FlatArray<TIND>        scalRowBuffer,
            FlatArray<TIND>        scalColBuffer) const
{
  auto blockRows = GetBlockRows(kBlock);
  auto blockCols = GetBlockCols(kBlock);

  auto const numScalRows = ItRows(blockRows, [&](auto l, auto scalRow) { scalRowBuffer[l] = scalRow; });

  auto const numScalCols = ItCols(blockCols, [&](auto l, auto scalCol) { scalColBuffer[l] = scalCol; });

  return std::make_tuple(scalRowBuffer.Range(0, numScalRows),
                         scalColBuffer.Range(0, numScalCols),
                         FlatMatrix<TSCAL const> (numScalRows, numScalCols, GetBlockData(kBlock)));
} // DynBlockSparseMatrix::GetBlockRCV

} //namespace amg

#endif // FILE_AMG_DYN_BLOCK_IMPL_HPP