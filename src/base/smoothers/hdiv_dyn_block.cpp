#include "hdiv_dyn_block.hpp"

namespace amg
{

template<class TSCAL>
HDivDynBlockSmoother<TSCAL>::
HDivDynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>  aA,
                     shared_ptr<SparseMatrix<TSCAL>>          B,
                     shared_ptr<SparseMatrix<TSCAL>>          BT,
                     shared_ptr<BitArray>                     freeDofs,
                     int                                      numLocSteps,
                     bool                                     commInThread,
                     bool                                     overlapComm,
                     double                                   uRel)
  : BaseSmoother(aA)
  , _A(aA)
  , _omega(uRel)
{
  auto const &A = *_A; // getDynA();

  auto const numRowBlocks = A.GetNumBlocks();


  for (auto kBlock : Range(numRowBlocks))
  {
    // for all scal. rows of block, get cols of BT
    // get all rows of B where we have cols of BT
    // block is:
    //   kBLock of A   | rows kBlock/BT-cols
    //   B-rows        | 0
    // total size of the block is (scal rows/cols of A in kBlock,)
  }
}


template<class TSCAL>
INLINE void
HDivDynBlockSmoother<TSCAL>::
updateBlockRHS (unsigned          const &kBlock,
                FlatVector<TSCAL>       &x,
                FlatVector<TSCAL> const &b) const
{
  FlatVector<TSCAL> res  = _rowBuffer0.Range();
  FlatVector<TSCAL> resU = res.Range();
  FlatVector<TSCAL> resP = res.Range();

  FlatMatrix<TSCAL const> A_ff(nScalRows, nScalCols, A.GetBlockData(kBlock));
  FlatMatrix<TSCAL const> B_fc(nNCells, nSCalRows, GetFaceB)



  // compute res_u = omega * ( b - A_kBlock u )
  resU = b.Range();
  
  resU -= rowBlock * u;

  // compute res_p = 0 - B_kblock u   // NOT: not scaled w. omega, but should we?
  resP = 


  // compute up_u, up_b

  // update x -> x + x_up
}


} // namespace amg