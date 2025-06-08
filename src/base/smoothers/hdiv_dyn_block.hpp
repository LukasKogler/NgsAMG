#ifndef FILE_AMG_HDIV_DYN_BLOCK_SMOOTHER_HPP
#define FILE_AMG_HDIV_DYN_BLOCK_SMOOTHER_HPP

#include <base.hpp>
#include <dyn_block.hpp>

#include "base_smoother.hpp"

namespace amg
{
    
/**
 * A smoother for incompressible HDiv. Instead of using penalized system
 * A + eps^{-1} BTB, here we have B,BT explicitly and invert small SP matrices.
 *
 * When we update a block for certain face-DOFS, we are enforcing
 *       B_{left,*} (u_f + u_up, u_n) = B_{right,*} (u_f + u_up, u_n)
 * That is, we find an update that equalizes divergence on the "left" and "right"
 * cells of that face.
 *
 * If we had p,q in the system as unknowns, we could just compute the residual
 * and enforce
 *      B u_up = -B (u_f, u_n) = res_p
 * with pressure-residual res_p. What we do here is like computing entries of
 * res_p whenever we update DOFS for a face and then immediately forgetting
 * that we have p,q and continue to work only on u.
 *
 * Does not work in parallel at the moment because we need access to all
 * neighboring DOFS u_n, i.e. all DOFs of both "left" and "right" cells,
 * which we don't have in parallel.
 *
 * The advantage of this over inverting penalized system diagonals is better
 * numerical stability. The eps^{-1} penalty introduces contributions into A that
 * are scaled very differently from the rest. Convergence tends to
 * improve as eps goes to zero but deteriorates again once a certain threshold
 * is reached.
 */
template<class TSCAL>
class HDivDynBlockSmoother : public BaseSmoother
{
public:
  HDivDynBlockSmoother(shared_ptr<DynBlockSparseMatrix<TSCAL>>  A,
                       shared_ptr<SparseMatrix<TSCAL>>          B,
                       shared_ptr<SparseMatrix<TSCAL>>          BT,
                       shared_ptr<BitArray>                     freeDofs = nullptr,
                       int                                      numLocSteps = 1,
                       bool                                     commInThread = true,
                       bool                                     overlapComm = true,
                       double                                   uRel = 1.0);

  INLINE void
  updateBlockRHS (unsigned          const &kBlock,
                  FlatVector<TSCAL>       &x,
                  FlatVector<TSCAL> const &b) const;

private:
  shared_ptr<DynBlockSparseMatrix<TSCAL>> _A;

  double _omega;

  // # of constraints (dual vertices/primal cells) for each block,
  // 1 block can contain multiple dual edges/primal faces so this can be >2 QD 
  Array<int> _numConstraints;
  
  Table<int> ;

}; // class HDivDynBlockSmoother

} // namespace amg

#endif // FILE_AMG_HDIV_DYN_BLOCK_SMOOTHER_HPP