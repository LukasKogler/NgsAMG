#ifndef FILE_AMG_REPL
#define FILE_AMG_REPL

namespace amg {

  shared_ptr<SparseMatrix<double>> AssBlockRepl_wip
  (int BS, const shared_ptr<const BaseAlgebraicMesh> & mesh,
   const shared_ptr<const AMGOptions> & opts,
   std::function<void(size_t vi, size_t vj, FlatMatrix<double>&)> get_block,
   std::function<bool(size_t vi, size_t vj)> filter_block,
   bool smooth, double rho);

  
  /** 
      Assembles the replatement matrix (edge-contributions)
        (bool) smooth: instead of Ahat, assemble I-rho D^-1 Ahat (to use for smoothing)
	(lambda) filter_block(v, edge) -> bool   /// false -> filter out the block for this edge!
	   rows without blocks: 0 if not smooth, identity if smooth
        (lambda) get_block: get_block(l, edge, mat) -> void /// write repl-block for edge into mat [l is for orientation]
   **/
  template<int BS>
  shared_ptr<SparseMatrix<double>> AssBlockRepl
  (const shared_ptr<const BaseAlgebraicMesh> & mesh,
   const shared_ptr<const AMGOptions> & opts,
   std::function<void(size_t vi, size_t vj, FlatMatrix<double>&)> get_block,
   std::function<bool(size_t vi, size_t vj)> filter_block,
   bool smooth = true, double rho = 0.0)
  {
    return move(AssBlockRepl_wip(BS, mesh, opts, get_block, filter_block, smooth, rho));
  }

  
} // end namespace amg


#endif
