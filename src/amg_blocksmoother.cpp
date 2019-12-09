#ifndef FILE_AMG_BS_CPP
#define FILE_AMG_BS_CPP

#include "amg.hpp"

namespace amg
{
    
  /** BSmoother **/

  template<class TM>
  BSmoother<TM> :: BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks, FlatArray<TM> md)
    : spmat(_spmat), blocks(_blocks)
  {
    const auto & A(*spmat);
    size_t n_blocks = blocks.Size();

    /** Diag buffer **/
    size_t totmem = 
      ParallelReduce (n_blocks,
                      [&] (size_t i) { return sqr (blocks[i].Size()); },
                      [] (size_t a, size_t b) { return a+b; },
                      size_t(0));
    buffer.SetSize(totmem);
    dinv.SetSize(n_blocks);
    totmem = 0;
    maxbs = 0;
    for (auto k : Range(n_blocks)) {
      size_t h = blocks[k].Size();
      dinv[k] = FlatMatrix<TM> (h, h, buffer.Addr(totmem));
      totmem += sqr(h);
      maxbs = max(maxbs, h);
    }

    /** Get and invert diagonal blocks **/
    SharedLoop2 sl(n_blocks);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 for (int block_nr : sl) {
	   auto block_dofs = blocks[block_nr];
	   auto D = dinv[block_nr];
	   for (int i : Range(block_dofs)) {
	     for (int j : Range(block_dofs))
	       { D(i,j) = A(block_dofs[i], block_dofs[j]); }
	     if (md.Size())
	       { D(i,i) = md[block_dofs[i]]; }
	   }
	   CalcInverse(D);
	 }
       } );

    /** Do block coloring here for shm parallelization **/
    Table<int> ext_blocks;
    { /** For dependencies of blocks, because of resiudal upates, we also need one layer around each block! **/
      TableCreator<int> ceb(n_blocks);
      for (auto block_nr : Range(n_blocks)) {
	auto block_dofs = blocks[block_nr];
	ceb.Add(block_nr, block_dofs);
	/** I do not care about multiples **/
	for (auto dof : block_dofs)
	  for (auto col : A.GetRowIndices(dof))
	    { ceb.Add(block_nr, col); }
      }
      ext_blocks = ceb.MoveTable();
    }
    Array<int> bcarray(n_blocks);
    ComputeColoring( bcarray, A.Height(), [&](auto i) { return ext_blocks[i]; });
    TableCreator<int> cbc(n_blocks);
    for (; !cbc.Done(); cbc++) {
      for (auto k : Range(bcarray))
	{ cbc.Add(bcarray[k], k); }
    }
    block_colors = cbc.MoveTable();
    // loops.SetSize(block_colors.Size());
  } // BSmoother::BSmoother(..)

  template<class TM> template<class TLAM>
  void BSmoother<TM> :: Smooth_impl (BaseVector & x, const BaseVector & b, TLAM get_col, int steps) const
  {
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    const auto & A(*spmat);
    Array<SharedLoop2> loops(block_colors.Size());
    for (int step_nr : Range(steps)) {
      for (auto k : Range(loops))
	{ loops[k].Reset(block_colors[k].Range()); }
      task_manager -> CreateJob
	( [&] (const TaskInfo & ti) 
	  {
	    VectorMem<100,TV> hxmax(maxbs);
	    VectorMem<100,TV> hymax(maxbs);
	    for (int J : Range(block_colors))
	      {
		int color = get_col(J);
		for (auto block_nr : loops[color]) {
		  auto block_dofs = blocks[block_nr];
		  int bs = block_dofs.Size();
		  FlatVector<TV> hup = hxmax.Range(0,bs);
		  FlatVector<TV> hr = hymax.Range(0,bs);
		  for (int j = 0; j < bs; j++) {
		    auto jj = block_dofs[j];
		    hr(j) = fb(jj) - A.RowTimesVector(jj, fx);
		  }
		  hup = dinv[block_nr] * hr;
		  fx(block_dofs) += hup;
		}
	      }
	  });
    }
  } //BSmoother::Smooth_impl

  template<class TM>
  void BSmoother<TM> :: Smooth (BaseVector & x, const BaseVector & b, int steps) const
  {
    Smooth_impl(x, b, [&](auto i) { return i; } , steps);
  } //BSmoother::Smooth


  template<class TM>
  void BSmoother<TM> :: SmoothBack (BaseVector & x, const BaseVector & b, int steps) const
  {
    const auto maxcol = block_colors.Size() - 1;
    Smooth_impl(x, b, [&](auto i) { return maxcol - i; } , steps);
  } //BSmoother::SmoothBack


  template<class TM> template<class TLAM>
  INLINE void BSmoother<TM> :: SmoothRES_impl (BaseVector & x, BaseVector & res, TLAM get_col, int steps) const
  {
    auto fx = x.FV<TV>();
    auto fres = res.FV<TV>();
    const auto & A(*spmat);
    Array<SharedLoop2> loops(block_colors.Size());
    for (int step_nr : Range(steps)) {
      for (auto k : Range(loops))
	{ loops[k].Reset(block_colors[k].Range()); }
      task_manager -> CreateJob
	( [&] (const TaskInfo & ti) 
	  {
	    VectorMem<100,TV> hxmax(maxbs);
	    VectorMem<100,TV> hymax(maxbs);
	    for (int J : Range(block_colors))
	      {
		int color = get_col(J);
		for (auto block_nr : loops[color]) {
		  auto block_dofs = blocks[block_nr];
		  int bs = block_dofs.Size();
		  FlatVector<TV> hup = hxmax.Range(0,bs);
		  FlatVector<TV> hr = hymax.Range(0,bs);
		  hr = fres(block_dofs);
		  hup = dinv[block_nr] * hr;
		  for (int j = 0; j < bs; j++)
		    { A.AddRowTransToVector(block_dofs[j], hup(j), fres); }
		  fx(block_dofs) += hup;
		}
	      }
	  });
    }
  }


  template<class TM>
  void BSmoother<TM> :: SmoothRES (BaseVector & x, BaseVector & res, int steps) const
  {
    SmoothRES_impl(x, res, [&](auto i) { return i; } , steps);
  } //BSmoother::SmoothRES


  template<class TM>
  void BSmoother<TM> :: SmoothBackRES (BaseVector & x, BaseVector & res, int steps) const
  {
    const auto maxcol = block_colors.Size() - 1;
    SmoothRES_impl(x, res, [&](auto i) { return maxcol - i; } , steps);
  } //BSmoother::SmoothBackRES

  /** END BSmoother **/


  /** HybridBS **/

  template<class TM>
  HybridBS<TM> :: HybridBS (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h,
			      Table<int> && blocks, bool _overlap, bool _in_thread)
    : HybridSmoother2<TM>(_A, eqc_h, _overlap, _in_thread)
  {
    Array<Table<int>> fblocks;
    if (eqc_h->GetCommunicator().Size() > 2)
      { fblocks = FilterBlocks(move(blocks)); }
    else
      { fblocks.SetSize(3); fblocks[0] = move(blocks); }
    loc_smoothers.SetSize(3);
    Array<TM> mod_diag = this->CalcModDiag(nullptr);
    if(mod_diag.Size()) {
      for (auto k : Range(loc_smoothers))
	{ loc_smoothers[k] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[k]), mod_diag); }
    }
    else {
      for (auto k : Range(loc_smoothers))
	{ loc_smoothers[k] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[k])); }
    }
  } // HybridBS::HybridBS(..)


  template<class TM>
  Array<Table<int>> HybridBS<TM> :: FilterBlocks (Table<int> && _blocks)
  {
    const auto & pds = *A->GetParallelDofs();

    Table<int> blocks = move(_blocks);
    auto n_blocks = blocks.Size();
    int maxbs = 100;
    Array<int> fb(maxbs);

    auto iterate_blocks = [&](auto lam_loc, auto lam_ex) {
      for (auto block_nr : Range(n_blocks)) {
	auto block = blocks[block_nr];
	if (block.Size() > maxbs)
	  { maxbs = 2*block.Size(); fb.SetSize(maxbs); }
	bool ismaster = true, isloc = true;
	int cnt_master = 0;
	for (auto dof : block) {
	  auto dps = pds.GetDistantProcs(dof);
	  if (dps.Size()) {
	    isloc = false;
	    if (!pds.IsMasterDof(dof))
	      { ismaster = false; break; }
	    fb[cnt_master++] = dof;
	  }
	}
	if (cnt_master > 0) {
	  fb.SetSize(cnt_master);
	  if (isloc)
	    { lam_loc(fb); }
	  else if (ismaster)
	    { lam_ex(fb); }
	}
      } // Range(n_blocks)
    };
    
    int cnt_loc = 0, cnt_ex = 0;
    iterate_blocks([&](auto & fb) { cnt_loc++; },
		   [&](auto & fb) { cnt_ex++; });
    int nl1 = cnt_loc/2, nl2 = cnt_loc - nl1;
    TableCreator<int> cl1(nl1), cl2(nl2), cex(cnt_ex);
    int cnt_loc1 = 0, cnt_loc2 = 0; cnt_ex = 0;
    for (; !cex.Done(); cl1++, cl2++, cex++)
      iterate_blocks(
		     [&](auto & fb) {
		       if (++cnt_loc1 < nl1)
			 { cl1.Add(cnt_loc1, fb); }
		       else
			 { cl2.Add(cnt_loc2++, fb); }
		     },
		     [&](auto & fb) {
		       cex.Add(cnt_ex++, fb);
		     } );
    
    Array<Table<int>> fblocks(3);
    fblocks[0] = cl1.MoveTable();
    fblocks[1] = cex.MoveTable();
    fblocks[2] = cl2.MoveTable();

    return fblocks;
  } // HybridBS::FilterBlocks


  template<class TM>
  Array<MemoryUsage> HybridBS<TM> :: GetMemoryUsage() const
  {
    // TODO
    Array<MemoryUsage> memu;
    // Array<MemoryUsage> memu = HybridSmoother2<TM>::GetMemoryUsage();
    // for (auto& ls : locsmoothers)
      // { memu.Append(ls->GetMemoryUsage()); }
    return memu;
  } // BSmoother::GetMemoryUsage


  template<class TM>
  void HybridBS<TM> :: SmoothLocal (int stage, BaseVector &x, const BaseVector &b) const
  {
    loc_smoothers[stage]->Smooth(x, b);
  } // HybridBS::SmoothLocal


  template<class TM>
  void HybridBS<TM> :: SmoothBackLocal (int stage, BaseVector &x, const BaseVector &b) const
  {
    loc_smoothers[stage]->SmoothBack(x, b);
  } // HybridBS::SmoothBackRESLocal


  template<class TM>
  void HybridBS<TM> :: SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    loc_smoothers[stage]->SmoothRES(x, res);
  } // HybridBS::SmoothRESLocal


  template<class TM>
  void HybridBS<TM> :: SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    loc_smoothers[stage]->SmoothBackRES(x, res);
  } // HybridBS::SmoothBackRESLocal


  /** END HybridBS **/

} // namespace amg

#include "amg_tcs.hpp"

#endif
