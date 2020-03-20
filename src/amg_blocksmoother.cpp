#define FILE_AMG_BS_CPP

#include "amg_blocksmoother.hpp"

namespace amg
{
    
  /** BSmoother **/

  template<class TM>
  BSmoother<TM> :: BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks, 
			      bool _parallel, bool _use_sl2, FlatArray<TM> md)
    : spmat(_spmat), blocks(move(_blocks)), parallel(_parallel), use_sl2(_use_sl2)
  {
    static Timer t("BSmoother"); RegionTimer rt(t);
    
    const auto & A(*spmat);
    size_t n_blocks = blocks.Size();

    for (auto row : blocks)
      { QuickSort(row); }

    // cout << " BSmoother with " << n_blocks << " blocks! " << endl;
    // cout << blocks << endl;

    /** Diag buffer **/
    size_t totmem = 
      ParallelReduce (n_blocks,
                      [&] (size_t i) { return sqr (blocks[i].Size()); },
                      [] (size_t a, size_t b) { return a+b; },
                      size_t(0));
    // cout << " totmem = " << totmem << endl;
    buffer.SetSize(totmem);
    dinv.SetSize(n_blocks);
    auto otm = totmem;
    totmem = 0;
    maxbs = 0;
    for (auto k : Range(n_blocks)) {
      size_t h = blocks[k].Size();
      // dinv[k] = FlatMatrix<TM> (h, h, buffer.Addr(totmem));
      new (&dinv[k]) FlatMatrix<TM> (h, h, buffer.Addr(totmem));
      totmem += sqr(h);
      maxbs = max2(maxbs, h);
    }

    // cout << " maxbs " << maxbs << endl;
    // cout << " alloced, now inv " << endl;

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
	   // cout << " diag " << block_nr << endl << D << endl;
	   CalcInverse(D);
	   // cout << "inv diag " << block_nr << endl << D << endl;
	 }
       } );

    // cout << " have dinvs " << endl;

    if (parallel) {
      /** Do block coloring here for shm parallelization **/
      Table<int> ext_blocks;
      { /** For dependencies of blocks, because of resiudal upates, we also need one layer around each block! **/
	TableCreator<int> ceb(n_blocks);
	Array<int> dnums(20*maxbs);
	for (; !ceb.Done(); ceb++) {
	  for (auto block_nr : Range(n_blocks)) {
	    auto block_dofs = blocks[block_nr];
	    // ceb.Add(block_nr, block_dofs);
	    /** I do not care about multiples **/
	    // for (auto dof : block_dofs)
	    // for (auto col : A.GetRowIndices(dof))
	    // { ceb.Add(block_nr, col); }
	    /** Actually, I think I do **/
	    dnums.SetSize0();
	    dnums.Append(block_dofs);
	    // QuickSort(dnums); // blocks are sorted
	    int pos;
	    for (auto dof : block_dofs)
	      for (auto col : A.GetRowIndices(dof)) {
		pos = find_in_sorted_array(col, dnums);
		if (pos == -1)
		  { insert_into_sorted_array(col, dnums); }
	      }
	    ceb.Add(block_nr, dnums);
	  }
	}
	ext_blocks = ceb.MoveTable();
      }
      // cout << " ext_blocks: " << endl << ext_blocks << endl;
      Array<int> bcarray(n_blocks);

      int maxcolor = 0;
      if (n_blocks > 0) /** need coloring even vor 1 thread b.c smooth/smoothback order of blocks **/
	{ maxcolor = ComputeColoring( bcarray, A.Height(), [&](auto i) { return ext_blocks[i]; }); }

      // cout << " blocks colored with " << maxcolor << " colors" << endl;
      // cout << " blk colors: " << endl; prow2(bcarray); cout << endl;

      TableCreator<int> cbc(maxcolor+1);
      for (; !cbc.Done(); cbc++) {
	for (auto k : Range(bcarray))
	  { cbc.Add(bcarray[k], k); }
      }
      block_colors = cbc.MoveTable();

      if (use_sl2) {
	// loops.SetSize(block_colors.Size()); // no operator= with const reference for sharedloops
	loops = move(Array<SharedLoop2>(block_colors.Size()));
      }
      else {
	color_balance.SetSize(block_colors.Size());
	for (auto cnum : Range(block_colors)) {
	  color_balance[cnum].Calc(block_colors[cnum].Size(),
				   [&] (auto bi) LAMBDA_INLINE {
				     int costs = 0;
				     auto blocknr = block_colors[cnum][bi];
				     for (auto d : blocks[blocknr])
				       { costs += A.GetRowIndices(d).Size(); }
				     return costs;
				   });
	}
      }
    }
  } // BSmoother::BSmoother(..)


  template<class TM>
  BSmoother<TM> :: BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks, Table<int> && _block_ext_dofs,
			      bool _parallel, bool _use_sl2, FlatArray<TM> md)
    : spmat(_spmat), blocks(move(_blocks)), parallel(_parallel), use_sl2(_use_sl2)
  {
    static Timer t("BSmoother"); RegionTimer rt(t);

    use_sl2 = true; // TODO if this i false !!
    
    const auto & A(*spmat);
    size_t n_blocks = blocks.Size();

    Table<int> block_ext_dofs = move(_block_ext_dofs);

    for (auto row : blocks)
      { QuickSort(row); }

    for (auto row : block_ext_dofs)
      { QuickSort(row); }

    // cout << " BSmoother with " << n_blocks << " blocks! " << endl;

    // for (auto k : Range(n_blocks)) {
    //   cout << endl << " block " << k << endl;
    //   cout << " in-dofs = "; prow2(blocks[k]); cout << endl;
    //   cout << " ex-dofs = "; prow2(block_ext_dofs[k]); cout << endl;
    // }
    // cout << endl;


    /** Diag buffer **/
    size_t totmem = 
      ParallelReduce (n_blocks,
                      [&] (size_t i) { return sqr (blocks[i].Size()); },
                      [] (size_t a, size_t b) { return a+b; },
                      size_t(0));
    // cout << " totmem = " << totmem << endl;
    buffer.SetSize(totmem);
    dinv.SetSize(n_blocks);
    auto otm = totmem;
    totmem = 0;
    maxbs = 0;
    for (auto k : Range(n_blocks)) {
      size_t h = blocks[k].Size();
      // dinv[k] = FlatMatrix<TM> (h, h, buffer.Addr(totmem));
      new (&dinv[k]) FlatMatrix<TM> (h, h, buffer.Addr(totmem));
      totmem += sqr(h);
      maxbs = max2(maxbs, h);
    }

    // cout << " maxbs " << maxbs << endl;
    // cout << " alloced, now inv " << endl;

    /** Get and invert diagonal blocks **/
    LocalHeap clh (ngcore::task_manager->GetNumThreads() * 20*1024*1024, "BSMem");
    SharedLoop2 sl(n_blocks);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
	 for (int block_nr : sl) {
	   auto block_dofs = blocks[block_nr];
	   auto D = dinv[block_nr];
	   auto ext_dofs = block_ext_dofs[block_nr];
	   int nex = ext_dofs.Size(), nin = block_dofs.Size();
	   for (int i : Range(block_dofs)) {
	     for (int j : Range(block_dofs))
	       { D(i,j) = A(block_dofs[i], block_dofs[j]); }
	     if (md.Size())
	       { D(i,i) = md[block_dofs[i]]; }
	   }
	   if ( (nex > 0) && (nin > 0) ) {
	     FlatMatrix<TM> ee(nex, nex, lh), ei(nex, nin, lh), ee_ei(nex, nin, lh);
	     for (int i : Range(nex)) {
	       for (int j : Range(nex))
		 { ee(i, j) = A(ext_dofs[i], ext_dofs[j]); }
	       for (int j : Range(nin))
		 { ei(i, j) = A(ext_dofs[i], block_dofs[j]); }
	     }
	     // cout << endl << "block_nr " << block_nr << " nin " << nin << " nex " << nex << endl;
	     // cout << "ei: " << endl << ei << endl;
	     // cout << "ee: " << endl << ee << endl;
	     CalcInverse(ee);
	     // cout << "eei: " << endl << ee << endl;
	     ee_ei = ee * ei;
	     // cout << " ee_ei: " << endl << ee_ei << endl;
	     // cout << " diag " << block_nr << endl << D << endl;
	     // up = -Trans(ei) * ee_ei;
	     // cout << " update " << up << endl;
	     D *= 2;
	     D -= Trans(ei) * ee_ei;
	     // cout << "updated diag " << block_nr << endl << D << endl;
	   }
	   CalcInverse(D);
	   // cout << "inv diag " << block_nr << endl << D << endl;
	 }
       } );

    // cout << " have dinvs " << endl;

    /** Do block coloring here for shm parallelization **/
    Table<int> ext_blocks;
    { /** For dependencies of blocks, because of resiudal upates, we also need one layer around each block! **/
      TableCreator<int> ceb(n_blocks);
      Array<int> dnums(20*maxbs);
      for (; !ceb.Done(); ceb++) {
	for (auto block_nr : Range(n_blocks)) {
	  auto block_dofs = blocks[block_nr];
	  // ceb.Add(block_nr, block_dofs);
	  /** I do not care about multiples **/
	  // for (auto dof : block_dofs)
	    // for (auto col : A.GetRowIndices(dof))
	    // { ceb.Add(block_nr, col); }
	  /** Actually, I think I do **/
	  dnums.SetSize0();
	  dnums.Append(block_dofs);
	  // QuickSort(dnums); // blocks are sorted
	  int pos;
	  for (auto dof : block_dofs)
	    for (auto col : A.GetRowIndices(dof)) {
	      pos = find_in_sorted_array(col, dnums);
	      if (pos == -1)
		{ insert_into_sorted_array(col, dnums); }
	    }
	  ceb.Add(block_nr, dnums);
	}
      }
      ext_blocks = ceb.MoveTable();
    }
    // cout << " ext_blocks: " << endl << ext_blocks << endl;
    Array<int> bcarray(n_blocks);

    int maxcolor = 0;
    if (n_blocks > 0) /** need coloring even vor 1 thread b.c smooth/smoothback order of blocks **/
      { maxcolor = ComputeColoring( bcarray, A.Height(), [&](auto i) { return ext_blocks[i]; }); }

    // cout << " blocks colored with " << maxcolor << " colors" << endl;
    // cout << " blk colors: " << endl; prow2(bcarray); cout << endl;

    TableCreator<int> cbc(maxcolor+1);
    for (; !cbc.Done(); cbc++) {
      for (auto k : Range(bcarray))
	{ cbc.Add(bcarray[k], k); }
    }
    block_colors = cbc.MoveTable();

    if (parallel && use_sl2) {
      // loops.SetSize(block_colors.Size()); // no operator= with const reference for sharedloops
      loops = move(Array<SharedLoop2>(block_colors.Size()));
    }
  } // BSmoother::BSmoother(..)


  template<class TM> template<class TLAM>
  INLINE void BSmoother<TM> :: IterateBlocks (bool reverse, TLAM smooth_block) const
  {
    if (parallel) {
      if (use_sl2) {
	const int ncolors = block_colors.Size();
	for (auto k : Range(loops))
	  { loops[k].Reset(block_colors[k].Range()); }
	task_manager -> CreateJob
	  ( [&] (const TaskInfo & ti) 
	    {
	      VectorMem<100,TV> hxmax(maxbs);
	      VectorMem<100,TV> hymax(maxbs);
	      for (int J : Range(block_colors))
		{
		  int color = reverse ? ncolors-1-J : J;
		  for (auto bcind : loops[color]) {
		    auto block_nr = block_colors[color][bcind];
		    smooth_block(block_nr, hxmax, hymax);
		  }
		}
	    } );
      }
      else { // use_sl2
	const int ncolors = block_colors.Size();
	for (int J : Range(block_colors)) {
	  int color = reverse ? ncolors-1-J : J;
	  ParallelForRange(color_balance[color], [&] (IntRange r) LAMBDA_INLINE {
		VectorMem<100,TV> hxmax(maxbs);
		VectorMem<100,TV> hymax(maxbs);
		auto & cblocks = block_colors[color];
		for (auto block_nr : cblocks.Range(r)) {
		  smooth_block(block_nr, hxmax, hymax);
		}
	    });
	}
      }
    }
    else { //parallel
      VectorMem<100,TV> hxmax(maxbs);
      VectorMem<100,TV> hymax(maxbs);
      if (reverse) {
	for (int block_num = blocks.Size() - 1; block_num >= 0; block_num--)
	  { smooth_block(block_num, hxmax, hymax); }
      }
      else {
	for (auto block_num : Range(blocks.Size()))
	  { smooth_block(block_num, hxmax, hymax); }
      }
    }
  } // BSmoother::IterateBlocks


  INLINE Timer & SRHS_thack () { static Timer t("BSmoother::SmoothRHS(FW/BW)"); return t; }
  template<class TM>
  void BSmoother<TM> :: Smooth_impl (BaseVector & x, const BaseVector & b, int steps, bool reverse) const
  {
    RegionTimer rt(SRHS_thack());
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    const auto & A(*spmat);
    for (int step_nr : Range(steps)) {
      IterateBlocks(reverse, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  auto block_dofs = blocks[block_nr];
	  int bs = block_dofs.Size();
	  if ( bs > 0 ) { // avoid range check in debug mode
	    // cout << " up block " << block_nr << ", bs = " << bs << endl;
	    // prow(block_dofs); cout << endl;
	    FlatVector<TV> hup = hxmax.Range(0,bs);
	    FlatVector<TV> hr = hymax.Range(0,bs);
	    for (int j = 0; j < bs; j++) {
	      auto jj = block_dofs[j];
	      hr(j) = fb(jj) - A.RowTimesVector(jj, fx);
	    }
	    // cout << " res " << endl << hr << endl;
	    hup = dinv[block_nr] * hr;
	    // cout << " dinv: " << endl << dinv[block_nr] << endl;
	    // cout << " up " << endl << hup << endl;
	    fx(block_dofs) += hup;
	    // cout << " new x " << endl << fx(block_dofs) << endl;
	  }
	});
    }
  } //BSmoother::Smooth_impl

  template<class TM>
  void BSmoother<TM> :: Smooth (BaseVector & x, const BaseVector & b, int steps) const
  {
    Smooth_impl(x, b, steps, false);
  } //BSmoother::Smooth


  template<class TM>
  void BSmoother<TM> :: SmoothBack (BaseVector & x, const BaseVector & b, int steps) const
  {
    Smooth_impl(x, b, steps, true);
  } //BSmoother::SmoothBack


  INLINE Timer & SRES_thack () { static Timer t("BSmoother::SmoothRES(FW/BW)"); return t; }
  template<class TM>
  INLINE void BSmoother<TM> :: SmoothRES_impl (BaseVector & x, BaseVector & res, int steps, bool reverse) const
  {
    RegionTimer rt(SRES_thack());
    auto fx = x.FV<TV>();
    auto fres = res.FV<TV>();
    const auto & A(*spmat);
    for (int step_nr : Range(steps)) {
      IterateBlocks(reverse, [&](auto block_nr, auto & hxmax, auto & hymax) {
	  auto block_dofs = blocks[block_nr];
	  int bs = block_dofs.Size();
	  if ( bs > 0 ) { // avoid range check in debug mode
	    // cout << " up block " << block_nr << ", bs = " << bs << endl;
	    FlatVector<TV> hup = hxmax.Range(0,bs);
	    FlatVector<TV> hr = hymax.Range(0,bs);
	    hr = fres(block_dofs);
	    // cout << " res " << endl << hr << endl;
	    hup = dinv[block_nr] * hr;
	    // cout << " up " << endl << hup << endl;
	    for (int j = 0; j < bs; j++)
	      { A.AddRowTransToVector(block_dofs[j], -hup(j), fres); }
	    fx(block_dofs) += hup;
	    // cout << " new x " << endl << fx(block_dofs) << endl;
	  }
	});
    }
  } // BSmoother::SmoothRES_impl


  template<class TM>
  void BSmoother<TM> :: SmoothRES (BaseVector & x, BaseVector & res, int steps) const
  {
    SmoothRES_impl(x, res, steps, false);
  } //BSmoother::SmoothRES


  template<class TM>
  void BSmoother<TM> :: SmoothBackRES (BaseVector & x, BaseVector & res, int steps) const
  {
    SmoothRES_impl(x, res, steps, true);
  } //BSmoother::SmoothBackRES

  /** END BSmoother **/


  /** HybridBS **/

  template<class TM>
  HybridBS<TM> :: HybridBS (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h,
			    Table<int> && blocks, bool _overlap, bool _in_thread, bool _parallel, bool _sl2)
    : HybridSmoother2<TM>(_A, eqc_h, _overlap, _in_thread)
  {
    Array<Table<int>> fblocks;
    if (eqc_h->GetCommunicator().Size() > 2)
      { fblocks = FilterBlocks(move(blocks)); }
    else { fblocks.SetSize(3); fblocks[0] = move(blocks); }
    loc_smoothers.SetSize(3);
    Array<TM> mod_diag = this->CalcModDiag(nullptr);
    if(mod_diag.Size()) {
      for (auto k : Range(loc_smoothers))
	{ loc_smoothers[k] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[k]), _parallel, _sl2, mod_diag); }
    }
    else {
      for (auto k : Range(loc_smoothers))
	{ loc_smoothers[k] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[k]), _parallel, _sl2); }
    }
  } // HybridBS::HybridBS(..)


  template<class TM>
  HybridBS<TM> :: HybridBS (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h, Table<int> && blocks,
			    Table<int> && block_ext_dofs, bool _overlap, bool _in_thread, bool _parallel, bool _sl2)
    : HybridSmoother2<TM>(_A, eqc_h, _overlap, _in_thread)
  {
    Array<Table<int>> fblocks;
    if (eqc_h->GetCommunicator().Size() > 2)
      { throw Exception("dampened BS toto for MPI"); }
    else { fblocks.SetSize(3); fblocks[0] = move(blocks); }
    loc_smoothers.SetSize(3);
    Array<TM> mod_diag = this->CalcModDiag(nullptr);
    if(mod_diag.Size()) {
      throw Exception("dampened BS toto for MPI");
      // for (auto k : Range(loc_smoothers))
	// { loc_smoothers[k] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[k]), mod_diag); }
    }
    else {
      loc_smoothers[0] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[0]), move(block_ext_dofs), _parallel, _sl2);
      loc_smoothers[1] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[1]), _parallel, _sl2);
      loc_smoothers[2] = make_shared<BSmoother<TM>>(A->GetM(), move(fblocks[2]), _parallel, _sl2);
    }
  } // HybridBS::HybridBS(..)


  template<class TM>
  Array<Table<int>> HybridBS<TM> :: FilterBlocks (Table<int> && _blocks)
  {
    auto & dccm = *A->GetMap(); // the DCCMap - tells us which DOFs are "master"
    const auto & pds = *A->GetParallelDofs();

    Table<int> blocks = move(_blocks);
    auto n_blocks = blocks.Size();
    int maxbs = 100;
    Array<int> fb(maxbs);

    const auto & mdofs = *dccm.GetMasterDOFs();

    // cout << "filter " << n_blocks << " blocks " << endl;
    // cout << blocks << endl;
    auto iterate_blocks = [&](auto lam_loc, auto lam_ex) LAMBDA_INLINE {
      for (auto block_nr : Range(n_blocks)) {
	auto block = blocks[block_nr];
	// if (block.Size() > maxbs)
	  // { maxbs = 2*block.Size(); fb.SetSize(maxbs); }
	fb.SetSize(block.Size());
	bool isloc = true;
	int cnt_master = 0;
	for (auto dof : block)
	  if ( mdofs.Test(dof) ) {
	    fb[cnt_master++] = dof;
	    if (pds.GetDistantProcs(dof).Size())
	      { isloc = false; }
	  }
	if ( cnt_master > 0 ) {
	  fb.SetSize(cnt_master);
	  if ( isloc )
	    { lam_loc(fb); }
	  else
	    { lam_ex(fb); }
	}
      } // Range(n_blocks)
    };
    int cnt_loc = 0, cnt_ex = 0;
    iterate_blocks([&](auto & fb) LAMBDA_INLINE { cnt_loc++; },
		   [&](auto & fb) LAMBDA_INLINE { cnt_ex++; } );
    int nl1 = cnt_loc/2, nl2 = cnt_loc - nl1;
    // cout << " split into " << nl1 << " " << nl2 << " " << cnt_ex << endl;
    TableCreator<int> cl1(nl1), cl2(nl2), cex(cnt_ex);
    int cnt_loc1 = 0, cnt_loc2 = 0; cnt_ex = 0;
    for ( ; !cex.Done(); cl1++, cl2++, cex++ ) {
      cnt_loc1 = cnt_loc2 = cnt_ex = 0;
      iterate_blocks([&](auto & fb) LAMBDA_INLINE {
	  if ( cnt_loc1 < nl1 )
	    { cl1.Add(cnt_loc1++, fb); }
	  else
	    { cl2.Add(cnt_loc2++, fb); }
	},
	[&](auto & fb) LAMBDA_INLINE {
	  cex.Add(cnt_ex++, fb);
	} );
    }
    // cout << "after filter, there are " << nl1 << " + " << cnt_ex << " + " << nl2 << " = " << nl1+nl2+cnt_ex << " blocks left " << endl;

    Array<Table<int>> fblocks(3);
    fblocks[0] = cl1.MoveTable();
    fblocks[1] = cex.MoveTable();
    fblocks[2] = cl2.MoveTable();

    // cout << " blocks loc 1 :" << endl;
    // cout << fblocks[0] << endl;
    // cout << " blocks ex    :" << endl;
    // cout << fblocks[1] << endl;
    // cout << " blocks loc 2 :" << endl;
    // cout << fblocks[2] << endl;

    // cout << " orig blocks sz :" << endl;
    // for (auto k : Range(blocks))
    //   { cout << "(" << k << "::" << blocks[k].Size() <<") "; }
    // cout << endl;

    // cout << " blocks loc1 sz :" << endl;
    // for (auto k : Range(fblocks[0]))
    //   { cout << "(" << k << "::" << fblocks[0][k].Size() <<") "; }
    // cout << endl;

    // cout << " blocks ex :" << endl;
    // for (auto k : Range(fblocks[1]))
    //   { cout << "(" << k << "::" << fblocks[1][k].Size() <<") "; }
    // cout << endl;

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
    loc_smoothers[2-stage]->SmoothBack(x, b);
  } // HybridBS::SmoothBackRESLocal


  template<class TM>
  void HybridBS<TM> :: SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    loc_smoothers[stage]->SmoothRES(x, res);
  } // HybridBS::SmoothRESLocal


  template<class TM>
  void HybridBS<TM> :: SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    loc_smoothers[2-stage]->SmoothBackRES(x, res);
  } // HybridBS::SmoothBackRESLocal


  /** END HybridBS **/

} // namespace amg


namespace amg
{
  template class HybridBS<double>;
  template class HybridBS<Mat<2,2,double>>;
  template class HybridBS<Mat<3,3,double>>;
#ifdef ELASTICITY
  template class HybridBS<Mat<6,6,double>>;
#endif
} // namespace amg

