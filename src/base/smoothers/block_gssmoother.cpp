#define FILE_AMG_BLOCKSMOOTHER_CPP

#include "block_gssmoother.hpp"
#include <utils_arrays_tables.hpp>
#include <utils_denseLA.hpp>

// I don't want to bother untangling this right now,
// just include the header
#include "loc_block_gssmoother_impl.hpp"

namespace amg
{

/** BSmoother **/

template<class TM>
BSmoother<TM>::
BSmoother (shared_ptr<SparseMatrix<TM>> _spmat,
           Table<int> && _blocks,
            bool _parallel,
            bool _use_sl2,
            bool _pinv,
            FlatArray<TM> md)
  : BaseSmoother(_spmat)
  , spmat(_spmat)
  , blocks(std::move(_blocks))
  , parallel(_parallel)
  , use_sl2(_use_sl2)
  , pinv(_pinv)
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

  for (auto k : Range(n_blocks))
  {
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

  ParallelJob([&] (const TaskInfo & ti)
  {
    constexpr int BS = ngbla::Height<TM>();
    LocalHeap lh(3*sqr(sizeof(double)*maxbs*ngbla::Height<TM>()), "AAA", false);
    for (int block_nr : sl)
    {
      HeapReset hr(lh);
      auto block_dofs = blocks[block_nr];
      auto D = dinv[block_nr];
      /** For some reason, sometimes (I think for badly conditioned mats) CalcInverse with Mat<N,N> gives garbage wile it works with double entries **/
      if constexpr(BS>1)
      {
        lh.CleanUp();
        const int n = block_dofs.Size(); const int N = BS * n;
        FlatMatrix<double> flatD(N, N, lh);
        for (int i : Range(block_dofs))
        {
          const int I = BS * i;
          /** TODO: these loops are not ideal... **/
          for (int j : Range(block_dofs)) {
            const int J = BS * j;
            const auto & aetr = A(block_dofs[i], block_dofs[j]);
            Iterate<BS>([&](auto ii){ Iterate<BS>([&](auto jj) { flatD(I+ii, J+jj) = aetr(ii,jj); }); });
          } // j
          if (md.Size())
          {
            // const auto & mdetr = md[block_dofs[i]];
            // Iterate<BS>([&](auto ii){ flatD(I+ii, I+ii) = mdetr(ii,ii); });
            flatD.Rows(I, I + BS).Cols(I, I + BS) = md[block_dofs[i]];
          } // md.Size()
        } // i
        // cout << " diag " << block_nr << endl << flatD << endl;
        if (pinv)
          { CalcPseudoInverseTryNormal(flatD, lh); }
        else
          { CalcInverse(flatD); }
        // cout << " inv diag " << block_nr << endl << flatD << endl;
        for (int i : Range(block_dofs))
        {
          const int I = BS * i;
          for (int j : Range(block_dofs))
          {
            const int J = BS * j;
            Iterate<BS>([&](auto ii){ Iterate<BS>([&](auto jj) { D(i,j)(ii,jj) = flatD(I+ii, J+jj); }); });
          }
        }
      }
      else
      {
        for (int i : Range(block_dofs))
        {
          for (int j : Range(block_dofs))
            { D(i,j) = A(block_dofs[i], block_dofs[j]); }
          if (md.Size())
            { D(i,i) = md[block_dofs[i]]; }
        }
        // cout << " diag " << block_nr << endl << D << endl;
        if (pinv) {
          if constexpr(ngbla::Height<TM>() > 1)
          {
            int nd = block_dofs.Size(), ND = ngbla::Height<TM>() * nd;
            FlatMatrix<double> scal_diag(ND, ND, lh);
            mat_to_scal(nd, D, scal_diag);
            CalcPseudoInverseTryNormal (scal_diag, lh);
            scal_to_mat(nd, scal_diag, D);
          }
          else
            { CalcPseudoInverseTryNormal (D, lh); }
        }
        else
          { CalcInverse(D); }
        // cout << "inv diag " << block_nr << endl << D << endl;
      }
    }
  });

  // cout << " have dinvs " << endl;

  if (parallel)
  {
    /** Do block coloring here for shm parallelization **/
    Table<int> ext_blocks;

    { /** For dependencies of blocks, because of resiudal upates, we also need one layer around each block! **/
      TableCreator<int> ceb(n_blocks);
      Array<int> dnums(20*maxbs);
      for (; !ceb.Done(); ceb++) {
        for (auto block_nr : Range(n_blocks))
        {
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
          {
            for (auto col : A.GetRowIndices(dof))
            {
              pos = find_in_sorted_array(col, dnums);
              if (pos == -1)
                { insert_into_sorted_array(col, dnums); }
            }
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
    for (; !cbc.Done(); cbc++)
    {
      for (auto k : Range(bcarray))
        { cbc.Add(bcarray[k], k); }
    }
    block_colors = cbc.MoveTable();

    if (use_sl2)
    {
      // loops.SetSize(block_colors.Size()); // no operator= with const reference for sharedloops
      loops = std::move(Array<SharedLoop2>(block_colors.Size()));
    }
    else
    {
      color_balance.SetSize(block_colors.Size());

      for (auto cnum : Range(block_colors))
      {
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


template<class TM> template<class TLAM>
INLINE void BSmoother<TM> :: IterateBlocks (bool reverse, TLAM smooth_block) const
{
  if (parallel) 
  {
    if (use_sl2)
    {
      const int ncolors = block_colors.Size();

      for (auto k : Range(loops))
        { loops[k].Reset(block_colors[k].Range()); }

      task_manager -> CreateJob([&] (const TaskInfo & ti)
      {
        VectorMem<100,TV> hxmax(maxbs);
        VectorMem<100,TV> hymax(maxbs);
        for (int J : Range(block_colors))
        {
          int color = reverse ? ncolors-1-J : J;
          for (auto bcind : loops[color])
          {
            auto block_nr = block_colors[color][bcind];
            smooth_block(block_nr, hxmax, hymax);
          }
        }
      });
    }
    else // !use_sl2
    {
      const int ncolors = block_colors.Size();

      for (int J : Range(block_colors))
      {
        int color = reverse ? ncolors-1-J : J;
        ParallelForRange(color_balance[color], [&] (IntRange r) LAMBDA_INLINE {
          VectorMem<100,TV> hxmax(maxbs);
          VectorMem<100,TV> hymax(maxbs);
          auto & cblocks = block_colors[color];

          for (auto block_nr : cblocks.Range(r))
            { smooth_block(block_nr, hxmax, hymax); }
        });
      }
    }
  }
  else
  { // !parallel
    VectorMem<100,TV> hxmax(maxbs);
    VectorMem<100,TV> hymax(maxbs);

    if (reverse)
    {
      for (int block_num = blocks.Size() - 1; block_num >= 0; block_num--)
        { smooth_block(block_num, hxmax, hymax); }
    }
    else
    {
      for (auto block_num : Range(blocks.Size()))
        { smooth_block(block_num, hxmax, hymax); }
    }
  }
} // BSmoother::IterateBlocks


INLINE Timer<TTracing, TTiming>& SRHS_thack () { static Timer t("BSmoother::SmoothRHS(FW/BW)"); return t; }
template<class TM>
INLINE void
BSmoother<TM>::
Smooth_impl (BaseVector & x, const BaseVector & b, int steps, bool reverse) const
{
  RegionTimer rt(SRHS_thack());

  auto fx = x.FV<TV>();
  auto fb = b.FV<TV>();
  const auto & A(*spmat);

  for (int step_nr : Range(steps))
  {
    IterateBlocks(reverse, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE
    {
      auto block_dofs = blocks[block_nr];
      int bs = block_dofs.Size();

      if ( bs > 0 )
      {
        // cout << " up block " << block_nr << ", bs = " << bs << endl;
        // prow(block_dofs); cout << endl;
        FlatVector<TV> hup = hxmax.Range(0,bs);
        FlatVector<TV> hr = hymax.Range(0,bs);
        for (int j = 0; j < bs; j++)
        {
          auto jj = block_dofs[j];
          hr(j) = fb(jj) - A.RowTimesVector(jj, fx);
        }
        // cout << " rhs " << endl << fb(block_dofs) << endl;
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
void
BSmoother<TM>::
SmoothInternal (BaseVector &x, BaseVector const &b, int steps) const
{
  Smooth_impl(x, b, steps, false);
} //BSmoother::SmoothInternal


template<class TM>
void
BSmoother<TM>::
SmoothBackInternal (BaseVector &x, BaseVector const &b, int steps) const
{
  Smooth_impl(x, b, steps, true);
} //BSmoother::SmoothBackInternal


INLINE Timer<TTracing, TTiming>& SRES_thack () { static Timer t("BSmoother::SmoothRES(FW/BW)"); return t; }

template<class TM>
INLINE void
BSmoother<TM> :: SmoothRES_impl (BaseVector & x, BaseVector & res, int steps, bool reverse) const
{
  RegionTimer rt(SRES_thack());

  auto fx = x.FV<TV>();
  auto fres = res.FV<TV>();
  const auto & A(*spmat);

  for (int step_nr : Range(steps))
  {
    IterateBlocks(reverse, [&](auto block_nr, auto & hxmax, auto & hymax)
    {
      auto block_dofs = blocks[block_nr];
      int bs = block_dofs.Size();
      
      if ( bs > 0 )
      { // avoid range check in debug mode
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
void
BSmoother<TM>::
SmoothRESInternal (BaseVector & x, BaseVector & res, int steps) const
{
  SmoothRES_impl(x, res, steps, false);
} //BSmoother::SmoothRESInternal


template<class TM>
void
BSmoother<TM>::
SmoothBackRESInternal (BaseVector & x, BaseVector & res, int steps) const
{
  SmoothRES_impl(x, res, steps, true);
} //BSmoother::SmoothBackRESInternal


template<class TM>
void
BSmoother<TM>::
MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  // IterateBlocks(false, [&](auto block_nr, auto & hxmax, auto & hymax) {
  auto fx = x.FV<TV>();
  auto fb = b.FV<TV>();
  ParallelForRange (blocks.Size(), [&] ( IntRange r )
  {
    VectorMem<100,TV> hxmax(maxbs);
    VectorMem<100,TV> hymax(maxbs);
    for (auto block_nr : r) {
      auto block_dofs = blocks[block_nr];
      int bs = block_dofs.Size();
      FlatVector<TV> hup = hxmax.Range(0,bs);
      FlatVector<TV> hr = hymax.Range(0,bs);
      hr = fb(block_dofs);
      hup = dinv[block_nr] * hr;
      fx(block_dofs) += s * hup;
    }
  });
} // BSmoother::MultAdd

template<class TM>
void
BSmoother<TM>::
Smooth (BaseVector       &x,
			  BaseVector const &b,
			  BaseVector       &res,
			  bool              res_updated,
			  bool              update_res,
			  bool              x_zero) const
{
  if (update_res)
  {
    if (res_updated)
    {
      SmoothRESInternal(x, res, 1);
    }
    else
    {
      // TODO: benchmark which version is fater!
      this->CalcResiduum(x, b, res, x_zero);
      SmoothRESInternal(x, res, 1);

      // SmoothInternal(x, b, 1);
      // this->CalcResiduum(x, b, res);
    }
  }
  else
  {
    // if res_updated, just forget about the residual vector
    SmoothInternal(x, b, 1);
  }
} // BSmoother::Smooth

template<class TM>
void
BSmoother<TM>::
SmoothBack (BaseVector       &x,
			      BaseVector const &b,
			      BaseVector       &res,
			      bool              res_updated,
			      bool              update_res,
			      bool              x_zero) const
{
  if (update_res)
  {
    if (res_updated)
    {
      SmoothBackRESInternal(x, res, 1);
    }
    else
    {
      // TODO: benchmark which version is fater!
      this->CalcResiduum(x, b, res, x_zero);
      SmoothBackRESInternal(x, res, 1);

      // SmoothBackInternal(x, b, 1);
      // this->CalcResiduum(x, b, res);
    }
  }
  else
  {
    // if res_updated, just forget about the residual vector
    SmoothBackInternal(x, b, 1);
  }
} // BSmoother::SmoothBack


/** END BSmoother **/


/** HybridBS **/


template<class TM>
HybridBS<TM>::
HybridBS (shared_ptr<BaseMatrix> _A, // shared_ptr<EQCHierarchy> eqc_h,
          Table<int> && blocks,
          bool _pinv,
          bool _overlap,
          bool _in_thread,
          bool _parallel,
          bool _sl2,
          bool _bs2,
          bool _no_blocks,
          bool _symm_loc,
          int _nsteps_loc)
  : HybridSmoother<TM>(_A, _nsteps_loc, _in_thread, _overlap)
  , bs2(_bs2)
  , no_blocks(_no_blocks)
  , symm_loc(_symm_loc)
{
  Array<Table<int>> fblocks;

  auto comm = this->GetParallelDofs()->GetCommunicator();

  bool filterBlocks = comm.Size() > (IsRankZeroIdle(this->GetParallelDofs()) ? 1 : 2);

  if (filterBlocks)
    { fblocks = FilterBlocks(std::move(blocks)); }
  else
    { fblocks.SetSize(3); fblocks[0] = std::move(blocks); }

  // cout << " HybridBS constr, blocks: " << endl;
  // for (auto k : Range(fblocks))
  // {
  //   cout << fblocks[k].Size() << endl;
  // }

  Array<TM> mod_diag = this->CalcModDiag(nullptr);

  auto &A = this->GetHybSparseA();

  if (bs2)
  {
    loc_smoother_2 = make_shared<BSmoother2<TM>>(A.GetSpM(),
                                                 fblocks,
                                                 false,
                                                 false,
                                                 _pinv,
                                                 _no_blocks,
                                                 mod_diag);
   
    if (no_blocks)
    {
      // replace M in the hybrid matrix with the forward application of the loc_smoother (saves memory and should not be slower)
      A.ReplaceM(loc_smoother_2->GetAMatrix());
    }
  }
  else
  {
    loc_smoothers.SetSize(fblocks.Size());

    if(mod_diag.Size())
    {
      for (auto k : Range(loc_smoothers))
      {
        loc_smoothers[k] = make_shared<BSmoother<TM>>(A.GetSpM(),
                                                      std::move(fblocks[k]),
                                                      _parallel,
                                                      _sl2,
                                                      _pinv,
                                                      mod_diag);
      }
    }
    else
    {
      for (auto k : Range(loc_smoothers))
      {
        loc_smoothers[k] = make_shared<BSmoother<TM>>(A.GetSpM(),
                                                      std::move(fblocks[k]),
                                                      _parallel,
                                                      _sl2,
                                                      _pinv);
      }
    }
  }
} // HybridBS::HybridBS(..)


template<class TM>
Array<Table<int>> HybridBS<TM> :: FilterBlocks (Table<int> && _blocks)
{
  auto const &A = this->GetHybSparseA();

  auto       &dccm = A.GetDCCMap(); // the DCCMap - tells us which DOFs are "master"
  auto const &pds = *A.GetParallelDofs();

  Table<int> blocks = std::move(_blocks);
  auto n_blocks = blocks.Size();
  int maxbs = 100;
  Array<int> fb(maxbs);

  const auto & mdofs = *dccm.GetMasterDOFs();

  // cout << "filter " << n_blocks << " blocks " << endl;
  // cout << blocks << endl;
  auto iterate_blocks = [&](auto lam_loc, auto lam_ex) LAMBDA_INLINE
  {
    for (auto block_nr : Range(n_blocks))
    {
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

  int nl1 = (this->symm_loc?cnt_loc:cnt_loc/2), nl2 = cnt_loc - nl1;

  // cout << " split into " << nl1 << " " << nl2 << " " << cnt_ex << endl;
  TableCreator<int> cl1(nl1), cl2(nl2), cex(cnt_ex);
  int cnt_loc1 = 0, cnt_loc2 = 0; cnt_ex = 0;

  for ( ; !cex.Done(); cl1++, cl2++, cex++ )
  {
    cnt_loc1 = cnt_loc2 = cnt_ex = 0;
    iterate_blocks([&](auto & fb) LAMBDA_INLINE
    {
      if ( this->symm_loc || ( cnt_loc1 < nl1 ) )
        { cl1.Add(cnt_loc1++, fb); }
      else
        { cl2.Add(cnt_loc2++, fb); }
    },
    [&](auto & fb) LAMBDA_INLINE
    {
      cex.Add(cnt_ex++, fb);
    });
  }
  // cout << "after filter, there are " << nl1 << " + " << cnt_ex << " + " << nl2 << " = " << nl1+nl2+cnt_ex << " blocks left " << endl;

  Array<Table<int>> fblocks(this->symm_loc ? 2 : 3);
  fblocks[0] = cl1.MoveTable();
  fblocks[1] = cex.MoveTable();
  if (!this->symm_loc)
    { fblocks[2] = cl2.MoveTable(); }

  // cout << " blocks loc 1 :" << endl;
  // cout << fblocks[0] << endl;
  // cout << " blocks ex    :" << endl;
  // cout << fblocks[1] << endl;
  // cout << " blocks loc 2 :" << endl;
  // cout << fblocks[2] << endl;

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
void
HybridBS<TM>::
SmoothStageRHS (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  if ( bs2 )
  {
    if ( symm_loc )
    {
      // SYMM:      F,-,F / -,FB,- / B,-,B
      
      switch(stage)
      {
        case(SMOOTH_STAGE::LOC_PART_1):
        {
          Array<int> groups({ 0 });
          if (no_blocks)
            { loc_smoother_2->Smooth_savebL(groups, x, b, res, x_zero); } // FW implied by savebL
          else
            { loc_smoother_2->SmoothSimple(groups, x, b, 1, false, false); }
          break;
        }
        case(SMOOTH_STAGE::EX_PART):
        {
          Array<int> groups({ 1 });
          if (no_blocks)
            { loc_smoother_2->Smooth_SYMM(groups, x, b, res, 1, x_zero); }
          else {
            loc_smoother_2->SmoothSimple(groups, x, b, 1, false, false);
            loc_smoother_2->SmoothSimple(groups, x, b, 1, true, false); // no more zig!
          }
          break;
        }
        case(SMOOTH_STAGE::LOC_PART_2):
        {
          Array<int> groups({ 0 });
          if (no_blocks)
            { loc_smoother_2->SmoothBack_usebL(groups, x, b, res); }
          else
            { loc_smoother_2->SmoothSimple(groups, x, b, 1, true, false); }
          break;
        }
      }
    }
    else
    {
      // NON SYMM:  F,-,- / -,F,- / -,-,F

      bool const reverse = direction == BACKWARD;

      switch(stage)
      {
        case(SMOOTH_STAGE::LOC_PART_1):
        {
          Array<int> groups({0});
          loc_smoother_2->SmoothSimple(groups, x, b, 1, reverse, false);
          break;
        }
        case(SMOOTH_STAGE::EX_PART):
        {
          Array<int> groups({1});
          loc_smoother_2->SmoothSimple(groups, x, b, 1, reverse, false);
          break;
        }
        case(SMOOTH_STAGE::LOC_PART_2):
        {
          Array<int> groups({2});
          loc_smoother_2->SmoothSimple(groups, x, b, 1, reverse, false);
          break;
        }
      }
    }
  }
  else
  {
    if ( symm_loc )
    {
      // SYMM:      F,-,F / -,FB,- / B,-,B
      
      switch(stage)
      {
        case(SMOOTH_STAGE::LOC_PART_1):
        {
          loc_smoothers[0]->SmoothInternal(x, b);
          break;
        }
        case(SMOOTH_STAGE::EX_PART):
        {
          loc_smoothers[1]->SmoothInternal(x, b);
          loc_smoothers[1]->SmoothBackInternal(x, b);
          break;
        }
        case(SMOOTH_STAGE::LOC_PART_2):
        {
          loc_smoothers[0]->SmoothBackInternal(x, b);
          break;
        }
      }
    }
    else
    {
      // NON SYMM:  F,-,- / -,F,- / -,-,F

      int idxS(stage);
      
      if ( direction == FORWARD )
        { loc_smoothers[idxS]->SmoothInternal(x, b); } 
      else
        { loc_smoothers[idxS]->SmoothBackInternal(x, b); }
    }
  }
} // HybridBS::SmoothStageRHS


template<class TM>
void
HybridBS<TM>::
SmoothStageRes (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  if ( bs2 )
  {
    if ( symm_loc )
    {
      // SYMM:      F,-,F / -,FB,- / B,-,B

      switch(stage)
      {
        case(SMOOTH_STAGE::LOC_PART_1):
        {
          Array<int> groups({ 0 });
          if (no_blocks)
            { loc_smoother_2->Smooth_savebL(groups, x, res, res, x_zero); } // wrong ?! need b,res here, not res,res
          else
            { loc_smoother_2->SmoothRESSimple(groups, x, res, 1, false, false); }
          break;
        }
        case(SMOOTH_STAGE::EX_PART):
        {
          Array<int> groups({ 1 });
          if (no_blocks)
            { loc_smoother_2->Smooth_SYMM_CR(groups, x, b, res, 1, x_zero); }
          else
            { loc_smoother_2->SmoothRESSimple(groups, x, res, 1, false, true); }
          break;
        }
        case(SMOOTH_STAGE::LOC_PART_2):
        {
          Array<int> groups({ 0 });
          if (no_blocks)
            { loc_smoother_2->SmoothBack_usebL_saveL(groups, x, b, res); } // does not update RES!
          else
            { loc_smoother_2->SmoothRESSimple(groups, x, res, 1, true, false); }
          break;
        }
      }
    }
    else
    {
      // NON SYMM:  F,-,- / -,F,- / -,-,F

      int idxS(stage);
      Array<int> groups({ idxS });

      bool const reverse = direction == BACKWARD;

      loc_smoother_2->SmoothRESSimple(groups, x, res, 1, reverse, false);
    }
  }
  else
  {
    if ( symm_loc )
    {
      // SYMM:      F,-,F / -,FB,- / B,-,B

      switch(stage)
      {
        case(SMOOTH_STAGE::LOC_PART_1):
        {
          loc_smoothers[0]->SmoothRESInternal(x, res);
          break;
        }
        case(SMOOTH_STAGE::EX_PART):
        {
          loc_smoothers[1]->SmoothRESInternal(x, res);
          loc_smoothers[1]->SmoothBackRESInternal(x, res);
          break;
        }
        case(SMOOTH_STAGE::LOC_PART_2):
        {
          loc_smoothers[0]->SmoothBackRESInternal(x, res);
          break;
        }
      }
    }
    else
    {
      // NON SYMM:  F,-,- / -,F,- / -,-,F
      int idxS(stage);
      bool const reverse = direction == BACKWARD;

      if ( reverse )
        { loc_smoothers[idxS]->SmoothBackRESInternal(x, res); }
      else
        { loc_smoothers[idxS]->SmoothRESInternal(x, res); }
    }
  }
} // HybridBS::SmoothStageRes


template<class TM>
void HybridBS<TM> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  x.Distribute();
  b.Cumulate();

  auto parb = dynamic_cast<const ParallelBaseVector*>(&b);

  if (parb == nullptr)
    { throw Exception("HybridGSS3::MultAdd b not parallel!"); }

  auto parx = dynamic_cast<ParallelBaseVector*>(&x);
  if (parx == nullptr)
    { throw Exception("HybridGSS3::MultAdd x not parallel!"); }

  if (bs2)
    { loc_smoother_2->MultAdd(s, *parb->GetLocalVector(), *parx->GetLocalVector()); }
  else
  {
    for (auto k : Range(loc_smoothers))
      { loc_smoothers[k]->MultAdd(s, *parb->GetLocalVector(), *parx->GetLocalVector()); }
  }

  b.Distribute();
} // HybridBS::MultAdd

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

