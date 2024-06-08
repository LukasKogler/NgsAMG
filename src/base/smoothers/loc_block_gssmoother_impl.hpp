#ifndef FILE_AMG_BLOCKSMOOTHER_LOC_IMPL_HPP
#define FILE_AMG_BLOCKSMOOTHER_LOC_IMPL_HPP

#include <utils_arrays_tables.hpp>
#include <utils_denseLA.hpp>

namespace amg
{

  /** BSBlock **/

  template<class TM>
  INLINE tuple<int, int> BSmoother2<TM>::BSBlock :: GetAllocSize (const SparseMatrixTM<TM> & A, FlatArray<int> block_dofs, bool _LU, bool _md)
  {
    LU = _LU; md = _md;
    int bufs_inds = 0, bufs_vals;
    for (auto dof : block_dofs) { // cols, vals
      auto allcols = A.GetRowIndices(dof);
      iterate_anotb(allcols, block_dofs, [&](auto indi) {
	  bufs_inds++;
	});
    }
    bufs_vals = bufs_inds;
    firsti.Assign(FlatArray<int>(block_dofs.Size(), NULL)); // [NO!! not +1!!] TERRIBLE HACK!!! but this way we can save info for Alloc later!
    dofnrs.Assign(FlatArray<int>(bufs_inds, NULL));         // TERRIBLE HACK!!! but this way we can save info for Alloc later!
    bufs_inds += 1 + 2 * block_dofs.Size();  // firsti, rownrs
    if (LU)
      { bufs_inds += block_dofs.Size(); }    // double firstis !
    bufs_vals += 2 * sqr(block_dofs.Size()); // diag + inv
    if (md)
      { bufs_vals += block_dofs.Size(); }    // add. diag vals (hybrid smoothers)
    return make_tuple(bufs_inds, bufs_vals);
  } // BSBlock::GetBufferSizes


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: Alloc (LocalHeap & lh)
  {
    int nds = firsti.Size(), nctot = dofnrs.Size();
    dofnrs.Assign(nds, lh);
    firsti.Assign(LU ? (1+2*nds) : (1+nds), lh);
    cols.Assign(nctot, lh);
    diag.AssignMemory(nds, nds, lh);
    diag_inv.AssignMemory(nds, nds, lh);
    vals.Assign(nctot, lh);
    if (md)
      { mdadd.Assign(nds, lh); }
  } // BSBlock::Alloc


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: Alloc (int*& ptr_i, TM*& ptr_v)
  {
    int nds = firsti.Size(), nctot = dofnrs.Size(), nfis = LU ? (1+2*nds) : (1+nds);
    dofnrs.Assign(FlatArray<int>(nds, ptr_i)); ptr_i += nds;
    firsti.Assign(FlatArray<int>(nfis, ptr_i)); ptr_i += nfis;
    cols.Assign(FlatArray<int>(nctot, ptr_i)); ptr_i += nctot;
    diag.AssignMemory(nds, nds, ptr_v); ptr_v += sqr(nds);
    diag_inv.AssignMemory(nds, nds, ptr_v); ptr_v += sqr(nds);
    vals.Assign(FlatArray<TM>(nctot, ptr_v)); ptr_v += nctot;
    if (md)
      { mdadd.Assign(FlatArray<TM>(nds, ptr_v)); ptr_v += nds; }
  } // BSBlock::Alloc


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: SetFromSPMat (const SparseMatrixTM<TM> & A, FlatArray<int> dofs, LocalHeap & lh, bool pinv,
						       FlatArray<TM> mdvals)
  {
    dofnrs = dofs;
    QuickSort(dofnrs); // e.g V.GetDofNrs(el) can be un-sorted, but we want it sorted below!
    firsti = 0;
    diag_inv = 0.0; // can have zero entries that are not set below!
    int ci = 0; // col counter
    for (auto kd : Range(dofnrs)) {
      auto dof = dofnrs[kd];
      auto allcols = A.GetRowIndices(dof);
      auto allvals = A.GetRowValues(dof);
      for (auto j : Range(allcols)) {
	int col = allcols[j];
	if ( md && (dof == col) ) {
	  diag_inv(kd, kd) = allvals[j];
	  mdadd[kd] = mdvals[dof] - allvals[j];
	}
	else {
	  int pos = find_in_sorted_array(col, dofnrs);
	  if ( pos == -1 ) {
	    cols[ci] = col;
	    vals[ci++] = allvals[j];
	    firsti[kd+1]++;
	  }
	  else
	    { diag_inv(kd, pos) = allvals[j]; }
	}
      }
    }
    diag = diag_inv;
    if (md)
      for (auto k : Range(dofnrs))
	{ diag_inv(k, k) = mdvals[dofnrs[k]]; }
    // cout << endl << "block for dofs "; prow(dofs); cout << endl;
    // cout << "dofnrs         "; prow(dofnrs); cout << endl;
    // cout << " diag " << endl << diag << endl;
    // cout << " diag_inv " << endl << diag_inv << endl;
    // get firsti, cols and vals
    for (auto kd : Range(dofnrs))
      { firsti[kd+1] += firsti[kd]; }
    // cout << " firsti " << endl << firsti << endl;
    // cout << " vals " << endl; prow2(vals); cout << endl;
    // cout << " cols " << endl; prow2(cols); cout << endl;
    // cout << " mdadd " << endl; prow2(mdadd); cout << endl;
    // invert diagonal
    if (pinv) {
      HeapReset hr(lh);
      CalcPseudoInverseTryNormal(diag_inv, lh);
      // if constexpr (ngbla::Height<TM>()>1)
      // {
      //   int nd = dofnrs.Size(), ND = ngbla::Height<TM>() * nd;
      //   FlatMatrix<double> scal_diag(ND, ND, lh);
      //   mat_to_scal(nd, diag_inv, scal_diag);
      //   CalcPseudoInverseNew (scal_diag, lh);
      //   scal_to_mat(nd, scal_diag, diag_inv);
      // }
      // else
      //   { CalcPseudoInverseNew (diag_inv, lh); }
    }
    else
      { CalcInverse(diag_inv); }
    // cout << " diag inv " << endl << diag_inv << endl;
  } // BSmoother2<TM>::BSBlock::SetFromSPMat


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: SetLUFromSPMat (int block_nr, FlatArray<int> d2blk,
							 const SparseMatrixTM<TM> & A, FlatArray<int> dofs, LocalHeap & lh, bool pinv,
							 FlatArray<TM> mdvals)
  {
    if (md)
      { throw Exception("MD still TODO (non-overlapping block version)!"); }
    dofnrs = dofs;
    QuickSort(dofnrs); // e.g V.GetDofNrs(el) can be un-sorted, but we want it sorted below!
    firsti = 0;
    diag_inv = 0.0; // can have zero entries that are not set below!
    int ci_left = 0, ci_right = 0; // col counter
    // cout << " block_nr = " << block_nr << endl;
    // cout << " dofnrs = "; prow(dofnrs); cout << endl;
    for (auto kd : Range(dofnrs)) {
      auto dof = dofnrs[kd];
      auto allcols = A.GetRowIndices(dof);
      for (int col : allcols) {
	// int pos = find_in_sorted_array(col, dofnrs);
	if (d2blk[col] < block_nr) // part of "L" -> inc offset for U cols/vals
	  { ci_right++; }
      }
    }
    int nds = dofnrs.Size();
    for (auto kd : Range(dofnrs)) {
      auto dof = dofnrs[kd];
      auto allcols = A.GetRowIndices(dof);
      auto allvals = A.GetRowValues(dof);
      int fii_left = kd+1, fii_right = nds + fii_left;
      for (auto j : Range(allcols)) {
	int col = allcols[j];
	// cout << col << " in blk " << d2blk[col] << endl;
	if (md && (dof == col)) {
	  diag_inv(kd, kd) = allvals[j];
	  mdadd[kd] = mdvals[dof] - allvals[j];
	}
	else {
	  if (d2blk[col] == block_nr) {
	    int pos = find_in_sorted_array(col, dofnrs);
	    diag_inv(kd, pos) = allvals[j];
	  }
	  else if (d2blk[col] < block_nr) { // part of "L"
	    // cout << col << " -> L " << ci_left << endl;
	    cols[ci_left] = col;
	    vals[ci_left++] = allvals[j];
	    firsti[fii_left]++;
	  } else { // part of "R"
	    // cout << col << " -> R " << ci_right << endl;
	    cols[ci_right] = col;
	    vals[ci_right++] = allvals[j];
	    firsti[fii_right]++;
	  }
	}
      }
    }
    diag = diag_inv;
    if (md)
      for (auto k : Range(dofnrs))
	{ diag_inv(k, k) = mdvals[dofnrs[k]]; }
    // cout << endl
	 // << "block for dofs "; prow(dofs); cout << endl;
    // cout << "dofnrs         "; prow(dofnrs); cout << endl;
    // cout << " diag " << endl << diag << endl;
    // cout << " diag inv " << endl << diag_inv << endl;
    // get firsti, cols and vals
    for (auto kd : Range(2*nds))
      { firsti[kd+1] += firsti[kd]; }
    // invert diagonal
    if (pinv) {
      HeapReset hr(lh);
      CalcPseudoInverseTryNormal(diag_inv, lh);
      // if constexpr (ngbla::Height<TM>()>1) {
      //   int nd = dofnrs.Size(), ND = ngbla::Height<TM>() * nd;
      //   FlatMatrix<double> scal_diag(ND, ND, lh);
      //   mat_to_scal(nd, diag_inv, scal_diag);
      //   CalcPseudoInverseNew (scal_diag, lh);
      //   scal_to_mat(nd, scal_diag, diag_inv);
      // }
      // else
    	// { CalcPseudoInverseNew (diag_inv, lh); }
    }
    else
      { CalcInverse(diag_inv); }
    // cout << " cols = "; prow(cols); cout << endl;
    // cout << " diag inv " << endl << diag_inv << endl;
  } // BSmoother2<TM>::BSBlock::SetLUFromSPMat


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: Prefetch () const
  {
#ifdef NETGEN_ARCH_AMD64
#ifdef __GNUC__
    char *pi = reinterpret_cast<char*>(firsti.Data()),
      *pin = reinterpret_cast<char*>(cols.Data() + cols.Size());
    while (pi < pin) {
      _mm_prefetch (reinterpret_cast<void*>(pi), _MM_HINT_T2);
      pi += 64;
    }
    char *vi = reinterpret_cast<char*>(diag_inv.Data()),
      *vin = reinterpret_cast<char*>(vals.Data() + vals.Size()); // misses all but one entry of last TM
    while (vi < vin) {
      _mm_prefetch (reinterpret_cast<void*>(vi), _MM_HINT_T2);
      vi += 64;
    }
#endif
#endif
  } // BSmoother2<TM>::BSBlock::BSmoother2<TM>::BSBlock ::


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
							   FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const
  {
    for (auto k : Range(dofnrs)) {
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    if (LU) { // L and U part stored seperately
      int nds = dofnrs.Size();
      for (auto k : Range(dofnrs))
	for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	  { smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    // DOES NOT WORK WITH PINV
    // if (md)
      // for (auto k : Range(dofnrs))
	// { smallrhs(k) += mdadd[k] * bigsol(dofnrs[k]); }
    // smallsol = diag_inv * smallrhs;
    // for (auto k : Range(dofnrs))
      // { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
    smallrhs -= diag * bigsol(dofnrs);
    smallsol = diag_inv * smallrhs;
    bigsol(dofnrs) += omega * smallsol;
  } // BSBlock::RichardsonUpdate


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_FW_zig (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
								  FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const
  {
    // b - L * x_new
    for (auto k : Range(dofnrs)) {
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    smallsol = diag_inv * smallrhs;
    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_BW_zig (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
								  FlatVector<TV> smallrhs, FlatVector<TV> bigrhs) const
  {
    int nds = dofnrs.Size();

    // b - U * x_new
    for (auto k : Range(dofnrs))
    {
      smallrhs(k) = bigrhs(dofnrs[k]); // contains b or (b-Lx if we are being tricky)

      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
      	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }

    smallsol = diag_inv * smallrhs;

    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
  } // BSBlock::RichardsonUpdate


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_FW_zig_saveU (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
									FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    // b - L * x_new
    for (auto k : Range(dofnrs)) {
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    smallsol = diag_inv * smallrhs;
    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
    // (L+D) x_new = res -->  res_new = b-Ax_new = -Ux_new
    // so set res to 0, later blocks add -= L^T x_new
    for (auto dnr : dofnrs)
      { bigres(dnr) = 0.0; }
    // res -= (D+L^T) x_new
    for (auto k : Range(dofnrs)) {
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ bigres(cols[l]) -= vals[l] * smallsol[k]; }
    }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_FW_zig_savebL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
									 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    // save b - L * x_new to "res"
    for (auto k : Range(dofnrs))
    {
      smallrhs(k) = bigrhs(dofnrs[k]);

      for (auto l : Range(firsti[k], firsti[k+1]))
      	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }

      bigres(dofnrs[k]) = smallrhs(k);
    }

    smallsol = diag_inv * smallrhs;

    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
  } // BSBlock::RichardsonUpdate_FW_zig_savebL


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_BW_zig_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
									FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    int nds = dofnrs.Size();
    // b - U * x_new
    for (auto k : Range(dofnrs)) {
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    smallsol = diag_inv * smallrhs;
    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
    // (U+D) x_new = res -->  res_new = b-Ax_new = -Lx_new
    // so set res to 0, later blocks add -= U^T x_new
    for (auto dnr : dofnrs)
      { bigres(dnr) = 0.0; }
    // res -= L x_new
    for (auto k : Range(dofnrs)) {
      for (auto l : Range(firsti[nds], firsti[nds+1]))
	{ bigres(cols[l]) -= vals[l] * smallsol[k]; }
    }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_saveU (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
								 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    // current res
    int nds = dofnrs.Size();
    for (auto k : Range(dofnrs)) { // b-Lx
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    for (auto k : Range(dofnrs)) { // -Ux
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    smallsol = diag_inv * smallrhs;
    TV tmp;
    for (auto k : Range(dofnrs)) {
      tmp = bigsol(dofnrs[k]);
      bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k);
      smallsol(k) = bigsol(dofnrs[k]) - tmp; // the update, for new RES
    }
    // (L+D) x_new = res -->  res_new = b-Ax_new = -U(x_new-x)
    // so set res to 0, later blocks add -= L^T x_new
    for (auto dnr : dofnrs)
      { bigres(dnr) = 0.0; }
    // res -= L^T (x_new-x)
    for (auto k : Range(dofnrs)) {
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ bigres(cols[l]) -= vals[l] * smallsol[k]; }
    }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
								 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    // current res
    int nds = dofnrs.Size();
    for (auto k : Range(dofnrs)) { // b-Lx
      smallrhs(k) = bigrhs(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    for (auto k : Range(dofnrs)) { // -Ux
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }
    smallsol = diag_inv * smallrhs;
    TV tmp;
    for (auto k : Range(dofnrs)) {
      tmp = bigsol(dofnrs[k]);
      bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k);
      smallsol(k) = bigsol(dofnrs[k]) - tmp; // the update, for new RES
    }
    // (U+D) x_new = res -->  res_new = b-Ax_new = -L(x_new-x)
    // so set res to 0, later blocks add -= U^T x_new
    for (auto dnr : dofnrs)
      { bigres(dnr) = 0.0; }
    // res -= U^T (x_new-x)
    for (auto k : Range(dofnrs)) {
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ bigres(cols[l]) -= vals[l] * smallsol[k]; }
    }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_savebL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
								 FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    if (omega != 1.0)
      { throw Exception("OMEGA does not work yet..."); }

    int nds = dofnrs.Size();

    for (auto k : Range(dofnrs))
    {
      // b
      smallrhs(k) = bigrhs(dofnrs[k]);

      // L x_new
      for (auto l : Range(firsti[k], firsti[k+1]))
      	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }

      // stash B - L x_new
      bigres(dofnrs[k]) = smallrhs(k);
    }

    // U x_old
    for (auto k : Range(dofnrs))
    { 
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
      	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
    }

    smallsol = diag_inv * smallrhs;

    for (auto k : Range(dofnrs))
      { bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k); }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_subtractU_saveL (double omega, FlatVector<TV> smallsol, FlatVector<TV> bigsol,
									   FlatVector<TV> smallrhs, FlatVector<TV> bigrhs, FlatVector<TV> bigres) const
  {
    if (omega != 1.0)
      { throw Exception("OMEGA does not work yet..."); }
    // current res: b-Lx in rhs
    int nds = dofnrs.Size();
    for (auto k : Range(dofnrs)) { // b-Lx
      smallrhs(k) = bigrhs(dofnrs[k]);
      // -Ux
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ smallrhs(k) -= vals[l] * bigsol(cols[l]); }
      bigres(dofnrs[k]) = 0.0; // order important: bigrhs/bigres can be same vector
    }
    smallsol = diag_inv * smallrhs;
    TV tmp;
    for (auto k : Range(dofnrs)) {
      tmp = bigsol(dofnrs[k]);
      bigsol(dofnrs[k]) = (1-omega) * bigsol(dofnrs[k]) + omega * smallsol(k);
      smallsol(k) = bigsol(dofnrs[k]) - tmp;
    }
    // res -= U (x_new-x)
    for (auto k : Range(dofnrs))
      for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	{ bigres(cols[l]) -= vals[l] * smallsol[k]; }
  } // BSBlock::RichardsonUpdate_FW_zig


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: RichardsonUpdate_RES (double omega, FlatVector<TV> smallupdate, FlatVector<TV> bigsol,
							       FlatVector<TV> smallres, FlatVector<TV> bigres) const
  {
    int nds = dofnrs.Size();
    // static Timer t("RUP"); RegionTimer rt(t);
    for (auto k : Range(dofnrs))
      { smallres(k) = bigres(dofnrs[k]); }
    smallupdate = omega * diag_inv * smallres;
    bigsol(dofnrs) += smallupdate;
    for (auto k : Range(dofnrs)) {
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ bigres(cols[l]) -= Trans(vals[l]) * smallupdate(k); }
    }
    // we know that new res (for omega 1 and no md!!) is (mdadd - U) (x_new - x)
    // but I do not think we can use that here without having L/U stored seperately
    // also: does PINV have an effect??
    if (LU) { // L and U part stored seperately
      for (auto k : Range(dofnrs))
	for (auto l : Range(firsti[nds+k], firsti[nds+k+1]))
	  { bigres(cols[l]) -= Trans(vals[l]) * smallupdate(k); }
    }
    bigres(dofnrs) -= diag * smallupdate;
    // this version only works for !md, !pinv
    // for (auto k : Range(dofnrs))
    // { bigres(dofnrs[k]) *= (1.0 - omega); }
  } // BSBlock::RichardsonUpdate_RES


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: MultAdd (double scal, FlatVector<TV> xsmall, FlatVector<TV> xlarge,
						  FlatVector<TV> ysmall, FlatVector<TV> ylarge) const
  {
    xsmall = xlarge(dofnrs);
    // ylarge(dofnrs) += scal * diag_inv * xsmall;
    // for (auto k : Range(dofnrs))
      // { xsmall(k) = xlarge(dofnrs[k]); }
    ysmall = diag_inv * xsmall;
    ylarge(dofnrs) += scal * ysmall;
    // for (auto k : Range(dofnrs))
      // { ylarge(dofnrs[k]) += scal * ysmall(k); }
  } // BSBlock::MultAdd


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: Mult (FlatVector<TV> xsmall, FlatVector<TV> xlarge,
					       FlatVector<TV> ysmall, FlatVector<TV> ylarge) const
  {
    for (auto k : Range(dofnrs))
      { xsmall(k) = xlarge(dofnrs[k]); }
    ysmall = diag_inv * xsmall;
    for (auto k : Range(dofnrs))
      { ylarge(dofnrs[k]) = ysmall(k); }
  } // BSBlock::Mult


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: MultAdd_mat (double scal, FlatVector<TV> xsmall, FlatVector<TV> xlarge,
  						      FlatVector<TV> ysmall, FlatVector<TV> ylarge) const
  {
    // static Timer t("MAM"); RegionTimer rt(t);
    for (auto k : Range(dofnrs)) {
      xsmall(k) = xlarge(dofnrs[k]);
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ ylarge(dofnrs[k]) += scal * vals[l] * xlarge(cols[l]); }
    }
    if (LU) { // L and U part stored seperately
      int nds = dofnrs.Size();
      for (auto k : Range(dofnrs))
	for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	  { ylarge(dofnrs[k]) += scal * vals[l] * xlarge(cols[l]); }
    }
    ylarge(dofnrs) += scal * diag * xsmall;
  } // BSBlock::MultAdd


  template<class TM>
  INLINE void BSmoother2<TM>::BSBlock :: Mult_mat (FlatVector<TV> xsmall, FlatVector<TV> xlarge,
  						   FlatVector<TV> ysmall, FlatVector<TV> ylarge) const
  {
    for (auto k : Range(dofnrs)) {
      xsmall(k) = xlarge(dofnrs[k]);
      ylarge(dofnrs[k]) = 0.0;
      for (auto l : Range(firsti[k], firsti[k+1]))
	{ ylarge(dofnrs[k]) += vals[l] * xlarge(cols[l]); }
    }
    if (LU) { // L and U part stored seperately
      int nds = dofnrs.Size();
      for (auto k : Range(dofnrs))
	for (auto l : Range(firsti[nds+k], firsti[nds+k+1])) // <- firsti offset for U
	  { ylarge(dofnrs[k]) += vals[l] * xlarge(cols[l]); }
    }
    ylarge(dofnrs) += diag * xsmall;
    // for (auto k : Range(dofnrs))
    // { ylarge(dofnrs[k]) += ngbla::InnerProduct(diag.Row(k), xsmall); }
  } // BSBlock::Mult

  /** END BSBlock **/


  /** BSmoother2 **/

  template<class TM> template<class TLAM>
  INLINE void BSmoother2<TM> :: IterateBlocks (FlatArray<int> groups, bool reverse, TLAM smooth_block) const
  {
    VectorMem<100,TV> hxmax(maxbs);
    VectorMem<100,TV> hymax(maxbs);

    int sg = groups.Size();

    for (auto k : Range(groups))
    {
      if (reverse)
      {
        int group_num = groups[sg-1]-k;
        int finext = fi_blocks[group_num+1], fifirst = fi_blocks[group_num];

        for (int block_num = finext-1; block_num > fifirst; block_num--)
        {
          blocks[block_num - 1].Prefetch();
          smooth_block(block_num, hxmax, hymax);
        }

        if (finext > fifirst)
          { smooth_block(fifirst, hxmax, hymax); }
      }
      else
      {
        int grk = groups[k];

        for (auto block_num : Range(fi_blocks[grk], fi_blocks[grk+1]))
        {
          smooth_block(block_num, hxmax, hymax);
        }
      }
    }

  } // BSmoother2::IterateBlocks


  template<class TM>
  INLINE void BSmoother2<TM> :: SmoothWO (FlatArray<int> groups, BaseVector & x, const BaseVector & b,
					  BaseVector &res, int steps, bool res_updated,
					  bool update_res, bool x_zero, bool reverse, bool symm) const
  {
    // cout << " SmoothWO " << res_updated << " " << update_res << " " << x_zero << " " << reverse << endl;
    if (res_updated && update_res) // symmetric smooting done by SmoothSimple/SmoothRESSimple
      { SmoothRESSimple(groups, x, res, steps, reverse, symm); }
    else {
      SmoothSimple(groups, x, b, steps, reverse, symm);
      if (update_res)
	{ res = b - *(GetAMatrix()) * x; }
    }
  } // BSmoother2::SmoothWO


  template<class TM>
  INLINE void BSmoother2<TM> :: SmoothSimple (FlatArray<int> groups, BaseVector & x, const BaseVector & b, int steps, bool reverse, bool symm) const
  {
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto smooth_lam = [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
      int n = blocks[block_nr].dofnrs.Size();
      FlatVector<TV> hx = hxmax.Range(0, n);
      FlatVector<TV> hb = hymax.Range(0, n);
      blocks[block_nr].RichardsonUpdate(1.0, hx, fx, hb, fb);
    };
    for (int step_nr : Range(steps)) {
      IterateBlocks(groups, reverse, smooth_lam);
      if (symm)
      	{ IterateBlocks(groups, !reverse, smooth_lam); }
    }
  } //BSmoother2::SmoothSimple


  template<class TM>
  INLINE void BSmoother2<TM> :: SmoothRESSimple (FlatArray<int> groups, BaseVector & x, BaseVector & res, int steps, bool reverse, bool symm) const
  {
    auto fx = x.FV<TV>();
    auto fr = res.FV<TV>();
    auto lam_smooth = [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
      int n = blocks[block_nr].dofnrs.Size();
      FlatVector<TV> hx = hxmax.Range(0, n);
      FlatVector<TV> hr = hymax.Range(0, n);
      blocks[block_nr].RichardsonUpdate_RES(1.0, hx, fx, hr, fr);
    };
    for (int step_nr : Range(steps)) {
      IterateBlocks(groups, reverse, lam_smooth);
      if (symm)
	{ IterateBlocks(groups, !reverse, lam_smooth); }
    }
  } // BSmoother2::SmoothRESSimple


  template<class TM>
  INLINE void
  BSmoother2<TM>::
  SmoothNO (FlatArray<int> groups, BaseVector &x, const BaseVector &b,
					  BaseVector &res, int steps, bool res_updated,
					  bool update_res, bool x_zero, bool reverse, bool symm) const
  {
    // cout << " SmoothNO " << res_updated << " " << update_res << " " << x_zero << " " << reverse << endl;
    if (res_updated)
    { // smooth with "res" as RHS and ZIG for update
      if (update_res)
      { // writes res-A*update = b-Ax_new into "res" vector
        *myresb = res;
        // for omega!=1, need to zero out update-vec
        //   even then, not really sure this works?
        if (symm)
          { Smooth_SYMM_CR(groups, *update, *myresb, res, steps, true); }
        else
          { Smooth_CR(groups, *update, *myresb, res, steps, true, reverse); }
      }
      else
      { // do not calc updated residuum
        *update = 0;

        if (symm)
          { Smooth_SYMM (groups, *update, res, *myresb, steps, true); } // use myresb as working vector
        else
          { Smooth (groups, *update, res, steps, false, reverse); }
      }
      x += *update;
    }
    else
    { // no up-to-date RES. smooth with RHS, update RES if necessary!
      if (symm)
	      { Smooth_SYMM (groups, x, b, res, steps, x_zero); } // use res as working vector
      else
      	{ Smooth (groups, x, b, steps, x_zero, reverse); }
      if (update_res)
      	{ res = b - (*GetAMatrix()) * x; }
    }
  } // BSmoother2::Smooth


  template<class TM>
  INLINE void BSmoother2<TM> :: Smooth (FlatArray<int> groups, BaseVector & x, const BaseVector & b,
					int steps, bool zig, bool reverse) const
  {
    /** Smooth FW/BW **/
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    for (auto step : Range(steps)) {
      IterateBlocks(groups, reverse, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hb = hymax.Range(0, n);
	  if (zig) { // zero initial guess - save U/L part of residuum
	    if (reverse)
	      { blocks[block_nr].RichardsonUpdate_FW_zig(1.0, hx, fx, hb, fb); }
	    else
	      { blocks[block_nr].RichardsonUpdate_BW_zig(1.0, hx, fx, hb, fb); }
	  }
	  else
	    { blocks[block_nr].RichardsonUpdate(1.0, hx, fx, hb, fb); }
	});
      zig = false; // only ZIG in INITIAL loop
    }
  } // BSmoother2<TM>::Smooth


  INLINE Timer<TTracing, TTiming>& SCR_thack () { static Timer t("BSmoother2::Smooth_CR "); return t; }
  template<class TM>
  INLINE void
  BSmoother2<TM>::
  Smooth_CR (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					   int steps, bool zig, bool reverse) const
  {
    /** Smooth FW/BW, calc updated RES **/
    RegionTimer rt(SCR_thack());
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>(); // write new res into fr
    for (auto step : Range(steps)) {
      IterateBlocks(groups, reverse, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hr = hymax.Range(0, n);
	  // saveLDU -> update all res
	  if (zig) {
	    if (reverse)
	      { blocks[block_nr].RichardsonUpdate_BW_zig_saveL(1.0, hx, fx, hr, fb, fr); }
	    else
	      { blocks[block_nr].RichardsonUpdate_FW_zig_saveU(1.0, hx, fx, hr, fb, fr); }
	  }
	  else
	    if (reverse)
	      { blocks[block_nr].RichardsonUpdate_saveL(1.0, hx, fx, hr, fb, fr);}
	    else
	      { blocks[block_nr].RichardsonUpdate_saveU(1.0, hx, fx, hr, fb, fr);}
	});
      zig = false; // only ZIG in INITIAL loop
    }
  } // BSmoother2<TM>::Smooth_CR


  template<class TM>
  INLINE void BSmoother2<TM> :: Smooth_SYMM (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					     int steps, bool zig) const
  {
    /** Smooth FW+BW **/
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>(); //
    for (auto step : Range(steps)) {
      // FW step: save b-Lx_mid in "res"
      IterateBlocks(groups, false, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hr = hymax.Range(0, n);
	  if (zig)
	    { blocks[block_nr].RichardsonUpdate_FW_zig_savebL(1.0, hx, fx, hr, fb, fr); }
	  else
	    { blocks[block_nr].RichardsonUpdate_savebL(1.0, hx, fx, hr, fb, fr); }
	});
      // BW step: use b-Lx_mid as RHS with with "zero initial guess (=skip Lx)" in back smoothing
      IterateBlocks(groups, true, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hr = hymax.Range(0, n);
	  blocks[block_nr].RichardsonUpdate_BW_zig(1.0, hx, fx, hr, fr);
	});
      zig = false; // only ZIG in INITIAL loop
    }
  } // BSmoother2<TM>::Smooth_SYMM


  template<class TM>
  INLINE void
  BSmoother2<TM>::
  Smooth_SYMM_CR (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
      						int steps, bool zig) const
  {
    /** Smooth FW+BW, calc updated RES. **/
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>(); //
    for (auto step : Range(steps)) {
      // FW step: save b-Lx_mid
      IterateBlocks(groups, false, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hr = hymax.Range(0, n);
	  if (zig)
	    { blocks[block_nr].RichardsonUpdate_FW_zig_savebL(1.0, hx, fx, hr, fb, fr); }
	  else
	    { blocks[block_nr].RichardsonUpdate_savebL(1.0, hx, fx, hr, fb, fr); }
	});
      // BW step: use b-Lx_mid as RHS with with "zero initial guess (=skip Lx)" in back smoothing
      IterateBlocks(groups, true, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hr = hymax.Range(0, n);
	  // v-Lx is in res ->  as rhs
	  blocks[block_nr].RichardsonUpdate_subtractU_saveL(1.0, hx, fx, hr, fr, fr);
	});
      zig = false; // only ZIG in INITIAL loop
    }
  } // BSmoother2<TM>::Smooth_SYMM_CR


  INLINE Timer<TTracing, TTiming>& SSL_thack () { static Timer t("BSmoother2::Smooth_subL"); return t; }
  template<class TM>
  INLINE void BSmoother2<TM> :: Smooth_savebL (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res,
					       bool zig) const
  {
    /** FW step, save b-Lx in res **/
    RegionTimer rt(SSL_thack());
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>();
    IterateBlocks(groups, false, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
      int n = blocks[block_nr].dofnrs.Size();
      FlatVector<TV> hx = hxmax.Range(0, n);
      FlatVector<TV> hr = hymax.Range(0, n);
      if (zig)
        { blocks[block_nr].RichardsonUpdate_FW_zig_savebL(1.0, hx, fx, hr, fb, fr); }
      else
        { blocks[block_nr].RichardsonUpdate_savebL(1.0, hx, fx, hr, fb, fr); }
    });
  } // BSmoother2<TM> :: Smooth_savebL


  INLINE Timer<TTracing, TTiming>& SBL_thack () { static Timer t("SmoothBack_usebL"); return t; }
  template<class TM>
  INLINE void BSmoother2<TM> :: SmoothBack_usebL (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res) const
  {
    /** BW step, assume res==b-Lx, use it as RHS for a backward step with ZIG **/
    RegionTimer rt(SBL_thack());
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>();
    IterateBlocks(groups, true, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE
    {
      int n = blocks[block_nr].dofnrs.Size();
      FlatVector<TV> hx = hxmax.Range(0, n);
      FlatVector<TV> hr = hymax.Range(0, n);
      blocks[block_nr].RichardsonUpdate_BW_zig(1.0, hx, fx, hr, fr);
    });
  } // BSmoother2<TM> :: SmoothBack_usebL


  INLINE Timer<TTracing, TTiming>& SBuLsL_thack () { static Timer t("SmoothBack_usebL_saveL"); return t; }
  template<class TM>
  INLINE void BSmoother2<TM> :: SmoothBack_usebL_saveL (FlatArray<int> groups, BaseVector & x, const BaseVector & b, BaseVector & res) const
  {
    // THIS IS WRONG, IT DOES NOT SAVE ANYTHING TO "RES"!!
    /** BW step, assume res==b-Lx, use it as RHS for a backward step with ZIG **/
    RegionTimer rt(SBuLsL_thack());
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    auto fr = res.FV<TV>();
    IterateBlocks(groups, true, [&](auto block_nr, auto & hxmax, auto & hymax) LAMBDA_INLINE {
	int n = blocks[block_nr].dofnrs.Size();
	FlatVector<TV> hx = hxmax.Range(0, n);
	FlatVector<TV> hr = hymax.Range(0, n);
	blocks[block_nr].RichardsonUpdate_BW_zig(1.0, hx, fx, hr, fr);
      });
  } // BSmoother2<TM> :: SmoothBack_usebL_saveL

  /** END BSmoother2 **/

} // namespace amg

#endif //  FILE_AMG_BLOCKSMOOTHER_LOC_IMPL_HPP
