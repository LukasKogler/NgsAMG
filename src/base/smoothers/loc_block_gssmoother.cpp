#define FILE_LOC_BGSS_CPP

#include "loc_block_gssmoother.hpp"
#include "loc_block_gssmoother_impl.hpp"

#include <utils_io.hpp>
#include <utils_denseLA.hpp>

namespace amg
{

  /** BSmoother2 **/

  template<class TM>
  BSmoother2<TM> :: BSmoother2 (shared_ptr<SparseMatrix<TM>> _spmat,  Table<int> && _blocks,
				bool _parallel, bool _use_sl2, bool _pinv, bool _blocks_no,
				FlatArray<TM> md)
    : BaseSmoother(nullptr, nullptr), parallel(_parallel), use_sl2(_use_sl2), pinv(_pinv), blocks_no(_blocks_no)
  {
    static Timer t("BSmoother2"); RegionTimer rt(t);

    Array<Table<int>> block_array(1);
    block_array[0] = std::move(_blocks);

    SetUp(_spmat, block_array, md);
  } // BSmoother2::BSmoother2(..)


  template<class TM>
  BSmoother2<TM> :: BSmoother2 (shared_ptr<SparseMatrix<TM>> _spmat,  FlatArray<Table<int>> _block_array,
				bool _parallel, bool _use_sl2, bool _pinv, bool _blocks_no,
				FlatArray<TM> md)
    : BaseSmoother(nullptr, nullptr), parallel(_parallel), use_sl2(_use_sl2), pinv(_pinv), blocks_no(_blocks_no)
  {
    static Timer t("BSmoother2"); RegionTimer rt(t);

    SetUp(_spmat, _block_array, md);
  } // BSmoother2::BSmoother2(..)


  template<class TM>
  void BSmoother2<TM> :: SetUp (shared_ptr<SparseMatrix<TM>> _spmat, FlatArray<Table<int>> _block_array,
				FlatArray<TM> md)
  {
    // if (parallel)
    //   { throw Exception("New BS parallel TODO!!"); }

    const auto & A(*_spmat);
    height = A.Height();
    maxbs = 0;

    bool have_md = md.Size() > 0;

    auto block_array = std::move(_block_array);
    n_groups = block_array.Size();
    n_blocks_group.SetSize(n_groups);
    fi_blocks.SetSize(n_groups+1); fi_blocks[0] = 0;
    for (auto k : Range(n_groups)) {
      n_blocks_group[k] = block_array[k].Size();
      fi_blocks[1+k] = fi_blocks[k] + n_blocks_group[k];
    }
    // cout << " NBC " << endl << n_blocks_group << endl;
    // cout << " fi_blocks " << endl << fi_blocks << endl;
    size_t n_blocks_tot = std::accumulate(n_blocks_group.begin(), n_blocks_group.end(),
					  size_t(0), [](size_t a, size_t b) { return a + b; });
    blocks.SetSize(n_blocks_tot);

    allgroups.SetSize(n_groups);
    for (auto k : Range(allgroups))
      { allgroups[k] = k; }

    /** color and sort blocks (WITHIN GROUPS) **/
    Array<int> block_nrs(n_blocks_tot);
    for (auto k : Range(block_nrs))
      { block_nrs[k] = k; }

    /** count block memory **/
    tuple<size_t, size_t> tm_tot(0,0);
    for (auto group : Range(n_groups)) {
      auto [ tm_inds_group, tm_vals_group ] =
	ParallelReduce (n_blocks_tot,
			[&](size_t k) {
			  int group_nr = merge_pos_in_sorted_array(size_t(block_nrs[k]), fi_blocks) - 1;
			  FlatArray<int> bds = block_array[group_nr][block_nrs[k] - fi_blocks[group_nr]];
			  return blocks[k].GetAllocSize(A, bds, blocks_no, have_md);
			},
			[](tuple<int, int> a, tuple<int, int> b) -> tuple<int, int>
			{ return make_tuple( get<0>(a) + get<0>(b), get<1>(a) + get<1>(b) ); },
			make_tuple(0, 0));
      tm_tot = make_tuple(get<0>(tm_tot) + tm_inds_group,
			  get<1>(tm_tot) + tm_vals_group);
    }
    buffer_inds.SetSize(get<0>(tm_tot));
    buffer_vals.SetSize(get<1>(tm_tot));

    // cout << " tot buf size " << get<0>(tm_tot) << " " << get<1>(tm_tot) << endl;

    /** assign block memory **/
    size_t cnti = 0, cntv = 0;
    // LocalHeap need a bit of extra memory because of alignment. I do not like it ...
    // int bufs = tm_inds*sizeof(int) + tm_vals*sizeof(TM) + 6*LocalHeap::ALIGN*n_blocks;
    // buffer_lh = LocalHeap(bufs, "KAAAAAAAAAAAAN");
    int* ptr_i = buffer_inds.Data();
    TM* ptr_v = buffer_vals.Data();
    for (auto group_nr : Range(n_groups)) {
      for (auto k : Range(n_blocks_group[group_nr])) {
	int blocknr = fi_blocks[group_nr] + k;
	blocks[blocknr].Alloc(ptr_i, ptr_v);
	// blocks[blocknr].Alloc(buffer_lh);
	maxbs = max(maxbs, blocks[k].dofnrs.Size());
      }
    }

    // cout << " maxbs " << endl << maxbs << endl;

    Array<int> dof2block;
    if (blocks_no) {
      bool ok = true;
      dof2block.SetSize(height); dof2block = -1;
      for (auto group_nr : Range(n_groups)) {
	auto & block_dofs = block_array[group_nr];
	for (auto k : Range(block_dofs)) {
	  int blocknr = fi_blocks[group_nr] + k;
	  for ( auto dof : block_dofs[k] ) {
	    if (dof2block[dof] != -1)
	      { ok = false; break; }
	    dof2block[dof] = blocknr;
	  }
	}
      }
      if (!ok)
	{ throw Exception("Blocks do overlap, but are assumbed to be non-overlapping!!"); }
    }

    /** write data into blocks **/
    SharedLoop2 sl(n_blocks_tot);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 constexpr int BS = ngbla::Height<TM>();
	 size_t hs = max(size_t(10*1024*1024), size_t(6*sqr(sizeof(double)*maxbs*ngbla::Height<TM>())));
	 LocalHeap lh (hs, "AAA", false);
	 for (int kb : sl) {
	   auto block_nr = block_nrs[kb];
	   int group_nr = merge_pos_in_sorted_array(size_t(block_nr), fi_blocks) - 1;
	   FlatArray<int> bds = block_array[group_nr][block_nr - fi_blocks[group_nr]];
	   auto & block = blocks[kb];
	   if (blocks_no) {
	     blocks[kb].SetLUFromSPMat (kb, dof2block, A, bds, lh, this->pinv, md);
	   }
	   else
	     { blocks[kb].SetFromSPMat (A, bds, lh, this->pinv, md); }
	 }
       } );

    // cout << " have set all " << endl;

    /** working vectors **/
    myresb.AssignPointer(CreateColVector());
    update.AssignPointer(CreateColVector());
  } // BSmoother2::SetUp


  INLINE Timer<TTracing, TTiming>& S_thack () { static Timer t("BSmoother2::Smooth"); return t; }
  template<class TM>
  void BSmoother2<TM> :: Smooth (BaseVector &x, const BaseVector &b,
				 BaseVector &res, bool res_updated,
				 bool update_res, bool x_zero) const
  {
    RegionTimer rt(S_thack());
    if (blocks_no)
      { SmoothNO(allgroups, x, b, res, 1, res_updated, update_res, x_zero, false, false); }
    else
      { SmoothWO(allgroups, x, b, res, 1, res_updated, update_res, x_zero, false, false); }
  } // BSmoother2::Smooth


  INLINE Timer<TTracing, TTiming>& SK_thack () { static Timer t("BSmoother2::SmoothK"); return t; }
  template<class TM>
  void BSmoother2<TM> :: SmoothK (int steps, BaseVector &x, const BaseVector &b,
				  BaseVector &res, bool res_updated,
				  bool update_res, bool x_zero) const
  {
    RegionTimer rt(SK_thack());
    if (blocks_no)
      { SmoothNO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, false, false); }
    else
      { SmoothWO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, false, false); }
  } // BSmoother2::SmoothK


  INLINE Timer<TTracing, TTiming>& SSK_thack () { static Timer t("BSmoother2::SmoothSymmK"); return t; }
  template<class TM>
  void BSmoother2<TM> :: SmoothSymmK (int steps, BaseVector &x, const BaseVector &b,
				      BaseVector &res, bool res_updated,
				      bool update_res, bool x_zero) const
  {
    RegionTimer rt(SSK_thack());
    if (blocks_no)
      { SmoothNO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, false, true); }
    else
      { SmoothWO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, false, true); }
  } // BSmoother2::SmoothK


  INLINE Timer<TTracing, TTiming>& SB_thack () { static Timer t("BSmoother2::SmoothBack"); return t; }
  template<class TM>
  void BSmoother2<TM> :: SmoothBack (BaseVector &x, const BaseVector &b,
				     BaseVector &res, bool res_updated,
				     bool update_res, bool x_zero) const
  {
    RegionTimer rt(SB_thack());
    if (blocks_no)
      { SmoothNO(allgroups, x, b, res, 1, res_updated, update_res, x_zero, true, false); }
    else
      { SmoothWO(allgroups, x, b, res, 1, res_updated, update_res, x_zero, true, false); }
  } // BSmoother2::SmoothBack


  INLINE Timer<TTracing, TTiming>& SBK_thack () { static Timer t("BSmoother2::SmoothBackK"); return t; }
  template<class TM>
  void BSmoother2<TM> :: SmoothBackK (int steps, BaseVector &x, const BaseVector &b,
				      BaseVector &res, bool res_updated,
				      bool update_res, bool x_zero) const
  {
    RegionTimer rt(SBK_thack());
    if (blocks_no)
      { SmoothNO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, true, false); }
    else
      { SmoothWO(allgroups, x, b, res, steps, res_updated, update_res, x_zero, true, false); }
  } // BSmoother2::SmoothBackK


  INLINE Timer<TTracing, TTiming>& SG_thack () { static Timer t("BSmoother2::SmoothGroups"); return t; }
  template<class TM>
  void BSmoother2<TM> :: SmoothGroups (FlatArray<int> groups, int steps, BaseVector &x, const BaseVector &b,
				       BaseVector &res, bool res_updated,
				       bool update_res, bool x_zero, bool reverse, bool symm) const
  {
    RegionTimer rt(SG_thack());
    if (blocks_no)
      { SmoothNO(groups, x, b, res, steps, res_updated, update_res, x_zero, true, symm); }
    else
      { SmoothWO(groups, x, b, res, steps, res_updated, update_res, x_zero, true, symm); }
  } // BSmoother2<TM> :: SmoothGroups


  template<class TM>
  void BSmoother2<TM> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    // IterateBlocks(false, [&](auto block_nr, auto & hxmax, auto & hymax) {
    auto fx = x.FV<TV>();
    auto fb = b.FV<TV>();
    ParallelForRange (blocks.Size(), [&] ( IntRange r ) {
    	VectorMem<100,TV> hxmax(maxbs);
    	VectorMem<100,TV> hbmax(maxbs);
    	for (auto block_nr : r) {
      	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hb = hbmax.Range(0, n);
	  blocks[block_nr].MultAdd(s, hb, fb, hx, fx);
	}
      });
  } // BSmoother2::MultAdd


  template<class TM>
  void BSmoother2<TM> :: Mult_Sysmat (const BaseVector & x, BaseVector & y) const
  {
    static Timer t("Mult_Sysmat"); RegionTimer rt(t);
    if (!blocks_no) { // would overwrite y entries
      y.FV<typename mat_traits<TV>::TSCAL>() = 0.0;
      MultAdd_Sysmat(1.0, x, y);
      return;
    }
    auto fx = x.FV<TV>();
    auto fy = y.FV<TV>();
    ParallelForRange(blocks.Size(), [&] ( IntRange r ) {
	VectorMem<100,TV> hxmax(maxbs);
	VectorMem<100,TV> hymax(maxbs);
	for (auto block_nr : r) {
	  int n = blocks[block_nr].dofnrs.Size();
	  FlatVector<TV> hx = hxmax.Range(0, n);
	  FlatVector<TV> hy = hymax.Range(0, n);
	  blocks[block_nr].Mult_mat(hx, fx, hy, fy);
	}
      });
  } // BSmoother2::Mult_Sysmat


  template<class TM>
  void BSmoother2<TM> :: MultAdd_Sysmat (double s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t("MultAdd_Sysmat"); RegionTimer rt(t);
    auto fx = x.FV<TV>();
    auto fy = y.FV<TV>();
    if (blocks_no) {
      ParallelForRange (blocks.Size(), [&] ( IntRange r ) {
	  VectorMem<100,TV> hxmax(maxbs);
	  VectorMem<100,TV> hymax(maxbs);
	  for (auto block_nr : r) {
	    int n = blocks[block_nr].dofnrs.Size();
	    FlatVector<TV> hx = hxmax.Range(0, n);
	    FlatVector<TV> hy = hymax.Range(0, n);
	    blocks[block_nr].MultAdd_mat(s, hx, fx, hy, fy);
	  }
	});
    }
    else {
      IterateBlocks(allgroups, false, [&](auto block_nr, auto & hx, auto & hy) {
	  blocks[block_nr].MultAdd_mat(s, hx, fx, hy, fy);
	});
    }
  } // BSmoother2::MultAdd


  template<class TM>
  shared_ptr<BaseMatrix> BSmoother2<TM> :: GetAMatrix () const
  {
    return make_shared<BSmoother2SysMat<TM>>(const_pointer_cast<BSmoother2<TM>>(dynamic_pointer_cast<const BSmoother2<TM>>(shared_from_this())));
  } // BSmoother2::GetAMatrix

INLINE void printEvals (FlatMatrix<double> M, LocalHeap & lh, std::string const &prefix = "", ostream &os = std::cout)
{
  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(M, evals, evecs);

  os << prefix << "evals "; prow(evals, os); os << endl;

} // CalcPseudoInverseWithTol


template<int N>
void printEvals (FlatMatrix<Mat<N, N, double>> mat, LocalHeap & lh, std::string const &prefix = "", ostream &os = std::cout)
{
  auto const H = mat.Height();
  auto const W = mat.Width();

  FlatMatrix<double> B(mat.Height() * N, mat.Width() * N, lh);

  ToFlat<N>(mat, B);

  printEvals(B, lh, prefix, os);
}


template<class TM>
void
BSmoother2<TM>::BSBlock ::
PrintTo (ostream & os, string prefix) const
{
  LocalHeap lh(3*1024*1024, "whatever");

  std::string const prefix2 = prefix  + "  ";

  os << prefix << "height = " << dofnrs.Size() << endl;
  os << prefix << "DOFNrs = "; prow(dofnrs, os); os << endl; 
  os << prefix << " diag: " << endl;
  // print_tm_mat(os, diag);
  printEvals(diag, lh, prefix, os);


  os << prefix << " diag INV: " << endl;
  // print_tm_mat(os, diag_inv);
  printEvals(diag_inv, lh, prefix, os);
}


template<class TM>
void
BSmoother2<TM> ::
PrintTo (ostream & os, string prefix) const
{
  std::string const prefix2 = prefix  + "  ";
  std::string const prefix3 = prefix2 + "  ";

  os << prefix << " BSmoother2, BS = " << ngbla::Height<TM>() << ", H = " << GetAMatrix()->Height() << std::endl;
  os << prefix << "    using piv = " << pinv << std::endl;
  os << prefix << " Have " << blocks.Size() << " blocks, maxbs = " << maxbs << endl;
  for (auto k : Range(blocks))
  {
    os << prefix2 << " Blcok #" << k << "/" << blocks.Size() << ": " << endl;
    blocks[k].PrintTo(os, prefix3);
  }
  os << endl;
}



  /** END BSmoother2 **/

  template class BSmoother2<double>;
  template class BSmoother2<Mat<2,2,double>>;
  template class BSmoother2<Mat<3,3,double>>;
#ifdef ELASTICITY
  template class BSmoother2<Mat<6,6,double>>;
#endif
} // namespace amg
