
#define FILE_AMGSM2_CPP

#ifdef USE_TAU
#include <Profile/Profiler.h>
// #include "TAU.h"
#endif

#include "amg.hpp"

namespace amg
{


  /** Local Gauss-Seidel **/


  template<class TM>
  GSS2<TM> :: GSS2 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset, FlatArray<TM> add_diag)
    : spmat(mat), freedofs(subset)
  {
    H = spmat->Height();
    dinv.SetSize (H); 
    const auto& A(*spmat);
    ParallelFor (H, [&](size_t i) {
  	if (!freedofs || freedofs->Test(i)) {
  	  dinv[i] = A(i,i);
  	  CalcInverse (dinv[i]);
  	}
  	else
  	  { dinv[i] = TM(0.0); }
      });
  }


  template<class TM>
  void GSS2<TM> :: SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const
  {
#ifdef USE_TAU
    TAU_PROFILE("SmoothRHSInternal", TAU_CT(*this), TAU_DEFAULT);
#endif

    static Timer t(string("GSS2<bs=")+to_string(BS())+">::SmoothRHS");
    RegionTimer rt(t);

    const auto& A(*spmat);

    auto fds = freedofs.get();

    auto fvx = x.FV<TV>();
    auto fvb = b.FV<TV>();

    auto up_row = [&](auto rownr) LAMBDA_INLINE {
      auto r = fvb(rownr) - A.RowTimesVector(rownr, fvx);
      fvx(rownr) += dinv[rownr] * r;
    };

    double tl = MPI_Wtime();
    if (!backwards) {
      for (size_t rownr = 0; rownr<H; rownr++)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  // if (rownr + 1 < H)
	  //   A.PrefetchRow(rownr+1);
	  up_row(rownr);
	}
    }
    else {
      for (int rownr = H-1; rownr>=0; rownr--)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  // if (rownr > 1)
	  //   A.PrefetchRow(rownr-1);
	  up_row(rownr);
	}
    }
    tl = MPI_Wtime() - tl;

    cout << " rows, K rows / sec : " << H << ", " << double(H) / 1000 / tl << endl;
    
  } // SmoothRHSInternal


  template<class TM>
  void GSS2<TM> :: SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const
  {
#ifdef USE_TAU
    TAU_PROFILE("SmoothRESInternal", TAU_CT(*this), TAU_DEFAULT);
#endif

    static Timer t(string("GSS2<bs=")+to_string(BS())+">::SmoothRES");
    RegionTimer rt(t);

    const auto& A(*spmat);

    auto fds = freedofs.get();

    auto fvx = x.FV<TV>();
    auto fvr = res.FV<TV>();

    auto up_row = [&](auto rownr) LAMBDA_INLINE {
      auto w = -dinv[rownr] * fvr(rownr);
      A.AddRowTransToVector(rownr, w, fvr);
      fvx(rownr) -= w;
    };
    
    if (!backwards) {
      for (size_t rownr = 0; rownr<H; rownr++)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  up_row(rownr);
	}
    }
    else {
      for (int rownr = H-1; rownr>=0; rownr--)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  up_row(rownr);
	}
    }

  } // SmoothRESInternal


  /** HybridMatrix **/


  template<class TM>
  void HybridMatrix<TM> :: MultAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultAdd(s, x, y);
    if (S != nullptr)
      { S->MultAdd(s, x, y); }
  }

  template<class TM>
  void HybridMatrix<TM> :: MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultAdd(s, x, y);
    if (S != nullptr)
      { S->MultAdd(s, x, y); }
  }

  template<class TM>
  void HybridMatrix<TM> :: MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultTransAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, x, y);
    if (S != nullptr)
      { S->MultTransAdd(s, x, y); }
  }

  template<class TM>
  void HybridMatrix<TM> :: MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultTransAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, x, y);
    if (S != nullptr)
      { S->MultTransAdd(s, x, y); }
  }


  template<class TM>
  HybridMatrix<TM> :: HybridMatrix (shared_ptr<BaseMatrix> mat)
  {
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat)) {
      // can still have dummy-pardofs (ngs-amg w. NP==1) or no ex-procs (NP==2)
      dummy = false;
      if (auto loc_spmat = dynamic_pointer_cast<SparseMatrix<TM>> (parmat->GetMatrix())) {
	pardofs = parmat->GetParallelDofs();
	SetParallelDofs(pardofs);
	SetUpMats(loc_spmat);
      }
      else
	{ throw Exception("HybridMatrix needs a sparse Matrix!"); }
    }
    else if (auto spmat = dynamic_pointer_cast<SparseMatrix<TM>>(mat)) {
      // actually only a local matrix
      dummy = true;
      M = spmat;
      if (M == nullptr)
	{ throw Exception("u gave me an actual TM mat - WTF am i supposed to do with that??"); }
      S = nullptr;
    }
  }


  template<class TM>
  void HybridMatrix<TM> :: SetUpMats (shared_ptr<SparseMatrix<TM>> anA)
  {
    string cn = string("HybridMatrix<" + to_string(mat_traits<TM>::HEIGHT) + string(">"));
    static Timer t(cn + "::SetUpMats"); RegionTimer rt(t);

    auto& A(*anA);

    auto H = A.Height();

    auto& pds(*pardofs);

    NgsAMG_Comm comm(pds.GetCommunicator());
    auto ex_procs = pds.GetDistantProcs();
    nexp = ex_procs.Size();
    nexp_smaller = 0;
    while (nexp_smaller<nexp && ex_procs[nexp_smaller]<comm.Rank()) nexp_smaller++;
    nexp_larger = nexp - nexp_smaller;
    auto rank = comm.Rank();
    
    if (nexp == 0) {
      // we are still not "dummy", because we are still working with parallel vectors
      M = anA;
      S = nullptr;
      return;
    }

    BitArray mf_dofs(H); mf_dofs.Clear();
    BitArray mf_exd(H); mf_exd.Clear();
    for (auto k:Range(H)) {
      if (pds.IsMasterDof(k)) {
	mf_dofs.Set(k);
	if (pds.GetDistantProcs(k).Size())
	  { mf_exd.Set(k); }
      }
    }

    Table<int> mx_dofs; // for each row: master-dofs for which I want to send/recv the diag-block
    Table<int> mx_loc; // same as mx_dofs, but in ex-dof numbering
    {
      TableCreator<int> cmxd(nexp);
      TableCreator<int> cmxd_loc(nexp);
      for (; !cmxd.Done(); cmxd++ ) {
	for (auto k : Range(nexp)) {
	  auto p = ex_procs[k];
	  auto m = min(p,rank);
	  auto exdofs = pds.GetExchangeDofs(p);
	  for (auto j:Range(exdofs.Size())) {
	    auto d = exdofs[j];
	    auto dps = pds.GetDistantProcs(d);
	    if (dps[0] >= m) { // either I or "p" are master of this DOF
	      cmxd.Add(k,d);
	      cmxd_loc.Add(k,j);
	    }
	  }
	}
	cmxd_loc++;
      }
      mx_dofs = cmxd.MoveTable();
      mx_loc = cmxd_loc.MoveTable();
    }

    /** build A-matrix **/
    typedef SparseMatrixTM<TM> TSPMAT_TM;
    /** Additional mf_exd x mf_exd mat **/
    Array<TSPMAT_TM*> send_add_diag_mats(nexp_smaller);
    Array<MPI_Request> rsdmat (nexp_smaller);
    { // create & send diag-blocks to masters
      static Timer t(cn + "::SetUpMats - create diag"); RegionTimer rt(t);
      auto SPM_DIAG = [&] (auto dofs) {
	auto iterate_coo = [&](auto fun) {
	  for (auto i : Range(dofs.Size())) {
	    auto d = dofs[i];
	    auto ris = A.GetRowIndices(d);
	    auto rvs = A.GetRowValues(d);
	    for (auto j : Range(ris.Size())) {
	      auto pos = find_in_sorted_array(ris[j], dofs);
	      if (pos != -1) {
		fun(i, pos, rvs[j]);
	      }
	    }
	  }
	};
	Array<int> perow(dofs.Size()); perow = 0;
	iterate_coo([&](auto i, auto j, auto val) { perow[i]++; });
	TSPMAT_TM* dspm = new TSPMAT_TM(perow, perow.Size());
	perow = 0;
	iterate_coo([&](auto i, auto j, auto val) {
	    dspm->GetRowIndices(i)[perow[i]] = j;
	    dspm->GetRowValues(i)[perow[i]++] = val;
	  });
	return dspm;
      };
      for (auto kp : Range(nexp_smaller)) {
	send_add_diag_mats[kp] = SPM_DIAG(mx_dofs[kp]);
	rsdmat[kp] = comm.ISend(*send_add_diag_mats[kp], ex_procs[kp], MPI_TAG_AMG);
      }
    } // create & send diag-blocks to masters


    
    Array<shared_ptr<TSPMAT_TM>> recv_mats(nexp_larger);
    { // recv diag-mats
      static Timer t(cn + "::SetUpMats - recv diag"); RegionTimer rt(t);
      for (auto kkp : Range(nexp_larger))
	{
	  comm.Recv(recv_mats[kkp], ex_procs[nexp_smaller + kkp], MPI_TAG_AMG);
	  // cout << kkp << " " << kp << endl;
	  // cout << "mat from " << ex_procs[nexp_smaller + kkp] << endl << *recv_mats[kkp] << endl;
	}
    }
    


    { // merge diag-mats
      static Timer t(cn + "::SetUpMats - merge"); RegionTimer rt(t);

      Array<int> perow(H); perow = 0;
      Array<size_t> at_row(nexp_larger);
      Array<int> row_matis(nexp_larger);
      Array<FlatArray<int>> all_cols;
      Array<int> mrowis(50); mrowis.SetSize0(); // col-inds for a row of orig mat (but have to remove a couple first)

      auto iterate_rowinds = [&](auto fun, bool map_exd) {
	at_row = 0; // restart counting through rows of recv-mats
	for (auto rownr : Range(H)) {
	  if (mf_dofs.Test(rownr)) { // I am master of this dof - merge recved rows with part of original row
	    row_matis.SetSize0(); // which mats I received have this row?
	    if (mf_exd.Test(rownr)) { // local master
	      for (auto kkp : Range(nexp_larger)) {
		auto kp = nexp_smaller + kkp;
		auto exds = mx_dofs[kp];
		if (at_row[kkp] == exds.Size()) continue; // no more rows to take from there
		auto ar = at_row[kkp]; // the next row for that ex-mat
		size_t ar_dof = exds[ar]; // the dof that row belongs to
		if (ar_dof>rownr) continue; // not yet that row's turn
		row_matis.Append(kkp);
		// cout << "row " << at_row[kp] << " from " << kp << " " << kkp << endl;
		at_row[kkp]++;
	      }
	      all_cols.SetSize0(); all_cols.SetSize(1+row_matis.Size()); // otherwise tries to copy FA I think
	      for (auto k:Range(all_cols.Size()-1)) {
		auto kkp = row_matis[k];
		auto kp = nexp_smaller + kkp;
		auto cols = recv_mats[kkp]->GetRowIndices(at_row[kkp]-1);
		if (map_exd) { // !!! <- remap col-nrs of received rows, only do this ONCE!!
		  auto mxd = mx_dofs[kp];
		  for (auto j:Range(cols.Size()))
		    cols[j] = mxd[cols[j]];
		}
		all_cols[k].Assign(cols);
	      }

	      // only master-master (not master-all) goes into M
	      auto aris = A.GetRowIndices(rownr);
	      mrowis.SetSize(aris.Size()); int c = 0;
	      for (auto col : aris)
		if (pardofs->IsMasterDof(col))
		  mrowis[c++] = col;
	      mrowis.SetSize(c);
	      all_cols.Last().Assign(mrowis);

	      // cout << "merge cols: " << endl;
	      // for (auto k : Range(all_cols.Size()))
	      // 	{ cout << k << " || "; prow2(all_cols[k]); cout << endl; }
	      // cout << endl;

	      auto merged_cols = merge_arrays(all_cols, [](const auto&a, const auto &b){return a<b; });

	      // cout << "merged cols: "; prow2(merged_cols); cout << endl;

	      fun(rownr, row_matis, merged_cols);
	    } // mf-exd.Test(rownr);
	    else { // local row - pick out only master-cols
	      auto aris = A.GetRowIndices(rownr);
	      mrowis.SetSize(aris.Size()); int c = 0;
	      for (auto col : aris)
		if (pardofs->IsMasterDof(col))
		  mrowis[c++] = col;
	      mrowis.SetSize(c);
	      fun(rownr, row_matis, mrowis);
	    }
	  } // mf_dofs.Test(rownr)
	}
      };

      iterate_rowinds([&](auto rownr, const auto &matis, const auto &rowis) { perow[rownr] = rowis.Size(); }, true);

      M = make_shared<SparseMatrix<TM>>(perow);

      // cout << "M NZE : " << M->NZE() << endl;

      iterate_rowinds([&](auto rownr, const auto & matis, const auto & rowis) {
	  auto ris = M->GetRowIndices(rownr); ris = rowis;
	  auto rvs = M->GetRowValues(rownr); rvs = 0;
	  // cout << "rownr, rowis: " << rownr << ", "; prow2(rowis); cout << endl;
	  // cout << "rownr, matis: " << rownr << ", "; prow2(matis); cout << endl;
	  // cout << ris.Size() << " " << rvs.Size() << " " << rowis.Size() << endl;
	  auto add_vals = [&](auto cols, auto vals) {
	    for (auto l : Range(cols)) {
	      auto pos = find_in_sorted_array<int>(cols[l], ris);
	      // cout << "look for " << cols[l] << " in "; prow(ris); cout << " -> pos " << pos << endl;
	      if (pos != -1)
		{ rvs[pos] += vals[l]; }
	    }
	  };
	  add_vals(A.GetRowIndices(rownr), A.GetRowValues(rownr));
	  for (auto kkp : matis) {
	    // cout << "row " << at_row[kkp] -1 << " from kkp " << kkp << endl;
	    add_vals(recv_mats[kkp]->GetRowIndices(at_row[kkp]-1),
		     recv_mats[kkp]->GetRowValues(at_row[kkp]-1));
	  }
	}, false);

    } // merge diag-mats


    { /** build S-matrix **/

      static Timer t(cn + "::SetUpMats - S-mat"); RegionTimer rt(t);

      // ATTENTION: explicitely only for symmetric matrices!
      auto master_of = [&](auto k) {
	auto dps = pardofs->GetDistantProcs(k);
	return (dps.Size() && dps[0] < comm.Rank()) ? dps[0] : comm.Rank();
      };
      auto is_loc = [&](auto k)
	{ return pardofs->GetDistantProcs(k).Size() == 0; };
      const auto me = comm.Rank();
      Array<int> perow(A.Height());
      auto iterate_coo = [&](auto fun) { // could be better if we had GetMaxExchangeDof(), or GetExchangeDofs()
      	perow = 0;
      	for (auto rownr : Range(A.Height())) {
      	  auto ris = A.GetRowIndices(rownr);
      	  auto rvs = A.GetRowValues(rownr);
      	  auto mrow = master_of(rownr);
      	  for (auto j : Range(ris)) {
      	    if (master_of(ris[j]) != mrow) {
      	      fun(rownr, ris[j], rvs[j]);
      	    }
      	  }
      	}
      };
      iterate_coo([&](auto i, auto j, auto val) {
	  perow[i]++;
	});
      auto sp_S = make_shared<SparseMatrix<TM>>(perow); S = sp_S;
      iterate_coo([&](auto i, auto j, auto val) {
	  sp_S->GetRowIndices(i)[perow[i]] = j;
	  sp_S->GetRowValues(i)[perow[i]++] = val;
	});
    } // build S-matrix


    {
      static Timer t(cn + "::SetUpMats - finish send"); RegionTimer rt(t);
      MyMPI_WaitAll(rsdmat);
      for (auto kp : Range(nexp_smaller))
	delete send_add_diag_mats[kp];
    }

    rr_gather.SetSize(nexp_larger);
    rr_scatter.SetSize(nexp_smaller);

  } // HybridMatrix :: SetUpMats


  template<class TM>
  void HybridMatrix<TM> :: gather_vec (const BaseVector & vec) const
  {
#ifdef USE_TAU
    TAU_PROFILE("gather_vec", TAU_CT(*this), TAU_DEFAULT);
#endif

    if (dummy)
      { return; }

    ParallelBaseVector * parvec = dynamic_cast_ParallelBaseVector(const_cast<BaseVector*>(&vec));

    // ngs-amg wraps dummy-pardofs around local mats for non-mpi case, but we still apply to local vecs
    if (parvec == nullptr)
      { return; }

    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::gather_vec");
    RegionTimer rt(t);

    if (parvec->GetParallelStatus() == CUMULATED)
      { return; }

    auto ex_procs = paralleldofs->GetDistantProcs();

    for (auto kp : Range(nexp_smaller))
      { parvec->ISend(ex_procs[kp], rr_scatter[kp]); }

    for (auto kkp : Range(nexp_larger))
      { parvec->IRecvVec(ex_procs[nexp_smaller + kkp], rr_gather[kkp]); }

    MyMPI_WaitAll(rr_scatter);

    for (auto j : Range(nexp_larger)) {
      auto kkp = MyMPI_WaitAny(rr_gather);
      parvec->AddRecvValues(ex_procs[nexp_smaller + kkp]);
    }
    
  } // gather_vec


  template<class TM> void
  HybridMatrix<TM> :: scatter_vec (const BaseVector & vec) const
  {
#ifdef USE_TAU
    TAU_PROFILE("scatter_vec", TAU_CT(*this), TAU_DEFAULT);
#endif

    // if (!scatter_done)
    //   { finish_scatter(); }

    if (dummy)
      { return; }

    ParallelBaseVector * parvec = dynamic_cast_ParallelBaseVector(const_cast<BaseVector*>(&vec));

    // ngs-amg wraps dummy-pardofs around local mats for non-mpi case, but we still apply to local vecs
    if (parvec == nullptr)
      { return; }

    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">>::scatter_vec");
    RegionTimer rt(t);

    auto ex_procs = paralleldofs->GetDistantProcs();

    parvec->SetParallelStatus(CUMULATED); parvec->Distribute();

    for (auto kkp : Range(nexp_larger))
      { parvec->ISend(ex_procs[nexp_smaller + kkp], rr_gather[kkp], true); }

    for (auto kp : Range(nexp_smaller))
      { parvec->IRecvVec(ex_procs[kp], rr_scatter[kp]); }

    MyMPI_WaitAll(rr_gather);

    for (auto j : Range(nexp_smaller)) {
      auto kp = MyMPI_WaitAny(rr_scatter);
      parvec->AddRecvValues(ex_procs[kp]);
    }

    parvec->SetParallelStatus(CUMULATED);
  } // scatter_vec


  template<class TM> void
  HybridMatrix<TM> :: finish_scatter () const
  {
    MyMPI_WaitAll(rr_gather);
    scatter_done = true;
  } // finish_scatter

    
  /** HybridSmoother **/

  template<class TM>
  HybridSmoother<TM> :: HybridSmoother (shared_ptr<BaseMatrix> _A, bool _csr)
    : can_smooth_res(_csr)
  {
    A = make_shared<HybridMatrix<TM>> (_A);

    auto pardofs = A->GetParallelDofs();

    SetParallelDofs(pardofs);

    if (pardofs != nullptr)
      { Sx = make_shared<S_BaseVectorPtr<double>> (pardofs->GetNDofLocal(), pardofs->GetEntrySize()); }
    else
      { Sx = make_shared<S_BaseVectorPtr<double>> (A->Height(), A->BS()); }
  }



  template<class TM>
  void HybridSmoother<TM> :: Smooth (BaseVector  &x, const BaseVector &b,
				 BaseVector  &res, bool res_updated,
				 bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 2 : 0, x, b, res, res_updated, update_res, x_zero);
  }


  template<class TM>
  void HybridSmoother<TM> :: SmoothBack (BaseVector  &x, const BaseVector &b,
				     BaseVector &res, bool res_updated,
				     bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 2 : 1, x, b, res, res_updated, update_res, x_zero);
  }


  template<class TM>
  void HybridSmoother<TM> :: SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
					     bool res_updated, bool update_res, bool x_zero) const
  {

    static Timer t(string("HybridGSS2<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::Smooth");
    RegionTimer rt(t);

    static Timer tpre(string("HybridGSS2<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::Smooth - pre");
    static Timer tpost(string("HybridGSS2<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::Smooth - post");

    /** most of the time RU == UR, if not, reduce to such a case **/
    if (res_updated && !update_res) { // RU && !UR
      SmoothInternal(type, x, b, res, false, false, x_zero); // this is actually cheaper I think
      return;
    }
    if (!res_updated && update_res) { // !RU + UR
      // should happen very infrequently - we can affort mat x vector 
      if (x_zero)
  	{ res = b; }
      else
  	{ res = b - *A * x; } // what about freedofs?
      SmoothInternal(type, x, b, res, true, update_res, x_zero);
      return;
    }

    if (type == 2) {
      SmoothInternal(0, x, b, res, res_updated, update_res, x_zero);
      SmoothInternal(1, x, b, res, res_updated, update_res, false);
      return;
    }
    else if (type == 3) {
      SmoothInternal(1, x, b, res, res_updated, update_res, x_zero);
      SmoothInternal(0, x, b, res, res_updated, update_res, false);
      return;
    }

    /** !!! RU==UR case !!! **/
    if ( (!can_smooth_res) && update_res) { // temporary hack for block smoothers (they cant update res)
      SmoothInternal(type, x, b, res, false, false, x_zero);
      static Timer thack("hacky res. update"); RegionTimer rt(thack);
      res = b - *A * x;
    }


    auto get_loc_ptr = [&](const auto& x) -> BaseVector* { // FML i hate this
      BaseVector* ncp = const_cast<BaseVector*>(&x);
      if (auto parvec = dynamic_cast_ParallelBaseVector(ncp) )
	{ return parvec->GetLocalVector().get(); }
      else
	{ return ncp; }
    };

    const auto H = A->Height();
    auto S = A->GetS();
    auto & xloc = *get_loc_ptr(x);
    const auto & bloc = *get_loc_ptr(b);
    auto & resloc = *get_loc_ptr(res);
    auto & Sxloc = *get_loc_ptr(*Sx);

    { // gather RHS and calculate b-Sx as RHS for local smooth if update res, otherwise stash Sx
      RegionTimer rt(tpre);
#ifdef USE_TAU
    TAU_PROFILE("PREP", TAU_CT(*this), TAU_DEFAULT);
#endif
      x.Cumulate(); // should do nothing most of the time
      if (S != nullptr) {
	if (!res_updated) { // feed in b-S*x as RHS, not res-update
	  b.Distribute();
	  if (x_zero)
	    { resloc = bloc; } // TODO: take bloc for this cast
	  else
	    { resloc = bloc - *S * xloc; }
	  res.SetParallelStatus(DISTRIBUTED);
	}
	else if (!x_zero) // stash S*x_old, because afterwards we get out b-Mx_new-Sx_old
	  { Sxloc = *S * xloc; }
	if (res.GetParallelStatus() == DISTRIBUTED)
	  { A->gather_vec(res); res.SetParallelStatus(CUMULATED); res.Distribute(); }
      }
      else if (b.GetParallelStatus() == DISTRIBUTED) {
	auto& ncb = const_cast<BaseVector&>(b);
	A->gather_vec(ncb); ncb.SetParallelStatus(CUMULATED); ncb.Distribute();
      }
    }
    
    if (type == 0) {
      if (update_res)
	{ SmoothRESLocal(xloc, resloc); }
      else
	{ SmoothLocal(xloc, (S == nullptr) ? bloc : resloc); }
    }
    else if (type == 1) {
      if (update_res)
	{ SmoothBackRESLocal(xloc, resloc); }
      else
	{ SmoothBackLocal(xloc, (S == nullptr) ? bloc : resloc); }
    }
    else {
      throw Exception("GSS invalid type");
    }

    { // scatter updates and finish update residuum, res -= S * (x - x_old)
      RegionTimer rt(tpost);
#ifdef USE_TAU
    TAU_PROFILE("POST", TAU_CT(*this), TAU_DEFAULT);
#endif
      A->scatter_vec(x);
      if (S != nullptr) {
	if (update_res) {
	  res.Distribute();
	  if (!x_zero) // stash S*x_old, because afterwards we get out b-Mx_new-Sx_old
	    { resloc += Sxloc; }
	  S->MultAdd(-1, xloc, resloc);
	}
      }
    }

  } // HybridSmoother::SmoothInternal


  /** HybridGSS2 **/


  template<class TM>
  HybridGSS2<TM> :: HybridGSS2 (shared_ptr<BaseMatrix> _A, shared_ptr<BitArray> _subset)
    : HybridSmoother<TM>(_A, true)
  {

    auto& M = *A->GetM();

    Array<TM> add_diag = this->CalcAdditionalDiag();

    auto pardofs = A->GetParallelDofs();

    if (pardofs != nullptr)
      for (auto k : Range(M.Height()))
	if ( ((!_subset) || (_subset->Test(k))) && (pardofs->IsMasterDof(k)) )
	  M(k,k) += add_diag[k];


    shared_ptr<BitArray> mss = _subset;
    if (_subset && pardofs) {
      mss = make_shared<BitArray>(_subset->Size());
      for (auto k : Range(M.Height())) {
	if (_subset->Test(k) && pardofs->IsMasterDof(k)) { mss->Set(k); }
	else { mss->Clear(k); }
      }
    }

    // if (_subset)
    //   { cout << "smooth on " << mss->NumSet() << " of " << mss->Size() << " dofs!" << endl; }
    // else
    //   { cout << "smooth on all " << A->GetM()->Height() << "dofs!!" << endl; }

    jac = make_shared<GSS2<TM>>(A->GetM(), mss);

    if (pardofs != nullptr)
      for (auto k : Range(M.Height()))
	if ( ((!_subset) || (_subset->Test(k))) && (pardofs->IsMasterDof(k)) )
	  M(k,k) -= add_diag[k];

  }

  template<class TM> INLINE void AddODToD (const TM & v, TM & w) {
    Iterate<mat_traits<TM>::HEIGHT>([&](auto i) {
	Iterate<mat_traits<TM>::HEIGHT>([&](auto j) {
	    w(i.value, i.value) += 0.5 * fabs(v(i.value, j.value));
	  });
      });
  }
  template<> INLINE void AddODToD<double> (const double & v, double & w)
  { w += 0.5 * fabs(v); }
  // template<> INLINE void AddODToD<Complex> (const double & v, double & w)
  // { w += 0.5 * fabs(v); }
  template<class TM>
  Array<TM> HybridSmoother<TM> :: CalcAdditionalDiag ()
  {
    static Timer t(string("HybridGSS<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::CalcAdditionalDiag");
    RegionTimer rt(t);

    Array<TM> add_diag(A->Height()); add_diag = 0;
 
    auto S = dynamic_pointer_cast<SparseMatrixTM<TM>>(A->GetS());
    const auto & M = *A->GetM();
   
    if (S == nullptr)
      { return add_diag; }

    for (auto k : Range(S->Height())) {

      auto rvs = S->GetRowValues(k);
      for (auto l : Range(rvs)) {
	AddODToD(rvs[l], add_diag[k]);
	  // AddODToD(rvs[l], add_diag[ris[l]]);
      }
    }

    AllReduceDofData (add_diag, MPI_SUM, A->GetParallelDofs());  

    if constexpr( is_same<TM, double>::value )
      {
	for (auto k : Range(M.Height())) {
	  if ( (add_diag[k] != 0) && (M(k,k) != 0) ) {
	    if (M(k,k) > 3 * add_diag[k])
	      { add_diag[k] = 0; }
	  }
	}
      }

    return add_diag;
  }


  template<class TM>
  void HybridGSS2<TM> :: SmoothLocal (BaseVector &x, const BaseVector &b) const
  { jac->Smooth(x, b); }

  template<class TM>
  void HybridGSS2<TM> :: SmoothBackLocal (BaseVector &x, const BaseVector &b) const
  { jac->SmoothBack(x, b); }

  template<class TM>
  void HybridGSS2<TM> :: SmoothRESLocal (BaseVector &x, BaseVector &res) const
  { jac->SmoothRES(x, res); }

  template<class TM>
  void HybridGSS2<TM> :: SmoothBackRESLocal (BaseVector &x, BaseVector &res) const
  { jac->SmoothBackRES(x, res); }


  /** HybridBlockSmoother **/


  template<class TM>
  HybridBlockSmoother<TM> :: HybridBlockSmoother (shared_ptr<BaseMatrix> _A, shared_ptr<Table<int>> _blocktable)
    : HybridSmoother<TM>(_A)
  {
    auto S = A->GetS();
    if (S != nullptr) { // filter non-master dofs out of blocktable
      auto pardofs = A->GetParallelDofs();
      Array<int> dofs;
      auto filtered_blocks = [&](auto lam){
	for (auto bnr : Range(_blocktable->Size())) {
	  auto row = (*_blocktable)[bnr];
	  int c = 0;
	  dofs.SetSize(row.Size()); dofs.SetSize0();
	  for (auto dof : row)
	    if (pardofs->IsMasterDof(dof))
	      dofs[c++] = dof;
	  dofs.SetSize(c);
	  if (c != row.Size()) {
	    cout << "adjusted block " << bnr << endl;
	    prow2(row); cout << endl;
	    prow2(dofs); cout << endl;
	  }
	  lam(dofs);
	}
      };
      Array<int> bs(_blocktable->Size());
      size_t cblocks = 0;
      filtered_blocks([&](const auto& dofs) {
	  if(dofs.Size())
	    bs[cblocks++] = dofs.Size();
	});
      bs.SetSize(cblocks); cblocks = 0;
      auto b2 = make_shared<Table<int>> (bs);
      auto &newblocks(*b2);
      filtered_blocks([&](const auto& dofs) {
	  if(dofs.Size())
	    newblocks[cblocks++] = dofs;
	});
      _blocktable = b2;
    }

    Array<TM> add_diag = this->CalcAdditionalDiag();

    auto &M(*A->GetM());
    auto pardofs = A->GetParallelDofs();

    if (pardofs != nullptr)
      for (auto k : Range(M.Height()))
	if (pardofs->IsMasterDof(k))
	  M(k,k) += add_diag[k];

    jac = make_shared<BlockJacobiPrecond<TM, typename HybridMatrix<TM>::TV, typename HybridMatrix<TM>::TV>> (M, _blocktable);

    if (pardofs != nullptr)
      for (auto k : Range(M.Height()))
	if (pardofs->IsMasterDof(k))
	  M(k,k) -= add_diag[k];
  }


  template<class TM>
  void HybridBlockSmoother<TM> :: SmoothLocal (BaseVector &x, const BaseVector &b) const
  { jac->GSSmooth(x, b); }

  template<class TM>
  void HybridBlockSmoother<TM> :: SmoothBackLocal (BaseVector &x, const BaseVector &b) const
  { jac->GSSmoothBack(x, b); }

  // template<class TM>
  // void HybridBlockSmoother<TM> :: Smooth (BaseVector  &x, const BaseVector &b,
  // 					  BaseVector  &res, bool res_updated,
  // 					  bool update_res, bool x_zero) const
  // {
  //   jac->GSSmooth(x, b);
  // }

  // template<class TM>
  // void HybridBlockSmoother<TM> :: SmoothBack (BaseVector  &x, const BaseVector &b,
  // 					      BaseVector &res, bool res_updated,
  // 					      bool update_res, bool x_zero) const
  // {
  //   jac->GSSmoothBack(x, b);
  // }

} // namespace amg


#include <python_ngstd.hpp>


namespace amg
{

  void ExportSmoothers2 (py::module & m)
  {

    py::class_<HybridSmoother<double>, shared_ptr<HybridSmoother<double>>, BaseMatrix>
      (m, "HybridSmoother", "scalar hybrid smoother")
      .def("Smooth", [](shared_ptr<HybridSmoother<double>> & sm, shared_ptr<BaseVector> & sol,
			shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res)
	   { sm->Smooth(*sol, *rhs, *res); }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothBack", [](shared_ptr<HybridSmoother<double>> & sm, shared_ptr<BaseVector> & sol,
			    shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res)
	   { sm->SmoothBack(*sol, *rhs, *res); }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothSymm", [](shared_ptr<HybridSmoother<double>> & sm, shared_ptr<BaseVector> & sol,
			    shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res) {
	     sm->Smooth(*sol, *rhs, *res);
	     sm->SmoothBack(*sol, *rhs, *res);
	   }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothSymmReverse", [](shared_ptr<HybridSmoother<double>> & sm, shared_ptr<BaseVector> & sol,
				   shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res) {
	     sm->SmoothBack(*sol, *rhs, *res);
	     sm->Smooth(*sol, *rhs, *res);
	   }, py::arg("sol"), py::arg("rhs"), py::arg("res"));


    py::class_<HybridGSS2<double>, shared_ptr<HybridGSS2<double>>, HybridSmoother<double>>
      (m, "HybridGSS2", "scalar hybrid Gauss-Seidel")
      .def(py::init<>
	   ( [] (shared_ptr<BaseMatrix> mat, shared_ptr<BitArray> freedofs) {
	     return make_shared<HybridGSS2<double>>(mat, freedofs);
	   }), py::arg("mat"), py::arg("freedofs") = nullptr);


    py::class_<HybridBlockSmoother<double>, shared_ptr<HybridBlockSmoother<double>>, HybridSmoother<double>>
      (m, "HybridBlockSmoother", "scalar block smoother")
      .def(py::init<>
	   ( [] (shared_ptr<BaseMatrix> mat, py::object blocks)
	     {
	       shared_ptr<Table<int>> blocktable;
	       {
		 py::gil_scoped_acquire aq;
		 size_t size = py::len(blocks);
           
		 Array<int> cnt(size);
		 size_t i = 0;
		 for (auto block : blocks)
		   cnt[i++] = py::len(block);
           
		 i = 0;
		 blocktable = make_shared<Table<int>>(cnt);
		 for (auto block : blocks)
		   {
		     auto row = (*blocktable)[i++];
		     size_t j = 0;
		     for (auto val : block)
		       row[j++] = val.cast<int>();
		   }
	       }
	       return make_shared<HybridBlockSmoother<double>>(mat, blocktable);
	     }), py::arg("mat"), py::arg("blocks") );
      
  }

} // namespace amg

#include "amg_tcs.hpp"
