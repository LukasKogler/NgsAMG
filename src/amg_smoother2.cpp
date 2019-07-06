
#define FILE_AMGSM2_CPP

#include "amg.hpp"

namespace amg
{

  class NoMatrix : public BaseMatrix
  {
  protected:
    bool complex; size_t h, w;
  public:
    NoMatrix (size_t _h, size_t _w, bool _c = false)
      : h(_h), w(_w), complex(_c)
    { ; }
    virtual bool IsComplex() const override { return complex; }
    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const override
    { ; }
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const override
    { ; }
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override
    { ; }
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const override
    { ; }
    virtual int VHeight () const override { return h; }
    virtual int VWidth () const override { return w; }
  };

  template<class TM>
  void HybridMatrix<TM> :: MultAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    x.Cumulate();
    y.Distribute();
    M->MultAdd(s, x, y);
    S->MultAdd(s, x, y);
  }

  template<class TM>
  void HybridMatrix<TM> :: MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    x.Cumulate();
    y.Distribute();
    M->MultAdd(s, x, y);
    S->MultAdd(s, x, y);
  }

  template<class TM>
  void HybridMatrix<TM> :: MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, x, y);
    S->MultTransAdd(s, x, y);
  }

  template<class TM>
  void HybridMatrix<TM> :: MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, x, y);
    S->MultTransAdd(s, x, y);
  }

  template<class TM>
  HybridMatrix<TM> :: HybridMatrix (shared_ptr<ParallelMatrix> parmat)
  {
    auto loc_mat = parmat->GetMatrix();

    if (!dynamic_pointer_cast<BaseSparseMatrix> (loc_mat))
      { throw Exception("HybridMatrix needs a sparse Matrix!"); }

    auto loc_spmat = dynamic_pointer_cast<SparseMatrixTM<TM>> (loc_mat);

    if (loc_spmat == nullptr)
      { throw Exception(string("HybridMatrix needs a ") + typeid(SparseMatrixTM<TM>).name() +
			string(", but got handed a ") + typeid(*loc_mat).name()); }

    pardofs = parmat->GetParallelDofs();

    SetParallelDofs(pardofs);

    if (pardofs == nullptr)
      { throw Exception("HybridMatrix needs valid ParallelDofs!"); }

    if (parmat->GetOpType() != C2D)
      { throw Exception("HybridMatrix probably only for C2D (I think)"); }

    comm = pardofs->GetCommunicator();

    SetUpMats(*loc_spmat);

  }


  template<class TM>
  HybridMatrix<TM> :: HybridMatrix (shared_ptr<SparseMatrixTM<TM>> _M)
    : dummy(true), S(nullptr), pardofs(nullptr)
  {
    M = dynamic_pointer_cast<SparseMatrix<TM>>(_M);
    if (M == nullptr)
      { throw Exception("u gave me an actual TM mat - WTF am i supposed to do with that??"); }
    S = make_shared<NoMatrix>(M->Height(), M->Width());
    comm = AMG_ME_COMM;
  }

  template<class TM>
  void HybridMatrix<TM> :: SetUpMats (SparseMatrixTM<TM> & A)
  {
    string cn = string("HybridMatrix<" + to_string(mat_traits<TM>::HEIGHT) + string(">"));
    static Timer t(cn + "::SetUpMats"); RegionTimer rt(t);

    auto H = A.Height();

    auto& pds(*pardofs);

    auto ex_procs = pds.GetDistantProcs();
    nexp = ex_procs.Size();
    nexp_smaller = 0;
    while (nexp_smaller<nexp && ex_procs[nexp_smaller]<comm.Rank()) nexp_smaller++;
    nexp_larger = nexp - nexp_smaller;
    auto rank = comm.Rank();
    
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
	      // cout << " add " << rownr << " " << ris[j] << ", masters " << mrow << " " << master_of(ris[j]) << ", val " << rvs[j] << endl;
      	      fun(rownr, ris[j], rvs[j]);
      	    }
      	  }
      	}
      };
      iterate_coo([&](auto i, auto j, auto val) {
	  perow[i]++; // perow[j]++;
	});
      S = sp_S = make_shared<SparseMatrix<TM>>(perow);
      iterate_coo([&](auto i, auto j, auto val) {
	  sp_S->GetRowIndices(i)[perow[i]] = j;
	  sp_S->GetRowValues(i)[perow[i]++] = val;
	  // sp_S->GetRowIndices(j)[perow[j]] = i;
	  // sp_S->GetRowValues(j)[perow[j]++] = val;
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
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::gather_vec");
    RegionTimer rt(t);

    ParallelBaseVector * parvec = const_cast<ParallelBaseVector*>(dynamic_cast_ParallelBaseVector(&vec));

    if (!parvec)
      { return; }

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
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">>::scatter_vec");
    RegionTimer rt(t);

    ParallelBaseVector * parvec = const_cast<ParallelBaseVector*>(dynamic_cast_ParallelBaseVector(&vec));

    if (!parvec)
      { return; }

    auto ex_procs = paralleldofs->GetDistantProcs();

    parvec->SetParallelStatus(CUMULATED); parvec->Distribute();

    for (auto kkp : Range(nexp_larger))
      { parvec->ISend(ex_procs[nexp_smaller + kkp], rr_gather[kkp]); }

    for (auto kp : Range(nexp_smaller))
      { parvec->IRecvVec(ex_procs[kp], rr_scatter[kp]); }

    MyMPI_WaitAll(rr_gather);

    for (auto j : Range(nexp_smaller)) {
      auto kp = MyMPI_WaitAny(rr_scatter);
      parvec->AddRecvValues(ex_procs[kp]);
    }

    parvec->SetParallelStatus(CUMULATED);
  } // scatter_vec

    
  template<class TM> INLINE void AddODToD (const TM & v, TM & w) {
    Iterate<mat_traits<TM>::HEIGHT>([&](auto i) {
	Iterate<mat_traits<TM>::HEIGHT>([&](auto j) {
	    w(i.value, i.value) += 0.5 * fabs(v(i.value, j.value));
	  });
      });
  }

  template<> INLINE void AddODToD<double> (const double & v, double & w)
  { w += fabs(v); }

  // template<> INLINE void AddODToD<Complex> (const double & v, double & w)
  // { w += 0.5 * fabs(v); }


  template<class TM>
  Array<TM> HybridGSS2<TM> :: CalcAdditionalDiag ()
  {
    static Timer t(string("HybridGSS<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::CalcAdditionalDiag");
    RegionTimer rt(t);

    Array<TM> add_diag(A->Height()); add_diag = 0;
 
    auto S = A->GetSPS();
    if (S == nullptr)
      { return add_diag; }

    for (auto k : Range(S->Height())) {
      auto rvs = S->GetRowValues(k);
      for (auto l : Range(rvs)) {
	AddODToD(rvs[l], add_diag[k]);
	  // AddODToD(rvs[l], add_diag[ris[l]]);
      }
    }

    AllReduceDofData (add_diag, MPI_SUM, pardofs);  

    return add_diag;
  }


  template<class TM>
  HybridGSS2<TM> :: HybridGSS2 (shared_ptr<BaseMatrix> _A, shared_ptr<BitArray> _subset)
  {
    shared_ptr<SparseMatrix<TM>> spm;
    auto parmat = dynamic_pointer_cast<ParallelMatrix> (_A);
    if (parmat != nullptr) {
      pardofs = parmat->GetParallelDofs();
      spm = dynamic_pointer_cast<SparseMatrix<TM>> (parmat->GetMatrix());
      A = make_shared<HybridMatrix<TM>> (parmat);
    }
    else {
      spm = dynamic_pointer_cast<SparseMatrix<TM>> (_A);
      A = make_shared<HybridMatrix<TM>> (spm);
      if (spm == nullptr)
	{ throw Exception("HybridGSS2 could not cast correctly!!"); }
    }
    SetParallelDofs(pardofs);

    origA = _A;
    auto& M = *A->GetM();

    Array<TM> add_diag = CalcAdditionalDiag();

    if (parmat != nullptr)
      for (auto k : Range(spm->Height()))
	if (pardofs->IsMasterDof(k))
	  M(k,k) += add_diag[k];

    cout << "add_diag: " << endl; prow2(add_diag); cout << endl;

    shared_ptr<BitArray> mss = _subset;
    if (_subset && pardofs) {
      mss = make_shared<BitArray>(_subset->Size());
      for (auto k : Range(spm->Height())) {
	if (_subset->Test(k) && pardofs->IsMasterDof(k)) { mss->Set(k); }
	else { mss->Clear(k); }
      }
    }

    jac = make_shared<JacobiPrecond<TM>>(M, mss, false); // false to make sure it does not cumulate diag
    // jac = make_shared<JacobiPrecond<TM>>(*spm, mss, false); // false to make sure it does not cumulate diag

    if (parmat != nullptr)
      for (auto k : Range(spm->Height()))
	if (pardofs->IsMasterDof(k))
	  M(k,k) -= add_diag[k];

  }

  template<class TM>
  void HybridGSS2<TM> :: SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
					bool res_updated, bool update_res, bool x_zero) const
  {
    // cout << typeid(x).name() << " " << typeid(b).name() << " " << typeid(res).name() << endl;

    auto get_loc_ptr = [&](const auto& x) -> BaseVector* { // FML i hate this
      if (auto parvec = dynamic_cast<const ParallelBaseVector*>(&x))
	{ return parvec->GetLocalVector().get(); }
      else
	{ return const_cast<BaseVector*>(&x); }
    };

    auto & xloc = *get_loc_ptr(x);
    const auto & bloc = *get_loc_ptr(b);
    auto & resloc = *get_loc_ptr(res);

    // A->gather_vec(x); // does nothing for cumulated x (as it usually is)
    x.Cumulate();


    if (!res_updated) {
      b.Distribute();
      if (x_zero)
    	{ resloc = bloc; }
      else
    	{ resloc = bloc - *A->GetS() * xloc; }
      res.SetParallelStatus(DISTRIBUTED);
    }

    // A->gather_vec(res); // res.SetParallelStatus(CUMULATED); res.Distribute();
    
    res.Cumulate(); res.Distribute();

    switch(type) {
    case(0) : {
      jac->GSSmooth(xloc, resloc);
      break;
    }
    case(1) : {
      jac->GSSmoothBack(xloc, resloc);
      break;
    }
    case(2) : {
      jac->GSSmooth(xloc, resloc);
      jac->GSSmoothBack(xloc, resloc);
      break;
    }
    case(3) : {
      jac->GSSmoothBack(xloc, resloc);
      jac->GSSmooth(xloc, resloc);
      break;
    }
    default : {
      throw Exception("HGSS invalid type");
      break;
    }
    }

    A->scatter_vec(x);

    // x.SetParallelStatus(CUMULATED); x.Distribute(); x.Cumulate();

    // cout << "x now: " << x.GetParallelStatus() << endl;
    // for (auto k : Range(A->Height())) {
    //   cout << k<< "> " << xloc.template FV<TV>()(k) << ", dps = "; prow2(pardofs->GetDistantProcs(k)); cout << endl;
    // }
    // cout << endl;
    

    if (update_res) {
      res = b - *A * x;

      // auto res2 = res.CreateVector();
      // res2 = b - *origA * x;
      // auto res3 = res.CreateVector();
      // res3 = b - *A * x;
      // auto res4 = res.CreateVector();
      // res4 = res3;
      // res4 -= res2;

      // cout << "DIFF AX orig // A: " << endl;
      // for (auto k : Range(res.FVDouble().Size())) {
      // 	auto diff = fabs(res4.FVDouble()[k]);
      // 	auto dps = pardofs->GetDistantProcs(k);
      // 	if (diff > 1e-8) {
      // 	  cout << k << ": " << res2.FVDouble()[k] << " // " <<  res3.FVDouble()[k] << " // " << fabs(diff) << " // dps ";
      // 	  prow2(dps);
      // 	  cout << endl;
      // 	}
      // }
      // cout << endl;
      

      // cout << "CUMUL DIFF AX orig // A: " << L2Norm(res4) << " // rel " << L2Norm(res4)/L2Norm(res) << " // rel " << L2Norm(res4)/L2Norm(b) << endl;
      // res2.Cumulate(); res3.Cumulate(); res4.Cumulate();
      // for (auto k : Range(res.FVDouble().Size())) {
      // 	auto diff = fabs(res4.FVDouble()[k]);
      // 	auto dps = pardofs->GetDistantProcs(k);
      // 	if (diff > 1e-8) {
      // 	  cout << k << ": " << res2.FVDouble()[k] << " // " <<  res3.FVDouble()[k] << " // " << fabs(diff) << " // dps ";
      // 	  prow2(dps);
      // 	  cout << endl;
      // 	}
      // }
      // cout << endl;

    }
  }


  template<class TM>
  void HybridGSS2<TM> :: Smooth (BaseVector  &x, const BaseVector &b,
				 BaseVector  &res, bool res_updated,
				 bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 2 : 0, x, b, res, res_updated, update_res, x_zero);
  }


  template<class TM>
  void HybridGSS2<TM> :: SmoothBack (BaseVector  &x, const BaseVector &b,
				     BaseVector &res, bool res_updated,
				     bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 3 : 1, x, b, res, res_updated, update_res, x_zero);
  }


} // namespace amg


#include <python_ngstd.hpp>


namespace amg
{

  void ExportSmoothers2 (py::module & m)
  {
    py::class_<HybridGSS2<double>, shared_ptr<HybridGSS2<double>>, BaseMatrix>
      (m, "HybridGSS2", "scalar hybrid Gauss-Seidel")
      .def(py::init<>
	   ( [] (shared_ptr<BaseMatrix> mat, shared_ptr<BitArray> freedofs) {
	     return make_shared<HybridGSS2<double>>(mat, freedofs);
	   }), py::arg("mat"), py::arg("freedofs") = nullptr)
      .def("Smooth", [](shared_ptr<HybridGSS2<double>> & sm, shared_ptr<BaseVector> & sol,
			shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res)
	   { sm->Smooth(*sol, *rhs, *res); }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothBack", [](shared_ptr<HybridGSS2<double>> & sm, shared_ptr<BaseVector> & sol,
			    shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res)
	   { sm->Smooth(*sol, *rhs, *res); }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothSymm", [](shared_ptr<HybridGSS2<double>> & sm, shared_ptr<BaseVector> & sol,
			    shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res) {
	     sm->Smooth(*sol, *rhs, *res);
	     sm->SmoothBack(*sol, *rhs, *res);
	   }, py::arg("sol"), py::arg("rhs"), py::arg("res"))
      .def("SmoothSymmReverse", [](shared_ptr<HybridGSS2<double>> & sm, shared_ptr<BaseVector> & sol,
				   shared_ptr<BaseVector> & rhs, shared_ptr<BaseVector> & res) {
	     sm->SmoothBack(*sol, *rhs, *res);
	     sm->Smooth(*sol, *rhs, *res);
	   }, py::arg("sol"), py::arg("rhs"), py::arg("res"));

  }

} // namespace amg
