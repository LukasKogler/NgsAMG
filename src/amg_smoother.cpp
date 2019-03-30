
#define FILE_AMGSM_CPP

#include "amg.hpp"

namespace amg {

  /** HybridGSS **/

  template<int BS>
  HybridGSS<BS> :: HybridGSS ( const shared_ptr<const HybridGSS<BS>::TSPMAT> & amat,
					 const shared_ptr<ParallelDofs> & par_dofs,
					 const shared_ptr<const BitArray> & atake_dofs)
    : BaseSmoother(par_dofs), free_dofs(atake_dofs), parallel_dofs(par_dofs),
      comm(par_dofs->GetCommunicator()), spmat(amat), A(*spmat)
  {
    
    name =  string("HybridGSS<") + to_string(BS) + string(">");
    auto & pds = *parallel_dofs;
    this->H = spmat->Height();
    this->mf_dofs = BitArray(H); mf_dofs.Clear();
    this->mf_exd  = BitArray(H); mf_exd.Clear();
    for (auto k:Range(H)) {
      if (pds.IsMasterDof(k) && ( (!free_dofs) || free_dofs->Test(k) )) {
	mf_dofs.Set(k);
	if (pds.GetDistantProcs(k).Size()) mf_exd.Set(k);
      }
    }

    // size_t nf = (free_dofs ? free_dofs->NumSet() : H);
    // cout << "make smoother, free " << nf << " of " << H << endl;
    // if (free_dofs) cout << *free_dofs << endl;

    // cout << "mat is: " << endl << A << endl;
    
    SetUpMat();
    CalcDiag();
  } // HybridGSS

  template<int BS>
  HybridGSS<BS> :: ~HybridGSS()
  {
    if (addA != nullptr) delete addA;
    if (CLD != nullptr) delete CLD;
  }

  

  template<int BS>
  void HybridGSS<BS> :: SetUpMat ()
  {
    auto & pds = *parallel_dofs;
    auto rank = comm.Rank();
    // auto np = comm.Size();

    auto ex_procs = pds.GetDistantProcs();
    this->nexp = ex_procs.Size();
    this->nexp_smaller = 0;
    while (nexp_smaller<nexp && ex_procs[nexp_smaller]<rank) nexp_smaller++;
    this->nexp_larger = nexp - nexp_smaller;

    
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
	    if (free_dofs && !free_dofs->Test(d)) continue;
	    auto dps = pds.GetDistantProcs(d);
	    if (dps[0]>=m) {
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

    // cout << "mx_dofs : " << endl << mx_dofs << endl;
    // cout << "mx_loc : " << endl << mx_loc << endl;

    /** Additional mf_exd x mf_exd mat **/
    Array<TSPMAT*> send_add_diag_mats(nexp_smaller);
    Array<MPI_Request> rsdmat (nexp_smaller);
    { // create & send diag-blocks to masters
      Array<int> perow;
      auto SPM_DIAG = [&] (auto dofs) {
	int ds = dofs.Size();
	perow.SetSize(ds); perow = 0;
	for (auto i : Range(dofs.Size())) {
	  auto d = dofs[i];
	  auto ris = A.GetRowIndices(d);
	  int riss = ris.Size();
	  int c = 0;
	  for (auto j : Range(riss)) {
	    auto dj = ris[j];
	    while ( c<ds && dofs[c]<dj ) c++;
	    if ( c == ds ) break;
	    else if ( dofs[c] == dj ) {
	      perow[i]++;
	    }
	  }
	}
	TSPMAT* dspm = new TSPMAT(perow, dofs.Size());
	for (auto i : Range(dofs.Size())) {
	  auto d = dofs[i];
	  auto ris = A.GetRowIndices(d);
	  auto rvs = A.GetRowValues(d);
	  int riss = ris.Size();
	  auto dris = dspm->GetRowIndices(i);
	  auto drvs = dspm->GetRowValues(i);
	  int c = 0; int rc = 0;
	  for (auto j : Range(riss)) {
	    auto dj = ris[j];
	    while ( c<ds && dofs[c]<dj ) c++;
	    if ( c == ds ) break;
	    else if ( dofs[c] == dj ) {
	      dris[rc] = c;
	      drvs[rc++] = rvs[j];
	    }
	  }
	}
	return dspm;
      };
      for (auto kp : Range(nexp_smaller))
	{
	  // Generate diagonal block to send
	  auto p = ex_procs[kp];
	  // cout << "make block to send to proc " << p << ", kp " << kp << endl;
	  auto dofs = mx_dofs[kp];
	  send_add_diag_mats[kp] = SPM_DIAG(dofs);
	  // cout << "have block " << endl;
	  rsdmat[kp] = comm.ISend(*send_add_diag_mats[kp], p, MPI_TAG_AMG);
	}
    }
    // recv diag-mats
    Array<shared_ptr<TSPMAT>> recv_mats(nexp_larger);
    for (auto kkp : Range(nexp_larger)) {
      auto kp = nexp_smaller+kkp;
      auto p = ex_procs[kp];
      comm.Recv(recv_mats[kkp], p, MPI_TAG_AMG);
    }
    // merge diag-mats to get addA!
    Array<int> perow(H); perow = 0;
    Array<size_t> at_row(nexp_larger); 
    Array<int> row_matis(nexp_larger);
    Array<FlatArray<int>> all_cols;
    at_row = 0;
    for (auto rownr : Range(H)) {
      if (!mf_exd.Test(rownr)) continue;
      // which mats I received have this row?
      row_matis.SetSize(0);
      for (auto kkp : Range(nexp_larger)) {
	auto kp = nexp_smaller+kkp;
	auto exds = mx_dofs[kp];
	auto ar = at_row[kkp]; // the next row for that ex-mat
	if (at_row[kkp]==exds.Size()) continue; // no more rows to take from there
	size_t ar_dof = exds[ar]; // the dof that row belongs to
	if (ar_dof>rownr) continue; // not yet that row's turn
	row_matis.Append(kkp);
	at_row[kkp]++;
      }
      // merge the rows I got
      all_cols.SetSize(0); all_cols.SetSize(row_matis.Size()); // otherwise tries to copy FA I think
      for (auto k:Range(all_cols.Size())) {
	auto kkp = row_matis[k];
	auto kp = nexp_smaller + kkp;
	auto mxd = mx_dofs[kp];
	auto cols = recv_mats[kkp]->GetRowIndices(at_row[kkp]-1);
	for (auto j:Range(cols.Size()))
	  cols[j] = mxd[cols[j]];
	all_cols[k].Assign(cols);
      }
      auto merged_cols = merge_arrays(all_cols, [](const auto&a, const auto &b){return a<b; });
      perow[rownr] = merged_cols.Size();
    }
    addA = new TSPMAT(perow, H);
    at_row = 0;
    for (auto rownr : Range(H)) {
      row_matis.SetSize(0);
      if (!mf_exd.Test(rownr)) continue;
      // which mats I received have this row?
      for (auto kkp : Range(nexp_larger)) {
	auto kp = nexp_smaller+kkp;
	auto exds = mx_dofs[kp];
	auto ar = at_row[kkp]; // the next row for that ex-mat
	if (at_row[kkp]==exds.Size()) continue; // no more rows to take from there
	size_t ar_dof = exds[ar]; // the dof that row belongs to
	if (ar_dof>rownr) continue; // not yet that row's turn
	row_matis.Append(kkp);
	at_row[kkp]++;
      }
      // merge the rows I got
      all_cols.SetSize(0); all_cols.SetSize(row_matis.Size()); // otherwise tries to copy FA I think
      for (auto k:Range(all_cols.Size())) {
	auto kkp = row_matis[k];
	all_cols[k].Assign(recv_mats[kkp]->GetRowIndices(at_row[kkp]-1));
      }
      auto merged_cols = merge_arrays(all_cols, [](const auto&a, const auto &b){return a<b; });
      auto ris = addA->GetRowIndices(rownr);
      auto rvs = addA->GetRowValues(rownr);
      // cerr << "mc (" << merged_cols.Size() << "): "; prow(merged_cols, cerr); cerr << endl;
      // cerr << "ris (" << ris.Size() << "): "; prow(ris, cerr); cerr << endl;
      ris = merged_cols;
      rvs = 0;
      for (auto l: Range(all_cols.Size())) {
	auto kp = row_matis[l];
	auto dris = recv_mats[kp]->GetRowIndices(at_row[kp]-1);
	auto drvs = recv_mats[kp]->GetRowValues(at_row[kp]-1);
	for (auto j : Range(dris.Size())) {
	  rvs[ris.Pos(dris[j])] += drvs[j];
	}
      }
    }

    /** 
	Now build the off-diagonal matrix C_DL:
	  - cols are all mx_dofs[0..nexp_smaller)
	  - rows are everything, except diag within mx_dof-rows
     **/
    size_t CH = H;
    size_t CW = 0;
    for (auto k:Range(nexp_smaller))
      CW += mx_dofs[k].Size();
    perow.SetSize(CH); perow = 0;
    // row-inds == col-inds!
    for (auto kp : Range(nexp_smaller)) {
      auto exd = mx_dofs[kp];
      for (auto dof:exd) {
	auto ris = A.GetRowIndices(dof);
	for (auto j : Range(ris.Size())) {
	  auto dj = ris[j];
	  if ( free_dofs && !free_dofs->Test(dj) ) continue;
	  if (!exd.Contains(dj)) perow[dj]++;
	}
      }
    }
    CLD = new TSPMAT(perow, CW);
    perow = 0;
    // we "reorder" cols to [0..nex1-1], [0..nex_2-1], ...
    // so we can just multadd the buffer!
    int ccol = 0;
    for (auto kp : Range(nexp_smaller)) {
      auto exd = mx_dofs[kp];
      for (auto kdof : Range(exd.Size())) {
	auto dof = exd[kdof];
	auto ris = A.GetRowIndices(dof);
	auto rvs = A.GetRowValues(dof);
	for (auto j : Range(ris.Size())) {
	  auto dj = ris[j];
	  if ( free_dofs && !free_dofs->Test(dj) ) continue;
	  auto cdj = perow[dj];
	  if (!exd.Contains(dj)) {
	    CLD->GetRowIndices(dj)[cdj] = ccol;
	    CLD->GetRowValues(dj)[cdj] = Trans(rvs[j]);
	    perow[dj]++;
	  }
	}
	ccol++;
      }
    }

    // cout << "mx_dofs: " << endl << mx_dofs << endl;
    // cout << "orig mat: " << endl << A << endl;
    // cout << "addA: " << endl << *addA << endl;
    // cout << "CLD: " << endl << *CLD << endl;

    rr_gather.SetSize(nexp_larger);
    rr_scatter.SetSize(nexp_smaller);
    buf_cnt.SetSize(nexp);
    buf_os.SetSize(nexp+1);
    buf_os[0] = 0;
    for (auto kp : Range(nexp)) {
      buf_cnt[kp] = mx_dofs[kp].Size();
      buf_os[kp+1] = buf_os[kp] + buf_cnt[kp];
    }
    buffer.SetSize(buf_os.Last());
    size_t xbs = 0; for (auto kp:Range(nexp_smaller)) xbs += mx_dofs[kp].Size();
    x_buffer.SetSize(xbs);
    ax_buffer.SetSize(xbs);

    // cout << "buf_cnt: " << endl << buf_cnt << endl;
    // cout << "buf_os: " << endl << buf_os << endl;
    
    // cout << "wait all send " << endl;
    MyMPI_WaitAll(rsdmat);
    // cout << "delete mats " << endl;
    for (auto kp : Range(nexp_smaller))
      delete send_add_diag_mats[kp];

    
  } // SetUpMat




  template<int BS>
  void add_diag (double val, Vec<BS*BS,double>& a, const Mat<BS,BS,double>& b)
  {
    for (auto k:Range(BS)) {
      const int i1 = k*(BS+1); 
      for (auto j : Range(BS))
	a(i1) += val * fabs(b(k,j));
    }
  }
  void add_diag (double val, Vec<1,double>& a, const double& b) { a(0) += val * fabs(b); }
  template<int BS>
  void set_v2m (Mat<BS,BS,double>& a, const Vec<BS*BS,double>& b)
  { for (auto k : Range(BS*BS)) a(k) = b(k); }
  void set_v2m (double& a, const Vec<1,double>& b)
  { a = b(0); }
  template<int BS>
  void set_m2v (Vec<BS*BS,double>& a, const Mat<BS,BS,double>& b)
  { for (auto k : Range(BS*BS)) a(k) = b(k); }
  void set_m2v (Vec<1,double>& a, const double& b)
  { a(0) = b; }
  template<int BS>
  void HybridGSS<BS> :: CalcDiag ()
  {
    const TSPMAT & spm(*spmat);
    const ParallelDofs & pds(*parallel_dofs);
    TableCreator<int> cvp(H);
    for (;!cvp.Done(); cvp++) {
      for (auto k:Range(H))
	for (auto p:pds.GetDistantProcs(k))
	  cvp.Add(k,p);
    }
    constexpr int MS = BS*BS;
    ParallelDofs block_pds(pds.GetCommunicator(), cvp.MoveTable(), MS, false);
    shared_ptr<ParallelDofs> spbp(&block_pds, NOOP_Deleter);
    ParallelVVector<Vec<MS,double>> pvec(spbp, DISTRIBUTED);
    for (auto k : Range(H)) {
      auto & diag_etr = spm(k,k);
      auto & dvec = pvec(k);
      set_m2v(dvec, diag_etr);
      auto mproc = pds.GetMasterProc(k);
      if (!free_dofs || free_dofs->Test(k)) {
	auto ris = spm.GetRowIndices(k);
	auto rvs = spm.GetRowValues(k);
	for (auto j : Range(ris.Size())) {
	  auto mj = pds.GetMasterProc(j);
	  if (mproc!=mj) {
	    add_diag(0.5, dvec, rvs[j]);
	  }
	}
      }
    }
    pvec.Cumulate();
    diag.SetSize(H);
    for (auto k : Range(H)) {
      auto & diag_etr = diag[k];
      const auto & pve = pvec(k);
      set_v2m(diag_etr, pve);
    }
    cout << "final diags: " << endl << diag << endl;
    for (auto k : Range(H)) {
      if (!free_dofs || free_dofs->Test(k))
	CalcInverse(diag[k]);
    }
    cout << "final inved diags: " << endl << diag << endl;
  } // CalcDiag


  
  template<int BS> void
  HybridGSS<BS> :: gather_vec (const BaseVector & vec) const
  {
    FlatVector<TV> tvec = vec.FV<TV>();
    auto & pds = *parallel_dofs;
    auto ex_procs = pds.GetDistantProcs();
    for (auto kp : Range(nexp_smaller)) {
      auto p = ex_procs[kp];
      int sz = buf_os[kp+1] - buf_os[kp];
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      int c = 0;
      auto exdofs = pds.GetExchangeDofs(p);
      for (auto d:exdofs) {
	if (free_dofs && !free_dofs->Test(d)) continue;
	auto master = pds.GetMasterProc(d);
	if (p==master) {
	  p_buffer[c++] = tvec(d);
	}
      }
      // cout << "gather, send buf to " << p << ", kp " << kp << " " << p_buffer.Size() << ": "; prow(p_buffer); cout << endl;
      MPI_Request req = MyMPI_ISend(p_buffer, p, MPI_TAG_AMG, comm);
      MPI_Request_free(&req);
    }
    if (nexp_larger==0) return;
    for (auto kkp : Range(nexp_larger)) {
      auto kp = nexp_smaller + kkp;
      auto p = ex_procs[kp];
      int sz = buf_os[kp+1] - buf_os[kp];
      // cout << "gather, recv " << sz << " from " << p << ", kp " << kp << endl;
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      rr_gather[kkp] = MyMPI_IRecv(p_buffer, p, MPI_TAG_AMG, comm);
    }
    int nrr = nexp_larger;
    MPI_Request* rrptr = &rr_gather[0];
    int ind;
    for (int nreq = 0; nreq<nexp_larger; nreq++) {
      // cout << "wait for message " << nreq << " of " << nexp_larger << endl;
      MPI_Waitany(nrr, rrptr, &ind, MPI_STATUS_IGNORE);
      int kp = nexp_smaller+ind;
      auto p = ex_procs[kp];
      auto exdofs = pds.GetExchangeDofs(p);
      int c = 0;
      int sz = buf_os[kp+1] - buf_os[kp];
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      // cout << "gather, got buf from " << p << ", kp " << kp << " " << p_buffer.Size() << ": "; prow(p_buffer); cout << endl;
      // cout << "apply to dofs ";
      for (auto d:exdofs) {
	if (free_dofs && !free_dofs->Test(d)) continue;
	if (!pds.IsMasterDof(d)) continue;
	// TV old = tvec(d);
	tvec(d) += p_buffer[c++];
	// cout << "tvec(" << d << ") += " << p_buffer[c-1] << ": "
	//      << old << " -> " << tvec(d) << endl;
      }
      // cout << endl;
    }
  } // gather_vec

  template<int BS> void
  HybridGSS<BS> :: scatter_vec (const BaseVector & vec) const
  {
    FlatVector<TV> fvec = vec.FV<TV>();
    auto & pds = *parallel_dofs;
    auto ex_procs = pds.GetDistantProcs();
    for (int kkp : Range(nexp_larger)) {
      int kp = nexp_smaller + kkp;
      auto p = ex_procs[kp];
      int sz = buf_os[kp+1] - buf_os[kp];
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      // cout << "scatter, send update to " << p << ", kp " << kp << " " << p_buffer.Size() << ": "; prow(p_buffer); cout << endl;
      MPI_Request reqs = MyMPI_ISend(p_buffer, p, MPI_TAG_AMG, comm);
      MPI_Request_free(&reqs);
    }
    if (nexp_smaller==0) return;
    for (int kp : Range(nexp_smaller)) {
      int sz = buf_os[kp+1] - buf_os[kp];
      // cout << "scatter, recv " << sz << " from " << p << ", kp " << kp << endl;
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      rr_scatter[kp] = MyMPI_IRecv(p_buffer, ex_procs[kp], MPI_TAG_AMG, comm);
    }
    int nrr = nexp_smaller;
    MPI_Request * rrptr = &rr_scatter[0];
    int ind;
    for (int nreq = 0; nreq<nexp_smaller; nreq++) {
      MPI_Waitany(nrr, rrptr, &ind, MPI_STATUS_IGNORE);
      int kp = ind;
      auto p = ex_procs[kp];
      auto exdofs = pds.GetExchangeDofs(p);
      int sz = buf_os[kp+1] - buf_os[kp];
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      // cout << "apply update from " << p << ", kp " << kp << " " << p_buffer.Size() << ": "; prow(p_buffer); cout << endl;
      int c = 0;
      // cout << "to dofs: ";
      for (auto d:exdofs) {
	if (free_dofs && !free_dofs->Test(d)) continue;
	auto master = pds.GetMasterProc(d);
	// if (master==p) cout << d << " ";
	if (master==p) fvec(d) += p_buffer[c++];
      }
      // cout << endl;
    }
  } // scatter_vec


  /**
    L,D,L.T    - refers to diag-block of A
    C          - the off-diag part of A
    C_DL, C_LD - dist-loc // loc-dist part of C
    w          - update for x
    We assume that x is CUMULATED
    FW works like this:
      if RU:
  	    I)   initial state:  res = b-Ax
  	         (if res is DISTR: -> Gather)
  	    II)  loop: 
  	           res = b-Ax-Lw
  	           w = diag_inv * res -> write w to buffer
  		   update later rows for Lw [half row ARTTV]
  		   if UR: update prev. rows for L.Tw [half row ARTTV; includes C_DLw_L]
  		 afterwards:
  		   res = b-Ax-Lw-Dw-L.Tw-C_DLw_L // missing: -C_LDw_D
  	    III) scatter w, up UR: update res -= Wc
  		   C_LD w_d is easy (no w-vec needed)
  	 else :
  	    I)   initial state: res = b-C_DLxL - C_LDxD 
	               C_DL xL = A_D: (0,xL).T (RTV), distribute x, buffer old vals for x_D
		       C_LD xD with CLD->MultAdd(..)
  	         (if x_zero, skip mults; if also b is CUMUL, also skip gather)
  	    II)  loop:
  	           res = b-Cx
  		   update res -= Lx = [half row RTV]  // only on L-rows!!
		   L(A)x = L(A)_LLx_L_new + [L(A)_LD x_D_new=0!!] [half RTV]
  		   w = diag_inv * (res-L.Tx) [half row RTV]
		   write w to buffer
  		   if UR: update prev.rows for L.Tw [half row ARTTV; includes D+L.T part of C_DLw_L]
  		 afterwards:
		   missing C_LDw_D
		   missing LTD(C_DL)w_L
  		   res = b-Cx-Lx_new-L.Tx_new-C_DLw_L // missingL -C_LDw_D
		   !!restore buffered x_old_D values
  	    III) as above
  **/

  template<int BS> Timer& hgss_timer_hack (string bname, int type) {
    static Timer t0(bname+string("::FW"));
    static Timer t1(bname+string("::BW"));
    if (type==0) return t0;
    return t1;
  };
  template<int BS> void
  HybridGSS<BS> :: smoothfull (int _type, BaseVector  &x, const BaseVector &b, BaseVector &res,
			       bool _res_updated, bool _update_res, bool _x_zero) const
  {
    string tname = name + string("::Smooth");
    static Timer t1 (tname);
    RegionTimer RT1(t1);
    RegionTimer RT2(hgss_timer_hack<BS>(tname, _type));

    const int type = _type;
    const bool res_updated = _res_updated;
    const bool update_res = _update_res;
    const bool x_zero = _x_zero;
    
    if (type==2) {
      smoothfull(0, x, b, res, res_updated, true, _x_zero);
      smoothfull(1, x, b, res, true, update_res, false);
      return;
    }
    else if (type==3) {
      smoothfull(1, x, b, res, res_updated, true, _x_zero);
      smoothfull(0, x, b, res, true, update_res, false);
      return;
    }
    
    // cout << "called: " << type << " " << res_updated << " " << update_res << " " << x_zero << endl;
    // cout << "stats " << x.GetParallelStatus() << " " << b.GetParallelStatus() << " " << res.GetParallelStatus() << endl;

    FlatVector<TV> tvx = x.FV<TV>();
    FlatVector<TV> tvb = b.FV<TV>();
    FlatVector<TV> tvr = res.FV<TV>();
    const auto & pds(*parallel_dofs);
    auto ex_procs = pds.GetDistantProcs();
    auto index_of_proc = [&ex_procs](auto p) {
      return ex_procs.Pos(p); // TODO: could be faster with binary search
    };

    // if(free_dofs) cout << "fds: " << endl << *free_dofs << endl;
    
    // auto savex = res.CreateVector();
    // FlatVector<TV> tvsx = savex.FV<TV>();
    // tvsx = tvx;
    // savex.SetParallelStatus(x.GetParallelStatus());
    
    // auto print_vec = [&](const auto &vec) {
    //   for (auto k : Range(H)) {
    // 	auto dps = pds.GetDistantProcs(k);
    // 	cout << k << ": ";
    // 	cout << vec(k) << " ";
    // 	if (!free_dofs) cout << "1";
    // 	else cout << free_dofs->Test(k);
    // 	cout << " mf " << mf_dofs.Test(k);
    // 	cout << " mfex " << mf_exd.Test(k);
    // 	cout << "  || dps: ";
    // 	prow(dps);
    // 	cout << endl;
    //   }
    //   cout << endl;
    // };

    // auto check_res = [&](auto & res) {
    //   FlatVector<TV> tvr = res.template FV<TV>();
    //   auto stat = res.GetParallelStatus();
    //   Array<TV> save_d(H), save_c(H);
    //   for (auto k : Range(H))
    // 	save_d[k] = tvr(k);
    //   res.Cumulate();
    //   for (auto k : Range(H))
    // 	save_c[k] = tvr(k);
    //   tvr = tvb;
    //   res.SetParallelStatus(b.GetParallelStatus());
    //   res.Distribute();
    //   A.MultAdd(-1.0, *savex, res);
    //   cout << "DISTR ex res: " << endl; print_vec(tvr);
    //   res.Cumulate();
    //   cout << "CUMUL ex res: " << endl; print_vec(tvr);
    //   for (auto k : Range(H)) {
    // 	if (free_dofs && !free_dofs->Test(k)) continue;
    // 	TV diff = tvr(k)-save_c[k];
    // 	TV tvrk = tvr(k);
    // 	if (TVNorm(tvrk)*TVNorm(diff)<1e-14) continue;
    // 	cout << "diff/dis/cumu/ex res " << k << ": " << diff << " " << save_d[k] << " " << save_c[k] <<
    // 	  " " << tvr(k) << endl;
    //   }
    //   cout << endl;
    //   for (auto k : Range(H))
    // 	tvr(k) = save_d[k];
    //   res.SetParallelStatus(stat);
    // };

    // if (res_updated) {
    //   cout << "check input res!" << endl;
    //   check_res(res);
    //   cout << "check input res done" << endl;
    // }

    // cout << " x in : " << endl;
    // print_vec(tvx);

    // cout << "res in: " << endl; // cout << res << endl;
    // print_vec(tvr);

    // cout << "rhs in: " << endl; // cout << res << endl;
    // print_vec(tvb);

    // auto temp = res.CreateVector();
    // FlatVector<TV> tvt = temp.FV<TV>();
    // tvt = 0;
    // temp.SetParallelStatus(DISTRIBUTED);
    // for (auto k:Range(H)) {
    //   if(!mf_dofs.Test(k)) continue;
    //   auto ris = A.GetRowIndices(k);
    //   auto rvs = A.GetRowValues(k);
    //   for (auto j : Range(ris.Size())) {
    // 	auto d = ris[j];
    // 	if (!mf_dofs.Test(d)) continue;
    // 	tvt(k) -= rvs[j] * tvx(d);
    //   }
    //   if(mf_exd.Test(k)) {
    // 	auto ris = addA->GetRowIndices(k);
    // 	auto rvs = addA->GetRowValues(k);
    // 	for (auto j : Range(ris.Size())) {
    // 	  auto d = ris[j];
    // 	  if (!mf_dofs.Test(d)) continue;
    // 	  tvt(k) -= rvs[j] * tvx(d);
    // 	}
    //   }
    // }

    if (x.GetParallelStatus()!=CUMULATED) {
      // Note: this can happen in Preconditioner::Test()
      x.Cumulate();
    }

    /** STAGE (I) **/
    if (!res_updated) {
      tvr = tvb;
      res.SetParallelStatus(b.GetParallelStatus());
      res.Distribute();
      // cout << "rhs before mods: " << endl; // cout << res << endl;
      // print_vec(tvr);
      if (!x_zero) {
	// Calc C_LDx_D = A_DDhat x_Dhat + A_DL x_L
	int cx = 0;
	for (auto kp : Range(nexp_smaller)) {
	  auto p = ex_procs[kp];
	  auto ds = pds.GetExchangeDofs(p);
	  int c = 0;
	  int sz = buf_os[kp+1] - buf_os[kp];
	  FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
	  for (auto j : Range(ds.Size())) { // stash diag-vals
	    auto d = ds[j]; if(free_dofs && !free_dofs->Test(d)) continue;
	    auto m = pds.GetMasterProc(d); if (m!=p) continue;
	    p_buffer[c++] = x_buffer[cx++] = tvx(d); tvx(d) = 0;
	  }
	}
	auto sz = buf_os[nexp_smaller];
	VFlatVector<TV> tvw (sz, &buffer[0]);
	// cout << "CLD update, buffer to " << sz << ": "; prow(buffer); cout << endl;
	// print_vec(tvr);
	CLD->MultAdd(-1.0, tvw, res);
	// cout << "rhs with CLD: " << endl; // cout << res << endl;
	// print_vec(tvr);
	// Store A_DL x_L ( no A_DDhat x_Dhat!!) into buffer
	int cax = 0;
	for (auto kp : Range(nexp_smaller)) {
	  auto p = ex_procs[kp];
	  auto ds = pds.GetExchangeDofs(p);
	  int sz = buf_os[kp+1] - buf_os[kp];
	  FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
	  for (auto j : Range(ds.Size())) { // stash diag-vals
	    auto d = ds[j]; if(free_dofs && !free_dofs->Test(d)) continue;
	    auto m = pds.GetMasterProc(d); if (m!=p) continue;
	    auto ax = A.RowTimesVector(d, tvx);
	    // cout << "row " << d << " times xloc: " << ax << endl;
	    tvr(d) -= (ax_buffer[cax++] = ax);
	  }
	}
	// cout << "stashed x_vals: " << endl; prow(x_buffer); cout << endl;
      }
    }
    // cout << "res after prep (not gathered): " << endl;
    // print_vec(tvr);

    // gather res to master
    if (res.GetParallelStatus()==DISTRIBUTED) {
      gather_vec(res); // if not res_up, includes C_DL x_L_old
      res.SetParallelStatus(CUMULATED);
      res.Distribute();
    }
    else {
      res.Distribute();
    }
    // res has now full values on masters, 0 on others

    // cout << "res after prep: " << endl; // cout << res << endl;
    // print_vec(tvr);

    // cout << " x after prep (vals stashed) : " << endl;
    // print_vec(tvx);

    // check if so far ok:
    // if(!res_updated) {
    //   tvt += tvr;
    //   cout << "tvt is :" << endl; print_vec(tvt);
    //   BaseVector & bvt(*temp);
    //   check_res(bvt);
    //   cout << "check if res = b - OFFDIAG * x is done!" << endl;
    // }    
    
    
    // cout << "res after prep + comp: " << endl; // cout << res << endl;
    // print_vec(tvr);
    
    /**
       INIT: b - Ax_old - L(A_LL) x_new
       -> w = dinv * res [SIMPLE]
       update next rows for INIT
       update prev. rows for RES
       -> updates add values to residual at D-dofs [A_LL w_L + C_DL w_L]
    **/
    auto update_row_resu = [&](auto rownr) {
      if (!mf_dofs.Test(rownr)) return;
      TV w = diag[rownr] * tvr(rownr);
      tvx(rownr) += w;
      auto ris = A.GetRowIndices(rownr);
      auto rvs = A.GetRowValues(rownr);
      auto pos = ris.Pos(rownr);
      auto sz = ris.Size();
      // L update for future rows
      // [0..pos) or [pos+1,..sz)
      const int imin = (type==1) ? 0 : pos+1;
      const int imax = (type==1) ? pos : sz;
      for (int l = imin; l < imax; l++) {
	// cout << "tvr( " << ris[l] << "/" << tvr.Size() << ") -= " << rvs[l] << " * " << w << endl;
	// cerr << "tvr( " << ris[l] << "/" << tvr.Size() << ") -= " << rvs[l] << " * " << w << endl;
	tvr(ris[l]) -= Trans(rvs[l]) * w;
      }
      if (update_res) {  // D + L.T update for prev. rows
	// [0..pos] or [pos..sz)
	const int imin = (type==0) ? 0 : pos;
	const int imax = (type==0) ? pos+1 : sz;
	for (int l = imin; l < imax; l++) {
	  tvr(ris[l]) -= Trans(rvs[l]) * w;
	}
      }
      if (mf_exd.Test(rownr)) {
	auto dps = pds.GetDistantProcs(rownr);
	for (auto k : Range(dps.Size())) {
	  int ind = index_of_proc(dps[k]);
	  if (type == 0) {
	    auto offset = buf_os[ind];
	    buffer[offset + buf_cnt[ind]++] = w;
	  }
	  else {
	    auto offset = buf_os[ind+1];
	    buffer[offset - ++buf_cnt[ind]] = w;
	  }
	}
	auto ris = addA->GetRowIndices(rownr);
	auto rvs = addA->GetRowValues(rownr);
	auto pos = ris.Pos(rownr);
	auto sz = ris.Size();
	// L update for future rows
	const int imin = (type==1) ? 0 : pos+1;
	const int imax = (type==1) ? pos : sz;
      	for (int l = imin; l < imax; l++) {
	  tvr(ris[l]) -= Trans(rvs[l]) * w;
	}
	if (update_res) {  // L.T update for prev. rows
	  const int imin = (type==0) ? 0 : pos;
	  const int imax = (type==0) ? pos+1 : sz;
	  for (int l = imin; l < imax; l++) {
	    tvr(ris[l]) -= Trans(rvs[l]) * w;
	  }
	}
      }
    };

    /**
       INIT: b - C_DL x_L - C_LD x_D = b - Cx
       update res:
               res -= L*x = L(A_LL)x_L_new !! C_LD x_D is 0 bc x is DISTR!
       now res is: b - Cx - L(A_LL)x_L_new
       missing: L.T(A_LL) x_old -> second half of RTV (agail, x_D=0)!
       -> w = dinv * (res - add_res)
       update prev. rows for RES -> adds values to residual at D-dofs [A_LL w_L + C_DL w_L]
    **/
    int pos(-1), pos_add(-1), sz(-1), sz_add(-1);
    auto update_row_resnotu = [&](auto rownr) {
      if (free_dofs && !free_dofs->Test(rownr)) return;
      const bool exrow = mf_exd.Test(rownr);
      auto & resval = tvr(rownr);
      // cout << "type " << typeid(resval).name() << endl;
      auto ris = A.GetRowIndices(rownr);
      auto rvs = A.GetRowValues(rownr);
      pos = ris.Pos(rownr);
      sz = ris.Size();
      // L * x_new -> exclude diag
      const int imin = (type==0) ? 0 : pos+1;
      const int imax = (type==0) ? pos : sz;
      for (int l = imin; l < imax; l++) resval -= rvs[l] * tvx(ris[l]);
      // if free & not master, only do L-update (per def only local mat needed!)
      if (!mf_dofs.Test(rownr)) return;
      TV tot_rv(0);
      // update res curr.row
      // (L.T+D) * x_old -> include diag
      const int imin2 = (type==1) ? 0 : pos;
      const int imax2 = (type==1) ? pos+1 : sz;
      for (int l = imin2; l < imax2; l++) tot_rv -= rvs[l] * tvx(ris[l]);
      if (exrow) {
	auto ris = addA->GetRowIndices(rownr);
	auto rvs = addA->GetRowValues(rownr);
	pos_add = ris.Pos(rownr);
	sz_add = ris.Size();
	// L * x_new -> exclude diag
	const int imin = (type==0) ? 0 : pos_add+1;
	const int imax = (type==0) ? pos_add : sz_add;
	// cout << " res L up ex " << imin << " " << imax << endl;
      	for (int l = imin; l < imax; l++) resval -= rvs[l] * tvx(ris[l]);
	// (L.T+D) * x_old -> include diag
      	const int imin2 = (type==1) ? 0 : pos_add;
	const int imax2 = (type==1) ? pos_add+1 : sz_add;
	for (int l = imin2; l < imax2; l++) tot_rv -= rvs[l] * tvx(ris[l]);
      }
      // calc & store update
      tot_rv += resval;
      TV w = diag[rownr] * tot_rv;
      tvx(rownr) += w;
      if (exrow) {
      	auto dps = pds.GetDistantProcs(rownr);
	for (auto k : Range(dps.Size())) {
	  int ind = index_of_proc(dps[k]);
	  if (type == 0) {
	    auto offset = buf_os[ind];
	    buffer[offset + buf_cnt[ind]++] = w;
	  }
	  else {
	    auto offset = buf_os[ind+1];
	    buffer[offset - ++buf_cnt[ind]] = w;
	  }
	}
      }
      // update res for prev. rows: (L.T+D), include diag
      if (update_res) {
	auto xval = tvx(rownr);
      	const int imin = (type==0) ? 0 : pos;
	const int imax = (type==0) ? pos+1 : sz;
	for (int l = imin; l < imax; l++) tvr(ris[l]) -= Trans(rvs[l]) * xval;
	if (exrow) {
	  auto ris = addA->GetRowIndices(rownr);
	  auto rvs = addA->GetRowValues(rownr);
	  const int imin = (type==0) ? 0 : pos_add;
	  const int imax = (type==0) ? pos_add+1 : sz_add;
	  for (int l = imin; l < imax; l++) tvr(ris[l]) -= Trans(rvs[l]) * xval;
	}
      }
    };

    buf_cnt = 0;
    if (type==0) { // FW
      if (res_updated)
	for (size_t rownr = 0; rownr<H; rownr++)
	  update_row_resu(rownr);
      else
	for (size_t rownr = 0; rownr<H; rownr++)
	  update_row_resnotu(rownr);
    }
    else { // BW
      if (res_updated)
	for (int rownr = H-1; rownr>=0; rownr--)
	  update_row_resu(rownr);
      else
	for (int rownr = H-1; rownr>=0; rownr--)
	  update_row_resnotu(rownr);
    }
      

    // // compensate C_DL x_L_old
    // if ( (!res_updated) && (update_res) && (!x_zero) ) {
    //   cout << "buffer (b - A_DO x_O)" << endl; prow(buffer); cout << endl;
    //   int cax = 0;
    //   for (auto kp : Range(nexp_smaller)) {
    // 	auto p = ex_procs[kp];
    // 	auto ds = pds.GetExchangeDofs(p);
    // 	// int c = 0;
    // 	int sz = buf_os[kp+1] - buf_os[kp];
    // 	FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
    // 	for (auto j : Range(ds.Size())) { // stash diag-vals
    // 	  auto d = ds[j]; if(free_dofs && !free_dofs->Test(d)) continue;
    // 	  auto m = pds.GetMasterProc(d); if (m!=p) continue;
    // 	  // cout << "b-ax " << d << ": " << p_buffer[c] << endl;
    // 	  cout << "b-ax " << d << ": " << ax_buffer[cax] << endl;
    // 	  cout << "tvb( " << d << "): " << tvb(d) << endl;
    // 	  // buffer = b-A_DLx_L
    // 	  tvr(d) = ax_buffer[cax++]; // = +AODLx_L
    // 	  cout << "ax should be " << tvr(d) << endl;
    // 	}
    //   }
    // }

    /** STAGE (III) **/
    // scatter & apply updates
    scatter_vec(x);
    // restore buffered x-values!
    if ( (!res_updated) && (!x_zero) ) {
      int cx = 0;
      for (auto kp : Range(nexp_smaller)) {
	auto p = ex_procs[kp];
	auto ds = pds.GetExchangeDofs(p);
	for (auto j : Range(ds.Size())) { // stash diag-vals
	  auto d = ds[j]; if(free_dofs && !free_dofs->Test(d)) continue;
	  auto m = pds.GetMasterProc(d); if (m!=p) continue;
	  if (update_res) tvr(d) += ax_buffer[cx];
	  tvx(d) += x_buffer[cx++];
	}
      }
    }
    // update residuum (missing CLD*w_D)

    if ( update_res && (nexp_smaller>0) ) {
      auto sz = buf_os[nexp_smaller];
      VFlatVector<TV> tvw (sz, &buffer[0]);
      // cout << "res before CLD update" << endl;
      // cout << res.GetParallelStatus() << endl;
      // print_vec(tvr);
      // cout << "buffer-fv: " << endl << tvw << endl;
      CLD->MultAdd(-1.0, tvw, res);
      // cout << "res after CLD update" << endl;
      // print_vec(tvr);
    }

    // cout << " x out : " << endl;
    // print_vec(tvx);

    // Array<TV> rcp(H); rcp = 0;
    // for (auto k : Range(H)) {
    //   rcp[k] = tvx(k);
    // }
    // x.Distribute();
    // x.Cumulate();
    // cout << "check against : " << endl;
    // print_vec(tvx);
    // for (auto k : Range(H)) {
    //   if (free_dofs && !free_dofs->Test(k)) continue;
    //   TV diff = rcp[k] - tvx(k);
    //   if (TVNorm(diff)>1e-14) cout << "diff x " << k << ": " << diff << endl;
    // }
    // cout << endl;


    // if (update_res) {
    //   cout << "output res:" << endl;
    //   print_vec(tvr);
    //   cout << "check output res!" << endl;
    //   tvsx = tvx;
    //   savex.SetParallelStatus(x.GetParallelStatus());
    //   check_res(res);
    //   cout << "check output res done" << endl;
    // }
    
  } // smoothfull

} // namespace amg

#include "amg_tcs.hpp"
