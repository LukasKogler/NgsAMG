#define FILE_AMGSM_CPP

#include "amg_smoother.hpp"

namespace amg {

  /** BaseSmoother **/

  void BaseSmoother :: Mult (const BaseVector & b, BaseVector & x) const
  {
    x = 0.0;
    MultAdd(1.0, b, x);
  } // BaseSmoother :: Mult

  void BaseSmoother :: MultTrans (const BaseVector & b, BaseVector & x) const
  {
    x = 0.0;
    MultAdd(1.0, b, x);
  } // BaseSmoother :: MultTrans

  void BaseSmoother :: MultTransAdd (double scal, const BaseVector & b, BaseVector & x) const
  {
    MultAdd(scal, b, x);
  } // BaseSmoother :: MultTrans


  void BaseSmoother :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    throw Exception("BaseSmoother :: MultAdd not overloaded!");
  }


  /** END BaseSmoother **/


  /** HybridGSS **/

  template<int BS>
  HybridGSS<BS> :: HybridGSS ( const shared_ptr<HybridGSS<BS>::TSPMAT> & amat,
			       const shared_ptr<ParallelDofs> & par_dofs,
			       const shared_ptr<BitArray> & atake_dofs)
    : BaseSmoother(make_shared<ParallelMatrix>(amat, par_dofs, par_dofs, PARALLEL_OP::C2D), par_dofs),
      free_dofs(atake_dofs), parallel_dofs(par_dofs),
      comm(par_dofs->GetCommunicator()), spmat(amat), A(*spmat)
  {
    name =  string("HybridGSS<") + to_string(BS) + string(">");
    auto & pds = *parallel_dofs;
    this->H = spmat->Height();
    this->mf_dofs = BitArray(H); mf_dofs.Clear();
    this->mf_exd  = BitArray(H); mf_exd.Clear();
    for (auto k:Range(H)) {
      if (pds.IsMasterDof(k) && ( (!free_dofs) || free_dofs->Test(k) )) {
	mf_dofs.SetBit(k);
	if (pds.GetDistantProcs(k).Size()) mf_exd.SetBit(k);
      }
    }
    SetUpMat();
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
    static Timer t(name+"::SetUpMat"); RegionTimer rt(t);
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
    Array<shared_ptr<TSPMAT_TM>> recv_mats(nexp_larger);
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
    rsds.SetSize(max2(nexp_smaller, nexp_larger));
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
  void add_diag (double val, double& a, const double& b) { a += val * fabs(b); }

  template<int BS> void set_v2m (Mat<BS,BS,double>& a, const Vec<BS*BS,double>& b)
  { for (auto k : Range(BS*BS)) a(k) = b(k); }
  void set_v2m (double& a, const Vec<1,double>& b) { a = b(0); }
  void set_v2m (double& a, const double& b) { a = b; }

  template<int BS> void set_m2v (Vec<BS*BS,double>& a, const Mat<BS,BS,double>& b)
  { for (auto k : Range(BS*BS)) a(k) = b(k); }
  void set_m2v (Vec<1,double>& a, const double& b)
  { a(0) = b; }
  void set_m2v (double& a, const double& b)
  { a = b; }

  template<int BS>
  void HybridGSS<BS> :: CalcDiag ()
  {
    static Timer t(name+"::CalcDiag"); RegionTimer rt(t);
    {
      static Timer tbar(name+"::CalcDiag-barrier"); RegionTimer rt(tbar);
      parallel_dofs->GetCommunicator().Barrier();
    }
    const TSPMAT & spm(*spmat);
    const ParallelDofs & pds(*parallel_dofs);
    TableCreator<int> cvp(H);
    for (;!cvp.Done(); cvp++) {
      for (auto k:Range(H))
	for (auto p:pds.GetDistantProcs(k))
	  cvp.Add(k,p);
    }
    constexpr int MS = BS*BS;
    static Timer tpd(name+"::CalcDiag-pardofs constr"); tpd.Start();
    shared_ptr<ParallelDofs> block_pds = nullptr;
    if constexpr(BS==1) {
	block_pds = parallel_dofs;
      }
    else {
      block_pds = make_shared<ParallelDofs>(pds.GetCommunicator(), cvp.MoveTable(), MS, false);
    }
    tpd.Stop();
    ParallelVVector<typename strip_vec<Vec<MS,double>>::type> pvec(block_pds, DISTRIBUTED);
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
    static Timer tcumu(name+"::CalcDiag-pardofs cumulate"); tcumu.Start();
    pvec.Cumulate();
    tcumu.Stop();
    diag.SetSize(H);
    for (auto k : Range(H)) {
      auto & diag_etr = diag[k];
      const auto & pve = pvec(k);
      set_v2m(diag_etr, pve);
    }
    // cout << "final diags: " << endl << diag << endl;
    static Timer tcalc(name+"::CalcDiag-calc"); tcalc.Start();
    for (auto k : Range(H)) {
      if (!free_dofs || free_dofs->Test(k))
	CalcInverse(diag[k]);
    }
    tcalc.Stop();
    // cout << "final inved diags: " << endl << diag << endl;
  } // HybridGSS<BS>::CalcDiag

  template<int BS, int RMIN, int RMAX>
  void StabHGSS<BS, RMIN, RMAX> :: CalcRegDiag ()
  {
    static Timer t(name+"::CalcRegDiag"); RegionTimer rt(t);
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
    //cout << "final diags: " << endl << diag << endl;
    constexpr int NR = RMAX-RMIN;
    Matrix<double> rblock(NR, NR), evecs(NR,NR);
    Vector<double> evals(NR), kv(NR);
    size_t nr1 = 0, nr2 = 0, nnr = 0;
    double trace = 0.0;
    for (auto k : Range(H)) {
      if (!free_dofs || free_dofs->Test(k)) {
	bool zero_block = true;
	auto & block = diag[k];
	for (int i : Range(NR))
	  if (block(RMIN+i, RMIN+i) != 0.0)
	    { zero_block = false; break; }
	if (zero_block) {
	  nr1++;
	  for (int i : Range(NR))
	    block(RMIN+i, RMIN+i) = 1.0;
	}
	else {
	  trace = 0;
	  for (auto k : Range(RMIN)) trace += block(k,k);
	  for (int k = RMAX; k < BS; k++) trace += block(k,k);
	  trace /= BS;
	  for (int i : Range(NR))
	    for (int j : Range(NR))
	      rblock(i,j) = block(RMIN+i, RMIN+j);
	  LapackEigenValuesSymmetric(rblock, evals, evecs);
	  bool reged = false;
	  for (int l = 0; l < NR; l++) {
	    if (fabs(evals(0)/evals(NR-1)) < 1e-15) {
	      reged = true;
	      kv = evecs.Rows(l, l+1);
	      double fac = trace / L2Norm(kv);
	      for (int i : Range(NR))
		for (int j : Range(NR))
		  block(RMIN+i, RMIN+j) += fac*kv(i)*kv(j);
	    }
	  }
	  if(reged) nr2++;
	  else nnr++;
	}
	// cerr << "stab inv block " << endl; print_tm(cerr, block); cerr << endl;
	CalcInverse(block);
	// cerr << "stab inved block " << endl; print_tm(cerr, block); cerr << endl;
      }
    }
    // cout << "REGED " << nr1+nr2 << " OF " << nr1+nr2+nnr << endl;
    // cout << "final inved diags: " << endl << diag << endl;
  } // StabHGSS<BS>::CalcDiag
  
  
  template<int BS>
  void HybridGSS<BS> :: gather_vec (const BaseVector & vec) const
  {
    static Timer t(string("HybridGSS<")+to_string(BS)+">::gather_vec");
    RegionTimer rt(t);
    FlatVector<TV> tvec = vec.FV<TV>();
    auto & pds = *parallel_dofs;
    auto ex_procs = pds.GetDistantProcs();
    rsds.SetSize0();
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
      rsds.Append(comm.ISend(p_buffer, p, MPI_TAG_AMG));
      // MPI_Request_free(&req);
    }
    if (nexp_larger==0) { MyMPI_WaitAll(rsds); return; }
    for (auto kkp : Range(nexp_larger)) {
      auto kp = nexp_smaller + kkp;
      auto p = ex_procs[kp];
      int sz = buf_os[kp+1] - buf_os[kp];
      // cout << "gather, recv " << sz << " from " << p << ", kp " << kp << endl;
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      rr_gather[kkp] = comm.IRecv(p_buffer, p, MPI_TAG_AMG);
    }
    int nrr = nexp_larger;
    MPI_Request* rrptr = rr_gather.Data();
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
    MyMPI_WaitAll(rsds);
  } // gather_vec

  template<int BS> void
  HybridGSS<BS> :: scatter_vec (const BaseVector & vec) const
  {
    static Timer t(string("HybridGSS<")+to_string(BS)+">::scatter_vec");
    RegionTimer rt(t);
    FlatVector<TV> fvec = vec.FV<TV>();
    auto & pds = *parallel_dofs;
    auto ex_procs = pds.GetDistantProcs();
    rsds.SetSize0();
    for (int kkp : Range(nexp_larger)) {
      int kp = nexp_smaller + kkp;
      auto p = ex_procs[kp];
      int sz = buf_os[kp+1] - buf_os[kp];
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      // cout << "scatter, send update to " << p << ", kp " << kp << " " << p_buffer.Size() << ": "; prow(p_buffer); cout << endl;
      rsds.Append(comm.ISend(p_buffer, p, MPI_TAG_AMG));
      // MPI_Request_free(&reqs); // TODO: am i SURE that this is OK??
    }
    if (nexp_smaller==0) { MyMPI_WaitAll(rsds); return; }
    for (int kp : Range(nexp_smaller)) {
      int sz = buf_os[kp+1] - buf_os[kp];
      // cout << "scatter, recv " << sz << " from " << p << ", kp " << kp << endl;
      FlatArray<TV> p_buffer (sz, &(buffer[buf_os[kp]]));
      rr_scatter[kp] = comm.IRecv(p_buffer, ex_procs[kp], MPI_TAG_AMG);
    }
    int nrr = nexp_smaller;
    MPI_Request * rrptr = rr_scatter.Data();
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
    MyMPI_WaitAll(rsds);
  } // scatter_vec

  
  template<int BS> Array<MemoryUsage>
  HybridGSS<BS> ::  GetMemoryUsage() const
  {
    string name = string("buffers,HybridGSS<") + to_string(BS) + string(">");
    size_t nbytes = 0, nblocks = 1;
    nbytes += buffer.Size() * sizeof(TV);
    nbytes += x_buffer.Size() * sizeof(TV);
    nbytes += ax_buffer.Size() * sizeof(TV);
    nbytes += buf_os.Size() * sizeof(int);
    nbytes += buf_cnt.Size() * sizeof(int);
    nbytes += rr_gather.Size() * sizeof(MPI_Request);
    nbytes += rr_scatter.Size() * sizeof(MPI_Request);
    nbytes += diag.Size() * sizeof(TM);
    Array<MemoryUsage> mus = { { name+string("buffers"), nbytes, nblocks } };
    if (addA != nullptr) {
      auto mu_addA = addA->GetMemoryUsage();
      for (auto & mu : mu_addA) mu.AddName("-addA");
      mus += mu_addA;
    }
    if (CLD != nullptr) {
      auto mu_CLD = CLD->GetMemoryUsage();
      for (auto & mu : mu_CLD) mu.AddName("-CLD");
      mus += mu_CLD;
    }
    return mus;
  }


  template<int H, int W>
  INLINE void SubsAxFromy (FlatVec<H> y, const Mat<H,W> & A, const Vec<W> & x)
  {
    Iterate<H>([&](auto i) {
	Iterate<W>([&](auto j) {
	    y(i.value) -= A(i.value,j.value) * x(j.value);
	  });
      });
  }
  template<int H, int W>
  INLINE void SubsAxFromy (Vec<H> &y, const Mat<H,W> & A, FlatVec<W> x)
  {
    Iterate<H>([&](auto i) {
	Iterate<W>([&](auto j) {
	    y(i.value) -= A(i.value,j.value) * x(j.value);
	  });
      });
  }
  template<int H, int W>
  INLINE void SubsAxFromy (FlatVec<H> &y, const Mat<H,W> & A, FlatVec<W> x)
  {
    Iterate<H>([&](auto i) {
	Iterate<W>([&](auto j) {
	    y(i.value) -= A(i.value,j.value) * x(j.value);
	  });
      });
  }
  INLINE void SubsAxFromy (double& y, const double & A, const double & x)
  { y -= A * x; }

  template<int H, int W>
  INLINE void SubsATxFromy (FlatVec<W> y, const Mat<H,W> & A, const Vec<H> & x)
  {
    Iterate<H>([&](auto j) {
	Iterate<W>([&](auto i) {
	    y(i.value) -= A(j.value,i.value) * x(j.value);
	  });
      });
  }
  template<int H, int W>
  INLINE void SubsATxFromy (Vec<W> & y, const Mat<H,W> & A, FlatVec<H> x)
  {
    Iterate<H>([&](auto j) {
	Iterate<W>([&](auto i) {
	    y(i.value) -= A(j.value,i.value) * x(j.value);
	  });
      });
  }
  template<int H, int W>
  INLINE void SubsATxFromy (FlatVec<W> y, const Mat<H,W> & A, FlatVec<H> x)
  {
    Iterate<H>([&](auto j) {
	Iterate<W>([&](auto i) {
	    y(i.value) -= A(j.value,i.value) * x(j.value);
	  });
      });
  }
  INLINE void SubsATxFromy (double& y, const double & A, const double & x)
  { y -= A * x; }



  template<int BS>
  void HybridGSS<BS> :: Smooth (BaseVector  &x, const BaseVector &b,
				BaseVector  &res, bool res_updated,
				bool update_res, bool x_zero) const
  {
    if (symmetric)
      { smoothfull(2, x, b, res, res_updated, update_res, x_zero); }
    else
      { smoothfull(0, x, b, res, res_updated, update_res, x_zero); }
  }


  template<int BS>
  void HybridGSS<BS> :: SmoothBack (BaseVector  &x, const BaseVector &b,
				    BaseVector &res, bool res_updated,
				    bool update_res, bool x_zero) const
  {
    if (symmetric)
      { smoothfull(2, x, b, res, res_updated, update_res, x_zero); }
    else
      { smoothfull(1, x, b, res, res_updated, update_res, x_zero); }
  }


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
  template<int BS> struct TEREF { typedef FlatVec<BS, double> T; };
  template<> struct TEREF<1> { typedef double& T; };
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
    string tname = string("HybridGSS<")+to_string(BS)+">::Smooth";
    static Timer t1 (tname);
    RegionTimer RT1(t1);
    RegionTimer RT2(hgss_timer_hack<BS>(tname, _type));
    static Timer tprep (tname+"-prep");
    static Timer tcloc (tname+"-calc");
    static Timer tpost (tname+"-post");

    const int type = _type;
    const bool res_updated = _res_updated;
    const bool update_res = _update_res;
    const bool x_zero = _x_zero;

    // auto comm = parallel_dofs->GetCommunicator();
    // auto rk = comm.Rank();
    // auto check_res = [&]() {
    //   auto pv1 = res.CreateVector(); auto & v1(*pv1); auto fv1 = v1.FV<double>(); auto tv1 = v1.FV<TV>();
    //   auto pv2 = res.CreateVector(); auto & v2(*pv2); auto fv2 = v2.FV<double>(); auto tv2 = v2.FV<TV>();
    //   tv1 = tvr; v1.SetParallelStatus(res.GetParallelStatus()); v1.Cumulate();
    //   tv2 = tvb; v2.SetParallelStatus(b.GetParallelStatus()); v2.Distribute();
    //   if (free_dofs) {
    // 	for (auto k : Range(fv1.Size()))
    // 	  if (!free_dofs->Test(k))
    // 	    { tvx(k) = 0; }
    //   }
    //   x.Cumulate();
    //   spmat->MultAdd(-1, x, v2);
    //   v2.Cumulate();
    //   fv1 -= fv2;
    //   if (free_dofs) {
    // 	for (auto k : Range(fv1.Size()))
    // 	  if (!free_dofs->Test(k))
    // 	    { fv1(k) = 0; }
    //   }
    //   auto nl = sqrt(InnerProduct(fv1, fv1));
    //   auto ng = sqrt(InnerProduct(v1, v1));
    //   auto nb = sqrt(InnerProduct(b,b));
    //   if (ng > 1e-8)
    // 	cout << "DIFF NORM RANK " << rk << ": " << nl << " " << ng << "   , norm b " << nb << endl;
    //   if (nl > 1e-8)
    // 	for (auto k : Range(fv1.Size()))
    // 	  { if (fabs(fv1(k)) > 1e-12) { cout << k << ": " << fv1(k) << " "; if (free_dofs) cout << free_dofs->Test(k); cout  << endl; } }
    //   comm.Barrier();
    // };

    if (type==2) {
      smoothfull(0, x, b, res, res_updated, true, x_zero);
      smoothfull(1, x, b, res, true, update_res, false);
      return;
    }
    else if (type==3) {
      smoothfull(1, x, b, res, res_updated, true, x_zero);
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

    tprep.Start();
    
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
	VFlatVector<TV> tvw (sz, buffer.Data());
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
    tprep.Stop();
    tcloc.Start();

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

    auto update_row_resu2 = [&](auto rownr,
				auto bf_in_min, auto bf_in_max,
				auto af_ex_min, auto af_ex_max) {
      if (!mf_dofs.Test(rownr)) return;
      TV w = diag[rownr] * tvr(rownr);
      tvx(rownr) += w;
      auto ris = A.GetRowIndices(rownr);
      auto rvs = A.GetRowValues(rownr);
      auto pos = ris.Pos(rownr);
      auto sz = ris.Size();
      // L update for future rows
      // [0..pos) or [pos+1,..sz)
      const int imin = af_ex_min(pos,sz);
      const int imax = af_ex_max(pos,sz);
      for (int l = imin; l < imax; l++) {
	// cout << "tvr( " << ris[l] << "/" << tvr.Size() << ") -= " << rvs[l] << " * " << w << endl;
	// cerr << "tvr( " << ris[l] << "/" << tvr.Size() << ") -= " << rvs[l] << " * " << w << endl;
	SubsATxFromy(tvr(ris[l]),rvs[l],w);
	// tvr(ris[l]) -= Trans(rvs[l]) * w;
      }
      if (update_res) {  // D + L.T update for prev. rows
	// [0..pos] or [pos..sz)
	const int imin = bf_in_min(pos,sz);
	const int imax = bf_in_max(pos,sz);
	for (int l = imin; l < imax; l++) {
	  SubsATxFromy(tvr(ris[l]), rvs[l], w);
	  // tvr(ris[l]) -= Trans(rvs[l]) * w;
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
	const int imin = af_ex_min(pos, sz);
	const int imax = af_ex_max(pos, sz);
      	for (int l = imin; l < imax; l++) {
	  SubsATxFromy(tvr(ris[l]), rvs[l], w);
	  // tvr(ris[l]) -= Trans(rvs[l]) * w;
	}
	if (update_res) {  // L.T update for prev. rows
	  const int imin = bf_in_min(pos, sz);
	  const int imax = bf_in_max(pos, sz);
	  for (int l = imin; l < imax; l++) {
	    SubsATxFromy(tvr(ris[l]), rvs[l], w);
	    // tvr(ris[l]) -= Trans(rvs[l]) * w;
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
    int pos_add(-1), sz_add(-1);
    auto update_row_resnotu = [&](auto rownr) {
      if (free_dofs && !free_dofs->Test(rownr)) return;
      const bool exrow = mf_exd.Test(rownr);
      auto & resval = tvr(rownr);
      // cout << "type " << typeid(resval).name() << endl;
      auto ris = A.GetRowIndices(rownr);
      auto rvs = A.GetRowValues(rownr);
      int pos = ris.Pos(rownr);
      int sz = ris.Size();
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

    auto update_row_resnotu2 = [&](auto rownr,
				   auto bf_ex_min, auto bf_ex_max,
				   auto bf_in_min, auto bf_in_max,
				   auto af_in_min, auto af_in_max) {
      if (free_dofs && !free_dofs->Test(rownr)) return;
      const bool exrow = mf_exd.Test(rownr);
      typename TEREF<BS>::T resval = tvr(rownr);
      // cout << "type " << typeid(resval).name() << endl;
      auto ris = A.GetRowIndices(rownr);
      auto rvs = A.GetRowValues(rownr);
      int pos = ris.Pos(rownr);
      int sz = ris.Size();
      // L * x_new -> exclude diag
      const int imin = bf_ex_min(pos,sz);
      const int imax = bf_ex_max(pos,sz);
      for (int l = imin; l < imax; l++) {
	SubsAxFromy(resval, rvs[l], tvx(ris[l]));
	// resval -= rvs[l] * tvx(ris[l]);
      }
      // if free & not master, only do L-update (per def only local mat needed!)
      if (!mf_dofs.Test(rownr)) return;
      TV tot_rv(0);
      // update res curr.row
      // (L.T+D) * x_old -> include diag
      const int imin2 = af_in_min(pos,sz);
      const int imax2 = af_in_max(pos,sz);
      for (int l = imin2; l < imax2; l++) {
	SubsAxFromy(tot_rv, rvs[l], tvx(ris[l]));
	// tot_rv -= rvs[l] * tvx(ris[l]);
      }
      if (exrow) {
	auto ris = addA->GetRowIndices(rownr);
	auto rvs = addA->GetRowValues(rownr);
	pos_add = ris.Pos(rownr);
	sz_add = ris.Size();
	// L * x_new -> exclude diag
	const int imin = bf_ex_min(pos_add,sz_add);
	const int imax = bf_ex_max(pos_add,sz_add);
	// cout << " res L up ex " << imin << " " << imax << endl;
      	for (int l = imin; l < imax; l++) {
	  SubsAxFromy(resval, rvs[l], tvx(ris[l]));
	  // resval -= rvs[l] * tvx(ris[l]);
	}
	// (L.T+D) * x_old -> include diag
	const int imin2 = af_in_min(pos_add,sz_add);
	const int imax2 = af_in_max(pos_add,sz_add);
      	for (int l = imin2; l < imax2; l++) {
	  SubsAxFromy(tot_rv, rvs[l], tvx(ris[l]));
	  //tot_rv -= rvs[l] * tvx(ris[l]);
	}
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
      	const int imin = bf_in_min(pos,sz);
	const int imax = bf_in_max(pos,sz);
	for (int l = imin; l < imax; l++) {
	  SubsATxFromy(tvr(ris[l]), rvs[l], xval);
	  //tvr(ris[l]) -= Trans(rvs[l]) * xval;
	}
	if (exrow) {
	  auto ris = addA->GetRowIndices(rownr);
	  auto rvs = addA->GetRowValues(rownr);
	  const int imin = bf_in_min(pos_add,sz_add);
	  const int imax = bf_in_max(pos_add,sz_add);
	  for (int l = imin; l < imax; l++) {
	    SubsATxFromy(tvr(ris[l]), rvs[l], xval);
	    //tvr(ris[l]) -= Trans(rvs[l]) * xval;
	  }
	}
      }
    };

    
    buf_cnt = 0;
    // if (type==0) { // FW
    //   if (res_updated)
    // 	for (size_t rownr = 0; rownr<H; rownr++)
    // 	  update_row_resu(rownr);
    //   else
    // 	for (size_t rownr = 0; rownr<H; rownr++)
    // 	  update_row_resnotu(rownr);
    // }
    // else { // BW
    //   if (res_updated)
    // 	for (int rownr = H-1; rownr>=0; rownr--)
    // 	  update_row_resu(rownr);
    //   else
    // 	for (int rownr = H-1; rownr>=0; rownr--)
    // 	  update_row_resnotu(rownr);
    // }

    if (type==0) { // FW
      auto bf_in_min = [](auto min, auto max){ return 0; };
      auto bf_in_max = [](auto min, auto max){ return min+1; };
      auto af_in_min = [](auto min, auto max){ return min; };
      auto af_in_max = [](auto min, auto max){ return max; };
      auto bf_ex_min = [](auto min, auto max){ return 0; };
      auto bf_ex_max = [](auto min, auto max){ return min; };
      auto af_ex_min = [](auto min, auto max){ return min+1; };
      auto af_ex_max = [](auto min, auto max){ return max; };
      if (res_updated)
	for (size_t rownr = 0; rownr<H; rownr++)
	  update_row_resu2(rownr, bf_in_min, bf_in_max, af_ex_min, af_ex_max);
      else
	for (size_t rownr = 0; rownr<H; rownr++)
	  update_row_resnotu2(rownr, bf_ex_min, bf_ex_max, bf_in_min, bf_in_max,
			      af_in_min, af_in_max);
    }
    else { // BW
      auto bf_in_min = [](auto min, auto max){ return min; };
      auto bf_in_max = [](auto min, auto max){ return max; };
      auto af_in_min = [](auto min, auto max){ return 0; };
      auto af_in_max = [](auto min, auto max){ return min+1; };
      auto bf_ex_min = [](auto min, auto max){ return min+1; };
      auto bf_ex_max = [](auto min, auto max){ return max; };
      auto af_ex_min = [](auto min, auto max){ return 0; };
      auto af_ex_max = [](auto min, auto max){ return min; };
      if (res_updated)
	for (int rownr = H-1; rownr>=0; rownr--)
	  update_row_resu2(rownr, bf_in_min, bf_in_max, af_ex_min, af_ex_max);
      else
	for (int rownr = H-1; rownr>=0; rownr--)
	  update_row_resnotu2(rownr, bf_ex_min, bf_ex_max, bf_in_min, bf_in_max,
			      af_in_min, af_in_max);
    }
    tcloc.Stop();
    tpost.Start();

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
      VFlatVector<TV> tvw (sz, buffer.Data());
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
    tpost.Stop();
  } // smoothfull


  void HiptMairSmoother :: Smooth (BaseVector &x, const BaseVector &b, BaseVector  &res,
				   bool res_updated, bool update_res, bool x_zero) const
  {
    // cout << " smrange " << typeid(*smrange).name() << endl;
    // cout << " smpot " << typeid(*smpot).name() << endl;
    // cout << " FW b " << endl << b << endl;
    // cout << " FW res " << endl << res << endl;
    // cout << " FW x " << endl << x << endl;
    // cout << " HMS FW " << endl;
    smrange->Smooth(x, b, res, res_updated, true, x_zero);
    // cout << " FW x smoothed " << endl << x << endl;
    // cout << " HMS FW " << endl;
    DT->Mult(res, *rhspot);
    // cout << " FW rhs-pot " << endl << *rhspot << endl;
    *solpot = 0;
    // cout << " HMS FW " << endl;
    smpot->Smooth(*solpot, *rhspot, *respot, false, false, true);
    // cout << " FW sol-pot " << endl << *solpot << endl;
    // cout << " HMS FW " << endl;
    D->MultAdd(1.0, *solpot, x);
    // cout << " FW x incl pot " << endl << x << endl;
    if (update_res)
      { res = b - (*Arange) * x; }
  }


  void HiptMairSmoother :: SmoothBack (BaseVector &x, const BaseVector &b, BaseVector &res,
				       bool res_updated, bool update_res, bool x_zero) const
  {

    if (res_updated)
      { DT->Mult(res, *rhspot); }
    else {
      if (x_zero)
	{ DT->Mult(b, *rhspot); }
      else {
	res = b - (*Arange) * x;
	DT->Mult(res, *rhspot);
      }
    }
    // cout << " BW rhs-pot " << endl << *rhspot << endl;
    *solpot = 0;
    // cout << " HMS BW " << endl;
    smpot->SmoothBack(*solpot, *rhspot, *respot, false, false, true);
    // cout << " BW sol-pot " << endl << *solpot << endl;
    // cout << " HMS BW " << endl;
    D->MultAdd(1.0, *solpot, x);
    // cout << " BW x " << endl << *solpot << endl;
    // cout << " HMS BW " << endl;
    smrange->SmoothBack(x, b, res, false, update_res, false);
    // cout << " BW x fin " << endl << x << endl;
  }

  void HiptMairSmoother :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    throw Exception("HiptMairSmoother::MultAdd not implemented (should be easy)");
  } // HiptMairSmoother::MultAdd

} // namespace amg


/** 

#include <python_ngstd.hpp>

namespace amg {

  void ExportSmoothers (py::module & m)
  {
    py::class_<HybridGSS<1>, shared_ptr<HybridGSS<1>>, BaseMatrix>
      (m, "HybridGSS", "scalar hybrid Gauss-Seidel")
      .def(py::init<>
	   ( [] (shared_ptr<BaseMatrix> mat, shared_ptr<BitArray> freedofs) {
	     shared_ptr<SparseMatrix<double>> spm;
	     shared_ptr<ParallelDofs> pardofs;
	     if ( auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat) ) {
	       spm = dynamic_pointer_cast<SparseMatrix<double>>(parmat->GetMatrix());
	       if (spm == nullptr)
		 { throw Exception("wrong mat type for hgss"); }
	       pardofs = parmat->GetParallelDofs();
	     }
	     else { // ok .. make dummy pardofs
	       spm = dynamic_pointer_cast<SparseMatrix<double>>(mat);
	       if (spm == nullptr)
		 { throw Exception("wrong mat type for hgss"); }
	       Array<int> perow (spm->Height() ); perow = 0;
	       Table<int> pds (perow);
	       pardofs = make_shared<ParallelDofs> ( AMG_ME_COMM , move(pds), GetEntryDim(spm.get()), false);
	     }
	     auto sm = make_shared<HybridGSS<1>>(spm, pardofs, freedofs);
	     BaseSmoother & bsm(*sm); bsm.Finalize();
	     return sm;
	   }), py::arg("mat"), py::arg("freedofs") = nullptr)
      .def ( "SmoothSym",
	     [&](shared_ptr<HybridGSS<1>> & sm, shared_ptr<BaseVector> & x,
		 shared_ptr<BaseVector> & b, shared_ptr<BaseVector> & res,
		 int k)
	     {
	       res->FVDouble() = b->FVDouble();
	       res->SetParallelStatus(b->GetParallelStatus());
	       sm->Smooth(*x, *b, *res, true, true, false);
	       for (int j = 0; j+1<k; j++) {
		 sm->SmoothBack(*x, *b, *res, true, true, false);
		 sm->Smooth(*x, *b, *res, true, true, false);
	       }
	       sm->SmoothBack(*x, *b, *res, true, false, false);
	     }, py::arg("sol"), py::arg("rhs"), py::arg("res"), py::arg("numits") = 1);


    py::class_<HybridGSS<3>, shared_ptr<HybridGSS<3>>, BaseMatrix>
      (m, "HybridGS3", "scalar hybrid Gauss-Seidel")
      .def(py::init<>
	   ( [] (shared_ptr<BaseMatrix> mat, shared_ptr<BitArray> freedofs) {
	     shared_ptr<SparseMatrix<Mat<3>>> spm;
	     shared_ptr<ParallelDofs> pardofs;
	     if ( auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat) ) {
	       spm = dynamic_pointer_cast<SparseMatrix<Mat<3>>>(parmat->GetMatrix());
	       if (spm == nullptr)
		 { throw Exception("wrong mat type for hgss"); }
	       pardofs = parmat->GetParallelDofs();
	     }
	     else { // ok .. make dummy pardofs
	       spm = dynamic_pointer_cast<SparseMatrix<Mat<3>>>(mat);
	       if (spm == nullptr)
		 { throw Exception("wrong mat type for hgss"); }
	       Array<int> perow (spm->Height() ); perow = 0;
	       Table<int> pds (perow);
	       pardofs = make_shared<ParallelDofs> ( AMG_ME_COMM , move(pds), GetEntryDim(spm.get()), false);
	     }
	     auto sm = make_shared<HybridGSS<3>>(spm, pardofs, freedofs);
	     BaseSmoother & bsm(*sm); bsm.Finalize();
	     return sm;
	   }), py::arg("mat"), py::arg("freedofs") = nullptr)
      .def ( "SmoothSym",
	     [&](shared_ptr<HybridGSS<3>> & sm, shared_ptr<BaseVector> & x,
		 shared_ptr<BaseVector> & b, shared_ptr<BaseVector> & res,
		 shared_ptr<BaseMatrix> & mat,
		 int k)
	     {

	       for (int j = 0; j+1<k; j++) {
		 sm->Smooth(*x, *b, *res, true, false, true);
		 sm->SmoothBack(*x, *b, *res, false, false, false);
	       }

	       // for (int j = 0; j<k; j++) {
	       // 	 sm->Smooth(*x, *b, *res, false, false, false);
	       // 	 sm->SmoothBack(*x, *b, *res, false, false, false);
	       // }

	       
	     }, py::arg("sol"), py::arg("rhs"), py::arg("res"), py::arg("mat"),py::arg("numits") = 1);

  }

} // namespace amg
**/

#include "amg_tcs.hpp"
