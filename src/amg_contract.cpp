#define FILE_AMGCTR_CPP

#ifdef USE_TAU
#include "TAU.h"
#endif

#include "amg.hpp"

#include <metis.h>
typedef idx_t idxtype;   


namespace amg
{

  Table<int> PartitionProcsMETIS (BlockTM & mesh, int nparts, bool sep_p0)
  {
    static Timer t("PartitionProcsMETIS"); RegionTimer rt(t);
    const auto & eqc_h(*mesh.GetEQCHierarchy());
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();
    Table<int> groups;
    if ( (nparts==1) || (sep_p0 && nparts==2)) {
      Array<int> perow(nparts); perow = 0; perow[0]++; perow.Last() += comm.Size()-1;
      groups = Table<int>(perow); perow = 0;
      groups[0][perow[0]++] = 0;
      for (auto k : Range(comm.Size()-1)) groups[nparts-1][perow[nparts-1]++] = k+1;
      return groups;
    }
    int root = 0;
    Array<size_t> all_nvs ( (comm.Rank()==root) ? comm.Size() : 0);
    size_t nv_loc = mesh.GetNN<NT_VERTEX>();
    MyMPI_Gather(nv_loc, all_nvs, comm, root);
    // per dp: dist-PROC, NV_SHARED,NE that would become loc (second not used ATM)
    auto ex_procs = eqc_h.GetDistantProcs();
    Array<INT<3,size_t>> data (ex_procs.Size()); data = 0;
    for (auto k : Range(data.Size())) data[k][0] = ex_procs[k];
    for (auto eqc : Range(neqcs)) {
      auto dps = eqc_h.GetDistantProcs(eqc);
      if (dps.Size()==1) {
	auto pos = ex_procs.Pos(dps[0]);
	data[pos][1] = mesh.GetENN<NT_VERTEX>(eqc);
      }
    }
    if (neqcs>0) {
      // these edges definitely become local through contracting
      auto pad_edges = mesh.GetCNodes<NT_EDGE>(0);
      for (const auto & edge : pad_edges) {
	AMG_Node<NT_VERTEX> vmax = max(edge.v[0], edge.v[1]);
	auto eq = mesh.GetEqcOfNode<NT_VERTEX>(vmax);
	if (eqc_h.GetDistantProcs(eq).Size() == 1) {
	  auto dp = eqc_h.GetDistantProcs(eq)[0];
	  auto pos = ex_procs.Pos(dp);
	  data[pos][2]++;
	}
      }
    }
    if (comm.Rank() != root) {
      /** Send  data to root **/
      comm.Send(data, root, MPI_TAG_AMG);
    }
    if (comm.Rank() == root) {
      /** Recv data from all ranks **/
      Array<Array<INT<3,size_t>>> gdata(comm.Size());
      gdata[root] = move(data);
      for (auto k : Range(comm.Size())) {
	if (k!=root) comm.Recv(gdata[k], k, MPI_TAG_AMG);
      }
      // generate metis graph structure
      Array<idx_t> partition (comm.Size()); partition = -1;
      Array<idx_t> v_weights(comm.Size());
      for (auto k : Range(comm.Size()))
	v_weights[k] = all_nvs[k];
      Array<idx_t> edge_firsti(comm.Size()+1); edge_firsti = 0;
      for (auto k : Range(comm.Size()))
	edge_firsti[k+1] = edge_firsti[k] + gdata[k].Size();
      Array<idx_t> edge_idx(edge_firsti.Last());
      Array<idx_t> edge_wt(edge_firsti.Last());
      int c = 0;
      for (auto k : Range(comm.Size())) {
	auto & row = gdata[k];
	for (auto j : Range(row.Size())) {
	  edge_idx[c] = row[j][0];
	  edge_wt[c] = row[j][1];
	  c++;
	}
      }
      idx_t nvts = idx_t(comm.Size());      // nr of vertices
      idx_t ncon = 1;                       // nr of balancing constraints
      idx_t* xadj = &(edge_firsti[0]);      // edge-firstis
      idx_t* adjncy = &(edge_idx[0]);       // edge-connectivity
      idx_t* vwgt = &(v_weights[0]);        // "computation cost"
      idx_t* vsize = NULL;                  // "comm. cost"
      idx_t* adjwgts = &(edge_wt[0]);       // edge-weights
      idx_t  m_nparts = sep_p0 ? nparts-1 : nparts; // nr of parts
      real_t* tpwgts = NULL;                // weights for each part (equal if NULL)
      real_t* ubvec = NULL;                 // tolerance
      idx_t metis_options[METIS_NOPTIONS];  // metis-options
      idx_t objval;                         // value of the edgecut/totalv of the partition
      idx_t * part = &partition[0];         // where to write the partition
      METIS_SetDefaultOptions(metis_options);
      metis_options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;         // minimize communication volume
      metis_options[METIS_OPTION_NCUTS] = (comm.Size()>1000) ? 1 : 2;  // metis will generate this many partitions and return the best
      // cout << "nvts: " << nvts << endl;
      // cout << "nparts: " << nparts << endl;
      // cout << "v_weights: " << endl; prow2(v_weights); cout << endl;
      // cout << "edge_firsti: " << endl; prow2(edge_firsti); cout << endl;
      // cout << "edge_idx: " << endl; prow(edge_idx); cout << endl;
      // cout << "edge_wt: " << endl; prow2(edge_wt); cout << endl;
      {
	static Timer t("METIS_PartGraphKway"); RegionTimer rt(t);
	int metis_code = METIS_PartGraphKway (&nvts, &ncon, xadj, adjncy, vwgt, vsize, adjwgts, &m_nparts, tpwgts,
					      ubvec, metis_options, &objval, part);
	if (metis_code != METIS_OK) {
	  switch(metis_code) {
	  case(METIS_ERROR_INPUT) : { cout << "METIS_ERROR_INPUT" << endl; break; }
	  case(METIS_ERROR_MEMORY) : { cout << "METIS_ERROR_MEMORY" << endl; break; }
	  case(METIS_ERROR) : { cout << "METIS_ERROR" << endl; break; }
	  default: { cout << "unknown metis return??" << endl; break; }
	  }
	  throw Exception("METIS did not succeed!!");
	}
      }
      // sort partition by min, rank it has
      
      TableCreator<int> cgs; // not (nparts), because in some cases metis gives enpty parts!!
      Array<int> arra(nparts); arra = comm.Size(); // empty grps will be sorted at the end
      for (auto k : Range(comm.Size())) arra[partition[k]] = min2(arra[partition[k]], k);
      Array<int> arrb(nparts); for (auto k : Range(nparts)) arrb[k] = k;
      QuickSortI(arra, arrb); for (auto k : Range(nparts)) arra[arrb[k]] = k;
      if (sep_p0) {
	for (; !cgs.Done(); cgs++) {
	  cgs.Add(0,0);
	  for (auto p : Range(1, comm.Size())) {
	    cgs.Add(arra[partition[p]]+1,p);
	  }
	}
      }
      else {
	for (; !cgs.Done(); cgs++) {
	  for (auto p : Range(comm.Size())) {
	    cgs.Add(arra[partition[p]],p);
	  }
	}
      }
      groups = cgs.MoveTable();
    }
    comm.Bcast(groups, root);
    // cout << "groups: " << endl << groups << endl;
    return groups;
  }

  
  template<class TV>
  CtrMap<TV> :: CtrMap (shared_ptr<ParallelDofs> _pardofs, shared_ptr<ParallelDofs> _mapped_pardofs,
			Array<int> && _group, Table<int> && _dof_maps)
    : BaseDOFMapStep(_pardofs, _mapped_pardofs), group(move(_group)), master(group[0]), dof_maps(move(_dof_maps)) 
  {
    auto comm = pardofs->GetCommunicator();
    is_gm = comm.Rank() == master;
  }

  template<class TV>
  CtrMap<TV> :: ~CtrMap ()
  {
    for (auto k:Range(mpi_types.Size()))
      MPI_Type_free(&mpi_types[k]);
  }


  INLINE Timer & timer_hack_ctr_f2c () { static Timer t("CtrMap::F2C"); return t; }
  INLINE Timer & timer_hack_ctr_c2f () { static Timer t("CtrMap::C2F"); return t; }


  template<class TV>
  void CtrMap<TV> :: TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("CtrMap::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_ctr_f2c());
    /** 
	We HAVE to null out coarse vec, because there are DOFs that get vals from multiple
	sources that have to be added up, and because we can have DOFs that cannot be mapped "within-group-locally".
	These DOFs have to be nulled so we have no garbage vals left.
     **/
    if (x_coarse != nullptr)
      { x_coarse->FVDouble() = 0; x_coarse->SetParallelStatus(DISTRIBUTED); }
    AddF2C(1.0, x_fine, x_coarse);
  }


  template<class TV>
  void CtrMap<TV> :: AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("CtrMap::AddF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_ctr_f2c());
    /** 
	We can have DOFs that cannot be mapped "within-group-locally".
	typically this does not matter, because F2C works with DISTRIBUTED vectors
	anyways
     **/
    x_fine->Distribute();
    auto fvf = x_fine->FV<TV>();
    auto& comm(pardofs->GetCommunicator());
    if (!is_gm) {
      if (fvf.Size() > 0)
	{ MPI_Send(x_fine->Memory(), fvf.Size(), MyGetMPIType<TV>(), group[0], MPI_TAG_AMG, comm); }
      return;
    }
    x_coarse->Distribute();
    auto fvc = x_coarse->FV<TV>();
    auto loc_map = dof_maps[0];
    int nreq_tot = 0;
    for (size_t kp = 1; kp < group.Size(); kp++)
      if (dof_maps[kp].Size()>0) { nreq_tot++; reqs[kp] = MyMPI_IRecv(buffers[kp], group[kp], MPI_TAG_AMG, comm); }
      else reqs[kp] = MPI_REQUEST_NULL;
    for (auto j : Range(loc_map.Size()))
      fvc(loc_map[j]) += fac * fvf(j);
    MPI_Request* rrptr = &reqs[0]; int nrr = group.Size();
    int kp; reqs[0] = MPI_REQUEST_NULL;
    for (int nreq = 0; nreq < nreq_tot; nreq++) {
      MPI_Waitany(nrr, rrptr, &kp, MPI_STATUS_IGNORE);
      auto map = dof_maps[kp]; auto buf = buffers[kp];
      for (auto j : Range(map.Size()))
	fvc(map[j]) += fac * buf[j];
    }
    // cout << " x coarse is: " << endl << fvc << endl;
  }


  template<class TV>
  void CtrMap<TV> :: TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("CtrMap::TransferC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_ctr_c2f());
    auto& comm(pardofs->GetCommunicator());
    x_fine->SetParallelStatus(CUMULATED);
    auto fvf = x_fine->FV<TV>();
    if (!is_gm)
      {
	if (fvf.Size() > 0)
	  { MPI_Recv(x_fine->Memory(), fvf.Size(), MyGetMPIType<TV>(), group[0], MPI_TAG_AMG, comm, MPI_STATUS_IGNORE); }
	return;
      }

    x_coarse->Cumulate();
    auto fvc = x_coarse->FV<TV>();
    for (size_t kp = 1; kp < group.Size(); kp++) {
      if (dof_maps[kp].Size() > 0)
	{ MPI_Isend( x_coarse->Memory(), 1, mpi_types[kp], group[kp], MPI_TAG_AMG, comm, &reqs[kp]); }
      else
	{ reqs[kp] = MPI_REQUEST_NULL; }
    }
    auto loc_map = dof_maps[0];
    for (auto j : Range(loc_map.Size()))
      { fvf(j) = fvc(loc_map[j]); }
    reqs[0] = MPI_REQUEST_NULL; MyMPI_WaitAll(reqs);
  }


  template<class TV>
  void CtrMap<TV> :: AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("CtrMap::AddC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    /**
       some values are transfered to multiple ranks 
       does not matter because coarse grid vectors are typically CUMULATED
     **/
    RegionTimer rt(timer_hack_ctr_c2f());
    auto& comm(pardofs->GetCommunicator());
    x_fine->Cumulate();
    auto fvf = x_fine->FV<TV>();
    if (!is_gm)
      {
	if (fvf.Size() > 0) {
	  auto b0 = buffers[0];
	  comm.Recv(b0, group[0], MPI_TAG_AMG);
	  for (auto k : Range(fvf.Size()))
	    { fvf(k) += fac * b0[k]; }
	}
	return;
      }

    x_coarse->Cumulate();
    auto fvc = x_coarse->FV<TV>();
    for (size_t kp = 1; kp < group.Size(); kp++) {
      if (dof_maps[kp].Size() > 0)
	{ MPI_Isend( x_coarse->Memory(), 1, mpi_types[kp], group[kp], MPI_TAG_AMG, comm, &reqs[kp]); }
      else
	{ reqs[kp] = MPI_REQUEST_NULL; }
    }
    auto loc_map = dof_maps[0];
    for (auto j : Range(loc_map.Size()))
      { fvf(j) += fac * fvc(loc_map[j]); }
    reqs[0] = MPI_REQUEST_NULL; MyMPI_WaitAll(reqs);
  }


  template<class TV>
  bool CtrMap<TV> :: DoSwap (bool in)
  {
    auto comm = pardofs->GetCommunicator();
    int doit = in;
    if (is_gm) {
      for (auto p : group)
	if (p != comm.Rank())
	  { comm.Send(in, p, MPI_TAG_AMG); }
    }
    else
      { comm.Recv(doit, master, MPI_TAG_AMG); }
    return doit != 0;
  }


  template<class TV>
  shared_ptr<ProlMap<SparseMatrixTM<typename CtrMap<TV>::TM>>> CtrMap<TV> :: SwapWithProl (shared_ptr<ProlMap<SparseMatrixTM<TM>>> pm_in)
  {

    NgsAMG_Comm comm = pardofs->GetCommunicator();

    cout << "SWAP WITH PROL, AM RANK " << comm.Rank() << " of " << comm.Size() << endl;
    cout << "loc glob nd " << pardofs->GetNDofLocal() << " " << pardofs->GetNDofGlobal() << endl;
    
    // TODO: new (mid) paralleldofs !?

    shared_ptr<SparseMatrixTM<TM>> split_A;
    // shared_ptr<ParallelDofs> mid_pardofs;


    if (!is_gm) {
      cout << " get split A from " << master << endl;

      comm.Recv(split_A, master, MPI_TAG_AMG);

      cout << " got split A: " << endl << *split_A << endl;

      // do sth like that if whe actually need non-dummy ones
      // Table<int> pd_tab; comm.Recv(pd_tab, master, MPI_TAG_AMG);
      // mid_pardofs = make_shared<ParallelDofs> (move(pd_tab), comm, pardofs->GetEntrySize(), false);
    }
    else { // is_gm
      cout << "swap 2 " << endl;

      auto P = pm_in->GetProl();

      cout << "prol dims " << P->Height() << " x " << P->Width() << endl;
      cout << " mapped loc glob nd " << mapped_pardofs->GetNDofLocal() << " " << mapped_pardofs->GetNDofGlobal() << endl;
    
    
      /**
	 For each member:
         - get rows of P corresponding to it's DOFs (keep col-indices) [Pchunk]
	 - col-inds in Pchunk are the new DOFs of that member, map them, write new dof_map for that member
	 - reverse-map col-inds of Pchunk
	 - send Pchunk to member
      **/

      const auto& rP(*P);
      Array<Array<int>> pdm(group.Size()); // dof-maps after prol
      BitArray colspace(P->Width());
      Array<int> rmap(P->Width());

      /**
	 Get rows contained in dmap out of prol, re-map colnrs from 0..N, where N is the number
	 of DOFs in the pre-image of the rows.

	 Write new dof_map into new_dmap.

	 Also make dp-table for new ParallelDofs.
      **/
      auto get_rows = [&](FlatArray<int> dmap, Array<int> & new_dmap, int for_rank) LAMBDA_INLINE {
	cout << " gr 1 " << endl;
	colspace.Clear();
	int cnt_cols = 0;
	Array<int> perow(dmap.Size());
	for (auto k : Range(dmap)) {
	  auto rownr = dmap[k];
	  auto ri = rP.GetRowIndices(rownr);
	  perow[k] = ri.Size();
	  for (auto col : ri) {
	    if (!colspace.Test(col))
	      { colspace.Set(col); cnt_cols++; }
	  }
	}
	cout << " gr 2 " << endl;
	rmap = -1;
	new_dmap.SetSize(cnt_cols); cnt_cols = 0;
	for (auto k : Range(P->Width())) {
	  if (colspace.Test(k))
	    { rmap[k] = cnt_cols; new_dmap[cnt_cols++] = k; }
	}
	cout << " gr 3 " << endl;
	auto Pchunk = make_shared<SparseMatrixTM<TM>>(perow, cnt_cols);
	const auto &rPchunk (*Pchunk);
	for (auto k : Range(dmap)) {
	  auto rownr = dmap[k];
	  auto ri_o = rP.GetRowIndices(rownr);
	  auto ri_n = rPchunk.GetRowIndices(k);
	  auto rv_o = rP.GetRowValues(rownr);
	  auto rv_n = rPchunk.GetRowValues(k);
	  int c = 0;
	  for (auto j : Range(ri_o)) {
	    auto newcol = rmap[ri_o[j]];
	    if (newcol != -1) {
	      ri_n[c] = newcol;
	      rv_n[c++] = rv_o[j];
	    }
	  }
	}

	cout << " gr ret " << endl;
	return Pchunk;
      }; // get_rows
      cout << "swap 3 " << endl;


      /**
	 New dist-procs: DUMMIES, should not ever need them
       
	 But we could do this if we need them:
	 -        C
	 - MID  ---->  FIN
	 -  ^            ^
	 P  |            | P_old
	 -  |            |
	 - INI  ---->  CRS
	 -    C_old

	 Have access to INI, CRS and FIN pardofs. Need to construct MID pardofs.

	 For each FIN DOF, masters know who, in their group, has them.
       
	 So, for each contracted ex-proc, write pre-contracted members for all ex-dofs into a table
	 and exchange tables. Merge these tables into one NDOF-sized one, then break that up into
	 group.Size() small tables which we distribute.
      **/
      {
	// Array<MPI_Request> sreq(group.Size());

	// auot cexps = mapped_pardofs->GetDistantProcs();
	// Array<int> perow(cexps.Size());
	// for (auto k : Range(perow))
	// 	{ perow[k] = 1 + mapped_pardofs->GetExchangeDofs(group[k]).Size(); }

	// auto proc_to_ind = [&](auto p) LAMBDA_INLINE { return find_in_sorted_array(p, cexps); }
	// auto iterate_dofs = [&](auto fun) LAMBDA_INLINE {
	// 	for (auto k : Range(group)) {
	// 	  FlatArray<int> map = dof_maps[k];
	// 	  for (auto j : Range(map)) {
	// 	    for (auto p : mapped_parodfs->GetDistantProcs(map[j]))
	// 	      { fun(map[j], p); }
	// 	  }
	// 	}
	// };
	// iterate_dofs([&](auto d, auto p) { perow[proc_to_ind(p)]++; });
	// Table<int> buffers(perow);
	// for (auto k : Range(perow))
	// 	{ perow[k] = mapped_pardofs->GetExchangeDofs(group[k]).Size(); }
	// // "firsti"
	// iterate_dofs([&](auto d, auto p) { perow[proc_to_ind(p)]++; });

	// MyMPI_WaitAll(sreq);
      }

      cout << " group "; prow(group); cout << endl;

      for (auto kp : Range(group)) {
	cout << " mem " << kp << " rk " << group[kp] << endl;
	auto Apart = get_rows(dof_maps[kp], pdm[kp], group[kp]);
	cout << "Apart for mem " << kp << " rk " << group[kp] << ": " << endl;
	cout << "pchunk dims for mem   " << kp << " rk " << group[kp] << ": " << Apart->Height() << " x " << Apart->Width() << endl;
	// cout << *Apart << endl;
	if (group[kp] == comm.Rank())
	  { split_A = Apart; }
	else
	  { comm.Send(*Apart, group[kp], MPI_TAG_AMG); }
	cout << "DONE sending Apart for mem " << kp << " rk " << group[kp] << ": " << endl;
      }
      cout << "swap 4 " << endl;

      Array<int> perow(group.Size());
      for (auto k : Range(perow))
	{ perow[k] = pdm[k].Size(); }
      dof_maps = Table<int>(perow);
      for (auto k : Range(perow))
	{ dof_maps[k] = pdm[k]; }
      cout << "swap 5 " << endl;

    } // is_gm

    // dummy (!!) pardofs
    Array<int> perow(split_A->Width()); perow = 0;
    Table<int> dps(perow);
    auto mid_pardofs = make_shared<ParallelDofs> (comm, move(dps), pardofs->GetEntrySize(), pardofs->IsComplex());
 
    auto prol_map = make_shared<ProlMap<SparseMatrixTM<TM>>> (split_A, pardofs, mid_pardofs);

    pardofs = mid_pardofs;
    mapped_pardofs = (pm_in == nullptr) ? nullptr : pm_in->GetMappedParDofs();

    cout << "SWP done " << endl;

    cout << "SPLIT PROL DIMS  " << split_A->Height() << " x " << split_A->Width() << endl;
    return prol_map;
  } // CtrMap::SplitProl


  template<class TV>
  void CtrMap<TV> :: SetUpMPIStuff ()
  {
    if (is_gm) {
      mpi_types.SetSize(group.Size());
      Array<int> ones; size_t max_s = 0;
      for (auto k : Range(group.Size())) max_s = max2(max_s, dof_maps[k].Size());
      ones.SetSize(max_s); ones = 1;
      for (auto k : Range(group.Size())) {
	auto map = dof_maps[k]; auto ms = map.Size();
	MPI_Type_indexed(ms, &ones[0], &map[0], MyGetMPIType<TV>(), &mpi_types[k]);
	MPI_Type_commit(&mpi_types[k]);
      }
      reqs.SetSize(group.Size()); reqs = MPI_REQUEST_NULL;
      Array<int> perow(group.Size()); perow[0] = 0;
      for (auto k : Range(size_t(1), group.Size())) perow[k] = dof_maps[k].Size();
      buffers = Table<TV>(perow);
    }
    else {
      Array<int> perow(1); perow[0] = pardofs->GetNDofLocal();
      buffers = Table<TV>(perow);
    }
  }


  INLINE Timer & timer_hack_ctrmat (int nr) {
    switch(nr) {
    case (0): { static Timer t("CtrMap::AssembleMatrix"); return t; }
    case (1): { static Timer t("CtrMap::AssembleMatrix - gather mats"); return t; }
    case (2): { static Timer t("CtrMap::AssembleMatrix - merge graph"); return t; }
    case (3): { static Timer t("CtrMap::AssembleMatrix - fill"); return t; }
    default: { break; }
    }
    static Timer t("CtrMap::AssembleMatrix - ???"); return t;
  }
  template<class TV> shared_ptr<typename CtrMap<TV>::TSPM>
  CtrMap<TV> :: DoAssembleMatrix (shared_ptr<typename CtrMap<TV>::TSPM> mat) const
  {

    const_cast<CtrMap<TV>&>(*this).SetUpMPIStuff();

    RegionTimer rt(timer_hack_ctrmat(0));
    NgsAMG_Comm comm(pardofs->GetCommunicator());

    if (!is_gm) {
      comm.Send(*mat, group[0], MPI_TAG_AMG);
      return nullptr;
    }

    // cout << "CTR MAT FOR " << pardofs->GetNDofGlobal() << " NDOF TO " << mapped_pardofs->GetNDofGlobal() << endl;
    // cout << "LOCAL DOFS " << pardofs->GetNDofLocal() << " NDOF TO " << mapped_pardofs->GetNDofLocal() << endl;
    // if (pardofs->GetNDofGlobal() < 10000)
    //   { cout << "CTR MAT FOR " << pardofs->GetNDofGlobal() << " NDOF " << endl; }

    timer_hack_ctrmat(1).Start();
    Array<shared_ptr<TSPM_TM> > dist_mats(group.Size());
    dist_mats[0] = mat;
    for(auto k:Range((size_t)1, group.Size())) {
      // cout << " get mat from " << k << " of " << group.Size() << endl;
      comm.Recv(dist_mats[k], group[k], MPI_TAG_AMG);
      // cout << " got mat from " << k << " of " << group.Size() << ", rank " << group[k] << endl;
      // cout << *dist_mats[k] << endl;
    }
    timer_hack_ctrmat(1).Stop();

    size_t cndof = mapped_pardofs->GetNDofLocal();

    // reverse map: maps coarse dof to (k,j) such that disp_mats[k].Row(j) maps to this row!
    TableCreator<INT<2, size_t>> crm(cndof);
    for (; !crm.Done(); crm++)
      for(auto k:Range(dof_maps.Size())) {
	auto map = dof_maps[k];
	for(auto j:Range(map.Size()))
	  crm.Add(map[j],INT<2, size_t>({k,j}));
      }
    auto reverse_map = crm.MoveTable();
    
    timer_hack_ctrmat(2).Start();
    Array<int*> merge_ptrs(group.Size()); Array<int> merge_sizes(group.Size());
    Array<int> perow(cndof);
    // we already buffer the col-nrs here, so we do not need to merge twice
    size_t max_nze = 0; for(auto k:Range(dof_maps.Size())) max_nze += dist_mats[k]->NZE();
    Array<int> colnr_buffer(max_nze); max_nze = 0; int* col_ptr = &(colnr_buffer[0]);
    Array<int> mc_buffer; // use this for merging with rows of LOCAL mat (which I should not change!!)
    Array<int> inds;
    auto QS_COL_VAL = [&inds](auto & cols, auto & vals) {
      auto S = cols.Size(); inds.SetSize(S); for (auto i:Range(S)) inds[i] = i;
      QuickSortI(cols, inds);
      for (int i:Range(S)) { // in-place permute
	// swap through one circle; use target as buffer as buffer
	if (inds[i] == -1) continue;
	int check = i; int from_here;
	while ( ( from_here = inds[check]) != i ) { // stop when we are back
	  swap(cols[check], cols[from_here]);
	  swap(vals[check], vals[from_here]);
	  inds[check] = -1; // have right ntry for pos. check
	  check = from_here; // check that position next
	}
	inds[check] = -1;
      }
    };
    for (auto rownr : Range(cndof)) {
      auto rmrow = reverse_map[rownr];
      int n_merge = rmrow.Size();
      if (n_merge == 0) { perow[rownr] = 0; continue; } // empty row
      else if (n_merge == 1) { // row from only one proc
	int km = rmrow[0][0], jm = rmrow[0][1];
	auto dmap = dof_maps[km];
	auto ris = dist_mats[km]->GetRowIndices(jm); auto riss = ris.Size();
	perow[rownr] = riss;
	FlatArray<int> cols(riss, col_ptr+max_nze); max_nze += riss;
	int last = 0; bool needs_sort = false;
	for (auto l : Range(riss)) {
	  cols[l] = dmap[ris[l]];
	  if (cols[l] < last) needs_sort = true;
	  last = cols[l];
	}
	if (needs_sort) QuickSort(cols);
      }
      else { // we have to merge rows
	perow[rownr] = 0;
	merge_ptrs.SetSize(n_merge); merge_sizes.SetSize(n_merge);
	for (auto j : Range(n_merge)) {
	  int km = rmrow[j][0], jm = rmrow[j][1];
	  auto map = dof_maps[km];
	  auto ris = dist_mats[km]->GetRowIndices(jm); auto riss = ris.Size();
	  auto rvs = dist_mats[km]->GetRowValues(jm);
	  int last = 0; bool needs_sort = false;
	  if (km!=0) { // use ris as storage
	    for (auto l : Range(riss)) {
	      ris[l] = map[ris[l]]; // re-map to contracted dof-nrs
	      if (ris[l] < last) needs_sort = true;
	      last = ris[l];
	    }
	    if (needs_sort) QS_COL_VAL(ris, rvs);
	    merge_ptrs[j] = &ris[0];
	  }
	  else  {
	    mc_buffer.SetSize(riss);
	    for (auto l : Range(riss)) {
	      mc_buffer[l] = map[ris[l]]; // re-map to contracted dof-nrs
	      if (mc_buffer[l] < last) needs_sort = true;
	      last = mc_buffer[l];
	    }
	    if (needs_sort) QuickSort(mc_buffer);
	    merge_ptrs[j] = &mc_buffer[0];
	  }
	  merge_sizes[j] = riss;
	}
	MergeArrays(merge_ptrs, merge_sizes, [&](auto k) { perow[rownr]++; colnr_buffer[max_nze++] = k; });
      }
    }
    timer_hack_ctrmat(2).Stop();

    // cout << "perow: " << endl; prow2(perow); cout << endl;
    // cout << "colnr-buffer: " << endl; prow(colnr_buffer); cout << endl;
    
    timer_hack_ctrmat(3).Start();
    max_nze = 0;
    auto cmat = make_shared<TSPM> (perow, cndof);
    for (auto rownr : Range(cndof)) {
      auto ris = cmat->GetRowIndices(rownr);
      auto rvs = cmat->GetRowValues(rownr);
      for (auto j : Range(ris.Size())) ris[j] = colnr_buffer[max_nze++];
      // cout << "set ris row " << rownr << endl; prow2(ris); cout << endl;
      auto rmrow = reverse_map[rownr];
      if (rmrow.Size()==1) {
	int km = rmrow[0][0], jm = rmrow[0][1];
	auto dmap = dof_maps[km];
	auto dris = dist_mats[km]->GetRowIndices(jm); ;
	auto drvs = dist_mats[km]->GetRowValues(jm);
	for (auto j : Range(dris.Size())) {
	  // cout << "j " << j << " dri " << dris[j] << " maps to " << dmap[dris[j]] << endl;
	  auto pos = find_in_sorted_array (dmap[dris[j]], ris);
	  // cout << "find " << "dmap[" << dris[j] << "] = " << dmap[dris[j]] << " at pos " << pos << endl;
	  rvs[pos] = drvs[j];
	}
      }
      else {
	rvs = 0;
	for (auto j : Range(rmrow.Size())) {
	  int km = rmrow[j][0], jm = rmrow[j][1];
	  auto dmap = dof_maps[km];
	  auto dris = dist_mats[km]->GetRowIndices(jm);
	  auto drvs = dist_mats[km]->GetRowValues(jm);
	  for (auto j : Range(dris.Size())) {
	    auto pos = (km==0) ? find_in_sorted_array (dmap[dris[j]], ris) : //have to re-map ...
	      find_in_sorted_array (dris[j], ris); // already mapped from merge!
	    rvs[pos] += drvs[j];
	  }
	}
      }
    }
    timer_hack_ctrmat(3).Stop();

    // cout << "contracted mat " << cmat->Height() << " x " << cmat->Width() << endl;

    return cmat;
  }
  
  
  INLINE Timer & timer_hack_gcmc () { static Timer t("GridContractMap constructor"); return t; }
  template<class TMESH> GridContractMap<TMESH> :: GridContractMap (Table<int> && _groups, shared_ptr<TMESH> _mesh)
    : GridMapStep<TMESH>(_mesh), eqc_h(_mesh->GetEQCHierarchy()), groups(_groups), node_maps(4), annoy_nodes(4)
  {
    RegionTimer rt(timer_hack_gcmc());
    BuildCEQCH();
    BuildNodeMaps();
  } // GridContractMap (..)



  
  /** 
  	There are annoying edges: 
  	   if edge is AC -- CB (with A\cut B=empty)
  	   then by now only contr(C) know about the edge,
  	   but it has to be in [contr(A)\cut contr(B)]\union contr(C)
  	Overall, there are 4 types of edges:
  	  - II-edges: go from in in meq to in in ceq
  	  - CI-edges: go from cross in meq to in in cut(coarse)
  	  - CC-edges: go from cross in meq to cross in ceq
  	  - annoy-edges: !!!!these can be IN or CROSS!!!!
  	NOTE: CC cannot go to another ceq!!
  	We do this:
  	- make table from annoying edges
  	- ReduceTable with annoying ones (all cross per definition)
  	
  	- now we have all edges we need and we can make contracted egde tables
  	  and edge maps:
  	  ordering of I-edges within each ceqc is:
  	     CEQ: [ IImeq1, IImeq2, IImeq3, ... ; CImeq1, CImeq2, ... ; ANNOY ]
                 (only meqs that map to ceq)  |  (possibly all meqs)
  	  ordering of C-edges within each ceq is:
  	     CEQ: [ CCmeq1, CCmeq2, ... ; ANNOY]
  	  NOTE: Map of any annoying edges has to be done by Pos(.)
  	        for the contr. annoy-edges array.
  		Rest of map is deterministic
  **/
  INLINE Timer & timer_hack_nmap () { static Timer t("GridContractMap :: BuildNodeMaps"); return t; }
  template<class TMESH> void GridContractMap<TMESH> :: BuildNodeMaps ()
  {
    RegionTimer rt(timer_hack_nmap());

    const auto & f_eqc_h(*this->eqc_h);
    auto comm = f_eqc_h.GetCommunicator();

    // cout << "local mesh: " << endl << *this->mesh << endl;
    
    if (!is_gm) {
      shared_ptr<BlockTM> btm = this->mesh;
      // cout << "send mesh to " << my_group[0] << endl;
      comm.Send(btm, my_group[0], MPI_TAG_AMG);
      // cout << "send mesh done" << endl;
      mapped_mesh = nullptr;
      if constexpr(std::is_same<TMESH, BlockTM>::value == 0) {
	  this->mesh->MapData(*this);
	}
      return;
    }

    const auto & c_eqc_h(*this->c_eqc_h);
    
    const TMESH & f_mesh(*this->mesh);
    auto p_c_mesh = make_shared<BlockTM>(this->c_eqc_h);
    auto & c_mesh(*p_c_mesh);

    // per definition
    for (NODE_TYPE NT : {NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL} ) {
      c_mesh.has_nodes[NT] = f_mesh.has_nodes[NT];
      c_mesh.nnodes_glob[NT] = f_mesh.nnodes_glob[NT];
    }
    
    int mgs = my_group.Size();
    Array<shared_ptr<BlockTM>> mg_btms(mgs); // (BlockTM on purpose)
    mg_btms[0] = this->mesh;
    for (size_t k = 1; k < my_group.Size(); k++) {
      // cout << "get mesh from " << my_group[k] << endl;
      comm.Recv(mg_btms[k], my_group[k], MPI_TAG_AMG);
      // cout << "got mesh from " << my_group[k] << endl;
      // cout << *mg_btms[k] << endl;
    }

    // constexpr int lhs = 1024*1024;
    // LocalHeap lh_max (lhs, "Max");
    // auto cut = [&lh_max](const auto & a, const auto & b) {
    //   Array<typename std::remove_reference<decltype(a[0])>::type > out(min2(a.Size(), b.Size()),lh_max);
    //   out.SetSize0();
    //   size_t ca = 0, cb = 0;
    //   while ( (ca<a.Size()) && (cb<b.Size()) )
    // 	if (a[ca]<b[cb]) ca++;
    // 	else if (a[ca]>b[cb]) cb++;
    // 	else { out.Append(a[ca]); ca++; cb++; }
    //   return std::move(out);
    // };
    auto cut_min = [](const auto & a, const auto & b) {
      size_t ca = 0, cb = 0;
      while ( (ca<a.Size()) && (cb<b.Size()) )
    	if (a[ca]<b[cb]) ca++;
    	else if (a[ca]>b[cb]) cb++;
    	else { return a[ca]; }
      return -1;
    };

    size_t mneqcs = map_mc.Size();
    size_t cneqcs = c_eqc_h.GetNEQCS();

    /** for when we need a unique proc to take an eqc from!!  **/
    Array<int> eqc_sender(mneqcs);
    for (auto k : Range(mneqcs)) {
      auto mems = mmems[k];
      // eqc_sender[k] = cut(mems, my_group)[0];
      eqc_sender[k] = cut_min(mems, my_group);
    }    

    // cout << "eqc_sender: " << endl; prow2(eqc_sender); cout << endl;
    
    /** vertices **/
    auto & v_dsp = c_mesh.disp_eqc[NT_VERTEX];
    v_dsp.SetSize(cneqcs+1); v_dsp = 0;
    Array<size_t> firsti_v(mneqcs);
    firsti_v = 0;
    for (auto k : Range(my_group.Size())) {
      for (auto j : Range(mg_btms[k]->GetNEqcs())) {
	auto eqc_vs = mg_btms[k]->template GetENodes<NT_VERTEX>(j);
	if (my_group[k]==eqc_sender[map_om[k][j]]) {
	  v_dsp[map_oc[k][j]+1] += eqc_vs.Size();
	  firsti_v[map_om[k][j]] = eqc_vs.Size();
	}
      }
    }

    for (auto k : Range(cneqcs)) {
      c_mesh.nnodes_eqc[NT_VERTEX][k] = v_dsp[k+1];
      c_mesh.nnodes_cross[NT_VERTEX][k] = 0;
      v_dsp[k+1] += v_dsp[k];
    }
    size_t cnv = v_dsp.Last();
    mapped_NN[NT_VERTEX] = cnv;
    c_mesh.nnodes[NT_VERTEX] = cnv;
    c_mesh.verts.SetSize(cnv);
    for (auto k : Range(cnv)) c_mesh.verts[k] = k;
    c_mesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (cneqcs, &(v_dsp[0]), &(c_mesh.verts[0]));
    // cout << "v_dsp: " << endl; prow2(v_dsp); cout << endl;
    // cout << "c_mesh.eqc_verts: " << endl << c_mesh.eqc_verts << endl;
    auto & ceqc_verts(c_mesh.eqc_verts);
    
    Array<size_t> sz(cneqcs); sz = 0;
    for (auto meq : Range(mneqcs)) {
      auto ceq = map_mc[meq];
      auto bs = firsti_v[meq];
      firsti_v[meq] = ceqc_verts.IndexArray()[ceq] + sz[ceq];
      sz[ceq] += bs;
    }

    sz.SetSize(my_group.Size());
    for (auto k : Range(my_group.Size())) {
      sz[k] = 0; for (auto row : Range(mg_btms[k]->GetNEqcs())) sz[k] += mg_btms[k]->template GetENodes<NT_VERTEX>(row).Size();
    }
    node_maps[NT_VERTEX] = Table<amg_nts::id_type>(sz);
    auto & vmaps = node_maps[NT_VERTEX];
    vmaps.AsArray() = -1;
    for (auto k : Range(my_group.Size())) {
      for (auto eqc : Range(mg_btms[k]->GetNEqcs())) {
	auto eqc_vs = mg_btms[k]->template GetENodes<NT_VERTEX>(eqc);
	for (auto l : Range(eqc_vs.Size())) {
	  vmaps[k][eqc_vs[l]] = firsti_v[map_om[k][eqc]]+l;
	}
      }
    }

    // cout << "contr vmap: " << endl;
    // for (auto k : Range(my_group.Size())) {
    //   cout << "map for " << k << ", rank " << my_group[k] << ":  ";
    //   prow2(vmaps[k]); cout << endl;
    // }
    // cout << endl;

    /** 
	Abandon hope all ye who enter here - this might
	be the ugliest code I have ever seen.
    **/
    
    Array<size_t> annoy_have(mneqcs);
    Array<size_t> ci_have(mneqcs);
    Table<size_t> ci_pos(mneqcs+1, cneqcs); // (i,j) nr of C in meq that become I in ceq
    Array<size_t> ci_get(cneqcs);
    Array<size_t> annoy_havec(cneqcs);
    // eq0, v0, eq1, v1
    typedef INT<4,int> ANNOYE;
    Table<ANNOYE> tannoy_edges;
    auto eq_of_v = [&c_mesh](auto v) { return c_mesh.template GetEqcOfNode<NT_VERTEX>(v); };
    auto map_cv_to_ceqc = [&c_mesh](auto v) { return c_mesh.template MapNodeToEQC<NT_VERTEX>(v); };
    {
      TableCreator<ANNOYE> ct(cneqcs);
      while (!ct.Done()) {
	annoy_have = 0; ci_get = 0; ci_have = 0; annoy_havec = 0;
	if (cneqcs) ci_pos.AsArray() = 0;
	for (auto k : Range(my_group.Size())) {
	  auto eqmap = map_om[k];
	  auto neq = eqmap.Size();
	  for (auto eq : Range(neq)) {
	    auto meq = map_om[k][eq];
	    int ceq = map_oc[k][eq];
	    if (my_group[k]!=eqc_sender[map_om[k][eq]]) continue;
	    auto es = mg_btms[k]->template GetCNodes<NT_EDGE>(eq);
	    for (auto l : Range(es.Size())) {
	      const auto& v = es[l].v;
	      auto cv1 = vmaps[k][v[0]];
	      auto cv2 = vmaps[k][v[1]];
	      if (cv1>cv2) swap(cv1, cv2);
	      int ceq1 = eq_of_v(cv1);
	      int ceq2 = eq_of_v(cv2);
	      if ( (ceq1==ceq2) && (ceq1==ceq) ) { // CI edge
		ci_pos[meq][ceq1]++;
		ci_get[ceq1]++;
		ci_have[meq]++;
		continue;
	      }
	      auto cutid = c_eqc_h.GetCommonEQC(ceq1, ceq2);
	      if (ceq==cutid) continue; // CC edge
	      auto ceq1_id = c_eqc_h.GetEQCID(ceq1);
	      auto ceq2_id = c_eqc_h.GetEQCID(ceq2);
	      // master of coarse(C) adds the edge
	      if (c_eqc_h.IsMasterOfEQC(ceq)) {
		ANNOYE ce = {ceq1_id, map_cv_to_ceqc(cv1), ceq2_id, map_cv_to_ceqc(cv2)};
		ct.Add(cutid, ce);
	      }
	      annoy_have[meq]++;
	      annoy_havec[cutid]++;
	    }
	  }
	}
	ct++;
      }
      tannoy_edges = ct.MoveTable();
    }

    // cout << "tannoy_edges: " << endl << tannoy_edges << endl;
    auto annoy_edges = ReduceTable<ANNOYE, ANNOYE>
      (tannoy_edges, this->c_eqc_h, [](const auto & in) {
	Array<ANNOYE> out;
	if (in.Size() == 0) return out;
	int ts = 0; for (auto k : Range(in.Size())) ts += in[k].Size();
	if (ts == 0) return out;
	out.SetSize(ts); ts = 0;
	for (auto k : Range(in.Size()))
	  { auto row = in[k]; for (auto j : Range(row.Size())) out[ts++] = row[j]; }
	QuickSort(out, [](const auto & a, const auto & b) {
	    const bool isin[2] = {a[0]==a[2], b[0]==b[2]};
	    if (isin[0] && !isin[1]) return true;
	    else if (isin[1] && !isin[0]) return false;
	    for (int l : {0,2,1,3})
	      { if (a[l]<b[l]) return true; else if (b[l]<a[l]) return false; }
	    return false;
	  });
	return out;
      });
    
    // cout << "reduced annoy_edges: " << endl << annoy_edges << endl;


    Array<INT<2, size_t>> annoy_count(cneqcs);
    for (auto ceq : Range(cneqcs)) {
      annoy_count[ceq] = 0;
      for (auto & edge: annoy_edges[ceq]) {
	// if (edge.eqc[0] == edge.eqc[1]) annoy_count[ceq][0]++;
	if (edge[0] == edge[2]) annoy_count[ceq][0]++;
	else break;
      }
      annoy_count[ceq][1] = annoy_edges[ceq].Size() - annoy_count[ceq][0];
    }
    /** allocate edge-maps **/
    Array<size_t> s_emap(my_group.Size());  // size for emap
    for (auto k : Range(my_group.Size())) {
      // s_emap[k] = recv_es[k].Size();
      s_emap[k] = mg_btms[k]->template GetNN<NT_EDGE>();
    }
    node_maps[NT_EDGE] = Table<amg_nts::id_type>(s_emap);
    auto & emaps = node_maps[NT_EDGE];
    emaps.AsArray() = -1; // TODO: remove...
      
    /** count edge types in CEQs **/
    Array<size_t> ii_pos(mneqcs);
    Array<size_t> cc_pos(mneqcs);
    Array<INT<5,size_t>> ccounts(cneqcs); // [II,CI,IANNOY,CC,CANNOY]
    ccounts = INT<5,size_t>(0);
    BitArray has_set(mneqcs); has_set.Clear();
    for (auto k : Range(my_group.Size())) {
      auto eqmap = map_om[k];
      auto neq = eqmap.Size();
      for (auto eq : Range(neq)) {
	auto meq = map_om[k][eq];
	auto ceq = map_mc[meq];
	bool is_sender = (my_group[k]==eqc_sender[meq]);
	if (!is_sender) continue;
	has_set.Set(meq);
	// auto ces = recv_cetab[k][eq];
	auto eqes = mg_btms[k]->template GetENodes<NT_EDGE>(eq);
	auto ces = mg_btms[k]->template GetCNodes<NT_EDGE>(eq);
	// ii_pos[meq] = recv_etab[k][eq].Size();
	ii_pos[meq] = eqes.Size();
	// ccounts[ceq][0] += recv_etab[k][eq].Size();
      	ccounts[ceq][0] += eqes.Size();
      	ccounts[ceq][1] = ci_get[ceq];
      	ccounts[ceq][2] = annoy_count[ceq][0];
	cc_pos[meq] = ces.Size() - ci_have[meq] - annoy_have[meq];
	ccounts[ceq][3] += ces.Size() - ci_have[meq] - annoy_have[meq];
      	ccounts[ceq][4] = annoy_count[ceq][1];
      }
    }

    // cout << "ccounts: " << endl << ccounts << endl;

    /** displacements, edge and edge-map allocation**/
    // Array<size_t> disp_ie(cneqcs+1); disp_ie = 0;
    auto & disp_ie = c_mesh.disp_eqc[NT_EDGE];
    // Array<size_t> disp_ce(cneqcs+1); disp_ce = 0;
    auto & disp_ce = c_mesh.disp_cross[NT_EDGE];
    size_t cniie, cncie, cnannoyi, cncce, cnannoyc;
    cniie = cncie = cnannoyi = cncce = cnannoyc = 0;
    for (auto k : Range(cneqcs)) {
      cniie += ccounts[k][0];
      cncie += ccounts[k][1];
      cnannoyi += ccounts[k][2];
      disp_ie[k+1] = disp_ie[k] + ccounts[k][0] + ccounts[k][1] + ccounts[k][2];
      cncce += ccounts[k][3];
      cnannoyc += ccounts[k][4];
      disp_ce[k+1] = disp_ce[k] + ccounts[k][3] + ccounts[k][4];
    }
    size_t cnie = cniie + cncie + cnannoyi;
    size_t cnce = cncce + cnannoyc;
    size_t cne = cnie+cnce;

    
    // cout << "CNE CNIE CNCE: " << cne << " " << cnie << " " << cnce << endl;
    // cout << "II CI ANNOYI CC ANNOYC: " << cniie << " " << cncie << " "
    // 	 << cnannoyi << " " << cncce << " " << cnannoyc << endl;
    // cout << "disp_ie: " << endl << disp_ie << endl;
    // cout << "disp_ce: " << endl << disp_ce << endl;

    
    mapped_NN[NT_EDGE] = cne;
    c_mesh.nnodes[NT_EDGE] = cne;
    c_mesh.edges.SetSize(cne);
    auto cedges = c_mesh.template GetNodes<NT_EDGE>();
    for (auto & e:cedges) e = {{{-1,-1}}, -1}; // TODO:remove

    /** Literally no idea what I did here **/
    if (ccounts.Size()) {
      ccounts[0][1] += ccounts[0][0];
      ccounts[0][2] += ccounts[0][1];
      ccounts[0][4] += ccounts[0][3];
    }
    for (size_t ceq = 1; ceq < cneqcs; ceq++) {
      ccounts[ceq][0] += ccounts[ceq-1][2];
      ccounts[ceq][1] += ccounts[ceq][0];
      ccounts[ceq][2] += ccounts[ceq][1];
      ccounts[ceq][3] += ccounts[ceq-1][4];
      ccounts[ceq][4] += ccounts[ceq][3];
    }
    for (int ceq = int(cneqcs)-1; ceq > 0; ceq--) {
      ccounts[ceq][2] = ccounts[ceq][1];
      ccounts[ceq][1] = ccounts[ceq][0];
      ccounts[ceq][0] = ccounts[ceq-1][2];
      ccounts[ceq][4] = ccounts[ceq][3];
      ccounts[ceq][3] = ccounts[ceq-1][4];
    }
    if (ccounts.Size()) {
      ccounts[0][2] = ccounts[0][1];
      ccounts[0][1] = ccounts[0][0];
      ccounts[0][0] = 0;
      ccounts[0][4] = ccounts[0][3];
      ccounts[0][3] = 0;
    }
    // cout << endl << "ccounts - pos: " << endl << ccounts << endl;
    Array<INT<2, size_t>> annoy_pos(cneqcs); // have to search here with Pos
    for (auto ceq : Range(cneqcs)) {
      annoy_pos[ceq][0] = ccounts[ceq][2];
      annoy_pos[ceq][1] = cnie + ccounts[ceq][4];
    }
    for (auto meq : Range(mneqcs)) {
      auto ceq = map_mc[meq];
      auto cii = ii_pos[meq];
      ii_pos[meq] = ccounts[ceq][0];
      ccounts[ceq][0] += cii;
      auto ccc = cc_pos[meq];
      cc_pos[meq] = cnie + ccounts[ceq][3];
      ccounts[ceq][3] += ccc;
    }

    /** prefix ci_pos **/
    for (auto meq : Range(mneqcs)) {
      for (auto ceq : Range(cneqcs)) {
	ci_pos[meq+1][ceq] += ci_pos[meq][ceq];
      }
    }
    for (auto ceq : Range(cneqcs)) {
      for (int meq = mneqcs-2; meq>=0;meq--) {
	ci_pos[meq+1][ceq] = ccounts[ceq][1] + ci_pos[meq][ceq];
      }
      ci_pos[0][ceq] = ccounts[ceq][1];
    }

    // fill all and make maps for edges
    Array<size_t> cci(cneqcs);
    for (auto k : Range(my_group.Size())) {
      auto eqmap = map_om[k];
      auto vmap = vmaps[k];
      auto emap = emaps[k];
      auto neq = eqmap.Size();
      auto lam = [&emap, &vmap, &cedges](auto id, auto & edge) {
	AMG_Node<NT_VERTEX> v0 = vmap[edge.v[0]];
	AMG_Node<NT_VERTEX> v1 = vmap[edge.v[1]];
	if (v0>v1) swap(v0,v1);
	cedges[id] = {{{v0,v1}}, id};
	emap[edge.id] = id;
      };
      for (auto eq : Range(neq)) {
	auto meq = map_om[k][eq];
	size_t ceq = map_mc[meq];
	// II edges
	// auto ies = recv_etab[k][eq];
	auto ies = mg_btms[k]->template GetENodes<NT_EDGE>(eq);
	for (auto l : Range(ies.Size())) {
	  amg_nts::id_type id = ii_pos[meq] + l;
	  lam(id, ies[l]);
	}
	// CI, CC and ANNOYING EDGES
	size_t cutid = 0;
	amg_nts::id_type id = 0;
	size_t ccc = 0;
	// auto ces = recv_cetab[k][eq];
	auto ces = mg_btms[k]->template GetCNodes<NT_EDGE>(eq);
	cci = 0;
	for (auto l : Range(ces.Size())) {
	  auto edge = ces[l];
	  auto cv0 = vmap[edge.v[0]];
	  auto cv1 = vmap[edge.v[1]];
	  if (cv0 > cv1) swap(cv0, cv1);
	  auto ceq0 = eq_of_v(cv0);
	  auto ceq0_id = c_eqc_h.GetEQCID(ceq0);
	  auto ceq1 = eq_of_v(cv1);
	  auto ceq1_id = c_eqc_h.GetEQCID(ceq1);
	  if (ceq0 == ceq1) {
	    if (ceq0 == ceq) { // CI
	      id = ci_pos[meq][ceq0] + cci[ceq0];
	      // cout << "(CI-edge " << cci[ceq0] << " to ceq " << ceq0 << ") ";
	      cci[ceq0]++;
	    }
	    else { // IANNOY!!
	      // weighted_cross_edge wce({INT<2>(ceq0_id, ceq1_id),
	      // 	    bare_edge(map_cv_to_ceqc(cv0), map_cv_to_ceqc(cv1)), 0.0});
	      INT<4, int> wce = {ceq0_id, map_cv_to_ceqc(cv0), ceq1_id, map_cv_to_ceqc(cv1)};
	      auto pos = annoy_edges[ceq0].Pos(wce);
	      id = annoy_pos[ceq0][0] + pos;
	      // cout << "(Iannoy-edge, pos " << pos << ") , cross edge was "
	      // 	   << wce << endl;
	    }
	  }
	  else if ( ceq == (cutid = c_eqc_h.GetCommonEQC(ceq0, ceq1)) ) { // CC
	    id = cc_pos[meq] + ccc;
	    // cout << "(CC-edge " << ccc << " ) ";
	    ccc++;
	  }
	  else { // CANNOY!!
	    const auto & count = annoy_count[cutid];
	    auto aces = FlatArray<ANNOYE>(count[1], &(annoy_edges[cutid][count[0]]));
	    // clang-6 doesnt like this?? (see also amg_coarsen.cpp!)
	    // weighted_cross_edge wce({INT<2>(ceq0_id, ceq1_id),
	    // 	  bare_edge(map_cv_to_ceqc(cv0), map_cv_to_ceqc(cv1)), 0.0});
	    INT<4, int> wce = {ceq0_id, map_cv_to_ceqc(cv0), ceq1_id, map_cv_to_ceqc(cv1)};
	    auto pos = aces.Pos(wce);;
	    // cout << "(Cannoy-edge, pos " << pos << ") , cross edge was "
	    // 	 << wce << endl;
	    id = annoy_pos[cutid][1] + pos;
	  }
	  // cout << "member " << k << ", eq " << eq << ", meq " << meq
	  //      << ", ceq " << ceq << ", cedge " << l << " -> id " << id << endl;
	  lam(id, edge);
	}
      }
    }

    // okay, now finish writing annoy_edges and constrct annoy_nodes:
    sz.SetSize(cneqcs);
    for (auto k : Range(cneqcs))
      sz[k] = annoy_edges[k].Size();
    annoy_nodes[NT_EDGE] = Table<amg_nts::id_type>(sz);
    sz = 0;
    INT<2, size_t> count;
    for (auto ceq : Range(cneqcs)) {
      auto as = annoy_edges[ceq];
      auto pos = annoy_pos[ceq];
      for (auto l : Range(as.Size())) {
	auto eq0 = c_eqc_h.GetEQCOfID(as[l][0]);
	AMG_Node<NT_VERTEX> v0 = ceqc_verts[eq0][as[l][1]];
	auto eq1 = c_eqc_h.GetEQCOfID(as[l][2]);
	AMG_Node<NT_VERTEX> v1 = ceqc_verts[eq1][as[l][3]];
	bool is_in = (l < annoy_count[ceq][0]);
	amg_nts::id_type id = is_in ? pos[0]+l : pos[1] + (l - annoy_count[ceq][0]);
	annoy_nodes[NT_EDGE][ceq][sz[ceq]++] = id;
	// cout << "ANNOY ceq (in? " << is_in << ")" << ceq << " edge " << l << endl;
	// cout << " pos: " << pos << endl;
	// cout << " counts: " << annoy_count[ceq] << endl;
	// cout << "ae: " << as[l] << endl;
	// cout << "edge " << cedges[id];
	cedges[id] = {{{v0, v1}}, id};
	// cout << " -> " << cedges[id] << endl;
      }
    }

    // cout << "contr emap: " << endl;
    // for (auto k : Range(my_group.Size())) {
    //   cout << "map for " << k << ", rank " << my_group[k] << ":  ";
    //   prow2(emaps[k]); cout << endl;
    // }
    // cout << endl;
    
    for (auto k : Range(cneqcs)) {
      c_mesh.nnodes_eqc[NT_EDGE][k] = disp_ie[k+1] - disp_ie[k];
      c_mesh.nnodes_cross[NT_EDGE][k] = disp_ce[k+1] - disp_ce[k];
    }
    c_mesh.eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (cneqcs, &c_mesh.disp_eqc[NT_EDGE][0], &c_mesh.edges[0]);
    c_mesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (cneqcs, &c_mesh.disp_cross[NT_EDGE][0], &c_mesh.edges[c_mesh.disp_eqc[NT_EDGE].Last()]);
    // cout << "contr eqc_edges: " << endl << c_mesh.eqc_edges << endl;
    // cout << "contr cross_edges: " << endl << c_mesh.cross_edges << endl;
    mapped_NN[NT_FACE] = mapped_NN[NT_CELL] = 0;

    if constexpr(std::is_same<TMESH, BlockTM>::value == 1) {
        mapped_mesh = move(p_c_mesh);
      }
    else {
      // cout << "MAKE MAPPED ALGMESH!!" << endl;
      this->mapped_mesh = make_shared<TMESH> ( move(*p_c_mesh), mesh->MapData(*this) );
      // cout << "MAPPED ALGMESH: " << endl;
      // cout << *mapped_mesh << endl;
    }
  }

  INLINE Timer & timer_hack_beq () { static Timer t("GridContractMap :: BuildCEQCH"); return t; }
  template<class TMESH> void GridContractMap<TMESH> :: BuildCEQCH ()
  {
    RegionTimer rt(timer_hack_beq());

    const auto & eqc_h(*this->eqc_h);
    auto comm = eqc_h.GetCommunicator();

    this->proc_map.SetSize(comm.Size());
    auto n_groups = groups.Size();
    for (auto grp_nr : Range(n_groups)) {
      auto row = groups[grp_nr];
      for (auto j : Range(row.Size())) {
	proc_map[row[j]] = grp_nr;
      }
    }
    this->my_group.Assign(groups[proc_map[comm.Rank()]]);
    this->is_gm = my_group[0] == comm.Rank();

    if (!is_gm) {
      /** Send DP-tables to master and return **/
      int master = my_group[0];
      comm.Send(eqc_h.GetDPTable(), master, MPI_TAG_AMG);
      comm.Send(eqc_h.GetEqcIds(), master, MPI_TAG_AMG);
      return;
    }
    
    /** New MPI-Comm **/
    netgen::NgArray<int> cmembs(groups.Size()); // haha, this has to be a netgen-array
    for (auto k : Range(groups.Size())) cmembs[k] = groups[k][0];
    NgsAMG_Comm c_comm(netgen::MyMPI_SubCommunicator(comm, cmembs), true);

    /** gather eqc-tables **/
    auto & reft = eqc_h.GetDPTable();
    Array<int> sz;
    if (reft.Size()) sz.SetSize(reft.Size());
    for (auto k : Range(reft.Size()))
      sz[k] = reft[k].Size();
    Table<int> eqcs_table(sz);
    for (auto k : Range(reft.Size()))
      for (auto j : Range(sz[k]))
	eqcs_table[k][j] = reft[k][j];
    Array<Table<int>> all_dist_eqcs(my_group.Size());
    all_dist_eqcs[0] = std::move(eqcs_table);
    Array<Array<size_t>> all_eqc_ids(my_group.Size());
    all_eqc_ids[0].SetSize(eqc_h.GetNEQCS());
    for (auto j : Range(eqc_h.GetNEQCS())) all_eqc_ids[0][j] = eqc_h.GetEQCID(j);
    for (auto j : Range((size_t)1,my_group.Size())) {
      comm.Recv(all_dist_eqcs[j], my_group[j], MPI_TAG_AMG);
      all_eqc_ids[j].SetSize(all_dist_eqcs[j].Size());
      comm.Recv(all_eqc_ids[j], my_group[j], MPI_TAG_AMG);
    }

    /** merge gathered eqc tables **/
    Array<int> gids;
    gids = std::move(all_eqc_ids[0]);
    for (auto j : Range((size_t)1, my_group.Size())) {
      for (auto l : Range(all_eqc_ids[j].Size())) {
	if (!gids.Contains(all_eqc_ids[j][l]))
	  gids.Append(all_eqc_ids[j][l]);
      }
    }    
    QuickSort(gids);
    size_t mneqcs = gids.Size();
    sz.SetSize(my_group.Size());
    for (auto j : Range(my_group.Size())) {
      sz[j] = all_eqc_ids[j].Size();
    }
    map_om = Table<int> (sz);
    map_oc = Table<int> (sz);
    sz.SetSize(mneqcs);
    sz = 0;
    for (auto j : Range(my_group.Size())) {
      auto nid = all_eqc_ids[j].Size();
      for (auto l : Range(nid)) {
	map_om[j][l] = gids.Pos(all_eqc_ids[j][l]);
	sz[map_om[j][l]] = all_dist_eqcs[j][l].Size()+1;
      }
    }
    // cout << "map_om: "  << endl << map_om << endl;
    // Table<int> mdps(sz);
    mmems = Table<int>(sz);
    for (auto j : Range(my_group.Size())) {
      auto nid = all_eqc_ids[j].Size();
      for (auto l : Range(nid)) {
	for (auto i : Range(all_dist_eqcs[j][l].Size()))
	  mmems[map_om[j][l]][i] = all_dist_eqcs[j][l][i];
	mmems[map_om[j][l]].Last() = my_group[j];
      }
    }
    for (auto j : Range(mmems.Size()))
      QuickSort(mmems[j]);
    /** contract eqc table + make map **/
    auto crs_dps = [&](const auto & dps) {
      Array<int> out;
      for (auto k : Range(dps.Size())) {
	auto dcr = proc_map[dps[k]];
	if ( (dcr!=c_comm.Rank()) && (!out.Contains(dcr)) )
	  out.Append(dcr);
      }
      QuickSort(out);
      return out;
    };
    this-> map_mc.SetSize(mneqcs);
    Array<Array<int>> ceqcs;
    sz.SetSize(0);
    for (auto j : Range(mmems.Size())) {
      auto cdps = crs_dps(mmems[j]);
      bool is_new = true;
      int l; int ceqss = ceqcs.Size();
      for (l=0; l<ceqss&&is_new; l++)
	if (ceqcs[l]==cdps) {
	  is_new = false;
	}
      map_mc[j] = is_new?l:l-1; //incremented one extra time
      if (is_new) {
	sz.Append(cdps.Size());
	ceqcs.Append(std::move(cdps));
      }
    }    
    Table<int> ceqcs_table(sz), ceq2(sz);
    for (auto k : Range(ceqcs_table.Size())) {
      for (auto j : Range(ceqcs_table[k].Size())) {
	ceqcs_table[k][j] = ceqcs[k][j];
	ceq2[k][j] = ceqcs[k][j];
      }
    }
    this->c_eqc_h = make_shared<EQCHierarchy>(std::move(ceqcs_table), c_comm);
    
    // EQCHierarchy re-sorts the DP-table!!
    auto & ctab = c_eqc_h->GetDPTable(); 
    Array<int> remap(ctab.Size());
    for (auto k : Range(ctab.Size())) {
      auto set = ceq2[k];
      bool found = false;
      int l = -1;
      while (!found) {
	l++;
	if (ctab[l]==set) found=true;
      }
      remap[k] = l;
    }
    for (auto k : Range(map_mc.Size()))
      map_mc[k] = remap[map_mc[k]];
    for (auto k : Range(map_oc.Size())) {
      for (auto j : Range(map_oc[k].Size())) {
	map_oc[k][j] = map_mc[map_om[k][j]];
      }
    }

  }

} // namespace amg

#include "amg_tcs.hpp"
