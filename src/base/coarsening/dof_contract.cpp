
#include "dof_contract.hpp"

#include<utils_arrays_tables.hpp>

namespace amg
{

template<class TV>
CtrMap<TV> :: CtrMap (UniversalDofs originalDofs,
                      UniversalDofs mappedDofs,
                      Array<int> && _group,
                      Table<int> && _dof_maps)
  : BaseDOFMapStep(originalDofs, mappedDofs)
  , group(std::move(_group))
  , master(group[0])
  , dof_maps(std::move(_dof_maps))
{
  is_gm = ( GetUDofs().GetCommunicator().Rank() == master );
}


template<class TV>
CtrMap<TV> :: ~CtrMap ()
{
  for (auto k : Range(mpi_types))
    { MPI_Type_free(&mpi_types[k]); }
}


template<class TV>
shared_ptr<BaseDOFMapStep> CtrMap<TV> :: Concatenate (shared_ptr<BaseDOFMapStep> other)
{
  return nullptr;
  // if ( auto rmap = dynamic_pointer_cast<CtrMap<TV>>(other) ) {
  //   Array<shared_ptr<BaseDOFMapStep>> sub_steps ({ shared_from_this(), other });
  //   return make_shrade<ConcDMS> (steps);
  // }
  // else
  //   { return nullptr; }
} // CtrMao::Concatenate


INLINE Timer<TTracing, TTiming>& timer_hack_ctr_f2c () { static Timer t("CtrMap::F2C"); return t; }
INLINE Timer<TTracing, TTiming>& timer_hack_ctr_c2f () { static Timer t("CtrMap::C2F"); return t; }


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
  const auto & comm(GetParallelDofs()->GetCommunicator());
  if (!is_gm) {
    if (fvf.Size() > 0)
{ MPI_Send(x_fine->Memory(), fvf.Size(), GetMPIType<TV>(), group[0], MPI_TAG_AMG, comm); }
    return;
  }
  x_coarse->Distribute();
  auto fvc = x_coarse->FV<TV>();
  auto loc_map = dof_maps[0];
  int nreq_tot = 0;
  for (size_t kp = 1; kp < group.Size(); kp++)
    if (dof_maps[kp].Size()>0) { nreq_tot++; reqs[kp] = comm.IRecv(buffers[kp], group[kp], MPI_TAG_AMG); }
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

  // cout << "CTR trans C2F" << endl;

  RegionTimer rt(timer_hack_ctr_c2f());
  const auto & comm(GetParallelDofs()->GetCommunicator());
  x_fine->SetParallelStatus(CUMULATED);
  auto fvf = x_fine->FV<TV>();
  if (!is_gm)
    {
if (fvf.Size() > 0)
  { MPI_Recv(x_fine->Memory(), fvf.Size(), GetMPIType<TV>(), group[0], MPI_TAG_AMG, comm, MPI_STATUS_IGNORE); }
// cout << "short x: " << endl;
// prow(x_fine->FVDouble());
// cout << endl;
return;
    }

  // cout << "long x: " << endl;
  // prow(x_coarse->FVDouble());
  // cout << endl;

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
  // cout << "loc data: " << endl;
  // for (auto j : Range(loc_map.Size()))
  //   { cout << "[" << j << " " << loc_map[j] << " " << fvc(loc_map[j]) << "] "; }
  // cout << endl;
  reqs[0] = MPI_REQUEST_NULL; MyMPI_WaitAll(reqs);

  // cout << "short x: " << endl;
  // prow(x_fine->FVDouble());
  // cout << endl;
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
  const auto & comm(GetParallelDofs()->GetCommunicator());
  x_fine->Cumulate();
  auto fvf = x_fine->FV<TV>();

  if (!is_gm) {
    if (fvf.Size() > 0) {
      auto b0 = buffers[0];
      comm.Recv(b0, group[0], MPI_TAG_AMG);
      // cout << "got buf: "; prow(b0); cout << endl;
      for (auto k : Range(fvf.Size()))
        { fvf(k) += fac * b0[k]; }
    }
    // cout << "x fine " << x_fine->Size() << endl;
    // cout << *x_fine << endl;
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

  // cout << "loc data: " << endl;
  // for (auto j : Range(loc_map.Size()))
    // { cout << "[" << j << " " << loc_map[j] << " " << fvc(loc_map[j]) << "] "; }
  // cout << endl;

  reqs[0] = MPI_REQUEST_NULL;
  
  MyMPI_WaitAll(reqs);
  // cout << "x fine " << x_fine->Size() << endl;
  // cout << *x_fine << endl;
}


template<class TV>
void CtrMap<TV> :: PrintTo (std::ostream & os, string prefix) const
{
  os << prefix << " CtrMap, BS = " << VecHeight<TV>() << endl;
  os << prefix << " group = "; prow(group, os); os << endl;
  os << prefix << " dof_maps: " << endl;
  for (auto k : Range(dof_maps)) {
    os << prefix << k << ", s = " << dof_maps[k].Size() << " :: "; prow2(dof_maps[k], os); os << endl;
  }
  os << endl;
} // CtrMap::PrintTo


template<class TV>
bool CtrMap<TV> :: DoSwap (bool in)
{
  auto comm = GetUDofs().GetCommunicator();
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
shared_ptr<BaseDOFMapStep> CtrMap<TV> :: PullBack (shared_ptr<BaseDOFMapStep> other)
{
  if ( (is_gm) && (other == nullptr) )
  {
    throw Exception("CtrMap master needs map for PullBack");
    return nullptr;
  }

  auto prol_map = [&]() -> shared_ptr<ProlMap<TM>> {
    if ( is_gm )
    {
      return my_dynamic_pointer_cast<ProlMap<TM>>(other,
                                                  "Invalid BDMS-type for CtrMap PullBack");
    }
    else
    {
      return nullptr;
    }
  }();

  return SwapWithProl(prol_map);
} // CtrMap::PullBack


template<class TV>
int
CtrMap<TV>::
GetBlockSize () const
{
  return VecHeight<TV>();
} // CtrMap::GetBlockSize


template<class TV>
int
CtrMap<TV>::
GetMappedBlockSize () const
{
  return VecHeight<TV>();
} // CtrMap::GetMappedBlockSize


template<class TV>
shared_ptr<ProlMap<typename CtrMap<TV>::TM>>
CtrMap<TV>::
SwapWithProl (shared_ptr<ProlMap<TM>> pm_in)
{

  NgsAMG_Comm comm = GetUDofs().GetCommunicator();
  
  // cout << "SWAP WITH PROL, AM RANK " << comm.Rank() << " of " << comm.Size() << endl;
  // cout << "loc glob nd " << pardofs->GetNDofLocal() << " " << pardofs->GetNDofGlobal() << endl;
  
  // // TODO: new (mid) paralleldofs !?

  shared_ptr<SparseMatrix<TM>> split_A;
  // shared_ptr<ParallelDofs> mid_pardofs;


  if (!is_gm) {
    // cout << " get split A from " << master << endl;

    shared_ptr<SparseMatrixTM<TM>> split_A_TM;

    comm.Recv(split_A_TM, master, MPI_TAG_AMG);

    split_A = make_shared<SparseMatrix<TM>>(std::move(*split_A_TM));

    // cout << " got split A: " << endl;
    // print_tm_spmat(cout, *split_A); cout << endl;

    // do sth like that if whe actually need non-dummy ones
    // Table<int> pd_tab; comm.Recv(pd_tab, master, MPI_TAG_AMG);
    // mid_pardofs = make_shared<ParallelDofs> (std::move(pd_tab), comm, pardofs->GetEntrySize(), false);
  }
  else { // is_gm

    auto P = pm_in->GetProl();

    // cout << "prol dims " << P->Height() << " x " << P->Width() << endl;
    // cout << " mapped loc glob nd " << mapped_pardofs->GetNDofLocal() << " " << mapped_pardofs->GetNDofGlobal() << endl;
  
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

    // cout << " PROL IN: " << endl;
    // print_tm_spmat(cout, *P); cout << endl;

    /**
 Get rows contained in dmap out of prol, re-map colnrs from 0..N, where N is the number
  of DOFs in the pre-image of the rows.

  Write new dof_map into new_dmap.

  Also make dp-table for new ParallelDofs.
    **/
    auto get_rows = [&](FlatArray<int> dmap, Array<int> & new_dmap, int for_rank) LAMBDA_INLINE {
colspace.Clear();
int cnt_cols = 0;
Array<int> perow(dmap.Size());
for (auto k : Range(dmap)) {
  auto rownr = dmap[k];
  auto ri = rP.GetRowIndices(rownr);
  perow[k] = ri.Size();
  for (auto col : ri) {
    if (!colspace.Test(col))
      { colspace.SetBit(col); cnt_cols++; }
  }
}
rmap = -1;
new_dmap.SetSize(cnt_cols); cnt_cols = 0;
for (auto k : Range(P->Width())) {
  if (colspace.Test(k))
    { rmap[k] = cnt_cols; new_dmap[cnt_cols++] = k; }
}
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
return Pchunk;
    }; // get_rows


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

    // cout << " group "; prow(group); cout << endl;

    for (auto kp : Range(group))
    {
      // cout << " mem " << kp << " rk " << group[kp] << endl;
      auto Apart = get_rows(dof_maps[kp], pdm[kp], group[kp]);
      // cout << "Apart for mem " << kp << " rk " << group[kp] << ": " << endl;
      // cout << "pchunk dims for mem   " << kp << " rk " << group[kp] << ": " << Apart->Height() << " x " << Apart->Width() << endl;
      // cout << *Apart << endl;
      if (group[kp] == comm.Rank())
      {
        split_A = make_shared<SparseMatrix<TM>>(std::move(*Apart));
      }
      else
        { comm.Send(*Apart, group[kp], MPI_TAG_AMG); }
      // cout << "DONE sending Apart for mem " << kp << " rk " << group[kp] << ": " << endl;
    }

    Array<int> perow(group.Size());
    for (auto k : Range(perow))
{ perow[k] = pdm[k].Size(); }
    dof_maps = Table<int>(perow);
    for (auto k : Range(perow))
{ dof_maps[k] = pdm[k]; }

  } // is_gm

  // dummy (!!) pardofs
  Array<int> perow(split_A->Width()); perow = 0;
  Table<int> dps(perow);
  auto pardofs = GetUDofs().GetParallelDofs();
  // auto mid_pardofs = make_shared<ParallelDofs> (comm, std::move(dps), pardofs->GetEntrySize(), pardofs->IsComplex(), IsRankZeroIdle(pardofs));
  auto mid_pardofs = CreateParallelDOFs(comm, std::move(dps), pardofs->GetEntrySize(), pardofs->IsComplex(), IsRankZeroIdle(pardofs));

  auto prol_map = make_shared<ProlMap<TM>> (split_A, pardofs, mid_pardofs);

  // cout << "new prol_map: " << endl;

  // print_tm_spmat(cout, *split_A); cout << endl;

  // pardofs = mid_pardofs;
  // mapped_pardofs = (pm_in == nullptr) ? nullptr : pm_in->GetMappedParDofs();

  _originDofs = UniversalDofs(mid_pardofs);
  _mappedDofs = UniversalDofs((pm_in == nullptr) ? nullptr : pm_in->GetMappedParDofs());

  return prol_map;
} // CtrMap::SplitProl


template<class TV>
void CtrMap<TV> :: SetUpMPIStuff ()
{
  if (is_gm) {
    Array<int> perow(group.Size());
    Array<int> ones;
    size_t max_s = 0;
    
    for (auto k : Range(group.Size()))
      { max_s = max2(max_s, dof_maps[k].Size()); }

    mpi_types.SetSize(group.Size());
    ones.SetSize(max_s); ones = 1;
    reqs.SetSize(group.Size()); reqs = MPI_REQUEST_NULL;

    for (auto k : Range(group.Size())) {
      auto map = dof_maps[k]; auto ms = map.Size();
      MPI_Type_indexed(ms, (ms == 0) ? NULL : &ones[0], (ms == 0) ? NULL : &map[0], GetMPIType<TV>(), &mpi_types[k]);
      MPI_Type_commit(&mpi_types[k]);
    }
    
    perow[0] = 0;
    for (auto k : Range(size_t(1), group.Size()))
      { perow[k] = dof_maps[k].Size(); }

    buffers = Table<TV>(perow);

  }
  else {
    Array<int> perow(1);
    perow[0] = GetParallelDofs()->GetNDofLocal();
    buffers = Table<TV>(perow);
  }
}


INLINE Timer<TTracing, TTiming>& timer_hack_ctrmat (int nr) {
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
  NgsAMG_Comm comm(GetUDofs().GetCommunicator());

  // cout << " pardfos " << pardofs << endl << *pardofs << endl;

  if (!is_gm) {
    // cout << " I DROP, send mat to " << group[0] << endl;
    // cout << " MAT " << *mat << endl;
    comm.Send(*mat, group[0], MPI_TAG_AMG);
    // cout << " sent mat to " << group[0] << endl;
    return nullptr;
  }

  // cout << " mapped pardofs " << mapped_pardofs << endl << *mapped_pardofs << endl;

  // cout << "CTR MAT FOR " << pardofs->GetNDofGlobal() << " NDOF TO " << mapped_pardofs->GetNDofGlobal() << endl;
  // cout << "LOCAL DOFS " << pardofs->GetNDofLocal() << " NDOF TO " << mapped_pardofs->GetNDofLocal() << endl;
  // if (pardofs->GetNDofGlobal() < 10000)
  //   { cout << "CTR MAT FOR " << pardofs->GetNDofGlobal() << " NDOF " << endl; }

  timer_hack_ctrmat(1).Start();
  Array<shared_ptr<TSPM_TM> > dist_mats(group.Size());
  dist_mats[0] = mat;
  for(auto k : Range((size_t)1, group.Size())) {
    // cout << " get mat from " << k << " of " << group.Size() << endl;
    comm.Recv(dist_mats[k], group[k], MPI_TAG_AMG);
    // cout << " got mat from " << k << " of " << group.Size() << ", rank " << group[k] << endl;
    // cout << *dist_mats[k] << endl;
  }
  timer_hack_ctrmat(1).Stop();

  const auto & mapped_pardofs = GetMappedParDofs();
  size_t cndof = mapped_pardofs->GetNDofLocal();

  // reverse map: maps coarse dof to (k,j) such that disp_mats[k].Row(j) maps to this row!
  TableCreator<INT<2, size_t>> crm(cndof);
  for (; !crm.Done(); crm++)
    for(auto k : Range(dof_maps.Size())) {
auto map = dof_maps[k];
for(auto j : Range(map.Size()))
  crm.Add(map[j],INT<2, size_t>({k,j}));
    }
  auto reverse_map = crm.MoveTable();
  
  timer_hack_ctrmat(2).Start();
  Array<int*> merge_ptrs(group.Size()); Array<int> merge_sizes(group.Size());
  Array<int> perow(cndof);
  // we already buffer the col-nrs here, so we do not need to merge twice
  size_t max_nze = 0; for(auto k : Range(dof_maps.Size())) max_nze += dist_mats[k]->NZE();
  Array<int> colnr_buffer(max_nze); int* col_ptr = (max_nze == 0) ? nullptr : &(colnr_buffer[0]); max_nze = 0;
  Array<int> mc_buffer; // use this for merging with rows of LOCAL mat (which I should not change!!)
  Array<int> inds;
  auto QS_COL_VAL = [&inds](auto & cols, auto & vals) {
    auto S = cols.Size(); inds.SetSize(S); for (auto i : Range(S)) inds[i] = i;
    QuickSortI(cols, inds);
    for (int i : Range(S)) { // in-place permute
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

template class CtrMap<double>;
template class CtrMap<Vec<2, double>>;
template class CtrMap<Vec<3, double>>;

#ifdef ELASTICITY
template class CtrMap<Vec<6, double>>;
#endif

}