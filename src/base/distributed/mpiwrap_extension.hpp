#ifndef FILE_MPIWRAP_EXTENSION
#define FILE_MPIWRAP_EXTENSION

#include <base.hpp>

namespace amg
{
#ifdef NG_MPI_WRAPPER
#define NG_MPI_REQUEST_NULL 0
#else
#define NG_MPI_REQUEST_NULL NG_MPI_REQUEST_NULL
#endif


enum {
  NG_MPI_TAG_AMG = 1120,
  AMG_TAG_BASE = 1020
};

class BlockTM;

class NgsAMG_Comm : public NgMPI_Comm
{
protected:
  using NgMPI_Comm::valid_comm;
public:
  NgsAMG_Comm (const NgsAMG_Comm & c) : NgMPI_Comm(c) { ; }
  NgsAMG_Comm (const NgMPI_Comm & c) : NgMPI_Comm(c) { ; }
  NgsAMG_Comm (NgMPI_Comm && c) : NgMPI_Comm(c) { ; }
  NgsAMG_Comm (NG_MPI_Comm comm, bool owns)
    : NgMPI_Comm(comm, owns) { ; }
  NgsAMG_Comm () : NgsAMG_Comm(NgMPI_Comm()) { ; }
  ~NgsAMG_Comm () { ; }

  using NgMPI_Comm :: Send;
  using NgMPI_Comm :: Recv;
  using NgMPI_Comm :: ISend;
  using NgMPI_Comm :: IRecv;

  INLINE bool isValid() const { return valid_comm; }

private:
  INLINE Timer<TTracing, TTiming>& thack_send_tab () const { static Timer t("Send Table"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Send (FlatTable<T> tab, int dest, int tag) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_send_tab());
    size_t size = tab.Size();
    Send(size, dest, tag);
    if(!size) return;
    Send(tab.IndexArray(), dest, tag);
    Send(tab.AsArray(), dest, tag);
    return;
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_recv_tab () const { static Timer t("Recv Table"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Recv (Table<T> & tab, int src, int tag) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_recv_tab());
    size_t size = -1;
    Recv(size, src, tag);
    if(!size) {
tab = Table<T>();
return;
    }
    Array<size_t> index(size+1);
    Recv(index, src, tag);
    Array<int> sizes(size);
    for(auto k:Range(size))
sizes[k] = index[k+1]-index[k];
    tab = Table<T>(sizes);
    Recv(tab.AsArray(), src, tag);
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_bcast_fa () const { static Timer t("Bcast Array"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Bcast (FlatArray<T> ar, int root) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_bcast_fa());
    if (ar.Size() == 0) return;
    NG_MPI_Bcast (ar.Data(), ar.Size(), GetMPIType<T>(), root, comm);
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_bcast_tab () const { static Timer t("Bcast Table"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Bcast (Table<T> & tab, int root) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_bcast_tab());
    IVec<2, size_t> ss = {tab.Size(), (tab.Size() ? tab.AsArray().Size() : 0)};
    // MyMPI_Bcast(ss, *this, root);
    NgMPI_Comm::Bcast(ss, root);
    if (Rank() != root) {
Array<int> perow(ss[0]);
perow = 0; perow.Last() = ss[1];
tab = Table<T>(perow);
    }
    if (ss[1]>0) {
Bcast(tab.IndexArray(), root);
Bcast(tab.AsArray(), root);
    }
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_isend_spm () const { static Timer t("ISend SPM"); return t; }
  mutable Array<size_t> hacky_w_buffer;
public:
  void cleanup_hacky_w_buffer ()
  { hacky_w_buffer = Array<size_t>(0); }
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  NG_MPI_Request ISend (const ngla::SparseMatrixTM<T> & spm, int dest, int tag) const {
    if (!valid_comm)
      { return NG_MPI_REQUEST_NULL; }
    RegionTimer rt(thack_isend_spm());
    size_t H = spm.Height(); // size_t W = spm.Width();
    // if (H != W) throw Exception("cant rly deal with H!=W ISend"); // need to send W, need temporary mem
    hacky_w_buffer.Append(spm.Width());
    auto& W = hacky_w_buffer.Last();
    NG_MPI_Request req = ISend (W, dest, tag);
    NG_MPI_Request_free(&req);
    req = ISend (spm.GetFirstArray(), dest, tag);
    size_t nzes = spm.GetFirstArray().Last();
    if ( (H > 0) && (W > 0) ) {
NG_MPI_Request_free(&req);
int* cp = spm.GetRowIndices(0).Data();
FlatArray<int> cols(nzes, cp);
req = ISend (cols, dest, tag);
NG_MPI_Request_free(&req);
T* vp = spm.GetRowValues(0).Data();
FlatArray<T> vals(nzes, vp);
req = ISend (vals, dest, tag);
    }
    return req;
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_send_spm () const { static Timer t("Send SPM"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Send (const ngla::SparseMatrixTM<T> & spm, int dest, int tag) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_send_spm());
    NG_MPI_Request req = ISend(spm, dest, tag); // well this is just lazy ...
    NG_MPI_Wait(&req, NG_MPI_STATUS_IGNORE);
  }

private:
  INLINE Timer<TTracing, TTiming>& thack_recv_spm () const { static Timer t("Recv SPM"); return t; }
public:
  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  void Recv (shared_ptr<ngla::SparseMatrixTM<T> >& spm, int src, int tag) const {
    if (!valid_comm)
      { return; }
    RegionTimer rt(thack_recv_spm());
    size_t W = -1; Recv(W, src, tag);
    Array<size_t> firstia;
    Recv(firstia, src, tag);
    size_t H = firstia.Size() - 1;
    Array<int> nperow(H);
    for (auto k:Range(H))
{ nperow[k] = firstia[k+1]-firstia[k]; }
    spm = make_shared<ngla::SparseMatrixTM<T>>(nperow, W);
    if ( (H==0) || (W==0) )
{ return; }
    int* cp = spm->GetRowIndices(0).Data();
    size_t nzes = 0; for (auto k:Range(H)) { nzes += nperow[k]; }
    FlatArray<int> cols(nzes, cp);
    Recv (cols, src, tag);
    T* vp = spm->GetRowValues(0).Data();
    FlatArray<T> vals(nzes, vp);
    Recv (vals, src, tag);
    // cout << "GOT SPM: " << endl << *spm << endl;
  }

  void Send (shared_ptr<BlockTM> & mesh, int dest, int tag) const;
  void Recv (shared_ptr<BlockTM> & mesh, int src,  int tag) const;

  template<typename T, typename T2 = decltype(GetMPIType<T>())>
  INLINE void Gather(T val, FlatArray<T> mem, int root)
  {
    if (!valid_comm) {
      val = mem[0];
    }
    else{
      if ((Rank() == root) && (mem.Size() < Size()) )
        { throw Exception("Not enough memory for NG_MPI_Gather!"); }
      NG_MPI_Gather(&val, 1, GetMPIType<T>(), mem.Data(), 1, GetMPIType<T>(), root, comm);
    }
  }

  template<typename T>
  inline NG_MPI_Request ISend (const FlatTable<T> tab, int dest, int tag)
  {
    if (!valid_comm)
      { return NG_MPI_REQUEST_NULL; }
    NG_MPI_Request req;
    size_t size = tab.Size();
    req = ISend(size, dest, tag);
    if (!size) return req;
    NG_MPI_Request_free(&req);
    req = ISend(tab.IndexArray(), dest, tag);
    NG_MPI_Request_free(&req);
    req = ISend(tab.AsArray(), dest, tag);
    return req;
  }

  template <typename T, typename T2 = decltype(GetMPIType<T>())>
  void AllReduceFA (FlatArray<T> d, const NG_MPI_Op op = NG_MPI_SUM) const
  {
    if (!valid_comm)
      { return; }
    NG_MPI_Allreduce ( NG_MPI_IN_PLACE, d.Data(), int(d.Size()), GetMPIType<T>(), op, comm);
  }

  NgsAMG_Comm CreateSubCommunicatorGlobal (FlatArray<int> procs) const
  {
  #ifdef NG_MPI_WRAPPER
    return NgMPI_Comm::SubCommunicator(procs);
  #else
    /**
      *  Unlike NgMPI_Comm::SubCommunicator, this uses NG_MPI_Comm_create
      *  instead of NG_MPI_Comm_create_group - i.e., this must be called by ALL
      *  procs in the communicator !!.
      */
    NG_MPI_Comm subcomm;
    NG_MPI_Group gcomm, gsubcomm;
    NG_MPI_Comm_group(comm, &gcomm);
    NG_MPI_Group_incl(gcomm, procs.Size(), procs.Data(), &gsubcomm);
    NG_MPI_Comm_create(comm, gsubcomm, &subcomm);
    return (subcomm == NG_MPI_COMM_NULL) ? NgsAMG_Comm() : NgsAMG_Comm(subcomm, true);
  #endif
  }
}; // class NgsAMG_Comm

template<class T, class TLAM>
void MyAllReduceDofData (const ParallelDofs & pardofs, FlatArray<T> data, TLAM lam)
{
  auto comm = pardofs.GetCommunicator();
  auto ex_procs = pardofs.GetDistantProcs();
  if (!ex_procs.Size())
    { return; }
  Array<int> perow(ex_procs.Size());
  for (auto k : Range(perow))
    { perow[k] = pardofs.GetExchangeDofs(ex_procs[k]).Size(); }
  Table<T> send(perow), recv(perow);
  for (auto k : Range(perow)) {
    auto buf = send[k];
    auto ex_dofs = pardofs.GetExchangeDofs(ex_procs[k]);
    for (auto j : Range(buf))
{ buf[j] = data[ex_dofs[j]]; }
  }
  Array<NG_MPI_Request> reqs(2 * ex_procs.Size());
  for (auto k : Range(perow)) {
    reqs[2*k]     = comm.ISend(send[k], ex_procs[k], NG_MPI_TAG_AMG);
    reqs[2*k+1]   = comm.IRecv(recv[k], ex_procs[k], NG_MPI_TAG_AMG);
  }
  MyMPI_WaitAll(reqs);

  /** Note: we iterate by rank in comm, such that we have guaranteed same results everywhere **/
  Array<int> tra (ex_procs.Last() + 1 - ex_procs[0]); tra = -1;
  const auto ep0 = ex_procs[0];
  for (auto k : Range(ex_procs))
    { tra[ex_procs[k] - ep0] = k; }
  auto to_row = [&tra, ep0] (auto p) LAMBDA_INLINE { return tra[p - ep0]; };
  perow = 0;
  for (auto dof_num : Range(pardofs.GetNDofLocal())) {
    auto dps = pardofs.GetDistantProcs(dof_num);
    if (dps.Size()) {
T dof_val(0);
bool loc_done = false;
for (auto j : Range(dps)) {
  if ( (comm.Rank() < dps[j]) && (!loc_done) ) {
    loc_done = true;
    lam(dof_val, data[dof_num]);
  }
  auto kp = to_row(dps[j]);
  lam(dof_val, recv[kp][perow[kp]++]);
}
if (!loc_done) {
  lam(dof_val, data[dof_num]);
}
data[dof_num] = dof_val;
    }
  }

} // MyAllReduceDofData


template<class TIN, class TOUT>
INLINE void ExchangePairWise (NgsAMG_Comm comm, FlatArray<int> dps, const TIN& send_data, TOUT & recv_data)
{
  if (!comm.isValid())
    { return; }
  Array<NG_MPI_Request> reqs(2 * dps.Size());
  for (auto k : Range(dps)) {
    reqs[2*k]     = comm.ISend(send_data[k], dps[k], NG_MPI_TAG_AMG);
    reqs[2*k + 1] = comm.IRecv(recv_data[k], dps[k], NG_MPI_TAG_AMG);
  }
  MyMPI_WaitAll(reqs);
} // SendRecvData


// works with array<array<..>> when no memory has been allocated in arrays
template<class TIN, class TOUT>
INLINE void ExchangePairWise2 (NgsAMG_Comm comm, FlatArray<int> dps, const TIN& send_data, TOUT & recv_data)
{
  if (!comm.isValid())
    { return; }
  Array<NG_MPI_Request> reqs(dps.Size());
  for (auto k : Range(dps))
    { reqs[k]     = comm.ISend(send_data[k], dps[k], NG_MPI_TAG_AMG); }
  for (auto k : Range(dps))
    { comm.Recv(recv_data[k], dps[k], NG_MPI_TAG_AMG); }
  MyMPI_WaitAll(reqs);
} // SendRecvData


INLINE Table<int> BuildDPTable (const ParallelDofs & pds)
{
  TableCreator<int> ct(pds.GetNDofLocal());
  for (; !ct.Done(); ct++) {
    for (auto k : Range(pds.GetNDofLocal()))
      { ct.Add(k, pds.GetDistantProcs(k)); }
  }
  return ct.MoveTable();
}

// // never tested (did not end up needing it)
// template <class TGET_DPS>
// void EnumerateGloballyByDPs (NgMPI_Comm comm, size_t ndof, TGET_DPS get_dps,
// 			       Array<size_t> & global_nums, size_t & num_glob_dofs) const
// {
//   size_t num_master_dofs = 0;
//   global_nums.SetSize(ndof);
//   global_nums = -1;
//   auto is_master = [&](auto i) {
//     FlatArray<int> dps = get_dps(i);
//     return !( (dps.Size() > 0) && (comm.Rank() < dps[0]) );
//   };
//   for (auto i : Range(ndof))
//     if (is_master(i) )
// 	global_nums[i] = num_master_dofs++;
//   Array<size_t> first_master_dof(comm.Size());
//   comm.AllGather (num_master_dofs, first_master_dof);
//   num_glob_dofs = 0;
//   for (auto i : Range(first_master_dof)) {
// 	size_t cur = first_master_dof[i];
// 	first_master_dof[i] = num_glob_dofs;
// 	num_glob_dofs += cur;
//     }
//   int rank = comm.Rank();
//   for (auto i : Range(ndof))
//     if (global_nums[i] != -1)
// 	{ global_nums[i] += first_master_dof[rank]; }
//   ScatterDofDataByDPs (comm, get_dps, global_nums);
// } // EnumerateGloballyByDPs

// works when recv-buffer is not allocated/sized correctly
template<class TIN, class TOUT>
INLINE void ExchangePairWise_norecbuffer (NgsAMG_Comm comm, FlatArray<int> dps, const TIN& send_data, TOUT & recv_data)
{
  Array<NG_MPI_Request> reqs(dps.Size());
  for (auto k : Range(dps))
    { reqs[k] = comm.ISend(send_data[k], dps[k], NG_MPI_TAG_AMG); }
  for (auto k : Range(dps))
    { comm.Recv(recv_data[k], dps[k], NG_MPI_TAG_AMG); }
  MyMPI_WaitAll(reqs);
} // SendRecvData

} // namespace amg


namespace ngstd
{


// // this doesnt work!!
// template<typename T>
// inline NG_MPI_Request MyMPI_ISend(const ngla::SparseMatrixTM<T> & spm, int dest, int tag, NgMPI_Comm comm)
// {
//   NG_MPI_Request req;
//   /** send height, width & NZE **/
//   size_t h = spm.Height();
//   size_t w = spm.Width();
//   size_t nze = spm.NZE();
//   const auto sym_spm = dynamic_cast<const ngla::SparseMatrixSymmetric<T>*>(&spm);
//   size_t is_sym = (sym_spm!=NULL)?1:0;
//   IVec<4, size_t> data({h, w, nze, is_sym});
//   req = MyMPI_ISend(data, dest, tag, comm);
//   NG_MPI_Request_free(&req);
//   /** send row-indices **/
//   req = MyMPI_ISend(spm.firsti, dest, tag, comm);
//   NG_MPI_Request_free(&req);
//   /** send col-nrs & vals **/
//   int* cptr = NULL;
//   T* vptr = NULL;
//   for(size_t k=0;k<h && cptr==NULL; k++)
//     if(spm.GetRowIndices(k).Size()) {
// 	cptr = &spm.GetRowIndices(k).Data();
// 	vptr = &spm.GetRowValues(k).Data();
//     }
//   req = MyMPI_ISend(FlatArray<int>(nze, cptr), dest, tag, comm);
//   NG_MPI_Request_free(&req);
//   req = MyMPI_ISend(FlatArray<T>(nze, vptr), dest, tag, comm);
//   return req;
// }

} // namespace ngstd

#endif
