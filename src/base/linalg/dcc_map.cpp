#include <reducetable.hpp>

#include "dcc_map.hpp"

namespace amg
{
/** DCCMap **/

template<class TSCAL>
DCCMap<TSCAL> :: DCCMap (shared_ptr<ParallelDofs> _pardofs)
  : pardofs(_pardofs), block_size(_pardofs->GetEntrySize())
{
  ;
} // DCCMap (..)


template<class TSCAL>
void
DCCMap<TSCAL>::
AllocMPIStuff ()
{

  /** alloc buffers **/
  {
    Array<int> perow;
    auto alloc_bs = [&](auto & a, auto & b) {
perow.SetSize(b.Size());
for (auto k : Range(perow))
  { perow[k] = block_size * b[k].Size(); }
// a = (typename remove_reference<decltype(a)>::type)(perow);
a = std::move(Table<TSCAL>(perow));
    };
    alloc_bs(m_buffer, m_ex_dofs); // m_buffer = -1;
    alloc_bs(g_buffer, g_ex_dofs); // g_buffer = -1;
  }

  /** alloc requests **/
  m_reqs.SetSize(m_ex_dofs.Size()); m_reqs = NG_MPI_REQUEST_NULL;
  g_reqs.SetSize(g_ex_dofs.Size()); g_reqs = NG_MPI_REQUEST_NULL;

  /** initialize petsistent communication **/
  m_send.SetSize(m_ex_dofs.Size()); m_recv.SetSize(m_ex_dofs.Size());
  g_send.SetSize(g_ex_dofs.Size()); g_recv.SetSize(g_ex_dofs.Size());

  int cm = 0, cg = 0;
  auto comm = pardofs->GetCommunicator();
  auto ex_procs = pardofs->GetDistantProcs();
  for (auto kp : Range(ex_procs)) {
    if (g_ex_dofs[kp].Size()) { // init G send/recv
MPI_Send_init( &g_buffer[kp][0], block_size * g_ex_dofs[kp].Size(), GetMPIType<TSCAL>(), ex_procs[kp], NG_MPI_TAG_AMG + 4, comm, &g_send[cg]);
MPI_Recv_init( &g_buffer[kp][0], block_size * g_ex_dofs[kp].Size(), GetMPIType<TSCAL>(), ex_procs[kp], NG_MPI_TAG_AMG + 5, comm, &g_recv[cg]);
cg++;
    }
    if (m_ex_dofs[kp].Size()) { // init G send/recv
MPI_Send_init( &m_buffer[kp][0], block_size * m_ex_dofs[kp].Size(), GetMPIType<TSCAL>(), ex_procs[kp], NG_MPI_TAG_AMG + 5, comm, &m_send[cm]);
MPI_Recv_init( &m_buffer[kp][0], block_size * m_ex_dofs[kp].Size(), GetMPIType<TSCAL>(), ex_procs[kp], NG_MPI_TAG_AMG + 4, comm, &m_recv[cm]);
cm++;
    }
  }
  g_send.SetSize(cg); g_recv.SetSize(cg);
  m_send.SetSize(cm); m_recv.SetSize(cm);
}


template<class TSCAL>
DCCMap<TSCAL> :: ~DCCMap ()
{
  // MyMPI_WaitAll(g_reqs);
  // MyMPI_WaitAll(m_reqs);
}


template<class TSCAL>
void
DCCMap<TSCAL>::
StartDIS2CO (BaseVector & vec) const
{
#ifdef USE_TAU
  // tauNTC("");
  TAU_PROFILE("StartDIS2CO", TAU_CT(*this), TAU_DEFAULT);
  // tauNTC("");
#endif
  BufferG(vec);
  if (vec.GetParallelStatus() != DISTRIBUTED) // just nulling-out entries is enough
    { vec.SetParallelStatus(DISTRIBUTED); return; }
  auto ex_dofs = pardofs->GetDistantProcs();
  auto comm = pardofs->GetCommunicator();
  if (g_send.Size())
    { NG_MPI_Startall(g_send.Size(), g_send.Data()); }
  if (m_recv.Size())
    { NG_MPI_Startall(m_recv.Size(), m_recv.Data()); }
  // for (auto kp : Range(ex_dofs.Size())) {
  //   if (g_ex_dofs[kp].Size()) // send G vals
  // 	{ g_reqs[kp] = MyMPI_ISend(g_buffer[kp], ex_dofs[kp], NG_MPI_TAG_AMG + 3, comm); }
  //   else
  // 	{ g_reqs[kp] = NG_MPI_REQUEST_NULL; }
  //   if (m_ex_dofs[kp].Size()) // recv M vals
  // 	{ m_reqs[kp] = MyMPI_IRecv(m_buffer[kp], ex_dofs[kp], NG_MPI_TAG_AMG + 3, comm); }
  //   else
  // 	{ m_reqs[kp] = NG_MPI_REQUEST_NULL; }
  // }
} // DCCMap::StartDIS2CO


template<class TSCAL>
void
DCCMap<TSCAL>::
ApplyDIS2CO (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("ApplyDIS2CO", TAU_CT(*this), TAU_DEFAULT);
#endif
  if (vec.GetParallelStatus() != DISTRIBUTED) // just nulling-out entries is enough
    { vec.SetParallelStatus(DISTRIBUTED); return; }
  MyMPI_WaitAll(m_recv);
  ApplyM(vec);
} // DCCMap::ApplyDIS2CO


template<class TSCAL>
void
DCCMap<TSCAL>::
FinishDIS2CO () const
{
#ifdef USE_TAU
  TAU_PROFILE("FinishDIS2CO", TAU_CT(*this), TAU_DEFAULT);
#endif
  MyMPI_WaitAll(g_send);
} // DCCMap::FinishDIS2CO


template<class TSCAL>
void
DCCMap<TSCAL>::
StartCO2CU (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("StartCO2CU", TAU_CT(*this), TAU_DEFAULT);
#endif
  if (vec.GetParallelStatus() != DISTRIBUTED)
    { return; }
  BufferM(vec);
  if (m_send.Size())
    { NG_MPI_Startall(m_send.Size(), m_send.Data()); }
  if (g_recv.Size())
    { NG_MPI_Startall(g_recv.Size(), g_recv.Data()); }
} // DCCMap::StartCO2CU


template<class TSCAL>
void
DCCMap<TSCAL>::
ApplyCO2CU (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("ApplyCO2CU", TAU_CT(*this), TAU_DEFAULT);
#endif
  if (vec.GetParallelStatus() != DISTRIBUTED)
    { return; }
  MyMPI_WaitAll(g_recv);
  ApplyG(vec);
  vec.SetParallelStatus(CUMULATED);
} // DCCMap::ApplyCO2CU


template<class TSCAL>
void
DCCMap<TSCAL>::
FinishCO2CU () const
{
#ifdef USE_TAU
  TAU_PROFILE("FinishCO2CU", TAU_CT(*this), TAU_DEFAULT);
#endif
  MyMPI_WaitAll(m_send);
} // DCCMap::FinishCO2CU


template<class TSCAL>
void
DCCMap<TSCAL>::
WaitD2C () const
{
  // cout << "wait d2c " << endl;
  MyMPI_WaitAll(g_send);
  MyMPI_WaitAll(m_recv);
  // cout << " m buf now " << endl << m_buffer << endl;
}


template<class TSCAL>
void
DCCMap<TSCAL>::
WaitM () const
{
  MyMPI_WaitAll(m_reqs);
}


template<class TSCAL>
void
DCCMap<TSCAL>::
WaitG () const
{
  MyMPI_WaitAll(g_reqs);
}


template<class TSCAL, class TDOFS, class TBUFS, class TLAM>
INLINE void
iterate_buf_vec (int block_size, TDOFS & dofs, TBUFS & bufs, BaseVector & vec, TLAM lam)
{
  auto iterate_rows = [&](auto fun) LAMBDA_INLINE {
    for (auto kp : Range(bufs.Size()))
    {
      fun(dofs[kp], bufs[kp]);
    }
  };

  auto fv = vec.FV<TSCAL>();

  if (block_size == 1) {
    iterate_rows([&](auto dnrs, auto buf) LAMBDA_INLINE
    {
      for (auto k : Range(dnrs))
      {
        lam(buf[k], fv(dnrs[k]));
      }
    });
  }
  else
  {
    iterate_rows([&](auto dnrs, auto buf) LAMBDA_INLINE
    {
      int c = 0;
      for (auto k : Range(dnrs))
      {
        int base_etr = block_size * dnrs[k];
        for (auto l : Range(block_size))
          { lam(buf[c++], fv(base_etr++)); }
      }
    });
  }
} // iterate_buf_vec


template<class TSCAL>
void
DCCMap<TSCAL>::
BufferG (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("BufferG", TAU_CT(*this), TAU_DEFAULT);
#endif
  iterate_buf_vec<TSCAL>(block_size, g_ex_dofs, g_buffer, vec, [&](auto & buf_etr, auto & vec_etr) LAMBDA_INLINE {
buf_etr = vec_etr; vec_etr = 0;
    });
} // DCCMap::BufferG


template<class TSCAL>
void
DCCMap<TSCAL>::
ApplyM (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("ApplyM", TAU_CT(*this), TAU_DEFAULT);
#endif
  iterate_buf_vec<TSCAL>(block_size, m_ex_dofs, m_buffer, vec, [&](auto buf_etr, auto & vec_etr) LAMBDA_INLINE {
vec_etr += buf_etr;
    });
} // DCCMap::ApplyM


template<class TSCAL>
void
DCCMap<TSCAL>::
BufferM (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("BufferM", TAU_CT(*this), TAU_DEFAULT);
#endif
  iterate_buf_vec<TSCAL>(block_size, m_ex_dofs, m_buffer, vec, [&](auto & buf_etr, auto vec_etr) LAMBDA_INLINE {
buf_etr = vec_etr;
    });
} // DCCMap :: BufferM


template<class TSCAL>
void
DCCMap<TSCAL>::
ApplyG (BaseVector & vec) const
{
#ifdef USE_TAU
  TAU_PROFILE("ApplyG", TAU_CT(*this), TAU_DEFAULT);
#endif
  iterate_buf_vec<TSCAL>(block_size, g_ex_dofs, g_buffer, vec, [&](auto buf_etr, auto & vec_etr) LAMBDA_INLINE {
vec_etr = buf_etr;
    });
} // DCCMap::ApplyG

/** END DCCMap **/


/** ChunkedDCCMap **/

template<class TSCAL>
ChunkedDCCMap<TSCAL>::
ChunkedDCCMap (shared_ptr<EQCHierarchy> eqc_h,
               shared_ptr<ParallelDofs> _pardofs,
               int _MIN_CHUNK_SIZE)
  : DCCMap<TSCAL>(_pardofs), MIN_CHUNK_SIZE(_MIN_CHUNK_SIZE)
{
  /** m_ex_dofs and g_ex_dofs **/
  CalcDOFMasters(eqc_h);

  /** requests and buffers **/
  this->AllocMPIStuff();

} // ChunkedDCCMap (..)


template<class TSCAL>
void
ChunkedDCCMap<TSCAL>::
CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h)
{
  static Timer t("ChunkedDCCMap::CalcDOFMasters"); RegionTimer rt(t);

  auto neqcs = eqc_h->GetNEQCS();
  auto comm = eqc_h->GetCommunicator();

  /** This is a subtle but IMPORTANT difference. eqc_h can have more distprocs than pardofs.
    Happens e.g. when two procs share only one single vertex on a dirichlet boundary. This vertex drops
    from the first coarse level, it is not a distproc of the coarse paralleldofs anymore but it's eqc is still in eqchierarchy! **/
  // auto ex_procs = eqc_h->GetDistantProcs();
  auto ex_procs = pardofs->GetDistantProcs();

  auto nexp = ex_procs.Size();

  // cout << " ChunkedDCCMap on EQCH: " << eqc_h << endl << *eqc_h << endl;
  // cout << " ChunkedDCCMap on PDS: " << pardofs << endl << *pardofs << endl;
  // cout << " ChunkedDCCMap EX_PROCS = "; prow2(ex_procs);	cout << endl;

  // clear non-local dofs from master_of, and split into eqc-blocks
  m_dofs = make_shared<BitArray>(pardofs->GetNDofLocal()); m_dofs->Set();
  Table<int> eq_ex_dofs;
  {
    TableCreator<int> create_eq_dofs(eqc_h->GetNEQCS());
    for (; !create_eq_dofs.Done(); create_eq_dofs++)
for (auto p : pardofs->GetDistantProcs()) {
  auto ex_dofs = pardofs->GetExchangeDofs(p);
  Array<int> feqdp({p});
  auto face_eq =  eqc_h->FindEQCWithDPs(feqdp);
  for (auto d : ex_dofs) {
    m_dofs->Clear(d);
    auto dps = pardofs->GetDistantProcs(d);
    if (dps.Size() == 1) { create_eq_dofs.Add(face_eq, d); }
    else if (dps[0] == p) { create_eq_dofs.Add(eqc_h->FindEQCWithDPs(dps), d); }
  }
}
    eq_ex_dofs = create_eq_dofs.MoveTable();
    for (auto row : eq_ex_dofs)
{ QuickSort(row); }
  }

  // cout << "eq_ex_dofs" << endl << eq_ex_dofs << endl;

  Table<int> members; // ALL members of each eqc
  Table<int> chunk_so; // for each eqc: chunk0-size, chunk1-size, ... ()
  {
    Array<int> perow(neqcs); perow = 0;
    if (neqcs > 1)
for (auto k : Range(size_t(1), neqcs))
  { perow[k] = 1 + eqc_h->GetDistantProcs(k).Size(); }
    members = Table<int> (perow);
    if (neqcs > 1)
for (auto k : Range(size_t(1), neqcs))
  if (!eqc_h->IsMasterOfEQC(k))
    { perow[k] = 0; }
    chunk_so = Table<int> (perow);
    if (neqcs > 1)
for (auto k : Range(size_t(1), neqcs)) {
  auto dps = eqc_h->GetDistantProcs(k);
  FlatArray<int> row = members[k];
  row.Part(size_t(0), dps.Size()) = dps;
  row.Last() = comm.Rank();
  QuickSort(row);
  chunk_so[k] = 0;
  if (eqc_h->IsMasterOfEQC(k)) { // decide chunk-distribution
    int exds = eq_ex_dofs[k].Size(); int nmems = members[k].Size();
    int chunk_size = max2(MIN_CHUNK_SIZE, exds / nmems);
    int n_chunks = exds / chunk_size + (exds % chunk_size) ? 1 : 0;
    chunk_size = (exds == 0) ? 1 : exds / n_chunks;
    int rest = exds % chunk_size;
    // start handing out chunks with an arbitrary member (but no RNG, so it is reproducible)
    int start = (1029 * eqc_h->GetEQCID(k) / 7) % nmems;
    // for (auto j : Range(n_chunks))
    //   { chunk_so[k][(start + j) % nmems] = chunk_size + ( (j < rest) ? 1 : 0 ); }

    if (chunk_so[k].Size()) {
      // chunk_so[k].Last() = exds;
      chunk_so[k][0] = exds;
    }

  }
}
    chunk_so = ReduceTable<int, int>(chunk_so, eqc_h, [&](auto in) {
  Array<int> out;
  if (in.Size())
    { out.SetSize(in[0].Size()); out = in[0]; }
  return out;
});
  }

  // cout << "members" << endl << members << endl << endl;
  // cout << "chunk_so" << endl << chunk_so << endl << endl;

  // I think we are ready to construct m_ex_dofs and g_ex_dofs !!
  {
    TableCreator<int> c_m_exd(nexp), c_g_exd(nexp);
    for (; !c_m_exd.Done(); c_m_exd++, c_g_exd++)
if (neqcs > 1)
  for (auto k : Range(size_t(1), neqcs)) {
    auto csk = chunk_so[k];
    auto memsk = members[k];
    auto dps = eqc_h->GetDistantProcs(k);
    auto dofsk = eq_ex_dofs[k];
    int offset = 0;
    // cout << "eqc " << k << ", dps "; prow2(dps); cout << endl;
    // cout << ", member "; prow2(memsk); cout << endl;
    // cout << " dofsk "; prow2(dofsk); cout << endl;
    for (auto j : Range(csk)) {
      // cout << "chunk for mem mem " << j << " has sz " << csk[j] << endl;
      if (csk[j] > 0) {
  auto exp = memsk[j];
  // cout << " mem is proc " << exp << endl;
  if (exp == comm.Rank()) { // MY chunk!
    // cout << " thats me!" << endl;
    for (auto p : dps) {
      auto kp = find_in_sorted_array(p, ex_procs); /** If the chunk size is > 0, p HAS to be in ex_procs of pardofs! **/
      // cout << " p " << p << " is kp " << kp << " in "; prow2(ex_procs); cout << endl;
      for (auto l : Range(csk[j])) {
        c_m_exd.Add(kp, dofsk[offset + l]);
        // cout << " my dof, set " << dofsk[offset + l] << endl;
        m_dofs->SetBit(dofsk[offset + l]);
      }
    }
  }
  else { // chunk for someone else
    auto kp = find_in_sorted_array(exp, ex_procs);
    // cout << exp << " is " << kp << " in "; prow2(ex_procs); cout << endl;
    for (auto l : Range(csk[j]))
      { c_g_exd.Add(kp, dofsk[offset + l]); }
  }
  offset += csk[j];
      }
    }
  }
    m_ex_dofs = c_m_exd.MoveTable();
    for (auto row : m_ex_dofs) // only already sorted on coarse levels !
{ QuickSort(row); }
    g_ex_dofs = c_g_exd.MoveTable();
    for (auto row : g_ex_dofs)
{ QuickSort(row); }
  }

  // cout << "m_ex_dofs" << endl << m_ex_dofs << endl << endl;
  // cout << "g_ex_dofs" << endl << g_ex_dofs << endl << endl;

} // ChunkedDCCMap::CalcDOFMasters

/** END ChunkedDCCMap **/


/** BasicDCCMap **/

template<class TSCAL>
BasicDCCMap<TSCAL>::
BasicDCCMap (shared_ptr<ParallelDofs> _pardofs)
  : DCCMap<TSCAL>(_pardofs)
{
  /** m_ex_dofs and g_ex_dofs **/
  CalcDOFMasters();

  /** requests and buffers **/
  this->AllocMPIStuff();

} // BasicDCCMap (..)


template<class TSCAL>
void
BasicDCCMap<TSCAL>::
CalcDOFMasters ()
{
  static Timer t("BasicDCCMap::CalcDOFMasters"); RegionTimer rt(t);

  auto comm = pardofs->GetCommunicator();

  /** This is a subtle but IMPORTANT difference. eqc_h can have more distprocs than pardofs.
    Happens e.g. when two procs share only one single vertex on a dirichlet boundary. This vertex drops
    from the first coarse level, it is not a distproc of the coarse paralleldofs anymore but it's eqc is still in eqchierarchy! **/
  auto ex_procs = pardofs->GetDistantProcs();

  auto nexp = ex_procs.Size();

  // clear non-local dofs from master_of, and split into eqc-blocks
  m_dofs = make_shared<BitArray>(pardofs->GetNDofLocal());
  // I think we are ready to construct m_ex_dofs and g_ex_dofs !!
  {
    TableCreator<int> c_m_exd(nexp), c_g_exd(nexp);
    for (; !c_m_exd.Done(); c_m_exd++, c_g_exd++) {
for (auto k : Range(pardofs->GetNDofLocal())) {
  auto dps = pardofs->GetDistantProcs(k);
  if (dps.Size() > 0 && (dps[0] < comm.Rank()) ) {
    auto kp = find_in_sorted_array(dps[0], ex_procs);
    c_g_exd.Add(kp, k);
    m_dofs->Clear(k);
  }
  else { // local or master
    m_dofs->SetBit(k);
    for (auto p : dps) {
      auto kp = find_in_sorted_array(p, ex_procs);
      c_m_exd.Add(kp, k);
    }
  }
}
    }
    m_ex_dofs = c_m_exd.MoveTable();
    for (auto row : m_ex_dofs) // only already sorted on coarse levels !
{ QuickSort(row); }
    g_ex_dofs = c_g_exd.MoveTable();
    for (auto row : g_ex_dofs)
{ QuickSort(row); }
  }

  // cout << "m_ex_dofs" << endl << m_ex_dofs << endl << endl;
  // cout << "g_ex_dofs" << endl << g_ex_dofs << endl << endl;

} // BasicDCCMap::CalcDOFMasters

/** END BasicDCCMap **/


template class DCCMap<double>;
template class BasicDCCMap<double>;
template class ChunkedDCCMap<double>;

}; // namespace amg