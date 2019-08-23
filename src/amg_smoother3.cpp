#define FILE_AMGSM3_CPP

#ifdef USE_TAU
#include <Profile/Profiler.h>
// #include "TAU.h"
#endif

#include "amg.hpp"

#include "amg_smoother_impl.hpp"

namespace amg
{

  static mutex glob_mut;
  static bool thread_ready = true, thread_done = false, end_thread = false;
  static function<void(void)> thread_exec_fun = [](){};
  static std::condition_variable cv;

  
  void thread_fpf () {
#ifdef USE_TAU
    TAU_REGISTER_THREAD();
    TAU_PROFILE_SET_NODE(NgMPI_Comm(MPI_COMM_WORLD).Rank());
#endif
    while( !end_thread ) {
      // cout << "--- ulock" << endl;
      std::unique_lock<std::mutex> lk(glob_mut);
      // cout << "--- wait" << endl;
      cv.wait(lk, [&](){ return thread_ready; });
      // cout << "--- woke up,  " << thread_ready << " " << thread_done << " " << end_thread << endl;
      if (!end_thread && !thread_done) {
	thread_exec_fun();
	thread_done = true; thread_ready = false;
	cv.notify_one();
      }
    }
    // cout << "--- am done!" << endl;
  }

  static std::thread glob_mpi_thread = std::thread(thread_fpf);

  class ThreadGuard {
  public:
    std::thread * t;
    // ThreadGuard(std::thread* at) : t(at) { ; }
    ThreadGuard(std::thread* at) : t(at) {
      // cout << " MAKE class ThreadGuard !!!" << endl;
#ifdef USE_TAU
      // has nothing to do with the MPI thread itself
      // TAU_ENABLE_INSTRUMENTATION();
      // cout << " call reg thread main" << endl;
      // TAU_REGISTER_THREAD();
      // cout << " call set node main" << endl;
      TAU_PROFILE_SET_NODE(NgMPI_Comm(MPI_COMM_WORLD).Rank());
#endif
    }
    ~ThreadGuard () { thread_ready = true; end_thread = true; cv.notify_one(); t->join(); }
  };

  static ThreadGuard tg(&glob_mpi_thread);


  // like parallelvector.hpp, but force inline
  INLINE ParallelBaseVector * dynamic_cast_ParallelBaseVector2 (BaseVector * x)
  {
    if (AutoVector * ax = dynamic_cast<AutoVector*> (x))
      { return dynamic_cast<ParallelBaseVector*> (&**ax); }
    return dynamic_cast<ParallelBaseVector*> (x);
  }

  // FML i hate this
  INLINE BaseVector* get_loc_ptr (const BaseVector& x)
  {
    if (auto parvec = dynamic_cast_ParallelBaseVector2(const_cast<BaseVector*>(&x)) )
      { return parvec->GetLocalVector().get(); }
    else
      { return const_cast<BaseVector*>(&x); }
  }

  // FML i hate this
  INLINE BaseVector& get_loc_ref (const BaseVector& x)
  {
    if (auto parvec = dynamic_cast_ParallelBaseVector2(const_cast<BaseVector*>(&x)) )
      { return *parvec->GetLocalVector(); }
    else
      { return const_cast<BaseVector&>(x); }
  }


  /** Local Gauss-Seidel **/

  template<class TM>
  GSS3<TM> :: GSS3 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset, FlatArray<TM> add_diag)
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

    cout << "GSS3 invs: " << endl; prow2(dinv); cout << endl;
			     
    auto numset = freedofs ? freedofs->NumSet() : A.Height();
    first_free = 0; next_free = A.Height();
    if (freedofs != nullptr) {
      if (freedofs->NumSet() == 0)
	{ next_free = 0; }
      else if (freedofs->NumSet() != freedofs->Size()) {
	int c = 0;
	while(c < freedofs->Size()) {
	  if (freedofs->Test(c))
	    { first_free = c; break; }
	  c++;
	}
	if (freedofs->Size()) {
	  c = freedofs->Size() - 1;
	  while( c != size_t(-1) ) {
	    if (freedofs->Test(c))
	      { next_free = c + 1; break; }
	    c--;
	  }
	}
      }
    }


    if ( numset < 0.2 * A.Height()) {
      row_nrs.SetSize(numset); numset = 0;
      for (auto k : Range(A.Height()))
	if (freedofs->Test(k))
	  { row_nrs[numset++] = k; }
    }
    // cout << " set " << ( (freedofs != nullptr) ? to_string(freedofs->NumSet()) : string("ALL!")) << endl;
    // cout << " first next " << first_free << " " << next_free << " height " << A.Height() << endl;

  } // GSS3 (..)


  template<class TM>
  void GSS3<TM> :: SmoothRHSInternal (size_t first, size_t next, BaseVector &x, const BaseVector &b, bool backwards) const
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
    bool out = false;
    auto up_row = [&](auto rownr) LAMBDA_INLINE {
      auto r = fvb(rownr) - A.RowTimesVector(rownr, fvx);
      // if (out)
      // 	{ cout << "gss3 up row " << rownr << " dof " << rownr << " res " << fvb(rownr) << " " << A.RowTimesVector(rownr, fvx) << " " << r << " update " << dinv[rownr] * r << endl; }
      // 	if (rownr == 20295) {
      // 	  auto ri = A.GetRowIndices(rownr);
      // 	  auto rv = A.GetRowValues(rownr);
      // 	  cout << "row: " << endl;
      // 	  for (auto j : Range(ri)) {
      // 	    cout << "[" << j << " " << ri[j] << " " << rv[j] << " " << fvx(ri[j]) <<  "] ";
      // 	  }
      // 	      cout << endl;
      // 	}
      fvx(rownr) += dinv[rownr] * r;
    };


    if (!backwards) {
      const size_t use_first = max2(first, first_free);
      const size_t use_next = min2(next, next_free);
      // const size_t use_first = 0;
      // const size_t use_next = A.Height();
      for (size_t rownr = use_first; rownr < use_next; rownr++)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  up_row(rownr);
	}
    }
    else {
      const int use_first = max2(first, first_free);
      const int use_next = min2(next, next_free);
      // const int use_first = 0;
      // const int use_next = A.Height();
      out = (first == 0) && (next == A.Height());
      // cout << " gss 3 rhs back " << first << " " << next << " " << A.Height() << endl;
      for (int rownr = use_next - 1; rownr >= use_first; rownr--)
	if (!fds || fds->Test(rownr)) {
	  A.PrefetchRow(rownr);
	  up_row(rownr);
	}
    }
  } // SmoothRHSInternal


  template<class TM>
  void GSS3<TM> :: SmoothRESInternal (size_t first, size_t next, BaseVector &x, BaseVector &res, bool backwards) const
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
    
    double ti = MPI_Wtime();
    double tl = 0;

    double nrows = 0;

    if (row_nrs.Size()) {
      tl = MPI_Wtime();
      if (!backwards) {
	if (row_nrs.Size() > 1) {
	  for (auto k : Range(row_nrs.Size()-1)) {
	    A.PrefetchRow(row_nrs[k+1]);
	    up_row(row_nrs[k]);
	  }
	}
	if (row_nrs.Size())
	  { up_row(row_nrs.Last()); }
      }
      else {
	for (int k = row_nrs.Size() - 1; k > 0; k--) {
	  A.PrefetchRow(row_nrs[k-1]);
	  up_row(row_nrs[k]);
	}
	if (row_nrs.Size())
	  { up_row(row_nrs[0]); }
      }
      tl = MPI_Wtime()-tl;
    }
    else {
      if (!backwards) {
	const size_t use_first = max2(first, first_free);
	const size_t use_next = min2(next, next_free);
	nrows = use_next - use_first;
	// const size_t use_first = 0;
	// const size_t use_next = A.Height();
	tl = MPI_Wtime();
	for (size_t rownr = use_first; rownr < use_next; rownr++)
	  if (!fds || fds->Test(rownr)) {
	    A.PrefetchRow(rownr);
	    up_row(rownr);
	  }
	tl = MPI_Wtime() - tl;
      }
      else {
	const int use_first = max2(first, first_free);
	const int use_next = min2(next, next_free);
	nrows = use_next - use_first;
	// const int use_first = 0;
	// const int use_next = A.Height();
	tl = MPI_Wtime();
	for (int rownr = use_next - 1; rownr >= use_first; rownr--)
	  if (!fds || fds->Test(rownr)) {
	    A.PrefetchRow(rownr);
	    up_row(rownr);
	  }
	tl = MPI_Wtime() - tl;
      }
    }

    // double nrows = fds ? fds->NumSet() : A.Height();
    // nrows = fds ? fds->NumSet() : A.Height();
    ti = MPI_Wtime() - ti;

    // if (nrows) {
    //   cout << endl << "-------" << endl;
    //   cout << "rows " << nrows << ", secs " << ti << " " << tl << endl;
    //   cout << "tot K rows/sec " << nrows / 1000 / ti << endl;
    //   cout << "loop K rows/sec " << nrows / 1000 / tl << endl;
    //   cout << "-------" << endl;
    // }

  } // SmoothRESInternal


  /** GSS4 **/
  template<class TM>
  GSS4<TM> :: GSS4 (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset, FlatArray<TM> add_diag)
  {
    if (subset && (subset->NumSet() != A->Height()) ) {
      /** xdofs / resdofs **/
      // Array<int> resdofs;
      size_t cntx = 0, cntres = 0;
      BitArray res_subset(subset->Size()); res_subset.Clear();
      for (auto k : Range(A->Height())) {
	if (subset->Test(k)) {
	  cntx++; res_subset.Set(k);
	  for (auto j : A->GetRowIndices(k))
	    { res_subset.Set(j); }
	}
      }
      xdofs.SetSize(cntx); cntx = 0;
      // resdofs.SetSize(res_subset->NumSet()); cntres = 0;
      for (auto k : Range(A->Height())) {
	if (subset->Test(k))
	  { xdofs[cntx++] = k; }
	// if (res_subset->Test(k))
	//   { resdofs[cntres++] = k; }
      }
      /** compress A **/
      Array<int> perow(xdofs.Size()); perow = 0;
      for (auto k : Range(xdofs))
	for (auto col : A->GetRowIndices(xdofs[k]))
	  { if (res_subset.Test(col)) { perow[k]++; } }
      cA = make_shared<SparseMatrix<TM>>(perow); perow = 0;
      for (auto k : Range(xdofs)) {
	auto ri = cA->GetRowIndices(k);
	auto rv = cA->GetRowValues(k);
	auto Ari = A->GetRowIndices(xdofs[k]);
	auto Arv = A->GetRowValues(xdofs[k]);
	int c = 0;
	for (auto j : Range(Ari)) {
	  auto col = Ari[j];
	  if (res_subset.Test(col)) {
	    ri[c] = col;
	    rv[c++] = Arv[j];
	  }
	}
      }
    } // if (subset)
    else {
      xdofs.SetSize(A->Height()); //resdofs.SetSize(A->Height());
      for (auto k : Range(A->Height()))
	{ xdofs[k] = /*resdofs[k] = */ k; }
      cA = A;
    }
    /** invert diag **/
    const auto& ncA(*A);
    dinv.SetSize(xdofs.Size());
    const bool add = add_diag.Size() > 0;
    for (auto k : Range(xdofs)) {
      auto dof = xdofs[k];
      dinv[k] = ncA(dof, dof);
      if (add)
	{ dinv[k] += add_diag[dof]; }
      CalcInverse(dinv[k]);
    }
    cout << "GSS4 invs: " << endl; prow2(dinv); cout << endl;
  } // GSS4(..)
  

  template<class TM>
  void GSS4<TM> :: SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const
  {
#ifdef USE_TAU
    TAU_PROFILE("GSS4::RES", "", TAU_DEFAULT);
#endif
    static Timer t("GSS4::RES"); RegionTimer rt(t);

    const auto& A(*cA);
    auto fvx = x.FV<TV>();
    auto fvr = res.FV<TV>();
    
    double ti = MPI_Wtime();
    iterate_rows( [&](auto rownr) LAMBDA_INLINE {
	auto w = -dinv[rownr] * fvr(xdofs[rownr]);
	A.AddRowTransToVector(rownr, w, fvr);
	fvx(xdofs[rownr]) -= w;
      }, backwards);
    ti = MPI_Wtime() - ti;

    // double nrows = xdofs.Size();
    // if (nrows) {
    //   cout << endl << "--- GSS4 ---" << endl;
    //   cout << "rows " << nrows / 1000 / ti << endl;
    //   cout << "K rows/sec " << nrows / 1000 / ti << endl;
    //   cout << endl << "------" << endl;
    // }

  } // GSS4::SmoothRESInternal


  template<class TM>
  void GSS4<TM> :: SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const
  {
#ifdef USE_TAU
    TAU_PROFILE("GSS4::RHS", "", TAU_DEFAULT);
#endif
    static Timer t("GSS4::RHS"); RegionTimer rt(t);

    const auto& A(*cA);
    auto fvx = x.FV<TV>();
    auto fvb = b.FV<TV>();
    iterate_rows([&](auto rownr) LAMBDA_INLINE {
	auto r = fvb(xdofs[rownr]) - A.RowTimesVector(rownr, fvx);
	// cout << "gss4 up row " << rownr << " dof " << xdofs[rownr] << " res " << fvb(xdofs[rownr]) << " " << A.RowTimesVector(rownr, fvx) << " " << r << " update " << dinv[rownr] * r << endl;

	// if (rownr == 1246) {
	//   auto ri = A.GetRowIndices(rownr);
	//   auto rv = A.GetRowValues(rownr);
	//   cout << "row: " << endl;
	//   for (auto j : Range(ri)) {
	//     cout << "[" << j << " " << ri[j] << " " << rv[j] << " " << fvx(ri[j]) <<  "] ";
	//   }
	//       cout << endl;
	// }
	
	fvx(xdofs[rownr]) += dinv[rownr] * r;
      }, backwards);

  } // GSS4::SmoothRHSInternal


#ifdef USE_TAU
  void tauNTC (string x) {
    TAU_ENABLE_INSTRUMENTATION();
    TAU_PROFILE_SET_NODE(NgMPI_Comm(MPI_COMM_WORLD).Rank());
    int node = TAU_PROFILE_GET_NODE();
    int thread = TAU_PROFILE_GET_THREAD();
    cout << x << " TAU node / thread id " << node << " " << thread << endl; 
  }
#endif


  void thread_fun (std::mutex * m, std::condition_variable * cv, bool * done, bool * ready, bool * end, Array<MPI_Request> * reqs)
  {
#ifdef USE_TAU
    TAU_REGISTER_THREAD();
    {
      // tauNTC("--- ");
      TAU_PROFILE("thread_fun", "", TAU_DEFAULT);
      // tauNTC("--- ");
#endif

    // cout << "--- thread makes ulock" << endl;
    while(!*end) {
      std::unique_lock<std::mutex> lk(*m);
      // cout << "--- thread sleep, " << *done << " " << *ready << " " << *end << endl;
      cv->wait(lk, [ready](){ return *ready; });
      // cout << "--- thread woke up, " << *done << " " << *ready << " " << *end << endl;
      if (!*end && !*done) {
	// cout << "--- thread calls waitall" << endl;
	MyMPI_WaitAll(*reqs);
	// cout << "--- thread waitall done, notify" << endl;
	*done = true; *ready = false;
	cv->notify_one();
	// cout << "--- go back to sleep" << endl;
      }
    }
    // cout << "thread finishes" << endl;

#ifdef USE_TAU
    }
#endif
  }

  /** DCCMap **/

  template<class TSCAL>
  DCCMap<TSCAL> :: DCCMap (shared_ptr<EQCHierarchy> eqc_h, shared_ptr<ParallelDofs> _pardofs)
    : pardofs(_pardofs), block_size(_pardofs->GetEntrySize()),
      thread_done(false), thread_ready(false), end_thread(false), mpi_thread(nullptr)
  {
    ;
  } // DCCMap (..)


  template<class TSCAL>
  void DCCMap<TSCAL> :: AllocMPIStuff ()
  {

    /** alloc buffers **/
    {
      Array<int> perow;
      auto alloc_bs = [&](auto & a, auto & b) {
	perow.SetSize(b.Size());
	for (auto k : Range(perow))
	  { perow[k] = block_size * b[k].Size(); }
	a = (typename remove_reference<decltype(a)>::type)(perow);
      };
      alloc_bs(m_buffer, m_ex_dofs); // m_buffer = -1;
      alloc_bs(g_buffer, g_ex_dofs); // g_buffer = -1;
    }

    /** alloc requests **/
    m_reqs.SetSize(m_ex_dofs.Size()); m_reqs = MPI_REQUEST_NULL;
    g_reqs.SetSize(g_ex_dofs.Size()); g_reqs = MPI_REQUEST_NULL;

    /** initialize petsistent communication **/
    m_send.SetSize(m_ex_dofs.Size()); m_recv.SetSize(m_ex_dofs.Size());
    g_send.SetSize(g_ex_dofs.Size()); g_recv.SetSize(g_ex_dofs.Size());

    int cm = 0, cg = 0;
    auto comm = pardofs->GetCommunicator();
    auto ex_procs = pardofs->GetDistantProcs();
    for (auto kp : Range(ex_procs)) {
      if (g_ex_dofs[kp].Size()) { // init G send/recv
	MPI_Send_init( &g_buffer[kp][0], block_size * g_ex_dofs[kp].Size(), MyGetMPIType<TSCAL>(), ex_procs[kp], MPI_TAG_AMG + 4, comm, &g_send[cg]);
	MPI_Recv_init( &g_buffer[kp][0], block_size * g_ex_dofs[kp].Size(), MyGetMPIType<TSCAL>(), ex_procs[kp], MPI_TAG_AMG + 5, comm, &g_recv[cg]);
	cg++;
      }
      if (m_ex_dofs[kp].Size()) { // init G send/recv
	MPI_Send_init( &m_buffer[kp][0], block_size * m_ex_dofs[kp].Size(), MyGetMPIType<TSCAL>(), ex_procs[kp], MPI_TAG_AMG + 5, comm, &m_send[cm]);
	MPI_Recv_init( &m_buffer[kp][0], block_size * m_ex_dofs[kp].Size(), MyGetMPIType<TSCAL>(), ex_procs[kp], MPI_TAG_AMG + 4, comm, &m_recv[cm]);
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
  void DCCMap<TSCAL> :: StartDIS2CO (BaseVector & vec)
  {
#ifdef USE_TAU
    // tauNTC("");

    TAU_PROFILE("StartDIS2CO", TAU_CT(*this), TAU_DEFAULT);

    // tauNTC("");
#endif
    cout << "start d2c" << endl;

			   cout << " in stat " << vec.GetParallelStatus() << endl;

			   BufferG(vec);

    if (vec.GetParallelStatus() != DISTRIBUTED) // just nulling-out entries is enough
      { vec.SetParallelStatus(DISTRIBUTED); return; }

    auto ex_dofs = pardofs->GetDistantProcs();
    auto comm = pardofs->GetCommunicator();

    if (g_send.Size())
      { MPI_Startall(g_send.Size(), g_send.Data()); }
    if (m_recv.Size())
      { MPI_Startall(m_recv.Size(), m_recv.Data()); }

    // for (auto kp : Range(ex_dofs.Size())) {
    //   if (g_ex_dofs[kp].Size()) // send G vals
    // 	{ g_reqs[kp] = MyMPI_ISend(g_buffer[kp], ex_dofs[kp], MPI_TAG_AMG + 3, comm); }
    //   else
    // 	{ g_reqs[kp] = MPI_REQUEST_NULL; }

    //   if (m_ex_dofs[kp].Size()) // recv M vals
    // 	{ m_reqs[kp] = MyMPI_IRecv(m_buffer[kp], ex_dofs[kp], MPI_TAG_AMG + 3, comm); }
    //   else
    // 	{ m_reqs[kp] = MPI_REQUEST_NULL; }
    // }

  } // DCCMap::StartDIS2CO


  template<class TSCAL>
  void DCCMap<TSCAL> :: ApplyDIS2CO (BaseVector & vec)
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
  void DCCMap<TSCAL> :: FinishDIS2CO (bool shortcut)
  {
#ifdef USE_TAU
    TAU_PROFILE("FinishDIS2CO", TAU_CT(*this), TAU_DEFAULT);
#endif

    // if (shortcut)
    //   { for (auto & req : g_reqs) { MPI_Request_free(&req); } }
    // else
    //   { MyMPI_WaitAll(g_reqs); }

    MyMPI_WaitAll(g_send);

  } // DCCMap::FinishDIS2CO


  template<class TSCAL>
  void DCCMap<TSCAL> :: StartCO2CU (BaseVector & vec)
  {
#ifdef USE_TAU
    TAU_PROFILE("StartCO2CU", TAU_CT(*this), TAU_DEFAULT);
#endif

    if (vec.GetParallelStatus() != DISTRIBUTED)
      { return; }

    BufferM(vec);

    cout << "m_ex_dofs: " << endl << m_ex_dofs << endl;
    cout << "m_buffer: " << endl << m_buffer << endl;

    if (m_send.Size())
      { MPI_Startall(m_send.Size(), m_send.Data()); }
    if (g_recv.Size())
      { MPI_Startall(g_recv.Size(), g_recv.Data()); }

    // auto ex_dofs = pardofs->GetDistantProcs();
    // auto comm = pardofs->GetCommunicator();
    // for (auto kp : Range(ex_dofs.Size())) {
    //   if (g_ex_dofs[kp].Size()) // recv G vals
    // 	{ g_reqs[kp] = MyMPI_IRecv(g_buffer[kp], ex_dofs[kp], MPI_TAG_AMG + 4, comm); }
    //   else
    // 	{ g_reqs[kp] = MPI_REQUEST_NULL; }

    //   if (m_ex_dofs[kp].Size()) // send M vals
    // 	{ m_reqs[kp] = MyMPI_ISend(m_buffer[kp], ex_dofs[kp], MPI_TAG_AMG + 4, comm); }
    //   else
    // 	{ m_reqs[kp] = MPI_REQUEST_NULL; }
    // }

  } // DCCMap::StartCO2CU

    
  template<class TSCAL>
  void DCCMap<TSCAL> :: ApplyCO2CU (BaseVector & vec)
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
  void DCCMap<TSCAL> :: FinishCO2CU (bool shortcut)
  {
#ifdef USE_TAU
    TAU_PROFILE("FinishCO2CU", TAU_CT(*this), TAU_DEFAULT);
#endif

    // if (shortcut)
    //   { for (auto & req : m_reqs) { MPI_Request_free(&req); } }
    // else
    //   { MyMPI_WaitAll(m_reqs); }

    MyMPI_WaitAll(m_send);
  } // DCCMap::FinishCO2CU


  /** utility **/
  template<class TSCAL, class TDOFS, class TBUFS, class TLAM>
  INLINE void iterate_buf_vec (int block_size, TDOFS & dofs, TBUFS & bufs, BaseVector & vec, TLAM lam)
  {
    auto iterate_rows = [&](auto fun) LAMBDA_INLINE {
      for (auto kp : Range(bufs.Size())) {
	fun(dofs[kp], bufs[kp]);
      }
    };
    auto fv = vec.FV<TSCAL>();
    if (block_size == 1) {
      iterate_rows([&](auto dnrs, auto buf) LAMBDA_INLINE {
	  for (auto k : Range(dnrs)) {
	    lam(buf[k], fv(dnrs[k]));
	  }
	});
    }
    else {
      iterate_rows([&](auto dnrs, auto buf) LAMBDA_INLINE {
	  int c = 0;
	  for (auto k : Range(dnrs)) {
	    int base_etr = block_size * dnrs[k];
	    for (auto l : Range(block_size))
	      { lam(buf[c++], fv(base_etr++)); }
	  }
	});
    }
  } // iterate_buf_vec


  template<class TSCAL>
  void DCCMap<TSCAL> :: WaitD2C ()
  {
    cout << "wait d2c " << endl;
    MyMPI_WaitAll(g_send);
    MyMPI_WaitAll(m_recv);
    cout << " m buf now " << endl << m_buffer << endl;
  }

  template<class TSCAL>
  void DCCMap<TSCAL> :: WaitM ()
  {
    MyMPI_WaitAll(m_reqs);
  }


  template<class TSCAL>
  void DCCMap<TSCAL> :: WaitG ()
  {
    MyMPI_WaitAll(g_reqs);
  }


  template<class TSCAL>
  void DCCMap<TSCAL> :: BufferG (BaseVector & vec)
  {
#ifdef USE_TAU
    TAU_PROFILE("BufferG", TAU_CT(*this), TAU_DEFAULT);
#endif
    // cout << endl << "buffer G" << endl;
    iterate_buf_vec<TSCAL>(block_size, g_ex_dofs, g_buffer, vec, [&](auto & buf_etr, auto & vec_etr) LAMBDA_INLINE {
	// cout << " buffer " << vec_etr << endl;
	buf_etr = vec_etr; vec_etr = 0;
      });
    cout << endl << "buffer G" << endl;
    cout << "g_ex_dofs: " << endl << g_ex_dofs << endl;
    cout << "g_buffer: " << endl << g_buffer << endl;
  } // DCCMap::BufferG


  template<class TSCAL>
  void DCCMap<TSCAL> :: ApplyM (BaseVector & vec)
  {
#ifdef USE_TAU
    TAU_PROFILE("ApplyM", TAU_CT(*this), TAU_DEFAULT);
#endif

    cout << endl << "apply M" << endl;
    cout << "m_ex_dofs " << endl << m_ex_dofs << endl;
    cout << "m_buffer" << endl << m_buffer << endl;
    

    iterate_buf_vec<TSCAL>(block_size, m_ex_dofs, m_buffer, vec, [&](auto buf_etr, auto & vec_etr) LAMBDA_INLINE {
	// cout << vec_etr << " += " << buf_etr << " = " << vec_etr + buf_etr << endl;
	vec_etr += buf_etr;
      });
  } // DCCMap::ApplyM


  template<class TSCAL>
  void DCCMap<TSCAL> :: BufferM (BaseVector & vec)
  {
#ifdef USE_TAU
    TAU_PROFILE("BufferM", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << endl << "buffer M" << endl;
    iterate_buf_vec<TSCAL>(block_size, m_ex_dofs, m_buffer, vec, [&](auto & buf_etr, auto vec_etr) LAMBDA_INLINE {
	// cout << " buffer " << vec_etr << endl;
	buf_etr = vec_etr;
      });
    cout << endl << "buffer M" << endl;
    cout << "m_ex_dofs " << endl << m_ex_dofs << endl;
    cout << "m_buffer" << endl << m_buffer << endl;

  } // DCCMap :: BufferM


  template<class TSCAL>
  void DCCMap<TSCAL> :: ApplyG (BaseVector & vec)
  {
#ifdef USE_TAU
    TAU_PROFILE("ApplyG", TAU_CT(*this), TAU_DEFAULT);
#endif

    cout << endl << "Apply G" << endl;
    cout << "g_ex_dofs: " << endl << g_ex_dofs << endl;
    cout << "g_buffer: " << endl << g_buffer << endl;

    iterate_buf_vec<TSCAL>(block_size, g_ex_dofs, g_buffer, vec, [&](auto buf_etr, auto & vec_etr) LAMBDA_INLINE {
	// cout << "set " << buf_etr << " to " << vec_etr << endl;
	vec_etr = buf_etr;
      });
  } // DCCMap::ApplyG


  /** ChunkedDCCMap **/

  template<class TSCAL>
  ChunkedDCCMap<TSCAL> :: ChunkedDCCMap (shared_ptr<EQCHierarchy> eqc_h, shared_ptr<ParallelDofs> _pardofs,
		 int _MIN_CHUNK_SIZE)
    : DCCMap<TSCAL>(eqc_h, _pardofs), MIN_CHUNK_SIZE(_MIN_CHUNK_SIZE)
  {
    /** m_ex_dofs and g_ex_dofs **/
    CalcDOFMasters(eqc_h);

    /** requests and buffers **/
    this->AllocMPIStuff();

  } // ChunkedDCCMap (..)


  template<class TSCAL>
  void ChunkedDCCMap<TSCAL> :: CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h)
  {
    static Timer t("ChunkedDCCMap::CalcDOFMasters"); RegionTimer rt(t);

    auto neqcs = eqc_h->GetNEQCS();
    auto comm = eqc_h->GetCommunicator();
    auto ex_procs = eqc_h->GetDistantProcs();
    auto nexp = ex_procs.Size();

    // cout << " ChunkedDCCMap on EQCH: " << endl << *eqc_h << endl;
    
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
	    chunk_size = exds / n_chunks;
	    int rest = exds % chunk_size;
	    // start handing out chunks with an arbitrary member (but no RNG, so it is reproducible)
	    int start = (1029 * eqc_h->GetEQCID(k) / 7) % nmems;
	    for (auto j : Range(n_chunks))
	      { chunk_so[k][(start + j) % nmems] = chunk_size + ( (j < rest) ? 1 : 0 ); }

	    // if (chunk_so[k].Size())
	    //   { chunk_so[k][0] = exds; }
	    
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
	    for (auto j : Range(csk)) {
	      // cout << "chunk for mem mem " << j << " has sz " << csk[j] << endl;
	      if (csk[j] > 0) {
		auto exp = memsk[j];
		// cout << " mem is proc " << exp << endl;
		if (exp == comm.Rank()) { // MY chunk!
		  // cout << " thats me!" << endl;
		  for (auto p : dps) {
		    auto kp = find_in_sorted_array(p, ex_procs);
		    for (auto l : Range(csk[j])) {
		      c_m_exd.Add(kp, dofsk[offset + l]);
		      // cout << " my dof, set " << dofsk[offset + l] << endl;
		      m_dofs->Set(dofsk[offset + l]);
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


  /** HybridMatrix **/

  template<class TM>
  HybridMatrix2<TM> :: HybridMatrix2 (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map)
    : dcc_map(_dcc_map)
  {
    if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat)) {
      // can still have dummy-pardofs (ngs-amg w. NP==1) or no ex-procs (NP==2)
      dummy = false;
      if (auto loc_spmat = dynamic_pointer_cast<SparseMatrix<TM>> (parmat->GetMatrix())) {
	pardofs = parmat->GetParallelDofs();
	SetParallelDofs(pardofs);
	SetUpMats(loc_spmat);
	g_zero = false;
	int nzg = (G != nullptr) ? 1 : 0;
	nzg = parmat->GetParallelDofs()->GetCommunicator().AllReduce(nzg, MPI_SUM);
	g_zero = (nzg == 0);
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
      G = nullptr; g_zero = true;
    }
  }


  template<class TM>
  void HybridMatrix2<TM> :: SetUpMats (shared_ptr<SparseMatrix<TM>> anA)
  {
    string cn = string("HybridMatrix<" + to_string(mat_traits<TM>::HEIGHT) + string(">"));
    static Timer t(cn + "::SetUpMats"); RegionTimer rt(t);

    auto& A(*anA);
    auto& pds(*pardofs);

    auto H = A.Height();
    NgsAMG_Comm comm(pds.GetCommunicator());
    auto ex_procs = pds.GetDistantProcs();
    auto nexp = ex_procs.Size();

    if (nexp == 0) {
      // we are still not "dummy", because we are still working with parallel vectors
      M = anA;
      G = nullptr;
      return;
    }

    typedef SparseMatrixTM<TM> TSPMAT_TM;
    Array<TSPMAT_TM*> send_add_diag_mats(nexp);
    Array<MPI_Request> rsdmat (nexp);
    { // create & send diag-blocks for M to masters
      static Timer t(cn + "::SetUpMats - create diag"); RegionTimer rt(t);
      auto SPM_DIAG = [&] (auto dofs) LAMBDA_INLINE {
	auto iterate_coo = [&](auto fun) LAMBDA_INLINE {
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
	iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE { perow[i]++; });
	TSPMAT_TM* dspm = new TSPMAT_TM(perow, perow.Size());
	perow = 0;
	iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE {
	    dspm->GetRowIndices(i)[perow[i]] = j;
	    dspm->GetRowValues(i)[perow[i]++] = val;
	  });
	return dspm;
      };
      for (auto kp : Range(nexp)) {
	send_add_diag_mats[kp] = SPM_DIAG(dcc_map->GetGDOFs(kp));
	rsdmat[kp] = comm.ISend(*send_add_diag_mats[kp], ex_procs[kp], MPI_TAG_AMG);
      }
    } // create & send diag-blocks for M to masters

    Array<shared_ptr<TSPMAT_TM>> recv_mats(nexp);
    { // recv diag-mats
      static Timer t(cn + "::SetUpMats - recv diag"); RegionTimer rt(t);
      for (auto kp : Range(nexp))
	{
	  comm.Recv(recv_mats[kp], ex_procs[kp], MPI_TAG_AMG);
	}
    }

    { // merge diag-mats
      static Timer t(cn + "::SetUpMats - merge"); RegionTimer rt(t);

      auto& mf_dofs = *dcc_map->GetMasterDOFs();

      Array<int> perow(H); perow = 0;
      Array<size_t> at_row(nexp);
      Array<int> row_matis(nexp);
      Array<FlatArray<int>> all_cols;
      Array<int> mrowis(50); mrowis.SetSize0(); // col-inds for a row of orig mat (but have to remove a couple first)

      auto iterate_rowinds = [&](auto fun, bool map_exd) LAMBDA_INLINE {
	at_row = 0; // restart counting through rows of recv-mats
	for (auto rownr : Range(H)) {
	  if (mf_dofs.Test(rownr)) { // I am master of this dof - merge recved rows with part of original row
	    row_matis.SetSize0(); // which mats I received have this row?
	    if (pds.GetDistantProcs(rownr).Size()) { // local master
	      for (auto kp : Range(nexp)) {
		auto exds = dcc_map->GetMDOFs(kp);
		if (at_row[kp] == exds.Size()) continue; // no more rows to take from there
		auto ar = at_row[kp]; // the next row for that ex-mat
		size_t ar_dof = exds[ar]; // the dof that row belongs to
		if (ar_dof > rownr) continue; // not yet that row's turn
		row_matis.Append(kp);
		// cout << "row " << at_row[kp] << " from " << kp << endl;
		at_row[kp]++;
	      }
	      all_cols.SetSize0(); all_cols.SetSize(1+row_matis.Size()); // otherwise tries to copy FA I think
	      for (auto k:Range(all_cols.Size()-1)) {
		auto kp = row_matis[k];
		auto cols = recv_mats[kp]->GetRowIndices(at_row[kp]-1);
		if (map_exd) { // !!! <- remap col-nrs of received rows, only do this ONCE!!
		  auto mxd = dcc_map->GetMDOFs(kp);
		  for (auto j:Range(cols.Size()))
		    { cols[j] = mxd[cols[j]]; }
		}
		all_cols[k].Assign(cols);
	      }

	      // only master-master (not master-all) goes into M
	      auto aris = A.GetRowIndices(rownr);
	      mrowis.SetSize(aris.Size()); int c = 0;
	      for (auto col : aris)
		if (mf_dofs.Test(col))
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
		if (mf_dofs.Test(col))
		  mrowis[c++] = col;
	      mrowis.SetSize(c);
	      fun(rownr, row_matis, mrowis);
	    }
	  } // mf_dofs.Test(rownr)
	}
      };

      iterate_rowinds([&](auto rownr, const auto &matis, const auto &rowis) LAMBDA_INLINE {
	  perow[rownr] = rowis.Size(); }, true);

      M = make_shared<SparseMatrix<TM>>(perow);

      // cout << "M NZE : " << M->NZE() << endl;

      iterate_rowinds([&](auto rownr, const auto & matis, const auto & rowis) {
	  auto ris = M->GetRowIndices(rownr); ris = rowis;
	  auto rvs = M->GetRowValues(rownr); rvs = 0;
	  // cout << "rownr, rowis: " << rownr << ", "; prow2(rowis); cout << endl;
	  // cout << "rownr, matis: " << rownr << ", "; prow2(matis); cout << endl;
	  // cout << ris.Size() << " " << rvs.Size() << " " << rowis.Size() << endl;
	  auto add_vals = [&](auto cols, auto vals) LAMBDA_INLINE {
	    for (auto l : Range(cols)) {
	      auto pos = find_in_sorted_array<int>(cols[l], ris);
	      // cout << "look for " << cols[l] << " in "; prow(ris); cout << " -> pos " << pos << endl;
	      if (pos != -1)
		{ rvs[pos] += vals[l]; }
	    }
	  };
	  add_vals(A.GetRowIndices(rownr), A.GetRowValues(rownr));
	  for (auto kp : matis) {
	    // cout << "row " << at_row[kp] -1 << " from kp " << kp << endl;
	    add_vals(recv_mats[kp]->GetRowIndices(at_row[kp]-1),
		     recv_mats[kp]->GetRowValues(at_row[kp]-1));
	  }
	}, false);

    } // merge diag-mats

    // cout << endl << "ORIG A MAT: " << endl << A << endl << endl ;
    // cout << endl  << "M done: " << endl << *M << endl << endl ;
    
    { // build G-matrix
      static Timer t(cn + "::SetUpMats - S-mat"); RegionTimer rt(t);
      // ATTENTION: explicitely only for symmetric matrices!
      Array<int> master_of(H); // DOF to master mapping (0 .. nexp-1 ordering // -1 -> I am master)
      master_of = -1;
      for (auto kp : Range(ex_procs))
	for (auto d : dcc_map->GetGDOFs(kp))
	  { master_of[d] = kp; }
      Array<int> perow(A.Height());
      auto iterate_coo = [&](auto fun) LAMBDA_INLINE { // could be better if we had GetMaxExchangeDof(), or GetExchangeDofs()
      	perow = 0;
      	for (auto rownr : Range(A.Height())) {
      	  auto ris = A.GetRowIndices(rownr);
      	  auto rvs = A.GetRowValues(rownr);
      	  auto mrow = master_of[rownr];
      	  for (auto j : Range(ris)) {
      	    if (master_of[ris[j]] != mrow) {
      	      fun(rownr, ris[j], rvs[j]);
      	    }
      	  }
      	}
      };
      iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE {
	  perow[i]++;
	});
      G = make_shared<SparseMatrix<TM>>(perow);
      iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE {
	  G->GetRowIndices(i)[perow[i]] = j;
	  G->GetRowValues(i)[perow[i]++] = val;
	});
    } // build G-matrix

    // cout << endl  << "G done: " << endl << *G << endl << endl ;

    {
      static Timer t(cn + "::SetUpMats - finish send"); RegionTimer rt(t);
      MyMPI_WaitAll(rsdmat);
      for (auto kp : Range(nexp))
	delete send_add_diag_mats[kp];
    }

  } // HybridMatrix2::SetUpMats


  template<class TM>
  void HybridMatrix2<TM> :: MultAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();

    M->MultAdd(s, get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultAdd(s, get_loc_ref(x), get_loc_ref(y)); }
  }

  template<class TM>
  void HybridMatrix2<TM> :: MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();

    M->MultAdd(s, get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultAdd(s, get_loc_ref(x), get_loc_ref(y)); }
  }

  template<class TM>
  void HybridMatrix2<TM> :: Mult (const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.SetParallelStatus(DISTRIBUTED);

    M->Mult(get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultAdd(1.0, get_loc_ref(x), get_loc_ref(y)); }
  }

  template<class TM>
  void HybridMatrix2<TM> :: MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultTransAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultTransAdd(s, get_loc_ref(x), get_loc_ref(y)); }
  }

  template<class TM>
  void HybridMatrix2<TM> :: MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultTransAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.Distribute();
    M->MultTransAdd(s, get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultTransAdd(s, get_loc_ref(x), get_loc_ref(y)); }
  }

  template<class TM>
  void HybridMatrix2<TM> :: MultTrans (const BaseVector & x, BaseVector & y) const
  {
    static Timer t(string("HybridMatrix<bs=")+to_string(BS())+">::MultTransAdd");
    RegionTimer rt(t);

    x.Cumulate();
    y.SetParallelStatus(DISTRIBUTED);

    M->MultTrans(get_loc_ref(x), get_loc_ref(y));
    if (G != nullptr)
      { G->MultTransAdd(1.0, get_loc_ref(x), get_loc_ref(y)); }
  }


  /** HybridSmoother2 **/

  template<class TM>
  HybridSmoother2<TM> :: HybridSmoother2 (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h,
					  bool _overlap, bool _in_thread)
    : overlap(_overlap), in_thread(_in_thread)
  {

    shared_ptr<DCCMap<typename mat_traits<TM>::TSCAL>> dcc_map = nullptr;
    if (auto pds = _A->GetParallelDofs())
      { dcc_map = make_shared<ChunkedDCCMap<typename mat_traits<TM>::TSCAL>>(eqc_h, pds, 10); }

    A = make_shared<HybridMatrix2<TM>> (_A, dcc_map);
    origA = _A;

    auto pardofs = A->GetParallelDofs();

    SetParallelDofs(pardofs);

    if (A->GetG() != nullptr)
      { Gx = make_shared<S_BaseVectorPtr<double>> (A->Height(), A->BS()); }
    
  }


  template<class TM>
  void HybridSmoother2<TM> :: Smooth (BaseVector  &x, const BaseVector &b,
				 BaseVector  &res, bool res_updated,
				 bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 2 : 0, x, b, res, res_updated, update_res, x_zero);
  }


  template<class TM>
  void HybridSmoother2<TM> :: SmoothBack (BaseVector  &x, const BaseVector &b,
				     BaseVector &res, bool res_updated,
				     bool update_res, bool x_zero) const
  {
    SmoothInternal(smooth_symmetric ? 2 : 1, x, b, res, res_updated, update_res, x_zero);
  }


  template<class TM>
  void HybridSmoother2<TM> :: SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
					     bool res_updated, bool update_res, bool x_zero) const
  {

    static Timer t(string("HybSm<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::Smooth");
    RegionTimer rt(t);

    static Timer tpre(string("HybSm<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::S - pre");
    static Timer tpost(string("HybSm<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::S - post");

    /** most of the time RU == UR, if not, reduce to such a case **/
    if (res_updated && !update_res) { // RU && !UR
      SmoothInternal(type, x, b, res, false, false, x_zero); // this is actually cheaper I think
      return;
    }
    else if (!res_updated && update_res) { // !RU + UR
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

#ifdef USE_TAU
    TAU_PROFILE("", TAU_CT(*this), TAU_DEFAULT)
#endif

    // if (!update_res) {
    //   res = b - *A * x;
    //   SmoothInternal(type, x, b, res, true, true, x_zero);
    //   return;
    // }
 
    // if (update_res) {
    //   res = 0;
    //   SmoothInternal(type, x, b, res, false, false, x_zero);
    //   res = b - *A * x;
    //   return;
    // }

    auto dcc_map = A->GetMap();
    const auto H = A->Height();
    auto G = A->GetG();
    auto & xloc = *get_loc_ptr(x);
    auto & ncb = const_cast<BaseVector&>(b);
    const auto & bloc = *get_loc_ptr(b);
    auto & ncbloc = const_cast<BaseVector&>(bloc);
    auto & resloc = *get_loc_ptr(res);
    auto & Gxloc = *Gx;

    /** 
	!MPI &  RES           -> use res
	!MPI & !RES           -> use b
	 MPI &  RES           -> use res
	 MPI & !RES &  x_zero -> use b ( b - G x = b - G 0 = b)
	 MPI & !RES & !x_zero -> use res ( = b - G x)
     **/

    const bool g_zero = A->HasG();
    const bool need_gx = (!x_zero) && (!g_zero);
    const bool use_b = (!update_res) && (!need_gx);

    // cout << update_res << " " << x_zero << " / " << g_zero << " " << need_gx << " " << use_b << " " << endl;

    { // calculate b-Gx as RHS for local smooth if update res, otherwise stash Gx
#ifdef USE_TAU
    TAU_PROFILE("PREP", TAU_CT(*this), TAU_DEFAULT)
#endif

      RegionTimer rt(tpre);

      x.Cumulate(); // should do nothing most of the time

      if (!res_updated) { // feed in b-G*x as RHS, not res-update
	if ( need_gx ) {
	  b.Distribute();
	  if (G != nullptr) // rank 0
	    { resloc = bloc - *G * xloc; }
	  else {
	    resloc = bloc;
	  }
	  res.SetParallelStatus(DISTRIBUTED);
	}
      }
      else if ( need_gx && (G != nullptr) )// stash G*x_old, because afterwards we get out b-Mx_new-Gx_old
	{ Gxloc = *G * xloc; }

    }

    auto smooth_stage = [&](int stage) LAMBDA_INLINE {
      if (type == 0) {
    	if (update_res)
    	  { SmoothRESLocal(stage, xloc, resloc); }
    	else
    	  { SmoothLocal(stage, xloc, use_b ? ncbloc : resloc); }
      }
      else if (type == 1) {
    	if (update_res)
    	  { SmoothBackRESLocal(stage, xloc, resloc); }
    	else
    	  { SmoothBackLocal(stage, xloc, use_b ? ncbloc : resloc); }
      }
    };

    // if (dcc_map != nullptr)
    //   { dcc_map->StartDIS2CO(*sm_rhs); }
    // else { throw Exception("oops"); }
    // // // smooth_stage(0);

    // if (dcc_map != nullptr)
    //   { dcc_map->ApplyDIS2CO(*sm_rhs); }
    // else { throw Exception("oops"); }


    // resloc = bloc; res.SetParallelStatus(b.GetParallelStatus());
    // x.Cumulate();
    // res.Distribute();
    // if (G != nullptr)
    //   { resloc -= *G * xloc; }

    // cout << "4 SMINTERNAL" << type << " " << x.GetParallelStatus() << " " << b.GetParallelStatus() << " " << res.GetParallelStatus() << " / " << res_updated << " " << update_res << " " << x_zero << endl;

    auto Apardofs = A->GetParallelDofs();
    auto prv = [Apardofs](auto& x) {
      cout << endl << "type: " << typeid(x).name() << endl;
      cout << endl << "stat: " << x.GetParallelStatus() << endl;
      for (auto k : Range(x.FVDouble().Size()))
	{ cout << k << ": " << x.FVDouble()[k] << "  ||  "; prow(Apardofs->GetDistantProcs(k)); cout << endl; }
      cout << endl;
    };

    // cout << " smooth, up res " << update_res << endl;
    // cout << " x in " << endl; prv(x);
    // cout << " b in " << endl; prv(b);
    // cout << " r in " << endl; prv(res);


    BaseVector * mpivec = use_b ? &ncb : &res;
    condition_variable* pcv = &cv;
    bool vals_buffered = false;
    bool need_d2c = mpivec->GetParallelStatus() == DISTRIBUTED;

    if (need_d2c) {
      if (overlap && in_thread) {
	std::lock_guard<std::mutex> lk(glob_mut);
	thread_done = false; thread_ready = true;
	thread_exec_fun = [&]() {
	  dcc_map->StartDIS2CO(*mpivec);
	  dcc_map->WaitD2C();
	};
	cv.notify_one();
      }
      else
	{ dcc_map->StartDIS2CO(*mpivec); }
  
      if (!overlap)
	{ dcc_map->WaitD2C(); }
    }

    smooth_stage(0);
    cout << " x 0 " << endl << x << endl;

    if (need_d2c) {
      if (overlap) {
	if (in_thread) {
	  std::unique_lock<std::mutex> lk(glob_mut);
	  cv.wait(lk, [&]{ return thread_done; });
	}
	else
	  { dcc_map->WaitD2C(); }
      }
      dcc_map->ApplyM(*mpivec);
    }

    cout << " x 1 " << endl << x << endl;
    smooth_stage(1);

    cout << " x 2 " << endl << x << endl;
    x.SetParallelStatus(DISTRIBUTED);
    mpivec = &x;

    if (overlap && in_thread) {
      std::lock_guard<std::mutex> lk(glob_mut); // get lock
      thread_done = false; thread_ready = true;
      thread_exec_fun = [mpivec, dcc_map] () {
	dcc_map->StartCO2CU(*mpivec);
	dcc_map->ApplyCO2CU(*mpivec);
	dcc_map->FinishCO2CU(false); // probably dont take a shortcut ??
      };
      cv.notify_one();
    }
    else
      { dcc_map->StartCO2CU(*mpivec); }
    cout << " x 3 " << endl << x << endl;

    if (!overlap) {
      dcc_map->ApplyCO2CU(*mpivec);
      dcc_map->FinishCO2CU(false); // probably dont take a shortcut ??
    }

    smooth_stage(2);
    cout << " x 4 " << endl << x << endl;

    if (overlap) {
      if (in_thread) {
	std::unique_lock<std::mutex> lk(glob_mut);
	cv.wait(lk, [&]{ return thread_done; });
      }
      else {
	dcc_map->ApplyCO2CU(*mpivec);
	dcc_map->FinishCO2CU(false); // probably dont take a shortcut ??
      }
    }
    cout << " x 5 " << endl << x << endl;

    { // scatter updates and finish update residuum, res -= S * (x - x_old)
      RegionTimer rt(tpost);
#ifdef USE_TAU
    TAU_PROFILE("POST", TAU_CT(*this), TAU_DEFAULT)
#endif
      if (update_res && (!g_zero) ) {
    	res.Distribute();
    	if (G != nullptr) {
    	  if (!x_zero) // stashed G*x_old
    	    { resloc += Gxloc; }
	  G->MultAdd(-1, xloc, resloc);
    	}
      }
    }
    cout << " x 6 " << endl << x << endl;

  } // HybridSmoother2<TM> :: SmoothInternal


  /** HybridGSS3 **/

  template<class TM>
  HybridGSS3<TM> :: HybridGSS3 (shared_ptr<BaseMatrix> _A, shared_ptr<EQCHierarchy> eqc_h, shared_ptr<BitArray> _subset,
				bool _overlap, bool _in_thread)
    : HybridSmoother2<TM>(_A, eqc_h, _overlap, _in_thread)
  {
    auto& M = *A->GetM();

    Array<TM> add_diag = this->CalcAdditionalDiag();

    auto pardofs = A->GetParallelDofs();

    auto m_dofs = A->GetMap()->GetMasterDOFs();

    if (pardofs != nullptr)
      for (auto k : Range(add_diag.Size()))
	if ( ((!_subset) || (_subset->Test(k))) && ((!m_dofs) || (m_dofs->Test(k))) )
	  { M(k,k) += add_diag[k]; }

    shared_ptr<BitArray> loc = _subset, ex = nullptr;
    if (pardofs != nullptr) {
      loc = make_shared<BitArray>(M.Height()); loc->Clear();
      ex = make_shared<BitArray>(M.Height()); ex->Clear();
      for (auto k : Range(M.Height())) {
	if (m_dofs->Test(k)) {
	  auto dps = pardofs->GetDistantProcs(k);
	  if (dps.Size()) { ex->Set(k); }
	  else { loc->Set(k); }
	}
      }
      if (_subset != nullptr) {
	loc->And(*_subset);
	ex->And(*_subset);
	// loc->Or(*ex);
      }
    }

    // if (_subset)
    //   { cout << "subset: " << endl << *_subset << endl; }
    // else
    //   { cout << " NO SUBSET!" << endl; }

    // cout << "m_dofs: " << endl << *m_dofs << endl;
    if (_subset) {
      cout << "rank " << A->GetParallelDofs()->GetCommunicator().Rank() << " numsets " << loc->NumSet() << " " << ex->NumSet() << " " << loc->Size() << " "
	   << double(loc->NumSet()) / loc->Size() << " " << double(ex->NumSet()) / ex->Size() << endl;
    }
    
    jac_loc = make_shared<GSS3<TM>>(A->GetM(), loc);
    if ( (pardofs != nullptr) && (ex->NumSet() != 0) ) {
      jac_exo  = make_shared<GSS3<TM>>(A->GetM(), ex);
    }

    if (pardofs != nullptr)
      for (auto k : Range(add_diag.Size()))
	if ( ((!_subset) || (_subset->Test(k))) && ((!m_dofs) || (m_dofs->Test(k))) )
	  { M(k,k) -= add_diag[k]; }

    if ( (pardofs != nullptr) && (ex->NumSet() != 0) ) {
      jac_ex  = make_shared<GSS4<TM>>(A->GetM(), ex, add_diag);
    }

  } // HybridGSS3 (..)


  template<class TM> INLINE void AddODToD (const TM & v, TM & w) {
    Iterate<mat_traits<TM>::HEIGHT>([&](auto i) LAMBDA_INLINE {
	Iterate<mat_traits<TM>::HEIGHT>([&](auto j) LAMBDA_INLINE {
    w(i.value, i.value) += 0.5 * fabs(v(i.value, j.value));
	  });
      });
  }
  template<> INLINE void AddODToD<double> (const double & v, double & w)
  { w += 0.5 * fabs(v); }
  template<class TM>
  Array<TM> HybridSmoother2<TM> :: CalcAdditionalDiag ()
  {
    static Timer t(string("HybridGSS<bs=")+to_string(mat_traits<TM>::HEIGHT)+">>::CalcAdditionalDiag");
    RegionTimer rt(t);

    Array<TM> add_diag;

    auto pardofs = A->GetParallelDofs();
    if ( (pardofs == nullptr) || (pardofs->GetDistantProcs().Size() == 0) )
      { return add_diag; }

    add_diag.SetSize(A->Height()); add_diag = 0;
 
    auto G = A->GetG();
    const auto & M = *A->GetM();
   
    if (G == nullptr)
      { return add_diag; }

    for (auto k : Range(G->Height())) {
      // auto ris = G->GetRowIndices(k);
      auto rvs = G->GetRowValues(k);
      for (auto l : Range(rvs)) {
	AddODToD(rvs[l], add_diag[k]);
	// AddODToD(rvs[l], add_diag[ris[l]]);
      }
    }

    AllReduceDofData (add_diag, MPI_SUM, A->GetParallelDofs());  

    // if constexpr( is_same<TM, double>::value )
    //   {
    // 	for (auto k : Range(M.Height())) {
    // 	  if ( (add_diag[k] != 0) && (M(k,k) != 0) ) {
    // 	    if (M(k,k) > 3 * add_diag[k])
    // 	      { add_diag[k] = 0; }
    // 	  }
    // 	}
    //   }

    cout << "add_diag" << endl; prow2(add_diag); cout << endl;

    return add_diag;
  }


  template<class TM>
  void HybridGSS3<TM> :: SmoothLocal (int stage, BaseVector &x, const BaseVector &b) const
  {
    if (stage == 0)
      { jac_loc->Smooth(0, A->Height()/2, x, b); }
    else if ( (stage == 1) && (jac_ex != nullptr) ) {
      jac_ex->Smooth(x, b);
      // jac_exo->Smooth(0, A->Height(), x, b);
    }
  else if (stage == 2)
      { jac_loc->Smooth(A->Height()/2, A->Height(), x, b); }
  }


  template<class TM>
  void HybridGSS3<TM> :: SmoothBackLocal (int stage, BaseVector &x, const BaseVector &b) const
  {
    if (stage == 0)
      { jac_loc->SmoothBack(A->Height()/2, A->Height(), x, b); }
    else if ( (stage == 1) && (jac_ex != nullptr) ) {
      jac_ex->SmoothBack(x, b);
      // jac_exo->SmoothBack(0, A->Height(), x, b);
    }
    else if (stage == 2)
      { jac_loc->SmoothBack(0, A->Height()/2, x, b); }
  }


  template<class TM>
  void HybridGSS3<TM> :: SmoothRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    if (stage == 0)
      { jac_loc->SmoothRES(0, A->Height()/2, x, res); }
    else if ( (stage == 1) && (jac_ex != nullptr) ) {
      jac_ex->SmoothRES(x, res);
      // jac_exo->SmoothRES(0, A->Height(), x, res);
    }
    else if (stage == 2)
      { jac_loc->SmoothRES(A->Height()/2, A->Height(), x, res); }
  }


  template<class TM>
  void HybridGSS3<TM> :: SmoothBackRESLocal (int stage, BaseVector &x, BaseVector &res) const
  {
    if (stage == 0)
      { jac_loc->SmoothBackRES(A->Height()/2, A->Height(), x, res); }
    else if ( (stage == 1) && (jac_ex != nullptr) )  {
      jac_ex->SmoothBackRES(x, res);
      // jac_exo->SmoothBackRES(0, A->Height(), x, res);
    }
    else if (stage == 2)
      { jac_loc->SmoothBackRES(0, A->Height()/2, x, res); }
  }

} // namespace amg

#include "amg_tcs.hpp"
