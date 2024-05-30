#include <base.hpp>
#include <universal_dofs.hpp>
#include <utils_arrays_tables.hpp>

#include "eqchierarchy.hpp"
#include "mpiwrap_extension.hpp"

namespace amg
{

EQCHierarchy :: EQCHierarchy ()
  : is_dummy(true)
  , comm() // creates a dummy Communicator object
  , rank(0)
  , np(1)
  , neqcs(1)
  , neqcs_glob(1)
  , _isRankZeroIdle(false)
  , _isTrulyParallel(false)
{
  eqc_ids.SetSize(1); eqc_ids = 1; // Note: ID should never be zero!
  idf_2_ind.SetSize(2); idf_2_ind = 0;
  Array<int> perow(1); perow = 0;
  dist_procs = Table<int>(perow);
  all_dist_procs.SetSize0();
  merge.SetSize(1); merge = 0;
  mat_merge.AssignMemory(1, 1, merge.Data());
  intersect.SetSize(1); intersect = 0;
  mat_intersect.AssignMemory(1, 1, intersect.Data());
  hierarchic_order.SetSize(1); hierarchic_order.SetBit(0);
} // EQCHierarchy(..)


  EQCHierarchy :: EQCHierarchy (const shared_ptr<MeshAccess> & ma, Array<NODE_TYPE> nts, bool do_cutunion)
    : is_dummy(false)
    , comm(ma->GetCommunicator())
    , _isRankZeroIdle(true) // when setting up from MA, rank 0 MUST be idle!
    , _isTrulyParallel(comm.Size() > 2)
  {
    static Timer t("EQCHierarchy::Constructor 1"); RegionTimer rt(t);
    Table<int> vanilla_dps;
    Array<int> size_of_dps(100); size_of_dps.SetSize0();
    Array<int> index_of_block(100); index_of_block.SetSize0();
    Array<NODE_TYPE> nt_of_block(100); nt_of_block.SetSize0();
    this->rank = comm.Rank(); this->np = comm.Size();
    neqcs = 0;	neqcs_glob = 0;
    bool has_any = false;
    for (NODE_TYPE NT : nts) {
      auto nnodes = ma->GetNNodes(NT);
      if (nnodes) has_any = true;
    }
    if (has_any) { size_of_dps.Append(0); index_of_block.Append(0); nt_of_block.Append(NT_VERTEX); neqcs++; }
    for (NODE_TYPE NT : nts) {
      auto nnodes = ma->GetNNodes(NT);
      for (auto k : Range(nnodes)) {
	auto dps = ma->GetDistantProcs(NodeId(NT, k));
	if (!dps.Size()) continue;
	int pos = -1;
	for (size_t l = 1; l < index_of_block.Size() && (pos==-1);l++) // 0 is seperate
	  if (ma->GetDistantProcs(NodeId(NT, index_of_block[l]))==dps)
	    pos = l;
	if (pos>=0) continue;
	index_of_block.Append(k);
	nt_of_block.Append(NT);
	size_of_dps.Append(dps.Size());
	neqcs++;
      }
    }
    vanilla_dps = Table<int>(size_of_dps);
    if(neqcs > 1)
      for (auto k : Range(size_t(1), neqcs))
	{ vanilla_dps[k] = ma->GetDistantProcs(NodeId(nt_of_block[k], index_of_block[k])); }
    if (do_cutunion)
      this->SetupFromInitialDPs(std::move(vanilla_dps));
    else
      this->SetupFromDPs(std::move(vanilla_dps));

  }


  EQCHierarchy :: EQCHierarchy (const shared_ptr<ParallelDofs> & apd, bool do_cutunion,
				size_t _max_nd, shared_ptr<BitArray> select)
    : is_dummy(false)
    , comm(apd->GetCommunicator())
    , _isRankZeroIdle(::amg::IsRankZeroIdle(apd))
    , _isTrulyParallel(comm.Size() > ( ::amg::IsRankZeroIdle(apd) ? 2 : 1 ) )
  {
    static Timer t("EQCHierarchy::Constructor 2"); RegionTimer rt(t);
    // if(apd==nullptr) {
    //   this->comm = ngs_comm;
    //   if(MyMPI_GetNTasks()>1) throw Exception("EQCHierarchy with no paralleldofs in non-seq environ!");
    // }
    // else {
    //   this->comm = apd->GetCommunicator();
    // }

    this->rank = comm.Rank();
    this->np = comm.Size();

    Table<int> vanilla_dps;
    size_t max_nd = (_max_nd == -1) ? apd->GetNDofLocal() : _max_nd;
    // cout << " size_t max_nd = " << (_max_nd == -1) << " ? " << apd->GetNDofLocal() << " : " <<  _max_nd << endl;
    if (apd != nullptr)
      {
	int ndof = apd->GetNDofLocal();
	neqcs = 0;
	neqcs_glob = 0;
	Array<int> index_of_block(50); //first point in block
	index_of_block.SetSize(0);
	Array<int> size_of_dps(50);
	size_of_dps.SetSize(0);
	if (ndof > 0) { size_of_dps.Append(0); index_of_block.Append(-1); neqcs++; }
	for (auto k : Range(max_nd)) {
	  if (select && (!select->Test(k))) // DOF is not selected (e.g definedon/bddc stuff, etc...)
	    { continue; }
	  auto dp = apd->GetDistantProcs(k);
	  if (!dp.Size())
	    { continue; }
	  int pos = -1;
	  for (size_t l = 1; l<index_of_block.Size() && (pos == -1); l++)
	    if (apd->GetDistantProcs(index_of_block[l]) == dp)
	      { pos = l; }
	  if (pos >= 0)
	    { continue; }
	  //new eqc found!
	  index_of_block.Append(k);
	  size_of_dps.Append(dp.Size());
	  neqcs++;
	}
	vanilla_dps = Table<int>(size_of_dps);
	if (neqcs > 1) { //size_t range...
	  for (auto k:Range(size_t(1), neqcs)) {
	    // if (vanilla_dps[k].Size()) { // i think i dont need this anymore since local EQ is hard-coded to 0
	    auto dps = apd->GetDistantProcs(index_of_block[k]);
	    vanilla_dps[k] = dps;
	    //}
	  }
	}
      }

    // cout << " have vanilla_DPS " << endl << vanilla_dps << endl;

    if (do_cutunion)
      this->SetupFromInitialDPs(std::move(vanilla_dps));
    else
      this->SetupFromDPs(std::move(vanilla_dps));

    return;
  } // end EQCHierarchy(pardofs,...)


  EQCHierarchy :: EQCHierarchy (Table<int> && eqcs, NgsAMG_Comm acomm, bool do_cutunion, bool isRankZeroIdle)
    : is_dummy(false)
    , comm(acomm)
    , _isRankZeroIdle(isRankZeroIdle)
    , _isTrulyParallel(comm.Size() > ( isRankZeroIdle ? 2 : 1 ) )
  {
    static Timer t("EQCHierarchy::Constructor 3"); RegionTimer rt(t);

    this->rank = comm.Rank();
    this->np = comm.Size();

    if (do_cutunion)
      this->SetupFromInitialDPs(std::move(eqcs));
    else
      this->SetupFromDPs(std::move(eqcs));

    return;
  } // end EQCHierarchy(dps,...)


  EQCHierarchy :: EQCHierarchy (FlatArray<int> alldps, FlatTable<int> aloc_mems, NgsAMG_Comm acomm, bool do_cutunion)
    : is_dummy(false), comm(acomm), rank(acomm.Rank()), np(acomm.Size())
  {
    // cout << " new EQCHierarchy with " << alldps.Size() << " dps = "; prow(alldps); cout << endl;
    all_dist_procs.SetSize(alldps.Size()); all_dist_procs = alldps;

    Table<int> eqcs;

    /** Communicate the eqs to potential neibs **/
    TableCreator<int> c_msgs(alldps.Size()); // every message: [size, mems, size, mems, ...]
    Array<int> eq_mems;
    for (; !c_msgs.Done(); c_msgs++)
      for (auto k : Range(aloc_mems)) {
	eq_mems.SetSize(aloc_mems[k].Size()); // member arrays here
	eq_mems = aloc_mems[k];
	insert_into_sorted_array_nodups(int(comm.Rank()), eq_mems);
	// auto eq_mems = aloc_mems[k];
	// cout << " eq " << k << " procs "; prow(eq_mems); cout << endl;
	for (auto p : aloc_mems[k]) { // only dist-procs here
	  auto kp = find_in_sorted_array(p, alldps);
	  // cout << p << " " << kp << endl;
	  if (kp != -1) {
	    c_msgs.Add(kp, aloc_mems[k].Size()); // we SEND only dp-arrays
	    // cout << " add " << eq_mems.Size() << endl;
	    for (auto pj : eq_mems)
	      if (pj != p) // !! sends ALL mems...
		{ c_msgs.Add(kp, pj); /* cout << pj << " " << endl; */ }
	  }
	}
      }
    auto s_msgs = c_msgs.MoveTable();

    // cout << " s_msgs" << endl << s_msgs << endl;

    Array<Array<int>> r_msgs(s_msgs.Size());
    ExchangePairWise_norecbuffer(comm, alldps, s_msgs, r_msgs);

    // cout << " r_msgs done " << endl;
    // for (auto k : Range(r_msgs))
    //   { cout << k << ": "; prow(r_msgs[k]); cout << endl; }
    // cout << endl;

    /** okay, now merge received stuff and local stuff **/
    Array<int> reps_loc(50); reps_loc.SetSize0(); // local reps
    Array<INT<3>> reps_rec(50); reps_rec.SetSize0(); // received reps

    auto get_loc = [&](int k, Array<int> & eq_dps) {
      eq_dps.SetSize(aloc_mems[k].Size());
      int c = 0;
      for(auto p : aloc_mems[k])
	if (p != comm.Rank())
	  { eq_dps[c++] = p; }
      eq_dps.SetSize(c);
    };
    Array<int> eq_dps1, eq_dps2;
    auto find_loc = [&](FlatArray<int> eq_dps) -> bool {
      if (!eq_dps.Size())
	{ return true; }
      for (auto rl : reps_loc) {
	get_loc(rl, eq_dps2);
	if (eq_dps == eq_dps2)
	  { return true; }
      }
      return false;
    };
    auto get_rec = [&](auto rcs) { return r_msgs[rcs[0]].Part(rcs[1], rcs[2]); };
    auto find_rec = [&](FlatArray<int> eq_dps) {
      if (!eq_dps.Size())
	{ return true; }
      for (auto rcs : reps_rec) {
	auto eq_rcs = get_rec(rcs);
	if (eq_rcs == eq_dps)
	  { return true; }
      }
      return false;
    };

    // get reps
    for (auto k : Range(aloc_mems)) {
      get_loc(k, eq_dps1);
      if (!find_loc(eq_dps1))
	{ reps_loc.Append(k); }
    }
    for (auto k : Range(r_msgs)) {
      auto & msg = r_msgs[k];
      int c = 0;
      while(c < msg.Size()) {
	int ndps = msg[c++];
	FlatArray<int> eq_dps = msg.Part(c, ndps);
	if (!find_loc(eq_dps))
	  if (!find_rec(eq_dps))
	    { reps_rec.Append(INT<3, int>({ int(k), c, ndps })); }
	bool found = find_loc(eq_dps);
	c += ndps;
      }
    }


    // create table
    // TableCreator<int> c_eqcs(1 + reps_loc.Size() + reps_rec.Size());
    // for (; !c_eqcs.Done(); c_eqcs++) {
    //   for (auto k : Range(reps_loc)) {
    // 	get_loc(reps[k], eq_dps1);
    // 	c_eqcs.Add(1+k, eq_dps1);
    //   }
    //   for (auto k : Range(reps_rec))
    // 	{ c_eqcs.Add(1+reps_loc.Size()+k, get_rec(reps_rec[k])); }
    // }
    // cout << " reps_loc = "; prow(reps_loc); cout << endl;
    // cout << " reps_rec = "; prow(reps_rec); cout << endl;
    // create table - sorted
    Array<INT<3>> all_reps;
    for (auto rl : reps_loc)
      { all_reps.Append(INT<3>({ rl, -1, -1 })); }
    all_reps.Append(reps_rec);
    // cout << " all_reps " << endl << all_reps << endl;

    QuickSort(all_reps, [&](auto tupi, auto tupj) {
	bool iloc = tupi[1] == -1, jloc = tupj[1] == -1;
	if (iloc)
	  { get_loc(tupi[0], eq_dps1); }
	if (jloc)
	  { get_loc(tupj[0], eq_dps2); }
	FlatArray<int> arri(iloc ? eq_dps1 : get_rec(tupi)),
	  arrj(jloc ? eq_dps2 : get_rec(tupj));
	return lex_smaller(arri, arrj);
      });


    TableCreator<int> c_eqcs(1 + all_reps.Size());
    for (; !c_eqcs.Done(); c_eqcs++)
      for (auto k : Range(all_reps))
	if (all_reps[k][1] == -1) {
	  get_loc(all_reps[k][0], eq_dps1);
	  c_eqcs.Add(1+k, eq_dps1);
	}
	else
	  { c_eqcs.Add(1+k, get_rec(all_reps[k])); }

    eqcs = c_eqcs.MoveTable();

    // cout << " ready for usual part, now wait " << endl;
    // comm.Barrier();
    // cout << " ok, now the usual part " << endl;

    // the usual
    if (do_cutunion)
      this->SetupFromInitialDPs(std::move(eqcs));
    else
      this->SetupFromDPs(std::move(eqcs));

    // cout << " done w. usual part, now wait " << endl;
    // comm.Barrier();
    // cout << " ok, now all done w. the usual part " << endl;

    _isRankZeroIdle = false;

    if (comm.Size())
    {
      int numOnRankZero = 0;

      if (comm.Rank() == 0)
      {
        numOnRankZero = GetNEQCS();
      }

      comm.Scatter(numOnRankZero);

      _isRankZeroIdle = numOnRankZero == 0;
    }

    _isTrulyParallel = ( comm.Size() > ( _isRankZeroIdle ? 2 : 1 ) );
  } // EQCHierarchy(alldps, ...)



  void EQCHierarchy :: SetupFromInitialDPs (Table<int> && _vanilla_dps)
  {
    static Timer t("EQCHierarchy::SetupFromInitialDPs");
    RegionTimer rt(t);

    auto vanilla_dps = std::move(_vanilla_dps);

    // cout << "vanilla_dps: " << endl << vanilla_dps << endl;

    //nr of vanilla eqcs
    size_t nveqcs = vanilla_dps.Size();
    Array<int> v_exps(50); v_exps.SetSize0(); // all neighbours
    for (auto row : vanilla_dps) {
      for (auto p : row) {
	auto ind = find_in_sorted_array(p, v_exps);
	if (ind == decltype(ind)(-1)) {
	  v_exps.Append(p); QuickSort(v_exps);
	}
      }
    }

    /** do (local) pairwise merge of dps, add eqc if there is already one that is a superset **/
    Array<Array<int> > new_dps;
    {
      static Timer tm("merge loc"); RegionTimer rt(tm);
      auto merge = [](auto & a, auto & b, auto & c)
	{
	  c.SetSize0();
	  size_t k = 0;
	  size_t j = 0;
	  while( (k<a.Size()) && (j<b.Size()) )
	    if (a[k]<b[j])
	      c.Append(a[k++]);
	    else if (a[k]>b[j])
	      c.Append(b[j++]);
	    else { c.Append(a[k++]); j++; }
	  while( k<a.Size() )
	    c.Append(a[k++]);
	  while(j<b.Size() )
	    c.Append(b[j++]);
	};
      size_t n1 = nveqcs;
      for (auto k:Range(n1))
	{
	  new_dps.Append(Array<int>(vanilla_dps[k].Size()));
	  for (auto j:Range(new_dps[k].Size()))
	    new_dps[k][j] = vanilla_dps[k][j];
	}
      Array<int> mdp(np);//merged dps
      for (size_t e1=0;e1<n1;e1++)
	for (size_t e2=0;e2<e1;e2++)
	  {
	    merge(new_dps[e1], new_dps[e2], mdp);
	    bool valid = false;
	    bool seq = true;
	    bool is_new = true;
	    for (size_t e3 = 0; (e3<n1) && is_new; e3++)
	      {
		seq = true;
		for (size_t l=0;l<mdp.Size()&&seq;l++)
		  if (!new_dps[e3].Contains(mdp[l]))
		    seq = false;
		if (seq)
		  valid = true;
		if (seq && (new_dps[e3].Size()==mdp.Size()))
		  is_new = false;
	      }
	    if (valid && is_new)
	      {
		new_dps.Append(Array<int>(mdp.Size()));
		for (auto j:Range(mdp.Size()))
		  new_dps[n1][j] = mdp[j];
		n1++;
	      }
	  }
      // cout << "new_dps: " << endl;
      // for (auto & row : new_dps) { prow(row); cout << endl; }
    }

    /** communicate newly created eqcs **/
    {
      static Timer tex("exchange"); RegionTimer rt(tex);
      int n_v_exp = v_exps.Size();
      // msg for each exp:     n_eqs | sizeeq1, eq1 | sizeeq2, eq2 | ...
      Array<int> msg_sz(n_v_exp); msg_sz = 1;
      for (int k = nveqcs; k < new_dps.Size(); k++) {
	auto& row = new_dps[k];
	for (auto p : row) {
	  auto ind = find_in_sorted_array(p, v_exps);
	  msg_sz[ind] += 1 + row.Size();
	}
      }
      Table<int> buffer(msg_sz);
      for (auto row : buffer) row[0] = 0;
      msg_sz = 1;
      for (int nneq = nveqcs; nneq < new_dps.Size(); nneq++) {
	auto& dps = new_dps[nneq];
	for (auto p : dps) {
	  auto ind = find_in_sorted_array(p, v_exps);
	  auto buf_row = buffer[ind];
	  buf_row[0]++;
	  int sz = dps.Size();
	  buf_row[msg_sz[ind]++] = sz;
	  FlatArray<int> buf_chunk(sz, &buf_row[msg_sz[ind]]);
	  buf_chunk = dps;
	  // cout << "add "; prow(buf_chunk); cout << " to " << ind << " (proc " << p << ")" << endl;
	  msg_sz[ind] += sz;
	}
      }
      // cout << "send-buf: " << endl << buffer << endl;
      Array<MPI_Request> reqs(n_v_exp);
      Array<Array<int>> rbuf(n_v_exp);
      for (auto k : Range(n_v_exp))
	{ reqs[k] = comm.ISend(buffer[k], v_exps[k], MPI_TAG_AMG); }
      for (auto k : Range(n_v_exp))
       	{ comm.Recv(rbuf[k], v_exps[k], MPI_TAG_AMG); }
      // cout << "rbuf: " << endl;
      // for (auto & row : rbuf)
      // 	{ prow(row); cout << endl; }
      MyMPI_WaitAll(reqs);


      for (auto k : Range(n_v_exp)) {
	int cnt = 1;
	auto & brow = rbuf[k];
	auto pk = v_exps[k];
	for (auto j : Range(brow[0])) {
	  int sz = brow[cnt++];
	  FlatArray<int> chunk (sz, &brow[cnt]);
	  chunk[find_in_sorted_array(comm.Rank(), chunk)] = pk;
	  QuickSort(chunk);
	  int ind = -1;
	  for (int l = 0; (l < new_dps.Size()) && (ind == -1) ; l++) {
	    if (new_dps[l] == chunk) ind = l;
	  }
	  if (ind == -1) {
	    new_dps.Append(Array<int>(chunk.Size()));
	    new_dps.Last() = chunk;
	    // cout << "added dps: "; prow(new_dps.Last()); cout << endl;
	  }
	  cnt += sz;
	}
      }
    }
    //do pairwise intersect (local operation)
    Array<int> common_dist_procs(this->np);
    bool changed = true;
    {
      static Timer ti("intersect"); RegionTimer rt(ti);
      while(changed)
	{
	  changed = false;
	  //for (auto k:Range(new_dps.Size()))
	  for (size_t k=0;k<new_dps.Size();k++)
	    {
	      //auto & rowk = new_dps[k];
	      //changed loops - range would keep old range (?)
	      //for (auto j:Range(new_dps.Size()))
	      for (size_t j=0;j<new_dps.Size();j++)
		{
		  //moved this inside bc. if resize still ref to old one!
		  auto & rowk = new_dps[k];
		  auto & rowj = new_dps[j];
		  common_dist_procs.SetSize(0);
		  for (auto p:rowk)
		    if (rowj.Contains(p))
		      common_dist_procs.Append(p);
		  QuickSort(common_dist_procs);
		  int pos = -1;
		  for (size_t l=0;l<new_dps.Size() && (pos==-1);l++)
		    if (new_dps[l]==common_dist_procs)
		      pos = l;
		  if (pos==-1) //we have a new eqc!!
		    {
		      new_dps.Append(Array<int>(common_dist_procs.Size()));
		      for (auto j:Range(common_dist_procs.Size()))
			new_dps[new_dps.Size()-1][j] = common_dist_procs[j];
		      changed = true;
		    }
		}
	    }
	}
    }

    TableCreator<int> ct1(new_dps.Size());
    for (; !ct1.Done(); ct1++) {
      for (auto k:Range(new_dps.Size()))
    	for (auto j:Range(new_dps[k].Size()))
    	  ct1.Add(k,new_dps[k][j]);
    }
    Table<int> t1 = ct1.MoveTable();

    this->SetupFromDPs(std::move(t1));
  } // end SetupFromInitialDPs


  void EQCHierarchy :: SetupFromDPs(Table<int> && anew_dps)
  {
    static Timer t("EQCHierarchy::SetupFromDPs"); RegionTimer rt(t);
    Table<int> new_dps = std::move(anew_dps);
    Array<int> perm(new_dps.Size());
    for (auto k : Range(new_dps.Size()))
      { perm[k] = k; }
    QuickSort(perm, [&new_dps](auto l, auto j) -> bool LAMBDA_INLINE {
	auto dp1 = new_dps[l]; auto dps1 = dp1.Size();
	auto dp2 = new_dps[j]; auto dps2 = dp2.Size();
    	if (dps1 > dps2) return false;
    	else if (dps1<dps2) return true;
    	else
	  for (auto k:Range(dps1)) {
	    auto p1 = dp1[k]; auto p2 = dp2[k];
	    if (p1<p2) return true; else if (p2<p1) return false;
	  }
    	return false;
      });
    auto permute = [&perm] (auto & a){
      if (!a.Size())
  	return;
      int neqcs = perm.Size();
      Array<typename std::remove_reference<decltype(a[0])>::type > a_save(neqcs);
      for (auto k:Range(neqcs))
  	a_save[k] = a[k];
      for (auto k:Range(neqcs))
  	a[k] = a_save[perm[k]];
    };

    neqcs = new_dps.Size();
    Array<int> size_of_dps(neqcs);
    for (auto k:Range(neqcs))
      size_of_dps[k] = new_dps[k].Size();
    permute(size_of_dps);
    // cout << "make dp table with " << size_of_dps.Size() << endl << size_of_dps << endl;
    dist_procs = Table<int>(size_of_dps);
    all_dist_procs = Array<int>(50);
    all_dist_procs.SetSize(0);
    for (auto k:Range(neqcs))
      size_of_dps[k]++;
    for (auto k:Range(neqcs))
      for (auto j:Range(dist_procs[k].Size())) {
  	dist_procs[k][j] = new_dps[perm[k]][j];
      }

    for (auto k:Range(neqcs))
      for (auto j:Range(dist_procs[k].Size())) {
	if (!all_dist_procs.Contains(dist_procs[k][j]))
	  all_dist_procs.Append(dist_procs[k][j]);
      }
    QuickSort(all_dist_procs);

    /**
       Get EQC-ids
       0..np-1 are local eqc-ids
       rest of blocks ordered by lowest rank
    **/
    int nblocks_loc = 0;
    for (auto k:Range(dist_procs.Size()))
      if ( (dist_procs[k].Size()) && (rank<dist_procs[k][0]) )
  	nblocks_loc++;
    Array<int> nblocks(np);
    nblocks = 0;

    MPI_Allgather(&nblocks_loc, 1, MPI_INT, nblocks.Data(), 1, MPI_INT, comm);

    int tagbase_loc = AMG_TAG_BASE + np;
    for (auto k:Range(rank))
      tagbase_loc += nblocks[k];

    Array<MPI_Request> reqs;
    eqc_ids.SetSize(dist_procs.Size());
    eqc_ids = -1;
    for (auto k:Range(dist_procs.Size()))
      if ( !dist_procs[k].Size() )
  	{ eqc_ids[k] = AMG_TAG_BASE + rank; }
      else if ( rank<dist_procs[k][0] ) {
  	  eqc_ids[k] = tagbase_loc++;
  	  for (auto p:dist_procs[k])
	    { reqs.Append(comm.ISend(FlatArray<size_t>(1, &eqc_ids[k]), p, MPI_TAG_SOLVE)); }
  	}
      else
	{ reqs.Append(comm.IRecv(FlatArray<size_t>(1, &eqc_ids[k]), dist_procs[k][0], MPI_TAG_SOLVE)); }

    MyMPI_WaitAll(reqs);

    neqcs_glob = tagbase_loc;
    for (auto k:Range(rank, np))
      { neqcs_glob += nblocks[k]; }

    /** map: ID -> index **/
    idf_2_ind.SetSize(neqcs_glob);
    idf_2_ind = -1;
    for (auto k:Range(dist_procs.Size()))
      { idf_2_ind[eqc_ids[k]] = k; }

    neqcs_glob -= AMG_TAG_BASE; // this is actually one too much usually (rank 0 has no loc eq-class)

    /** sub-comms; not needed currently!!  **/
    // MPI_Group g_ngs;
    // MPI_Comm_group(comm, &g_ngs);
    // groups.SetSize(dist_procs.Size());
    // comms.SetSize(dist_procs.Size());
    // Array<int> members(np);
    // for (auto k:Range(dist_procs.Size()))
    //   {
    // 	members.SetSize(0);
    // 	members.Append(rank);
    // 	for (auto p:dist_procs[k])
    // 	  members.Append(p);
    // 	QuickSort(members); // i doint think i need this?; actually, i do!!
    // 	MPI_Group_incl(g_ngs, members.Size(), &members[0], &groups[k]);
    // 	MPI_Comm_create_group(comm, groups[k], eqc_ids[k], &comms[k]);
    //   }

    /** EQC merging and intersection **/

    auto merge_arrays = [] (auto & a, auto & b, auto & c)
      {
  	size_t i1 = 0;
  	size_t i2 = 0;
  	c.SetSize(0);
  	while( (i1<a.Size()) && (i2<b.Size()) )
  	  if (a[i1]==b[i2])
  	    {
  	      c.Append(a[i1++]);
  	      i2++;
  	    }
  	  else if (a[i1]<b[i2])
  	    c.Append(a[i1++]);
  	  else
  	    c.Append(b[i2++]);
  	while(i1<a.Size())
  	  c.Append(a[i1++]);
  	while(i2<b.Size())
  	  c.Append(b[i2++]);
      };


    /** Generate possible merges (and set hierarchy) **/
    Array<int> arr(dist_procs.Size());
    merge.SetSize(neqcs*neqcs);
    merge = NO_EQC;
    for (auto eqc1:Range(neqcs))
      {
  	for (auto eqc2:Range(eqc1, neqcs))
  	  {
  	    merge_arrays(dist_procs[eqc1], dist_procs[eqc2], arr);
  	    auto pos = NO_EQC;
  	    for (auto k:Range(neqcs))
  	      if (dist_procs[k]==arr)
  		pos = k;
	    merge[eqc1+neqcs*eqc2] = pos;
  	    merge[eqc2+neqcs*eqc1] = pos;
  	  }
      }
    mat_merge.AssignMemory(neqcs, neqcs, merge.Data());
    hierarchic_order.SetSize(neqcs*neqcs);
    hierarchic_order.Clear();
    for (auto eqc1:Range(neqcs))
      for (auto eqc2:Range(neqcs))
	{
	  bool subset = true;
	  if ( (!dist_procs[eqc2].Size()) && dist_procs[eqc1].Size())
	    subset = false;
	  else
	    for (size_t l=0; (l<dist_procs[eqc1].Size()) && subset; l++)
	      if ( dist_procs[eqc2].Pos(dist_procs[eqc1][l])==size_t(-1))
		subset = false;
	  if (!subset)
	    continue;
	  hierarchic_order.SetBit(eqc1*neqcs+eqc2);
	  //hierarchic_order.SetBit(eqc2*neqcs+eqc1); //haha, I'm a moron...
	}

    auto intersect_arrays = [] (auto & a, auto & b, auto & c)
      {
  	size_t i1 = 0;
  	size_t i2 = 0;
  	c.SetSize(0);
  	while( (i1<a.Size()) && (i2<b.Size()) )
  	  {
  	    if (a[i1]==b[i2])
  	      {
  		c.Append(a[i1]);
  		i1++;
  		i2++;
  	      }
  	    else if (a[i1]<b[i2])
  	      i1++;
  	    else
  	      i2++;
  	  }
      };

    /** Generate possible intersections **/
    intersect.SetSize(neqcs*neqcs);
    intersect = NO_EQC;
    for (auto eqc1:Range(neqcs))
      {
  	for (auto eqc2:Range(eqc1, neqcs))
  	  {
  	    intersect_arrays(dist_procs[eqc1], dist_procs[eqc2], arr);
  	    auto pos = -1;
  	    for (auto k:Range(neqcs))
  	      if (dist_procs[k]==arr)
  		pos = k;
	    intersect[eqc1+neqcs*eqc2] = pos;
  	    intersect[eqc2+neqcs*eqc1] = pos;
  	  }
      }
    mat_intersect.AssignMemory(neqcs, neqcs, intersect.Data());

    return;
  } // end SetupFromDPs(Table<int> && new_dps)


  std::ostream & operator<<(std::ostream &os, const EQCHierarchy& eqc_h)
  {
    if (eqc_h.IsDummy())
    {
      os << "EQCHierarchy @" << &eqc_h << " on dummy-comm !" << endl;
    }
    else
    {
      os << "EQCHierarchy @" << &eqc_h << " on comm " << eqc_h.comm << ", rank " << eqc_h.rank << " of " << eqc_h.np << endl;
      os << "NEQCS: loc=" << eqc_h.neqcs << ", glob=" << eqc_h.neqcs_glob << endl;
      os << "rank "<< eqc_h.rank << " of " << eqc_h.np << endl;
      os << "rank in world : " << NgMPI_Comm(MPI_COMM_WORLD, false).Rank() << endl;
      os << "EQCS: id   || dps " << endl;
      for (auto k:Range(eqc_h.neqcs)) {
        os << k << ": " << eqc_h.eqc_ids[k] << "  || ";
        for (auto p:eqc_h.dist_procs[k])
          os << p << " ";
          os << endl;
      }
    }
    return os;
  }


} // namespace amg
