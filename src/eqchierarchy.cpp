#include "amg.hpp"

namespace amg {

  NgsAMG_Comm make_me_comm () {
    NgMPI_Comm wcomm(MPI_COMM_WORLD);
    if (wcomm.Size() == 1)
      { return NgsAMG_Comm(wcomm); }
    else {
      netgen::NgArray<int> me(1); me[0] = wcomm.Rank();
      return NgsAMG_Comm(netgen::MyMPI_SubCommunicator(wcomm, me ), true);
    }
  }

  static NgsAMG_Comm mecomm (make_me_comm());
  NgsAMG_Comm AMG_ME_COMM = mecomm;

  EQCHierarchy :: EQCHierarchy (const shared_ptr<MeshAccess> & ma, Array<NODE_TYPE> nts, bool do_cutunion)
    : comm(ma->GetCommunicator())
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


  EQCHierarchy :: EQCHierarchy (const shared_ptr<ParallelDofs> & apd,
				bool do_cutunion, size_t _max_nd)
    : comm(apd->GetCommunicator())
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
	for (auto k:Range(max_nd)) {
	  auto dp = apd->GetDistantProcs(k);
	  if (!dp.Size()) continue; 
	  int pos = -1;
	  for (size_t l = 1; l<index_of_block.Size() && (pos == -1); l++)
	    if (apd->GetDistantProcs(index_of_block[l]) == dp)
	      { pos = l; }
	  if (pos>=0) continue;
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

    if (do_cutunion)
      this->SetupFromInitialDPs(std::move(vanilla_dps));
    else
      this->SetupFromDPs(std::move(vanilla_dps));
      
    return;
  } // end EQCHierarchy(pardofs,...)
  

  EQCHierarchy :: EQCHierarchy (Table<int> && eqcs, NgsAMG_Comm acomm, bool do_cutunion)
    : comm(acomm)
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

  void EQCHierarchy :: SetupFromInitialDPs (Table<int> && _vanilla_dps)
  {
    static Timer t("EQCHierarchy::SetupFromInitialDPs");
    RegionTimer rt(t);

    auto vanilla_dps = move(_vanilla_dps);

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
	{ reqs[k] = MyMPI_ISend(buffer[k], v_exps[k], MPI_TAG_AMG, comm); }
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
    Table<int> new_dps = move(anew_dps);
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

    MPI_Allgather(&nblocks_loc, 1, MPI_INT, &nblocks[0], 1, MPI_INT, comm);

    int tagbase_loc = AMG_TAG_BASE + np;
    for (auto k:Range(rank))
      tagbase_loc += nblocks[k];
    
    Array<MPI_Request> reqs;
    eqc_ids.SetSize(dist_procs.Size());
    eqc_ids = -1;
    for (auto k:Range(dist_procs.Size()))
      if ( !dist_procs[k].Size() )
  	eqc_ids[k] = AMG_TAG_BASE + rank;
      else if ( rank<dist_procs[k][0] )
  	{
  	  eqc_ids[k] = tagbase_loc++;
  	  for (auto p:dist_procs[k]) {
  	    reqs.Append(MyMPI_ISend(FlatArray<size_t>(1, &eqc_ids[k]), p, MPI_TAG_SOLVE, comm));
	  }
  	}
      else {
	reqs.Append(MyMPI_IRecv(FlatArray<size_t>(1, &eqc_ids[k]), dist_procs[k][0], MPI_TAG_SOLVE, comm));
      }
    
    MyMPI_WaitAll(reqs);    

    neqcs_glob = tagbase_loc;
    for (auto k:Range(rank, np))
      neqcs_glob += nblocks[k];

    /** map: ID -> index **/
    idf_2_ind.SetSize(neqcs_glob);
    idf_2_ind = -1;
    for (auto k:Range(dist_procs.Size()))
      idf_2_ind[eqc_ids[k]] = k;

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
    mat_merge.AssignMemory(neqcs, neqcs, &merge[0]);
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
	  hierarchic_order.Set(eqc1*neqcs+eqc2);
	  //hierarchic_order.Set(eqc2*neqcs+eqc1); //haha, I'm a moron...
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
    mat_intersect.AssignMemory(neqcs, neqcs, &intersect[0]);

    return;
  } // end SetupFromDPs(Table<int> && new_dps)


  std::ostream & operator<<(std::ostream &os, const EQCHierarchy& eqc_h)
  {
    os << "EQCHierarchy on comm " << eqc_h.comm << ", rank " << eqc_h.rank << " of " << eqc_h.np << endl;
    os << "NEQCS: loc=" << eqc_h.neqcs << ", glob=" << eqc_h.neqcs_glob << endl;
    os << "rank "<< eqc_h.rank << " of " << eqc_h.np << endl;
    os << "rank in world : " << NgMPI_Comm(MPI_COMM_WORLD).Rank() << endl;
    os << "EQCS: id   || dps " << endl;
    for (auto k:Range(eqc_h.neqcs)) {
      os << k << ": " << eqc_h.eqc_ids[k] << "  || ";
      for (auto p:eqc_h.dist_procs[k])
	os << p << " ";
      os << endl;
    }
    return os;
  }
  
  
} // namespace amg
