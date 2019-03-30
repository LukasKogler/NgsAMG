#ifndef FILE_REDUCE_TABLE
#define FILE_REDUCE_TABLE

#include <cxxabi.h>

namespace amg
{
  
  const int TAG_PARTABLE = MPI_TAG_AMG;    

  INLINE Timer & RTTimerHack() {
    static Timer timer("ReduceTable");
    return timer;
  }
  
  
  /**
     @brief Reduce table of T's to a table of TR's in parallel with reduction-action defined by lambda
     @details The general procedure is LINEARIZE | GATHER | REDUCE (on fake-de-linearized data) | LINEARIZE | SCATTER | DE-LINEARIZE.
     LINEARIZE means collecting lines of the matrix so only one message has to be sent to each neighbouring proc.
     In the GATHER-step rows are gathered on the proc with the lowest rank in the row's eqc.
     This allows lambda to be non-deterministic because it is only called once for each block of data. Non-deterministic
     lambdas might arise due to random numbers or, in some cases, rounding errors.
     No assumptions on the size of the reduced data is made (so e.g merging or intersecting the rows of the table is possible).
     This uses MPI_Alltoall for a couple of short messages and non-blocking MPI_Isends for all others.
     Custom MPI-Datatypes are not used (which means unnecessary copies of data).
     MPI_Neighbour-functions are not used either.	      
     If lambda is deterministic or the size of the reduced data is known in advance (e.g sorting or summing up weights),
     one of the other ReduceTable-functions might be more efficient (If I have implemented them yet ...).
     !!! NUMBER AND ORDER OF ROWS HAS TO BE CONSISTENT !!!

     @ param table_in The table to be reduced. 
     @ param eqcs Row k of table_in belongs to EQC eqcs[k]. 
     @ param eqc_h The eqc-hierarchy to be used for communication context
     @ param lambda The reduction operation to be called. 
     It has to table a table (or array of arrays, one for each proc in the corresponding eqc) of T's and return
     an array of TR's to an array. The table may be empty and the row-sizes of the table may vary.
     lambda may be non-deterministic and no assumptions on the lenth of the output are made.      
  **/
  template<class T, class TR, class LAM>
  Table<TR> ReduceTable (const Table<T> & table_in, FlatArray<size_t> eqcs, const shared_ptr<const EQCHierarchy> & eqc_h, LAM lambda)
  {
     //cout << "-------------" << "REDUCE TABLE; version 1" << "-------------" << endl;

    NgsMPI_Comm comm = eqc_h->GetCommunicator();
    int rank = comm.Rank();
    int np = comm.Size();

    // cout << "rank " << rank << " of " << np << "         waiting for rest in reduce.." << endl;
    
    // MPI_Barrier(comm);
    // double tstart = MPI_Wtime();

    auto demangle = [](const char* name) -> string {
      int status = -4;
      std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    	  };
      return string(res.get());
    };    
    string timer_prefix = "ReduceTable; T = ";
    string tname = demangle(typeid(T).name());
    timer_prefix += tname;
    static Timer tt(timer_prefix);
    // static Timer t0(timer_prefix + " - linearize1");
    // static Timer t1(timer_prefix + " - gather");
    // static Timer t2(timer_prefix + " - reduce");
    // static Timer t3(timer_prefix + " - linearize2");
    // static Timer t4(timer_prefix + " - scatter");    
    RegionTimer rt(RTTimerHack());
    RegionTimer rt1(tt);

    // cout << "reduce table; rank " << rank << " of " << np << " on comm " << comm << endl;
    
    //cout << "DP-table: " << endl << const_cast<EQCHierarchy*>(eqc_h.get())->GetDPTable() << endl;
    
    //cout << "/** Generate proc-tables - TODO: move this into eqc_h?? **/" << endl;
    /** Generate proc-tables - TODO: move this into eqc_h?? **/    
    Table<int> master_of_rows; //proc p is master of rows...
    Table<int> slave_of_rows;  //proc p is slave of this row of mine ...
    //BitArray master_of_row(table_in.Size());
    {
      //master_of_row.Clear();
      auto nrows = table_in.Size();
      TableCreator<int> cmof(np);
      TableCreator<int> csof(np);
      while (!cmof.Done()) {
	for (auto row:Range(nrows)) {
	  auto dps = eqc_h->GetDistantProcs(eqcs[row]);
	  if (!dps.Size()) {
	    cmof.Add(rank, row);
	    continue;
	  }
	  if (rank<dps[0]) { //i am master of this; others are my slaves
	    cmof.Add(rank,   row);
	    for (auto k:Range(dps.Size()))
	      csof.Add(dps[k], row);
	  } else { //p is master of this row
	    cmof.Add(dps[0], row);
	  }
	}
	cmof++; csof++;
      }
      master_of_rows = cmof.MoveTable();
      slave_of_rows = csof.MoveTable();
    }
    
    //cerr << "master_of_rows:" << endl << master_of_rows << endl;

    //cout << "/** Linearize data and Create meta-data **/ " << endl;
    /** Linearize data and Create meta-data **/
    Array<int> gather_send_total_msg_size(np);
    gather_send_total_msg_size = 0;
    Array<int> gather_send_nr_blocks(np);
    gather_send_nr_blocks = 0;
    Table<int> gather_send_block_sizes;
    Table<T> gather_send_data;
    {
      for (auto p:Range(rank)) { //to master
	gather_send_nr_blocks[p] = master_of_rows[p].Size();
	for (auto k:Range(master_of_rows[p].Size()))
	  gather_send_total_msg_size[p] += table_in[master_of_rows[p][k]].Size();
      }
      //cerr << "gather_send_nr_blocks: " << endl << gather_send_nr_blocks << endl;
      gather_send_block_sizes = Table<int>(gather_send_nr_blocks);
      gather_send_data = Table<T>(gather_send_total_msg_size);
      // for (auto row:gather_send_data)
      // 	row = -1000.0;
      for (auto p:Range(rank)) { //only linearize data for lower ranks - local data doesnt get copied
	int index = 0;
	for (auto k:Range(master_of_rows[p].Size())) {
	  auto row = table_in[master_of_rows[p][k]];
	  gather_send_block_sizes[p][k] = row.Size();
	  for (auto j:Range(row.Size())) {
	    gather_send_data[p][index++] = row[j];
	  }	      
	}
      }
    }

    // MPI_Barrier(comm);
    // t0.Stop();
    // t1.Start();

    // cout << "gather_send_total_msg_size: " << endl << gather_send_total_msg_size << endl;
    // cout << "gather_send_nr_blocks: " << endl << gather_send_nr_blocks << endl;
    // cout << "gather_send_block_sizes: " << endl << gather_send_block_sizes << endl;
    // cout << "gather_send_data: " << endl << gather_send_data << endl;

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP1!" << endl;
    // auto pf2 = timer_prefix + " - gather, part ";
    // static Timer t11(pf2 + ", #msg");
    // t11.Start();
    
    //cout << " /** Gather meta-data **/ " << endl;
    /** Gather meta-data **/
    Array<int> gather_recv_total_msg_size(np);
    // MPI_Alltoall( &(gather_send_total_msg_size[0]), 1, MPI_INT, &(gather_recv_total_msg_size[0]), 1, MPI_INT, comm);
  //cout << "gather_recv_total_msg_size: " << endl << gather_recv_total_msg_size << endl;
    
    Array<int> gather_recv_nr_blocks(np);
    // MPI_Alltoall( &(gather_send_nr_blocks[0]), 1, MPI_INT, &(gather_recv_nr_blocks[0]), 1, MPI_INT, comm);
    //cout << "gather_recv_nr_blocks: " << endl << gather_recv_nr_blocks << endl;

    //TODO: i think gather/recv_nr_blocks is already in m_of_row/s_of_row??
    gather_recv_total_msg_size = 0;
    gather_recv_nr_blocks = 0;
    Array<MPI_Request> gr1;
    for (auto p:Range(rank+1, np)) {
      if (slave_of_rows[p].Size()>0) {
	// cout << "get 2 from " << p << endl;
	gr1.Append( MyMPI_IRecv( gather_recv_total_msg_size[p], p, MPI_TAG_AMG, comm) );
	gr1.Append( MyMPI_IRecv( gather_recv_nr_blocks[p], p, MPI_TAG_AMG, comm) );
      }
    }
    for (auto p:Range(rank)) {
      if (master_of_rows[p].Size()>0) {
	// cout << "send 2 to " << p << endl;
	comm.Send( gather_send_total_msg_size[p], p, MPI_TAG_AMG);
	comm.Send( gather_send_nr_blocks[p], p, MPI_TAG_AMG);
      }
    }
    if (gr1.Size()) MyMPI_WaitAll(gr1);
    
    // t11.Stop();
    // static Timer t12(pf2 + ", make st");
    // t12.Start();

    // cout << "gather_send_total_msg_size: " << endl; prow2(gather_send_total_msg_size); cout << endl;
    // cout << "gather_send_nr_blocks: " << endl; prow2(gather_send_nr_blocks); cout << endl;
    
    auto make_displs = [] (auto & bs, auto & displs) {
      displs[0] = 0;
      for (auto k:Range(bs.Size()-1))
	displs[k+1] = displs[k] + bs[k];
    };
    
    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP2!" << endl;

    Table<int> gather_recv_block_sizes(gather_recv_nr_blocks);
    Array<int> gather_send_displs(np); //mpi needs ints...
    (void) make_displs(gather_send_nr_blocks, gather_send_displs);
    Array<int> gather_recv_displs(np);
    (void) make_displs(gather_recv_nr_blocks, gather_recv_displs);
    // use .data() instead!!
    /**
    MPI_Alltoallv( gather_send_block_sizes.Data(), &(gather_send_nr_blocks[0]), &(gather_send_displs[0]), MPI_INT,
		   gather_recv_block_sizes.Data(), &(gather_recv_nr_blocks[0]), &(gather_recv_displs[0]), MPI_INT,
		   comm);
    **/

    // t12.Stop();
    // static Timer t122(pf2 + ", send sz");
    // t122.Start();
    
    {
      bool hasblock = false;
      for (int k=rank+1;k<np&&!hasblock;k++)
	if (gather_recv_nr_blocks[k]>0) {
	  gather_recv_block_sizes.AsArray() = 0;
	  hasblock = true;
	}
    }
    Array<MPI_Request> gr2;
    for (auto p:Range(rank+1, np)) {
      if (gather_recv_nr_blocks[p]>0)
	gr2.Append( MyMPI_IRecv( gather_recv_block_sizes[p], p, MPI_TAG_AMG, comm) );
    }
    for (auto p:Range(rank)) {
      if (gather_send_nr_blocks[p]>0)
	comm.Send( gather_send_block_sizes[p], p, MPI_TAG_AMG);
    }

    if (gr2.Size()) MyMPI_WaitAll(gr2);

    // t122.Stop();
    // static Timer t13(pf2 + ", msg mem");
    // t13.Start();

    //cout << "boop" << endl;

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP3!" << endl;

    // cout << "gather_recv_block_sizes: " << endl << gather_recv_block_sizes << endl;
    
    // cout << "/** Create buffer for gathered data **/" << endl;
    /** Create buffer for gathered data **/
    Table<T> gather_recv_data(gather_recv_total_msg_size);
    //TODO: remove
    // for (auto row:gather_recv_data)
    //   row = -1000000;

    // t13.Stop();
    // static Timer t14(pf2 + ", rcv msg");
    // t14.Start();
    
    //cout << " /** Gather data **/ " << endl;
    Array<MPI_Request> gather_reqs;
    //send to master
    for (auto p:Range(rank)) {
      if (gather_send_total_msg_size[p]>0)
	gather_reqs.Append( MyMPI_ISend( gather_send_data[p], p, MPI_TAG_AMG, comm) );
    }
    for (auto p:Range(rank+1, np)) {
      if (gather_recv_total_msg_size[p]>0)
	gather_reqs.Append( MyMPI_IRecv( gather_recv_data[p], p, MPI_TAG_AMG, comm) );
    }
    //copy data I am master of
    for (auto k:Range(gather_recv_data[rank].Size()))
      gather_recv_data[rank][k] = gather_send_data[rank][k];
    if (gather_reqs.Size()) MyMPI_WaitAll(gather_reqs);
    
    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP3.5!" << endl;
    
    // t14.Stop();

    // MPI_Barrier(comm);
    // t1.Stop();
    // t2.Start();

    // cout << "gather_recv_data: " << endl << gather_recv_data << endl;
    
    //cout << "/** Pick out data for each row of orig. tab **/" << endl;
    /** Pick out data for each row of orig. tab and call lambda **/

    Array<int> unreduced_data_sizes(table_in.Size());
    if (table_in.Size()) {
      unreduced_data_sizes = 0;
      for (auto k:Range(table_in.Size())) {
	auto dps = eqc_h->GetDistantProcs(eqcs[k]);
	if ( !(dps.Size() && rank>dps[0]) ) //I am the only one or master 
	  unreduced_data_sizes[k] =  dps.Size() +  1;
      }    
    }
    // cout << "unreduced_data_sizes: " << endl << unreduced_data_sizes << endl;
    Table<FlatArray<T> > unreduced_data(unreduced_data_sizes);

    // cout << "have table" << endl;
    // MPI_Barrier(comm);
    // cout << "have table" << endl;

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP3.6!" << endl;


    {
      Array<int> s(np);
      s = 0;
      Array<int> displ(np);
      displ = 0;
      for (auto k:Range(table_in.Size())) {
	auto dps = eqc_h->GetDistantProcs(eqcs[k]);
	if (dps.Size() && rank>dps[0]) //I am slave of this row!!
	  continue;
	// local data

	// auto all_procs = eqc_h->GetAllProcs(eqcs[k]);
	// auto col = all_procs.Pos(rank);

	Array<int> all_procs(dps.Size()+1);
	all_procs.SetSize(0);
	all_procs.Append(rank);
	for (auto p:dps)
	  all_procs.Append(p);
	QuickSort(all_procs);
	auto col = all_procs.Pos(rank);
       
	//cout << "row " << k << " , dps " << endl << eqc_h->GetDistantProcs(eqcs[k]) << endl << "dist procs: " << endl << eqc_h->GetAllProcs(eqcs[k]) << endl << "col " << col << endl;
	auto & a = unreduced_data[k][col];
	auto b = table_in[k];
	//cout << "b is " << endl << b << endl;
	//cout << "b size: " << b.Size() << endl;
	a.Assign(b);      
	if (!dps.Size()) // only local data!! 
	  continue;      
	// dist data
	for (auto p:dps) {
	  auto bs = gather_recv_block_sizes[p][s[p]++];
	  unreduced_data[k][all_procs.Pos(p)].Assign(FlatArray<T>(bs, &(gather_recv_data[p][displ[p]])));
	  displ[p] += bs;
	}
      }
    }

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP3.7!" << endl;
    
    // cout << "unreduced data: " << endl;
    // for (auto row:Range(table_in.Size()))
    //   {
    // 	cout << row << ": ";
    // 	for (auto bn:Range(unreduced_data[row].Size())) {
    // 	  for (auto j:unreduced_data[row][bn])
    // 	    cout << j << " ";
    // 	  cout << " || ";
    // 	}
    // 	cout << endl;
    //   }
    // cout << endl << endl;
    
    
    //cout << "/** Call lambda on gathered data **/" << endl;
    /** Call lambda on gathered data **/
    Array<Array<TR> > reduced_data(table_in.Size());
    for (auto row:Range(table_in.Size()))
      if (unreduced_data[row].Size()) {
    	//reduced_data[row] = std::move( lam(unreduced_data[row]) );
    	auto r = lambda(unreduced_data[row]);
    	reduced_data[row] = std::move(r);
      }

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP3.8!" << endl;

    // MPI_Barrier(comm);
    // t2.Stop();
    // t3.Start();
    
    //cout << "slave_of_rows:" << endl << slave_of_rows << endl;
    //cout << "/** Linerize reduced data, create meta-data  **/" << endl;
    /** Linerize reduced data, create meta-data  **/
    Array<int> scatter_send_total_msg_size(np);    
    scatter_send_total_msg_size = 0;
    Array<int> scatter_send_nr_blocks(np);
    scatter_send_nr_blocks = 0;
    Table<int> scatter_send_block_sizes;
    Table<TR> scatter_send_data;
    for (auto p:Range(rank+1, np)) {
      scatter_send_nr_blocks[p] = slave_of_rows[p].Size();
    }
    //cout << "scatter_send_nr_blocks:" << endl << scatter_send_nr_blocks << endl;
    scatter_send_block_sizes = Table<int>(scatter_send_nr_blocks);
    for (auto p:Range(rank+1, np)) {
      for (auto k:Range(slave_of_rows[p].Size())) {
	auto block_size = reduced_data[slave_of_rows[p][k]].Size();
	scatter_send_total_msg_size[p] += (scatter_send_block_sizes[p][k] = block_size );
      }
    }
    scatter_send_data = Table<TR>(scatter_send_total_msg_size);
    for (auto p:Range(rank+1, np)) {
      int displ = 0;
      for (auto k:Range(slave_of_rows[p].Size())) {
	int block_size = reduced_data[slave_of_rows[p][k]].Size();
	FlatArray<TR>(block_size, &(scatter_send_data[p][displ])) = reduced_data[slave_of_rows[p][k]];
	displ += block_size;
      }
    }

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP4!" << endl;

    // cout << "scatter_send_total_msg_size" << endl << scatter_send_total_msg_size << endl;
    // cout << "scatter_send_nr_blocks" << endl << scatter_send_nr_blocks << endl;
    // cout << "scatter_send_block_sizes" << endl << scatter_send_block_sizes << endl;
    // cout << "scatter_send_data:" << endl << scatter_send_data << endl;

    
    // MPI_Barrier(comm);
    // t3.Stop();
    // t4.Start();
    

    //cout << "/** Scatter meta-data **/" << endl;
    /** Scatter meta-data **/
    Array<int> scatter_recv_total_msg_size(np);
    // MPI_Alltoall(&(scatter_send_total_msg_size[0]), 1, MPI_INT, &(scatter_recv_total_msg_size[0]), 1, MPI_INT, comm);
    //cout << "scatter_recv_total_msg_size: " << endl << scatter_recv_total_msg_size << endl;
    Array<int> scatter_recv_nr_blocks(np);
    // MPI_Alltoall(&(scatter_send_nr_blocks[0]), 1, MPI_INT, &(scatter_recv_nr_blocks[0]), 1, MPI_INT, comm);
    //cout << "scatter_recv_nr_blocks:" << endl << scatter_recv_nr_blocks << endl;

    // cout << "rank np: " << rank << " " << np << endl;
    
    scatter_recv_total_msg_size = 0;
    scatter_recv_nr_blocks = 0;
    Array<MPI_Request> gr3;
    // for (auto p:Range(1,rank)) {
    for (int p=0;p<rank;p++) {
      // cout << "get 2x from " << p << endl; 
      if (master_of_rows[p].Size()>0) {
	gr3.Append( MyMPI_IRecv( scatter_recv_total_msg_size[p], p, MPI_TAG_AMG, comm) );
	gr3.Append( MyMPI_IRecv( scatter_recv_nr_blocks[p], p, MPI_TAG_AMG, comm) );
      }
    }
    for (auto p:Range(rank+1, np)) {
      // cout << "send 2x to " << p << endl; 
      if (slave_of_rows[p].Size()>0) {
	comm.Send( scatter_send_total_msg_size[p], p, MPI_TAG_AMG);
	comm.Send( scatter_send_nr_blocks[p], p, MPI_TAG_AMG);
      }
    }
    // cout << "wait for " << gr3.Size() << "msgs.." << endl;
    if (gr3.Size()) MyMPI_WaitAll(gr3);
    
    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP4.5!" << endl;

    // for (auto k:Range(np)) {
    //   if (scatter_recv_total_msg_size[k]>0 ||
    // 	 scatter_recv_nr_blocks[k]>0)
    // 	cout << "from " << k << " recv total " << scatter_recv_total_msg_size[k] <<
    // 	  " in " << scatter_recv_nr_blocks[k] << " blocks " << endl;
    // }
    // cout << endl;
    
    Table<int> scatter_recv_block_sizes(scatter_recv_nr_blocks);
    Array<int> scatter_send_displs(np);
    (void) make_displs(scatter_send_nr_blocks, scatter_send_displs);
    Array<int> scatter_recv_displs(np);
    (void) make_displs(scatter_recv_nr_blocks, scatter_recv_displs);
    /**
    MPI_Alltoallv( scatter_send_block_sizes.Data(), &(scatter_send_nr_blocks[0]), &(scatter_send_displs[0]), MPI_INT,
		   scatter_recv_block_sizes.Data(), &(scatter_recv_nr_blocks[0]), &(scatter_recv_displs[0]), MPI_INT,
		   comm);
    **/
    //cout << "scatter_recv_block_sizes" << endl << scatter_recv_block_sizes << endl;

    {
      bool hasblock = false;
      for (int k=0;k<rank&&!hasblock;k++) {
	if ( scatter_recv_nr_blocks[k] > 0 ) {
	  // cout << "get from " << k << ", set 0!" << endl;
	  scatter_recv_block_sizes.AsArray() = 0;
	  hasblock = true;
	}
      }
    }

    Array<MPI_Request> gr4;
    for (auto p:Range(rank)) {
      if (scatter_recv_total_msg_size[p]) {
	// cout << "get BS from " << p << endl;
	gr4.Append( MyMPI_IRecv( scatter_recv_block_sizes[p], p, MPI_TAG_AMG, comm) );
      }
    }
    for (auto p:Range(rank+1, np)) {
      if (scatter_send_total_msg_size[p]) {
	// cout << "send to " << p << endl;
	comm.Send( scatter_send_block_sizes[p], p, MPI_TAG_AMG);
      }
    }
    if (gr4.Size()) MyMPI_WaitAll(gr4);
    
    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP5!" << endl;
    
    Table<TR> scatter_recv_data(scatter_recv_total_msg_size);

    // cout << "GATHER DATA SIZES R | S:" << endl;
    // if (rank)
    //   for (auto k:Range(np))
    // 	{
    // 	  cout << k << ": " << gather_recv_data[k].Size() << " | " << gather_send_data[k].Size() << endl;
    // 	}						    
    // cout << "SCATTER DATA SIZES R | S:" << endl;
    // if (rank)
    //   for (auto k:Range(np))
    // 	{
    // 	  cout << k << ": " << scatter_recv_data[k].Size() << " | " << scatter_send_data[k].Size() << endl;
    // 	}						
    
    //cout << "/** Scatter data **/" << endl;
    /** Scatter data **/
    Array<MPI_Request> req_scatter_data;
    for (auto p:Range(rank+1, np)) {
      //cout << "send " << scatter_send_data[p].Size() << " to " << p << endl;
      if (scatter_send_total_msg_size[p]>0) {
	req_scatter_data.Append( MyMPI_ISend(scatter_send_data[p], p, MPI_TAG_AMG, comm) );
      }
    }
    // for (auto p:Range(1, rank)) {
    for (auto p=0;p<rank;p++) {
      //cout << "recv " << scatter_recv_data[p].Size() << " from " << p << endl;
      if (scatter_recv_total_msg_size[p]>0) {
	req_scatter_data.Append( MyMPI_IRecv(scatter_recv_data[p], p, MPI_TAG_AMG, comm) );
      }
    }
    //cout << "wait" << endl;
    if (req_scatter_data.Size()) MyMPI_WaitAll(req_scatter_data);

    // if (ASC_AMG_OUTPUT_LEVEL>=STATUS_UPDATES_ALL)
    //   cout << "TAB PIEP6!" << endl;

    //cout << "/** Pick apart scattered data and copy local reduced data **/" << endl;
    /** Pick apart scattered data and copy local reduced data **/
    Table<TR> reduced_table;
    Array<int> reduced_table_sizes(table_in.Size());
    reduced_table_sizes = -1;
    for (auto p:Range(rank)) { //recvd data
      for (auto k:Range(master_of_rows[p].Size()))
	reduced_table_sizes[master_of_rows[p][k]] = scatter_recv_block_sizes[p][k];
    }
    
    // cout << "OWN MASTER: " << endl << master_of_rows[rank] << endl;
    for (auto k:Range(master_of_rows[rank].Size())) { //local reduced data
      reduced_table_sizes[master_of_rows[rank][k]] = reduced_data[master_of_rows[rank][k]].Size();
    }

    reduced_table = Table<TR>(reduced_table_sizes);      
    for (auto p:Range(rank)) { //copy recvd data
      size_t displ = 0;
      for (auto k:Range(master_of_rows[p].Size())) {
	auto bs = scatter_recv_block_sizes[p][k];
	reduced_table[master_of_rows[p][k]] = FlatArray<TR>(bs, &(scatter_recv_data[p][displ]));
	displ += bs;
      }
    }
    for (auto k:Range(master_of_rows[rank].Size())) { //copy local reduced data
      reduced_table[master_of_rows[rank][k]] = reduced_data[master_of_rows[rank][k]];
    }

    // cout << "TAB OUT: " << endl;
    // for (auto k:Range(reduced_table.Size())) {
    //   cout << k << ": ";
    //   for (auto j:Range(reduced_table[k].Size()))
    //     cout << reduced_table[k][j] << " ";
    //   cout << endl;
    // }
    // cout << endl;
    
    return reduced_table;
  }

  
  template<class T, class TR, class LAM>
  Table<TR> ReduceTable (const Table<T> & table_in, const shared_ptr<const EQCHierarchy> & eqc_h, LAM lambda)
  {
    Array<size_t> eqcs(eqc_h->GetNEQCS());
    for (auto k:Range(eqcs.Size())) eqcs[k] = k;
    return ReduceTable<T, TR>(table_in, eqcs, eqc_h, lambda);
  }


  /**
     Input: eqc-wise table, whith one row per eqc. master has all data for eqch eqc 
     Output: table where everyone has all rows.
     if has_mem==true, does not allocate a new table for output
      
     TODO: give option not to copy local data
  **/
  template<class T> Table<T> ScatterEqcData (Table<T> && table_in, const EQCHierarchy & eqc_h, bool has_mem = false)
  {
    auto table = move(table_in);
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();
    auto ex_procs = eqc_h.GetDistantProcs();
    int n_smaller = 0; for (auto p : ex_procs) if(comm.Rank()>p) n_smaller++;
    // int n_larger = ex_procs.Size() - n_smaller;
    // block_sizes, block_data
    Array<int> perow(ex_procs.Size()); perow = 0;
    Array<int> nblocks(ex_procs.Size()); nblocks = 0;
    for ( auto eqc : Range(neqcs)) {
      if (eqc_h.IsMasterOfEQC(eqc)) {
	auto eq_dps = eqc_h.GetDistantProcs(eqc);
	// eq_dps[pos:] > comm.Rank() -> need to send
	auto row = table[eqc];
	for ( int kp = 0; kp < eq_dps.Size(); kp++) {
	  auto posk = find_in_sorted_array(eq_dps[kp], ex_procs);
	  perow[posk] += row.Size() + 1;
	  nblocks[posk]++;
	}
      }
    }
    Table<T> send_buffers(perow); perow = 0;
    Array<int> count_blocks(ex_procs.Size()); count_blocks = 0;
    for ( auto eqc : Range(neqcs)) {
      if (eqc_h.IsMasterOfEQC(eqc)) {
	auto eq_dps = eqc_h.GetDistantProcs(eqc);
	auto row = table[eqc];
	for ( int kp = 0; kp < eq_dps.Size(); kp++) {
	  auto posk = find_in_sorted_array(eq_dps[kp], ex_procs);
	  send_buffers[posk][count_blocks[posk]++] = row.Size();
	  for (auto l : Range(row.Size()))
	    send_buffers[posk][nblocks[posk] + perow[posk]++] = row[l];
	}
      }
    }
    
    cout << " send data: " << endl;
    for (auto kp : Range(ex_procs.Size())) {
      cout << " to " << kp << " rk " << ex_procs[kp] << ": ";
      prow2(send_buffers[kp]); cout << endl;
    }
    cout << endl;

    // TODO: this sends a couple of empty messages from ranks that are not master of anything
    for (int kp = n_smaller; kp < ex_procs.Size(); kp++) {
      auto req = MyMPI_ISend(send_buffers[kp], ex_procs[kp], MPI_TAG_AMG, comm);
      MPI_Request_free(&req);
    }

    // TODO: build buffer->row maps, then waitiany if this is critical
    Array<Array<T>> recv_buffers(ex_procs.Size());
    for (int kp = 0; kp < n_smaller; kp++) {
      Array<T> & rb = recv_buffers[kp];
      comm.Recv(rb, ex_procs[kp], MPI_TAG_AMG);
      // comm.Recv(recv_buffers[kp], ex_procs[kp], MPI_TAG_AMG);
    }

    cout << " got data: " << endl;
    for (auto kp : Range(ex_procs.Size())) {
      cout << " from " << kp << " rk " << ex_procs[kp] << ": ";
      prow2(recv_buffers[kp]); cout << endl;
    }
    cout << endl;
    
    Table<T> table_out;
    
    if ( has_mem == false ) { // alloc memory for output
      perow.SetSize(neqcs); perow = 0; count_blocks = 0;
      for ( auto eqc : Range(neqcs)) {
	if (eqc_h.IsMasterOfEQC(eqc)) {
	  perow[eqc] = table[eqc].Size();
	}
	else {
	  auto eq_dps = eqc_h.GetDistantProcs(eqc);
	  int posk = find_in_sorted_array(eq_dps[0], ex_procs);
	  perow[eqc] += recv_buffers[posk][count_blocks[posk]++];
	}
      }
      table_out = Table<T>(perow);
    }
    else {
      table_out = move(table);
    }

    nblocks = count_blocks; perow = 0; count_blocks = 0;
    for ( auto eqc : Range(neqcs)) {
      if (eqc_h.IsMasterOfEQC(eqc) && (has_mem == false) ) {
	table_out[eqc] = table[eqc]; // data copy
      }
      else {
	auto eq_dps = eqc_h.GetDistantProcs(eqc);
	int posk = find_in_sorted_array(eq_dps[0], ex_procs);
	auto & buffer = recv_buffers[posk];
	auto outrow = table_out[eqc];
	auto bsize = buffer[count_blocks[posk]++];
	if (bsize != outrow.Size()) cout << " MISFIT " << eqc << " " << posk << " " << bsize << " " << outrow.Size() << endl;
	int offset = nblocks[posk];
	for ( auto l : Range(bsize) )
	  outrow[l] = buffer[offset + perow[posk]++];
      }
    }

    return move(table_out);
  }


} // end namsepace asc_amg_h1par

#endif
