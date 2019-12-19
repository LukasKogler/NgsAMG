
namespace amg
{
  /** AgglomerateCoarseMap **/

  template<class TMESH>
  AgglomerateCoarseMap<TMESH> :: AgglomerateCoarseMap (shared_ptr<TMESH> _mesh)
    : BaseCoarseMap(), GridMapStep<TMESH>(_mesh)
  {
    assert(mesh != nullptr); // obviously this would be bad
    for (NODE_TYPE NT : {NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL} )
      { NN[NT] = mapped_NN[NT] = 0; }
    NN[NT_VERTEX] = mesh->template GetNN<NT_VERTEX>();
    NN[NT_EDGE] = mesh->template GetNN<NT_EDGE>();
  } // AgglomerateCoarseMap ()


  template<class TMESH>
  shared_ptr<TopologicMesh> AgglomerateCoarseMap<TMESH> :: GetMappedMesh () const
  {
    if (mapped_mesh == nullptr)
      { const_cast<AgglomerateCoarseMap<TMESH>&>(*this).BuildMappedMesh(); }
    return mapped_mesh;
  } // AgglomerateCoarseMap::GetMappedMesh


  template<class TMESH>
  void AgglomerateCoarseMap<TMESH> :: BuildMappedMesh ()
  {
    static Timer t("AgglomerateCoarseMap"); RegionTimer rt(t);

    mesh->CumulateData();

    Array<Agglomerate> aggs;
    Array<int> v_to_agg;
    FormAgglomerates(aggs, v_to_agg);

    is_center = make_shared<BitArray>(mesh->template GetNN<NT_VERTEX>());
    is_center->Clear();
    for (const auto & agg : aggs)
      { is_center->SetBit(agg.center()); }

    auto mapped_btm = new BlockTM(mesh->GetEQCHierarchy());

    mapped_btm->has_nodes[NT_VERTEX] = mapped_btm->has_nodes[NT_EDGE] = true;
    mapped_btm->has_nodes[NT_FACE] = mapped_btm->has_nodes[NT_CELL] = false;

    MapVerts(*mapped_btm, aggs, v_to_agg);

    MapEdges(*mapped_btm, aggs, v_to_agg);

    if constexpr(std::is_same<TMESH, BlockTM>::value == 1) {
	mapped_mesh = shared_ptr<BlockTM>(mapped_btm);
      }
    else {
      // auto scd = mesh->MapData(*this);
      auto cdata = mesh->AllocMappedData(*this);
      mapped_mesh = make_shared<TMESH> ( move(*mapped_btm), cdata );
      mesh->MapDataNoAlloc(*this);
      // get<0>(mapped_mesh->Data())->Cumulate();
      // get<1>(mapped_mesh->Data())->Cumulate();
    }


  } // AgglomerateCoarseMap::BuildMappedMesh


  template<class TMESH>
  void AgglomerateCoarseMap<TMESH> :: MapVerts (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg)
  {
    static Timer t("MapVerts"); RegionTimer rt(t);

    /**
       We need:
          I) No non-master vertex is in any agglomerate I have.
	 II) For each member v of an agg with center c:
	            i) eqc(v) <= eqc(c)
		   ii) I am master of c (and therefore also of v)
	     [So no agglomerates that cross process boundaries]
	     -> each coarse vertex, defined by an agglomerate is in the eqc of the (fine) center vertex v of that agglomerate
     **/

    const auto & M = *mesh;
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();

    // cout << " EQCH: " << endl << eqc_h << endl;
    // cout << " fmesh NV " << M.nnodes[NT_VERTEX] << " " << M.nnodes_glob[NT_VERTEX] << endl;
    // cout << " fmesh eqc_verts " << endl << M.eqc_verts << endl;
    // cout << " agglomerates: " << endl << agglomerates << endl;
    // cout << " v_to_agg: " << endl; prow2(v_to_agg); cout << endl;

    auto& vmap = node_maps[0];
    vmap.SetSize(M.template GetNN<NT_VERTEX>()); vmap = -1;

    /** EX-data: [crs_cnt, Nsame, Nchange, (locvmap, eqid)-array 
	crs_cnt .. #of verts in eqc on coarse level
	Nsame,Nchange .. #of verts that stay in this eqc / that change eqc **/
    Array<int> cnt_vs(neqcs); cnt_vs = 0;

    Array<int> agg_map(agglomerates.Size()); agg_map = -1;
    Array<int> agg_eqc(agglomerates.Size()); agg_eqc = -1;

    /** **/
    Array<int> exds(neqcs); exds = 0;
    Table<int> exdata;
    if (neqcs > 1)
      for (auto k : Range(size_t(1), neqcs))
	{ exds[k] = 3 + 2 * M.template GetENN<NT_VERTEX>(k); }
    exdata = Table<int> (exds);
    if (neqcs > 1) {
      for (auto eqc : Range(size_t(1), neqcs)) {
	if (eqc_h.IsMasterOfEQC(eqc)) {
	  auto exrow = exdata[eqc];
	  exrow[0] = 0;
	  auto & cnt_same = exrow[1]; cnt_same = 0;
	  auto & cnt_change = exrow[2]; cnt_change = 0;
	  auto & loc_map = exrow.Part(3); // cout << " loc map s " << loc_map.Size() << endl;
	  auto eqc_verts = M.template GetENodes<NT_VERTEX>(eqc);
	  for (auto j : Range(eqc_verts)) {
	    auto v = eqc_verts[j];
	    auto agg_nr = v_to_agg[v];
	    // cout << "eqc j v agg_nr " << eqc << " " << j << " " << v << " " << agg_nr << endl;
	    if (agg_nr == -1) // dirichlet
	      { loc_map[2*j] = -1; }
	    else { // not dirichlet
	      auto & cid = agg_map[agg_nr];
	      auto & ceq = agg_eqc[agg_nr];
	      if (agg_map[agg_nr] == -1) {
		auto & agg = agglomerates[agg_nr];
		// cout << "agg: " << agg << endl;
		ceq = M.template GetEqcOfNode<NT_VERTEX>(agg.center());
		agg_map[agg_nr] = cnt_vs[ceq]++;
	      }
	      // cout << " ceq " << ceq << " cid " << cid << endl;
	      loc_map[2*j]   = cid;
	      loc_map[2*j+1] = eqc_h.GetEQCID(ceq);
	      if (eqc == ceq) { cnt_same++; }
	      else { cnt_change++; }
	    }
	  }
	}
      }
      for (auto eqc : Range(size_t(1), neqcs))
	{ exdata[eqc][0] = cnt_vs[eqc]; }
    }

    // cout << "(unred) exdata: " << endl << exdata << endl;

    auto reqs = eqc_h.ScatterEQCData(exdata);

    /** map agglomerates that do not touch a subdomain boundary **/
    for (auto & agg : agglomerates) { // per definition I am master of all verts in this agg
      auto eqc = M.template GetEqcOfNode<NT_VERTEX>(agg.center());
      if (eqc == 0) { // ALL mems must be local!
	auto cid = cnt_vs[0]++;
	for (auto m : agg.members())
	  { vmap[m] = cid; }
      }
    }

    /** finish EQC scatter, make displ-array **/
    MyMPI_WaitAll(reqs);

    // cout << "(red)   exdata: " << endl << exdata << endl;

    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs))
	{ cnt_vs[eqc] = exdata[eqc][0]; }
    Array<int> disps(1+neqcs); disps = 0;
    for (auto k : Range(neqcs))
      { disps[k+1] = disps[k] + cnt_vs[k]; }

    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs)) {
	auto exrow = exdata[eqc];
	auto & cnt_same = exrow[1];
	auto & cnt_change = exrow[2];
	auto & loc_map = exrow.Part(3);
	auto eqc_verts = M.template GetENodes<NT_VERTEX>(eqc);
	for (auto j : Range(eqc_verts)) {
	  auto v = eqc_verts[j];
	  auto cid_loc = loc_map[2*j];
	  if (cid_loc != -1) { // dirichlet, once again!
	    // cout << v << ", " << j << " in " << eqc << ", locmap " << loc_map[2*j] << flush;
	    // cout << " " << loc_map[2*j+1] << flush;
	    int ceq = eqc_h.GetEQCOfID(loc_map[2*j+1]);
	    // cout << ", ceq " << ceq;
	    auto cid = disps[ceq] + cid_loc;
	    // cout << ", disp " << disps[ceq] << " -> " << cid << endl;
	    vmap[v] = cid;
	    auto agg_nr = v_to_agg[v];
	    if (agg_nr != -1) { // a local agg, but might also contain verts from eqc 0!
	      auto & agg = agglomerates[agg_nr];
	      if (agg.center() == v) {
		for (auto mem : agg.members())
		  { vmap[mem] = cid; }
	      }
	    }
	  }
	}
      }
    

    mapped_NN[NT_VERTEX] = disps.Last();
    cmesh.nnodes[NT_VERTEX] = disps.Last();
    cmesh.verts.SetSize(cmesh.nnodes[NT_VERTEX]);
    auto & cverts = cmesh.verts;
    for (auto k : Range(cmesh.nnodes[NT_VERTEX]) )
      { cverts[k] = k; }
    auto & disp_veq = cmesh.disp_eqc[NT_VERTEX];
    disp_veq = move(disps);
    cmesh.nnodes_eqc[NT_VERTEX] = move(cnt_vs);
    cmesh.nnodes_cross[NT_VERTEX].SetSize(neqcs); cmesh.nnodes_cross[NT_VERTEX] = 0;
    cmesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, cmesh.disp_eqc[NT_VERTEX].Data(), cmesh.verts.Data());
    auto ncv_master = 0;
    for (auto eqc : Range(neqcs))
      if (eqc_h.IsMasterOfEQC(eqc))
	{ ncv_master += cmesh.GetENN<NT_VERTEX>(eqc); }
    cmesh.nnodes_glob[NT_VERTEX] = comm.AllReduce(ncv_master, MPI_SUM);

    // cout << " fmesh NV " << M.nnodes[NT_VERTEX] << " " << M.nnodes_glob[NT_VERTEX] << endl;
    // cout << " fmesh eqc_verts " << endl << M.eqc_verts << endl;

    // cout << " vmap: " << endl; prow2(vmap); cout << endl;
    // cout << " EQCH: " << endl << eqc_h << endl;
    // cout << " cmesh NV " << cmesh.nnodes[NT_VERTEX] << " " << cmesh.nnodes_glob[NT_VERTEX] << endl;
    // cout << " cmesh eqc_verts " << endl << cmesh.eqc_verts << endl;

    // {
    //   Array<int> cnt(cmesh.template GetNN<NT_VERTEX>()); cnt = 1;
    //   cmesh.template AllreduceNodalData<NT_VERTEX>(cnt, [&](auto in) { return sum_table(in); });
    //   cout << " vertex test cnt: " << endl;
    //   prow2(cnt); cout << endl;
    // }

  } // AgglomerateCoarseMap::MapVerts


  template<class TMESH>
  void AgglomerateCoarseMap<TMESH> :: MapVerts2 (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg)
  {
    static Timer t("MapVerts"); RegionTimer rt(t);

    /** This version works only for purely in-eqc aggs (and even then I am not 100% sure that it is bug free) **/

    const auto & M = *mesh;
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();
    
    auto& vmap = node_maps[0];
    vmap.SetSize(M.template GetNN<NT_VERTEX>()); vmap = -1;
    
    // start agglomerate scatter
    size_t n_agg_nonloc = 0;
    Array<int> exds(neqcs); // [n_aggs, v2agg_loc]
    if (neqcs)
      { exds[0] = 0; }
    if (neqcs > 1)
      for (auto k : Range(size_t(1), neqcs))
	{ exds[k] = 1 + M.template GetENN<NT_VERTEX>(k); }
    Table<int> exdata(exds); exds = 0;
    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs)) {
	if (!eqc_h.IsMasterOfEQC(eqc))
	  { continue; }
	auto exd_row = exdata[eqc];
	auto & n_aggs = exd_row[0]; n_aggs = 0;
	auto vmap = exd_row.Part(1);
	auto eq_vs = M.template GetENodes<NT_VERTEX>(eqc);
	for (auto j : Range(eq_vs)) {
	  auto vj = eq_vs[j];
	  auto agg_nr = v_to_agg[vj];
	  if (agg_nr != -1) {
	    auto & aggj = agglomerates[agg_nr];
	    if (aggj.center() == vj) {
	      for (auto v : aggj.members()) {
		auto loc_vn = M.template MapENodeToEQC<NT_VERTEX>(eqc, v);
		// cout << v << ", loc nr " << loc_vn << " in " << eqc << " -> " << n_aggs << endl;
		vmap[loc_vn] = n_aggs;
	      }
	      n_aggs++; n_agg_nonloc++;
	    }
	  }
	  else {
	    auto loc_vn = M.template MapENodeToEQC<NT_VERTEX>(eqc, vj);
	    // cout << vj << ", loc nr " << loc_vn << " in " << eqc << " -> DIRI !" << endl;
	    vmap[loc_vn] = -1;
	  }
	}
      }

    // cout << " exdata loc" << endl << exdata << endl;

    auto reqs = eqc_h.ScatterEQCData(exdata);

    // cout << " exdata red" << endl << exdata << endl;

    Array<int> cnv(neqcs); cnv = 0;
    Array<int> cvos(1+neqcs); cvos = 0;

    // map loc. verts
    size_t n_agg_loc = 0;
    for (auto& agg : agglomerates) {
      if (M.template GetEqcOfNode<NT_VERTEX>(agg.center())==0) {
	for (auto mem : agg.members())
	  { vmap[mem] = n_agg_loc; /**cout << "mem " << mem << " of " << agg.id << " -> " << n_agg_loc << endl;**/ }
	n_agg_loc++;
      }
    }
    if (neqcs) {
      cnv[0] = n_agg_loc;
      cvos[1] = cnv[0];
    }

    // finish vmap scatter
    MyMPI_WaitAll(reqs);

    // re-map to global enum
    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs)) {
	cnv[eqc] = exdata[eqc][0];
	cvos[1+eqc] = cvos[eqc] + cnv[eqc];
	auto eq_vs = M.template GetENodes<NT_VERTEX>(eqc);
	auto os = cvos[eqc];
	auto exdr = exdata[eqc];
	auto loc_vm = exdr.Part(1);
	for (auto j : Range(eq_vs))
	  if (loc_vm[j] != -1)
	    { vmap[eq_vs[j]] = os + loc_vm[j]; }
      }


    // comm.Barrier();
    // cout << "cnv : "; prow2(cnv); cout << endl;
    // cout << "cvos: "; prow2(cvos); cout << endl;
    // cout << "vmap: " << endl << vmap << endl;
    // comm.Barrier();

    
    mapped_NN[NT_VERTEX] = cvos.Last();
    cmesh.nnodes[NT_VERTEX] = cvos.Last();
    cmesh.nnodes_glob[NT_VERTEX] = comm.AllReduce(cmesh.nnodes[NT_VERTEX], MPI_SUM);
    cmesh.verts.SetSize(cmesh.nnodes[NT_VERTEX]);
    auto & cverts = cmesh.verts;
    for (auto k : Range(cmesh.nnodes[NT_VERTEX]) )
      { cverts[k] = k; }
    auto & disp_veq = cmesh.disp_eqc[NT_VERTEX];
    disp_veq = move(cvos);
    cmesh.nnodes_eqc[NT_VERTEX] = move(cnv);
    cmesh.nnodes_cross[NT_VERTEX].SetSize(neqcs); cmesh.nnodes_cross[NT_VERTEX] = 0;
    cmesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, cmesh.disp_eqc[NT_VERTEX].Data(), cmesh.verts.Data());

    // comm.Barrier();
    // cout << " cmesh NV " << cmesh.nnodes[NT_VERTEX] << " " << cmesh.nnodes_glob[NT_VERTEX] << endl;
    // cout << " cmesh eqc_verts " << endl << cmesh.eqc_verts << endl;
  } // AgglomerateCoarseMap::MapVerts

  
  template<class TMESH>
  void AgglomerateCoarseMap<TMESH> :: MapEdges (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v2agg)
  {
    /** Say there is a coarse edge between coarse vertices A and B. Everyone looks at all their edges that map to this coarse edge.
	     -) If I have some fine edge in a different EQC that A-B, add the coarse eqc to a table of "additional edges".
	        If I do not, increment counter for eq/cross edges in the coarse eqc. If they are local edges, I can instantly map them!
	     -) Mark all these fine edges as done.
	MPI-gather additional edges.
	Go through the gathered coarse edges. Look at all fine edges that map to it:
	     -) If all of them are in the same eqc, decreace counter for eq/cross edges by one  and again mark fine edges as done.
	Now we have the correct coarse counts.
	Go through gathered coarse edges again and map all fine edges concerned.
	Go through all UNMAPPED fine edges again, their coarse ids are defined by the offsets + the (consistent) ordering of the
	fine edge with the lowest number for each coarse edge.
     **/
    static Timer t("MapEdges"); RegionTimer rt(t);

    const auto & M = *mesh;
    const auto FNV = M.template GetNN<NT_VERTEX>();
    const auto FNE = M.template GetNN<NT_EDGE>();
    const auto & fecon = *M.GetEdgeCM();

    const auto & CM = cmesh;
    const auto CNV = CM.template GetNN<NT_VERTEX>();

    auto sp_eqc_h = M.GetEQCHierarchy();
    const auto & eqc_h = *sp_eqc_h;
    auto neqcs = eqc_h.GetNEQCS();
    auto comm = eqc_h.GetCommunicator();

    const auto & vmap = node_maps[NT_VERTEX];
    auto & emap = node_maps[NT_EDGE]; emap.SetSize(GetNN<NT_EDGE>()); emap = -1;
  
    auto fedges = M.template GetNodes<NT_EDGE>();

    Table<int> c2fv;
    {
      TableCreator<int> cc2fv(CNV);
      for (; !cc2fv.Done(); cc2fv++) {
	for (auto v : Range(vmap)) {
	  auto cv = vmap[v];
	  if (cv != -1)
	    { cc2fv.Add(cv,v); }
	}
      }
      c2fv = cc2fv.MoveTable();
    }

    // cout << " c2fv TABLE: " << endl << c2fv << endl << endl;

    /** Iterates through fine edges. If fine and coarse are loc, call lambda on edge.
	If not, call lambda only if edge is the designated one. Designated is the one
	with the lowest number in the highest eqc. **/
    BitArray mark(FNE), desig(FNE);
    auto wrap_elam = [&](auto eqc, auto edges, auto check_lam, auto order_matters,
			 auto edge_lam) LAMBDA_INLINE {
      for (const auto & e : edges) {
	// cout << " wrap_elam around " << e << " " << mark.Test(e.id) << " " << desig.Test(e.id) << " " << check_lam(e) << endl;
	if ( (!mark.Test(e.id)) && check_lam(e) ) {
	  int cv0 = vmap[e.v[0]], cv1 = vmap[e.v[1]];
	  if ( (cv0 != -1) && (cv1 != -1) && (cv0 != cv1) ) {
	    auto ceq0 = CM.template GetEqcOfNode<NT_VERTEX>(cv0);
	    auto ceq1 = CM.template GetEqcOfNode<NT_VERTEX>(cv1);
	    bool c_cross = (ceq0 != ceq1);
	    auto ceq = c_cross ? eqc_h.GetCommonEQC(ceq0, ceq1) : ceq0;
	    /** look through all fine edges that map to the same coarse edge, pick the one with the smallest number
		in the largest eqc and call lambda on that one. **/
	    auto max_eq = eqc; /** auto min_eq = eqc; **/ auto spec_feid = e.id;
	    auto it_feids = [&](auto ind_lam) LAMBDA_INLINE {
	      FlatArray<int> memsa = c2fv[cv0], memsb = c2fv[cv1];
	      for (auto x : memsa) {
		auto eids = fecon.GetRowValues(x);
		iterate_intersection(fecon.GetRowIndices(x), memsb,
				     [&](auto i, auto j) LAMBDA_INLINE {
				       auto feid = int(eids[i]);
				       mark.SetBit(feid);
				       ind_lam(feid);
				     });
	      }
	    };
	      
	    if ( (order_matters) && (ceq > 0) && (!desig.Test(e.id)) ) { /** Not a local edge, designated edge not found. **/
	      FlatArray<int> memsa = c2fv[cv0], memsb = c2fv[cv1];
	      it_feids ([&](auto feid) LAMBDA_INLINE {
		  /** Find and mark one fine edge as designated, mark rest as done! **/
		  auto eq_fe = M.template GetEqcOfNode<NT_EDGE>(feid);
		  if ( (eq_fe != max_eq) && (eqc_h.IsLEQ(eq_fe, max_eq)) ) {
		    /** must be the lowest edge id in the "biggest" eqc (if there is one. if not, edge changes anyways) **/
		    max_eq =  eq_fe;
		    spec_feid = feid;
		  }
		});
	      }
	    if (spec_feid != e.id) // call this later
	      { desig.SetBit(spec_feid); continue; }
	    /** Either order does not matter (count), or we must be either local, or the designated fine edge! **/
	    if (cv0 > cv1) { swap(cv0, cv1); swap(ceq0, ceq1); }
	    edge_lam (spec_feid, max_eq, ceq, c_cross,
		      cv0, ceq0, cv1, ceq1, it_feids);
	  }
	}
      }
    };
    /** ~ iterate edges unique **/
    auto it_es_u = [&](bool doloc, auto check_lam, auto edge_lam, bool order_matters) LAMBDA_INLINE {
      for (int eqc = ( doloc ? 0 : 1 ); eqc < neqcs; eqc++ ) {
	wrap_elam(eqc, M.template GetENodes<NT_EDGE>(eqc), check_lam, order_matters, edge_lam);
	wrap_elam(eqc, M.template GetCNodes<NT_EDGE>(eqc), check_lam, order_matters, edge_lam);
      }
    };
  
    auto sort_tup = [&](auto & t) LAMBDA_INLINE {
      if (t[0] == t[1]) { if (t[2] > t[3]) { swap(t[2], t[3]); } }
      else if (t[0] > t[1]) { swap(t[1], t[0]); swap(t[2], t[3]); }
    };
    auto less_tup = [&](const auto & a, const auto & b) LAMBDA_INLINE {
      const bool ain = a[0] == a[1], bin = b[0] == b[1];
      if (ain && !bin) { return true; } else if (bin && !ain) { return false; }
      else if (a[0] < b[0]) { return true; } else if (a[0] > b[0]) { return false; }
      else if (a[1] < b[1]) { return true; } else if (a[1] > b[1]) { return false; }
      else if (a[2] < b[2]) { return true; } else if (a[2] > b[2]) { return false; }
      else if (a[3] < b[3]) { return true; } else if (a[3] > b[3]) { return false; }
      else { return false; }
    };
    
    Array<int> cnt_eq(neqcs), cnt_cr(neqcs); cnt_eq = 0; cnt_cr = 0;
    Table<INT<4,int>> add_ce;
    TableCreator<INT<4,int>> cce(neqcs);
    auto edge_lam = [&](bool first_time) {
      return [&, first_time](auto spec_feid, auto feqc, auto ceqc, auto ccross, // first_time captured by value !!
		 auto cv0, auto ceq0, auto cv1, auto ceq1, auto it_feids ) LAMBDA_INLINE {
	// cout << " elam" << first_time << ": " << feqc << " -> " << ceqc << ", cedge " <<
	// "[" << cv0 << ", " << ceq0 << "] - [" << cv1 << ", " << ceq1 << "]" << endl;
	if (ceqc != feqc) { // can only happen in parallel
	  auto ceq0_id = eqc_h.GetEQCID(ceq0);
	  auto ceq1_id = eqc_h.GetEQCID(ceq1);
	  auto cv0_loc = CM.template MapENodeToEQC<NT_VERTEX>(ceq0, cv0);
	  auto cv1_loc = CM.template MapENodeToEQC<NT_VERTEX>(ceq1, cv1);
	  INT<4,int> tup( { ceq0_id, ceq1_id, cv0_loc, cv1_loc } );
	  // cout << " c tup " << tup << endl;
	  sort_tup(tup);
	  // cout << "sorted c tup " << tup << endl;
	  cce.Add ( ceqc, tup );
	}
	// cannot map local edges here because I cannot alloc coarse edge array without finished counts...
	if (first_time) {
	  if (ccross) { cnt_cr[ceqc]++; }
	  else { cnt_eq[ceqc]++; }
	}
	// it_feids( [&](auto feid) LAMBDA_INLINE { ; } ); // marks!
	// cout << "feids: ";
	it_feids( [&](auto feid) LAMBDA_INLINE { /*cout << fedges[feid] << " "; */; } ); // !!! DO NOT REMOVE - marks !!!
	// cout << endl;
      };
    };
    auto check_all = [&](const auto & edge) LAMBDA_INLINE { return true; };
    mark.Clear();
    it_es_u (true, check_all, edge_lam(true), false); cce++; // we only count - order does not matter !
    for (; !cce.Done(); cce++) {
      if (neqcs > 1) { // if local, skip these steps - we would not add any edges
	mark.Clear();
	it_es_u (true, check_all, edge_lam(false), false); // have to sort and merge this anyways - order does not matter !
      }
    }
    auto X = cce.MoveTable();
    for (auto row : X) // should be cheap (few add_edges!). rows are duplicate-less, but diffrent rows can contain same edge
      { QuickSort(row, less_tup); }

    // cout << endl << "LOC ADD_CE: " << endl;
    // cout << X << endl << endl;

    add_ce = ReduceTable<INT<4,int>,INT<4,int>>(X, sp_eqc_h, [&less_tup](const auto & tab) LAMBDA_INLINE {
	return merge_arrays(tab, less_tup);
      });

    // cout << endl << "RED ADD_CE: " << endl;
    // cout << add_ce << endl << endl;

    // cout << endl << "LOC CNTS cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
    // cout << endl << "LOC CNTS cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
    
    /** Now we have to iterate through add_cedges, and decrease counters if we have a fine edge mapping to it.
	Counters for add_edges are correct already. **/
    Array<int> cnt_add_eq(neqcs), cnt_add_cr(neqcs);
    cnt_add_eq = 0; cnt_add_cr = 0;
    for (auto eqc : Range(neqcs)) {
      auto exrow = add_ce[eqc];
      int n_add_in_es = exrow.Size();
      for (auto j : Range(exrow)) {
	auto & tup = exrow[j];
	auto eq0 = eqc_h.GetEQCOfID(tup[0]);
	auto v0 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[2], eq0);
	auto eq1 = (tup[1] != tup[0]) ? eqc_h.GetEQCOfID(tup[1]) : eq0;
	auto v1 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[3], eq1);
	auto memsa = c2fv[v0]; auto memsb = c2fv[v1];
	// cout << "count " << eqc << " " << j << ", tup " << tup << endl;
	auto ccross = tup[1] != tup[0];
	if (ccross) { cnt_add_cr[eqc]++; }
	else {cnt_add_eq[eqc]++; }
	// cout << " memsa: "; prow(memsa); cout << endl;
	// cout << " memsb: "; prow(memsb); cout << endl;
	for (auto x : memsa) { /** find out if we have already counted this coarse edge (there might not be) **/
	  // cout << x << " con to "; prow(fecon.GetRowIndices(x)); cout << endl;
	  if (!is_intersect_empty(fecon.GetRowIndices(x), memsb)) { // there is SOME fine edge, which we have counted already
	    if (ccross) { cnt_cr[eqc]--; /* cout << "dec cnt_cr[" << eqc << "], now " << cnt_cr[eqc] << endl; */ }
	    else { cnt_eq[eqc]--; /*cout << "dec cnt_eq[" << eqc << "], now " << cnt_eq[eqc] << endl; */ }
	    break;
	  }
	}
      }
    }

    /** Counters are now correct, we can set up the offset arrays **/
    Array<int> os_eq(1+neqcs), os_cr(1+neqcs);
    os_eq[0] = 0; os_cr[0] = 0;
    for (auto k : Range(neqcs)) {
      os_eq[1+k] = os_eq[k] + cnt_eq[k] + cnt_add_eq[k];
      os_cr[1+k] = os_cr[k] + cnt_cr[k] + cnt_add_cr[k];
    }
    size_t neq = os_eq.Last(), ncr = os_cr.Last(), nce = neq + ncr;

    // cout << endl << " nce neq ncr " << nce << " " << neq << " " << ncr << endl;
    // cout << "cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
    // cout << "cnt_add_eq " << endl; prow2(cnt_add_eq); cout << endl;
    // cout << "os_eq " << endl; prow2(os_eq); cout << endl;
    // cout << "cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
    // cout << "cnt_add_cr " << endl; prow2(cnt_add_cr); cout << endl;
    // cout << "os_cr " << endl; prow2(os_cr); cout << endl << endl;


    auto & cedges = cmesh.edges; cedges.SetSize(nce);
    for (auto & cedge : cedges)
      { cedge.id = -4242; cedge.v = { -42, -42 }; }

    mark.Clear(); desig.Clear();

    /** we have to set add_edges manually - we might have some which are not mapped to locally **/
    if (neqcs > 1) {
      auto set_cedge = [&](auto id, auto& tup) {
	auto eq0 = eqc_h.GetEQCOfID(tup[0]);
	auto v0 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[2], eq0);
	auto eq1 = (tup[1] != tup[0]) ? eqc_h.GetEQCOfID(tup[1]) : eq0;
	auto v1 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[3], eq1);
	cedges[id].id = id;
	cedges[id].v = (v0 < v1) ? INT<2, int>({v0, v1}) : INT<2, int>({v1, v0});
	// cout << " add edge id " << id << ", tup " << tup << " -> " << cedges[id] << endl;
	// cout << " map fedges ";
	FlatArray<int> memsa = c2fv[v0], memsb = c2fv[v1];
	for (auto x : memsa) {
	  auto eids = fecon.GetRowValues(x);
	  iterate_intersection(fecon.GetRowIndices(x), memsb,
			       [&](auto i, auto j) LAMBDA_INLINE {
				 auto feid = int(eids[i]);
				 emap[feid] = id; mark.SetBit(feid); // i think i dont ned mark because i check emap below ??
				 // cout << feid << " ";
			       });
	}
	// cout << endl;
      };
      for (auto eqc : Range(size_t(1), neqcs)) {
     	auto exrow = add_ce[eqc];
	auto cid_eq = os_eq[eqc] + cnt_eq[eqc];
     	for (auto j : Range(cnt_add_eq[eqc]))
	  { set_cedge(cid_eq++, exrow[j]); }
	auto cid_cr = neq + os_cr[eqc] + cnt_cr[eqc];
     	for (auto j : Range(cnt_add_eq[eqc], int(exrow.Size())))
	  { set_cedge(cid_cr++, exrow[j]); }
      }
    }

    cnt_eq = 0; cnt_cr = 0; // mark already cleared
    it_es_u(true, // god dammit, i HAVE to loop through ALL edges because i only alloc coarse edges after finished counts
	    [&](const auto & e) LAMBDA_INLINE { return emap[e.id] == -1; }, // edges mapping to add_edges are already mapped
	    [&](auto spec_id, auto feqc, auto ceqc, auto ccross,
		auto cv0, auto ceq0, auto cv1, auto ceq1, auto it_feids ) LAMBDA_INLINE {
	      // cout << "map ordinary, ccross " << ccross << ", eqcs " << feqc << " -> " << ceqc << ", cedge " <<
		// "[" << cv0 << ", " << ceq0 << "] - [" << cv1 << ", " << ceq1 << "]" << endl;
	      int loc_id = ccross ? cnt_cr[ceqc]++ : cnt_eq[ceqc]++;
	      auto cid = (ccross ? (neq + os_cr[ceqc]) : os_eq[ceqc]) + loc_id;
	      cedges[cid].id = cid;
	      // cedges[cid].v = (cv0 < cv1) ? INT<2, int>({cv0, cv1}) : INT<2, int>({cv1, cv0});
	      cedges[cid].v = INT<2, int>({cv0, cv1});
	      // cout << " feids: ";
	      it_feids([&](auto feid) LAMBDA_INLINE { /* cout << fedges[feid] << " "; */ emap[feid] = cid; });
	      // cout << " -> " << "cross " << ccross << ", ceq " << ceqc << ", locid " << loc_id << ", fill id " << cid << ", edge " << cedges[cid] << endl;
	      // cout << "(crs loc e " << eqc_h.GetEQCID(ceq0) << " " << eqc_h.GetEQCID(ceq1) <<
		// " " << CM.template MapENodeToEQC<NT_VERTEX>(cv0) << " " << CM.template MapENodeToEQC<NT_VERTEX>(cv1) << endl;
	    }, true); // order matters here!
    
    // cout << endl << "FIN 1 cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
    // cout << "FIN 1 os_eq " << endl; prow2(os_eq); cout << endl;
    // cout << "FIN 1 cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
    // cout << "FIN 1 os_cr " << endl; prow2(os_cr); cout << endl << endl;

    for (auto k : Range(neqcs)) {
      cnt_eq[k] += cnt_add_eq[k];
      cnt_cr[k] += cnt_add_cr[k];
    }
    
    // cout << endl << "FIN 2 cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
    // cout << "FIN 2 os_eq " << endl; prow2(os_eq); cout << endl;
    // cout << "FIN 2 cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
    // cout << "FIN 2 os_cr " << endl; prow2(os_cr); cout << endl << endl;

    mapped_NN[NT_EDGE] = nce;
    cmesh.nnodes[NT_EDGE] = nce;
    cmesh.disp_eqc[NT_EDGE] = move(os_eq);
    cmesh.disp_cross[NT_EDGE] = move(os_cr);
    cmesh.nnodes_glob[NT_EDGE] = 0;
    cmesh.nnodes_eqc[NT_EDGE] = cnt_eq;
    cmesh.nnodes_cross[NT_EDGE] = cnt_cr;
    cmesh.eqc_edges  = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_eqc[NT_EDGE].Data(), cmesh.edges.Data());
    cmesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_cross[NT_EDGE].Data(), cmesh.edges.Part(neq).Data());
    cmesh.nnodes_glob[NT_EDGE] = 0;
    for (auto eqc : Range(neqcs))
      if (eqc_h.IsMasterOfEQC(eqc))
	{ cmesh.nnodes_glob[NT_EDGE] += cnt_eq[eqc] + cnt_cr[eqc]; }
    cmesh.nnodes_glob[NT_EDGE] = comm.AllReduce(cmesh.nnodes_glob[NT_EDGE], MPI_SUM);

    cmesh.nnodes[NT_FACE] = 0;
    cmesh.nnodes[NT_CELL] = 0;

    // cout << " fmesh NE " << M.nnodes[NT_EDGE] << " " << M.nnodes_glob[NT_EDGE] << endl << endl;
    // cout << " fmesh disp e eqc " << endl; prow2(M.disp_eqc[NT_EDGE]); cout << endl << endl;
    // cout << " fmesh disp e cr " << endl; prow2(M.disp_cross[NT_EDGE]); cout << endl << endl;
    // cout << " fmesh eqc e " << endl; cout << M.eqc_edges << endl << endl;
    // cout << " fmesh cr e " << endl; cout << M.cross_edges << endl << endl;

    // cout << " cmesh NE " << cmesh.nnodes[NT_EDGE] << " " << cmesh.nnodes_glob[NT_EDGE] << endl << endl;
    // cout << " cmesh eqc e " << endl; cout << cmesh.eqc_edges << endl << endl;
    // cout << " cmesh cr e " << endl; cout << cmesh.cross_edges << endl << endl;

    Array<int> ce_cnt(cmesh.nnodes[NT_EDGE]); ce_cnt = 1;

    // cout << "eqc_h: " << endl << eqc_h << endl;
    // cout << " CM eqc_h: " << endl << *CM.GetEQCHierarchy() << endl;
    
    // cout << " test allred edge data " << endl;
    // CM.template AllreduceNodalData<NT_EDGE>(ce_cnt, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });
    // cout << " have tested allred edge data! " << endl;

    // cout << " ce_cnt: " << endl; prow2(ce_cnt); cout << endl;
    
    // Array<INT<2>> ce_cnt2(cmesh.nnodes[NT_EDGE]); ce_cnt2 = 0;
    // int I = eqc_h.GetCommunicator().Rank() - 1;
    // if ( (I == 0) || (I == 1) )
    //   for (auto k : Range(neqcs)) {
    // 	for(const auto & edge : cmesh.template GetENodes<NT_EDGE>(k))
    // 	  { ce_cnt2[edge.id][I] = 1; }
    // 	for(const auto & edge : cmesh.template GetCNodes<NT_EDGE>(k))
    // 	  { ce_cnt2[edge.id][I] = 2; }
    //   }
    // CM.template AllreduceNodalData<NT_EDGE>(ce_cnt2, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    // cout << " ce_cnt2: " << endl; prow2(ce_cnt2); cout << endl;

  } // AgglomerateCoarseMap::MapEdges


  template<class TMESH>
  void AgglomerateCoarseMap<TMESH> :: MapEdges2 (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v2agg)
  {
    /**
       This version can only work if all eqcs are in-eqc, and only master of any vertex can assign it to an agg.
       (Even then not 100% sure it is bug-free)

       I know that edges do not change eqc, and cannot go from in-eq to cross-eq, so everything should be easy,
       master of eqch EQC maps (locally), then scatter and add offsets.
       But only masters of each EQC have the agglomerates IN that EQC. So, when i am master of an EQC i CAN
       map IN-EQC edges of that eqc BUT NOT NECESSARILY CROSS-EQC edges.
       For Cross-edges, I sometimes also need to know the agglomerates in EQCs I am not master of!
       EX: an edge {1,2} - {2,3}. Aggs in {1,2} are on 1, aggs in {2,3} are on 3, but 2 is master of the edge.
       Therefore, we set for all agglomerates:
            - agg.id = vmap[agg.center()]
       We scatter agg. data for non-local EQCs, and iterate through cross-edges of each EQC an extra time.
       If a cross-edge connects two verts that do not drop, we know it WILL be mapped, and that we are it's master,
       so we can map it (locally) as we wish before scattering the local map.
       For each of its vertices, we are either that master of it's eqc (in which case we have the agg), or we are not
       (and the agg is in the exchange-data!)
     **/

    static Timer t("MapEdges"); RegionTimer rt(t);

    const auto & M = *mesh;
    const auto & CM = cmesh;
    const auto & fecon = *M.GetEdgeCM();
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto neqcs = eqc_h.GetNEQCS();
    auto comm = eqc_h.GetCommunicator();

    const auto & vmap = node_maps[NT_VERTEX];
    auto & emap = node_maps[NT_EDGE]; emap.SetSize(GetNN<NT_EDGE>()); emap = -1;

    Array<int> cnt_eqe(neqcs); cnt_eqe = 0;
    Array<int> cnt_cre(neqcs); cnt_cre = 0;

    /** Scatter agglomerates **/
    Table<int> ex_aggs; // [offsets, loc nr of mems aggs] -> (1+CNV) + FNV
    {
      size_t mexv = 0;
      Array<int> exds(neqcs);
      for (auto k : Range(neqcs))
	{ exds[k] = (k == 0) ? 0 : 1 + M.template GetENN<NT_VERTEX>(k) + CM.template GetENN<NT_VERTEX>(k); }
      ex_aggs = Table<int> (exds); // [offsets, loc nr of mems aggs] -> max. (1+CNV) + FNV (some vertices collapse and are not in an agg.)
      if (neqcs > 1)
	for (auto eqc : Range(size_t(1), neqcs)) {
	  if (eqc_h.IsMasterOfEQC(eqc)) {
	    auto fexvs = M.template GetENodes<NT_VERTEX>(eqc);
	    auto naggs = CM.template GetENN<NT_VERTEX>(eqc);
	    auto exrow = ex_aggs[eqc];
	    auto exos = exrow.Part(0, 1 + naggs); exos = 0;
	    for (auto v : fexvs) {
	      auto aggnr = v2agg[v];
	      if ( aggnr != -1 ) {
		auto & agg = agglomerates[aggnr];
		if (v == agg.center()) {
		  auto cv = vmap[v];
		  // cout << " eqc " << eqc << ", v " << v << ", aggnr " << aggnr << endl;
		  // cout << " agg " << agg << endl;
		  // cout << " cv " << cv << " loc nr " << CM.template MapENodeToEQC<NT_VERTEX>(eqc, cv) << endl;
		  exos[1 + CM.template MapENodeToEQC<NT_VERTEX>(eqc, cv)] = agg.members().Size();
		}
	      }
	    }
	    for (auto l : Range(naggs))
	      { exos[l+1] += exos[l]; }
	    // cout << endl << " offsets "; prow2(exos); cout << endl;
	    auto aggm = exrow.Part(1 + naggs); aggm = -1;
	    for (auto v : fexvs) {
	      auto aggnr = v2agg[v];
	      if ( aggnr != -1 ) {
		auto & agg = agglomerates[aggnr];
		if ( v == agg.center()) {
		  auto cv = vmap[v];
		  auto loc_cv = CM.template MapENodeToEQC<NT_VERTEX>(eqc, cv);
		  auto mems = agg.members();
		  auto exmems = exrow.Part(1 + naggs + exos[loc_cv], mems.Size());
		  // cout << " eqc " << eqc << ", v " << v << ", aggnr " << aggnr << endl;
		  // cout << " agg " << agg << endl;
		  // cout << " cv " << cv << " loc nr " << CM.template MapENodeToEQC<NT_VERTEX>(eqc, cv) << ", os " << exos[loc_cv] << endl;
		  // for (auto l : Range(exmems))
		    // { cout << l << ", mem " << mems[l] << " -> loc " << M.template MapENodeToEQC<NT_VERTEX>(eqc, mems[l]) << endl; }
		  for (auto l : Range(exmems))
		    { exmems[l] = M.template MapENodeToEQC<NT_VERTEX>(eqc, mems[l]); }
		}
	      }
	    }
	  }
	}
      MyMPI_WaitAll(eqc_h.ScatterEQCData(ex_aggs));
    }

    // cout << endl << "ex_aggs: " << endl << ex_aggs << endl << endl;

    /** ex_aggs are in (eqc-) local enumeration -> make this a glob enum!  **/
    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs)) {
	if (!eqc_h.IsMasterOfEQC(eqc)) {
	  auto naggs = CM.template GetENN<NT_VERTEX>(eqc);
	  if (naggs) {
	    auto exrow = ex_aggs[eqc];
	    auto remap_this = exrow.Part(naggs + 1);
	    // !! a bit hacky, i can do this super cheaply because i KNOW that i just need to add something !! //
	    auto os = M.template MapENodeFromEQC<NT_VERTEX>(0, eqc);
	    for (auto& v : remap_this)
	      { v += os; }
	  }
	}
      }

    // cout << endl << "re-mapped ex_aggs: " << endl << ex_aggs << endl << endl;

    Array<int> cneibs1(30), cneibs2(30);
    Array<int> se_v1(30); se_v1.SetSize0();
    Array<int> shared_edges(30);
    auto map_it = [&](size_t n_aggs, auto get_eqc, auto get_mems, auto get_cvnum) {
      for (auto agg_nr : Range(n_aggs)) {
	auto eqc = get_eqc(agg_nr);
	auto mems = get_mems(agg_nr);
	auto cvnum = get_cvnum(agg_nr);
	// cout << endl << " map agg in eqc " << eqc << ", cvnum " << cvnum << ", mems "; prow2(mems); cout << endl;
	cneibs1.SetSize0(); cneibs2.SetSize0();
	for (auto mem : mems) {
	  for (auto n : fecon.GetRowIndices(mem)) {
	    auto neib_agg_nr = v2agg[n];
	    if (neib_agg_nr != -1) { // neighbor is in an agg I have
	      auto & neib_agg = agglomerates[neib_agg_nr];
	      // if ( (cvnum < vmap[neib_agg.center()]) && (!cneibs1.Contains(neib_agg_nr)) ) // only need it once
		// cout << " have a new neib agg " << neib_agg << endl;
	      if ( (cvnum < vmap[neib_agg.center()]) && (!cneibs1.Contains(neib_agg_nr)) ) // only need it once
		{ cneibs1.Append(neib_agg_nr); }
	    }
	    else { // neighbor must be in an ex-agg I do not have locally
	      auto neib_cvnum = vmap[n];
	      // cout << "  neib " << n << " -> " << vmap[n] << endl;
	      // if ( (neib_cvnum != -1) && (cvnum < neib_cvnum) && (!cneibs2.Contains(neib_cvnum)) )
		// cout << " have a new neib cv " << neib_cvnum << endl;
	      if ( (neib_cvnum != -1) && (cvnum < neib_cvnum) && (!cneibs2.Contains(neib_cvnum)) )
		{ cneibs2.Append(neib_cvnum); }
	      }
	    }
	  }
	auto find_map_edges = [&](auto ce_cross, auto ce_eqc, auto neib_cv, const auto & neib_mems) LAMBDA_INLINE {
	  shared_edges.SetSize0();
	  for (auto agg_mem : mems) {
	    auto mem_neibs = fecon.GetRowIndices(agg_mem);
	    intersect_sorted_arrays(mem_neibs, neib_mems, se_v1);
	    for (auto n : se_v1) {
	      int eid = int(fecon(agg_mem, n));
	      if (!shared_edges.Contains(eid))
		{ shared_edges.Append(eid); }
	    }
	  }
	  auto ceid = ce_cross ? cnt_cre[ce_eqc]++ : cnt_eqe[ce_eqc]++;
	  // cout << " set edges "; prow(shared_edges); cout << " TO CEID " << ceid << endl;
	  for (auto eid : shared_edges)
	    { emap[eid] = ceid; }
	};
	/** map edges connecting to an agg I have **/
	// cout << "  map neib aggs " << endl;
	for (auto neib : cneibs1) {
	  auto & neib_agg = agglomerates[neib];
	  auto neib_cv = vmap[neib_agg.center()]; // the coarse edge is (cv - neib_cv)
	  auto neib_eqc = CM.template GetEqcOfNode<NT_VERTEX>(neib_cv);
	  bool ce_cross = (neib_eqc != eqc);
	  auto ce_eqc = (ce_cross) ? eqc_h.GetCommonEQC(eqc, neib_eqc) : eqc;
	  bool master = eqc_h.IsMasterOfEQC(ce_eqc);
	  // cout << "   neib_agg " << " cv " << neib_cv << ", eqc " << neib_eqc << "  " << neib_agg << endl;
	  // cout << "   ce_eqc " << ce_eqc << " cross " << ce_cross << " master " << master << endl;
	  if (master) // i have ALL verts in both aggs -> i have all edges connecting them, I am master of cut eqc -> can map locally
	    { find_map_edges(ce_cross, ce_eqc, neib_cv, neib_agg.members()); }
	  // cout << "   map ok " << endl;
	}
	/** map edges connecting to an ex-agg I do not also have locally **/
	// cout << "  map neib ex_aggs " << endl;
	for (auto neib_cv : cneibs2) {
	  auto neib_eqc = CM.template GetEqcOfNode<NT_VERTEX>(neib_cv);
	  auto loc_neib_num = CM.template MapENodeToEQC<NT_VERTEX>(neib_eqc, neib_cv);
	  bool ce_cross = (neib_eqc != eqc);
	  auto ce_eqc = (ce_cross) ? eqc_h.GetCommonEQC(eqc, neib_eqc) : eqc;
	  bool master = eqc_h.IsMasterOfEQC(ce_eqc);
	  // cout << "   neib_agg  cv " << neib_cv << ", eqc " << neib_eqc << " " << endl;
	  // cout << "   ce_eqc " << ce_eqc << " cross " << ce_cross << " master " << master << endl;
	  if (master) {
	    auto exrow = ex_aggs[neib_eqc];
	    auto ncvineq = CM.template GetENN<NT_VERTEX>(neib_eqc);
	    auto os = ncvineq + 1;
	    auto neib_mems = exrow.Range(os+exrow[loc_neib_num], os+exrow[loc_neib_num+1]);
	    // cout << "   neib_mems "; prow2(neib_mems); cout << endl;
	    find_map_edges(ce_cross, ce_eqc, neib_cv, neib_mems);
	  }
	}
      }
    };

    // auto map_it = [&](size_t n_aggs, auto get_eqc, auto get_mems, auto get_cvnum);

    /** Map (eqc-locally): { my_aggs } X { my_aggs, ex_aggs } **/
    // cout << " map loc aggs " << endl;

    map_it ( agglomerates.Size(),
	     [&](auto i) LAMBDA_INLINE { return M.template GetEqcOfNode<NT_VERTEX>(agglomerates[i].center()); },
	     [&](auto i) LAMBDA_INLINE { return agglomerates[i].members(); },
	     [&](auto i) LAMBDA_INLINE { return vmap[agglomerates[i].center()]; } );

    // cout << " emap after aggs " << endl; prow2(emap); cout << endl;

    /** Map (eqc-locally): { ex_aggs } X { my_aggs, ex_aggs } **/
    if (neqcs > 1)
      for (auto eqc : Range(size_t(1), neqcs)) {
	if (!eqc_h.IsMasterOfEQC(eqc)) { // if i was master of this eqc, I would have the agglomerates
	  // cout << " map ex aggs im eqc " << eqc << endl;
	  auto naggs = CM.template GetENN<NT_VERTEX>(eqc);
	  auto exrow = ex_aggs[eqc];
	  auto ex_os = exrow.Part(0, 1 + naggs);
	  auto ex_mems = exrow.Part(1+naggs);
	  map_it ( naggs,
		   [&](auto i) LAMBDA_INLINE { return eqc; },
		   [&](auto i) LAMBDA_INLINE { return ex_mems.Range(ex_os[i], ex_os[i+1]); },
		   [&](auto i) LAMBDA_INLINE { return CM.template MapENodeFromEQC<NT_VERTEX>(i, eqc); } );
	}
      }


    /** scatter EQC counts and loc maps **/
    Array<INT<2>> cnts(neqcs);
    for (auto k : Range(cnts)) {
      cnts[k][0] = cnt_eqe[k];
      cnts[k][1] = cnt_cre[k];
    }
    // cout << " cnts pre scatter " << endl; prow2(cnts); cout << endl;
    auto reqs = eqc_h.ScatterEQCData(cnts);

    // cout << " loc emap " << endl; prow2(emap); cout << endl << endl;
    M.template ScatterNodalData<NT_EDGE>(emap);
    // cout << " red emap " << endl; prow2(emap); cout << endl << endl;

    MyMPI_WaitAll(reqs);
    // cout << " cnts post scatter " << endl; prow2(cnts); cout << endl << endl;
    for (auto k : Range(cnts)) {
      cnt_eqe[k] = cnts[k][0];
      cnt_cre[k] = cnts[k][1];
    }

    size_t neqe = std::accumulate(cnt_eqe.begin(), cnt_eqe.end(), 0);
    size_t ncre = std::accumulate(cnt_cre.begin(), cnt_cre.end(), 0);
    Array<int> disp_eqe(1+neqcs); disp_eqe = 0;
    Array<int> disp_cre(1+neqcs); disp_cre = 0;
    for (auto k : Range(neqcs)) {
      disp_eqe[1+k] = disp_eqe[k] + cnt_eqe[k];
      disp_cre[1+k] = disp_cre[k] + cnt_cre[k];
    }

    // cout << "neqeq ncre neqeq+ncre  " << neqe << " " << ncre << " " << neqe+ncre << endl;
    // cout << "cnt_eqe "; prow2(cnt_eqe); cout << endl;
    // cout << "disp_eqe "; prow2(disp_eqe); cout << endl;
    // cout << "cnt_cre "; prow2(cnt_cre); cout << endl;
    // cout << "disp_cre "; prow2(disp_cre); cout << endl;

    BitArray ceset(neqe + ncre); ceset.Clear();
    auto & cedges = cmesh.edges; // need non-const ref
    // Array<AMG_Node<NT_EDGE>> cedges(neqe + ncre);
    cedges.SetSize(neqe + ncre);

    auto remap_and_set = [&](const auto os, auto e_array) {
      for (const auto & fedge : e_array) {
	const auto loc_ceid = emap[fedge.id] ;
	// cout << " remap eqc fedge " << fedge << " -> " << loc_ceid << endl;
	if (loc_ceid != -1) {
	  const auto ceid = os + loc_ceid;
	  // cout << "  ceid " << ceid << endl;
	  emap[fedge.id] = ceid;
	  if (!ceset.Test(ceid)) {
	    auto & cedge = cedges[ceid];
	    const auto cv0 = vmap[fedge.v[0]]; const auto cv1 = vmap[fedge.v[1]];
	    cedge.id = ceid;
	    cedge.v = (cv0 < cv1) ? decltype(cedge.v)({ cv0, cv1 }) : decltype(cedge.v)({ cv1, cv0 });
	    ceset.SetBit(ceid);
	    // cout << "  -> cedge " << cedge << endl;
	  }
	}
      }
    };
    for (auto k : Range(neqcs)) {
      // cout << endl << "map eqc edges " << k << endl;
      remap_and_set(disp_eqe[k], M.template GetENodes<NT_EDGE>(k));
      // cout << endl << "map cross edges " << k << endl;
      remap_and_set(neqe + disp_cre[k], M.template GetCNodes<NT_EDGE>(k));
    }

    mapped_NN[NT_EDGE] = neqe + ncre;
    cmesh.nnodes[NT_EDGE] = neqe + ncre;
    cmesh.edges = move(cedges);
    cmesh.disp_eqc[NT_EDGE] = move(disp_eqe);
    cmesh.disp_cross[NT_EDGE] = move(disp_cre);
    cmesh.nnodes_glob[NT_EDGE] = 0;
    cmesh.nnodes_eqc[NT_EDGE] = cnt_eqe;
    cmesh.nnodes_cross[NT_EDGE] = cnt_cre;
    cmesh.eqc_edges  = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_eqc[NT_EDGE].Data(), cmesh.edges.Data());
    cmesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_cross[NT_EDGE].Data(), cmesh.edges.Part(neqe).Data());
    cmesh.nnodes_glob[NT_EDGE] = 0;
    for (auto eqc : Range(neqcs))
      if (eqc_h.IsMasterOfEQC(eqc))
	{ cmesh.nnodes_glob[NT_EDGE] += cnt_eqe[eqc] + cnt_cre[eqc]; }
    cmesh.nnodes_glob[NT_EDGE] = comm.AllReduce(cmesh.nnodes_glob[NT_EDGE], MPI_SUM);

    cmesh.nnodes[NT_FACE] = 0;
    cmesh.nnodes[NT_CELL] = 0;

    // cout << " fmesh NE " << M.nnodes[NT_EDGE] << " " << M.nnodes_glob[NT_EDGE] << endl;
    // cout << " fmesh eqc e " << endl; cout << M.eqc_edges << endl;
    // cout << " fmesh cr e " << endl; cout << M.cross_edges << endl;

    // cout << " cmesh NE " << cmesh.nnodes[NT_EDGE] << " " << cmesh.nnodes_glob[NT_EDGE] << endl;
    // cout << " cmesh eqc e " << endl; cout << cmesh.eqc_edges << endl;
    // cout << " cmesh cr e " << endl; cout << cmesh.cross_edges << endl;


    // Array<int> ce_cnt(cmesh.nnodes[NT_EDGE]); ce_cnt = 1;

    // cout << "eqc_h: " << endl << eqc_h << endl;
    
    // cout << " test allred edge data " << endl;
    // CM.template AllreduceNodalData<NT_EDGE>(ce_cnt, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });
    // cout << " have tested allred edge data! " << endl;

    // cout << " ce_cnt: " << endl; prow2(ce_cnt); cout << endl;
    
    // Array<INT<2>> ce_cnt2(cmesh.nnodes[NT_EDGE]); ce_cnt2 = 0;
    // int I = eqc_h.GetCommunicator().Rank() - 1;
    // if ( (I == 0) || (I == 1) )
    //   for (auto k : Range(neqcs)) {
    // 	for(const auto & edge : cmesh.template GetENodes<NT_EDGE>(k))
    // 	  { ce_cnt2[edge.id][I] = 1; }
    // 	for(const auto & edge : cmesh.template GetCNodes<NT_EDGE>(k))
    // 	  { ce_cnt2[edge.id][I] = 2; }
    //   }
    // CM.template AllreduceNodalData<NT_EDGE>(ce_cnt2, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    // cout << " ce_cnt2: " << endl; prow2(ce_cnt2); cout << endl;

  } // AgglomerateCoarseMap::MapEdges


  /** Agglomerator **/


  template<class FACTORY>
  Agglomerator<FACTORY> :: Agglomerator (shared_ptr<typename FACTORY::TMESH> _mesh, shared_ptr<BitArray> _free_verts, Options && _settings)
    : AgglomerateCoarseMap<TMESH>(_mesh), free_verts(_free_verts), settings(_settings)
  {
    assert(mesh != nullptr); // obviously this would be bad
  } // Agglomerator(..)


  template<class FACTORY> template<class TMU>
  INLINE FlatArray<TMU> Agglomerator<FACTORY> :: GetEdgeData ()
  {
    if constexpr(std::is_same<TMU, TM>::value)
      { return get<1>(mesh->Data())->Data(); }
    else {
      auto full_edata = get<1>(mesh->Data())->Data();
      traces.SetSize(full_edata.Size());
      for (auto k : Range(traces))
	{ traces[k] = calc_trace(full_edata[k]); }
      return traces;
    }
  }

  template<class FACTORY> template<class TMU>
  INLINE void Agglomerator<FACTORY> :: FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    static_assert ( (std::is_same<TMU, TM>::value || std::is_same<TMU, double>::value), "Only 2 options make sense!");

    static Timer t("FormAgglomerates"); RegionTimer rt(t);

    static Timer tdiag("FormAgglomerates - diags");
    static Timer tecw("FormAgglomerates - init. ecw");
    static Timer t1("FormAgglomerates - 1"); // prep
    static Timer t2("FormAgglomerates - 2"); // first loop
    static Timer t3("FormAgglomerates - 3"); // second loop

    constexpr int N = mat_traits<TMU>::HEIGHT;
    const auto & M = *mesh; M.CumulateData();
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    const auto NV = M.template GetNN<NT_VERTEX>();
    const auto & econ = *M.GetEdgeCM();

    const double MIN_ECW = settings.edge_thresh;
    const bool dist2 = settings.dist2;
    const bool geom = settings.cw_geom;
    const int MIN_NEW_AGG_SIZE = 2;

    FlatArray<TVD> vdata = get<0>(mesh->Data())->Data();
    FlatArray<TMU> edata = GetEdgeData<TMU>();

    // cout << " agg coarsen params: " << endl;
    // cout << "min_ecw = " << MIN_ECW << endl;
    // cout << "dist2 = " << dist2 << endl;
    // cout << "geom = " << geom << endl;
    // cout << " min new agg size = " << MIN_NEW_AGG_SIZE << endl;
    // cout << " neibs per v " << double(2 * M.template GetNN<NT_EDGE>())/NV << endl;

    /** replacement-matrix diagonals **/
    Array<TMU> repl_diag(M.template GetNN<NT_VERTEX>()); repl_diag = 0;

    /** collapse weights for edges - we use these as upper bounds for weights between agglomerates (TODO: should not need anymore) **/
    Array<double> ecw(M.template GetNN<NT_EDGE>()); ecw = 0;

    /** vertex -> agglomerate map **/
    BitArray marked(M.template GetNN<NT_VERTEX>()); marked.Clear();
    Array<int> dist2agg(M.template GetNN<NT_VERTEX>()); dist2agg = -1;
    v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;

    Array<TMU> agg_diag;

    auto get_vwt = [&](auto v) {
      if constexpr(is_same<TVD, double>::value) { return vdata[v]; }
      else { return vdata[v].wt; }
    };

    /** Can we add something from eqa to eqb?? **/
    auto eqa_to_eqb = [&](auto eqa, auto eqb) { // PER DEFINITION, we are master of eqb!
      return eqc_h.IsLEQ(eqa, eqb);
    };

    Array<int> neibs_in_agg;
    size_t cnt_prtm = 0;
    Array<int> common_neibs(20), aneibs(20), bneibs(20);
    Array<FlatArray<int>> neib_tab(20);
    TMU Qij, Qji, emat, Q, Ein, Ejn, Esum, addE, Q2, Aaa, Abb;
    SetIdentity(Qij); SetIdentity(Qji); SetIdentity(emat); SetIdentity(Q); SetIdentity(Ein); SetIdentity(Ejn);
    SetIdentity(Esum); SetIdentity(addE); SetIdentity(Q2); SetIdentity(Aaa); SetIdentity(Abb);
    
    /** Add a contribution from a neighbour common to both vertices of an edge to the edge's matrix **/
    auto add_neib_edge = [&](TVD h_data, const auto & amems, const auto & bmems, auto N, auto & mat) LAMBDA_INLINE {
      constexpr int N2 = mat_traits<TMU>::HEIGHT;
      auto rowis = econ.GetRowIndices(N);
      // cout << " add neib " << N << endl;
      // cout << " amems "; prow(amems); cout << endl;
      // cout << " bmems "; prow(bmems); cout << endl;
      Ein = 0;
      intersect_sorted_arrays(rowis, amems, neibs_in_agg);
      // cout << " conections to a: "; prow(neibs_in_agg); cout << endl;
      for (auto amem : neibs_in_agg) {
	ModQij(vdata[N], vdata[amem], Q2);
	// Ein += Trans(Q2) * edata[int(econ(amem,N))] * Q2;
	Add_AT_B_A(1.0, Ein, Q2, edata[int(econ(amem,N))]);
      }
      // prt_evv<N2>(Ein, "--- Ein", false);
      Ejn = 0;
      intersect_sorted_arrays(rowis, bmems, neibs_in_agg);
      // cout << " conections to b: "; prow(neibs_in_agg); cout << endl;
      for (auto bmem : neibs_in_agg) {
	ModQij(vdata[N], vdata[bmem], Q2);
	// Ejn += Trans(Q2) * edata[int(econ(bmem,N))] * Q2;
	Add_AT_B_A(1.0, Ejn, Q2, edata[int(econ(bmem,N))]);
      }
      // prt_evv<N2>(Ejn, "--- Ejn", false);
      Esum = Ein + Ejn;
      // prt_evv<N2>(Esum, "--- Esum", false);
      if constexpr(is_same<TMU, double>::value) { CalcInverse(Esum); }
      else { CalcPseudoInverse<mat_traits<TMU>::HEIGHT>(Esum); } // CalcInverse(Esum); // !! pesudo inv for 3d elast i think !!
      // addE = Ein * Esum * Ejn;
      addE = TripleProd(Ein, Esum, Ejn);
      // prt_evv<N2>(addE, "--- addE", false);
      ModQHh(h_data, vdata[N], Q2); // QHN
      // mat += 2 * Trans(Q2) * addE * Q2;
      Add_AT_B_A (2.0, mat, Q2, addE);
      // TM update = 2 * Trans(Q2) * addE * Q2;
      // prt_evv<N2>(update, "--- update", false);
      // cout << " UPDATED MAT EVS: " << endl;
      // prt_evv<N2>(mat, "intermed. emat");
    }; // add_neib_edge
    /** Calculate the strength of connection between two agglomerates
	(or an agglomerate and a vertex, or two vertices) **/
    auto CalcSOC = [&](auto ca, FlatArray<int> memsa, const auto & diaga,
		       auto cb, FlatArray<int> memsb, const auto & diagb,
		       bool common_neib_boost) LAMBDA_INLINE {
      // bool doout = ( (ca == 154) || (cb == 154) ) && (NV == 192);
      // if ( ( (ca == 154) || (cb == 154) ) && (NV == 192) )
	// { common_neib_boost = false; }
      // cout << " calc SOC, ca " << ca << " with " << memsa.Size() << " mems " << endl;
      // prow2(memsa, cout); cout << endl;
      // prt_evv<N>(diaga, "diag ca", false);
      // cout << " diag: " << diaga << endl;
      // cout << " v data " << vdata[ca] << endl;
      // cout << " calc SOC, cb " << cb << " with " << memsb.Size() << " mems " << endl;
      // prow2(memsb, cout); cout << endl;
      // cout << " diag: " << diagb << endl;
      // cout << " v data " << vdata[cb] << endl;
      // if (doout) {
      // prt_evv<N>(diagb, "diag cb", false);
      // }
      const auto memsas = memsa.Size();
      const auto memsbs = memsb.Size();
      bool vv_case = (memsas == memsbs) && (memsas == 1);
      TVD H_data = FACTORY::CalcMPData(vdata[ca], vdata[cb]);
      ModQHh(H_data, vdata[ca], Q);
      // Aaa = Trans(Q) * diaga * Q;
      Aaa = AT_B_A(Q, diaga);
      ModQHh(H_data, vdata[cb], Q);
      // Abb = Trans(Q) * diagb * Q;
      Abb = AT_B_A(Q, diagb);
      double max_wt = 0;
      int NA = 1, NB = 1;
      common_neibs.SetSize0();
      if ( vv_case ) {// simple vertex-vertex case
	int eid = int(econ(ca, cb));
	emat = edata[eid]; max_wt = 1;
	intersect_sorted_arrays(econ.GetRowIndices(ca), econ.GetRowIndices(cb), common_neibs);
	NA = econ.GetRowIndices(ca).Size();
	NB = econ.GetRowIndices(cb).Size();
	// prt_evv<N>(emat, "no boost emat", false);
	// cout << " boost from neibs: "; prow(common_neibs); cout << endl;
	// prt_evv<N>(emat, "pure emat");
	if (common_neib_boost) { // on the finest level, this is porbably 0 in most cases, but still expensive
	  for (auto v : common_neibs) {
	    add_neib_edge(H_data, memsa, memsb, v, emat);
	    // prt_evv<N>(emat, string("emat with boost from") + to_string(v), false);
	  }
	}
      }
      else { // find all edges connecting the agglomerates and most shared neibs
	emat = 0;
	for (auto amem : memsa) { // add up emat contributions
	  intersect_sorted_arrays(econ.GetRowIndices(amem), memsb, common_neibs);
	  for (auto bmem : common_neibs) {
	    int eid = int(econ(amem, bmem));
	    TVD h_data = FACTORY::CalcMPData(vdata[amem], vdata[bmem]);
	    ModQHh (H_data, h_data, Q);
	    // emat += Trans(Q) * edata[eid] * Q;
	    Add_AT_B_A(1.0, emat, Q, edata[eid]);
	    max_wt = max2(max_wt, ecw[eid]);
	  }
	}
	NA = econ.GetRowIndices(ca).Size(); // (!) not really correct, need # of all edges from ca to outside ca
	NB = econ.GetRowIndices(cb).Size(); // (!) not really correct, need # of all edges from cb to outside cb
	// prt_evv<N>(emat, "pure emat");
	if (common_neib_boost) {
	  // if ( (ca != 5) || (cb != 62) ) { // contribs from common neighbors
	  auto get_all_neibs = [&](auto mems, auto & ntab, auto & out) LAMBDA_INLINE {
	    ntab.SetSize0(); ntab.SetSize(mems.Size());
	    for (auto k : Range(mems))
	      { ntab[k].Assign(econ.GetRowIndices(mems[k])); }
	    merge_arrays(ntab, out, [&](const auto & i, const auto & j) LAMBDA_INLINE { return i < j; });
	  };
	  // cout << " memsa " << memsa.Size(); cout << " "; prow(memsa); cout << endl;
	  get_all_neibs(memsa, neib_tab, aneibs); NA = aneibs.Size();
	  get_all_neibs(memsb, neib_tab, bneibs); NB = bneibs.Size();
	  intersect_sorted_arrays(aneibs, bneibs, common_neibs); // can contain members of a/b
	  // prt_evv<N>(emat, "no boost emat", false);
	  // cout << " boost from neibs: ";
	  for (auto N : common_neibs) {
	    auto pos = find_in_sorted_array(N, memsa);
	    if (pos == -1) {
	      pos = find_in_sorted_array(N, memsb);
	      if (pos == -1) {
		// cout << N << " ";
		add_neib_edge(H_data, memsa, memsb, N, emat);
	      }
	    }
	  }
	  // cout << endl;
	}
      }
      // prt_evv<N>(Aaa, "Ai", false);
      // prt_evv<N>(Abb, "Aj", false);
      // prt_evv<N>(emat, "emat", false);
      double mmev = -1;
      if (geom) {
	// auto mmev_h = MIN_EV_HARM(Aaa, Abb, emat);
	// cout << " harm " << mmev_h << endl;
	// FACTORY::CalcQs(vdata[ca], vdata[cb], Qij, Qji);
	ModQs(vdata[ca], vdata[cb], Qij, Qji);
	mmev = MIN_EV_FG(Aaa, Abb, Qij, Qji, emat);
	// cout << " geo " << mmev << " , harm " << mmev_h << endl;
	// mmev *= sqrt(NA*NB);
	mmev *= sqrt(max(NA, NB)/min(NA, NB));
	// cout << " geo, neib adjusted " << mmev << endl;
      }
      else {
	mmev = MIN_EV_HARM(Aaa, Abb, emat);
	// cout << " harm " << mmev << endl;
	// FACTORY::CalcQs(vdata[ca], vdata[cb], Qij, Qji);
	// auto mmev_g = MIN_EV_FG(Aaa, Abb, Qij, Qji, emat);
	// cout << " geo " << mmev_g << " , harm " << mmev << endl;
      }
      double vw0 = get_vwt(ca);
      double vw1 = get_vwt(cb);
      double maxw = max(vw0, vw1);
      double minw = min(vw0, vw1);
      double fac = (fabs(maxw) < 1e-15) ? 1.0 : minw/maxw;
      // cout << " mmev fac ret " << mmev << " * " << fac << " = " <<  fac * min2(mmev, max_wt) << endl;
      return fac * min2(mmev, max_wt);
      // return fac * mmev;
    }; // CalcSOC

    Array<int> dummya(1), dummyb(1);
    auto CalcSOC_av = [&](const auto & agg, auto v, auto cnb) LAMBDA_INLINE {
      dummyb = v;
      return CalcSOC(agg.center(), agg.members(), agg_diag[agg.id], v, dummyb, repl_diag[v], cnb);
    };

    auto CalcSOC_aa = [&](const auto & agga, const auto & aggb, auto cnb) LAMBDA_INLINE {
      return CalcSOC(agga.center(), agga.members(), agg_diag[agga.id],
		     aggb.center(), aggb.members(), agg_diag[aggb.id],
		     cnb);
    };

    auto add_v_to_agg = [&](auto & agg, auto v) LAMBDA_INLINE {
      /** 
	  Variant 1:
 	     diag = sum_{k in agg, j not in agg} Q(C->mid(k,j)).T Ekj Q(C->mid(k,j))
	     So, when we add a new member, we have to remove contributions from edges that are now
	     internal to the agglomerate, and add new edges instead!
	     !!! Instead we add Q(C->j).T Ajj Q(C->j) and substract Q(C->mid(k,j)).T Ekj Q(C->mid(k,j)) TWICE !!
	     We know that any edge from a new agg. member to the agg is IN-EQC, so we KNOW we LOCALLY have the edge
	     and it's full matrix.
	     However, we might be missing some new out-of-agg connections from the other vertex. So we just add the entire
	     diag and substract agg-v twice.
	  Probably leads to problems with "two-sided" conditions like:
	     alpha_ij * (aii+ajj) / (aii*ajj)
	  So we would need something like:
	     alpha_ij / sqrt(aii*ajj)
	  Alternatively, we take a two-sided one, and use
	    CW(agg, v) = min( max_CW_{n in Agg}(n,v), CW_twoside(agg, v))
	    [ So agglomerating vertices can only decrease the weight of edges but never increase it ]
      **/
      // cout << "--- add vertex " << v << " to agg " << agg.id << " around " << agg.center() << endl;
      auto vneibs = econ.GetRowIndices(v);
      auto eids = econ.GetRowValues(v);
      double fac = 2;
      if ( (!geom) && (agg.members().Size() == 1) )
	{ fac = 1.5; } // leave half the contrib of first edge in
      for (auto j : Range(vneibs)) {
	auto neib = vneibs[j];
	auto pos = find_in_sorted_array(neib, agg.members());
	if (pos != -1) {
	  int eid = int(eids[j]);
	  TVD mid_vd = FACTORY::CalcMPData(vdata[neib], vdata[v]);
	  ModQHh(vdata[agg.center()], mid_vd, Q); // Qij or QHh??
	  agg_diag[agg.id] -= fac * Trans(Q) * edata[eid] * Q;
	  // agg_diag[agg.id] -= 2 * Trans(Q) * edata[eid] * Q;
	}
      }
      // FACTORY::CalcQij(vdata[agg.center()], vdata[v], Q);
      ModQHh(vdata[agg.center()], vdata[v], Q);
      // cout << "---- add " << Trans(Q) * repl_diag[v] * Q << endl;
      agg_diag[agg.id] += 1.0 * Trans(Q) * repl_diag[v] * Q;
      // cout << "---- agg.diag now " << agg_diag[agg.id] << endl;
      agg.AddSort(v);
    }; // add_v_to_agg

    Array<int> neib_ecnt(30), qsis(30);
    Array<double> ntraces(30);
    auto init_agglomerate = [&](auto v, auto v_eqc, bool force) LAMBDA_INLINE {
      // cout << endl << " INIT AGG FOR " << v << " from eqc " << v_eqc << " (force " << force << ")" << endl;
      auto agg_nr = agglomerates.Size();
      agglomerates.Append(Agglomerate(v, agg_nr)); // TODO: does this do an allocation??
      agg_diag.Append(repl_diag[v]);
      marked.SetBit(v);
      v_to_agg[v] = agg_nr;
      dist2agg[v] = 0;
      auto& agg = agglomerates.Last();
      auto & aggd = agg_diag[agg_nr];
      auto neibs_v = econ.GetRowIndices(v);
      int cnt_mems = 1;
      auto may_check_neib = [&](auto N) -> bool LAMBDA_INLINE
      { return (!marked.Test(N)) && eqa_to_eqb(M.template GetEqcOfNode<NT_VERTEX>(N), v_eqc) ;  };
      /** First, try to find ONE neib which we can add. Heuristic: Try descending order of trace(emat).
	  I think we should give common neighbour boost here. **/
      auto neibs_e = econ.GetRowValues(v);
      ntraces.SetSize0(); ntraces.SetSize(neibs_v.Size());
      qsis.SetSize0(); qsis.SetSize(neibs_v.Size());
      int first_N_ind = -1;
      for (auto k : Range(ntraces)) {
	ntraces[k] = calc_trace(edata[int(neibs_e[k])]);
	qsis[k] = k;
      }
      QuickSortI(ntraces, qsis, [&] (const auto & i, const auto & j) LAMBDA_INLINE { return i > j; });
      for (auto j : Range(ntraces)) {
	auto indN = qsis[j];
	auto N = neibs_v[indN];
	// cout << " try for first neib " << N << " with trace " << ntraces[indN] << endl;
	if ( may_check_neib(N) ) {
	  dummyb = N;
	  auto soc = CalcSOC (v, agg.members(), aggd,
			      N, dummyb, repl_diag[N],
			      true); // neib boost probably worth it ...
	  // cout << " soc " << soc << endl;
	  if (soc > MIN_ECW) {
	    // cout << "FIRST N " << N << " from eqc " << M.template GetEqcOfNode<NT_VERTEX>(N) << ", with soc " << soc << endl;
	    first_N_ind = indN;
	    v_to_agg[N] = agg_nr;
	    dist2agg[N] = 1;
	    marked.SetBit(N);
	    add_v_to_agg (agg, N);
	    cnt_mems++;
	    break;
	  }
	}
      }
      if (first_N_ind != -1) { // if we could not add ANY neib, nohing has changed
	/** We perform a greedy strategy: Keep adding neighbours have the highest number of connections
	    leading into the aggregate. If not dist2, only check neighbours common to the two first verts in the aggregate,
	    otherwise check all neibs of v.
	    Its not perfect - if i check a neib, and do not add it, but later on add a common neib, i could potentially
	    have added it in the first place. On the other hand - di I WANT to re-check all the time?
	    Can I distinguish between "weak" and "singular" connections? Then I could only re-check the singular ones? **/
	int qss = neibs_v.Size();
	neib_ecnt.SetSize0(); neib_ecnt.SetSize(qss); neib_ecnt = 1; // per definition every neighbour has one edge
	qsis.SetSize0(); qsis.SetSize(qss); qsis = -1; qss = 0;
	for (auto j : Range(neibs_v))
	  if (!may_check_neib(neibs_v[j]))
	    { neib_ecnt[j] = -1; }
	  else
	    { qsis[qss++] = j; }
	int first_valid_ind = 0;
	/** inc edge count for common neibs of v and first meber**/
	iterate_intersection(econ.GetRowIndices(neibs_v[first_N_ind]), neibs_v,
			     [&](auto i, auto j) {
			       if (neib_ecnt[j] != -1)
				 { neib_ecnt[j]++; }
			     });
	/**  lock out all neibs of v that are not neibs of N **/
	int inc_fv = 0;
	if ( !dist2 ) {
	  iterate_anotb(neibs_v, econ.GetRowIndices(neibs_v[first_N_ind]),
			[&](auto inda) LAMBDA_INLINE {
			  // cout << " inda " << inda << endl;
			  if (neib_ecnt[inda] != -1) {
			    neib_ecnt[inda] = -1;
			    // cout << "(dist1) remove " << neibs_v[inda] << endl;
			    inc_fv++;
			  }
			});
	}
	QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });
	first_valid_ind += inc_fv;
	qss -= inc_fv;
	// cout << " init " << first_valid_ind << " " << qss << endl;
	// prow(neibs_v); cout << endl;
	// prow(neib_ecnt); cout << endl;
	// prow(qsis); cout << endl;
	while(qss>0) {
	  auto n_ind = qsis[first_valid_ind + qss - 1]; 
	  auto N = neibs_v[n_ind]; dummyb = N;
	  auto soc = CalcSOC (v, agg.members(), aggd,
			      N, dummyb, repl_diag[N],
			      true); // neib boost should not be necessary
	  // cout << " check " << N << " #con " << neib_ecnt[n_ind] << ", soc " << soc << ", ok ? " << bool(soc>MIN_ECW) << " // " << n_ind << " " << first_valid_ind << " " << qss << endl;
	  qss--; // done with this neib
	  neib_ecnt[n_ind] = -1; // just to check if i get all - i should not access this anymore anyways
	  if (soc > MIN_ECW) { // add N to agg, update edge counts and re-sort neibs
	    v_to_agg[N] = agg_nr;
	    dist2agg[N] = 1;
	    marked.SetBit(N);
	    add_v_to_agg (agg, N); cnt_mems++;
	    // cout << " add " << N << " from eqc " << M.template GetEqcOfNode<NT_VERTEX>(N) << endl;
	    // cout << " neibs of N " << N << " are: "; prow(econ.GetRowIndices(N)); cout << endl;
	    // cout << " neibs_v: "; prow(neibs_v); cout << endl;
	    // cout << " old ecnt :"; prow(neib_ecnt); cout << endl;
	    iterate_intersection(econ.GetRowIndices(N), neibs_v, // all neibs of v and N now have an additional edge into the agglomerate
				 [&](auto inda, auto indb) LAMBDA_INLINE {
				   if (neib_ecnt[indb] != -1)
				     { neib_ecnt[indb]++; }
				 });
	    // cout << " new ecnt :"; prow(neib_ecnt); cout << endl;
	    // int inc_fv = 0;
	    // if ( (cnt_mems == 2) && (!dist2) ) { // lock all neibs of v that are not neibs of N
	    //   // cout << " can stay: "; prow(econ.GetRowIndices(N)); cout << endl;
	    //   iterate_anotb(neibs_v, econ.GetRowIndices(N),
	    // 		    [&](auto inda) LAMBDA_INLINE {
	    // 		      // cout << " inda " << inda << endl;
	    // 		      if (neib_ecnt[inda] != -1) {
	    // 			neib_ecnt[inda] = -1;
	    // 			// cout << "(dist1) remove " << neibs_v[inda] << endl;
	    // 			inc_fv++;
	    // 		      }
	    // 		    });
	    // }
	    // // cout << " first " << first_valid_ind << " qss " << qss << ", qsis: "; prow2(qsis); cout << endl;
	    QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });
	    // first_valid_ind += inc_fv;
	    // qss -= inc_fv;
	    // cout << "sorted; first " << first_valid_ind << " qss " << qss << ", qsis: "; prow2(qsis); cout << endl;
	  }
	  // cout << " after " << N << " " << n_ind << " " << first_valid_ind << " " << qss << endl;
	  // cout << "neibs: "; prow(neibs_v); cout << endl;
	  // cout << "#edge: "; prow(neib_ecnt); cout << endl;
	  // cout << "qsis:  "; prow(qsis); cout << endl;
	} // while(qss)
      } // first_N_ind != -1
      // cout << "agg has " << cnt_mems << " mems, would like at least " << MIN_NEW_AGG_SIZE << endl;
      if ( force || (cnt_mems >= MIN_NEW_AGG_SIZE) ) {
	// cout << " agg accepted! " << endl;
	// cout << agg << endl;
	return true;
      }
      else { // remove the aggregate again - 
    	for (auto M : agg.members()) {
    	  v_to_agg[M] = -1;
    	  dist2agg[M] = -1;
    	  marked.Clear(M);
    	}
    	agglomerates.SetSize(agg_nr);
    	agg_diag.SetSize(agg_nr);
    	return false;
      }
    }; // init_agglomerate (..)


    /** Should I make a new agglomerate around v? **/
    auto check_v = [&](auto v) LAMBDA_INLINE {
      auto myeq = M.template GetEqcOfNode<NT_VERTEX>(v);
      if ( (marked.Test(v)) || (!eqc_h.IsMasterOfEQC(myeq)) ) // only master-verts !!
	{ return false; }
      else {
	auto neibs = econ.GetRowIndices(v);
	for (auto n : neibs) {
	  if ( (marked.Test(n)) && eqa_to_eqb(myeq, M.template GetEqcOfNode<NT_VERTEX>(n)) )  {
	    auto n_agg_nr = v_to_agg[n];
	    if (n_agg_nr != -1) // can still do this later ...
	      { return false; }
	    // if (n_agg_nr != -1) { // could be dirichlet
	    //   auto soc = CalcSOC_av(agglomerates[n_agg_nr], v);
	    //   if (soc > MIN_ECW) // 
	    // 	{ return false; }
	    // }
	  }
	}
	return init_agglomerate(v, myeq, false);
      }
    };

    t1.Start();

    /** Deal with dirichlet vertices **/
    if (free_verts != nullptr) {
      for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
	if (!free_verts->Test(k))
	  { marked.SetBit(k); }
      }
      // cout << "DIRICHLET VERTS: " << endl;
      // for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
      // 	if (!free_verts->Test(k)) {
      // 	  cout << k << " " << endl;
      // 	}
      // }
    }

    /** Calc replacement matrix diagonals **/
    tdiag.Start();

    M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
	ModQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
	const auto & em = edata[edge.id];
	// repl_diag[edge.v[0]] += Trans(Qij) * em * Qij;
	Add_AT_B_A(1.0, repl_diag[edge.v[0]], Qij, em);
	// repl_diag[edge.v[1]] += Trans(Qji) * em * Qji;
	Add_AT_B_A(1.0, repl_diag[edge.v[1]], Qji, em);
      }, true); // only master, we cumulate this afterwards
    M.template AllreduceNodalData<NT_VERTEX>(repl_diag, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    tdiag.Stop();


    /** Calc initial edge weights. TODO: I dont think i need this anymore at all! **/
    tecw.Start();
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	dummya = edge.v[0]; dummyb = edge.v[1];
	// ecw[edge.id] = CalcSOC(edge.v[0], dummya, repl_diag[edge.v[0]],
	// 		       edge.v[1], dummyb, repl_diag[edge.v[1]],
	// 		       true);
	ecw[edge.id] = 1;
      });
    tecw.Stop();

    size_t n_strong_e = 0;
    M.template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_strong_e++; } }, true);
    double s_e_per_v = (M.template GetNN<NT_VERTEX>() == 0) ? 0 : 2 * double(n_strong_e) / double(M.template GetNN<NT_VERTEX>());
    size_t approx_nagg = max2(size_t(1), size_t(NV / ( 1 + s_e_per_v )));
    agglomerates.SetSize(1.2 * approx_nagg); agglomerates.SetSize0();
    agg_diag.SetSize(1.2 * approx_nagg); agg_diag.SetSize0();

    t1.Stop(); t2.Start();

    /** Iterate through vertices and start new agglomerates if the vertex is at least at distance 1 from
	any agglomerate (so distance 2 from any agg. center) **/
    {
      Array<int> vqueue(M.template GetNN<NT_VERTEX>()); vqueue.SetSize0();
      BitArray checked(M.template GetNN<NT_VERTEX>()); checked.Clear();
      BitArray queued(M.template GetNN<NT_VERTEX>()); queued.Clear();
      size_t nchecked = 0; const auto NV = M.template GetNN<NT_VERTEX>();
      int cntq = 0, cntvr = NV-1;
      int vnum;
      while (nchecked < NV) {
	/** if there are any queued vertices, handle them first, otherwise take next vertex by reverse counting
	    ( ex-vertices have higher numbers) **/
	bool from_q = cntq < vqueue.Size();
	if (from_q)
	  { vnum = vqueue[cntq++]; }
	else
	  { vnum = cntvr--; }
	if (!checked.Test(vnum)) {
	  // cout << " from queue ? " << from_q << endl;
	  bool newagg = check_v(vnum);
	  checked.SetBit(vnum);
	  nchecked++;
	  if (newagg) { // enqueue neibs of neibs of the agg (if they are not checked, queued or at least marked yet)
	    const auto & newagg = agglomerates.Last();
	    int oldqs = vqueue.Size();
	    for (auto mem : newagg.members()) {
	      auto mem_neibs = econ.GetRowIndices(mem);
	      for (auto x : mem_neibs) {
		auto y = econ.GetRowIndices(x);
		for (int jz = int(y.Size()) - 1; jz>= 0; jz--) { // enqueue neibs of neibs - less local ones first
		  auto z = y[jz];
		  if (  (!marked.Test(z)) && (!checked.Test(z)) && (!queued.Test(z)) )
		    { vqueue.Append(z); queued.SetBit(z); }
		}
	      }
	    }
	    int newqs = vqueue.Size();
	    if (newqs > oldqs + 1) // not sure ?
	      { QuickSort(vqueue.Part(oldqs, (newqs-oldqs)), [&](auto i, auto j) { return i>j; }); }
	  }
	}
      }
    }

    // {
    //  //   cout << endl << " FIRST loop done " << endl;
    //  cout << "frac marked: " << double(marked.NumSet()) / marked.Size() << endl;
    //  cout << " INTERMED agglomerates : " << agglomerates.Size() << endl;
    //  cout << agglomerates << endl;
    //  Array<int> naggs;
    //  auto resize_to = [&](auto i) {
    // 	auto olds = naggs.Size();
    // 	if (olds < i) {
    // 	  naggs.SetSize(i);
    // 	  for (auto j : Range(olds, i))
    // 	    { naggs[j] = 0; }
    // 	}
    //  };
    //  for (const auto & agg : agglomerates) {
    // 	auto ags = agg.members().Size();
    // 	resize_to(1+ags);
    // 	naggs[ags]++ ;
    //  }
    //  cout << " INTERMED agg size distrib: "; prow2(naggs); cout << endl;
    // }

    t2.Stop(); t3.Start();

    /** Assign left over vertices to some neighbouring agglomerate, or, if not possible, start a new agglomerate with them.
	Also try to weed out any dangling vertices. **/
    Array<int> neib_aggs(20), notake(20), index(20);
    Array<double> na_soc(20);
    M.template ApplyEQ<NT_VERTEX>([&] (auto eqc, auto v) LAMBDA_INLINE {
	// TODO: should do this twice, once with max_dist 1, once 2 (second round, steal verts from neibs ?)
	auto neibs = econ.GetRowIndices(v);
	if (!marked.Test(v)) {
	  if (neibs.Size() == 1) {  // Check for hanging vertices
	    // cout << " vertex " << v << " unmarked in loop 2 " << endl;
	    // Collect neighbouring agglomerates we could append ourselfs to
	    auto N = neibs[0];
	    auto neib_agg_id = v_to_agg[neibs[0]];
	    if (neib_agg_id != -1) { // neib is in an agglomerate - I must also be master of N [do not check marked - might be diri!]
	      auto & neib_agg = agglomerates[neib_agg_id];
	      auto N_eqc = M.template GetEqcOfNode<NT_VERTEX>(N);
	      bool could_add = eqa_to_eqb(eqc, N_eqc);
	      bool can_add = (dist2agg[N] == 0); // (neib_agg.center() == N); // I think in this case this MUST be ok ??
	      if ( could_add && (!can_add) ) { // otherwise, re-check SOC (but this should have been checked already)
		auto soc = CalcSOC_av (neib_agg, v, true);
		if (soc > MIN_ECW)
		  { can_add = true; }
	      }
	      can_add &= could_add;
	      if (can_add) { // lucky!
		// cout << " add hanging " << v << " to agg of " << N << " = " << neib_agg << endl;
		marked.SetBit(v);
		v_to_agg[v] = neib_agg_id;
		dist2agg[v] = 1 + dist2agg[N]; // must be correct - N is the only neib
		add_v_to_agg (neib_agg, v);
	      }
	      else { // unfortunate - new single agg!
		init_agglomerate(v, eqc, true);
	      }
	    }
	    else { // neib is not in an agg - force start a new one at neib and add this vertex (which must be OK!)
	      auto N_eqc = M.template GetEqcOfNode<NT_VERTEX>(N);
	      if ( (eqc_h.IsMasterOfEQC(N_eqc)) && (eqa_to_eqb(eqc, N_eqc) ) ) { // only if it is OK eqc-wise
		init_agglomerate(N, N_eqc, true); // have to force new agg even if we dont really want to ...
		auto new_agg_id = v_to_agg[N];
		if (!marked.Test(v)) { // might already have been added by init_agg, but probably not
		  marked.SetBit(v);
		  v_to_agg[v] = new_agg_id;
		  dist2agg[v] = 1;
		  add_v_to_agg(agglomerates[new_agg_id], v);
		}
	      }
	      else { // unfortunate - new single agg!
		init_agglomerate(v, eqc, true);
	      }
	    }
	  }
	  else {
	    neib_aggs.SetSize0();
	    notake.SetSize0();
	    na_soc.SetSize0();
	    for (auto n : neibs) {
	      if ( (marked.Test(n)) && (dist2agg[n] <= (dist2 ? 2 : 1))) { // max_dist 2 maybe, idk ??
		auto agg_nr = v_to_agg[n];
		if (agg_nr != -1) { // can this even be -1 if "marked" is set ??
		  auto & n_agg = agglomerates[agg_nr];
		  if ( eqa_to_eqb(eqc, M.template GetEqcOfNode<NT_VERTEX>(n_agg.center())) &&
		       (!notake.Contains(agg_nr)) && (!neib_aggs.Contains(agg_nr)) ) {
		    // cout << " calc des with " << n_agg << endl;
		    // cout << " calc soc " << v << " to agg " << agg_nr << ", " << n_agg << endl;
		    auto soc = CalcSOC_av(n_agg, v, true); // use neib_boost here - we want as few new aggs as possible
		    // cout << " soc " << soc << endl;
		    if (soc > MIN_ECW) {
		      /** Desirability of adding v to A:
			  1 / (1 + |A|) * Sum_{k in A} alpha(e_vk)/dist[k]
			  So desirability grows when we have many strong connections to an agglomerate and decreases
			  the farther away from the center of A we are and the larger A already is. **/
		      intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
		      // for (auto k : neibs_in_agg)
		      // { cout << "[" << k << " | " << dist2agg[k] << " | " << ecw[int(econ(v,k))] << "]" << endl; }
		      double mindist = 20;
		      double des = 0;
		      for (auto k : neibs_in_agg)
			{ des += ecw[int(econ(v,k))] / ( 1 + dist2agg[k] );
			  mindist = min2(mindist, double(dist2agg[k]));
			}
		      // cout << " init des " << des << endl;
		      // des = des * (soc/MIN_ECW) / (n_agg.members().Size());
		      des = (soc/MIN_ECW) / (mindist * n_agg.members().Size());
		      // cout << "soc/des " << soc << " " << des << endl;
		      neib_aggs.Append(agg_nr);
		      na_soc.Append(des);
		    }
		    else // just so we do not compute SOC twice for no reason
		      { notake.Append(agg_nr); }
		  }
		}
	      }
	    }
	    if (neib_aggs.Size()) { // take the most desirable neib
	      // cout << " decide between " << neib_aggs.Size() << endl;
	      // 							cout << " neib ses "; for(auto n : neib_aggs) { cout << agglomerates[n].members().Size() << " "; }
	      // 																		    cout << endl;
	      auto mi = ind_of_max(na_soc);
	      auto agg_nr = neib_aggs[mi]; auto& n_agg = agglomerates[agg_nr];
	      // cout << " decided for " << n_agg.members().Size() << endl;
	      add_v_to_agg(n_agg, v);
	      marked.SetBit(v);
	      v_to_agg[v] = agg_nr;
	      intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
	      int mindist = 1000;
	      for (auto k : neibs_in_agg)
		{ mindist = min2(mindist, dist2agg[k]); }
	      dist2agg[v] = 1 + mindist;
	      // cout << " added to agg " << n_agg << endl;
	    }
	    else {
	      init_agglomerate(v, eqc, true); // have to force new agg even if we dont really want to ...
	      // Maybe we should check if we should steal some vertices from surrounding aggs, at least those with
	      // dist >= 2. Maybe init_agg(v, eqc, steal=true) ??
	    }
	  } // neibs.Size > 1
	} // if (!marked.Test(v))
      }, true); // also only master verts!

    t3.Stop();

  } // Agglomerator::FormAgglomerates_impl


  template<class FACTORY>
  void Agglomerator<FACTORY> :: FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    if (settings.robust) /** cheap, but not robust for some corner cases **/
      { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
    else /** (much) more expensive, but also more robust **/
      { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
  } // Agglomerator::FormAgglomerates


  // template<class FACTORY>
  // void Agglomerator<FACTORY> :: FormAgglomeratesOld (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  // {

  //   static Timer t("FormAgglomerates"); RegionTimer rt(t);

  //   static Timer tdiag("FormAgglomerates - diags");
  //   static Timer tecw("FormAgglomerates - init. ecw");
  //   static Timer t1("FormAgglomerates - 1");
  //   static Timer t2("FormAgglomerates - 2");
  //   static Timer t3("FormAgglomerates - 3");

  //   t1.Start();
    
  //   constexpr int N = mat_traits<TM>::HEIGHT;

  //   const auto & M = *mesh; M.CumulateData();

  //   const auto & eqc_h = *M.GetEQCHierarchy();
  //   auto comm = eqc_h.GetCommunicator();

  //   const auto NV = M.template GetNN<NT_VERTEX>();

  //   const auto & econ = *M.GetEdgeCM();
  //   FlatArray<TVD> vdata = get<0>(mesh->Data())->Data();
  //   FlatArray<TM> edata = get<1>(mesh->Data())->Data();

  //   const double MIN_ECW = settings.edge_thresh;
  //   const bool dist2 = settings.dist2;
  //   const bool geom = settings.cw_geom;
  //   const int MIN_NEW_AGG_SIZE = 2;

  //   // cout << " agg coarsen params: " << endl;
  //   // cout << "min_ecw = " << MIN_ECW << endl;
  //   // cout << "dist2 = " << dist2 << endl;
  //   // cout << "geom = " << geom << endl;
  //   // cout << " min new agg size = " << MIN_NEW_AGG_SIZE << endl;
  //   // cout << " neibs per v " << double(2 * M.template GetNN<NT_EDGE>())/NV << endl;

  //   auto get_vwt = [&](auto v) {
  //     if constexpr(is_same<TVD, double>::value) { return vdata[v]; }
  //     else { return vdata[v].wt; }
  //   };

  //   Array<TM> agg_diag;

  //   /** replacement-matrix diagonals **/
  //   Array<TM> repl_diag(M.template GetNN<NT_VERTEX>()); repl_diag = 0;

  //   /** collapse weights for edges - we use these as upper bounds for weights between agglomerates **/
  //   Array<double> ecw(M.template GetNN<NT_EDGE>()); ecw = 0;

  //   /** vertex -> agglomerate map **/
  //   BitArray marked(M.template GetNN<NT_VERTEX>()); marked.Clear();
  //   Array<int> dist2agg(M.template GetNN<NT_VERTEX>()); dist2agg = -1;
  //   // Array<int> v_to_agg(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;
  //   v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;

  //   // cout << endl << "AGGLOMERATE, init data " << endl;

  //   // cout << " edata : " << endl; prow2(edata); cout << endl;

  //   // cout << " vdata : " << endl; prow2(vdata); cout << endl;

  //   // if (M.template GetNN<NT_VERTEX>() < 1000) {
  //   //   cout << endl << "AGGLOMERATE, econ " << endl;
  //   //   cout << econ << endl;
  //   // }

  //   // cout << "AGGLOMERATE, calc repl diag " << endl;

  //   tdiag.Start();

  //   TM Qij, Qji;
  //   M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
  // 	FACTORY::CalcQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
  // 	const auto & em = edata[edge.id];
  // 	// repl_diag[edge.v[0]] += Trans(Qij) * em * Qij;
  // 	AddTripleProd(1.0, repl_diag[edge.v[0]], Trans(Qij), em, Qij);
  // 	// repl_diag[edge.v[1]] += Trans(Qji) * em * Qji;
  // 	AddTripleProd(1.0, repl_diag[edge.v[1]], Trans(Qji), em, Qji);
  //     }, true); // only master, we cumulate this afterwards
  //   M.template AllreduceNodalData<NT_VERTEX>(repl_diag, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

  //   tdiag.Stop();

  //   // cout << "repl_diag: " << endl << repl_diag << endl;

  //   /**
  //      Calculate the strength of connection between two agglomerates
  //      (or an agglomerate and a vertex)
  //    **/
  //   TM emat, Q;
  //   Array<int> neibs_in_agg;

  //   TM Ein, Ejn, Esum, addE, Q2;
  //   auto add_neib_edge = [&](TVD h_data, const auto & amems, const auto & bmems, auto N, auto & mat) LAMBDA_INLINE {
  //     constexpr int N2 = mat_traits<TM>::HEIGHT;
  //     auto rowis = econ.GetRowIndices(N);
  //     // cout << " add neib " << N << endl;
  //     // cout << " amems "; prow(amems); cout << endl;
  //     // cout << " bmems "; prow(bmems); cout << endl;
  //     Ein = 0;
  //     intersect_sorted_arrays(rowis, amems, neibs_in_agg);
  //     // cout << " conections to a: "; prow(neibs_in_agg); cout << endl;
  //     for (auto amem : neibs_in_agg) {
  // 	FACTORY::CalcQij(vdata[N], vdata[amem], Q2);
  // 	Ein += Trans(Q2) * edata[int(econ(amem,N))] * Q2;
  //     }
  //     // prt_evv<N2>(Ein, "--- Ein", false);
  //     Ejn = 0;
  //     intersect_sorted_arrays(rowis, bmems, neibs_in_agg);
  //     // cout << " conections to b: "; prow(neibs_in_agg); cout << endl;
  //     for (auto bmem : neibs_in_agg) {
  // 	FACTORY::CalcQij(vdata[N], vdata[bmem], Q2);
  // 	Ejn += Trans(Q2) * edata[int(econ(bmem,N))] * Q2;
  //     }
  //     // prt_evv<N2>(Ejn, "--- Ejn", false);
  //     Esum = Ein + Ejn;
  //     // prt_evv<N2>(Esum, "--- Esum", false);
  //     if constexpr(is_same<TM, double>::value) { CalcInverse(Esum); }
  //     else { CalcPseudoInverse<mat_traits<TM>::HEIGHT>(Esum); } // CalcInverse(Esum); // !! pesudo inv for 3d elast i think !!
  //     addE = Ein * Esum * Ejn;
  //     // prt_evv<N2>(addE, "--- addE", false);
  //     FACTORY::CalcQHh(h_data, vdata[N], Q2); // QHN
  //     mat += 2 * Trans(Q2) * addE * Q2;
  //     // TM update = 2 * Trans(Q2) * addE * Q2;
  //     // prt_evv<N2>(update, "--- update", false);
  //     // cout << " UPDATED MAT EVS: " << endl;
  //     // prt_evv<N2>(mat, "intermed. emat");
  //   };
  //   size_t cnt_prtm = 0;
  //   Array<int> common_neibs(20), aneibs(20), bneibs(20);
  //   Array<FlatArray<int>> neib_tab(20);
  //   auto CalcSOC = [&](auto ca, FlatArray<int> memsa, const auto & diaga,
  // 		       auto cb, FlatArray<int> memsb, const auto & diagb,
  // 		       bool common_neib_boost) LAMBDA_INLINE {
  //     // bool doout = ( (ca == 154) || (cb == 154) ) && (NV == 192);
  //     // if ( ( (ca == 154) || (cb == 154) ) && (NV == 192) )
  // 	// { common_neib_boost = false; }
  //     // cout << " calc SOC, ca " << ca << " with " << memsa.Size() << " mems " << endl;
  //     // prow2(memsa, cout); cout << endl;
  //     // prt_evv<N>(diaga, "diag ca", false);

  //     // cout << " diag: " << diaga << endl;
  //     // cout << " v data " << vdata[ca] << endl;
  //     // cout << " calc SOC, cb " << cb << " with " << memsb.Size() << " mems " << endl;
  //     // prow2(memsb, cout); cout << endl;
  //     // cout << " diag: " << diagb << endl;
  //     // cout << " v data " << vdata[cb] << endl;
  //     // if (doout) {
  //     // prt_evv<N>(diagb, "diag cb", false);
  //     // }
  //     const auto memsas = memsa.Size();
  //     const auto memsbs = memsb.Size();
  //     bool vv_case = (memsas == memsbs) && (memsas == 1);
  //     TVD H_data = FACTORY::CalcMPData(vdata[ca], vdata[cb]);
  //     FACTORY::CalcQHh(H_data, vdata[ca], Q);
  //     TM Aaa = Trans(Q) * diaga * Q;
  //     FACTORY::CalcQHh(H_data, vdata[cb], Q);
  //     TM Abb = Trans(Q) * diagb * Q;
  //     double max_wt = 0;
  //     int NA = 1, NB = 1;
  //     common_neibs.SetSize0();
  //     if ( vv_case ) {// simple vertex-vertex case
  // 	int eid = int(econ(ca, cb));
  // 	emat = edata[eid]; max_wt = 1;
  // 	intersect_sorted_arrays(econ.GetRowIndices(ca), econ.GetRowIndices(cb), common_neibs);
  // 	NA = econ.GetRowIndices(ca).Size();
  // 	NB = econ.GetRowIndices(cb).Size();
  // 	// prt_evv<N>(emat, "no boost emat", false);
  // 	// cout << " boost from neibs: "; prow(common_neibs); cout << endl;
  // 	// prt_evv<N>(emat, "pure emat");
  // 	if (common_neib_boost) { // on the finest level, this is porbably 0 in most cases, but still expensive
  // 	  for (auto v : common_neibs) {
  // 	    add_neib_edge(H_data, memsa, memsb, v, emat);
  // 	    // prt_evv<N>(emat, string("emat with boost from") + to_string(v), false);
  // 	  }
  // 	}
  //     }
  //     else { // find all edges connecting the agglomerates and most shared neibs
  // 	emat = 0;
  // 	for (auto amem : memsa) { // add up emat contributions
  // 	  intersect_sorted_arrays(econ.GetRowIndices(amem), memsb, common_neibs);
  // 	  for (auto bmem : common_neibs) {
  // 	    int eid = int(econ(amem, bmem));
  // 	    TVD h_data = FACTORY::CalcMPData(vdata[amem], vdata[bmem]);
  // 	    FACTORY::CalcQHh (H_data, h_data, Q);
  // 	    emat += Trans(Q) * edata[eid] * Q;
  // 	    max_wt = max2(max_wt, ecw[eid]);
  // 	  }
  // 	}
  // 	NA = econ.GetRowIndices(ca).Size(); // (!) not really correct, need # of all edges from ca to outside ca
  // 	NB = econ.GetRowIndices(cb).Size(); // (!) not really correct, need # of all edges from cb to outside cb
  // 	// prt_evv<N>(emat, "pure emat");
  // 	if (common_neib_boost) {
  // 	  // if ( (ca != 5) || (cb != 62) ) { // contribs from common neighbors
  // 	  auto get_all_neibs = [&](auto mems, auto & ntab, auto & out) LAMBDA_INLINE {
  // 	    ntab.SetSize0(); ntab.SetSize(mems.Size());
  // 	    for (auto k : Range(mems))
  // 	      { ntab[k].Assign(econ.GetRowIndices(mems[k])); }
  // 	    merge_arrays(ntab, out, [&](const auto & i, const auto & j) LAMBDA_INLINE { return i < j; });
  // 	  };
  // 	  // cout << " memsa " << memsa.Size(); cout << " "; prow(memsa); cout << endl;
  // 	  get_all_neibs(memsa, neib_tab, aneibs); NA = aneibs.Size();
  // 	  get_all_neibs(memsb, neib_tab, bneibs); NB = bneibs.Size();
  // 	  intersect_sorted_arrays(aneibs, bneibs, common_neibs); // can contain members of a/b
  // 	  // prt_evv<N>(emat, "no boost emat", false);
  // 	  // cout << " boost from neibs: ";
  // 	  for (auto N : common_neibs) {
  // 	    auto pos = find_in_sorted_array(N, memsa);
  // 	    if (pos == -1) {
  // 	      pos = find_in_sorted_array(N, memsb);
  // 	      if (pos == -1) {
  // 		// cout << N << " ";
  // 		add_neib_edge(H_data, memsa, memsb, N, emat);
  // 	      }
  // 	    }
  // 	  }
  // 	  // cout << endl;
  // 	}
  //     }

  //     // prt_evv<N>(Aaa, "Ai", false);
  //     // prt_evv<N>(Abb, "Aj", false);
  //     // prt_evv<N>(emat, "emat", false);
	
  //     double mmev = -1;
  //     if (geom) {
  // 	// auto mmev_h = MIN_EV_HARM(Aaa, Abb, emat);
  // 	// cout << " harm " << mmev_h << endl;
  // 	FACTORY::CalcQs(vdata[ca], vdata[cb], Qij, Qji);
  // 	mmev = MIN_EV_FG(Aaa, Abb, Qij, Qji, emat);
  // 	// cout << " geo " << mmev << " , harm " << mmev_h << endl;
  // 	// mmev *= sqrt(NA*NB);
  // 	mmev *= sqrt(max(NA, NB)/min(NA, NB));
  // 	// cout << " geo, neib adjusted " << mmev << endl;
  //     }
  //     else {
  // 	mmev = MIN_EV_HARM(Aaa, Abb, emat);
  // 	// cout << " harm " << mmev << endl;
  // 	// FACTORY::CalcQs(vdata[ca], vdata[cb], Qij, Qji);
  // 	// auto mmev_g = MIN_EV_FG(Aaa, Abb, Qij, Qji, emat);
  // 	// cout << " geo " << mmev_g << " , harm " << mmev << endl;
  //     }

  //     double vw0 = get_vwt(ca);
  //     double vw1 = get_vwt(cb);
  //     double maxw = max(vw0, vw1);
  //     double minw = min(vw0, vw1);
  //     double fac = (fabs(maxw) < 1e-15) ? 1.0 : minw/maxw;

  //     // cout << " mmev fac ret " << mmev << " * " << fac << " = " <<  fac * min2(mmev, max_wt) << endl;

  //     return fac * min2(mmev, max_wt);
  //     // return fac * mmev;
  //   };

  //   Array<int> dummya(1), dummyb(1);
  //   auto CalcSOC_av = [&](const auto & agg, auto v, auto cnb) LAMBDA_INLINE {
  //     dummyb = v;
  //     return CalcSOC(agg.center(), agg.members(), agg_diag[agg.id], v, dummyb, repl_diag[v], cnb);
  //   };

  //   auto CalcSOC_aa = [&](const auto & agga, const auto & aggb, auto cnb) LAMBDA_INLINE {
  //     return CalcSOC(agga.center(), agga.members(), agg_diag[agga.id],
  // 		     aggb.center(), aggb.members(), agg_diag[aggb.id],
  // 		     cnb);
  //   };

  //   auto add_v_to_agg = [&](auto & agg, auto v) LAMBDA_INLINE {
  //     /** 
  // 	  Variant 1:
  // 	     diag = sum_{k in agg, j not in agg} Q(C->mid(k,j)).T Ekj Q(C->mid(k,j))
  // 	     So, when we add a new member, we have to remove contributions from edges that are now
  // 	     internal to the agglomerate, and add new edges instead!
  // 	     !!! Instead we add Q(C->j).T Ajj Q(C->j) and substract Q(C->mid(k,j)).T Ekj Q(C->mid(k,j)) TWICE !!
  // 	     We know that any edge from a new agg. member to the agg is IN-EQC, so we KNOW we LOCALLY have the edge
  // 	     and it's full matrix.
  // 	     However, we might be missing some new out-of-agg connections from the other vertex. So we just add the entire
  // 	     diag and substract agg-v twice.
  // 	  Probably leads to problems with "two-sided" conditions like:
  // 	     alpha_ij * (aii+ajj) / (aii*ajj)
  // 	  So we would need something like:
  // 	     alpha_ij / sqrt(aii*ajj)
  // 	  Alternatively, we take a two-sided one, and use
  // 	    CW(agg, v) = min( max_CW_{n in Agg}(n,v), CW_twoside(agg, v))
  // 	    [ So agglomerating vertices can only decrease the weight of edges but never increase it ]
  //     **/
  //     // cout << "--- add vertex " << v << " to agg " << agg.id << " around " << agg.center() << endl;
  //     auto vneibs = econ.GetRowIndices(v);
  //     auto eids = econ.GetRowValues(v);
  //     double fac = 2;
  //     if ( (!geom) && (agg.members().Size() == 1) )
  // 	{ fac = 1.5; } // leave half the contrib of first edge in
  //     for (auto j : Range(vneibs)) {
  // 	auto neib = vneibs[j];
  // 	auto pos = find_in_sorted_array(neib, agg.members());
  // 	if (pos != -1) {
  // 	  int eid = int(eids[j]);
  // 	  TVD mid_vd = FACTORY::CalcMPData(vdata[neib], vdata[v]);
  // 	  FACTORY::CalcQHh(vdata[agg.center()], mid_vd, Q); // Qij or QHh??

  // 	  // agg_diag[agg.id] -= 2 * Trans(Q) * edata[eid] * Q;
  // 	  agg_diag[agg.id] -= fac * Trans(Q) * edata[eid] * Q;

  // 	  // this 
  // 	  // agg_diag[agg.id] -= 2 * Trans(Q) * edata[eid] * Q;

  // 	}
  //     }
  //     // FACTORY::CalcQij(vdata[agg.center()], vdata[v], Q);
  //     FACTORY::CalcQHh(vdata[agg.center()], vdata[v], Q);
  //     // cout << "---- add " << Trans(Q) * repl_diag[v] * Q << endl;
  //     agg_diag[agg.id] += 1.0 * Trans(Q) * repl_diag[v] * Q;
  //     // cout << "---- agg.diag now " << agg_diag[agg.id] << endl;
  //     agg.AddSort(v);
  //   };

  //   tecw.Start();

  //   M.template Apply<NT_EDGE>([&](const auto & edge) {
  // 	dummya = edge.v[0]; dummyb = edge.v[1];
  // 	// ecw[edge.id] = CalcSOC(edge.v[0], dummya, repl_diag[edge.v[0]],
  // 	// 		       edge.v[1], dummyb, repl_diag[edge.v[1]],
  // 	// 		       true);
  // 	ecw[edge.id] = 1;
  //     });

  //   tecw.Stop();
    
    
  //   size_t n_strong_e = 0;
  //   M.template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_strong_e++; } }, true);
  //   n_strong_e = comm.AllReduce(n_strong_e, MPI_SUM);
  //   double s_e_per_v = 2 * double(n_strong_e) / double(M.template GetNNGlobal<NT_VERTEX>());
  //   size_t approx_nagg = max2(size_t(1), size_t(NV / ( 1 + s_e_per_v )));
  //   agglomerates.SetSize(1.2 * approx_nagg); agglomerates.SetSize0();
  //   agg_diag.SetSize(1.2 * approx_nagg); agg_diag.SetSize0();

  //   // auto add_v_to_agg2 = [&](const auto & agg, auto v) LAMBDA_INLINE {
  //   //   /**
  //   // 	 Variant 2:
  //   // 	    diag = sum_{k in agg} Q(C->k).T Akk Q(C->k)    [Akk is the diagonal entry of the repl. mat]
  //   // 	 I would think that this locks too much. Maybe instead [1/|agg| * sum] ?
  //   //   **/
  //   // };


  //   // constexpr bool DIM3 = FACTORY::DIM == 3;
  //   constexpr bool DIM3 = true; // probably wont hurt ?

  //   /** Can we add something from eqa to eqb?? **/
  //   auto eqa_to_eqb = [&](auto eqa, auto eqb) { // PER DEFINITION, we are master of eqb!
  //     return eqc_h.IsLEQ(eqa, eqb);
  //   };


  //   Array<int> neib_ecnt(30), qsis(30);
  //   Array<double> ntraces(30);
  //   auto init_agglomerate = [&](auto v, auto v_eqc, bool force) LAMBDA_INLINE {
  //     // cout << endl << " INIT AGG FOR " << v << " from eqc " << v_eqc << " (force " << force << ")" << endl;
  //     auto agg_nr = agglomerates.Size();
  //     agglomerates.Append(Agglomerate(v, agg_nr)); // TODO: does this do an allocation??
  //     agg_diag.Append(repl_diag[v]);
  //     marked.SetBit(v);
  //     v_to_agg[v] = agg_nr;
  //     dist2agg[v] = 0;
  //     auto& agg = agglomerates.Last();
  //     auto & aggd = agg_diag[agg_nr];
  //     auto neibs_v = econ.GetRowIndices(v);
  //     int cnt_mems = 1;

  //     auto may_check_neib = [&](auto N) -> bool LAMBDA_INLINE
  //     { return (!marked.Test(N)) && eqa_to_eqb(M.template GetEqcOfNode<NT_VERTEX>(N), v_eqc) ;  };

  //     /** First, try to find ONE neib which we can add. Heuristic: Try descending order of trace(emat).
  // 	  I think we should give common neighbour boost here. **/
  //     auto neibs_e = econ.GetRowValues(v);
  //     ntraces.SetSize0(); ntraces.SetSize(neibs_v.Size());
  //     qsis.SetSize0(); qsis.SetSize(neibs_v.Size());
  //     int first_N_ind = -1;
  //     for (auto k : Range(ntraces)) {
  // 	ntraces[k] = calc_trace(edata[int(neibs_e[k])]);
  // 	qsis[k] = k;
  //     }
  //     QuickSortI(ntraces, qsis, [&] (const auto & i, const auto & j) LAMBDA_INLINE { return i > j; });
  //     for (auto j : Range(ntraces)) {
  // 	auto indN = qsis[j];
  // 	auto N = neibs_v[indN];
  // 	// cout << " try for first neib " << N << " with trace " << ntraces[indN] << endl;
  // 	if ( may_check_neib(N) ) {
  // 	  dummyb = N;
  // 	  auto soc = CalcSOC (v, agg.members(), aggd,
  // 			      N, dummyb, repl_diag[N],
  // 			      true); // neib boost probably worth it ...
  // 	  // cout << " soc " << soc << endl;
  // 	  if (soc > MIN_ECW) {
  // 	    // cout << "FIRST N " << N << " from eqc " << M.template GetEqcOfNode<NT_VERTEX>(N) << ", with soc " << soc << endl;
  // 	    first_N_ind = indN;
  // 	    v_to_agg[N] = agg_nr;
  // 	    dist2agg[N] = 1;
  // 	    marked.SetBit(N);
  // 	    add_v_to_agg (agg, N);
  // 	    cnt_mems++;
  // 	    break;
  // 	  }
  // 	}
  //     }
      
  //     if (first_N_ind != -1) { // if we could not add ANY neib, nohing has changed
  // 	/** We perform a greedy strategy: Keep adding neighbours have the highest number of connections
  // 	    leading into the aggregate. If not dist2, only check neighbours common to the two first verts in the aggregate,
  // 	    otherwise check all neibs of v.

  // 	    Its not perfect - if i check a neib, and do not add it, but later on add a common neib, i could potentially
  // 	    have added it in the first place. On the other hand - di I WANT to re-check all the time?

  // 	    Can I distinguish between "weak" and "singular" connections? Then I could only re-check the singular ones? **/
  // 	int qss = neibs_v.Size();
  // 	neib_ecnt.SetSize0(); neib_ecnt.SetSize(qss); neib_ecnt = 1; // per definition every neighbour has one edge
  // 	qsis.SetSize0(); qsis.SetSize(qss); qsis = -1; qss = 0;
  // 	for (auto j : Range(neibs_v))
  // 	  if (!may_check_neib(neibs_v[j]))
  // 	    { neib_ecnt[j] = -1; }
  // 	  else
  // 	    { qsis[qss++] = j; }
  // 	int first_valid_ind = 0;

  // 	/** inc edge count for common neibs of v and first meber**/
  // 	iterate_intersection(econ.GetRowIndices(neibs_v[first_N_ind]), neibs_v,
  // 			     [&](auto i, auto j) {
  // 			       if (neib_ecnt[j] != -1)
  // 				 { neib_ecnt[j]++; }
  // 			     });

  // 	/**  lock out all neibs of v that are not neibs of N **/
  // 	int inc_fv = 0;
  // 	if ( !dist2 ) {
  // 	  iterate_anotb(neibs_v, econ.GetRowIndices(neibs_v[first_N_ind]),
  // 			[&](auto inda) LAMBDA_INLINE {
  // 			  // cout << " inda " << inda << endl;
  // 			  if (neib_ecnt[inda] != -1) {
  // 			    neib_ecnt[inda] = -1;
  // 			    // cout << "(dist1) remove " << neibs_v[inda] << endl;
  // 			    inc_fv++;
  // 			  }
  // 			});
  // 	}

  // 	QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });

  // 	first_valid_ind += inc_fv;
  // 	qss -= inc_fv;

  // 	// cout << " init " << first_valid_ind << " " << qss << endl;
  // 	// prow(neibs_v); cout << endl;
  // 	// prow(neib_ecnt); cout << endl;
  // 	// prow(qsis); cout << endl;

  // 	while(qss>0) {
  // 	  auto n_ind = qsis[first_valid_ind + qss - 1]; 
  // 	  auto N = neibs_v[n_ind]; dummyb = N;
  // 	  auto soc = CalcSOC (v, agg.members(), aggd,
  // 			      N, dummyb, repl_diag[N],
  // 			      true); // neib boost should not be necessary
  // 	  // cout << " check " << N << " #con " << neib_ecnt[n_ind] << ", soc " << soc << ", ok ? " << bool(soc>MIN_ECW) << " // " << n_ind << " " << first_valid_ind << " " << qss << endl;
  // 	  qss--; // done with this neib
  // 	  neib_ecnt[n_ind] = -1; // just to check if i get all - i should not access this anymore anyways
  // 	  if (soc > MIN_ECW) { // add N to agg, update edge counts and re-sort neibs
  // 	    v_to_agg[N] = agg_nr;
  // 	    dist2agg[N] = 1;
  // 	    marked.SetBit(N);
  // 	    add_v_to_agg (agg, N); cnt_mems++;
  // 	    // cout << " add " << N << " from eqc " << M.template GetEqcOfNode<NT_VERTEX>(N) << endl;
  // 	    // cout << " neibs of N " << N << " are: "; prow(econ.GetRowIndices(N)); cout << endl;
  // 	    // cout << " neibs_v: "; prow(neibs_v); cout << endl;
  // 	    // cout << " old ecnt :"; prow(neib_ecnt); cout << endl;
  // 	    iterate_intersection(econ.GetRowIndices(N), neibs_v, // all neibs of v and N now have an additional edge into the agglomerate
  // 				 [&](auto inda, auto indb) LAMBDA_INLINE {
  // 				   if (neib_ecnt[indb] != -1)
  // 				     { neib_ecnt[indb]++; }
  // 				 });
  // 	    // cout << " new ecnt :"; prow(neib_ecnt); cout << endl;
  // 	    // int inc_fv = 0;
  // 	    // if ( (cnt_mems == 2) && (!dist2) ) { // lock all neibs of v that are not neibs of N
  // 	    //   // cout << " can stay: "; prow(econ.GetRowIndices(N)); cout << endl;
  // 	    //   iterate_anotb(neibs_v, econ.GetRowIndices(N),
  // 	    // 		    [&](auto inda) LAMBDA_INLINE {
  // 	    // 		      // cout << " inda " << inda << endl;
  // 	    // 		      if (neib_ecnt[inda] != -1) {
  // 	    // 			neib_ecnt[inda] = -1;
  // 	    // 			// cout << "(dist1) remove " << neibs_v[inda] << endl;
  // 	    // 			inc_fv++;
  // 	    // 		      }
  // 	    // 		    });
  // 	    // }
  // 	    // // cout << " first " << first_valid_ind << " qss " << qss << ", qsis: "; prow2(qsis); cout << endl;
  // 	    QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });
  // 	    // first_valid_ind += inc_fv;
  // 	    // qss -= inc_fv;
  // 	    // cout << "sorted; first " << first_valid_ind << " qss " << qss << ", qsis: "; prow2(qsis); cout << endl;
  // 	  }
  // 	  // cout << " after " << N << " " << n_ind << " " << first_valid_ind << " " << qss << endl;
  // 	  // cout << "neibs: "; prow(neibs_v); cout << endl;
  // 	  // cout << "#edge: "; prow(neib_ecnt); cout << endl;
  // 	  // cout << "qsis:  "; prow(qsis); cout << endl;
  // 	} // while(qss)
  //     } // first_N_ind != -1

  //     // cout << "agg has " << cnt_mems << " mems, would like at least " << MIN_NEW_AGG_SIZE << endl;
  //     if ( force || (cnt_mems >= MIN_NEW_AGG_SIZE) ) {
  // 	// cout << " agg accepted! " << endl;
  // 	// cout << agg << endl;
  // 	return true;
  //     }
  //     else { // remove the aggregate again - 
  //   	for (auto M : agg.members()) {
  //   	  v_to_agg[M] = -1;
  //   	  dist2agg[M] = -1;
  //   	  marked.Clear(M);
  //   	}
  //   	agglomerates.SetSize(agg_nr);
  //   	agg_diag.SetSize(agg_nr);
  //   	return false;
  //     }
  //   }; // init_agglomerate (..)

  //   // BitArray neibs_checked (NV); neibs_checked.Clear();
  //   // Array<int> common_neibs1(30), common_neibs2(30), common_neibs3(30), common_neibs4(30);
  //   // Array<int> queue1(30), queue2(30), queue3(30);
  //   // auto init_agglomerate = [&](auto v, auto v_eqc, bool force) LAMBDA_INLINE {
  //   //   auto agg_nr = agglomerates.Size();
  //   //   agglomerates.Append(Agglomerate(v, agg_nr));
  //   //   agg_diag.Append(repl_diag[v]);
  //   //   marked.SetBit(v);
  //   //   v_to_agg[v] = agg_nr;
  //   //   dist2agg[v] = 0;
  //   //   auto& agg = agglomerates.Last();
  //   //   auto & aggd = agg_diag[agg_nr];
  //   //   auto neibs_v = econ.GetRowIndices(v);
  //   //   for (auto n : neibs_v)
  //   // 	{ neibs_checked.Clear(n); }
  //   //   neibs_checked.Clear(v);
  //   //   auto do_check_neib = [&](auto N) LAMBDA_INLINE {
  //   // 	return (!neibs_checked.Test(N)) && (eqc_h.IsLEQ(M.template GetEqcOfNode<NT_VERTEX>(N)), v_eqc);
  //   //   };
  //   //   auto wrap_rec_lam = [&](auto & neibs, auto & queue, auto rec_lam, bool dist2) LAMBDA_INLINE {
  //   // 	return [&](auto ROOT, auto & LAST_NEIBS, auto enqueue ) LAMBDA_INLINE {
  //   // 	  intersect_sorted_arrays(econ.GetRowIndices(ROOT), LAST_NEIBS, neibs);
  //   // 	  auto my_queue = Queue(neibs);
  //   // 	  for (auto N : my_queue) {
  //   // 	  for (int lN = 0; lN < neibs.Size(); lN++) {
  //   // 	    auto N = neibs[lN];
  //   // 	    if (do_check_neib(N)) {
  //   // 	      neibs_checked.SetBit(N);
  //   // 	      dummyb = N;
  //   // 	      auto soc = CalcSOC_av (v, agg.members(), aggd,
  //   // 				     N, dummyb, repl_diag[N],
  //   // 				     true, geom);
  //   // 	      if (soc > MIN_ECW) {
  //   // 		add_v_to_agg (agg, N);
  //   // 		rec_lam(N, neibs); // <- CALL WRAPPED LAMBDA
  //   // 		if (dist2) {
  //   // 		  intersect_sorted_array(econ.GetRowIndices(N), LAST_NEIBS, common_neibs4);
  //   // 		  for (auto N : common_neibs4)
  //   // 		    if (!neibs_checked.Test(N)) {
  //   // 		      { neibs[lN] = N; lN--; /** neibs.Append(N); **/ }
  //   // 		    }
  //   // 		}
  //   // 	      }
  //   // 	    }
  //   // 	  }
  //   // 	};
  //   //   };
  //   //   auto NO_OP_LAM = [&](auto ROOT, auto & LAST_NEIBS) LAMBDA_INLINE { ; };
  //   //   auto lam_d3 = wrap_rec_lam(common_neibs3, NO_OP_LAM, dist2);
  //   //   auto lam_d2 = wrap_rec_lam(common_neibs2, common_neibs3, dist2, lam_d3);
  //   //   auto lam_d1 = wrap_rec_lam(neibs_v, commin_neibs2, dist2, lam_d2);
  //   //   lam_d1();
  //   //   // for (auto K : neibs_v) {
  //   //   // 	if (do_check_neib(K)) {
  //   //   // 	  neibs_checked.SetBit(K);
  //   //   // 	  dummyb = K;
  //   //   // 	  auto soc1 = CalcSOC_av (v, agg.members(), aggd,
  //   //   // 				  K, dummyb, repl_diag[K],
  //   //   // 				  true, geom);
  //   //   // 	  if (soc1 > MIN_ECW) {
  //   //   // 	    add_v_to_agg (agg, K);
  //   //   // 	    /** [V,K] is an edge, so common neibs are CN2 **/
  //   //   // 	    intersect_sorted_arrays(econ.GetRowIndices(K), neibs_v, common_neibs2);
  //   //   // 	    for (int lJ = 0; lJ < common_neibs2.Size(); lJ++) {
  //   //   // 	      auto J = common_neibs2[lJ]; dummyb = J;
  //   //   // 	      if (do_check_neib(J)) {
  //   //   // 	      	neibs_checked.Set(J);
  //   //   // 		auto soc2 = CalcSOC_av (v, agg.members(), aggd,
  //   //   // 					J, dummyb, repl_diag[J],
  //   //   // 					true, geom);
  //   //   // 		if (soc2 > MIN_ECW) {
  //   //   // 		  add_v_to_agg (agg, J);
  //   //   // 		  /** [V,K,J] is a triple, so common neibs are CN3 **/
  //   //   // 		  intersect_sorted_arrays(econ.GetRowIndices(J), common_neibs2, common_neibs3);
  //   //   // 		  for (int lM = 0; lM < common_neibs3.Size(); lM++) {
  //   //   // 		    auto M = common_neibs3[indM]; dummyb = M;
  //   //   // 		    if (do_check_neib(M)) {
  //   //   // 		      neibs_checked.Set(M);
  //   //   // 		      auto soc3 = CalcSOC_av (v, agg.members(), aggd,
  //   //   // 					      M, dummyB, repl_diag[M],
  //   //   // 					      true, geom);
  //   //   // 		      if (soc3 > MIN_ECW) {
  //   //   // 			add_v_to_agg (agg, M);
  //   //   // 			/** [V,K,M] is a triple, so common neibs are CN3 (fir dist2) **/
  //   //   // 			if (dist2) {
  //   //   // 			  intersect_sorted_arrays(econ.GetRowIndices(M), common_neibs2, common_neibs4);
  //   //   // 			  for (auto N : common_neibs4)
  //   //   // 			    if (!checked_neibs.Test(N))
  //   //   // 			      { common_neibs3.Append(N); }
  //   //   // 			} // DIM3 dist2
  //   //   // 		      } // DIM3 strong
  //   //   // 		    } // DIM3 check
  //   //   // 		  } // DIM3 loop
  //   //   // 		  /** V - J is an edge, so common neibs of V, J are CN2 (for dist2) **/
  //   //   // 		  if (dist2) {
  //   //   // 		    intersect_sorted_arrays(econ.GetRowIndices(J), neibs_v, common_neibs4);
  //   //   // 		    for (auto N : common_neibs4)
  //   //   // 		      if (!neibs_checked.Test(N))
  //   //   // 			{ common_neibs2.Append(N); }
  //   //   // 		  } // DIM2 dist2
  //   //   // 		} // DIM2 strong
  //   //   // 	      } // DIM2 check
  //   //   // 	    } // CN 2 loop
  //   //   // 	  } // DIM1 strong
  //   //   // 	} // DIM1 check
  //   //   // } // DIM1 loop
  //   //   if ( force || (agg.members.Size() > MIN_NEW_AGG_SIZE) ) {
  //   // 	return true;
  //   //   }
  //   //   else {
  //   // 	for (auto M : agg.members()) {
  //   // 	  v_to_agg[M] = -1;
  //   // 	  dist2agg[M] = -1;
  //   // 	  marked.ClearBit(M);
  //   // 	}
  //   // 	agglomerates.SetSize(agg_nr);
  //   // 	agg_diag.SetSize(agg_nr);
  //   // 	return false;
  //   //   }
  //   // };


  //   /** Deal with dirichlet vertices **/
  //   if (free_verts != nullptr) {
  //     for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
  // 	if (!free_verts->Test(k))
  // 	  { marked.SetBit(k); }
  //     }
  //     // cout << "DIRICHLET VERTS: " << endl;
  //     // for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
  //     // 	if (!free_verts->Test(k)) {
  //     // 	  cout << k << " " << endl;
  //     // 	}
  //     // }
  //   }

  //   t1.Stop(); t2.Start();


  //   /** Iterate through vertices and start new agglomerates if the vertex is at least at distance 1 from
  // 	any agglomerate (so distance 2 from any agg. center) **/
  //   {
  //     Array<int> vqueue(M.template GetNN<NT_VERTEX>()); vqueue.SetSize0();
  //     BitArray checked(M.template GetNN<NT_VERTEX>()); checked.Clear();
  //     BitArray queued(M.template GetNN<NT_VERTEX>()); queued.Clear();
  //     size_t nchecked = 0; const auto NV = M.template GetNN<NT_VERTEX>();
  //     int cntq = 0, cntvr = NV-1;
  //     int vnum;
  //     while (nchecked < NV) {
  // 	/** if there are any queued vertices, handle them first, otherwise take next vertex by reverse counting
  // 	    ( ex-vertices have higher numbers) **/
  // 	bool from_q = cntq < vqueue.Size();
  // 	if (from_q)
  // 	  { vnum = vqueue[cntq++]; }
  // 	else
  // 	  { vnum = cntvr--; }
  // 	if (!checked.Test(vnum)) {
  // 	  // cout << " from queue ? " << from_q << endl;
  // 	  bool newagg = check_v(vnum);
  // 	  checked.SetBit(vnum);
  // 	  nchecked++;
  // 	  if (newagg) { // enqueue neibs of neibs of the agg (if they are not checked, queued or at least marked yet)
  // 	    const auto & newagg = agglomerates.Last();
  // 	    int oldqs = vqueue.Size();
  // 	    for (auto mem : newagg.members()) {
  // 	      auto mem_neibs = econ.GetRowIndices(mem);
  // 	      for (auto x : mem_neibs) {
  // 		auto y = econ.GetRowIndices(x);
  // 		for (int jz = int(y.Size()) - 1; jz>= 0; jz--) { // enqueue neibs of neibs - less local ones first
  // 		  auto z = y[jz];
  // 		  if (  (!marked.Test(z)) && (!checked.Test(z)) && (!queued.Test(z)) )
  // 		    { vqueue.Append(z); queued.SetBit(z); }
  // 		}
  // 	      }
  // 	    }
  // 	    int newqs = vqueue.Size();
  // 	    if (newqs > oldqs + 1) // not sure ?
  // 	      { QuickSort(vqueue.Part(oldqs, (newqs-oldqs)), [&](auto i, auto j) { return i>j; }); }
  // 	  }
  // 	}
  //     }
  //   }

  //    // {
  //    //  //   cout << endl << " FIRST loop done " << endl;
  //    //  cout << "frac marked: " << double(marked.NumSet()) / marked.Size() << endl;
  //    //  cout << " INTERMED agglomerates : " << agglomerates.Size() << endl;
  //    //  cout << agglomerates << endl;
  //    //  Array<int> naggs;
  //    //  auto resize_to = [&](auto i) {
  //    // 	auto olds = naggs.Size();
  //    // 	if (olds < i) {
  //    // 	  naggs.SetSize(i);
  //    // 	  for (auto j : Range(olds, i))
  //    // 	    { naggs[j] = 0; }
  //    // 	}
  //    //  };
  //    //  for (const auto & agg : agglomerates) {
  //    // 	auto ags = agg.members().Size();
  //    // 	resize_to(1+ags);
  //    // 	naggs[ags]++ ;
  //    //  }
  //    //  cout << " INTERMED agg size distrib: "; prow2(naggs); cout << endl;
  //    // }

  //   cnt_prtm = 0;

  //   t2.Stop(); t3.Start();

  //   Array<int> neib_aggs(20);
  //   Array<int> notake(20);
  //   Array<double> na_soc(20);
  //   Array<int> index(20);
  //   // Array<int> neibs_in_agg(20):
  //   M.template ApplyEQ<NT_VERTEX>([&] (auto eqc, auto v) LAMBDA_INLINE {
  // 	// TODO: should do this twice, once with max_dist 1, once 2 (second round, steal verts from neibs ?)
  // 	auto neibs = econ.GetRowIndices(v);
  // 	if (!marked.Test(v)) {
  // 	  if (neibs.Size() == 1) {  // Check for hanging vertices
  // 	    // cout << " vertex " << v << " unmarked in loop 2 " << endl;
  // 	    // Collect neighbouring agglomerates we could append ourselfs to
  // 	    auto N = neibs[0];
  // 	    auto neib_agg_id = v_to_agg[neibs[0]];
  // 	    if (neib_agg_id != -1) { // neib is in an agglomerate - I must also be master of N [do not check marked - might be diri!]
  // 	      auto & neib_agg = agglomerates[neib_agg_id];
  // 	      auto N_eqc = M.template GetEqcOfNode<NT_VERTEX>(N);
  // 	      bool could_add = eqa_to_eqb(eqc, N_eqc);
  // 	      bool can_add = (dist2agg[N] == 0); // (neib_agg.center() == N); // I think in this case this MUST be ok ??
  // 	      if ( could_add && (!can_add) ) { // otherwise, re-check SOC (but this should have been checked already)
  // 		auto soc = CalcSOC_av (neib_agg, v, true);
  // 		if (soc > MIN_ECW)
  // 		  { can_add = true; }
  // 	      }
  // 	      can_add &= could_add;
  // 	      if (can_add) { // lucky!
  // 		// cout << " add hanging " << v << " to agg of " << N << " = " << neib_agg << endl;
  // 		marked.SetBit(v);
  // 		v_to_agg[v] = neib_agg_id;
  // 		dist2agg[v] = 1 + dist2agg[N]; // must be correct - N is the only neib
  // 		add_v_to_agg (neib_agg, v);
  // 	      }
  // 	      else { // unfortunate - new single agg!
  // 		init_agglomerate(v, eqc, true);
  // 	      }
  // 	    }
  // 	    else { // neib is not in an agg - force start a new one at neib and add this vertex (which must be OK!)
  // 	      auto N_eqc = M.template GetEqcOfNode<NT_VERTEX>(N);
  // 	      if ( (eqc_h.IsMasterOfEQC(N_eqc)) && (eqa_to_eqb(eqc, N_eqc) ) ) { // only if it is OK eqc-wise
  // 		init_agglomerate(N, N_eqc, true); // have to force new agg even if we dont really want to ...
  // 		auto new_agg_id = v_to_agg[N];
  // 		if (!marked.Test(v)) { // might already have been added by init_agg, but probably not
  // 		  marked.SetBit(v);
  // 		  v_to_agg[v] = new_agg_id;
  // 		  dist2agg[v] = 1;
  // 		  add_v_to_agg(agglomerates[new_agg_id], v);
  // 		}
  // 	      }
  // 	      else { // unfortunate - new single agg!
  // 		init_agglomerate(v, eqc, true);
  // 	      }
  // 	    }
  // 	  }
  // 	  else {
  // 	    neib_aggs.SetSize0();
  // 	    notake.SetSize0();
  // 	    na_soc.SetSize0();
  // 	    for (auto n : neibs) {
  // 	      if ( (marked.Test(n)) && (dist2agg[n] <= (dist2 ? 2 : 1))) { // max_dist 2 maybe, idk ??
  // 		auto agg_nr = v_to_agg[n];
  // 		if (agg_nr != -1) { // can this even be -1 if "marked" is set ??
  // 		  auto & n_agg = agglomerates[agg_nr];
  // 		  if ( eqa_to_eqb(eqc, M.template GetEqcOfNode<NT_VERTEX>(n_agg.center())) &&
  // 		       (!notake.Contains(agg_nr)) && (!neib_aggs.Contains(agg_nr)) ) {
  // 		    // cout << " calc des with " << n_agg << endl;
  // 		    // cout << " calc soc " << v << " to agg " << agg_nr << ", " << n_agg << endl;
  // 		    auto soc = CalcSOC_av(n_agg, v, true); // use neib_boost here - we want as few new aggs as possible
  // 		    // cout << " soc " << soc << endl;
  // 		    if (soc > MIN_ECW) {
  // 		      /** Desirability of adding v to A:
  // 			  1 / (1 + |A|) * Sum_{k in A} alpha(e_vk)/dist[k]
  // 			  So desirability grows when we have many strong connections to an agglomerate and decreases
  // 			  the farther away from the center of A we are and the larger A already is. **/
  // 		      intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
  // 		      // for (auto k : neibs_in_agg)
  // 		      // { cout << "[" << k << " | " << dist2agg[k] << " | " << ecw[int(econ(v,k))] << "]" << endl; }
  // 		      double mindist = 20;
  // 		      double des = 0;
  // 		      for (auto k : neibs_in_agg)
  // 			{ des += ecw[int(econ(v,k))] / ( 1 + dist2agg[k] );
  // 			  mindist = min2(mindist, double(dist2agg[k]));
  // 			}
  // 		      // cout << " init des " << des << endl;
  // 		      // des = des * (soc/MIN_ECW) / (n_agg.members().Size());
  // 		      des = (soc/MIN_ECW) / (mindist * n_agg.members().Size());
  // 		      // cout << "soc/des " << soc << " " << des << endl;
  // 		      neib_aggs.Append(agg_nr);
  // 		      na_soc.Append(des);
  // 		    }
  // 		    else // just so we do not compute SOC twice for no reason
  // 		      { notake.Append(agg_nr); }
  // 		  }
  // 		}
  // 	      }
  // 	    }
  // 	    if (neib_aggs.Size()) { // take the most desirable neib
  // 	      // cout << " decide between " << neib_aggs.Size() << endl;
  // 	      // 							cout << " neib ses "; for(auto n : neib_aggs) { cout << agglomerates[n].members().Size() << " "; }
  // 	      // 																		    cout << endl;
  // 	      auto mi = ind_of_max(na_soc);
  // 	      auto agg_nr = neib_aggs[mi]; auto& n_agg = agglomerates[agg_nr];
  // 	      // cout << " decided for " << n_agg.members().Size() << endl;
  // 	      add_v_to_agg(n_agg, v);
  // 	      marked.SetBit(v);
  // 	      v_to_agg[v] = agg_nr;
  // 	      intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
  // 	      int mindist = 1000;
  // 	      for (auto k : neibs_in_agg)
  // 		{ mindist = min2(mindist, dist2agg[k]); }
  // 	      dist2agg[v] = 1 + mindist;
  // 	      // cout << " added to agg " << n_agg << endl;
  // 	    }
  // 	    else {
  // 	      init_agglomerate(v, eqc, true); // have to force new agg even if we dont really want to ...
  // 	      // Maybe we should check if we should steal some vertices from surrounding aggs, at least those with
  // 	      // dist >= 2. Maybe init_agg(v, eqc, steal=true) ??
  // 	    }
  // 	  } // neibs.Size > 1
  // 	} // if (!marked.Test(v))
  //     }, true); // also only master verts!

  //   // comm.Barrier();

  //   // {
  //   //   // cout << endl;
  //   //   cout << " FINAL agglomerates : " << agglomerates.Size() << endl;
  //   //   cout << agglomerates << endl;
  //   //   Array<int> naggs;
  //   //   auto resize_to = [&](auto i) {
  //   // 	auto olds = naggs.Size();
  //   // 	if (olds < i) {
  //   // 	  naggs.SetSize(i);
  //   // 	  for (auto j : Range(olds, i))
  //   // 	    { naggs[j] = 0; }
  //   // 	}
  //   //   };
  //   //   for (const auto & agg : agglomerates) {
  //   // 	auto ags = agg.members().Size();
  //   // 	resize_to(1+ags);
  //   // 	naggs[ags]++ ;
  //   //   }
  //   //   cout << " FINAL agg size distrib: "; prow2(naggs); cout << endl;
  //   // }

  //   // comm.Barrier();

  //   t3.Stop();

  // } // Agglomerator::FormAgglomeratesOld

} // namespace amg

