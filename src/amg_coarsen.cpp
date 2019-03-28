#define FILE_AMGCRS_CPP
#include "amg.hpp"

namespace amg
{

  VWCoarseningData::CollapseTracker :: CollapseTracker (size_t nv, size_t ne)
    : vertex_ground(nv), edge_collapse(ne), vertex_collapse(nv), edge_fixed(ne),
      vertex_fixed(nv), v2e(nv)
  {
    if (nv) {
      vertex_ground.Clear();
      vertex_collapse.Clear();
      vertex_fixed.Clear();
      v2e = nullptr;
    }
    if (ne) {
      edge_fixed.Clear();
      edge_collapse.Clear();
    }
  }

  template<class TMESH>
  VWiseCoarsening<TMESH> :: VWiseCoarsening (shared_ptr<VWiseCoarsening<TMESH>::Options> opts)
    : VWCoarseningData(opts)
  { ; }

  template<class TMESH>
  BlockVWC<TMESH> :: BlockVWC (shared_ptr<BlockVWC<TMESH>::Options> opts)
    : VWiseCoarsening<TMESH>(opts)
  { ; }

  
  /** We should not try to map a flat-mesh !!**/
  template<> shared_ptr<CoarseMap<FlatTM>> VWiseCoarsening<FlatTM> :: Coarsen (shared_ptr<FlatTM> mesh)
  { return nullptr; }

  
  template<class TMESH>
  SeqVWC<TMESH> :: SeqVWC (shared_ptr<SeqVWC<TMESH>::Options> opts)
    : VWiseCoarsening<TMESH>(opts)
  { ; }


  INLINE Timer & HierVWCTimerHack (){ static Timer t("Collapse - hierarchic"); return t; }
  template<class TMESH>
  HierarchicVWC<TMESH> :: HierarchicVWC (shared_ptr<HierarchicVWC::Options> opts)
    : VWiseCoarsening<TMESH>(nullptr)
    { options = (opts == nullptr) ? make_shared<Options>() : opts; }
  
  INLINE Timer & SeqVWCTimerHack (){ static Timer t("Collapse - seq, DAG"); return t; }
  template<class TMESH>
  void SeqVWC<TMESH> :: Collapse (const TMESH & mesh, SeqVWC<TMESH>::CollapseTracker & coll)
  {
    RegionTimer rt(SeqVWCTimerHack());
    size_t NV = mesh.template GetNN<NT_VERTEX>();
    if (!NV) return;
    auto verts = mesh.template GetNodes<NT_VERTEX>();
    auto edges = mesh.template GetNodes<NT_EDGE>();
    size_t NE = mesh.template GetNN<NT_EDGE>();
    const double MIN_CW(options->min_cw);
    shared_ptr<BitArray> free_verts = options->free_verts;
    const double* ecw(&options->ecw[0]);
    const double* vcw(&options->vcw[0]);
    auto GetECW = [ecw](const auto & x) { return ecw[x.id]; };
    auto GetVCW = [vcw](const auto & x) { return vcw[x]; };
    // cout << "collapse seq, NV NE " << NV << " " << NE << endl;
    // vertices are now always OS..OS+NV-1!!
    // cout << "verts: "; prow(verts); cout << endl;
    const int min_v = (verts.Size()>0) ? verts[0] : 0;
    if (free_verts != nullptr)
      for (auto k:Range(min_v, min_v+int(NV)))
	if (!free_verts->Test(k))
	  { coll.GroundVertex(k); coll.FixVertex(k); }
    // cout << "min_v: " << min_v << ", NV : " << NV << endl;
    // cout << "edges: "; prow(edges); cout << endl;
    TableCreator<int> v2ec(NV);
    for ( ; !v2ec.Done(); v2ec++)
      for (auto k:Range(edges.Size())) {
	const auto & edge = edges[k];
	v2ec.Add(edge.v[0]-min_v, k);
    	v2ec.Add(edge.v[1]-min_v, k);
      }
    Table<int> v2e = v2ec.MoveTable(); // TODO: can we just use "econ" here?
    for (auto rn:Range(v2e.Size())) {
      auto row = v2e[rn];
      if (!row.Size()) continue;
      QuickSort(row, [&](auto k, auto j) {
    	  double w1 = GetECW(edges[k]), w2 = GetECW(edges[j]);
	  // double fac = w1/w2;
	  // if (fac>0.8 && fac<1.2) return vs_in_agg1 < vs_in_agg2;
    	  if (w1 == w2) return k < j;
    	  return w1 < w2;
    	});
    }
    auto edge_valid = [&] (const auto & edge) -> bool {
      return ( (GetECW(edge) > MIN_CW) &&
    	       (!coll.IsVertexCollapsed(edge.v[0])) &&
    	       (!coll.IsVertexCollapsed(edge.v[1])) &&
    	       (!coll.IsVertexFixed(edge.v[0])) &&
    	       (!coll.IsVertexFixed(edge.v[1])) &&
	       (!coll.IsVertexGrounded(edge.v[1])) &&
	       (!coll.IsVertexGrounded(edge.v[1]))
    	       );
    };
    TableCreator<int> ced(NE);
    for ( ; !ced.Done(); ced++)  
      for (auto k:Range(NV)) {
    	auto vedges = v2e[k];
    	for (size_t j = 0; j+1 < vedges.Size(); j++)
    	  ced.Add(vedges[j+1], vedges[j]);
      }
    auto edge_dag = ced.MoveTable();
    RunParallelDependency(edge_dag,
    			  [&] (int k) {
    			    const auto & edge = edges[k];
			    // cout << " check edge " << edge << " fix, valid " << ", wt " << GetECW(edge) << " "
			    // 	 << coll.IsEdgeFixed(edge) << " " << edge_valid(edge) << endl;
    			    if ( (!coll.IsEdgeFixed(edge)) &&
    				(edge_valid(edge)) ) {
			      coll.CollapseEdge(edge);
			    }
    			  });
    for (auto vertex:verts)
      if ((GetVCW(vertex) > MIN_CW) && (!coll.IsVertexFixed(vertex)) &&
    	  (!coll.IsVertexGrounded(vertex)) && (!coll.IsVertexCollapsed(vertex)) )
    	{ /**cout << "warning, ground V!!" << endl; **/ coll.GroundVertex(vertex); }
  }


  // INLINE Timer & BlockVWCTimerHack (){ static Timer t("Collapse - blockwise"); return t; }
  // template<class TMESH>
  // void BlockVWC<TMESH> :: Collapse (const TMESH & mesh, BlockVWC::CollapseTracker & coll)
  // {
  //   // cout << "collapse blockwise " << endl;
  //   RegionTimer rt(BlockVWCTimerHack());
  //   const auto & eqc_h = *mesh.GetEQCHierarchy();
  //   SeqVWC<FlatTM> block_crs (options);
  //   auto neqcs = eqc_h.GetNEQCS();
  //   for (auto eqc_num : Range(neqcs)) {
  //     FlatTM mesh_block = mesh.GetBlock(eqc_num);
  //     // cout << "collapse block for eq " << eqc_num << " of " << neqcs << endl;
  //     block_crs.Collapse(mesh_block, coll);
  //   }
  // }

  INLINE Timer & BlockVWCTimerHack (){ static Timer t("Collapse - blockwise"); return t; }
  template<class TMESH>
  void BlockVWC<TMESH> :: Collapse (const TMESH & mesh, BlockVWC::CollapseTracker & coll)
  {
    RegionTimer rt(BlockVWCTimerHack());
    const auto & eqc_h = *mesh.GetEQCHierarchy();
    auto free = options->free_verts;
    if (free!=nullptr) for (auto k:Range(mesh.template GetNN<NT_VERTEX>())) if (!free->Test(k)) { coll.GroundVertex(k); coll.FixVertex(k); }

    auto block_opts = make_shared<typename SeqVWC<FlatTM>::Options>();
    block_opts->min_cw = options->min_cw;
    block_opts->vcw = Array<double>(options->vcw.Size(), &(options->vcw[0])); block_opts->vcw.NothingToDelete();
    block_opts->ecw = Array<double>(options->ecw.Size(), &(options->ecw[0])); block_opts->ecw.NothingToDelete();
    block_opts->free_verts = nullptr;
    SeqVWC<FlatTM> block_crs (block_opts);

    // entry>0: collapse edge (entry-1) // entry<=0: ground vertex (-entry)
    auto neqcs = eqc_h.GetNEQCS();
    Array<int> perow(neqcs); perow = 0;
    for (size_t eqc_num = 0; eqc_num < neqcs; eqc_num++) {
      if (eqc_h.IsMasterOfEQC(eqc_num)) {
	FlatTM mesh_block = mesh.GetBlock(eqc_num);
	block_crs.Collapse(mesh_block, coll);
	if (eqc_num!=0) {
	  for (const auto & e : mesh_block.GetNodes<NT_EDGE>())
	    if (coll.IsEdgeCollapsed(e)) perow[eqc_num]++;
	  for (auto v : mesh_block.GetNodes<NT_VERTEX>())
	    if (coll.IsVertexGrounded(v)) perow[eqc_num]++;
	}
      }
    }
    Table<int> sync(perow); perow = 0;
    for (size_t eqc_num = 1; eqc_num < neqcs; eqc_num++) {
      if (eqc_h.IsMasterOfEQC(eqc_num)) {
	FlatTM mesh_block = mesh.GetBlock(eqc_num);
	for (const auto & e : mesh_block.GetNodes<NT_EDGE>())
	  if (coll.IsEdgeCollapsed(e)) {
	    sync[eqc_num][perow[eqc_num]++] = 1+mesh.template MapNodeToEQC<NT_EDGE>(e.id);
	    // cout << e << " is coll -> " << eqc_num << " " << mesh.template MapNodeToEQC<NT_EDGE>(e.id) << endl;
	  }
	for (auto v : mesh_block.GetNodes<NT_VERTEX>())
	  if (coll.IsVertexGrounded(v)) {
	    // cout << v << " is grnd -> " << eqc_num << " " << mesh.template MapNodeToEQC<NT_VERTEX>(v) << endl;
	    sync[eqc_num][perow[eqc_num]++] = -mesh.template MapNodeToEQC<NT_VERTEX>(v);
	  }
      }
    }
    // cout << "sync: " << endl << sync << endl;
    Table<int> syncsync = ScatterEqcData(move(sync), eqc_h);
    // cout << "syncsync: " << endl << syncsync << endl;
    for (size_t eqc_num = 1; eqc_num < neqcs; eqc_num++) {
      if (!eqc_h.IsMasterOfEQC(eqc_num)) {
	auto srow = syncsync[eqc_num]; auto ss = srow.Size();
	auto c = 0;
	for (auto l : Range(srow.Size())) {
	  auto val = srow[l];
	  if ( val <= 0 ) { // ground vertex
	    auto vnr = mesh.template MapENodeFromEQC<NT_VERTEX>(-val, eqc_num);
	    // cout << "ground vertex " << -val << " -> " << vnr << endl;
	    if (!coll.IsVertexGrounded(vnr)) { coll.template ClearNode<NT_VERTEX>(vnr); coll.GroundVertex(vnr); }
	  }
	  else {
	    int enr = val-1;
	    auto eqc_es = mesh.template GetENodes<NT_EDGE>(eqc_num);
	    const AMG_Node<NT_EDGE> & edge = eqc_es[enr];
	    // cout << "collapse edge " << enr << " -> " << edge << endl;
	    if (!coll.IsEdgeCollapsed(edge)) { coll.template ClearNode<NT_EDGE>(edge); coll.CollapseEdge(edge); }
	  }
	}
      }
    }
    
    
  }


  INLINE Timer & HierarchicVWCTimerHack (){ static Timer t("Collapse - hierarchic"); return t; }
  template<class TMESH>
  void HierarchicVWC<TMESH> :: Collapse (const TMESH & mesh, HierarchicVWC::CollapseTracker & coll)
  {
    cout << "HCOL mesh: " << endl << mesh << endl;
    RegionTimer rt(HierVWCTimerHack());
    auto block_opts = make_shared<typename BlockVWC<TMESH>::Options>();
    block_opts->min_cw = options->min_cw;
    // block_opts->free_verts = nullptr; // I am taking care of that!
    block_opts->vcw = Array<double>(options->vcw.Size(), &(options->vcw[0])); block_opts->vcw.NothingToDelete();
    block_opts->ecw = Array<double>(options->ecw.Size(), &(options->ecw[0])); block_opts->ecw.NothingToDelete();
    BlockVWC<TMESH> block_crs (block_opts);
    
    const auto & eqc_h = *mesh.GetEQCHierarchy();
    auto neqcs = eqc_h.GetNEQCS();
    auto comm = eqc_h.GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    auto NV = mesh.template GetNN<NT_VERTEX>();
    auto NE = mesh.template GetNN<NT_EDGE>();

    Array<double> mark_v(NV);
    if (NV) mark_v = 0.0;
    Array<int> vmark_2_e(NV);
    if (NV) vmark_2_e = -1;
    BitArray want_edge(NE);
    if (NE) want_edge.Clear();

    const double* ecw(&options->ecw[0]);
    const double* vcw(&options->vcw[0]);
    auto GetECW = [ecw](const auto & x) { return ecw[x.id]; };
    auto GetVCW = [vcw](const auto & x) { return vcw[x]; };
    const double MIN_CW(options->min_cw);
    auto free = options->free_verts;

    if (free!=nullptr) {
      // cout << free->NumSet() << " of " << free->Size() << " are free!" << endl;
      const auto & fv = *free; for (auto k:Range(NV)) if (!fv.Test(k)) { cout << " g+f vertex " << k << endl; coll.GroundVertex(k); coll.FixVertex(k); }
    }

    // if (free) cout << "total free: " << free->NumSet() << " of " << free->Size() << endl;
    
    double nvalid = 0;
    for (auto k : Range(NE)) {
      bool val = (ecw[k]>MIN_CW);
      if (val) nvalid++;
      // cout << "ecw " << k << " of " << NE << ": " << ecw[k] << ", valid: " << val << endl;
    }
    nvalid = (nvalid==0) ? 0 : nvalid/NE;
    // cout << "total valid NE: " << nvalid << endl;

    if (options->pre_coll) {
      block_crs.Collapse(mesh, coll);
      size_t ncol = 0;
      for (const auto & e : mesh.template GetNodes<NT_EDGE>()) { if (coll.IsEdgeCollapsed(e)) ncol++; }
      double cfr = (mesh.template GetNN<NT_VERTEX>()==0) ? 0 : (1.0*ncol)/mesh.template GetNN<NT_VERTEX>();
      // cout << "blockwise ncol: " << ncol << " of " << mesh.GetNN<NT_EDGE>() << ", frac " << cfr << endl;
    }
    
    
    /** mark vertices **/
    auto all_edges = mesh.template GetNodes<NT_EDGE>();
    for (auto eqc : Range(neqcs) ) {
      auto cross_edges = mesh.template GetCNodes<NT_EDGE> (eqc);
      // cout << " eqc " << eqc << ",ce " << cross_edges.Size() << endl;
      for (const auto & edge : cross_edges) {
	// should not matter - block-coll should have fixed these!
	// cout << "check edge " << edge << endl;
	if (coll.IsVertexFixed(edge.v[0]) || coll.IsVertexFixed(edge.v[1]) ||
	   coll.IsEdgeFixed(edge)) continue;
	const auto ew = GetECW(edge);
	if (ew < MIN_CW) continue;
	auto e0 = mesh.template GetEqcOfNode<NT_VERTEX>(edge.v[0]);
	auto e1 = mesh.template GetEqcOfNode<NT_VERTEX>(edge.v[1]);
	// cout << "eqcs " << e0 << " " << e1 << endl;
	// cout << "goes to eqc " << eqc_h.GetMergedEQC(e0, e1) << endl;
	if (!eqc_h.IsMergeValid(e0,e1)) { /* cout << "cant merge " << e0 << " " << e1 << endl; */ continue; }
	bool want_it = true; double wt = 0;
	for (auto l : Range(2)) {
	  auto v = edge.v[l];
	  if (coll.IsVertexCollapsed(v)) {
	    if ( (wt = max2(wt, GetECW(coll.CollapsedEdge(v)))) > ew ) { /* cout << "no 1 " << l << endl; */ want_it = false; }
	  }
	  else if (coll.IsVertexGrounded(edge.v[l])) {
	    if ( (wt = max2(wt, GetVCW(v))) > ew ) { /* cout << "no 2 " << l << endl; */ want_it = false; }
	  }
	}
	if ( (!want_it) || (!eqc_h.IsMasterOfEQC(eqc)) ) { //still, mark local max
	  for (auto l : Range(2)) mark_v[edge.v[l]] = max2(wt, mark_v[edge.v[l]]);
	  continue;
	}
	if ( (ew > mark_v[edge.v[0]] ) && (ew > mark_v[edge.v[1]]) ) {
	  // TODO: in orig code, I cleared want_edge from CollapsedEdge(edge.v[l]) ??
	  for (auto l : Range(2)) {
	    auto oeid = vmark_2_e[edge.v[l]];
	    if ( oeid != -1 ) {
	      want_edge.Clear(oeid);
	      const auto & oe = all_edges[oeid];
	      vmark_2_e[oe.v[0]] = -1;
	      vmark_2_e[oe.v[1]] = -1;
	    }
	    vmark_2_e[edge.v[l]] = edge.id;
	    mark_v[edge.v[l]] = ew;
	  }
	  want_edge.Set(edge.id);
	}
      }
    }
    int nce = 0; for (auto k : Range(neqcs)) nce += mesh.template GetCNN<NT_EDGE>(k);
    // cout << "want edge: " << want_edge.NumSet() << " of " << nce << endl;
    // cout << want_edge << endl;
    // cout << "marked vs: " << endl; prow(mark_v); cout << endl;
    
    // overwrite values with index of max. proc!
    mesh.template AllreduceNodalData<NT_VERTEX> (mark_v, [](const auto & tab_in) {
	Array<double> out;
	const int nrows = tab_in.Size();
	if (nrows==0) return out;
	const int rows = tab_in[0].Size();
	if (rows==0) return out;
	out.SetSize(rows); out = nrows-1;
	if (nrows==1) { return out; }
	auto rowo = tab_in[nrows-1];
	for (int k = nrows-2; k>=0; k--) {
	  auto rowk = tab_in[k];
	  for (auto j : Range(rows)) {
	    if (rowo[j]<rowk[j]) {
	      rowo[j] = rowk[j]; out[j] = k;
	    }
	  }
	}
	return out;
      }, false); // do nothing for local values

    int until = 0;
    for (auto eqc : Range(neqcs)) {
      int last = until;
      until += mesh.template GetENN<NT_VERTEX>(eqc);
      auto dps = eqc_h.GetDistantProcs(eqc);
      auto pos = 0; while ( pos<dps.Size() && comm.Rank() > dps[pos]) pos++;
      if (until > last)
	for (auto l : Range(last, until)) {
	  mark_v[l] = ( (eqc==0) || (mark_v[l]==pos)) ? 1 : 0;
	}
    }

    // cout << "assigned vs: " << endl; prow2(mark_v); cout << endl;

    // also, edge always comes BEFORE its vertices
    // entry >0: cross-edge entry-1 in coll
    // entry <=0: vertex -entry is coll to SOME edge
    TableCreator<int> csync(neqcs);
    for (; !csync.Done(); csync++) {
      for (auto eqc : Range(neqcs)) {
	auto cross_edges = mesh.template GetCNodes<NT_EDGE>(eqc);
	for (auto ke : Range(cross_edges.Size())) {
	  const auto & e = cross_edges[ke];
	  if (!want_edge.Test(e.id)) continue;
	  if ( (mark_v[e.v[0]]==0) || (mark_v[e.v[1]]==0) ) continue;
	  csync.Add(eqc, 1+ke);
	  int eq0 = mesh.template GetEqcOfNode<NT_VERTEX>(e.v[0]);
	  int vl0 = mesh.template MapNodeToEQC<NT_VERTEX>(e.v[0]);
	  csync.Add(eq0, -vl0);
	  int eq1 = mesh.template GetEqcOfNode<NT_VERTEX>(e.v[1]);
	  int vl1 = mesh.template MapNodeToEQC<NT_VERTEX>(e.v[1]);
	  csync.Add(eq1, -vl1);
	}
      }
    }
    auto sync = csync.MoveTable();

    // cout << "sync: " << endl << sync << endl;

    auto syncsync = ReduceTable<int, int>(sync, mesh.template GetEQCHierarchy(), [&](auto & in) { // syncsync ... haha
	Array<int> out;
	int nrows = in.Size(); if (nrows==0) return out;
	int tot_entries = 0; for (auto k:Range(nrows)) tot_entries += in[k].Size();
	if (tot_entries == 0) return out;
	out.SetSize(tot_entries);
	tot_entries = 0;
	for (auto k:Range(nrows)) {
	  auto row = in[k];
	  for (auto j : Range(row.Size())) {
	    out[tot_entries++] = row[j];
	  }
	}
	return out;
      });

    // cout << "syncsync: " << endl << syncsync << endl;
    
    for (auto eqc : Range(neqcs)) {
      auto cross_edges = mesh.template GetCNodes<NT_EDGE>(eqc);
      auto verts = mesh.template GetENodes<NT_VERTEX>(eqc);
      auto row = syncsync[eqc];
      for (int val : row) {
	if (val>0) {
	  int ke = val-1;
	  AMG_Node<NT_EDGE> &e = cross_edges[ke];
	  // cout << "should cross coll edge " << e << endl;
	  for (auto l : Range(2))
	    { auto v = e.v[l]; if (coll.IsVertexCollapsed(v)) {
		// cout << "--- first un-coll " << coll.CollapsedEdge(v) << endl;
		coll.UncollapseEdge(coll.CollapsedEdge(v)); }}
	  if (!coll.IsEdgeCollapsed(e)) { /*cout << "actually cross coll edge " << e << endl;*/  coll.CollapseEdge(e); } // why should this be already collapsed??
	  coll.FixEdge(e);
	}
	else {
	  int kv = -val;
	  AMG_Node<NT_VERTEX> v = verts[kv];
	  // cout << "cross fix vertex " << kv << " of eqc " << eqc << ", vnr " << v << endl;
	  if (coll.IsVertexCollapsed(v)) { const auto & ce = coll.CollapsedEdge(v); if (!coll.IsEdgeFixed(ce)) coll.UncollapseEdge(ce); }
	  else if (coll.IsVertexGrounded(v) && !coll.IsVertexFixed(v)) coll.UngroundVertex(v);
	  coll.FixVertex(v);
	}
      }
    }

    size_t ncol = 0;
    for (const auto & e : mesh.template GetNodes<NT_EDGE>()) {
      // cout << "ege " << e << " coll? " << coll.IsEdgeCollapsed(e) << endl;
      if (coll.IsEdgeCollapsed(e)) ncol++;
    }
    double cfr = (mesh.template GetNN<NT_VERTEX>()==0) ? 0 : (1.0*ncol)/mesh.template GetNN<NT_VERTEX>();
    cout << "hierarch ncol: " << ncol << " of " << mesh.template GetNN<NT_EDGE>() << ", frac " << cfr << endl;

    if (options->post_coll) {
      block_crs.Collapse(mesh, coll);
      size_t ncol = 0;
      for (const auto & e : mesh.template GetNodes<NT_EDGE>()) { if (coll.IsEdgeCollapsed(e)) ncol++; }
      double cfr = (mesh.template GetNN<NT_VERTEX>()==0) ? 0 : (1.0*ncol)/mesh.template GetNN<NT_VERTEX>();
      cout << "post hierarch ncol: " << ncol << " of " << mesh.template GetNN<NT_EDGE>() << ", frac " << cfr << endl;
    }
  } // HierarchicVWC :: Collapse

  template<class TMESH>
  CoarseMap<TMESH> :: CoarseMap (shared_ptr<TMESH> _mesh, VWCoarseningData::CollapseTracker &coll)
    : BaseCoarseMap(), GridMapStep<TMESH>(_mesh)
  {
    // cout << "coarse map, V" << endl;
    BuildVertexMap(coll);
    // cout << "coarse map, E?" << _mesh->template HasNodes<NT_EDGE>()<< endl;
    if (_mesh->template HasNodes<NT_EDGE>()) BuildEdgeMap(coll);
    // if (_mesh->template HasNodes<NT_FACE>()) BuildMap<NT_FACE>(coll);
    // if (_mesh->template HasNodes<NT_CELL>()) BuildMap<NT_CELL>(coll);
    // cout << " make mapped mesh " << endl;
    // cout << " mm:  " << mm << endl;
    // cout << " mm id:  " << typeid(*mm).name() << endl;
    auto mapped_btm = mesh->MapBTM(*this);
    if constexpr(std::is_same<TMESH, BlockTM>::value==1) {
	mapped_mesh.reset(mapped_btm);
      }
    else {
      // this->mapped_mesh = std::apply( [&mapped_btm](auto& ...x) { return make_shared<TMESH> ( move(*mapped_btm), x...); }, mesh->MapData(*this));
      // cout << "make coarse alg-mesh, mesh now is " << endl << *mesh << endl;
      this->mapped_mesh = make_shared<TMESH> ( move(*mapped_btm), mesh->MapData(*this) );
    }
		  
    // if constexpr(std::is_base_of<BlockAlgMesh<>, TMESH>::value == 1) {
    // 	this->mapped_mesh = make_shared<TMESH> ( move(*mapped_btm), mesh->MapData(*this) );
    //   }
    // else {
    //   mapped_mesh = move(mapped_btm);
    // }
  }

  template<class TMESH>
  void CoarseMap<TMESH> :: BuildVertexMap (VWCoarseningData::CollapseTracker& coll)
  {
    auto is_invalid = [](auto val)->bool {return val==decltype(val)(-1); };
    static Timer t("CoarseMap - BuildMap<NT_VERTEX>");
    RegionTimer rt(t);
    auto & vmap = node_maps[NT_VERTEX];
    BlockTM & bmesh(*mesh);
    auto & NV = NN[NT_VERTEX];
    NV = bmesh.template GetNN<NT_VERTEX>();
    auto & NVC = mapped_NN[NT_VERTEX];
    NVC = 0;
    auto sp_eqc_h = bmesh.GetEQCHierarchy();
    const auto & eqc_h = *(sp_eqc_h);
    auto comm = eqc_h.GetCommunicator();
    size_t neqcs = eqc_h.GetNEQCS();
    Array<size_t> eqcs(neqcs);
    for (auto k:Range(neqcs)) eqcs[k] = k;

    // {
    //   cout << endl;
    //   cout << " check coll status: " << endl;
    //   auto alles = bmesh.GetNodes<NT_EDGE>();
    //   for (auto k : Range(bmesh.GetNN<NT_EDGE>())) {
    // 	cout << "edge " << k << "/" <<  bmesh.GetNN<NT_EDGE>() << " coll " << coll.IsEdgeCollapsed(alles[k]) << " , edge: " << alles[k] << endl;
    //   }
    //   cout << endl;
    // }
      
    /**
       collapsed cross edges
       eq,vnr;eq,vnr
    **/
    typedef INT<4> tcce;
    Table<tcce> cce;
    auto less_cce = [](auto a, auto b){
      for (int l : {0,2,1,3}) {
	if (a[l]<b[l]) return true;
	else if (a[l]>b[l]) return false;
      }
      return false;
    };
    {
      TableCreator<tcce> ccce(neqcs);
      while (!ccce.Done()) {
	for (auto eqc:eqcs) {
	  if (!eqc_h.IsMasterOfEQC(eqc)) continue;
	  auto pad_edges = bmesh.template GetCNodes<NT_EDGE>(eqc);
	  for (const auto & edge:pad_edges)
	    if (coll.IsEdgeCollapsed(edge)) {
	      auto v1 = edge.v[0];
	      int v1l = bmesh.template MapNodeToEQC<NT_VERTEX>(v1);
	      auto e1 = bmesh.template GetEqcOfNode<NT_VERTEX>(v1);
	      int e1id = eqc_h.GetEQCID(e1);
	      INT<2, int> V1({e1id,v1l});
	      auto v2 = edge.v[1];
	      int v2l = bmesh.template MapNodeToEQC<NT_VERTEX>(v2);
	      auto e2 = bmesh.template GetEqcOfNode<NT_VERTEX>(v2);
	      int e2id = eqc_h.GetEQCID(e2);
	      INT<2, int> V2({e2id,v2l});
	      const bool smaller = (V1<V2);
	      auto meq = eqc_h.GetMergedEQC(e1, e2);
	      // cout << "edge was " << edge << ", ";
	      if (smaller) {
		tcce E({V1[0], V1[1], V2[0], V2[1]});
		// cout << " add E " << E << " (v1) " << endl;
		ccce.Add(meq, E);
	      }
	      else {
		tcce E({V2[0], V2[1], V1[0], V1[1]});
		// cout << " add E " << E << " (v2) " << endl;
		ccce.Add(meq, E);
	      }
	    }
	}
	ccce++;
      }
      cce = ccce.MoveTable();
    }
    for (auto k:Range(cce.Size()))
      QuickSort(cce[k], less_cce);
    // cout << endl << "cce: " << endl << cce << endl;
    auto rcce = ReduceTable<tcce,tcce> (cce, eqcs, sp_eqc_h,
					[&](const auto & tab) { return merge_arrays(tab, less_cce); });
    // cout << endl << "rcce: " << endl << rcce << endl;
    vmap.SetSize(NV);
    if (NV) vmap = -1;
    Array<size_t> v_cnt(eqcs.Size());
    v_cnt = 0;
    /** 
	Mark vertices that are cross-collapsed have to do this because 
	cross-edges are written not to their own eqc, but to the eqc 
	of the resulting coarse vertex.
	(-> this way no wrongfully  single map for off-proc collapsed vertex)
    **/
    for (auto eqc : Range(eqcs.Size())) {
      auto row = rcce[eqc];
      v_cnt[eqc] = row.Size();
      FlatTM block = bmesh.GetBlock(eqc);
      for (auto j:Range(row.Size())) {
	bool has_one = false;
	for (int l:{0,2}) {
	  auto v_eqc = eqc_h.GetEQCOfID(row[j][l+0]);
	  auto v_locnum = row[j][l+1];
	  /** We have to have the coarse vertex, so we have to have at least one of the vertices! **/
	  if (is_invalid(v_eqc)) continue;
	  vmap[bmesh.MapENodeFromEQC<NT_VERTEX>(v_locnum, v_eqc)] = j; // what goes here does not matter (just not -1)
	}
      }
    }

    /** 
	per eqc: [singles, in-e, cross-e] 
    **/
    size_t displ = 0;
    size_t last_displ = 0;
    for (auto eqc : Range(eqcs.Size())) {
      FlatTM block = bmesh.GetBlock(eqc);
      auto verts = block.template GetNodes<NT_VERTEX>();
      auto lld = last_displ;
      for (auto v : verts)
	if ( (is_invalid(vmap[v])) && (!coll.IsVertexCollapsed(v)) && (!coll.IsVertexGrounded(v)) ) {
	  // cout << "set vmap2 " << v << " -> " << displ << endl;
	  vmap[v] = displ++;
	}
      // 	else if (coll.IsVertexCollapsed(v)) cout << v << " is coll!" << endl;
      // 	else if (coll.IsVertexGrounded(v)) cout << v << " is grounded!" << endl;
      // 	else cout << v << " is already mapped????" << endl;
      // cout << " eqc " << eqc << " has " << displ-lld << " type1" << endl;
      lld = displ;
      auto row = rcce[eqc];
      /** vmap for collapsed (eqc-)edges **/
      auto edges = block.template GetNodes<NT_EDGE>();
      for (const auto & edge : edges) {
	if (coll.IsEdgeCollapsed(edge)) {
	  // for (auto l:Range(2)) { cout << "set vmap3-" << l << " " << edge.v[l] << " -> " << displ << endl; }
	  for (auto l:Range(2)) { vmap[edge.v[l]] = displ; }
	  displ++;
	}
      }
      // cout << " eqc " << eqc << " has " << displ-lld << " type2" << endl;
      lld = displ;
      /** modify vmap for collapsed (cross-)edges **/
      for (auto j:Range(row.Size())) {
	bool has_one = false;
	// cout << " eqc/j " << eqc << " " << j << " rcce entry: " << row[j] << endl;
	for (int l:{0,2}) {
	  auto v_eqc = eqc_h.GetEQCOfID(row[j][l+0]);
	  auto v_locnum = row[j][l+1];
	  /** We have to have the coarse vertex, so we have to have at least one of the vertices! **/
	  if (is_invalid(v_eqc)) continue;
	  vmap[bmesh.MapENodeFromEQC<NT_VERTEX>(v_locnum, v_eqc)] = displ;
	  // cout << "set vmap1-" << l << " " << bmesh.MapENodeFromEQC<NT_VERTEX>(v_locnum, v_eqc) << " -> " << displ << endl;
	}
	displ++;
      }
      // cout << " eqc " << eqc << " has " << displ-lld << " type3" << endl;
      lld = displ;
      v_cnt[eqc] = displ - last_displ;
      last_displ = displ;
    }
    // cout << "have vert map: " << endl; prow2(vmap); cout << endl;
    /** TODO: weave this in earlier, but for now do not touch **/
    mapped_eqc_firsti[NT_VERTEX].SetSize(eqcs.Size()+1);
    mapped_eqc_firsti[NT_VERTEX][0] = 0;
    for (auto k:Range(eqcs.Size()))
      mapped_eqc_firsti[NT_VERTEX][k+1] = mapped_eqc_firsti[NT_VERTEX][k] + v_cnt[k];
    NVC = mapped_eqc_firsti[NT_VERTEX].Last();
    mapped_cross_firsti[NT_VERTEX].SetSize(eqcs.Size()+1);
    // no cross-vertices per definition
    mapped_cross_firsti[NT_VERTEX] = mapped_eqc_firsti[NT_VERTEX].Last();

    // cout << "mapped eqc_v_firsti : " << endl << mapped_eqc_firsti[NT_VERTEX] << endl;

    // cout << endl << "have vertex map: " << endl; prow2(vmap); cout << endl;

    // Array<int> ones(eqcs.Size()); ones = 1;
    // Table<int> cnt(ones);
    // for (auto k : Range(eqcs.Size()))
    //   { cnt[k][0] = mapped_eqc_firsti[NT_VERTEX][k+1] - mapped_eqc_firsti[NT_VERTEX][k]; }
    // int cntcs = 0;
    // bool isok = true;
    // auto dontcare = ReduceTable<int, int> (cnt, eqcs, sp_eqc_h,
    // 					   [&](auto & tab) {
    // 					     int nrows = tab.Size();
    // 					     Array<int> out(nrows);
    // 					     cntcs++;
    // 					     if (!nrows) return out;
    // 					     if (nrows==1) { out[0] = tab[0][0]; return out; }
    // 					     bool this_isok = true;
    // 					     for (auto k : Range(size_t(1), tab.Size()))
    // 					       if (tab[k][0] != tab[0][0]) { this_isok = false; }
    // 					     if (!this_isok) {
    // 					       isok = this_isok;
    // 					       cout << "NOT OK CHECK EQC S nr : " << cntcs-1 << endl;
    // 					       print_ft(cout, tab); cout << endl;
    // 					     }
    // 					     return out;
    // 					   });
    // if (!isok) throw Exception("INVALID V SIZES");
  } // CoarseMap :: BuildVertexMap

  template<class TMESH>
  void CoarseMap<TMESH> :: BuildEdgeMap (VWCoarseningData::CollapseTracker& coll)
  {
    /**
       Consistent map for coarse nodes:
       - Locally build table of all nodes that change eqc. (all CNodes)
       - "AllGather" the rows. Having an edge multiple times does not matter here.
         (because we use HTs)
       - Two HashTables per eqc (one eqc, one cross)
       This gives map to local-per-eqc-and-type enumeration.
       Add offstes to get the final map!
     **/
    // cout << "CoarseMap - map edges" << endl;
    static Timer t("CoarseMap - map edges");
    RegionTimer rt(t);
    const BlockTM & bmesh = *mesh;
    const auto & vmap = node_maps[NT_VERTEX];
    auto & node_map = node_maps[NT_EDGE];
    auto & NNODES = NN[NT_EDGE];
    NNODES = bmesh.template GetNN<NT_EDGE>();
    auto & NNODES_COARSE = mapped_NN[NT_EDGE];
    NNODES_COARSE = 0;
    auto & coarse_nodes = mapped_E;
    node_map.SetSize(NNODES); node_map = -1;
    auto sp_eqc_h = bmesh.GetEQCHierarchy();
    const auto & eqc_h = *sp_eqc_h;
    auto comm = eqc_h.GetCommunicator();
    size_t neqcs = eqc_h.GetNEQCS();
    Array<size_t> eqcs(neqcs);
    for (auto k:Range(neqcs)) eqcs[k] = k;
    // eqc-changing coarse edges
    // cout << "check coll " << endl;
    // for (const auto & e : bmesh.template GetNodes<NT_EDGE>())
    //   cout << "edge " << e << " , coll ? " << coll.IsEdgeCollapsed(e) << endl;
    // cout << endl;
    Table<AMG_CNode<NT_EDGE>> tadd_edges;
    {
      TableCreator<AMG_CNode<NT_EDGE>> ct(neqcs);
      auto lam_e = [&](auto feq, const auto & ar) {
	for (const auto & e : ar) {
	  if ( coll.IsEdgeCollapsed(e) || (vmap[e.v[0]]==-1) || (vmap[e.v[1]]==-1) )
	    continue;
	  auto v0 = vmap[e.v[0]];
	  auto v1 = vmap[e.v[1]];
	  AMG_Node<NT_VERTEX> mv0 = CN_to_EQC<NT_VERTEX>(v0);
	  AMG_Node<NT_VERTEX> mv1 = CN_to_EQC<NT_VERTEX>(v1);
	  int ceq0 = EQC_of_CN<NT_VERTEX>(v0);
	  int ceq1 = EQC_of_CN<NT_VERTEX>(v1);
	  auto ceq0id = eqc_h.GetEQCID(ceq0);
	  auto ceq1id = eqc_h.GetEQCID(ceq1);
	  auto ceq = eqc_h.GetCommonEQC(ceq0, ceq1);
	  if (eqc_h.IsLEQ(ceq, feq)) continue;// can be computed locally
	  if ((mv0>mv1) ||
	     ((mv0==mv1) && (ceq0id>ceq1id)) ){
	    swap(v0,v1);
	    swap(mv0,mv1);
	    swap(ceq0id, ceq1id);
	  }
	  AMG_CNode<NT_EDGE> cnode;
	  cnode.v = {mv0, mv1};
	  cnode.eqc = {ceq0id, ceq1id};
	  ct.Add(ceq, cnode);
	}
      };
      while (!ct.Done()) {
	for (auto feq : Range(neqcs)) {
	  if (!eqc_h.IsMasterOfEQC(feq)) continue;
	  auto pedges = bmesh.template GetCNodes<NT_EDGE>(feq);
	  lam_e(feq, pedges);
	  auto iedges = bmesh.template GetENodes<NT_EDGE>(feq);
	  lam_e(feq, iedges);
	}
	ct++;
      }
      tadd_edges = ct.MoveTable();
    }
    // cout << "tadd_edges: " << endl << tadd_edges << endl;
    auto add_edges = ReduceTable<AMG_CNode<NT_EDGE>,AMG_CNode<NT_EDGE>>(tadd_edges, eqcs, sp_eqc_h, [](const auto & tab) {
    	Array<AMG_CNode<NT_EDGE>> out;
    	if (!tab.Size()) return out;
    	size_t s = 0;
    	for (auto row:tab) s += row.Size();
    	out.SetSize(s); s = 0;
    	for (auto row:tab) {
    	  for (auto j:Range(row.Size())) {
    	    out[s++] = row[j]; } }
    	return out;
      });
    // cout << "add_edges: " << endl << add_edges << endl;
    // make hash-tables
    typedef ClosedHashTable<INT<2, int>, int> HT1;
    Array<HT1*> hash_ie(neqcs);
    typedef ClosedHashTable<INT<4, int>, int> HT2;
    Array<HT2*> hash_ce(neqcs);
    auto & disp_ie = mapped_eqc_firsti[NT_EDGE];
    disp_ie.SetSize(neqcs+1);
    disp_ie = 0;
    auto & disp_ce = mapped_cross_firsti[NT_EDGE];
    disp_ce.SetSize(neqcs+1);
    disp_ce = 0;
    for (auto k:Range(neqcs)) {
      size_t c1 = 0;
      for (const auto & v:add_edges[k]) if (v.eqc[0]==v.eqc[1]) c1++;
      FlatTM block = bmesh.GetBlock(k);
      size_t nie_max = c1 + block.GetNN<NT_EDGE>() * 1.2;
      hash_ie[k] = new HT1(nie_max);
      size_t nce_max = add_edges[k].Size() - c1 + block.GetCNN<NT_EDGE>() * 1.2;
      hash_ce[k] = new HT2(nce_max);
    }
    size_t pos = 0;
    for (auto eqc:Range(neqcs)) {
      auto row = add_edges[eqc];
      for (auto j:Range(row.Size())) {
    	auto e = row[j];
    	auto eq0id = e.eqc[0];
    	auto eq1id = e.eqc[1];
    	auto vc0_loc = e.v[0];
    	auto vc1_loc = e.v[1];
    	auto eq0 = eqc_h.GetEQCOfID(eq0id);
    	auto eq1 = eqc_h.GetEQCOfID(eq1id);
    	bool isin = eq0==eq1;
    	if (isin) {
    	  auto & hash =  *hash_ie[eqc];
    	  INT<2, int> ec(vc0_loc, vc1_loc);
    	  if (hash.PositionCreate(ec, pos)) {
    	    hash.SetData(pos, disp_ie[eqc+1]);
    	    disp_ie[eqc+1]++;
    	  }
    	}
    	else {
    	  auto & hash =  *hash_ce[eqc];
    	  INT<4, int> ec(eq0id, eq1id, vc0_loc, vc1_loc);
    	  if (hash.PositionCreate(ec, pos)) {
    	    hash.SetData(pos, disp_ce[eqc+1]);
    	    disp_ce[eqc+1]++;
    	  }
    	}
      }
    }
    // local-eqc-map
    Array<INT<3,int> > node_map2(NNODES); // in/cross; eqc; loc-nr
    node_map2 = -1;
    auto fine_nodes = bmesh.template GetNodes<NT_EDGE>();
    for (const auto & e : fine_nodes) {
      if ( coll.IsEdgeCollapsed(e) || (vmap[e.v[0]]==-1) || (vmap[e.v[1]]==-1) )
    	continue;
      int vc0 = vmap[e.v[0]];
      int vc1 = vmap[e.v[1]];
      int vc0_loc = CN_to_EQC<NT_VERTEX>(vc0);
      int vc1_loc = CN_to_EQC<NT_VERTEX>(vc1);
      int eq0 = EQC_of_CN<NT_VERTEX>(vc0);
      int eq1 = EQC_of_CN<NT_VERTEX>(vc1);
      int eq0id = eqc_h.GetEQCID(eq0);
      int eq1id = eqc_h.GetEQCID(eq1);
      if ( (vc0_loc>vc1_loc) ||
    	  ( (vc0_loc==vc1_loc) && (eq0id > eq1id)) ) {
    	swap(vc0,vc1);
    	swap(vc0_loc,vc1_loc);
    	swap(eq0id, eq1id);
      }
      int eq = eqc_h.GetCommonEQC(eq0, eq1);
      bool isin = eq0==eq1;
      node_map2[e.id][0] = isin ? 1 : 0;
      node_map2[e.id][1] = eq;
      if (isin) {
    	auto & hash = *hash_ie[eq];
    	auto & count = disp_ie[eq+1];
    	INT<2, int> ec(vc0_loc, vc1_loc);
    	if (hash.PositionCreate(ec,pos)) {
    	  node_map2[e.id][2] = count;
    	  hash.SetData(pos, count++);
    	}
    	else {
    	  hash.GetData(pos, node_map2[e.id][2]);
    	}
      }
      else {
    	auto & hash = *hash_ce[eq];
    	auto & count = disp_ce[eq+1];
    	INT<4, int> ec(eq0id, eq1id, vc0_loc, vc1_loc);
    	if (hash.PositionCreate(ec,pos)) {
    	  node_map2[e.id][2] = count;
    	  hash.SetData(pos, count++);
    	}
    	else {
    	  hash.GetData(pos, node_map2[e.id][2]);
    	}
      }
    }
    for (auto k:Range(size_t(1), neqcs+1)) {
      disp_ie[k] += disp_ie[k-1];
    }
    size_t NECI = disp_ie.Last();
    for (auto k:Range(size_t(1), neqcs+1)) {
      disp_ce[k] += disp_ce[k-1];
    }
    size_t NECC = disp_ce.Last();
    node_map.SetSize(NNODES); if (NNODES) node_map = -1;
    // cout << "NECI NECC " << NECI << " " << NECC << endl;
    NNODES_COARSE = NECI + NECC;
    coarse_nodes.SetSize(NNODES_COARSE);
    for (auto k:Range(NNODES)) {
      const auto & node = fine_nodes[k];
      auto t = node_map2[k];
      if (t[0]==-1) continue;
      auto cv0 = vmap[node.v[0]];
      auto cv1 = vmap[node.v[1]];
      if (cv0>cv1) swap(cv0,cv1);
      node_map[k] = (t[0]==1 ? disp_ie[t[1]] : NECI+disp_ce[t[1]] ) + t[2];
      coarse_nodes[node_map[k]] = {cv0, cv1};
      cout << " loc edge " << node << " became c-ndode " << node_map[k] << " " << cv0 << " " << cv1 << endl;
    }
    // ok, now add_edges
    for (auto eqc:Range(neqcs)) {
      auto row = add_edges[eqc];
      for (auto j:Range(row.Size())) {
    	auto e = row[j];
    	auto eq0id = e.eqc[0];
    	auto eq1id = e.eqc[1];
    	auto vc0_loc = e.v[0];
    	auto vc1_loc = e.v[1];
    	auto eq0 = eqc_h.GetEQCOfID(eq0id);
    	auto eq1 = eqc_h.GetEQCOfID(eq1id);
    	int vc0 = CN_of_EQC<NT_VERTEX>(eq0, vc0_loc);
    	int vc1 = CN_of_EQC<NT_VERTEX>(eq1, vc1_loc);
	bool swapped = (vc0>vc1);
    	// if (vc0>vc1) { swap(vc0,vc1); swap(eq0id,eq1id); }
    	if (vc0>vc1) swap(vc0,vc1);
    	bool isin = eq0==eq1;
	// cout << "add edge " << eqc << " " << j << " (swap " << swapped << ") : ";
    	if (isin) {
    	  auto & hash =  *hash_ie[eqc];
    	  INT<2, int> ecl(vc0_loc, vc1_loc);
    	  auto ct = hash[ecl];
    	  auto cid = disp_ie[eqc]+ct;
    	  coarse_nodes[cid] = {vc0, vc1};
	  // cout << " add edge " << e << " became eqc-node " << cid << " " << vc0 << " " << vc1 << endl;
	  // cout << "locs " << vc0_loc << " " << vc1_loc << endl;
    	}
    	else {
    	  auto & hash =  *hash_ce[eqc];
    	  INT<4, size_t> ecl(eq0id, eq1id, vc0_loc, vc1_loc);
    	  auto ct = hash[ecl];
    	  auto cid = NECI + disp_ce[eqc]+ct;
    	  coarse_nodes[cid] = {vc0, vc1};
	  // cout << " add edge " << e << " became cross-node " << eqc << " " << cid << " " << vc0 << " " << vc1 << endl;
	  // cout << "eqs " << eq0 << " " << eq1 << endl;
	  // cout << "locs " << vc0_loc << " " << vc1_loc << endl;
    	}
      }
    }
    // dont need mapped_eqc_nodes ?
    for (auto x : hash_ie) delete x;
    for (auto x : hash_ce) delete x;
    // cout << endl << "have coarse edges: " << endl; prow2(coarse_nodes); cout << endl;
    // cout << endl << "have edge map: " << endl; prow2(node_map); cout << endl;
  } // CoarseMap :: BuildEdgeMap

} // namespace amg

#include "amg_tcs.hpp"
