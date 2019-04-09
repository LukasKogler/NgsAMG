#include "amg.hpp"

namespace amg {

  /** BaseAlgebraicMesh **/

  BaseAlgebraicMesh :: BaseAlgebraicMesh (size_t nv, size_t ne, size_t nf)
    : n_verts(nv), n_edges(ne), n_faces(nf),
      n_verts_glob(n_verts), n_edges_glob(n_edges), n_faces_glob(n_faces)
  {
    has_verts = (nv!=0);
    has_edges = (ne!=0);
    has_faces = (nf!=0);
  }

  shared_ptr<BaseAlgebraicMesh>
  BaseAlgebraicMesh :: Map (CoarseMap & cmap)
  { return nullptr; }

  shared_ptr<BaseAlgebraicMesh>
  BaseAlgebraicMesh :: Map (GridContractMap & cmap)
  { return nullptr; }

  shared_ptr<BaseAlgebraicMesh>
  BaseAlgebraicMesh :: Map (NodeDiscardMap & cmap)
  { return nullptr; }

  
  shared_ptr<BaseAlgebraicMesh>
  BaseAlgebraicMesh :: GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap,
					   Array<double> && awv, Array<double> && awe)
  {
    auto cmesh = GenerateCoarseMesh(cmap);
    cmesh->SetVertexWeights(std::move(awv));
    cmesh->SetEdgeWeights(std::move(awe));
    return cmesh;
  }
  
  shared_ptr<BaseAlgebraicMesh>
  BaseAlgebraicMesh :: GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap)
  {
    auto CNV = cmap->GetNVC();
    auto CNE = cmap->GetNEC();
    size_t CNF = 0;
    auto cmesh = make_shared<BaseAlgebraicMesh> (CNV, CNE, CNF);
    const auto & cedges = cmap->GetCEDGES();
    Array<idedge> CE(CNE);
    for(auto k:Range(CNE))
      CE[k] = idedge({cedges[k], k});
    cmesh->SetEdges(std::move(CE));
    return std::move(cmesh);
  }

  
  /** End BaseAlgebraicMesh **/
  
#ifdef PARALLEL
  
  /** BlockedAlgebraicMesh **/
  
  BlockedAlgebraicMesh :: BlockedAlgebraicMesh
  ( size_t nv, size_t ne, size_t nf,
    Array<edge> && edges, Array<face> && faces,
    Array<double> && awv, Array<double> && awe, Array<double> && awf,
    Array<size_t> && eqc_v, const shared_ptr<EQCHierarchy> & aeqc_h)
    : BaseAlgebraicMesh(nv, ne, nf), eqc_h(aeqc_h), c_eqc_h(aeqc_h)
  {
    static Timer t("BlockedAlgebraicMesh constructor");
    RegionTimer rt(t);
    
    n_verts_glob = 0;
    n_edges_glob = 0;
    n_faces_glob = 0;
    
    this->BuildVertexTables(std::move(awv), std::move(eqc_v));
    this->BuildEdgeTables(std::move(edges), std::move(awe));
    // this->BuildFaceTables(std::move(wv), std::move(wf));

    this->BuildMeshBlocks();

    return;
  } // end BlockedAlgebraicMesh(..)

  void BlockedAlgebraicMesh :: BuildVertexTables(Array<double> && awv, Array<size_t> && eqc_v) {

    static Timer t("AlgMesh - VertexTable");
    RegionTimer rt(t);
    
    /** vertex-eqc table **/
    TableCreator<size_t> cvt(eqc_h->GetNEQCS());
    {
      while(!cvt.Done()) {
	for(auto k:Range(n_verts))
	  cvt.Add(eqc_v[k], k);
	cvt++;
      }
      this->eqc_verts = cvt.MoveTable();
    }

    /** vertex-mappings **/
    g2l_v.SetSize(n_verts);
    vertex_in_eqc.SetSize(n_verts);
    for(auto k:Range(eqc_verts.Size()))
      for(auto j:Range(eqc_verts[k].Size()))
	{
	  vertex_in_eqc[eqc_verts[k][j]] = k;
	  g2l_v[eqc_verts[k][j]] = j;
	}

    verts = Array<vertex>(n_verts, &(eqc_verts[0][0]));
    verts.NothingToDelete();

    /** eqcs & types **/
    int neqcs = eqc_verts.Size();
    this->eqcs.SetSize(neqcs);
    // this->eqc_types.SetSize(neqcs);
    for(auto k:Range(neqcs)) {
      eqcs[k] = k;
      // int ndp = eqc_h->GetDistantProcs(k).Size();
      // if(ndp>1)
      // 	eqc_types[k] = WIRE_EQC;
      // else if(ndp==1)
      // 	eqc_types[k] = FACE_EQC;
      // else
      // 	eqc_types[k] = VOL_EQC;
    }

    /** cumulate v-wts **/
    // NOTE: vertex-data is saved as distributed!
    // this->wv = CumulateVertexData(awv);
    this->wv = move(awv);


    /** vertex-mpi types **/
    // auto all_dps = eqc_h->GetDistantProcs();
    // TableCreator<int> cinds(all_dps.Size());
    // while(!cinds.Done()) {
    //   for(auto k:Range(n_verts)) {
    // 	auto dps = eqc_h->GetDistantProcs(vertex_in_eqc[k]);
    // 	if(!dps.Size()) continue;
    // 	for(auto p:dps)
    // 	  cinds.Add(all_dps.Pos(p), k);
    //   }
    //   cinds++;
    // }
    // Table<int> inds = cinds.MoveTable();
    
    // this->v_types.SetSize(eqcs.Size());
    // MPI_Datatype tscal = MyGetMPIType<double>();
    // Array<int> blocklen;

    // for(auto k:Range(inds.Size())) {
    //   blocklen.SetSize(inds[k].Size());
    //   blocklen = 1;
    //   int nev = eqc_verts[k].Size();
    //   MPI_Type_indexed( nev, &blocklen[0], &inds[k][0], tscal, &v_types[k]);
    //   MPI_Type_commit(&v_types[k]);
    // }


    return;
  } // end BuildVertexTables

  void BlockedAlgebraicMesh :: BuildEdgeTables
  (const Array<edge> & aedges, const Array<double> & awe)
  {

    static Timer t("AlgMesh - EdgeTable");
    RegionTimer rt(t);

    size_t n_edges_p = aedges.Size();
    
    /** local edge tables **/
    BitArray is_pad (n_edges_p);
    is_pad.Clear();
    auto lam1 = [this, &aedges, &is_pad](int k)->size_t { 
      if(vertex_in_eqc[aedges[k][0]]==vertex_in_eqc[aedges[k][1]])
	return vertex_in_eqc[aedges[k][0]];
      else
	is_pad.Set(k);
      return NO_EQC;
    };
    auto lam2 = [this, &aedges, &awe](int k)->weighted_edge 
      { return (weighted_edge) {edge({g2l_v[aedges[k][0]],g2l_v[aedges[k][1]]}),awe[k]}; };
    Table<weighted_edge> etent = eqc_h->PartitionData<weighted_edge, decltype(lam1), decltype(lam2)>(n_edges_p, lam1, lam2);

    for(auto row:etent)
      QuickSort(row, [](auto & a, auto & b) { return a<b; });

    /** merge arrays, sum up weights **/
    auto merge_arrays = [](auto & tab_in)
      {
	int max_size = 0;
	for(auto k:Range(tab_in.Size()))
	  if(tab_in[k].Size()>max_size)
	    max_size = tab_in[k].Size();
	max_size += (105*max_size)/100; 
	Array<typename std::remove_reference<decltype(tab_in[0][0])>::type > out(max_size);
	out.SetSize(0);
	Array<int> count(tab_in.Size());
	count = 0;
	int ndone = 0;
	BitArray hasmin(tab_in.Size());
	hasmin.Clear();
	int rofmin = -1;
	for(int k=0;((k<tab_in.Size())&&(rofmin==-1));k++)
	  if(tab_in[k].Size())
	    rofmin = k;
	if(rofmin==-1) //empty input
	  return out;
	for(auto k:Range(tab_in.Size()))
	  if(!tab_in[k].Size())
	    ndone++;
	auto min_datum = tab_in[rofmin][0];
	while(ndone<tab_in.Size()) {
	  for(auto k:Range(tab_in.Size()))
	    if(count[k]<tab_in[k].Size()) {
	      if(tab_in[k][count[k]]==min_datum) {
		hasmin.Set(k);
	      }
	      else if(tab_in[k][count[k]]<min_datum) {
		hasmin.Clear();
		hasmin.Set(k);
		min_datum = tab_in[k][count[k]];
		rofmin = k;
	      }
	    }
	  min_datum.wt = 0;
	  for(auto k:Range(tab_in.Size()))
	    if(hasmin.Test(k)) {
	      min_datum.wt += tab_in[k][count[k]++].wt;
	      if(count[k]==tab_in[k].Size())
		ndone++;
	    }
	  out.Append(min_datum);
	  rofmin = -1;
	  for(int k=0;((k<tab_in.Size())&&(rofmin==-1));k++)
	    if(count[k]<tab_in[k].Size()) {
	      rofmin = k;
	      min_datum = tab_in[k][count[k]];
	    }
	  hasmin.Clear();
	}
	return out;
      };
    
    /** merge local tables **/
    Array<size_t> eqc_array(etent.Size());
    for(auto k:Range(etent.Size()))
      eqc_array[k] = k;
    Table<weighted_edge> rtab = ReduceTable<weighted_edge, weighted_edge>(etent, eqc_array, eqc_h, merge_arrays);
    Table<weighted_edge>* in_eqc_etab = &rtab;



    /** CROSS-EQC EDGES **/    

    /** local pad edges **/
    int n_tent_pad = is_pad.NumSet();
    int count = 0;
    Array<int> tpi(n_tent_pad); //temp pad index
    for(auto k:Range(is_pad.Size()))
      if(is_pad.Test(k))
	tpi[count++] = k;

    auto lam_pad_eqc = [this, &aedges, &tpi](int k)->size_t
      { 
	auto e = aedges[tpi[k]];
	auto eqc0 = eqcs[vertex_in_eqc[e[0]]];
	auto eqc1 = eqcs[vertex_in_eqc[e[1]]];
	return eqc_h->GetCommonEQC(eqc0, eqc1);
      };    
    auto lam_pad_edge = [this, &aedges, &awe, &tpi](int k)->weighted_cross_edge 
      { 
	
	auto e = aedges[tpi[k]];
	auto w = awe[tpi[k]];
	int eqc[2];
	for(auto k:Range(2))
	  eqc[k] = eqc_h->GetEQCID(eqcs[vertex_in_eqc[e[k]]]);
        int l = (eqc[0]<eqc[1])?0:1;
	INT<2> eqcs(eqc[l], eqc[1-l]);
	edge e2(g2l_v[e[l]], g2l_v[e[1-l]]);
	return weighted_cross_edge({eqcs, e2, w} );
      };
    
    Table<weighted_cross_edge> e_pad_tent = eqc_h->PartitionData<weighted_cross_edge, decltype(lam_pad_eqc), decltype(lam_pad_edge)>(n_tent_pad, lam_pad_eqc, lam_pad_edge);

    for(auto row:e_pad_tent)
      QuickSort(row, [](auto & a, auto & b) { return a<b; });

    /** merge cross edge s**/
    Table<weighted_cross_edge> red_cwt = ReduceTable<weighted_cross_edge,weighted_cross_edge>(e_pad_tent, eqc_array, eqc_h, merge_arrays);
    Table<weighted_cross_edge> * cross_eqc_etab =  &red_cwt;


    /** Write edge-array and construct flattables for edges **/
    /** Displacements for in-eqc edge table **/
    int row_vol = -1;
    for(auto k:Range(eqcs.Size()))
      if(!eqc_h->GetDistantProcs(eqcs[k]).Size())
	row_vol = k;
    /** size+1 for eqc-part **/
    displs_eqc_edges.SetSize(eqcs.Size()+1);
    displs_eqc_edges = 0;
    size_t cur_eqc_d = 0;
    for(auto k:Range(eqcs.Size()))
      if(k!=row_vol) {
	cur_eqc_d += (*in_eqc_etab)[k].Size();
	displs_eqc_edges[k+1] = cur_eqc_d;
      }
      else {
	cur_eqc_d += etent[k].Size();
	displs_eqc_edges[k+1] = cur_eqc_d;
      }

    /** Displacements for cross-eqc edge table **/
    /** size+1 for eqc-part **/
    displs_pad_edges.SetSize(eqcs.Size()+1);
    displs_pad_edges = 0;
    size_t cur_pad_d = 0;
    for(auto k:Range(eqcs.Size()))
      if(k!=row_vol) {
	cur_pad_d += (*cross_eqc_etab)[k].Size();
	displs_pad_edges[k+1] = cur_pad_d;
      }
      else {
	cur_pad_d += e_pad_tent[k].Size();
	displs_pad_edges[k+1] = cur_pad_d;
      }
    
    n_edges = cur_eqc_d + cur_pad_d;

    /** weights needed for coarse level - array**/
    this->we.SetSize(n_edges);

    /** in-eqc-edges - table! **/
    edges.SetSize(n_edges);

    if(cur_eqc_d) {
      eqc_edges = FlatTable<idedge>(eqcs.Size(), &(displs_eqc_edges[0]), &(edges[0]));
    }
    else {
      eqc_edges = FlatTable<idedge>(eqcs.Size(), &(displs_eqc_edges[0]), nullptr);
    }
    

    /** cross-eqc-edges - table! **/    
    if(cur_pad_d) {
      FlatArray<idedge> dummy(cur_pad_d, &(edges[cur_eqc_d]));
      padding_edges.Assign(dummy);
      eqc_pad_edges = FlatTable<idedge>(eqcs.Size(), &(displs_pad_edges[0]), &(edges[cur_eqc_d]));
    }
    else {
      FlatArray<idedge> dummy(cur_pad_d, nullptr);
      padding_edges.Assign(dummy);
      // eqc_pad_edges = Table<edge>(eqcs.Size(),0); //assign, then table dies->frees mem in dest.
      eqc_pad_edges = FlatTable<idedge>(eqcs.Size(), &(displs_pad_edges[0]), nullptr);
    }

    // for(auto k:Range(n_edges))
    //   edges[k] = edge({INT<2>(-1,-1), -1});
    
    /** TODO: Currently, paralleltable does not do anything for local data; maybe change this? **/
    /** Write In-EQC (non-local) edges **/
    size_t ide = 0;
    size_t vol_here = 0;
    for(auto k:Range(in_eqc_etab->Size()))
      if(k!=row_vol) {
	for(auto j:Range((*in_eqc_etab)[k].Size())) {
	  we[ide] = (*in_eqc_etab)[k][j].wt;
	  edge & edge_loc = (*in_eqc_etab)[k][j].e;
	  edge edge_glob = edge(eqc_verts[k][edge_loc[0]],eqc_verts[k][edge_loc[1]]);
	  eqc_edges[k][j] = idedge({edge_glob, ide++});
	}
      }
      else {
	vol_here = ide;
	ide += etent[k].Size();
      }

    /** Copy In-EQC (local) edges from tentative eqc-edges **/
    if(row_vol!=-1)
      for(auto k:Range(etent[row_vol].Size())) {
	auto & e = etent[row_vol][k];
	edge edge_glob = edge(eqc_verts[row_vol][e.e[0]],eqc_verts[row_vol][e.e[1]]);
	we[vol_here] = e.wt;
	eqc_edges[row_vol][k] = idedge({edge_glob, vol_here++});
      }
    
    /** Write Cross-EQC (non-local) edges **/
    vertex v_temp[2];
    size_t pvol_here = -1;
    for(auto k:Range(cross_eqc_etab->Size()))
      if(k!=row_vol)
	for(auto j:Range((*cross_eqc_etab)[k].Size())) {
	  auto e = (*cross_eqc_etab)[k][j];
	  we[ide] = e.wt;
	  for(auto l:Range(2)) {
	    v_temp[l] = eqc_verts[eqcs.Pos(eqc_h->GetEQCOfID(e.eqc[l]))][e.e[l]];
	  }
	  int l = (v_temp[0]<v_temp[1])?0:1;
	  eqc_pad_edges[k][j] = idedge({ edge(v_temp[l], v_temp[1-l]), ide++ });
	}
      else {
	pvol_here = ide;
	ide += e_pad_tent[k].Size();
      }

    /** Write Cross-EQC (local) edges **/
    if(row_vol!=-1)
      for(auto k:Range(e_pad_tent[row_vol].Size())) {
	auto e = e_pad_tent[row_vol][k];
	we[pvol_here] = e.wt;
	for(auto l:Range(2))
	  v_temp[l] = eqc_verts[eqc_h->GetEQCOfID(e.eqc[l])][e.e[l]];
	int l = (v_temp[0]<v_temp[1])?0:1;
	eqc_pad_edges[row_vol][k] = {edge(v_temp[l], v_temp[1-l]), pvol_here++};
      }

    return;
  } // end BuildEdgeTables

  void BlockedAlgebraicMesh :: BuildMeshBlocks()
  {
    mesh_blocks.SetSize(eqcs.Size());
    for(auto k:Range(eqcs.Size())) {
      mesh_blocks[k] = make_shared<FlatAlgebraicMesh>
	(eqc_verts[k], eqc_edges[k], FlatArray<face>(0, NULL),
	 wv, we, wf,
	 cwv, cwe, cwf,
	 eqc_pad_edges[k]);
      mesh_blocks[k]->SetFreeVertices(v_free);
    }
    size_t anv = 0;
    size_t ane = 0;
    for(auto [eqc, block] : *this) {
      if(!eqc_h->IsMasterOfEQC(eqc)) continue;
      anv += block->NN<NT_VERTEX>();
      ane += block->NN<NT_EDGE>();
      ane += block->NN_Cross<NT_EDGE>();
    }
    auto comm = eqc_h->GetCommunicator();
    n_verts_glob = comm.AllReduce(anv, MPI_SUM);
    n_edges_glob = comm.AllReduce(ane, MPI_SUM);
    return;
    // padding_block = make_shared<FlatAlgebraicMesh>
    //   (padding_verts, padding_edges, FlatArray<face>(0, NULL),
    //    wv, we, wf,
    //    cwv, cwe, cwf);
  } // end BuildMeshBlockso

  shared_ptr<BaseAlgebraicMesh>
  BlockedAlgebraicMesh :: Map (NodeDiscardMap & cmap)
  {
    static Timer t("BlockedAlgebraicMesh :: Map (NodeDiscard))");
    RegionTimer rt(t);
    auto cmesh = make_shared<BlockedAlgebraicMesh>();
    /** EQCHierarchy **/
    cmesh->eqc_h = this->eqc_h;
    cmesh->c_eqc_h = this->c_eqc_h;
    cmesh->eqcs.SetSize(eqcs.Size());
    cmesh->eqcs = eqcs;
    /** verts **/
    cmesh->has_verts = true;
    cmesh->n_verts = cmap.mapped_NN[NT_VERTEX];
    Array<size_t> vsz(eqcs.Size());
    for(auto k:Range(eqcs.Size()))
      vsz[k] = cmap.mapped_eqc_firsti[NT_VERTEX][k+1]-cmap.mapped_eqc_firsti[NT_VERTEX][k];
    Table<vertex> ceqc_verts(vsz);
    for(auto k:Range(cmesh->n_verts))
      ceqc_verts.AsArray()[k] = k;
    cmesh->eqc_verts = move(ceqc_verts);
    cmesh->verts = Array<vertex>(cmesh->n_verts, &cmesh->eqc_verts.AsArray()[0]);
    cmesh->verts.NothingToDelete();      
    cmesh->g2l_v.SetSize(cmesh->n_verts);
    cmesh->vertex_in_eqc.SetSize(cmesh->n_verts);
    size_t cnt = 0;
    for(auto k:Range(eqcs.Size())) {
      auto row = cmesh->eqc_verts[k];
      for(auto j:Range(row.Size())) {
    	cmesh->vertex_in_eqc[cnt] = k;
    	cmesh->g2l_v[cnt++] = j;
      }
    }
    /** edges **/
    cmesh->has_edges = true;
    cmesh->n_edges = cmap.mapped_NN[NT_EDGE];
    cmesh->edges.SetSize(cmesh->n_edges);
    for(auto k:Range(cmesh->n_edges))
      cmesh->edges[k] = idedge({cmap.mapped_E[k], static_cast<size_t>(k)});
    cmesh->displs_eqc_edges = std::move(cmap.mapped_eqc_firsti[NT_EDGE]);
    cmesh->displs_pad_edges = std::move(cmap.mapped_cross_eqc_firsti[NT_EDGE]);
    cmesh->eqc_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					 &(cmesh->displs_eqc_edges[0]),
					 &(cmesh->edges[0]));
    cmesh->eqc_pad_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					     &(cmesh->displs_pad_edges[0]),
					     &(cmesh->edges[cmesh->displs_eqc_edges.Last()]));
    cmesh->BuildMeshBlocks();
    return cmesh;
  }


  shared_ptr<BaseAlgebraicMesh>
  BlockedAlgebraicMesh :: Map (GridContractMap & cmap)
  { return cmap.GETCM()->Contract(shared_ptr<BlockedAlgebraicMesh>(this, NOOP_Deleter)); }
  
  shared_ptr<BaseAlgebraicMesh>
  BlockedAlgebraicMesh :: Map (CoarseMap & cmap)
  {
    static Timer t("BlockedAlgebraicMesh :: Map (CoarseMap))");
    RegionTimer rt(t);
    
    auto cmesh = make_shared<BlockedAlgebraicMesh>();

    /** EQCHierarchy **/
    cmesh->eqc_h = this->eqc_h;
    cmesh->c_eqc_h = this->c_eqc_h;
    cmesh->eqcs.SetSize(eqcs.Size());
    cmesh->eqcs = eqcs;

    /** verts **/
    cmesh->has_verts = true;
    cmesh->n_verts = cmap.mapped_NN[NT_VERTEX];
    Array<size_t> vsz(eqcs.Size());
    for(auto k:Range(eqcs.Size()))
      vsz[k] = cmap.mapped_eqc_firsti[NT_VERTEX][k+1]-cmap.mapped_eqc_firsti[NT_VERTEX][k];
    Table<vertex> ceqc_verts(vsz);
    for(auto k:Range(cmesh->n_verts))
      ceqc_verts.AsArray()[k] = k;
    cmesh->eqc_verts = move(ceqc_verts);
    cmesh->verts = Array<vertex>(cmesh->n_verts, &cmesh->eqc_verts.AsArray()[0]);
    cmesh->verts.NothingToDelete();      
    cmesh->g2l_v.SetSize(cmesh->n_verts);
    cmesh->vertex_in_eqc.SetSize(cmesh->n_verts);
    size_t cnt = 0;
    for(auto k:Range(eqcs.Size())) {
      auto row = cmesh->eqc_verts[k];
      for(auto j:Range(row.Size())) {
    	cmesh->vertex_in_eqc[cnt] = k;
    	cmesh->g2l_v[cnt++] = j;
      }
    }
    // cmesh->vertex_in_eqc = std::move(cmap.cv_eqcs);

    /** edges **/
    cmesh->has_edges = true;
    cmesh->n_edges = cmap.mapped_NN[NT_EDGE];
    cmesh->edges.SetSize(cmesh->n_edges);
    for(auto k:Range(cmesh->n_edges))
      cmesh->edges[k] = idedge({cmap.mapped_E[k], static_cast<size_t>(k)});
    cmesh->displs_eqc_edges = std::move(cmap.disp_ie);
    cmesh->displs_pad_edges = std::move(cmap.disp_ce);
    cmesh->eqc_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					 &(cmesh->displs_eqc_edges[0]),
					 &(cmesh->edges[0]));
    cmesh->eqc_pad_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					     &(cmesh->displs_pad_edges[0]),
					     &(cmesh->edges[cmesh->displs_eqc_edges.Last()]));
    cmesh->BuildMeshBlocks();
    
    return cmesh;
  }

  
  shared_ptr<BaseAlgebraicMesh>
  BlockedAlgebraicMesh :: GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap,
					      Array<double> && awv, Array<double> && awe)
  {
    // auto cmesh = GenerateCoarseMesh(cmap);
    static Timer t("BlockedAlgebraicMesh :: GenerateCoarseMesh");
    RegionTimer rt(t);
    auto cmesh = make_shared<BlockedAlgebraicMesh>();
    {

    /** EQCHierarchy **/
    cmesh->eqc_h = this->eqc_h;
    cmesh->c_eqc_h = this->c_eqc_h;
    cmesh->eqcs.SetSize(eqcs.Size());
    cmesh->eqcs = eqcs;


    /** verts **/
    const auto & vmap = cmap->GetVMAP();
    cmesh->has_verts = true;
    cmesh->n_verts = cmap->GetNVC();
    // auto L1 = [&cmap](size_t k)->size_t {
    //   return cmap->GetCVEQC(k);
    // };
    // auto L2 = [](size_t k)->size_t{ return k; };
    // cmesh->eqc_verts = eqc_h->PartitionData<size_t, decltype(L1), decltype(L2)>
    //   (cmesh->NV(), L1, L2);
    // cmesh->verts = Array<size_t>(cmesh->n_verts, &(cmesh->eqc_verts[0][0]));
    // cmesh->verts.NothingToDelete();
    // cmesh->g2l_v.SetSize(cmesh->n_verts);
    // cmesh->vertex_in_eqc.SetSize(cmesh->n_verts);
    // for(auto k:Range(cmesh->eqc_verts.Size())) {
    //   auto row = cmesh->eqc_verts[k];
    //   for(auto j:Range(row.Size())) {
    // 	cmesh->g2l_v[row[j]] = j;
    // 	cmesh->vertex_in_eqc[row[j]] = k;
    //   }
    // }

    cmesh->eqc_verts = std::move(cmap->ceqc_verts);
    cmesh->verts = Array<vertex>(cmap->NVC, &cmesh->eqc_verts.AsArray()[0]);
    cmesh->verts.NothingToDelete();      
    cmesh->g2l_v.SetSize(cmap->NVC);
    size_t cnt = 0;
    for(auto k:Range(cmesh->eqc_verts.Size())) {
      auto row = cmesh->eqc_verts[k];
      for(auto j:Range(row.Size())) {
    	cmesh->g2l_v[cnt++] = j;
      }
    }
    cmesh->vertex_in_eqc = std::move(cmap->cv_eqcs);

    /** edges **/
    const auto & emap = cmap->GetEMAP();
    const auto & cedges = cmap->GetCEDGES();
    cmesh->has_edges = true;
    cmesh->n_edges = cmap->NEC;
    cmesh->edges.SetSize(cmap->NEC);
    for(auto k:Range(cmap->NEC))
      cmesh->edges[k] = idedge({cmap->coarse_edges[k], k});
    cmesh->displs_eqc_edges = std::move(cmap->disp_ie);
    cmesh->displs_pad_edges = std::move(cmap->disp_ce);
    cmesh->eqc_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					 &(cmesh->displs_eqc_edges[0]),
					 &(cmesh->edges[0]));
    cmesh->eqc_pad_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					     &(cmesh->displs_pad_edges[0]),
					     &(cmesh->edges[cmesh->displs_eqc_edges.Last()]));
    
    // cmesh->BuildEdgeTables(std::move(cedges), std::move(awe)); 
    }
    
    cmesh->wv = std::move(awv);
    cmesh->CumulateVertexData(cmesh->wv);
    cmesh->we = std::move(awe);
    cmesh->CumulateEdgeData(cmesh->we);
    cmesh->BuildMeshBlocks();
    return cmesh;
  }
  
  shared_ptr<BaseAlgebraicMesh>
  BlockedAlgebraicMesh :: GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap)
  {

    static Timer t("BlockedAlgebraicMesh :: GenerateCoarseMesh");
    RegionTimer rt(t);
    
    auto cmesh = make_shared<BlockedAlgebraicMesh>();

    /** EQCHierarchy **/
    cmesh->eqc_h = this->eqc_h;
    cmesh->c_eqc_h = this->c_eqc_h;
    cmesh->eqcs.SetSize(eqcs.Size());
    cmesh->eqcs = eqcs;


    /** verts **/
    const auto & vmap = cmap->GetVMAP();
    cmesh->has_verts = true;
    cmesh->n_verts = cmap->GetNVC();
    // auto L1 = [&cmap](size_t k)->size_t {
    //   return cmap->GetCVEQC(k);
    // };
    // auto L2 = [](size_t k)->size_t{ return k; };
    // cmesh->eqc_verts = eqc_h->PartitionData<size_t, decltype(L1), decltype(L2)>
    //   (cmesh->NV(), L1, L2);
    // cmesh->verts = Array<size_t>(cmesh->n_verts, &(cmesh->eqc_verts[0][0]));
    // cmesh->verts.NothingToDelete();
    // cmesh->g2l_v.SetSize(cmesh->n_verts);
    // cmesh->vertex_in_eqc.SetSize(cmesh->n_verts);
    // for(auto k:Range(cmesh->eqc_verts.Size())) {
    //   auto row = cmesh->eqc_verts[k];
    //   for(auto j:Range(row.Size())) {
    // 	cmesh->g2l_v[row[j]] = j;
    // 	cmesh->vertex_in_eqc[row[j]] = k;
    //   }
    // }

    cmesh->eqc_verts = std::move(cmap->ceqc_verts);
    cmesh->verts = Array<vertex>(cmap->NVC, &cmesh->eqc_verts.AsArray()[0]);
    cmesh->verts.NothingToDelete();      
    cmesh->g2l_v.SetSize(cmap->NVC);
    size_t cnt = 0;
    for(auto k:Range(cmesh->eqc_verts.Size())) {
      auto row = cmesh->eqc_verts[k];
      for(auto j:Range(row.Size())) {
    	cmesh->g2l_v[cnt++] = j;
      }
    }
    cmesh->vertex_in_eqc = std::move(cmap->cv_eqcs);

    /** edges **/
    const auto & emap = cmap->GetEMAP();
    const auto & cedges = cmap->GetCEDGES();
    cmesh->has_edges = true;
    cmesh->n_edges = cmap->NEC;
    cmesh->edges.SetSize(cmap->NEC);
    for(auto k:Range(cmap->NEC))
      cmesh->edges[k] = idedge({cmap->coarse_edges[k], k});
    cmesh->displs_eqc_edges = std::move(cmap->disp_ie);
    cmesh->displs_pad_edges = std::move(cmap->disp_ce);
    cmesh->eqc_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					 &(cmesh->displs_eqc_edges[0]),
					 &(cmesh->edges[0]));
    cmesh->eqc_pad_edges = FlatTable<idedge>(eqc_h->GetNEQCS(),
					     &(cmesh->displs_pad_edges[0]),
					     &(cmesh->edges[cmesh->displs_eqc_edges.Last()]));
    
    // cmesh->BuildEdgeTables(std::move(cedges), std::move(awe)); 

    cmesh->BuildMeshBlocks();

    return cmesh;
  } // end BlockedAlgebraicMesh :: GenerateCoarseMesh


  /** End BlockedAlgebraicMesh**/


  
  /** FlatAlgebraicMesh **/

  
  FlatAlgebraicMesh :: FlatAlgebraicMesh
  (FlatArray<vertex> av, FlatArray<idedge> ae, FlatArray<face> af,
   FlatArray<double> awv, FlatArray<double> awe, FlatArray<double> awf,
   FlatArray<double> acwv, FlatArray<double> acwe, FlatArray<double> acwf,
   FlatArray<idedge> ce)
    : BaseAlgebraicMesh(av.Size(), ae.Size(), af.Size())
  {
    if(n_verts) {
      verts = Array<vertex>(av.Size(), &(av[0]));
      verts.NothingToDelete();
    }
    if(n_edges) {
      edges = Array<idedge>(ae.Size(), &(ae[0]));
      edges.NothingToDelete();
    }
    if(n_faces) {
      faces = Array<face>(af.Size(), &(af[0]));
      faces.NothingToDelete();
    }
    n_cross_edges = ce.Size();
    if(n_cross_edges) {
      cross_edges = Array<idedge>(ce.Size(), &(ce[0]));
      cross_edges.NothingToDelete();
    }
    
    // using global wts!
    cwv = Array<double>(acwv.Size(), &(acwv[0]));
    wv = Array<double>(awv.Size(), &(awv[0]));
    cwe = Array<double>(acwe.Size(), &(acwe[0]));
    we = Array<double>(awe.Size(), &(awe[0]));
    cwf = Array<double>(acwf.Size(), &(acwf[0]));
    wf = Array<double>(awf.Size(), &(awf[0]));
    
    return;
  }

#endif

} // end namespace amg
