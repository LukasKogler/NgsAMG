#define FILE_AMGCTR_CPP

#include "amg.hpp"

#include <metis.h>
typedef idx_t idxtype;   

namespace amg
{

  Table<int> PartitionProcsMETIS (BlockTM & mesh, int nparts)
  {
    static Timer t("PartitionProcsMETIS"); RegionTimer rt(t);
    const auto & eqc_h(*mesh.GetEQCHierarchy());
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();
    Table<int> groups;
    if (nparts==1) {
      Array<int> perow(1); perow[0] = comm.Size();
      groups = Table<int>(perow);
      for (auto k : Range(comm.Size())) groups[0][k] = k;
      return move(groups);
    }
    int root = 0;
    Array<size_t> all_nvs ( (comm.Rank()==root) ? comm.Size() : 0);
    size_t nv_loc = mesh.GetNN<NT_VERTEX>();
    MyMPI_Gather(nv_loc, all_nvs, comm, root);
    // per dp: dist-PROC, NV_SHARED,NE that would become loc (second not used ATM)
    Array<INT<3,size_t>> data (eqc_h.GetDistantProcs().Size()); data = 0;
    auto ex_procs = eqc_h.GetDistantProcs();
    for (auto eqc : Range(neqcs)) {
      auto dps = eqc_h.GetDistantProcs(eqc);
      if (dps.Size()==1) {
	auto pos = ex_procs.Pos(dps[0]);
	data[pos][0] = dps[0];
	data[pos][1] = mesh.GetENN<NT_VERTEX>(eqc);
      }
    }
    if (neqcs>0) {
      // these edges definitely become local through contracting
      auto pad_edges = mesh.GetCNodes<NT_EDGE>(0);
      for (const auto & edge : pad_edges) {
	AMG_Node<NT_VERTEX> vmax = max(edge.v[0], edge.v[1]);
	auto eq = mesh.GetEqcOfNode<NT_VERTEX>(vmax);
	if (eqc_h.GetDistantProcs(eq).Size()!=1) cout << "try eq " << eq << " not s 1!!" << endl;
	auto dp = eqc_h.GetDistantProcs(eq)[0];
	auto pos = ex_procs.Pos(dp);
	data[pos][2]++;
      }
    }
    cout << "send data to " << root << endl;
    prow2(data); cout << endl;
    if (comm.Rank() != root) {
      /** Send  data to root **/
      comm.Send(data, root, MPI_TAG_AMG);
    }
    if (comm.Rank() == root) {
      /** Recv data from all ranks **/
      Array<Array<INT<3,size_t>>> gdata(comm.Size());
      gdata[root] = move(data);
      for (auto k : Range(comm.Size())) {
	if (k!=root) comm.Recv(gdata[k], k, MPI_TAG_AMG);
      }
      // generate metis graph structure
      Array<idx_t> partition (comm.Size()); partition = -1;
      Array<idx_t> v_weights(comm.Size());
      for (auto k : Range(comm.Size()))
	v_weights[k] = all_nvs[k];
      Array<idx_t> edge_firsti(comm.Size()+1); edge_firsti = 0;
      for (auto k : Range(comm.Size()))
	edge_firsti[k+1] = edge_firsti[k] + gdata[k].Size();
      Array<idx_t> edge_idx(edge_firsti.Last());
      Array<idx_t> edge_wt(edge_firsti.Last());
      int c = 0;
      for (auto k : Range(comm.Size())) {
	auto & row = gdata[k];
	for (auto j : Range(row.Size())) {
	  edge_idx[c] = row[j][0]-1;
	  edge_wt[c] = row[j][1];
	  c++;
	}
      }
      // cout << "v_weights: " << endl; prow2(v_weights); cout << endl;
      // cout << "edge_firsti: " << endl; prow2(edge_firsti); cout << endl;
      // cout << "edge_idx: " << endl; prow(edge_idx); cout << endl;
      // cout << "edge_wt: " << endl; prow2(edge_wt); cout << endl;
      idx_t nvts = idx_t(comm.Size());      // nr of vertices
      idx_t ncon = 1;                       // nr of balancing constraints
      idx_t* xadj = &(edge_firsti[0]);      // edge-firstis
      idx_t* adjncy = &(edge_idx[0]);       // edge-connectivity
      idx_t* vwgt = &(v_weights[0]);        // "computation cost"
      idx_t* vsize = NULL;                  // "comm. cost"
      idx_t* adjwgts = &(edge_wt[0]);       // edge-weights
      idx_t  m_nparts = nparts;             // nr of parts
      real_t* tpwgts = NULL;                // weights for each part (equal if NULL)
      real_t* ubvec = NULL;                 // tolerance
      idx_t metis_options[METIS_NOPTIONS];  // metis-options
      idx_t objval;                         // value of the edgecut/totalv of the partition
      idx_t * part = &partition[0];         // where to write the partition
      METIS_SetDefaultOptions(metis_options);
      metis_options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;         // minimize communication volume
      metis_options[METIS_OPTION_NCUTS] = (comm.Size()>1000) ? 1 : 2;  // metis will generate this many partitions and return the best
      {
	static Timer t("METIS_PartGraphKway"); RegionTimer rt(t);
	int metis_code = METIS_PartGraphKway (&nvts, &ncon, xadj, adjncy, vwgt, vsize, adjwgts, &m_nparts, tpwgts,
					      ubvec, metis_options, &objval, part);
	if (metis_code != METIS_OK) {
	  switch(metis_code) {
	  case(METIS_ERROR_INPUT) : { cout << "METIS_ERROR_INPUT" << endl; break; }
	  case(METIS_ERROR_MEMORY) : { cout << "METIS_ERROR_MEMORY" << endl; break; }
	  case(METIS_ERROR) : { cout << "METIS_ERROR" << endl; break; }
	  default: { cout << "unknown metis return??" << endl; break; }
	  }
	  throw Exception("METIS did not succeed!!");
	}
      }
      // sort partition by min, rank it has
      TableCreator<int> cgs(nparts);
      Array<int> arra(nparts); arra = comm.Size();
      for (auto k : Range(comm.Size())) arra[partition[k]] = min2(arra[partition[k]], k);
      Array<int> arrb(nparts); for (auto k : Range(nparts)) arrb[k] = k;
      QuickSortI(arra, arrb); for (auto k : Range(nparts)) arra[arrb[k]] = k;
      for (; !cgs.Done(); cgs++) {
	for (auto p : Range(comm.Size())) {
	  cgs.Add(arra[partition[p]],p);
	}
      }
      groups = cgs.MoveTable();
    }
    comm.Bcast(groups, root);
    cout << "groups: " << endl << groups << endl;
    return move(groups);
  }

  INLINE Timer & timer_hack_gcmc () { static Timer t("GridContractMap constructor"); return t; }
  template<class TMESH> GridContractMap<TMESH> :: GridContractMap (Table<int> && _groups, shared_ptr<TMESH> _mesh)
    : GridMapStep<TMESH>(_mesh), eqc_h(_mesh->GetEQCHierarchy()), groups(_groups), node_maps(4), annoy_nodes(4)
  {
    RegionTimer rt(timer_hack_gcmc());

    cout << "groups : " << endl << groups << endl;

    BuildCEQCH();

    cout << "c_eqc_h: ";
    if (c_eqc_h!=nullptr) cout << endl << *c_eqc_h << endl;
    else cout << "NULLPTR" << endl;
	  
    cout << "BuildNodeMaps " << endl;
    eqc_h->GetCommunicator().Barrier();
    cout << "BuildNodeMaps " << endl;

    BuildNodeMaps();

    cout << "BuildNodeMaps done " << endl;
    eqc_h->GetCommunicator().Barrier();
    cout << "BuildNodeMaps done " << endl;

    throw Exception("GridContractMap not yet usable!!!");
  } // GridContractMap (..)

  /** 
  	There are annoying edges: 
  	   if edge is AC -- CB (with A\cut B=empty)
  	   then by now only contr(C) know about the edge,
  	   but it has to be in [contr(A)\cut contr(B)]\union contr(C)
  	Overall, there are 4 types of edges:
  	  - II-edges: go from in in meq to in in ceq
  	  - CI-edges: go from cross in meq to in in cut(coarse)
  	  - CC-edges: go from cross in meq to cross in ceq
  	  - annoy-edges: !!!!these can be IN or CROSS!!!!
  	NOTE: CC cannot go to another ceq!!
  	We do this:
  	- make table from annoying edges
  	- ReduceTable with annoying ones (all cross per definition)
  	
  	- now we have all edges we need and we can make contracted egde tables
  	  and edge maps:
  	  ordering of I-edges within each ceqc is:
  	     CEQ: [ IImeq1, IImeq2, IImeq3, ... ; CImeq1, CImeq2, ... ; ANNOY ]
                 (only meqs that map to ceq)  |  (possibly all meqs)
  	  ordering of C-edges within each ceq is:
  	     CEQ: [ CCmeq1, CCmeq2, ... ; ANNOY]
  	  NOTE: Map of any annoying edges has to be done by Pos(.)
  	        for the contr. annoy-edges array.
  		Rest of map is deterministic
  **/
  INLINE Timer & timer_hack_nmap () { static Timer t("GridContractMap :: BuildNodeMaps"); return t; }
  template<class TMESH> void GridContractMap<TMESH> :: BuildNodeMaps ()
  {
    RegionTimer rt(timer_hack_nmap());

    const auto & f_eqc_h(*this->eqc_h);
    auto comm = f_eqc_h.GetCommunicator();

    cout << "fine eqch: " << endl << f_eqc_h << endl;
    
    if (!is_gm) {
      shared_ptr<BlockTM> btm = this->mesh;
      cout << "send mesh to " << my_group[0] << endl;
      comm.Send(btm, my_group[0], MPI_TAG_AMG);
      cout << "send mesh done" << endl;
      return;
    }

    const auto & c_eqc_h(*this->c_eqc_h);
    cout << "coarse fine eqch: " << endl << c_eqc_h << endl;
    
    const TMESH & fmesh(*this->mesh);
    auto p_c_mesh = make_shared<BlockTM>(this->c_eqc_h);
    auto & c_mesh(*p_c_mesh);

    int mgs = my_group.Size();
    Array<shared_ptr<BlockTM>> mg_btms(mgs);
    mg_btms[0] = this->mesh;
    for (int k = 1; k < my_group.Size(); k++) {
      cout << "get mesh from " << my_group[k] << endl;
      comm.Recv(mg_btms[k], my_group[k], MPI_TAG_AMG);
      cout << "got mesh from " << my_group[k] << endl;
      cout << *mg_btms[k] << endl;
    }

    // constexpr int lhs = 1024*1024;
    // LocalHeap lh_max (lhs, "Max");
    // auto cut = [&lh_max](const auto & a, const auto & b) {
    //   Array<typename std::remove_reference<decltype(a[0])>::type > out(min2(a.Size(), b.Size()),lh_max);
    //   out.SetSize0();
    //   size_t ca = 0, cb = 0;
    //   while ( (ca<a.Size()) && (cb<b.Size()) )
    // 	if (a[ca]<b[cb]) ca++;
    // 	else if (a[ca]>b[cb]) cb++;
    // 	else { out.Append(a[ca]); ca++; cb++; }
    //   return std::move(out);
    // };
    auto cut_min = [](const auto & a, const auto & b) {
      size_t ca = 0, cb = 0;
      while ( (ca<a.Size()) && (cb<b.Size()) )
    	if (a[ca]<b[cb]) ca++;
    	else if (a[ca]>b[cb]) cb++;
    	else { return a[ca]; }
      return -1;
    };

    size_t mneqcs = map_mc.Size();
    size_t cneqcs = c_eqc_h.GetNEQCS();

    /** for when we need a unique proc to take an eqc from!!  **/
    Array<int> eqc_sender(mneqcs);
    for (auto k : Range(mneqcs)) {
      auto mems = mmems[k];
      // eqc_sender[k] = cut(mems, my_group)[0];
      eqc_sender[k] = cut_min(mems, my_group);
    }    

    cout << "eqc_sender: " << endl; prow2(eqc_sender); cout << endl;
    
    /** vertices **/
    auto & v_dsp = c_mesh.disp_eqc[NT_VERTEX];
    v_dsp.SetSize(cneqcs+1); v_dsp = 0;
    Array<size_t> firsti_v(mneqcs);
    firsti_v = 0;
    for (auto k : Range(my_group.Size())) {
      for (auto j : Range(mg_btms[k]->GetNEqcs())) {
	auto eqc_vs = mg_btms[k]->GetENodes<NT_VERTEX>(j);
	if (my_group[k]==eqc_sender[map_om[k][j]]) {
	  v_dsp[map_oc[k][j]+1] += eqc_vs.Size();
	  firsti_v[map_om[k][j]] = eqc_vs.Size();
	}
      }
    }

    for (auto k : Range(cneqcs)) {
      c_mesh.nnodes_eqc[NT_VERTEX][k] = v_dsp[k+1];
      c_mesh.nnodes_cross[NT_VERTEX][k] = 0;
      v_dsp[k+1] += v_dsp[k];
    }
    size_t cnv = v_dsp.Last();
    c_mesh.nnodes[NT_VERTEX] = cnv;
    c_mesh.verts.SetSize(cnv);
    for (auto k : Range(cnv)) c_mesh.verts[k] = k;
    c_mesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (cneqcs, &(v_dsp[0]), &(c_mesh.verts[0]));
    cout << "v_dsp: " << endl; prow2(v_dsp); cout << endl;
    cout << "c_mesh.eqc_verts: " << endl << c_mesh.eqc_verts << endl;
    auto & ceqc_verts(c_mesh.eqc_verts);
    
    Array<size_t> sz(cneqcs); sz = 0;
    for (auto meq : Range(mneqcs)) {
      auto ceq = map_mc[meq];
      auto bs = firsti_v[meq];
      firsti_v[meq] = ceqc_verts.IndexArray()[ceq] + sz[ceq];
      sz[ceq] += bs;
    }

    sz.SetSize(my_group.Size());
    for (auto k : Range(my_group.Size())) {
      sz[k] = 0; for (auto row:Range(mg_btms[k]->GetNEqcs())) sz[k] += mg_btms[k]->GetENodes<NT_VERTEX>(row).Size();
    }
    node_maps[NT_VERTEX] = Table<size_t>(sz);
    auto & vmaps = node_maps[NT_VERTEX];
    vmaps.AsArray() = -1;
    for (auto k : Range(my_group.Size())) {
      for (auto eqc : Range(mg_btms[k]->GetNEqcs())) {
	auto eqc_vs = mg_btms[k]->GetENodes<NT_VERTEX>(eqc);
	for (auto l : Range(eqc_vs.Size())) {
	  vmaps[k][eqc_vs[l]] = firsti_v[map_om[k][eqc]]+l;
	}
      }
    }
    cout << "vmaps: " << endl << vmaps << endl;

    auto & c_eqc_dps = c_eqc_h.GetDPTable();
    Array<size_t> annoy_have(mneqcs);
    Array<size_t> ci_have(mneqcs);
    Table<size_t> ci_pos(mneqcs+1, cneqcs); // (i,j) nr of C in meq that become I in ceq
    Array<size_t> ci_get(cneqcs);
    Array<size_t> annoy_havec(cneqcs);
    // eq0, v0, eq1, v1
    Table<INT<4,int>> tannoy_edges;
    auto eq_of_v = [&c_mesh](auto v) { return c_mesh.GetEqcOfNode<NT_VERTEX>(v); };
    auto map_cv_to_ceqc = [&c_mesh](auto v) { return c_mesh.MapNodeToEQC<NT_VERTEX>(v); };
    {
      TableCreator<INT<4,int>> ct(cneqcs);
      while(!ct.Done()) {
	annoy_have = 0; ci_get = 0; ci_have = 0; annoy_havec = 0;
	if(cneqcs) ci_pos.AsArray() = 0;
	for(auto k:Range(my_group.Size())) {
	  auto eqmap = map_om[k];
	  auto neq = eqmap.Size();
	  for(auto eq:Range(neq)) {
	    auto meq = map_om[k][eq];
	    auto ceq = map_oc[k][eq];
	    if(my_group[k]!=eqc_sender[map_om[k][eq]]) continue;
	    auto es = mg_btms[k]->GetCNodes<NT_EDGE>(eq);
	    for(auto l:Range(es.Size())) {
	      const auto& v = es[l].v;
	      auto cv1 = vmaps[k][v[0]];
	      auto cv2 = vmaps[k][v[1]];
	      if(cv1>cv2) swap(cv1, cv2);
	      auto ceq1 = eq_of_v(cv1);
	      auto ceq2 = eq_of_v(cv2);
	      if( (ceq1==ceq2) && (ceq1==ceq) ) { // CI edge
		ci_pos[meq][ceq1]++;
		ci_get[ceq1]++;
		ci_have[meq]++;
		continue;
	      }
	      auto cutid = c_eqc_h.GetCommonEQC(ceq1, ceq2);
	      if(ceq==cutid) continue; // CC edge
	      auto ceq1_id = c_eqc_h.GetEQCID(ceq1);
	      auto ceq2_id = c_eqc_h.GetEQCID(ceq2);
	      auto cdps = c_eqc_h.GetDistantProcs(cutid);
	      // master of coarse(C) adds the edge
	      if(c_eqc_h.IsMasterOfEQC(ceq)) {
		INT<4,int> ce = {ceq1_id, map_cv_to_ceqc(cv1), ceq2_id, map_cv_to_ceqc(cv2)};
		ct.Add(cutid, ce);
	      }
	      annoy_have[meq]++;
	      annoy_havec[cutid]++;
	    }
	  }
	}
	ct++;
      }
      tannoy_edges = ct.MoveTable();
    }

    cout << "tannoy_edges: " << endl << tannoy_edges << endl;
    auto annoy_edges = ReduceTable<INT<4,int>, INT<4,int>>
      (tannoy_edges, this->c_eqc_h, [](const auto & in) {
	Array<INT<4,int>> out;
	if (in.Size() == 0) return out;
	int ts = 0; for (auto k : Range(in.Size())) ts += in[k].Size();
	if (ts == 0) return out;
	cout << "got in: " << endl; print_ft(cout, in); cout << endl;
	out.SetSize(ts); ts = 0;
	for (auto k : Range(in.Size()))
	  { auto row = in[k]; for (auto j : Range(row.Size())) out[ts++] = row[j]; }
	cout << "out to sort: " << endl; prow2(out); cout << endl;
	QuickSort(out, [](const auto & a, const auto & b) {
	    const bool isin[2] = {a[0]==a[2], b[0]==b[2]};
	    if (isin[0] && !isin[1]) return true;
	    if (isin[1] && !isin[0]) return false;
	    for (int l : {0,2,1,3})
	      { if (a[l]<b[l]) return true; if(b[l]<a[l]) return false; }
	    return false;
	  });
	cout << "sorted out: " << endl; prow2(out); cout << endl;
	return out;
      });

    cout << "reduced annoy_edges: " << endl << annoy_edges << endl;


  }

  INLINE Timer & timer_hack_beq () { static Timer t("GridContractMap :: BuildCEQCH"); return t; }
  template<class TMESH> void GridContractMap<TMESH> :: BuildCEQCH ()
  {
    RegionTimer rt(timer_hack_beq());

    const auto & eqc_h(*this->eqc_h);
    auto comm = eqc_h.GetCommunicator();

    this->proc_map.SetSize(comm.Size());
    auto n_groups = groups.Size();
    for (auto grp_nr : Range(n_groups)) {
      auto row = groups[grp_nr];
      for (auto j : Range(row.Size())) {
	proc_map[row[j]] = grp_nr;
      }
    }
    this->my_group.Assign(groups[proc_map[comm.Rank()]]);
    this->is_gm = my_group[0] == comm.Rank();

    cout << "my_group: "; prow2(my_group); cout << endl;
    cout << "is_gm ? " << is_gm << endl;
    
    if (!is_gm) {
      /** Send DP-tables to master and return **/
      int master = my_group[0];
      comm.Send(eqc_h.GetDPTable(), master, MPI_TAG_AMG);
      comm.Send(eqc_h.GetEqcIds(), master, MPI_TAG_AMG);
      return;
    }
    
    /** New MPI-Comm **/
    netgen::Array<int> cmembs(groups.Size()); // haha, this has to be a netgen-array
    for (auto k : Range(groups.Size())) cmembs[k] = groups[k][0];
    NgsAMG_Comm c_comm(netgen::MyMPI_SubCommunicator(comm, cmembs), true);

    /** gather eqc-tables **/
    auto & reft = eqc_h.GetDPTable();
    Array<int> sz;
    if (reft.Size()) sz.SetSize(reft.Size());
    for (auto k : Range(reft.Size()))
      sz[k] = reft[k].Size();
    Table<int> eqcs_table(sz);
    for (auto k : Range(reft.Size()))
      for (auto j : Range(sz[k]))
	eqcs_table[k][j] = reft[k][j];
    Array<Table<int>> all_dist_eqcs(my_group.Size());
    all_dist_eqcs[0] = std::move(eqcs_table);
    Array<Array<size_t>> all_eqc_ids(my_group.Size());
    all_eqc_ids[0].SetSize(eqc_h.GetNEQCS());
    for (auto j : Range(eqc_h.GetNEQCS())) all_eqc_ids[0][j] = eqc_h.GetEQCID(j);
    for (auto j : Range((size_t)1,my_group.Size())) {
      comm.Recv(all_dist_eqcs[j], my_group[j], MPI_TAG_AMG);
      all_eqc_ids[j].SetSize(all_dist_eqcs[j].Size());
      comm.Recv(all_eqc_ids[j], my_group[j], MPI_TAG_AMG);
    }

    /** merge gathered eqc tables **/
    Array<int> gids;
    gids = std::move(all_eqc_ids[0]);
    for (auto j : Range((size_t)1, my_group.Size())) {
      for (auto l : Range(all_eqc_ids[j].Size())) {
	if (!gids.Contains(all_eqc_ids[j][l]))
	  gids.Append(all_eqc_ids[j][l]);
      }
    }    
    QuickSort(gids);
    size_t mneqcs = gids.Size();
    sz.SetSize(my_group.Size());
    for (auto j : Range(my_group.Size())) {
      sz[j] = all_eqc_ids[j].Size();
    }
    map_om = Table<int> (sz);
    map_oc = Table<int> (sz);
    sz.SetSize(mneqcs);
    sz = 0;
    for (auto j : Range(my_group.Size())) {
      auto nid = all_eqc_ids[j].Size();
      for (auto l : Range(nid)) {
	map_om[j][l] = gids.Pos(all_eqc_ids[j][l]);
	sz[map_om[j][l]] = all_dist_eqcs[j][l].Size()+1;
      }
    }
    // cout << "map_om: "  << endl << map_om << endl;
    // Table<int> mdps(sz);
    mmems = Table<int>(sz);
    for (auto j : Range(my_group.Size())) {
      auto nid = all_eqc_ids[j].Size();
      for (auto l : Range(nid)) {
	for (auto i : Range(all_dist_eqcs[j][l].Size()))
	  mmems[map_om[j][l]][i] = all_dist_eqcs[j][l][i];
	mmems[map_om[j][l]].Last() = my_group[j];
      }
    }
    for (auto j : Range(mmems.Size()))
      QuickSort(mmems[j]);
    /** contract eqc table + make map **/
    auto crs_dps = [&](const auto & dps) {
      Array<int> out;
      auto cr = proc_map[comm.Rank()];
      for (auto k : Range(dps.Size())) {
	auto dcr = proc_map[dps[k]];
	if ( (dcr!=c_comm.Rank()) && (!out.Contains(dcr)) )
	  out.Append(dcr);
      }
      QuickSort(out);
      return out;
    };
    this-> map_mc.SetSize(mneqcs);
    Array<Array<int>> ceqcs;
    sz.SetSize(0);
    for (auto j : Range(mmems.Size())) {
      auto cdps = crs_dps(mmems[j]);
      bool is_new = true;
      int l;
      for (l=0; l<ceqcs.Size()&&is_new; l++)
	if (ceqcs[l]==cdps) {
	  is_new = false;
	}
      map_mc[j] = is_new?l:l-1; //incremented one extra time
      if (is_new) {
	sz.Append(cdps.Size());
	ceqcs.Append(std::move(cdps));
      }
    }    
    Table<int> ceqcs_table(sz), ceq2(sz);
    for (auto k : Range(ceqcs_table.Size())) {
      for (auto j : Range(ceqcs_table[k].Size())) {
	ceqcs_table[k][j] = ceqcs[k][j];
	ceq2[k][j] = ceqcs[k][j];
      }
    }
    this->c_eqc_h = make_shared<EQCHierarchy>(std::move(ceqcs_table), c_comm);

    // EQCHierarchy re-sorts the DP-table!!
    auto & ctab = c_eqc_h->GetDPTable(); 
    Array<int> remap(ctab.Size());
    for (auto k : Range(ctab.Size())) {
      auto set = ceq2[k];
      bool found = false;
      int l = -1;
      while (!found) {
	l++;
	if (ctab[l]==set) found=true;
      }
      remap[k] = l;
    }
    for (auto k : Range(map_mc.Size()))
      map_mc[k] = remap[map_mc[k]];
    for (auto k : Range(map_oc.Size())) {
      for (auto j : Range(map_oc[k].Size())) {
	map_oc[k][j] = map_mc[map_om[k][j]];
      }
    }
    
  }

  template class GridContractMap<H1Mesh>;
  
} // namespace amg

#undef FILE_AMGCTR_CPP
