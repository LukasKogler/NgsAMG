#include "amg.hpp"

namespace amg
{
  TopologicMesh :: TopologicMesh (size_t nv, size_t ne, size_t nf, size_t nc)
  {
    nnodes[0] = nnodes_glob[0] = nv;
    nnodes[1] = nnodes_glob[1] = ne;
    nnodes[2] = nnodes_glob[2] = nf;
    nnodes[3] = nnodes_glob[3] = nc;
  }

  TopologicMesh :: TopologicMesh (TopologicMesh && other)
    : verts(move(other.verts)), edges(move(other.edges)),
      faces(move(other.faces)), cells(move(other.cells))
  {
    for(auto l:Range(4)) {
      has_nodes[l] = other.has_nodes[l]; other.has_nodes[l] = false;
      nnodes[l] = other.nnodes[l]; other.nnodes[l] = 0;
      nnodes_glob[l] = other.nnodes_glob[l]; other.nnodes_glob[l] = 0;
    }
    econ = other.econ; other.econ = nullptr;
  }

  shared_ptr<SparseMatrix<double>> TopologicMesh :: GetEdgeCM() const
  {
    if(econ != nullptr) return econ;
    auto nv = GetNN<NT_VERTEX>();
    Array<int> econ_s(nv);
    econ_s = 0;
    for(auto & edge: GetNodes<NT_EDGE>())
      for(auto l:Range(2))
	econ_s[edge.v[l]]++;
    Table<INT<2> > econ_i(econ_s);
    econ_s = 0;
    for(auto & edge: GetNodes<NT_EDGE>())
      for(auto l:Range(2))
	econ_i[edge.v[l]][econ_s[edge.v[l]]++] = INT<2>(edge.v[1-l], edge.id);
    for(auto row:econ_i)
      QuickSort(row, [](auto & a, auto & b) { return a[0]<b[0]; });
    econ = make_shared<SparseMatrix<double>>(econ_s, nv);
    for(auto k:Range(nv)) {
      auto rinds = econ->GetRowIndices(k);
      auto rvals = econ->GetRowValues(k);
      rinds = -1;
      rvals = -1;
      for(auto j:Range(econ_i[k].Size())) {
	rinds[j] = econ_i[k][j][0];
	rvals[j] = econ_i[k][j][1]; //e-ids
      }
    }
    return econ;
  }

  FlatTM :: FlatTM(FlatArray<AMG_Node<NT_VERTEX>> av, FlatArray<AMG_Node<NT_EDGE>> ae,
		   FlatArray<AMG_Node<NT_FACE>> af  , FlatArray<AMG_Node<NT_CELL>> ac,
		   FlatArray<AMG_Node<NT_EDGE>> ace , FlatArray<AMG_Node<NT_FACE>> acf)
    : TopologicMesh ()
  {
    for (auto i:Range(4)) nnodes_glob[i] = -1;
    nnodes[NT_VERTEX] = av.Size();
    if(nnodes[NT_VERTEX]) {
      verts = Array<AMG_Node<NT_VERTEX>>(av.Size(), &(av[0]));
      verts.NothingToDelete();
    }
    nnodes[NT_EDGE] = ae.Size();
    if(nnodes[NT_EDGE]) {
      edges = Array<AMG_Node<NT_EDGE>>(ae.Size(), &(ae[0]));
      edges.NothingToDelete();
    }
    nnodes[NT_FACE] = af.Size();
    if(nnodes[NT_FACE]) {
      faces = Array<AMG_Node<NT_FACE>>(af.Size(), &(af[0]));
      faces.NothingToDelete();
    }
    nnodes[NT_CELL] = ac.Size();
    if(nnodes[NT_CELL]) {
      cells = Array<AMG_Node<NT_CELL>>(ac.Size(), &(ac[0]));
      cells.NothingToDelete();
    }
    nnodes_cross[NT_EDGE] = ace.Size();
    if(nnodes_cross[NT_EDGE]) {
      cross_edges = Array<AMG_Node<NT_EDGE>>(ace.Size(), &(ace[0]));
      cross_edges.NothingToDelete();
    }
    nnodes_cross[NT_FACE] = ace.Size();
    if(nnodes_cross[NT_FACE]) {
      cross_faces = Array<AMG_Node<NT_FACE>>(acf.Size(), &(acf[0]));
      cross_faces.NothingToDelete();
    }
  }

  BlockTM :: BlockTM (shared_ptr<EQCHierarchy> _eqc_h)
    : TopologicMesh(), eqc_h(_eqc_h), disp_eqc(4), disp_cross(4)
  {
    auto neqcs = eqc_h->GetNEQCS();
    for (auto l:Range(4)) {
      nnodes_cross[l] = Array<size_t>(neqcs); nnodes_cross[l] = 0;
      nnodes_eqc[l] = Array<size_t>(neqcs); nnodes_eqc[l] = 0;
      disp_eqc[l] = Array<size_t> (neqcs+1); disp_eqc[l] = 0;
      disp_cross[l] = Array<size_t> (neqcs+1); disp_cross[l] = 0;
    }
    eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &(disp_eqc[NT_VERTEX][0]), &verts[0]);
    eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &(disp_eqc[NT_EDGE][0]), &edges[0]);
    eqc_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &(disp_eqc[NT_FACE][0]), &faces[0]);
    // eqc_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &(disp_eqc[NT_CELL][0]), &cells[0]);
    disp_cross[NT_EDGE] = Array<size_t> (neqcs+1); disp_cross[NT_EDGE] = 0;
    cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &(disp_cross[NT_EDGE][0]), &edges[0]);
    disp_cross[NT_FACE] = Array<size_t> (neqcs+1); disp_cross[NT_FACE] = 0;
    cross_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &(disp_cross[NT_FACE][0]), &faces[0]);
  }

  BlockTM :: BlockTM ( BlockTM && other)
    : TopologicMesh (move(other))
  {
    eqc_h = other.eqc_h; other.eqc_h = nullptr;
    disp_eqc.SetSize(4);
    disp_cross.SetSize(4);
    for (auto l:Range(4)) {
      nnodes_eqc[l] = move(other.nnodes_eqc[l]);
      disp_eqc[l] = move(other.disp_eqc[l]);
      nnodes_cross[l] = move(other.nnodes_cross[l]);
      disp_cross[l] = move(other.disp_cross[l]);
    }
    auto neqcs = eqc_h->GetNEQCS();
    eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &disp_eqc[NT_VERTEX][0], &verts[0]);
    eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &disp_eqc[NT_EDGE][0], &edges[0]);
    eqc_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &disp_eqc[NT_FACE][0], &faces[0]);
    cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &disp_cross[NT_EDGE][0], &edges[disp_eqc[NT_EDGE].Last()]);
    cross_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &disp_cross[NT_FACE][0], &faces[disp_eqc[NT_FACE].Last()]);
  } // BlockTM (..)

  BlockTM :: BlockTM ()
    : TopologicMesh(), eqc_h(nullptr), disp_eqc(4), disp_cross(4)
  { ; }

  void NgsAMG_Comm :: Send (shared_ptr<BlockTM> & amesh, int dest, int tag) const
  {
    const auto & mesh(*amesh);
    const auto & eqc_h(*mesh.GetEQCHierarchy());
    auto neqcs = eqc_h.GetNEQCS();
    // has_nodes // nnodes_loc // nnodes_glob // nnodes_eqc[0..4] // nnodes_cross[0..4]
    Array<int> ar(4 * (1 + 1 + 1 + 2*neqcs));
    int c = 0;
    for (auto l:Range(4)) {
      ar[c++] = mesh.has_nodes[l];
      ar[c++] = mesh.nnodes_glob[l];
      ar[c++] = mesh.nnodes[l];
      for (auto eqc : Range(neqcs)) {
	ar[c++] = mesh.nnodes_eqc[l][eqc];
	ar[c++] = mesh.nnodes_cross[l][eqc];
      }
    }
    Send(ar, dest, tag);
    if (mesh.has_nodes[0]) {
      Send(mesh.verts, dest, tag); //TODO: dont need to do this!
      Send(mesh.disp_eqc[0], dest, tag);
    }
    if (mesh.has_nodes[1]) {
      Send(mesh.edges, dest, tag);
      Send(mesh.disp_eqc[1], dest, tag);
      Send(mesh.disp_cross[1], dest, tag);
    }
    if (mesh.has_nodes[2]) {
      Send(mesh.faces, dest, tag);
      Send(mesh.disp_eqc[2], dest, tag);
      Send(mesh.disp_cross[2], dest, tag);
    }
    if (mesh.has_nodes[3]) {
      Send(mesh.cells, dest, tag);
      Send(mesh.disp_eqc[3], dest, tag);
      Send(mesh.disp_cross[3], dest, tag);
    }
  }

  void NgsAMG_Comm :: Recv (shared_ptr<BlockTM> & amesh, int src, int tag) const
  {
    Array<int> ar; Recv(ar, src, tag);
    int neqcs = (ar.Size()/4 - 3)/2;
    amesh = make_shared<BlockTM>();
    auto & mesh(*amesh);
    int c = 0;
    // mesh.disp_eqc.SetSize(4); mesh.disp_cross.SetSize(4);
    for (auto l:Range(4)) {
      mesh.has_nodes[l] = ar[c++];
      mesh.nnodes_glob[l] = ar[c++];
      mesh.nnodes[l] = ar[c++];
      mesh.nnodes_eqc[l].SetSize(neqcs);
      mesh.nnodes_cross[l].SetSize(neqcs);
      for (auto eqc : Range(neqcs)) {
	mesh.nnodes_eqc[l][eqc] = ar[c++];
	mesh.nnodes_cross[l][eqc] = ar[c++];
      }
      if(!mesh.has_nodes[l]) {
	mesh.disp_eqc[l].SetSize(neqcs+1); mesh.disp_eqc[l] = 0;
	mesh.disp_cross[l].SetSize(neqcs+1); mesh.disp_cross[l] = 0;
      }
    }
    if (mesh.has_nodes[0]) {
      Recv(mesh.verts, src, tag);
      Recv(mesh.disp_eqc[0], src, tag);
      mesh.disp_cross[0].SetSize(neqcs+1); mesh.disp_cross[0] = 0;
    }
    mesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &mesh.disp_eqc[NT_VERTEX][0], &mesh.verts[0]);
    if (mesh.has_nodes[1]) {
      Recv(mesh.edges, src, tag);
      Recv(mesh.disp_eqc[1], src, tag);
      Recv(mesh.disp_cross[1], src, tag);
    }
    mesh.eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &mesh.disp_eqc[NT_EDGE][0], &mesh.edges[0]);
    mesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &mesh.disp_cross[NT_EDGE][0], &mesh.edges[mesh.disp_eqc[NT_EDGE].Last()]);
    if (mesh.has_nodes[2]) {
      Recv(mesh.faces, src, tag);
      Recv(mesh.disp_eqc[2], src, tag);
      Recv(mesh.disp_cross[2], src, tag);
    }
    mesh.eqc_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &mesh.disp_eqc[NT_FACE][0], &mesh.faces[0]);
    mesh.cross_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &mesh.disp_cross[NT_FACE][0], &mesh.faces[mesh.disp_eqc[NT_FACE].Last()]);
    if (mesh.has_nodes[3]) {
      Recv(mesh.cells, src, tag);
      Recv(mesh.disp_eqc[3], src, tag);
      Recv(mesh.disp_cross[3], src, tag);
    }
    // mesh.eqc_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &mesh.disp_eqc[NT_CELL][0], &mesh.cells[0]);
    // mesh.cross_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &mesh.disp_cross[NT_CELL][0], &mesh.cells[mesh.disp_eqc[NT_CELL].Last()]);
  }


  const FlatTM BlockTM :: GetBlock (size_t eqc_num) const
  {
    return FlatTM(eqc_verts[eqc_num], eqc_edges[eqc_num], eqc_faces[eqc_num],
		  FlatArray<AMG_Node<NT_CELL>>(0,nullptr) /*eqc_cells[eqc_num]*/, 
		  cross_edges[eqc_num], cross_faces[eqc_num]);
  }

  BlockTM* BlockTM :: MapBTM (const BaseCoarseMap & cmap) const
  {
    static Timer t("MapBTM"); RegionTimer rt(t);
    // cout << "map mesh " << *this << endl;
    auto coarse_mesh = new BlockTM(this->eqc_h);
    const auto & eqc_h = *this->eqc_h;
    auto comm = eqc_h.GetCommunicator();
    auto neqcs = eqc_h.GetNEQCS();
    auto & cmesh = *coarse_mesh;
    for (auto l:Range(4)) cmesh.has_nodes[l] = has_nodes[l];
    // vertices
    cmesh.nnodes[NT_VERTEX] = cmap.GetMappedNN<NT_VERTEX>();
    cmesh.verts.SetSize(cmesh.nnodes[NT_VERTEX]);
    auto & cverts = cmesh.verts;
    for (auto k : Range(cmesh.nnodes[NT_VERTEX]) ) cverts[k] = k;
    // cout << "cmesh NV: " << cmesh.nnodes[NT_VERTEX] << endl;
    // cout << "cmesh verts: "; prow(cmesh.verts); cout << endl;
    // vertex tables
    auto & disp_veq = cmesh.disp_eqc[NT_VERTEX];
    disp_veq.SetSize(neqcs+1); disp_veq = cmap.GetMappedEqcFirsti<NT_VERTEX>();
    // cout << "cmesh v-disp: " << endl; prow(disp_veq); cout << endl;
    cmesh.nnodes_glob[NT_VERTEX] = 0;
    cmesh.nnodes_eqc[NT_VERTEX].SetSize(neqcs);
    for (auto eqc : Range(neqcs)) {
      auto nn = disp_veq[eqc+1] - disp_veq[eqc];
      if (eqc_h.IsMasterOfEQC(eqc)) cmesh.nnodes_glob[NT_VERTEX] += nn;
      cmesh.nnodes_eqc[NT_VERTEX][eqc] = nn;
    }
    cmesh.nnodes_glob[NT_VERTEX] = comm.AllReduce(cmesh.nnodes_glob[NT_VERTEX], MPI_SUM);
    // cout << "cmesh glob NV: " << cmesh.nnodes_glob[NT_VERTEX] << endl;
    cmesh.nnodes_cross[NT_VERTEX].SetSize(0); cmesh.nnodes_cross[NT_VERTEX] = 0;
    cmesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &cmesh.disp_eqc[NT_VERTEX][0], &cmesh.verts[0]);
    // edges
    cmesh.nnodes[NT_EDGE] = cmap.GetMappedNN<NT_EDGE>();
    auto & cedges = cmesh.edges;
    auto mapped_etup = cmap.GetMappedEdges();
    // cout << "mehs cmesh NE: " << nnodes[NT_EDGE] << " " << cmesh.nnodes[NT_EDGE] << endl;
    // cout << "ets" << mapped_etup.Size() << endl;
    // cout << "mapped_etup: " << endl; prow2(mapped_etup); cout << endl;
    cedges.SetSize(cmesh.nnodes[NT_EDGE]);
    for (auto k : Range(cmesh.nnodes[NT_EDGE]) ) { auto & e = cedges[k]; e.v = mapped_etup[k]; e.id = k; }
    // edge table - eqc
    auto & disp_eeq = cmesh.disp_eqc[NT_EDGE];
    disp_eeq.SetSize(neqcs+1); disp_eeq = cmap.GetMappedEqcFirsti<NT_EDGE>();
    cmesh.nnodes_glob[NT_EDGE] = 0;
    cmesh.nnodes_eqc[NT_EDGE].SetSize(neqcs);
    for (auto eqc : Range(neqcs)) {
      auto nn = disp_eeq[eqc+1] - disp_eeq[eqc];
      if (eqc_h.IsMasterOfEQC(eqc)) cmesh.nnodes_glob[NT_EDGE] += nn;
      cmesh.nnodes_eqc[NT_EDGE][eqc] = nn;
    }
    cmesh.eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &cmesh.disp_eqc[NT_EDGE][0], &cmesh.edges[0]);
    // edge table - cross
    auto & disp_ceq = cmesh.disp_cross[NT_EDGE];
    disp_ceq.SetSize(neqcs+1); disp_ceq = cmap.GetMappedCrossFirsti<NT_EDGE>();
    cmesh.nnodes_cross[NT_EDGE].SetSize(neqcs);
    for (auto eqc : Range(neqcs)) {
      auto nn = disp_ceq[eqc+1] - disp_ceq[eqc];
      if (eqc_h.IsMasterOfEQC(eqc)) cmesh.nnodes_glob[NT_EDGE] += nn;
      cmesh.nnodes_cross[NT_EDGE][eqc] = nn;
    }
    cmesh.nnodes_glob[NT_EDGE] = comm.AllReduce(cmesh.nnodes_glob[NT_EDGE], MPI_SUM);
    cmesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &cmesh.disp_cross[NT_EDGE][0], &cmesh.edges[cmesh.disp_eqc[NT_EDGE].Last()]);
    // faces/cells (dummy)
    cmesh.nnodes[NT_FACE] = cmap.GetMappedNN<NT_FACE>();
    cmesh.nnodes[NT_CELL] = cmap.GetMappedNN<NT_CELL>();
    return coarse_mesh;
  }

  namespace amg_nts
  {
    template<ngfem::NODE_TYPE NT>
    INLINE decltype(AMG_Node<NT>::v) MAToAMG_Node (const MeshAccess & ma, size_t node_num) {
      constexpr int NODE_SIZE = sizeof(AMG_Node<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
      INT<NODE_SIZE, AMG_Node<NT_VERTEX>> vs;
      auto ng_node = ma.GetNode<NT>(node_num);
      for (auto i : Range(NODE_SIZE)) vs[i] = ng_node.vertices[i];
      return move(vs);
    }
  } // namespace amg_nts

  shared_ptr<BlockTM> MeshAccessToBTM (shared_ptr<MeshAccess> ma, shared_ptr<EQCHierarchy> eqc_h,
				       FlatArray<int> vert_sort,
				       bool build_edges, FlatArray<int> edge_sort,
				       bool build_faces, FlatArray<int> face_sort,
				       bool build_cells, FlatArray<int> cell_sort)
  {
    static Timer t("MeshAccessToBTM"); RegionTimer rt(t);
    if(build_cells)
      throw Exception("Cant build cells yet, sorry!!");
    if(build_faces && (eqc_h->GetCommunicator().Rank()==0))
      cout << "Faces probably don't work consistently yet!!" << endl;
    auto mesh = make_shared<BlockTM>(eqc_h);
    mesh->SetVs (ma->GetNV(), [&](auto vnr)->FlatArray<int>{ return ma->GetDistantProcs(NodeId(NT_VERTEX, vnr)); },
		 [vert_sort](auto i, auto j){ vert_sort[i] = j; });
    const MeshAccess & mar = *ma;
    if (build_edges) {
      size_t num_edges_total = ma->GetNEdges();
      BitArray fine_edge(num_edges_total); fine_edge.Clear();
      int ne = ma->GetNE();
      for (auto i : Range(ne)) {
	ElementId ei(VOL, i);
	auto eledges = ma->GetElEdges (ei);		
	for(size_t j=0;j<eledges.Size();j++) fine_edge.Set(eledges[j]);
      }
      // do I need sth. like that??
      // ma->AllReduceNodalData (NT_EDGE, fine_edge, MPI_LOR);
      // size_t num_edges = ma->GetNEdges();
      // cout << "fine edges: " << fine_edge.NumSet() << " of " << fine_edge.Size() << endl << fine_edge << endl;
      size_t num_edges = fine_edge.NumSet(), cnt = 0;
      mesh->SetNodes<NT_EDGE>( num_edges,
			       [&](auto node_num) {
				 while ( (cnt < num_edges_total) && (!fine_edge.Test(cnt)) ) cnt++;
				 if (cnt == num_edges_total) // we are starting over, reset cnt
				   { cnt = 0; while ( (cnt < num_edges_total) && (!fine_edge.Test(cnt)) ) cnt++; }
				 // cout << "get " << node_num << " which is " << cnt << " of " << num_edges_total << flush;
				 auto node = amg_nts::MAToAMG_Node<NT_EDGE>(mar, cnt++);
				 // cout << " -> have " << endl;
				 constexpr int NODE_SIZE = sizeof(AMG_Node<NT_EDGE>::v)/sizeof(AMG_Node<NT_VERTEX>);
				 for (int k = 0; k < NODE_SIZE; k++) node[k] = vert_sort[node[k]];
				 if (node[0] > node[1]) Swap(node[1], node[0]);
				 return move(node);},
			       [edge_sort](auto i, auto j) { edge_sort[i] = j; } );
    }
    if (build_faces) {
      throw Exception("face-nodes only work with non-refined mesh, remove this if that is the case!!");
      mesh->SetNodes<NT_FACE>( ma->GetNFaces(),
			       [mar, vert_sort](auto node_num) {
				 auto node = amg_nts::MAToAMG_Node<NT_FACE>(mar, node_num);
				 constexpr int NODE_SIZE = sizeof(AMG_Node<NT_FACE>::v)/sizeof(AMG_Node<NT_VERTEX>);
				 for (int k = 0; k < NODE_SIZE; k++) node[k] = vert_sort[node[k]];
				 node.Sort();
				 return move(node);},
			       [face_sort](auto i, auto j) { face_sort[i] = j; } );
    }
    return mesh;
  }

  std::ostream & operator<<(std::ostream &os, const TopologicMesh& p)
  {
    os << " TopMesh, NNODES: "; for(auto l:Range(4)) os << p.nnodes[l] << " "; os << endl;
    if(p.nnodes[0]) os << "verts: " << p.nnodes[0] << endl; prow(p.verts); cout << endl;
    if(p.nnodes[1]) os << "edges: " << p.nnodes[1] << endl; prow(p.edges); cout << endl;
    if(p.nnodes[2]) os << "faces: " << p.nnodes[2] << endl; prow(p.faces); cout << endl;
    if(p.nnodes[3]) os << "cells: " << p.nnodes[3] << endl; prow(p.cells); cout << endl;
    return os;
  }

  std::ostream & operator<<(std::ostream &os, const FlatTM& p)
  {
    os << " FlatTM, NNODES: "; for(auto l:Range(4)) os << p.nnodes[l] << " "; cout << endl;
    if(p.nnodes[0]) os << "verts: " << p.nnodes[0] << endl; prow(p.verts); cout << endl;
    if(p.nnodes[1]) os << "edges: " << p.nnodes[1] << endl; prow(p.edges); cout << endl;
    if(p.nnodes[2]) os << "faces: " << p.nnodes[2] << endl; prow(p.faces); cout << endl;
    if(p.nnodes[3]) os << "cells: " << p.nnodes[3] << endl; prow(p.cells); cout << endl;
    return os;
  }

  std::ostream & operator<<(std::ostream &os, const BlockTM& p)
  {
    os << endl << "------" << endl;
    os << " BlockTM, NNODES glob:  "; for(auto l:Range(4)) os << p.nnodes_glob[l] << " "; cout << endl;
    os << " BlockTM, NNODES loc :   "; for(auto l:Range(4)) os << p.nnodes[l] << " "; cout << endl;
    os << " BlockTM, NNODES eqc:   " << endl; for(auto l:Range(4)) { os << "NT " << l << ": "; prow2(p.nnodes_eqc[l], os); os << endl; }
    os << " BlockTM, NNODES cross: " << endl; for(auto l:Range(4)) { os << "NT " << l << ": "; prow2(p.nnodes_cross[l], os); os << endl; }
    os << " BlockTM is on ";
    if(p.eqc_h!=nullptr) { cout << *p.eqc_h; cout << endl; }
    else cout << " NO EQCH !!" << endl;
    os << "eqc_verts: " << p.eqc_verts; cout << endl;
    os << "eqc_edges: " << p.eqc_edges; cout << endl;
    os << "cross_edges: " << p.cross_edges; cout << endl;
    os << "------" << endl << endl;
    return os;
  }

} // namespace amg
