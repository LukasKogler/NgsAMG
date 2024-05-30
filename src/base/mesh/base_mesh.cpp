#define FILE_AMG_MESH_CPP

#include <base.hpp>
#include <utils.hpp>
#include <utils_io.hpp>
// #include <base_map.hpp>
// #include <base_agg.hpp>

#include "base_mesh.hpp"

namespace amg
{
TopologicMesh :: TopologicMesh (shared_ptr<EQCHierarchy> _eqc_h, size_t nv, size_t ne, size_t nf, size_t nc)
  : eqc_h(_eqc_h)
{
  nnodes[0] = nnodes_glob[0] = nv;
  nnodes[1] = nnodes_glob[1] = ne;
  nnodes[2] = nnodes_glob[2] = nf;
  nnodes[3] = nnodes_glob[3] = nc;
}

TopologicMesh :: TopologicMesh (TopologicMesh && other)
  : eqc_h(other.eqc_h),
    verts(std::move(other.verts)), edges(std::move(other.edges)),
    faces(std::move(other.faces)), cells(std::move(other.cells))
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

  // cout << " GetEdgeCM(), edges over? " << endl;
  // for(auto & edge: GetNodes<NT_EDGE>())
  // {
  //   for(auto l:Range(2))
  //   {
  //     auto v = edge.v[l];
  //     if ( v < 0 ||  v > GetNN<NT_VERTEX>() )
  //     {
  //       cout << " EDGE OVER! " << edge << endl;
  //     }
  //   }
  // }

  auto nv = GetNN<NT_VERTEX>();
  Array<int> econ_s(nv);
  econ_s = 0;
  for(auto & edge: GetNodes<NT_EDGE>())
    for(auto l:Range(2))
econ_s[edge.v[l]]++;
  Table<IVec<2> > econ_i(econ_s);
  econ_s = 0;
  for(auto & edge: GetNodes<NT_EDGE>())
    for(auto l:Range(2))
econ_i[edge.v[l]][econ_s[edge.v[l]]++] = IVec<2>(edge.v[1-l], edge.id);
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

// void TopologicMesh :: MapAdditionalData (const BaseGridMapStep & amap)
// {
//   ;
// }

template<typename T> void AssignArray (Array<T> & a, FlatArray<T> & b)
{
  if (b.Size())
    { a = Array<T>(b.Size(), &b[0]); }
  else
    { a = Array<T>(b.Size(), nullptr); }
}

BlockTM :: BlockTM (shared_ptr<EQCHierarchy> _eqc_h)
  : TopologicMesh(_eqc_h), disp_eqc(4), disp_cross(4)
{
  auto neqcs = eqc_h->GetNEQCS();
  for (auto l:Range(4)) {
    nnodes_cross[l] = Array<size_t>(neqcs); nnodes_cross[l] = 0;
    nnodes_eqc[l] = Array<size_t>(neqcs); nnodes_eqc[l] = 0;
    disp_eqc[l] = Array<size_t> (neqcs+1); disp_eqc[l] = 0;
    disp_cross[l] = Array<size_t> (neqcs+1); disp_cross[l] = 0;
  }
  eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &(disp_eqc[NT_VERTEX][0]), nullptr);
  eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &(disp_eqc[NT_EDGE][0]), nullptr);
  eqc_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &(disp_eqc[NT_FACE][0]), nullptr);
  // eqc_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &(disp_eqc[NT_CELL][0]), &cells[0]);
  disp_cross[NT_EDGE] = Array<size_t> (neqcs+1); disp_cross[NT_EDGE] = 0;
  cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &(disp_cross[NT_EDGE][0]), nullptr);
  disp_cross[NT_FACE] = Array<size_t> (neqcs+1); disp_cross[NT_FACE] = 0;
  cross_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &(disp_cross[NT_FACE][0]), nullptr);
}

BlockTM :: BlockTM ( BlockTM && other)
  : TopologicMesh (std::move(other))
{
  eqc_h = other.eqc_h; other.eqc_h = nullptr;
  disp_eqc.SetSize(4);
  disp_cross.SetSize(4);
  for (auto l:Range(4)) {
    nnodes_eqc[l] = std::move(other.nnodes_eqc[l]);
    disp_eqc[l] = std::move(other.disp_eqc[l]);
    nnodes_cross[l] = std::move(other.nnodes_cross[l]);
    disp_cross[l] = std::move(other.disp_cross[l]);
  }
  auto neqcs = eqc_h->GetNEQCS();
  eqc_verts = MakeFT<AMG_Node<NT_VERTEX>> (neqcs, disp_eqc[NT_VERTEX], verts, 0);
  eqc_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, disp_eqc[NT_EDGE], edges, 0);
  eqc_faces = MakeFT<AMG_Node<NT_FACE>> (neqcs, disp_eqc[NT_FACE], faces, 0);
  cross_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, disp_cross[NT_EDGE], edges, disp_eqc[NT_EDGE].Last());
  cross_faces = MakeFT<AMG_Node<NT_FACE>> (neqcs, disp_cross[NT_FACE], faces, disp_eqc[NT_FACE].Last());
} // BlockTM (..)


BlockTM :: BlockTM ()
  : TopologicMesh(nullptr), disp_eqc(4), disp_cross(4)
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
  // mesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, &mesh.disp_eqc[NT_VERTEX][0], &mesh.verts[0]);
  mesh.eqc_verts = MakeFT<AMG_Node<NT_VERTEX>> (neqcs, mesh.disp_eqc[NT_VERTEX], mesh.verts, 0);
  if (mesh.has_nodes[1]) {
    Recv(mesh.edges, src, tag);
    Recv(mesh.disp_eqc[1], src, tag);
    Recv(mesh.disp_cross[1], src, tag);
  }
  // mesh.eqc_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &mesh.disp_eqc[NT_EDGE][0], &mesh.edges[0]);
  mesh.eqc_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, mesh.disp_eqc[NT_EDGE], mesh.edges, 0);
  // mesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, &mesh.disp_cross[NT_EDGE][0], &mesh.edges[mesh.disp_eqc[NT_EDGE].Last()]);
  mesh.cross_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, mesh.disp_cross[NT_EDGE], mesh.edges, mesh.disp_eqc[NT_EDGE].Last());
  if (mesh.has_nodes[2]) {
    Recv(mesh.faces, src, tag);
    Recv(mesh.disp_eqc[2], src, tag);
    Recv(mesh.disp_cross[2], src, tag);
  }
  // mesh.eqc_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &mesh.disp_eqc[NT_FACE][0], &mesh.faces[0]);
  mesh.eqc_faces = MakeFT<AMG_Node<NT_FACE>> (neqcs, mesh.disp_eqc[NT_FACE], mesh.faces, 0);
  // mesh.cross_faces = FlatTable<AMG_Node<NT_FACE>> (neqcs, &mesh.disp_cross[NT_FACE][0], &mesh.faces[mesh.disp_eqc[NT_FACE].Last()]);
  mesh.cross_faces = MakeFT<AMG_Node<NT_FACE>> (neqcs, mesh.disp_cross[NT_FACE], mesh.faces, mesh.disp_eqc[NT_FACE].Last());
  if (mesh.has_nodes[3]) {
    Recv(mesh.cells, src, tag);
    Recv(mesh.disp_eqc[3], src, tag);
    Recv(mesh.disp_cross[3], src, tag);
  }
  // mesh.eqc_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &mesh.disp_eqc[NT_CELL][0], &mesh.cells[0]);
  // mesh.cross_cells = FlatTable<AMG_Node<NT_CELL>> (neqcs, &mesh.disp_cross[NT_CELL][0], &mesh.cells[mesh.disp_eqc[NT_CELL].Last()]);
}


namespace amg_nts
{
  template<ngfem::NODE_TYPE NT>
  INLINE decltype(AMG_Node<NT>::v) MAToAMG_Node (const MeshAccess & ma, size_t node_num) {
    constexpr int NODE_SIZE = sizeof(AMG_Node<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
    IVec<NODE_SIZE, AMG_Node<NT_VERTEX>> vs;
    auto ng_node = ma.GetNode<NT>(node_num);
    for (auto i : Range(NODE_SIZE)) vs[i] = ng_node.vertices[i];
    return std::move(vs);
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
  if (ma->GetCommunicator().Size() > 1)
    mesh->SetVs (ma->GetNV(), [&](auto vnr)->FlatArray<int>{ return ma->GetDistantProcs(NodeId(NT_VERTEX, vnr)); },
      [vert_sort](auto i, auto j){ vert_sort[i] = j; });
  else {
    auto & M(*mesh);
    M.has_nodes[NT_VERTEX] = true;
    auto nv = ma->GetNV();
    M.nnodes[NT_VERTEX] = nv;
    M.verts.SetSize(nv);
    for (auto k:Range(nv)) {
M.verts[k] = k;
vert_sort[k] = k;
    }
    Array<size_t> & disp(M.disp_eqc[NT_VERTEX]);
    disp.SetSize(2);
    disp[0] = 0; disp[1] = nv;
    Array<int> dummy;
    mesh->SetVs (ma->GetNV(), [&](auto vnr)->FlatArray<int>{ return dummy; },
      [vert_sort](auto i, auto j){ vert_sort[i] = j; });
    // M.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>>(1, &(M.disp_eqc[NT_VERTEX][0]), &(M.verts[0]));
    M.eqc_verts = MakeFT<AMG_Node<NT_VERTEX>>(1, M.disp_eqc[NT_VERTEX], M.verts, 0);
    M.nnodes_eqc[NT_VERTEX].SetSize(1); M.nnodes_eqc[NT_VERTEX][0] = nv;
    M.nnodes_cross[NT_VERTEX].SetSize(1); M.nnodes_cross[NT_VERTEX][0] = 0;
    M.nnodes_glob[NT_VERTEX] = M.nnodes[NT_VERTEX];
  }
  const MeshAccess & mar = *ma;
  if (build_edges) {
    size_t num_edges_total = ma->GetNEdges();
    BitArray fine_edge(num_edges_total); fine_edge.Clear();
    int ne = ma->GetNE();
    for (auto i : Range(ne)) {
ElementId ei(VOL, i);
auto eledges = ma->GetElEdges (ei);
for(size_t j=0;j<eledges.Size();j++) fine_edge.SetBit(eledges[j]);
    }
    // do I need sth. like that??
    // ma->AllReduceNodalData (NT_EDGE, fine_edge, NG_MPI_LOR);
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
        return std::move(node);},
            [edge_sort](auto i, auto j) { edge_sort[i] = j; } );
  }
  if (build_faces) {
    throw Exception("face-nodes only work with non-refined mesh, remove this if that is the case!!");
    mesh->SetNodes<NT_FACE>( ma->GetNFaces(),
            [&](auto node_num) {
        auto node = amg_nts::MAToAMG_Node<NT_FACE>(mar, node_num);
        constexpr int NODE_SIZE = sizeof(AMG_Node<NT_FACE>::v)/sizeof(AMG_Node<NT_VERTEX>);
        for (int k = 0; k < NODE_SIZE; k++) node[k] = vert_sort[node[k]];
        node.Sort();
        return std::move(node);},
            [face_sort](auto i, auto j) { face_sort[i] = j; } );
  }
  return mesh;
}


void TopologicMesh :: printTo (std::ostream &os) const
{
  os << " TopMesh, NNODES: "; for(auto l:Range(4)) os << nnodes[l] << " "; os << endl;
  if(nnodes[0]) os << "verts: " << nnodes[0] << endl; prow(verts); os << endl;
  if(nnodes[1]) os << "edges: " << nnodes[1] << endl; prow(edges); os << endl;
  if(nnodes[2]) os << "faces: " << nnodes[2] << endl; prow(faces); os << endl;
  if(nnodes[3]) os << "cells: " << nnodes[3] << endl; prow(cells); os << endl;
}

void BlockTM :: printTo (std::ostream &os) const
{
  os << endl << "------" << endl;
  os << "MeshType " << typeid(*this).name() << endl;
  os << endl << "------" << endl;
  os << " BlockTM, NNODES glob:  "; for(auto l:Range(4)) os << nnodes_glob[l] << " "; os << endl;
  os << " BlockTM, NNODES loc :   "; for(auto l:Range(4)) os << nnodes[l] << " "; os << endl;

  os << " BlockTM, NNODES eqc:   " << endl;
  for(auto l:Range(4))
    { os << "NT " << l << ": "; prow2(nnodes_eqc[l], os); os << endl; }
  os << " BlockTM, NNODES cross: " << endl;
  for(auto l:Range(4))
    { os << "NT " << l << ": "; prow2(nnodes_cross[l], os); os << endl; }
  os << " The EQCHierarchy is ";
  if(eqc_h!=nullptr)
    { os << *eqc_h << endl; }
  else
    { os << " NO EQCH !!" << endl; }

  os << " NODES: " << endl;
  os << "  eqc_verts: " << endl;
  printTable(eqc_verts, os, "   ", 30);

  os << "  eqc_edges: " << endl;
  printTable(eqc_edges, os, "   ", 15);

  os << "  cross_edges: " << endl;
  printTable(cross_edges, os, "   ", 15);

  os << endl << "edge-connectivity matrix: " << endl;
  os << *this->GetEdgeCM() << endl;

  os << "------" << endl << endl;

}

} // namespace amg
