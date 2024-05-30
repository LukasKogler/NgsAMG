#ifndef FILE_BASE_MESH_HPP
#define FILE_BASE_MESH_HPP

#include <base.hpp>
#include <utils.hpp>               // param-pack utils
#include <utils_arrays_tables.hpp> // for algos
#include <reducetable.hpp>

#include "alg_mesh_nodes.hpp"

namespace amg
{
// class BaseGridMapStep;
// // class BaseAgglomerateCoarseMap;
// class GridContractMap;

// Only topology!
class TopologicMesh
{
  friend class NgsAMG_Comm;
  friend class BaseCoarseMap;
  friend class BaseAgglomerateCoarseMap;
  friend class LocCoarseMap;
  friend class GridContractMap;
  // friend class AgglomerateCoarseMap;
public:
  TopologicMesh (shared_ptr<EQCHierarchy> _eqc_h, size_t nv = 0, size_t ne = 0, size_t nf = 0, size_t nc = 0);
  TopologicMesh (TopologicMesh && other);
  virtual ~TopologicMesh() { ; }
  shared_ptr<EQCHierarchy> GetEQCHierarchy () const { return eqc_h; }
  template<NODE_TYPE NT> INLINE bool HasNodes () const { return has_nodes[NT]; }
  template<NODE_TYPE NT> INLINE size_t GetNN () const { return nnodes[NT]; }
  template<NODE_TYPE NT> INLINE size_t GetNNGlobal () const  { return nnodes_glob[NT]; }
  template<NODE_TYPE NT> INLINE AMG_Node<NT> GetNode (size_t node_num) const {
    if constexpr(NT==NT_VERTEX) return verts[node_num];
    else if constexpr(NT==NT_EDGE) return edges[node_num];
    else if constexpr(NT==NT_FACE) return faces[node_num];
    else if constexpr(NT==NT_CELL) return cells[node_num];
  }
  template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetNodes () const {
    if constexpr(NT==NT_VERTEX) return verts;
    if constexpr(NT==NT_EDGE) return edges;
    if constexpr(NT==NT_FACE) return faces;
    if constexpr(NT==NT_CELL) return cells;
  }
  virtual shared_ptr<SparseMatrix<double>> GetEdgeCM () const;
  virtual void ResetEdgeCM () const { econ = nullptr; }
  void SetFreeNodes (shared_ptr<BitArray> _free_nodes) const { free_nodes = _free_nodes; }
  shared_ptr<BitArray> GetFreeNodes () const { return free_nodes; }

  virtual void printTo(std::ostream &os) const;

  // // I think this is not currently used anywhere, but I probably need it for stokes?
  // virtual void MapAdditionalData (const BaseGridMapStep & amap);

  template<NODE_TYPE NT>
  void SetNodeArray (Array<AMG_Node<NT>> && _node_array) {
    has_nodes[NT] = true;
    if constexpr(NT==NT_VERTEX) verts = _node_array;
    if constexpr(NT==NT_EDGE) edges = _node_array;
    if constexpr(NT==NT_FACE) faces = _node_array;
    if constexpr(NT==NT_CELL) cells = _node_array;
  }
  template<NODE_TYPE NT>
  void SetNN(size_t _nn, size_t _nn_glob) {
    nnodes[NT] = _nn;
    nnodes_glob[NT] = _nn_glob;
  }
protected:
  shared_ptr<EQCHierarchy> eqc_h;
  bool has_nodes[4] = {false, false, false, false};
  size_t nnodes[4];
  size_t nnodes_glob[4];
  Array<AMG_Node<NT_VERTEX>> verts; // can be empty (when its just)
  Array<AMG_Node<NT_EDGE>> edges;
  Array<AMG_Node<NT_FACE>> faces;
  Array<AMG_Node<NT_CELL>> cells;
  mutable shared_ptr<SparseMatrix<double>> econ = nullptr;
  mutable shared_ptr<BitArray> free_nodes = nullptr; // this is kinda hacky ...
}; // class TopologicMesh


INLINE std::ostream & operator<<(std::ostream &os, const TopologicMesh& p)
{
  p.printTo(os);
  return os;
}

class BlockTM : public TopologicMesh
{
  friend class NgsAMG_Comm;
  friend class BaseCoarseMap;
  friend class BaseAgglomerateCoarseMap;
  friend class LocCoarseMap;
  friend class GridContractMap;
public:
  BlockTM (shared_ptr<EQCHierarchy> _eqc_h);
  BlockTM (BlockTM && other);
  BlockTM ();

  ~BlockTM () = default;

  // not needed currently (for stokes maybe??)
  // virtual void ContractData(GridContractMap& cmap) const {}

  virtual void printTo(std::ostream &os) const override;

  template<NODE_TYPE NT> INLINE size_t GetENN () const { return disp_eqc[NT].Last(); }
  template<NODE_TYPE NT> INLINE size_t GetENN (size_t eqc_num) const { return nnodes_eqc[NT][eqc_num]; }
  template<NODE_TYPE NT> INLINE size_t GetCNN () const { return disp_cross[NT].Last(); }
  template<NODE_TYPE NT> INLINE size_t GetCNN (size_t eqc_num) const { return nnodes_cross[NT][eqc_num]; }
  template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetENodes (size_t eqc_num) const;
  template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetCNodes (size_t eqc_num) const;
  template<NODE_TYPE NT> INLINE size_t GetEqcOfENode (size_t node_num) const
  { return merge_pos_in_sorted_array(node_num, disp_eqc[NT]) - 1; }
  template<NODE_TYPE NT> INLINE size_t GetEqcOfCNode (size_t node_num) const
  { return merge_pos_in_sorted_array(node_num - GetENN<NT>(), disp_cross[NT]) - 1; }
  template<NODE_TYPE NT> INLINE size_t GetEQCOfNode (size_t node_num) const
  { return (node_num < GetENN<NT>()) ? GetEqcOfENode<NT>(node_num) : GetEqcOfCNode<NT>(node_num); }
  template<NODE_TYPE NT> INLINE int MapENodeToEQC (size_t node_num) const
  { auto eq = GetEQCOfNode<NT>(node_num); return node_num - disp_eqc[NT][eq]; }
  template<NODE_TYPE NT> INLINE int MapENodeToEQC (int eq, size_t node_num) const
  { return node_num - disp_eqc[NT][eq]; }
  template<NODE_TYPE NT> INLINE int MapCNodeToEQC (int eq, size_t node_num) const
  { return node_num - GetENN<NT>() - disp_cross[NT][eq]; }
  template<NODE_TYPE NT> INLINE int MapCNodeToEQC (size_t node_num) const
  { auto eq = GetEQCOfNode<NT>(node_num); return MapCNodeToEQC<NT>(eq, node_num); }
  template<NODE_TYPE NT> INLINE int MapNodeToEQC (size_t node_num) const
  { return (node_num < GetENN<NT>()) ? MapENodeToEQC<NT>(node_num) : MapCNodeToEQC<NT>(node_num); }
  template<NODE_TYPE NT> INLINE std::tuple<int, int> MapENodeToEQLNR (int node_num) const {
    auto eq = GetEqcOfENode<NT>(node_num);
    return make_tuple(eq, MapENodeToEQC<NT>(eq, node_num));
  }
  template<NODE_TYPE NT> INLINE std::tuple<int, int> MapCNodeToEQLNR (int node_num) const {
    auto eq = GetEqcOfCNode<NT>(node_num);
    return make_tuple(eq, MapCNodeToEQC<NT>(eq, node_num));
  }
  template<NODE_TYPE NT> INLINE std::tuple<int, int> MapNodeToEQLNR (int node_num) const
  { return (node_num < GetENN<NT>()) ? MapENodeToEQLNR<NT>(node_num) : MapCNodeToEQLNR<NT>(node_num); }
  template<NODE_TYPE NT> INLINE int MapENodeFromEQC (size_t node_num, size_t eqc_num) const
  { return node_num + disp_eqc[NT][eqc_num]; }
  template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
  INLINE int MapCNodeFromEQC (size_t node_num, size_t eqc_num) const
  { return GetENN<NT>() + node_num + disp_cross[NT][eqc_num]; }
protected:
  using TopologicMesh::eqc_h;
  /** eqc-wise data **/
  Array<size_t> nnodes_eqc[4];
  FlatTable<AMG_Node<NT_VERTEX>> eqc_verts = FlatTable<AMG_Node<NT_VERTEX>>(0, nullptr, nullptr);
  FlatTable<AMG_Node<NT_EDGE>> eqc_edges = FlatTable<AMG_Node<NT_EDGE>>(0, nullptr, nullptr);
  FlatTable<AMG_Node<NT_FACE>> eqc_faces = FlatTable<AMG_Node<NT_FACE>>(0, nullptr, nullptr);
  Array<Array<size_t>> disp_eqc; // displacement in node array
  /** padding data **/
  Array<size_t> nnodes_cross[4];
  /** cross data **/
  FlatTable<AMG_Node<NT_EDGE>> cross_edges = FlatTable<AMG_Node<NT_EDGE>>(0, nullptr, nullptr);
  FlatTable<AMG_Node<NT_FACE>> cross_faces = FlatTable<AMG_Node<NT_FACE>>(0, nullptr, nullptr);
  Array<Array<size_t>> disp_cross; // displacement in node array
  template<ngfem::NODE_TYPE NT> friend
  void BuildNodes (shared_ptr<MeshAccess> ma, BlockTM& tmesh,
        FlatArray<FlatArray<int>> node_sort);
  friend shared_ptr<BlockTM> MeshAccessToBTM (shared_ptr<MeshAccess> ma, shared_ptr<EQCHierarchy> eqc_h,
          FlatArray<int> vert_sort,
          bool build_edges, FlatArray<int> edge_sort,
          bool build_faces, FlatArray<int> face_sort,
          bool build_cells, FlatArray<int> cell_sort);
public:
  INLINE size_t GetNEqcs () const { return eqc_verts.Size(); } // !! ugly, but necessary in contract.cpp, where I dont have eqc for sent mesh OMG
  // Apply a lambda-function to each node
  template<NODE_TYPE NT, class TLAM>
  INLINE void Apply (TLAM lam, bool master_only = false) const {
    if (master_only) {
      for (auto eqc : Range(GetNEqcs()))
        if ( eqc_h->IsMasterOfEQC(eqc) ) {
          for (const auto& node : GetENodes<NT>(eqc))
            { lam(node); }
          if constexpr(NT!=NT_VERTEX) for (const auto& node : GetCNodes<NT>(eqc))
            { lam(node); }
        }
    }
    else {
      for (const auto & node : GetNodes<NT>())
        { lam(node); }
    }
  }
  // Apply a lambda-function to each eqc/node - pair
  template<NODE_TYPE NT, class TX, class TLAM>
  INLINE void ApplyEQ (TX&& eqcs, TLAM lam, bool master_only = false) const {
    for (auto eqc : eqcs) {
      if ( !master_only || eqc_h->IsMasterOfEQC(eqc) ) {
        for (const auto& node : GetENodes<NT>(eqc))
          { lam(eqc, node); }
        if constexpr(NT != NT_VERTEX) {
          for (const auto& node : GetCNodes<NT>(eqc))
            { lam(eqc, node); }
        }
      }
    }
  }
  template<NODE_TYPE NT, class TLAM>
  INLINE void ApplyEQ (TLAM lam, bool master_only = false) const {
    ApplyEQ<NT>(Range(GetNEqcs()), lam, master_only);
  }
  template<NODE_TYPE NT, class TX, class TLAM>
  INLINE void ApplyEQ2 (TX&& eqcs, TLAM lam, bool master_only = false) const {
    for (auto eqc : eqcs) {
      if ( !master_only || eqc_h->IsMasterOfEQC(eqc) ) {
        lam(eqc, GetENodes<NT>(eqc));
        if constexpr(NT != NT_VERTEX)
          { lam(eqc, GetCNodes<NT>(eqc)); }
      }
    }
  }
  template<NODE_TYPE NT, class TLAM>
  INLINE void ApplyEQ2 (TLAM lam, bool master_only = false) const {
    ApplyEQ2<NT>(Range(GetNEqcs()), lam, master_only);
  }
  // ugly stuff
  template<NODE_TYPE NT, class T>
  void ScatterNodalData (Array<T> & avdata) const {
    int neqcs = eqc_h->GetNEQCS();
    if (neqcs < 2) // nothing to do!
{ return; }
    int nreq = 0;
    Array<int> cnt(neqcs); cnt = 0;
    for (auto k : Range(1, neqcs)) {
cnt[k] = GetENN<NT>(k) + GetCNN<NT>(k);
nreq += eqc_h->IsMasterOfEQC(k) ? (eqc_h->GetDistantProcs(k).Size()) : 1;
    }
    Table<T> ex_data(cnt);
    Array<NG_MPI_Request> req(nreq); nreq = 0;
    auto comm = eqc_h->GetCommunicator();
    auto & disp_eq = disp_eqc[NT];
    auto & disp_c  = disp_cross[NT];
    for (auto k : Range(1, neqcs)) {
auto exrow = ex_data[k];
if (eqc_h->IsMasterOfEQC(k)) {
  exrow.Part(0, GetENN<NT>(k)) = avdata.Part(disp_eq[k], GetENN<NT>(k));
  exrow.Part(GetENN<NT>(k)) = avdata.Part(GetENN<NT>() + disp_c[k], GetCNN<NT>(k));
  // cout << " for eqc " << k << " send exrow to "; prow(eqc_h->GetDistantProcs(k)); cout << endl;
  // cout << GetENN<NT>() << " " << GetENN<NT>(k) << " " << GetCNN<NT>(k) << endl;
  // cout << disp_eq[k] << " " << disp_c[k] << endl;
  // prow2(exrow); cout << endl;
  for (auto p : eqc_h->GetDistantProcs(k))
    { req[nreq++] = comm.ISend(exrow, p, NG_MPI_TAG_AMG); }
}
else
  { req[nreq++] = comm.IRecv(exrow, eqc_h->GetDistantProcs(k)[0], NG_MPI_TAG_AMG); }
    }
    // cout << " nreq " << req.Size() << " " << nreq << endl;
    MyMPI_WaitAll(req);
    for (auto k : Range(1, neqcs)) {
if (!eqc_h->IsMasterOfEQC(k)) {
  auto exrow = ex_data[k];
  // cout << " for eqc " << k << " got exrow "; prow(eqc_h->GetDistantProcs(k)); cout << endl;
  // cout << GetENN<NT>() << " " << GetENN<NT>(k) << " " << GetCNN<NT>(k) << endl;
  // cout << disp_eq[k] << " " << disp_c[k] << endl;
  // prow2(exrow); cout << endl;
  avdata.Part(disp_eq[k], GetENN<NT>(k)) = exrow.Part(0, GetENN<NT>(k));
  avdata.Part(GetENN<NT>() + disp_c[k], GetCNN<NT>(k)) = exrow.Part(GetENN<NT>(k));
}
    }
  }
  template<NODE_TYPE NT, class T, class TRED>
  void AllreduceNodalData (Array<T> & avdata, TRED red, bool apply_loc = false) const {
    // TODO: this should be much easier - data is already in eqc-wise form (since nodes are ordered that way now)!
    // cout << "alred. nodal data, NT=" << NT << ", NN " << GetNN<NT>() << " ndata " << avdata.Size() << " appl loc " << apply_loc << endl;
    // cout << "data in: " << endl; prow2(avdata); cout << endl;
    int neqcs = eqc_h->GetNEQCS();
    if (neqcs == 0) return; // nothing to do!
    if (neqcs == 1 && !apply_loc) return; // nothing to do!
    Array<int> rowcnt(neqcs);
    for (auto k : Range(neqcs))
{ rowcnt[k] = nnodes_eqc[NT][k] + nnodes_cross[NT][k]; }
    // for (auto k : Range(neqcs))
// { cout << "rowcnt[" << k << "] = " << nnodes_eqc[NT][k] << " + " << nnodes_cross[NT][k] << endl; }
    if (!apply_loc) rowcnt[0] = 0;
    Table<T> data(rowcnt);
    int C = 0;
    auto loop_eqcs = [&] (auto lam, auto & data) {
for (auto k : Range(apply_loc?0:1, neqcs)) {
  C = 0;
  lam(GetENodes<NT>(k), data[k]);
  if constexpr(NT!=NT_VERTEX) lam(GetCNodes<NT>(k), data[k]);
}
    };
    loop_eqcs([&](auto nodes, auto row)
  { if constexpr(NT==NT_VERTEX) for (auto l:Range(nodes.Size())) row[C++] = avdata[nodes[l]];
    else for (auto l:Range(nodes.Size())) row[C++] = avdata[nodes[l].id]; },
  data);
    // cout << "ARND data: " << endl << data << endl;
    Table<T> reduced_data = ReduceTable<T,T,TRED>(data, eqc_h, red);
    // cout << "ARND reduced data: " << endl << reduced_data << endl;
    loop_eqcs([&](auto nodes, auto row)
  { if constexpr(NT==NT_VERTEX) for (auto l:Range(nodes.Size())) avdata[nodes[l]] = row[C++];
    else for (auto l:Range(nodes.Size())) {
        // cout << l << " " << nodes[l] << " to " << C << " " << row.Size() << endl;
        avdata[nodes[l].id] = row[C++];
      }
  },
  reduced_data);
    // cout << "data out: " << endl; prow2(avdata); cout << endl;
  } // CumulateNodalData

  template<class TGET, class TSET>
  INLINE void SetVs  (size_t annodes, TGET get_dps, TSET set_sort);

  INLINE void SetVs (size_t nnodes, Table<int> dist_procs, FlatArray<int> node_sort)
    { SetVs(nnodes, [&dist_procs](auto i){ return dist_procs[i]; }, [&node_sort](auto i, auto j){ node_sort[i] = j; }); }

  template<ngfem::NODE_TYPE NT, class TGET, class TSET, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
  INLINE void SetNodes (size_t annodes, TGET get_node, TSET set_sort);

  template<ngfem::NODE_TYPE NT, class TGET, class TSET, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
  INLINE void SetNodes (Array<decltype(AMG_Node<NT>::v)> nodes, FlatArray<int> node_sort)
    { SetNodes (nodes.Size(), [&nodes](auto i) {return nodes[i]; }, [&node_sort](auto i, auto j){ node_sort[i] = j; }); }
  
  template<ngfem::NODE_TYPE NT>
  INLINE void SetEQOS (Array<size_t> && _adisp);
}; // class BlockTM

template<> INLINE FlatArray<AMG_Node<NT_VERTEX>> BlockTM::GetENodes<NT_VERTEX> (size_t eqc_num) const
{ return (eqc_num == size_t(-1)) ? verts : eqc_verts[eqc_num]; }
template<> INLINE FlatArray<AMG_Node<NT_EDGE>> BlockTM::GetENodes<NT_EDGE> (size_t eqc_num) const
{ return (eqc_num == size_t(-1)) ? eqc_edges.AsArray() : eqc_edges[eqc_num]; }
template<> INLINE FlatArray<AMG_Node<NT_FACE>> BlockTM::GetENodes<NT_FACE> (size_t eqc_num) const
{ return (eqc_num == size_t(-1)) ? eqc_faces.AsArray() : eqc_faces[eqc_num]; }

template<> INLINE FlatArray<AMG_Node<NT_EDGE>> BlockTM::GetCNodes<NT_EDGE> (size_t eqc_num) const
{ return (eqc_num == size_t(-1)) ? cross_edges.AsArray() : cross_edges[eqc_num]; }
template<> INLINE FlatArray<AMG_Node<NT_FACE>> BlockTM::GetCNodes<NT_FACE> (size_t eqc_num) const
{ return (eqc_num == size_t(-1)) ? cross_faces.AsArray() : cross_faces[eqc_num]; }


shared_ptr<BlockTM> MeshAccessToBTM (shared_ptr<MeshAccess> ma, shared_ptr<EQCHierarchy> eqc_h,
              FlatArray<int> vert_sort,
              bool build_edges, FlatArray<int> edge_sort,
              bool build_faces, FlatArray<int> face_sort,
              bool build_cells, FlatArray<int> cell_sort);

} // namespace amg

#include "base_mesh_impl.hpp"

#endif // FILE_BASE_MESH_HPP