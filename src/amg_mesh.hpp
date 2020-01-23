#ifndef FILE_AMG_MESH_HPP
#define FILE_AMG_MESH_HPP

#include "amg.hpp"

namespace amg
{

  class PairWiseCoarseMap;
  template<class TMESH> class VDiscardMap;
  template<class TMESH> class AgglomerateCoarseMap;
  
  // Only topology!
  class TopologicMesh
  {
    friend class NgsAMG_Comm;
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
    friend std::ostream & operator<<(std::ostream &os, const TopologicMesh& p);
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
  }; // class TopologicMesh

  
  class FlatTM : public TopologicMesh
  {
  public:

    FlatTM (shared_ptr<EQCHierarchy> eqc_h = nullptr);

    FlatTM (FlatArray<AMG_Node<NT_VERTEX>> av, FlatArray<AMG_Node<NT_EDGE>> ae,
	    FlatArray<AMG_Node<NT_FACE>> af  , FlatArray<AMG_Node<NT_CELL>> ac,
	    FlatArray<AMG_Node<NT_EDGE>> ace , FlatArray<AMG_Node<NT_FACE>> acf,
	    shared_ptr<EQCHierarchy> eqc_h = nullptr);
    ~FlatTM () { ; }
    template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE size_t GetCNN ()
    { return nnodes_cross[NT]; }
    template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE AMG_Node<NT> GetCNode (size_t node_num)
    {
      if constexpr(NT==NT_EDGE) return cross_edges[node_num];
      else if constexpr(NT==NT_FACE) return cross_faces[node_num];
    }
    template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE FlatArray<AMG_Node<NT>> GetCNodes ()
    {
      if constexpr(NT==NT_EDGE) return cross_edges;
      else if constexpr(NT==NT_FACE) return cross_faces;
    }
    friend std::ostream & operator<<(std::ostream &os, const FlatTM& p);
  protected:
    size_t nnodes_cross[4];
    Array<AMG_Node<NT_EDGE>> cross_edges;
    Array<AMG_Node<NT_FACE>> cross_faces;
  };


  class BlockTM : public TopologicMesh
  {
    template<class TMESH> friend class GridContractMap;
    template<class TMESH> friend class VDiscardMap;
    template<class TMESH> friend class AgglomerateCoarseMap;
    friend class NgsAMG_Comm;
  public:
    BlockTM (shared_ptr<EQCHierarchy> _eqc_h);
    BlockTM (BlockTM && other);
    BlockTM ();
    ~BlockTM () { ; }
    template<NODE_TYPE NT> INLINE size_t GetENN () const { return disp_eqc[NT].Last(); }
    template<NODE_TYPE NT> INLINE size_t GetENN (size_t eqc_num) const { return nnodes_eqc[NT][eqc_num]; }
    template<NODE_TYPE NT> INLINE size_t GetCNN () const { return disp_cross[NT].Last(); }
    template<NODE_TYPE NT> INLINE size_t GetCNN (size_t eqc_num) const { return nnodes_cross[NT][eqc_num]; }
    template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetENodes (size_t eqc_num) const;
    template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetCNodes (size_t eqc_num) const;
    template<NODE_TYPE NT> INLINE size_t GetEqcOfNode (size_t node_num) const {
      // array[pos-1] <= elem < array[pos]
      // return merge_pos_in_sorted_array(node_num, disp_eqc[NT]) - 1;
      return (node_num < GetENN<NT>()) ? merge_pos_in_sorted_array(node_num, disp_eqc[NT]) - 1 :
	merge_pos_in_sorted_array(node_num - GetENN<NT>(), disp_cross[NT]) - 1;
    }
    template<NODE_TYPE NT> INLINE int MapENodeToEQC (size_t node_num) const 
    { auto eq = GetEqcOfNode<NT>(node_num); return node_num - disp_eqc[NT][eq]; }
    template<NODE_TYPE NT> INLINE int MapENodeToEQC (int eq, size_t node_num) const 
    { return node_num - disp_eqc[NT][eq]; }
    template<NODE_TYPE NT> INLINE int MapCNodeToEQC (size_t node_num) const 
    { auto eq = GetEqcOfNode<NT>(node_num); return node_num - disp_cross[NT][eq]; }
    template<NODE_TYPE NT> INLINE int MapCNodeToEQC (int eq, size_t node_num) const 
    { return node_num - disp_cross[NT][eq]; }
    template<NODE_TYPE NT> INLINE int MapNodeToEQC (size_t node_num) const 
    { return (node_num < GetENN<NT>()) ? MapENodeToEQC<NT>(node_num) : MapCNodeToEQC<NT>(node_num); } 
    // template<NODE_TYPE NT> INLINE int MapNodeToEQC (size_t node_num) const 
    // { auto eq = GetEqcOfNode<NT>(node_num); return node_num - ( (node_num < disp_eqc[NT].Last()) ? disp_eqc[NT][eq] : disp_cross[NT][eq] ); }
    template<NODE_TYPE NT> INLINE int MapENodeFromEQC (size_t node_num, size_t eqc_num) const 
    { return node_num + disp_eqc[NT][eqc_num]; }
    template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE int MapCNodeFromEQC (size_t node_num, size_t eqc_num) const 
    { return node_num + disp_cross[NT][eqc_num]; }
    template<NODE_TYPE NT> INLINE int MapNodeFromEQC (size_t node_num, size_t eqc_num) const 
    { return (node_num < GetENN<NT>(eqc_num)) ? MapENodeFromEQC<NT>(node_num, eqc_num) : MapCNodeFromEQC<NT>(node_num, eqc_num); }
    const FlatTM GetBlock (size_t eqc_num) const;
    /** Creates new(!!) block-tm! **/
    // BlockTM* Map (CoarseMap & cmap) const;
    BlockTM* MapBTM (const PairWiseCoarseMap & cmap) const;
    shared_ptr<BlockTM> Map (const PairWiseCoarseMap & cmap) const
    { return shared_ptr<BlockTM>(MapBTM(cmap)); }
    // INLINE size_t GetNEqcs () const { return (eqc_h == nullptr) ? 0 : eqc_h->GetNEQCS(); }
  protected:
    using TopologicMesh::eqc_h;
    /** eqc-block-views of data **/
    // Array<shared_ptr<FlatTM>> mesh_blocks;
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
    friend std::ostream & operator<<(std::ostream &os, const BlockTM& p);
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
	    for (const auto& node : GetENodes<NT>(eqc)) lam(node);
	    if constexpr(NT!=NT_VERTEX) for (const auto& node : GetCNodes<NT>(eqc)) lam(node);
	  }
      }
      else { for (const auto & node : GetNodes<NT>()) lam(node); }
    }
    // Apply a lambda-function to each eqc/node - pair
    template<NODE_TYPE NT, class TX, class TLAM>
    INLINE void ApplyEQ (TX&& eqcs, TLAM lam, bool master_only = false) const {
      for (auto eqc : eqcs)
	if ( !master_only || eqc_h->IsMasterOfEQC(eqc) ) {
	  for (const auto& node : GetENodes<NT>(eqc)) lam(eqc, node);
	  if constexpr(NT!=NT_VERTEX) for (const auto& node : GetCNodes<NT>(eqc)) lam(eqc, node);
	}
    }
    template<NODE_TYPE NT, class TLAM>
    INLINE void ApplyEQ (TLAM lam, bool master_only = false) const {
      ApplyEQ<NT>(Range(GetNEqcs()), lam, master_only);
      // for (auto eqc : Range(GetNEqcs()))
      // 	if ( !master_only || eqc_h->IsMasterOfEQC(eqc) ) {
      // 	  for (const auto& node : GetENodes<NT>(eqc)) lam(eqc, node);
      // 	  if constexpr(NT!=NT_VERTEX) for (const auto& node : GetCNodes<NT>(eqc)) lam(eqc, node);
      // 	}
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
      Array<MPI_Request> req(nreq); nreq = 0;
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
	    { req[nreq++] = comm.ISend(exrow, p, MPI_TAG_AMG); }
	}
	else
	  { req[nreq++] = comm.IRecv(exrow, eqc_h->GetDistantProcs(k)[0], MPI_TAG_AMG); }
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
    INLINE void SetVs  (size_t annodes, TGET get_dps, TSET set_sort)
    {
      has_nodes[NT_VERTEX] = true;
      auto neqcs = eqc_h->GetNEQCS();
      const auto & eqc_h = *this->eqc_h;
      if (neqcs==0) { nnodes_glob[NT_VERTEX] = eqc_h.GetCommunicator().AllReduce(size_t(0), MPI_SUM); return; } // rank 0
      // if (neqcs==0) return; // rank 0
      static Timer t("BlockTM - Set Vertices");
      RegionTimer rt(t);
      this->nnodes[NT_VERTEX] = annodes;
      size_t nv = this->nnodes[NT_VERTEX];
      this->verts.SetSize(nv);
      for (auto k:Range(nv)) this->verts[k] = k;
      Array<size_t> & disp(this->disp_eqc[NT_VERTEX]);
      disp.SetSize(neqcs+1);
      Array<size_t> vcnt(neqcs+1);
      disp = 0; vcnt = 0;
      auto lam_veq = [&](auto fun) LAMBDA_INLINE {
	for (auto vnr : Range(nv)) {
	  auto dps = get_dps(vnr);
	  auto eqc = eqc_h.FindEQCWithDPs(dps);
	  fun(vnr,eqc);
	}
      };
      lam_veq([&](auto vnr, auto eqc) LAMBDA_INLINE {
	  disp[eqc+1]++;
	});
      disp[0] = 0;
      for(auto k:Range(size_t(1), neqcs)) {
	disp[k+1] += disp[k];
      }
      vcnt = disp;
      this->eqc_verts = FlatTable<AMG_Node<NT_VERTEX>>(neqcs, &(this->disp_eqc[NT_VERTEX][0]), &(this->verts[0]));
      lam_veq([&](auto vnr, auto eqc) LAMBDA_INLINE { set_sort(vnr, vcnt[eqc]++); });
      Array<int> v2eq;
      v2eq.SetSize(nv);
      size_t cnt = 0;
      for (auto k:Range(neqcs)) {
	auto d = disp[k];
	for (auto j:Range(disp[k+1]-disp[k]))
	  v2eq[cnt++] = d+j;
      }
      nnodes_eqc[NT_VERTEX].SetSize(neqcs);
      nnodes_cross[NT_VERTEX].SetSize(neqcs); nnodes_cross[NT_VERTEX] = 0;
      for (auto eqc : Range(neqcs))
	nnodes_eqc[NT_VERTEX][eqc] = disp_eqc[NT_VERTEX][eqc+1]-disp_eqc[NT_VERTEX][eqc];
      size_t nv_master = 0;
      for (auto eqc : Range(neqcs))
	if (eqc_h.IsMasterOfEQC(eqc))
	  nv_master += eqc_verts[eqc].Size();
      nnodes_glob[NT_VERTEX] = eqc_h.GetCommunicator().AllReduce(nv_master, MPI_SUM);
    } // SetVs
    INLINE void SetVs (size_t nnodes, Table<int> dist_procs, FlatArray<int> node_sort)
    { SetVs(nnodes, [&dist_procs](auto i){ return dist_procs[i]; }, [&node_sort](auto i, auto j){ node_sort[i] = j; }); }
    template<ngfem::NODE_TYPE NT, class TGET, class TSET, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE void SetNodes (size_t annodes, TGET get_node, TSET set_sort)
    {
      has_nodes[NT] = true;
      const auto & eqc_h = *this->eqc_h;
      auto neqcs = eqc_h.GetNEQCS();
      if (neqcs==0) { nnodes_glob[NT] = eqc_h.GetCommunicator().AllReduce(size_t(0), MPI_SUM); return; } // rank 0
      // if (neqcs==0) return; // rank 0
      string tname = "BlockTM - Set ";
      if constexpr(NT==NT_EDGE) tname += "Edges";
      else if constexpr(NT==NT_FACE) tname += "Faces";
      else if constexpr(NT==NT_CELL) tname += "Cells";
      else tname += " Unknown??";
      static Timer t(tname);
      RegionTimer rt(t);
      constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
      auto lam_neq = [&](auto fun_eqc, auto fun_cross) LAMBDA_INLINE {
	constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
	INT<NODE_SIZE,int> eqcs;
	for (auto node_num : Range(annodes)) {
	  auto vs = get_node(node_num);
	  auto eq_in = GetEqcOfNode<NT_VERTEX>(vs[0]);
	  auto eq_cut = eq_in;
	  for (auto i:Range(NODE_SIZE)) {
	    auto eq_v = GetEqcOfNode<NT_VERTEX>(vs[i]);
	    eqcs[i] = eqc_h.GetEQCID(eq_v);
	    eq_in = (eq_in==eq_v) ? eq_in : -1;
	    eq_cut = (eq_cut==eq_v) ? eq_cut : eqc_h.GetCommonEQC(eq_cut, eq_v);
	  }
	  AMG_CNode<NT> node = {{vs}, eqcs};
	  if (eq_in!=size_t(-1)) fun_eqc(node_num, node, eq_in);
	  else fun_cross(node_num, node, eq_cut);
	}
      };
      // create tent ex-nodes
      TableCreator<AMG_CNode<NT>> cten(neqcs);
      Array<size_t> &node_disp_eqc(this->disp_eqc[NT]), &node_disp_cross(this->disp_cross[NT]);
      node_disp_eqc.SetSize(neqcs+1); node_disp_cross.SetSize(neqcs+1);
      {
	constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
	auto add_node_eqc = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
	  if (eqc==0) { node_disp_eqc[1]++; return; }
	  for(auto i:Range(NODE_SIZE)) node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]);
	  cten.Add(eqc, node);
	};
	auto add_node_cross = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
	  if (eqc==0) { node_disp_cross[1]++; return; }
	  for(auto i:Range(NODE_SIZE)) node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]);
	  cten.Add(eqc, node);
	};
	for(;!cten.Done();cten++) {
	  node_disp_eqc = 0;
	  node_disp_cross = 0;
	  lam_neq(add_node_eqc, add_node_cross);
	}
      }
      // merge ex-nodes
      Table<AMG_CNode<NT>> tent_ex_nodes = cten.MoveTable();
      // TODO: do this for arbitrary nodes
      auto smaller = [&](const auto & a, const auto & b) -> bool {
	bool isina = a.eqc[0]==a.eqc[1];
	bool isinb = b.eqc[0]==b.eqc[1];
	if (isina && !isinb) return true;
	else if (!isina && isinb) return false;
	else if (isina && isinb) return a.v<b.v;
	else if (a.eqc[0]<b.eqc[0]) return true; else if (a.eqc[0]>b.eqc[0]) return false;
	else if (a.eqc[1]<b.eqc[1]) return true; else if (a.eqc[1]>b.eqc[1]) return false;
	else return a.v<b.v;
      };
      for (auto k:Range(size_t(1), neqcs))
	QuickSort(tent_ex_nodes[k], smaller);
      // cout << "tent_ex_nodes : " << endl << tent_ex_nodes << endl;
      auto merge_it = [&](auto & input) LAMBDA_INLINE { return merge_arrays(input, smaller); };
      auto ex_nodes = ReduceTable<AMG_CNode<NT>,AMG_CNode<NT>> (tent_ex_nodes, this->eqc_h, merge_it);
      // cout << "ex_nodes : " << endl << ex_nodes << endl;
      size_t tot_nnodes, tot_nnodes_eqc, tot_nnodes_cross;
      tot_nnodes_eqc = node_disp_eqc[1];
      tot_nnodes_cross = node_disp_cross[1];
      tot_nnodes = tot_nnodes_eqc+tot_nnodes_cross;
      for (auto k:Range(size_t(1), neqcs)) {
	auto row = ex_nodes[k];
	auto rows = row.Size();
	size_t n_in = 0;
	auto isin = [](auto X) {
	  auto eq = X.eqc[0];
	  for (int k = 1; k < NODE_SIZE; k++) if (X.eqc[k]!=eq) return false;
	  return true;
	};
	while(n_in<rows) { if(!isin(row[n_in])) break; n_in++; }
	tot_nnodes += rows;
	node_disp_eqc[k+1] = node_disp_eqc[k] + n_in;
	tot_nnodes_eqc += n_in;
	auto n_cross = rows-n_in;
	node_disp_cross[k+1] = node_disp_cross[k] + n_cross;
	tot_nnodes_cross += n_cross;
      }
      auto get_node_array = [&]()->Array<AMG_Node<NT>>&{
	if constexpr(NT==NT_EDGE) return this->edges;
	if constexpr(NT==NT_FACE) return this->faces;
	if constexpr(NT==NT_CELL) return this->cells;
      };
      this->nnodes[NT] = tot_nnodes;
      Array<AMG_Node<NT>> & nodes(get_node_array());
      nodes.SetSize(tot_nnodes);
      for (auto k:Range(neqcs+1)) {
	node_disp_cross[k] += tot_nnodes_eqc;
      }
      // add ex-nodes
      Array<size_t> nin(neqcs);
      nin[0] = -1; // just to ensure this blows up if we access here
      for (auto k:Range(size_t(1), neqcs)) {
	auto row = ex_nodes[k];
	auto n_in = node_disp_eqc[k+1]-node_disp_eqc[k];
	nin[k] = n_in;
	auto cr = 0;
	auto os_in = node_disp_eqc[k];
	for (auto j:Range(n_in)) {
	  auto id = os_in+j;
	  auto & node = nodes[id];
	  node.id = id;
	  const auto & exn_v = row[cr].v;
	  for (auto l:Range(NODE_SIZE)) node.v[l] = MapENodeFromEQC<NT_VERTEX>(exn_v[l], k);
	  cr++;
	}
	auto n_cross = row.Size()-n_in;
	auto os_cross = node_disp_cross[k];
	if (n_cross) for (auto j:Range(n_cross)) {
	    auto id = os_cross+j;
	    auto & node = nodes[id];
	    node.id = id;
	    auto & exn_v  = row[cr].v;
	    auto & exn_eq = row[cr].eqc;
	    for (auto l:Range(NODE_SIZE)) node.v[l] = MapENodeFromEQC<NT_VERTEX>(exn_v[l], eqc_h.GetEQCOfID(exn_eq[l]));
	    cr++;
	  }
      }
      // add local nodes + write node_map!
      size_t cnt0 = 0;
      auto add_node_eqc2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
	if (eqc==0) {
	  set_sort(node_num,id);
	  amg_nts::id_type id = cnt0;
	  nodes[id] = {{node.v}, id};
	  set_sort(node_num, id);
	  cnt0++;
	}
	else {
	  for(auto i:Range(NODE_SIZE)) node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]);
	  auto pos = ex_nodes[eqc].Pos(node);
	  auto loc_id = pos-nin[eqc];
	  set_sort(node_num,MapENodeFromEQC<NT>(loc_id, eqc));
	}
      };
      size_t cnt0c = tot_nnodes_eqc;
      auto add_node_cross2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
	if (eqc==0) {
	  set_sort(node_num,id);
	  amg_nts::id_type id = cnt0c;
	  nodes[id] = {{node.v}, id};
	  set_sort(node_num, id);
	  cnt0c++;
	}
	else {
	  for(auto i:Range(NODE_SIZE)) node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]);
	  auto pos = ex_nodes[eqc].Pos(node);
	  auto loc_id = pos-nin[eqc];
	  set_sort(node_num,MapCNodeFromEQC<NT>(loc_id, eqc));
	}
      };
      lam_neq(add_node_eqc2, add_node_cross2);
      for (auto k:Range(neqcs+1)) {
	node_disp_cross[k] -= tot_nnodes_eqc;
      }
      auto writeit = [neqcs, tot_nnodes, tot_nnodes_eqc](auto & _arr, auto & _disp_eqc, auto & _tab_eqc,
							 auto & _disp_cross, auto & _tab_cross) LAMBDA_INLINE
	{
	  if (_disp_eqc.Last() > _disp_eqc[0])
	    { _tab_eqc = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_eqc[0]), &(_arr[0])); }
	  else
	    { _tab_eqc = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_eqc[0]), nullptr); }
	  if (tot_nnodes > tot_nnodes_eqc)
	    { _tab_cross = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_cross[0]), &(_arr[tot_nnodes_eqc])); }
	  else
	    { _tab_cross = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_cross[0]), nullptr); }
	};
      if constexpr(NT==NT_EDGE) {
	  writeit(edges, disp_eqc[NT], eqc_edges, disp_cross[NT], cross_edges);
	}
      else if constexpr(NT==NT_FACE) {
	  writeit(faces, disp_eqc[NT], eqc_faces, disp_cross[NT], cross_faces);
	}
      nnodes_eqc[NT].SetSize(neqcs);
      nnodes_cross[NT].SetSize(neqcs);
      for (auto eqc : Range(neqcs)) {
	nnodes_eqc[NT][eqc] = disp_eqc[NT][eqc+1]-disp_eqc[NT][eqc];
	nnodes_cross[NT][eqc] = disp_cross[NT][eqc+1]-disp_cross[NT][eqc];
      }
      size_t nn_master = 0;
      for (auto eqc : Range(neqcs))
	if (eqc_h.IsMasterOfEQC(eqc))
	  nn_master += GetENN<NT>(eqc) + GetCNN<NT>(eqc);
      nnodes_glob[NT] = eqc_h.GetCommunicator().AllReduce(nn_master, MPI_SUM);
    } // SetNodes
    template<ngfem::NODE_TYPE NT, class TGET, class TSET, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE void SetNodes (Array<decltype(AMG_Node<NT>::v)> nodes, FlatArray<int> node_sort)
    { SetNodes (nodes.Size(), [&nodes](auto i) {return nodes[i]; }, [&node_sort](auto i, auto j){ node_sort[i] = j; }); }
  };

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
  
  template<class TMESH> class GridContractMap;

  /** Nodal data that can be attached to a Mesh to form an AlgebraicMesh **/
  template<NODE_TYPE NT, class T, class CRTP>
  class AttachedNodeData
  {
  protected:
    BlockTM * mesh;
    Array<T> data;
    PARALLEL_STATUS stat;
  public:
    static constexpr NODE_TYPE TNODE = NT;
    using TDATA = T;
    using TCRTP = CRTP;
    AttachedNodeData (Array<T> && _data, PARALLEL_STATUS _stat) : data(move(_data)), stat(_stat) {}
    virtual ~AttachedNodeData () { ; }
    void SetMesh (BlockTM * _mesh) { mesh = _mesh; }
    PARALLEL_STATUS GetParallelStatus () const { return stat; }
    void SetParallelStatus (PARALLEL_STATUS astat) { stat = astat; }
    Array<T> & GetModData () { return data; }
    FlatArray<T> Data () const { return data; }
    /**
       If this was template<class TMAP> void map_data (const TMAP & ...), derived classes would have to write
       template<> void map_data (const BaseCoarseMap ...), which I don't like because it is higher level code.
       
       TODO: would this be a cleaner ?
          here:    template<> void map_data (const BaseCoarseMap ...) { MapDataToCoarseLevel (map); }
	           void MapDataToCoarseLevel (const BaseCoarseMap ...) const = 0
	  in derived: void MapDataToCoarseLevel (const BaseCoarseMap ...) const override;
    **/
    template<class TMESH>
    INLINE void map_data (const GridContractMap<TMESH> & map, CRTP & cdata) const
    { map.template MapNodeData<NT, T>(data, stat, &cdata.data); cdata.SetParallelStatus(stat); }
    template<class TMESH>
    INLINE void map_data (const VDiscardMap<TMESH> & map, CRTP & cdata) const
    { map.template MapNodeData<NT, T>(data, stat, &cdata.data); cdata.SetParallelStatus(stat); }
    template<class TMAP>
    CRTP* Map (const TMAP & map) const
    {
      CRTP* cdata = new CRTP(Array<T>(map.template GetMappedNN<NT>()), NOT_PARALLEL);
      static_cast<const CRTP&>(*this).map_data(map, *cdata);
      return cdata;
    };
    virtual void Cumulate () const {
      if (stat == DISTRIBUTED) {
	AttachedNodeData<NT,T,CRTP>& nc_ref(const_cast<AttachedNodeData<NT,T,CRTP>&>(*this));
	// cout << "NT = " << NT << " dis data " << endl; prow2(data); cout << endl;
	mesh->AllreduceNodalData<NT, T> (nc_ref.data, [](auto & tab){return move(sum_table(tab)); }, false);
	// cout << "NT = " << NT << " cumul. data " << endl; prow2(data); cout << endl;
	nc_ref.stat = CUMULATED;
      }
    }
    virtual void Distribute () const {
      AttachedNodeData<NT,T,CRTP>& nc_ref(const_cast<AttachedNodeData<NT,T,CRTP>&>(*this));
      if (stat == CUMULATED) {
	const auto & eqc_h = *mesh->GetEQCHierarchy();
	for (auto eqc : Range(eqc_h.GetNEQCS())) {
	  if (eqc_h.IsMasterOfEQC(eqc)) continue;
	  auto block = mesh->GetBlock(eqc);
	  if constexpr(NT==NT_VERTEX) {
	      for (auto v : block.GetNodes<NT>()) data[v] = 0;
	    }
	  else {
	    for (auto node : block.GetNodes<NT>()) data[node.id] = 0;
	    for (auto node : block.GetCNodes<NT>()) data[node.id] = 0;
	  }
	}
	nc_ref.stat = DISTRIBUTED;
      }
    }
  };

  /** Mesh topology + various attached nodal data **/
  template<class TMESH> class GridMapStep;
  template<class TMESH> class CoarseMap;
  template<class... T>
  class BlockAlgMesh : public BlockTM
  {
  public:
    BlockAlgMesh (BlockTM && _mesh, std::tuple<T*...> _data)
      : BlockTM(move(_mesh)), node_data(_data)
    { std::apply([&](auto& ...x){(..., x->SetMesh(this));}, node_data); }
    BlockAlgMesh (BlockTM && _mesh, T*... _data)
      : BlockAlgMesh (move(_mesh), std::tuple<T*...>(_data...))
    { ; }
    ~BlockAlgMesh () { std::apply([](auto& ...x){ (...,delete x); }, node_data);}

    const std::tuple<T*...>& Data () const { return node_data; }
    const std::tuple<T*...>& AttachedData () const { return node_data; }

    INLINE void CumulateData () const { std::apply([&](auto& ...x){ (x->Cumulate(),...); }, node_data); }
    INLINE void DistributeData () const { std::apply([&](auto& ...x){ (x->Distribute(),...); }, node_data); }

    // template<class TMAP,
	     // typename T_ENABLE = typename std::enable_if<std::is_base_of<GridMapStep<BlockAlgMesh<T...>>, TMAP>::value==1>::type>
    template<class TMAP>
    std::tuple<T*...> MapData (const TMAP & map) const
    {
      static Timer t("BlockAlgMesh::MapData"); RegionTimer rt(t);
      return std::apply([&](auto& ...x){ return make_tuple<T*...>(x->Map(map)...); }, node_data);
    };

    template<class TMAP>
    std::tuple<T*...> AllocMappedData (const TMAP & map) const
    { return make_tuple<T*...>(new T(Array<typename T::TDATA>(map.template GetMappedNN<T::TNODE>()), DISTRIBUTED)...); }

    shared_ptr<BlockAlgMesh<T...>> Map (CoarseMap<BlockAlgMesh<T...>> & map) {
      static Timer t("BlockAlgMesh::Map (coarse)"); RegionTimer rt(t);
      shared_ptr<BlockTM> crs_btm(BlockTM::MapBTM(map));
      auto cdata = std::apply([&](auto& ...x) {
	  return make_tuple<T*...>(new T(Array<typename T::TDATA>(map.template GetMappedNN<T::TNODE>()), DISTRIBUTED)...);
	}, node_data);
      auto cmesh = make_shared<BlockAlgMesh<T...>> (move(*crs_btm), move(cdata));
      auto & cm_data = cmesh->Data();
      Iterate<count_ppack<T...>::value>([&](auto i){ get<i.value>(node_data)->map_data(map, *get<i.value>(cm_data)); });
      return cmesh;
    }

    void MapDataNoAlloc (AgglomerateCoarseMap<BlockAlgMesh<T...>> & map) {
      auto cmesh = static_pointer_cast<BlockAlgMesh<T...>> (map.GetMappedMesh());
      auto & cm_data = cmesh->Data();
      Iterate<count_ppack<T...>::value>([&](auto i){ get<i.value>(node_data)->map_data(map, *get<i.value>(cm_data)); });
    }
    
    template<typename... T2> friend std::ostream & operator<<(std::ostream &os, BlockAlgMesh<T2...> & m);
  protected:
    std::tuple<T*...> node_data;
  };

  template<NODE_TYPE NT, class T, class CRTP>
  std::ostream & operator<<(std::ostream &os, AttachedNodeData<NT,T, CRTP>& nd)
    {
      os << "Data for NT=" << NT;;
      os << ", status: " << nd.GetParallelStatus();
      os << ", data:" << endl; prow2(nd.Data()); os << endl;
      return os;
    }

  template<class... T>
  std::ostream & operator<<(std::ostream &os, BlockAlgMesh<T...> & m)
  {
    os << "BlockAlgMesh, BTM:" << endl;
    BlockTM& btm(m); os << btm << endl;
    os << "BlockAlgMesh, BTM, data:" << endl;
    apply([&](auto&&... x) {(os << ... << *x) << endl; }, m.Data());
    return os;
  }
  
} // namespace amg


#endif
