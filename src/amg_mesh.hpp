#ifndef FILE_AMGMESH
#define FILE_AMGMESH


namespace amg
{

  class CoarseMap;
  
  // Only topology!
  class TopologicMesh
  {
    friend class NgsAMG_Comm;
  public:
    TopologicMesh (size_t nv = 0, size_t ne = 0, size_t nf = 0, size_t nc = 0);
    TopologicMesh (TopologicMesh && other);
    virtual ~TopologicMesh() { ; }
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
    virtual shared_ptr<SparseMatrix<double>> GetEdgeCM() const;
    friend std::ostream & operator<<(std::ostream &os, const TopologicMesh& p);
  protected:
    bool has_nodes[4] = {false, false, false, false};
    size_t nnodes[4];
    size_t nnodes_glob[4];
    Array<AMG_Node<NT_VERTEX>> verts; // can be empty (when its just)
    Array<AMG_Node<NT_EDGE>> edges;
    Array<AMG_Node<NT_FACE>> faces;
    Array<AMG_Node<NT_CELL>> cells;
    shared_ptr<SparseMatrix<double>> econ = nullptr;
  }; // class TopologicMesh

  
  class FlatTM : public TopologicMesh
  {
  public:
    FlatTM () { ; }
    FlatTM (FlatArray<AMG_Node<NT_VERTEX>> av, FlatArray<AMG_Node<NT_EDGE>> ae,
	    FlatArray<AMG_Node<NT_FACE>> af  , FlatArray<AMG_Node<NT_CELL>> ac,
	    FlatArray<AMG_Node<NT_EDGE>> ace , FlatArray<AMG_Node<NT_FACE>> acf);
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
    friend class NgsAMG_Comm;
  public:
    BlockTM (shared_ptr<EQCHierarchy> _eqc_h);
    BlockTM (BlockTM && other);
    BlockTM ();
    ~BlockTM () { ; }
    shared_ptr<EQCHierarchy> GetEQCHierarchy () const { return eqc_h; }
    INLINE size_t GetNEqcs () const { return eqc_verts.Size(); }
    template<NODE_TYPE NT> INLINE size_t GetENN (size_t eqc_num) const
    { return (eqc_num==size_t(-1)) ? disp_eqc[NT].Last() : nnodes_eqc[NT][eqc_num]; }
    template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetENodes (size_t eqc_num) const;
    template<NODE_TYPE NT> INLINE size_t GetCNN (size_t eqc_num) const
    { return (eqc_num==size_t(-1)) ? disp_cross[NT].Last()-disp_eqc[NT].Last() : nnodes_cross[NT][eqc_num]; }
    template<NODE_TYPE NT> INLINE FlatArray<AMG_Node<NT>> GetCNodes (size_t eqc_num) const;
    template<NODE_TYPE NT> INLINE size_t GetEqcOfNode (size_t node_num) const {
      size_t eq = 0; // TODO: binary search??
      if (node_num < disp_eqc[NT].Last()) while(disp_eqc[NT][eq+1] <= node_num ) eq++;
      else while(disp_cross[NT][eq+1] <= node_num ) eq++;
      return eq;
    }
    template<NODE_TYPE NT> INLINE int MapNodeToEQC (size_t node_num) const 
    { auto eq = GetEqcOfNode<NT>(node_num); return node_num - ( (node_num < disp_eqc[NT].Last()) ? disp_eqc[NT][eq] : disp_cross[NT][eq] ); }
    template<NODE_TYPE NT> INLINE int MapENodeFromEQC (size_t node_num, size_t eqc_num) const 
    { return node_num + disp_eqc[NT][eqc_num]; }
    template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    INLINE int MapCNodeFromEQC (size_t node_num, size_t eqc_num) const 
    { return node_num + disp_cross[NT][eqc_num]; }
    const FlatTM GetBlock (size_t eqc_num) const;
    /** Creates new(!!) block-tm! **/
    // BlockTM* Map (CoarseMap & cmap) const;
    virtual BlockTM* Map (const CoarseMap & cmap) const;
  protected:
    shared_ptr<EQCHierarchy> eqc_h;
    /** eqc-block-views of data **/
    // Array<shared_ptr<FlatTM>> mesh_blocks;
    /** eqc-wise data **/
    Array<size_t> nnodes_eqc[4];
    FlatTable<AMG_Node<NT_VERTEX>> eqc_verts;
    FlatTable<AMG_Node<NT_EDGE>> eqc_edges;
    FlatTable<AMG_Node<NT_FACE>> eqc_faces;
    Array<Array<size_t>> disp_eqc; // displacement in node array
    /** padding data **/
    Array<size_t> nnodes_cross[4];
    /** cross data **/
    FlatTable<AMG_Node<NT_EDGE>> cross_edges;
    FlatTable<AMG_Node<NT_FACE>> cross_faces;
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
    // ugly stuff
    template<NODE_TYPE NT, class T, class TRED>
    void AllreduceNodalData (Array<T> & avdata, TRED red, bool apply_loc = false) const {
      int neqcs = eqc_h->GetNEQCS();
      if (neqcs == 0) return; // nothing to do!
      if (neqcs == 1 && !apply_loc) return; // nothing to do!
      Array<int> rowcnt(neqcs);
      for (auto k : Range(neqcs)) rowcnt[k] = nnodes_eqc[NT][k] + nnodes_cross[NT][k];
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
		  else for (auto l:Range(nodes.Size())) row[C++] = avdata[nodes[l].id];
		},
		data);
      Table<T> reduced_data = ReduceTable<T,T,TRED>(data, eqc_h, red);
      loop_eqcs([&](auto nodes, auto row)
		{ if constexpr(NT==NT_VERTEX) for (auto l:Range(nodes.Size())) avdata[nodes[l]] = row[C++];
		  else for (auto l:Range(nodes.Size())) avdata[nodes[l].id] = row[C++]; },
		reduced_data);
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
      auto lam_veq = [&](auto fun) {
	for (auto vnr : Range(nv)) {
	  auto dps = get_dps(vnr);
	  auto eqc = eqc_h.FindEQCWithDPs(dps);
	  fun(vnr,eqc);
	}
      };
      lam_veq([&](auto vnr, auto eqc) {
	  disp[eqc+1]++;
	});
      disp[0] = 0;
      for(auto k:Range(size_t(1), neqcs)) {
	disp[k+1] += disp[k];
      }
      vcnt = disp;
      this->eqc_verts = FlatTable<AMG_Node<NT_VERTEX>>(neqcs, &(this->disp_eqc[NT_VERTEX][0]), &(this->verts[0]));
      lam_veq([&](auto vnr, auto eqc) { set_sort(vnr, vcnt[eqc]++); });
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
      cout << "nnodes loc, master, glob: " << nnodes[NT_VERTEX] << " " << nv_master << " " << nnodes_glob[NT_VERTEX] << endl;
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
      auto lam_neq = [&](auto fun_eqc, auto fun_cross) {
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
	auto add_node_eqc = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) {
	  if (eqc==0) { node_disp_eqc[1]++; return; }
	  for(auto i:Range(NODE_SIZE)) node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]);
	  cten.Add(eqc, node);
	};
	auto add_node_cross = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) {
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
	else if (isinb && !isina) return false;
	else if (isina && isinb) return a.v<b.v;
	else return a.eqc<b.eqc;
      };
      for (auto k:Range(size_t(1), neqcs))
	QuickSort(tent_ex_nodes[k], smaller);
      auto merge_it = [&](auto & input) { return merge_arrays(input, smaller); };
      auto ex_nodes = ReduceTable<AMG_CNode<NT>,AMG_CNode<NT>> (tent_ex_nodes, this->eqc_h, merge_it);
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
      auto add_node_eqc2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) {
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
      auto add_node_cross2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) {
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
      auto writeit = [neqcs, tot_nnodes_eqc](auto & _arr, auto & _disp_eqc, auto & _tab_eqc,
					     auto & _disp_cross, auto & _tab_cross)
	{
	  _tab_eqc = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_eqc[0]), &(_arr[0]));
	  _tab_cross = FlatTable<AMG_Node<NT>>(neqcs, &(_disp_cross[0]), &(_arr[tot_nnodes_eqc]));
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
  
  
  /** Nodal data that can be attached to a Mesh to form an AlgebraicMesh **/
  template<NODE_TYPE NT, class T, class CRTP>
  class AttachedNodeData
  {
  protected:
    BlockTM * mesh;
    Array<T> data;
    PARALLEL_STATUS stat;
  public:
    AttachedNodeData (Array<T> && _data, PARALLEL_STATUS _stat) : data(move(_data)), stat(_stat) {}
    void SetMesh (BlockTM * _mesh) { mesh = _mesh; }
    PARALLEL_STATUS GetParallelStatus () const { return stat; }
    FlatArray<T> Data () const { return data; }
    // INLINE void map_data (const CoarseMap & cmap, FlatArray<T> fdata, FlatArray<T> cdata); 
    CRTP* Map (const CoarseMap & map) const
    {
      Array<T> cdata;
      auto cstat = static_cast<const CRTP&>(*this).map_data(map, cdata);
      auto cm_wd = new CRTP(move(cdata), cstat);
      return cm_wd;
    };
    virtual void Cumulate() const {
      if (stat == DISTRIBUTED) {
	AttachedNodeData<NT,T,CRTP>& nc_ref(const_cast<AttachedNodeData<NT,T,CRTP>&>(*this));
	mesh->AllreduceNodalData<NT, double> (nc_ref.data, [](auto & tab){return move(sum_table(tab)); });
	nc_ref.stat = CUMULATED;
      }
    }
    virtual void Distribute() const {
      AttachedNodeData<NT,T,CRTP>& nc_ref(const_cast<AttachedNodeData<NT,T,CRTP>&>(*this));
      if (stat == CUMULATED) {
	const auto eqc_h = *mesh->GetEQCHierarchy();
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
  template<class... T>
  class BlockAlgMesh : public BlockTM
  {
  public:
    BlockAlgMesh (BlockTM && _mesh, T*... _data)
      : BlockTM(move(_mesh)), node_data(_data...)
    { std::apply([&](auto& ...x){(..., x->SetMesh(this));}, node_data); }
    BlockAlgMesh<T...>* MapBALG (const CoarseMap & cmap) const
    { return std::apply([&](auto&& ...x){ return new BlockAlgMesh<T...>(move(*BlockTM::Map(cmap)), x->Map(cmap)...); }, node_data); };
    virtual BlockTM* Map (const CoarseMap & cmap) const override { return MapBALG(cmap); }
    const std::tuple<T*...>& Data () const { return node_data; }
    template<typename... T2> friend std::ostream & operator<<(std::ostream &os, BlockAlgMesh<T2...> & m);
    ~BlockAlgMesh () { std::apply([](auto& ...x){ (...,delete x); }, node_data);}
  protected:
    std::tuple<T*...> node_data;
  };

  template<NODE_TYPE NT, class T, class CRTP>
  std::ostream & operator<<(std::ostream &os, AttachedNodeData<NT,T, CRTP>& nd)
    {
      os << endl << "Data for NT : " << NT << endl;
      os << endl << "Status: " << nd.GetParallelStatus() << endl;
      os << endl << "Data: "; prow(nd.Data()); os << endl;
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
