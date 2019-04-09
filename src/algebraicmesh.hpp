#ifndef FILE_ALGMESH
#define FILE_ALGMESH

namespace amg {

  // TODO: alg-mesh should not HAVE weights, only collapse-weights!
  
  class Coarsener;
  class CoarseMapping;
  class CoarseMap;
  class GridContractMap;
  class NodeDiscardMap;
  template<int D, int R> class edge_data;
  
  /** no +-operator for INT<k,double> implemented ... kill me now **/
  template<class T> void hacked_add(T & a, const T & b);

  template<>
  INLINE void hacked_add<double>(double & a, const double & b) { a+=b; }
  template<>
  INLINE void hacked_add<int>(int & a, const int & b) { a+=b; }
  template<>
  INLINE void hacked_add<size_t>(size_t & a, const size_t & b) { a+=b; }
  template<>
  INLINE void hacked_add<INT<2,double> >(INT<2,double> & a, const INT<2,double> & b)
  { for(int l=0; l<2; l++) a[l] += b[l];}
  template<>
  INLINE void hacked_add<INT<3,double> >(INT<3,double> & a, const INT<3,double> & b)
  { for(int l=0; l<3; l++) a[l] += b[l];}
  template<>
  INLINE void hacked_add<INT<4,double> >(INT<4,double> & a, const INT<4,double> & b)
  { for(int l=0; l<4; l++) a[l] += b[l];}
  template<>
  INLINE void hacked_add<INT<5,double> >(INT<5,double> & a, const INT<5,double> & b)
  { for(int l=0; l<5; l++) a[l] += b[l];}
  template<>
  INLINE void hacked_add<INT<6,double> >(INT<6,double> & a, const INT<6,double> & b)
  { for(int l=0; l<6; l++) a[l] += b[l];}
  template<>
  INLINE void hacked_add<Vec<3,double> >(Vec<3,double> & a, const Vec<3,double> & b)
  { a += b; }
  template<>
  INLINE void hacked_add<Vec<4,double> >(Vec<4,double> & a, const Vec<4,double> & b)
  { a += b; }




  class BaseAlgebraicMesh
  {
  public:

    BaseAlgebraicMesh(size_t nv, size_t ne, size_t nf);

    BaseAlgebraicMesh(BaseAlgebraicMesh && other) :
      n_verts(other.n_verts),
      n_edges(other.n_edges),
      n_faces(other.n_faces),
      verts(move(other.verts)),
      edges(move(other.edges)),
      wv(move(other.wv)),
      we(move(other.we)),
      cwv(move(other.cwv)),
      cwe(move(other.cwe)),
      v_free(move(other.v_free))
    { }
    
    BaseAlgebraicMesh():BaseAlgebraicMesh(0,0,0){}

    virtual ~BaseAlgebraicMesh(){}

    INLINE size_t NV() const { return n_verts; }
    INLINE size_t NE() const { return n_edges; }
    INLINE size_t NF() const { return n_faces; }

    INLINE size_t GNV() const { return n_verts_glob; }
    INLINE size_t GNE() const { return n_edges_glob; }
    INLINE size_t GNF() const { return n_faces_glob; }

    
    template<NODE_TYPE NT> INLINE size_t NN () const;
    template<NODE_TYPE NT> INLINE amg_ntype<NT> GetNode (size_t nn);
    template<NODE_TYPE NT> INLINE const Array<amg_ntype<NT>> & GetNodes () const;

    // template<NODE_TYPE NT> void SetCWs (Array<double> wts);
    // template<NODE_TYPE NT> INLINE double CW (size_t node_num);

    

    
    // INLINE const double GetECW(const vertex & vert) const { return cwe[vert.id]; }
    // INLINE const double GetECW(const edge & edge)   const { return cwe[edge.id]; }
    // INLINE const double GetECW(const face & face)   const { return cwe[face.id]; }

    INLINE const Array<vertex> & Vertices() const { return verts; }
    INLINE const Array<idedge> & Edges() const { return edges; }
    // INLINE Array<face> & Faces() { return faces; }
    
    INLINE const double GetVCW(size_t vert_nr) const { return cwv
	[vert_nr]; }
    INLINE const double GetECW(size_t edge_nr) const { return cwe[edge_nr]; }
    // INLINE const double GetFCW(size_t face_nr) const { return cwf[face_nr]; }
    
    INLINE const double GetECW(const idedge & edge) const { return cwe[edge.id]; }
    // INLINE const double GetFCW(const face & face) const { return cwf[face_nr]; }

    
    INLINE const double GetVW(size_t vert_nr) const { return wv[vert_nr]; }
    INLINE const double GetEW(size_t edge_nr) const { return we[edge_nr]; }
    INLINE const double GetEW(const idedge & edge) const { return we[edge.id]; }
    // INLINE const double GetFW(size_t edge_nr) const { return wf[edgenr]; }

    // INLINE FlatArray<vertex> E2V(const edge & edge) const { return e2v[edge.id]; }
    // INLINE FlatArray<vertex> F2E(const face & face) const { return e2v[face.id]; }

    INLINE bool IsVertexFree(size_t vert_nr) const {
      if(v_free==nullptr) return true;
      return v_free->Test(vert_nr);
    }
    
    virtual void SetFreeVertices(const shared_ptr<BitArray> & a_barray)
    {
      v_free = a_barray;
    }

    shared_ptr<BitArray> GetFreeVertices() const { return v_free; } 
    
    virtual shared_ptr<BaseAlgebraicMesh>
    GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap);
    virtual shared_ptr<BaseAlgebraicMesh>
    GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap,
			Array<double> && awv, Array<double> && awe);

    // new style maps
    virtual shared_ptr<BaseAlgebraicMesh> Map (CoarseMap & cmap);
    virtual shared_ptr<BaseAlgebraicMesh> Map (GridContractMap & map);
    virtual shared_ptr<BaseAlgebraicMesh> Map (NodeDiscardMap & map);

    void SetEdges (Array<idedge> && ae) { edges = std::move(ae); }
    
    virtual void SetVertexWeights(Array<double> && awv) { wv = awv; }
    const Array<double> & GetVertexWeights() { return wv; }

    virtual void SetEdgeWeights(Array<double> && awe) { we = awe; }
    const Array<double> & GetEdgeWeights() { return we; }

    // virtual void SetFaceWeights(Array<double> && awf) { wf = awf; }
    // const Array<double> & GetFaceWeights() { return wf; }

    virtual void SetVertexCollapseWeights(Array<double> && acwv) { cwv = acwv; }
    virtual void SetEdgeCollapseWeights(Array<double> && acwe) { cwe = acwe; }
    // virtual void SetFaceCollapseWeights(Array<double> && acwf) { cwf = acwf; }

    shared_ptr<SparseMatrix<double>> GetEdgeConnectivityMatrix() const {
      if(econ != nullptr) return econ;
      int nv = NV();
      Array<int> econ_s(nv);
      econ_s = 0;
      for(auto & edge: Edges())
	for(auto l:Range(2))
	  econ_s[edge.v[l]]++;
      Table<INT<2> > econ_i(econ_s);
      econ_s = 0;
      for(auto & edge: Edges())
	for(auto l:Range(2))
	  econ_i[edge.v[l]][econ_s[edge.v[l]]++] = INT<2>(edge.v[1-l], edge.id);
      for(auto row:econ_i)
	QuickSort(row, [](auto & a, auto & b) { return a[0]<b[0]; });
      auto econ = make_shared<SparseMatrix<double>>(econ_s, nv);
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

    
  protected:

    /** wether verts/edges/faces are built **/
    bool has_verts, has_edges, has_faces;
    
    /** nr of nodes **/
    int n_verts, n_edges, n_faces;
    int n_verts_glob, n_edges_glob, n_faces_glob;

    Array<vertex> verts;
    Array<idedge> edges;
    Array<face> faces;
    
    /** weights for nodes **/
    Array<double> wv, we, wf;

    /** collapse-weights for nodes **/
    Array<double> cwv, cwe, cwf;

    /** for dirichlet-conditions **/
    shared_ptr<BitArray> v_free = nullptr;

    /** edge-connectivity; build on demand **/
    shared_ptr<SparseMatrix<double>> econ = nullptr;
    
  };

  template<> INLINE size_t BaseAlgebraicMesh::NN<NT_VERTEX> () const { return n_verts; }
  template<> INLINE size_t BaseAlgebraicMesh::NN<NT_EDGE> () const { return n_edges; }
  template<> INLINE amg_ntype<NT_VERTEX> BaseAlgebraicMesh::GetNode<NT_VERTEX> (size_t nn) { return verts[nn]; }
  template<> INLINE amg_ntype<NT_EDGE> BaseAlgebraicMesh::GetNode<NT_EDGE> (size_t nn) { return edges[nn]; }
  template<> INLINE const Array<amg_ntype<NT_VERTEX>> & BaseAlgebraicMesh::GetNodes<NT_VERTEX> () const { return verts; }
  template<> INLINE const Array<amg_ntype<NT_EDGE>> & BaseAlgebraicMesh::GetNodes<NT_EDGE> () const { return edges; }
  

  /** Algebraic Mesh that does not own any memory **/
  class FlatAlgebraicMesh : public BaseAlgebraicMesh
  {
  public:
    FlatAlgebraicMesh(){}
    FlatAlgebraicMesh(FlatArray<vertex> av, FlatArray<idedge> ae, FlatArray<face> af,
		      FlatArray<double> awv, FlatArray<double> awe, FlatArray<double> awf,
		      FlatArray<double> acwv, FlatArray<double> acwe, FlatArray<double> acwf,
		      FlatArray<idedge> ce);
    ~FlatAlgebraicMesh(){}
    
    template<NODE_TYPE NT> INLINE size_t NN_Cross ();
    template<NODE_TYPE NT> INLINE amg_ntype<NT> GetNode_Cross (size_t nn);
    template<NODE_TYPE NT> INLINE const Array<amg_ntype<NT>> & GetNodes_cross ();
    
    
  private:
    size_t n_cross_edges;
    Array<idedge> cross_edges;
    friend std::ostream & operator<<(std::ostream &os, const FlatAlgebraicMesh& p);
  };

  template<> INLINE size_t FlatAlgebraicMesh::NN_Cross<NT_VERTEX> () { return 0; }
  template<> INLINE size_t FlatAlgebraicMesh::NN_Cross<NT_EDGE> () { return n_cross_edges; }
  template<> INLINE amg_ntype<NT_VERTEX> FlatAlgebraicMesh::GetNode_Cross<NT_VERTEX> (size_t nn) { throw Exception("cross vertices make no sense!"); }
  template<> INLINE const Array<amg_ntype<NT_VERTEX>> & FlatAlgebraicMesh::GetNodes_cross<NT_VERTEX> () { throw Exception("cross vertices make no sense!"); }
  template<> INLINE amg_ntype<NT_EDGE> FlatAlgebraicMesh::GetNode_Cross<NT_EDGE> (size_t nn) { return cross_edges[nn]; }
  template<> INLINE const Array<amg_ntype<NT_EDGE>> & FlatAlgebraicMesh::GetNodes_cross<NT_EDGE> () { return cross_edges; }

#ifdef PARALLEL

  /**
     MPI-parallel AlgMesh.
     Partitions Mesh into EQC-wise blocks.
     Can only handle verts & edges currently
     faces are not considered!

     vertex-wts are "distributed"
     edge-wts "cumulated"
     COLWts are all cumulated
   **/
  class BlockedAlgebraicMesh : public BaseAlgebraicMesh
  {
    // TODO: only for debug...
    friend class CoarseMapping;
  public:

    BlockedAlgebraicMesh(){}

    BlockedAlgebraicMesh( size_t nv, size_t ne, size_t nf,
			  Array<edge> && edges, Array<face> && faces,
			  Array<double> && awv, Array<double> && awe, Array<double> && awf,
			  Array<size_t> && eqc_v, const shared_ptr<EQCHierarchy> & aeqc_h);

    BlockedAlgebraicMesh( BlockedAlgebraicMesh && other ) :
      BaseAlgebraicMesh(move(other)),
      eqcs(move(other.eqcs)),
      eqc_h(other.eqc_h),
      c_eqc_h(other.eqc_h),
      eqc_verts(move(other.eqc_verts)),
      vertex_in_eqc(move(other.vertex_in_eqc)),
      g2l_v(move(other.g2l_v)),
      displs_pad_edges(move(other.displs_pad_edges)),
      displs_eqc_edges(move(other.displs_eqc_edges))
    {
      size_t nei = (eqcs.Size()==0) ? 0 : displs_eqc_edges.Last();
      size_t nec = n_edges - nei;
      eqc_edges = FlatTable<idedge>(eqcs.Size(), &(displs_eqc_edges[0]), (nei==0)?(idedge*)nullptr:&(edges[0]));
      padding_edges.Assign(FlatArray<idedge>(nec, (nec==0)?(idedge*)nullptr:&(edges[nei])));
      eqc_pad_edges = FlatTable<idedge>(eqcs.Size(), &(displs_pad_edges[0]), ((nec==0)?(idedge*)nullptr:&(edges[nei])));
      BuildMeshBlocks();
    }
    
    ~BlockedAlgebraicMesh(){}

    virtual shared_ptr<BaseAlgebraicMesh>
    GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap);
    virtual shared_ptr<BaseAlgebraicMesh>
    GenerateCoarseMesh (const shared_ptr<CoarseMapping> & cmap,
			Array<double> && awv, Array<double> && awe);
    virtual shared_ptr<BaseAlgebraicMesh> Map (CoarseMap & cmap);
    virtual shared_ptr<BaseAlgebraicMesh> Map (GridContractMap & map);
    virtual shared_ptr<BaseAlgebraicMesh> Map (NodeDiscardMap & map);
    
    /** Cumulates Weights! **/
    // TODO: does function-parameter make problems with inlining?? back to auto??
    template<class T>
    void CumulateVertexData (Array<T> & avdata,
			     std::function<void(T&, const T&)> lam_add
			     = [](T & a, const T & b){ hacked_add(a,b); }) const;
    // void CumulateVertexData (Array<T> & avdata) const;

    template<class T>
    void CumulateEdgeData (Array<T> & avdata,
			   std::function<void(T&, const T&)> lam_add
			   = [](T & a, const T & b){ hacked_add(a,b); }) const;
    // void CumulateFaceData (Array<double> & afdata) const { return; }

    friend shared_ptr<BlockedAlgebraicMesh>
    ContractionMap::Contract (const shared_ptr<const BlockedAlgebraicMesh> & alg_mesh);
    friend shared_ptr<BlockedAlgebraicMesh>
    ContractionMap::ContractV2 (const shared_ptr<const BlockedAlgebraicMesh> & alg_mesh);
  
    const shared_ptr<const EQCHierarchy> & GetEQCHierarchy () const { return c_eqc_h; }
    const Array<size_t> & GetEQCS() const { return eqcs; }

    const auto & DP() const { return displs_eqc_edges; }
    const auto & DPP() const { return displs_pad_edges; }

    // const shared_ptr<FlatAlgebraicMesh> & operator[] (size_t k) const
    // { return mesh_blocks[eqcs.Pos(k)]; }

    shared_ptr<FlatAlgebraicMesh> operator[] (size_t k) const
    { return mesh_blocks[eqcs.Pos(k)]; }

    // {
    //   mesh_blocks[k] = make_shared<FlatAlgebraicMesh>(eqc_verts[k], eqc_edges[k], FlatArray<face>(0, NULL),
    // 						      wv, we, wf, cwv, cwe, cwf, eqc_pad_edges[k]);
    //   mesh_blocks[k]->SetFreeVertices(v_free);
    //   return move(mesh_blocks[k]);
    // }

    
    INLINE FlatArray<idedge> GetPaddingEdges(size_t eqc) const
    { return eqc_pad_edges[eqcs.Pos(eqc)]; }
    INLINE FlatArray<idedge> GetPaddingEdges() const
    { return eqc_pad_edges.AsArray(); }

    // const shared_ptr<FlatAlgebraicMesh> & GetPadding ()
    // { return padding_block; }

    INLINE size_t GetEQCOfV(vertex v) const { return vertex_in_eqc[v]; }

    INLINE size_t MapVToEQC(vertex v) const { return g2l_v[v]; }

    // NOTE: i dont think i need this!
    // INLINE size_t MapEToEQC(idedge e) const { return g2l_e[e.id]; }
    // INLINE size_t MapEToEQC(size_t enr) const { return g2l_e[enr]; }

    void BuildMeshBlocks();

    virtual void SetFreeVertices(const shared_ptr<BitArray> & a_barray)
    {
      v_free = a_barray;
      for(auto & block:mesh_blocks)
	block->SetFreeVertices(a_barray);
    }

    const Table<vertex> & GetEqcVerts() const { return eqc_verts; }


    class Iterator
    {
      const BlockedAlgebraicMesh & mesh;
      size_t at_step;
    public:
      Iterator (const BlockedAlgebraicMesh & _mesh, size_t _at_step) : mesh(_mesh), at_step(_at_step) { ; }
      Iterator & operator++ () { ++at_step; return *this; }
      std::tuple<size_t, shared_ptr<FlatAlgebraicMesh>>  operator* ()
      {
	// auto algm = make_shared<FlatAlgebraicMesh>(mesh.eqc_verts[at_step], mesh.eqc_edges[at_step], FlatArray<face>(0, NULL),
	// 					   mesh.wv, mesh.we, mesh.wf, mesh.cwv, mesh.cwe, mesh.cwf, mesh.eqc_pad_edges[at_step]);
	// algm->SetFreeVertices(mesh.v_free);
	//return std::tuple<size_t, shared_ptr<FlatAlgebraicMesh>>(at_step, algm);
	return std::make_tuple(at_step, mesh[at_step]);
      }
      bool operator != (const Iterator  it2) { return at_step != it2.at_step; }
    };
    friend class Iterator;
    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, eqcs.Size()); }
    
    
  protected:

    shared_ptr<EQCHierarchy> eqc_h;
    shared_ptr<const EQCHierarchy> c_eqc_h;

    void BuildVertexTables(Array<double> && awv, Array<size_t> && eqc_v);
    void BuildEdgeTables(const Array<edge> & aedges, const Array<double> & awe);
    void BuildEdgeTables(Array<edge> && aedges, Array<double> && awe)
    {
      BuildEdgeTables(aedges, awe);
      return;
    }
    // void BuildFaceTables(Array<face> && faces, Array<double> && wf);

    Array<size_t> eqcs;
    // Array<EQC_TYPE> eqc_types;

    /** eqc-wise vertex data **/
    Table<vertex> eqc_verts;
    Array<size_t> vertex_in_eqc;
    Array<size_t> g2l_v;

    // Array<MPI_Datatype> v_types;
    
    /** eqc-block-views of data **/
    Array<shared_ptr<FlatAlgebraicMesh>> mesh_blocks;
    /** padding-block-view of data **/
    // shared_ptr<FlatAlgebraicMesh> padding_block;

    /** eqc-wise edge data **/
    // Array<int> edge_in_eqc;
    // Array<int> g2l_e;
    FlatTable<idedge> eqc_edges; /** ColWeights, need for coarsening **/
        
    /** padding data (only edges!) **/
    FlatArray<idedge> padding_edges;
    Array<vertex> padding_verts; // empty!
    
    /** cross/padding data 2.0!! **/
    FlatTable<idedge> eqc_pad_edges;

    Array<size_t> displs_pad_edges;
    Array<size_t> displs_eqc_edges;

    friend std::ostream & operator<<(std::ostream &os, const BlockedAlgebraicMesh& p);
    
    
  };

  
  INLINE Timer & HackedCVDTimer() {
    static Timer t("BlockedAlgebraicMesh :: CumulateVertexData");
    return t;
  }

  /** templated, implement in header... **/
  template<class T> void BlockedAlgebraicMesh ::
  CumulateVertexData (Array<T> & avdata, std::function<void(T&, const T&)> lam_add) const
  {
    RegionTimer rt(HackedCVDTimer());
    
    Array<size_t> dsz(eqc_verts.Size());
    for(auto k:Range(eqc_verts.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      dsz[k] = (dps.Size()) ? eqc_verts[k].Size() : 0;
    }

    Table<T> eqcdata(dsz);
    for(auto k:Range(eqc_verts.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      if(!dps.Size()) continue;
      for(auto j:Range(eqc_verts[k].Size())) {
	eqcdata[k][j] = avdata[eqc_verts[k][j]];
      }
    }
    
    auto RT = ReduceTable<T, T>(eqcdata, eqcs, eqc_h, [&lam_add](auto &in) {
	bool isok = true;
	if( (!in.Size()) || (!in[0].Size()))
	  return Array<T>(0);
	Array<T> out(in[0].Size());
	out = 0;
	for(auto k:Range(in.Size())) {
	  auto ink = in[k];
	  for(auto j:Range(in[k].Size()))
	    lam_add(out[j], ink[j]);
	}
	return out;
      });

    for(auto k:Range(eqc_verts.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      if(!dps.Size()) continue;
      for(auto j:Range(eqc_verts[k].Size()))
	avdata[eqc_verts[k][j]] = RT[k][j];
    }

    return;
  } // end BlockedAlgebraicMesh :: CumulateVertexData



  INLINE Timer & HackedCEDTimer() {
    static Timer t("BlockedAlgebraicMesh :: CumulateEdgeData");
    return t;
  }
  
  /** templated, implement in header... **/
  template<class T> void BlockedAlgebraicMesh ::
  CumulateEdgeData (Array<T> & avdata, std::function<void(T&, const T&)> lam_add) const
  {

    RegionTimer rt(HackedCEDTimer());
    
    Array<size_t> dsz(eqc_edges.Size());
    for(auto k:Range(eqc_edges.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      dsz[k] = (dps.Size()) ? eqc_edges[k].Size()+eqc_pad_edges[k].Size() : 0;
    }

    Table<T> eqcdata(dsz);
    for(auto k:Range(eqc_verts.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      if(!dps.Size()) continue;
      for(auto j:Range(eqc_edges[k].Size())) {
	eqcdata[k][j] = avdata[eqc_edges[k][j].id];
      }
      size_t os = eqc_edges[k].Size();
      for(auto j:Range(eqc_pad_edges[k].Size())) {
	eqcdata[k][os+j] = avdata[eqc_pad_edges[k][j].id];
      }
    }

    auto RT = ReduceTable<T, T>(eqcdata, eqcs, eqc_h, [&lam_add](auto &in) {
	bool isok = true;
	if( (!in.Size()) || (!in[0].Size()))
	  return Array<T>(0);
	// {
	//   cout << endl;
	//   cout << " got data: " << endl;
	//   for(auto k:Range(in.Size())) {
	//     if(in[k].Size() != in[0].Size()) cout << "INCONSISTENT SZ!!!" << endl;
	//     cout << k << " (" << in[k].Size() << "): ";
	//     auto row = in[k];
	//     for(auto v:row) cout << v << " "; // v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " || ";
	//     cout << endl;
	//   }
	//   cout << endl;
	//   cout << endl;
	// }
	Array<T> out(in[0].Size());
	out = 0;
	for(auto k:Range(in.Size())) {
	  auto ink = in[k];
	  for(auto j:Range(in[k].Size()))
	    lam_add(out[j], ink[j]);
	}
	// cout << "out: " << endl << out << endl;
	return out;
      });

    // for(auto k:Range(RT.Size())) {
    //   cout << k << ": ";amg_dumb_utils::prow(RT[k]);cout << endl;
    // }
    // cout << endl;
    // cout << endl << "CUMULATE EDGE DATA, RT: " << endl;
    // for(auto k:Range(RT.Size())) {
    //   cout << k << ": ";
    //   auto row = RT[k];
    //   for(auto v:row) cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " | ";
    //   cout << endl;
    // }
    // cout << endl;
    // cout << endl << "eqc_edges" << endl << eqc_edges  << endl;
    // cout << endl << "eqc_pad_edges" << endl << eqc_pad_edges  << endl;
    
    for(auto k:Range(eqc_verts.Size())) {
      auto dps = eqc_h->GetDistantProcs(eqcs[k]);
      if(!dps.Size()) continue;
      for(auto j:Range(eqc_edges[k].Size())) {
	avdata[eqc_edges[k][j].id] = RT[k][j];
      }
      size_t os = eqc_edges[k].Size();
      for(auto j:Range(eqc_pad_edges[k].Size())) {
	avdata[eqc_pad_edges[k][j].id] = RT[k][os+j];
      }
    }

    return;
  } // end BlockedAlgebraicMesh :: CumulateEdgeData



  
  

  

  std::ostream & operator<<(std::ostream &os, const BlockedAlgebraicMesh& p);

#endif  
  
  std::ostream & operator<<(std::ostream &os, const FlatAlgebraicMesh& p);
  
} // end namespace amg

#endif
