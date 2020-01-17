#ifndef FILE_AMGCRS
#define FILE_AMGCRS

namespace amg
{

  template<class TMESH> class CoarseMap;

  template<class TMESH> class CoarseningAlgorithm
  {
  public:
    virtual shared_ptr<CoarseMap<TMESH>> Coarsen (shared_ptr<TMESH> mesh) = 0;
  };

  class VWCoarseningData
  {
  public:
    struct Options
    {
      double min_ecw = 0.05, min_vcw = 0.5;
      shared_ptr<BitArray> free_verts = nullptr;
      Array<double> vcw, ecw;
    };
    VWCoarseningData (shared_ptr<Options> opts = nullptr)
    { options = (opts == nullptr) ? make_shared<Options>() : opts; }
    shared_ptr<Options> options;
    class CollapseTracker
    {
    public:
      CollapseTracker (size_t nv, size_t ne);
      ~CollapseTracker () { }
      INLINE bool IsVertexGrounded (AMG_Node<NT_VERTEX> v) { return vertex_ground.Test(v); }
      INLINE bool IsVertexCollapsed (AMG_Node<NT_VERTEX> v) { return vertex_collapse.Test(v); }
      INLINE bool IsEdgeCollapsed (const AMG_Node<NT_EDGE> & e) { return edge_collapse.Test(e.id); }
      INLINE const AMG_Node<NT_EDGE>& CollapsedEdge (const AMG_Node<NT_VERTEX> & v) { return *v2e[v]; }
      INLINE int GetNVertsCollapsed () { return vertex_collapse.NumSet(); }
      INLINE int GetNVertsGrounded () { return vertex_ground.NumSet(); }
      INLINE int GetNEdgesCollapsed () { return edge_collapse.NumSet(); }
      INLINE void FixEdge (const AMG_Node<NT_EDGE> & e) { edge_fixed.SetBit(e.id); }
      INLINE bool IsEdgeFixed (const AMG_Node<NT_EDGE> & e) { return edge_fixed.Test(e.id); }
      INLINE void FixVertex (const AMG_Node<NT_VERTEX> v) { vertex_fixed.SetBit(v); }
      INLINE bool IsVertexFixed (const AMG_Node<NT_VERTEX> v) { return vertex_fixed.Test(v); }
      template<NODE_TYPE NT> INLINE void ClearNode (const AMG_Node<NT> & node) {
	if constexpr(NT==NT_VERTEX) {
	    if (IsVertexCollapsed(node))
	      { UncollapseEdge(CollapsedEdge(node)); }
	    else { UngroundVertex(node); }
	  }
	else if constexpr(NT==NT_EDGE) {
	    if (IsEdgeCollapsed(node))
	      { UncollapseEdge(node); }
	    else if (IsVertexCollapsed(node.v[0]))
	      { UncollapseEdge(CollapsedEdge(node.v[0])); }
	    else if (IsVertexCollapsed(node.v[1]))
	      { UncollapseEdge(CollapsedEdge(node.v[1])); }
	  }
      }
      void CheckCollapseConsistency ();
      void PrintCollapse ()
      {
	cout << " ----- collapse report -----" << endl;
	cout << "nv " << vertex_ground.Size() << "  ne " << edge_collapse.Size() << endl;
	cout << "vert grnd " << vertex_ground.NumSet() << " / " << vertex_ground.Size() << endl;
	cout << "vert col " << vertex_collapse.NumSet() << " / " << vertex_collapse.Size() << endl;
	cout << "edge col " << edge_collapse.NumSet() << " / " << edge_collapse.Size() << endl;
	cout << " --- collapse report end ---" << endl;
	//cout << edge_collapse << endl << endl;
      }
      INLINE void GroundVertex (AMG_Node<NT_VERTEX> v)
      { vertex_ground.SetBit(v); }
      INLINE void UngroundVertex (AMG_Node<NT_VERTEX> v)
      { vertex_ground.Clear(v); }
      INLINE void CollapseEdge (const AMG_Node<NT_EDGE> & e)
      {
	if(edge_collapse.Test(e.id)) return;
#ifdef ASC_AMG_DEBUG
	if(edge_fixed.Test(e.id))
	  cout << "WARNING; TRYING TO COLLAPSE FIXED EDGE!!" << endl;      
	if(vertex_collapse.Test(e.v[0]) || vertex_collapse.Test(e.v[1]))
	  cout << "WARNING; TRYING TO COLLAPSE EDGE INTO OTHER EDGE, others fixed??" << endl;
	for(auto k:Range(2))
	  if(vertex_collapse.Test(e.v[k]))
	    if(v2e[e.v[k]]!=nullptr)
	      cout << "edge of " << k << ": " << *v2e[e.v[k]] << ", fixed? " << edge_fixed.Test(v2e[e.v[k]]->id) << " " << endl;
	    else
	      cout << "edge if " << k << " does not exist " << " " << endl;	  
#endif
	edge_collapse.SetBit(e.id);
	vertex_collapse.SetBit(e.v[0]);
	vertex_collapse.SetBit(e.v[1]);
	v2e[e.v[0]] = &e;
	v2e[e.v[1]] = &e;
      }
      INLINE void UncollapseEdge (const AMG_Node<NT_EDGE> & e)
      {
#ifdef ASC_AMG_DEBUG
	if(edge_fixed.Test(e.id)) cout << "WARNING; TRYING TO UN-COLLAPSE FIXED EDGE " << e << " !!" << endl;
#endif
	edge_collapse.Clear(e.id);
	vertex_collapse.Clear(e.v[0]);
	vertex_collapse.Clear(e.v[1]);
	v2e[e.v[0]] = nullptr;
	v2e[e.v[1]] = nullptr;
      }
      const Array<const AMG_Node<NT_EDGE>*> & GetVertex2EdgeMapping() const { return v2e; }
    private:
      BitArray vertex_ground;
      BitArray edge_collapse;
      BitArray vertex_collapse;
      BitArray edge_fixed; //edge_fixed == vertex_fixed[v1]|vertex_fixed[v2]?
      BitArray vertex_fixed;
      Array<const AMG_Node<NT_EDGE>*> v2e;
    }; // class CollapseTracker
  };

  template<class TMESH>
  class VWiseCoarsening : public VWCoarseningData, public CoarseningAlgorithm<TMESH> 
  {
  public:
    using Options = typename VWCoarseningData::Options;
    using CollapseTracker = typename VWCoarseningData::CollapseTracker;
    VWiseCoarsening (shared_ptr<Options> options = nullptr);
    ~VWiseCoarsening () { ; }
    virtual shared_ptr<CoarseMap<TMESH>> Coarsen (shared_ptr<TMESH> mesh)
    {
      CollapseTracker coll(mesh->template GetNN<NT_VERTEX>(), mesh->template GetNN<NT_EDGE>());
      Collapse(*mesh, coll);
      return make_shared<CoarseMap<TMESH>>(mesh, coll);
    }
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll) = 0;
  };

  template<class TMESH>
  class SeqVWC : public VWiseCoarsening<TMESH>
  {
  public:
    using Options = typename VWiseCoarsening<TMESH>::Options;
    using VWiseCoarsening<TMESH>::options;
    virtual shared_ptr<CoarseMap<TMESH>> Coarsen (shared_ptr<TMESH> mesh) override
    {
      static_assert(std::is_base_of<BlockTM, TMESH>::value==0, "Can not coarsen a parallel blocked mesh with SeqVWC!");
      return VWiseCoarsening<TMESH>::Coarsen(mesh);
    }
    using CollapseTracker = typename VWiseCoarsening<TMESH>::CollapseTracker;
    SeqVWC (shared_ptr<Options> opts);
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll) override;
  };
  
  template<class TMESH>
  class BlockVWC : public VWiseCoarsening<TMESH>
  {
    static_assert(std::is_base_of<BlockTM, TMESH>::value==1, "Can only use BlockVWC for Blocked meshes!");
  public:
    using Options = typename VWiseCoarsening<TMESH>::Options;
    using VWiseCoarsening<TMESH>::options;
    using CollapseTracker = typename VWiseCoarsening<TMESH>::CollapseTracker;
    BlockVWC (shared_ptr<Options> opts);
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll) override;
  };

  template<class TMESH>
  class HierarchicVWC : public VWiseCoarsening<TMESH>
  {
    static_assert(std::is_base_of<BlockTM, TMESH>::value==1, "Can only use BlockVWC for Blocked meshes!");
  public:
    struct Options : public VWiseCoarsening<BlockTM>::Options
    {
      bool pre_coll = false;
      bool post_coll = true;
    };
    // using Options = typename VWiseCoarsening<BlockTM>::Options;
    using CollapseTracker = typename VWiseCoarsening<BlockTM>::CollapseTracker;
    HierarchicVWC (shared_ptr<Options> opts);
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll) override;
    shared_ptr<HierarchicVWC::Options> options;
  };


  class BaseCoarseMap : public BaseGridMapStep
  {
  public:
    BaseCoarseMap (shared_ptr<TopologicMesh> mesh, shared_ptr<TopologicMesh> mapped_mesh = nullptr)
      : BaseGridMapStep(mesh, mapped_mesh)
    { ; } //{ NN = 0; mapped_NN = 0; }
    virtual ~BaseCoarseMap () { ; }
    template<NODE_TYPE NT> INLINE size_t GetNN () const { return NN[NT]; }
    template<NODE_TYPE NT> INLINE size_t GetMappedNN () const { return mapped_NN[NT]; }
    template<NODE_TYPE NT> INLINE FlatArray<int> GetMap () const { return node_maps[NT]; }
    shared_ptr<BaseCoarseMap> Concatenate (shared_ptr<BaseCoarseMap> right_map);
  protected:
    Array<Array<int>> node_maps = Array<Array<int>> (4);
    size_t NN[4] = {0,0,0,0};
    size_t mapped_NN[4] = {0,0,0,0};
  };
  
  
  class PairWiseCoarseMap : public BaseCoarseMap// workaround
  {
  public:
    PairWiseCoarseMap (shared_ptr<TopologicMesh> mesh, shared_ptr<TopologicMesh> mapped_mesh = nullptr)
      : BaseCoarseMap(mesh, mapped_mesh)
    { ; }
    template<NODE_TYPE NT> INLINE size_t CN_in_EQC (size_t eqc) const
    { return mapped_eqc_firsti[NT][eqc+1] -  mapped_eqc_firsti[NT][eqc]; }
    template<NODE_TYPE NT> INLINE size_t EQC_of_CN (size_t node_num) const {
      size_t eq = 0;
      while(mapped_eqc_firsti[NT][eq+1] <= node_num ) {
  	eq++;
      }
      return eq;
    }
    template<NODE_TYPE NT> INLINE int CN_to_EQC (size_t node_num) const
    { return node_num - mapped_eqc_firsti[NT][EQC_of_CN<NT>(node_num)]; }
    template<NODE_TYPE NT> INLINE int CN_of_EQC (size_t eqc, size_t loc_nr) const
    { return mapped_eqc_firsti[NT][eqc] +  loc_nr; }
    template<NODE_TYPE NT> FlatArray<size_t> GetMappedEqcFirsti () const { return mapped_eqc_firsti[NT]; }
    template<NODE_TYPE NT> FlatArray<size_t> GetMappedCrossFirsti () const { return mapped_cross_firsti[NT]; }
    INLINE FlatArray<decltype(AMG_Node<NT_EDGE>::v)> GetMappedEdges () const { return mapped_E; }
  protected:
    using BaseCoarseMap::node_maps, BaseCoarseMap::NN, BaseCoarseMap::mapped_NN;
    Array<Array<size_t> > mapped_eqc_firsti = Array<Array<size_t> >(4);
    Array<Array<size_t> > mapped_cross_firsti = Array<Array<size_t> >(4);
    Array<decltype(AMG_Node<NT_EDGE>::v)> mapped_E;
  };


  template<class TMESH>
  class CoarseMap : public PairWiseCoarseMap
  {
  public:
    CoarseMap (shared_ptr<TMESH> _mesh, VWCoarseningData::CollapseTracker &coll);
    ~CoarseMap () {}
  protected:
    using PairWiseCoarseMap::node_maps, PairWiseCoarseMap::NN, PairWiseCoarseMap::mapped_NN,
      PairWiseCoarseMap::mapped_eqc_firsti, PairWiseCoarseMap::mapped_cross_firsti, PairWiseCoarseMap::mapped_E;
    using BaseGridMapStep::mesh, BaseGridMapStep::mapped_mesh;
    void BuildVertexMap (VWCoarseningData::CollapseTracker& coll);
    void BuildEdgeMap (VWCoarseningData::CollapseTracker& coll);
  };


} // namespace amg

#endif
