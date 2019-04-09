#ifndef FILE_AMGCRS
#define FILE_AMGCRS

namespace amg
{

  template<class TMESH>
  class CoarseningAlgorithm
  {
  public:
    virtual shared_ptr<CoarseMap> Coarsen (shared_ptr<TMESH> mesh) = 0;
  };

  class VWCoarseningData
  {
  public:
    struct Options
    {
      double min_cw = 0.05;
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
      ~CollapseTracker() { }
      INLINE bool IsVertexGrounded (AMG_Node<NT_VERTEX> v) { return vertex_ground.Test(v); }
      INLINE bool IsVertexCollapsed (AMG_Node<NT_VERTEX> v) { return vertex_collapse.Test(v); }
      INLINE bool IsEdgeCollapsed (const AMG_Node<NT_EDGE> & e) { return edge_collapse.Test(e.id); }
      INLINE const AMG_Node<NT_EDGE>& CollapsedEdge (const AMG_Node<NT_VERTEX> & v) { return *v2e[v]; }
      INLINE int GetNVertsCollapsed () { return vertex_collapse.NumSet(); }
      INLINE int GetNVertsGrounded () { return vertex_ground.NumSet(); }
      INLINE int GetNEdgesCollapsed () { return edge_collapse.NumSet(); }
      INLINE void FixEdge (const AMG_Node<NT_EDGE> & e) { edge_fixed.Set(e.id); }
      INLINE bool IsEdgeFixed (const AMG_Node<NT_EDGE> & e) { return edge_fixed.Test(e.id); }
      INLINE void FixVertex (const AMG_Node<NT_VERTEX> v) { vertex_fixed.Set(v); }
      INLINE bool IsVertexFixed(const AMG_Node<NT_VERTEX> v) { return vertex_fixed.Test(v); }
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
      { vertex_ground.Set(v); }
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
	edge_collapse.Set(e.id);
	vertex_collapse.Set(e.v[0]);
	vertex_collapse.Set(e.v[1]);
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
  class VWiseCoarsening : public CoarseningAlgorithm<TMESH>, public VWCoarseningData
  {
  public:
    // using Options = typename VWCoarseningData::Options;
    using VWCoarseningData::Options;
    using VWCoarseningData::options;
    using CollapseTracker = typename VWCoarseningData::CollapseTracker;
    VWiseCoarsening (shared_ptr<Options> options = nullptr);
    ~VWiseCoarsening () { ; }
    virtual shared_ptr<CoarseMap> Coarsen (shared_ptr<TMESH> mesh);
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll) = 0;
  };

  template<class TMESH>
  class SeqVWC : public VWiseCoarsening<TMESH>
  {
  public:
    using Options = typename VWiseCoarsening<TMESH>::Options;
    using VWiseCoarsening<TMESH>::options;
    using CollapseTracker = typename VWiseCoarsening<TMESH>::CollapseTracker;
    SeqVWC (shared_ptr<Options> opts);
    virtual void Collapse (const TMESH & mesh, CollapseTracker & coll);
  };
  
  class BlockVWC : public VWiseCoarsening<BlockTM>
  {
  public:
    using Options = typename VWiseCoarsening<BlockTM>::Options;
    using VWiseCoarsening<BlockTM>::options;
    using CollapseTracker = typename VWiseCoarsening<BlockTM>::CollapseTracker;
    BlockVWC (shared_ptr<Options> opts);
    virtual void Collapse (const BlockTM & mesh, CollapseTracker & coll);
  };

  class HierarchicVWC : public VWiseCoarsening<BlockTM>
  {
  public:
    struct Options : public VWiseCoarsening<BlockTM>::Options
    {
      bool pre_coll = false;
      bool post_coll = true;
    };
    // using Options = typename VWiseCoarsening<BlockTM>::Options;
    using CollapseTracker = typename VWiseCoarsening<BlockTM>::CollapseTracker;
    HierarchicVWC (shared_ptr<Options> opts);
    virtual void Collapse (const BlockTM & mesh, CollapseTracker & coll);
    shared_ptr<HierarchicVWC::Options> options;
  };



  class CoarseMap : public GridMapStep<BlockTM>
  {
  public:
    CoarseMap (shared_ptr<BlockTM> _mesh, VWCoarseningData::CollapseTracker &coll);
    virtual ~CoarseMap () {}
    template<NODE_TYPE NT> INLINE size_t GetNN () const { return NN[NT]; }
    template<NODE_TYPE NT> INLINE size_t GetMappedNN () const { return mapped_NN[NT]; }
    template<NODE_TYPE NT> INLINE FlatArray<int> GetMap() const { return node_maps[NT]; }
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
    INLINE FlatArray<decltype(AMG_Node<NT_EDGE>::v)> GetMappedEdges() const { return mapped_E; }
    template<NODE_TYPE NT> FlatArray<size_t> GetMappedEqcFirsti () const { return mapped_eqc_firsti[NT]; }
    template<NODE_TYPE NT> FlatArray<size_t> GetMappedCrossFirsti () const { return mapped_cross_firsti[NT]; }
  protected:
    using GridMapStep<BlockTM>::mesh, GridMapStep<BlockTM>::mapped_mesh;
    void BuildVertexMap (VWCoarseningData::CollapseTracker& coll);
    void BuildEdgeMap (VWCoarseningData::CollapseTracker& coll);
    
    Array<Array<int>> node_maps = Array<Array<int>> (4);
    Array<size_t> NN = Array<size_t>(4);
    Array<size_t> mapped_NN  = Array<size_t>(4);
    Array<Array<size_t> > mapped_eqc_firsti = Array<Array<size_t> >(4);
    Array<Array<size_t> > mapped_cross_firsti = Array<Array<size_t> >(4);
    
    Array<decltype(AMG_Node<NT_EDGE>::v)> mapped_E;

    // Array<decltype(AMG_Node<NT_FACE>::v)> mapped_F;
    // Array<decltype(AMG_Node<NT_CELL>::v)> mapped_C;
    // template<NODE_TYPE NT, typename T2 = typename std::enable_if<NT!=NT_VERTEX>::type>
    // Array<decltype(AMG_Node<NT>::v)> & GetMappedNodes () {
    //   if constexpr(NT==NT_VERTEX) return mapped_E;
    //   if constexpr(NT==NT_FACE) return mapped_F;
    //   if constexpr(NT==NT_CELL) return mapped_C;
    // };
    
  };


#ifndef FILE_AMGCOARSEN_CPP
  extern template class SeqVWC<FlatTM>;
#endif

} // namespace amg

#endif
