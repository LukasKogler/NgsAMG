#ifndef FILE_AMGCRS2_HPP
#define FILE_AMGCRS2_HPP

namespace amg
{
  
  struct Agglomerate 
  {
    int id, ctr;
    ArrayMem<int, 30> mems;
    Agglomerate () { id = -2; ctr = -1; }
    Agglomerate (int c, int d) { mems.SetSize(1); mems[0] = c; id = d; ctr = c; }
    INLINE Agglomerate (Agglomerate && other) {
      id = move(other.id);
      ctr = move(other.ctr);
      mems = move(other.mems);
    }
    INLINE Agglomerate& operator = (Agglomerate && other) {
      id = move(other.id);
      ctr = move(other.ctr);
      mems = move(other.mems);
      return *this;
    }
    // we never want to copy/copy construct
    INLINE Agglomerate (const Agglomerate & other) = delete;
    INLINE Agglomerate& operator = (const Agglomerate & other) = delete;
    ~Agglomerate () { ; }
    INLINE int center () const { return ctr; }
    INLINE FlatArray<int> members () const { return mems; }
    INLINE void AddSort (int v) { insert_into_sorted_array(v, mems); }
    INLINE void Add (int v) { AddSort(v); }
  };
  INLINE std::ostream & operator<<(std::ostream &os, const Agglomerate& agg) {
    os << "agg. id " << agg.id << ", n members " << agg.mems.Size() << ", center " << agg.center() << ", all mems: ";
    prow2(agg.mems, os);
    return os;
  }

  template<class TMESH>
  class AgglomerateCoarseMap : public BaseCoarseMap, public GridMapStep<TMESH>
  {
    friend class BlockTM;

  public:

    AgglomerateCoarseMap (shared_ptr<TMESH> _mesh);

    virtual shared_ptr<TopologicMesh> GetMappedMesh () const override;

    shared_ptr<BitArray> GetAggCenter () const { return is_center; }

  protected:

    using BaseCoarseMap::node_maps, BaseCoarseMap::NN, BaseCoarseMap::mapped_NN;

    using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;

    virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) = 0;

    shared_ptr<BitArray> is_center;

    void BuildMappedMesh ();
    void MapVerts  (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
    void MapVerts2 (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
    void MapEdges (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
    void MapEdges2 (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
    void MapEdges_old (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);

    // public:
  //   template<NODE_TYPE NT> INLINE size_t GetNN () const { return mesh->template GetNN<NT>(); }
  //   template<NODE_TYPE NT> INLINE FlatArray<int> GetMap () const { return node_maps[NT]; }
  //   template<NODE_TYPE NT> INLINE size_t GetMappedNN () const { return mapped_nn[NT]; }

  };


  /**
     MIS(2) based agglomeration. Strength of connection is computed on the fly.
  **/

  template<class FACTORY>
  class Agglomerator : public AgglomerateCoarseMap<typename FACTORY::TMESH> 
  {

    using TMESH = typename FACTORY::TMESH;
    using TM = typename FACTORY::TM;
    using TVD = typename FACTORY::T_V_DATA;

  public:

    struct Options
    {
      double edge_thresh = 0.025;
      double vert_thresh = 0.0; // not considered!
      bool cw_geom = false;
      double dist2 = false;
      int min_new_aggs = 3;
    };

    Agglomerator (shared_ptr<typename FACTORY::TMESH> _mesh, shared_ptr<BitArray> _free_verts, Options && _opts);

  protected:

    using AgglomerateCoarseMap<typename FACTORY::TMESH>::mesh;

    shared_ptr<BitArray> free_verts;

    Options settings;

    virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;
  }; // Agglomerator


  // template<class TMESH>
  // class CoarseMap2 : public BaseCoarseMap, public GridMapStep<TMESH>
  // {
  // public:
  //   CoarseMap2 (shared_ptr<TMESH> _mesh, Table<int> aggs);
  // protected:
  //   using BaseCoarseMap::node_maps, BaseCoarseMap::NN, BaseCoarseMap::mapped_NN,
  //     BaseCoarseMap::mapped_eqc_firsti, BaseCoarseMap::mapped_cross_firsti, BaseCoarseMap::mapped_E;
  //   using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;
  //   void BuildVertexMap (VWCoarseningData::CollapseTracker& coll);
  //   void BuildEdgeMap (VWCoarseningData::CollapseTracker& coll);
  // }; // CoarseMap2



} // namespace amg

#endif