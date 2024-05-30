#ifndef FILE_AGGLOMERATE_MAP_HPP
#define FILE_AGGLOMERATE_MAP_HPP

#include "base_coarse.hpp"
#include "agglomerator.hpp"

namespace amg
{

// an agglomerate map that only knows about the topology and not the data of the mesh
class BaseAgglomerateCoarseMap : public BaseCoarseMap
{
  friend class BlockTM;

public:
  BaseAgglomerateCoarseMap (shared_ptr<TopologicMesh> mesh);

  void InitializeAgg (const AggOptions & opts, int level);

  void SetFreeVerts (shared_ptr<BitArray> &_free_verts);
  void SetSolidVerts (shared_ptr<BitArray> &_solid_verts);
  void SetFixedAggs (Table<int> && _fixed_aggs);
  void SetAllowedEdges (shared_ptr<BitArray> &_allowed_edges);

  // virtual void Finalize();

  virtual shared_ptr<TopologicMesh> GetMappedMesh () const override;

  virtual ~BaseAgglomerateCoarseMap () = default;

  shared_ptr<BitArray> GetAggCenter () const { return _is_center; }

protected:
  void BuildMappedMesh ();
  void MapVerts  (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
  void MapVerts_sv (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
  void MapEdges (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);

  BaseAgglomerator &getAgglomerator ()
  {
    return *_agglomerator;
  }

  virtual unique_ptr<BaseAgglomerator> createAgglomerator () = 0;

  virtual void allocateMappedMesh (shared_ptr<EQCHierarchy> &eqc_h) = 0;
  virtual void fillCoarseMesh () = 0;

  using BaseGridMapStep::mesh, BaseGridMapStep::mapped_mesh;
  using BaseCoarseMap::node_maps, BaseCoarseMap::NN, BaseCoarseMap::mapped_NN;

  unique_ptr<BaseAgglomerator> _agglomerator;

  /** settings **/
  bool print_vmap = false;     // debugging output
  shared_ptr<BitArray> _solid_verts = nullptr;
  shared_ptr<BitArray> _is_center = nullptr;
}; // class BaseAgglomerateCoarseMap


// this class brings in the data attached to the mesh
template<class TMESH> 
class AgglomerateCoarseMap : public BaseAgglomerateCoarseMap
{
public:
  AgglomerateCoarseMap (shared_ptr<TMESH> _mesh)
    : BaseAgglomerateCoarseMap(_mesh)
  {
    ;
  } // AgglomerateCoarseMap(..)

  virtual ~AgglomerateCoarseMap () = default;

protected:
  virtual void allocateMappedMesh (shared_ptr<EQCHierarchy> &eqc_h) override
  {
    mapped_mesh = make_shared<TMESH>(eqc_h);
  } // AgglomerateCoarseMap::allocateMappedMesh

  virtual void fillCoarseMesh () override
  {
    auto f_mesh = dynamic_pointer_cast<TMESH>(GetMesh());
    auto c_mesh = dynamic_pointer_cast<TMESH>(GetMappedMesh());

    assert(f_mesh != nullptr);
    assert(c_mesh != nullptr);

    c_mesh->AllocateAttachedData();

    auto fd = f_mesh->ModData();
    auto cd = c_mesh->ModData();

    ApplyComponentWise([&](auto &a, auto &b) {
      a->map_data(*this, b);
    }, fd, cd);

    // f_mesh->MapDataNoAlloc(*this);
  } // AgglomerateCoarseMap::fillCoarseMesh

}; // class AgglomerateCoarseMap


// finally, this class brings in the Agglomerator
template<class TMESH, class TAGGLOMERATOR>
class DiscreteAgglomerateCoarseMap : public AgglomerateCoarseMap<TMESH>
{
  // this is the trick:
  //   the T_AGG_MESH knows about attached data as "AttachedData",
  //   not as its concrete derived classes; that makes the Agglomerator
  //   independent of any "<xxx>_map.hpp" headers!
  using T_AGG_MESH = typename TMESH::T_MESH_W_DATA;
  using AGGLOMERATOR = TAGGLOMERATOR;

public:
  DiscreteAgglomerateCoarseMap (shared_ptr<TMESH> _mesh)
    : AgglomerateCoarseMap<TMESH>(_mesh)
  {
    ;
  } // AgglomerateCoarseMap(..)

  virtual ~DiscreteAgglomerateCoarseMap () = default;

  template<class TOPTS>
  INLINE void Initialize (const TOPTS & opts, int level)
  {
    this->InitializeAgg(opts, level);
    getConcreteAgglomerator().Initialize(opts, level);
  }

protected:
  virtual unique_ptr<BaseAgglomerator> createAgglomerator () override
  {
    auto tm = dynamic_pointer_cast<TMESH>(this->GetMesh());

    assert(tm != nullptr);

    return make_unique<AGGLOMERATOR>(tm);
  }

  AGGLOMERATOR &getConcreteAgglomerator()
  {
    auto agg = dynamic_cast<AGGLOMERATOR*>(&this->getAgglomerator());

    assert(agg != nullptr);

    return *agg;
  }

}; // class DiscreteAgglomerateCoarseMap


} // namespace amg

#endif // FILE_AGGLOMERATE_MAP_HPP
