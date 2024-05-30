#ifndef FILE_STOKES_MAP_HPP
#define FILE_STOKES_MAP_HPP

#include <base_coarse.hpp>
#include <agglomerate_map.hpp>

#include <grid_contract.hpp>

namespace amg
{

template<class TMESH>
class StokesCoarseMap : public AgglomerateCoarseMap<TMESH>
{
protected:
  // if I wanted, I could have something like this too:
  // Array<int> loop_maps;

public:
  StokesCoarseMap(shared_ptr<TMESH> mesh)
    : AgglomerateCoarseMap<TMESH>(mesh)
  {}

  virtual ~StokesCoarseMap() = default;

  // ghost-verts
  void MapAdditionalDataA ();

  // loops
  void MapAdditionalDataB ();

protected:
  virtual void fillCoarseMesh () override
  {
    // topology, "normal" attached data
    AgglomerateCoarseMap<TMESH>::fillCoarseMesh();

    // as a hopefully temporary workaround, I split up MapAdditionalData into two parts -
    // one that can and must be done immediately, and one tht happens after the DOF-maps are complete
    // and we know the actual coarse flows.

    // map solid vertices
    MapAdditionalDataA();
  } // AgglomerateCoarseMap::fillCoarseMesh

  virtual shared_ptr<BaseCoarseMap> Concatenate (shared_ptr<BaseCoarseMap> right_map) override;

  virtual unique_ptr<BaseAgglomerator> createAgglomerator () override
  {
    /**
     *  Should never get here, but we need this class to be non-virtual
     *  so we can have the "Concatenate" method.
     *  This entire thing where the "Map" knows about the "Agglomerator"
     *  is kind of garbage, but whatever...
     */
    throw Exception("Called into StokesCoarseMap::createAgglomerator");
    return nullptr;
  }

}; // class StokesCoarseMap


template<class TMESH, class TAGGLOMERATOR>
class DiscreteStokesCoarseMap : public StokesCoarseMap<TMESH>
{
  // this is the trick:
  //   the T_AGG_MESH knows about attached data as "AttachedData",
  //   not as its concrete derived classes; that makes the Agglomerator
  //   independent of any "<xxx>_map.hpp" headers!
  using T_AGG_MESH = typename TMESH::T_MESH_W_DATA;
  using AGGLOMERATOR = TAGGLOMERATOR;

public:
  DiscreteStokesCoarseMap (shared_ptr<TMESH> _mesh)
    : StokesCoarseMap<TMESH>(_mesh)
  {
    ;
  } // AgglomerateCoarseMap(..)

  virtual ~DiscreteStokesCoarseMap () = default;

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
}; // class StokesCoarseMap


template<class TMESH>
class StokesContractMap : public AlgContractMap<TMESH>
{
  using PARENT = GridContractMap;
protected:
  Table<int> loop_maps, dofed_edge_maps;
public:

  using PARENT::IsMaster, PARENT::GetGroup, PARENT::GetNodeMap, PARENT::GetEQCHierarchy,
    PARENT::GetMappedEQCHierarchy, PARENT::GetProcMap, PARENT::GetMappedNN, PARENT::MapNodeData;

  StokesContractMap (Table<int> && groups, shared_ptr<TMESH> mesh)
    : AlgContractMap<TMESH>(std::move(groups), mesh, true) // need edge orientation here !
  {
    // trigger coarse mesh to be created at this point
    auto cm = this->GetMappedMesh();
  }

  ~StokesContractMap () { ; }

  INLINE FlatTable<int> GetLoopMaps () const { return loop_maps; }
  INLINE FlatTable<int> GetDofedEdgeMaps () const { return dofed_edge_maps; }

  void SetLoopMaps (Table<int> && amap) { loop_maps = std::move(amap); }
  void SetDofedEdgeMaps (Table<int> && amap) { dofed_edge_maps = std::move(amap); }

  void MapAdditionalData ();

  virtual void FillContractedMesh() override
  {
    AlgContractMap<TMESH>::FillContractedMesh();

    MapAdditionalData();
  }

  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

}; // class StokesContractMap


} // namespace amg

#endif // FILE_STOKES_MAP_HPP
