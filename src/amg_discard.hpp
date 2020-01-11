#ifndef FILE_AMG_DISCARD_HPP
#define FILE_AMG_DISCARD_HPP

namespace amg
{

  class BaseDiscardMap : public BaseCoarseMap
  {
  protected:
    shared_ptr<BitArray> dropped_nodes[4];
    size_t dropped_NN[4];
  public:

    BaseDiscardMap ()
      : BaseCoarseMap()
    {
      for (auto k : Range(4)) {
	dropped_NN[k] = 0;
	dropped_nodes[k] = nullptr;
      }
    }

    ~BaseDiscardMap () { ; }

    template<NODE_TYPE NT> shared_ptr<BitArray> GetDroppedNodes () const
    { return dropped_nodes[NT]; }

    template<NODE_TYPE NT> size_t GetNDroppedNodes () const
    { return dropped_NN[NT]; }

  }; // class BaseDiscardMap

  template<class TMESH>
  class VDiscardMap : public BaseDiscardMap,
		      public GridMapStep<TMESH>
  {
  protected:
    size_t max_bs;
    using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;
    using BaseCoarseMap::NN, BaseCoarseMap::mapped_NN;

    shared_ptr<Table<size_t>> vertex_blocks;

  public:

    VDiscardMap (shared_ptr<TMESH> _mesh, size_t _max_bs = 5);

    shared_ptr<Table<size_t>> GetVertexBlocks () const { return vertex_blocks; }

    virtual shared_ptr<TopologicMesh> GetMappedMesh () const override;

  protected:

    void CalcDiscard ();
    void SetUpMM ();

  public:

    template<NODE_TYPE NT, typename T>
    void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const
    {
      if constexpr( (NT != NT_VERTEX) && (NT != NT_EDGE) )
		    { throw Exception("VDiscardMap does not have a map for that NT!!"); }
      auto & map = node_maps[NT];
      auto & cd (*cdata);
      cd.SetSize(mapped_NN[NT]);
      for (auto k : Range(map))
	if (map[k] != size_t(-1))
	  { cd[map[k]] = data[k]; }
    } // MapNodeData

  }; // class VDiscardMap

} // namespace amg

#endif
