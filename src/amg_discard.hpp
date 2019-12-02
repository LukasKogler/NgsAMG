#ifndef FILE_AMG_DISCARD_HPP
#define FILE_AMG_DISCARD_HPP

namespace amg
{

  template<class TMESH>
  class VDiscardMap : public GridMapStep<TMESH>
  {
  public:

    VDiscardMap (shared_ptr<TMESH> _mesh, size_t _max_bs = 5);

    shared_ptr<BitArray> GetDroppedVerts () const { return dropped_vert; }

    shared_ptr<Table<size_t>> GetVertexBlocks () const { return vertex_blocks; }

    // TODO: mapped_nnodes[NT_VERTEX] shoulde be comnputable much easier!
    template<NODE_TYPE NT> size_t GetMappedNN () const { return mapped_nnodes[NT]; }
    template<NODE_TYPE NT> FlatArray<int> GetNodeMap () const { return node_maps[NT]; }
    template<NODE_TYPE NT> FlatArray<int> GetMap () const { return node_maps[NT]; }

    size_t GetNDroppedVerts () const { return dropped_vert->NumSet(); }

    virtual shared_ptr<TopologicMesh> GetMappedMesh () const override;

  protected:

    void CalcDiscard ();
    void SetUpMM ();

    size_t max_bs;
    using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;

    shared_ptr<BitArray> dropped_vert;
    shared_ptr<Table<size_t>> vertex_blocks;

    Array<size_t> mapped_nnodes;
    Array<Array<int>> node_maps;

  public:

    template<NODE_TYPE NT, typename T>
    void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const
    {
      if constexpr( (NT != NT_VERTEX) && (NT != NT_EDGE) )
		    { throw Exception("VDiscardMap does not have a map for that NT!!"); }
      const auto mapped_nn = mapped_nnodes[NT];
      auto & map = node_maps[NT];
      auto & cd (*cdata);
      cd.SetSize(mapped_nn);
      for (auto k : Range(map))
	if (map[k] != size_t(-1))
	  { cd[map[k]] = data[k]; }
    } // MapNodeData

  }; // class VDiscardMap

} // namespace amg

#endif
