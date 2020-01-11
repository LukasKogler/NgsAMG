#ifndef FILE_AMGH1_HPP
#define FILE_AMGH1_HPP

namespace amg
{

  /** data which we attach to each vertex in the mesh **/
  class H1VData : public AttachedNodeData<NT_VERTEX, double, H1VData>
  {
  public:
    using AttachedNodeData<NT_VERTEX, double, H1VData>::map_data;
    H1VData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, double, H1VData>(move(_data), _stat) {}
    template<class TMAP> INLINE void map_data_impl (const TMAP & cmap, H1VData & ch1v) const;
    INLINE void map_data (const BaseCoarseMap & cmap, H1VData & ch1v) const
    { map_data_impl(cmap, ch1v); }
    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1VData & ch1v) const
    { map_data_impl(cmap, ch1v); }
  }; // class H1VData


  /** data which we attach to each edge in the mesh **/
  class H1EData : public AttachedNodeData<NT_EDGE, double, H1EData>
  {
  public:
    using AttachedNodeData<NT_EDGE, double, H1EData>::map_data;
    H1EData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, double, H1EData>(move(_data), _stat) {}
    template<class TMESH> INLINE void map_data_impl (const TMESH & cmap, H1EData & ch1e) const;
    INLINE void map_data (const BaseCoarseMap & cmap, H1EData & ch1e) const
    { map_data_impl(cmap, ch1e); }
    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1EData & ch1e) const
    { map_data_impl(cmap, ch1e); }
  }; // class H1EData

  using H1Mesh = BlockAlgMesh<H1VData, H1EData>;

  // template<int ADIM>
  class H1AMGFactory : public VertexAMGFactory<H1AMGFactory, H1Mesh, H1Energy<1>::DPV>
  {
  public:
    using BASE = VBasedAMGFactory<H1Factory>;
    using TMESH = H1Mesh<DIM>;
    using ENERGY = H1Energy<DIM>;
    using DIM = ADIM;
    using BS = ENERGY::DPV;
    using Options = BASE::Options;

  protected:
    using BASE::options;

  public:
    H1AMGFactory (shared_ptr<Options> _opts)
      : BASE(_opts)
    { ; }
  }; // class H1AMGFactory

} // namespace amg

#endif
