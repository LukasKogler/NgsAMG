#ifndef FILE_AMGH1_HPP
#define FILE_AMGH1_HPP

#include <alg_mesh.hpp>

#include <vertex_factory.hpp>
#include <vertex_factory_impl.hpp>

#include "h1_energy.hpp"
#include <utils_buffering.hpp>

namespace amg
{

// /** data which we attach to each vertex in the mesh **/
// class H1VData : public AttachedNodeData<NT_VERTEX, double, H1VData>
// {
// public:
//   using AttachedNodeData<NT_VERTEX, double, H1VData>::map_data;
//   H1VData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, double, H1VData>(std::move(_data), _stat) {}
//   template<class TMAP> INLINE void map_data_impl (const TMAP & cmap, H1VData & ch1v) const;
//   INLINE void map_data (const BaseCoarseMap & cmap, H1VData & ch1v) const
//   { map_data_impl(cmap, ch1v); }
//   template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1VData & ch1v) const
//   { map_data_impl(cmap, ch1v); }
// }; // class H1VData

/** data which we attach to each vertex in the mesh **/
// class H1VertexData
// {
// public:
//   unsigned cnt;
//   double wt;
//   H1VertexData (double _wt) : cnt(_wt), wt(_wt) { ; }
//   H1VertexData () : H1VertexData(0) { ; }
//   H1VertexData (int _cnt, double _wt) : cnt(_cnt), wt(_wt) { ; }
//   H1VertexData (H1VertexData && other) : cnt(std::move(other.cnt)), wt(std::move(other.wt)) { ; }
//   H1VertexData (H1VertexData && other) : cnt(other.cnt), wt(other.wt) { ; }
//   INLINE void operator  = (double x) { cnt = x; wt = x; }
//   INLINE void operator  = (const H1VertexData & other) { cnt = other.cnt; wt = other.wt;; }
//   INLINE void operator += (const H1VertexData & other) { cnt += other.cnt; wt += other.wt; }
//   INLINE void operator == (const H1VertexData & other) { return (cnt == other.cnt) && (wt == other.wt); }
// }; // class H1VertexData

class H1VData : public AttachedNodeData<NT_VERTEX, IVec<2, double>>
{
public:
  static constexpr NODE_TYPE TNODE = NT_VERTEX;
  // using AttachedNodeData<NT_VERTEX, IVec<2, double>, H1VData>::map_data;
  H1VData (Array<IVec<2, double>> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, IVec<2, double>>(std::move(_data), _stat) {}
  template<class TMAP> INLINE void map_data_impl (const TMAP & cmap, H1VData & ch1v) const;
  INLINE void map_data (const BaseCoarseMap & cmap, H1VData *ch1v) const
    { map_data_impl(cmap, *ch1v); }
  template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1VData *ch1v) const
    { map_data_impl(cmap, *ch1v); }
}; // class H1VData


/** data which we attach to each edge in the mesh **/
class H1EData : public AttachedNodeData<NT_EDGE, double>
{
public:
  static constexpr NODE_TYPE TNODE = NT_EDGE;
  // using AttachedNodeData<NT_EDGE, double, H1EData>::map_data;
  H1EData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, double>(std::move(_data), _stat) {}
  template<class TMESH> INLINE void map_data_impl (const TMESH & cmap, H1EData & ch1e) const;
  INLINE void map_data (const BaseCoarseMap & cmap, H1EData *ch1e) const
    { map_data_impl(cmap, *ch1e); }
  template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1EData *ch1e) const
    { map_data_impl(cmap, *ch1e); }
}; // class H1EData


using H1Mesh = BlockAlgMesh<H1VData, H1EData>;


// class H1GridContractMap : public GridContractMap
// {
// public:
//   H1GridContractMap (Table<int> && groups, shared_ptr<BlockTM> mesh, bool oriented = false)
//     : GridContractMap(std::move(groups), mesh, oriented)
//   {}

//   ~H1GridContractMap() = default;

// protected:
//   virtual shared_ptr<BlockTM> AllocateContractedMesh (shared_ptr<EQCHierarchy> _cEQCH) override
//   {
//     return make_shared<H1Mesh>(_cEQCH);
//   }

//   virtual void FillContractedMesh () override
//   {
//     auto f_mesh = dynamic_pointer_cast<H1Mesh>(GetMesh());
//     auto c_mesh = dynamic_pointer_cast<H1Mesh>(GetMappedMesh());

//     Assert(f_mesh != nullptr):
//     Assert(c_mesh != nullptr):

//     c_mesh->AllocateAttachedData();

//     std::apply( [&](auto& ..fdata, auto& ..cdata) {
//       auto fstat = fdata.GetParallelStatus();
//       MapNodeData<fdata::NT>(fdata, fstat, cdata);
//       cdata.SetParallelStatus(stat);
//     },
//     f_mesh->AttachedData(),
//     c_mesh->AttachedData());
//   }

// }; // class H1GridContractMap


template<int ADIM>
class H1AMGFactory : public VertexAMGFactory<H1Energy<ADIM, IVec<2,double>, double>, H1Mesh, ADIM>
{
public:
  static constexpr int DIM = ADIM;
  using ENERGY = H1Energy<DIM, IVec<2,double>, double>;
  using TMESH = H1Mesh;
  static constexpr int BS = ENERGY::DPV;
  using BASE = VertexAMGFactory<ENERGY, TMESH, BS>;
  using Options = typename BASE::Options;

protected:
  using BASE::options;

  virtual shared_ptr<GridContractMap> AllocateContractMap (Table<int> && groups, shared_ptr<TMESH> mesh) const override
  {
    return make_shared<AlgContractMap<H1Mesh>>(std::move(groups), mesh);
    // return make_shared<H1GridContractMap>(std::move(groups), mesh);
  }

public:
  H1AMGFactory (shared_ptr<Options> _opts)
    : BASE(_opts)
  { ; }
}; // class H1AMGFactory

} // namespace amg

#endif
