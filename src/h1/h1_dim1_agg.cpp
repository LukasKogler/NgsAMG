#include "h1.hpp"
#include <amg_pc_vertex.hpp>

#include "h1_impl.hpp"
#include "h1_energy_impl.hpp"
// #include <amg_pc_vertex_impl.hpp>

#include <agglomerate_map.hpp>

#include <spw_agg.hpp>
#include <spw_agg_impl.hpp>

#include <spw_agg_map.hpp>

#include <mis_agg.hpp>
#include <mis_agg_impl.hpp>

#include <mis_agg_map.hpp>

namespace amg
{
  using T_MESH           = H1AMGFactory<1>::TMESH;
  using T_ENERGY         = H1AMGFactory<1>::ENERGY;
  using T_MESH_WITH_DATA = typename T_MESH::T_MESH_W_DATA;

  template class SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;
  template class MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;

  template class DiscreteAgglomerateCoarseMap<T_MESH, SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;
  template class DiscreteAgglomerateCoarseMap<T_MESH, MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;

  // template class SPWAgglomerateCoarseMap<H1AMGFactory<1>::TMESH, H1AMGFactory<1>::ENERGY>;
  // template class MISAgglomerateCoarseMap<H1AMGFactory<1>::TMESH, H1AMGFactory<1>::ENERGY>;
} // namespace amg
