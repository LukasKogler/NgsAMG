#define FILE_AMGH1_CPP
#define FILE_AMGH1_CPP_DIM1

#include <dof_map.hpp>
#include "h1.hpp"
#include <amg_pc_vertex.hpp>
#include <amg_pc_vertex_impl.hpp>

#include "h1_impl.hpp"
#include "h1_energy_impl.hpp"

#include <amg_register.hpp>
// #define AMG_EXTERN_TEMPLATES
// #include "amg_tcs.hpp"
// #undef AMG_EXTERN_TEMPLATES

#include "plate_test_agg_impl.hpp"

namespace amg
{

  using T_MESH           = H1AMGFactory<2>::TMESH;
  using T_ENERGY         = H1AMGFactory<2>::ENERGY;
  using T_MESH_WITH_DATA = typename T_MESH::T_MESH_W_DATA;

  extern template class SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;
  extern template class MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;

  extern template class DiscreteAgglomerateCoarseMap<T_MESH, SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;
  extern template class DiscreteAgglomerateCoarseMap<T_MESH, MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;

  template class H1AMGFactory<2>;
  template class VertexAMGPC<H1AMGFactory<2>>;
  // template class ElmatVAMG<H1AMGFactory<2>, double, double>;

  template class PlateTestAgglomerator<T_MESH_WITH_DATA>;
  template class DiscreteAgglomerateCoarseMap<T_MESH, PlateTestAgglomerator<T_MESH_WITH_DATA>>;

  // not compiling elmat for 2d
  using PCCBASE = VertexAMGPC<H1AMGFactory<2>>;
  // using PCC = ElmatVAMG<H1AMGFactory<2>, double, double>;

  // RegisterPreconditioner<PCC> register_h1amg_1d ("ATAmg.h1_scal");
  RegisterAMGSolver<PCCBASE> register_h1amg_2d ("h1_2d");

} // namespace amg

