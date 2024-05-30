
/** 2D, NC-space, laplace + div div **/

#include <base.hpp>

#include <spw_agg.hpp>
#include <spw_agg_map.hpp>

#include <mis_agg.hpp>
#include <mis_agg_map.hpp>

#include "nc_stokes_factory.hpp"

#include "nc_stokes_mesh.hpp"

#include "nc_stokes_pc.hpp"

// #include <nodal_factory_impl.hpp>
#include <stokes_factory_impl.hpp>
// #include <spw_agg_impl.hpp>
// #include <mis_agg_impl.hpp>
#include <stokes_pc_impl.hpp>

#include "nc_stokes_mesh_impl.hpp"

#include "nc_stokes_pc_impl.hpp"

namespace amg
{

using TMESH   = NCGGStokesMesh<2>;
using TENERGY = NCGGStokesEnergy<2>;

/** COARSENING **/

using T_MESH_WITH_DATA = TMESH::T_MESH_W_DATA;

extern template class SPWAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;
extern template class MISAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;


/** FACTORY **/

extern template class   StokesAMGFactory<TMESH, TENERGY>;
extern template class NCStokesAMGFactory<TMESH, TENERGY>;

using TFACTORY = NCStokesAMGFactory<TMESH, TENERGY>;
 
 
/** PRECONDITIONER **/

template class NCStokesAMGPC<TFACTORY>;
using STOKES_PC = NCStokesAMGPC<TFACTORY>;

RegisterPreconditioner<STOKES_PC> reg_stokes_gg_2d ("NgsAMG.stokes_gg_2d");

} // namespace amg


#include "python_stokes.hpp"

namespace amg
{
void ExportStokes_gg_2d (py::module & m) __attribute__((visibility("default")));
  
void ExportStokes_gg_2d (py::module & m)
{
  string stokes_desc = "Stokes preconditioner for grad-grad + div-div penalty in NC space";

  ExportStokesAMGClass<STOKES_PC> (m, "stokes_gg_2d", stokes_desc, [&](auto & pyclass) {});
} // ExportStokes_gg_2d

} // namespace amg
