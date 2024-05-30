
/** 2D, NC-space, laplace + div div **/

#include <base.hpp>

#include <spw_agg.hpp>
#include <spw_agg_map.hpp>

#include <mis_agg.hpp>
#include <mis_agg_map.hpp>

#include "nc_stokes_factory.hpp"

#include "nc_stokes_mesh.hpp"

#include "nc_stokes_pc.hpp"

#include <nodal_factory_impl.hpp>
#include <stokes_factory_impl.hpp>
#include "nc_stokes_factory_impl.hpp"
#include "nc_stokes_mesh_impl.hpp"

namespace amg
{

using TMESH   = NCGGStokesMesh<2>;
using TENERGY = NCGGStokesEnergy<2>;

/** COARSENING **/

using T_MESH_WITH_DATA = TMESH::T_MESH_W_DATA;

extern template class SPWAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;
extern template class MISAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;


/** FACTORY **/

template class   StokesAMGFactory<TMESH, TENERGY>;
template class NCStokesAMGFactory<TMESH, TENERGY>;

} // namespace amg
