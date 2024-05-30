
/** 2D, NC-space, laplace + div div **/

#include <base.hpp>

#include <spw_agg.hpp>
#include <spw_agg_map.hpp>

#include "nc_stokes_mesh.hpp"

#include <spw_agg_impl.hpp>

// #include "nc_stokes_mesh_impl.hpp"

namespace amg
{

using TMESH   = NCGGStokesMesh<2>;
using TENERGY = NCGGStokesEnergy<2>;

using T_MESH_WITH_DATA = TMESH::T_MESH_W_DATA;

template class SPWAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;

} // namespace amg

