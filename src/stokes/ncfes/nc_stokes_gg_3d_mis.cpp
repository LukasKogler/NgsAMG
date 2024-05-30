
/** 2D, NC-space, laplace + div div **/

#include <base.hpp>

#include <mis_agg.hpp>

#include "nc_stokes_mesh.hpp"

#include <mis_agg_impl.hpp>

// #include "nc_stokes_mesh_impl.hpp"

namespace amg
{

using TMESH   = NCGGStokesMesh<3>;
using TENERGY = NCGGStokesEnergy<3>;

using T_MESH_WITH_DATA = TMESH::T_MESH_W_DATA;

template class MISAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;

} // namespace amg

