#ifdef ELASTICITY

#define FILE_SPWAGG_EL3D_CPP

#include "elasticity_mesh.hpp"

#include <amg_pc_vertex.hpp>

#include "elasticity_energy_impl.hpp"

#include <spw_agg.hpp>
#include <spw_agg_impl.hpp>

namespace amg
{


using TVD = ElastVData<3>;
using TED = ElasticityEdgeData<3>;
using TENERGY = EpsEpsEnergy<3, TVD, TED>;

using TMESH = BlockTMWithData<AttachedNodeData<NT_VERTEX, TVD>,
                              AttachedNodeData<NT_EDGE, TED>>;


template class SPWAgglomerator<TENERGY, TMESH, TENERGY::NEED_ROBUST>;

} // namespace amg

#endif // ELASTICITY
