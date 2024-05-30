#ifdef ELASTICITY

#define FILE_MISAGG_EL2D_CPP

#include "elasticity_mesh.hpp"

#include <amg_pc_vertex.hpp>

#include "elasticity_energy_impl.hpp"

#include <mis_agg.hpp>
#include <mis_agg_impl.hpp>

namespace amg
{

using TVD = ElastVData<2>;
using TED = ElasticityEdgeData<2>;
using TENERGY = EpsEpsEnergy<2, TVD, TED>;

using TMESH = BlockTMWithData<AttachedNodeData<NT_VERTEX, TVD>,
                              AttachedNodeData<NT_EDGE, TED>>;


template class MISAgglomerator<TENERGY, TMESH, TENERGY::NEED_ROBUST>;

} // namespace amg

#endif // ELASTICITY
