#ifndef FILE_SPW_AGG_MAP_HPP
#define FILE_SPW_AGG_MAP_HPP

#ifdef SPW_AGG

#include "agglomerate_map.hpp"

#include "spw_agg.hpp"

namespace amg
{

template<class TMESH, class ENERGY>
using SPWAgglomerateCoarseMap = DiscreteAgglomerateCoarseMap<TMESH, SPWAgglomerator<ENERGY, typename TMESH::T_MESH_W_DATA, ENERGY::NEED_ROBUST>>;


} // namespace amg

#endif // SPW_AGG

#endif // FILE_SPW_AGG_MAP_HPP
