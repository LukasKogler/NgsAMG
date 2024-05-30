#ifndef FILE_PLATE_TEST_AGG_MAP_HPP
#define FILE_PLATE_TEST_AGG_MAP_HPP

#ifdef MIS_AGG

#include "agglomerate_map.hpp"

#include "plate_test_agg.hpp"


namespace amg
{

template<class TMESH>
using PlateTestAgglomerateCoarseMap = DiscreteAgglomerateCoarseMap<TMESH, PlateTestAgglomerator<typename TMESH::T_MESH_W_DATA>>;

} // namespace amg

#endif // MIS_AGG

#endif // FILE_PLATE_TEST_AGG_MAP_HPP
