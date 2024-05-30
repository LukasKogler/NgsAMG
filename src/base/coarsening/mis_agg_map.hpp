#ifndef FILE_MIS_AGG_MAP_HPP
#define FILE_MIS_AGG_MAP_HPP

#ifdef MIS_AGG

#include "agglomerate_map.hpp"

#include "mis_agg.hpp"


namespace amg
{

template<class TMESH, class ENERGY>
using MISAgglomerateCoarseMap = DiscreteAgglomerateCoarseMap<TMESH, MISAgglomerator<ENERGY, typename TMESH::T_MESH_W_DATA, ENERGY::NEED_ROBUST>>;


// template<class TMESH, class ENERGY>
// class MISAgglomerateCoarseMap : public AgglomerateCoarseMap<TMESH>
// {
//   // this is the trick:
//   //   the T_AGG_MESH knows about attached data as "AttachedData",
//   //   not as its concrete derived classes; that makes the Agglomerator
//   //   independent of any "<xxx>_map.hpp" headers!
//   using T_AGG_MESH = typename TMESH::T_MESH_W_DATA;
//   using AGGLOMERATOR = MISAgglomerator<ENERGY, T_AGG_MESH, ENERGY::NEED_ROBUST>;

// public:
//   MISAgglomerateCoarseMap(shared_ptr<TMESH> mesh)
//     : AgglomerateCoarseMap<TMESH>(mesh)
//   { ; }

//   virtual ~MISAgglomerateCoarseMap() = default;

//   template<class TOPTS>
//   INLINE void Initialize (const TOPTS & opts, int level)
//   {
//     this->InitializeAgg(opts, level);
//     getMISAgglomerator().InitializeMIS(opts, level);
//   }

// protected:

//   virtual unique_ptr<BaseAgglomerator> createAgglomerator () override
//   {
//     auto tm = dynamic_pointer_cast<TMESH>(this->GetMesh());

//     assert(tm != nullptr);

//     return make_unique<AGGLOMERATOR>(tm);
//   }

//   AGGLOMERATOR &getMISAgglomerator()
//   {
//     auto agg = dynamic_cast<AGGLOMERATOR*>(&this->getAgglomerator());

//     assert(agg != nullptr);

//     return *agg;
//   }

// }; // class MISAgglomerateCoarseMap

} // namespace amg

#endif // MIS_AGG

#endif // FILE_MIS_AGG_MAP_HPP
