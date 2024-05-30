#ifdef ELASTICITY
#ifndef FILE_AMG_ELAST_IMPL_HPP
#define FILE_AMG_ELAST_IMPL_HPP

#include "elasticity_mesh.hpp"
#include "elasticity_energy.hpp"
#include "elasticity_energy_impl.hpp"

// #include "amg_factory_nodal_impl.hpp"
// #include "amg_factory_vertex_impl.hpp"
// #include "amg_pc.hpp"
// #include "amg_pc_vertex.hpp"
// #include "amg_pc_vertex_impl.hpp"

#include "compressedfespace.hpp"

/** Need this only if we also include the PC headers **/

namespace amg
{

template<int DIM>
void AttachedEED<DIM> :: map_data (const BaseCoarseMap & cmap, AttachedEED<DIM> *pceed) const
{
  auto & ceed(*pceed);
  static Timer t(string("AttachedEED::map_data")); RegionTimer rt(t);
  auto & M = static_cast<ElasticityMesh<DIM>&>(*this->mesh); M.CumulateData();
  const auto & eqc_h = *M.GetEQCHierarchy();
  auto& fecon = *M.GetEdgeCM();
  ElasticityMesh<DIM> & CM = static_cast<ElasticityMesh<DIM>&>(*ceed.mesh);
  auto cedges = CM.template GetNodes<NT_EDGE>();
  const size_t NV = M.template GetNN<NT_VERTEX>();
  const size_t NE = M.template GetNN<NT_EDGE>();
  const size_t CNV = cmap.template GetMappedNN<NT_VERTEX>();
  const size_t CNE = cmap.template GetMappedNN<NT_EDGE>();
  auto e_map = cmap.template GetMap<NT_EDGE>();
  auto v_map = cmap.template GetMap<NT_VERTEX>();
  get<0>(M.Data())->Cumulate(); // should be cumulated anyways
  auto fvd = get<0>(M.Data())->Data();
  get<0>(CM.Data())->Cumulate(); // we need full vertex positions !
  auto cvd = get<0>(CM.Data())->Data();
  auto fed = this->Data();
  auto ced = ceed.Data();
  typedef Mat<BS, BS, double> TM;
  // TODO: we are modifying coarse v-wts here. HACKY!!
  Array<TM> add_cvw (CNV); add_cvw = 0;
  Vec<DIM> posH, posh, tHh;
  TM TEST = 0;
  TM QHh(0), FMQ(0);
  ceed.data.SetSize(CNE); // ced. is a flat-array, so directly access ceed.data
  ced = 0.0;
  M.template ApplyEQ<NT_EDGE>([&] (auto eqc, const auto & fedge) LAMBDA_INLINE {
    auto cenr = e_map[fedge.id];
    if (cenr != -1) {
      const auto & cedge = cedges[cenr];
      auto& cemat = ced[cenr];
      auto cmid = ElasticityAMGFactory<DIM>::ENERGY::CalcMPData(cvd[cedge.v[0]], cvd[cedge.v[1]]);
      auto fmid = ElasticityAMGFactory<DIM>::ENERGY::CalcMPData(fvd[fedge.v[0]], fvd[fedge.v[1]]);
      ElasticityAMGFactory<DIM>::ENERGY::CalcQHh(cmid, fmid, QHh);
      FMQ = fed[fedge.id] * QHh;
      cemat += Trans(QHh) * FMQ;
    }
    else { /** connection to ground goes into vertex weight **/
      INT<2, int> cvs ( { v_map[fedge.v[0]], v_map[fedge.v[1]] } );;
      if (cvs[0] != cvs[1]) { // max. and min. one is -1
        int l = (cvs[0] == -1) ? 1 : 0;
        int cvnr = v_map[fedge.v[l]];
        ElasticityAMGFactory<DIM>::ENERGY::CalcQij(fvd[fedge.v[l]], fvd[fedge.v[1-l]], QHh); // from [l] to [1-l] should be correct
        ElasticityAMGFactory<DIM>::ENERGY::AddQtMQ(1.0, add_cvw[cvnr], QHh, fed[fedge.id]);
      }
    }
  }, true); // master only
  CM.template AllreduceNodalData<NT_VERTEX>(add_cvw, [](auto & in) { return sum_table(in); }, false);
  for (auto k : Range(CNV)) { // cvd and add_cvw are both "CUMULATED"
    cvd[k].wt += add_cvw[k];
  }
  ceed.SetParallelStatus(DISTRIBUTED);
} // AttachedEED::map_data


// this was for pairwise coarse-map
// template<int DIM> template<class TMESH>
// INLINE void AttachedEVD<DIM> :: map_data (const CoarseMap<TMESH> & cmap, AttachedEVD<DIM> *pcevd) const
// {
//   auto &cevd(*pcevd);
//   /** ECOL coarsening -> set midpoints in edges **/
//   static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
//   Cumulate();
//   auto & cdata = cevd.data; cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0;
//   auto vmap = cmap.template GetMap<NT_VERTEX>();
//   Array<int> touched(vmap.Size()); touched = 0;
//   mesh->template Apply<NT_EDGE>([&](const auto & e) { // set coarse data for all coll. vertices
//     auto CV = vmap[e.v[0]];
//     if ( (CV != -1) || (vmap[e.v[1]] == CV) ) {
//       touched[e.v[0]] = touched[e.v[1]] = 1;
//       cdata[CV] = ElasticityAMGFactory<DIM>::ENERGY::CalcMPDataWW(data[e.v[0]], data[e.v[1]]);
//     }
//   }, true); // if stat is CUMULATED, only master of collapsed edge needs to set wt 
//   mesh->template AllreduceNodalData<NT_VERTEX>(touched, [](auto & in) { return std::move(sum_table(in)); } , false);
//   mesh->template Apply<NT_VERTEX>([&](auto v) { // set coarse data for all "single" vertices
//     auto CV = vmap[v];
//     if ( (CV != -1) && (touched[v] == 0) )
//       { cdata[CV] = data[v]; }
//   }, true);
//   cevd.SetParallelStatus(DISTRIBUTED);
// } // AttachedEVD::map_data


template<int DIM> template<class TMESH>
INLINE void AttachedEVD<DIM> :: map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedEVD<DIM> *pcevd) const
{
  auto &cevd(*pcevd);
  /** AGG coarsening -> set midpoints in agg centers **/
  static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
  Cumulate();
  auto & cdata = cevd.data; cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>());
  auto vmap = cmap.template GetMap<NT_VERTEX>();
  const auto & M = *mesh;
  const auto & CM = static_cast<BlockTM&>(*cmap.GetMappedMesh()); // okay, kinda hacky, the coarse mesh already exists, but only as BlockTM i think
  const auto & ctrs = *cmap.GetAggCenter();
  typename ElasticityAMGFactory<DIM>::ENERGY::TM Q;
  
  auto aggs = cmap.template GetMapC2F<NT_VERTEX>();
  // CM.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto cV) LAMBDA_INLINE
  // {
  //   Vec<DIM, double> cPos(.0);
  //   auto agg = aggs[cV];

  //   for (auto k : Range(agg))
  //   {
  //     cPos += data[agg[k]].pos;
  //   }
  //   cPos /= agg.Size();
  //   cdata[cV].pos = cPos;

  // }, true);

  auto const &eqc_h = *M.GetEQCHierarchy();

  cdata = 0.0;

  CM.template ApplyEQ2<NT_VERTEX> ( [&](auto eqc, auto cVerts) LAMBDA_INLINE
  {
    bool const isMaster = eqc_h.IsMasterOfEQC(eqc);

    for (auto cV : cVerts)
    {
      if (isMaster)
      {
        auto agg = aggs[cV];
        auto fV  = agg[0];

        cdata[cV].pos         = data[fV].pos;
        cdata[cV].wt          = data[fV].wt;
        cdata[cV].rot_scaling = data[fV].rot_scaling;
      }
      else
      {
        cdata[cV] = 0.0;
      }
    }
  }, false);

  // M.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto v) LAMBDA_INLINE
  // {
  //   /** set crs v pos - ctrs can add weight already **/
  //   auto cv = vmap[v];
  //   if ( ( cv != -1 ) && ( ctrs.Test(v) ) )
  //   { // not correct when we are not doing root-pos!
  //     cdata[cv].pos = data[v].pos;
  //     cdata[cv].wt += data[v].wt;
  //     // cdata[cv].rot_scaling = data[v].rot_scaling;
  //     cdata[cv].rot_scaling = 1.0;//data[v].rot_scaling;
  //   }
  // }, true);

  M.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto v) LAMBDA_INLINE
  {
    /** add l2 weights for non ctrs - I already need crs pos here **/
    auto cv = vmap[v];
    if ( ( cv != -1 ) && ( !ctrs.Test(v) ) )
    {
      ElasticityAMGFactory<DIM>::ENERGY::CalcQHh(cdata[cv], data[v], Q);
      ElasticityAMGFactory<DIM>::ENERGY::AddQtMQ(1.0, cdata[cv].wt, Q, data[v].wt);
    }
  }, true);

  cevd.SetParallelStatus(DISTRIBUTED);
} // AttachedEVD::map_data

} // namespace amg


#endif // FILE_AMG_ELAST_IMPL_HPP
#endif // ELASTICITY
