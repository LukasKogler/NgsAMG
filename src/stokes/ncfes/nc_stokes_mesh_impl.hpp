#ifndef FILE_NC_STOKES_MESH_IMPL_HPP
#define FILE_NC_STOKES_MESH_IMPL_HPP

#include "nc_stokes_mesh.hpp"
#include "stokes_mesh_impl.hpp"

namespace amg
{

/** Stokes Attached Data **/

template<class TED>
void
AttachedSED<TED> :: map_data (const BaseCoarseMap & cmap, AttachedSED<TED> *ceed) const
{
  auto fmesh = dynamic_pointer_cast<NCGGStokesMesh<DIM>>(cmap.GetMesh());
  if (fmesh == nullptr)
  {
    throw Exception("Need a NCGGStokesMesh (fine) in AttachedSVD::map_data!");
  }

  auto cmesh = dynamic_pointer_cast<NCGGStokesMesh<DIM>>(cmap.GetMappedMesh());
  if (cmesh == nullptr)
  {
    throw Exception("Need a NCGGStokesMesh (coarse) in AttachedSVD::map_data!");
  }

  auto vmap   = cmap.template GetMap<NT_VERTEX>();
  auto emap   = cmap.template GetMap<NT_EDGE>();

  fmesh->CumulateData();

  auto fvdata = get<0>(fmesh->Data())->Data();
  auto cvdata = get<0>(cmesh->Data())->Data();

  auto  cedges = cmesh->template GetNodes<NT_EDGE>();

  auto &cdata  = ceed->data;

  cdata.SetSize(cmap.template GetMappedNN<NT_EDGE>());
  cdata = 0.0;

  typename NCGGStokesEnergy<DIM>::TM QHh;
  SetIdentity(QHh);

  /**
    * Consider this case:
    *       ___
    *      /   \      If A and C get merged, flow of coarse edge B-[A/C] is ZERO.
    *   A A     |     This complicates setting up coarse loops.
    *   A B B   |     Loop D->B->C->D has no useful coarse level equivalent!
    *     B B  /     So we find out here when a flow is "basically" zero [[ abs(flow(crs_e)) < eps * sum(abs(flow(f_e))) ]]
    *     D C C      we set it to EXACTLY zero. Then we can just check if(edge.flow==0) { ... crs loop is garbage }.
    * 
    *  TODO: not sure if this in practice catches all cases... what if flow reduces gradually over many levels??
    *        maybe I should compute some kind of "absflow" for every edge and keep it in the edge data...
    */

  Array<double> absflow(cdata.Size());
  absflow = 0.0;

  fmesh->template Apply<NT_EDGE>([&](const auto & e) {
    auto cenr = emap[e.id];
    if (cenr != -1) {
      const auto & ce = cedges[cenr];
      // vdh = NCGGStokesEnergy<C::DIM>::CalcMPData(fvdata[e.v[0]], fvdata[e.v[1]]);
      // vdH = NCGGStokesEnergy<C::DIM>::CalcMPData(cvdata[ce.v[0]], cvdata[ce.v[1]]);
      // NCGGStokesEnergy<C::DIM>::ModQHh(vdH, vdh, QHh);
      double fac = 1.0;

      if (vmap[e.v[0]] == ce.v[0]) {
        cdata[cenr].edi += data[e.id].edi;
        cdata[cenr].edj += data[e.id].edj;
      }
      else {
        cdata[cenr].edj += data[e.id].edi;
        cdata[cenr].edi += data[e.id].edj;
        fac = -1;
      }

      absflow[cenr] += L2Norm(data[e.id].flow);

      Iterate<DIM>([&](auto i) {
        cdata[cenr].flow[i.value] += fac * InnerProduct(data[e.id].flow, QHh.Col(i.value));
      });
    }
  }, true); // master only !

  // cout << " CED, D"; prow2(cdata); cout << endl;
  ceed->SetParallelStatus(DISTRIBUTED);
  // probably can do it locally on master, but we cumulate this sooner or later anyways and this way it is
  // consistent, so less confusing
  ceed->Cumulate();
  cmesh->template AllreduceNodalData<NT_EDGE>(absflow, [](auto & tab){return std::move(sum_table(tab)); }, false);
  // cout << endl << " ABSflow " << endl; prow2(absflow); cout << endl;

  const double eps = 1e-12;
  for (auto k : Range(absflow)) {
    if ( L2Norm(cdata[k].flow) < eps * absflow[k] ) {
      cout << " CE HAS ZERO FLOW " << k << " flow = " << cdata[k].flow << ", absflow = " << absflow[k] << endl;
      cdata[k].flow = 0.0;
    }
  }

} // AttachedSED :: map_data

/** END Stokes Attached Data **/

} // namespace amg

#endif // FILE_NC_STOKES_MESH_IMPL_HPP