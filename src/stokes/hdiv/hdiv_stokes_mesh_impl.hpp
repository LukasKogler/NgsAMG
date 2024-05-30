#ifndef FILE_NC_STOKES_MESH_IMPL_HPP
#define FILE_NC_STOKES_MESH_IMPL_HPP

#include "hdiv_stokes_mesh.hpp"
#include "stokes_mesh_impl.hpp"

namespace amg
{

/** Stokes Attached Data **/

template<class TED>
void
AttachedSED<TED> :: map_data (const BaseCoarseMap & cmap, AttachedSED<TED> *ceed) const
{
  auto fmesh = my_dynamic_pointer_cast<HDivGGStokesMesh<DIM>>(cmap.GetMesh(),
                "Need a NCGGStokesMesh (fine) in AttachedSVD::map_data!");
  auto cmesh = my_dynamic_pointer_cast<HDivGGStokesMesh<DIM>>(cmap.GetMappedMesh(),
                "Need a NCGGStokesMesh (coarse) in AttachedSVD::map_data!");

  auto vmap   = cmap.template GetMap<NT_VERTEX>();
  auto emap   = cmap.template GetMap<NT_EDGE>();

  auto cedges = cmesh->template GetNodes<NT_EDGE>();

  fmesh->CumulateData();

  auto fvdata = get<0>(fmesh->Data())->Data();
  auto cvdata = get<0>(cmesh->Data())->Data();

  auto &cdata  = ceed->data;

  cdata.SetSize(cmap.template GetMappedNN<NT_EDGE>());
  cdata = 0.0;

  fmesh->template Apply<NT_EDGE>([&](const auto & e)
  {
    const auto cenr = emap[e.id];

    if (cenr != -1)
    {
      const auto &ce   = cedges[cenr];
      if (vmap[e.v[0]] == ce.v[0]) {
        cdata[cenr].edi += data[e.id].edi;
        cdata[cenr].edj += data[e.id].edj;
      }
      else {
        cdata[cenr].edj += data[e.id].edi;
        cdata[cenr].edi += data[e.id].edj;
      }
    }
  }, true); // master only !

  ceed->SetParallelStatus(DISTRIBUTED);
} // AttachedSED :: map_data

/** END Stokes Attached Data **/

} // namespace amg

#endif // FILE_NC_STOKES_MESH_IMPL_HPP