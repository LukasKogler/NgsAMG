
/** 3D, NC-space, laplace + div div **/

#include <base.hpp>

#include <spw_agg.hpp>
#include <spw_agg_map.hpp>

#include <mis_agg.hpp>
#include <mis_agg_map.hpp>

#include "hdiv_stokes_factory.hpp"

#include "hdiv_stokes_mesh.hpp"

#include "hdiv_stokes_pc.hpp"

// #include <nodal_factory_impl.hpp>
#include <stokes_factory_impl.hpp>
// #include <spw_agg_impl.hpp>
// #include <mis_agg_impl.hpp>
#include <stokes_pc_impl.hpp>

#include "hdiv_stokes_mesh_impl.hpp"

#include "hdiv_stokes_pc_impl.hpp"

namespace amg
{

using TMESH   = HDivGGStokesMesh<3>;
using TENERGY = HDivGGStokesEnergy<3>;

/** COARSENING **/

using T_MESH_WITH_DATA = TMESH::T_MESH_W_DATA;

extern template class SPWAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;
extern template class MISAgglomerator<TENERGY, T_MESH_WITH_DATA, TENERGY::NEED_ROBUST>;


/** FACTORY **/

extern template class StokesAMGFactory<TMESH, TENERGY>;

using TFACTORY = HDivStokesAMGFactory<TMESH, TENERGY>;
 
 
/** PRECONDITIONER **/

template class HDivStokesAMGPC<TFACTORY>;
using STOKES_PC = HDivStokesAMGPC<TFACTORY>;

RegisterPreconditioner<STOKES_PC> reg_stokes_hdiv_gg_3d ("NgsAMG.stokes_hdiv_gg_3d");

} // namespace amg


#include "python_stokes.hpp"

namespace amg
{

void ExportStokes_hdiv_gg_3d (py::module & m) __attribute__((visibility("default")));

void ExportStokes_hdiv_gg_3d (py::module & m)
{
  string stokes_desc = "Stokes preconditioner for grad-grad + div-div penalty in HDiv space";

  ExportStokesAMGClass<STOKES_PC> (m, "stokes_hdiv_gg_3d", stokes_desc, [&](auto & pyClass)
  {
    // pyClass.def("SetVectorsToPreserve", [](STOKES_PC &c,
    //                                        py::object pyVecs) {
    //   auto vecs = makeCArray<shared_ptr<BaseVector>>(pyVecs);
    //   c.SetVectorsToPreserve(vecs);
    // },
    // py::arg("vecs"));

    pyClass.def("GetPresVec", [](STOKES_PC &c,
                                 int level,
                                 int k) -> shared_ptr<BaseVector>
    {
      if (c.stashedPresVecs.Size() == 0)
      {
        cout << " pres-vecs not stashed, set log_level to debug! " << endl;
        return nullptr;
      }

      auto map = c.GetAMGMatrix()->GetMap();
      auto vec = map->CreateVector(0);

      if (level > 0)
      {
        auto pVec = c.stashedPresVecs[level][k];
        map->TransferAtoB(level, 0, pVec.get(), vec.get());
      }
      else
      {
        auto pVec = c.stashedPresVecs[level][k];
        c.stashedEmb->TransferC2F(vec.get(), pVec.get());
      }

      return vec;
    },
    py::arg("level") = 1,
    py::arg("k") = 0);
  });
} // ExportStokes_gg_2d

} // namespace amg
