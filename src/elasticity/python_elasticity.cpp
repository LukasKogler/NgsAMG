#ifdef ELASTICITY

#include "elasticity.hpp"
#include <amg_pc_vertex.hpp>

#include "python_amg.hpp"

namespace amg
{
  void ExportElast2d (py::module & m) __attribute__((visibility("default")));

  void ExportElast2d (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<ElasticityAMGFactory<2>, double, double>,
                   ElmatVAMG<ElasticityAMGFactory<2>, double, double>>(m, "elast_2d", "", [&](auto & pyClass) { ; } );
  };

  void ExportElast3d (py::module & m) __attribute__((visibility("default")));

  void ExportElast3d (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<ElasticityAMGFactory<3>, double, double>,
                   ElmatVAMG<ElasticityAMGFactory<3>, double, double>>(m, "elast_3d", "", [&](auto & pyClass) {
      pyClass.def("GetRotationOfBF", [](ElmatVAMG<ElasticityAMGFactory<3>, double, double> &pre, shared_ptr<BaseVector> vec,
          int level, size_t dof, int comp, int rank)
      {
        if (level == 0)
          { return; }

        auto amgMat = pre.GetAMGMatrix();
        auto map = amgMat->GetMap();
        auto firstStep = map->GetStep(0);
        auto multiStep = my_dynamic_pointer_cast<MultiDofMapStep>(firstStep, "MULTI-DMS");
        auto rotStep = multiStep->GetMap(1);
        auto tempVec = map->CreateVector(1);
        amgMat->GetBF(*tempVec, level, dof, comp, rank, 1);
        // this is OK in 3d where #displacements == #rotations!
        rotStep->TransferC2F(vec.get(), tempVec.get());
      },
      py::arg("vec")   = nullptr,
      py::arg("level") = int(0),
      py::arg("dof")   = size_t(0),
      py::arg("comp")  = int(0),
      py::arg("rank")  = int(0) );
    });
  };
  
} // namespace amg

#endif // ELASTICITY
