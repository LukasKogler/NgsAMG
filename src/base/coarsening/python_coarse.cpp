#include <base.hpp>
#include <python_ngstd.hpp>

// using namespace ngsolve;

#include "python_amg.hpp"

#include <dof_map.hpp>

namespace amg
{
  void ExportMaps (py::module & m)
  {
    
    auto baseStepClass = py::class_<BaseDOFMapStep, shared_ptr<BaseDOFMapStep>>(m, "BaseDOFMapStep", "Maps vectors between levels and can project a matrix.");

    baseStepClass.def("C2F", [&](BaseDOFMapStep         &step,
                                 shared_ptr<BaseVector>  fVec,
                                 shared_ptr<BaseVector>  cVec) {
      step.TransferC2F(fVec.get(), cVec.get());
    },
    py::arg("fVec") = nullptr,
    py::arg("cVec") = nullptr
    );
  
    baseStepClass.def("F2C", [&](BaseDOFMapStep         &step,
                                 shared_ptr<BaseVector>  fVec,
                                 shared_ptr<BaseVector>  cVec) {
      step.TransferF2C(fVec.get(), cVec.get());
    },
    py::arg("fVec") = nullptr,
    py::arg("cVec") = nullptr
    );

    baseStepClass.def("Concatenate", [&](BaseDOFMapStep         &step,
                                         shared_ptr<BaseDOFMapStep>  other) -> shared_ptr<BaseDOFMapStep> {
      return step.Concatenate(other);
    });

    baseStepClass.def("ProjectMatrix", [&](BaseDOFMapStep         &step,
                                           shared_ptr<BaseMatrix>  fineMat) -> shared_ptr<BaseMatrix> {
      if (auto fineSparse = dynamic_pointer_cast<BaseSparseMatrix>(fineMat))
        { return step.AssembleMatrix(fineSparse); }
      else
        { return nullptr; }
    },
    py::arg("fineMat")
    );

    baseStepClass.def("PrintTo", [&](BaseDOFMapStep      &step,
                                     std::string   const &ofn) {
      std::ofstream ofs(ofn);
      step.PrintTo(ofs);
    },
    py::arg("outfile"));
    
    
    auto dofMapClass = py::class_<DOFMap, shared_ptr<DOFMap>>(m, "DOFMap", "A collection of DOFMapSteps");

    dofMapClass.def(py::init<>([](py::object pySteps) -> shared_ptr<DOFMap>{
      auto steps = makeCArray<shared_ptr<BaseDOFMapStep>>(pySteps);
      auto map = make_shared<DOFMap>();
      for (auto step : steps)
      {
        map->AddStep(step);
      }
      map->Finalize();
      return map;
    }),
    py::arg("steps") = py::list()
    );

    dofMapClass.def("GetNLevels", [&](DOFMap &map) { return map.GetNLevels(); });
    
    dofMapClass.def("CreateVector", [&](DOFMap &map,
                                        int     level) -> shared_ptr<BaseVector> {
      return map.CreateVector(level);
    },
    py::arg("level") = 0
    );

    dofMapClass.def("Transfer", [&](DOFMap                 &map,
                                    int                     lOrig,
                                    shared_ptr<BaseVector>  vecIn,
                                    int                     lDest,
                                    shared_ptr<BaseVector>  vecOut) {
      map.TransferAtoB(lOrig, lDest, vecIn.get(), vecOut.get());
    },
    py::arg("lOrig")  = 0,
    py::arg("vecIn")  = nullptr,
    py::arg("lDest")  = 0,
    py::arg("vecOut") = nullptr
    );

    dofMapClass.def("SubMap", [&](DOFMap &map,
                                  int startLevel,
                                  int endLevel) -> shared_ptr<DOFMap> {
      return map.SubMap(startLevel, endLevel);
    },
    py::arg("startLevel") = 0,
    py::arg("endLevel") = 1
    );

    dofMapClass.def("GetStep", [&](DOFMap &map,
                                   int     level) -> shared_ptr<BaseDOFMapStep> {
      return map.GetStep(level);
    },
    py::arg("level") = 0
    );

    dofMapClass.def("ConcStep", [&](DOFMap &map,
                                    int     fineLevel,
                                    int     coarseLevel) -> shared_ptr<BaseDOFMapStep> {
      if (fineLevel >= coarseLevel)
        { return nullptr; }
      else
        { return map.ConcStep(fineLevel, coarseLevel, false); }
    },
    py::arg("fineLevel") = 0,
    py::arg("coarseLevel") = 0
    );

  }

} // namespace amg    