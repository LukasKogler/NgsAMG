#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

#include "python_amg.hpp"


namespace amg {
  extern void ExportMCS3D (py::module & m);
  extern void ExportH1Scal (py::module & m);
  extern void ExportH1Dim2 (py::module & m);
  extern void ExportH1Dim3 (py::module & m);
}

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";
  // amg::ExportAMGClass<amg::EmbedWithElmats<amg::H1AMGFactory, double, double>>(m, "h1_scal", "scalar h1 amg PC");
  amg::ExportH1Scal(m);
  amg::ExportH1Dim2(m);
  amg::ExportH1Dim3(m);
#ifdef ELASTICITY
  amg::ExportAMGClass<amg::EmbedWithElmats<amg::ElasticityAMGFactory<2>, double, double>>(m, "elast_2d", "2d elasticity amg");
  amg::ExportAMGClass<amg::EmbedWithElmats<amg::ElasticityAMGFactory<3>, double, double>>(m, "elast_3d", "3d elasticity amg");
  amg::ExportMCS3D(m);
#endif
}
