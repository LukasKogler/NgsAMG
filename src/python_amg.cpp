#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

#include "python_amg.hpp"


namespace amg {

  extern void ExportSmoothers (py::module & m);

  extern void ExportH1Scal (py::module & m);
  extern void ExportH1Dim2 (py::module & m);
  extern void ExportH1Dim3 (py::module & m);
#ifdef AUX_AMG
  extern void ExportMCS_gg_2d (py::module & m);
  extern void ExportMCS_gg_3d (py::module & m);
#endif
#ifdef ELASTICITY
  extern void ExportElast2d (py::module & m);
  extern void ExportElast3d (py::module & m);
#ifdef AUX_AMG
  extern void ExportMCS_epseps_2d (py::module & m);
  extern void ExportMCS_epseps_3d (py::module & m);
#endif
#endif

#ifdef STOKES
  extern void ExportStokes_gg_2d (py::module & m);
  extern void ExportNCSpace (py::module &m);
#endif // STOKES

  extern void ExportHackyStuff (py::module &m);

}

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";

  amg::ExportSmoothers(m);
  
  amg::ExportH1Scal(m);
  amg::ExportH1Dim2(m);
  amg::ExportH1Dim3(m);
#ifdef AUX_AMG
  amg::ExportMCS_gg_2d(m);
  amg::ExportMCS_gg_3d(m);
#endif

#ifdef ELASTICITY
  amg::ExportElast2d(m);
  amg::ExportElast3d(m);
#ifdef AUX_AMG
  amg::ExportMCS_epseps_2d(m);
  amg::ExportMCS_epseps_3d(m);
#endif
#endif

#ifdef STOKES
  amg::ExportStokes_gg_2d(m);
  amg::ExportNCSpace(m);
#endif // STOKES

  amg::ExportHackyStuff(m);

} // PYBIND11_MODULE
