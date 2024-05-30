#include <base.hpp>
#include <python_ngstd.hpp>
#include <python_comp.hpp>

// using namespace ngsolve;

#include <ncfespace.hpp>

#include "python_amg.hpp"


namespace amg {

extern void ExportUtils (py::module &m);

extern void ExportSmoothers (py::module & m);
extern void ExportMaps (py::module & m);
extern void ExportSolve (py::module & m);

extern void ExportH1 (py::module & m);

#ifdef ELASTICITY
extern void ExportElast2d (py::module & m);
extern void ExportElast3d (py::module & m);
#endif

extern void ExportKrylovAMG (py::module & m);

#ifdef STOKES_AMG
extern void ExportStokes_gg_2d (py::module &m);
extern void ExportStokes_gg_3d (py::module &m);
extern void ExportStokes_hdiv_gg_2d (py::module &m);
extern void ExportStokes_hdiv_gg_3d (py::module &m);
#endif
}

PYBIND11_MODULE (NgsAMG, m) {
  m.attr("__name__") = "NgsAMG";

  m.doc() = "Auxiliary Topology based Algebraic Multigrid Methods";

  ngcomp::ExportFESpace<typename amg::NoCoH1FESpace>(m, "NoCoH1");

  amg::ExportUtils(m);

  amg::ExportSmoothers(m);
  amg::ExportMaps(m);
  amg::ExportSolve(m);

  amg::ExportH1(m);

#ifdef ELASTICITY
  amg::ExportElast2d(m);
  amg::ExportElast3d(m);
#endif

#ifdef STOKES_AMG
  amg::ExportStokes_gg_2d(m);
  amg::ExportStokes_gg_3d(m);
  amg::ExportStokes_hdiv_gg_2d(m);
  amg::ExportStokes_hdiv_gg_3d(m);
#endif
} // PYBIND11_MODULE
