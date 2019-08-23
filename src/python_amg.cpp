#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

namespace amg {

  void ExportH1Scal (py::module & m)
  {
    typedef EmbedWithElmats<H1AMGFactory, double, double> PCC;
    auto h1s_class = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, "h1_scal", "scalar h1 amg PC");
    h1s_class.def(py::init([&](shared_ptr<BilinearForm> bfa, py::kwargs kwargs) {
	  // auto flags = CreateFlagsFromKwArgs(kwargs, h1s_class);
	  auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
	  return make_shared<PCC>(bfa, flags, "noname-pre");
	}), py::arg("bf"))
      .def_static("__flags_doc__", [] ()
		  { return py::dict();})
      .def("GetNLevels", [](PCC &pre, size_t rank) {
	  return pre.GetAMGMat()->GetNLevels(rank);
	}, py::arg("rank")=int(0))
      .def("GetNDof", [](PCC &pre, size_t level, size_t rank) {
	  return pre.GetAMGMat()->GetNDof(level, rank);
	}, py::arg("level"), py::arg("rank")=int(0))
      .def("GetBF", [](PCC &pre, shared_ptr<BaseVector> vec,
		       size_t level, size_t rank, size_t dof) {
	     pre.GetAMGMat()->GetBF(level, rank, dof, *vec);
	   }, py::arg("vec")=nullptr, py::arg("level")=size_t(0),
	   py::arg("rank")=size_t(0), py::arg("dof")=size_t(0))
      .def("CINV", [](PCC &pre, shared_ptr<BaseVector> csol,
		      shared_ptr<BaseVector> rhs) {
	     pre.GetAMGMat()->CINV(csol, rhs);
	   }, py::arg("sol")=nullptr, py::arg("rhs")=nullptr);
  } // ExportH1Scal

} // namespace amg

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";
  amg::ExportH1Scal(m);
}
