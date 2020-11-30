#ifndef FILE_PYTHON_AMG_HPP
#define FILE_PYTHON_AMG_HPP

#include <python_ngstd.hpp>

#include "amg_pc.hpp"

namespace amg {

  template<class PCC, class TLAM>
  void ExportAMGClass (py::module & m, string stra, string strb, TLAM lam)
  {
    auto amg_class = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, stra.c_str() , strb.c_str());
    amg_class.def(py::init([&](shared_ptr<BilinearForm> bfa, py::kwargs kwargs) {
	  // auto flags = CreateFlagsFromKwArgs(kwargs, h1s_class);
	  auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
	  return make_shared<PCC>(bfa, flags, "noname-pre");
	}), py::arg("bf"));

    /** TODO: add doc **/
    amg_class.def_static("__flags_doc__", [] ()
			 { return py::dict();});
    
    /** For Visualization **/
    amg_class.def("GetNLevels", [](PCC &pre, size_t rank) {
	return pre.GetAMGMatrix()->GetNLevels(rank);
      }, py::arg("rank") = int(0));
    amg_class.def("GetNDof", [](PCC &pre, size_t level, size_t rank) {
	return pre.GetAMGMatrix()->GetNDof(level, rank);
      }, py::arg("level"), py::arg("rank") = int(0));
    amg_class.def("GetBF", [](PCC &pre, shared_ptr<BaseVector> vec,
			      size_t level, size_t rank, size_t dof) {
		    pre.GetAMGMatrix()->GetBF(level, rank, dof, *vec);
		  }, py::arg("vec") = nullptr, py::arg("level") = size_t(0),
		  py::arg("rank") = size_t(0), py::arg("dof") = size_t(0));
    amg_class.def("CINV", [](PCC &pre, shared_ptr<BaseVector> csol,
			     shared_ptr<BaseVector> rhs) {
		    pre.GetAMGMatrix()->CINV(csol, rhs);
		  }, py::arg("sol") = nullptr, py::arg("rhs") = nullptr);

    amg_class.def("InitLevel", [](PCC & pre, shared_ptr<BitArray> freedofs) {
	amg::BaseAMGPC & base_pre(pre);
	base_pre.InitLevel(freedofs);
      }, py::arg("freedofs") = nullptr);

    amg_class.def("FinalizeLevel", [](PCC & pre, shared_ptr<BaseMatrix> mat) {
	amg::BaseAMGPC & base_pre(pre);
	base_pre.FinalizeLevel(mat.get());
      }, py::arg("mat") = nullptr);

    lam(amg_class);

  } // ExportH1Scal

} // namespace amg

#endif
