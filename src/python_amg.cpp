#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

namespace amg { void ExportH1AMG (py::module & m); }

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";
  amg::ExportH1AMG(m);  
}

// template<int D>
// void ExportElasticityAMG(string name, py::module m)
// {
//   py::class_<ElasticityAMGPreconditioner<D>, shared_ptr<ElasticityAMGPreconditioner<D>>, BaseMatrix>
//     (m, name.c_str(), "elasticity AMG")
//     .def ("__init__",
// 	  [](ElasticityAMGPreconditioner<D>* instance,
// 	     shared_ptr<BilinearForm> bfa, shared_ptr<BilinearForm> cut_bfa, Flags & flags,
// 	     int max_levels, py::list ass_levels, py::list smooth_levels,
// 	     string smoother_type, py::list smoother_types, int max_cv, py::dict options)
// 	  {
// 	    new (instance) ElasticityAMGPreconditioner<D>(bfa, flags);
// 	    if(cut_bfa!=nullptr) instance->SetCutBLF(cut_bfa);
// 	    auto opts = instance->GetOptions();
// 	    if(max_levels>0)
// 	      opts->SetMaxNLevels(max_levels);
// 	    auto cass = makeCArray<int>(ass_levels);
// 	    if (cass.Size()) {
// 	      for(auto k:Range(opts->GetMaxNLevels()))
// 		opts->SetAssembleLevel(k, 0);
// 	      for(auto level:cass)
// 		opts->SetAssembleLevel(level,1);
// 	    }
// 	    auto csm = makeCArray<int>(smooth_levels);
// 	    if (csm.Size()) {
// 	      for(auto k:Range(opts->GetMaxNLevels()))
// 		opts->SetSmoothLevel(k, 0);
// 	      if(csm.Size()>1 || csm[0]!=-1)
// 		for(auto level:csm)
// 		  opts->SetSmoothLevel(level, 1);
// 	    }
// 	    if (smoother_type!="") {
// 	      opts->SetSmootherType(smoother_type);
// 	    }
// 	    auto cst = makeCArray<string>(smoother_types);
// 	    for(auto k:Range(cst.Size()))
// 	      opts->SetSmootherTypeOfLevel(k, cst[k]);
// 	    opts->SetCoarsestMaxVerts(max_cv);
// 	    if(options.contains("prol_max_pr"))
// 	      opts->SetProlMaxPerRow(py::extract<int>(options["prol_max_pr"])());
// 	    if(options.contains("prol_min_wt"))
// 	      opts->SetProlMinWt(py::extract<double>(options["prol_min_wt"])());
// 	    if(options.contains("prol_min_frac"))
// 	      opts->SetProlMinFrac(py::extract<double>(options["prol_min_frac"])());
// 	  },
// 	  py::arg("bf"), py::arg("cut_bf")=nullptr, py::arg("flags")=py::dict(), py::arg("max_levels")=int(0),
//           py::arg("ass_levels")=py::list(), py::arg("smooth_levels")=py::list(),
// 	  py::arg("smoother_type")="", py::arg("smoother_types")=py::list(), py::arg("max_cv")=int(100),
// 	  py::arg("amg_options")=py::dict())
//     .def ("Test", [](ElasticityAMGPreconditioner<D> &pre) { pre.MyTest();} )
//     .def ("Update", [](ElasticityAMGPreconditioner<D> &pre) { pre.Update();} )
//     .def_property_readonly("mat", [](ElasticityAMGPreconditioner<D> &self)
// 			   { return self.GetMatrixPtr(); })
//     .def("GetNLevels", [](ElasticityAMGPreconditioner<D> &pre, size_t rank) {
// 	return pre.GetNLevels(rank);
//       }, py::arg("rank")=int(0))
//     .def("GetNDof", [](ElasticityAMGPreconditioner<D> &pre, size_t level, size_t rank) {
// 	return pre.GetNDof(level, rank);
//       }, py::arg("level"), py::arg("rank")=int(0))
//     .def("GetBF", [](ElasticityAMGPreconditioner<D> &pre, shared_ptr<BaseVector> vec,
// 		     size_t level, size_t rank, size_t dof) {
// 	   pre.GetBF(level, rank, dof, *vec);
// 	 })
//     .def("GetEV", [](ElasticityAMGPreconditioner<D> &pre, shared_ptr<BaseVector> vec,
// 		     size_t level, size_t rank, size_t dof) {
// 	   pre.GetEV(level, rank, dof, *vec);
// 	 })
//     .def("CSOL", [](ElasticityAMGPreconditioner<D> &pre, shared_ptr<BaseVector> sol,
// 		    shared_ptr<BaseVector> rhs) {
// 	   pre.CINV(sol, rhs);
// 	 })
//     .def ("VCycle",
// 	  [](ElasticityAMGPreconditioner<D> & pre, shared_ptr<BaseVector> sol,
// 	     shared_ptr<BaseVector> rhs, py::list levels) {
// 	    auto clevels = makeCArray<int>(levels);
// 	    if(!clevels.Size())
// 	      pre.GetAMGMatrix()->SmoothV(sol, rhs);
// 	    else
// 	      pre.GetAMGMatrix()->SmoothV(sol, rhs, clevels);
// 	  }, py::arg("sol"), py::arg("rhs"), py::arg("levels")=py::list())
//     .def ("SetVCycle",
// 	  [](ElasticityAMGPreconditioner<D> & pre, py::list levels) {
// 	    auto clevels = makeCArray<int>(levels);
// 	    pre.GetAMGMatrix()->SetStdLevels(clevels);
// 	  }, py::arg("levels"))
//     ;
// }


// PYBIND11_MODULE(ngs_amg, m) {
//   m.attr("__name__") = "ngs_amg";
//   ExportElasticityAMG<2>("ElasticityAMGPreconditioner2d", m);
//   ExportElasticityAMG<3>("ElasticityAMGPreconditioner3d", m);
// }
