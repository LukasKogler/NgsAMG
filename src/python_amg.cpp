#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

namespace amg {
  template <class TOPTS>
  void opts_from_kwa (shared_ptr<TOPTS> opts, py::kwargs & kwa)
  {
    for (auto item : kwa) {
      string name = item.first.cast<string>();
      if (name == "max_levels") { opts->max_n_levels = item.second.cast<int>(); }
      else if (name == "max_cv") { opts->max_n_verts = item.second.cast<int>(); }
      else if (name == "v_dofs") { opts->v_dofs = item.second.cast<string>(); }
      else if (name == "v_pos") { opts->v_pos = item.second.cast<string>(); }
      else if (name == "energy") { opts->energy = item.second.cast<string>(); }
      else if (name == "edges") { opts->edges = item.second.cast<string>(); }
      else if (name == "clev") { opts->clev_type = item.second.cast<string>(); }
      else if (name == "clev_inv") { opts->clev_type = item.second.cast<string>(); }
      else { cout << "warning, invalid AMG option: " << name << endl; break; }
    }
    // opts->v_pos = "VERTEX";
  }

  template<class AMG_CLASS>
  void ExportEmbedVAMG (py::module & m, string name, string description)
  {
    py::class_<EmbedVAMG<AMG_CLASS>, shared_ptr<EmbedVAMG<AMG_CLASS> >, BaseMatrix>
      (m, name.c_str(), description.c_str())
      .def(py::init<>
	   ( [] (shared_ptr<BilinearForm> blf, py::kwargs kwa) {
	     auto opts = make_shared<typename EmbedVAMG<AMG_CLASS>::Options>();
	     opts->v_pos = "VERTEX";
	     opts_from_kwa(opts, kwa);
	     return new EmbedVAMG<AMG_CLASS>(blf, opts);
	   }), py::arg("blf") = nullptr)
    .def ("Test", [](EmbedVAMG<AMG_CLASS> &pre) { pre.MyTest();} )
       .def("GetNLevels", [](EmbedVAMG<AMG_CLASS> &pre, size_t rank) {
	   return pre.GetNLevels(rank);
	 }, py::arg("rank")=int(0))
       .def("GetNDof", [](EmbedVAMG<AMG_CLASS> &pre, size_t level, size_t rank) {
	   return pre.GetNDof(level, rank);
	 }, py::arg("level"), py::arg("rank")=int(0))
       .def("GetBF", [](EmbedVAMG<AMG_CLASS> &pre, shared_ptr<BaseVector> vec,
			size_t level, size_t rank, size_t dof) {
	      pre.GetBF(level, rank, dof, *vec);
	    });
  }
  
} // namespace amg

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";

  amg::ExportEmbedVAMG<amg::H1AMG>(m, "AMG_H1", "Ngs-AMG for scalar H1-problems");
  amg::ExportEmbedVAMG<amg::ElasticityAMG<2>>(m, "AMG_EL2", "Ngs-AMG for 2d elasticity");
  amg::ExportEmbedVAMG<amg::ElasticityAMG<3>>(m, "AMG_EL3", "Ngs-AMG for 3d elasticity");
  
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
