#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

namespace amg {
  template <class TOPTS>
  void opts_from_kwa (shared_ptr<TOPTS> opts, py::kwargs & kwa)
  {
    auto int_to_il = [](int in)->INFO_LEVEL {
      if (in==0) return NONE;
      if (in==1) return BASIC;
      if (in==2) return DETAILED;
      if (in==3) return EXTRA;
      return NONE;
    };
    auto capitalize_it = [](string in) -> string {
      string out; std::locale loc;
      for (auto x : in) out += std::toupper(x, loc);
      return out;
    };
    for (auto item : kwa) {
      string name = item.first.cast<string>();
      bool edges_set = false;
      if (name == "max_levels") { opts->max_n_levels = item.second.cast<int>(); }
      else if (name == "max_cv") { opts->max_n_verts = item.second.cast<int>(); }
      else if (name == "v_dofs") { opts->v_dofs = item.second.cast<string>(); }
      else if (name == "v_pos") { opts->v_pos = capitalize_it(item.second.cast<string>()); }
      else if (name == "energy") {
	opts->energy = capitalize_it(item.second.cast<string>());
	if (!edges_set)
	  if (opts->energy == "ELMAT") opts->edges = "ELMAT";
	// else if (opts->energy == "ALG") opts->edges = "ALG"; // not implemented yet
      }
      else if (name == "edges") { edges_set = true; opts->edges = capitalize_it(item.second.cast<string>()); }
      else if (name == "clev") { opts->clev_type = capitalize_it(item.second.cast<string>()); }
      else if (name == "clev_inv") { opts->clev_type = item.second.cast<string>(); }
      else if (name == "skip_ass") { opts->skip_ass_first = item.second.cast<int>(); }
      else if (name == "ass_lev") { py::list py_list = item.second.cast<py::list>(); opts->ass_levels = move(makeCArray<int>(py_list)); }
      else if (name == "ass_skip_lev") { py::list py_list = item.second.cast<py::list>(); opts->ass_skip_levels = move(makeCArray<int>(py_list)); }
      else if (name == "ass_frac") { opts->ass_after_frac = item.second.cast<double>(); }
      else if (name == "force_ass") { py::list py_list = item.second.cast<py::list>(); opts->force_ass = true; opts->ass_levels = move(makeCArray<int>(py_list)); }
      else if (name == "enable_sm") { opts->enable_sm = item.second.cast<bool>(); }
      else if (name == "skip_sm") { opts->skip_smooth_first = item.second.cast<int>(); }
      else if (name == "sm_lev") { py::list py_list = item.second.cast<py::list>(); opts->sm_levels = move(makeCArray<int>(py_list)); }
      else if (name == "sm_skip_lev") { py::list py_list = item.second.cast<py::list>(); opts->sm_skip_levels = move(makeCArray<int>(py_list)); }
      else if (name == "force_sm") { py::list py_list = item.second.cast<py::list>(); opts->force_sm = true; opts->sm_levels = move(makeCArray<int>(py_list)); }
      else if (name == "sm_frac") { opts->smooth_after_frac = item.second.cast<double>(); }
      else if (name == "enable_redist") { opts->enable_ctr = item.second.cast<bool>(); }
      else if (name == "ctr_min_nv") { opts->ctr_min_nv = item.second.cast<size_t>(); }
      else if (name == "ctr_seq_nv") { opts->ctr_seq_nv = item.second.cast<size_t>(); }
      else if (name == "ctr_frac") { opts->ctr_after_frac = item.second.cast<double>(); }
      else if (name == "log_level") { opts->info_level = int_to_il(item.second.cast<int>()); }
      else if (name == "crs_v_thresh") { opts->min_vcw = item.second.cast<double>(); }
      else if (name == "crs_e_thresh") { opts->min_ecw = item.second.cast<double>(); }
      else if (name == "dpv") { opts->block_s.SetSize(1); opts->block_s[0] = item.second.cast<int>(); }
      else { throw Exception(string("warning, invalid AMG option: ")+name); }
    }
    // opts->v_pos = "VERTEX";
  }

  py::list py_list_int3p1 (Array<INT<3>> & ar, Array<int> & ar2) {
    py::list pl(ar.Size());
    for (auto k : Range(ar.Size())) {
      auto v = ar[k]; auto v2 = ar2[k];
      pl[k] = py::make_tuple(v[0], v[1], v[2], v2);
    }
    return pl;
  }

  template<class T>
  py::list py_list (Array<T> & ar) {
    py::list pl(ar.Size());
    for (auto k : Range(ar.Size()))
      pl[k] = ar[k];
    return pl;
  }

  template<class AMG_CLASS, class TLAM>
  void Export1 (py::module & m, string name, string description, TLAM lam_opts)
  {
    auto dict_from_info = [](auto spi) -> py::object {
      py::dict pd;
      auto il_to_str = [](INFO_LEVEL in) {
	if (in==NONE) return "NONE";
	if (in==BASIC) return "BASIC";
	if (in==DETAILED) return "DETAILED";
	if (in==EXTRA) return "EXTRA";
	return "UNKNOWN";
      };
      pd["log_level"] = il_to_str(spi->ilev);
      if ( spi->has_comm && spi->glob_comm.Rank() !=0 ) return py::none();
      if (spi->ilev >= BASIC) {
	pd["levels"] = py_list_int3p1(spi->lvs, spi->isass);
	pd["VC"] = spi->v_comp;
	pd["VCcs"] = py_list(spi->vcc);
	pd["OC"] = spi->op_comp;
	pd["OCcs"] = py_list(spi->occ);
	pd["NVs"] = py_list(spi->NVs);
      }
      if (spi->ilev >= DETAILED) {
	pd["NEs"] = py_list(spi->NEs);
	pd["NPs"] = py_list(spi->NPs);
	pd["MCMat"] = spi->mem_comp1;
	pd["MCSm"] = spi->mem_comp2;
	pd["MCMatcs"] = py_list(spi->mcc1);
	pd["MCSmocs"] = py_list(spi->mcc2);
	pd["OC_LOC"] = spi->op_comp_l;
	pd["OC_LOCcs"] = py_list(spi->occ_l);
	pd["VC_LOC"] = spi->v_comp_l;
	pd["VC_LOCcs"] = py_list(spi->vcc_l);
      }
      if (spi->ilev >= EXTRA) {
      }
      return move(pd);
    };
    py::class_<AMG_CLASS, shared_ptr<AMG_CLASS >, BaseMatrix>
      (m, name.c_str(), description.c_str())
      .def(py::init<>
	   ( [&] (shared_ptr<BilinearForm> blf, py::kwargs kwa) {
	     auto opts = make_shared<typename AMG_CLASS::Options>();
	     opts->v_pos = "VERTEX";
	     opts_from_kwa(opts, kwa);
	     lam_opts(opts);
	     return new AMG_CLASS(blf, opts);
	   }), py::arg("blf") = nullptr)
      .def ("GetLogs", [dict_from_info](AMG_CLASS &pre) { return dict_from_info(pre.GetInfo()); } )
      .def ("Test", [](AMG_CLASS &pre) { pre.MyTest();} )
      .def("GetNLevels", [](AMG_CLASS &pre, size_t rank) {
	  return pre.GetNLevels(rank);
	}, py::arg("rank")=int(0))
      .def("GetNDof", [](AMG_CLASS &pre, size_t level, size_t rank) {
	  return pre.GetNDof(level, rank);
	}, py::arg("level"), py::arg("rank")=int(0))
      .def("GetBF", [](AMG_CLASS &pre, shared_ptr<BaseVector> vec,
		       size_t level, size_t rank, size_t dof) {
	     pre.GetBF(level, rank, dof, *vec);
	   });
  }
  
} // namespace amg

PYBIND11_MODULE (ngs_amg, m) {
  m.attr("__name__") = "ngs_amg";

  amg::Export1<amg::EmbedVAMG<amg::H1AMG>>(m, "AMG_H1", "Ngs-AMG for scalar H1-problems", [](auto & o) {});

#ifdef ELASTICITY
  amg::Export1<amg::EmbedVAMG<amg::ElasticityAMG<2>, double, amg::STABEW<2>>>(m, "AMG_EL2", "Ngs-AMG for 2d elasticity", [](auto & o) { o->keep_vp = true; });
  amg::Export1<amg::EmbedVAMG<amg::ElasticityAMG<3>, double, amg::STABEW<3>>>(m, "AMG_EL3", "Ngs-AMG for 2d elasticity", [](auto & o) { o->keep_vp = true; });
#else
  m.def("AMG_EL2", [&] (shared_ptr<BilinearForm> blf, py::kwargs kwa) {
      throw Exception("Elasticity AMG not available.");
      return py::none();      
    }, py::arg("blf") = nullptr);
  m.def("AMG_EL3", [&] (shared_ptr<BilinearForm> blf, py::kwargs kwa) {
      throw Exception("Elasticity AMG not available.");
      return py::none();      
    }, py::arg("blf") = nullptr);
#endif  
}
