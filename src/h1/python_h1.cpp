#include "h1.hpp"
#include <amg_pc_vertex.hpp>

#include "python_amg.hpp"

namespace amg
{
  extern template class VertexAMGPC<H1AMGFactory<1>>;
  extern template class VertexAMGPC<H1AMGFactory<2>>;
  extern template class VertexAMGPC<H1AMGFactory<2>>;
  extern template class ElmatVAMG<H1AMGFactory<1>, double, double>;
  extern template class VertexAMGPC<H1AMGFactory<2>>;
  extern template class VertexAMGPC<H1AMGFactory<3>>;

  void ExportH1 (py::module & m) __attribute__((visibility("default")));

  template<class PCCBASE, class PCC>
  void ExportH1AMG (std::string const &name, py::module & m)
  {
    ExportAMGClass<PCCBASE, PCC>(m, name, "", [&](auto & amg_class) {

      // (experimental) "purely algebraic" mode
      amg_class.def(py::init([&](shared_ptr<BaseMatrix> A, shared_ptr<BitArray> freedofs, py::kwargs kwargs) {
        auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
        // // need to set this so ngcomp::Preconditioner does not try to register itself
        // // with the nonexistent BLF
        // flags.SetFlag("not_register_for_auto_update", true);
        auto pc = make_shared<PCC>(A, flags, "h1_scal_frommat");
        pc->InitLevel(freedofs);
        pc->FinalizeLevel(A);
        return pc;
      }), py::arg("mat"), py::arg("freedofs") = nullptr);

      amg_class.def("GetElmatEVs", [](PCCBASE &pre) -> std::tuple<double, double, double> {
        auto evs = pre.GetElmatEVs();
        return std::make_tuple(evs[0], evs[1], evs[2]);
      });

    } );
  };

  void ExportH1 (py::module & m)
  {
    ExportH1AMG<VertexAMGPC<H1AMGFactory<1>>, ElmatVAMG<H1AMGFactory<1>, double, double>>("h1_scal", m);
    ExportH1AMG<VertexAMGPC<H1AMGFactory<2>>, VertexAMGPC<H1AMGFactory<2>>>("h1_2d", m);
    ExportH1AMG<VertexAMGPC<H1AMGFactory<3>>, VertexAMGPC<H1AMGFactory<3>>>("h1_3d", m);
  }
}