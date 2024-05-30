#ifndef FILE_PYTHON_AMG_HPP
#define FILE_PYTHON_AMG_HPP

#include "utils_sparseLA.hpp"
#include <base.hpp>
#include <python_ngstd.hpp>

#include <amg_pc.hpp>

namespace amg {

template<class PCC, class PCCREAL, class TLAM>
void ExportAMGClass (py::module & m, string stra, string strb, TLAM lam)
{
  auto amg_class = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, stra.c_str() , strb.c_str());

  amg_class.def(py::init([&](shared_ptr<BilinearForm> bfa, py::kwargs kwargs)
  {
    // auto flags = CreateFlagsFromKwArgs(kwargs, h1s_class);
    auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
    return make_shared<PCCREAL>(bfa, flags, "noname-pre");
  }), py::arg("bf"));

  /** TODO: add doc **/
  amg_class.def_static("__flags_doc__", [] ()
      { return py::dict();});

  /** For Visualization **/

  amg_class.def("GetNLevels", [](PCC &pre, size_t rank) {
    return pre.GetAMGMatrix()->GetNLevels(rank);
  }, py::arg("rank") = int(0));

  amg_class.def("GetNProcs", [](PCC & pre, size_t level) {
    auto uDofs = pre.GetAMGMatrix()->GetMap()->GetUDofs();
    if (uDofs.IsParallel())
    {
      auto comm = uDofs.GetCommunicator();
      int nps = (comm.Rank() > 0) ? 0 : pre.GetAMGMatrix()->GetMap()->GetUDofs(level).GetCommunicator().Size();
      return comm.AllReduce(nps, NG_MPI_MAX);
    }
    else
    {
      return 1;
    }
  }, py::arg("level") = int(0));

  amg_class.def("GetBlockSize", [](PCC & pre, int level) {
auto [nd, bs] = pre.GetAMGMatrix()->GetNDof(level, 0);
return bs;
    }, py::arg("level") = int(0));
  amg_class.def("GetNDof", [](PCC &pre, int level, int rank) {
auto [nd, bs] = pre.GetAMGMatrix()->GetNDof(level, rank);
return nd;
    }, py::arg("level"), py::arg("rank") = int(0));
  amg_class.def("GetNDBS", [](PCC &pre, int level, int rank) {
auto tup = pre.GetAMGMatrix()->GetNDof(level, rank);
return py::cast(tup);
// auto [nd, bs] = pre.GetAMGMatrix()->GetNDof(level, rank);
// return py::tuple(py::cast(nd), py::cast(bs));
    }, py::arg("level"), py::arg("rank") = int(0));
  amg_class.def("GetBF", [](PCC &pre, shared_ptr<BaseVector> vec,
          int level, size_t dof, int comp, int rank) {
      pre.GetAMGMatrix()->GetBF(*vec, level, dof, comp, rank);
    }, py::arg("vec") = nullptr, py::arg("level") = int(0),
    py::arg("dof") = size_t(0), py::arg("comp") = int(0), py::arg("rank") = int(0) );
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

  amg_class.def("GetMap", [](PCC & pre) {
return pre.GetAMGMatrix()->GetMap();
    });

  amg_class.def("RegularizeMatrix", [](PCC &pre, shared_ptr<BaseMatrix> mat) {
    auto pardofs = mat->GetParallelDofs();
    auto localMat = GetLocalMat(mat);
    auto localSparse = my_dynamic_pointer_cast<BaseSparseMatrix>(localMat, "RegularizeMatrix needs a SPARSE matrix!");
    pre.RegularizeMatrix(localSparse, pardofs);
  });

  amg_class.def("GetSmoother", [](PCC & pre, int level) -> shared_ptr<BaseSmoother> {
shared_ptr<BaseSmoother> sm = nullptr;
if (level < pre.GetAMGMatrix()->GetSmoothers().Size())
  { sm =  const_pointer_cast<BaseSmoother>(pre.GetAMGMatrix()->GetSmoother(level)); }
else { cout << " only have " << pre.GetAMGMatrix()->GetSmoothers().Size() << " smoothers " << endl; }
return sm;
    }), py::arg("level") = 0;

  amg_class.def("GetAMGMatrix", [](PCC &pre) -> shared_ptr<AMGMatrix> { return pre.GetAMGMatrix(); });

  lam(amg_class);

} // ExportH1Scal

} // namespace amg

#endif
