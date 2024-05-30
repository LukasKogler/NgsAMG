#include <base.hpp>

#include "base_smoother.hpp"
#include "dyn_block_smoother.hpp"
#include "hybrid_matrix.hpp"
#include "hybrid_smoother.hpp"
#include "gssmoother.hpp"
#include "block_gssmoother.hpp"
#include "loc_block_gssmoother.hpp"

#include <python_ngstd.hpp>

namespace amg
{

extern template class HybridDISmoother<double>;
extern template class HybridDISmoother<Mat<2,2,double>>;
extern template class HybridDISmoother<Mat<3,3,double>>;
extern template class HybridDISmoother<Mat<6,6,double>>;

std::tuple<shared_ptr<BaseMatrix>, shared_ptr<ParallelDofs>> MatrixToLocalComponents (shared_ptr<BaseMatrix> A)
{
  if ( auto pm = dynamic_pointer_cast<ParallelMatrix>(A) )
  {
    return std::make_tuple(pm->GetMatrix(), pm->GetRowParallelDofs());
  }
  else
  {
    return std::make_tuple(A, nullptr);
  }
}

void ExportSmoothers (py::module & m)
{
  auto smc = py::class_<BaseSmoother, shared_ptr<BaseSmoother>, BaseMatrix>(m, "BaseSmoother", "");

  smc.def("GetMatrix", [](BaseSmoother &sm) -> shared_ptr<BaseMatrix> { return sm.GetAMatrix(); });

  smc.def("Smooth", [](BaseSmoother & sm, shared_ptr<BaseVector> x, shared_ptr<BaseVector> b, shared_ptr<BaseVector> res,
                       bool res_updated, bool update_res, bool x_zero)
    {
      shared_ptr<BaseVector> ures = res;
      if ( ures == nullptr )
        { ures = sm.CreateColVector(); }
      sm.Smooth(*x, *b, *ures, res_updated, update_res, x_zero);
    },
    py::arg("x"),
    py::arg("rhs"),
    py::arg("res") = nullptr,
    py::arg("res_updated") = false,
    py::arg("update_res") = false,
    py::arg("x_zero") = false);
  smc.def("SmoothK", [](BaseSmoother & sm, int steps, shared_ptr<BaseVector> x, shared_ptr<BaseVector> b, shared_ptr<BaseVector> res,
                        bool res_updated, bool update_res, bool x_zero)
    {
      shared_ptr<BaseVector> ures = res;
      if ( ures == nullptr )
        { ures = sm.CreateColVector(); }
      sm.SmoothK(steps, *x, *b, *ures, res_updated, update_res, x_zero);
    },
    py::arg("steps") = 1,
    py::arg("x"),
    py::arg("rhs"),
    py::arg("res") = nullptr,
    py::arg("res_updated") = false,
    py::arg("update_res") = false,
    py::arg("x_zero") = false);
  
  smc.def("SmoothBack", [](BaseSmoother & sm, shared_ptr<BaseVector> x, shared_ptr<BaseVector> b, shared_ptr<BaseVector> res,
                           bool res_updated, bool update_res, bool x_zero)
    {
      shared_ptr<BaseVector> ures = res;
      if ( ures == nullptr )
        { ures = sm.CreateColVector(); }
      sm.SmoothBack(*x, *b, *ures, res_updated, update_res, x_zero);
    },
    py::arg("x"),
    py::arg("rhs"),
    py::arg("res") = nullptr,
    py::arg("res_updated") = false,
    py::arg("update_res") = false,
    py::arg("x_zero") = false);
  
  smc.def("SmoothBackK", [](BaseSmoother & sm, int steps, shared_ptr<BaseVector> x, shared_ptr<BaseVector> b, shared_ptr<BaseVector> res,
                            bool res_updated, bool update_res, bool x_zero)
    {
      shared_ptr<BaseVector> ures = res;
      if ( ures == nullptr )
        { ures = sm.CreateColVector(); }
      sm.SmoothBackK(steps, *x, *b, *ures, res_updated, update_res, x_zero);
    },
    py::arg("steps") = 1,
    py::arg("x"),
    py::arg("rhs"),
    py::arg("res") = nullptr,
    py::arg("res_updated") = false,
    py::arg("update_res") = false,
    py::arg("x_zero") = false);
  
  smc.def("Print", [](shared_ptr<BaseSmoother> self) { self->PrintTo(cout); });
  
  smc.def_property_readonly("sys_mat", [](shared_ptr<BaseSmoother> self) { return self->GetAMatrix(); });
  
  smc.def("PrintPointers", [](shared_ptr<BaseSmoother> self, shared_ptr<BaseMatrix> amat) {
    cout << endl << " --- " << endl;
    cout << " smoother " << self << endl;
    cout << " smoother pds " << self->GetParallelDofs() << endl;
    cout << " smoother sys mat " << self->GetAMatrix() << endl;
    cout << " smoother sys mat pds " << self->GetAMatrix()->GetParallelDofs() << endl;
    if (amat) {
      cout << " amat " << amat << endl;
      cout << " amat pds " << amat->GetParallelDofs() << endl;
    }
    cout << " --- " << endl;
  }, py::arg("mat") = nullptr);

  auto proxy_smc = py::class_<ProxySmoother, shared_ptr<ProxySmoother>, BaseSmoother>(m, "ProxySmoother", "");
  proxy_smc.def(py::init<>([](shared_ptr<BaseSmoother> sm, int nsteps, bool symm)
    {
      return make_shared<ProxySmoother>(sm, nsteps, symm);
    }),
    py::arg("smoother"),
    py::arg("nsteps") = 1,
    py::arg("symm") = false
  );


  auto hpt_smc = py::class_<HiptMairSmoother, shared_ptr<HiptMairSmoother>, BaseSmoother>(m, "HiptMairSmoother", "");
  hpt_smc.def(py::init<>([](shared_ptr<BaseSmoother> pot_smoother, shared_ptr<BaseSmoother> range_smoother,
          shared_ptr<BaseMatrix> pot_mat, shared_ptr<BaseMatrix> range_mat,
          shared_ptr<BaseMatrix> D, shared_ptr<BaseMatrix> DT) {
          if (DT == nullptr)
            { D = make_shared<Transpose>(D); }
          return make_shared<HiptMairSmoother>(pot_smoother, range_smoother, pot_mat, range_mat, D, DT);
        }),
        py::arg("pot_smoother"),
        py::arg("range_smoother"),
        py::arg("pot_mat"),
        py::arg("range_mat"),
        py::arg("D"),
        py::arg("DT") = nullptr);


  m.def("CreateHybridGSS", [](shared_ptr<BaseMatrix> A, shared_ptr<BitArray> freedofs,
      bool pinv, bool overlap, bool in_thread,
      bool symm, bool symm_loc, int nsteps, int nsteps_loc) { // -> shared_ptr<BaseSmoother> {
    auto parA = dynamic_pointer_cast<ParallelMatrix>(A);
    auto locA = (parA == nullptr) ? A : parA->GetMatrix();
    auto spA = dynamic_pointer_cast<BaseSparseMatrix>(locA);
    if (spA == nullptr)
      { throw Exception("Need a SPARSE matrix for HybridGSS!"); }
    shared_ptr<BaseSmoother> smoother;
    Switch<MAX_SYS_DIM>
      ( GetEntryDim(spA.get())-1, [&] (auto BSM)
  {
    constexpr int BS = BSM + 1;
    typedef typename strip_mat<Mat<BS, BS, double>>::type TM;
    if constexpr ( (BS == 0) || (BS == 4) || (BS == 5)
#ifndef ELASTICITY
        || (BS == 6)
#endif
        ) {
        throw Exception("Smoother for that dim is not compiled!!");
        return;
      }
    else {
      if (parA == nullptr) {
        auto spm = static_pointer_cast<SparseMatrix<TM>>(spA);
        smoother = make_shared<GSS3<TM>>(spm, freedofs, pinv);
      }
      else {
        auto pds = parA->GetParallelDofs();
        // auto eqc_h = make_shared<EQCHierarchy> (pds);
        smoother = make_shared<HybridGSSmoother<TM>> (parA, freedofs, pinv, overlap, in_thread, symm_loc, nsteps_loc);
        smoother->Finalize();
      }
    }
  } );
    if ( (nsteps > 1) || symm )
      { smoother = make_shared<ProxySmoother>(smoother, nsteps, symm); }
    // cout << " CHGSS, smoother = " << endl;
    // smoother->PrintTo(cout, "  ");
    return smoother;
  },
  py::arg("mat") = nullptr,
  py::arg("freedofs") = nullptr,
  py::arg("pinv") = false,
  py::arg("NG_MPI_overlap") = false,
  py::arg("NG_MPI_thread") = false,
  py::arg("symm") = false,
  py::arg("symm_loc") = false,
  py::arg("nsteps") = 1,
  py::arg("nsteps_loc") = 1
  ); // CreateHybridGSS


  m.def("CreateHybridBlockGSS", [](shared_ptr<BaseMatrix> A, py::object blocks, bool NG_MPI_overlap,
            bool NG_MPI_thread, bool shm, bool sl2, bool bs2, bool pinv, bool blocks_no,
            bool symm, bool symm_loc, int nsteps, int nsteps_loc)
  {
    auto parA = dynamic_pointer_cast<ParallelMatrix>(A);
    auto locA = (parA == nullptr) ? A : parA->GetMatrix();
    auto spA = dynamic_pointer_cast<BaseSparseMatrix>(locA);
    if (spA == nullptr)
      { throw Exception("Need a SPARSE matrix for HybridGSS!"); }

    Table<int> * blocktable;
    {
      py::gil_scoped_acquire aq;
      size_t size = py::len(blocks);

      Array<int> cnt(size);
      size_t i = 0;
      for (auto block : blocks)
  cnt[i++] = py::len(block);

      i = 0;
      blocktable = new Table<int>(cnt);
      for (auto block : blocks)
  {
    auto row = (*blocktable)[i++];
    size_t j = 0;
    for (auto val : block)
      row[j++] = val.cast<int>();
  }
    }

    shared_ptr<BaseSmoother> smoother;
    Switch<MAX_SYS_DIM>
      ( GetEntryDim(spA.get())-1, [&] (auto BSM)
  {
    constexpr int BS = BSM + 1;
    typedef typename strip_mat<Mat<BS, BS, double>>::type TM;
    if constexpr ( (BS == 0) || (BS == 4) || (BS == 5)
#ifndef ELASTICITY
        || (BS == 6)
#endif
        ) {
        throw Exception("Smoother for that dim is not compiled!!");
        return;
      }
    else {
      if (parA == nullptr) {
        auto spm = static_pointer_cast<SparseMatrix<TM>>(spA);
        if (bs2)
    { smoother = make_shared<BSmoother2<TM>>(spm, std::move(*blocktable), shm, sl2, pinv, blocks_no); }
        else
    { smoother = make_shared<BSmoother<TM>>(spm, std::move(*blocktable), shm, sl2, pinv); }
      }
      else {
        auto pds = parA->GetParallelDofs();
        // auto eqc_h = make_shared<EQCHierarchy> (pds);
        smoother = make_shared<HybridBS<TM>>(parA, std::move(*blocktable), pinv, NG_MPI_overlap, NG_MPI_thread, shm, sl2,
                bs2, blocks_no, symm_loc, nsteps_loc);
      }
    }
  } );
    if ( (nsteps > 1) || symm )
      { smoother = make_shared<ProxySmoother>(smoother, nsteps, symm); }
    return smoother;
  },
  py::arg("mat"),
  py::arg("blocks"),
  py::arg("NG_MPI_overlap") = false,
  py::arg("NG_MPI_thread") = false,
  py::arg("shm") = true,
  py::arg("sl2") = true,
  py::arg("bs2") = false,
  py::arg("pinv") = false,
  py::arg("blocks_no") = false,
  py::arg("symm") = false,
  py::arg("symm_loc") = false,
  py::arg("nsteps") = 1,
  py::arg("nsteps_loc") = 1
  ); // CreateHybridBlockGSS


  m.def("CreateHybridDISmoother", [](shared_ptr<BaseMatrix> A, shared_ptr<BitArray> freedofs, bool NG_MPI_overlap, bool NG_MPI_thread,
              bool symm, int nsteps) {
    auto parA = dynamic_pointer_cast<ParallelMatrix>(A);
    auto locA = (parA == nullptr) ? A : parA->GetMatrix();
    auto spA = dynamic_pointer_cast<BaseSparseMatrix>(locA);
    if (spA == nullptr)
      { throw Exception("Need a SPARSE matrix for HybridGSS!"); }
    shared_ptr<BaseSmoother> smoother;
    if (parA != nullptr) {
      Switch<MAX_SYS_DIM>
      ( GetEntryDim(spA.get())-1, [&] (auto BSM)
  {
    constexpr int BS = BSM + 1;
    typedef typename strip_mat<Mat<BS, BS, double>>::type TM;
    if constexpr ( (BS == 0) || (BS == 4) || (BS == 5)
#ifndef ELASTICITY
        || (BS == 6)
#endif
        ) {
        throw Exception("Smoother for that dim is not compiled!!");
        return nullptr;
      }
    else {
      auto pds = parA->GetParallelDofs();
      smoother =  make_shared<HybridDISmoother<TM>>(A, freedofs, NG_MPI_overlap, NG_MPI_thread);
    }
  });
    }
    else {
      auto inv = A->InverseMatrix(freedofs);
      smoother =  make_shared<RichardsonSmoother>(A, inv, 1.0);
    }
    if ( (nsteps > 1) || symm )
      { smoother = make_shared<ProxySmoother>(smoother, nsteps, symm); }
    return smoother;
  },
  py::arg("mat"),
  py::arg("freedofs") = nullptr,
  py::arg("NG_MPI_overlap") = false,
  py::arg("NG_MPI_thread") = false,
  py::arg("symm") = false,
  py::arg("nsteps") = 1
  ); // CreateHybridDISmoother


  m.def("CreateJacobiSmoother", [](shared_ptr<BaseMatrix> A, shared_ptr<BitArray> freedofs) -> shared_ptr<BaseSmoother>
    {
      shared_ptr<BaseSmoother> smoother;
      auto [Aloc, pardofs] = MatrixToLocalComponents(A);
      auto spm = dynamic_pointer_cast<BaseSparseMatrix>(Aloc);
      if (spm == nullptr)
        { throw Exception("Need a sparse matrix for Jacobi!"); }
      Switch<MAX_SYS_DIM> (GetEntryDim(spm.get())-1, [&] (auto BSM) {
        constexpr int BS = BSM + 1;
        if constexpr(!isSmootherSupported<BS>()) {
          throw Exception("Smoother for that dim is not compiled!!");
          return;
        } else {
          using TM = typename strip_mat<Mat<BS, BS, double>>::type;
          auto tm_spm = static_pointer_cast<SparseMatrixTM<TM>>(spm);
          smoother = make_shared<JacobiSmoother<TM>>(tm_spm, freedofs);
          smoother->Finalize();
          return;
        }
      });
      return smoother;
    },
    py::arg("mat"),
    py::arg("freedofs") = nullptr
  );

  m.def("CreateDynBlockSmoother", [](shared_ptr<BaseMatrix> A,
                                     shared_ptr<BitArray>  freedofs,
                                     bool const &NG_MPI_overlap,
                                     bool const &NG_MPI_thread) -> shared_ptr<BaseSmoother>
  {
    shared_ptr<BaseSmoother> sm = nullptr;

    cout << " A = " << A << endl;

    if (auto dynSPA = dynamic_pointer_cast<DynBlockSparseMatrix<double>>(A))
    {
      cout << " dynSPA = " << dynSPA << endl;
      sm = make_shared<DynBlockSmoother<double>>(dynSPA, freedofs);
    }
    else if (auto sparseA = dynamic_pointer_cast<SparseMatrix<double>>(A))
    {
      auto dynSPA = make_shared<DynBlockSparseMatrix<double>>(*sparseA);
      sm = make_shared<DynBlockSmoother<double>>(dynSPA, freedofs);
    }
    else if (auto parA = dynamic_pointer_cast<ParallelMatrix>(A))
    {
      auto hybSPA = make_shared<DynamicBlockHybridMatrix<double>>(A);
      sm = make_shared<HybridDynBlockSmoother<double>>(parA, freedofs, 1, NG_MPI_thread, NG_MPI_overlap);
    }

    return sm;
  },
  py::arg("mat"),
  py::arg("freedofs") = nullptr,
  py::arg("NG_MPI_overlap") = true,
  py::arg("NG_MPI_thread") = true);

} // ExportSmoothers

} // namespace amg
