#include <solve.hpp>

using namespace ngsolve;

#include <python_ngstd.hpp>

#include "amg.hpp"

#include "amg_smoother.hpp"
#include "amg_smoother3.hpp"
#include "amg_blocksmoother.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

#include <python_ngstd.hpp>

namespace amg
{

  void ExportSmoothers (py::module & m)
  {
    auto smc = py::class_<BaseSmoother, shared_ptr<BaseSmoother>, BaseMatrix>(m, "BaseSmoother", "");
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



    m.def("CreateHybridGSS", [](shared_ptr<BaseMatrix> A, shared_ptr<BitArray> freedofs,
				bool overlap, bool in_thread) { // -> shared_ptr<BaseSmoother> {
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
		      smoother = make_shared<GSS3<TM>>(spm, freedofs);
		    }
		    else {
		      auto pds = parA->GetParallelDofs();
		      auto eqc_h = make_shared<EQCHierarchy> (pds);
		      smoother = make_shared<HybridGSS3<TM>> (parA, eqc_h, freedofs, false, overlap, in_thread);
		    }
		  }
		} );
	    return smoother;
	  },
	  py::arg("mat") = nullptr,
	  py::arg("freedofs") = nullptr,
	  py::arg("mpi_overlap") = false,
	  py::arg("mpi_thread") = false
	  ); // CreateHybridGSS


    m.def("CreateHybridBlockGSS", [](shared_ptr<BaseMatrix> A, py::object blocks, bool mpi_overlap,
				     bool mpi_thread, bool shm, bool sl2)
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
		      smoother = make_shared<BSmoother<TM>>(spm, move(*blocktable), shm, sl2);
		    }
		    else {
		      auto pds = parA->GetParallelDofs();
		      auto eqc_h = make_shared<EQCHierarchy> (pds);
		      smoother = make_shared<HybridBS<TM>>(parA, eqc_h, move(*blocktable), mpi_overlap, mpi_thread, shm, sl2);
		    }
		  }
		} );
	    return smoother;
	  },
	  py::arg("mat"),
	  py::arg("blocks"),
	  py::arg("mpi_overlap") = false,
	  py::arg("mpi_thread") = false,
	  py::arg("shm") = true,
	  py::arg("sl2") = true
	  ); // CreateHybridBlockGSS

  } // ExportSmoothers

} // namespace amg
