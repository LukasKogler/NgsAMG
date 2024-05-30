#include "dyn_block.hpp"
#include "universal_dofs.hpp"
#include "utils_sparseLA.hpp"
#include "utils_sparseMM.hpp"

#include "python_amg.hpp"
#include <memory>

namespace amg
{

extern shared_ptr<BaseMatrix> restrictMatrixToBlocks(shared_ptr<BaseMatrix> A,
                                                      FlatTable<int> rowBlocks,
                                                      FlatTable<int> colBlocks,
                                                      double const &tol = 1e-20);

extern
void
AMGBFCheck(shared_ptr<BaseMatrix> const &A,
           shared_ptr<BaseMatrix> const &M,
           double     const &thresh = 1e-10);


extern
shared_ptr<SparseMatrix<double>>
makeLOToHOFacetEmbedding(shared_ptr<FESpace> orig,
                         shared_ptr<FESpace> dest,
                         shared_ptr<BitArray> destFree = nullptr);

void ExportUtils (py::module &m)
{
  m.def("SparseMM", [&](shared_ptr<BaseMatrix> A,
                                 shared_ptr<BaseMatrix> B) -> shared_ptr<BaseMatrix>
  {
    return MatMultABGeneric(A,B);
  });

  m.def("ToSparseMatrix", [&](shared_ptr<BaseMatrix> A,
                                       bool const &compress,
                                       double const &compressTol) -> shared_ptr<BaseMatrix>
  {
    auto spA = BaseMatrixToSparse(A);
    if (compress)
    {
      spA = CompressAGeneric(spA);
    }
    return spA;
  },
  py::arg("mat"),
  py::arg("compress") = false,
  py::arg("tol") = 1e-20);

  m.def("CompressSparseMatrix", [&](shared_ptr<BaseMatrix> A,
                                             double const &compressTol) -> shared_ptr<BaseMatrix>
  {
    return CompressAGeneric(A, compressTol);
  },
  py::arg("mat"),
  py::arg("tol") = 1e-20);

  m.def("RestrictMatrixToBlocks", [&](shared_ptr<BaseMatrix> A,
                                                py::object rowBlocks,
                                                py::object colBlocks,
                                                double const &compressTol) -> shared_ptr<BaseMatrix>
  {
    Table<int> * rblocktable;
    {
      py::gil_scoped_acquire aq;
      size_t size = py::len(rowBlocks);

      Array<int> cnt(size);
      size_t i = 0;
      for (auto block : rowBlocks)
        { cnt[i++] = py::len(block); }

      i = 0;
      rblocktable = new Table<int>(cnt);
      for (auto block : rowBlocks)
      {
        auto row = (*rblocktable)[i++];
        size_t j = 0;
        for (auto val : block)
          { row[j++] = val.cast<int>(); }
      }
    }

    Table<int> * cblocktable;
    {
      py::gil_scoped_acquire aq;
      size_t size = py::len(colBlocks);

      Array<int> cnt(size);
      size_t i = 0;
      for (auto block : colBlocks)
        { cnt[i++] = py::len(block); }

      i = 0;
      cblocktable = new Table<int>(cnt);
      for (auto block : colBlocks)
      {
        auto row = (*cblocktable)[i++];
        size_t j = 0;
        for (auto val : block)
          { row[j++] = val.cast<int>(); }
      }
    }

    return restrictMatrixToBlocks(A, *rblocktable, *cblocktable, compressTol);
  },
  py::arg("mat"),
  py::arg("row_blocks"),
  py::arg("col_blocks"),
  py::arg("tol") = 1e-20);


  m.def("LOToHOFacetEmbedding", [&](shared_ptr<FESpace> orig,
                                             shared_ptr<FESpace> dest,
                                             shared_ptr<BitArray> destFree) -> shared_ptr<BaseMatrix>
  {
    shared_ptr<SparseMatrix<double>> emb = makeLOToHOFacetEmbedding(orig, dest, destFree);
    return emb;
  },
  py::arg("orig"),
  py::arg("dest"),
  py::arg("destFree") = nullptr);


  // auto dynC = py::class_<DynBlockSparseMatrix<double>, shared_ptr<BaseMatrix>, BaseMatrix>(m, "DynBlockSparseMatrix", "Dynamic blocked sparse matrix");
  
  m.def("ConvertDynBlock", [](shared_ptr<BaseMatrix> A) -> shared_ptr<BaseMatrix>
  {
    if (A == nullptr)
    {
      return nullptr;
    }

    auto [rowUD, colUD, locMat, opType] = UnwrapParallelMatrix(A);

    auto dynA = ConvertToDynSPM(*locMat);

    cout << " -> dynA = " << dynA << endl;

    return WrapParallelMatrix(dynA, rowUD, colUD, opType);
  }, py::arg("mat"));


  m.def("GetMemoryUse", [](shared_ptr<BaseMatrix> A) -> size_t
  {
    if (A == nullptr)
    {
      return 0;
    }

    cout << " GetMemoryUse, unwrap " << endl;
    auto [rowUD, colUD, locMat, opType] = UnwrapParallelMatrix(A);

    cout << " GetMemoryUse, unwrap OK, locMat = " << locMat << endl;
    size_t locMU;

    if (auto spA = dynamic_cast<SparseMatrixTM<double> const *>(locMat.get()))
    {
      // sparse-mat getmemoryuse does not count matrix-graph data for assembled matrices,
      // the matrix-graph must be shared/owned by someone else.
      // for our purposes here we WANT it to count so we can compare memory use
      locMU = spA->NZE() * (sizeof(double) + sizeof(int)) + spA->Height() * sizeof(size_t);
    }
    else
    {
      auto mUse = locMat->GetMemoryUsage();

      // cout << " MUSE parts, s = " << mUse.Size() << ": " << endl;
      // for (auto j : Range(mUse))
      // {
      //   cout << mUse[j].Name() << ": " << mUse[j].NBytes() << ", in MB = " << mUse[j].NBytes()/1024.0/1024 << endl;
      // }

      locMU = std::accumulate(mUse.begin(), mUse.end(), size_t(0), [&](auto const &partialSum, auto const &mU) { return partialSum + mU.NBytes(); });
    }


    return colUD.GetCommunicator().AllReduce(locMU, MPI_SUM);
  }, py::arg("mat"));


  m.def("AMGBFCheck", [](shared_ptr<BaseMatrix> A,
                         shared_ptr<BaseMatrix> M,
                         double tol) -> void
  {
    AMGBFCheck(A, M, tol);
  },
  py::arg("A"),
  py::arg("M"),
  py::arg("tol") = 1e-10);

}

} // namespace amg

