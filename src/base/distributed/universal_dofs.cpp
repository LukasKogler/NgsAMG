
#include <base.hpp>

#include <utils_sparseLA.hpp>
#include <utils_io.hpp>

#include "universal_dofs.hpp"
#include "hybrid_matrix.hpp"

namespace amg
{

bool IsRankZeroIdle(ParallelDofs const &parDOFs)
{
#ifdef NGS_COMPATIBILITY
  return parDOFs.GetCommunicator().Size() > 1;
#else
  return parDOFs.IsRankZeroIdle();
#endif
}

bool IsRankZeroIdle(shared_ptr<ParallelDofs> const &parDOFs)
{
  return (parDOFs != nullptr) && IsRankZeroIdle(*parDOFs);
}

bool IsRankZeroIdle(ParallelDofs const *parDOFs)
{
  return (parDOFs != nullptr) && IsRankZeroIdle(*parDOFs);
}

shared_ptr<ParallelDofs>
CreateParallelDOFs(NgMPI_Comm comm,
                   Table<int> && adist_procs, 
                	 int dim,
                   bool iscomplex,
                   bool isRankZeroIdle)
{
#ifdef NGS_COMPATIBILITY
  if (!isRankZeroIdle)
  {
    if (comm.Size() > 1)
    {
      throw Exception("Idle-Rank-Zero in parallel not available!");
    }
    else
    {
      std::cout << "  WARNING! Idle-Rank-Zero mode not available!" << std::endl;
    }
  }
  return make_shared<ParallelDofs>(comm, std::move(adist_procs), dim, iscomplex);
#else
  return make_shared<ParallelDofs>(comm, std::move(adist_procs), dim, iscomplex, isRankZeroIdle);
#endif
}

/** UniversalDofs **/

unique_ptr<BaseVector> UniversalDofs :: CreateVector (PARALLEL_STATUS stat) const
{
    return IsValid() ? CreateSuitableVector(_N, _BS, _pds, stat) : nullptr;
} // UniversalDofs::CreateVector


shared_ptr<BaseVector> UniversalDofs :: CreateSPVector (PARALLEL_STATUS stat) const
{
    return IsValid() ? CreateSuitableSPVector(_N, _BS, _pds, stat) : nullptr;
} // UniversalDofs::CreateSPVector


void UniversalDofs :: printToStream (ostream &os, bool printPDs) const
{
  if (IsValid()) {
    os << " UniversalDofs @ " << this << std::endl;
    os << "   ND loc   = " << GetND()      << std::endl;
    os << "   ND glob  = " << GetNDGlob()  << std::endl;
    os << "   BS       = " << GetBS()      << std::endl;
    os << "   parallel = " << IsParallel() << std::endl;
    os << "   rk0 idle = " << IsRankZeroIdle() << std::endl;
    if (GetParallelDofs() != nullptr) { 
      os << "   pardofs @ " << GetParallelDofs() << std::endl;
      if (printPDs)
        { os << *GetParallelDofs() << std::endl; }
    }
    else
    {
      os << "  UniversalDofs without ParallelDofs! " << std::endl;
    }
  }
  else
    { os << " UniversalDofs @ " << this << ", DUMMY!!" << std::endl; }
} // UniversalDofs::printToStream


std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (BaseMatrix const *A)
{
  if (A == nullptr)
  {
    return std::make_tuple(UniversalDofs(), UniversalDofs(), nullptr, PARALLEL_OP::C2D);
  }

  shared_ptr<ParallelDofs>        rowPDs = nullptr;
  shared_ptr<ParallelDofs>        colPDs = nullptr;
  BaseMatrix               const *locMat = A;
  PARALLEL_OP                     op     = C2C;

  if (auto pmat = dynamic_cast<ParallelMatrix const*>(A))
  {
    locMat = pmat->GetMatrix().get();
    rowPDs = pmat->GetRowParallelDofs();
    colPDs = pmat->GetColParallelDofs();
    op     = pmat->GetOpType();

    return std::make_tuple(UniversalDofs(rowPDs, A->Width(),  rowPDs->GetEntrySize()),
                           UniversalDofs(colPDs, A->Height(), colPDs->GetEntrySize()),
                           const_cast<BaseMatrix&>(*locMat).SharedFromThis<BaseMatrix>(),
                           op);
  }
  else if (auto hybA = dynamic_cast<HybridBaseMatrix<double> const *>(A))
  {
    locMat = hybA->GetLocalOp().get();
    rowPDs = hybA->GetParallelDofs();
    colPDs = hybA->GetParallelDofs();
  
    return std::make_tuple(UniversalDofs(rowPDs, A->Width(),  rowPDs->GetEntrySize()),
                           UniversalDofs(colPDs, A->Height(), colPDs->GetEntrySize()),
                           const_cast<BaseMatrix&>(*locMat).SharedFromThis<BaseMatrix>(),
                           op);
  }
  else
  {
    return std::make_tuple(UniversalDofs(rowPDs, A->Width(),  GetEntryWidth(locMat)),
                           UniversalDofs(colPDs, A->Height(), GetEntryHeight(locMat)),
                           const_cast<BaseMatrix&>(*locMat).SharedFromThis<BaseMatrix>(),
                           op);
  }
}

// input-UD, outpu-ID, local-mat, op-type
std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (shared_ptr<BaseMatrix> A)
{
  return UnwrapParallelMatrix(A.get());

  // shared_ptr<ParallelDofs> rowPDs = nullptr;
  // shared_ptr<ParallelDofs> colPDs = nullptr;
  // shared_ptr<BaseMatrix>   locMat = A;
  // PARALLEL_OP              op     = C2C;

  // if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(A))
  // {
  //   locMat = pmat->GetMatrix();
  //   rowPDs = pmat->GetRowParallelDofs();
  //   colPDs = pmat->GetColParallelDofs();
  //   op     = pmat->GetOpType();

  //   return std::make_tuple(UniversalDofs(rowPDs, A->Width(),  rowPDs->GetEntrySize()),
  //                          UniversalDofs(colPDs, A->Height(), colPDs->GetEntrySize()),
  //                          locMat,
  //                          op);
  // }
  // else
  // {
  //   return std::make_tuple(UniversalDofs(rowPDs, A->Width(),  GetEntryWidth(locMat.get())),
  //                          UniversalDofs(colPDs, A->Height(), GetEntryHeight(locMat.get())),
  //                          locMat,
  //                          op);
  // }
}

// input-UD, outpu-ID, local-mat, op-type
std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (BaseMatrix const &A)
{
  return UnwrapParallelMatrix(&A);
}


UniversalDofs MatToUniversalDofs (BaseMatrix const &A, DOF_SPACE SPACE )
{
  shared_ptr<ParallelDofs> pds = nullptr;
  BaseMatrix const *locMat = &A;
  if (auto pmat = dynamic_cast<ParallelMatrix const*>(&A)) {
    pds = (SPACE == DOF_SPACE::ROWS) ? pmat->GetRowParallelDofs() : pmat->GetColParallelDofs();
    locMat = pmat->GetMatrix().get();
  }
  if (SPACE == DOF_SPACE::ROWS)
    { return UniversalDofs(pds, A.Width(),  GetEntryWidth(locMat)); }
  else
    { return UniversalDofs(pds, A.Height(), GetEntryHeight(locMat)); }
}


shared_ptr<BaseMatrix> WrapParallelMatrix(shared_ptr<BaseMatrix> localMat, UniversalDofs rowUDofs, UniversalDofs colUDofs, PARALLEL_OP op)
{
  if ( (colUDofs.IsParallel() != rowUDofs.IsParallel()) ||
       (colUDofs.GetND() != localMat->Height()) ||
       (rowUDofs.GetND() != localMat->Width()) )
  {
    cout << " RowUDofs: " << endl << rowUDofs << endl;
    cout << " colUDofs: " << endl << colUDofs << endl;
    cout << " localMat = " << localMat << endl;
    if (localMat) {
      auto &locRef = *localMat;
      cout << " localMat type = " << typeid(locRef).name() << endl;
      cout << " localMat dims = " << localMat->Height() << " x " << localMat->Width() << endl;
    }
    throw Exception("WrapParallelMatrix invalid input!");
  }

  if ( (!colUDofs.IsParallel()) || (!rowUDofs.IsParallel()) )
    { return localMat; }  

  return make_shared<ParallelMatrix>(localMat, rowUDofs.GetParallelDofs(), colUDofs.GetParallelDofs(), op);
} // WrapParallelMatrix

/** END UniversalDofs **/

} // namespace amg
