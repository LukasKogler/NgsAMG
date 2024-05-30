#ifndef FILE_UNIVERSAL_DOFS_HPP
#define FILE_UNIVERSAL_DOFS_HPP

#include <base.hpp>

namespace amg
{

bool IsRankZeroIdle(ParallelDofs const &parDOFs);
bool IsRankZeroIdle(shared_ptr<ParallelDofs> const &parDOFs);
bool IsRankZeroIdle(ParallelDofs const *parDOFs);

shared_ptr<ParallelDofs>
CreateParallelDOFs(NgMPI_Comm acomm,
                   Table<int> && adist_procs, 
                	 int dim = 1,
                   bool iscomplex = false,
                   bool isRankZeroIdle = false);
      
/**
 *  This class encapsulates information about a set of DOFs,
 *  i.e their number, block-size, the communicator they live in, 
 *  and, if parallel, the parallel-dofs.
 * 
 *  This is useful to:
 *     i) reduce the number of  "if (parallel_dofs != nullptr)" checks
 *    ii) have some object we can hand around to get block-size and NDOF
 *        out of when we are in serial and parallel_dofs are nullptr
 *   iii) (probably) have some place to store an "NgsAMG_Comm", as opposed to an Ngs_Comm
 *        instead of either re-creating it all the time or storing it in other places
*/
class UniversalDofs
{
public:
  UniversalDofs(bool valid, NgsAMG_Comm comm, shared_ptr<ParallelDofs> pds, size_t N, size_t BS)
    : _valid(valid)
    , _comm(comm)
    , _pds(pds)
    , _N(N)
    , _BS(BS)
  {
    _idleRankZero = ( IsValid() && IsParallel() ) ? ::amg::IsRankZeroIdle(GetParallelDofs()) : false;
    _isTrulyParallel = GetCommunicator().Size() > ( _idleRankZero ? 2 : 1 );
  }

  UniversalDofs(shared_ptr<ParallelDofs> pds, size_t N, size_t BS)
    : UniversalDofs(true,
                    pds != nullptr ? NgsAMG_Comm(pds->GetCommunicator()) : NgsAMG_Comm(),
                    pds,
                    N,
                    BS)
  { ; }

  UniversalDofs(size_t N, size_t BS)
    : UniversalDofs(true, NgsAMG_Comm(), nullptr, N, BS)
  { ; }

  UniversalDofs(shared_ptr<ParallelDofs> pds)
    : UniversalDofs(pds != nullptr,
                    pds != nullptr ? NgsAMG_Comm(pds->GetCommunicator()) : NgsAMG_Comm(),
                    pds,
                    pds != nullptr ? pds->GetNDofLocal() : -1,
                    pds != nullptr ? pds->GetEntrySize() : -1)
  { ; }

  UniversalDofs()
    : UniversalDofs(false, NgsAMG_Comm(), nullptr, -1, -1)
  { ; }


  ~UniversalDofs () = default;

  INLINE bool IsRankZeroIdle         () const { return _idleRankZero; }
  INLINE bool IsValid                () const { return _valid; }
  INLINE size_t GetND                () const { return _N; }
  INLINE size_t GetNDGlob            () const { return (_pds == nullptr) ? _N : _pds->GetNDofGlobal(); }
  INLINE size_t GetBS                () const { return _BS; }
  INLINE size_t GetNDScal            () const { return GetND() * GetBS();  }
  INLINE size_t GetNDScalGlob        () const { return GetNDGlob() * GetBS();  }
  // INLINE NgsAMG_Comm GetCommunicator () const { return (_pds == nullptr) ? NgsAMG_Comm() : _pds->GetCommunicator(); }
  INLINE NgsAMG_Comm GetCommunicator () const { return _comm; }
  INLINE bool IsParallel             () const { return GetCommunicator().Size() > 1; }
  INLINE bool IsTrulyParallel        () const { return _isTrulyParallel; }

  INLINE shared_ptr<ParallelDofs> const &GetParallelDofs () const { return _pds; }
  INLINE shared_ptr<ParallelDofs>       &GetParallelDofs ()       { return _pds; }

  /** Some of the ParllelDofs methods so we can easily swap in UDofs instead of ParallelDofs **/
  INLINE size_t GetNDofLocal  () const { return _N; }
  INLINE size_t GetNDofGlobal () const { return GetNDGlob(); }
  INLINE FlatArray<int> GetDistantProcs (int dof) const { return (_pds != nullptr) ? _pds->GetDistantProcs(dof) : FlatArray<int>(); }
  INLINE FlatArray<int> GetDistantProcs ()        const { return (_pds != nullptr) ? _pds->GetDistantProcs()    : FlatArray<int>(); }
  INLINE bool IsMasterDof (int dof) const { return (_pds == nullptr) || _pds->IsMasterDof(dof); }

  INLINE bool operator== (const UniversalDofs & other) const
  {
    return (_valid == other._valid) && (_pds == other._pds ) &&
           (_N == other._N) && (_BS == other._BS) &&
           (_idleRankZero == other._idleRankZero) && (_isTrulyParallel == other._isTrulyParallel);
  }

  unique_ptr<BaseVector> CreateVector(PARALLEL_STATUS stat = DISTRIBUTED) const;
  shared_ptr<BaseVector> CreateSPVector(PARALLEL_STATUS stat = DISTRIBUTED) const;

  void printToStream (ostream &os, bool printPDs = false) const;

private:
  bool _valid;
  NgsAMG_Comm _comm;
  shared_ptr<ParallelDofs> _pds;
  size_t _N;
  size_t _BS;

  bool _idleRankZero;
  bool _isTrulyParallel;
}; // class UniversalDofs


enum DOF_SPACE : unsigned char
{
  ROWS = 0,
  COLS = 1
};


UniversalDofs MatToUniversalDofs (BaseMatrix const &A, DOF_SPACE SPACE = DOF_SPACE::ROWS);

// returnsL (input/row-UD, output/col-UD, local-mat, op-type)
std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (BaseMatrix const &A);
std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (shared_ptr<BaseMatrix> A);
std::tuple<UniversalDofs, UniversalDofs, shared_ptr<BaseMatrix>, PARALLEL_OP>
UnwrapParallelMatrix (BaseMatrix const *A);

shared_ptr<BaseMatrix> WrapParallelMatrix (shared_ptr<BaseMatrix> localMat, UniversalDofs rowUDofs, UniversalDofs colUDofs, PARALLEL_OP op);


INLINE int rankInMatComm(BaseMatrix const &A)
{
  return A.GetCommunicator() ? A.GetCommunicator()->Rank() : 0;
} 

INLINE ostream &operator << (ostream &os, UniversalDofs const &uDofs)
{
  uDofs.printToStream(os);
  return os;
}

} // namespace amg

#endif // FILE_UNIVERSAL_DOFS_HPP
