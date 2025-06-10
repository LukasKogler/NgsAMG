#ifndef FILE_DOF_CONTRACT_HPP
#define FILE_DOF_CONTRACT_HPP

#include <base.hpp>

#include "dof_map.hpp"

namespace amg
{

template<class TV>
class CtrMap : public BaseDOFMapStep,
               public std::enable_shared_from_this<CtrMap<TV>>
{
public:

  using TM = typename strip_mat<Mat<VecHeight<TV>(), VecHeight<TV>(), typename mat_traits<TV>::TSCAL>>::type;
  using TSPM = stripped_spm<TM>;
  using TSPM_TM = stripped_spm_tm<TM>;

  CtrMap (UniversalDofs originalDofs, UniversalDofs mappedDofs, Array<int> && group, Table<int> && dof_maps);
  ~CtrMap ();
  
  virtual void TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const override;
  virtual void AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const override;
  virtual void TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const override;
  virtual void AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const override;
  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

  virtual shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const override
  {
    // TODO: static cast this??
    shared_ptr<TSPM> spm = dynamic_pointer_cast<TSPM>(mat);
    if (spm == nullptr)
      { throw Exception("CtrMap cast did not work!!"); }
    return DoAssembleMatrix(spm);
  }
  
  virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;
  
  bool DoSwap (bool in);

  // prol after contract becomes new prol before new contract
  shared_ptr<ProlMap<TM>> SwapWithProl (shared_ptr<ProlMap<TM>> pm);

  virtual shared_ptr<BaseDOFMapStep> PullBack (shared_ptr<BaseDOFMapStep> other) override;

  int GetBlockSize ()       const override;
  int GetMappedBlockSize () const override;

  INLINE bool IsMaster () const { return is_gm; }

  // TODO: bad hack because NgsAMG_Comm -> NG_MPI_Comm -> NgMPI_Comm in pardofs constructor (ownership lost!)
  NgsAMG_Comm _comm_keepalive_hack;

  /** Allocates MPI-Buffers and sets up MPI-types **/
  void SetUpMPIStuff ();

protected:

  shared_ptr<TSPM> DoAssembleMatrix (shared_ptr<TSPM> mat) const;
  // using BaseDOFMapStep::pardofs, BaseDOFMapStep::mapped_pardofs;
  using BaseDOFMapStep::_originDofs, BaseDOFMapStep::_mappedDofs;

  Array<int> group;
  int master;
  bool is_gm;
  Table<int> dof_maps;
  Array<NG_MPI_Request> reqs;
  Array<NG_MPI_Datatype> NG_MPI_types;
  Table<TV> buffers;
}; // class CtrMap

} // namespace amg

#endif // FILE_DOF_CONTRACT_HPP