#ifndef FILE_DOF_MAP_HPP
#define FILE_DOF_MAP_HPP

#include <base.hpp>

#include <dyn_block.hpp>
#include <universal_dofs.hpp>
#include <utils_sparseLA.hpp> // for prolmap
#include <utils_sparseMM.hpp> // for prolmap
#include <utils_sparseLA.hpp>

namespace amg
{

/**
    At the minimum, we have to be able to:
    -) transfer vectors between levels
    -) construct the coarse level matrix
    -) be able to create vectors for both levels
**/
class BaseDOFMapStep
{
protected:
  UniversalDofs _originDofs, _mappedDofs;

  void SetUDofs(UniversalDofs &&uDofs)       { _originDofs = uDofs; }
  void SetMappedUDofs(UniversalDofs &&uDofs) { _mappedDofs = uDofs; }

public:
  BaseDOFMapStep(UniversalDofs originDofs,
                  UniversalDofs mappedDofs);

  BaseDOFMapStep()
    : BaseDOFMapStep(UniversalDofs(),
                      UniversalDofs())
  {}

  virtual ~BaseDOFMapStep () = default;

  virtual void TransferF2C (BaseVector const* x_fine, BaseVector *x_coarse) const = 0;
  virtual void AddF2C (double fac, BaseVector const* x_fine, BaseVector *x_coarse) const = 0;
  virtual void TransferC2F (BaseVector *x_fine, BaseVector const* x_coarse) const = 0;
  virtual void AddC2F (double fac, BaseVector *x_fine, BaseVector const* x_coarse) const = 0;

  unique_ptr<BaseVector> CreateVector () const;
  unique_ptr<BaseVector> CreateMappedVector () const;

  virtual shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const = 0;

  /** "me - other" -> "conc" **/
  virtual bool CanConcatenate (shared_ptr<BaseDOFMapStep> other) { return false; }
  virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) { return nullptr; }

  /** "me - other" -> "new other - new me" **/
  virtual shared_ptr<BaseDOFMapStep> PullBack (shared_ptr<BaseDOFMapStep> other) { return nullptr; }

  UniversalDofs const &GetUDofs () const { return _originDofs; }
  UniversalDofs const &GetMappedUDofs () const { return _mappedDofs; }

  shared_ptr<ParallelDofs> GetParallelDofs () const
    { return _originDofs.GetParallelDofs(); }

  shared_ptr<ParallelDofs> GetMappedParDofs () const
    { return _mappedDofs.GetParallelDofs() ; }

  virtual void Finalize () { ; }
  virtual void PrintTo (std::ostream & os, string prefix = "") const { ; }

  virtual int GetBlockSize ()       const = 0;
  virtual int GetMappedBlockSize () const = 0;
};

std::ostream & operator<<(std::ostream &os, const BaseDOFMapStep& p);


/**
    This maps DOFS between levels. We can add DOF-map-steps of various kind.
    Each step has to start where the last one left off.

    OLD INFO (now happens outside of class):
      With SetLevelCutoffs, we decide where to assemble matrices.
      The first, finest matrix is on level 0.
      The matrix on level k is assembled by steps 0 .. cutoff[k]-1
      Before actually assembling coarse level matrices, we try to concatenate
      steps where possible.
**/
class DOFMap
{
protected:
  Array<shared_ptr<BaseDOFMapStep>> steps;
  Array<size_t> cutoffs;
  bool finalized;
  int nsteps_glob;
  Array<shared_ptr<BaseVector>> vecs;

  // only built on demand!
  BitArray have_dnums;
  Array<Array<int>> glob_dnums;

public:

  DOFMap ();

  ~DOFMap () = default;

  void AddStep (const shared_ptr<BaseDOFMapStep> step);

  INLINE size_t GetNSteps () const { return steps.Size(); }
  INLINE size_t GetNLevels () const { return GetNSteps() + 1; } // we count one "empty" level if we drop

  shared_ptr<BaseDOFMapStep> GetStep (int k) const { return steps[k]; }

  void ReplaceStep (int k, shared_ptr<BaseDOFMapStep> newStep);

  void Finalize ();
  // void Finalize (FlatArray<size_t> acutoffs, shared_ptr<BaseDOFMapStep> embed_step);

  INLINE void TransferF2C (size_t lev_start, const BaseVector *x_fine, BaseVector *x_coarse) const
  { steps[lev_start]->TransferF2C(x_fine, x_coarse);}

  INLINE void AddF2C (size_t lev_start, double fac, const BaseVector *x_fine, BaseVector *x_coarse) const
  { steps[lev_start]->AddF2C(fac, x_fine, x_coarse);}

  INLINE void TransferC2F (size_t lev_dest, BaseVector *x_fine, BaseVector const* x_coarse) const
  { steps[lev_dest]->TransferC2F(x_fine, x_coarse);}

  INLINE void AddC2F (size_t lev_dest, double fac, BaseVector *x_fine, BaseVector const* x_coarse) const
  { steps[lev_dest]->AddC2F(fac, x_fine, x_coarse);}

  void TransferAtoB (int la, int lb, BaseVector const* vin, BaseVector * vout) const;

  unique_ptr<BaseVector> CreateVector (size_t l) const
  { return (l>steps.Size()) ? nullptr : ((l==steps.Size()) ? steps.Last()->CreateMappedVector() : steps[l]->CreateVector()); }

  UniversalDofs const& GetUDofs (size_t L = 0) const
  {
    static UniversalDofs dummyDofs;

    if ( L < steps.Size() )
    {
      return steps[L]->GetUDofs();
    }
    else if (L == steps.Size())
    {
      return steps.Last()->GetMappedUDofs();
    }
    else
    {
      return dummyDofs;
    }
    // return ( L == steps.Size() ) ? steps.Last()->GetMappedUDofs() : steps[L]->GetUDofs();
  }


  UniversalDofs const& GetMappedUDofs () const { return steps.Last()->GetMappedUDofs(); }

  shared_ptr<ParallelDofs> GetParDofs (size_t L = 0) const { return GetUDofs(L).GetParallelDofs(); }
  shared_ptr<ParallelDofs> GetMappedParDofs () const { return GetMappedUDofs().GetParallelDofs(); }

  Array<shared_ptr<BaseMatrix>> AssembleMatrices (shared_ptr<BaseMatrix> finest_mat) const;

  shared_ptr<DOFMap> SubMap (int from, int to = -1) const;

  shared_ptr<BaseDOFMapStep> ConcStep (int la, int lb, bool symbolic=true) const;

  FlatArray<int> GetGlobDNums (int level) const;
}; // DOFMap


/** Multiple steps that act as one **/
class ConcDMS : public BaseDOFMapStep
{
protected:
  Array<shared_ptr<BaseDOFMapStep>> sub_steps;
  Array<shared_ptr<BaseVector>> spvecs;
  Array<BaseVector*> vecs;

public:
  ConcDMS (FlatArray<shared_ptr<BaseDOFMapStep>> _sub_steps,
            FlatArray<shared_ptr<BaseVector>> _vecs = FlatArray<shared_ptr<BaseVector>>(0, nullptr));

  virtual ~ConcDMS () = default;

  virtual void TransferF2C (BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void AddF2C (double fac, BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void TransferC2F (BaseVector *x_fine, BaseVector const* x_coarse) const override;
  virtual void AddC2F (double fac, BaseVector *x_fine, BaseVector const* x_coarse) const override;

  virtual shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const override;

  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

  INLINE int GetNSteps () const { return sub_steps.Size(); }
  INLINE shared_ptr<BaseDOFMapStep> GetStep (int k) const { return sub_steps[k]; }
  INLINE FlatArray<shared_ptr<BaseDOFMapStep>> GetSteps () const { return sub_steps; }

  virtual void Finalize () override;
  // virtual bool CanPullBack (shared_ptr<BaseDOFMapStep> other) override;
  // virtual shared_ptr<BaseDOFMapStep> PullBack (shared_ptr<BaseDOFMapStep> other) override;

  virtual int GetBlockSize ()       const override;
  virtual int GetMappedBlockSize () const override;
};


class BaseProlMap : public BaseDOFMapStep
{
public:
  BaseProlMap (UniversalDofs originUDofs,
               UniversalDofs mappedUDofs)
    : BaseDOFMapStep(originUDofs, mappedUDofs)
  {}

  virtual ~BaseProlMap () = default;

  virtual shared_ptr<BaseMatrix> GetBaseProl      () const = 0;
  virtual shared_ptr<BaseMatrix> GetBaseProlTrans () const = 0;

  void TransferF2C (BaseVector const* x_fine, BaseVector *x_coarse)             const override;
  void AddF2C      (double fac, BaseVector const* x_fine, BaseVector *x_coarse) const override;
  void TransferC2F (BaseVector *x_fine, BaseVector const* x_coarse)             const override;
  void AddC2F      (double fac, BaseVector *x_fine, BaseVector const* x_coarse) const override;
};


class BaseSparseProlMap : public BaseProlMap
{
public:
  BaseSparseProlMap (UniversalDofs originUDofs,
                     UniversalDofs mappedUDofs)
    : BaseProlMap(originUDofs, mappedUDofs)
  {}

  virtual ~BaseSparseProlMap () = default;

  virtual shared_ptr<BaseSparseMatrix> GetBaseSparseProl      () const = 0;
  virtual shared_ptr<BaseSparseMatrix> GetBaseSparseProlTrans () const = 0;

  shared_ptr<BaseMatrix> GetBaseProl      () const override { return GetBaseSparseProl(); };
  shared_ptr<BaseMatrix> GetBaseProlTrans () const override { return GetBaseSparseProlTrans(); };

  virtual shared_ptr<BaseDOFMapStep> ConcatenateFromLeft (BaseSparseProlMap &other) = 0;
}; // class BaseSparseProlMap


/**
 *  This maps DOFs via a prolongation matrix (which is assumed to be hierarchic).
 *  ProlMaps can concatenate by multiplying their prolongation matrices.
 */
template<class ATM>
class ProlMap : public BaseSparseProlMap
{
public:
  using TM = ATM;

  static constexpr int H = TMHeight<TM>();
  static constexpr int W = TMWidth<TM>();

  using TM_F = StripTM<H, H>;
  using TM_P = TM;
  using TM_R = StripTM<W, H>;
  using TM_C = StripTM<W, W>;

  using SPM_TM_F = SparseMatTM<H, H>;
  using SPM_TM_P = SparseMatTM<H, W>;
  using SPM_TM_R = SparseMatTM<W, H>;
  using SPM_TM_C = SparseMatTM<W, W>;

  using SPM_F = SparseMat<H, H>;
  using SPM_P = SparseMat<H, W>;
  using SPM_R = SparseMat<W, H>;
  using SPM_C = SparseMat<W, W>;

  using TMAT = SPM_P;

  ProlMap (shared_ptr<TMAT> aprol,
           shared_ptr<SPM_R> aprolT,
           UniversalDofs originDofs,
           UniversalDofs mappedDofs);

  ProlMap (shared_ptr<TMAT> aprol,
           UniversalDofs originDofs,
           UniversalDofs mappedDofs);

  virtual ~ProlMap () = default;

  static_assert(std::is_same<TM, StripTM<H,W>>::value, "Need STRIP-TM!");

public:

  virtual void TransferF2C (BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void AddF2C (double fac, BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void TransferC2F (BaseVector *x_fine, BaseVector const* x_coarse) const override;
  virtual void AddC2F (double fac, BaseVector *x_fine, BaseVector const* x_coarse) const override;

  virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;
  virtual shared_ptr<BaseDOFMapStep> ConcatenateFromLeft (BaseSparseProlMap &other) override;

  virtual shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const override;

  shared_ptr<SPM_P> GetProl () const;
  shared_ptr<SPM_R> GetProlTrans () const;

  virtual shared_ptr<BaseSparseMatrix> GetBaseSparseProl () const override
  {
    return GetProl();
  }

  virtual shared_ptr<BaseSparseMatrix> GetBaseSparseProlTrans () const override
  {
    return GetProlTrans();
  }

  void SetProl (shared_ptr<TMAT> aprol);
  void BuildPT (bool force = false);
  virtual void Finalize () override;

  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

  virtual int GetBlockSize ()       const override;
  virtual int GetMappedBlockSize () const override;

protected:

  shared_ptr<SPM_P> _prol;
  shared_ptr<SPM_R> _prolT;
}; // class ProlMap


template<class TSCAL>
class DynBlockProlMap : public BaseProlMap
{
public:
  DynBlockProlMap (shared_ptr<DynBlockSparseMatrix<TSCAL>> prol,
                   UniversalDofs originDofs,
                   UniversalDofs mappedDofs,
                   bool buildPT = false);

  DynBlockProlMap (shared_ptr<DynBlockSparseMatrix<TSCAL>> prol,
                   shared_ptr<DynBlockSparseMatrix<TSCAL>> prolT,
                   UniversalDofs originDofs,
                   UniversalDofs mappedDofs);

  virtual ~DynBlockProlMap () = default;

  shared_ptr<DynBlockSparseMatrix<TSCAL>> GetDynSparseProl      () const { return _prol; };
  shared_ptr<DynBlockSparseMatrix<TSCAL>> GetDynSparseProlTrans () const { return _prolT; };

  shared_ptr<BaseMatrix> GetBaseProl      () const override { return GetDynSparseProl(); };
  shared_ptr<BaseMatrix> GetBaseProlTrans () const override { return GetDynSparseProlTrans(); };

  bool CanConcatenate (shared_ptr<BaseDOFMapStep> other) override;
  shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;
  /** "me - other" -> "new other - new me" **/
  shared_ptr<BaseDOFMapStep> PullBack (shared_ptr<BaseDOFMapStep> other) override;
  shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const override;

  void Finalize () override;
  void PrintTo (std::ostream & os, string prefix = "") const override;
  
  int GetBlockSize ()       const override;
  int GetMappedBlockSize () const override;

private:
  shared_ptr<DynBlockSparseMatrix<TSCAL>> _prol;
  shared_ptr<DynBlockSparseMatrix<TSCAL>> _prolT;
}; // class DynBlockProlMap


shared_ptr<DynBlockProlMap<double>>
ConvertToDynSparseProlMap(ProlMap<double>     const &sparseProlMap,
                          DynVectorBlocking<> const &fineBlocking,
                          DynVectorBlocking<> const &coarseBlocking);


/**
 * One primal DOF map with one or more attached secondary ones.
 */
class MultiDofMapStep : public BaseDOFMapStep
{
protected:
  Array<shared_ptr<BaseDOFMapStep>> maps;
public:
  MultiDofMapStep (FlatArray<shared_ptr<BaseDOFMapStep>> _maps);

  virtual ~MultiDofMapStep () { ; }

  INLINE int GetNMaps () const { return maps.Size(); }
  const shared_ptr<BaseDOFMapStep> & GetPrimMap () const { return maps[0]; }
  const shared_ptr<BaseDOFMapStep> & GetMap (int k) const { return maps[k]; }

  virtual void TransferF2C (BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void AddF2C (double fac, BaseVector const* x_fine, BaseVector *x_coarse) const override;
  virtual void TransferC2F (BaseVector *x_fine, BaseVector const* x_coarse) const override;
  virtual void AddC2F (double fac, BaseVector *x_fine, BaseVector const* x_coarse) const override;

  virtual bool CanConcatenate (shared_ptr<BaseDOFMapStep> other) override;
  virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;
  /** "me - other" -> "new other - new me" **/
  virtual shared_ptr<BaseDOFMapStep> PullBack (shared_ptr<BaseDOFMapStep> other) override;
  virtual shared_ptr<BaseMatrix> AssembleMatrix (shared_ptr<BaseMatrix> mat) const override;
  virtual void Finalize () override;
  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

  virtual int GetBlockSize ()       const override;
  virtual int GetMappedBlockSize () const override;
};


shared_ptr<BaseDOFMapStep> MakeSingleStep2 (FlatArray<shared_ptr<BaseDOFMapStep>> sub_steps);

extern template class ProlMap<double>;
extern template class ProlMap<Mat<2,2,double>>;
extern template class ProlMap<Mat<3,3,double>>;

#ifdef ELASTICITY
extern template class ProlMap<Mat<1,3,double>>;
extern template class ProlMap<Mat<1,6,double>>;
extern template class ProlMap<Mat<2,3,double>>;
extern template class ProlMap<Mat<3,6,double>>;
extern template class ProlMap<Mat<6,6,double>>;
#endif // ELASTICITY
}

#endif // FILE_DOF_MAP_HPP
