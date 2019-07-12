

#ifndef FILE_AMG_MAP
#define FILE_AMG_MAP

namespace amg {

  class BaseGridMapStep
  {
  public:
    virtual ~BaseGridMapStep () { ; }
    virtual shared_ptr<TopologicMesh> GetMesh () const = 0;
    virtual shared_ptr<TopologicMesh> GetMappedMesh () const = 0;
    // virtual void CleanupMeshes () const = 0;
  };

  /** This maps meshes and their NODES between levels. **/
  class BaseGridMapStep;
  class GridMap
  {
  public:
    void AddStep (shared_ptr<BaseGridMapStep> step) { steps.Append(step); }
    shared_ptr<BaseGridMapStep> GetStep (size_t nr) { return steps[nr]; }
    void CleanupStep (int level) { steps[level] = nullptr; }
    auto begin() const { return steps.begin(); }
    auto end() const { return steps.end(); }
  private:
    Array<shared_ptr<BaseGridMapStep>> steps;
  };

  template<class TMESH>
  class GridMapStep : public BaseGridMapStep
  {
  public:
    GridMapStep (shared_ptr<TMESH> _mesh) : mesh(_mesh), mapped_mesh(nullptr) { ; };
    virtual shared_ptr<TopologicMesh> GetMesh () const override { return mesh; }
    virtual shared_ptr<TopologicMesh> GetMappedMesh () const override { return mapped_mesh; }
    // virtual void CleanupMeshes () const { mesh = nullptr; mapped_mesh = nullptr; }
  protected:
    shared_ptr<TMESH> mesh, mapped_mesh;
  };

  /**
     At the minimum, we have to be able to:
     -) transfer vectors between levels
     -) construct the coarse level matrix
     -) be able to create vectors for both levels
  **/
  class BaseDOFMapStep : public BaseMatrix
  {
  protected:
    shared_ptr<ParallelDofs> pardofs, mapped_pardofs;
  public:
    BaseDOFMapStep (shared_ptr<ParallelDofs> _pardofs, shared_ptr<ParallelDofs> _mapped_pardofs)
      : pardofs(_pardofs), mapped_pardofs(_mapped_pardofs) {}
    virtual ~BaseDOFMapStep () { ; }
    virtual void TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const = 0;
    virtual void AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const = 0;
    virtual void TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const = 0;
    virtual void AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const = 0;
    shared_ptr<BaseVector> CreateVector () const {
      return make_shared<S_ParallelBaseVectorPtr<double>>
	(pardofs->GetNDofLocal(), pardofs->GetEntrySize(), pardofs, DISTRIBUTED);
    }
    shared_ptr<BaseVector> CreateMappedVector () const {
      if (mapped_pardofs == nullptr) return nullptr;
      return make_shared<S_ParallelBaseVectorPtr<double>>
	(mapped_pardofs->GetNDofLocal(), mapped_pardofs->GetEntrySize(), mapped_pardofs, DISTRIBUTED);
    }
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) { return nullptr; }
    shared_ptr<ParallelDofs> GetParDofs () const { return pardofs; }
    shared_ptr<ParallelDofs> GetMappedParDofs () const { return mapped_pardofs; }
    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const = 0;
  };

  /**
     This maps DOFS between levels. We can add DOF-map-steps of various kind.
     Each step has to start where the last one left off.
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

  public:
    DofMap () { ; }

    void AddStep(const shared_ptr<BaseDOFMapStep> step) { steps.Append(step); }

    INLINE size_t GetNSteps () const { return steps.Size(); }
    INLINE size_t GetNLevels () const { return GetNSteps() + 1; } // we count one "empty" level if we drop

    shared_ptr<BaseDOFMapStep> GetStep (int k) { return steps[k]; }

    // void Finalize (FlatArray<size_t> acutoffs, shared_ptr<BaseDOFMapStep> embed_step);

    INLINE void TransferF2C (size_t lev_start, const BaseVector *x_fine, BaseVector * x_coarse) const
    { steps[lev_start]->TransferF2C(x_fine, x_coarse);}

    INLINE void AddF2C (size_t lev_start, double fac, const BaseVector *x_fine, BaseVector * x_coarse) const
    { steps[lev_start]->AddF2C(fac, x_fine, x_coarse);}

    INLINE void TransferC2F (size_t lev_dest, BaseVector *x_fine, const BaseVector * x_coarse) const
    { steps[lev_dest]->TransferC2F(x_fine, x_coarse);}

    INLINE void AddC2F (size_t lev_dest, double fac, BaseVector *x_fine, const BaseVector * x_coarse) const
    { steps[lev_dest]->AddC2F(fac, x_fine, x_coarse);}

    shared_ptr<BaseVector> CreateVector (size_t l) const
    { return (l>steps.Size()) ? nullptr : ((l==steps.Size()) ? steps.Last()->CreateMappedVector() : steps[l]->CreateVector()); }

    shared_ptr<ParallelDofs> GetParDofs (size_t L = 0) const { return (L==steps.Size()) ? steps.Last()->GetMappedParDofs() : steps[L]->GetParDofs(); }

    shared_ptr<ParallelDofs> GetMappedParDofs () const { return steps.Last()->GetMappedParDofs(); }

    Array<shared_ptr<BaseSparseMatrix>> AssembleMatrices (shared_ptr<BaseSparseMatrix> finest_mat) const;

  }; // DOFMap


  /** Multiple steps that act as one **/
  class ConcDMS : public BaseDOFMapStep
  {
  protected:
    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    Array<shared_ptr<BaseVector>> vecs;

  public:
    ConcDMS (Array<shared_ptr<BaseDOFMapStep>> & _sub_steps);
    
    virtual void TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const override;
    virtual void AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const override;
    virtual void TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const override;
    virtual void AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const override;

    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override;
  };


  /**
     This maps DOFs via a prolongation matrix (which is assumed to be hierarchic).
     ProlMaps can concatenate by multiplying their prolongation matrices.
  **/
  INLINE Timer & timer_hack_prol_f2c () { static Timer t("ProlMap::TransferF2C"); return t; }
  INLINE Timer & timer_hack_prol_c2f () { static Timer t("ProlMap::TransferC2F"); return t; }
  template<class TMAT>
  class ProlMap : public BaseDOFMapStep
  {
  public:

#ifdef ELASTICITY // workaround until NGSolve master gets updated
    using SPM_TM_F = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::HEIGHT, double>>;
    using SPM_TM_P = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>;
    using SPM_TM_C = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::WIDTH, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>;

    using SPM_F = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::HEIGHT, double>>::type>;
    using SPM_P = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>::type>;
    using SPM_C = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::WIDTH, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>::type>;
#else
    using SPM_TM_F = SparseMatrixTM<double>;
    using SPM_TM_P = SparseMatrixTM<double>;
    using SPM_TM_C = SparseMatrixTM<double>;
    using SPM_F = SparseMatrix<double>;
    using SPM_P = SparseMatrix<double>;
    using SPM_C = SparseMatrix<double>;
#endif

    static_assert(std::is_same<SPM_TM_P, TMAT>::value, "Use SPM_TM for ProlMap!!");

    ProlMap (shared_ptr<TMAT> aprol, shared_ptr<ParallelDofs> fpd, shared_ptr<ParallelDofs> cpd)
      : BaseDOFMapStep(fpd, cpd), prol(aprol), prol_trans(nullptr)
    { ; }

  public:

    virtual void TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const override;
    virtual void AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const override;
    virtual void TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const override;
    virtual void AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const override;

    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;

    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override;

    INLINE shared_ptr<TMAT> GetProl () const { return prol; }
    INLINE void SetProl (shared_ptr<TMAT> aprol) { prol = aprol; }

  protected:
    shared_ptr<TMAT> prol;
    shared_ptr<trans_spm_tm<TMAT>> prol_trans;
  };

  
}


#endif
