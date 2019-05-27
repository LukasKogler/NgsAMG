

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
  class BaseDOFMapStep
  {
  protected:
    shared_ptr<ParallelDofs> pardofs, mapped_pardofs;
  public:
    BaseDOFMapStep (shared_ptr<ParallelDofs> _pardofs, shared_ptr<ParallelDofs> _mapped_pardofs)
      : pardofs(_pardofs), mapped_pardofs(_mapped_pardofs) {}
    virtual ~BaseDOFMapStep () { ; }
    virtual void TransferF2C(const shared_ptr<const BaseVector> & x_fine,
			     const shared_ptr<BaseVector> & x_coarse) const = 0;
    virtual void TransferC2F(const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const = 0;
    shared_ptr<BaseVector> CreateVector() const
    { return make_shared<S_ParallelBaseVectorPtr<double>> (pardofs->GetNDofLocal(), pardofs->GetEntrySize(), pardofs, DISTRIBUTED); }
    shared_ptr<BaseVector> CreateMappedVector() const
    { return (mapped_pardofs!=nullptr) ? make_shared<S_ParallelBaseVectorPtr<double>> (mapped_pardofs->GetNDofLocal(), mapped_pardofs->GetEntrySize(), mapped_pardofs, DISTRIBUTED) : nullptr; }
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) { return nullptr; }
    shared_ptr<ParallelDofs> GetParDofs() const { return pardofs; }
    shared_ptr<ParallelDofs> GetMappedParDofs() const { return mapped_pardofs; }
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
  public:
    void AddStep(const shared_ptr<BaseDOFMapStep> step) { steps.Append(step); }
    shared_ptr<BaseDOFMapStep> GetStep (int k) { return steps[k]; }
    void SetCutoffs (FlatArray<size_t> acutoffs);
    INLINE void TransferF2C (size_t lev_start, const shared_ptr<const BaseVector> & x_fine,
			     const shared_ptr<BaseVector> & x_coarse) const
    { steps[lev_start]->TransferF2C(x_fine, x_coarse);}
    INLINE void TransferC2F (size_t lev_dest, const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const
    { steps[lev_dest]->TransferC2F(x_fine, x_coarse);}
    shared_ptr<BaseVector> CreateVector (size_t l) const
    { return (l>steps.Size()) ? nullptr : ((l==steps.Size()) ? steps.Last()->CreateMappedVector() : steps[l]->CreateVector()); }
    size_t GetNLevels () const { return steps.Size() + 1; } // we count one "empty" level if we drop
    shared_ptr<ParallelDofs> GetParDofs (size_t L = 0) const { return (L==steps.Size()) ? steps.Last()->GetMappedParDofs() : steps[L]->GetParDofs(); }
    shared_ptr<ParallelDofs> GetMappedParDofs () const { return steps.Last()->GetMappedParDofs(); }
    Array<shared_ptr<BaseSparseMatrix>> AssembleMatrices (shared_ptr<BaseSparseMatrix> finest_mat) const;
  private:
    void ConcSteps ();
    Array<shared_ptr<BaseDOFMapStep>> steps;
    Array<size_t> cutoffs;
  };


  class ConcDMS : public BaseDOFMapStep
  {
  protected:
    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    Array<shared_ptr<BaseVector>> vecs;
  public:
    ConcDMS (const Array<shared_ptr<BaseDOFMapStep>> & _sub_steps) :
      BaseDOFMapStep(_sub_steps[0]->GetParDofs(), _sub_steps.Last()->GetMappedParDofs()),
      sub_steps(_sub_steps)
    {
      vecs.SetSize(sub_steps.Size()-1);
      if(sub_steps.Size()>1) // TODO: test this out -> i think Range(1,0) is broken??
	for(auto k:Range(size_t(1),sub_steps.Size()))
	  vecs[k-1] = sub_steps[k]->CreateVector();
    }
    virtual void TransferF2C (const shared_ptr<const BaseVector> & x_fine,
			      const shared_ptr<BaseVector> & x_coarse) const override
    {
      if (sub_steps.Size()==1) {
	sub_steps[0]->TransferF2C(x_fine, x_coarse);
	return;
      }
      sub_steps[0]->TransferF2C(x_fine, vecs[0]);
      for (int l = 1; l<int(sub_steps.Size())-1; l++)
	sub_steps[l]->TransferF2C(vecs[l-1], vecs[l]);
      sub_steps.Last()->TransferF2C(vecs.Last(), x_coarse);
    }
    virtual void TransferC2F (const shared_ptr<BaseVector> & x_fine,
			      const shared_ptr<const BaseVector> & x_coarse) const override
    {
      if (sub_steps.Size()==1) {
	sub_steps[0]->TransferC2F(x_fine, x_coarse);
	return;
      }
      sub_steps.Last()->TransferC2F(vecs.Last(), x_coarse);
      for (int l = sub_steps.Size()-2; l>0; l--)
	sub_steps[l]->TransferC2F(vecs[l-1], vecs[l]);
      sub_steps[0]->TransferC2F(x_fine, vecs[0]);
    }
    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override
    {
      shared_ptr<BaseSparseMatrix> cmat = mat;
      for (auto& step : sub_steps)
	cmat = step->AssembleMatrix(cmat);
      return cmat;
    }
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

    using SPM_TM_F = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::HEIGHT, double>>;
    using SPM_TM_P = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>;
    using SPM_TM_C = stripped_spm_tm<Mat<mat_traits<typename TMAT::TENTRY>::WIDTH, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>;

    using SPM_F = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::HEIGHT, double>>::type>;
    using SPM_P = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::HEIGHT, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>::type>;
    using SPM_C = SparseMatrix<typename strip_mat<Mat<mat_traits<typename TMAT::TENTRY>::WIDTH, mat_traits<typename TMAT::TENTRY>::WIDTH, double>>::type>;

    static_assert(std::is_same<SPM_TM_P, TMAT>::value, "Use SPM_TM for ProlMap!!");
    
    ProlMap (shared_ptr<ParallelDofs> fpd, shared_ptr<ParallelDofs> cpd)
      : BaseDOFMapStep(fpd, cpd), prol(nullptr)
    { ; }
    // me left -- other right
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override
    {
      auto pmother = dynamic_pointer_cast<ProlMap<SPM_TM_C>>(other);
      if (pmother==nullptr) { return nullptr; }
      return pmother->ConcBack(*this);
    }
    // me right -- other left
    template<class TMATO>
    shared_ptr<BaseDOFMapStep> ConcBack (ProlMap<TMATO> & other) {
      auto oprol = other.GetProl();
      auto pstep = make_shared<ProlMap<mult_spm_tm<TMATO, TMAT>>> (other.GetParDofs(), this->GetMappedParDofs());
      shared_ptr<mult_spm_tm<TMATO, TMAT>> pp = MatMultAB (*oprol, *prol);
      pstep->SetProl(pp);
      return pstep;
    }
    virtual void TransferF2C (const shared_ptr<const BaseVector> & x_fine,
			      const shared_ptr<BaseVector> & x_coarse) const override
    {
      RegionTimer rt(timer_hack_prol_f2c());
      x_coarse->FVDouble() = 0.0;
      prol->MultTransAdd(1.0, *x_fine, *x_coarse);
      x_coarse->SetParallelStatus(DISTRIBUTED);
    }
    virtual void TransferC2F (const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const override
    {
      RegionTimer rt(timer_hack_prol_c2f());
      x_coarse->Cumulate();
      prol->Mult(*x_coarse, *x_fine);
      x_fine->SetParallelStatus(CUMULATED);
    }
    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override
    {
      auto tfmat = dynamic_pointer_cast<SPM_F>(mat);
      if (tfmat==nullptr) {
	throw Exception(string("Cannot cast to ") + typeid(SPM_F).name());
      }
      return DoAssembleMatrix (tfmat);
    }
    shared_ptr<SPM_C> DoAssembleMatrix (shared_ptr<SPM_F> mat) const
    {
      auto & ncp = const_cast<shared_ptr<SPM_TM_P>&>(prol);
      ncp = make_shared<SPM_P>(move(*ncp));

      auto rm = RestrictMatrixTM<SPM_TM_F, SPM_TM_P> (*mat, *prol);
      return make_shared<SPM_C>(move(*rm));
    }
    shared_ptr<TMAT> GetProl () const { return prol; }
    void SetProl (shared_ptr<TMAT> aprol) { prol = aprol; }
  protected:
    shared_ptr<TMAT> prol;
  };

}


#endif
