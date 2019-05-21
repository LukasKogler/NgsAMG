
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
    virtual void TransferF2C (const shared_ptr<const BaseVector> & x_fine,
			      const shared_ptr<BaseVector> & x_coarse) const = 0;
    virtual void TransferC2F (const shared_ptr<BaseVector> & x_fine,
			      const shared_ptr<const BaseVector> & x_coarse) const = 0;
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
  public:
    void AddStep(const shared_ptr<BaseDOFMapStep> step) { steps.Append(step); }
    shared_ptr<BaseDOFMapStep> GetStep (int k) { return steps[k]; }
    void Finalize (FlatArray<size_t> acutoffs, shared_ptr<BaseDOFMapStep> embed_step);
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
    // virtual shared_ptr<BaseVector> CreateVector() const override { return sub_steps[0]->CreateVector(); }
    // virtual shared_ptr<BaseVector> CreateMappedVector() const override { return sub_steps.Last()->CreateMappedVector(); }
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
    using TFMAT = typename amg_spm_traits<TMAT>::T_LEFT;
    using TCMAT = typename amg_spm_traits<TMAT>::T_RIGHT;

    ProlMap (shared_ptr<TMAT> aprol, shared_ptr<ParallelDofs> fpd, shared_ptr<ParallelDofs> cpd, bool apw = true)
      : BaseDOFMapStep(fpd, cpd), prol(aprol), ispw(apw), has_smf(false), SMFUNC([](auto x) { ; }), LOGFUNC([](auto x) { ; })
    { ; }

    virtual void TransferF2C (const shared_ptr<const BaseVector> & x_fine,
			      const shared_ptr<BaseVector> & x_coarse) const override;
    
    virtual void TransferC2F (const shared_ptr<BaseVector> & x_fine,
			      const shared_ptr<const BaseVector> & x_coarse) const override;

    // virtual shared_ptr<BaseVector> CreateVector () const override;
    // virtual shared_ptr<BaseVector> CreateMappedVector () const override;
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override;

    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override;

    INLINE shared_ptr<TMAT> GetProl () const { return prol; }
    INLINE void SetProl (shared_ptr<TMAT> aprol) { prol = aprol; }

    INLINE bool IsPW () const { return ispw; }
    INLINE bool CanSM () const { return has_smf; }
    INLINE bool WantSM () const { return has_smf && force_smf; }
    // void SetSmoothed (const VWiseAMG* aamg, shared_ptr<TopologicMesh> amesh);
    // void SetSmoothed (void (*SMFUNC) (ProlMap<TMAT> * map, shared_ptr<TopologicMesh> mesh), shared_ptr<TopologicMesh> amesh);
    void SetSmoothed ( std::function<void(ProlMap<TMAT> * map)> ASMFUNC, bool force = true);
    void SetLog ( std::function<void(shared_ptr<BaseSparseMatrix> prol)> ALOGFUNC);
    std::function<void(shared_ptr<BaseSparseMatrix>)> GetLog () { return LOGFUNC; }
    void ClearSmoothed ( );
    void Smooth (); 

    void SetCnt (int acnt) const { cnt = acnt; }
    int GetCnt () const { return cnt; }

  protected:
    shared_ptr<TMAT> prol;
    mutable bool ispw = true;
    mutable bool has_smf = false;
    mutable bool force_smf = false;
    mutable bool has_lf = false;
    mutable int cnt = 0;
    
    std::function<void(ProlMap<TMAT> * map)> SMFUNC;
    std::function<void(shared_ptr<BaseSparseMatrix> prol)> LOGFUNC;
    // shared_ptr<TopologicMesh> mesh;
  };


  template <class SPML, class SPMR>
  class TwoProlMap : public ProlMap<mult_spm<SPML, SPMR>>
  {
  public:    
    using SPM = mult_spm<SPML, SPMR>;

    TwoProlMap (shared_ptr<ProlMap<SPML>> pml, shared_ptr<ProlMap<SPMR>> pmr)
      : ProlMap<SPM>(nullptr, pml->GetParDofs(), pmr->GetMappedParDofs()),
      lmap(pml), rmap(pmr)
    { ; }

    ~TwoProlMap () { ; }
    
    shared_ptr<ProlMap<SPML>> GetLMap () const { return lmap; }
    void SetLMap (shared_ptr<ProlMap<SPML>> almap) { lmap = almap; }
    shared_ptr<ProlMap<SPMR>> GetRMap () const { return rmap; }
    void SetRMap (shared_ptr<ProlMap<SPMR>> armap) { rmap = armap; }

    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override
    { // can only happen for [PP, S(..)] case with first map
      // AAAARGH MORE HACKS AAAAAAA
      const_cast<shared_ptr<SPM>&>(prol) = MatMultAB<SPML, SPMR>(*lmap->GetProl(), *rmap->GetProl());
      this->SetCnt(lmap->GetCnt() + rmap->GetCnt());
      this->ispw = rmap->IsPW() && lmap->IsPW();
      const_cast<shared_ptr<ProlMap<SPML>>&>(lmap) = nullptr;
      const_cast<shared_ptr<ProlMap<SPMR>>&>(rmap) = nullptr;
      return ProlMap<SPM> :: AssembleMatrix(mat);      
    }
    
  protected:
    using ProlMap<SPM>::prol;
    shared_ptr<ProlMap<SPML>> lmap;
    shared_ptr<ProlMap<SPMR>> rmap;
  };
  
  

  
}


#endif
