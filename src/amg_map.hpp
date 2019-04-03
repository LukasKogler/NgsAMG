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
    virtual shared_ptr<BaseVector> CreateVector() const = 0;
    //{ return make_shared<ParallelVVector<double>>(pardofs->GetNDofLocal(), pardofs, CUMULATED); }
    virtual shared_ptr<BaseVector> CreateMappedVector() const = 0;
    //{ return (mapped_pardofs!=nullptr) ? make_shared<ParallelVVector<double>>(mapped_pardofs->GetNDofLocal(), mapped_pardofs, CUMULATED) : nullptr; }
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
    virtual shared_ptr<BaseVector> CreateVector() const override { return sub_steps[0]->CreateVector(); }
    virtual shared_ptr<BaseVector> CreateMappedVector() const override { return sub_steps.Last()->CreateMappedVector(); }
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
  template<class TMAT>
  class ProlMap : public BaseDOFMapStep
  {
  public:
    using TFMAT = typename amg_spm_traits<TMAT>::T_RIGHT;
    using TCMAT = typename amg_spm_traits<TMAT>::T_LEFT;
    ProlMap (shared_ptr<ParallelDofs> fpd, shared_ptr<ParallelDofs> cpd)
      : BaseDOFMapStep(fpd, cpd), prol(nullptr)
    { ; }
    virtual shared_ptr<BaseVector> CreateVector() const override
    { return make_shared<ParallelVVector<typename TMAT::TVY>>(pardofs->GetNDofLocal(), pardofs, CUMULATED); }
    virtual shared_ptr<BaseVector> CreateMappedVector() const override
    { return (mapped_pardofs!=nullptr) ? make_shared<ParallelVVector<typename TMAT::TVX>>(mapped_pardofs->GetNDofLocal(), mapped_pardofs, CUMULATED) : nullptr; }
    // me left -- other right
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override
    {
      auto pmother = dynamic_pointer_cast<ProlMap<TCMAT>>(other);
      if (pmother==nullptr) { return nullptr; }
      return pmother->ConcBack(*this);
    }
    // me right -- other left
    template<class TMATO>
    shared_ptr<BaseDOFMapStep> ConcBack (ProlMap<TMATO> & other) {
      cout << " me " << this << endl;
      cout << "ConcBack, me dims: " << prol->Height() << " " << prol->Width() << endl;
      auto oprol = other.GetProl();
      cout << " other " << &other << endl;
      cout << "ConcBack, other dims: " << oprol->Height() << " " << oprol->Width() << endl;
      auto pstep = make_shared<ProlMap<typename mult_spm<TMATO, TMAT>::type>> (other.GetParDofs(), this->GetMappedParDofs());
      shared_ptr<typename mult_spm<TMATO, TMAT>::type> pp = MatMultAB (*other.prol, *prol);
      cout << "conc NDS: " << pstep->GetParDofs()->GetNDofGlobal() << " -> " << pstep->GetMappedParDofs()->GetNDofGlobal() << endl;
      // cout << "conc prol: " << endl; print_tm_spmat(cout, *pp); cout << endl;
      pstep->SetProl(pp);
      return pstep;
    }
    virtual void TransferF2C (const shared_ptr<const BaseVector> & x_fine,
			      const shared_ptr<BaseVector> & x_coarse) const override
    {
      x_coarse->FVDouble() = 0.0;
      prol->MultTransAdd(1.0, *x_fine, *x_coarse);
      x_coarse->SetParallelStatus(DISTRIBUTED);
    }
    virtual void TransferC2F (const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const override
    {
      x_coarse->Cumulate();
      cout << "MA 1, coarse: " << *x_coarse << endl;
      cout << "MA 1, fine: " << *x_fine << endl;
      prol->Mult(*x_coarse, *x_fine);
      cout << "MA 2, coarse: " << *x_coarse << endl;
      cout << "MA 2, fine: " << *x_fine << endl;
      x_fine->SetParallelStatus(CUMULATED);
    }
    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override
    {
      auto tfmat = dynamic_pointer_cast<TFMAT>(mat);
      if (tfmat==nullptr) {
	string exname = "Cannot cast ";
	// exname += typeid(*mat).name();
	exname += " to "; ;
	exname += typeid(TFMAT).name();
	exname += "!!";
	throw Exception(exname);
      }
      return DoAssembleMatrix (tfmat);
    }
    shared_ptr<TCMAT> DoAssembleMatrix (shared_ptr<TFMAT> mat) const
    { return RestrictMatrix<TFMAT, TMAT> (*mat, *prol); }
    shared_ptr<TMAT> GetProl () const { return prol; }
    void SetProl (shared_ptr<TMAT> aprol) { if(prol==nullptr) { cout << "me: " << this << endl; cout << "set prol, dims: " << aprol->Height() << " " << aprol->Width() << endl;;} prol = aprol; }
  protected:
    shared_ptr<TMAT> prol;
  };

}


#endif
