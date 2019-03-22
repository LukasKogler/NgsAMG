#ifndef FILE_AMG_MAP
#define FILE_AMG_MAP

namespace amg {

  class BaseGridMapStep
  {
  public:
    virtual shared_ptr<TopologicMesh> GetMesh () const = 0;
    virtual shared_ptr<TopologicMesh> GetMappedMesh () const = 0;
  };

  /** This maps meshes and their NODES between levels. **/
  class BaseGridMapStep;
  class GridMap
  {
  public:
    void AddStep (shared_ptr<BaseGridMapStep> step) { steps.Append(step); }
    shared_ptr<BaseGridMapStep> GetStep (size_t nr) { return steps[nr]; }
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
    virtual void TransferF2C(const shared_ptr<const BaseVector> & x_fine,
			     const shared_ptr<BaseVector> & x_coarse) const = 0;
    virtual void TransferC2F(const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const = 0;
    shared_ptr<BaseVector> CreateVector() const
    { return make_shared<ParallelVVector<double>>(pardofs->GetNDofLocal(), pardofs, CUMULATED); }
    shared_ptr<BaseVector> CreateMappedVector() const
    { return (mapped_pardofs!=nullptr) ? make_shared<ParallelVVector<double>>(mapped_pardofs->GetNDofLocal(), mapped_pardofs, CUMULATED) : nullptr; }
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) { return nullptr; }
    shared_ptr<ParallelDofs> GetParDofs() const { return pardofs; }
    shared_ptr<ParallelDofs> GetMappedParDofs() const { return mapped_pardofs; }
    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const = 0;
  };

  class BaseDOFMapStep;
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
    void AddStep(const shared_ptr<BaseDOFMapStep> step)
    { steps.Append(step); }
    shared_ptr<BaseDOFMapStep> GetStep(int k)
    { return steps[k]; }
    void SetCutoffs (FlatArray<size_t> acutoffs)
    { cutoffs.SetSize(acutoffs.Size()); cutoffs = acutoffs; }
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
    void ConcSteps () { throw Exception("You forgot to do this, dummy!"); };
    // bool drops = false; //TODO: did I need this?
    Array<shared_ptr<BaseDOFMapStep>> steps;
    Array<size_t> cutoffs;
    Array<shared_ptr<BaseVector>> temp_vecs;
  };



  /**
     This maps DOFs via a prolongation matrix (which is assumed to be hierarchic).
     ProlMaps can concatenate by multiplying their prolongation matrices.
  **/
  template<class TMAT>
  class ProlMap : public BaseDOFMapStep
  {
  public:
    ProlMap (shared_ptr<TMAT> _prol,
	     shared_ptr<ParallelDofs> fpd,
	     shared_ptr<ParallelDofs> cpd)
      : BaseDOFMapStep(fpd, cpd), prol(_prol)
    { }

    using TFMAT = typename amg_spm_traits<TMAT>::T_RIGHT;
    using TCMAT = typename amg_spm_traits<TMAT>::T_LEFT;
    // me left -- other right
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override
    {
      cout << " me1 @" << this << endl;
      cout << " other1 @" << other << endl;
      constexpr int W = mat_traits<typename TMAT::TVY>::HEIGHT;
      // using TMR = SparseMatrix<Mat<W,W,double>, Vec<W,double>, Vec<W,double>>;
      auto pmother = dynamic_pointer_cast<ProlMap<TCMAT>>(other);
      if (pmother==nullptr) {
	cout << " cannot concatenate!!" << endl;
	return nullptr;
      }
      return pmother->ConcBack(*this);
    }

    // me right -- other left
    template<class TMATO>
    shared_ptr<BaseDOFMapStep> ConcBack (ProlMap<TMATO> & other) {
      // cout << " me @" << this << endl;
      // cout << " other @" << &other << endl;
      shared_ptr<typename mult_spm<TMATO, TMAT>::type> pp = MatMultAB (*other.prol, *prol);
      // cout << "concatenated prol: " << endl << *pp << endl;
      return make_shared<ProlMap<typename mult_spm<TMATO, TMAT>::type>> (pp, other.GetParDofs(), this->GetMappedParDofs());
    }
    virtual void TransferF2C(const shared_ptr<const BaseVector> & x_fine,
			     const shared_ptr<BaseVector> & x_coarse) const override
    {
      x_coarse->FVDouble() = 0.0;
      prol->MultTransAdd(1.0, *x_fine, *x_coarse);
      x_coarse->SetParallelStatus(DISTRIBUTED);
    }
    virtual void TransferC2F(const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const override
    {
      x_coarse->Cumulate();
      prol->Mult(*x_coarse, *x_fine);
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
  private:
    shared_ptr<TMAT> prol;
  };

}


#endif
