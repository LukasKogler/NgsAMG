#ifdef STOKES

#ifndef FILE_AMG_FACTORY_STOKES_HPP
#define FILE_AMG_FACTORY_STOKES_HPP

namespace amg
{

  /** Stokes AMG (ENERGY + div-div penalty):
      We assume that we have DIM DOFs per facet of the mesh. Divergence-The divergence 
      - DOFs are assigned to edges of the dual mesh. **/

  template<class ATMESH, class AENERGY>
  class StokesAMGFactory : public NodalAMGFactory<NT_EDGE, ATMESH, AENERGY::DPV>
  {
  public:
    using TMESH = ATMESH;
    using ENERGY = AENERGY;
    static constexpr int BS = ENERGY::DPV;
    static constexpr int DIM = ENERGY::DIM;
    using BASE_CLASS = NodalAMGFactory<NT_EDGE, ATMESH, AENERGY::DPV>;
    using Options = typename BASE_CLASS::Options;
    using TM = typename ENERGY::TM;
    using TSPM_TM = stripped_spm_tm<TM>;
    using TCM_TM = stripped_spm_tm<Mat<BS, 1, double>>;
    using TCM = SparseMatrix<Mat<BS, 1, double>>;
    using TCTM_TM = stripped_spm_tm<Mat<1, BS, double>>;
    using TCTM = SparseMatrix<Mat<1, BS, double>>;
    using TPM = SparseMatrix<double>;
    using TPM_TM = SparseMatrixTM<double>;


    struct StokesLC : public BaseAMGFactory::LevelCapsule
    {
      shared_ptr<TPM> pot_mat;   // matrix in the potential space
      shared_ptr<ParallelDofs> pot_pardofs;   // ParallelDofs in the potential space
      shared_ptr<TCTM> curl_mat_T;  // discrete curl matrix on this level 
      shared_ptr<TCM> curl_mat;  // discrete curl matrix on this level 
    }; // struct StokesLC


  protected:
    using BASE_CLASS::options;

  public:

    StokesAMGFactory (shared_ptr<Options> _opts);

    ~StokesAMGFactory() { ; }

  protected:
    /** Misc overloads **/
    virtual BaseAMGFactory::State* AllocState () const override;
    virtual shared_ptr<BaseAMGFactory::LevelCapsule> AllocCap () const override;
    virtual void MapLevel2 (shared_ptr<BaseDOFMapStep> & dof_step, shared_ptr<BaseAMGFactory::AMGLevel> & f_cap, shared_ptr<BaseAMGFactory::AMGLevel> & c_cap) override;
    virtual shared_ptr<BaseDOFMapStep> MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps,
						 shared_ptr<BaseAMGFactory::AMGLevel> & f_cap, shared_ptr<BaseAMGFactory::AMGLevel> & c_cap) override;

    /** Coarse **/
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) override;
    virtual shared_ptr<StokesBCM> BuildAggMap (BaseAMGFactory::State & state, shared_ptr<StokesLC> & mapped_cap);
    // virtual shared_ptr<BaseCoarseMap> BuildECMap (BaseAMGFactory::State & state);
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
						  shared_ptr<BaseAMGFactory::LevelCapsule> ccap) override;
    virtual shared_ptr<BaseDOFMapStep> RangePWProl (shared_ptr<StokesBCM> cmap, shared_ptr<StokesLC> fcap, shared_ptr<StokesLC> ccap);
    virtual shared_ptr<BaseDOFMapStep> PotPWProl (shared_ptr<StokesBCM> cmap, shared_ptr<StokesLC> fcap, shared_ptr<StokesLC> ccap,
						  shared_ptr<ProlMap<TSPM_TM>> range_prol);
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseAMGFactory::LevelCapsule> fcap) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap,
							shared_ptr<BaseAMGFactory::LevelCapsule> fcap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);
    shared_ptr<TSPM_TM> BuildPWProl_impl (shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
					  FlatArray<int> vmap, FlatArray<int> emap,
					  FlatTable<int> v_aggs);
    shared_ptr<TSPM_TM> SmoothProlMap_impl (shared_ptr<ProlMap<TSPM_TM>> pwprol, shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
					    FlatArray<int> vmap, FlatArray<int> emap, FlatTable<int> v_aggs);
  public:
    /*** Need these also for setting up the coarsest level **/
    virtual void BuildCurlMat (StokesLC & cap);
    virtual void ProjectToPotSpace (StokesLC & cap);
    virtual void BuildPotParDofs (StokesLC & cap);

  protected:
    /** Discard **/
    virtual bool TryDiscardStep (BaseAMGFactory::State & state) override { return false; }
    virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (BaseAMGFactory::State & state) { return nullptr; }
  }; // class StokesAMGFactory

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP
#endif // STOKES
