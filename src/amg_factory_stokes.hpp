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

  protected:
    using BASE_CLASS::options;

  public:

    StokesAMGFactory (shared_ptr<Options> _opts);

    ~StokesAMGFactory() { ; }

  protected:
    /** State **/
    virtual BaseAMGFactory::State* AllocState () const override;
    virtual void InitState (BaseAMGFactory::State & state, BaseAMGFactory::AMGLevel & lev) const override;

    /** Coarse **/
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (BaseAMGFactory::State & state) override;
    virtual shared_ptr<BaseCoarseMap> BuildAggMap (BaseAMGFactory::State & state);
    // virtual shared_ptr<BaseCoarseMap> BuildECMap (BaseAMGFactory::State & state);
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);
    shared_ptr<TSPM_TM> BuildPWProl_impl (shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds,
					  shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
					  FlatArray<int> vmap, FlatArray<int> emap,
					  FlatTable<int> v_aggs);
    shared_ptr<TSPM_TM> SmoothProlMap_impl (shared_ptr<ProlMap<TSPM_TM>> pwprol, shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
					    FlatArray<int> vmap, FlatArray<int> emap, FlatTable<int> v_aggs);

    /** Discard **/
    virtual bool TryDiscardStep (BaseAMGFactory::State & state) override { return false; }
    virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (BaseAMGFactory::State & state) { return nullptr; }
  }; // class StokesAMGFactory

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP
#endif // STOKES
