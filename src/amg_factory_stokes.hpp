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
    static constexpr int BS = ENERGY::DPV
    static constexpr int DIM = ENERGY::DIM;
    using BASE = NodalAMGFactory<NT_EDGE, TMESH, ENERGY>;
    using Options = typename BAE::Options;
    using TM = typename ENERGY::TM;
    using TSPM_TM = stripped_spm_tm<TM>;

  protected:
    using BASE::options;

  public:
    StokesAMGFactory (shared_ptr<Options> _opts)
      : BASE(_opts)
    { ; }

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
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);

    /** Discard **/
    virtual bool TryDiscardStep (BaseAMGFactory::State & state) override { return false; }
    virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (BaseAMGFactory::State & state);
  };

} // namespace amg
