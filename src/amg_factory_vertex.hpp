#ifndef FILE_AMG_FACTORY_VERTEX_HPP
#define FILE_AMG_FACTORY_VERTEX_HPP

#include "amg_factory.hpp"

namespace amg
{

  /** DOFs attached to vertices, edge-wise energy. **/
  class VertexAMGFactoryOptions;

  template<class AENERGY, class ATMESH, int ABS>
  class VertexAMGFactory : public NodalAMGFactory<NT_VERTEX, ATMESH, ABS>
  {
  public:
    using ENERGY = AENERGY;
    using TMESH = ATMESH;
    static constexpr int BS = ABS;
    using BASE_CLASS = NodalAMGFactory<NT_VERTEX, TMESH, BS>;
    using TM = typename ENERGY::TM;
    using TSPM_TM = stripped_spm_tm<TM>;

    using AMGLevel = typename BaseAMGFactory::AMGLevel;
    using Options = VertexAMGFactoryOptions;
    class State;

  protected:
    using BaseAMGFactory::options;
    
  public:
    VertexAMGFactory (shared_ptr<Options> opts);

    ~VertexAMGFactory ();

  protected:

    /** State **/
    virtual BaseAMGFactory::State* AllocState () const override;
    virtual void InitState (BaseAMGFactory::State & state, BaseAMGFactory::AMGLevel & lev) const override;

    /** Coarse **/
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (BaseAMGFactory::State & state) override;
    virtual shared_ptr<BaseCoarseMap> BuildAggMap (BaseAMGFactory::State & state);
    virtual shared_ptr<BaseCoarseMap> BuildECMap (BaseAMGFactory::State & state);
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);

    /** Discard **/
    virtual bool TryDiscardStep (BaseAMGFactory::State & state) override;
    virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (BaseAMGFactory::State & state);

  }; // VertexAMGFactory
    
} // namespace amg

#endif
