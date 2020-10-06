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
    virtual void InitState (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::AMGLevel> & lev) const override;

    /** Coarse **/
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) override;
    virtual shared_ptr<BaseCoarseMap> BuildAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap);
#ifdef PWAGG
    virtual shared_ptr<BaseCoarseMap> BuildPWAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap);
#endif
    virtual shared_ptr<BaseCoarseMap> BuildECMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap);
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap,
						  shared_ptr<BaseAMGFactory::LevelCapsule> fcap, shared_ptr<BaseAMGFactory::LevelCapsule> ccap) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseAMGFactory::LevelCapsule> fcap) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap, shared_ptr<BaseAMGFactory::LevelCapsule> fcap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);

    virtual void CalcECOLWeightsSimple (BaseAMGFactory::State & state, Array<double> & vcw, Array<double> & ecw);
    virtual void CalcECOLWeightsRobust (BaseAMGFactory::State & state, Array<double> & vcw, Array<double> & ecw);


    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap_impl (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap,
							     shared_ptr<BaseAMGFactory::LevelCapsule> fcap);
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap_impl_v2 (shared_ptr<ProlMap<TSPM_TM>> pw_step, shared_ptr<BaseCoarseMap> cmap,
								shared_ptr<BaseAMGFactory::LevelCapsule> fcap);

    /** Discard **/
    virtual bool TryDiscardStep (BaseAMGFactory::State & state) override;
    virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & c_cap);

  }; // VertexAMGFactory
    
} // namespace amg

#endif
