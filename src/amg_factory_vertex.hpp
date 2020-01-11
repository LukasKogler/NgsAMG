#ifndef FILE_AMG_FACTORY_VERTEX_HPP
#define FILE_AMG_FACTORY_VERTEX_HPP

namespace amg
{

  /** DOFs attached to vertices, edge-wise energy. **/
  class VertexAMGFactoryOptions;

  template<class FACTORY_CLASS, class ATMESH, int ABS>
  class VertexAMGFactory : public NodalAMGFactory<NT_VERTEX, ABS>
  {
    static_assert(ABS == FACTORY_CLASS::BS, "BS must match");
    static_assert(ATMESH == FACTORY_CLASS::TMESH, "TMESH must match");

  public:
    using BASE_CLASS = NodalAMGFactory<NT_VERTEX, FACTORY_CLASS::BS>;
    using Options = BASE_CLASS::Options;
    using BS = ABS;
    using TMESH = TMESH;
    using FACTORY_CLASS::ENERGY;
    using TM = ENERGY::TM;
    using TSPM_TM = stripped_spm_tm<TM>;

  protected:

    /** Coarse **/
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (State & state) override;
    virtual shared_ptr<BaseCoarseMap> BuildAggMap (State & state);
    virtual shared_ptr<BaseCoarseMap> BuildECMap (State & state);
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (State & state) override;
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh) override;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap) override;
    // shared_ptr<BaseDOFMapStep> SP_impl (shared_ptr<ProlMap<TSPM_TM>> pw_prol, shared_ptr<TMESH> fmesh, FlatArray<int> vmap);

  }; // VertexAMGFactory
    
} // namespace amg

#endif
