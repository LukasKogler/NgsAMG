#ifndef FILE_AMG_FACTORY_HPP
#define FILE_AMG_FACTORY_HPP

namespace amg
{

  /** Factories set up coarse level matrices by building a series of grid-transfer operators (GridMaps/DofMaps). **/
  class BaseAMGFactory
  {
  public:
    class Logger;      // logging/printing
    class Options;     // configurable from outside
    struct State;      // internal book-keeping
    struct AMGLevel;   // contains one "level"

  protected:

    shared_ptr<Options> options;
    shared_ptr<Logger> logger;

  public:

    BaseAMGFactory (shared_ptr<Options> _opts);

    void SetUpLevels (Array<AMGLevel> & finest_level, shared_ptr<DOFMap> & dmap);

  protected:

    void RSU (Array<AMGLevel> & amg_levels, shared_ptr<DOFMap> & dof_map, State & state);

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TopologicMesh> amesh) const = 0;

    virtual shared_ptr<BaseDOFMapStep> DoStep (AMGLevel & f_lev, AMGLevel & c_lev, State & state);

    /** Used for controlling coarse levels and deciding on redistributing **/
    virtual size_t ComputeMeshMeasure (const TopologicMesh & m) const = 0;
    virtual double ComputeLocFrac (const TopologicMesh & m) const = 0;

    virtual State* NewState (AMGLevel & lev);
    virtual State* AllocState () const = 0;
    virtual void InitState (State & state, AMGLevel & lev) const ;

    /** Discard **/
    virtual bool TryDiscardStep (State & state) = 0;

    /** Coarse **/
    virtual size_t ComputeGoal (const AMGLevel & f_lev, State & state) = 0;
    virtual bool TryCoarseStep (State & state) = 0;
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (State & state) = 0;
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds) = 0;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh) = 0;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap) = 0;

    /** Redist **/
    virtual bool TryContractStep (State & state);
    virtual double FindRDFac (shared_ptr<TopologicMesh> cmesh);
    virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh) const = 0;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<ParallelDofs> fpd) const = 0;

  }; // BaseAMGFactory


  /** BS DOFs per node of type NT **/
  template<NODE_TYPE ANT, class ATMESH, int ABS>
  class NodalAMGFactory : public BaseAMGFactory
  {
  public:
    static constexpr int NT = ANT;
    static constexpr int BS = ABS;
    using TMESH = ATMESH;
    using OPTIONS = BaseAMGFactory::Options;

  public:    

    NodalAMGFactory (shared_ptr<Options> _opts);

  protected:    

    virtual size_t ComputeMeshMeasure (const TopologicMesh & m) const override;
    virtual double ComputeLocFrac (const TopologicMesh & m) const override;

    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TopologicMesh> amesh) const override;

    virtual size_t ComputeGoal (const AMGLevel & f_lev, State & state) override;

    virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh) const override;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<ParallelDofs> fpd) const override;
  }; // NodalAMGFactory

} // namespace amg

#endif
