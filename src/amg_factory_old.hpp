#ifndef FILE_AMG_FACTORY_HPP
#define FILE_AMG_FACTORY_HPP

namespace amg
{
  /**
     Factories build grid-transfer operators (DofMaps) and coarse level matrices.
     
     They assume canonical ordering of DOFs and need a finished finest mesh to start from.
   **/


  /** Utility **/

  template<class TOPTS, class TMAP, class TSCO>
  INLINE void UpdateCoarsenOpts (shared_ptr<TOPTS> opts, shared_ptr<TMAP> coarse_map, TSCO sco);


  /** Sets up prolongation- and coarse level matrices. (abstract class) **/
  template<class TMESH, class TM>
  class AMGFactory
  {
  public:
    using TSPM_TM = stripped_spm_tm<TM>;

    struct Options;    // configurable from the outside

    class Logger;     // logging and printing of information

    struct State;      // some internal book-keeping

    shared_ptr<BitArray> free_verts; // TODO: hacky!

  protected:
    shared_ptr<Options> options;
    shared_ptr<Logger> logger;

    shared_ptr<TMESH> finest_mesh;
    shared_ptr<BaseDOFMapStep> embed_step;

  public:

    AMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts,
		shared_ptr<BaseDOFMapStep> _embed_step = nullptr);
		

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    void SetupLevels (Array<shared_ptr<BaseSparseMatrix>> & mats, shared_ptr<DOFMap> & dmap);

  protected:

    // e.g NV for H1 amg
    virtual size_t ComputeMeshMeasure (const TMESH & m) const = 0;
    virtual double ComputeLocFrac (const TMESH & m) const = 0;

    struct Capsule
    {
      int level;
      shared_ptr<TMESH> mesh;
      shared_ptr<ParallelDofs> pardofs;
      shared_ptr<BaseSparseMatrix> mat;
    };

    Array<shared_ptr<BaseSparseMatrix>> RSU (Capsule & cap, State & state, shared_ptr<DOFMap> dof_map);

    shared_ptr<BaseDOFMapStep> STEP_AGG  (const Capsule & f_cap, State & state, Capsule & c_cap); // One coarse level with Aggregation
    shared_ptr<BaseDOFMapStep> STEP_ECOL (const Capsule & f_cap, State & state, Capsule & c_cap); // One coarse level with edge-collapsing
    shared_ptr<BaseDOFMapStep> STEP_COMB (const Capsule & f_cap, State & state, Capsule & c_cap); // Possibly Combined agg/ecol (probably buggy!)
    
    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TMESH> amesh) const = 0;

    virtual void SetCoarseningOptions (VWCoarseningData::Options & opts, shared_ptr<TMESH> mesh) const = 0;

    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const = 0;
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<AgglomerateCoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const = 0;
    virtual void SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const = 0;
    // virtual void SmoothProlongation2 (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const = 0;
    virtual void SmoothProlongationAgg (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<AgglomerateCoarseMap<TMESH>> agg_map) const = 0;
    virtual void SmoothProlongation_RealMat (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TSPM_TM> fmat) const = 0;

    // virtual void ModCoarseEdata (shared_ptr<ProlMap<TSPM_TM>> spmap, shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh) const = 0;

    virtual shared_ptr<AgglomerateCoarseMap<TMESH>> BuildAggMap  (shared_ptr<TMESH> mesh, bool dist2) const = 0;

    virtual shared_ptr<GridContractMap<TMESH>> BuildContractMap (double factor, shared_ptr<TMESH> mesh) const;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const = 0;

    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<VDiscardMap<TMESH>> dmap) const = 0;
    virtual shared_ptr<TSPM_TM> BuildSProl (shared_ptr<VDiscardMap<TMESH>> dmap) const = 0;

    double FindCTRFac (shared_ptr<TMESH> cmesh);
    bool DoContractStep (State & state);
  };


  /** DPN dofs per node of the specified kind **/
  template<NODE_TYPE NT, class TMESH, class TM>
  class NodalAMGFactory : public AMGFactory<TMESH, TM>
  {
  public:
    using BASE = AMGFactory<TMESH, TM>;
    using TSPM_TM = typename BASE::TSPM_TM;

    struct Options;

    NodalAMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts,
		     shared_ptr<BaseDOFMapStep> _embed_step = nullptr);

    virtual size_t ComputeMeshMeasure (const TMESH & m) const override;
    virtual double ComputeLocFrac (const TMESH & m) const override;

    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TMESH> amesh) const override;

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const override;

  protected:
    using BASE::options;
  };


  template<class FACTORY_CLASS, class TMESH, class TM>
  class VertexBasedAMGFactory : public NodalAMGFactory<NT_VERTEX, TMESH, TM>
  {
  public:
    using BASE = NodalAMGFactory<NT_VERTEX, TMESH, TM>;
    using TSPM_TM = typename BASE::TSPM_TM;

    struct Options;

    VertexBasedAMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts,
			   shared_ptr<BaseDOFMapStep> _embed_step = nullptr);

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    virtual size_t ComputeMeshMeasure (const TMESH & m) const override;
    virtual double ComputeLocFrac (const TMESH & m) const override;

    using BASE::free_verts;

  protected:
    using BASE::options;

    // virtual shared_ptr<CoarseMap<TMESH>> BuildCoarseMap  (shared_ptr<TMESH> mesh) const override;
    virtual shared_ptr<AgglomerateCoarseMap<TMESH>> BuildAggMap  (shared_ptr<TMESH> mesh, bool dist2) const override;

    template<class TMAP> shared_ptr<TSPM_TM> BuildPWProl_impl (shared_ptr<TMAP> cmap, shared_ptr<ParallelDofs> fpd) const;
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const override
    { return BuildPWProl_impl(cmap, fpd); }
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<AgglomerateCoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const override
    { return BuildPWProl_impl(cmap, fpd); }

    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<VDiscardMap<TMESH>> dmap) const override;

    virtual shared_ptr<TSPM_TM> BuildSProl (shared_ptr<VDiscardMap<TMESH>> dmap) const override;

    virtual void SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const override;
    // virtual void SmoothProlongation2 (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const override;
    virtual void SmoothProlongationAgg (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<AgglomerateCoarseMap<TMESH>> agg_map) const override;
    virtual void SmoothProlongation_RealMat (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TSPM_TM> fmat) const override;
    // virtual void ModCoarseEdata (shared_ptr<ProlMap<TSPM_TM>> spmap, shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh) const;
  };

} // namespace amg

#endif // FILE_AMG_FACTORY_HPP
