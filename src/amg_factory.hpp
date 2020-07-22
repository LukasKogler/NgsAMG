#ifndef FILE_AMG_FACTORY_HPP
#define FILE_AMG_FACTORY_HPP

#include "amg.hpp"
#include "amg_mesh.hpp"
#include "amg_map.hpp"
#include "amg_coarsen.hpp"
#include "amg_discard.hpp"
#include "amg_contract.hpp"

namespace amg
{

  /** Factories set up coarse level matrices by building a series of grid-transfer operators (GridMaps/DofMaps). **/
  class BaseAMGFactory
  {
  public:
    class Logger;         // logging/printing
    class Options;        // configurable from outside
    struct State;         // internal book-keeping
    struct LevelCapsule;  // Contains everything for one level on it's own - at least a mesh and pardofs.
    struct AMGLevel;      // Contains one LevelCapsule, plus maps leading to the next level, plus potentially an embedding on the finest level

  protected:

    shared_ptr<Options> options;
    shared_ptr<Logger> logger;

  public:

    BaseAMGFactory (shared_ptr<Options> _opts);

    void SetUpLevels (Array<shared_ptr<AMGLevel>> & finest_level, shared_ptr<DOFMap> & dmap);

    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TopologicMesh> amesh) const = 0;

    virtual shared_ptr<LevelCapsule> AllocCap () const; // weird, but I need to call this from PC

  protected:

    void RSU (Array<shared_ptr<AMGLevel>> & amg_levels, shared_ptr<DOFMap> & dof_map, State & state);

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    virtual void MapLevel2 (shared_ptr<BaseDOFMapStep> & dof_step, shared_ptr<AMGLevel> & f_cap, shared_ptr<AMGLevel> & c_cap);
    virtual shared_ptr<BaseDOFMapStep> MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps,
						 shared_ptr<AMGLevel> & f_cap, shared_ptr<AMGLevel> & c_cap);

    virtual shared_ptr<BaseDOFMapStep> DoStep (shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev, State & state);

    /** Used for controlling coarse levels and deciding on redistributing **/
    virtual size_t ComputeMeshMeasure (const TopologicMesh & m) const = 0;
    virtual double ComputeLocFrac (const TopologicMesh & m) const = 0;

    virtual State* NewState (shared_ptr<AMGLevel> & lev);
    virtual State* AllocState () const = 0;
    virtual void InitState (State & state, shared_ptr<AMGLevel> & lev) const;

    /** Discard **/
    virtual bool TryDiscardStep (State & state) = 0;
    // virtual shared_ptr<BaseDiscardMap> BuildDiscardMap (State & state);

    /** Coarse **/
    virtual size_t ComputeGoal (const shared_ptr<AMGLevel> & f_lev, State & state) = 0;
    virtual bool TryCoarseStep (State & state);
    virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (State & state, shared_ptr<LevelCapsule> & mapped_cap) = 0;
    virtual shared_ptr<BaseDOFMapStep> PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<LevelCapsule> fcap, shared_ptr<LevelCapsule> ccap) = 0;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<LevelCapsule> fcap) = 0;
    virtual shared_ptr<BaseDOFMapStep> SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap, shared_ptr<LevelCapsule> fcap) = 0;

    /** Redist **/
    virtual bool TryContractStep (State & state);
    virtual double FindRDFac (shared_ptr<TopologicMesh> cmesh);
    virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh, shared_ptr<LevelCapsule> & mapped_cap) const = 0;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<ParallelDofs> fpd, shared_ptr<LevelCapsule> & mapped_cap) const = 0;

  }; // BaseAMGFactory


  /** Options **/

  template<class OC>
  struct SpecOpt
  {
  public:
    OC default_opt;
    Array<OC> spec_opt;
    SpecOpt () : spec_opt(0) { ; }
    SpecOpt (OC _default_opt) : default_opt(_default_opt), spec_opt(0) { ; }
    SpecOpt (OC _default_opt, Array<OC> & _spec_opt) : default_opt(_default_opt), spec_opt(_spec_opt) { ; }
    SpecOpt (SpecOpt<OC> && other) : default_opt(other.default_opt), spec_opt(move(other.spec_opt)) { ; }
    SpecOpt (const Flags & flags, OC defopt, string defkey, string speckey, Array<string> optkeys)
      : SpecOpt(defopt)
    { SetFromFlagsEnum(flags, defkey, speckey, move(optkeys)); }
    SpecOpt<OC> & operator = (OC defoc) {
      default_opt = defoc;
      spec_opt.SetSize0();
      return *this;
    }
    void SetFromFlagsEnum (const Flags & flags, string defkey, string speckey, Array<string> optkeys)
    {
      auto setoc = [&](auto & oc, string val) {
	int index;
	if ( (index = optkeys.Pos(val)) != -1)
	  { oc = OC(index); }
      };
      setoc(default_opt, flags.GetStringFlag(defkey, ""));
      auto & specfa = flags.GetStringListFlag(speckey);
      spec_opt.SetSize(specfa.Size()); spec_opt = default_opt;
      for (auto k : Range(spec_opt))
	{ setoc(spec_opt[k], specfa[k]); }
    }
    OC GetOpt (int level) { return (level < spec_opt.Size()) ? spec_opt[level] : default_opt; }
  };

  class BaseAMGFactory :: Options
  {
  public:
    /** General Level-Control **/
    size_t max_n_levels = 10;                   // maximun number of multigrid levels (counts first level, so at least 2)
    size_t min_meas = 0;                        // minimum measure of coarsest mesh
    size_t max_meas = 50;                       // maximal maesure of coarsest mesh

    /** Coarsening **/
    // TODO: deal with multistep/interleave properly
    bool enable_multistep = false;              // allow chaining multiple coarsening steps
    bool use_static_crs = true;                 // use static coarsening ratio
    double aaf = 0.1;                           // (static crs ratio) chain edge-collapse maps until mesh is decreased by factor aaf
    double first_aaf = 0.05;                    // (static crs ratio) (smaller) factor for first level. -1 for dont use
    double aaf_scale = 1;                       // (static crs ratio) scale aaf, e.g if 2:   first_aaf, aaf, 2*aaf, 4*aaf, .. (or aaf, 2*aaf, ...)
    bool enable_dyn_crs = true;                 // use dynamic coarsening ratios
    int n_levels_d2_agg = 1;                    // do this many levels MIS(2)-like aggregates (afterwards MIS(1)-like)

    /** Contract (Re-Distribute) **/
    bool enable_redist = true;                  // allow re-distributing on coarse levels
    bool enable_static_redist = false;          // redist after a fixed coarsening ratio
    double rdaf = 0.05;                         // contract after reducing measure by this factor
    double first_rdaf = 0.025;                  // see first_aaf
    double rdaf_scale = 1;                      // see aaf_scale
    double rd_crs_thresh = 0.9;                 // if coarsening slows down more than this, redistribute
    double rd_loc_thresh = 0.5;                 // if less than this fraction of vertices are purely local, redistribute
    double rd_pfac = 0.25;                      // per default, reduce active NP by this factor (rd_pfac / rdaf should be << 1 !)
    size_t rd_min_nv_th = 500;                  // re-distribute when there are less than this many vertices per proc left
    size_t rd_min_nv_gl = 500;                  // try to re-distribute such that at least this many NV per proc remain
    size_t rd_seq_nv = 1000;                    // always re-distribute to sequential once NV reaches this threshhold
    double rd_loc_gl = 0.8;                     // always try to redistribute such that at least this fraction will be local

    /** Discard **/
    bool enable_disc = true;                    // enable discarding of vertices (should eliminate hanging nodes)

    /** Smoothed Prolongation **/
    bool enable_sp = true;                      // enable prolongation-smoothing
    bool sp_needs_cmap = true;                  // do we need the coarse map for smoothed prol?
    double sp_min_frac = 0.1;                   // min. (relative) wt to include an edge
    int sp_max_per_row = 3;                     // maximum entries per row (should be >= 2!)
    double sp_omega = 1.0;                      // relaxation parameter for prol-smoothing

    /** Output **/
    bool keep_grid_maps = false;                // do we need grid maps later on (e.g for building block-smoothers)?

    /** Logging **/
    enum LOG_LEVEL : char { NONE   = 0,         // nothing
			    BASIC  = 1,         // summary info
			    NORMAL = 2,         // global level-wise info
			    EXTRA  = 3};        // local level-wise info
    LOG_LEVEL log_level = LOG_LEVEL::NORMAL;    // how much info do we collect
    bool print_log = true;                      // print log to shell
    string log_file = "";                       // which file to print log to (none if empty)

    Options () { ; }

    virtual void SetFromFlags (const Flags & flags, string prefix);

  }; // BaseAMGFactory::Options

  /** END Options **/


  /** State **/

  struct BaseAMGFactory::State
  {
    INT<3> level;
    /** most "current" objects **/
    shared_ptr<BaseAMGFactory::LevelCapsule> curr_cap;
    /** built maps **/
    shared_ptr<BaseDOFMapStep> dof_map;
    shared_ptr<BaseDiscardMap> disc_map;
    shared_ptr<BaseCoarseMap> crs_map;
    /** book-keeping **/
    bool first_redist_used = false;
    size_t last_redist_meas = 0;
    bool need_rd = false;
  }; // BaseAMGFactory::State

  /** END State **/


  /** AMGLevel **/

  struct BaseAMGFactory::LevelCapsule
  {
    shared_ptr<EQCHierarchy> eqc_h = nullptr;
    shared_ptr<TopologicMesh> mesh = nullptr;
    shared_ptr<ParallelDofs> pardofs = nullptr;
    shared_ptr<BaseSparseMatrix> mat = nullptr;
    shared_ptr<BitArray> free_nodes = nullptr;

    virtual ~LevelCapsule () { ; } // so we can cast down to child classes
  }; // struct BaseAMGFactory::LevelCapsule

  /** END AMGLevel **/


  /** AMGLevel **/

  struct BaseAMGFactory::AMGLevel
  {
    int level;
    shared_ptr<BaseAMGFactory::LevelCapsule> cap;
    shared_ptr<BaseDOFMapStep> embed_map = nullptr; // embedding
    bool embed_done = false;
    // map to next level: embed -> disc -> crs (-> ctr)
    shared_ptr<BaseDiscardMap> disc_map = nullptr;
    shared_ptr<BaseCoarseMap> crs_map = nullptr;
  }; // struct BaseAMGFactory::AMGLevel

  /** END AMGLevel**/


  /** Logger **/

  class BaseAMGFactory::Logger
  {
  public:
    using LOG_LEVEL = typename BaseAMGFactory::Options::LOG_LEVEL;
    Logger (LOG_LEVEL _lev, int max_levels = 10)
      : lev(_lev), ready(false)
    { Alloc(max_levels); }
    void LogLevel (BaseAMGFactory::AMGLevel & cap);
    void Finalize ();
    void PrintLog (ostream & out);
    void PrintToFile (string file_name);
  protected:
    LOG_LEVEL lev;
    bool ready;
    void Alloc (int N);
    /** BASIC level - summary info **/
    double v_comp;
    double op_comp;
    /** NORMAL level - global info per level **/
    Array<double> vcc;                   // vertex complexity components
    Array<double> occ;                   // operator complexity components
    Array<size_t> NVs;                   // # of vertices
    Array<size_t> NEs;                   // # of edges
    Array<size_t> NPs;                   // # of active procs
    Array<size_t> NZEs;                  // # of NZEs (can be != occ for systems)
    /** EXTRA level - local info per level **/
    int vccl_rank;                       // rank with max/min local vertex complexity
    double v_comp_l;                     // max. loc vertex complexity
    Array<double> vccl;                  // components for vccl
    int occl_rank;                       // rank with max/min local operator complexity
    double op_comp_l;                    // max. loc operator complexity
    Array<double> occl;                  // components for occl
    /** internal **/
    NgsAMG_Comm comm;
  }; // class Logger

  /** END Logger **/


} // namespace amg

#endif
