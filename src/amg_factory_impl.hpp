#ifndef FILE_AMG_FACTORY_IMPL_HPP
#define FILE_AMG_FACTORY_IMPL_HPP

namespace amg
{
  /** Options **/

  class BaseAMGFactory :: Options
  {
  protected:
    string prefix = "ngs_amg_";                 // prefix for flags

  public:
    /** General Level-Control **/
    size_t max_n_levels = 10;                   // maximun number of multigrid levels (counts first level, so at least 2)
    size_t max_meas = 50;                       // maximal maesure of coarsest mesh

    /** Coarsening **/
    // TODO: deal with multistep/interleave properly
    bool enable_multistep = true;               // allow chaining multiple coarsening steps
    bool use_static_crs = true;                 // use static coarsening ratio
    double aaf = 0.1;                           // (static crs ratio) chain edge-collapse maps until mesh is decreased by factor aaf
    double first_aaf = 0.05;                    // (static crs ratio) (smaller) factor for first level. -1 for dont use
    double aaf_scale = 1;                       // (static crs ratio) scale aaf, e.g if 2:   first_aaf, aaf, 2*aaf, 4*aaf, .. (or aaf, 2*aaf, ...)
    bool enable_dyn_crs = true;                 // use dynamic coarsening ratios

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

    Options (string _prefix = "ngs_amg_")
      : prefix(_prefix)
    { ; }

    virtual void SetFromFlags (const Flags & flags);

  }; // BaseAMGFactory::Options


  void BaseAMGFactory::Options :: SetFromFlags (const Flags & flags)
  {
    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals) {
      string val = flags.GetStringFlag(prefix + key, "");
      for (auto k : Range(vals)) {
	if (val == vals[k])
	  { opt = decltype(opt)(k); return; }
      }
    };
    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };
    auto set_num = [&](auto& v, string key)
      { v = flags.GetNumFlag(prefix + key, v); };
    
    set_num(max_n_levels, "max_levels");
    set_num(max_meas, "max_coarse_size");

    set_bool(enable_multistep, "enable_multistep");
    set_bool(enable_dyn_crs, "enable_dyn_crs");
    set_num(aaf, "aaf");
    set_num(first_aaf, "first_aaf");
    set_num(aaf_scale, "aafaaf_scale");

    set_bool(enable_redist, "enable_redist");
    set_num(rdaf, "rdaf");
    set_num(first_rdaf, "first_rdaf");
    set_num(rdaf_scale, "rdaf_scale");
    set_num(rd_crs_thresh, "rd_crs_thresh");
    set_num(rd_loc_thresh, "rd_loc_thresh");
    set_num(rd_pfac, "rd_pfac");
    set_num(rd_min_nv_th, "rd_min_nv_thresh");
    set_num(rd_min_nv_gl, "rd_min_nv_gl");
    set_num(rd_seq_nv, "rd_seq_nv");
    set_num(rd_loc_gl, "rd_loc_gl");

    set_bool(enable_sp, "enable_sp");
    set_bool(sp_needs_cmap, "sp_needs_cmap");
    set_num(sp_min_frac, "sp_min_frac");
    set_num(sp_max_per_row, "sp_max_per_row");
    set_num(sp_omega, "sp_omega");

    set_enum_opt(log_level, "log_level", {"none", "basic", "normal", "extra"});
    set_bool(print_log, "print_log");
    log_file = flags.GetStringFlag(prefix + string("log_file"), "");
  } // BaseAMGFactory::Options :: SetFromFlags


  /** END Options **/


  /** State **/

  struct BaseAMGFactory::State
  {
    INT<3> level;
    /** most "current" objects **/
    shared_ptr<TopologicMesh> curr_mesh;
    shared_ptr<ParallelDofs> curr_pds;
    /** built maps **/
    shared_ptr<BaseDOFMapStep> dof_map;
    shared_ptr<BaseDiscardMap> disc_map;
    shared_ptr<BaseCoarseMap> crs_map;
    shared_ptr<BitArray> free_nodes;
    /** book-keeping **/
    bool first_redist_used;
    size_t last_redist_meas;
  }; // BaseAMGFactory::State


  // template<NODE_TYPE NT, int BS, class TMESH>
  // class NodalAMGFactory<NT, BS,TMESH>::State : public BaseAMGFactory::State
  // {
  // }; // NodalAMGFactory::State

  /** END State **/


  /** AMGLevel **/

  struct BaseAMGFactory::AMGLevel
  {
    int level;
    shared_ptr<TopologicMesh> mesh = nullptr;
    shared_ptr<ParallelDofs> pardofs = nullptr;
    shared_ptr<BaseSparseMatrix> mat = nullptr;
    shared_ptr<BaseDOFMapStep> embed_map = nullptr; // embedding
    // map to next level: embed -> disc -> crs (-> ctr)
    shared_ptr<BaseDiscardMap> disc_map = nullptr;
    shared_ptr<BaseCoarseMap> crs_map = nullptr;
    shared_ptr<BitArray> free_nodes = nullptr;
  }; // BaseAMGFactory::AMGLevel

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


  void BaseAMGFactory::Logger :: Alloc (int N)
  {
    auto lam = [N](auto & a) { a.SetSize(N); a.SetSize0(); };
    lam(vcc); lam(occ); // need it as work array even for basic
    if ( (lev == LOG_LEVEL::NONE) || (lev == LOG_LEVEL::BASIC) )
      { return; }
    lam(NVs);
    lam(NEs);
    lam(NPs);
    if (lev == LOG_LEVEL::NORMAL)
      { return; }
    lam(vccl);
    lam(occl);
  } // Logger::Alloc


  void BaseAMGFactory::Logger :: LogLevel (BaseAMGFactory::AMGLevel & cap)
  {
    ready = false;
    if (cap.level == 0)
      { comm = cap.mesh->GetEQCHierarchy()->GetCommunicator(); }
    if (lev == LOG_LEVEL::NONE)
      { return; }
    auto lev_comm = cap.mesh->GetEQCHierarchy()->GetCommunicator();
    vcc.Append(cap.mesh->template GetNNGlobal<NT_VERTEX>());
    // cout << " occ-comp would be " << cap.mat->NZE() << " * " << GetEntrySize(cap.mat.get())
    // 	 << " = " << cap.mat->NZE() * GetEntrySize(cap.mat.get()) << endl;
    // cout << " mat " << cap.mat << " " << typeid(*cap.mat).name() << endl;
    if (lev_comm.Rank() == 0)
      { occ.Append(lev_comm.Reduce(cap.mat->NZE() * GetEntrySize(cap.mat.get()), MPI_SUM, 0)); }
    else
      { lev_comm.Reduce(cap.mat->NZE() * GetEntrySize(cap.mat.get()), MPI_SUM, 0); }
    if (lev == LOG_LEVEL::BASIC)
      { return; }
    NVs.Append(cap.mesh->template GetNNGlobal<NT_VERTEX>());
    NEs.Append(cap.mesh->template GetNNGlobal<NT_EDGE>());
    NPs.Append(cap.mesh->GetEQCHierarchy()->GetCommunicator().Size());
    NZEs.Append(lev_comm.Reduce(cap.mat->NZE(), MPI_SUM, 0));
    if (lev == LOG_LEVEL::NORMAL)
      { return; }
    vccl.Append(cap.mesh->template GetNN<NT_VERTEX>());
    occl.Append(cap.mat->NZE() * GetEntrySize(cap.mat.get()));
  } // Logger::LogLevel


  void BaseAMGFactory::Logger :: Finalize ()
  {
    ready = true;
    if (lev == LOG_LEVEL::NONE)
      { return; }
    op_comp = v_comp = 0;
    if (comm.Rank() == 0) { // only 0 has op-comp
      double vcc0 = vcc[0];
      for (auto& v : vcc)
	{ v /= vcc0; v_comp += v; }
      double occ0 = occ[0];
      for (auto& v : occ)
	{ v /= occ0; op_comp += v;}
    }
    if (lev == LOG_LEVEL::BASIC)
      { return; }
    if (lev == LOG_LEVEL::NORMAL)
      { return; }
    auto alam = [&](auto & rk, auto & val, auto & array) {
      if (comm.Size() > 1) {
	auto maxval = comm.AllReduce(val, MPI_MAX);
	rk = comm.AllReduce( (val == maxval) ? (int)comm.Rank() : (int)comm.Size(), MPI_MIN);
	if ( (comm.Rank() == 0) && (rk != 0) )
	  { comm.Recv(array, rk, MPI_TAG_AMG); }
	else if ( (comm.Rank() == rk) && (rk != 0) )
	  { comm.Send(array, 0, MPI_TAG_AMG); }
	val = maxval;
      }
      else
	{ rk = 0; }
    };
    v_comp_l = 0;
    double vccl0 = max2(vccl[0], 1.0);
    for (auto& v : vccl)
      { v /= vccl0; v_comp_l += v; }
    alam(vccl_rank, v_comp_l, vccl);
    op_comp_l = 0;
    double occl0 = max2(occl[0], 1.0);
    for (auto& v : occl)
      { v /= occl0; op_comp_l += v; }
    // cout << "loc OCC: " << op_comp_l << endl;
    // prow(occl); cout << endl;
    alam(occl_rank, op_comp_l, occl);
  } // Logger::Finalize


  void BaseAMGFactory::Logger :: PrintLog (ostream & out)
  {
    if (!ready)
      { Finalize(); }
    if (comm.Rank() != 0)
      { return; }
    if (lev == LOG_LEVEL::NONE)
      { return; }
    out << endl << " ---------- AMG Summary ---------- " << endl;
    if (lev >= LOG_LEVEL::BASIC) {
      out << "Vertex complexity: " << v_comp << endl;
      out << "Operator complexity: " << op_comp << endl;
    }
    if (lev >= LOG_LEVEL::NORMAL) {
      out << "Vertex complexity components: "; prow(vcc, out); out << endl;
      out << "Operator complexity components: "; prow(occ, out); out << endl;
      out << "# vertices "; prow(NVs); out << endl;
      out << "# edges: "; prow(NEs); out << endl;
      out << "# procs: "; prow(NPs); out << endl;
      out << "NZEs:"; prow(NZEs); out << endl;
    }
    if (lev >= LOG_LEVEL::EXTRA) {
      out << "max. loc. vertex complexity is " << v_comp_l << " on rank " << vccl_rank << endl;
      out << "max. loc. vertex complexity components: "; prow(vccl, out); out << endl;
      out << "max. loc. operator complexity is " << op_comp_l << " on rank " << occl_rank << endl;
      out << "max. loc. operator complexity components: "; prow(occl, out); out << endl;
    }
    out << " ---------- AMG Summary End ---------- " << endl << endl;
  } // Logger::PrintLog


  void BaseAMGFactory::Logger :: PrintToFile (string file_name)
  {
    if (comm.Rank() == 0) {
      ofstream out(file_name, ios::out);
      PrintLog(out);
    }
  } // Logger::PrintToFile

  /** END Logger **/


  /** BaseAMGFactory **/

  BaseAMGFactory :: BaseAMGFactory (shared_ptr<Options> _opts)
    : options(_opts)
  {
    logger = make_shared<Logger>(options->log_level);
  } // BaseAMGFactory(..)


  void BaseAMGFactory :: SetUpLevels (Array<BaseAMGFactory::AMGLevel> & amg_levels, shared_ptr<DOFMap> & dof_map)
  {
    static Timer t("SetupLevels"); RegionTimer rt(t);

    const auto & O(*options);

    logger = make_shared<Logger>(options->log_level);

    amg_levels.SetAllocSize(O.max_n_levels);
    auto & finest_level = amg_levels[0];

    // Array<AMGLevel> amg_levels(O.max_n_levels); amg_levels.SetSize(1);
    // amg_levels[0] = finest_level;

    /** rank 0 (and maybe sometimes others??) can also not have an embed_step, while others DO.
	ParallelDofs - constructor has to be called by every member of the communicator! **/
    auto fmat = amg_levels[0].mat;
    auto embed_step = amg_levels[0].embed_map;
    int have_embed = fmat->GetParallelDofs()->GetCommunicator().AllReduce((embed_step == nullptr) ? 0 : 1, MPI_SUM);
    amg_levels[0].pardofs = (have_embed == 0) ? fmat->GetParallelDofs() : BuildParallelDofs(amg_levels[0].mesh);

    auto state = NewState(amg_levels[0]);

    RSU(amg_levels, dof_map, *state);

    if (options->print_log)
      { logger->PrintLog(cout); }
    if (options->log_file.size() > 0)
      { logger->PrintToFile(options->log_file); }

    logger = nullptr;
    delete state;
  } // BaseAMGFactory::SetUpLevels


  void BaseAMGFactory :: RSU (Array<AMGLevel> & amg_levels, shared_ptr<DOFMap> & dof_map, State & state)
  {
    const auto & O(*options);

    auto & f_lev = amg_levels.Last();

    AMGLevel c_lev;

    logger->LogLevel (f_lev);

    shared_ptr<BaseDOFMapStep> step = DoStep(f_lev, c_lev, state);

    if ( (step == nullptr) || (c_lev.mesh == f_lev.mesh) || (c_lev.mat == f_lev.mat) )
      { return; } // step not performed correctly - coarsening is probably stuck
    else
      { dof_map->AddStep(step); }

    if (!O.keep_grid_maps){
      f_lev.disc_map = nullptr;
      f_lev.crs_map = nullptr;
    }

    /** Recursive call (or return) **/
    if ( (c_lev.mesh == nullptr) || (c_lev.mat == nullptr) ) // dropped out (redundand "||" ?)
      { amg_levels.Append(move(c_lev)); return; }
    else if ( (f_lev.level + 2 == O.max_n_levels) ||                // max n levels reached
    	      (O.max_meas >= ComputeMeshMeasure (*c_lev.mesh) ) ) { // max coarse size reached
      logger->LogLevel (c_lev);
      amg_levels.Append(move(c_lev));
      return;
    }
    else { // more coarse levels
      amg_levels.Append(move(c_lev));
      RSU (amg_levels, dof_map, state);
      return;
    }
  } // BaseAMGFactory::RSU


  shared_ptr<BaseDOFMapStep> BaseAMGFactory :: DoStep (AMGLevel & f_lev, AMGLevel & c_lev, State & state)
  {
    const auto & O(*options);

    size_t curr_meas = ComputeMeshMeasure(*state.curr_mesh), goal_meas = ComputeGoal(f_lev, state);

    if (f_lev.level == 0)
      { state.last_redist_meas = curr_meas; }

    shared_ptr<BaseDOFMapStep> embed_step = move(f_lev.embed_map), disc_step;

    // CalcCoarsenOpts(state); // TODO: proper update of coarse cols for ecol?

    if ( O.enable_redist && (f_lev.level != 0) ) {
      if ( TryDiscardStep(state) ) {
	disc_step = move(state.dof_map);
	curr_meas = ComputeMeshMeasure(*state.curr_mesh);
      }
    }

    if (O.keep_grid_maps)
      { f_lev.disc_map = state.disc_map; }

    shared_ptr<BaseDOFMapStep> prol_map, rd_map;
    { /** Coarse/Redist maps - constructed interleaved, but come out untangled. **/
      bool have_options = O.enable_redist;
      bool goal_reached = (curr_meas < goal_meas);
      Array<shared_ptr<BaseCoarseMap>> cm_chunks;
      Array<shared_ptr<BaseDOFMapStep>> rd_chunks;
      shared_ptr<TopologicMesh> sp_fm = state.curr_mesh;

      do { /** coarsen until goal reached or stuck **/
	cm_chunks.SetSize0();
	auto sp_fm = state.curr_mesh;
	auto comm = static_cast<BlockTM&>(*sp_fm).GetEQCHierarchy()->GetCommunicator();
	bool could_recover = ( O.enable_redist && (comm.Size() > 2) && (state.level[2] == 0) );
	bool crs_stuck = false;
	do { /** coarsen until stuck or goal reached **/
	  auto fm = state.curr_mesh;
	  auto fpds = state.curr_pds;
	  if ( TryCoarseStep(state) ) { // coarse map constructed successfully
	    state.level[1]++; state.level[2] = 0; // count up sub-coarse, reset redist
	    size_t f_meas = ComputeMeshMeasure(*fm), c_meas = ComputeMeshMeasure(*state.curr_mesh);
	    double meas_fac = (f_meas == 0) ? 0 : c_meas / double(f_meas);
	    goal_reached = (c_meas < goal_meas);
	    if ( (goal_reached) && (cm_chunks.Size() > 0) && (f_meas * c_meas < sqr(goal_meas)) ) {
	      // last step was closer than current step - reset to last step
	      state.curr_mesh = fm;
	      state.curr_pds = fpds;
	      state.crs_map = nullptr;
	      state.disc_map = nullptr;
	      state.dof_map = nullptr;
	      goal_reached = true;
	    }
	    else
	      { curr_meas = c_meas; }
	    if ( (!goal_reached) && could_recover)
	      { crs_stuck |= meas_fac > O.rd_crs_thresh; }  // bad coarsening - try to recover via redist
	    auto cm = move(state.crs_map);
	    auto gm = dynamic_pointer_cast<BaseGridMapStep>(cm);
	    if (cm == nullptr)
	      { throw Exception("Not a BGM??"); }
	    cm_chunks.Append(cm);
	  }
	  else // coarsening failed completely
	    { crs_stuck = true; }
	} while ( (!crs_stuck) && (!goal_reached) );

	/** concatenate coarse map chunks **/
	shared_ptr<BaseCoarseMap> c_step = nullptr;
	if ( cm_chunks.Size() )
	  { c_step = cm_chunks.Last(); }
	for (int l = cm_chunks.Size() - 2; l >= 0; l--)
	  { c_step = cm_chunks[l]->Concatenate(c_step); }

	if (f_lev.crs_map == nullptr)
	  { f_lev.crs_map = c_step; }

	if (c_step != nullptr) {
	  /** build pw-prolongation **/
	  auto pmap = BuildPWProlMap(c_step, fpds, state.curr_pds);

	  /** If we need meshes for sprol, we have to do this here.
	      Thats the best we can do. **/
	  if ( O.enable_sp && O.sp_needs_meshes && (sp_fm != nullptr) )
	    { pmap = SmoothProlMap(pmap, sp_fm, state.curr_mesh); sp_fm = nullptr; }

	  /** pull back prol through redists **/
	  for (int k = rd_chunks.Size()-1; k >=0; k--)
	    { pmap = rd_chunks[k]->PullBack(pmap); }
	  
	  prol_map = (prol_map == nullptr) ? pmap : prol_map->Concatenate(pmap);
	}

	/** try to recover via redistribution **/
	if (O.enable_redist && (state.level[2] == 0) ) {
	  if ( TryContractStep(state) ) { // redist successfull
	    state.level[2]++; // count up redist
	    rd_chunks.Apend(move(state.dof_step));
	  }
	  else // redist rejected
	    { could_recover = false; }
	}
	else // already distributed once
	  { could_recover = false; }
      } while ( (could_recover) && (curr_meas > goal_meas) );

      /** If not forced to before, smooth prol here.
	  TODO: check if other version would be possible here (that one should be better anyways!)  **/
      if ( (O.enable_sp) && (!O.sp_needs_meshes) && (prol_map != nullptr) )
	{ prol_map = SmoothedProlMap(prol_map, fm); }

      /** pack rd-maps into one step**/
      if (rd_chunks.Size() == 1)
	{ rd_map = rd_chunks[0]; }
      else
	{ rd_map = make_shared<ConcDMS>(rd_chunks); }
    }

    /** We now have emb - disc - prol - rd. Concatenate and pack into final map. **/
    Array<shared_ptr<BaseDOFMapStep>> sub_steps, init_steps;
    if (embed_step != nullptr)
      { init_steps.Append(embed_step); }
    if (disc_step != nullptr)
      { init_steps.Append(disc_step); }
    if (prol_map != nullptr)
      { init_steps.Append(prol_map); }
    if (rd_map != nullptr)
      { init_steps.Append(rd_map); }
    for (int k = 0; k < 4; k++) {
      shared_ptr<BaseDOFMapStep> conc_step = init_steps[k];
      int j = k+1;
      for ( ; j < 4; j++)
	if (auto x = conc_step->Concatenate(init_steps[j]))
	  { conc_step = x; k++; }
	else
	  { break; }
      sub_steps.Append(conc_step);
    }
    init_steps = nullptr;

    shared_ptr<BaseDOFMapStep> final_step = nullptr;
    if (sub_steps.Size() == 1)
      { final_step = sub_steps[0]; }
    else if (sub_steps.Size() > 1)
      { final_step = make_shared<ConcDMS>(sub_steps); }

    /** assemble coarse level matrix and set return vals **/
    c_lev.level = f_lev.level + 1;
    c_lev.mesh = state.curr_mesh;
    c_lev.pardofs = state.curr_pds;
    c_lev.mat = final_step->AssembleMatrix(f_lev.mat);

    return final_step;
  } // BaseAMGFactory::DoStep


  bool BaseAMGFactory :: TryCoarseStep (State & state)
  {
    /** build coarse map **/
    shared_ptr<BaseGridMapStep> cmap = BuildCoarseMap(state);

    if (cmap == nullptr) // could not build map
      { return false; }

    /** check if the step was ok **/
    bool accept_crs = true;

    auto cmesh = cmap->GetMappedMesh();
    auto comm = state.curr_mesh

    size_t f_meas = ComputeMeshMeasure(*state.mesh), c_meas = ComputeMeshMeasure(*cmesh);
    double cfac = (f_meas == 0) ? 0 : c_meas / f_meas;

    bool map_op = true;

    map_ok &= ( c_meas < f_meas ); // coarsening is stuck

    if (O.enable_redist)
      { map_ok &= (cfac < O.rd_crs_thresh); }

    if (!map_ok)
      { return false; }

    state.curr_mesh = cmesh;
    state.curr_pds = BuildParallelDofs(*cmesh);
    state.grid_map = cmap;

    return true;
  } // BaseAMGFactory::TryCoarseStep


  double BaseAMGFactory :: FindRDFac (shared_ptr<TopologicMesh> mesh)
  {
    const auto & O(*options);
    const auto & M(*mesh);
    const auto& eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();

    auto NV = M.template GetNNGlobal<NT_VERTEX>();

    /** Already "sequential" **/
    if (comm.Size() <= 2)
      { return 1; }

    /** Sequential threshold reached **/
    if (NV < options->rd_seq_nv)
      { return -1; }

    /** default redist-factor **/
    double rd_factor = O.rd_pfac;
    
    /** ensure enough vertices per proc **/
    rd_factor = min2(rd_factor, double(NV) / O.rd_min_nv_gl / comm.Size());

    /** try to heuristically ensure that enough vertices are local **/
    // size_t nv_loc = M.template GetNN<NT_VERTEX>() > 0 ? M.template GetENN<NT_VERTEX>(0) : 0;
    // double frac_loc = comm.AllReduce(double(nv_loc), MPI_SUM) / NV;
    double frac_loc = ComputeLocFrac(M);
    if (frac_loc < options->rd_loc_thresh) {
      size_t NP = comm.Size();
      size_t NF = comm.AllReduce (eqc_h.GetDistantProcs().Size(), MPI_SUM) / 2; // every face counted twice
      double F = frac_loc;
      double FGOAL = O.rd_loc_gl; // we want to achieve this large of a frac of local verts
      double FF = (1-frac_loc) / NF; // avg frac of a face
      double loc_fac = 1;
      if (F + NP/2 * FF > FGOAL) // merge 2 -> one face per group becomes local
	{ loc_fac = 0.5; }
      else if (F + NP/4 * 5 * FF > FGOAL) // merge 4 -> probably 4(2x2) to 6(tet) faces per group
	{ loc_fac = 0.25; }
      else // merge 8, in 3d probably 12 edges (2x2x2) per group. probably never want to do more than that
	{ loc_fac = 0.125; }
      rd_factor = min2(rd_factor, loc_fac);
    }

    /** ALWAYS at least factor 2 **/
    rd_factor = min2(0.5, rd_factor);

    return rd_factor;
  } // BaseAMGFactory::FindRDFac


  bool BaseAMGFactory :: TryContractStep (State & state, bool need_rd)
  {
    static Timer t("TryContractStep");
    RegionTimer rt(t);

    const auto & O(*options);
    const auto & M = *state.curr_mesh;
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    double meas = ComputeMeshMeasure(M);

    /** cannot redistribute if when is turned off or when we are already basically sequential **/
    if ( (!O.enable_rd) || (comm.Size() <= 2) )
      { return false; }

    bool want_redist = need_rd; // probably coarsening is slowing

    if (!want_redist) { // check if mesh is becoming too non-local
      double loc_frac = ComputeLocFrac(M);
      want_redist |= ( loc_frac < O.rd_loc_thresh );
    }

    if (!want_redist) { // check if we reach measure/proc threshhold
      want_redist |= (meas < comm.Size() * O.rd_min_nv_th);
    }
  
    if ( (!want_redist) && (O.enable_static_redist) ) { // check for static threshhold 
      double af = ( (!state.first_rd_used) && (O.first_rdaf != -1) ) ?
	O.first_rdaf : ( pow(O.rdaf_scale, state.level[0] - ( (O.first_rdaf == -1) ? 0 : 1) ) * O.rdaf );
      size_t goal_meas = max( size_t(min(af, 0.9) * state.last_redist_meas), max(O.rd_seq_nv, size_t(1)));
      want_redist |= (meas < goal_meas);
    }

    if (!want_redist)
      { return false; }

    auto rd_factor = FindRDFac (state.curr_mesh);

    auto rd_map = BuildContractMap(rd_factor, state.curr_mesh);

    if (state.free_nodes != nullptr)
      { throw Exception("free-node redist update todo"); /** do sth here..**/ }

    state.grid_step = rd_map;
    state.first_redist_used = true;
    state.last_redist_meas = meas;
    state.curr_mesh = rd_map->GetMappedMesh();
    state.dof_step = BuildDOFContractMap(rd_map, state.curr_pds);

    return true;
  } // BaseAMGFactory::TryContractStep


  BaseAMGFactory::State* BaseAMGFactory :: NewState (AMGLevel & lev)
  {
    auto s = AllocState();
    InitState(*s, lev);
    return s;
  }

  void BaseAMGFactory :: InitState (BaseAMGFactory::State& state, AMGLevel & lev)
  {
    state.level = { 0, 0, 0 };

    state.curr_mesh = lev.mesh;
    state.curr_pds = lev.pardofs;

    state.disc_map = nullptr;
    state.crs_map = nullptr;
    state.free_ndoes = lev.free_nodes;

    state.first_redist_used = false;
    state.last_redist_meas = ComputeMeshMeasure(*lev.mesh);
  }

  void BaseAMGFactory :: SetOptionsFromFlags (BaseAMGFactory::Options& opts, const Flags & flags, string prefix)
  {
    options->prefix = prefix;
    options->SetFromFlags(flags);
  } // BaseAMGFactory::SetOptionsFromFlags

  /** END BaseAMGFactory **/


  /** NodalAMGFactory **/

  template<NODE_TYPE NT, int BS, class TMESH>
  NodalAMGFactory<NT, BS, TMESH> :: NodalAMGFactory (shared_ptr<Options> _opts)
    : BaseAMGFactory(_opts)
  { ; }


  template<NODE_TYPE NT, int BS, class TMESH>
  size_t NodalAMGFactory<NT, BS, TMESH> :: ComputeMeshMeasure (const TopologicMesh & m) const
  {
    return m.template GetNNGlobal<NT>();
  } // NodalAMGFactory::ComputeMeshMeasure


  template<NODE_TYPE NT, int BS, class TMESH>
  double NodalAMGFactory<NT, BS, TMESH> :: ComputeLocFrac (const TopologicMesh & m) const
  {
    auto nng = m.template GetNNGlobal<NT_VERTEX>();
    size_t nnloc = (m.GetEQCHierarchy()->GetNEQCS() > 1) ? m.template GetENN<NT_VERTEX>(0) : 0;
    auto nnlocg = m.GetEQCHierarchy()->GetCommunicator().AllReduce(nnloc, MPI_SUM);
    return double(nnlocg) / nng;
  } // NodalAMGFactory::ComputeLocFrac


  template<NODE_TYPE NT, int BS, class TMESH>
  shared_ptr<ParallelDofs> NodalAMGFactory<NT, BS, TMESH> :: BuildParallelDofs (shared_ptr<TopologicMesh> amesh) const
  {
    const auto & mesh = *amesh;
    const auto & eqc_h = *mesh.GetEQCHierarchy();
    size_t neqcs = eqc_h.GetNEQCS();
    size_t ndof = mesh.template GetNN<NT>();
    TableCreator<int> cdps(ndof);
    for (; !cdps.Done(); cdps++) {
      for (auto eq : Range(neqcs)) {
	auto dps = eqc_h.GetDistantProcs(eq);
	auto verts = mesh.template GetENodes<NT>(eq);
	for (auto vnr : verts) {
	  for (auto p:dps) cdps.Add(vnr, p);
	}
      }
    }
    auto tab = cdps.MoveTable();
    auto pds = make_shared<ParallelDofs> (eqc_h.GetCommunicator(), move(tab) /* cdps.MoveTable() */, BS, false);
    return pds;
  } // NodalAMGFactory::BuildParallelDofs


  template<NODE_TYPE NT, int BS, class TMESH>
  size_t NodalAMGFactory<NT, BS, TMESH> :: ComputeGoal (const AMGLevel & f_lev, State & state)
  {
    auto &O(*options);

    // TODO: we have to respect enable_multistep/interleave here

    auto curr_meas = ComputeMeshMeasure(*fmesh);

    size_t goal_meas = (curr_meas == 0) ? 0 : 1;

    auto fmesh = f_lev.mesh;

    /** static coarsening ratio **/
    if (O.enable_static_crs) {
      double af = ( (f_lev.level == 0) && (O.first_aaf != -1) ) ?
	O.first_aaf : ( pow(O.aaf_scale, f_cap.level - ( (O.first_aaf == -1) ? 0 : 1) ) * O.aaf );
      goal_meas = max( size_t(min(af, 0.9) * curr_meas), max(O.max_meas, size_t(1)));
    }

    // dynamic coarsening ratio has to be handled by derived classes
    // TODO: There should also be a ComputeGoal in VertexAMGFactory which can take advantage of this!
    /** We want to find the right agglomerate size, as a heutristic take 1/(1+avg number of strong neighbours) **/
    // size_t curr_ne = fmesh->template GetNNGlobal<NT_EDGE>();
    // size_t curr_nv = fmesh->template GetNNGlobal<NT_VERTEX>();
    // double edge_per_v = 2 * double(curr_ne) / double(curr_nv);
    // if (O.enable_dyn_aaf) {
    //   const double MIN_ECW = coarsen_opts->min_ecw;
    //   const auto& ecw = coarsen_opts->ecw;
    //   size_t n_s_e = 0;
    //   cmesh->template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_s_e++; } }, true);
    //   n_s_e = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_s_e, MPI_SUM);
    //   double s_e_per_v = 2 * double(n_s_e) / double(cmesh->template GetNNGlobal<NT_VERTEX>());
    //   double dynamic_goal_fac = 1.0 / ( 1 + s_e_per_v );
    //   goal_meas = max( size_t(min2(0.5, dynamic_goal_fac) * curr_meas), max(O.max_meas, size_t(1)));
    // } // dyn_aaf

    return goal_meas;
  } // NodalAMGFactory::ComputeGoal


  template<NODE_TYPE NT, int BS, class TMESH>
  shared_ptr<BaseGridMapStep> NodalAMGFactory<NT, BS, TMESH> :: BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh) const
  {
    static Timer t("BuildContractMap"); RegionTimer rt(t);

    if (mesh == nullptr)
      { throw Exception("BuildContractMap needs a mesh!"); }
    auto m = dynamic_pointer_cast<TMESH>(mesh);
    if (m = nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*mesh).name() + string(" for BuildContractMap")); }

    // at least 2 groups - dont send everything from 1 to 0 for no reason
    int n_groups = (factor == -1) ? 2 : max2(int(2), int(1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * factor)));
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);

    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
  } // NodalAMGFactory::BuildContractMap


  template<NODE_TYPE NT, int BS, class TMESH>
  shared_ptr<BaseDOFMapStep> NodalAMGFactory<NT, BS, TMESH> :: BuildDOFContractMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    static Timer t("BuildDOFContractMap"); RegionTimer rt(t);

    if (cmap == nullptr)
      { throw Exception("BuildDOFContractMap needs a mesh!"); }
    auto cm = dynamic_pointer_cast<GridContractMap<TMESH>>(mesh);
    if (cm = nullptr)
      { throw Exception(string("Invalid map type ") + typeid(*mesh).name() + string(" for BuildContractMap")); }

    auto fg = cm->GetGroup();
    Array<int> group(fg.Size()); group = fg;
    Table<int> dof_maps;
    shared_ptr<ParallelDofs> cpd = nullptr;
    if (cm->IsMaster()) {
      // const TMESH& cmesh(*static_cast<const TMESH&>(*grid_step->GetMappedMesh()));
      shared_ptr<TMESH> cmesh = static_pointer_cast<TMESH>(cm->GetMappedMesh());
      cpd = BuildParallelDofs(cmesh);
      Array<int> perow (group.Size()); perow = 0;
      for (auto k : Range(group.Size())) perow[k] = cm->template GetNodeMap<NT>(k).Size();
      dof_maps = Table<int>(perow);
      for (auto k : Range(group.Size())) dof_maps[k] = cm->template GetNodeMap<NT>(k);
    }
    auto ctr_map = make_shared<CtrMap<typename strip_vec<Vec<BS, double>>::type>> (fpd, cpd, move(group), move(dof_maps));
    if (cm->IsMaster()) {
      ctr_map->_comm_keepalive_hack = cm->GetMappedEQCHierarchy()->GetCommunicator();
    }

    return move(ctr_map);
  } // NodalAMGFactory::BuildDOFContractMap


  /** END NodalAMGFactory **/

} // namespace amg

#endif
