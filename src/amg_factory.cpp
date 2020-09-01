#define FILE_AMG_FACTORY_CPP

#include "amg_factory.hpp"

namespace amg
{

  /** Options **/

  void BaseAMGFactory::Options :: SetFromFlags (const Flags & flags, string prefix)
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
    set_num(min_meas, "min_coarse_size");

    enable_multistep.SetFromFlags(flags, prefix + "enable_multistep");
    set_bool(enable_dyn_crs, "enable_dyn_crs");
    set_num(aaf, "aaf");
    set_num(first_aaf, "first_aaf");
    set_num(aaf_scale, "aafaaf_scale");
    d2_agg.SetFromFlags(flags, prefix+"d2_agg");

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

    enable_disc.SetFromFlags(flags, prefix + "enable_disc");

    enable_sp.SetFromFlags(flags, prefix +  "enable_sp");
    set_bool(sp_needs_cmap, "sp_needs_cmap");
    sp_min_frac.SetFromFlags(flags, prefix +  "sp_min_frac");
    sp_max_per_row.SetFromFlags(flags, prefix +  "sp_max_per_row");
    sp_omega.SetFromFlags(flags, prefix +  "sp_omega");

    set_bool(keep_grid_maps, "keep_grid_maps");

    set_bool(check_kvecs, "check_kvecs");

    set_enum_opt(log_level, "log_level", {"none", "basic", "normal", "extra"});
    set_bool(print_log, "print_log");
    log_file = flags.GetStringFlag(prefix + string("log_file"), "");
  } // BaseAMGFactory::Options :: SetFromFlags

  /** END Options **/


  /** Logger **/

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
      { comm = cap.cap->mesh->GetEQCHierarchy()->GetCommunicator(); }
    if (lev == LOG_LEVEL::NONE)
      { return; }
    auto lev_comm = cap.cap->mesh->GetEQCHierarchy()->GetCommunicator();
    vcc.Append(cap.cap->mesh->template GetNNGlobal<NT_VERTEX>());
    // cout << " occ-comp would be " << cap.mat->NZE() << " * " << GetEntrySize(cap.mat.get())
    // 	 << " = " << cap.mat->NZE() * GetEntrySize(cap.mat.get()) << endl;
    // cout << " mat " << cap.mat << " " << typeid(*cap.mat).name() << endl;
    if (lev_comm.Rank() == 0)
      { occ.Append(lev_comm.Reduce(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()), MPI_SUM, 0)); }
    else
      { lev_comm.Reduce(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()), MPI_SUM, 0); }
    if (lev == LOG_LEVEL::BASIC)
      { return; }
    NVs.Append(cap.cap->mesh->template GetNNGlobal<NT_VERTEX>());
    NEs.Append(cap.cap->mesh->template GetNNGlobal<NT_EDGE>());
    NPs.Append(cap.cap->mesh->GetEQCHierarchy()->GetCommunicator().Size());
    NZEs.Append(lev_comm.Reduce(cap.cap->mat->NZE(), MPI_SUM, 0));
    if (lev == LOG_LEVEL::NORMAL)
      { return; }
    vccl.Append(cap.cap->mesh->template GetNN<NT_VERTEX>());
    occl.Append(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()));
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
	// auto maxval = comm.AllReduce(val, MPI_MAX);
	auto maxval = val;
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


  shared_ptr<BaseAMGFactory::LevelCapsule> BaseAMGFactory :: AllocCap () const
  {
    return make_shared<BaseAMGFactory::LevelCapsule>();
  } // BaseAMGFactory :: AllocCap

  void BaseAMGFactory :: SetUpLevels (Array<shared_ptr<BaseAMGFactory::AMGLevel>> & amg_levels, shared_ptr<DOFMap> & dof_map)
  {
    static Timer t("SetupLevels"); RegionTimer rt(t);

    const auto & O(*options);

    logger = make_shared<Logger>(options->log_level);

    amg_levels.SetAllocSize(O.max_n_levels);
    auto & finest_level = amg_levels[0];

    // Array<AMGLevel> amg_levels(O.max_n_levels); amg_levels.SetSize(1);
    // amg_levels[0] = finest_level;

    auto state = NewState(amg_levels[0]);

    RSU(amg_levels, dof_map, *state);

    if (options->print_log)
      { logger->PrintLog(cout); }
    if (options->log_file.size() > 0)
      { logger->PrintToFile(options->log_file); }

    if (dof_map->GetNSteps() == 0)
      { throw Exception("NgsAMG failed to construct any coarse levels!"); }

    if (options->check_kvecs)
      { CheckKVecs(amg_levels, dof_map); }

 
    logger = nullptr;
    delete state;
  } // BaseAMGFactory::SetUpLevels


  void BaseAMGFactory :: RSU (Array<shared_ptr<AMGLevel>> & amg_levels, shared_ptr<DOFMap> & dof_map, State & state)
  {
    const auto & O(*options);

    auto & f_lev = amg_levels.Last();

    shared_ptr<AMGLevel> c_lev = make_shared<AMGLevel>();

    logger->LogLevel (*f_lev);

    shared_ptr<BaseDOFMapStep> step = DoStep(f_lev, c_lev, state);

    int step_bad = 0;

    if ( (step == nullptr) || (c_lev->cap->mesh == f_lev->cap->mesh) || (c_lev->cap->mat == f_lev->cap->mat) )
      { return; } // step not performed correctly - coarsening is probably stuck
    if (c_lev->cap->mesh != nullptr) { // not dropped out
      if (ComputeMeshMeasure(*c_lev->cap->mesh) == 0)
	{ step_bad = 1; } // e.g stokes: when coarsening down to 1 vertex, no more edges left!
      if (ComputeMeshMeasure(*c_lev->cap->mesh) < O.min_meas)
	{ step_bad = 1; } // coarse grid is too small
    }

    step_bad = f_lev->cap->eqc_h->GetCommunicator().AllReduce(step_bad, MPI_SUM);

    if (step_bad != 0)
      { return; }

    dof_map->AddStep(step);

    if (!O.keep_grid_maps){
      f_lev->disc_map = nullptr;
      f_lev->crs_map = nullptr;
    }

    // cout << " CLEV ECON " << endl;
    // cout << *c_lev->cap->mesh->GetEdgeCM() << endl;
    // cout << endl;

    /** Recursive call (or return) **/
    if ( (c_lev->cap->mesh == nullptr) || (c_lev->cap->mat == nullptr) ) // dropped out (redundand "||" ?)
      { amg_levels.Append(move(c_lev)); return; }
    else if ( (f_lev->level + 2 == O.max_n_levels) ||                // max n levels reached
    	      (O.max_meas >= ComputeMeshMeasure (*c_lev->cap->mesh) ) ) { // max coarse size reached
      logger->LogLevel (*c_lev);
      amg_levels.Append(move(c_lev));
      return;
    }
    else { // more coarse levels
      amg_levels.Append(move(c_lev));
      RSU (amg_levels, dof_map, state);
      return;
    }
  } // BaseAMGFactory::RSU


  shared_ptr<BaseDOFMapStep> BaseAMGFactory :: DoStep (shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev, State & state)
  {
    const auto & O(*options);

    const auto fcomm = f_lev->cap->eqc_h->GetCommunicator();
    const bool doco = (fcomm.Rank() == 0);

    size_t curr_meas = ComputeMeshMeasure(*state.curr_cap->mesh), goal_meas = ComputeGoal(f_lev, state);

    if (doco)
      { cout << " step from level " << f_lev->level << " goal is: " << curr_meas << " -> " << goal_meas << endl; }

    if (f_lev->level == 0)
      { state.last_redist_meas = curr_meas; }

    shared_ptr<BaseDOFMapStep> embed_map = f_lev->embed_map, disc_map;

    // CalcCoarsenOpts(state); // TODO: proper update of coarse cols for ecol?

    if ( O.enable_redist && (f_lev->level != 0) ) {
      if ( TryDiscardStep(state) ) {
	disc_map = move(state.dof_map);
	curr_meas = ComputeMeshMeasure(*state.curr_cap->mesh);
      }
    }

    if (O.keep_grid_maps)
      { f_lev->disc_map = state.disc_map; }

    shared_ptr<BaseDOFMapStep> prol_map, rd_map;
    { /** Coarse/Redist maps - constructed interleaved, but come out untangled. **/
      bool have_options = O.enable_redist;
      bool goal_reached = (curr_meas < goal_meas);
      Array<shared_ptr<BaseCoarseMap>> cm_chunks;

      INT<4> mesh_meas = {0,0,0,0};
	/** mesh0 -(ctr)-> mesh1 -(first_crs)-> mesh2 -(crs/ctr)-> mesh3
	    mesh0-mesh1, and mesh2-mesh3 can be the same **/
      shared_ptr<TopologicMesh> mesh0 = nullptr, mesh1 = nullptr, mesh2 = nullptr, mesh3 = nullptr;
      mesh0 = state.curr_cap->mesh; mesh_meas[0] = ComputeMeshMeasure(*mesh0);
      shared_ptr<LevelCapsule> m0cap = state.curr_cap;
      
      Array<shared_ptr<BaseDOFMapStep>> dof_maps;
      Array<int> prol_inds;
      shared_ptr<BaseCoarseMap> first_cmap;
      shared_ptr<LevelCapsule> fcm_cap = nullptr;
      Array<shared_ptr<BaseDOFMapStep>> rd_chunks;

      bool could_recover = O.enable_redist;
	
      do { /** coarsen until goal reached or stuck **/
	cm_chunks.SetSize0();
	auto comm = static_cast<BlockTM&>(*state.curr_cap->mesh).GetEQCHierarchy()->GetCommunicator();
	could_recover = ( O.enable_redist && (comm.Size() > 2) && (state.level[2] == 0) );
	bool crs_stuck = false, rded_out = false;
	auto cm_cap = state.curr_cap;
	do { /** coarsen until stuck or goal reached **/
	  if (doco)
	    cout << " inner C loop, curr " << curr_meas << " -> " << goal_meas << endl;
	  auto f_cap = state.curr_cap;
	  if ( TryCoarseStep(state) ) { // coarse map constructed successfully
	    state.level[1]++; state.level[2] = 0; // count up sub-coarse, reset redist
	    size_t f_meas = ComputeMeshMeasure(*f_cap->mesh), c_meas = ComputeMeshMeasure(*state.curr_cap->mesh);
	    double meas_fac = (f_meas == 0) ? 0 : c_meas / double(f_meas);
	    if (doco)
	      cout << " c step " << f_meas << " -> " << c_meas << ", frac = " << meas_fac << endl;
	    goal_reached = (c_meas < goal_meas);
	    if ( (goal_reached) && (cm_chunks.Size() > 0) && (f_meas * c_meas < sqr(goal_meas)) ) {
	      if (doco)
		cout << " roll back c!" << endl;
	      // last step was closer than current step - reset to last step
	      state.curr_cap = f_cap;
	      state.crs_map = nullptr;
	      state.disc_map = nullptr;
	      state.dof_map = nullptr;
	      goal_reached = true;
	    }
	    else { // okay, use this step
	      curr_meas = c_meas;
	      cm_chunks.Append(move(state.crs_map));
	      if (!O.enable_multistep.GetOpt(f_lev->level)) // not allowed to chain coarse steps
		{ goal_reached = true; }
	    }
	    if ( (!goal_reached) && could_recover)
	      { crs_stuck |= meas_fac > O.rd_crs_thresh; }  // bad coarsening - try to recover via redist
	  }
	  else { // coarsening failed completely
	    if (doco)
	      cout << "no cmap, stuck!" << endl;
	    crs_stuck = true;
	  }
	} while ( (!crs_stuck) && (!goal_reached) );

	if (doco)
	  cout << " -> back to outer loop, stuck " << crs_stuck << ", reached " << goal_reached << endl;
	
	state.need_rd = crs_stuck;

	/** concatenate coarse map chunks **/
	shared_ptr<BaseCoarseMap> c_step = nullptr;
	if ( cm_chunks.Size() )
	  { c_step = cm_chunks.Last(); }
	for (int l = cm_chunks.Size() - 2; l >= 0; l--)
	  { c_step = cm_chunks[l]->Concatenate(c_step); }

	if (doco)
	  cout << " have conc c-step ? " << c_step << endl;;

	if (f_lev->crs_map == nullptr)
	  { f_lev->crs_map = c_step; }

	if (c_step != nullptr) {
	  /** build pw-prolongation **/
	  shared_ptr<BaseDOFMapStep> pmap = PWProlMap(c_step, cm_cap, state.curr_cap);

	  /** Save the first coarse-map - we might be able to use it later to get a slightly better smoothed prol! **/
	  if (first_cmap == nullptr) {
	    fcm_cap = cm_cap;
	    first_cmap = c_step;
	    mesh1 = first_cmap->GetMesh(); mesh_meas[1] = ComputeMeshMeasure(*mesh1);
	    mesh2 = first_cmap->GetMappedMesh(); mesh_meas[2] = ComputeMeshMeasure(*mesh2);
	  }
	  mesh3 = c_step->GetMappedMesh(); mesh_meas[3] = ComputeMeshMeasure(*mesh3);

	  prol_inds.Append(dof_maps.Size());
	  dof_maps.Append(pmap);
	}

	/** try to recover via redistribution **/
	if (O.enable_redist && (state.level[2] == 0) ) {
	  if (doco)
	    cout << " try contract " << endl;
	  if ( TryContractStep(state) ) { // redist successfull
	    if (doco)
	      cout << " contract ok " << endl;
	    state.level[2]++; // count up redist
	    auto rdm = move(state.dof_map);
	    dof_maps.Append(rdm);
	    rd_chunks.Append(rdm);
	    if (state.curr_cap->mesh == nullptr)
	      { rded_out = true; break; }
	    mesh3 = state.curr_cap->mesh; mesh_meas[3] = ComputeMeshMeasure(*mesh3);
	  }
	  else { // redist rejected
	    if (doco)
	      cout << " no contract " << endl;
	    could_recover = false;
	  }
	}
	else // already distributed once
	  { could_recover = false; }
	if (doco)
	  cout << " continue ? " << bool((could_recover) && (!goal_reached)) << endl;
      } while ( (could_recover) && (!goal_reached) );

      if (doco)
	cout << " broke out of crs/ctr loops " << endl;

      // cout << " enable_sp: " << O.enable_sp << endl;

      /** Smooth the first prol-step, using the first coarse-map if we need coarse-map for smoothing,
	  or if it is preferrable to smoothing the concatenated prol with only fine mesh. **/
      bool sp_done = false, have_pwp = true;
      if ( O.enable_sp.GetOpt(f_lev->level) ) {
	// cout << " deal with sp!" << endl;
	/** need cmap || only one coarse map **/
	auto comm = mesh0->GetEQCHierarchy()->GetCommunicator();
	int do_sp_now = 0;
	if (comm.Rank() == 0) {
	  if (mesh2 != nullptr) { // mesh2: first coarse mesh - if this is nullptr, coarsening is stuck!
	    do_sp_now = O.sp_needs_cmap || (mesh3 == mesh2);
	    if (do_sp_now == 0) {
	      double frac21 = (mesh_meas[1] == 0) ? 0.0 : double(mesh_meas[2]) / mesh_meas[1];
	      double frac32 = (mesh_meas[2] == 0) ? 0.0 : double(mesh_meas[3]) / mesh_meas[2];
	      /** first cmap is a significant part of coarsening **/
	      do_sp_now = (frac21 < 1.333 * frac32) ? 1 : 0;
	    }
	  }
	  else
	    { do_sp_now = -1; have_pwp = false; }
	}
	comm.NgMPI_Comm::Bcast(do_sp_now);
	// cout << "do now?" << do_sp_now << endl;
	if (do_sp_now < 0) // have no coarse map at all!
	  { have_pwp = false; }
	else if (do_sp_now > 0) { // have coarse map and do now!
	  sp_done = true;
	  if (prol_inds.Size() > 0) { // might be rded out!
	    auto pmap = dof_maps[prol_inds[0]];
	    dof_maps[prol_inds[0]] = SmoothedProlMap(pmap, first_cmap, fcm_cap);
	    first_cmap = nullptr; // no need for this anymore
	  }
	}
	else // have some coarse map, but do later
	  { ; }
      }

      // cout << " sp done " << endl;

      // cout << "dof_maps: " << endl;
      // for (auto k :Range(dof_maps))
	// { cout << k << ": " << typeid(*dof_maps[k]).name() << endl; }
      // cout << "prol_inds: " << endl; prow(prol_inds); cout << endl;
      
      /** Untangle prol/ctr-maps to one prol followed by one ctr map. **/
      int do_last_pb = 0; bool last_map_rd = false;
      if ( ( (prol_inds.Size()>0) && (prol_inds.Last() != dof_maps.Size()-1) ) ||
	   ( (prol_inds.Size() == 0) && (dof_maps.Size() > 0) ) ) { // last map is redist
	last_map_rd = true;
	// cout << "lmrd " << endl; auto c = dof_maps.Last()->GetParDofs()->GetCommunicator();
	// cout << "AR on " << c.Rank() << " " << c.Size() << endl;
	if (dof_maps.Last()->GetParDofs()->GetCommunicator().AllReduce(do_last_pb, MPI_MAX)) { // cannot use mapped pardofs!
	    // cout << "pull back " << endl;
	    auto ctr_map = dof_maps.Last();
	    auto pind = dof_maps.Size() - 1;
	    auto pb_prol = ctr_map->PullBack(nullptr);
	    dof_maps.Append(ctr_map);
	    prol_inds.Append(pind);
	    dof_maps[pind] = pb_prol;
	  }
      }

      // cout << " last, non-master PB done " << endl;

      bool first_rd = true; /** Actually, I think this is garbage ... I THINK, I should just delete first_rd.**/
      int maxmi = dof_maps.Size() - 1 + ( last_map_rd ? -1 : 0 );
      for (int map_ind = maxmi, pii = prol_inds.Size() - 1; map_ind >= 0; map_ind--) {
	if ( (pii >= 0) && (map_ind == prol_inds[pii]) ) { // prol map
	  // cout << "(prol), mi " << map_ind << " pii " << pii << endl;
	  if (prol_map == nullptr)
	    { prol_map = dof_maps[map_ind]; }
	  else
	    { prol_map = dof_maps[map_ind]->Concatenate(prol_map); }
	  pii--;
	}
	else { // rd map
	  // cout << "(rd), mi " << map_ind;
	  if (first_rd) {
	    do_last_pb = prol_map != nullptr;
	    // cout << "pb " << do_last_pb << endl;
	    auto comm = dof_maps[map_ind]->GetParDofs()->GetCommunicator(); // cannot use mapped pardofs!
	    // cout << "AR on " << comm.Rank() << " " << comm.Size() << endl;
	    comm.AllReduce(do_last_pb, MPI_MAX);
	    // cout << "pb2 " << do_last_pb << endl;
	    first_rd = false;
	  }
	  if (prol_map != nullptr)
	    { prol_map = dof_maps[map_ind]->PullBack(prol_map); }
	}
      }

      // cout << " concs/pullbacks done " << endl;

      /** If not forced to before, smooth prol here.
	  TODO: check if other version would be possible here (that one should be better anyways!)  **/
      if ( O.enable_sp.GetOpt(f_lev->level) && (!sp_done) && (have_pwp) )
	{ prol_map = SmoothedProlMap(prol_map, m0cap); }

      /** pack rd-maps into one step**/
      if (rd_chunks.Size() == 1)
	{ rd_map = rd_chunks[0]; }
      else if (rd_chunks.Size())
	{ rd_map = make_shared<ConcDMS>(rd_chunks); }
    }

    /** We now have emb - disc - prol - rd. Concatenate and pack into final map. **/

    Array<shared_ptr<BaseDOFMapStep>> init_steps;
    if (embed_map != nullptr)
      { init_steps.Append(embed_map); }
    if (disc_map != nullptr)
      { init_steps.Append(disc_map); }
    if (prol_map != nullptr)
      { init_steps.Append(prol_map); }
    if (rd_map != nullptr)
      { init_steps.Append(rd_map); }

    // cout << " steps: " << endl;
    // cout << "embed_map " << embed_map << endl;
    // cout << "disc_map " << disc_map << endl;
    // cout << "prol_map " << prol_map << endl;
    // cout << "rd_map " << rd_map << endl;

    // const int iss = init_steps.Size();
    // for (int k = 0; k < iss; k++) {
    //   shared_ptr<BaseDOFMapStep> conc_step = init_steps[k];
    //   int j = k+1;
    //   for ( ; j < iss; j++)
    // 	if (auto x = conc_step->Concatenate(init_steps[j]))
    // 	  { cout << " conc " << k << " - " << j << endl; conc_step = x; k++; }
    // 	else
    // 	  { break; }
    //   // if (auto pm = dynamic_pointer_cast<ProlMap<SparseMatrixTM<double>>>(conc_step)) {
    // 	// cout << " CONC STEP " << k << " - " << j << ": " << endl;
    // 	// print_tm_spmat(cout, *pm->GetProl()); cout << endl;
    //   // }
    //   sub_steps.Append(conc_step);
    // }
    // init_steps = nullptr;
    // cout << "sub_steps: " << endl;
    // for (auto k :Range(sub_steps.Size()))
    //   { cout << k << ": " << typeid(*sub_steps[k]).name() << endl; }
    // if (sub_steps.Size() > 0) {
    //   shared_ptr<BaseDOFMapStep> final_step = nullptr;
    //   if (sub_steps.Size() == 1)
    // 	{ final_step = sub_steps[0]; }
    //   else
    // 	{ final_step = make_shared<ConcDMS>(sub_steps); }

    /** assemble coarse level matrix, set return vals, etc. **/
    c_lev->level = f_lev->level + 1;
    c_lev->cap = state.curr_cap; c_lev->cap->baselevel = c_lev->level;
    auto final_step = MapLevel(init_steps, f_lev, c_lev);

    // auto final_step = MakeSingleStep(init_steps);
    // c_lev->level = f_lev->level + 1;
    // c_lev->cap = state.curr_cap;
    // MapLevel(final_step, f_lev, c_lev);

    if (final_step == nullptr)
      { c_lev->cap = f_lev->cap; }

    return final_step;
  } // BaseAMGFactory::DoStep


  shared_ptr<BaseDOFMapStep> BaseAMGFactory :: MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps, shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev)
  {
    if (c_lev->cap->mesh == f_lev->cap->mesh)
      { return nullptr; }
    if ( (dof_steps.Size() > 1) && (f_lev->embed_map != nullptr) && (f_lev->embed_done) ) {
      /** The fine level matrix is already embedded! **/
      auto afs = MakeSingleStep(dof_steps.Part(1));
      if (afs == nullptr) { /** No idea how this would happen. **/
	c_lev->cap->mat = dof_steps[0]->AssembleMatrix(f_lev->cap->mat);
	dof_steps[0]->Finalize();
	return dof_steps[0];
      }
      else { /** Use all steps except embedding for coarse level matrix. **/
	c_lev->cap->mat = afs->AssembleMatrix(f_lev->cap->mat);
	Array<shared_ptr<BaseDOFMapStep>> ds2( { dof_steps[0], afs } );
	auto final_step = MakeSingleStep(ds2);
	final_step->Finalize();
	return final_step;
      }
    }
    else { /** Fine level matrix is not embedded, or there is no embedding. **/
      auto final_step = MakeSingleStep(dof_steps);
      if (final_step != nullptr) {
	c_lev->cap->mat = final_step->AssembleMatrix(f_lev->cap->mat);
	final_step->Finalize();
      }
      return final_step;
    }
  } // BaseAMGFactory::MapLevel


  void BaseAMGFactory :: MapLevel2 (shared_ptr<BaseDOFMapStep> & dof_step, shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev)
  {
    c_lev->cap->mat = dof_step->AssembleMatrix(f_lev->cap->mat);
  } // BaseAMGFactory::MapLevel


  bool BaseAMGFactory :: TryCoarseStep (State & state)
  {
    auto & O(*options);

    shared_ptr<LevelCapsule> c_cap = AllocCap();
    c_cap->baselevel = state.level[0];

    /** build coarse map **/
    shared_ptr<BaseCoarseMap> cmap = BuildCoarseMap(state, c_cap);

    if (cmap == nullptr) // could not build map
      { return false; }

    /** check if the step was ok **/
    bool accept_crs = true;

    auto cmesh = cmap->GetMappedMesh();

    size_t f_meas = ComputeMeshMeasure(*state.curr_cap->mesh), c_meas = ComputeMeshMeasure(*cmesh);

    double cfac = (f_meas == 0) ? 0 : double(c_meas) / f_meas;

    if (cmap->GetMesh()->GetEQCHierarchy()->GetCommunicator().Rank() == 0)
      { cout << " map maps " << f_meas << " -> " << c_meas <<  ", fac " << cfac << endl; }

    bool map_ok = true;

    map_ok &= ( c_meas < f_meas ); // coarsening is stuck

    if (O.enable_redist)
      { map_ok &= (cfac < O.rd_crs_thresh); }

    if (!map_ok)
      { return false; }

    /** Okay, we have a new map! **/
    state.curr_cap = c_cap;
    state.crs_map = cmap;

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


  bool BaseAMGFactory :: TryContractStep (State & state)
  {
    static Timer t("TryContractStep");
    RegionTimer rt(t);

    const auto & O(*options);
    const auto & M = *state.curr_cap->mesh;
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    double meas = ComputeMeshMeasure(M);

    /** cannot redistribute if when is turned off or when we are already basically sequential **/
    if ( (!O.enable_redist) || (comm.Size() <= 2) )
      { return false; }

    bool want_redist = state.need_rd; // probably coarsening is slowing

    if (!want_redist) { // check if mesh is becoming too non-local
      double loc_frac = ComputeLocFrac(M);
      want_redist |= ( loc_frac < O.rd_loc_thresh );
    }

    if (!want_redist) { // check if we reach measure/proc threshhold
      want_redist |= (meas < comm.Size() * O.rd_min_nv_th);
    }
  
    if ( (!want_redist) && (O.enable_static_redist) ) { // check for static threshhold 
      double af = ( (!state.first_redist_used) && (O.first_rdaf != -1) ) ?
	O.first_rdaf : ( pow(O.rdaf_scale, state.level[0] - ( (O.first_rdaf == -1) ? 0 : 1) ) * O.rdaf );
      size_t goal_meas = max( size_t(min(af, 0.9) * state.last_redist_meas), max(O.rd_seq_nv, size_t(1)));
      want_redist |= (meas < goal_meas);
    }

    if (!want_redist)
      { return false; }

    auto rd_factor = FindRDFac (state.curr_cap->mesh);

    shared_ptr<LevelCapsule> c_cap = AllocCap();
    c_cap->baselevel = state.level[0];

    auto rd_map = BuildContractMap(rd_factor, state.curr_cap->mesh, c_cap);

    if (state.curr_cap->free_nodes != nullptr)
      { throw Exception("free-node redist update todo"); /** do sth here..**/ }

    state.dof_map = BuildDOFContractMap(rd_map, state.curr_cap->pardofs, c_cap);
    state.first_redist_used = true;
    state.need_rd = false;
    state.last_redist_meas = meas;
    state.curr_cap = c_cap;

    return true;
  } // BaseAMGFactory::TryContractStep


  BaseAMGFactory::State* BaseAMGFactory :: NewState (shared_ptr<AMGLevel> & lev)
  {
    auto s = AllocState();
    InitState(*s, lev);
    return s;
  } // BaseAMGFactory::NewState


  void BaseAMGFactory :: InitState (BaseAMGFactory::State& state, shared_ptr<AMGLevel> & lev) const
  {
    state.level = { 0, 0, 0 };

    state.curr_cap = lev->cap;

    /** TODO: this is kind of unclean ... **/
    if (lev->embed_map != nullptr)
      { state.curr_cap->pardofs = lev->embed_map->GetMappedParDofs(); }

    state.disc_map = nullptr;
    state.crs_map = nullptr;
    state.curr_cap->free_nodes = lev->cap->free_nodes;

    state.first_redist_used = false;
    state.last_redist_meas = ComputeMeshMeasure(*lev->cap->mesh);
  } // BaseAMGFactory::InitState


  void BaseAMGFactory :: SetOptionsFromFlags (BaseAMGFactory::Options& opts, const Flags & flags, string prefix)
  {
    opts.SetFromFlags(flags, prefix);
  } // BaseAMGFactory::SetOptionsFromFlags

  /** END BaseAMGFactory **/


} // namespace amg
