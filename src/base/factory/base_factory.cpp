#define FILE_AMG_FACTORY_CPP

#include "base_factory.hpp"

#include <utils_io.hpp>

namespace amg
{

/** Options **/

void BaseAMGFactory::Options::SetFromFlags (const Flags & flags, string prefix)
{
  auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

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

  check_aux_mats.SetFromFlags(flags, prefix + "check_aux_mats");

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

  set_bool(sp_needs_cmap, "sp_needs_cmap");
  sp_min_frac.SetFromFlags(flags, prefix +  "sp_min_frac");
  sp_max_per_row.SetFromFlags(flags, prefix +  "sp_max_per_row");
  sp_omega.SetFromFlags(flags, prefix +  "sp_omega");
  use_emb_sp = flags.GetDefineFlagX(prefix + "use_emb_sp").IsTrue();

  set_bool(keep_grid_maps, "keep_grid_maps");

  set_bool(check_kvecs, "check_kvecs");

  SetEnumOpt(flags, log_level, pfit("log_level"), {"none", "basic", "normal", "extra", "debug"}, { NONE, BASIC, NORMAL, EXTRA, DBG });

  set_bool(print_log, "print_log");
  log_file = flags.GetStringFlag(prefix + string("log_file"), "");
} // BaseAMGFactory::Options::SetFromFlags

/** END Options **/


/** Logger **/

void BaseAMGFactory::Logger::Alloc (int N)
{
  auto lam = [N](auto & a) { a.SetSize(N); a.SetSize0(); };
  lam(vcc); lam(occ); // need it as work array even for basic
  if ( lev == LOG_LEVEL::NONE )
    { return; }
  lam(NVs);
  lam(NEs);
  lam(NPs);
  if (lev < LOG_LEVEL::EXTRA)
    { return; }
  lam(vccl);
  lam(occl);
} // Logger::Alloc


void BaseAMGFactory::Logger::LogLevel (BaseAMGFactory::AMGLevel & cap)
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
    { occ.Append(lev_comm.Reduce(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()), NG_MPI_SUM, 0)); }
  else
    { lev_comm.Reduce(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()), NG_MPI_SUM, 0); }
  NVs.Append(cap.cap->mesh->template GetNNGlobal<NT_VERTEX>());
  NEs.Append(cap.cap->mesh->template GetNNGlobal<NT_EDGE>());
  NPs.Append(cap.cap->mesh->GetEQCHierarchy()->GetCommunicator().Size());
  NZEs.Append(lev_comm.Reduce(cap.cap->mat->NZE(), NG_MPI_SUM, 0));
  if (lev < LOG_LEVEL::EXTRA)
    { return; }
  vccl.Append(cap.cap->mesh->template GetNN<NT_VERTEX>());
  occl.Append(cap.cap->mat->NZE() * GetEntrySize(cap.cap->mat.get()));
} // Logger::LogLevel


void BaseAMGFactory::Logger::Finalize ()
{
  ready = true;

  if (lev == LOG_LEVEL::NONE)
    { return; }

  op_comp = v_comp = 0;

  if (comm.Rank() == 0)
  { // only 0 has op-comp
    double vcc0 = vcc[0];
    for (auto& v : vcc)
      { v /= vcc0; v_comp += v; }
    double occ0 = occ[0];
    for (auto& v : occ)
      { v /= occ0; op_comp += v;}
  }

  if (lev < LOG_LEVEL::EXTRA)
    { return; }

  auto alam = [&](auto & rk, auto & val, auto & array) {
    if (comm.Size() > 1) {
      // auto maxval = comm.AllReduce(val, NG_MPI_MAX);
      auto maxval = val;
      rk = comm.AllReduce( (val == maxval) ? (int)comm.Rank() : (int)comm.Size(), NG_MPI_MIN);
      if ( (comm.Rank() == 0) && (rk != 0) )
        { comm.Recv(array, rk, NG_MPI_TAG_AMG); }
      else if ( (comm.Rank() == rk) && (rk != 0) )
        { comm.Send(array, 0, NG_MPI_TAG_AMG); }
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


void BaseAMGFactory::Logger::PrintLog (ostream & out)
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


void BaseAMGFactory::Logger::PrintToFile (string file_name)
{
  if (comm.Rank() == 0) {
    ofstream out(file_name, ios::out);
    PrintLog(out);
  }
} // Logger::PrintToFile

/** END Logger **/


/** BaseAMGFactory **/

BaseAMGFactory::BaseAMGFactory (shared_ptr<Options> _opts)
  : options(_opts)
{
  logger = make_shared<Logger>(options->log_level);
} // BaseAMGFactory(..)


shared_ptr<BaseAMGFactory::LevelCapsule> BaseAMGFactory::AllocCap () const
{
  return make_shared<BaseAMGFactory::LevelCapsule>();
} // BaseAMGFactory::AllocCap

void BaseAMGFactory::SetUpLevels (Array<shared_ptr<BaseAMGFactory::AMGLevel>> & amg_levels, shared_ptr<DOFMap> & dof_map)
{
  static Timer t("SetUpLevels"); RegionTimer rt(t);

  const auto & O(*options);

  amg_levels.SetAllocSize(O.max_n_levels);
  auto & finest_level = amg_levels[0];

  auto state = NewState(amg_levels[0]);

  RSU(amg_levels, dof_map, *state);

  if (options->print_log) {
    if (options->log_file.size() > 0)
      { logger->PrintToFile(options->log_file); }
    else
      { logger->PrintLog(cout); }
  }

  if (dof_map->GetNSteps() == 0)
    { throw Exception("Failed to construct any coarse levels!"); }

  if (options->check_kvecs)
    { CheckKVecs(amg_levels, dof_map); }

  Logger lg2(BaseAMGFactory::Options::LOG_LEVEL::DBG, 20);

  logger = nullptr;
} // BaseAMGFactory::SetUpLevels


void BaseAMGFactory::RSU (Array<shared_ptr<AMGLevel>> & amg_levels, shared_ptr<DOFMap> & dof_map, State & state)
{
  const auto & O(*options);

  auto & f_lev = amg_levels.Last();

  shared_ptr<AMGLevel> c_lev = make_shared<AMGLevel>();

  logger->LogLevel (*f_lev);

  cout << " DoStep " << amg_levels.Last()->level << "!" << endl;
  shared_ptr<BaseDOFMapStep> step = DoStep(f_lev, c_lev, state);
  cout << " DoStep " << amg_levels.Last()->level << " DONE!" << endl;

  int step_bad = 0;

  if ( (step == nullptr) || (c_lev->cap->mesh == f_lev->cap->mesh) || (c_lev->cap->mat == f_lev->cap->mat) )
  {
    // step not performed correctly - coarsening is probably stuck
    DoDebuggingTests(amg_levels, dof_map);
    return;
  }
  if (c_lev->cap->mesh != nullptr)
  { // not dropped out
    cout << " ComputeMeshMeasure!" << endl;
    auto const cMeas = ComputeMeshMeasure(*c_lev->cap->mesh);
    cout << " ComputeMeshMeasure BACK, cMeas = " << cMeas << "!" << endl;
    if (cMeas == 0)
      { step_bad = 1; } // e.g stokes: when coarsening down to 1 vertex, no more edges left!
    else if ( cMeas < O.min_meas)
      { step_bad = 1; } // coarse grid is too small
    else
    {
      cout << " ComputeMeshMeasure! fMeas" << endl;
      auto const fMeas = ComputeMeshMeasure(*f_lev->cap->mesh);
      cout << " ComputeMeshMeasure BACK, fMeas = " << fMeas << "!" << endl;

      if (cMeas == fMeas)
        { step_bad = 1; } // probably just a contract without coarse map
    }
  }

  cout << "step_bad F =" << step_bad << endl;

  step_bad = f_lev->cap->eqc_h->GetCommunicator().AllReduce(step_bad, NG_MPI_SUM);

  cout << "step_bad F all =" << step_bad << endl;

  if (step_bad != 0)
  {
    DoDebuggingTests(amg_levels, dof_map);
    return;
  }

  dof_map->AddStep(step);

  if (O.log_level == Options::LOG_LEVEL::DBG)
  {
    ofstream out("dof_step_rk_" + std::to_string(f_lev->cap->uDofs.GetCommunicator().Rank()) +
                          "_l_" + std::to_string(f_lev->cap->baselevel) + ".out");
    out << *step << endl;
  }


  f_lev->crs_map = O.keep_grid_maps ? state.crs_map : nullptr;

  // cout << " CLEV ECON " << endl;
  // cout << *c_lev->cap->mesh->GetEdgeCM() << endl;
  // cout << endl;

  /** Recursive call (or return) **/
  if ( (c_lev->cap->mesh == nullptr) || (c_lev->cap->mat == nullptr) ) { // dropped out (redundand "||" ?)
    amg_levels.Append(std::move(c_lev));
    DoDebuggingTests(amg_levels, dof_map);
    return;
    }
  else if ( (f_lev->level + 2 == O.max_n_levels) ||                // max n levels reached
          (O.max_meas >= ComputeMeshMeasure (*c_lev->cap->mesh) ) ) { // max coarse size reached
    logger->LogLevel (*c_lev);
    amg_levels.Append(std::move(c_lev));
    DoDebuggingTests(amg_levels, dof_map);
    return;
  }
  else { // more coarse levels
    // at this point we go to the next level and we reset the sub-levels
    state.level =  { c_lev->level, 0, 0 };
    amg_levels.Append(std::move(c_lev));
    RSU (amg_levels, dof_map, state);
    return;
  }
} // BaseAMGFactory::RSU


shared_ptr<BaseDOFMapStep> BaseAMGFactory::DoStep (shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev, State & state)
{
  const auto & O(*options);

  const auto fcomm = f_lev->cap->eqc_h->GetCommunicator();
  const bool doco = (fcomm.Rank() == 0) && (options->log_level > Options::LOG_LEVEL::NORMAL);

  size_t curr_meas = ComputeMeshMeasure(*state.curr_cap->mesh), goal_meas = ComputeGoal(f_lev, state);

  if (doco)
    { cout << " step from level " << f_lev->level << " goal is: " << curr_meas << " -> " << goal_meas << endl; }

  if (f_lev->level == 0)
    { state.last_redist_meas = curr_meas; }

  shared_ptr<BaseDOFMapStep> embed_map = f_lev->embed_map;

  shared_ptr<BaseDOFMapStep> prol_map, rd_map;

  state.level = { f_lev->level, 0, 0 };

  auto f_cap = f_lev->cap;

  /** coarse map **/
  // --emb--> fine_level --crs--> mid_level --ctr--> crs_level
  if ( TryCoarseStep(state) ) {
    // coarse map constructed successfully - count up level, etc.
    state.level[1]++;
    size_t f_meas = ComputeMeshMeasure(*f_cap->mesh), c_meas = ComputeMeshMeasure(*state.curr_cap->mesh);
    curr_meas = c_meas;
    double meas_fac = (f_meas == 0) ? 0 : c_meas / double(f_meas);
    if (doco)
      { cout << " c step " << f_meas << " -> " << c_meas << ", frac = " << meas_fac << endl; }

    // if goal was not reached or coarsening is slowing down too much,
    // we need to redistrubte if possible
    state.need_rd |= (goal_meas < c_meas);
    state.need_rd |= meas_fac > O.rd_crs_thresh;
    prol_map = BuildCoarseDOFMap(state.crs_map, f_cap, state.curr_cap, f_lev->embed_map);

    if (O.enable_redist && TryContractStep(state)) {
      // contract map constructed successfully
      state.level[2]++;
      rd_map = state.dof_map;
    }
    else {
      // probably contract is just not needed, but if state.need_rd we might be in trouble
    }
  }
  else { // coarsening failed completely - we give up I guess ?
    if (doco)
      { cout << "no cmap, stuck!" << endl; }
  }

  // { prol_map = SmoothedProlMap(prol_map, m0cap); }

  /** We now have emb - prol - rd. Concatenate and pack into final map. **/

  Array<shared_ptr<BaseDOFMapStep>> init_steps;
  if (embed_map != nullptr)
    { init_steps.Append(embed_map); }
  if (prol_map != nullptr)
    { init_steps.Append(prol_map); }
  if (rd_map != nullptr)
    { init_steps.Append(rd_map); }

  cout << " MAPS " << f_lev->level << "  -> " << f_lev->level+1 << ": " << endl << init_steps << endl;

  /** assemble coarse level matrix, set return vals, etc. **/
  c_lev->level = f_lev->level + 1;
  c_lev->cap = state.curr_cap; c_lev->cap->baselevel = c_lev->level;

  cout << " MapLevel!" << endl;
  auto final_step = MapLevel(init_steps, f_lev, c_lev);
  cout << " MapLevel OK!" << endl;

  if (final_step == nullptr)
    { c_lev->cap = f_lev->cap; }

  return final_step;
} // BaseAMGFactory::DoStep


void print_bmat(std::ofstream &of, shared_ptr<BaseMatrix> bm)
{
  DispatchSquareMatrixBS(*bm, [&](auto ABS) {
    constexpr int BS = ABS;
    auto sptm = my_dynamic_pointer_cast<stripped_spm_tm<Mat<BS,BS,double>>>(bm, "print_bmat");
    print_tm_spmat(of, *sptm);
  });
}

extern shared_ptr<BaseDOFMapStep> MakeSingleStep3 (FlatArray<shared_ptr<BaseDOFMapStep>> init_steps);

shared_ptr<BaseDOFMapStep>
BaseAMGFactory :: MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps, shared_ptr<AMGLevel> & f_lev, shared_ptr<AMGLevel> & c_lev)
{
  if (c_lev->cap->mesh == f_lev->cap->mesh)
    { return nullptr; }

  shared_ptr<BaseDOFMapStep> final_step;

  // TODO: with the stokes multi-step embedding we might need MakeSingleStep
  //       somewhere instead of MakeSingleStep2
  if ( (dof_steps.Size() > 1) && (f_lev->embed_map != nullptr) && (f_lev->embed_done) )
  {
    cout << "  MapLevel A " << endl;
    /** The fine level matrix is already embedded! **/
    auto afs = MakeSingleStep3(dof_steps.Part(1));
    if (afs == nullptr) { /** No idea how this would happen. **/
      c_lev->cap->mat = dof_steps[0]->AssembleMatrix(f_lev->cap->mat);
      dof_steps[0]->Finalize();
      final_step = dof_steps[0];
    }
    else { /** Use all steps except embedding for coarse level matrix. **/
      c_lev->cap->mat = afs->AssembleMatrix(f_lev->cap->mat);
      Array<shared_ptr<BaseDOFMapStep>> ds2( { dof_steps[0], afs } );
      final_step = MakeSingleStep3(ds2);
      final_step->Finalize();
    }
  }
  else
  {
    cout << "  MapLevel B " << endl;
    /** Fine level matrix is not embedded, or there is no embedding. **/

    final_step = MakeSingleStep3(dof_steps);

    if (final_step != nullptr)
    {
      c_lev->cap->mat = final_step->AssembleMatrix(f_lev->cap->mat);
      final_step->Finalize();
    }

    if (f_lev->level == 0)
    {
      if (auto multiEmb = dynamic_pointer_cast<MultiDofMapStep>(f_lev->embed_map))
      {
        Array<shared_ptr<BaseDOFMapStep>> otherSteps(dof_steps.Size());
        otherSteps[0] = multiEmb->GetMap(1);
        otherSteps.Part(1) = dof_steps.Part(1);

        Array<shared_ptr<BaseDOFMapStep>> multiStepSteps(2);
        multiStepSteps[0] = final_step;
        multiStepSteps[1] = MakeSingleStep3(otherSteps);
        multiStepSteps[1]->Finalize();

        auto multiStep = make_shared<MultiDofMapStep>(multiStepSteps);

        final_step = multiStep;
      }
    }
  }

  if (options->log_level == Options::LOG_LEVEL::DBG)
  {
    cout << "  MapLevel PRINT " << endl;

    if (f_lev->level == 0)
    {
      auto rk = f_lev->cap->uDofs.GetCommunicator().Rank();

      std::ofstream of("ngs_amg_mat_l0_rk" + std::to_string(rk) + ".out");
      cout << " print_bmat F-LEV " << endl;
      print_bmat(of, f_lev->cap->mat); of << endl;
      cout << " print_bmat F-LEV OK " << endl;
    }

    if (c_lev->cap->mat != nullptr)
    {
      auto rk = c_lev->cap->uDofs.GetCommunicator().Rank();
      std::ofstream of("ngs_amg_mat_l" + std::to_string(c_lev->level) + "_rk" + std::to_string(rk) + ".out");
      cout << " print_bmat C-LEV " << endl;
      print_bmat(of, c_lev->cap->mat); of << endl;
      cout << " print_bmat C-LEV " << endl;
    }
  }

  return final_step;
} // BaseAMGFactory::MapLevel


bool
BaseAMGFactory :: TryCoarseStep (State & state)
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

  if ( ( cmap->GetMesh()->GetEQCHierarchy()->GetCommunicator().Rank() == 0) &&
       ( options->log_level > Options::LOG_LEVEL::BASIC ) )
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


double
BaseAMGFactory :: FindRDFac (shared_ptr<TopologicMesh> mesh)
{
  const auto & O(*options);
  const auto & M(*mesh);
  const auto& eqc_h = *M.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();

  auto NV = M.template GetNNGlobal<NT_VERTEX>();
  auto meas = ComputeMeshMeasure(M);

  /** Already "sequential" **/
  int serialThresh = eqc_h.IsRankZeroIdle() ? 2 : 1;

  if (comm.Size() <= serialThresh)
    { return 1; }

  /** Sequential threshold reached **/
  if (meas < options->rd_seq_nv)
    { return -1; }

  /** default redist-factor **/
  double rd_factor = O.rd_pfac;

  /** ensure enough vertices per proc **/
  rd_factor = min2(rd_factor, double(meas) / O.rd_min_nv_gl / comm.Size());

  /** try to heuristically ensure that enough vertices are local **/
  // size_t nv_loc = M.template GetNN<NT_VERTEX>() > 0 ? M.template GetENN<NT_VERTEX>(0) : 0;
  // double frac_loc = comm.AllReduce(double(nv_loc), NG_MPI_SUM) / NV;
  double frac_loc = ComputeLocFrac(M);
  if (frac_loc < options->rd_loc_thresh)
  {
    size_t NP = comm.Size();
    size_t NF = comm.AllReduce (eqc_h.GetDistantProcs().Size(), NG_MPI_SUM) / 2; // every face counted twice
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


bool BaseAMGFactory::TryContractStep (State & state)
{
  static Timer t("TryContractStep");
  RegionTimer rt(t);

  const auto & O(*options);
  const auto & M = *state.curr_cap->mesh;
  const auto & eqc_h = *M.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();
  double meas = ComputeMeshMeasure(M);

  /** cannot redistribute if when is turned off or when we are truly, or effectively, sequential **/
  if ( (!O.enable_redist) || (!state.curr_cap->uDofs.IsTrulyParallel()) )
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
                  O.first_rdaf :
                  ( pow(O.rdaf_scale, state.level[0] - ( (O.first_rdaf == -1) ? 0 : 1) ) * O.rdaf );
    size_t goal_meas = max( size_t(min(af, 0.9) * state.last_redist_meas), max(O.rd_seq_nv, size_t(1)));
    want_redist |= (meas < goal_meas);
  }

  if (!want_redist)
    { return false; }

  auto rd_factor = FindRDFac (state.curr_cap->mesh);

  shared_ptr<LevelCapsule> c_cap = AllocCap();
  c_cap->baselevel = state.level[0];

  cout << " call BuildContractMap " << endl;
  auto rd_map = BuildContractMap(rd_factor, state.curr_cap->mesh, c_cap);
  cout << " -> GOT " << rd_map << endl;

  if (state.curr_cap->free_nodes != nullptr)
    { throw Exception("free-node redist update todo"); /** do sth here..**/ }

  cout << " BuildContractDOFMap " << endl;
  state.dof_map = BuildContractDOFMap(rd_map, state.curr_cap, c_cap);
  cout << " BuildContractDOFMap OK " << endl;
  state.first_redist_used = true;
  state.need_rd = false;
  state.last_redist_meas = meas;
  state.curr_cap = c_cap;

  return true;
} // BaseAMGFactory::TryContractStep


BaseAMGFactory::State* BaseAMGFactory::NewState (shared_ptr<AMGLevel> & lev)
{
  auto s = AllocState();
  InitState(*s, lev);
  return s;
} // BaseAMGFactory::NewState


void BaseAMGFactory::InitState (BaseAMGFactory::State& state, shared_ptr<AMGLevel> & lev) const
{
  state.level = { lev->level, 0, 0 };

  state.curr_cap = lev->cap;

  /** TODO: this is kind of unclean ... **/
  // if (lev->embed_map != nullptr)
  //   { state.curr_cap->pardofs = lev->embed_map->GetMappedParDofs(); }
  if (lev->embed_map != nullptr)
    { state.curr_cap->uDofs = lev->embed_map->GetMappedUDofs(); }

  state.crs_map = nullptr;
  state.curr_cap->free_nodes = lev->cap->free_nodes;

  state.first_redist_used = false;
  state.last_redist_meas = ComputeMeshMeasure(*lev->cap->mesh);
} // BaseAMGFactory::InitState


void BaseAMGFactory::SetOptionsFromFlags (BaseAMGFactory::Options& opts, const Flags & flags, string prefix)
{
  opts.SetFromFlags(flags, prefix);
} // BaseAMGFactory::SetOptionsFromFlags

/** END BaseAMGFactory **/

} // namespace amg
