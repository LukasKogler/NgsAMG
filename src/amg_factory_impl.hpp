#ifndef FILE_AMG_FACTORY_IMPL_HPP
#define FILE_AMG_FACTORY_IMPL_HPP

namespace amg
{

  /** --- Options --- **/

  template<class TMESH, class TM>
  struct AMGFactory<TMESH, TM> :: Options
  {
    /** Level-control **/
    size_t max_n_levels = 10;                   // maximun number of multigrid levels (counts first level, so at least 2)
    size_t max_meas = 50;                       // maximal maesure of coarsest mesh
    double aaf = 0.1;                           // assemble after factor
    double first_aaf = 0.05;                    // (smaller) factor for first level. -1 for dont use
    double aaf_scale = 1;                       // scale aaf, e.g if 2:   first_aaf, aaf, 2*aaf, 4*aaf, .. (or aaf, 2*aaf, ...)
    
    /** Discard - not back yet  **/

    /** Contract (Re-Distribute) **/
    bool enable_ctr = true;
    /** WHEN to contract **/
    double ctraf = 0.05;                        // contract after reducing measure by this factor
    double first_ctraf = 0.025;                 // see first_aaf
    double ctraf_scale = 1;                     // see aaf_scale
    double ctr_crs_thresh = 0.7;                // if coarsening slows down more than this, redistribute
    double ctr_loc_thresh = 0.5;                // if less than this fraction of vertices are purely local, redistribute
    /** HOW AGGRESSIVELY to contract **/
    double ctr_pfac = 0.25;                     // per default, reduce active NP by this factor (ctr_pfac / ctraf should be << 1 !)
    /** additional constraints for contract **/
    size_t ctr_min_nv_th = 500;                 // re-distribute when there are less than this many vertices per proc left
    size_t ctr_min_nv_gl = 500;                 // try to re-distribute such that at least this many NV per proc remain
    size_t ctr_seq_nv = 1000;                   // always re-distribute to sequential once NV reaches this threshhold
    double ctr_loc_gl = 0.8;                    // always try to redistribute such that at least this fraction will be local

    /** Smoothed Prolongation **/
    bool enable_sm = true;                      // emable prolongation-smoothing

    /** Build a new mesh from a coarse level matrix**/
    bool enable_rbm = false;                    // probably only necessary on coarse levels
    double rbmaf = 0.01;                        // rebuild mesh after measure decreases by this factor
    double first_rbmaf = 0.005;                 // see first_aaf
    double rbmaf_scale = 1;                     // see aaf_scale
    std::function<shared_ptr<TMESH>(shared_ptr<TMESH>, shared_ptr<BaseSparseMatrix>, shared_ptr<ParallelDofs>)> rebuild_mesh =
      [](shared_ptr<TMESH> mesh, shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs>) { return mesh; };

    /** Logging **/
    enum LOG_LEVEL : char { NONE   = 0,         // nothing
			    BASIC  = 1,         // summary info
			    NORMAL = 2,         // global level-wise info
			    EXTRA  = 3};        // local level-wise info
    LOG_LEVEL log_level = LOG_LEVEL::NORMAL;    // how much info do we collect
    bool print_log = true;                      // print log to shell
    string log_file = "";                       // which file to print log to (none if empty)
  };


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM> :: SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix)
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

    set_num(opts.max_n_levels, "max_levels");
    set_num(opts.max_meas, "max_coarse_size");
    set_num(opts.aaf, "aaf");
    set_num(opts.first_aaf, "first_aaf");
    set_num(opts.aaf_scale, "aaf_scale");

    set_bool(opts.enable_ctr, "enable_redist");
    set_num(opts.ctraf, "rdaf");
    set_num(opts.first_ctraf, "first_rdaf");
    set_num(opts.ctraf_scale, "rdaf_scale");
    set_num(opts.ctr_pfac, "rdaf_pfac");
    set_num(opts.ctr_min_nv_gl, "rd_min_nv");
    set_num(opts.ctr_min_nv_th, "rd_min_nv_thr");
    opts.ctr_min_nv_th = min2(opts.ctr_min_nv_th, opts.ctr_min_nv_gl / 2); // we always contract by at least factor 2
    set_num(opts.ctr_seq_nv, "rd_seq");

    set_bool(opts.enable_sm, "enable_sp");

    set_bool(opts.enable_rbm, "enable_rbm");
    set_num(opts.rbmaf, "rbmaf");
    set_num(opts.first_rbmaf, "first_rbmaf");
    set_num(opts.rbmaf_scale, "rbmaf_scale");

    set_enum_opt(opts.log_level, "log_level", {"none", "basic", "normal", "extra"});
    opts.log_file = flags.GetStringFlag("log_file", "");
    set_bool(opts.print_log, "print_log");
  }


  template<NODE_TYPE NT, class TMESH, class TM>
  struct NodalAMGFactory<NT, TMESH, TM> :: Options : public AMGFactory<TMESH, TM>::Options
  {

  };


  template<NODE_TYPE NT, class TMESH, class TM>
  void NodalAMGFactory<NT, TMESH, TM> :: SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix)
  {
    BASE::SetOptionsFromFlags(opts, flags, prefix);
  }


  template<class FACTORY_CLASS, class TMESH, class TM>
  struct VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: Options : public NodalAMGFactory<NT_VERTEX, TMESH, TM>::Options
  {
    /** Coarsening **/
    double min_ecw = 0.05;
    double min_vcw = 0.3;

    /** Smoothed Prolongation **/
    double sp_min_frac = 0.1;          // min. (relative) wt to include an edge
    int sp_max_per_row = 3;                 // maximum entries per row (should be >= 2!)
    double sp_omega = 1.0;               // relaxation parameter for prol-smoothing
  };


  template<class FACTORY_CLASS, class TMESH, class TM>
  void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix)
  {
    BASE::SetOptionsFromFlags (opts, flags, prefix);
    
    auto set_num = [&](auto& v, string key)
      { v = flags.GetNumFlag(prefix + key, v); };
    
    set_num(opts.min_ecw, "edge_thresh");
    set_num(opts.min_vcw, "vert_thresh");
    set_num(opts.sp_min_frac, "sp_thresh");
    set_num(opts.sp_max_per_row, "sp_max_per_row");
    set_num(opts.sp_omega, "sp_omega");
  }


  /** --- Logging --- **/

  template<class TMESH, class TM>
  class AMGFactory<TMESH, TM> :: Logger
  {
  public:
    using LOG_LEVEL = typename AMGFactory<TMESH, TM>::Options::LOG_LEVEL;

    Logger (LOG_LEVEL _lev, int max_levels = 10)
      : lev(_lev), ready(false)
    { Alloc(max_levels); }

    void LogLevel (AMGFactory<TMESH, TM>::Capsule cap);

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


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM>::Logger :: Alloc (int N)
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


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM>::Logger :: LogLevel (AMGFactory<TMESH, TM>::Capsule cap)
  {
    ready = false;
    
    if (cap.level == 0)
      { comm = cap.mesh->GetEQCHierarchy()->GetCommunicator(); }

    if (lev == LOG_LEVEL::NONE)
      { return; }

    auto lev_comm = cap.mesh->GetEQCHierarchy()->GetCommunicator();

    vcc.Append(cap.mesh->template GetNNGlobal<NT_VERTEX>());
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
    occl.Append(cap.mat->NZE());

  } // Logger::LogLevel


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM>::Logger :: Finalize ()
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
      auto maxval = comm.AllReduce(val, MPI_MAX);
      rk = comm.AllReduce( (val == maxval) ? (int)comm.Rank() : (int)comm.Size(), MPI_MIN);
      if ( (comm.Rank() == 0) && (rk != 0) )
	{ comm.Recv(array, rk, MPI_TAG_AMG); }
      else if ( (comm.Rank() == rk) && (rk != 0) )
	{ comm.Send(array, 0, MPI_TAG_AMG); }
      val = maxval;
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

    cout << "loc OCC: " << op_comp_l << endl;
				       prow(occl); cout << endl;

    alam(occl_rank, op_comp_l, occl);

  } // Logger::Finalize


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM>::Logger :: PrintLog (ostream & out)
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


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM>::Logger :: PrintToFile (string file_name)
  {
    if (comm.Rank() == 0) {
      ofstream out(file_name, ios::out);
      PrintLog(out);
    }
  } // Logger::PrintToFile


  /** --- State --- **/


  template<class TMESH, class TM>
  struct AMGFactory<TMESH, TM> :: State
  {
    /** Contract **/
    bool first_ctr_used = false;
    size_t last_ctr_nv;

    /** Rebuild Mesh **/
    bool first_rbm_used = false;
    size_t last_meas_rbm;

    /** hacky stuff **/
    bool coll_cross = false;
  };
  
  
  /** --- AMGFactory --- **/


  template<class TMESH, class TM>
  AMGFactory<TMESH, TM> :: AMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts, shared_ptr<BaseDOFMapStep> _embed_step)
    : options(_opts), finest_mesh(_finest_mesh), embed_step(_embed_step)
  { ; }


  template<class TMESH, class TM>
  void AMGFactory<TMESH, TM> :: SetupLevels (Array<shared_ptr<BaseSparseMatrix>> & mats, shared_ptr<DOFMap> & dof_map)
  {
    static Timer t("SetupLevels"); RegionTimer rt(t);

    if(mats.Size() != 1)
      { throw Exception("SetupLevels needs a finest level mat!"); }
    
    cout << "logger" << endl;
    logger = make_shared<Logger>(options->log_level);
    cout << "logger ok" << endl;

    cout << "embed_step: " << embed_step << endl;
    
    auto fmat = mats[0];
    cout << "pardofs" << endl;

    /** rank 0 (and maybe sometimes others??) can also not have an embed_step, while others DO.
	ParallelDofs - constructor has to be called by every member of the communicator! **/
    int have_embed = fmat->GetParallelDofs()->GetCommunicator().AllReduce((embed_step == nullptr) ? 0 : 1, MPI_SUM);
    shared_ptr<ParallelDofs> fpds = (have_embed == 0) ? fmat->GetParallelDofs() : BuildParallelDofs(finest_mesh);
    cout << "pardofs ok" << endl;

    cout << "calll RSU" << endl;
    auto coarse_mats = RSU({ 0, finest_mesh, fpds, fmat }, dof_map);
    cout << "RSU dne" << endl;

    // coarse mats in reverse order
    mats.SetSize(coarse_mats.Size() + 1);
    for (auto k : Range(mats.Size() - 1))
      { mats[k+1] = coarse_mats[coarse_mats.Size()-1-k]; }
    mats[0] = fmat;

    if (options->print_log)
      { logger->PrintLog(cout); }
    if (options->log_file.size() > 0)
      { logger->PrintToFile(options->log_file); }
    logger = nullptr;
      
  }


  template<class TMESH, class TM>
  Array<shared_ptr<BaseSparseMatrix>> AMGFactory<TMESH, TM> :: RSU (Capsule cap, shared_ptr<DOFMap> dof_map)
  {
    cout << "RSU2 " << cap.level << endl;

    logger->LogLevel(cap);

    shared_ptr<BaseDOFMapStep> ds;

    shared_ptr<TMESH> fmesh = cap.mesh, cmesh = move(cap.mesh);
    shared_ptr<ParallelDofs> fpds = move(cap.pardofs), cpds = nullptr;
    shared_ptr<BaseSparseMatrix> fmat = cap.mat, cmat = move(cap.mat);
    size_t curr_meas = ComputeMeshMeasure(*fmesh);

    if (cap.level == 0) { // (re-) initialize book-keeping
      state = make_shared<State>();
      state->first_ctr_used = state->first_rbm_used = false;
      state->last_ctr_nv = fmesh->template GetNNGlobal<NT_VERTEX>();
      state->last_meas_rbm = curr_meas;
      state->coll_cross = false;
    }

    auto comm = fmesh->GetEQCHierarchy()->GetCommunicator();
    
    double af = ( (cap.level == 0) && (options->first_aaf != -1) ) ?
      options->first_aaf : ( pow(options->aaf_scale, cap.level - ( (options->first_aaf == -1) ? 0 : 1) ) * options->aaf );

    size_t goal_meas = max( size_t(min(af, 0.9) * curr_meas), max(options->max_meas, size_t(1)));

    // if (comm.Rank() == 0) {
    //   cout << "using AF " << af << " from " << options->first_aaf << " " << options->aaf_scale << " " << options->aaf << endl;
    //   cout << "goal is: " << goal_meas << endl;
    // }

    INT<3> level = { cap.level, 0, 0 }; // coarse / sub-coarse / ctr 

    // Array<shared_ptr<shared_ptr<TMESH>>> sprol_meshes; // meshes we use for prol-smoothing
    // sprol_meshes.Append(fmesh);

    /**
       the mesh we use for prol-smoothing
       [ this can be different from "fmesh" when have to start out with a different map than coarse-map, e.g. contract ]
    **/
    shared_ptr<TMESH> sprol_mesh = fmesh;
    shared_ptr<BaseSparseMatrix> sprol_mat = fmat;

    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    shared_ptr<TSPM_TM> conc_pwp;

    double frac_loc = -1;
    double crs_meas_fac = 0;
    auto old_meas = curr_meas;

    auto do_coarsen_step = [&] () -> bool LAMBDA_INLINE {
	// if ( (level[1] != 0) && (crs_meas_fac > 0.6) )
	//   { state->coll_cross = true; }

	cout << "BCM" << endl;
	auto grid_step = BuildCoarseMap (cmesh);
	cout << "BCM" << endl;

	state->coll_cross = false;

	if (grid_step == nullptr) // coarsening did not work - we are stuck!
	  { return true; }

	auto _cmesh = static_pointer_cast<TMESH>(grid_step->GetMappedMesh());

	crs_meas_fac = ComputeMeshMeasure(*_cmesh) / (1.0 * curr_meas);

	frac_loc = _cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(double(_cmesh->template GetNN<NT_VERTEX>() > 0 ? _cmesh->template GetENN<NT_VERTEX>(0) : 0), MPI_SUM) / _cmesh->template GetNNGlobal<NT_VERTEX>();
	cout << IM(4) << "coarsen fac " << crs_meas_fac << ", went from " << curr_meas << " to " << ComputeMeshMeasure(*_cmesh) << ", frac loc " << frac_loc << endl;


	cout << " stuck1 " << (crs_meas_fac > options->ctr_crs_thresh) << endl;
	cout << " sutck2 " << (frac_loc < options->ctr_loc_thresh) << endl;

	if ( (options->enable_ctr) && // break out of inner coarsening loop due to slow-down
	     ( (crs_meas_fac > options->ctr_crs_thresh) ||
	       (frac_loc < options->ctr_loc_thresh) ) )
	  { return true; }

	cmesh = _cmesh;

	free_verts = nullptr; // relevant on finest level

	cout << "PWP" << endl;
	auto pwp = BuildPWProl(grid_step, cpds); // fine pardofs are correct
	cout << "PWP OK" << endl;

	conc_pwp = (conc_pwp == nullptr) ? pwp : MatMultAB(*conc_pwp, *pwp);
	cpds = BuildParallelDofs(cmesh);

	curr_meas = ComputeMeshMeasure(*cmesh);

	level[1]++;

	return false; // not stuck
    };


    auto do_contract_step = [&] () -> bool LAMBDA_INLINE {

      /** cannot redistribute if when is turned off or when we are already basically sequential **/
      if ( (!options->enable_ctr) || (cmesh->GetEQCHierarchy()->GetCommunicator().Size() <= 2) )
	{ return false; }

      cout << "do_contract_step" << endl;
      
      /** Find out how rapidly we should redistribute **/
      auto ccomm = cmesh->GetEQCHierarchy()->GetCommunicator();

      double fac = options->ctr_pfac;
      auto next_nv = cmesh->template GetNNGlobal<NT_VERTEX>();

      double af = ( (!state->first_ctr_used) && (options->first_ctraf != -1) ) ?
      options->first_ctraf : ( pow(options->ctraf_scale, cap.level - ( (options->first_ctraf == -1) ? 0 : 1) ) * options->ctraf );
      size_t goal_nv = max( size_t(min(af, 0.9) * state->last_ctr_nv), max(options->ctr_seq_nv, size_t(1)));
      double ctr_factor = 1;

      cout << IM(5) << "ctr goal from " << state->first_ctr_used << " " << options->first_ctraf << " " << options->ctraf_scale << " "
      << cap.level - ( (options->first_ctraf == -1) ? 0 : 1) << " " << options->ctraf << endl;
      cout << IM(5) << "goal " << goal_nv << endl;

      if (next_nv < options->ctr_seq_nv) // sequential threshold reached
	{ ctr_factor = -1; }
      else if ( (crs_meas_fac > options->ctr_crs_thresh) ||                                           // coarsening slows down
		(cmesh->template GetNNGlobal<NT_VERTEX>() < ccomm.Size() * options->ctr_min_nv_th) || // NV/NP too small
		(goal_nv > cmesh->template GetNNGlobal<NT_VERTEX>()) ||                               // static redistribute every now and then
		(frac_loc < options->ctr_loc_thresh) ) {                                              // not enough local vertices

	cout << IM(5) << " ctr bc " << crs_meas_fac  << " > " << options->ctr_crs_thresh << endl;
	cout << IM(5) << " ctr bc " <<  goal_nv << " > " << cmesh->template GetNNGlobal<NT_VERTEX>() << endl;
	cout << IM(5) << " ctr bc " << frac_loc << " < " << options->ctr_loc_thresh << endl;

	ctr_factor = fac;

	if (frac_loc < options->ctr_loc_gl) {
	  size_t NP = ccomm.Size();
	  size_t NF = ccomm.AllReduce (cmesh->GetEQCHierarchy()->GetDistantProcs().Size(), MPI_SUM) / 2; // every face counted twice
	  double F = frac_loc;
	  double FGOAL = options->ctr_loc_gl; // we want to achieve this large of a frac of local verts
	  double FF = (1-frac_loc) / NF; // avg frac of a face

	  if (F + NP/2 * FF > FGOAL) // merge 2 -> one face per group becomes local
	    { ctr_factor = 0.5; }
	  else if (F + NP/4 * 5 * FF > FGOAL) // merge 4 -> probably 4(2x2) to 6(tet) faces per group
	    { ctr_factor = 0.25; }
	  else // merge 8, in 3d probably 12 edges (2x2x2) per group. probably never want to do more than that
	    { ctr_factor = 0.125; }
	}
      }

      ctr_factor = min2(ctr_factor, double(next_nv) / options->ctr_min_nv_gl / ccomm.Size());

      if (ctr_factor == 1) // should not redistribute
	{ return false; }

      state->first_ctr_used = true;

      ctr_factor = min2(0.5, ctr_factor); // ALWAYS at least factor 2
	  
      cout << IM(4) << "contract by factor " << ctr_factor << endl;

      cout << "do_contract_step" << endl;
      auto ctr_map = BuildContractMap(ctr_factor, cmesh);
      cout << "do_contract_step" << endl;

      if (free_verts != nullptr) { // TODO: hacky
	/** should only happen very rarely! **/
	cout << " FIX FVS " << endl;
	Array<int> ff (fmesh->template GetNN<NT_VERTEX>());
	for (auto k : Range(ff))
	  ff[k] = free_verts->Test(k) ? 1 : 0;
	auto CM = static_pointer_cast<TMESH>(ctr_map->GetMappedMesh());
	auto ncv = (CM == nullptr) ? 0 : CM->template GetNN<NT_VERTEX>();
	Array<int> cf;
	ctr_map->template MapNodeData<NT_VERTEX, int> (ff, CUMULATED, &cf);
	if (CM == nullptr)
	  { free_verts = nullptr; }
	else {
	  free_verts = make_shared<BitArray>(ncv);
	  for (auto k : Range(cf))
	    if (cf[k] == 0)
	      { free_verts->Clear(k); }
	    else
	      { free_verts->Set(k); }
	}
      }

      fmesh = cmesh = static_pointer_cast<TMESH>(ctr_map->GetMappedMesh());

      auto ds = BuildDOFContractMap(ctr_map, fpds);

      fpds = ds->GetMappedParDofs(); cpds = ds->GetMappedParDofs();

      sub_steps.Append(ds);

      level[2]++;


      if (cmesh == nullptr) // dropped out
	{ return true; }

      double locnvloc = cmesh->template GetNN<NT_VERTEX>() > 0 ? cmesh->template GetENN<NT_VERTEX>(0) : 0;
      double frac_loc = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(locnvloc, MPI_SUM) / cmesh->template GetNNGlobal<NT_VERTEX>();

      cout << IM(4) << "NP down to " << cmesh->GetEQCHierarchy()->GetCommunicator().Size() << ", frac loc " << frac_loc << endl;

      return true;
    }; // do_contract_step


    /** chain maps until we have coarsened the mesh appropiately **/
    while (curr_meas > goal_meas) { // outer loop - when inner loop is stuck (or done), try to redistribute

      /** inner loop - do coarsening until stuck **/
      bool stuck = false;
      while ( (curr_meas > goal_meas) && (!stuck) ) {
	cout << "DCS " << endl;
	cout << "DCS cmesh " << cmesh << " " << cmesh->template GetNNGlobal<NT_VERTEX>() << endl;
	stuck = do_coarsen_step();
	cout << "DCS OK " << endl;
	cout << "DCS OK cmesh " << cmesh << " " << cmesh->template GetNNGlobal<NT_VERTEX>() << endl;
      }

      /** we PROBABLY have a coarse map **/
      if(conc_pwp != nullptr) {
	sub_steps.Append(make_shared<ProlMap<TSPM_TM>> (conc_pwp, fpds, cpds));
	fpds = cpds; conc_pwp = nullptr; fmesh = cmesh;
      }

      /** try contract **/
      cout << "DCTRS" << endl;
      auto ctr_worked = do_contract_step();
      cout << "DCTRS OK" << endl;

      /** contract is first map - need to smooth prol with first mapped mesh **/
      // if ( ctr_worked && (sub_steps.Size() == 1) )
      // 	{ sprol_mesh = cmesh; }
      // ACTUALLY - why?? just swap CP to PC then smooth with orig mesh I think

      /** if contract did not work, or if we are contracted out, break out of loop **/
      if ( (!ctr_worked) || (cmesh == nullptr))
	  { break; }
    }

    // cout << " done, steps are " << endl; prow(sub_steps); cout << endl;
    // for (auto step : sub_steps)
    //   cout << typeid(*step).name() << endl;

    /**
       If we were unable to do any maps, we should return here, or maybe do some fallback scheme,
       like weakening the coarsening criteria.
    **/

    /** 
	Swap contracts and prols, from right to left:
	   I)   P - C - P - C - P    // 
	   II)  P - C - PP - C       // swapped last C-P to P-C, multiply prols
	   III) PPP - C - C          // same
	   IV)  PPP - CC             // (TODO!!) concatenate C-maps 

	If we are contracted out, our last map is a C-map. It might, however, still be
	followed by a P-map, WHICH WE DON'T KNOW ABOUT!
    **/
    using TCTR = CtrMap<typename strip_vec<Vec<mat_traits<TM>::HEIGHT, double>>::type>;
    using TCRS = ProlMap<TSPM_TM>;
    { // swap C/P
      if ( auto ctr_last = dynamic_pointer_cast<TCTR>(sub_steps.Last()) ) {

	cout << "last step is ctr!" << endl;
	auto fpd = ctr_last->GetParDofs();
	cout << " in that comm " << fpd->GetCommunicator().Rank() << " of " << fpd->GetCommunicator().Size() << endl;

	// if there is no prol after it, all members of group enter with false
	if (ctr_last->DoSwap(false)) { // master must have entered with true - do swap!
	  cout << "do swap!" << endl;
	  // auto fpd = ctr_last->GetParDofs();
	  sub_steps.Append(nullptr);
	  sub_steps.Last() = sub_steps[sub_steps.Size()-2];
	  sub_steps[sub_steps.Size()-2] = ctr_last->SwapWithProl(nullptr);
	}
	else
	  { cout << " no swap " << endl; }
      }

      for (int step_nr = sub_steps.Size() - 1; step_nr > 0; step_nr--) {
	// if [step_nr - 1, step_nr] are [prol, ctr], swap them
	auto step_L = sub_steps[step_nr-1];
	auto step_R = sub_steps[step_nr];
	// cout << " check steps " << step_nr - 1 << " " << step_nr << endl;
	// cout << typeid(*step_L).name() << " " << typeid(*step_R).name() << endl;
	if ( auto ctr_L = dynamic_pointer_cast<TCTR>(step_L) ) {
	  if ( auto crs_R = dynamic_pointer_cast<TCRS>(step_R) ) { // C -- P -> swap to P -- C
	    // cout << " case 1!" << endl;
	    auto fpd = ctr_L->GetParDofs();
	    // cout << "doswap with true!" << endl;
	    // cout << " in that comm " << fpd->GetCommunicator().Rank() << " of " << fpd->GetCommunicator().Size() << endl;
	    ctr_L->DoSwap(true);
	    // cout << "am back!" << endl;
	    sub_steps[step_nr-1] = ctr_L->SwapWithProl(crs_R);
	    sub_steps[step_nr]   = ctr_L;
	  }
	  else // TODO: C -- C -> concatenate to single C
	    { /* cout << " case CC " << endl; */ ; }
	}
	else if ( auto crs_L = dynamic_pointer_cast<TCRS>(step_L) ) {
	  if ( auto crs_R = dynamic_pointer_cast<TCRS>(step_R) ) { // P -- P, concatenate to single P (actually: P -- nullptr)
	    // cout << " case 2" << endl;
	    auto conc_P = MatMultAB(*crs_L->GetProl(), *crs_R->GetProl());
	    auto conc_map = make_shared<TCRS>(conc_P, crs_L->GetParDofs(), crs_R->GetMappedParDofs());
	    
	    sub_steps[step_nr-1] = conc_map;
	    sub_steps[step_nr] = nullptr;
	  }
	  else // P -- C, nothing to do, leave sub_step entries as-is
	    { /* cout << " case PC " << endl; */ ; }
	}

      }
      cout << " swapping done, remove nullptrs" << endl;

      /**
	 remove nullptr-maps from sub_steps
	 now we have P -- C -- C -- C ... (until i can concatenate C -- C to C, then it should really just be just P - C)
      **/
      cout << "in steps " << endl;
      for (auto k : Range(sub_steps)) {
	auto & step = sub_steps[k];
	cout << step;
	if (step != nullptr) cout << " " << typeid(*step).name();
	cout << endl;
      }
      int c = 0;
      for (int j = 0; j < sub_steps.Size(); j++)
	if (sub_steps[j] != nullptr)
	  { sub_steps[c++] = sub_steps[j]; }
      sub_steps.SetSize(c);
    } // swap C/P

    /** smooth prolongation **/
    if (options->enable_sm) {
      auto pstep = dynamic_pointer_cast<TCRS>(sub_steps[0]);
      cout << " first step " << pstep << endl;
      cout << " first step type " << typeid(*pstep).name() << endl;
      if (pstep == nullptr)
	{ throw Exception("Something must be broken!!"); }
      if ( (options->enable_rbm) && (level[0] != 0) )
	{ sprol_mesh = options->rebuild_mesh(sprol_mesh, sprol_mat, pstep->GetParDofs()); }
      cout << "SPROL" << endl;
      SmoothProlongation(pstep, sprol_mesh);
      cout << "SPROL OK" << endl;
    } // sm-prol

    /** add embed_step in the beginning **/
    if (embed_step != nullptr) {
      cout << "EMB" << endl;
      auto conc_step = embed_step->Concatenate(sub_steps[0]);
      if (conc_step != nullptr)
	{ sub_steps[0] = conc_step; }
      else {
	sub_steps.Append(nullptr);
	for (int k = sub_steps.Size()-1; k > 0; k--)
	  { sub_steps[k] = sub_steps[k-1]; }
	sub_steps[0] = embed_step;
      }
      embed_step = nullptr; 
      cout << "EMB OK" << endl;
    }

    /** assemble coarse matrix and add DOFstep to map **/
    // cout << " sub_steps: " << endl << sub_steps << endl;
    // for (auto step : sub_steps)
    //   cout << typeid(*step).name() << endl;
    shared_ptr<BaseDOFMapStep> tot_step;
    cout << "TS" << endl;
    if (sub_steps.Size() > 1)
      { tot_step = make_shared<ConcDMS>(sub_steps); }
    else
      { tot_step = sub_steps[0]; }
    cout << "TSOK" << endl;
    
    // cout << " ass mat " << endl;
    cout << " ASS MAT " << endl;
    cmat = tot_step->AssembleMatrix(cmat);
    cout << " ASS MAT ok" << endl;
    // cout << " have ass mat! " << endl;

    dof_map->AddStep(tot_step);

    /** potentially clean up some stuff before recursive call **/
    fmesh = sprol_mesh = nullptr;
    fpds = nullptr;
    bool did_nothing = cmat == fmat;
    fmat = sprol_mat = nullptr;

    /** recursive setup call **/
    if (cmesh == nullptr) // I drop out!
      { return Array<shared_ptr<BaseSparseMatrix>>({ nullptr }); }
    else if (cmat == nullptr) // contracted out - redundant?
      { return Array<shared_ptr<BaseSparseMatrix>> ({cmat}); }
    else if (did_nothing) { // I could not do anyting, stop coarsening here I guess
      throw Exception("hold on, is this ok??");
      return Array<shared_ptr<BaseSparseMatrix>> (0);
    }
    else if ( (cap.level + 2 == options->max_n_levels) )
      { return Array<shared_ptr<BaseSparseMatrix>> ({cmat}); }
    else if ( (cap.level + 2 == options->max_n_levels) ||              // max n levels reached
    	      (options->max_meas >= ComputeMeshMeasure (*cmesh) ) ) {  // max coarse size reached
      logger->LogLevel ({cap.level + 1, cmesh, cpds, cmat}); // also log coarsest level
      return Array<shared_ptr<BaseSparseMatrix>> ({cmat});
    }
    else { // actual recursive call
      auto cmats = RSU( {cap.level + 1, cmesh, cpds, cmat}, dof_map );
      cmats.Append(cmat);
      return cmats;
    }

  } // AMGFactory::RSU


  template<class TMESH, class TM>
  shared_ptr<GridContractMap<TMESH>> AMGFactory<TMESH, TM> :: BuildContractMap (double factor, shared_ptr<TMESH> mesh) const
  {
    static Timer t("BuildContractMap"); RegionTimer rt(t);
    // at least 2 groups - dont send everything from 1 to 0 for no reason
    int n_groups = (factor == -1) ? 2 : max2(int(2), int(1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * factor)));
    cout << "pprocs" << endl;
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);
    cout << "pprocs" << endl;
    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
  }


  /** --- NodalAMGFactory --- **/
  

  template<NODE_TYPE NT, class TMESH, class TM>
  NodalAMGFactory<NT, TMESH, TM> :: NodalAMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts,
						     shared_ptr<BaseDOFMapStep> _embed_step)
    : BASE(_finest_mesh, _opts, _embed_step)
  { ; }


  template<NODE_TYPE NT, class TMESH, class TM>
  shared_ptr<ParallelDofs> NodalAMGFactory<NT, TMESH, TM> :: BuildParallelDofs (shared_ptr<TMESH> amesh) const
  {
    const auto & mesh = *amesh;
    const auto & eqc_h = *mesh.GetEQCHierarchy();
    size_t neqcs = eqc_h.GetNEQCS();
    size_t ndof = mesh.template GetNN<NT_VERTEX>();
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
    auto pds = make_shared<ParallelDofs> (eqc_h.GetCommunicator(), move(tab) /* cdps.MoveTable() */, mat_traits<TM>::HEIGHT, false);
    return pds;
  }


  template<NODE_TYPE NT, class TMESH, class TM>
  shared_ptr<BaseDOFMapStep> NodalAMGFactory<NT, TMESH, TM> :: BuildDOFContractMap (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    auto fg = cmap->GetGroup();
    Array<int> group(fg.Size()); group = fg;
    Table<int> dof_maps;
    shared_ptr<ParallelDofs> cpd = nullptr;
    if (cmap->IsMaster()) {
      // const TMESH& cmesh(*static_cast<const TMESH&>(*grid_step->GetMappedMesh()));
      shared_ptr<TMESH> cmesh = static_pointer_cast<TMESH>(cmap->GetMappedMesh());
      cpd = BuildParallelDofs(cmesh);
      Array<int> perow (group.Size()); perow = 0;
      for (auto k : Range(group.Size())) perow[k] = cmap->template GetNodeMap<NT>(k).Size();
      dof_maps = Table<int>(perow);
      for (auto k : Range(group.Size())) dof_maps[k] = cmap->template GetNodeMap<NT>(k);
    }
    auto ctr_map = make_shared<CtrMap<typename strip_vec<Vec<mat_traits<TM>::HEIGHT, double>>::type>> (fpd, cpd, move(group), move(dof_maps));
    if (cmap->IsMaster()) {
      ctr_map->_comm_keepalive_hack = cmap->GetMappedEQCHierarchy()->GetCommunicator();
    }
    return move(ctr_map);
  }


  /** --- VertexBasedAMGFactory --- **/


  template<class FACTORY_CLASS, class TMESH, class TM>
  VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: VertexBasedAMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<Options> _opts,
									    shared_ptr<BaseDOFMapStep> _embed_step)
    : BASE(_finest_mesh, _opts, _embed_step)
  { ; }


  template<class FACTORY_CLASS, class TMESH, class TM>
  size_t VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: ComputeMeshMeasure (const TMESH & m) const
  {
    return m.template GetNNGlobal<NT_VERTEX>();
  }


  template<class FACTORY_CLASS, class TMESH, class TM>
  shared_ptr<CoarseMap<TMESH>> VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildCoarseMap  (shared_ptr<TMESH> mesh) const
  {
    static Timer t("BuildCoarseMap"); RegionTimer rt(t);
    shared_ptr<VWiseCoarsening<TMESH>> calg;
    if (this->state->coll_cross) {
      if (mesh->GetEQCHierarchy()->GetCommunicator().Rank() == 0)
	{ cout << "collapse cross!" << endl; }
      auto coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
      this->SetCoarseningOptions(*coarsen_opts, mesh);
      calg = make_shared<HierarchicVWC<TMESH>> (coarsen_opts);
    }
    else {
      auto coarsen_opts = make_shared<typename BlockVWC<TMESH>::Options>();
      cout << "SCO " << endl;
      this->SetCoarseningOptions(*coarsen_opts, mesh);
      cout << "SCO OK" << endl;
      calg = make_shared<BlockVWC<TMESH>> (coarsen_opts);
      cout << "CALG OK" << endl;
    }
    return calg->Coarsen(mesh);
  }


  template<class FACTORY_CLASS, class TMESH, class TM>
  shared_ptr<typename VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM>::TSPM_TM>
  VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildPWProl (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const CoarseMap<TMESH> & rcmap(*cmap);
    const TMESH & fmesh = static_cast<TMESH&>(*rcmap.GetMesh()); 
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;
    fmesh.CumulateData();
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;
    const TMESH & cmesh = static_cast<TMESH&>(*rcmap.GetMappedMesh()); cmesh.CumulateData();
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;

    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();

    // Alloc Matrix
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;
    auto vmap = rcmap.template GetMap<NT_VERTEX>();
    Array<int> perow (NV); perow = 0;
    // -1 .. cant happen, 0 .. locally single, 1+ ..locally merged
    // -> cumulated: 0..single, 1+..merged
    Array<int> has_partner (NCV); has_partner = -1;
    for (auto vnr : Range(NV)) {
      auto cvnr = vmap[vnr];
      if (cvnr != -1)
	{ has_partner[cvnr]++; }
    }
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;
    cmesh.template AllreduceNodalData<NT_VERTEX, int>(has_partner, [](auto & tab){ return move(sum_table(tab)); });
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;
    for (auto vnr : Range(NV))
      { if (vmap[vnr] != -1) perow[vnr] = 1; }
    auto prol = make_shared<TSPM_TM>(perow, NCV);
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;

    // Fill Matrix
    for (auto vnr : Range(NV)) {
      if (vmap[vnr]!=-1) {
	auto ri = prol->GetRowIndices(vnr);
	auto rv = prol->GetRowValues(vnr);
	auto cvnr = vmap[vnr];
	ri[0] = cvnr;
	if (has_partner[cvnr]==0) { // single vertex
	  SetIdentity(rv[0]);
	}
	else { // merged vertex
	  self.CalcPWPBlock (fmesh, cmesh, rcmap, vnr, cvnr, rv[0]);
	}
      }
    }
    fmesh.GetEQCHierarchy()->GetCommunicator().Barrier();
    cout << "BPWP" << endl;

    // cout << "PWP MAT: " << endl;
    // print_tm_spmat(cout, *prol); cout << endl<< endl;

    return prol;
  } // VertexBasedAMGFactory::BuildPWProl


  template<class FACTORY_CLASS, class TMESH, class TM>
  void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const
  {
    static Timer t("SmoothProlongation"); RegionTimer rt(t);

    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const TMESH & fmesh(*mesh); fmesh.CumulateData();
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch==fine eqch !!
    const TSPM_TM & pwprol = *pmap->GetProl();

    Options &O (static_cast<Options&>(*options));
    const double MIN_PROL_FRAC = O.sp_min_frac;
    const int MAX_PER_ROW = O.sp_max_per_row;
    const double omega = O.sp_omega;

    // Construct vertex-map from prol graph (can be concatenated)
    size_t NFV = fmesh.template GetNN<NT_VERTEX>();
    Array<size_t> vmap (NFV); vmap = -1;
    size_t NCV = 0;
    for (auto k : Range(NFV)) {
      auto ri = pwprol.GetRowIndices(k);
      // we should be able to combine smoothing and discarding
      // if (ri.Size() > 1) { cout << "TODO: handle this case, dummy" << endl; continue; } // this is for double smooted prol
      // if (ri.Size() > 1) { throw Exception("comment this out"); }
      if (ri.Size() > 0) {
	vmap[k] = ri[0];
	NCV = max2(NCV, size_t(ri[0]+1));
      }
    }

    // For each fine vertex, sum up weights of edges that connect to the same CV
    //  (can be more than one edge, if the pw-prol is concatenated)
    // TODO: for many concatenated pws, does this dominate edges to other agglomerates??
    auto all_fedges = fmesh.template GetNodes<NT_EDGE>();
    Array<double> vw (NFV); vw = 0;
    auto neqcs = eqc_h.GetNEQCS();
    {
      INT<2, int> cv;
      auto doit = [&](auto the_edges) {
	for (const auto & edge : the_edges) {
	  if ( ((cv[0]=vmap[edge.v[0]]) != -1 ) &&
	       ((cv[1]=vmap[edge.v[1]]) != -1 ) &&
	       (cv[0]==cv[1]) ) {
	    auto com_wt = self.template GetWeight<NT_EDGE>(fmesh, edge);
	    vw[edge.v[0]] += com_wt;
	    vw[edge.v[1]] += com_wt;
	  }
	}
      };
      for (auto eqc : Range(neqcs)) {
	if (!eqc_h.IsMasterOfEQC(eqc)) continue;
	doit(fmesh.template GetENodes<NT_EDGE>(eqc));
	doit(fmesh.template GetCNodes<NT_EDGE>(eqc));
      }
    }
    fmesh.template AllreduceNodalData<NT_VERTEX>(vw, [](auto & tab){return move(sum_table(tab)); }, false);

    /** Find Graph for Prolongation **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    {
      Array<INT<2,double>> trow;
      Array<INT<2,double>> tcv;
      Array<size_t> fin_row;
      for (auto V:Range(NFV)) {
	auto CV = vmap[V];
	if ( is_invalid(CV) ) continue; // grounded -> TODO: do sth. here if we are free?
	if (vw[V] == 0.0) { // MUST be single
	  perow[V] = 1;
	  graph[V][0] = CV;
	  continue;
	}
	trow.SetSize(0);
	tcv.SetSize(0);
	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
	auto ovs = fecon.GetRowIndices(V);
	auto eis = fecon.GetRowValues(V);
	size_t pos;
	for (auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if (is_invalid(cov) || cov==CV) continue;
	  auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	  if (eqc_h.IsLEQ(EQ, oeq)) {
	    auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[eis[j]]);
	    if ( (pos = tcv.Pos(cov)) == size_t(-1)) {
	      trow.Append(INT<2,double>(cov, wt));
	      tcv.Append(cov);
	    }
	    else {
	      trow[pos][1] += wt;
	    }
	  }
	}
	QuickSort(trow, [](const auto & a, const auto & b) {
	    if (a[0]==b[0]) return false;
	    return a[1]>b[1];
	  });
	double cw_sum = (is_valid(CV)) ? vw[V] : 0.0;
	fin_row.SetSize(0);
	if (is_valid(CV)) fin_row.Append(CV); //collapsed vertex
	size_t max_adds = (is_valid(CV)) ? min2(MAX_PER_ROW-1, int(trow.Size())) : trow.Size();
	for (auto j:Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if (is_valid(CV)) {
	    // I don't think I actually need this: Vertex is collapsed to some non-weak (not necessarily "strong") edge
	    // therefore the relative weight comparison should eliminate all really weak connections
	    // if (fin_row.Size() && (trow[j][1] < MIN_PROL_WT)) break; 
	    if (trow[j][1] < MIN_PROL_FRAC*cw_sum) break;
	  }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	perow[V] = fin_row.Size();
	for (auto j:Range(fin_row.Size()))
	  graph[V][j] = fin_row[j];
      }
    }
    
    /** Create Prolongation **/
    auto sprol = make_shared<TSPM_TM>(perow, NCV);

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "Tobias", false); // ~2 MB LocalHeap
    Array<INT<2,size_t>> uve(30); uve.SetSize(0);
    Array<int> used_verts(20), used_edges(20);
    TM id; SetIdentity(id);
    for (int V:Range(NFV)) {
      auto CV = vmap[V];
      if (is_invalid(CV)) continue; // grounded -> TODO: do sth. here if we are free?
      if (perow[V] == 1) { // SINGLE or no good connections avail.
	sprol->GetRowIndices(V)[0] = CV;
	sprol->GetRowValues(V)[0] = pwprol.GetRowValues(V)[0];
      }
      else { // SMOOTH
	HeapReset hr(lh);
	// Find which fine vertices I can include
	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
	auto graph_row = graph[V];
	auto all_ov = fecon.GetRowIndices(V);
	auto all_oe = fecon.GetRowValues(V);
	uve.SetSize(0);
	for (auto j:Range(all_ov.Size())) {
	  auto ov = all_ov[j];
	  auto cov = vmap[ov];
	  if (is_valid(cov)) {
	    if (graph_row.Contains(cov)) {
	      auto eq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	      if (eqc_h.IsLEQ(EQ, eq)) {
		uve.Append(INT<2>(ov,all_oe[j]));
	      } } } }
	uve.Append(INT<2>(V,-1));
	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];}); // WHY??
	used_verts.SetSize(uve.Size()); used_edges.SetSize(uve.Size());
	for (auto k:Range(uve.Size()))
	  { used_verts[k] = uve[k][0]; used_edges[k] = uve[k][1]; }
	
	auto posV = find_in_sorted_array(int(V), used_verts);
      	size_t unv = used_verts.Size(); // # of vertices used
	FlatMatrix<TM> mat (1,unv,lh); mat(0, posV) = 0;
	FlatMatrix<TM> block (2,2,lh);
	for (auto l:Range(unv)) {
	  if (l==posV) continue;
	  self.CalcRMBlock (fmesh, all_fedges[used_edges[l]], block);
	  int brow = (V < used_verts[l]) ? 0 : 1;
	  mat(0,l) = block(brow,1-brow); // off-diag entry
	  mat(0,posV) += block(brow,brow); // diag-entry
	}

	TM diag;
	double tr = 1;
	if constexpr(mat_traits<TM>::HEIGHT == 1) {
	    diag = mat(0, posV);
	  }
	else {
	  diag = mat(0, posV);
	  tr = 0; Iterate<mat_traits<TM>::HEIGHT>([&](auto i) { tr += diag(i.value,i.value); });
	  tr /= mat_traits<TM>::HEIGHT;
	  diag /= tr;
	  // if (sing_diags) {
	  //   self.RegDiag(diag);
	  // }
	}
	CalcInverse(diag);
	
	auto sp_ri = sprol->GetRowIndices(V); sp_ri = graph_row;
	auto sp_rv = sprol->GetRowValues(V); sp_rv = 0;
	double fac = omega/tr;
	for (auto l : Range(unv)) {
	  int vl = used_verts[l];
	  auto pw_rv = pwprol.GetRowValues(vl);
	  int cvl = vmap[vl];
	  auto pos = find_in_sorted_array(cvl, sp_ri);
	  if (l==posV)
	    { sp_rv[pos] += pw_rv[0]; }
	  sp_rv[pos] -= fac * (diag * mat(0,l)) * pw_rv[0];

	}
      }
    }

    // cout << "SPROL MAT: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    pmap->SetProl(sprol);
  } // VertexBasedAMGFactory::SmoothProlongation
  

} // namespace amg

#endif // FILE_AMG_FACTORY_IMPL_HPP
