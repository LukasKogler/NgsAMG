#ifndef FILE_AMG_FACTORY_IMPL_HPP
#define FILE_AMG_FACTORY_IMPL_HPP

namespace amg
{

  /** Utility **/

  INLINE Timer & tuco_hack () { static Timer t("UpdateCoarsenOpts"); return t; }
  template<class TOPTS, class TMAP, class TSCO>
  void UpdateCoarsenOpts (shared_ptr<TOPTS> copts, shared_ptr<TMAP> map, TSCO sco)
  {
    /** Coarse weights are minimum of this computed from the coarse mesh and 
	weights of all fine nodes that map to the same coarse node.
	This under-estiates weights somethimes, but that should not be an issue. **/
    RegionTimer rt(tuco_hack());

    auto cofv = copts->free_verts;
    if (cofv != nullptr) {
      auto & ffv(*cofv);
      if constexpr(is_base_of<BaseCoarseMap, TMAP>::value) {
	auto set_wts_min = [] (const auto MIN_WT, auto map, auto & fcw, auto & ccw) LAMBDA_INLINE {
	  for (auto k : Range(map.Size())) {
	    auto mk = map[k];
	    if (mk != decltype(mk)(-1))
	      { ccw[mk] = min2(ccw[mk], fcw[k]); }
	  }
	};
	auto & fmesh = static_cast<BlockTM&>(*map->GetMesh());
	auto & cmesh = static_cast<BlockTM&>(*map->GetMappedMesh());
	auto fvcw = move(copts->vcw);
	auto fecw = move(copts->ecw);
	sco(copts, map->GetMappedMesh());
 	/** vertex-wts **/
	auto & cvcw = copts->vcw;
	set_wts_min (copts->min_vcw, map->template GetMap<NT_VERTEX>(), fvcw, cvcw);
	cmesh.template AllreduceNodalData<NT_VERTEX>(cvcw, [&](auto & in) LAMBDA_INLINE { return min_table(in); }, false);
	/** edge-wts **/
	auto & cecw = copts->ecw;
	set_wts_min (copts->min_ecw, map->template GetMap<NT_EDGE>(), fecw, cecw);
	cmesh.template AllreduceNodalData<NT_EDGE>(cecw, [&](auto & in) LAMBDA_INLINE { return min_table(in); }, false);
	/** free verts update **/
	auto comm = fmesh.GetEQCHierarchy()->GetCommunicator();
	auto vmap = map->template GetMap<NT_VERTEX>();
	auto cfv = make_shared<BitArray> (cmesh.template GetNN<NT_VERTEX>()); cfv->Set();
	int ndir = 0;
	for (auto k : Range(fmesh.template GetNN<NT_VERTEX>()))
	  if ( (!ffv.Test(k)) && (vmap[k] != -1))
	    { cfv->Clear(vmap[k]); ndir++; }
	ndir = comm.AllReduce(ndir, MPI_SUM);
	if (ndir == 0)
	  { copts->free_verts = nullptr; }
	else
	  { copts->free_verts = cfv; }
      }
      else {
	auto & fmesh = static_cast<BlockTM&>(*map->GetMesh());
	auto fvcw = move(copts->vcw); copts->vcw.SetSize(map->template GetMappedNN<NT_VERTEX>());
	map->template MapNodeData<NT_VERTEX, double>(fvcw, CUMULATED, &copts->vcw);
	auto fecw = move(copts->ecw); copts->ecw.SetSize(map->template GetMappedNN<NT_EDGE>());
	map->template MapNodeData<NT_EDGE, double>(fecw, CUMULATED, &copts->ecw);
	Array<int> ff (fmesh.template GetNN<NT_VERTEX>());
	Array<int> cf (map->template GetMappedNN<NT_VERTEX>());
	for (auto k : Range(ff))
	  { ff[k] = ffv.Test(k) ? 1 : 0; }
	map->template MapNodeData<NT_VERTEX, int> (ff, CUMULATED, &cf);
	if ( auto CM = static_pointer_cast<BlockTM>(map->GetMappedMesh()) ) { // not contracted out
	  const auto & cmesh = *CM;
	  auto ccomm = CM->GetEQCHierarchy()->GetCommunicator();
	  auto ncv = cmesh.template GetNN<NT_VERTEX>();
	  auto cfv = make_shared<BitArray>(ncv);
	  int ndir = 0;
	  for (auto k : Range(cf))
	    if (cf[k] == 0) { cfv->Clear(k); ndir++; }
	    else { cfv->SetBit(k); }
	  ndir = ccomm.AllReduce(ndir, MPI_SUM);
	  if (ndir == 0)
	    { copts->free_verts = nullptr; }
	  else
	    { copts->free_verts = cfv; }
	}
	else
	  { copts->free_verts = nullptr; }
      }
    }
    else
      { copts->free_verts = nullptr; }
  } // UpdateCoarsenOpts


  /** --- Options --- **/

  template<class TMESH, class TM>
  struct AMGFactory<TMESH, TM> :: Options
  {
    /** Level-control **/
    size_t max_n_levels = 10;                   // maximun number of multigrid levels (counts first level, so at least 2)
    size_t max_meas = 50;                       // maximal maesure of coarsest mesh
    bool combined_rsu = false;
    
    /** choice of coarsening algorithm **/
    enum CRS_ALG : char { ECOL,                 // edge collapsing
			  AGG };                // aggregaion
    CRS_ALG crs_alg = ECOL;
    int ecol_after_nlev = 1000;                 // switch to edge collapse after this many levels

    /** coarsening rate (mostly relevant for edge collapsing) **/
    bool enable_dyn_aaf = true;                 // dynamic coarsening rate
    double aaf = 0.1;                           // chain edge-collapse maps until mesh is decreased by factor aaf
    double first_aaf = 0.05;                    // (smaller) factor for first level. -1 for dont use
    double aaf_scale = 1;                       // scale aaf, e.g if 2:   first_aaf, aaf, 2*aaf, 4*aaf, .. (or aaf, 2*aaf, ...)

    /** agg opts **/
    int n_levels_d2_agg = 1;                    // do this many levels MIS(2)-like aggregates (afterwards MIS(1)-like)
    bool agg_wt_geom = true;                    // use geometric instead of harmonic mean when determining strength of connection
    bool agg_robust = true;                     // use more expensive, but also more robust edge weights

    /** Discard  **/
    bool enable_disc = true;
    double disc_thresh = 0.8;                   // try to discard verts when coarsening slows down to this (or slower)
    int disc_max_bs = 5;

    /** Contract (Re-Distribute) **/
    bool enable_ctr = true;
    /** WHEN to contract **/
    double ctraf = 0.05;                        // contract after reducing measure by this factor
    double first_ctraf = 0.025;                 // see first_aaf
    double ctraf_scale = 1;                     // see aaf_scale
    double ctr_crs_thresh = 0.9;                // if coarsening slows down more than this, redistribute
    double ctr_loc_thresh = 0.5;                // if less than this fraction of vertices are purely local, redistribute
    /** HOW AGGRESSIVELY to contract **/
    double ctr_pfac = 0.25;                     // per default, reduce active NP by this factor (ctr_pfac / ctraf should be << 1 !)
    /** additional constraints for contract **/
    size_t ctr_min_nv_th = 500;                 // re-distribute when there are less than this many vertices per proc left
    size_t ctr_min_nv_gl = 500;                 // try to re-distribute such that at least this many NV per proc remain
    size_t ctr_seq_nv = 1000;                   // always re-distribute to sequential once NV reaches this threshhold
    double ctr_loc_gl = 0.8;                    // always try to redistribute such that at least this fraction will be local

    /** Smoothed Prolongation **/
    bool enable_sm = true;                      // enable prolongation-smoothing
    bool force_osm = false;                     // force old smoothed prol
    bool realmsm = false;                          // real matrix for smoothed prol

    /** Build a new mesh from a coarse level matrix**/
    bool enable_rbm = false;                    // probably only necessary on coarse levels
    // bool mced_sp = true;
    // bool sp_neib_boost = true;                  
    // bool sp_tk = true;
    // Array<int> spk_ks;
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

    set_enum_opt(opts.crs_alg, "crs_alg", {"ecol", "agg" });
    set_num(opts.ecol_after_nlev, "ecol_after_nlev");
    set_num(opts.n_levels_d2_agg, "n_levels_d2_agg");
    set_num(opts.agg_wt_geom, "agg_wt_geom");
    set_num(opts.agg_robust, "agg_robust");

    set_bool(opts.enable_dyn_aaf, "dyn_aaf");
    set_num(opts.aaf, "aaf");
    set_num(opts.first_aaf, "first_aaf");
    set_num(opts.aaf_scale, "aaf_scale");
    
    set_num(opts.ctr_crs_thresh, "crs_thresh");
    set_num(opts.disc_max_bs, "disc_max_bs");

    set_bool(opts.enable_disc, "enable_disc");

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
    set_bool(opts.force_osm, "force_old_sp");
    set_bool(opts.realmsm, "sp_real_mat");

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
    double min_ecw2 = 0.03;
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
    set_num(opts.min_ecw2, "et2");
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
    INT<3> level;
    shared_ptr<TMESH> curr_mesh;
    shared_ptr<BaseDOFMapStep> dof_step;
    shared_ptr<ParallelDofs> curr_pds;
    double crs_meas_fac, frac_loc;
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
    
    logger = make_shared<Logger>(options->log_level);

    auto fmat = mats[0];

    /** rank 0 (and maybe sometimes others??) can also not have an embed_step, while others DO.
	ParallelDofs - constructor has to be called by every member of the communicator! **/
    int have_embed = fmat->GetParallelDofs()->GetCommunicator().AllReduce((embed_step == nullptr) ? 0 : 1, MPI_SUM);
    shared_ptr<ParallelDofs> fpds = (have_embed == 0) ? fmat->GetParallelDofs() : BuildParallelDofs(finest_mesh);

    State state;
    Capsule f_cap({ 0, move(finest_mesh), fpds, fmat });
    auto coarse_mats = RSU(f_cap, state, dof_map);
    
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
  double AMGFactory<TMESH, TM> :: FindCTRFac (shared_ptr<TMESH> mesh)
  {
    /** Find out how rapidly we should redistribute **/
    const auto & O(*options);
    const auto & M(*mesh);
    const auto& eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();

    auto NV = M.template GetNNGlobal<NT_VERTEX>();

    /** Already "sequential" **/
    if (comm.Size() <= 2)
      { return 1; }

    /** Sequential threshold reached **/
    if (NV < options->ctr_seq_nv)
      { return -1; }

    /** default redist-factor **/
    double ctr_factor = O.ctr_pfac;
    
    /** ensure enough vertices per proc **/
    ctr_factor = min2(ctr_factor, double(NV) / O.ctr_min_nv_gl / comm.Size());

    /** try to heuristically ensure that enough vertices are local **/
    // size_t nv_loc = M.template GetNN<NT_VERTEX>() > 0 ? M.template GetENN<NT_VERTEX>(0) : 0;
    // double frac_loc = comm.AllReduce(double(nv_loc), MPI_SUM) / NV;
    double frac_loc = ComputeLocFrac(M);
    if (frac_loc < options->ctr_loc_thresh) {
      size_t NP = comm.Size();
      size_t NF = comm.AllReduce (eqc_h.GetDistantProcs().Size(), MPI_SUM) / 2; // every face counted twice
      double F = frac_loc;
      double FGOAL = O.ctr_loc_gl; // we want to achieve this large of a frac of local verts
      double FF = (1-frac_loc) / NF; // avg frac of a face
      double loc_fac = 1;
      if (F + NP/2 * FF > FGOAL) // merge 2 -> one face per group becomes local
	{ loc_fac = 0.5; }
      else if (F + NP/4 * 5 * FF > FGOAL) // merge 4 -> probably 4(2x2) to 6(tet) faces per group
	{ loc_fac = 0.25; }
      else // merge 8, in 3d probably 12 edges (2x2x2) per group. probably never want to do more than that
	{ loc_fac = 0.125; }
      ctr_factor = min2(ctr_factor, loc_fac);
    }

    /** ALWAYS at least factor 2 **/
    ctr_factor = min2(0.5, ctr_factor);

    return ctr_factor;
  } // AMGFactory::FindCTRFac


  template<class TMESH, class TM>
  shared_ptr<BaseDOFMapStep> AMGFactory<TMESH, TM> :: STEP_AGG (const Capsule & f_cap, State & state, Capsule & c_cap)
  {
    const auto & O (*options);

    shared_ptr<BaseDOFMapStep> tot_step = nullptr;

    shared_ptr<TMESH> fmesh = f_cap.mesh, cmesh = move(f_cap.mesh);
    shared_ptr<ParallelDofs> fpds = move(f_cap.pardofs), cpds = nullptr;
    shared_ptr<BaseSparseMatrix> fmat = f_cap.mat, cmat = move(f_cap.mat);
    shared_ptr<BaseSparseMatrix> emb_fmat = fmat, disc_fmat = fmat;
    
    state.level = { f_cap.level, 0, 0 }; // coarse / sub-coarse / ctr 
    state.curr_mesh = fmesh;
    state.curr_pds = fpds;
    state.dof_step = nullptr;

    auto comm = fmesh->GetEQCHierarchy()->GetCommunicator();
    size_t fmeas = ComputeMeshMeasure(*fmesh), cmeas = fmeas;

    double & crs_meas_fac = state.crs_meas_fac; crs_meas_fac = 1;
    double & frac_loc = state.frac_loc; frac_loc = ComputeLocFrac(*fmesh);

    shared_ptr<ProlMap<TSPM_TM>> disc_step = nullptr, crs_step = nullptr;
    shared_ptr<BaseDOFMapStep> ctr_dstep = nullptr;

    /** Maybe discard vertices **/
    if ( O.enable_disc && (f_cap.level != 0) ) {
      cout << IM(5) << "Try to eliminate hanging vertices!" << endl;
      auto dmap = make_shared<VDiscardMap<TMESH>> (cmesh, O.disc_max_bs);
      auto n_d_v = dmap->GetNDroppedVerts();
      auto any_n_d_v = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_d_v, MPI_SUM);
      bool map_ok = any_n_d_v != 0; // someone somewhere eliminated some verices
      if (map_ok) {
	// mapped mesh is only built here !
	auto elim_vs = dmap->GetMesh()->template GetNNGlobal<NT_VERTEX>() - dmap->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>();
	auto dv_frac = double(dmap->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>()) / dmap->GetMesh()->template GetNNGlobal<NT_VERTEX>();
	map_ok &= (dv_frac < 0.98);
      }
      if (map_ok) {
	auto ccmesh = static_pointer_cast<TMESH>(dmap->GetMappedMesh());
	cout << IM(5) << " discard went from " << ComputeMeshMeasure(*cmesh) << " to " << ComputeMeshMeasure(*ccmesh) << endl;
	shared_ptr<TSPM_TM> disc_prol;
	if (O.enable_sm)
	  { disc_prol = BuildSProl(dmap); }
	else
	  { disc_prol = BuildPWProl(dmap); }
	size_t fmeas = ComputeMeshMeasure(*fmesh), cmeas = ComputeMeshMeasure(*ccmesh);
	double cfac = double(cmeas) / fmeas;
	crs_meas_fac *= cfac;
      	fmesh = cmesh = ccmesh;
	cpds = BuildParallelDofs(cmesh);
	disc_step = make_shared<ProlMap<TSPM_TM>>(disc_prol, fpds, cpds);
	fpds = cpds;
      }
    }

    /** Do coarsening  **/
    {
      /** Coarsen Mesh **/
      bool do_dist2 = f_cap.level < O.n_levels_d2_agg;
      auto agg_map = BuildAggMap(cmesh, do_dist2);
      auto agg_fm = static_pointer_cast<TMESH>(cmesh);
      auto agg_cm = static_pointer_cast<TMESH>(agg_map->GetMappedMesh());
      size_t fmeas = ComputeMeshMeasure(*agg_fm), cmeas = ComputeMeshMeasure(*agg_cm);
      /** If not stuck - build Prolongation **/
      if (cmeas < fmeas) { // we might be stuck
	double cfac = double(cmeas) / fmeas;
	crs_meas_fac *= cfac;
	auto pwp = BuildPWProl(agg_map, cpds); // fine pardofs are correct
	state.curr_pds = cpds = BuildParallelDofs(agg_cm);
	state.curr_mesh = fmesh = cmesh = agg_cm;
	crs_step = make_shared<ProlMap<TSPM_TM>> (pwp, fpds, cpds);
	if (O.enable_sm) {
	  if (O.force_osm) // force old smoothed prolongation
	    { SmoothProlongation (crs_step, agg_fm); }
	  else if (O.realmsm) { // use real map for smoothed prol
	    if (embed_step != nullptr)
	      { emb_fmat = embed_step->AssembleMatrix(fmat); }
	    if (disc_step != nullptr)
	      { disc_fmat = disc_step->AssembleMatrix(emb_fmat); }
	    auto sprol_mat = static_pointer_cast<TSPM_TM>(emb_fmat);
	    SmoothProlongation_RealMat (crs_step, sprol_mat);
	  }
	  else // use new smoothed prol
	    { SmoothProlongationAgg(crs_step, agg_map); }
	}
	free_verts = nullptr;
	state.level[1]++;
      }
    }

    /** Maybe Re-Distribute  **/
    if ( DoContractStep(state) ) {
      fmesh = cmesh = state.curr_mesh;
      ctr_dstep = move(state.dof_step);
      fpds = ctr_dstep->GetParDofs(); cpds = ctr_dstep->GetMappedParDofs();
    }

    /** Concatenate maps.
	Steps are (any can be nullptr): EMB - DISC - CRS - CTR
	    -) I can definitely concatenate disc-crs.
	    -) I can PROBABLY conc emb-(disc-crs).
	    -) I can never conc [emb-(disc-crs)]-CTR
	Assemble coarse mat:
	    -) Assemble to after CRS. Possibly, I already computed the matrix after EMB or DISC step.
	       In that case, assemble from there to after CRS.
	    -) Assemble after CTR
    **/
    { // Concatenate maps
      Array<shared_ptr<BaseDOFMapStep>> sub_steps(3);
      sub_steps.SetSize0();
      /** DISC - CRS **/
      shared_ptr<ProlMap<TSPM_TM>> dc_step = nullptr;
      if (crs_step != nullptr) {
	if (disc_step != nullptr) {
	  if (disc_fmat != fmat) // disc already done
	    { cmat = crs_step->AssembleMatrix(disc_fmat); }
	  dc_step = static_pointer_cast<ProlMap<TSPM_TM>>(disc_step->Concatenate(crs_step));
	}
	else {
	  if (emb_fmat != fmat)
	    { cmat = crs_step->AssembleMatrix(emb_fmat); }
	  dc_step = crs_step;
	}
      }
      else if (disc_step != nullptr) {
	if (disc_fmat != fmat)
	  { cmat = disc_fmat; }
	dc_step = disc_step;
      }
      /** EMB - (DISC - CRS) **/
      if (embed_step != nullptr) {
	if (dc_step != nullptr) {
	  if ( (cmat == fmat) && (emb_fmat != fmat) )
	    { cmat = dc_step->AssembleMatrix(emb_fmat); }
	  auto edc_step = embed_step->Concatenate(dc_step);
	  if (edc_step == nullptr) { // EMB - (DISC-CRS)
	    sub_steps.Append(embed_step);
	    sub_steps.Append(dc_step);
	  }
	  else { // (EMB-DISC-CRS)
	    sub_steps.Append(edc_step);
	  }
	}
	else { // EMB - NULL
	  sub_steps.Append(embed_step);
	}
	embed_step = nullptr;
      }
      else if (dc_step != nullptr)
	{ sub_steps.Append(dc_step); }
	

      /** [EMB - (DISC - CRS)] - CTR **/
      if (ctr_dstep != nullptr) {
	if (cmat != fmat)
	  { cmat = ctr_dstep->AssembleMatrix(cmat); }
	sub_steps.Append(ctr_dstep);
      }

      // cout << " sub steps are " << endl << sub_steps << endl;
      // for (auto step : sub_steps)
	// cout << typeid(*step).name() << endl;
      // cout << endl;

      if (sub_steps.Size() == 1)
	{ tot_step = sub_steps[0]; }
      else if (sub_steps.Size() > 1)
	{ tot_step = make_shared<ConcDMS>(sub_steps); }
      if ( (cmat == fmat) && (tot_step != nullptr) )
	{ cmat = tot_step->AssembleMatrix(fmat); }
    } // Concatenate maps, assemble cmat

    if (tot_step != nullptr) {
      c_cap.level = f_cap.level + 1;
      c_cap.mat = cmat;
      c_cap.mesh = cmesh;
      c_cap.pardofs = tot_step->GetMappedParDofs();
    }
    else {
      c_cap.level = f_cap.level;
      c_cap.mat = f_cap.mat;
      c_cap.mesh = f_cap.mesh;
      c_cap.pardofs = f_cap.pardofs;
    }

    // cout << " TS " << tot_step << endl;
    // cout << " f level " << endl;
    // cout << "mesh " << f_cap.mesh << endl;
    // cout << "mat " << f_cap.mat << endl;
    // cout << " c level " << endl;
    // cout << "mesh " << c_cap.mesh << endl;
    // cout << "mat  " << c_cap.mat << endl;
    // if (c_cap.mat)
    //   { cout << " c mat " << c_cap.mat->Height() << endl; } // << *c_cap.mat << endl;
    
    return tot_step;
  } // AMGFactory::STEP_AGG


  template<class TMESH, class TM>
  shared_ptr<BaseDOFMapStep> AMGFactory<TMESH, TM> :: STEP_ECOL (const Capsule & f_cap, State & state, Capsule & c_cap)
  {
    const auto & O (*options);

    shared_ptr<BaseDOFMapStep> tot_step;
    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    shared_ptr<TMESH> fmesh, cmesh, sprol_mesh;
    shared_ptr<ParallelDofs> fpds, cpds;
    shared_ptr<BaseSparseMatrix> fmat, cmat, sprol_mat;
    shared_ptr<ProlMap<TSPM_TM>> disc_step;

    state.level = { f_cap.level, 0, 0 }; // coarse / sub-coarse / ctr 
    state.curr_mesh = fmesh;
    state.curr_pds = fpds;
    state.dof_step = nullptr;

    fmesh = f_cap.mesh; cmesh = move(f_cap.mesh);
    fpds = move(f_cap.pardofs); cpds = nullptr;
    fmat = f_cap.mat; cmat = move(f_cap.mat);

    shared_ptr<TSPM_TM> conc_pwp;
    size_t curr_meas = -1, goal_meas = -1;
    double& crs_meas_fac = state.crs_meas_fac; crs_meas_fac = 1;
    double& frac_loc = state.frac_loc; frac_loc = ComputeLocFrac(*fmesh);
    size_t dcs_cnt = 0;

    shared_ptr<typename HierarchicVWC<TMESH>::Options> coarsen_opts;

    auto do_coarsen_step = [&] () LAMBDA_INLINE {
      if (coarsen_opts == nullptr) {
	coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
	this->SetCoarseningOptions(*coarsen_opts, cmesh);
      }
      auto calg = make_shared<BlockVWC<TMESH>> (coarsen_opts);
      auto grid_step = calg->Coarsen(cmesh);
      auto _cmesh = static_pointer_cast<TMESH>(grid_step->GetMappedMesh());
      const auto & ceqc_h = *_cmesh->GetEQCHierarchy();
      auto ccomm = ceqc_h.GetCommunicator();

      auto crs_meas = ComputeMeshMeasure(*_cmesh);
      crs_meas_fac = crs_meas / (1.0 * curr_meas);

      // auto cnv_glob = _cmesh->template GetNNGlobal<NT_VERTEX>();
      // auto cnv_loc = (ceqc_h.GetNEQCS() > 0) ? _cmesh->template GetENN<NT_VERTEX>(0) : 0;
      // frac_loc = ccomm.AllReduce(double(cnv_loc), MPI_SUM) / cnv_glob;

      frac_loc = ComputeLocFrac(*_cmesh);

      if ( (dcs_cnt > 0) && (crs_meas < goal_meas) && ( curr_meas * crs_meas < sqr(goal_meas) ) )
	{ return true; }
      if (crs_meas == curr_meas) // we always have to break here, even without redist!
	{ return true; }
      if ( O.enable_ctr && (ccomm.Size() > 2) && (crs_meas_fac > O.ctr_crs_thresh) ) // coarsening slows down - recover with re-dist?
	{ return true; }

      // the step has been accepted
      cmesh = _cmesh;
      free_verts = nullptr; // relevant on finest level
      if (coarsen_opts != nullptr) // only need to update coarsen opts if the alg uses them
	{ UpdateCoarsenOpts(coarsen_opts, grid_step, [&](auto a, auto b) { SetCoarseningOptions(*a, static_pointer_cast<TMESH>(b)); } ); }
      auto pwp = BuildPWProl(grid_step, cpds); // fine pardofs are correct
      conc_pwp = (conc_pwp == nullptr) ? pwp : MatMultAB(*conc_pwp, *pwp);
      cpds = BuildParallelDofs(cmesh);
      curr_meas = ComputeMeshMeasure(*cmesh);

      state.curr_mesh = cmesh;
      state.curr_pds = cpds;
      state.level[1]++;

      if ( O.enable_ctr && (ccomm.Size() > 2) && (frac_loc < O.ctr_loc_thresh) )
	{ return true; }

      dcs_cnt++;

      return false; // not stuck
    }; // do_coarsen_step

    /** the mesh/mat we use for prol-smoothing
	[ this can be different from "fmesh" when have to start out with a different map than coarse-map, e.g. contract
	or when we do a discard_step in the beginning ] **/
    sprol_mesh = fmesh; sprol_mat = fmat;

    auto comm = fmesh->GetEQCHierarchy()->GetCommunicator();
    size_t fmeas = ComputeMeshMeasure(*fmesh), cmeas = fmeas;

    /** Find out how much we want to coarsen the mesh before assembling the next matrix **/
    { // goal_meas
      curr_meas = ComputeMeshMeasure(*fmesh);
      double af = ( (f_cap.level == 0) && (O.first_aaf != -1) ) ?
	O.first_aaf : ( pow(O.aaf_scale, f_cap.level - ( (O.first_aaf == -1) ? 0 : 1) ) * O.aaf );
      goal_meas = max( size_t(min(af, 0.9) * curr_meas), max(O.max_meas, size_t(1)));
      size_t curr_ne = cmesh->template GetNNGlobal<NT_EDGE>();
      size_t curr_nv = cmesh->template GetNNGlobal<NT_VERTEX>();
      double edge_per_v = 2 * double(curr_ne) / double(curr_nv);
      /** We want to find the right agglomerate size, as a heutristic take 1/(1+avg number of strong neighbours) **/
      if (O.enable_dyn_aaf) {
	if (coarsen_opts == nullptr) {
	  coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
	  this->SetCoarseningOptions(*coarsen_opts, cmesh);
	}
	const double MIN_ECW = coarsen_opts->min_ecw;
	const auto& ecw = coarsen_opts->ecw;
	size_t n_s_e = 0;
	cmesh->template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_s_e++; } }, true);
	n_s_e = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_s_e, MPI_SUM);
	double s_e_per_v = 2 * double(n_s_e) / double(cmesh->template GetNNGlobal<NT_VERTEX>());
	double dynamic_goal_fac = 1.0 / ( 1 + s_e_per_v );
	goal_meas = max( size_t(min2(0.5, dynamic_goal_fac) * curr_meas), max(O.max_meas, size_t(1)));
      } // dyn_aaf
    } // goal_meas

    cout << IM(4) << "next goal for coarsening is " << curr_meas << " -> " << goal_meas << ", factor " << double(goal_meas)/curr_meas << endl;

    /**
       The final map that takes us to the next level should be:
           EMB - DISC - CRS - CTR
       All maps are "optional".
         - EMB is only on the finest level
	 - DISC is usually not on the finest level (and only if enable_disc == true)
	 - CRS should almost always be present, some exceptions:
	        - disc_map already takes us to our coarsening goal
		- coarsening locks for some reason (THIS CASE IS PROBABLY NOT HANDLED GRACEFULLY!!)
	 - CTR usually not every level (and only if enable_ctr == true)
	!! IMPORTANT !!
	 CRS and CTR are done interleaved in multiple steps, so CRS - CTR - CRS - CRS - CTR is possible.
	 These are then, however, transformed to "CRS - CTR" (see below) 
    **/

    /** DISC_STEP **/
    if ( O.enable_disc && (f_cap.level != 0) ) {
      cout << IM(5) << "Try to eliminate hanging vertices!" << endl;
      auto dmap = make_shared<VDiscardMap<TMESH>> (cmesh, O.disc_max_bs);
      auto n_d_v = dmap->GetNDroppedVerts();
      auto any_n_d_v = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_d_v, MPI_SUM);
      bool map_ok = any_n_d_v != 0; // someone somewhere eliminated some verices
      if (map_ok) {
	auto elim_vs = dmap->GetMesh()->template GetNNGlobal<NT_VERTEX>() - dmap->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>();
	auto dv_frac = double(dmap->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>()) / dmap->GetMesh()->template GetNNGlobal<NT_VERTEX>();
	map_ok &= (dv_frac < 0.98);
      }
      if (map_ok) { // use the map
	auto ccmesh = static_pointer_cast<TMESH>(dmap->GetMappedMesh());
	cout << IM(5) << " discard went from " << ComputeMeshMeasure(*cmesh) << " to " << ComputeMeshMeasure(*ccmesh) << endl;
	crs_meas_fac = ComputeMeshMeasure(*ccmesh) / (1.0 * curr_meas);
	frac_loc = ccmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(double(cmesh->template GetNN<NT_VERTEX>() > 0 ?ccmesh->template GetENN<NT_VERTEX>(0) : 0), MPI_SUM) / ccmesh->template GetNNGlobal<NT_VERTEX>();
	// TODO: kinda hacky, but this HAS to be after ant_n_d_v != 0, so the coarse mesh is really already built
	shared_ptr<TSPM_TM> disc_prol;
	if (O.enable_sm) {
	  auto pwdp = BuildPWProl(dmap);
	  disc_prol = BuildSProl(dmap);
	}
	else {
	  disc_prol = BuildPWProl(dmap);
	}
	sprol_mesh = fmesh = cmesh = ccmesh;
	cpds = BuildParallelDofs(cmesh);
	curr_meas = ComputeMeshMeasure(*cmesh);
	disc_step = make_shared<ProlMap<TSPM_TM>>(disc_prol, fpds, cpds);
	fpds = cpds;
	state.curr_mesh = fmesh;
	state.curr_pds = fpds;
	if (coarsen_opts != nullptr)
	  { UpdateCoarsenOpts(coarsen_opts, dmap, [&](auto a, auto b) { SetCoarseningOptions(*a, static_cast<TMESH>(b)); } ); }
      } // (disc_)map_ok
    } // enable_disc


    /** Interleaved CRS- and CTR- maps until we have coarsened the mesh appropiately **/
    while (curr_meas > goal_meas) { // outer loop - when inner loop is stuck (or done), try to redistribute
      /** inner loop - do coarsening until stuck **/
      bool stuck = false;
      dcs_cnt = 0;
      while ( (curr_meas > goal_meas) && (!stuck) )
	{ stuck = do_coarsen_step(); }
      /** we PROBABLY have a coarse map **/
      if(conc_pwp != nullptr) {
	sub_steps.Append(make_shared<ProlMap<TSPM_TM>> (conc_pwp, fpds, cpds));
	fpds = cpds; conc_pwp = nullptr; fmesh = cmesh;
      }
      /** try contract **/
      bool ctr_worked = DoContractStep(state);
      if ( ctr_worked ) {
	fmesh = cmesh = state.curr_mesh;
	fpds = state.dof_step->GetMappedParDofs(); cpds = state.dof_step->GetMappedParDofs();
	sub_steps.Append(move(state.dof_step));
	state.level[2]++;
      }
      /** if contract did not work, or if we are contracted out, break out of loop **/
      if ( (!ctr_worked) || (cmesh == nullptr))
	  { break; }
    }

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
    if (sub_steps.Size() > 1) // TODO: re-dist as first step does not currently work! (why?? I don't remember)
      { // swap C/P
	if ( auto ctr_last = dynamic_pointer_cast<TCTR>(sub_steps.Last()) ) {
	  auto fpd = ctr_last->GetParDofs();
	  // if there is no prol after it, all members of group enter with false
	  if (ctr_last->DoSwap(false)) { // master must have entered with true - do swap!
	    // auto fpd = ctr_last->GetParDofs();
	    sub_steps.Append(nullptr);
	    sub_steps.Last() = sub_steps[sub_steps.Size()-2];
	    sub_steps[sub_steps.Size()-2] = ctr_last->SwapWithProl(nullptr);
	  }
	}
	for (int step_nr = sub_steps.Size() - 1; step_nr > 0; step_nr--) {
	  // if [step_nr - 1, step_nr] are [prol, ctr], swap them
	  auto step_L = sub_steps[step_nr-1];
	  auto step_R = sub_steps[step_nr];
	  if ( auto ctr_L = dynamic_pointer_cast<TCTR>(step_L) ) {
	    if ( auto crs_R = dynamic_pointer_cast<TCRS>(step_R) ) { // C -- P -> swap to P -- C
	      auto fpd = ctr_L->GetParDofs();
	      ctr_L->DoSwap(true);
	      sub_steps[step_nr-1] = ctr_L->SwapWithProl(crs_R);
	      sub_steps[step_nr]   = ctr_L;
	    }
	    else // TODO: C -- C -> concatenate to single C ?
	      { ; }
	  }
	  else if ( auto crs_L = dynamic_pointer_cast<TCRS>(step_L) ) {
	    if ( auto crs_R = dynamic_pointer_cast<TCRS>(step_R) ) { // P -- P, concatenate to single P (actually: P -- nullptr)
	      // auto conc_P = MatMultAB(*crs_L->GetProl(), *crs_R->GetProl());
	      // auto conc_map = make_shared<TCRS>(conc_P, crs_L->GetParDofs(), crs_R->GetMappedParDofs());
	      auto conc_map = crs_L->Concatenate(crs_R);
	      sub_steps[step_nr-1] = conc_map;
	      sub_steps[step_nr] = nullptr;
	    }
	    else // P -- C, nothing to do, leave sub_step entries as-is
	      { ; }
	  }
	}
	/**
	   remove nullptr-maps from sub_steps
	   now we have P -- C -- C -- C ... (until i can concatenate C -- C to C, then it should really just be just P - C)
	**/
	int c = 0;
	for (int j = 0; j < sub_steps.Size(); j++)
	  if (sub_steps[j] != nullptr)
	    { sub_steps[c++] = sub_steps[j]; }
	sub_steps.SetSize(c);
      } // swap C/P

    /** smooth prolongation **/
    if ( (O.enable_sm) && (sub_steps.Size()) ) { // possibly only disc_prol, not in sub_steps
      // cout << " step " << sub_steps[0] << endl;
      // cout << typeid(*sub_steps[0]).name() << endl;
      auto pstep = dynamic_pointer_cast<TCRS>(sub_steps[0]);
      if ( (pstep == nullptr) && (sub_steps.Size() > 1) ) // not sure if sprol_mesh is correct then ...
	{ pstep = dynamic_pointer_cast<TCRS>(sub_steps[1]); }
      if (pstep == nullptr)
	{ throw Exception("Something must be broken!!"); }
      if ( (O.enable_rbm) && (state.level[0] != 0) ) {
	if (disc_step != nullptr) // TODO: this is re-assembled laster on, can we re-use this somehow?
	  { sprol_mat = disc_step->AssembleMatrix(sprol_mat); }
	sprol_mesh = O.rebuild_mesh(sprol_mesh, sprol_mat, pstep->GetParDofs());
      }
      SmoothProlongation(pstep, sprol_mesh);
    } // sm-prol


    /** Add DISC step --> should be DISC-CRS-CTR, (DISC-CRS should be concatenated) **/
    if (disc_step != nullptr) {
      if (sub_steps.Size()) {
	auto conc_step = disc_step->Concatenate(sub_steps[0]);
	if (conc_step != nullptr)
	  { sub_steps[0] = conc_step; }
	else {
	  sub_steps.Append(nullptr);
	  for (auto k : Range(size_t(1), sub_steps.Size()))
	    { sub_steps[sub_steps.Size() - k] = sub_steps[sub_steps.Size() - k - 1]; }
	  sub_steps[0] = disc_step;
	}
      }
      else
	{ sub_steps.Append(disc_step); }
    }

    /** Add EMB step --> EMB-DISC-CRS-CTR (EMB-DISC-CRS should be concatenated) **/
    if (embed_step != nullptr) {
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
    }

    /** assemble coarse matrix and add DOFstep to map **/
    if (sub_steps.Size() > 1)
      { tot_step = make_shared<ConcDMS>(sub_steps); }
    else if (sub_steps.Size() == 1)
      { tot_step = sub_steps[0]; }
    
    if (tot_step != nullptr) {
      cmat = tot_step->AssembleMatrix(fmat);
      c_cap.level = f_cap.level + 1;
      c_cap.mat = cmat;
      c_cap.mesh = cmesh;
      c_cap.pardofs = tot_step->GetMappedParDofs();
    }
    else {
      c_cap.level = f_cap.level;
      c_cap.mat = f_cap.mat;
      c_cap.mesh = f_cap.mesh;
      c_cap.pardofs = f_cap.pardofs;
    }

    return tot_step;
  } // AMGFactory::STEP_ECOL


  template<class TMESH, class TM>
  shared_ptr<BaseDOFMapStep> AMGFactory<TMESH, TM> :: STEP_COMB (const Capsule & f_cap, State & state, Capsule & c_cap)
  {
    throw Exception("TODO (should just be a bunch of copy/paste)");
    return nullptr;
  }


  template<class TMESH, class TM>
  Array<shared_ptr<BaseSparseMatrix>> AMGFactory<TMESH, TM> :: RSU (Capsule & f_cap, State & state, shared_ptr<DOFMap> dof_map)
  {
    const auto & O(*options);

    if (f_cap.level == 0) { // initialize book-keeping
      state.first_ctr_used = state.first_rbm_used = false;
      state.last_ctr_nv = f_cap.mesh->template GetNNGlobal<NT_VERTEX>();
      state.last_meas_rbm = ComputeMeshMeasure(*f_cap.mesh);
      state.coll_cross = false;
    }

    logger->LogLevel (f_cap);

    shared_ptr<BaseDOFMapStep> step;
    Capsule c_cap;
    c_cap.level = f_cap.level + 1;
    c_cap.mesh = nullptr;
    c_cap.pardofs = nullptr;
    c_cap.mat = nullptr;

    if (O.combined_rsu)
      { step = STEP_COMB(f_cap, state, c_cap); }
    else if ( (O.crs_alg == Options::CRS_ALG::ECOL) || (f_cap.level > O.ecol_after_nlev) )
      { step = STEP_ECOL(f_cap, state, c_cap); }
    else
      { step = STEP_AGG(f_cap, state, c_cap); }

    if ( (step == nullptr) || (c_cap.mesh == f_cap.mesh) || (c_cap.mat == f_cap.mat) ) // coarsening is probably stuck
      { return Array<shared_ptr<BaseSparseMatrix>> (0); }
    else
      { dof_map->AddStep(step); }
      
    /** potentially clean up some stuff before recursive call **/
    f_cap.mesh = nullptr;
    // f_cap.mat = nullptr;
    f_cap.pardofs = nullptr;
    
    /** Recursive call (or return) **/
    if ( (c_cap.mesh == nullptr) || (c_cap.mat == nullptr) ) // dropped out (redundand "||" ?)
      { return Array<shared_ptr<BaseSparseMatrix>>({ nullptr }); }
    else if ( (f_cap.level + 2 == O.max_n_levels) ||                // max n levels reached
    	      (O.max_meas >= ComputeMeshMeasure (*c_cap.mesh) ) ) { // max coarse size reached
      logger->LogLevel (c_cap);
      return Array<shared_ptr<BaseSparseMatrix>> ({c_cap.mat});
    }
    else { // more coarse levels
      auto cmats = RSU( c_cap, state, dof_map );
      cmats.Append(c_cap.mat);
      return cmats;
    }
  } // AMGFactory::RSU


  template<class TMESH, class TM>
  bool AMGFactory<TMESH, TM> :: DoContractStep (State & state)
  {
    static Timer t("DoContractStep");
    RegionTimer rt(t);

    const auto & O(*options);
    const auto & M = *state.curr_mesh;
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    auto NV = M.template GetNNGlobal<NT_VERTEX>();

    /** cannot redistribute if when is turned off or when we are already basically sequential **/
    if ( (!O.enable_ctr) || (comm.Size() <= 2) )
      { return false; }

    /** Find out how rapidly we should redistribute **/

    double ctr_factor = 1;

    double af = ( (!state.first_ctr_used) && (O.first_ctraf != -1) ) ?
      O.first_ctraf : ( pow(O.ctraf_scale, state.level[0] - ( (O.first_ctraf == -1) ? 0 : 1) ) * O.ctraf );
    size_t goal_nv = max( size_t(min(af, 0.9) * state.last_ctr_nv), max(O.ctr_seq_nv, size_t(1)));
    if ( (state.crs_meas_fac > O.ctr_crs_thresh) ||  // coarsening slows down
	 (NV < comm.Size() * O.ctr_min_nv_th) ||     // NV/NP too small
	 (goal_nv > NV) ||                           // static redistribute every now and then
	 (state.frac_loc < O.ctr_loc_thresh) )       // not enough local vertices
      { ctr_factor = FindCTRFac (state.curr_mesh); }

    if (ctr_factor == 1)
      { return false; }

    cout << IM(4) << "contract by factor " << ctr_factor << endl;

    auto ctr_map = BuildContractMap(ctr_factor, state.curr_mesh);

    if (free_verts != nullptr) {
      auto fmesh = state.curr_mesh;
      Array<int> ff (fmesh->template GetNN<NT_VERTEX>());
      Array<int> cf (ctr_map->template GetMappedNN<NT_VERTEX>());
      for (auto k : Range(ff))
	{ ff[k] = free_verts->Test(k) ? 1 : 0; }
      ctr_map->template MapNodeData<NT_VERTEX, int> (ff, CUMULATED, &cf);
      if ( auto CM = static_pointer_cast<TMESH>(ctr_map->GetMappedMesh()) ) {
	auto ncv = CM->template GetNN<NT_VERTEX>();
	free_verts = make_shared<BitArray>(ncv);
	for (auto k : Range(cf))
	  if (cf[k] == 0) { free_verts->Clear(k); }
	  else { free_verts->SetBit(k); }
      }
      else
	{ free_verts = nullptr; }
    }

    state.first_ctr_used = true;
    state.last_ctr_nv = NV;
    state.curr_mesh = static_pointer_cast<TMESH>(ctr_map->GetMappedMesh());
    state.dof_step = BuildDOFContractMap(ctr_map, state.curr_pds);

    return true;
  } // AMGFactory::DoContractStep


  template<class TMESH, class TM>
  shared_ptr<GridContractMap<TMESH>> AMGFactory<TMESH, TM> :: BuildContractMap (double factor, shared_ptr<TMESH> mesh) const
  {
    static Timer t("BuildContractMap"); RegionTimer rt(t);
    // at least 2 groups - dont send everything from 1 to 0 for no reason
    int n_groups = (factor == -1) ? 2 : max2(int(2), int(1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * factor)));
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);
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
  double VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: ComputeLocFrac (const TMESH & m) const
  {
    auto nvg = m.template GetNNGlobal<NT_VERTEX>();
    size_t nvloc = (m.GetEQCHierarchy()->GetNEQCS() > 1) ? m.template GetENN<NT_VERTEX>(0) : 0;
    auto nvlocg = m.GetEQCHierarchy()->GetCommunicator().AllReduce(nvloc, MPI_SUM);
    return double(nvlocg) / nvg;
  }

  template<class FACTORY_CLASS, class TMESH, class TM>
  shared_ptr<AgglomerateCoarseMap<TMESH>> VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildAggMap  (shared_ptr<TMESH> mesh, bool dist2) const
  {
    auto & O = static_cast<VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM>::Options&>(*options);
    typename Agglomerator<FACTORY_CLASS>::Options agg_opts;
    agg_opts.edge_thresh = O.min_ecw;
    O.min_ecw = O.min_ecw2;
    agg_opts.vert_thresh = O.min_vcw;
    agg_opts.cw_geom = O.agg_wt_geom;
    agg_opts.robust = O.agg_robust;
    agg_opts.dist2 = dist2;
    auto agglomerator = make_shared<Agglomerator<FACTORY_CLASS>>(mesh, free_verts, move(agg_opts));
    return agglomerator;
  }


  template<class FACTORY_CLASS, class TMESH, class TM> template<class TMAP>
  shared_ptr<typename VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM>::TSPM_TM>
  VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildPWProl_impl (shared_ptr<TMAP> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const auto & rcmap(*cmap);
    const TMESH & fmesh = static_cast<TMESH&>(*rcmap.GetMesh()); fmesh.CumulateData();
    const TMESH & cmesh = static_cast<TMESH&>(*rcmap.GetMappedMesh()); cmesh.CumulateData();

    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();

    // Alloc Matrix
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

    cmesh.template AllreduceNodalData<NT_VERTEX, int>(has_partner, [](auto & tab){ return move(sum_table(tab)); });

    for (auto vnr : Range(NV))
      { if (vmap[vnr] != -1) perow[vnr] = 1; }

    auto prol = make_shared<TSPM_TM>(perow, NCV);

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
	  // self.CalcPWPBlock (fmesh, cmesh, rcmap, vnr, cvnr, rv[0]);
	  self.CalcPWPBlock (fmesh, cmesh, vnr, cvnr, rv[0]);
	}
      }
    }
    
    // cout << "pwprol: " << endl;
    // print_tm_spmat(cout, *prol); cout << endl<< endl;

    return prol;
  } // VertexBasedAMGFactory::BuildPWProl


  template<class FACTORY_CLASS, class TMESH, class TM>
  shared_ptr<typename VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM>::TSPM_TM>
  VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildPWProl (shared_ptr<VDiscardMap<TMESH>> dmap) const
  {
    /** for every discarded DOF, pick the stronges connected, not discarded, DOF to prolongate from **/
    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const TMESH & fmesh = static_cast<TMESH&>(*dmap->GetMesh()); fmesh.CumulateData();
    const TMESH & cmesh = static_cast<TMESH&>(*dmap->GetMappedMesh()); cmesh.CumulateData();
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!

    // TODO: rework this so that we dont need to compute duplicates
    auto coarsen_opts = make_shared<typename BlockVWC<TMESH>::Options>();
    this->SetCoarseningOptions(*coarsen_opts, static_pointer_cast<TMESH>(dmap->GetMesh()));

    size_t NV  = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();

    auto vmap = dmap->template GetNodeMap<NT_VERTEX>();
    Array<int> perow (NV); perow = 1;


    auto prol = make_shared<TSPM_TM>(perow, NCV);

    auto & dropped_verts = *dmap->GetDroppedVerts();

    auto & ecw = coarsen_opts->ecw;

    for (auto k : Range(NV)) {
      if (!dropped_verts.Test(k)) {
	prol->GetRowIndices(k)[0] = vmap[k];
	SetIdentity(prol->GetRowValues(k)[0]);
      }
      else {
	auto neibs = fecon.GetRowIndices(k);
	auto nes = fecon.GetRowValues(k);
	// find the remaining neighbour with the largest connecting weight
	// note: we only drop vertices where ALL neighbours are in a geq EQC, so te result here is consistent
	int s_r_neib_num = -1;
	double srn_wt = 0;
	for (auto j : Range(neibs)) {
	  if (!dropped_verts.Test(neibs[j])) {
	    int eid = int(nes[j]);
	    if (ecw[eid] > srn_wt) {
	      s_r_neib_num = j;
	      srn_wt = ecw[eid];
	    }
	  }
	}
	prol->GetRowIndices(k)[0] = vmap[neibs[s_r_neib_num]];
	self.CalcPWPBlock (fmesh, cmesh, k, vmap[neibs[s_r_neib_num]], prol->GetRowValues(k)[0]);
	// prol->GetRowValues(k)[0] = 0;
      }
    }

    // cout << " discard prol: " << endl;
    // print_tm_spmat(cout, *prol); cout << endl<< endl;

    return prol;
  } // VertexBasedAMGFactory::BuildPWProl (discard)


  template<class FACTORY_CLASS, class TMESH, class TM>
  shared_ptr<typename VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM>::TSPM_TM>
  VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: BuildSProl (shared_ptr<VDiscardMap<TMESH>> dmap) const
  {
    /**
       
       !!! UNUSED, PROBABLY BUGGY!!!

       R .. remaining
       D .. dropped

       The prolongation matrix we are building is:

             I_R
       -A_DD^{-1} ADR

       This means the coarse matrix (when applying the prol to the replacement matrix) is the schur complement w.r.t to the remaining DOFs.
       
       Applied to the actual matrix is not the exact schur complement, especially if matrix and replacement matrix graphs
       have already diverged, for example after an earlier smoothed prolongation.

       However, it should be "close" (and will still reproduce kernel vectors), which is our motivation for doing it this way.
     **/

    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const TMESH & fmesh = static_cast<TMESH&>(*dmap->GetMesh()); fmesh.CumulateData();
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!

    // cout << " fecon: " << endl;
    // cout << fecon << endl;

    auto & dropped_verts = *dmap->GetDroppedVerts();
    auto & vert_blocks = *dmap->GetVertexBlocks();

    auto NV = fmesh.template GetNN<NT_VERTEX>();
    auto CNV = dmap->template GetMappedNN<NT_VERTEX>();
    auto vmap = dmap->template GetNodeMap<NT_VERTEX>();
    auto emap = dmap->template GetNodeMap<NT_EDGE>();

    /** 
	Create graph:
	- non-dropped verts only connect to their coarse vertex
	- for every vert_block, all vertices in it connect to all coarse vertices of neibs of any member (all have same neibs)
    **/
    TableCreator<int> cg (NV);
    for (; !cg.Done(); cg++ ) {
      for (auto k : Range(NV))
	if (!dropped_verts.Test(k))
	  { cg.Add(k, vmap[k]); }
      for (auto k : Range(vert_blocks.Size())) {
	auto block = vert_blocks[k];
	auto neibs = fecon.GetRowIndices(block[0]); // does not contain block[0], but does not matter
	for (auto n : neibs)
	  if (!dropped_verts.Test(n))
	    for (auto v : block)
	      { cg.Add(v, vmap[n]); }
      }
    }
    auto graph = cg.MoveTable();

    // cout << endl << " vert_blocks: " << endl;
    // cout << vert_blocks << endl << endl;

    // cout << " graph: " << endl;
    // cout << graph << endl << endl;

    Array<int> perow(NV);
    for (auto k : Range(perow))
      { perow[k] = graph[k].Size(); }

    auto sprol = make_shared<SparseMatrixTM<TM>> (perow, CNV);
    for (auto k : Range(NV)) // identity for non-dropped vertices
      if (!dropped_verts.Test(k)) {
	sprol->GetRowIndices(k)[0] = vmap[k];
	SetIdentity(sprol->GetRowValues(k)[0]);
      }

    LocalHeap lh(2000000, "Drummer", false); // ~2 MB LocalHeap
    
    Matrix<TM> edge_mat (2, 2);
    
    auto fedges = fmesh.template GetNodes<NT_EDGE>();
    
    for (auto block_nr : Range(vert_blocks.Size())) {

      HeapReset hr(lh);

      // assemble replacement-matrix for block x neibs ("neibs" include block)
      auto block = vert_blocks[block_nr];

      auto ri = fecon.GetRowIndices(block[0]); // ECM has no diagonal entries
      FlatArray<int> neibs(1 + ri.Size(), lh);
      auto m_pos = merge_pos_in_sorted_array(int(block[0]), ri);
      int c = 0;
      for (int k : Range(neibs))
	{ neibs[k] = (k == m_pos) ? block[0] : ri[c++]; }

      // cout << endl << endl << "block_nr: " << block_nr << endl;
      // cout << " block: "; prow2(block); cout << endl;
      // cout << " neibsL "; prow2(neibs); cout << endl;

      FlatMatrix<TM> repl_row (block.Size(), neibs.Size(), lh); repl_row = 0;
      for (auto k : Range(block)) {
	auto block_v = block[k];
	for (auto j : Range(neibs)) {
	  auto neib_v = neibs[j];
	  if (block_v != neib_v) {
	    const auto & edge = fedges[fecon(block_v, neib_v)];
	    auto posj = find_in_sorted_array(size_t(neib_v), block);
	    // cout << "bv nv " << block_v << " " << neib_v << endl;
	    // cout << " kj " << k << " " << j << endl;
	    // cout << " edge " << edge << endl;
	    // cout << " posj " << posj << endl;
	    if (posj == size_t(-1)) { // block x remain
	      auto posk = find_in_sorted_array(int(block_v), neibs); // pos of block_v in cols
	      self.CalcRMBlock (fmesh, edge, edge_mat);
	      const int l0 = (block_v < neib_v) ? 0 : 1;
	      const int l1 = 1 - l0;
	      // cout << " lj " << l0 << " " << l1 << endl;
	      // print_tm_mat(cout, edge_mat); cout << endl;
	      // only block x block and block x remain (not rxb, rxr)
	      repl_row(k,posk) += edge_mat(l0,l0);
	      repl_row(k,j) += edge_mat(l0,l1);
	      // cout << " repl now : " << endl;
	      // print_tm_mat(cout, repl_row); cout << endl;
	    }
	    else if (neib_v < block_v) { // block x block - do not calc edge mat twice!
	      self.CalcRMBlock (fmesh, edge, edge_mat);
	      repl_row(k,k) += edge_mat(0,0);
	      repl_row(k,j) += edge_mat(0,1);
	      repl_row(j,k) += edge_mat(1,0);
	      repl_row(j,j) += edge_mat(1,1);
	    }
	  }
	}
      }

      // cout << " repl_row: " << endl;
      // print_tm_mat(cout, repl_row);
      // cout << endl;

      FlatArray<int> block_pos (block.Size(), lh); int c1 = 0;
      FlatArray<int> not_block_pos (neibs.Size() - block.Size(), lh); int c2 = 0;
      for (auto k : Range(neibs)) {
	if ( (c1 < block.Size()) && (block[c1] == neibs[k]) )
	  { block_pos[c1++] = k; }
	else
	  { not_block_pos[c2++] = k; }
      }
      // for (auto k : Range(block))
      // 	{ block_pos[k] = find_in_sorted_array(int(block[k]), neibs); }

      // cout << "     block_pos: "; prow2(block_pos); cout << endl;
      // cout << " not_block_pos: "; prow2(not_block_pos); cout << endl;

      FlatMatrix<TM> diag (block.Size(), block.Size(), lh);
      if (block.Size() > 1) {
	diag = repl_row.Cols(block_pos);
	if constexpr(mat_traits<TM>::HEIGHT != 1) {
	    for (auto k : Range(block.Size()))
	      { RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT> (diag(k,k)); }
	    // cout << " NO PSINV!" << endl;
	  }
	CalcInverse(diag); diag *= -1;
      }
      else {
	if constexpr(mat_traits<TM>::HEIGHT != 1) {
	    TM tm_diag = repl_row(0, block_pos[0]);
	    CalcPseudoInverse<mat_traits<TM>::HEIGHT>(tm_diag);
	    diag(0,0) = tm_diag; diag *= -1;
	  }
	else {
	  diag(0,0) = -1.0 / repl_row(0, block_pos[0]);
	}
      }

      FlatMatrix<TM> hext (block.Size(), neibs.Size() - block.Size(), lh);
      hext = diag * repl_row.Cols(not_block_pos);

      // cout << "hext: " << endl; print_tm_mat(cout, hext); cout << endl << endl;

      for (auto k : Range(block)) {
	auto block_v = block[k];
	auto ris = sprol->GetRowIndices(block_v);
	auto rvs = sprol->GetRowValues(block_v);
	int crow = 0;
	for (auto j : Range(neibs)) {
	  auto neib_v = neibs[j];
	  if (!dropped_verts.Test(neib_v)) {
	    ris[crow] = vmap[neib_v];
	    rvs[crow] = hext(k, crow);
	    crow++;
	  }
	}
      }

    } // vert_blocks
    
    // cout << " disc sprol: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl << endl;

    return sprol;
  } // VertexBasedAMGFactory::BuildSProl (discard)


  template<class FACTORY_CLASS, class TMESH, class TM>
  void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SmoothProlongation_RealMat (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TSPM_TM> fmat) const
  {
    /** Smooth prolongation with real matrix (no MPI!!) **/
    static Timer t("SmoothProlongation_RealMat"); RegionTimer rt(t);
    const auto & pwp = *pmap->GetProl();
    auto W = pwp.Width();
    auto H = pwp.Height();
    Options &O (static_cast<Options&>(*options));
    const double omega = O.sp_omega;
    auto pds = pmap->GetParDofs();
    constexpr int N = mat_traits<TM>::HEIGHT;
    const auto & A = *fmat;
    // cout << "A dims " << A.Height() << " " << A.Width() << endl;
    // cout << "pwp " << H << " " << W << endl;
    // cout << " USE REAL MAT " << endl;
    Array<TM> d(H);
    for (auto k : Range(H))
      { d[k] = A(k,k); }
    pds->AllReduceDofData (d, MPI_SUM);
    Array<size_t> vmap (H); vmap = -1;
    for (auto k : Range(H)) {
      auto ri = pwp.GetRowIndices(k);
      if (ri.Size() > 0)
	{ vmap[k] = ri[0]; }
    }
    TM id; SetIdentity(id);
    Array<int> perow(H); perow = 0; Array<int> inr(10);
    for (auto k : Range(H)){
      if (vmap[k] == -1) { continue; }
      if constexpr(N==1) { CalcInverse(d[k]); }
      else { CalcPseudoInverse<N> (d[k]); }
      auto ris = A.GetRowIndices(k);
      inr.SetSize0();
      for (auto j : Range(ris)) {
	auto cv = vmap[ris[j]];
	if (cv != -1) {
	  auto pos = inr.Pos(cv);
	  if (pos == -1)
	    { perow[k]++; inr.Append(cv); }
	}
      }
      // cout << " inrow " << k << ": "; prow(inr); cout << endl;
    }
    auto sprol = make_shared<TSPM_TM>(perow, W); const auto & SP = *sprol;
    for (auto k : Range(H)){
      if (vmap[k] == -1) { continue; }
      auto ris = A.GetRowIndices(k);
      auto rvs = A.GetRowValues(k);
      inr.SetSize0();
      for (auto j : Range(ris)) {
	auto cv = vmap[ris[j]];
	if (cv != -1) {
	  auto pos = inr.Pos(cv);
	  if (pos == -1)
	    { inr.Append(cv); }
	}
      }
      // cout << " diag " << endl; print_tm(cout, A(k,k)); cout << endl;
      // cout << " diag inv " << endl; print_tm(cout, d[k]); cout << endl;
      QuickSort(inr); sprol->GetRowIndices(k) = inr;
      // cout << " a ri "; prow(ris); cout << endl;
      // cout << " ris :"; prow(sprol->GetRowIndices(k)); cout << endl;
      sprol->GetRowValues(k) = 0;
      for (auto j : Range(rvs)) {
	auto cv = vmap[ris[j]];
	// cout << k << " " << ris[j] << " -> " << cv << endl;
	if (cv != -1) {
	  // cout << " set " << k << " " << cv << endl;
	  // cout << " sprol old " << endl; print_tm(cout, (*sprol)(k,cv)); cout << endl;
	  if (ris[j] == k)
	    { (*sprol)(k,cv) += pwp(k, cv); }
	  (*sprol)(k,cv) -= omega * d[k] * rvs[j] * pwp(ris[j], cv);
	  // TM X = 0;
	  // if (ris[j] == k)
	    // { X += pwp(k, cv); }
	  // X -= d[k] * rvs[j] * pwp(ris[j], cv);
	  // cout << " update " << endl; print_tm(cout, X); cout << endl;
	  // X = d[k] * rvs[j];
	  // cout << " d*offd " << endl; print_tm(cout, X); cout << endl;
	  // cout << " sprol new " << endl; print_tm(cout, (*sprol)(k,cv)); cout << endl;
	}
      }
    }
    // cout << "SPROL (real mat): " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;
    pmap->SetProl(sprol);
  } // SmoothProlongation_RealMat


  template<class FACTORY_CLASS, class TMESH, class TM>
  void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const
  {
    /** Smooth prolongation with replacement matrix **/

    static Timer t("SmoothProlongation"); RegionTimer rt(t);

    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const TMESH & fmesh(*mesh); fmesh.CumulateData();
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!
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

    // { // some useless bs
    //   Array<int> ccnt(NCV); ccnt = 0;
    //   for (auto v : vmap)
    // 	{ if (v!=-1) { ccnt[v]++; } }
    //   Array<double> c(30); c = 0;
    //   for (auto v : ccnt)
    // 	{ c[v] += 1.0 / NCV; }
    //   cout << " ccnt: " << endl;
    //   cout << ccnt << endl;
    //   cout << " coarse cnts: " << endl;
    //   prow(c); cout << endl;
    // }


    /** Find Graph for Prolongation **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    {
      Array<INT<2,double>> trow;
      Array<int> tcv;
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
	  // cout << V << " " << ov << " " << cov << " " << EQ << " " << oeq << " " << eqc_h.IsLEQ(EQ, oeq) << endl;
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
	// cout << V << " trow "; prow(trow); cout << endl;
	QuickSort(fin_row);
	// cout << V << " fin_row "; prow(fin_row); cout << endl;
	perow[V] = fin_row.Size();
	for (auto j:Range(fin_row.Size()))
	  graph[V][j] = fin_row[j];
      }
    }
    
    /** Create Prolongation **/
    auto sprol = make_shared<TSPM_TM>(perow, NCV);

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "hold this", false); // ~2 MB LocalHeap
    Array<INT<2,size_t>> uve(30); uve.SetSize0();
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
	// cout << endl << "------" << endl << "ROW FOR " << V << " -> " << CV << endl << "------" << endl;
	HeapReset hr(lh);
	// Find which fine vertices I can include
	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
	auto graph_row = graph[V];
	auto all_ov = fecon.GetRowIndices(V);
	auto all_oe = fecon.GetRowValues(V);
	uve.SetSize0();
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
	  // if (V != -1) {
	    // cout << "add fedge " << all_fedges[used_edges[l]] << endl;
	  // }
	  self.CalcRMBlock (fmesh, all_fedges[used_edges[l]], block);
	  int brow = (V < used_verts[l]) ? 0 : 1;
	  mat(0,l) = block(brow,1-brow); // off-diag entry
	  mat(0,posV) += block(brow,brow); // diag-entry
	//   if (V != -1) {
	//     cout << "edge diag part mat " << endl;
	//     print_tm(cout, block(brow, brow)); cout << endl;
	//     int N = mat_traits<TM>::HEIGHT;
	//     Matrix<double> d(N,N), evecs(N,N);
	//     Vector<double> evals(N);
	//     d = mat(0, posV);
	//     LapackEigenValuesSymmetric(d, evals, evecs);
	//     cout << " diag evals now: " << endl;
	//     cout << evals << endl;
	//     cout << " diag evecs now: " << endl << evecs << endl;
	//     d = block(brow,brow);
	//     LapackEigenValuesSymmetric(d, evals, evecs);
	//     cout << " block diag etr evals: " << endl;
	//     cout << evals << endl;
	//     cout << " block diag etr evecs: " << endl << evecs << endl;
	//   }
	}

	// cout << "mat row: " << endl; print_tm_mat(cout, mat); cout << endl;


	TM diag;
	double tr = 1;
	if constexpr(mat_traits<TM>::HEIGHT == 1) {
	    diag = mat(0, posV);
	  }
	else {
	  diag = mat(0, posV);
	  tr = 0; Iterate<mat_traits<TM>::HEIGHT>([&](auto i) { tr += diag(i.value,i.value); });
	  tr /= mat_traits<TM>::HEIGHT;
	  diag /= tr; // avg eval of diag is now 1
	  mat /= tr;
	  // cout << "scale: " << tr << " " << 1.0/tr << endl;
	  // if (sing_diags) {
	  //   self.RegDiag(diag);
	  // }
	}

	// cout << "scaled mat row: " << endl; print_tm_mat(cout, mat); cout << endl;

	// if constexpr(mat_traits<TM>::HEIGHT!=1) {
	//     constexpr int M = mat_traits<TM>::HEIGHT;
	//     static Matrix<double> D(M,M), evecs(M,M);
	//     static Vector<double> evals(M);
	//     D = diag;
	//     LapackEigenValuesSymmetric(D, evals, evecs);
	//     cout << " diag eig-vals: " << endl;
	//     cout << evals << endl;
	//     cout << " evecs: " << endl;
	//     cout << evecs << endl;
	//   }
	if constexpr(mat_traits<TM>::HEIGHT==1) {
	    CalcInverse(diag);
	  }
	else {
	  // cout << " pseudo invert diag " << endl; print_tm(cout, diag); cout << endl;

	  constexpr int N = mat_traits<TM>::HEIGHT;

	  // FlatMatrix<double> evecs(N, N, lh), d(N, N, lh); FlatVector<double> evals(N, lh);
	  // d = diag;
	  // LapackEigenValuesSymmetric(d, evals, evecs);
	  // cout << "evecs: " << endl << evecs << endl;
	  // cout << "1 evals: " << evals << endl;
	  // for (auto & v : evals)
	  //   { v = (v > 0.1 * evals(N-1)) ? 1.0/sqrt(v) : 0; }
	  // cout << "2 evals: " << evals << endl;
	  // for (auto k : Range(N))
	  //   for (auto j : Range(N))
	  //     evecs(k,j) *= evals(k);
	  // diag = Trans(evecs) * evecs;

	  // prt_evv<N>(diag, "init dg", false);
	  
	  /** Scale "diag" such that it has 1s in it's diagonal, then SVD, eliminate small EVs,
	      Pseudo inverse, scale back. **/
	  double tr = calc_trace(diag) / N;
	  double eps = 1e-8 * tr;
	  int M = 0;
	  for (auto k : Range(N))
	    if (diag(k,k) > eps)
	      { M++; }
	  FlatArray<double> diag_diags(M, lh);
	  FlatArray<double> diag_diag_invs(M, lh);
	  FlatArray<int> nzeros(M, lh);
	  M = 0;
	  for (auto k : Range(N)) {
	    if (diag(k,k) > eps) {
	      auto rt = sqrt(diag(k,k));
	      diag_diags[M] = rt;
	      diag_diag_invs[M] = 1.0/rt;
	      nzeros[M] = k;
	      M++;
	    }
	  }
	  FlatMatrix<double> smallD(M,M,lh);
	  // cout << "smallD: " << endl;
	  for (auto i : Range(M))
	    for (auto j : Range(M))
	      { smallD(i,j) = diag(nzeros[i], nzeros[j]) * diag_diag_invs[i] * diag_diag_invs[j]; }
	  // cout << smallD << endl;
	  FlatMatrix<double> evecs(M,M,lh);
	  FlatVector<double> evals(M, lh);
	  LapackEigenValuesSymmetric(smallD, evals, evecs);
	  // cout << " small D evals (of " << M << "): "; prow(evals); cout << endl;
	  for (auto k : Range(M)) {
	    double f = (evals(k) > 0.1) ? 1/sqrt(evals(k)) : 0;
	    for (auto j : Range(M))
	      { evecs(k,j) *= f; }
	  }
	  smallD = Trans(evecs) * evecs;
	  diag = 0;
	  for (auto i : Range(M))
	    for (auto j : Range(M))
	      { diag(nzeros[i],nzeros[j]) = smallD(i,j) * diag_diag_invs[i] * diag_diag_invs[j]; }
	  // CalcPseudoInverse<mat_traits<TM>::HEIGHT>(diag);

	  // cout << " inv: " << endl; print_tm(cout, diag); cout << endl;
	  // prt_evv<N>(diag, "inved dg", false);

	}

	auto sp_ri = sprol->GetRowIndices(V); sp_ri = graph_row;
	auto sp_rv = sprol->GetRowValues(V); sp_rv = 0;
	// double fac = omega/tr;
	double fac = omega;
	for (auto l : Range(unv)) {
	  int vl = used_verts[l];
	  auto pw_rv = pwprol.GetRowValues(vl);
	  int cvl = vmap[vl];
	  auto pos = find_in_sorted_array(cvl, sp_ri);
	  if (l==posV)
	    { sp_rv[pos] += pw_rv[0]; }

	  // cout << " --- " << endl;
	  // cout << " pw_rv " << endl;
	  // print_tm(cout, pw_rv[0]); cout << endl;

	  // TM dm = fac * diag * mat(0,l);
	  // cout << " diaginv * metr " << endl;
	  // print_tm(cout, dm); cout << endl;

	  // TM dm2 = dm * pw_rv[0];
	  // cout << "update: " << endl;
	  // print_tm(cout, dm2); cout << endl;

	  // cout << " old sp etr (" << pos << "): " << endl;
	  // print_tm(cout, sp_rv[pos]); cout << endl;

	  sp_rv[pos] -= fac * (diag * mat(0,l)) * pw_rv[0];

	  // cout << " -> sprol entry " << V << " " << graph_row[pos] << ":" << endl;
	  // print_tm(cout, sp_rv[pos]); cout << endl;
	  // cout << "---" << endl;
	}
      }
    }

    // cout << "sprol:: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    pmap->SetProl(sprol);
  } // VertexBasedAMGFactory::SmoothProlongation
  

  // template<class FACTORY_CLASS, class TMESH, class TM>
  // void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SmoothProlongation2 (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const
  // {
  //   /** Smooth prolongation with replacement matrix **/
  //   static Timer t("SmoothProlongation2"); RegionTimer rt(t);
  //   const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
  //   const TMESH & fmesh(*mesh); fmesh.CumulateData();
  //   const auto & fecon = *fmesh.GetEdgeCM();
  //   const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!
  //   const TSPM_TM & pwprol = *pmap->GetProl();
  //   auto avd = get<0>(fmesh.Data());
  //   auto vdata = avd->Data();
  //   auto aed = get<1>(fmesh.Data());
  //   auto edata = aed->Data();
  //   Options &O (static_cast<Options&>(*options));
  //   const double MIN_PROL_FRAC = O.sp_min_frac;
  //   const int MAX_PER_ROW = O.sp_max_per_row;
  //   const double omega = O.sp_omega;
  //   // Construct vertex-map from prol graph (can be concatenated)
  //   size_t NFV = fmesh.template GetNN<NT_VERTEX>();
  //   Array<size_t> vmap (NFV); vmap = -1;
  //   size_t NCV = 0;
  //   for (auto k : Range(NFV)) {
  //     auto ri = pwprol.GetRowIndices(k);
  //     // we should be able to combine smoothing and discarding
  //     // if (ri.Size() > 1) { cout << "TODO: handle this case, dummy" << endl; continue; } // this is for double smooted prol
  //     // if (ri.Size() > 1) { throw Exception("comment this out"); }
  //     if (ri.Size() > 0) {
  // 	vmap[k] = ri[0];
  // 	NCV = max2(NCV, size_t(ri[0]+1));
  //     }
  //   }
  // /** edge-matrices with common neighbour boost **/
  // typename FACTORY_CLASS::T_V_DATA MPD;
  // TM Q, EiN, EjN, ES, addE, Qij, Qji;
  // Array<TM> s_emats(fmesh.template GetNN<NT_EDGE>());
  // fmesh.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & edge) LAMBDA_INLINE {
  // 	// cout << " calc bonus edge " << edge << endl;
  // 	s_emats[edge.id] = edata[edge.id];
  // 	// prt_evv<mat_traits<TM>::HEIGHT>(s_emats[edge.id], "init", false);
  // 	auto v0 = edge.v[0]; auto ri0 = fecon.GetRowIndices(v0); auto rv0 = fecon.GetRowValues(v0);
  // 	auto v1 = edge.v[1]; auto ri1 = fecon.GetRowIndices(v1); auto rv1 = fecon.GetRowValues(v1);
  // 	MPD = FACTORY_CLASS::CalcMPData(vdata[v0], vdata[v1]);
  // 	iterate_intersection(ri0, ri1, [&](auto i0, auto i1) {
  //   	    auto N = ri0[i0];
  // 	    FACTORY_CLASS::CalcQij(vdata[N], vdata[v0], Q);
  // 	    EiN = Trans(Q) * edata[int(rv0[i0])] * Q;
  // 	    FACTORY_CLASS::CalcQij(vdata[N], vdata[v1], Q);
  // 	    EjN = Trans(Q) * edata[int(rv0[i1])] * Q;
  // 	    ES = EiN + EjN;
  // 	    if constexpr(is_same<TM, double>::value) { CalcInverse(ES); }
  // 	    else { CalcPseudoInverse<mat_traits<TM>::HEIGHT>(ES); }
  // 	    FACTORY_CLASS::CalcQHh(MPD, vdata[N], Q);
  // 	    addE = EiN * ES * EjN;
  //   	    s_emats[edge.id] += 2 * Trans(Q) * addE * Q;
  // 	    TM b = 2 * Trans(Q) * addE * Q;
  // 	    // prt_evv<mat_traits<TM>::HEIGHT>(b, string("bonus") + to_string(N), false);
  // 	  });
  // 	// prt_evv<mat_traits<TM>::HEIGHT>(s_emats[edge.id], "final", false);
  //   }, false); // TODO: only add neibs i am master of, cumulate, THEN add fmesh edge mat
  // // For each fine vertex, sum up weights of edges that connect to the same CV
  // //  (can be more than one edge, if the pw-prol is concatenated)
  // // TODO: for many concatenated pws, does this dominate edges to other agglomerates??
  // auto all_fedges = fmesh.template GetNodes<NT_EDGE>();
  // Array<double> vw (NFV); vw = 0;
  // auto neqcs = eqc_h.GetNEQCS();
  // {
  //   INT<2, int> cv;
  //   fmesh.template ApplyEQ<NT_EDGE>( [&](auto eqc, const auto & edge) {
  // 	  if ( ((cv[0] = vmap[edge.v[0]]) != -1 ) &&
  // 	       ((cv[1] = vmap[edge.v[1]]) != -1 ) &&
  // 	       (cv[0] == cv[1]) ) {
  // 	    auto com_wt = self.template GetWeight<NT_EDGE>(fmesh, edge);
  // 	    vw[edge.v[0]] += com_wt;
  // 	    vw[edge.v[1]] += com_wt;
  // 	  }
  // 	}, true);
  // }
  // fmesh.template AllreduceNodalData<NT_VERTEX>(vw, [](auto & tab){return move(sum_table(tab)); }, false);
  // /** Find Graph for Prolongation **/
  // Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
  // Array<int> perow(NFV); perow = 0; // 
  // {
  //   Array<INT<2,double>> trow;
  //   Array<int> tcv;
  //   Array<size_t> fin_row;
  //   for (auto V:Range(NFV)) {
  // 	auto CV = vmap[V];
  // 	if ( is_invalid(CV) ) continue; // grounded -> TODO: do sth. here if we are free?
  // 	if (vw[V] == 0.0) { // MUST be single
  // 	  perow[V] = 1;
  // 	  graph[V][0] = CV;
  // 	  continue;
  // 	}
  // 	trow.SetSize(0);
  // 	tcv.SetSize(0);
  // 	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
  // 	auto ovs = fecon.GetRowIndices(V);
  // 	auto eis = fecon.GetRowValues(V);
  // 	size_t pos;
  // 	for (auto j:Range(ovs.Size())) {
  // 	  auto ov = ovs[j];
  // 	  auto cov = vmap[ov];
  // 	  if (is_invalid(cov) || cov==CV) continue;
  // 	  auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
  // 	  if (eqc_h.IsLEQ(EQ, oeq)) {
  // 	    auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[eis[j]]);
  // 	    if ( (pos = tcv.Pos(cov)) == size_t(-1)) {
  // 	      trow.Append(INT<2,double>(cov, wt));
  // 	      tcv.Append(cov);
  // 	    }
  // 	    else {
  // 	      trow[pos][1] += wt;
  // 	    }
  // 	  }
  // 	}
  // 	// cout << " row " << V << " tcv "; prow(tcv); cout << endl;
  // 	QuickSort(trow, [](const auto & a, const auto & b) {
  // 	    if (a[0]==b[0]) return false;
  // 	    return a[1]>b[1];
  // 	  });
  // 	double cw_sum = (is_valid(CV)) ? vw[V] : 0.0;
  // 	fin_row.SetSize(0);
  // 	if (is_valid(CV)) fin_row.Append(CV); //collapsed vertex
  // 	size_t max_adds = (is_valid(CV)) ? min2(MAX_PER_ROW-1, int(trow.Size())) : trow.Size();
  // 	for (auto j:Range(max_adds)) {
  // 	  cw_sum += trow[j][1];
  // 	  if (is_valid(CV)) {
  // 	    // I don't think I actually need this: Vertex is collapsed to some non-weak (not necessarily "strong") edge
  // 	    // therefore the relative weight comparison should eliminate all really weak connections
  // 	    // if (fin_row.Size() && (trow[j][1] < MIN_PROL_WT)) break; 
  // 	    if (trow[j][1] < MIN_PROL_FRAC*cw_sum) break;
  // 	  }
  // 	  fin_row.Append(trow[j][0]);
  // 	}
  // 	QuickSort(fin_row);
  // 	perow[V] = fin_row.Size();
  // 	// cout << " row " << V << " fin_row "; prow(fin_row); cout << endl;
  // 	for (auto j:Range(fin_row.Size()))
  // 	  graph[V][j] = fin_row[j];
  //   }
  // }
  // /** Create Prolongation **/
  // auto sprol = make_shared<TSPM_TM>(perow, NCV);
  // /** Fill Prolongation **/
  // LocalHeap lh(2000000, "hold this", false); // ~2 MB LocalHeap
  // Array<INT<2,size_t>> uve(30); uve.SetSize0();
  // Array<int> used_verts(20), used_edges(20);
  // TM id; SetIdentity(id);
  // for (int V:Range(NFV)) {
  //   auto CV = vmap[V];
  //   if (is_invalid(CV)) continue; // grounded -> TODO: do sth. here if we are free?
  //   if (perow[V] == 1) { // SINGLE or no good connections avail.
  // 	sprol->GetRowIndices(V)[0] = CV;
  // 	sprol->GetRowValues(V)[0] = pwprol.GetRowValues(V)[0];
  //   }
  //   else { // SMOOTH
  // 	// cout << endl << "------" << endl << "ROW FOR " << V << " -> " << CV << endl << "------" << endl;
  // 	HeapReset hr(lh);
  // 	// Find which fine vertices I can include
  // 	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
  // 	auto graph_row = graph[V];
  // 	auto all_ov = fecon.GetRowIndices(V);
  // 	auto all_oe = fecon.GetRowValues(V);
  // 	uve.SetSize0();
  // 	for (auto j:Range(all_ov.Size())) {
  // 	  auto ov = all_ov[j];
  // 	  auto cov = vmap[ov];
  // 	  if (is_valid(cov)) {
  // 	    if (graph_row.Contains(cov)) {
  // 	      auto eq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
  // 	      if (eqc_h.IsLEQ(EQ, eq)) {
  // 		uve.Append(INT<2>(ov,all_oe[j]));
  // 	      } } } }
  // 	uve.Append(INT<2>(V,-1));
  // 	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];}); // WHY??
  // 	used_verts.SetSize(uve.Size()); used_edges.SetSize(uve.Size());
  // 	for (auto k:Range(uve.Size()))
  // 	  { used_verts[k] = uve[k][0]; used_edges[k] = uve[k][1]; }
  // auto posV = find_in_sorted_array(int(V), used_verts);
  // size_t unv = used_verts.Size(); // # of vertices used
  // FlatMatrix<TM> mat (1,unv,lh); mat(0, posV) = 0;
  // FlatMatrix<TM> block (2,2,lh);
  // for (auto l:Range(unv)) {
  //   if (l==posV) continue;
  //   const auto & edge = all_fedges[used_edges[l]];
  //   int L = (V == edge.v[0]) ? 0 : 1;
  //   FACTORY_CLASS::CalcQs(vdata[edge.v[L]], vdata[edge.v[1-L]], Qij, Qji);
  //   // FACTORY_CLASS::CalcQij(vdata[edge.v[L]], vdata[edge.v[1-L]], Qij);
  //   // FACTORY_CLASS::CalcQij(vdata[edge.v[1-L]], vdata[edge.v[L]], Qji);
  //   Q = Trans(Qij) * s_emats[used_edges[l]];
  //   // Q = Trans(Qij) * edata[used_edges[l]];
  //   mat(0, l) = -Q * Qji;
  //   mat(0, posV) += Q * Qij;
  // }
  // TM diag;
  // double tr = 1;
  // if constexpr(mat_traits<TM>::HEIGHT == 1) {
  //     diag = mat(0, posV);
  //   }
  // else {
  //   diag = mat(0, posV);
  //   tr = 0; Iterate<mat_traits<TM>::HEIGHT>([&](auto i) { tr += diag(i.value,i.value); });
  //   tr /= mat_traits<TM>::HEIGHT;
  //   diag /= tr; // avg eval of diag is now 1
  //   mat /= tr;
  //   // cout << "scale: " << tr << " " << 1.0/tr << endl;
  //   // if (sing_diags) {
  //   //   self.RegDiag(diag);
  //   // }
  // }
  // // cout << "scaled mat row: " << endl; print_tm_mat(cout, mat); cout << endl;
  // // if constexpr(mat_traits<TM>::HEIGHT!=1) {
  // //     constexpr int M = mat_traits<TM>::HEIGHT;
  // //     static Matrix<double> D(M,M), evecs(M,M);
  // //     static Vector<double> evals(M);
  // //     D = diag;
  // //     LapackEigenValuesSymmetric(D, evals, evecs);
  // //     cout << " diag eig-vals: " << endl;
  // //     cout << evals << endl;
  // //     cout << " evecs: " << endl;
  // //     cout << evecs << endl;
  // //   }
  // if constexpr(mat_traits<TM>::HEIGHT==1) {
  //     CalcInverse(diag);
  //   }
  // else {
  //   // cout << " pseudo invert diag " << endl; print_tm(cout, diag); cout << endl;
  // 	  constexpr int N = mat_traits<TM>::HEIGHT;
  // 	  // FlatMatrix<double> evecs(N, N, lh), d(N, N, lh); FlatVector<double> evals(N, lh);
  // 	  // d = diag;
  // 	  // LapackEigenValuesSymmetric(d, evals, evecs);
  // 	  // cout << "evecs: " << endl << evecs << endl;
  // 	  // cout << "1 evals: " << evals << endl;
  // 	  // for (auto & v : evals)
  // 	  //   { v = (v > 0.1 * evals(N-1)) ? 1.0/sqrt(v) : 0; }
  // 	  // cout << "2 evals: " << evals << endl;
  // 	  // for (auto k : Range(N))
  // 	  //   for (auto j : Range(N))
  // 	  //     evecs(k,j) *= evals(k);
  // 	  // diag = Trans(evecs) * evecs;
  // 	  // prt_evv<N>(diag, "init dg", false);
  // 	  /** Scale "diag" such that it has 1s in it's diagonal, then SVD, eliminate small EVs,
  // 	      Pseudo inverse, scale back. **/
  // 	  double tr = calc_trace(diag) / N;
  // 	  double eps = 1e-8 * tr;
  // 	  int M = 0;
  // 	  for (auto k : Range(N))
  // 	    if (diag(k,k) > eps)
  // 	      { M++; }
  // 	  FlatArray<double> diag_diags(M, lh);
  // 	  FlatArray<double> diag_diag_invs(M, lh);
  // 	  FlatArray<int> nzeros(M, lh);
  // 	  M = 0;
  // 	  for (auto k : Range(N)) {
  // 	    if (diag(k,k) > eps) {
  // 	      auto rt = sqrt(diag(k,k));
  // 	      diag_diags[M] = rt;
  // 	      diag_diag_invs[M] = 1.0/rt;
  // 	      nzeros[M] = k;
  // 	      M++;
  // 	    }
  // 	  }
  // 	  FlatMatrix<double> smallD(M,M,lh);
  // 	  // cout << "smallD: " << endl;
  // 	  for (auto i : Range(M))
  // 	    for (auto j : Range(M))
  // 	      { smallD(i,j) = diag(nzeros[i], nzeros[j]) * diag_diag_invs[i] * diag_diag_invs[j]; }
  // 	  // cout << smallD << endl;
  // 	  FlatMatrix<double> evecs(M,M,lh);
  // 	  FlatVector<double> evals(M, lh);
  // 	  LapackEigenValuesSymmetric(smallD, evals, evecs);
  // 	  // cout << " small D evals (of " << M << "): "; prow(evals); cout << endl;
  // 	  for (auto k : Range(M)) {
  // 	    double f = (evals(k) > 0.1) ? 1/sqrt(evals(k)) : 0;
  // 	    for (auto j : Range(M))
  // 	      { evecs(k,j) *= f; }
  // 	  }
  // 	  smallD = Trans(evecs) * evecs;
  // 	  diag = 0;
  // 	  for (auto i : Range(M))
  // 	    for (auto j : Range(M))
  // 	      { diag(nzeros[i],nzeros[j]) = smallD(i,j) * diag_diag_invs[i] * diag_diag_invs[j]; }
  // 	  // CalcPseudoInverse<mat_traits<TM>::HEIGHT>(diag);
  // 	  // cout << " inv: " << endl; print_tm(cout, diag); cout << endl;
  // 	  // prt_evv<N>(diag, "inved dg", false);
  // 	}
  // 	auto sp_ri = sprol->GetRowIndices(V); sp_ri = graph_row;
  // 	auto sp_rv = sprol->GetRowValues(V); sp_rv = 0;
  // 	// double fac = omega/tr;
  // 	double fac = omega;
  // 	for (auto l : Range(unv)) {
  // 	  int vl = used_verts[l];
  // 	  auto pw_rv = pwprol.GetRowValues(vl);
  // 	  int cvl = vmap[vl];
  // 	  auto pos = find_in_sorted_array(cvl, sp_ri);
  // 	  if (l==posV)
  // 	    { sp_rv[pos] += pw_rv[0]; }
  // 	  // cout << " --- " << endl;
  // 	  // cout << " pw_rv " << endl;
  // 	  // print_tm(cout, pw_rv[0]); cout << endl;
  // 	  // TM dm = fac * diag * mat(0,l);
  // 	  // cout << " diaginv * metr " << endl;
  // 	  // print_tm(cout, dm); cout << endl;
  // 	  // TM dm2 = dm * pw_rv[0];
  // 	  // cout << "update: " << endl;
  // 	  // print_tm(cout, dm2); cout << endl;
  // 	  // cout << " old sp etr (" << pos << "): " << endl;
  // 	  // print_tm(cout, sp_rv[pos]); cout << endl;
  // 	  sp_rv[pos] -= fac * (diag * mat(0,l)) * pw_rv[0];
  // 	  // cout << " -> sprol entry " << V << " " << graph_row[pos] << ":" << endl;
  // 	  // print_tm(cout, sp_rv[pos]); cout << endl;
  // 	  // cout << "---" << endl;
  // 	}
  //     }
  //   }
  //   cout << "sprol (+NEIBS):: " << endl;
  //   print_tm_spmat(cout, *sprol); cout << endl;
  //   pmap->SetProl(sprol);
  // } // VertexBasedAMGFactory::SmoothProlongation2


  template<class FACTORY_CLASS, class TMESH, class TM>
  void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: SmoothProlongationAgg (shared_ptr<ProlMap<TSPM_TM>> pmap,
										 shared_ptr<AgglomerateCoarseMap<TMESH>> agg_map) const
  {
    /** Smooth prolongation with replacement matrix **/

    static Timer t("SmoothProlongationAgg"); RegionTimer rt(t);

    const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
    const TMESH & fmesh(static_cast<TMESH&>(*agg_map->GetMesh())); fmesh.CumulateData();
    auto avd = get<0>(fmesh.Data());
    auto vdata = avd->Data();
    auto aed = get<1>(fmesh.Data());
    auto edata = aed->Data();
    const auto & fecon = *fmesh.GetEdgeCM();
    auto all_fedges = fmesh.template GetNodes<NT_EDGE>();

    const TMESH & CM(static_cast<TMESH&>(*agg_map->GetMappedMesh()));

    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch == fine eqch !!
    auto neqcs = eqc_h.GetNEQCS();

    const TSPM_TM & pwprol = *pmap->GetProl();

    Options &O (static_cast<Options&>(*options));
    const double MIN_PROL_FRAC = O.sp_min_frac;
    const int MAX_PER_ROW = O.sp_max_per_row;
    const double omega = O.sp_omega;

    // Construct vertex-map from prol graph (can be concatenated)
    auto vmap = agg_map->template GetMap<NT_VERTEX>();
    size_t NFV = fmesh.template GetNN<NT_VERTEX>();
    const auto NCV = agg_map->template GetMappedNN<NT_VERTEX>();
    // Array<size_t> vmap (NFV); vmap = -1;
    // size_t NCV = 0;
    // for (auto k : Range(NFV)) {
    //   auto ri = pwprol.GetRowIndices(k);
    //   if (ri.Size() > 0) {
    // 	vmap[k] = ri[0];
    // 	NCV = max2(NCV, size_t(ri[0]+1));
    //   }
    // }

    /** For each fine vertex, find all coarse vertices we can (and should) prolongate from.
	The master of V does this. **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    Array<INT<2,double>> trow;
    Array<int> tcv, fin_row;
    fmesh.template ApplyEQ<NT_VERTEX>([&](auto EQ, auto V) LAMBDA_INLINE  {
	auto CV = vmap[V];
	if ( is_invalid(CV) ) // Dirichlet/grounded
	  { return; } 
	trow.SetSize0(); tcv.SetSize0(); fin_row.SetSize0();
	auto ovs = fecon.GetRowIndices(V);
	auto eis = fecon.GetRowValues(V);
	size_t pos; double in_wt = 0;
	for (auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if ( is_invalid(cov) )
	    { continue; }
	  if (cov == CV) {
	    in_wt += self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
	    continue;
	  }
	  // auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	  auto oeq = CM.template GetEqcOfNode<NT_VERTEX>(cov);
	  if (eqc_h.IsLEQ(EQ, oeq)) {
	    auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
	    if ( (pos = tcv.Pos(cov)) == size_t(-1)) {
	      trow.Append(INT<2,double>(cov, wt));
	      tcv.Append(cov);
	    }
	    else
	      { trow[pos][1] += wt; }
	  }
	}
	QuickSort(trow, [](const auto & a, const auto & b) LAMBDA_INLINE { return a[1]>b[1]; });
	double cw_sum = 0.2 * in_wt; // all edges in the same agg are automatically assembled (penalize so we dont pw-ize too many)
	fin_row.Append(CV);
	size_t max_adds = min2(MAX_PER_ROW-1, int(trow.Size()));
	for (auto j : Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if (trow[j][1] < MIN_PROL_FRAC * cw_sum)
	    { break; }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	for (auto j:Range(fin_row.Size()))
	  { graph[V][j] = fin_row[j]; }
	int nniscv = 0; // number neibs in same cv
	// cout << "ovs: ";
	for (auto v : ovs) {
	  auto neib_cv = vmap[v];
	  auto pos = find_in_sorted_array(int(neib_cv), fin_row);
	  if (pos != -1)
	    { /* cout << "[" << v << " -> " << neib_cv << "] "; */ perow[V]++; }
	  // else
	  //   { cout << "[not " << v << " -> " << neib_cv << "] "; }
	  if (neib_cv == CV)
	    { nniscv++; }
	}
	// cout << endl;
	perow[V]++; // V always in!
	if (nniscv == 0) { // keep this as is (PW prol)
	  // if (fin_row.Size() > 1) {
	  //   cout << "reset a V" << endl;
	  //   cout << V << " " << CV << endl;
	  //   cout << "graph: "; prow(graph[V]); cout << endl;
	  // }
	  graph[V] = -1;
	  // cout << "" << endl;
	  graph[V][0] = CV;
	  perow[V] = 1;
	}
      }, true); //
    
    /** Create RM **/
    auto rmat = make_shared<TSPM_TM>(perow, NCV);
    const auto & RM = *rmat;

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "hold this", false); // ~2 MB LocalHeap
    Array<INT<2,int>> une(20);
    TM Q, Qij, Qji, diag, rvl, ID; SetIdentity(ID);
    fmesh.template ApplyEQ<NT_VERTEX>([&](auto EQ, auto V) LAMBDA_INLINE {
	auto CV = vmap[V];
	if ( is_invalid(CV) ) // grounded/dirichlet
	  { return; }
	// cout << " ROW " << V << endl;
	auto all_grow = graph[V]; int grs = all_grow.Size();
	for (auto k : Range(all_grow))
	  if (all_grow[k] == -1)
	    { grs = k; break; }
	auto grow = all_grow.Part(0, grs);
	auto neibs = fecon.GetRowIndices(V); auto neibeids = fecon.GetRowValues(V);
	auto ris = RM.GetRowIndices(V); auto rvs = RM.GetRowValues(V);
	// cout << " grow: "; prow(grow); cout << endl;
	// cout << " neibs: "; prow(neibs); cout << endl;
	// cout << " riss " << ris.Size() << endl;
	int cn = 0;
	une.SetSize0();
	INT<2,int> ME ({ V, -1 });
	une.Append(ME);
	for (auto jn : Range(neibs)) {
	  int n = neibs[jn];
	  int CN = vmap[n];
	  auto pos = find_in_sorted_array(CN, grow);
	  if (pos != -1)
	    { une.Append(INT<2,int>({n, int(neibeids[jn])})); }
	}
	if (une.Size() == 1)
	  { SetIdentity(rvs[0]); ris[0] = V; return; }
	QuickSort(une, [](const auto & a, const auto & b) LAMBDA_INLINE { return a[0]<b[0]; });
	auto MEpos = une.Pos(ME);
	// cout << " une "; for (auto&x : une) { cout << "[" << x[0] << " " << x[1] << "] "; } cout << endl;
	rvs[MEpos] = 0; // cout << "MEpos " << MEpos << endl;
	double maxtr = 0;
	for (auto l : Range(une))
	  if (l != MEpos) { // a cheap guess
	    const auto & edge = all_fedges[une[l][1]];
	    int L = (V == edge.v[0]) ? 0 : 1;
	    if (vmap[edge.v[1-L]] == CV)
	      { maxtr = max2(maxtr, calc_trace(edata[une[l][1]])); }
	  }
	maxtr /= mat_traits<TM>::HEIGHT;
	for (auto l : Range(une)) {
	  if (l != MEpos) {
	    const auto & edge = all_fedges[une[l][1]];
	    int L = (V == edge.v[0]) ? 0 : 1;
	    // cout << " l " << l << " L " << L << " edge " << edge << ", un " << une[l][0] << " " << une[l][1] << endl;
	    FACTORY_CLASS::CalcQs(vdata[edge.v[L]], vdata[edge.v[1-L]], Qij, Qji);
	    // Q = Trans(Qij) * s_emats[used_edges[l]];
	    TM EMAT = edata[une[l][1]];
	    if constexpr(mat_traits<TM>::HEIGHT!=1) {
		// RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT);
		// RegTM<0, FACTORY_CLASS::DIM, mat_traits<TM>::HEIGHT>(EMAT);
		// if (vmap[une[l][0]] == CV)
		  // { RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT, maxtr); }
		if (vmap[une[l][0]] == CV) {
		  RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT);
		}
	      }
	    Q = Trans(Qij) * EMAT;
	    rvs[l] = Q * Qji;
	    rvs[MEpos] += Q * Qij;
	    ris[l] = une[l][0];
	  }
	}
	ris[MEpos] = V;

	// cout << " ROW " << V << " RI: "; prow(ris); cout << endl;
	// cout << " ROW " << V << " RV (no diag): " << endl;
	// for (auto&  v : rvs)
	//   { print_tm(cout, v); }
	// cout << " repl mat diag row " << V << endl;

	diag = rvs[MEpos];

	// prt_evv<mat_traits<TM>::HEIGHT> (diag, "diag", false);
	// if constexpr(mat_traits<TM>::HEIGHT!=1) {
	//     RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(diag);
	//   }
	// CalcPseudoInverse2<mat_traits<TM>::HEIGHT>(diag, lh);
	// prt_evv<mat_traits<TM>::HEIGHT> (diag, "inv diag", false);

	CalcInverse(diag);

	for (auto l : Range(une)) {
	  rvl = rvs[l];
	  if (l == MEpos) // pseudo inv * mat can be != ID
	    { rvs[l] = ID - omega * diag * rvl; }
	  else
	    { rvs[l] = omega * diag * rvl; }
	}

	// cout << " ROW " << V << " RV (with diag): ";
	// for (auto&  v : rvs)
	//   { print_tm(cout, v); }

      }, true); // for (V)
  

    // cout << endl << "repl mat (I-omega Dinv A): " << endl;
    // print_tm_spmat(cout, RM); cout << endl;

    auto sprol = pmap->GetProl();
    sprol = MatMultAB(RM, *sprol);

    /** Now, unfortunately, we have to distribute matrix entries of sprol. We cannot do this for RM.
	(we are also using more local fine edges that map to less local coarse edges) **/
    if (eqc_h.GetCommunicator().Size() > 2) {
      const auto & SP = *sprol;
      Array<int> perow(sprol->Height()); perow = 0;
      fmesh.template ApplyEQ<NT_VERTEX>( Range(neqcs), [&](auto EQC, auto V) {
	  auto ris = sprol->GetRowIndices(V).Size();
	  perow[V] = ris;
	}, false); // all - also need to alloc loc!
      fmesh.template ScatterNodalData<NT_VERTEX>(perow);
      auto cumul_sp = make_shared<TSPM_TM>(perow, NCV);
      Array<int> eqc_perow(neqcs); eqc_perow = 0;
      if (neqcs > 1)
	fmesh.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
	    eqc_perow[EQC] += perow[V];
	  }, false); // all!
      Table<INT<2,int>> ex_ris(eqc_perow);
      Table<TM> ex_rvs(eqc_perow); eqc_perow = 0;
      if (neqcs > 1)
	fmesh.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
	    auto rvs = sprol->GetRowValues(V);
	    auto ris = sprol->GetRowIndices(V);
	    for (auto j : Range(ris)) {
	      int jeq = CM.template GetEqcOfNode<NT_VERTEX>(ris[j]);
	      int jeq_id = eqc_h.GetEQCID(jeq);
	      int jlc = CM.template MapENodeToEQC<NT_VERTEX>(jeq, ris[j]);
	      ex_ris[EQC][eqc_perow[EQC]] = INT<2,int>({ jeq_id, jlc });
	      ex_rvs[EQC][eqc_perow[EQC]++] = rvs[j];
	    }
	  }, true); // master!
      auto reqs = eqc_h.ScatterEQCData(ex_ris);
      reqs += eqc_h.ScatterEQCData(ex_rvs);
      MyMPI_WaitAll(reqs);
      const auto & CSP = *cumul_sp;
      eqc_perow = 0;
      if (neqcs > 1)
	fmesh.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
	    auto rvs = CSP.GetRowValues(V);
	    auto ris = CSP.GetRowIndices(V);
	    for (auto j : Range(ris)) {
	      auto tup = ex_ris[EQC][eqc_perow[EQC]];
	      ris[j] = CM.template MapENodeFromEQC<NT_VERTEX>(tup[1], eqc_h.GetEQCOfID(tup[0]));
	      rvs[j] = ex_rvs[EQC][eqc_perow[EQC]++];
	    }
	  }, false); // master!
      if (neqcs > 0)
	for (auto V : fmesh.template GetENodes<NT_VERTEX>(0)) {
	  CSP.GetRowIndices(V) = SP.GetRowIndices(V);
	  CSP.GetRowValues(V) = SP.GetRowValues(V);
	}
      sprol = cumul_sp;
      // cout << "CUMULATED SPROL: " << endl;
      // print_tm_spmat(cout, *sprol); cout << endl;
    }

    // cout << "SPROL: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    pmap->SetProl(sprol);
  } // VertexBasedAMGFactory::SmoothProlongationAgg





  // template<class FACTORY_CLASS, class TMESH, class TM>
  // void VertexBasedAMGFactory<FACTORY_CLASS, TMESH, TM> :: ModCoarseEdata (shared_ptr<ProlMap<TSPM_TM>> spmap, shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh) const
  // {
  //   /** Does not add additional coarse edges, but DOES modify coarse edge matrices. **/
  //   if (!options->enable_mced)
  //     { return; }
  //   fmesh->CumulateData();
  //   const FACTORY_CLASS & self = static_cast<const FACTORY_CLASS&>(*this);
  //   Options &O (static_cast<Options&>(*options));
  //   const int MAX_PER_ROW = O.sp_max_per_row;
  //   const auto & FM = *fmesh;
  //   const auto & fecon = *FM.GetEdgeCM();
  //   auto f_avd = get<0>(FM.Data());
  //   auto fvd = f_avd->Data();
  //   auto f_aed = get<1>(FM.Data());
  //   auto fed = f_aed->Data();
  //   const auto & CM = *cmesh;
  //   const auto & cecon = *CM.GetEdgeCM();
  //   auto c_avd = get<0>(CM.Data()); c_avd->Cumulate();
  //   auto cvd = c_avd->Data();
  //   auto c_aed = get<1>(CM.Data()); c_aed->SetParallelStatus(DISTRIBUTED);
  //   auto ced = c_aed->Data();
  //   const auto & SP  = *spmap->GetProl();
  //   const auto & SPT = *spmap->GetProlTrans();
  //   Array<TM> old_ced(CM.template GetNN<NT_EDGE>()); old_ced = ced;
  //   /** Set uI = 0.5*QHI*wH, uJ = -0.5*QHJ*wH
  // 	Semms to over-estimate with parameter jumps. **/
  //   // TM Qij, Qji, QIJ, QJI, RiI, RjJ, RS, RE, R;
  //   // FM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & fedge) LAMBDA_INLINE {
  //   // 	// i<j always
  //   // 	int i = fedge.v[0], j = fedge.v[1];
  //   // 	FlatArray<int> rii = SP.GetRowIndices(i), rij = SP.GetRowIndices(j);
  //   // 	FlatVector<TM> rvi = SP.GetRowValues(i),  rvj = SP.GetRowValues(j);
  //   // 	/** (negative) Off-diagonal entry of fine replacement matrix:
  //   // 	    Aij = -Trans(Qij) * Eij * Qji **/
  //   // 	FACTORY_CLASS::CalcQs(fvd[fedge.v[0]], fvd[fedge.v[1]], Qij, Qji);
  //   // 	auto & emat = fed[fedge.id];
  //   // 	for (auto kj : Range(rij)) {
  //   // 	  auto J = rij[kj];
  //   // 	  auto neibs_J = cecon.GetRowIndices(J);
  //   // 	  auto J_eids = cecon.GetRowValues(J);
  //   // 	  for (auto ki : Range(rii)) {
  //   // 	    auto I = rii[ki];
  //   // 	    auto pos = find_in_sorted_array(I, neibs_J);
  //   // 	    if (pos != -1) {
  //   // 	      FACTORY_CLASS::CalcQs(cvd[I], cvd[J], QIJ, QJI);
  //   // 	      RiI = Qij * rvi[ki] * QJI;
  //   // 	      RjJ = Qji * rvj[kj] * QIJ;
  //   // 	      RS = 0.5 * (RiI + RjJ);
  //   // 	      RE = Trans(RS) * emat;
  //   // 	      R = RE * RS;
  //   // 	      ced[J_eids[pos]] += R;
  //   // 	    }
  //   // 	  }
  //   // 	}
  //   //   }, true); // prolongation is hierarchic, so only master needs to add (cumulated) edge mat to coarse
  //   /** If i am master of I and JI, I am also master of all i,j they prolongate to, so also of all eij! **/
  //   // Table<int> c2fv;
  //   // TM Qij, Qji, QIJ, QJI, BijI, BijJ, B_DIFF;
  //   // TM BEB_SUM, B_SUM, BSI_BS;
  //   // TM X, Y; // ... yes, no name. whatever.
  //   // CM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & cedge) {
  //   // 	cout << " calc cedge " << cedge << endl;
  //   // 	constexpr int N = mat_traits<TM>::HEIGHT;
  //   // 	int I = cedge.v[0], J = cedge.v[1];
  //   // 	FlatArray<int> is   = SPT.GetRowIndices(I), js   = SPT.GetRowIndices(J);
  //   //  	FlatVector<TM> PiIs = SPT.GetRowValues(I),  PjJs = SPT.GetRowValues(J);
  //   // 	cout << "is: "; prow(is); cout << endl;
  //   // 	cout << "js: "; prow(js); cout << endl;
  //   // 	FACTORY_CLASS::CalcQs(cvd[I], cvd[J], QIJ, QJI);
  //   // 	BEB_SUM = 0; B_SUM = 0;
  //   // 	for (auto ki : Range(is)) {
  //   // 	  auto i = is[ki];
  //   // 	  auto neibs_i = fecon.GetRowIndices(i);
  //   // 	  auto i_eids = fecon.GetRowValues(i);
  //   // 	  X = Trans(PiIs[ki]) * QJI;
  //   // 	  for (auto kj : Range(js)) {
  //   // 	    auto j = js[kj];
  //   // 	    auto pos = find_in_sorted_array(j, neibs_i); // find the edge i-j (if it exists)
  //   //  	    if (pos != -1) {
  //   // 	      FACTORY_CLASS::CalcQs(fvd[i], fvd[j], Qij, Qji);
  //   // 	      BijI = Qij * X; // BijI = Qij * PiI * QJI
  //   // 	      Y = Trans(PjJs[kj]) * QIJ;
  //   // 	      BijJ = Qji * Y; // BijJ = Qji * PjJ * QIJ
  //   // 	      B_DIFF = BijI - BijJ; // BijI - BijJ
  //   // 	      B_SUM += B_DIFF;
  //   // 	      X = fed[int(i_eids[pos])] * B_DIFF;
  //   // 	      BEB_SUM += Trans(B_DIFF) * X; // (BijI - BijJ)^T * Eij * (BijI - BijJ)
  //   // 	      Y = Trans(B_DIFF) * X;
  //   // 	      cout << i << " " << j << " "; prt_evv<N>(Y, "BEB contrib", false);
  //   // 	      cout << i << " " << j << "BEB contrib: " << endl;
  //   // 	      print_tm(cout, Y); cout << endl;
  //   // 	    }
  //   // 	  }
  //   // 	}
  //   // 	prt_evv<N>(BEB_SUM, "BEB_SUM", false);
  //   // 	if constexpr(N == 1) {
  //   // 	    CalcInverse(BEB_SUM);
  //   // 	  }
  //   // 	else // actually, no neg evals, but _neg is faster I think
  //   // 	  { CalcPseudoInverse_neg<N>(BEB_SUM); }
  //   // 	prt_evv<N>(BEB_SUM, "inv BEB_SUM", false);
  //   // 	BSI_BS = BEB_SUM * B_SUM;
  //   // 	auto & cemat = ced[cedge.id];
  //   // 	cemat = 0;
  //   // 	for (auto ki : Range(is)) {
  //   // 	  auto i = is[ki];
  //   // 	  auto neibs_i = fecon.GetRowIndices(i);
  //   // 	  auto i_eids = fecon.GetRowValues(i);
  //   // 	  X = Trans(PiIs[ki]) * QJI;
  //   // 	  for (auto kj : Range(js)) {
  //   // 	    auto j = js[kj];
  //   // 	    auto pos = find_in_sorted_array(j, neibs_i); // find the edge i-j (if it exists)
  //   //  	    if (pos != -1) {
  //   // 	      FACTORY_CLASS::CalcQs(fvd[i], fvd[j], Qij, Qji);
  //   // 	      BijI = Qij * X; // BijI = Qij * PiI * QJI
  //   // 	      Y = Trans(PjJs[kj]) * QIJ;
  //   // 	      BijJ = Qji * Y; // BijJ = Qji * PjJ * QIJ
  //   // 	      B_DIFF = BijI - BijJ; // BijI - BijJ
  //   // 	      /** [[]]^T * Eij * [ (I - B_DIFF * BEB_SUM^(-1) * B_SUM) * BijI] **/
  //   // 	      SetIdentity(X);
  //   // 	      X -= B_DIFF * BSI_BS;
  //   // 	      Y = BijI * X;
  //   // 	      X = Trans(Y) * fed[int(i_eids[pos])];
  //   // 	      cemat += X * Y;
  //   // 	      TM contrib = X*Y;
  //   // 	      prt_evv<N>(fed[int(i_eids[pos])], "fine emat", false);
  //   // 	      prt_evv<N>(contrib, "contrib to crs", false);
  //   // 	    }
  //   // 	  }
  //   // 	}
  //   // 	X = Trans(QJI) * cemat;
  //   // 	cemat = X * QJI;
  //   // 	cout << endl << " cedge "; cedge;
  //   // 	prt_evv<N>(old_ced[cedge.id], "old emat", false);
  //   // 	prt_evv<N>(ced[cedge.id], "new emat", false); cout << endl;
  //   //   }, true);
  //   TM Qij, Qji, QIJ, QJI, BijI, BijJ, BijIT_E, BD, E_BD;
  //   TM X, Y;
  //   Matrix<TM> M(2,2);
  //   CM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & cedge) {
  // 	cout << endl << " calc cedge " << cedge << endl;
  //   	auto & cemat = ced[cedge.id];
  //   	constexpr int N = mat_traits<TM>::HEIGHT;
  //   	int I = cedge.v[0], J = cedge.v[1];
  //   	FACTORY_CLASS::CalcQs(cvd[I], cvd[J], QIJ, QJI);
  //   	FlatArray<int> is   = SPT.GetRowIndices(I), js   = SPT.GetRowIndices(J);
  //    	FlatVector<TM> PiIs = SPT.GetRowValues(I),  PjJs = SPT.GetRowValues(J);
  // 	cout << "is: "; prow(is); cout << endl;
  // 	cout << "js: "; prow(js); cout << endl;
  //   	M = TM(0);
  //   	for (auto ki : Range(is)) {
  //   	  auto i = is[ki];
  //   	  auto neibs_i = fecon.GetRowIndices(i);
  //   	  auto i_eids = fecon.GetRowValues(i);
  //   	  X = Trans(PiIs[ki]) * QJI;
  //   	  for (auto kj : Range(js)) {
  //   	    auto j = js[kj];
  //   	    auto pos = find_in_sorted_array(j, neibs_i); // find the edge i-j (if it exists)
  //    	    if (pos != -1) {
  //   	      auto posij = find_in_sorted_array(i, js);
  //   	      auto posji = find_in_sorted_array(j, is);
  // 	      if ( (posji != -1) && (posij != -1) && (i>j) ) // every edge only once!
  // 		{ continue; }
  //   	      FACTORY_CLASS::CalcQs(fvd[i], fvd[j], Qij, Qji);
  //   	      auto & femat = fed[int(i_eids[pos])];
  // 	      cout << "-- contrib " << i << " " << j << endl;
  //   	      BijI = Qij * X; // BijI = Qij * PiI * QJI
  //   	      if (posij != -1) {
  //   		Y = Trans(PjJs[posij]) * QIJ;
  // 		BijI -= Qji * Y;
  //   	      }
  //   	      Y = Trans(PjJs[kj]) * QIJ;
  //   	      BijJ = Qji * Y; // BijJ = Qji * PjJ * QIJ
  //   	      if (posji != -1) {
  //   		Y = Trans(PiIs[posji]) * QJI;
  // 		BijJ -= Qij * Y;
  //   	      }
  //   	      BD = BijI - BijJ; // BijI - BijJ
  // 	      // cout << "BijI: " << endl; print_tm(cout, BijI);
  // 	      // cout << "BijJ: " << endl; print_tm(cout, BijJ);
  // 	      // cout << "BD: " << endl; print_tm(cout, BD);
  //   	      BijIT_E = Trans(BijI) * femat;
  //   	      E_BD = femat * BD;
  //   	      M(0,0) += BijIT_E * BijI;
  //   	      M(0,1) -= BijIT_E * BD; // M(0,1) = Trans(M(1,0))
  //   	      TM m00c = BijIT_E * BijI;
  //   	      // prt_evv<N>(m00c, "---- M00 contrib", false);
  //   	      TM m11c = Trans(BD) * E_BD;
  //   	      // prt_evv<N>(m11c, "---- M11 contrib", false);
  //   	      M(1,1) += Trans(BD) * E_BD;
  //   	      // prt_evv<N>(M(1,1), "---- intermed M11", false);
  //   	      // prt_evv<N>(M(0,0), "---- intermed M00", false);
  //   	    }
  //   	  }
  //   	}
  //   	// prt_evv<N>(M(1,1), "final M11", false);
  //   	if constexpr(N == 1) { // weird indentation if { on next line 
  //   	    CalcInverse(M(1,1)); }
  //   	else // actually, no neg evals, but _neg is faster I think
  //   	  { CalcPseudoInverse_neg<N>(M(1,1)); }
  //   	X = M(0,1) * M(1,1);
  //   	// prt_evv<N>(M(0,0), "M00", false);
  //   	// prt_evv<N>(M(1,1), "inv M11", false);
  //   	// print_tm(cout, M(1,1));
  //   	M(0,0) -= X * Trans(M(0,1));
  //   	// prt_evv<N>(cemat, "old emat", false);
  //   	cemat = M(0,0);
  //   	// prt_evv<N>(cemat, "new emat", false);
  //     }, true);
  //   // const auto FNV = FM.template GetNN<NT_VERTEX>();
  //   // Array<int> perow(FNV);
  //   // for (auto k : Range(perow))
  //   //   { perow[k] = 1 + fecon.GetRowIndices(k).Size(); }
  //   // auto Ahat = make_shared<SparseMatrixTM<TM>>(perow, FNV);
  //   // for (auto k : Range(perow)) {
  //   //   auto ris = Ahat->GetRowIndices(k);
  //   //   auto ecri = fecon.GetRowIndices(k); int c = 0;
  //   //   for (auto j : ecri)
  //   // 	{ ris[c++] = j; }
  //   //   ris[c++] = k;
  //   //   QuickSort(ris);
  //   //   cout << " ris " << k << ": "; prow(Ahat->GetRowIndices(k)); cout << endl;
  //   //   Ahat->GetRowValues(k) = TM(0);
  //   // }
  //   // Matrix<TM> em(2,2);
  //   // FM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & fedge) {
  //   // 	self.CalcRMBlock (FM, fedge, em);
  //   // 	cout << fedge.v[0] << " x " << fedge.v[1] << endl;
  //   // 	(*Ahat)(fedge.v[0], fedge.v[0]) += em(0,0);
  //   // 	(*Ahat)(fedge.v[0], fedge.v[1]) += em(0,1);
  //   // 	(*Ahat)(fedge.v[1], fedge.v[0]) += em(1,0);
  //   // 	(*Ahat)(fedge.v[1], fedge.v[1]) += em(1,1);
  //   //   }, false); // not sure
  //   // auto Ahat_crs = RestrictMatrixTM(SPT, *Ahat, SP);
  //   // const auto & AH = *Ahat_crs;
  //   // // /** Set uI = QHI*wH, calc inf of all uJ such that QIJ uI - QJI uJ = wH **/
  //   // TM QIJ, QJI, X;
  //   // CM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto & cedge) {
  //   //  	constexpr int N = mat_traits<TM>::HEIGHT;
  //   // 	FACTORY_CLASS::CalcQs(cvd[cedge.v[0]], cvd[cedge.v[1]], QIJ, QJI);
  //   //  	// prt_evv<N>(old_ced[cedge.id], "old emat", false);
  //   //  	prt_evv<N>(ced[cedge.id], "old emat", false);
  //   // 	X = - Trans(QJI) * AH(cedge.v[0], cedge.v[1]) * QIJ;
  //   // 	ced[cedge.id] = 0.5 * (Trans(X) + X);
  //   // 	prt_evv<N>(ced[cedge.id], "new emat", false);
  //   // 	print_tm(cout, ced[cedge.id]); cout << endl;
  //   //   }, true);
  // } // VertexBasedAMGFactory::ModCoarseEdata

} // namespace amg

#endif // FILE_AMG_FACTORY_IMPL_HPP
