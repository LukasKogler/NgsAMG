#ifndef FILE_AMG_FACTORY_IMPL_HPP
#define FILE_AMG_FACTORY_IMPL_HPP

namespace amg
{

  /** --- Options --- **/

  template<class TMESH, class TM>
  struct AMGFactory<TMESH> :: Options
  {
    /** Level-control **/
    size_t max_meas = 50;                // maximal maesure of coarsest mesh
    double aaf = 0.1;                    // assemble after factor
    double first_aaf = 0.05;             // (smaller) factor for first level. -1 for dont use
    double aaf_scale = 1;                // scale aaf, e.g if 2:   first_aaf, aaf, 2*aaf, 4*aaf, .. (or aaf, 2*aaf, ...)
    
    /** Discard - not back yet  **/

    /** Contract (Re-Distribute) **/
    bool enable_ctr = true;

    /** Build a new mesh from a coarse level matrix**/
    bool enable_rbm = false;             // probably only necessary on coarse levels
    double rbmaf = 0.01;                 // rebuild mesh after measure decreases by this factor
    double first_rbmaf = 0.005;          // see first_aaf
    double rbmaf_scale = 1;              // see aaf_scale
    std::function<shared_ptr<TMESH>(shared_ptr<TMESH>, shared_ptr<some_spm>)> rebuild_mesh =
      [](shared_ptr<TMESH> mesh, shared_ptr<some_spm> mat) { return mesh; };
  };


  /** --- AMGFactory --- **/


  template<class TMESH, class TM>
  AMGFactory<TMESH, TM> :: AMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<BaseDOFMapStep> _embed_step, shared_ptr<Options> _opts)
    : opts(_opts), finest_mesh(_finest_mesh), embed_step(_embed_step)
  { ; }


  template<class TMESH, class TM>
  Array<shared_ptr<BaseSparseMatrix>> AMGFactory<TMESH> :: RSU (Capsule cap, shared_ptr<DOFMap> dof_map)
  {
    shared_ptr<BaseDOFMapStep> ds;

    shared_ptr<TMESH> fmesh = capsule.mesh, cmesh = capsule.mesh;
    shared_ptr<ParallelDofs> fpds = capsule.pardofs, cpds = nullptr;

    size_t curr_meas = GetMeshMeasure(*fmesh);
    double af = (cap.level == 0) ? options->first_aaf : ( pow(options_aaf_scale, cap.level - (options->first_aaf == -1) ? 1 : 0) * options->aaf );
    size_t goal_meas = max( min(af, 0.9) * curr_meas, max(options->max_meas, 1));

    INT<3> level = { cap.level, 0, 0 }; // coarse / sub-coarse / ctr 

    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    shared_ptr<TSPM_TM> conc_pwp;

    while (curr_meas > goal_meas) {
      while (curr_meas > goal_meas) {

	auto grid_step = BuildCoarseMap (cmesh, (level[0] == level[1] == 0) ? finest_free_verts : nullptr );

	if (grid_step == nullptr)
	  { break; }

	bool non_acceptable = false;
	if (non_acceptable)
	  { break; }

	cmesh = const_pointer_cast<TMESH>(gstep->GetMappedMesh());

	auto pwp = BuildPWProl(gstep, fpds);

	conc_prol = (conc_prol == nullptr) ? pwp : MatMultAB(*conc_prol, *pwp);
	cpds = BuildParallelDofs(fmesh);

	level[1]++;
      }

      fpds = cpds; conc_pwp = nullptr; fmesh = cmesh;
      auto pstep = make_shared<ProlMap<TSPM_TM>> (pwp, fpds, cpds, true);
      SmoothProlongation(pstep, capsule.mesh);

      if (embed_step != nullptr) {
	auto real_step = embed_step->Concatenate(pstep);
	sub_steps.Append(real_step);
	embed_step = nullptr; 
      }
      else {
	sub_steps.Append(pstep);
      }

      // contract here
    }

    shared_ptr<BaseDOFMapStep> tot_step = (sub_steps.Size() > 1) ? make_shared<ConcDMS>(sub_steps) : sub_steps[0];
    
    dof_map->AddStep(tot_step);

    // coarse level matrix
    auto cmat = tot_step->AssembleMatrix(cap.mat);

    if (options->enable_rbm) {
      double af = (cap.level == 0) ? options->first_rbmaf : ( pow(options_rbmaf_scale, cap.level - (options->first_rbmaf == -1) ? 1 : 0) * options->aaf );
      size_t goal_meas = max( min(af, 0.9) * state.last_rbm_meas, max(options->max_meas, 1));
      if (curr_meas < goal_meas)
	{ cmesh = options->rebuild_mesh(cmesh, cmat); }
    }

    bool do_more_levels = (cmesh != nullptr) &&
      (cap.level < options->max_n_levels) &&
      (options->max_meas > GetMeshMeasure (*cmesh) );
    
    if (do_more_levels) {
      // recursively call setup
      cap.level++; cap.mesh = cmesh; cap.mat = cmat; cap.
      auto cmats = RSU( {cap.level + 1, cmesh, cmat, cpds, nullptr}, dof_map );
      cmats.Append(cmat);
      return cmats;
    }
    else {
      // no more coarse levels
      return Array<shared_ptr<BaseSparseMatrix>> ({cmat});
    }

  } // AMGFactory :: RSU


  template<class TMESH, class TM>
  shared_ptr<GridContractMap<TMESH>> AMGFactory<TMESH, TM> :: BuildContractMap (shared_ptr<TMESH> mesh) const
  {
    static Timer t("BuildContractMap"); RegionTimer rt(t);

  }


  /** --- NodalAMGFactory --- **/

  
  /** --- VertexBasedAMGFactory --- **/


  template<class TMESH, class TM>
  shared_ptr<CoarseMap<TMESH>> VertexBasedAMGFactory<TMESH, TM> :: BuildCoarseMap  (shared_ptr<TMESH> mesh) const
  {
    static Timer t("BuildCoarseMap"); RegionTimer rt(t);
    auto coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
    coarsen_opts->free_verts = free_verts;
    SetCoarseningOptions(*coarsen_opts);
    shared_ptr<VWiseCoarsening<TMESH>> calg;
    calg = make_shared<BlockVWC<TMESH>> (coarsen_opts);
    return calg->Coarsen(mesh);
  }

  
} // namespace amg

#endif
