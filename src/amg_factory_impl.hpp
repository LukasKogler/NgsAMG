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
    /** WHEN to contract **/
    double ctraf = 0.05;                 // contract after reducing measure by this factor
    double first_ctraf = 0.025;          // see first_aaf
    double ctraf_scale = 1;              // see aaf_scale
    double ctr_crs_thresh = 0.7;         // if coarsening slows down more than this, contract one step
    /** HOW AGGRESSIVELY to contract **/
    double ctr_pfac = 0.25;              // reduce active NP by this factor (ctr_pfac / ctraf should be << 1 !)
    /** Constraints for contract **/
    size_t ctr_min_nv = 500;             // re-distribute such that at least this many NV per proc remain
    size_t ctr_seq_nv = 500;             // re-distribute to sequential once NV reaches this threshhold
      
    /** Build a new mesh from a coarse level matrix**/
    bool enable_rbm = false;             // probably only necessary on coarse levels
    double rbmaf = 0.01;                 // rebuild mesh after measure decreases by this factor
    double first_rbmaf = 0.005;          // see first_aaf
    double rbmaf_scale = 1;              // see aaf_scale
    std::function<shared_ptr<TMESH>(shared_ptr<TMESH>, shared_ptr<some_spm>)> rebuild_mesh =
      [](shared_ptr<TMESH> mesh, shared_ptr<some_spm> mat) { return mesh; };
  };

  template<NODE_TYPE NT, class TMESH, class TM>
  struct NodalAMGFactory<NT, TMESH, TM> :: Options : public AMGFactory<TMESH, TM>::Options
  {

  };

  template<class TMESH, class TM>
  struct VertexBasedAMGFactory<TMESH, TM> :: Options : public NodalAMGFactory<NT_VERTEX, TMESH, TM>::Options
  {

  };


  /** --- State --- **/


  template<class TMESH, class TM>
  struct AMGFactory<TMESH> :: State
  {
    /** Contract **/
    bool first_ctr_used = false;
    size_t last_nv_ctr;

    /** Rebuild Mesh **/
    bool first_rbm_used = false;
    size_t last_meas_rbm;
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
    shared_ptr<BaseSparseMatrix> cmat = capsule.mat;
    size_t curr_meas = GetMeshMeasure(*fmesh);

    if (cap.level == 0) { // (re-) initialize book-keeping
      state.first_ctr_used = state.first_rbm_used = false;
      state.last_nv_ctr = fmesh->template GetNNGlobal<NT_VERTEX>();
      state.last_meas_rbm = curr_mesh;
    }

    double af = (cap.level == 0) ? options->first_aaf : ( pow(options_aaf_scale, cap.level - (options->first_aaf == -1) ? 1 : 0) * options->aaf );
    size_t goal_meas = max( min(af, 0.9) * curr_meas, max(options->max_meas, 1));

    INT<3> level = { cap.level, 0, 0 }; // coarse / sub-coarse / ctr 

    Array<shared_ptr<BaseDOFMapStep>> sub_steps;
    shared_ptr<TSPM_TM> conc_pwp;

    double crs_meas_fac = 1;

    while (curr_meas > goal_meas) {

      while (curr_meas > goal_meas) {

	auto grid_step = BuildCoarseMap (cmesh, (level[0] == level[1] == 0) ? finest_free_verts : nullptr );

	if (grid_step == nullptr)
	  { break; }


	auto _cmesh = const_pointer_cast<TMESH>(gstep->GetMappedMesh());

	crs_meas_fac = ComputeMeasure(_cmesh) / (1.0 * curr_meas);

	if (crs_meas_fac > 0.95)
	  { throw Exception("no proper check in place here"); }

	bool non_acceptable = false; // i guess implement sth like that at some point ?
	if (non_acceptable)
	  { break; }

	cmesh = _cmesh;

	auto pwp = BuildPWProl(gstep, fpds);

	conc_prol = (conc_prol == nullptr) ? pwp : MatMultAB(*conc_prol, *pwp);
	cpds = BuildParallelDofs(fmesh);

	curr_meas = ComputeMeasure(*cmesh);

	level[1]++;
      } // inner while - we PROBABLY have a coarse map

      if(conc_prol != nullptr) {
	
	fpds = cpds; conc_pwp = nullptr; fmesh = cmesh;
	auto pstep = make_shared<ProlMap<TSPM_TM>> (pwp, fpds, cpds, true);
	SmoothProlongation(pstep, capsule.mesh);

	if (embed_step != nullptr) {
	  auto real_step = embed_step->Concatenate(pstep);
	  sub_steps.Append(real_step);
	  embed_step = nullptr; 
	}
	else { sub_steps.Append(pstep); }

	cmat = sub_steps.Last()->AssembleMatrix(cmat);
      }
      else if (!options->enable_ctr) { // throw exceptions here for now, but in principle just break is also fine i think
	throw Exception("Could not coarsen, and cannot contract (it is disabled)");
	// break;
      }
      else if (cmesh->GetEQCHierarchy()->GetCommunicator().Size() == 2) {
	throw Exception("Could not coarsen, and cannot contract (only 2 NP left!)");
	// break;
      }
      
      if (options->enable_ctr) { // if we are stuck coarsening wise, or if we have reached a threshhold, redistribute
	double af = (cap.level == 0) ? options->first_ctraf : ( pow(options_ctraf_scale, cap.level - (options->first_ctraf == -1) ? 1 : 0) * options->ctraf );
	size_t goal_nv = max( min(af, 0.9) * state.last_ctr_nv, max(options->ctr_seq_nv, 1));
	if ( (crs_meas_fac > options->ctr_crs_thresh) ||
	     (goal_nv > cmesh->GetNNGlobal<NT_VERTEX>() ) ) {

	  double fac = options->ctr_pfac;
	  auto ccomm = cmesh->GetEQCHierarchy()->GetCommunicator();
	  double ctr_factor = (cmesh->GetNNGlobal<NT_VERTEX>() < options->ctr_seq_nv) ? -1 :
	    min2(fac, double(next_nv) / options->ctr_min_nv / ccomm.Size());
	  
	  auto ctr_map = BuildContractMap(ctr_factor, cmesh);

	  cmesh = ctr_map->GetMappedMesh();

	  auto ds = BuildDOFContractMap(ctr_map);
	  sub_steps.Append(ds);

	  cmat = ds->AssembleMatrix(cmat);
	}
      }
      else { break; } // outer loop not useful
    } // outher while

    if (sub_steps.Size() > 0) { // we were able to do some sort of step
      shared_ptr<BaseDOFMapStep> tot_step = (sub_steps.Size() > 1) ? make_shared<ConcDMS>(sub_steps) : sub_steps[0];
      dof_map->AddStep(tot_step);
    }
    
    // coarse level matrix

    if (options->enable_rbm) {
      double af = (cap.level == 0) ? options->first_rbmaf : ( pow(options_rbmaf_scale, cap.level - (options->first_rbmaf == -1) ? 1 : 0) * options->aaf );
      size_t goal_meas = max( min(af, 0.9) * state.last_meas_rbm, max(options->max_meas, 1));
      if (curr_meas < goal_meas)
	{ cmesh = options->rebuild_mesh(cmesh, cmat); state.last_meas_rbm = curr_meas; }
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
    else if (cmat == cap.mat) { // I could not do anyting, stop coarsening here I guess
      return Array<shared_ptr<BaseSparseMatrix>> ({cmat});
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
    int n_groups;

    if (this->ctr_factor == -1 ) { n_groups = 2; }
    else { n_groups = 1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * this->ctr_factor) ; }
    n_groups = max2(2, n_groups); // dont send everything from 1 to 0 for no reason
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);
    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
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
