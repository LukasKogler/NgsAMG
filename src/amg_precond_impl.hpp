#ifndef FILE_AMGPCIMPL
#define FILE_AMGPCIMPL

#include "amg.hpp"

namespace amg
{


  /** NodeWiseAMG **/
  
  template<class TMESH, NODE_TYPE NT, int DPN>
  void NodeWiseAMG<TMESH, NT, DPN> :: Finalize (shared_ptr<BaseMatrix> fine_mat, shared_ptr<BaseDOFMapStep> aembed_step)
  {
    static Timer t(this->GetName()+string("::Finalize")); RegionTimer rt(t);
    finest_mat = fine_mat;
    embed_step = aembed_step;
    Setup();
  }

  template<class TMESH, NODE_TYPE NT, int DPN>
  void NodeWiseAMG<TMESH, NT, DPN> :: Setup ()
  {
    string timer_name = GetName() + string("::Setup");
    static Timer t(timer_name);
    RegionTimer rt(t);
    shared_ptr<TMESH> fm = mesh;
    shared_ptr<ParallelDofs> fm_pd = BuildParDofs(fm);
    NgsAMG_Comm glob_comm = (fm_pd==nullptr) ? NgsAMG_Comm() : NgsAMG_Comm(fm_pd->GetCommunicator());
    auto dof_map = make_shared<DOFMap>();
    Array<INT<3>> levels;
    Array<INT<3>> ass_levels;
    auto MAX_NL = options->max_n_levels;
    auto MAX_NV = options->max_n_verts;
    Array<size_t> nvs;
    nvs.Append(fm->template GetNNGlobal<NT_VERTEX>());
    
    infos = make_shared<Info>(options->info_level, 3*MAX_NL); // over-estimate
    infos->SetPrintInfo(options->print_info, options->info_file);
    
    Array<size_t> cutoffs = {0};
    { // coarsen mesh!
      static Timer t(timer_name + string("-GridMaps")); RegionTimer rt(t);
      shared_ptr<BaseGridMapStep> grid_step;
      shared_ptr<CoarseMap<TMESH>> gstep_coarse;
      shared_ptr<GridContractMap<TMESH>> gstep_contr;
      shared_ptr<BaseDOFMapStep> dof_step;
      INT<3> level = 0; // coarse, contr, elim
      levels.Append(level); ass_levels.Append(level);
      infos->LogMesh(level, fm, true);
      bool contr_locked = true, disc_locked = true;
      size_t last_nv_ass = nvs[0], last_nv_smo = nvs[0], last_nv_ctr = nvs[0];
      size_t step_cnt = 0;
      while ( level[0] < MAX_NL-1 && ( (level[0]==0) || (fm->template GetNNGlobal<NT_VERTEX>() > MAX_NV) ) ) {
	// cout << "MESH level " << level << endl << *fm << endl;
	size_t curr_nv = fm->template GetNNGlobal<NT_VERTEX>();
	if ( (!contr_locked) && (grid_step = (gstep_contr = TryContract(level, fm))) != nullptr ) {
	  contr_locked = true;
	  dof_step = BuildDOFMapStep(level, gstep_contr, fm_pd);
	  last_nv_ctr = curr_nv;
	  level[1]++;
	}
       	else if ( (grid_step = (gstep_coarse = TryCoarsen(level, fm))) != nullptr ) {

	  bool smoothit = true;
	  if (options->enable_sm == false) { /*cout << "1F";*/ smoothit = false; }
	  else if (options->force_sm) { /*cout << "2";*/ smoothit = options->sm_levels.Contains(level[0]); }
	  else if (options->sm_levels.Contains(level[0])) { /*cout << "3T";*/ smoothit = true; }
	  else if (options->sm_skip_levels.Contains(level[0])) { /*cout << "4F";*/ smoothit = false; }
	  else if (level[0] < options->skip_smooth_first) { /*cout << "5F";*/ smoothit = false; }
	  else if (curr_nv > options->smooth_after_frac * last_nv_smo) { /*cout << "6F";*/ smoothit = false; }

	  if (smoothit) { last_nv_smo = curr_nv; }

	  // set sm-func for all prols
	  smoothit |= options->force_composite_smooth;
	  
	  auto prol_step = BuildDOFMapStep(level, gstep_coarse, fm_pd, smoothit);

	  dof_step = prol_step;
	  // infos->LogProl(level, prol_step->GetProl(), smoothit);
       	  level[0]++; level[1] = level[2] = 0;
       	}
	else if ( !disc_locked ) { throw Exception("Discard-Map not yet ported to new Version!"); }
       	else { throw Exception("No map variant worked!"); break; } // all maps failed
	fm = dynamic_pointer_cast<TMESH>(grid_step->GetMappedMesh());
	// cout << "------------------------------" << endl << "coarse mesh: " << endl << *fm << endl << "-------------------------" << endl;
	if (fm != nullptr && fm->template GetNNGlobal<NT_VERTEX>()==0) { // can happen due to discard/vertex ground
	  if (level[0] == 1) // e.g. everywhere diagonally dominant matrix -> only smooth on level 0 (level[0] == 1 after coarsening)
	    { dof_map->AddStep(dof_step); }
	  else if (cutoffs.Last() != step_cnt)
	    { cutoffs.Append(step_cnt); } // 0 is always in cutoffs
	  fm_pd = nullptr; break;
	}
	levels.Append(level);
	dof_map->AddStep(dof_step);
	step_cnt++;
	if (fm == nullptr) { fm_pd = nullptr; cutoffs.Append(step_cnt); break; } // no mesh due to contract
	size_t next_nv = fm->template GetNNGlobal<NT_VERTEX>();
	// fm_pd = BuildParDofs(fm); dof-step should already have coarse pardofs
	fm_pd = dof_step->GetMappedParDofs();
	nvs.Append(next_nv);
	double frac_crs = (1.0*next_nv) / curr_nv;
	bool assit = false;
	// Assemble next level ?
	// cout << "level " << level << ", assemble? ";
	if ( (level[1] != 0) || (level[2] != 0) ) { assit = false; } 
	else if (options->force_ass) { assit = options->ass_levels.Contains(level[0]); }
	else if (options->ass_levels.Contains(level[0]) ) { /*cout << "1Y";*/ assit = true; }
	else if (options->ass_skip_levels.Contains(level[0])) { /*cout << "2F";*/ assit = false; }
	else if (level[0] < options->skip_ass_first) { /*cout << "3F";*/ assit = false; }
	else if (next_nv < options->ass_after_frac * last_nv_ass) { /*cout << "4Y";*/ assit = true; }
	// cout << "  " << assit << endl;
	infos->LogMesh(level, fm, assit);
	if (assit) { ass_levels.Append(level); cutoffs.Append(step_cnt); last_nv_ass = next_nv; }
	// Unlock contract ?
	if ( (level[1] == 0) && (level[2]==0) ) {
	  if (level[0] < options->skip_ctr_first) { contr_locked = true; }
	  else if ( options->ctr_after_frac * last_nv_ctr > next_nv) { contr_locked = false; }
	  else if ( frac_crs > options->ctr_crs_thresh) { contr_locked = false; }
	  if (!contr_locked) {
	    if (next_nv <= options->ctr_seq_nv) { /*cout << "redis to 1 rank" << endl;*/ this->ctr_factor = -1; }
	    else {
	      double fac = options->ctr_pfac;
	      auto ccomm = fm->GetEQCHierarchy()->GetCommunicator();
	      fac = min2(fac, double(next_nv) / options->ctr_min_nv / ccomm.Size());
	      // cout << " next_nv / min_nv : " << double(next_nv) / options->ctr_min_nv << endl;
	      // cout << " ccomn sz: " << ccomm.Size() << endl;
	      // cout << "fac: " << fac << endl;
	      this->ctr_factor = fac;
	    }
	  }
	}
      } // grid-map loop
      if (cutoffs.Last() != step_cnt) // make sure last level is assembled
	{ cutoffs.Append(step_cnt); }
    }
      
    // cout << " cutoffs are: " << endl; prow2(cutoffs, cout); cout << endl;
    
    {
      static Timer t(timer_name + string("-FinalizeDOFMaps")); RegionTimer rt(t);
      dof_map->Finalize(cutoffs, embed_step);
    }

    static Timer tmats(timer_name + string("-AssembleMats"));
    tmats.Start();
    auto mats = dof_map->AssembleMatrices(dynamic_pointer_cast<BaseSparseMatrix>(finest_mat)); // cannot static_cast - virtual inheritance??
    tmats.Stop();

    // cout << "mats: " << endl; prow2(mats, cout); cout << endl;
    
    // {
    //   auto nlevs = dof_map->GetNLevels();
    //   for (auto k : Range(nlevs)) {
    // 	cout << "---" << endl << "dps for level " << k << ":" << endl;
    // 	cout << *dof_map->GetParDofs(k) << endl << "----" << endl;
    //   }
    //   cout << endl;
    // }

    Array<shared_ptr<BaseSmoother>> sms(cutoffs.Size()); sms.SetSize(0);
    {
      static Timer t(timer_name + string("-Smoothers")); RegionTimer rt(t);
      for (auto k : Range(mats.Size()-1)) {
	// cout << "make smoother " << k << "!!" << endl;
	// cout << "mat: " << mats[k] << endl;
	// cout << mats[k]->Height() << " x " << mats[k]->Width() << endl;
	auto pds = dof_map->GetParDofs(k);
	// cout << "pds: " << pds << endl;
	// cout << "ndglob: " << pds->GetNDofGlobal() << endl;
	auto sm = BuildSmoother(ass_levels[k], mats[k], pds, (k==0) ? options->finest_free_dofs : nullptr);
	sm->Finalize();
	sms.Append(sm);
	infos->LogMatSm(mats[k], sms.Last());
      }
    }

    infos->Finalize();

    Array<shared_ptr<const BaseSmoother>> const_sms(sms.Size()); const_sms = sms;
    Array<shared_ptr<const BaseMatrix>> bmats_mats(mats.Size()); bmats_mats = mats;
    this->amg_mat = make_shared<AMGMatrix> (dof_map, const_sms, bmats_mats);

    {
      static Timer t(timer_name + string("-CLevel")); RegionTimer rt(t);
      if (options->clev_type=="INV") {
    	if (mats.Last() != nullptr) {
	  // cout << "COARSE MAT: " << endl << *mats.Last() << endl;
	  auto cpds = dof_map->GetMappedParDofs();
	  auto comm = cpds->GetCommunicator();
	  shared_ptr<BaseSparseMatrix> cspm = static_pointer_cast<BaseSparseMatrix>(mats.Last());
	  // cout << "COARSE MAT: " << endl << *mats.Last() << endl;
	  // cout << "invert " << cspm->Height() << " " << cpds->GetNDofLocal() << endl;
	  cspm = RegularizeMatrix(cspm, cpds);
	  // cout << "invert " << cspm->Height() << " " << cpds->GetNDofLocal() << endl;
	  if (comm.Size()>0) {
	    // cout << "coarse inv " << endl;
	    if constexpr(MAX_SYS_DIM < DPN) {
		throw Exception(string("MAX_SYS_DIM = ") + to_string(MAX_SYS_DIM) + string(", need at least ") +
				to_string(DPN) + string(" for coarsest level exact Inverse!"));
	      }
	    else {
	      shared_ptr<BaseMatrix> cinv;
	      if (cpds->GetCommunicator().Size() <= 2) { // <= 1 if i remove dummy master
		cspm->SetInverseType(SPARSECHOLESKY);
		cinv = cspm->InverseMatrix();
	      }
	      else {
		auto cpm = make_shared<ParallelMatrix>(cspm, cpds);
		cpm->SetInverseType(options->clev_inv_type);
		cinv = cpm->InverseMatrix();
	      }
	      amg_mat->AddFinalLevel(cinv);
	    }
	  }
	  // else {
	  //   auto cinv = mats.Last().Inverse("sparsecholesky");
	  //   amg_mat->AddFinalLevel(cinv);
	  // }
	}
      }
      else if (options->clev_type=="NOTHING") {
	amg_mat->AddFinalLevel(nullptr);
      }
      else {
	throw Exception(string("coarsest level type ")+options->clev_type+string(" not implemented!"));
      }
    }
  }

  template<class TMESH, NODE_TYPE NT, int DPN> shared_ptr<GridContractMap<TMESH>>
  NodeWiseAMG<TMESH, NT, DPN> :: TryContract (INT<3> level, shared_ptr<TMESH> mesh) const
  {
    static Timer t(GetName()+string("::Redistribute")); RegionTimer rt(t);
    if (options->enable_ctr == false) return nullptr;
    if (level[0] == 0) return nullptr; // TODO: if I remove this, take care of contracting free vertices!
    if (level[1] != 0) return nullptr; // dont contract twice in a row
    if (mesh->GetEQCHierarchy()->GetCommunicator().Size()==1) return nullptr; // dont add an unnecessary step
    if (mesh->GetEQCHierarchy()->GetCommunicator().Size()==2) return nullptr; // keep this as long as 0 is seperated
    int n_groups;
    if (this->ctr_factor == -1 ) { n_groups = 2; }
    else { n_groups = 1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * this->ctr_factor) ; }
    n_groups = max2(2, n_groups); // dont send everything from 1 to 0 for no reason
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);
    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
  }

  template<class TMESH, NODE_TYPE NT, int DPN> shared_ptr<BaseDOFMapStep>
  NodeWiseAMG<TMESH, NT, DPN> :: BuildDOFMapStep (INT<3> level, shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    auto fg = cmap->GetGroup();
    Array<int> group(fg.Size()); group = fg;
    Table<int> dof_maps;
    shared_ptr<ParallelDofs> cpd = nullptr;
    if (cmap->IsMaster()) {
      // const TMESH& cmesh(*static_cast<const TMESH&>(*grid_step->GetMappedMesh()));
      shared_ptr<TMESH> cmesh = static_pointer_cast<TMESH>(cmap->GetMappedMesh());
      cpd = BuildParDofs(cmesh);
      Array<int> perow (group.Size()); perow = 0;
      for (auto k : Range(group.Size())) perow[k] = cmap->template GetNodeMap<NT>(k).Size();
      dof_maps = Table<int>(perow);
      for (auto k : Range(group.Size())) dof_maps[k] = cmap->template GetNodeMap<NT>(k);
    }
    auto ctr_map = make_shared<CtrMap<typename strip_vec<Vec<DPN, double>>::type>> (fpd, cpd, move(group), move(dof_maps));
    if (cmap->IsMaster()) {
      ctr_map->_comm_keepalive_hack = cmap->GetMappedEQCHierarchy()->GetCommunicator();
    }
    return move(ctr_map);
  }

  template<class TMESH, NODE_TYPE NT, int DPN> shared_ptr<ParallelDofs>
  NodeWiseAMG<TMESH, NT, DPN> :: BuildParDofs (shared_ptr<TMESH> amesh) const
  {
    static Timer t(this->GetName()+string("::BuildParDofs")); RegionTimer rt(t);
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
    return make_shared<ParallelDofs> (eqc_h.GetCommunicator(), cdps.MoveTable(), DPN, false);
  }

  /** VWiseAMG **/

  template<class AMG_CLASS, class TMESH, int DPN>
  void VWiseAMG<AMG_CLASS, TMESH, DPN> :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts,
								INT<3> level, shared_ptr<TMESH> _mesh) const
  {
    static Timer t(this->GetName()+string("::SetCoarseningOptions")); RegionTimer rt(t);
    const TMESH & mesh(*_mesh);
    const AMG_CLASS & self = static_cast<const AMG_CLASS&>(*this);
    const auto & options = static_cast<const Options&>(*this->options);
    auto NV = mesh.template GetNN<NT_VERTEX>();
    auto NE = mesh.template GetNN<NT_EDGE>();
    mesh.CumulateData();
    Array<double> vcw(NV); vcw = 0;
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	auto ew = self.template GetWeight<NT_EDGE>(mesh, edge);
	vcw[edge.v[0]] += ew;
	vcw[edge.v[1]] += ew;
      }, true);
    mesh.template AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
    mesh.template Apply<NT_VERTEX>([&](auto v) { vcw[v] += self.template GetWeight<NT_VERTEX>(mesh, v); });
    Array<double> ecw(NE);
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
	ecw[edge.id] = self.template GetWeight<NT_EDGE>(mesh, edge) / vw;
      }, false);
    for (auto v : Range(NV))
      vcw[v] = self.template GetWeight<NT_VERTEX>(mesh, v)/vcw[v];
    opts->vcw = move(vcw);
    opts->min_vcw = options.min_vcw;
    opts->ecw = move(ecw);
    opts->min_ecw = options.min_ecw;
  }

  template<class AMG_CLASS, class TMESH, int DPN> shared_ptr<BaseDOFMapStep>
  VWiseAMG<AMG_CLASS, TMESH, DPN> :: BuildDOFMapStep (INT<3> level, shared_ptr<CoarseMap<TMESH>> _cmap, shared_ptr<ParallelDofs> fpd,
						      bool smoothed_prol) const
  {
    const AMG_CLASS & self = static_cast<const AMG_CLASS&>(*this);
    const CoarseMap<TMESH> & cmap(*_cmap);
    const auto & options = static_cast<const Options&>(*this->GetOptions());

    const TMESH & cmesh = static_cast<TMESH&>(*cmap.GetMappedMesh());
    auto cpd = this->BuildParDofs(static_pointer_cast<TMESH>(cmap.GetMappedMesh()));

    auto pwp = this->BuildPWProl (_cmap, fpd);


    bool ispw = (!smoothed_prol) || options.composite_smooth;
    auto pmap = make_shared<ProlMap<TSPM_TM>> (pwp, fpd, cpd, ispw);
    pmap->SetCnt(1);
    this->GetInfo()->LogSMP(level, smoothed_prol);
    pmap->SetLog([this] (auto x) { this->GetInfo()->LogProl(x); });


    if (smoothed_prol) {
      auto fm = static_pointer_cast<TMESH>(_cmap->GetMesh());
      if (options.composite_smooth) {
	pmap->SetSMF ([this, fm](auto x) { SmoothProlongation_hack(x, fm); }, !options.force_composite_smooth);
      }
      else {
	SmoothProlongation(pmap, fm); 
      }
    }

    return pmap;
  }

  template<class AMG_CLASS, class TMESH, int DPN> shared_ptr<typename VWiseAMG<AMG_CLASS, TMESH, DPN>::TSPM_TM>
  VWiseAMG<AMG_CLASS, TMESH, DPN> :: BuildPWProl (shared_ptr<CoarseMap<TMESH>> _cmap, shared_ptr<ParallelDofs> fpd) const
  {
    const AMG_CLASS & self = static_cast<const AMG_CLASS&>(*this);
    const CoarseMap<TMESH> & cmap(*_cmap);
    const auto & options = static_cast<const Options&>(*this->GetOptions());
    const TMESH & fmesh = static_cast<TMESH&>(*cmap.GetMesh()); fmesh.CumulateData();
    const TMESH & cmesh = static_cast<TMESH&>(*cmap.GetMappedMesh()); cmesh.CumulateData();
    auto cpd = this->BuildParDofs(static_pointer_cast<TMESH>(cmap.GetMappedMesh()));

    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();

    // Alloc Matrix
    auto vmap = cmap.template GetMap<NT_VERTEX>();
    Array<int> perow (NV); perow = 0;
    // -1 .. cant happen, 0 .. locally single, 1..locally merged
    // -> cumulated: 0..single, 1+..merged
    Array<int> has_partner (NCV); has_partner = -1;
    for (auto vnr : Range(NV)) {
      auto cvnr = vmap[vnr];
      if (cvnr!=-1) has_partner[cvnr]++;
    }
    cmesh.template AllreduceNodalData<NT_VERTEX, int>(has_partner, [](auto & tab){ return move(sum_table(tab)); });
    for (auto vnr : Range(NV)) { if (vmap[vnr]!=-1) perow[vnr] = 1; }
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
	  self.CalcPWPBlock (fmesh, cmesh, cmap, vnr, cvnr, rv[0]); 
	}
      }
    }

    // cout << "PWP MAT: " << endl;
    // print_tm_spmat(cout, *prol); cout << endl<< endl;

    return prol;
  }


  template<class AMG_CLASS, class TMESH, int DPN> void 
  VWiseAMG<AMG_CLASS, TMESH, DPN> :: SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> afmesh) const
  {
    static Timer t(GetName()+string("::SmoothProlongation")); RegionTimer rt(t);
    static Timer tinv(GetName()+string("::SmoothProlongation - invert"));
    const auto & options = static_cast<const Options&>(*this->GetOptions());

    const AMG_CLASS & self = static_cast<const AMG_CLASS&>(*this);
    const TMESH & fmesh(*afmesh); fmesh.CumulateData();
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch==fine eqch !!
    const TSPM_TM & pwprol = *pmap->GetProl();

    const double MIN_PROL_FRAC = options.min_prol_frac;
    const int MAX_PER_ROW = options.max_per_row;
    const double omega = options.sp_omega;
    // const bool sing_diags = options.singular_diag;
    const bool sing_diags = false; // moved to CalcRMBlock for now..

    // Construct vertex-map from prol (can be concatenated)
    size_t NFV = fmesh.template GetNN<NT_VERTEX>();
    Array<size_t> vmap (NFV); vmap = -1;
    size_t NCV = 0;
    for (auto k : Range(NFV)) {
      auto ri = pwprol.GetRowIndices(k);
      // we should be able to combine smoothing and discarding
      if (ri.Size() > 1) { cout << "TODO: handle this case, dummy" << endl; continue; }
      if (ri.Size() > 1) { throw Exception("comment this out"); }
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
	    // auto com_wt = max2(get_wt(edge.id, edge.v[0]),get_wt(edge.id, edge.v[1]));
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
    // cout << "VW - distributed: " << endl << vw << endl;
    fmesh.template AllreduceNodalData<NT_VERTEX>(vw, [](auto & tab){return move(sum_table(tab)); }, false);
    // cout << "VW - cumulated: " << endl << vw << endl;

    
    /** Find Graph for Prolongation **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    {
      Array<INT<2,double>> trow;
      Array<INT<2,double>> tcv;
      Array<size_t> fin_row;
      for (auto V:Range(NFV)) {
	// if (freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if ( is_invalid(CV) ) continue; // grounded -> TODO: do sth. here if we are free?
	if (vw[V] == 0.0) { // MUST be single
	  // cout << "row " << V << "SINGLE " << endl;
	  perow[V] = 1;
	  graph[V][0] = CV;
	  continue;
	}
	trow.SetSize(0);
	tcv.SetSize(0);
	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
	// cout << "V " << V << " of " << NFV << endl;
	auto ovs = fecon.GetRowIndices(V);
	// cout << "ovs: "; prow2(ovs); cout << endl;
	auto eis = fecon.GetRowValues(V);
	// cout << "eis: "; prow2(eis); cout << endl;
	size_t pos;
	for (auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if (is_invalid(cov) || cov==CV) continue;
	  auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	  if (eqc_h.IsLEQ(EQ, oeq)) {
	    // auto wt = get_wt(eis[j], V);
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
	// cout << "tent row for V " << V << endl; prow2(trow); cout << endl;
	QuickSort(trow, [](const auto & a, const auto & b) {
	    if (a[0]==b[0]) return false;
	    return a[1]>b[1];
	  });
	// cout << "sorted tent row for V " << V << endl; prow2(trow); cout << endl;
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
	// cout << "fin row for V " << V << endl; prow2(fin_row); cout << endl;
	perow[V] = fin_row.Size();
	for (auto j:Range(fin_row.Size()))
	  graph[V][j] = fin_row[j];
	// if (fin_row.Size()==1 && CV==-1) {
	//   cout << "whoops for dof " << V << endl;
	// }
      }
    }
    
    /** Create Prolongation **/
    auto sprol = make_shared<TSPM_TM>(perow, NCV);

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "Tobias", false); // ~2 MB LocalHeap
    Array<INT<2,size_t>> uve(30); uve.SetSize(0);
    Array<int> used_verts(20), used_edges(20);
    TMAT id; SetIdentity(id);
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
		// cout << " valid: " << V << " " << EQ << " // " << ov << " " << eq << endl;
		uve.Append(INT<2>(ov,all_oe[j]));
	      } } } }
	uve.Append(INT<2>(V,-1));
	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];}); // WHY??
	used_verts.SetSize(uve.Size()); used_edges.SetSize(uve.Size());
	for (auto k:Range(uve.Size()))
	  { used_verts[k] = uve[k][0]; used_edges[k] = uve[k][1]; }
	
	// cout << "sprol row " << V << endl;
	// cout << "graph: "; prow2(graph_row); cout << endl;
	// cout << "used_verts: "; prow2(used_verts); cout << endl;
	// cout << "used_edges: "; prow2(used_edges); cout << endl;

	// auto posV = used_verts.Pos(V);
	auto posV = find_in_sorted_array(int(V), used_verts);
	// cout << "posV: " << posV << endl;
      	size_t unv = used_verts.Size(); // # of vertices used
	FlatMatrix<TMAT> mat (1,unv,lh); mat(0, posV) = 0;
	FlatMatrix<TMAT> block (2,2,lh);
	for (auto l:Range(unv)) {
	  if (l==posV) continue;
	  // get_repl(edges[used_edges[l]], block);
	  // cout << "edge: " << all_fedges[used_edges[l]] << endl;
	  self.CalcRMBlock (fmesh, all_fedges[used_edges[l]], block);
	  // cout << "block " << l << ": " << endl;
	  // print_tm_mat(cout, block); cout << endl;
	  int brow = (V < used_verts[l]) ? 0 : 1;
	  // int brow = (V == all_fedges[used_edges[l]].v[0]) ? 0 : 1;
	  mat(0,l) = block(brow,1-brow); // off-diag entry
	  mat(0,posV) += block(brow,brow); // diag-entry
	  // cout << "diag now: " << endl; print_tm(cout, mat(0,posV)); cout << endl;
	}
	// cout << "inv diag V = " << V << endl; print_tm(cout, diag); cout << endl;
	tinv.Start();
	TMAT diag;
	double tr = 1;
	if constexpr(mat_traits<TMAT>::HEIGHT == 1) {
	    diag = mat(0, posV);
	  }
	else {
	  diag = mat(0, posV);
	  tr = 0; Iterate<mat_traits<TMAT>::HEIGHT>([&](auto i) { tr += diag(i.value,i.value); });
	  tr /= mat_traits<TMAT>::HEIGHT;
	  diag /= tr;
	  // mat /= tr;
	  // diag = mat(0, posV);
	  if (sing_diags) {
	    self.RegDiag(diag);
	  }
	}
	// cout << "invert diag: " << endl; print_tm(cout, diag); cout << endl;
	CalcInverse(diag);
	// cout << "inverted diag: " << endl; print_tm(cout, diag); cout << endl;
	tinv.Stop();
	// for ( auto l : Range(unv)) {
	//   TMAT diag2 = mat(0, l);
	//   TMAT diag3 = diag * diag2;
	//   // cout << "diag * block " << l << ": " << endl; print_tm(cout, diag3); cout << endl;
	// }
	// FlatMatrix<TMAT> row (1, unv, lh);
	// // Matrix<TMAT> row (1, unv);
	// row = diag * mat;
	// cout << " diag * row : " << endl; print_tm_mat(cout, row); cout << endl;
	// row = - omega * diag * mat;
	// cout << " - omega * diag * row : " << endl; print_tm_mat(cout, row); cout << endl;
	// row(0, posV) = (1.0-omega) * id;

	// cout << "mat row: " << endl; print_tm_mat(cout, mat); cout << endl;
	// cout << "inv: " << endl; print_tm(cout, diag); cout << endl;
	// cout << " repl-row: " << endl; print_tm_mat(cout, row); cout << endl;
	
	auto sp_ri = sprol->GetRowIndices(V); sp_ri = graph_row;
	auto sp_rv = sprol->GetRowValues(V); sp_rv = 0;
	double fac = omega/tr;
	for (auto l : Range(unv)) {
	  int vl = used_verts[l];
	  auto pw_rv = pwprol.GetRowValues(vl);
	  int cvl = vmap[vl];
	  // cout << "v " << l << ", " << vl << " maps to " << cvl << endl;
	  // cout << "pw-row for vl: " << endl; print_tm(cout, pw_rv[0]); cout << endl;
	  auto pos = find_in_sorted_array(cvl, sp_ri);
	  // cout << "pos is " << pos << endl;
	  // sp_rv[pos] += row(0,l) * pw_rv[0];
	  // cout << " before " << endl; print_tm(cout, sp_rv[pos]); cout << endl;
	  if (l==posV)
	    { sp_rv[pos] += pw_rv[0]; }
	  sp_rv[pos] -= fac * (diag * mat(0,l)) * pw_rv[0];
	  // cout << " after "; print_tm(cout, sp_rv[pos]); cout << endl;
	  
	  // if (l==posV) {
	  //   sp_rv[pos] += (1-omega) * pw_rv[0];
	  //   // cout << " mid "; print_tm(cout, sp_rv[pos]); cout << endl;
	  // }
	  // else {
	  //   // TMAT m1 = mat(0,l);
	  //   // TMAT m2 = diag * m1;
	  //   // TMAT m3 = m2 * pw_rv[0];
	  //   // cout << " should add " << omega << " * "; print_tm(cout, m3); cout << endl;
	  //   sp_rv[pos] += - omega * diag * mat(0,l) * pw_rv[0];
	  //   // cout << " after "; print_tm(cout, sp_rv[pos]); cout << endl;
	  // }
	}
      }
    }

    // cout << "SPROL MAT: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    pmap->SetProl(sprol);
  }

  template<class AMG_CLASS, class TMESH, int DPN> shared_ptr<CoarseMap<typename VWiseAMG<AMG_CLASS, TMESH, DPN>::TMESH>>
  VWiseAMG<AMG_CLASS, TMESH, DPN> :: TryCoarsen  (INT<3> level, shared_ptr<TMESH> mesh) const
  {
    static Timer t(this->GetName()+string("::Coarsening")); RegionTimer rt(t);
    const auto& options = static_cast<const Options&> (*this->GetOptions());
    auto coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
    shared_ptr<VWCoarseningData::Options> basos = coarsen_opts;
    // auto coarsen_opts = make_shared<VWCoarseningData::Options>();
    if (level[0]==0) { coarsen_opts->free_verts = options.free_verts; }
    {
      static Timer t(this->GetName()+string("::SetCoarseningOptions")); RegionTimer rt(t);
      SetCoarseningOptions(basos, level, mesh);
    }
    // BlockVWC<TMESH> bvwc (coarsen_opts);
    // return bvwc.Coarsen(mesh);
    shared_ptr<VWiseCoarsening<TMESH>> calg;
    // if (level[0] % 3 == 0) calg = make_shared<HierarchicVWC<TMESH>> (coarsen_opts);
    // else calg = make_shared<BlockVWC<TMESH>> (coarsen_opts);
    calg = make_shared<BlockVWC<TMESH>> (coarsen_opts);
    return calg->Coarsen(mesh);
  }

  /** EmbedVAMG **/
  
  template<class AMG_CLASS, class HTVD, class HTED>
  EmbedVAMG<AMG_CLASS, HTVD, HTED> :: EmbedVAMG (shared_ptr<BilinearForm> blf, shared_ptr<EmbedVAMG<AMG_CLASS, HTVD, HTED>::Options> opts)
    : Preconditioner(blf, Flags() /*(opts->energy=="ELMAT" ? Flags() : Flags({"not_register_for_auto_update"}))*/), options(opts), bfa(blf),
      node_sort(4), node_pos(4), ht_vertex(nullptr), ht_edge(nullptr)
  {

    // Preconditioner constructor does not set paralleldofs!
    SetParallelDofs(blf->GetTrialSpace()->GetParallelDofs());

    if (options->energy != "ELMAT") {
      /** we might be setting up directly from the assembled matrix.
	  in that case call FinalizeLevel ourselfs **/
      if (auto mp = bfa->GetMatrixPtr()) {
	InitLevel(bfa->GetFESpace()->GetFreeDofs());
	FinalizeLevel(mp.get());
      }
    }
    else { // TODO: this probably does not work with standard Preconditioner interface
      if (auto mp = bfa->GetMatrixPtr()) {
	throw Exception("energy is set to ELMAT, but BLF is already assembled!!");
      }
      /** we are setting up from element matrices - allocate hash-tables
	  FinalizeLevel will be called from BLF-Assemble **/
      auto fes = blf->GetFESpace();
      shared_ptr<FESpace> lofes = fes->LowOrderFESpacePtr();
      if (lofes == nullptr) lofes = fes; // no LO-space
      size_t dof_per_v = 0;
      if (options->block_s.Size()) { // embedding
	for(auto v : options->block_s) dof_per_v += v;
      }
      else { // no embedding
	dof_per_v = mat_traits<typename AMG_CLASS::TV>::HEIGHT;
      }
      //TODO: good enough or not ??
      size_t NV = lofes->GetNDof()/dof_per_v;
      ht_vertex = new HashTable<int, HTVD>(NV);
      //TODO: is this ok??
      ht_edge = new HashTable<INT<2,int>, HTED>(8*NV);
      // TODO: this is a super dirty hack! (need for elmats..)
      if (options->keep_vp) {
      	auto & vpos(node_pos[NT_VERTEX]); vpos.SetSize(ma->GetNV());
      	for (auto k : Range(vpos.Size()))
      	  ma->GetPoint(k,vpos[k]);
      	// cout << "vpos init: " << endl; prow2(vpos); cout << endl;
      }
    }
  }


  template<class AMG_CLASS, class HTVD, class HTED>
  EmbedVAMG<AMG_CLASS, HTVD, HTED> :: EmbedVAMG (shared_ptr<BilinearForm> blf, const Flags & aflags, const string aname)
    : Preconditioner(blf, aflags, aname), options(make_shared<EmbedVAMG<AMG_CLASS, HTVD, HTED>::Options>()), bfa(blf),
      node_sort(4), node_pos(4), ht_vertex(nullptr), ht_edge(nullptr)
  {

    options->energy = aflags.GetStringFlag("ngs_amg_energy", "ALG");
    
    options->edges = aflags.GetStringFlag("ngs_amg_edges", "ALG");

    // Find out which chunk of the matrix we should use for coarsening:
    // We can be explicitely told which part to use.
    // use cases: nodal p2, i guess AMG on component of a composite fespace without seperate matrix assembly
    size_t min_def_dof = aflags.GetNumFlag("ngs_amg_lower", 0);
    size_t max_def_dof = aflags.GetNumFlag("ngs_amg_upper", blf->GetFESpace()->GetNDof());
    // Otherwise default to caorsening on low order space, except if we are explicitely told not to
    if ( (min_def_dof == 0) && (max_def_dof == blf->GetFESpace()->GetNDof()) &&
	 (!aflags.GetDefineFlagX("ngs_amg_lo").IsFalse()) )
      if (auto lospace = blf->GetFESpace()->LowOrderFESpacePtr()) // e.g compound has no LO space
	{ min_def_dof = 0; max_def_dof = lospace->GetNDof(); }

    if ( (min_def_dof != 0) || (max_def_dof != blf->GetFESpace()->GetNDof()) ) {
      options->on_dofs = make_shared<BitArray>(blf->GetFESpace()->GetNDof());
      options->on_dofs->Clear();
      for (auto k : Range(min_def_dof, max_def_dof))
	{ options->on_dofs->Set(k); }
    }

    int ll = aflags.GetNumFlag("ngs_amg_log_level", 0);
    switch (ll) {
    case(0) : { options->info_level = NONE; break; }
    case(1) : { options->info_level = BASIC; options->print_info = true; break; }
    case(2) : { options->info_level = DETAILED; options->print_info = true; break; }
    case(3) : { options->info_level = EXTRA; options->print_info = true; break; }
    default : { break; }
    }

    options->info_file = aflags.GetStringFlag("ngs_amg_log_file", "");
    
    options->mat_ready = aflags.GetDefineFlagX("ngs_amg_mat_ready").IsTrue();

    options->do_test = aflags.GetDefineFlagX("ngs_amg_test").IsTrue();

    ModifyInitialOptions();

    SetParallelDofs(blf->GetTrialSpace()->GetParallelDofs());

    if (options->energy != "ELMAT") {
      /** we might be setting up directly from the assembled matrix.
	  in that case call FinalizeLevel ourselfs **/
      if (auto mp = bfa->GetMatrixPtr()) { // is also not nullptr if we are being used as coarse solver
	if (options->mat_ready) {
	  InitLevel(bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()));
	  FinalizeLevel(mp.get());
	}
      }
    }

    if (options->keep_vp) {
      auto & vpos(node_pos[NT_VERTEX]); vpos.SetSize(ma->GetNV());
      for (auto k : Range(vpos.Size()))
	ma->GetPoint(k,vpos[k]);
      // cout << "vpos init: " << endl; prow2(vpos); cout << endl;
    }
  }


  template<class AMG_CLASS, class HTVD, class HTED>
  EmbedVAMG<AMG_CLASS, HTVD, HTED> :: EmbedVAMG (const PDE & apde, const Flags & aflags, const string aname)
    : Preconditioner(&apde, aflags, aname)
  { throw Exception("PDE-constructor for Ngs-AMG Preconditioners not implemented!"); }


  template<class AMG_CLASS, class HTVD, class HTED>
  EmbedVAMG<AMG_CLASS, HTVD, HTED> :: ~EmbedVAMG ()
  {
    if (ht_vertex != nullptr) delete ht_vertex;
    if (ht_edge   != nullptr) delete ht_edge;
  }
  
  template<class AMG_CLASS, class HTVD, class HTED>
  void EmbedVAMG<AMG_CLASS, HTVD, HTED> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    if (freedofs == nullptr) // postpone to FinalizeLevel
      { return; }

    if (bfa->UsesEliminateInternal()) {
      auto fes = bfa->GetFESpace();
      options->finest_free_dofs = make_shared<BitArray>(*freedofs);
      auto& ofd(*options->finest_free_dofs);
      for (auto k : Range(freedofs->Size()))
	if (ofd.Test(k)) {
	  COUPLING_TYPE ct = fes->GetDofCouplingType(k);
	  if ((ct & CONDENSABLE_DOF) != 0)
	    ofd.Clear(k);
	}
    }
    else
      { options->finest_free_dofs = freedofs; }

  }


  template<class AMG_CLASS, class HTVD, class HTED>
  void EmbedVAMG<AMG_CLASS, HTVD, HTED> :: FinalizeLevel (const BaseMatrix * mat)
  {

    static Timer t(string("EmbedVAMG::FinalizeLevel")); RegionTimer rt(t);

    if (options->sync)
      {
	if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat)) {
	  static Timer t(string("EmbedVAMG::FinalizeLevel - Sync")); RegionTimer rt(t);
	  pmat->GetParallelDofs()->GetCommunicator().Barrier();
	}
      }

    if (options->finest_free_dofs == nullptr) { // dummy freedofs
      options->finest_free_dofs = make_shared<BitArray>(mat->Height());
      options->finest_free_dofs->Set();
    }


    shared_ptr<BaseMatrix> fine_spm;
    if (mat != nullptr)
      { fine_spm = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { fine_spm = bfa->GetMatrixPtr(); }

    finest_mat = fine_spm; // embed-amg finest mat - need parallel matrix!

    // I) bddc is garbage with multidim anyways
    // II) if this comes, back it has to be after parmat-cast
    // if (GetEntryDim(fine_spm.get()) != bfa->GetFESpace()->GetDimension()) { // BDDC converts to SparseMatrix<SCAL>
    //   auto fesdim = bfa->GetFESpace()->GetDimension();
    //   auto matdim = GetEntryDim(fine_spm.get());
    //   if (matdim != 1)
    // 	{ throw Exception("can only convert from double mat"); }
    //   auto pfine_spm = dynamic_cast<SparseMatrix<double>*>(finest_mat.get());
    //   if (!pfine_spm)
    // 	{ throw Exception("can only convert from double mat"); }
    //   Array<int> perow(finest_mat->Height() * matdim / fesdim); perow = 0;
    //   for (auto k : Range(perow.Size()))
    // 	{ perow[k] = pfine_spm->GetRowIndices(fesdim / matdim * k).Size() * matdim / fesdim; }
    //   if (fesdim == 2) {
    // 	if (NgMPI_Comm(MPI_COMM_WORLD).Rank() == 0)
    // 	  { cout << "NGsAMG did not get the expected matrix format, converting!" << endl; }
    // 	auto spm = make_shared<SparseMatrix<Mat<2>>>(perow);
    // 	for (auto bk : Range(perow.Size())) {
    // 	  auto bri = spm->GetRowIndices(bk);
    // 	  auto brv = spm->GetRowValues(bk);
    // 	  auto ris = pfine_spm->GetRowIndices(bk * fesdim);
    // 	  auto rvs1 = pfine_spm->GetRowValues(bk * fesdim);
    // 	  auto rvs2 = pfine_spm->GetRowValues(bk * fesdim + 1);
    // 	  for (auto j : Range(ris.Size() / fesdim)) {
    // 	    bri[j] = ris[2*j] / fesdim;
    // 	    brv[j](0, 0) = rvs1[2*j];
    // 	    brv[j](0, 1) = rvs1[2*j+1];
    // 	    brv[j](1, 0) = rvs2[2*j];
    // 	    brv[j](1, 1) = rvs2[2*j+1];
    // 	  }
    // 	}
    // 	// cout << endl << "converted mat: " << endl << *spm << endl;
    // 	// cout << "heights " << fine_spm->Height() << " " << GetEntryDim(fine_spm.get()) << endl;
    // 	// cout << "heights " << spm->Height() << GetEntryDim(spm.get()) << endl;
    // 	// cout << "heights " << finest_mat->Height() << GetEntryDim(finest_mat.get()) << endl;
    // 	fine_spm = finest_mat = spm;
    //   }
    //   else if (fesdim == 3) {
    // 	throw Exception("unhandled fes dim");
    //   }
    //   else
    // 	{ throw Exception("unhandled fes dim"); }
    //   }
    
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
      { fine_spm = pmat->GetMatrix(); }
    else {
      if ( (GetEntryDim(fine_spm.get())==1) != (AMG_CLASS::DPN==1) )
	{ throw Exception("Not sure if this works..."); }
      Array<int> perow (fine_spm->Height() ); perow = 0;
      Table<int> pds (perow);
      NgsMPI_Comm c(MPI_COMM_WORLD);
      // TODO: directly MPI_Comm, is never cleaned up
      netgen::Array<int> me(1); me[0] = c.Rank();
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : netgen::MyMPI_SubCommunicator(c, me );
      // NOTE: this will have an effect on jaboci-preconditioner (if i ever get around to redoing the hybrid smoothers)
      // fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(pds), GetEntryDim(fine_spm.get()), false));
      fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(pds), GetEntryDim(fine_spm.get()), false));
    }
    
    auto mesh = BuildInitialMesh();
    amg_pc = make_shared<AMG_CLASS>(mesh, options);

    auto embed_step = BuildEmbedding();

    amg_pc->Finalize(fine_spm, embed_step); // VAMG finest mat

    if (options->do_test)
      {
	printmessage_importance = 1; 
	netgen::printmessage_importance = 1; 
	Test();
      }
  }


  template<class AMG_CLASS, class HTVD, class HTED>
  shared_ptr<BlockTM> EmbedVAMG<AMG_CLASS, HTVD, HTED> :: BuildTopMesh ()
  {
    static Timer t(this->GetName() + string("::BuildTopMesh")); RegionTimer rt(t);
    static Timer t1(this->GetName() + string("::BuildTopMesh part 1"));
    static Timer t2(this->GetName() + string("::BuildTopMesh part 1"));
    static Timer t3(this->GetName() + string("::BuildTopMesh part 1"));

    t1.Start();
    auto & O(*options);
    shared_ptr<BlockTM> top_mesh = nullptr;
    auto fpd = finest_mat->GetParallelDofs();
    shared_ptr<EQCHierarchy> eqc_h;
    // if (ma->GetCommunicator().Size() == 1) // second version has an optimization the first does not have!
    //   { eqc_h = make_shared<EQCHierarchy>(fpd, true); }
    // else
    //   { eqc_h = make_shared<EQCHierarchy>(ma, Array<NODE_TYPE>({NT_VERTEX}), true); }
    size_t maxset = 0;
    if ( (O.v_dofs == "NODAL") && (O.on_dofs != nullptr) ) {
      for (auto k : Range(O.on_dofs->Size()))
	if (O.on_dofs->Test(k))
	  { maxset = k+1; }
    }
    else if (O.v_dofs == "VARIABLE")
      { maxset = (O.v_blocks.Size() > 0) ? O.v_blocks[O.v_blocks.Size()-1][0]+1 : 0; }
    else
      { maxset =  fpd->GetNDofLocal(); }

    eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

    t1.Stop();
    t2.Start();
    if (O.edges == "MESH") { // convert Netgen-mesh to AMG-Mesh
      if (O.v_pos == "VERTEX") {
	/** edges to edges **/
	node_sort[0].SetSize(ma->GetNV());
	node_sort[1].SetSize(ma->GetNEdges());
	top_mesh = MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
				    false, node_sort[2], false, node_sort[3]);
	auto & vsort = node_sort[0];
	/** Vertex positions **/
	if (options->keep_vp) {
	  static Timer t(this->GetName() + string("::BuildTopMesh - VPOS")); RegionTimer rt(t);
	  auto & vpos(node_pos[NT_VERTEX]); vpos.SetSize(top_mesh->template GetNN<NT_VERTEX>());
	  for (auto k : Range(vpos.Size()))
	    ma->GetPoint(k,vpos[vsort[k]]);
	}
      }
      else if (O.v_pos == "EDGE") {
	/** actually, not sure how to do this **/
	throw Exception("Sorry, have not implemented this case yet.");
      }
      else if (O.v_pos == "FACE") {
	/** edges through vol-els **/
	throw Exception("Sorry, have not implemented this case yet.");
      }
      else if (O.v_pos == "CELL") {
	/** edges through faces **/
	throw Exception("Sorry, have not implemented this case yet.");
      }
    }
    else if (O.edges == "ELMAT") { // AMG-Mesh top. from hash-tables
      if ( (O.v_dofs == "NODAL") && (O.on_dofs != nullptr) )
	{ throw Exception("unhandled case elmat + sub-block of mat"); }
      else if (O.v_dofs == "VARIABLE")
	{ throw Exception("unhandled case elmat + variable dofs"); }
      top_mesh = make_shared<BlockTM>(eqc_h);
      size_t n_verts = fpd->GetNDofLocal();
      auto & vert_sort = node_sort[NT_VERTEX]; vert_sort.SetSize(n_verts);
      top_mesh->SetVs (n_verts, [&](auto vnr)->FlatArray<int>{return fpd->GetDistantProcs(vnr); },
		       [&vert_sort](auto i, auto j){ vert_sort[i] = j; });
      size_t n_edges = 0; for (auto key_val : *ht_edge) { n_edges++; }
      auto ht_it1 = ht_edge->begin(); // ohno, we have to loop three times haha
      auto ht_it2 = ht_edge->begin();
      auto ht_it3 = ht_edge->begin();
      auto ht_it = &ht_it1; int cntit = 0;
      top_mesh->SetNodes<NT_EDGE>(n_edges, [&](int num){
	  decltype(AMG_Node<NT_EDGE>::v) verts = (**ht_it).first;
	  verts[0] = vert_sort[verts[0]];
	  verts[1] = vert_sort[verts[1]];
	  if (verts[1] < verts[0]) swap(verts[1], verts[0]);
	  if (num == n_edges-1) { if (cntit==0) { ht_it = &ht_it2; cntit++; } else { ht_it = &ht_it3; } }
	  else { ++(*ht_it); }
	  return verts;
	}, [](auto node_num, auto id) { /* do nothing - dont care about edge-sort! */ });
    }
    else { // vertices/edges from matrix (is this even relevant??)

      top_mesh = make_shared<BlockTM>(eqc_h);
      
      size_t n_verts = 0, n_edges = 0;

      auto & vert_sort = node_sort[NT_VERTEX];

      // vertices
      auto set_vs = [&](auto nv, auto v2d) {
	vert_sort.SetSize(nv);
	top_mesh->SetVs (nv, [&](auto vnr)->FlatArray<int> { return fpd->GetDistantProcs(v2d(vnr)); },
			 [&vert_sort](auto i, auto j){ vert_sort[i] = j; });
      };

      // edges 
      auto create_edges = [&](auto v2d, auto d2v) {
	auto traverse_graph = [&](const auto& g, auto fun) { // vertex->dof,  // dof-> vertex
	  for (auto k : Range(n_verts)) {
	    int row = v2d(k); // for find_in_sorted_array
	    auto ri = g.GetRowIndices(row);
	    auto pos = find_in_sorted_array(row, ri); // no duplicates
	    if (pos+1 < ri.Size()) {
	      for (auto col : ri.Part(pos+1)) {
		auto j = d2v(col);
		if (j != -1) {
		  // cout << "dofs " << row << " " << col << " are vertices " << k << " " << j << endl;
		  fun(vert_sort[k],vert_sort[j]);
		}
		// else {
		//   cout << "dont use " << row << " " << j << " with dof " << col << endl;
		// }
	      }
	    }
	  }
	}; // traverse_graph
	auto bspm = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat);
	if (!bspm) { bspm = dynamic_pointer_cast<BaseSparseMatrix>( dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix()); }
	if (!bspm) { throw Exception("could not get BaseSparseMatrix out of finest_mat!!"); }
	n_edges = 0;
	traverse_graph(*bspm, [&](auto vk, auto vj) { n_edges++; });
	Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);
	n_edges = 0;
	traverse_graph(*bspm, [&](auto vk, auto vj) {
	    if (vk < vj) { epairs[n_edges++] = {vk, vj}; }
	    else { epairs[n_edges++] = {vj, vk}; }
	  });
	top_mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) { return epairs[num]; }, // (already v-sorted)
				     [](auto node_num, auto id) { /* dont care about edge-sort! */ });
      }; // create_edges

      if (O.v_dofs == "NODAL") {
	int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	const auto bs0 = O.block_s[0]; // is this not kind of redundant ?
	const auto fes_bs = fpd->GetEntrySize();
	if (O.on_dofs == nullptr) { // coarsening all dofs
	  n_verts = fpd->GetNDofLocal() * fes_bs / dpv;
	  auto d2v = [&](auto d) -> int { return ( (d%(fes_bs/bs0)) == 0) ? d*fes_bs/bs0 : -1; };
	  auto v2d = [&](auto v) { return bs0/fes_bs * v; };
	  set_vs (n_verts, v2d);
	  create_edges ( v2d , d2v );
	}
	else { // coarsening on a subset
	  // cout << "on_dofs: " << endl << *O.on_dofs << endl;
	  n_verts = O.on_dofs->NumSet() * fes_bs / dpv;
	  // cout << O.on_dofs->NumSet() << " * " << fes_bs << " / " << dpv << " = " << n_verts << endl;
	  Array<int> first_dof(n_verts);
	  Array<int> compress(maxset); compress = -1;
	  // cout << "maxset: " << maxset << endl;
	  for (size_t k = 0, j = 0; k < n_verts; j += bs0 / fes_bs)
	    if (O.on_dofs->Test(j))
	      { first_dof[k] = j; compress[j] = k++; }
	  auto v2d = [&](auto v) { return first_dof[v]; };
	  auto d2v = [&](auto d) -> int { return (d+1 > compress.Size()) ? -1 : compress[d]; };
	  set_vs (n_verts, v2d);
	  create_edges ( v2d , d2v );
	}

	if (NgMPI_Comm(MPI_COMM_WORLD).Rank() == 1) {
	  cout << "AMG performed on " << n_verts << " vertices, ndof local is: " << fpd->GetNDofLocal() << endl;
	  cout << "free dofs " << options->finest_free_dofs->NumSet() << " ndof local is: " << fpd->GetNDofLocal() << endl;
	}

      }
      else {
	auto& vblocks = O.v_blocks;
	n_verts = vblocks.Size();
	auto v2d = [&](auto v) { return vblocks[v][0]; };
	Array<int> compress(vblocks[n_verts-1][0]); compress = -1;
	for (auto k : Range(compress.Size())) { compress[v2d(k)] = k; }
	auto d2v = [&](auto d) -> int { return (d+1 > compress.Size()) ? -1 : compress[d]; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
      }
    }
    t2.Stop();
    t3.Start();

    /** Convert FreeDofs to FreeVerts (should probably do this above where I have v2d mapping1)**/
    static Timer tfd(this->GetName() + string("::BuildTopMesh - FDS")); RegionTimer rtfd(tfd);
    // TODO:: THIS NEEDS TO WORK WITH on_dofs // v_blocks!!
    auto fvs = make_shared<BitArray>(top_mesh->GetNN<NT_VERTEX>()); fvs->Clear();
    auto & vsort = node_sort[NT_VERTEX];
    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>())) if (O.finest_free_dofs->Test(k)) { fvs->Set(vsort[k]); }
    O.free_verts = fvs;
    t3.Stop();
    return top_mesh;
  }

} // namespace amg

#endif
