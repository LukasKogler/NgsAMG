#ifndef FILE_AMGPCIMPL
#define FILE_AMGPCIMPL

#include "amg.hpp"

namespace amg
{
  template<class AMG_CLASS>
  EmbedVAMG<AMG_CLASS> :: EmbedVAMG (shared_ptr<BilinearForm> blf, shared_ptr<EmbedVAMG<AMG_CLASS>::Options> opts)
    : Preconditioner(blf, Flags({"not_register_for_auto_update"})), options(opts), bfa(blf), fes(blf->GetFESpace())
  {
    Setup();
  }

  template<class AMG_CLASS, class TMESH, class TMAT>
  void VWiseAMG<AMG_CLASS, TMESH, TMAT> :: Finalize (shared_ptr<BaseMatrix> fine_mat, shared_ptr<BaseDOFMapStep> aembed_step)
  {
    finest_mat = fine_mat;
    embed_step = aembed_step;
    Setup();
  }

  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<ParallelDofs> 
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: BuildParDofs (shared_ptr<TMESH> amesh)
  {
    const auto & mesh = *amesh;
    const auto & eqc_h = *mesh.GetEQCHierarchy();
    size_t neqcs = eqc_h.GetNEQCS();
    size_t ndof = mesh.template GetNN<NT_VERTEX>();
    TableCreator<int> cdps(ndof);
    // TODO: this can be done a bit more efficiently
    for (; !cdps.Done(); cdps++) {
      for (auto eq : Range(neqcs)) {
	auto dps = eqc_h.GetDistantProcs(eq);
	auto verts = mesh.template GetENodes<NT_VERTEX>(eq);
	for (auto vnr : verts) {
	  for (auto p:dps) cdps.Add(vnr, p);
	}
      }
    }
    // auto pdt = cdps.MoveTable()
    // cout << "pd-tab: " << endl << pdt << endl;
    return make_shared<ParallelDofs> (eqc_h.GetCommunicator(), cdps.MoveTable(), /*move(pdt), */mat_traits<TV>::HEIGHT, false);
  }

  template<class AMG_CLASS, class TMESH, class TMAT>
  void VWiseAMG<AMG_CLASS, TMESH, TMAT> :: Setup ()
  {
    string timer_name = this->name + " Setup";
    static Timer t(timer_name);
    RegionTimer rt(t);
    shared_ptr<TMESH> fm = mesh;
    shared_ptr<ParallelDofs> fm_pd = BuildParDofs(fm);
    NgsAMG_Comm glob_comm = (fm_pd==nullptr) ? NgsAMG_Comm() : NgsAMG_Comm(fm_pd->GetCommunicator());
    auto grid_map = make_shared<GridMap>();
    shared_ptr<BaseGridMapStep> grid_step;
    shared_ptr<CoarseMap<TMESH>> gstep_coarse;
    shared_ptr<GridContractMap<TMESH>> gstep_contr;
    auto dof_map = make_shared<DOFMap>();
    shared_ptr<BaseDOFMapStep> dof_step;
    Array<INT<3>> levels;
    auto MAX_NL = options->max_n_levels;
    auto MAX_NV = options->max_n_verts;
    bool contr_locked = true, disc_locked = true;
    int cnt_lc = 0;
    Array<size_t> nvs;
    nvs.Append(fm->template GetNNGlobal<NT_VERTEX>());
    double frac_coarse = 0.0;
    const double contr_after_frac = options->contr_after_frac;
    size_t nv_lc = nvs[0];
    
    { // coarsen mesh!
      INT<3> level = 0; // coarse, contr, elim
      levels.Append(level);
      while ( level[0] < MAX_NL-1 && fm->template GetNNGlobal<NT_VERTEX>()>MAX_NV) {

	cout << "now level " << level << endl;
	
	if ( (!contr_locked) && (grid_step = (gstep_contr = TryContract(level, fm))) != nullptr ) {
	  contr_locked = true;
	  dof_step = BuildDOFMapStep(gstep_contr, fm_pd);
	  cout << "HAVE CONTR STEP!!" << endl;
	  level[1]++;
	}
       	else if ( (grid_step = (gstep_coarse = TryCoarsen(level, fm))) != nullptr ) {
	  cnt_lc++;
	  dof_step = BuildDOFMapStep(gstep_coarse, fm_pd);
	  if (level[0]==0 && embed_step!=nullptr) {
	    // cout << "embst: " << embed_step << endl;
	    // cout << "dof s " << dof_step << endl;
	    // cout << "concatenate embedding + first ProlStep!!" << endl;
	    dof_step = embed_step->Concatenate(dof_step);
	  }
       	  level[0]++; level[1] = level[2] = 0;
       	}
       	else { cout << "warning, no map variant worked!" << endl; break; } // all maps failed

	grid_map->AddStep(grid_step);
	dof_map->AddStep(dof_step);

	auto NV = fm->template GetNNGlobal<NT_VERTEX>();
	fm = dynamic_pointer_cast<TMESH>(grid_step->GetMappedMesh());

	
	if (fm==nullptr) { cout << "dropped out, break loop!" << endl; break; } // no mesh due to contract

	cout << "mesh for level " << level << endl << *fm << endl;

	auto CNV = fm->template GetNNGlobal<NT_VERTEX>();
	if (fm->GetEQCHierarchy()->GetCommunicator().Rank()==0) {
	  double fac = (NV==0) ? 0 : (1.0*CNV)/NV;
	  cout << "map NV " << NV << " -> " << CNV << ", factor " <<  fac << endl;
	}
	fm_pd = dof_step->GetMappedParDofs();

	nvs.Append((fm!=nullptr ? fm->template GetNNGlobal<NT_VERTEX>() : 0));
	if (level[1]==0 && level[2]==0) frac_coarse = (1.0*nvs.Last())/(1.0*nvs[nvs.Size()-2]);
	if(level[1]!=0) nv_lc = nvs.Last();
	
	if(cnt_lc>3) // we have not contracted for a long time
	  contr_locked = false;
	else if(frac_coarse>0.75 && level[0]>2 && cnt_lc>1) // coarsening is slowing down
	  contr_locked = false;
	// else if(nvs.Last()/fpd->GetCommunicator().Size()<MIN_V_PP && cnt_lc>1) // too few verts per proc
	//   contr_locked = false;
	else if((1.0*nvs.Last())/nv_lc < contr_after_frac && cnt_lc>1) // if NV reduces by a good factor
	  contr_locked = false;
	// if(level[0]<5) contr_locked = true;
	
      }
    }

    cout << "mesh-loop done, enter barrier!" << endl;
    glob_comm.Barrier();
    cout << "mesh-loop done, barrier done!" << endl;

    // cout << "finest level mat: " << finest_mat << endl;
    // cout << "type " << typeid(*finest_mat).name() << endl;
    auto fmat = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat);
    // cout << "fmat: " << fmat << endl;
    // cout << "type " << typeid(*fmat).name() << endl;
    auto mats = dof_map->AssembleMatrices(fmat);

    {
      auto nlevs = dof_map->GetNLevels();
      for (auto k : Range(nlevs)) {
	// cout << "---" << endl << "dps for level " << k << ":" << endl;
    	// cout << *dof_map->GetParDofs(k) << endl << "----" << endl;
      }
      cout << endl;
    }

    Array<shared_ptr<BaseSmoother>> sms;
    for (auto k : Range(mats.Size()-1)) {
      // cout << "make smoother!!" << endl;
      auto pds = dof_map->GetParDofs(k);
      shared_ptr<const TSPMAT> mat = dynamic_pointer_cast<TSPMAT>(mats[k]);
      sms.Append(make_shared<HybridGSS<mat_traits<TV>::HEIGHT>>(mat,pds,(k==0) ? options->finest_free_dofs : nullptr));
    }
    
    cout << "make AMG-mat!" << endl;
    Array<shared_ptr<const BaseSmoother>> const_sms(sms.Size()); const_sms = sms;
    Array<shared_ptr<const BaseMatrix>> bmats_mats(mats.Size()); bmats_mats = mats;
    amg_mat = make_shared<AMGMatrix> (dof_map, const_sms, bmats_mats);
    cout << "have AMG-mat!" << endl;
    
    if (options->clev_type=="inv") {
      if (mats.Last()!=nullptr) {
	auto max_l = mats.Size();
	auto cpds = dof_map->GetMappedParDofs();
	auto comm = cpds->GetCommunicator();
	if (comm.Size()>0) {
	  cout << "coarse inv " << endl;
	  auto cpm = make_shared<ParallelMatrix>(mats.Last(), cpds);
	  cpm->SetInverseType(options->clev_inv_type);
	  auto cinv = cpm->InverseMatrix();
	  cout << "coarse inv done" << endl;
	  amg_mat->AddFinalLevel(cinv);
	}
	// else {
	//   auto cinv = mats.Last().Inverse("sparsecholesky");
	//   amg_mat->AddFinalLevel(cinv);
	// }
      }
    }
    else if (options->clev_type=="nothing") {
      amg_mat->AddFinalLevel(nullptr);
    }
    else {
      throw Exception(string("coarsest level type ")+options->clev_type+string(" not implemented!"));
    }
    
    cout << "AMG LOOP DONE, enter barrier" << endl;
    glob_comm.Barrier();
    cout << "AMG LOOP DONE" << endl;
    
  }

  template<class AMG_CLASS, class TMESH, class TMAT>
  shared_ptr<ProlMap<typename VWiseAMG<AMG_CLASS, TMESH, TMAT>::TSPMAT>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: BuildDOFMapStep (shared_ptr<CoarseMap<TMESH>> _cmap, shared_ptr<ParallelDofs> fpd)
  {
    // coarse ParallelDofs
    const CoarseMap<TMESH> & cmap(*_cmap);
    const TMESH & fmesh = static_cast<TMESH&>(*cmap.GetMesh());
    const TMESH & cmesh = static_cast<TMESH&>(*cmap.GetMappedMesh());
    const AMG_CLASS& self = static_cast<const AMG_CLASS&>(*this);
    auto cpd = BuildParDofs(static_pointer_cast<TMESH>(cmap.GetMappedMesh()));
    // prolongation Matrix
    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();
    // cout << "DOF STEP, fmesh " << fmesh << endl;
    // cout << "DOF STEP, cmesh " << cmesh << endl;
    auto vmap = cmap.template GetMap<NT_VERTEX>();
    Array<int> perow (NV); perow = 0;
    // -1 .. cant happen, 0 .. locally single, 1..locally merged
    // -> cumulated: 0..single, 1+..merged
    Array<int> has_partner (NCV); has_partner = -1;
    for (auto vnr : Range(NV)) {
      auto cvnr = vmap[vnr];
      if (cvnr!=-1) has_partner[cvnr]++;
    }
    // cout << "sync partner" << endl; prow2(has_partner); cout << endl;
    cmesh.template AllreduceNodalData<NT_VERTEX, int>(has_partner, [](auto & tab){ return move(sum_table(tab)); });
    // cout << "partner synced" << endl;
    for (auto vnr : Range(NV)) { if (vmap[vnr]!=-1) perow[vnr] = 1; }
    auto prol = make_shared<TSPMAT>(perow, NCV);
    for (auto vnr : Range(NV)) {
      if (vmap[vnr]!=-1) {
	auto ri = prol->GetRowIndices(vnr);
	auto rv = prol->GetRowValues(vnr);
	auto cvnr = vmap[vnr];
	ri[0] = cvnr;
	if (has_partner[cvnr]==0) {
	  // single vertex
	  SetIdentity(rv[0]);
	}
	else {
	  // merged vertex
	  self.CalcPWPBlock (fmesh, cmesh, cmap, vnr, cvnr, rv[0]); 
	}
      }
    }
    // cout << "have pw-prol: " << endl << *prol << endl;

    auto pmap = make_shared<ProlMap<TSPMAT>> (prol, fpd, cpd);

    cout << "smooth prol..." << endl;
    SmoothProlongation(pmap, static_pointer_cast<TMESH>(cmap.GetMesh()));
    cout << "smooth prol done!" << endl;
    
    return pmap;
  } // VWiseAMG<...> :: BuildDOFMapStep ( CoarseMap )

  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<CtrMap<typename VWiseAMG<AMG_CLASS, TMESH, TMAT>::TV>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: BuildDOFMapStep (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd)
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
      for (auto k : Range(group.Size())) perow[k] = cmap->template GetNodeMap<NT_VERTEX>(k).Size();
      dof_maps = Table<int>(perow);
      for (auto k : Range(group.Size())) dof_maps[k] = cmap->template GetNodeMap<NT_VERTEX>(k);
    }
    auto ctr_map = make_shared<CtrMap<TV>> (fpd, cpd, move(group), move(dof_maps));
    if (cmap->IsMaster()) {
      ctr_map->_comm_keepalive_hack = cmap->GetMappedEQCHierarchy()->GetCommunicator();
    }
    return move(ctr_map);
  } // VWiseAMG<...> :: BuildDOFMapStep ( GridContractMap )

  
  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<CoarseMap<TMESH>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: TryCoarsen  (INT<3> level, shared_ptr<TMESH> mesh)
  {
    auto coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
    shared_ptr<VWCoarseningData::Options> basos = coarsen_opts;
    // auto coarsen_opts = make_shared<VWCoarseningData::Options>();
    if (level[0]==0) { coarsen_opts->free_verts = options->free_verts; }
    SetCoarseningOptions(basos, level, mesh);
    // BlockVWC<TMESH> bvwc (coarsen_opts);
    // return bvwc.Coarsen(mesh);
    HierarchicVWC<TMESH> hvwc (coarsen_opts);
    return hvwc.Coarsen(mesh);
  }

  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<GridContractMap<TMESH>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: TryContract (INT<3> level, shared_ptr<TMESH> mesh)
  {
    if (level[0] == 0) return nullptr; // TODO: if I remove this, take care of contracting free vertices!
    if (level[1] != 0) return nullptr; // dont contract twice in a row
    if (mesh->GetEQCHierarchy()->GetCommunicator().Size()==1) return nullptr; // dont add an unnecessary step
    if (mesh->GetEQCHierarchy()->GetCommunicator().Size()==2) return nullptr; // keep this as long as 0 is seperated
    int n_groups = 1 + (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1)/3; // 0 is extra
    n_groups = max2(2, n_groups); // dont send everything from 1 to 0 for no reason
    Table<int> groups = PartitionProcsMETIS (*mesh, n_groups);
    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
  }

  template<class AMG_CLASS, class TMESH, class TMAT> void
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: SmoothProlongation (shared_ptr<ProlMap<TSPMAT>> pmap, shared_ptr<TMESH> afmesh) const
  {
    // cout << "SmoothProlongation" << endl;
    // cout << "fecon-ptr: " << afmesh->GetEdgeCM() << endl;
    // cout << "mesh at " << afmesh << endl;
    const AMG_CLASS & self = static_cast<const AMG_CLASS&>(*this);
    const TMESH & fmesh(*afmesh);
    const auto & fecon = *fmesh.GetEdgeCM();
    const auto & eqc_h(*fmesh.GetEQCHierarchy()); // coarse eqch==fine eqch !!
    const TSPMAT & pwprol = *pmap->GetProl();

    // cout << "fmesh: " << endl << fmesh << endl;

    // cout << "fecon: " << endl << fecon << endl;
    
    // const double MIN_PROL_WT = options->min_prol_wt;
    const double MIN_PROL_FRAC = options->min_prol_frac;
    const int MAX_PER_ROW = options->max_per_row;
    const double omega = options->sp_omega;

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
    // cout << "vmap" << endl; prow2(vmap); cout << endl;
    
    // For each fine vertex, sum up weights of edges that connect to the same CV
    //  (can be more than one edge, if the pw-prol is concatenated)
    // TODO: for many concatenated pws, does this dominate edges to other agglomerates??
    auto all_fedges = fmesh.template GetNodes<NT_EDGE>();
    Array<double> vw (NFV); vw = 0;
    auto neqcs = eqc_h.GetNEQCS();
    {
      INT<2, int> cv;
      auto doit = [&](auto the_edges) {
	for(const auto & edge : the_edges) {
	  if( ((cv[0]=vmap[edge.v[0]]) != -1 ) &&
	      ((cv[1]=vmap[edge.v[1]]) != -1 ) &&
	      (cv[0]==cv[1]) ) {
	    // auto com_wt = max2(get_wt(edge.id, edge.v[0]),get_wt(edge.id, edge.v[1]));
	    auto com_wt = self.EdgeWeight(fmesh, edge);
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
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1;
    Array<int> perow(NFV); perow = 0; // 
    {
      Array<INT<2,double>> trow;
      Array<INT<2,double>> tcv;
      Array<size_t> fin_row;
      for(auto V:Range(NFV)) {
	// if(freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if (CV == -1) continue; // grounded -> TODO: do sth. here if we are free?
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
	for(auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if(cov==-1 || cov==CV) continue;
	  auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	  if(eqc_h.IsLEQ(EQ, oeq)) {
	    // auto wt = get_wt(eis[j], V);
	    auto wt = self.EdgeWeight(fmesh, all_fedges[eis[j]]);
	    if( (pos = tcv.Pos(cov)) == -1) {
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
	    if(a[0]==b[0]) return false;
	    return a[1]>b[1];
	  });
	// cout << "sorted tent row for V " << V << endl; prow2(trow); cout << endl;
	double cw_sum = (CV!=-1) ? vw[V] : 0.0;
	fin_row.SetSize(0);
	if(CV != -1) fin_row.Append(CV); //collapsed vertex
	size_t max_adds = (CV!=-1) ? min2(MAX_PER_ROW-1, int(trow.Size())) : trow.Size();
	for(auto j:Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if(CV!=-1) {
	    // I don't think I actually need this: Vertex is collapsed to some non-weak (not necessarily "strong") edge
	    // therefore the relative weight comparison should eliminate all really weak connections
	    // if(fin_row.Size() && (trow[j][1] < MIN_PROL_WT)) break; 
	    if(trow[j][1] < MIN_PROL_FRAC*cw_sum) break;
	  }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	// cout << "fin row for V " << V << endl; prow2(fin_row); cout << endl;
	perow[V] = fin_row.Size();
	for(auto j:Range(fin_row.Size()))
	  graph[V][j] = fin_row[j];
	// if(fin_row.Size()==1 && CV==-1) {
	//   cout << "whoops for dof " << V << endl;
	// }
      }
    }
    
    /** Create Prolongation **/
    auto sprol = make_shared<TSPMAT>(perow, NCV);

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "Tobias", false); // ~2 MB LocalHeap
    Array<INT<2,size_t>> uve(30); uve.SetSize(0);
    Array<int> used_verts(20), used_edges(20);
    TMAT id; SetIdentity(id);
    for(int V:Range(NFV)) {
      auto CV = vmap[V];
      if (CV == -1) continue; // grounded -> TODO: do sth. here if we are free?
      if (perow[V] == 1) { // SINGLE or no good connections avail.
	sprol->GetRowIndices(V)[0] = CV;
	SetIdentity(sprol->GetRowValues(V)[0]);
      }
      else { // SMOOTH
	HeapReset hr(lh);
	// Find which fine vertices I can include
	auto EQ = fmesh.template GetEqcOfNode<NT_VERTEX>(V);
	auto graph_row = graph[V];
	auto all_ov = fecon.GetRowIndices(V);
	auto all_oe = fecon.GetRowValues(V);
	uve.SetSize(0);
	for(auto j:Range(all_ov.Size())) {
	  auto ov = all_ov[j];
	  auto cov = vmap[ov];
	  if(cov != -1) {
	    if(graph_row.Contains(cov)) {
	      auto eq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	      if(eqc_h.IsLEQ(EQ, eq)) {
		// cout << " valid: " << V << " " << EQ << " // " << ov << " " << eq << endl;
		uve.Append(INT<2>(ov,all_oe[j]));
	      } } } }
	uve.Append(INT<2>(V,-1));
	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];}); // WHY??
	used_verts.SetSize(uve.Size()); used_edges.SetSize(uve.Size());
	for(auto k:Range(uve.Size()))
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
	for(auto l:Range(unv)) {
	  if(l==posV) continue;
	  // get_repl(edges[used_edges[l]], block);
	  // cout << "block " << l << " with edge " << used_edges[l] << " " << all_fedges[used_edges[l]] << endl;
	  self.CalcRMBlock (fmesh, all_fedges[used_edges[l]], block);
	  // cout << "block " << l << endl << block << endl;
	  int brow = (V < used_verts[l]) ? 0 : 1;
	  mat(0,l) = block(brow,1-brow); // off-diag entry
	  mat(0,posV) += block(brow,brow); // diag-entry
	}

	TMAT diag = mat(0, posV);
	CalcInverse(diag); // TODO: can this be singular (with embedding?)
	FlatMatrix<double> row (1, unv, lh);
	row = - omega * diag * mat;
	// cout << " repl-row without diag adj: " << endl << row << endl;
	row(0, posV) = (1-omega) * id;

	// cout << "mat: " << endl << mat << endl;
	// cout << "inv: " << endl << diag << endl;
	// cout << " repl-row: " << endl << row << endl;
	
	auto sp_ri = sprol->GetRowIndices(V); sp_ri = graph_row;
	auto sp_rv = sprol->GetRowValues(V); sp_rv = 0;
	for (auto l : Range(unv)) {
	  int vl = used_verts[l];
	  auto pw_rv = pwprol.GetRowValues(vl);
	  int cvl = vmap[vl];
	  // cout << "v " << l << ", " << vl << " maps to " << cvl << endl;
	  // cout << "pw-row for vl: " << endl; prow(pw_rv); cout << endl;
	  auto pos = find_in_sorted_array(cvl, sp_ri);
	  // cout << "pos is " << pos << endl;
	  sp_rv[pos] += row(0,l) * pw_rv[0];
	}
      }
    }

    // cout << "smoothed: " << endl << *sprol << endl;

    pmap->SetProl(sprol);
    
  }


  template<class AMG_CLASS> shared_ptr<BlockTM> 
  EmbedVAMG<AMG_CLASS> :: BuildTopMesh ()
  {
    // Array<Array<int>> node_sort(4);
    node_sort.SetSize(4);
    if (options->v_pos == "VERTEX") {
      auto pds = fes->GetParallelDofs();
      auto eqc_h = make_shared<EQCHierarchy>(pds, true);
      node_sort[0].SetSize(ma->GetNV());
      node_sort[1].SetSize(ma->GetNEdges());
      // node_sort[2].SetSize(ma->GetNFaces());
      auto top_mesh = MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
				       false, node_sort[2], false, node_sort[3]);
      auto & vsort = node_sort[0];
      cout << "v-sort: "; prow2(vsort); cout << endl;
      auto fes_fds = fes->GetFreeDofs();
      // cout << "fes fds: " << endl << *fes_fds << endl;
      auto fvs = make_shared<BitArray>(ma->GetNV());
      fvs->Clear();
      for (auto k : Range(ma->GetNV())) if (fes_fds->Test(k)) { fvs->Set(vsort[k]); }
      options->free_verts = fvs;
      options->finest_free_dofs = fes_fds;
      cout << "init free vertices: " << fvs->NumSet() << " of " << fvs->Size() << endl;
      return top_mesh;
    }
    return nullptr;
  }


  template<class AMG_CLASS> void EmbedVAMG<AMG_CLASS> :: FinalizeLevel (const BaseMatrix * mat)
  {
    if (finest_mat==nullptr) { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    Setup();
  }
  
  template<class AMG_CLASS> void EmbedVAMG<AMG_CLASS> :: Setup ()
  {
    auto mesh = BuildInitialMesh();
    amg_pc = make_shared<AMG_CLASS>(mesh, options);
    auto fmat = (finest_mat==nullptr) ? bfa->GetMatrixPtr() : finest_mat;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fmat))
      fmat = pmat->GetMatrix();
    if (finest_mat==nullptr) finest_mat = bfa->GetMatrixPtr();
    auto embed_step = BuildEmbedding();
    amg_pc->Finalize(fmat, embed_step);
  }


} // namespace amg

#endif
