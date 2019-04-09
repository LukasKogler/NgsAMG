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
    shared_ptr<CoarseMap> gstep_coarse;
    shared_ptr<GridContractMap<TMESH>> gstep_contr;
    auto dof_map = make_shared<DOFMap>();
    shared_ptr<BaseDOFMapStep> dof_step;
    Array<INT<3>> levels;
    auto MAX_NL = options->max_n_levels;
    auto MAX_NV = options->max_n_verts;

    { // coarsen mesh!
      INT<3> level = 0; // coarse, contr, elim
      levels.Append(level);
      while ( level[0] < MAX_NL-1 && fm->template GetNNGlobal<NT_VERTEX>()>MAX_NV) {
	if ( (grid_step = (gstep_contr = TryContract(level, fm))) != nullptr ) {
	  // dof_step = BuildDOFMapStep(gstep_contr, fm_pd);
	  level[1]++;
	}
       	else if ( (grid_step = (gstep_coarse = TryCoarsen(level, fm))) != nullptr ) {
	  dof_step = BuildDOFMapStep(gstep_coarse, fm_pd);
	  if (level[0]==0 && embed_step!=nullptr) {
	    // cout << "embst: " << embed_step << endl;
	    // cout << "dof s " << dof_step << endl;
	    // cout << "concatenate embedding + first ProlStep!!" << endl;
	    dof_step = embed_step->Concatenate(dof_step);
	  }
       	  level[0]++;
       	}
       	else { cout << "warning, no map variant worked!" << endl; break; } // all maps failed

	auto NV = fm->template GetNNGlobal<NT_VERTEX>();
	fm = dynamic_pointer_cast<TMESH>(grid_step->GetMappedMesh());
	auto CNV = fm->template GetNNGlobal<NT_VERTEX>();
	if (fm->GetEQCHierarchy()->GetCommunicator().Rank()==0) {
	  double fac = (NV==0) ? 0 : (1.0*CNV)/NV;
	  // cout << "map NV " << NV << " -> " << CNV << ", factor " <<  fac << endl;
	}
	fm_pd = dof_step->GetMappedParDofs();
	grid_map->AddStep(grid_step);
	dof_map->AddStep(dof_step);
	if (fm==nullptr) { break; } // no mesh due to contract
      }
    }

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
    for (auto k : Range(mats.Size())) {
      // cout << "make smoother!!" << endl;
      auto pds = dof_map->GetParDofs(k);
      shared_ptr<const TSPMAT> mat = dynamic_pointer_cast<TSPMAT>(mats[k]);
      sms.Append(make_shared<HybridGSS<mat_traits<TV>::HEIGHT>>(mat,pds,(k==0) ? options->finest_free_dofs : nullptr));
    }
    
    // cout << "make AMG-mat!" << endl;
    Array<shared_ptr<const BaseSmoother>> const_sms(sms.Size()); const_sms = sms;
    Array<shared_ptr<const BaseMatrix>> bmats_mats(mats.Size()); bmats_mats = mats;
    amg_mat = make_shared<AMGMatrix> (dof_map, const_sms, bmats_mats);
    // cout << "have AMG-mat!" << endl;
    
    if (mats.Last()!=nullptr) {
      auto max_l = mats.Size();
      auto cpds = dof_map->GetMappedParDofs();
      auto comm = cpds->GetCommunicator();
      if (comm.Size()>1) {
	// cout << "coarse inv " << endl;
	auto cpm = make_shared<ParallelMatrix>(mats.Last(), cpds);
	cpm->SetInverseType("masterinverse");
	// auto cinv = cpm->InverseMatrix();
	// cout << "coarse inv done" << endl;
	// amg_mat->AddFinalLevel(cinv);
	amg_mat->AddFinalLevel(nullptr);
      }
      else {
	throw Exception("Here we should do local SP-CHOL!");
      }
    }
  }

  template<class AMG_CLASS, class TMESH, class TMAT>
  shared_ptr<ProlMap<typename VWiseAMG<AMG_CLASS, TMESH, TMAT>::TSPMAT>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: BuildDOFMapStep (shared_ptr<CoarseMap> _cmap, shared_ptr<ParallelDofs> fpd)
  {
    // coarse ParallelDofs
    const CoarseMap & cmap(*_cmap);
    const TMESH & fmesh = static_cast<TMESH&>(*cmap.GetMesh());
    const TMESH & cmesh = static_cast<TMESH&>(*cmap.GetMappedMesh());
    const AMG_CLASS& self = static_cast<const AMG_CLASS&>(*this);
    auto cpd = BuildParDofs(static_pointer_cast<TMESH>(cmap.GetMappedMesh()));
    // prolongation Matrix
    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();
    // cout << "DOF STEP, fmesh " << fmesh << endl;
    // cout << "DOF STEP, cmesh " << cmesh << endl;
    auto vmap = cmap.GetMap<NT_VERTEX>();
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
    return make_shared<ProlMap<TSPMAT>> (prol, fpd, cpd);
  }
  
  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<CoarseMap>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: TryCoarsen  (INT<3> level, shared_ptr<TMESH> mesh)
  {
    auto coarsen_opts = make_shared<HierarchicVWC::Options>();
    shared_ptr<VWCoarseningData::Options> basos = coarsen_opts;
    // auto coarsen_opts = make_shared<VWCoarseningData::Options>();
    if (level[0]==0) { coarsen_opts->free_verts = options->free_verts; }
    SetCoarseningOptions(basos, level, mesh);
    // BlockVWC bvwc (coarsen_opts);
    // return bvwc.Coarsen(mesh);
    HierarchicVWC hvwc (coarsen_opts);
    return hvwc.Coarsen(mesh);
  }

  template<class AMG_CLASS, class TMESH, class TMAT> shared_ptr<GridContractMap<TMESH>>
  VWiseAMG<AMG_CLASS, TMESH, TMAT> :: TryContract (INT<3> level, shared_ptr<TMESH> mesh)
  {
    Table<int> groups = PartitionProcsMETIS (*mesh, mesh->GetEQCHierarchy()->GetCommunicator().Size()/2);
    return make_shared<GridContractMap<TMESH>>(move(groups), mesh);
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
      // cout << "v-sort: "; prow(vsort); cout << endl;
      auto fes_fds = fes->GetFreeDofs();
      // cout << "fes fds: " << endl << *fes_fds << endl;
      auto fvs = make_shared<BitArray>(ma->GetNV());
      fvs->Clear();
      for (auto k : Range(ma->GetNV())) if (fes_fds->Test(k)) { fvs->Set(vsort[k]); }
      options->free_verts = fvs;
      options->finest_free_dofs = fes_fds;
      // cout << "free vertices: " << endl << *fvs << endl;
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
