#include "amg.hpp"

namespace amg
{

  template<int D>
  void ElasticityAMGPreconditioner<D> :: InitLevel (shared_ptr<BitArray> afreedofs)
  {
    this->finest_freedofs = afreedofs;
    // finest_freedofs->Clear();
    // finest_freedofs->Invert();

    // cout << "nfree: " << finest_freedofs->NumSet() << " of " << finest_freedofs->Size() << endl;
    // cout << "initial FD: " << endl << *finest_freedofs << endl;
    
    this->hash_edge = new HashTable<INT<2, size_t>, edge_data>(size_t(8.0/dofpv(D)*afreedofs->Size()));
    
    auto ma = fes->GetMeshAccess();
    // auto nv = MyMPI_GetId() ? ma->GetNP() : 0;
    auto nv = ma->GetNP();
    this->p_coords.SetSize(nv);
    for(auto k:Range(nv))
      ma->GetPoint(k,p_coords[k]);

    this->amg_info->SetLogLevel(2);
    this->amg_options->SetMinCW(0.05);
    // this->amg_options->SetMaxNLevels(30);
    // this->amg_options->SetCoarsestVertFraction(1e-4);
    // this->amg_options->SetCoarsestMaxVerts(100);
    
    if(NgMPI_Comm(MPI_COMM_WORLD).Size()>1) this->amg_options->SetCoarseInvType("masterinverse");
    else this->amg_options->SetCoarseInvType("sparsecholesky");
    // this->amg_options->SetCoarseInvType("mumps"); // <- BROKEN!! uses comm_world!!



    
  } // end ElasticityAMGPreconditioner :: InitLevel


  template<int D>
  void ElasticityAMGPreconditioner<D> :: FinalizeLevel (const BaseMatrix * a_mat) {
    // cout << "finalize elasticity AMG; mat has dim " << a_mat->Height() << " x " << a_mat->Width() << endl;
    if(cut_blf==nullptr) this->finest_bmat = a_mat;
    else {
      this->finest_bmat = cut_blf->GetMatrixPtr().get();
      this->cut_prol = true;
      this->has_dummies = true;
    }
    // default settings for prolongation smoothing
    bool def_sm = false;
    for(auto l:Range(amg_options->GetMaxNLevels()))
      if (amg_options->GetSmoothLevel(l)==-1) {
	def_sm = true;
      }
    if(def_sm) {
      for(auto l : {1,2,4,6})
	amg_options->SetSmoothLevel(l,0);
      for(auto l : {0,3,5})
	amg_options->SetSmoothLevel(l,1);
    }

    // default settings for level assembling
    bool def_al = false;
    amg_options->SetAssembleLevel(0,1); // for my sanity :)
    for(auto l:Range(amg_options->GetMaxNLevels()))
      if (amg_options->GetAssembleLevel(l)==-1) {
	def_al = true;
      }
    if(def_al) {
      amg_options->SetAssembleLevel(0,1);
      for(auto l:Range(size_t(1), min2(size_t(9), amg_options->GetMaxNLevels())))
	if(amg_options->GetSmoothLevel(l)!=-1)
	  amg_options->SetAssembleLevel(l,amg_options->GetSmoothLevel(l)==1);
      // if(amg_options->GetMaxNLevels()>9)
      // 	for(auto l:Range(size_t(9), amg_options->GetMaxNLevels()))
      // 	  amg_options->SetAssembleLevel(l,l%2==1);
    }
    this->SetupAMGMatrix();
    
    // cout << "DONE SO FAR!" << endl;
    // this->amg_matrix->Finalize();
    return;
  }


  template<int D> shared_ptr<BaseDOFMapStep>
  ElasticityAMGPreconditioner<D> :: BuildDOFMapStep (shared_ptr<ParallelDofs> fpd, CoarseMap & map)
  {
#ifdef SCOREP
    SCOREP_USER_REGION("DOF-step, coarse", SCOREP_USER_REGION_TYPE_COMMON);
#endif
    auto fbmesh = dynamic_pointer_cast<ElasticityMesh>(map.GetMesh());
    auto cbmesh = dynamic_pointer_cast<ElasticityMesh>(map.GetMappedMesh());
    size_t nv = map.GetNN<NT_VERTEX>(), ncv = map.GetMappedNN<NT_VERTEX>();
    auto vmap = map.GetMap<NT_VERTEX>();
    auto emap = map.GetMap<NT_EDGE>();
    auto eqc_h = fbmesh->GetEQCHierarchy();

    Array<Vec<3,double>> & coords = std::get<0>(fbmesh->Data())->Data();
    Array<Vec<3,double>> & Ccoords = std::get<0>(cbmesh->Data())->Data();

    // cout << "coords | map: " << endl;
    // for(auto [co,v] : Zip(coords, Enumerate(vmap)))
    //   cout << get<0>(v) << " -> " << get<1>(v) << "  |  " << co << endl;
    // cout << endl;

    const bool cut_it = this->cut_prol;
    if(cut_it) this->cut_prol = false;

    // auto cfds = make_shared<BitArray>(rotpv(D)*ncv);
    // auto & dummy_rots = *cfds;
    // int ndummies = 0;
    // if((cut_blf==nullptr) || !(this->has_dummies))
    // else{
    //   cfds->Clear();
    //   ndummies = rotpv(D)*ncv;
    // }

    Array<int> dp_s (dofpv(D)*ncv);
    dp_s = 0;
    Array<size_t> offsets(dofpv(D));
    for(auto k:Range(dofpv(D)))
      offsets[k] = k*ncv;
    for(auto k:Range(ncv)) {
      size_t sz = eqc_h->GetDistantProcs(map.EQC_of_CN<NT_VERTEX>(k)).Size();
      for(auto l:Range(dofpv(D)))
	dp_s[k+offsets[l]] = sz;
    }
    Table<int> coarse_distprocs(dp_s);
    for(auto k:Range(ncv))
      for(auto [j,p]:Enumerate(eqc_h->GetDistantProcs(map.EQC_of_CN<NT_VERTEX>(k))))
	for(auto l:Range(dofpv(D)))
	  coarse_distprocs[k+offsets[l]][j] = p;
    auto cpd = make_shared<ParallelDofs>(eqc_h->GetCommunicator(), move(coarse_distprocs));    
 
    if( (nv!=0) && (nv==ncv) ) cerr << "Warning, " << ncv << "==NV==NCV (was this intended?)" << endl;

    // cout << "vmap: " << endl << vmap << endl;
    
    Array<int> nzepr((cut_it?disppv(D):dofpv(D))*nv);
    nzepr = 0;
    size_t n_dispf = disppv(D)*nv;
    auto find = [&](auto v, auto type, auto comp)
      { return type*n_dispf + nv*comp + v;};
    size_t n_dispc = disppv(D)*ncv;
    auto cind = [&](auto cv, auto type, auto comp)
      { return type*n_dispc + ncv*comp + cv;};

    // auto clear_dummy = [&](auto cv, auto rc) {
    //   if(ndummies==0) return;
    //   auto dof = cind(cv, 1, rc);
    //   if(!dummy_rots.Test(dof)) ndummies--;
    //   dummy_rots.Set(dof);
    // };


    // cout << "last_hierarch? " << last_hierarch << endl;
    ParallelVVector<double> bme(fpd, this->last_hierarch ? DISTRIBUTED : CUMULATED); // <0..begin, 0..mid, >0..end
    bme.FVDouble() = 0;
    auto doit = [&](const auto & es) {
      for(auto e : es) {
	auto CV = vmap[e.v[0]];
	if(CV==-1) continue;
	if(CV!=vmap[e.v[1]]) continue;
	bme.FVDouble()[e.v[0]] = -1;
	bme.FVDouble()[e.v[1]] = 1;
      }
    };
    for(auto eqc:Range(eqc_h->GetNEQCS())) {
      if(this->last_hierarch && !eqc_h->IsMasterOfEQC(eqc)) continue;
      doit((*fbmesh)[eqc]->template GetNodes<NT_EDGE>());
      doit((*fbmesh)[eqc]->template GetNodes_cross<NT_EDGE>());
    }
    bme.Cumulate();
    
    const auto & vertex_coarse(vmap);
    // cout << "coords: " << endl << coords << endl;
    // cout << "vmap: " << endl << vertex_coarse << endl;
    // cout << "bme: " << endl << bme << endl;
    // cout << "Ccoords: " << endl << Ccoords << endl;

    for(auto vnr:Range(nv)) {
      auto cvnr = vmap[vnr];
      if(cvnr==-1) continue;
      // if we cut, we add zeros here so we dont get empty rows/cols
      // in the coarse matrices
      bool is_sing = bme.FVDouble()[vnr]==0;
      // if(is_sing) cout << "vnr " << vnr << " is sing!" << endl;
      for(auto i:Range(disppv(D)))
	nzepr[find(vnr,0,i)] = 1 + ((is_sing&&!cut_it)?0:rotpv(D));
      if(!cut_it)
	for(auto i:Range(rotpv(D)))
	  nzepr[find(vnr,1,i)] = 1;
    }
    // cout << "mat.." << endl;
    auto prol = make_shared<SparseMatrix<double>>(nzepr, dofpv(D)*ncv);
    // cout << "fill.." << endl;
    FlatMatrix<double> skew_t(disppv(D), rotpv(D), my_lh);
    Vec<D> t;
    for(auto vnr:Range(nv)) {
      auto cvnr = vmap[vnr];
      bool is_sing = bme.FVDouble()[vnr]==0;
      if(cvnr==-1) continue;
      if(bme.FVDouble()[vnr]==0.0) {
	for(auto i:Range(disppv(D))) {
	  auto ii = find(vnr,0,i);
	  auto ri = prol->GetRowIndices(ii);
	  auto rv = prol->GetRowValues(ii);
	  ri[0] = cind(cvnr,0,i);
	  rv[0] = 1.0;
	  if(is_sing && cut_it)
	    for(auto j:Range(rotpv(D))) {
	      ri[1+j] = cind(cvnr, 1, j);
	      rv[1+j] = 0;
	    }
	}
	if(!cut_it)
	  for(auto i:Range(rotpv(D))) {
	    auto ii = find(vnr,1,i);
	    prol->GetRowIndices(ii)[0] = cind(cvnr,1,i);
	    prol->GetRowValues(ii)[0] = 1.0;
	  }
      }
      else {
	// t = (bme.FVDouble()[vnr]<0) ? Ccoords[cvnr]-coords[vnr] : coords[vnr]-Ccoords[cvnr];
	// t = Ccoords[cvnr]-coords[vnr];
	t = coords[vnr]-Ccoords[cvnr];
	// cout << "tang for coarse vnr " << cvnr << ": " << t << endl;
	skew_t = 0;
	if constexpr(D==3) {
	    skew_t(0,1) = -(skew_t(1,0) =  t[2]);
	    skew_t(0,2) = -(skew_t(2,0) = -t[1]);
	    skew_t(1,2) = -(skew_t(2,1) =  t[0]);
	    // bool tsqs[3] = {t[0]*t[0]>1e-20, t[1]*t[1]>1e-20, t[2]*t[2]>1e-20};
	    // if(tsqs[1]||tsqs[2]) clear_dummy(cvnr, 0);
	    // if(tsqs[0]||tsqs[2]) clear_dummy(cvnr, 1);
	    // if(tsqs[0]||tsqs[1]) clear_dummy(cvnr, 2);
	  }
	else {
	  skew_t(0,0) =  t[1];
	  skew_t(1,0) = -t[0];
	}
	
	// cout << "skew_t: " << endl << skew_t << endl;
	for(auto i:Range(disppv(D))) {
	  auto ii0 = find(vnr,0,i);
	  auto ri0 = prol->GetRowIndices(ii0);
	  auto rv0 = prol->GetRowValues(ii0);
	  ri0[0] = cind(cvnr,0,i);
	  rv0[0] = 1.0;
	  for(auto j:Range(rotpv(D))) {
	    ri0[j+1] = cind(cvnr,1,j);
	    rv0[j+1] = skew_t(i,j); //unscaled t!
	  }
	}
	if(!cut_it)
	  for(auto i:Range(rotpv(D))) {
	    auto ii0 = find(vnr,1,i);
	    prol->GetRowIndices(ii0)[0] = cind(cvnr,1,i);
	    prol->GetRowValues(ii0)[0] = 1.0;
	  }
      }
    }

    // TODO: something is wrong here...
    // bool smooth_prol = prol->Height()!=finest_freedofs->Size();
    // bool smooth_prol = true;
    bool smooth_prol = smooth_next_prol;

    // cout << "COARSE vmap, " << prol->Height() << " -> " << prol->Width() << ": " << endl << vmap << endl;
    // cout << "COARSE prol, " << prol->Height() << " -> " << prol->Width() << ": " << endl << *prol << endl;

    // if(ndummies==0) this->dummy_fds = nullptr;
    // else { cfds->Invert(); this->dummy_fds->Append(cfds); }
	
    return make_shared<ProlMap>(prol, fpd, cpd);
  }


  template<int D> shared_ptr<SparseMatrix<double>>
  ElasticityAMGPreconditioner<D> :: SmoothProl (shared_ptr<ProlMap> pw_map, shared_ptr<ElasticityMesh> mesh)
  {
    // if(!MyMPI_GetId())cout << "smooth the prol!" << endl;
    auto fpd = pw_map->GetParDofs();
    auto cpd = pw_map->GetMappedParDofs();
    auto prol = pw_map->GetProlongation();
    Array<Vec<3,double>> & cos = get<0>(mesh->Data())->Data();
    Array<edge_data> & emats = get<1>(mesh->Data())->Data();
    Array<INT<2,double>> vstr (mesh->template NN<NT_VERTEX>());
    vstr = 0;
    auto eqc_h = mesh->GetEQCHierarchy();
    for (auto [eqc, block] : *mesh) {
      if(!eqc_h->IsMasterOfEQC(eqc)) continue;
      auto lam = [&](auto & erow) {
	for (const auto & e : erow) {
	  auto w = CalcEdgeWeights(emats[e.id]);
	  vstr[e.v[0]] += w;
	  vstr[e.v[1]] += w;
	}
      };
      lam(block->template GetNodes<NT_EDGE>());
      lam(block->template GetNodes_cross<NT_EDGE>());
    }
    mesh->CumulateVertexData(vstr);
    auto get_repl = [&](const idedge & edge, FlatMatrix<double> & mat) {
      CalcReplMatrix(emats[edge.id], cos[edge.v[0]], cos[edge.v[1]], mat);
      // CalcReplMatrix(edge, mat);
      return false;
    };
    auto get_wt = [&](size_t eid, size_t vid) {
      auto ew = CalcEdgeWeights(emats[eid]);
      auto vw = vstr[vid];
      return min2(ew[0]/vw[0], ew[1]/vw[1]);
    };
    // TODO: this is not so clean, we re-construct vertex-map
    Array<int> vmap(mesh->template NN<NT_VERTEX>());
    vmap = -1;
    for(auto k:Range(mesh->template NN<NT_VERTEX>())) {
      auto ri = prol->GetRowIndices(k);
      if(ri.Size()) vmap[k] = ri[0];
    }
    auto s_prol = HProl_Vertex(mesh, vmap, fpd, cpd, amg_options, prol,
			       (prol->Height()==finest_freedofs->Size()) ? finest_freedofs : nullptr,
			       dofpv(D), get_repl, get_wt);
    
    // return make_shared<ProlMap>(s_prol, fpd, cpd);
    return s_prol;
  }

  
  template<int D> shared_ptr<BaseDOFMapStep>
  ElasticityAMGPreconditioner<D> :: BuildDOFMapStep (shared_ptr<ParallelDofs> pd, GridContractMap & map)
  {
    auto mesh = dynamic_pointer_cast<BlockedAlgebraicMesh>(map.GetMesh());
    auto cmesh = dynamic_pointer_cast<BlockedAlgebraicMesh>(map.GetMappedMesh());
    auto npvc = map.GETCM()->NodalPVContraction<NT_VERTEX, dofpv(D)>(pd);
    auto cm = make_shared<DOFContractMap>(npvc);
    cm->SetBlockSize(dofpv(D));
    return cm;
  }

  template<int D> shared_ptr<BaseDOFMapStep>
  ElasticityAMGPreconditioner<D> :: BuildDOFMapStep (shared_ptr<ParallelDofs> fpd, NodeDiscardMap & map)
  {
    auto mesh = dynamic_pointer_cast<ElasticityMesh>(map.GetMesh());
    auto cmesh = dynamic_pointer_cast<ElasticityMesh>(map.GetMappedMesh());

    // cout << "DOF step for discard map" << endl;

    // cout << "fpd: " << fpd << endl;
    // cout << "fpd ND: " << fpd->GetNDofLocal() << " " << fpd->GetNDofGlobal() << endl;

    // coarse distprocs are easy (no vertices change EQC)
    auto eqc_h = mesh->GetEQCHierarchy();
    auto NV = map.GetNN<NT_VERTEX>();
    auto CNV = map.GetMappedNN<NT_VERTEX>();
    // cout << "discard map: " << NV << " -> " << CNV << endl;
    auto vmap = map.GetMap<NT_VERTEX>();
    Array<int> dp_s (dofpv(D)*CNV);
    dp_s = 0;
    Array<size_t> offsets(dofpv(D));
    for(auto k:Range(dofpv(D)))
      offsets[k] = k*CNV;
    for(auto [V,MV]:Enumerate(vmap)) {
      if(MV==-1) continue;
      size_t sz = fpd->GetDistantProcs(V).Size();
      for(auto l:Range(dofpv(D)))
	dp_s[MV+offsets[l]] = sz;
    }
    Table<int> coarse_distprocs(dp_s);
    for(auto [V,MV]:Enumerate(vmap)) {
      if(MV==-1) continue;
      for(auto [j,p]:Enumerate(fpd->GetDistantProcs(V)))
	for(auto l:Range(dofpv(D)))
	  coarse_distprocs[MV+offsets[l]][j] = p;
    }
    auto cpd = make_shared<ParallelDofs>(eqc_h->GetCommunicator(), move(coarse_distprocs));    
    // cout << "have pardofs " << endl;

    // cout << "(discarded) cpd ND: " << cpd->GetNDofLocal() << " " << cpd->GetNDofGlobal() << endl;

    // For the prolongation, we eliminate all DOFs of dropped vertices by
    // computing the exact schur-complement (of the replacement matrix)
    auto get_dof = [&](auto V, auto comp) { return NV*comp + V; };
    auto get_m_dof = [&](auto CV, auto comp) { return CNV*comp + CV; };
    TableCreator<int> cg(dofpv(D)*NV);
    auto econ = mesh->GetEdgeConnectivityMatrix();
    auto vertex_blocks = map.GetVertexBlocks();
    while(!cg.Done()) {
      for(auto block:vertex_blocks) {
	// has to contain ALL relevant neibs! (if v1,v2 in block -> all neibs of v1 and v2 have to be same)
	auto neibs = econ->GetRowIndices(block[0]);
	// cout << "block: " << block << endl;
	// cout << "neibs: "; prow(econ->GetRowIndices(block[0])); cout << endl;
	for(auto l:Range(dofpv(D))) {
	  for(auto j:Range(dofpv(D))) {
	    for(auto V:block) {
	      for(auto N:neibs) {
		if (vmap[N]!=-1) {
		  cg.Add(get_dof(V,l), get_m_dof(vmap[N],j));
		}
	      }
	    }
	  }
	}
      }
      cg++;
    }
    auto graph = cg.MoveTable();
    Array<int> perow(graph.Size());
    // for(auto [a,b]:Zip(graph, perow)) // <- does not work!
    //   b = max2(size_t(1),a.Size());
    for(auto [k,b]:Enumerate(perow))
      b = max2(size_t(1),graph[k].Size());
    // cout << "have graph " << endl << graph << endl;;
    auto prol = make_shared<SparseMatrix<double>>(perow, dofpv(D)*CNV);
    for(auto [V,CV] : Enumerate(vmap))
      if(CV!=-1)
	for(auto l:Range(dofpv(D))) {
	  auto dof = get_dof(V,l);
	  prol->GetRowIndices(dof)[0] = get_m_dof(CV,l);
	  prol->GetRowValues(dof)[0] = 1.0;
	}
    LocalHeap lh(3*1024*1024, "Elisabeth II");
    FlatMatrix<double> r_block (2*dofpv(D), 2*dofpv(D), lh);
    int MAX_BS = 3;
    Array<int> r0(dofpv(D));
    Array<int> r1(dofpv(D));
    Array<int> rV(dofpv(D));
    Array<int> cVN(2*dofpv(D));
    Array<int> patch;
    for(auto l:Range(dofpv(D)))
      r1[l] = 1 + (r0[l] = 2*l);
    auto insert = [](auto val, auto & ar) {
      size_t pos = 0;
      while(pos+1<ar.Size() && val>ar[pos]) pos++;
      for(int k = ar.Size()-1; k > pos; k--)
	ar[k] = ar[k-1];
      ar[pos] = val;
    };
    for(auto block:vertex_blocks) {
      // cout << "fill block: "; prow(block); cout << endl;
      HeapReset hr(lh);
      for(auto V:block)
	for(auto l:Range(dofpv(D)))
	  prol->GetRowIndices(get_dof(V,l)) = graph[get_dof(V,l)];
      // cout << "RI: " << endl;
      // for(auto V:block)
      // 	{ cout << get_dof(V,0) << ": "; prow(prol->GetRowIndices(get_dof(V,0))); cout << endl; }
      auto n_block = block.Size();
      auto dpatch = econ->GetRowIndices(block[0]);
      auto patch = FlatArray<int>(dpatch.Size()+1, lh);
      patch.Part(0, dpatch.Size()) = dpatch;
      insert(block[0], patch);
      auto n_patch = patch.Size();
      auto n_rest = n_patch - n_block;
      auto rest = FlatArray<int>(n_rest, lh);
      n_rest = 0;
      auto cb = 0;
      for(auto V:patch)
	if(cb<block.Size() && block[cb]==V) cb++;
	else rest[n_rest++] = V;
      // cout << "patch (" << n_patch << "): "; prow(patch); cout << endl;
      // cout << "block (" << n_block << "): "; prow(block); cout << endl;
      // cout << "rest (" << n_rest << "): "; prow(rest); cout << endl;
      
      auto nd_patch = dofpv(D)*n_patch;
      auto nd_block = dofpv(D)*n_block;
      auto nd_rest = dofpv(D)*n_rest;

      FlatArray<int> patch_dofs(nd_patch, lh);
      FlatArray<int> block_dofs(nd_block, lh);
      FlatArray<int> rest_dofs(nd_rest, lh);
      nd_patch = nd_block = nd_rest = 0;
      for(auto l:Range(dofpv(D))) {
	for(auto V:patch)
	  patch_dofs[nd_patch++] = get_dof(V,l);
	for(auto V:block)
	  block_dofs[nd_block++] = get_dof(V,l);
	for(auto V:rest)
	  rest_dofs[nd_rest++] = get_dof(V,l);
      }
      // cout << "block_dofs (" << nd_block << " " << block_dofs.Size() << "): ";prow(block_dofs); cout << endl;
      // cout << "patch_dofs (" << nd_patch << " " << patch_dofs.Size() << "): ";prow(patch_dofs); cout << endl;
      // cout << "rest_dofs (" << nd_rest << " " << rest_dofs.Size() << "): ";prow(rest_dofs); cout << endl;
      if (nd_rest!=dofpv(D)*n_rest) throw Exception("rest mismatch!!");
      if (nd_rest<dofpv(D)) throw Exception("what the hell??");
      if (nd_rest==dofpv(D)) {
	/* 
	   In this case, we could simply use pw-prol from the coarse DOFs.
	   We have to consider this also in the graph above.
	   However, for now, SC will deliver the same (but with a thicker graph).
	*/
      }
      FlatArray<int> block_pos(nd_block, lh);
      for(auto [k,v]:Enumerate(block_pos))
	v = patch_dofs.Pos(block_dofs[k]);
      FlatArray<int> rest_pos(nd_rest, lh);
      for(auto [k,v]:Enumerate(rest_pos))
	v = patch_dofs.Pos(rest_dofs[k]);
      FlatMatrix<double> repl (nd_block, nd_patch, lh);
      repl = 0;
      FlatMatrix<double> diag (nd_block, nd_block, lh);
      diag = 0;
      Array<edge_data> & emats = get<1>(mesh->Data())->Data();
      // Array<Vec<3,double>> & cos = get<0>(mesh->Data())->Data();
      auto & cos = get<0>(mesh->Data())->Data();
      for (auto V : block) {
	auto kV = patch.Pos(V);
	auto kV_block = block.Pos(V);
	for (auto [kN,N] : Enumerate(patch)) {
	  if (V==N) continue;
	  if (block.Contains(N) && N>V) continue;
	  auto V0 = min2(V,size_t(N));
	  auto V1 = max2(V,size_t(N));
	  auto col0 = min2(kV,kN);
	  auto col1 = max2(kV,kN);
	  auto c = 0;
	  for (auto l:Range(dofpv(D))) {
	    cVN[c++] = l*n_patch + col0;
	    cVN[c++] = l*n_patch + col1;
	    rV[l] = l*n_block + kV_block;
	  }
	  auto e_num = (*econ)(V0,V1);
	  CalcReplMatrix(emats[e_num], cos[V0], cos[V1], r_block);
	  // cout << "block for " << V << " " << N << endl;
	  // cout << "cos: " << endl << cos[V0] << endl << cos[V1] << endl;
	  // cout << "edata: " << emats[e_num] << endl;
	  // cout << "block: " << endl << r_block << endl;
	  // cout << "rows to write to: "; prow(rV); cout << endl;
	  // cout << "cols to write to: "; prow(cVN); cout << endl;
	  repl.Rows(rV).Cols(cVN) += r_block.Rows((V==V0)?r0:r1);
	  // cout << "repl is now " << endl << repl << endl;
	}
      }
      diag = repl.Cols(block_pos);
      // cout << "diag mat: " << endl << diag << endl;
      CalcInverse(diag);
      // cout << "diag inv: " << endl << diag << endl;
      FlatMatrix<double> prol_vals (nd_block, nd_patch-nd_block, lh);
      prol_vals = 0;
      prol_vals = -1.0 * diag * repl.Cols(rest_pos);
      // cout << "prol_vals: " << endl << prol_vals << endl << endl;
      // for(auto V:block)
      // 	for(auto l:Range(dofpv(D)))
      // 	  for(auto [v1,v2] : Zip(prol->GetRowValues(get_dof(V,l)), Map(Range(prol_vals.Width(), [&](auto j){ return prol_vals(l,j); }))))
      // 	    v1 = v2;
      auto bs = block.Size();
      for(auto l:Range(dofpv(D))) {
	for (auto [k,V] : Enumerate(block)) {
	  auto pv = prol->GetRowValues(get_dof(V,l));
	  for(auto j:Range(prol_vals.Width()))
	    pv[j] = prol_vals(bs*l+k,j);
	}
      }
      // cout << "block done" << endl;
    }  

    // cout << "discard dof-step, have prol!" << endl;
    // cout << "DISCARD vmap, " << prol->Height() << " -> " << prol->Width() << ": " << endl << vmap << endl;
    // cout << "DISCARD prol, " << prol->Height() << " -> " << prol->Width() << ": " << endl << *prol << endl;
    
    return make_shared<ProlMap>(prol, fpd, cpd);
  }

  
  
  template<int D> shared_ptr<BaseGridMapStep>
  ElasticityAMGPreconditioner<D> :: TryElim (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh)
  {
    if (level[2] !=0 ) return nullptr;
    if (_mesh->GNV()<4) return nullptr;
    if constexpr(D==2) if(level[0] < 5) return nullptr;
    if constexpr(D==3) if(level[0] < 10) return nullptr;
    auto mesh = dynamic_pointer_cast<ElasticityMesh>(_mesh);
    // if (MyMPI_GetNTasks(mesh->GetEQCHierarchy()->GetCommunicator())>2) return nullptr;
    // cout << "make discard map " << endl;
    auto ndm = make_shared<NodeDiscardMap>(mesh);
    // cout << "have discard map!" << endl;
    // dummy cweights; TODO: get rid of it
    Array<double> vw(ndm->template GetMappedNN<NT_VERTEX>());
    vw = 0.0;
    ndm->GetMappedMesh()->SetVertexWeights(move(vw));
    Array<double> we(ndm->template GetMappedNN<NT_EDGE>());
    we = 0.0;
    ndm->GetMappedMesh()->SetEdgeWeights(move(we));
    if(1.0*ndm->GetMappedMesh()->GNV() > 0.95*_mesh->GNV()) return nullptr;
    return ndm;
  }

  template<int D> shared_ptr<BaseGridMapStep>
  ElasticityAMGPreconditioner<D> :: TryCoarsen (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh)
  {
    auto mesh = dynamic_pointer_cast<ElasticityMesh>(_mesh);
    auto coll = make_shared<CollapseTracker>(mesh->NV(), mesh->NE());

    // vcw is 0
    Array<double> vcw(mesh->NV());
    vcw = 0;
    mesh->SetVertexCollapseWeights(move(vcw));
    // compute ecw from meta-data
    auto eqc_h = mesh->GetEQCHierarchy();
    auto & edata = get<1>(mesh->Data());
    // auto & coords = get<0>(mesh->Data()).Data();
    auto & coords = std::get<0>(mesh->Data())->Data();
    edata->Cumulate();
    auto & emats = edata->Data();
    Array<INT<2,double>> vs(mesh->NV());
    vs = 0.0;
    for (auto [eqc,block] : *mesh) {
      if(!eqc_h->IsMasterOfEQC(eqc)) continue;
      auto lam = [&](auto & es) {
	for (auto e:es) {
	  auto wt = CalcEdgeWeights(emats[e.id]);
	  vs[e.v[0]] += wt;
	  vs[e.v[1]] += wt;
	}
      };
      lam(block->template GetNodes<NT_EDGE>());
      lam(block->template GetNodes_cross<NT_EDGE>());
    }
    mesh->CumulateVertexData(vs);
    Array<double> cwe(mesh->template NN<NT_EDGE>());
    // double t0 = -MPI_Wtime();
    // double tcalc = 0;
    // THIS WAS SO FUCKING SLOW!!!!
    // for (auto [ew,e,em] : Zip(cwe, mesh->Edges(), emats)) {
    const auto & _edges = mesh->template GetNodes<NT_EDGE>();
    for (int _k = 0; _k < _edges.Size(); _k++) {
      auto & ew = cwe[_k];
      const auto & e = _edges[_k];
      auto & em = emats[_k];
      // double t1 = MPI_Wtime();
      auto wt = CalcEdgeWeights(em);
      // tcalc += (MPI_Wtime()-t1);
      auto vs0 = vs[e.v[0]];
      auto w0 = min2(wt[0]/vs0[0], wt[1]/vs0[1]);
      auto vs1 = vs[e.v[1]];
      auto w1 = min2(wt[0]/vs1[0], wt[1]/vs1[1]);
      ew = w0+w1;
      // ew = sqrt(w0*w1);
      // ew = wt[1]/vs0[1] + wt[1]/vs1[1];
    }
    // t0 += MPI_Wtime();
        	
    const double MIN_CW(amg_options->GetMinCW());
    size_t nvalid = 0;
    for(auto v : cwe) if(v>MIN_CW) nvalid++;
    double frac = (1.0*nvalid) / (1.0*cwe.Size());
    int glob_rk = NgMPI_Comm(MPI_COMM_WORLD).Rank();
    if(cwe.Size() && frac<0.5) {
      cout << glob_rk << ", valid in total: " << nvalid << " of " << cwe.Size() << ", frac: " << frac << endl;
      Array<size_t> countve(mesh->NV());
      countve = 0;
      for(const auto & edge : mesh->Edges())
	{ countve[edge.v[0]]++; countve[edge.v[1]]++; }
      size_t maxve = 0;
      for(auto v:countve)
	maxve = max2(maxve,v);
      cout << glob_rk << ", max ve: " << maxve << endl;
      if(glob_rk) QuickSort(countve, [](const auto & a, const auto & b) { return a>b; });
    }
    if(!glob_rk) cout << " EPV : " << (1.0*mesh->GNE())/mesh->GNV() << endl;
    
    
    mesh->SetEdgeCollapseWeights(move(cwe));
    mesh->BuildMeshBlocks();
    
    // cout << "blockw  " << endl;
    amg_coarsening::Collapse_H1_Blockwise(mesh, amg_options, *coll);
    // cout << "cross?  " << this->use_hierarch << endl;
    if(this->use_hierarch)
      amg_coarsening::Collapse_H1_Cross(mesh, amg_options, *coll);
    this->last_hierarch = this->use_hierarch;
      
    auto cm = make_shared<CoarseMap>(mesh, coll);
    // dummy cweights; TODO: get rid of it
    vcw.SetSize(cm->template GetMappedNN<NT_VERTEX>());
    vcw = 0.0;
    cwe.SetSize(cm->template GetMappedNN<NT_EDGE>());
    cwe = 0.0;
    cm->GetMappedMesh()->SetVertexWeights(move(vcw));
    cm->GetMappedMesh()->SetEdgeWeights(move(cwe));
    
    // cout << MyMPI_GetId() << " coarse, nv (loc): " << cm->GetMesh()->NV() << " -> " << cm->GetMappedMesh()->NV() << endl;
    // if(!MyMPI_GetId()) cout << " coarse, nv (glob): " << cm->GetMesh()->GNV() << " -> " << cm->GetMappedMesh()->GNV() << endl;

    
    return cm;
  }

  template<int D> shared_ptr<BaseGridMapStep>
  ElasticityAMGPreconditioner<D> :: TryContract (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh)
  {
    // cout << "try contract on level " << level << endl;
    
    auto mesh = dynamic_pointer_cast<ElasticityMesh>(_mesh);
    std::apply([&](auto&&... x){ (x->Cumulate(), ...); }, mesh->Data());

    if(level[0]==amg_options->GetMaxNLevels()-1) return nullptr;
    // if(level[0]<5) return nullptr;
    if(level[0]==0 && cut_blf!=nullptr) return nullptr; //cannot contract here (dummy-fds are not properly contracted!)
    if(level[1]==1) return nullptr;
    if(mesh->GetEQCHierarchy()->GetCommunicator().Size()<=2) return nullptr;

    const auto & eqc_h = mesh->GetEQCHierarchy();
    auto comm = eqc_h->GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    /** build sif-procs (those, where interface not empty!) **/
    Array<size_t> sif; // entries: [proc, sif_sz]
    for(auto k:Range(rank+1, np)) {
      Array<int> dps(1);
      dps[0] = k;
      size_t eqc;
      if( (eqc = eqc_h->FindEQCWithDPs(dps)) == -1 )
	continue;
      auto NV =  (*mesh)[eqc]->NV();
      if(!NV) continue;
      sif.Append(k);
      sif.Append(NV);
    }
    size_t nvloc = (rank>0) ? (*mesh)[0]->NV() : 0;
    Array<size_t> nvs(np);
    MPI_Gather(&nvloc, 1, MyGetMPIType<size_t>(), &(nvs[0]), 1, MyGetMPIType<size_t>(), 0, comm);
    int do_it = 1;
    if(rank==0) {
      int sum_nv = 0;
      int cnt_toosmall = 0;
      for(auto v:nvs) {
	sum_nv += v;
	if(v<100) cnt_toosmall++;
      }
      if( (sum_nv/(np-1)) > 1000 ) do_it = 0;
      if(cnt_toosmall > 0.3*np) do_it = 1;
    }
    MPI_Bcast(&do_it, 1, MPI_INT, 0, comm);
    // if(do_it == 0) return nullptr;
    Array<int> pmap(np);
    Table<int> groups;
    if(rank>0) {
      size_t sz = sif.Size();
      comm.Send(sz, 0, MPI_TAG_AMG);
      comm.Send(sif, 0, MPI_TAG_AMG);
      MPI_Bcast(&(pmap[0]), np, MPI_INT, 0, comm);
    }
    else {
      size_t NV = np;
      size_t NE = 0;
      size_t NF = 0;
      Array<size_t> szs(np);
      szs[0] = 0;
      for(auto p:Range(1, np)) {
	comm.Recv(szs[p], p, MPI_TAG_AMG);
	NE += szs[p]/2;
      }
      Table<size_t> msgs (szs);
      for(auto p:Range(1, np)) {
	comm.Recv(msgs[p], p, MPI_TAG_AMG);
      }
      /** edges **/
      Array<idedge> ae(NE);
      Array<double> awe(NE);
      NE = 0;
      for(auto p:Range(1, np)) {
	auto row = msgs[p];
	size_t rs2 = row.Size()/2;
	for(auto k:Range(rs2)) {
	  ae[NE] = idedge({edge(p, row[2*k]), NE});
	  awe[NE++] = row[2*k+1];
	}
      }
      Array<double> nvs_dbl(nvs.Size());
      for(auto k:Range(nvs.Size())) nvs_dbl[k] = nvs[k];
      auto fam = make_shared<BaseAlgebraicMesh>(NV, NE, NF);
      fam->SetEdges(std::move(ae));
      fam->SetEdgeWeights(std::move(awe));
      fam->SetVertexWeights(std::move(nvs_dbl));
      /** nr of coarsening steps for collapsing **/
      // static size_t count_ncs = 0;
      static size_t NCS = 1;
      Array<shared_ptr<const CoarseMapping> > cmaps(NCS);
      /** ok, coarsen **/
      for(auto step:Range(NCS)) {
	if(fam->NV()==2) break; // already sequential...
	auto FNV = fam->NV();
	auto FNE = fam->NE();
	/** col-wts **/
	Array<double> cwv(FNV);
	cwv = 0;
	Array<double> cwe(FNE);
	const auto & edges = fam->Edges();
	for(const auto & e:edges) {
	  auto wv0 = fam->GetVW(e.v[0]);
	  auto wv1 = fam->GetVW(e.v[1]);
	  cwe[e.id] = (1.0*fam->GetEW(e.id)) / (1.0*min2(wv0, wv1));
	}
	fam->SetVertexCollapseWeights(std::move(cwv));
	fam->SetEdgeCollapseWeights(std::move(cwe));
	auto opts = make_shared<AMGOptions>();
	opts->SetMinCW(0.01);
	/** coarsen **/
	auto cmap = amg_coarsening::Coarsen_H1_Sequential(fam, opts);
	cmaps[step] = cmap;
	if(step==NCS-1) break;
	/** coarse wts **/
	auto CNV = cmap->GetNVC();
	auto CNE = cmap->GetNEC();
	Array<double> wv(CNV);
	wv = 0;
	Array<double> we(CNE);
	if(CNE) we = 0;
	const auto & cvmap = cmap->GetVMAP();
	const auto & emap = cmap->GetEMAP();
	// cvmap is never -1 in this case !!
	for(auto k:Range(FNV))
	  wv[cvmap[k]] += fam->GetVW(k);
	// emap is -1 for coll. edges
	for(auto k:Range(FNE))
	  if(emap[k]==-1) // if collapsed, interface becomes local!
	    wv[cvmap[edges[k].v[0]]] += fam->GetEW(k); //only add once!
	  else
	    we[emap[k]] += fam->GetEW(k);
	auto cfam = fam->GenerateCoarseMesh(cmap, std::move(wv), std::move(we));
	fam = cfam;
      }
      /** proc-map **/
      pmap = -1;
      for(auto k:Range(NV))
	pmap[k] = k;
      size_t NPC = np;
      for(auto l:Range(NCS)) {
	if(cmaps[l]==nullptr) break;
	const auto & vmap = cmaps[l]->GetVMAP();
	for(auto k:Range((size_t)1, NV)) {
	  pmap[k] = vmap[pmap[k]];
	}
	NPC = cmaps[l]->GetNVC();
      }
      /** broadcast proc-map **/
      MPI_Bcast(&(pmap[0]), NV, MPI_INT, 0, comm);
      // count_ncs++;
      // if(count_ncs==2) NCS = 2;
      if(NCS==1) NCS = 2;
    }
    size_t NPC = 1;
    for(auto k:Range(np)) NPC = max2(NPC, (size_t)(pmap[k]+1));
    Array<size_t> grps(NPC);
    grps = 0;
    for(auto k:Range(np)) {
      grps[pmap[k]]++;
    }
    groups = Table<int>(grps);
    groups.AsArray() = -1;
    grps = 0;
    for(auto k:Range(np)) {
      groups[pmap[k]][grps[pmap[k]]++] = k;
    }

    // // contract all to groups [0], [rest] ...
    // grps.SetSize(2);
    // grps[0] = 1;
    // grps[1] = np-1;
    // groups = Table<int>(grps);
    // groups[0][0] = 0;
    // for(auto k:Range(1, np))
    //   groups[1][k-1] = k;

    if(!NgMPI_Comm(MPI_COMM_WORLD).Rank()) {
      // cout << "contract, groups: " << endl << groups << endl;
      cout << "contract, " << np << " -> " << groups.Size() << "   " << (1.0*groups.Size())/np << endl;
    }
    // cout << "eqch: " << mesh->GetEQCHierarchy() << endl;
    // cout << *mesh->GetEQCHierarchy() << endl;
    
    return make_shared<GridContractMap>(mesh, move(groups));
  }


  
  template<int D>
  void ElasticityAMGPreconditioner<D> :: SetupAMGMatrix()
  {
    static Timer t("ElasticityAMGPreconditioner::SetupAMGMatrix");
    RegionTimer rt(t);
    auto dof_map = make_shared<DOFMap>();
    auto em = BuildInitialAlgMesh();
    shared_ptr<BaseAlgebraicMesh> fm = em;
    // shared_ptr<ParallelDofs> fpd = finest_bmat->GetParallelDofs();
    shared_ptr<ParallelDofs> fpd = finest_pds;
    NgMPI_Comm glob_comm = (fpd==nullptr) ? NgMPI_Comm() : NgMPI_Comm(fpd->GetCommunicator());
    int glob_rk = glob_comm.Rank();
    Array<INT<3>> levels;
    shared_ptr<BaseGridMapStep> grid_step;
    shared_ptr<BaseDOFMapStep> dof_step;
    size_t max_L = amg_options->GetMaxNLevels();
    bool discard_locked = true;
    bool contr_locked = true;
    Array<size_t> nvs;
    nvs.Append(fm->GNV());
    // Array<int> pw_levs({1,2,3,5,6,8,10,12,14,16});
    Array<int> smoothed_it;
    Array<shared_ptr<BaseAlgebraicMesh>> meshes;
    Array<shared_ptr<ProlMap>> pmaps;
    { // mesh-loop (only pw-prols)
      INT<3> level = 0; // coarse, contr, elim
      levels.Append(level);
      static Timer t("AMG - grid coarsening");
      RegionTimer rt(t);
#ifdef SCOREP
      SCOREP_USER_REGION("setup meshes", SCOREP_USER_REGION_TYPE_COMMON);
#endif
      int cctr = 0;
      smooth_next_prol = amg_options->GetSmoothLevel(level[0]);
      smoothed_it.Append(smooth_next_prol ? 1 : 0);
      meshes.SetSize(max_L+1);
      meshes.SetSize(0);
      pmaps.SetSize(max_L);
      pmaps.SetSize(0);
      meshes.Append(smooth_next_prol ? fm : nullptr);
      int cnt_lc = 0; //count since contr
      int cnt_ld = 0; //count since distr
      int MIN_V_PP = 500;
      double frac_coarse = 0.0;
      double frac_discard = 0.0;
      double frac_do_contr = 0.1;
      size_t nv_lc = nvs[0];
      while( level[0] < max_L-1 && fm->GNV()>amg_options->GetCoarsestMaxVerts())
	{
#ifdef SCOREP
	  SCOREP_User_RegionHandle sco_handle = SCOREP_USER_INVALID_REGION;
	  string title = "level " + std::to_string(level[0]) + " " + std::to_string(level[1]) + " " + std::to_string(level[2]);
	  SCOREP_USER_REGION_INIT(sco_handle, title.c_str(), SCOREP_USER_REGION_TYPE_COMMON);
	  SCOREP_USER_REGION_ENTER(sco_handle);
#endif
	  if( !discard_locked && (grid_step = TryElim(level, fm)) != nullptr ) {
	    discard_locked = true;
	    level[2] = 1;
	    cnt_ld = 0;
	  }
	  else if( !contr_locked && (grid_step = TryContract(level, fm)) != nullptr ) {
	    contr_locked = true;
	    level[1] = 1;
	    cnt_lc = 0;
	  }
	  else if( (grid_step = TryCoarsen(level, fm)) != nullptr ) {
	    level[0]++;
	    level[1] = level[2] = 0;
	    cnt_lc++; cnt_ld++;
	    this->use_hierarch = !this->use_hierarch;
	  }
	  else { cout << "warning, no map variant worked!" << endl; break; }

	  dof_step = grid_step->GetDOFMapStep(fpd, *this);
	  dof_map->AddStep(dof_step);

	  fpd = dof_step->GetMappedParDofs();
	  levels.Append(level);
	  fm = grid_step->GetMappedMesh();

	  nvs.Append((fm!=nullptr ? fm->GNV() : 0));
	  double frac = (1.0*nvs.Last())/(1.0*nvs[nvs.Size()-2]);
	  if(level[1]==0 && level[2]==0) frac_coarse = frac;
	  if(level[2]!=0) frac_discard = frac;
	  if(level[1]!=0) nv_lc = nvs.Last();
#ifdef SCOREP
	  SCOREP_USER_REGION_END(sco_handle);
#endif
	  if (fm==nullptr) {
	    break;
	  }
	  if(level[2]==0 && level[1]==0) {
	    meshes.Append(level[0]>2 ? fm : nullptr);
	    pmaps.Append(dynamic_pointer_cast<ProlMap>(dof_step));
	  }
	  if(cnt_lc>3) // we have not contracted for a long time
	    contr_locked = false;
	  else if(frac_coarse>0.7 && level[0]>2 && cnt_lc>1) // coarsening is slowing down
	    contr_locked = false;
	  else if(nvs.Last()/fpd->GetCommunicator().Size()<MIN_V_PP && cnt_lc>1) // too few verts per proc
	    contr_locked = false;
	  else if((1.0*nvs.Last())/nv_lc < frac_do_contr && cnt_lc>1) // if NV reduces by a good factor
	    contr_locked = false;
	  if(level[0]<5) contr_locked = true;
	  if(frac_coarse>0.7 && level[0]>2 && cnt_ld>0) discard_locked = false;
	  
	}
    }    

    for(auto k:Range(levels.Size())) {
      auto pds = dof_map->GetParDofs(k);
      if(pds==nullptr) continue;
      size_t loc = 0;
      for(auto dof:Range(pds->GetNDofLocal()))
	if(!pds->GetDistantProcs(dof).Size()) loc++;
      size_t loc_glob = pds->GetCommunicator().Reduce(loc, MPI_SUM);
      size_t tot_glob = pds->GetNDofGlobal();
      if(!pds->GetCommunicator().Rank()) {
	cout << "frac local DOFs of level " << levels[k][0] << " " << levels[k][1] << " "
	     << levels[k][2] << ": " << (1.0*loc_glob)/(1.0*tot_glob) << endl;
      }
    }	  

    // cout << "saved meshes: " << endl << meshes << endl;
    // cout << "(casted) saved meshes: " << endl;
    // for(auto mesh:meshes)
    //   if(mesh!=nullptr) cout << dynamic_pointer_cast<ElasticityMesh>(mesh) << endl;
    //   else cout << "(ok!)nullptr" << endl;
    // cout << endl;
    // cout << "pmaps: " << endl << pmaps << endl;
    // cout << "(casted) pmaps: " << endl;
    // for(auto pm:pmaps)
    //   if(pm!=nullptr) cout << dynamic_pointer_cast<ProlMap>(pm) << endl;
    //   else cout << "(ok!)nullptr" << endl;
    // cout << endl;
    
      /**      
       // old grid loop
	       while( level[0] < max_L-1 && fm->GNV()>amg_options->GetCoarsestMaxVerts())
	       {
	       #ifdef SCOREP
	       SCOREP_User_RegionHandle sco_handle = SCOREP_USER_INVALID_REGION;
	       string title = "level " + std::to_string(level[0]) + " " + std::to_string(level[1]) + " " + std::to_string(level[2]);
	       SCOREP_USER_REGION_INIT(sco_handle, title.c_str(), SCOREP_USER_REGION_TYPE_COMMON);
	       SCOREP_USER_REGION_ENTER(sco_handle);
	       // SCOREP_USER_REGION(title.c_str(), SCOREP_USER_REGION_TYPE_DYNAMIC);
	       #endif
	       // discard_locked = false;
	       // contr_locked = true;
	       if( !discard_locked && (grid_step = TryElim(level, fm)) != nullptr ) {
	       level[2] = 1;
	       }
	       else if( !contr_locked && (grid_step = TryContract(level, fm)) != nullptr ) {
	       level[1] = 1;
	       // fds.Append(nullptr);
	       cctr = 0;
	       }
	       else if( (grid_step = TryCoarsen(level, fm)) != nullptr ) {
	       discard_locked = (level[2]!=0) && (level[0]%2==0);
	       // contr_locked = (level[1]!=0);
	       if(contr_locked) cctr++;
	       contr_locked = (cctr<2);
	       level[0]++;
	       // smooth_next_prol = !pw_levs.Contains(level[0]);
	       smooth_next_prol = amg_options->GetSmoothLevel(level[0]);
	       if(level[1]==1) { // after contraction, insert one pw+skip-level
	       smooth_next_prol = false;
	       amg_options->SetAssembleLevel(level[0],0);
	       }
	       // this->use_hierarch = !this->use_hierarch;
	       this->use_hierarch = true;
	       smoothed_it.Append(smooth_next_prol ? 1 : 0);
	       level[1] = level[2] = 0;
	       }
	       else { cout << "warning, no map variant worked!" << endl; break; }
	       dof_step = grid_step->GetDOFMapStep(fpd, *this);
	       dof_map->AddStep(dof_step);
	       fpd = dof_step->GetMappedParDofs();
	       levels.Append(level);
	       fm = grid_step->GetMappedMesh();
	       nvs.Append((fm!=nullptr ? fm->GNV() : 0));
	       #ifdef SCOREP
	       SCOREP_USER_REGION_END(sco_handle);
	       #endif
	       if (fm==nullptr) {
	       break;
	       }
	       double frac = (1.0*nvs.Last())/(1.0*nvs[nvs.Size()-2]);
	       if(frac>0.75) contr_locked = false;
	       }
	       }
      **/


#ifdef SCOREP
    {
      SCOREP_USER_REGION("drop-sync (mesh)", SCOREP_USER_REGION_TYPE_COMMON);
      MPI_Barrier(ngs_comm);
    }
#endif

    Array<size_t> cutoffs;
    {
      static Timer t("AMG - prol smoothing");
      RegionTimer rt(t);
#ifdef SCOREP
      SCOREP_USER_REGION("Smoothed Prolongation", SCOREP_USER_REGION_TYPE_COMMON);
#endif
      /** 
	  Decide where we Smooth prolongations
	     if we are a coarse-map &&
	     if opts->smooth is set on that level &&
	     if we still have the mesh saved &&
	     if we did not contract the step before  &&
	  Decide to assemble matrix on level K:
	     if step K-1 is coarse   (assembling after contr. or discard makes no sense) &&
	     ( if level low && ass. is set || 
	     if level hight && if NV[k] < 0.25 * last_ass_nv )
	     
      **/
      bool skip = false;
      cutoffs.Append(0);
      bool ls_cd = false;
      size_t last_nv_ass = nvs[0];
      size_t last_nv_smo = nvs[0];
      bool do_cout = false;
      if(do_cout) { cout << "rk " << glob_rk << ", levels: " << endl << levels << endl; }
      for(auto k :Range(levels.Size())) {
	auto L = levels[k];
	if(L[0]==0) continue;
	// L[0]==1 maybe also skip smoothing
	if(L[1]!=0 || L[2]!=0) { ls_cd=true; if(do_cout) { cout << "level " << L << "is cd!" << endl; } continue; }
	bool assit = true, smoothit = true;
	auto curr_nv = nvs[k];

	if(do_cout) {
	  cout << endl << "k is " << k << ", level " << L << endl;
	  cout << "curr nv " << curr_nv << endl;
	  cout << "last nvs " << last_nv_ass << " " << last_nv_smo << endl;
	  cout << "facs * last nvs " << 0.25*last_nv_ass << " " << 0.5*last_nv_smo << endl;
	}
	
	if(amg_options->GetAssembleLevel(L[0])==0) { assit = false; if(do_cout) { cout << "ass not set" << endl; } }
	else if(amg_options->GetAssembleLevel(L[0])==1) { assit = true; if(do_cout) { cout << "ass set!" << endl; } }
	else {
	  if(ls_cd) { assit = false; if(do_cout) { cout << "last cd" << endl; } }
	  if( (L[0]>6 && curr_nv>0.25*last_nv_ass) ) { assit = false; if(do_cout) { cout << "nv too large" << endl; } }
	}
	if (assit) { last_nv_ass = curr_nv; cutoffs.Append(k); }
	
	if(amg_options->GetSmoothLevel(L[0]-1)==0) { smoothit = false; if(do_cout) { cout << "smooth not set " << endl; } }
	else if(amg_options->GetSmoothLevel(L[0]-1)==1) { smoothit = true; if(do_cout) { cout << "smooth set! " << endl; } }
	else {
	  if (ls_cd) { smoothit = false; if(do_cout) { cout << "last cd" << endl; } }
	  if( (L[0]>6 && curr_nv>0.5*last_nv_smo) ) { smoothit = false; if(do_cout) { cout << "nv too large " << endl; } }
	}
	if(L[0]<=2) smoothit = false; //we dont save the first couple of meshes
	if(smoothit && (meshes[L[0]-1]==nullptr)) { smoothit = false; if(do_cout) { cout << "dont have mesh " << L[0]-1 << endl; } }
	
	smoothed_it.Append(smoothit ? 1 : 0);
	if (smoothit) {
	  if(meshes[L[0]-1]==nullptr) {
	    cout << "WARNING wanted to smooth level " << L << ", but dont have mesh" << endl;
	    cerr << "WARNING wanted to smooth level " << L << ", but dont have mesh" << endl;
	    ls_cd = false;
	    continue;
	  }
	  last_nv_smo = curr_nv;
	  if(!ls_cd) {
	    auto & pm = pmaps[L[0]-1];
	    auto sprol = SmoothProl(pm, dynamic_pointer_cast<ElasticityMesh>(meshes[L[0]-1]));
	    double epr_loc = (sprol->Height()>0) ? (1.0*sprol->AsVector().Size())/sprol->Height()/dofpv(D) : 0;
	    double epr = pm->GetParDofs()->GetCommunicator().Reduce(epr_loc, MPI_MAX);
	    if(!glob_rk) {
	      cout << "s-prol (mat-)epr level " << L[0]-1 << " -> " << L[0] << ": " << epr << endl;
	    }
	    pm->SetProlongation(sprol);
	  }
	  meshes[L[0]-1] = nullptr;
	}
	ls_cd = false;
      }
      if(!glob_rk) cout << endl;
      if(cutoffs.Last()!=levels.Size()-1)
	cutoffs.Append(levels.Size()-1);
      if(do_cout) cout << "cutoffs: " << endl << cutoffs << endl;
    }
    

    // // Decide where to assemble matrices
    // size_t last_nv = nvs[0];
    // for(auto [k,l]:Enumerate(levels)) {
    //   if(k==0) continue;
    //   if(l[1]!=0 || l[2]!=0) continue;
    //   auto curr_nv = nvs[k];
    //   if( (l[0]<8 && amg_options->GetAssembleLevel(l[0])) ||
    // 	  (l[0]>=8 && curr_nv < 0.25 * last_nv) ) {
    // 	cutoffs.Append(k);
    // 	last_nv = curr_nv;
    //   }
    // }
    // // last level is always assembled
    

    if(glob_rk==0) {
      cout << endl;
      cout << endl;
      cout << "LEVEL PROG: " << endl;
      for(auto k:Range(size_t(1), levels.Size())) {
	cout << levels[k] << "   ";
	cout << k-1 << "->" << k << " | ";
	cout << nvs[k-1] << "->" << nvs[k] << " | ";
	cout << (1.0*nvs[k])/nvs[k-1] << " | ";
	if(levels[k][0]>levels[k-1][0]) cout << "coarse";
	if(levels[k][1]>levels[k-1][1]) cout << "contr";
	if(levels[k][2]>levels[k-1][2]) cout << "discard";
	if(levels[k][1]==0 && levels[k][2]==0) cout << " |  " << (smoothed_it[levels[k][0]]==1 ? "s" : "p");
	if(cutoffs.Contains(k)) cout << " | <- assemble";
	    
	cout << endl;
      }
      cout << endl;
      cout << endl;
      cout << endl;
    }
    dof_map->SetCutoffs(cutoffs);

    auto fpm = const_cast<ParallelMatrix*>(dynamic_cast<const ParallelMatrix*>(finest_bmat));
    shared_ptr<SparseMatrix<double>> fspm;
    if(fpm) fspm = dynamic_pointer_cast<SparseMatrix<double>>(fpm->GetMatrix());
    else fspm = shared_ptr<SparseMatrix<double>>(const_cast<SparseMatrix<double>*>(dynamic_cast<const SparseMatrix<double>*>(finest_bmat)), NOOP_Deleter);
    Array<shared_ptr<SparseMatrix<double>>> mats;//(max_L+1);
    {
      static Timer t("AMG - coarse mats");
      RegionTimer rt(t);
      // mats.SetSize(0);
      mats = dof_map->AssembleMatrices(fspm);
    }

#ifdef SCOREP
      {
	SCOREP_USER_REGION("drop-sync (assemble)", SCOREP_USER_REGION_TYPE_COMMON);
	MPI_Barrier(ngs_comm);
      }
#endif

      
    
    Array<double> tots;
    double all_tots = 0;
    for(auto K:Range(mats.Size())) {
      if(!mats[K]) continue;
      Array<MemoryUsage> mem = mats[0]->GetMemoryUsage();
      double tot = 0;
      for(auto k:Range(mem.Size())) {
	double nbs = mem[k].NBytes()/1024.0/1024.0;
	tot += nbs;
	// cout << mem[k].Name() << ": " << nbs << endl;
      }
      tots.Append(tot);
      all_tots += tot;
    }
    double tot_mem0 = glob_comm.AllReduce(tots[0], MPI_SUM);
    double all_tot_mem = glob_comm.AllReduce(all_tots, MPI_SUM);
    double max_tot_mem = glob_comm.AllReduce(all_tots, MPI_MAX);
    if(!glob_rk) {
      cout << "tot mem level 0 " << tot_mem0 << endl;
      cout << "tot mem all levels " << all_tot_mem << endl;
      cout << "max loc tot mem all level " << max_tot_mem << endl;
    }
    
    
    // { // test coarsest level kernel vecs!!
    //   for(auto [L,fspmat] : Enumerate(mats)) {
    // 	if(fspmat==nullptr || fspmat->Width()>2000 || fspmat->Height()==0 ) continue;
    // 	size_t h = fspmat->Height();
    // 	Matrix<double> full(h,h);
    // 	if(h) full = 0.0;
    // 	const SparseMatrix<double>* SPM = dynamic_cast<SparseMatrix<double>*>(fspmat.get());
    // 	for(auto k:Range(h))
    // 	  for(auto j:Range(h)) {
    // 	    full(k,j) = (*SPM)(k,j);
    // 	  }
    // 	Matrix<double> evecs(h,h);
    // 	Vector<double> evals(h);
    // 	LapackEigenValuesSymmetric(full, evals, evecs);
    // 	auto nzero = 0;
    // 	for(auto k:Range(evals.Size()))
    // 	  if(sqr(evals(k))<sqr(1e-13)) nzero++;
    // 	cout << endl;
    // 	cout << "LEVEL " << L << ", have " << nzero << " kernel evecs!!" << endl;
    // 	cout << "dim " << fspmat->Width() << ", smallest evals: " << endl; prow(evals.Range(0,dofpv(D))); cout << endl;
    // 	cout << endl;
    //   }
    // } // end test coarsest level evals
    
    Array<shared_ptr<const BaseMatrix>> bmats(mats.Size());
    Array<shared_ptr<const BaseSmoother>> smoothers;
    Array<double> nzes, nzepr, vc, lnzes;
    {
#ifdef SCOREP
      SCOREP_USER_REGION("statistics", SCOREP_USER_REGION_TYPE_COMMON);
#endif
      // cout << "assembled mats: " << endl << mats << endl;
      for(auto [m1,m2] : Zip(bmats, mats)) m1 = m2;
      auto get_nze = [&](auto mat, auto pd) { return pd->GetCommunicator().AllReduce(mat->AsVector().Size(), MPI_SUM); };
      
      for (auto [L,mat] : Enumerate(mats.Part(0, mats.Size()-1))) {
	nzes.Append(get_nze(mats[L], dof_map->GetParDofs(L)));
	lnzes.Append(mats[L]->AsVector().Size());
	nzepr.Append(nzes.Last()/(1.0*dof_map->GetParDofs(L)->GetNDofGlobal()));
	vc.Append( (L==0) ? 1.0 : (1.0*dof_map->GetParDofs(L)->GetNDofGlobal())/dofpv(D));
      }
      

    }
    
    // If we cut off the first prol, we have "dummy"-rotations on the coarse levels.
    // These only have 0-entries in the matrix.
    // We take all r-r diag-blocks of the coarse matrices and regularize them.
    // if a block is 0, we just add 1s in its' diagonal
    // a block cannot have rank 1!
    // If a block has rank 2, we compute the kernel vec kv and add kv*kv.T
    if(cut_blf!=nullptr) {
      static Timer t("AMG - regularize");
      RegionTimer rt(t);

      auto calc_det = [](const auto & mat) {
	return mat(0,0)*mat(1,1)*mat(2,2) + mat(0,1)*mat(1,2)*mat(2,0)+mat(0,2)*mat(1,0)*mat(2,1)
	- mat(2,0)*mat(1,1)*mat(0,2) - mat(2,1)*mat(1,2)*mat(0,0) - mat(2,2)*mat(1,0)*mat(0,1);
      };
      auto calc_trace = [](const auto & mat) {
	return mat(0,0)+mat(1,1)+mat(2,2);
      };

      Matrix<double> r_block(rotpv(D),rotpv(D)), evecs(rotpv(D),rotpv(D));
      Vector<double> rdof(rotpv(D)), evals(rotpv(D)), kv(rotpv(D));
      // Matrix<double> r_block(rotpv(D), rotpv(D));
      if(mats.Size()>1) {
	for(auto mat_nr:Range(size_t(1), mats.Size())) {
	  // cout << "regularize mat " << k << endl;
	  //if I drop, this can be a nullptr
	  if(mats[mat_nr]==nullptr) continue;
	  // cout << "mat " << mat_nr << " of " << mats.Size() << endl;
	  const auto & mat = *mats[mat_nr];
	  auto & write_mat = *mats[mat_nr];
	  auto pds = dof_map->GetParDofs(mat_nr);

	  size_t NV = mat.Height()/dofpv(D);
	  TableCreator<int> cvp(NV);
	  for(;!cvp.Done(); cvp++) {
	    for(auto k:Range(NV))
	      for(auto p:pds->GetDistantProcs(k))
		cvp.Add(k,p);
	  }
	  constexpr int MS = rotpv(D)*rotpv(D);
	  auto block_pds = make_shared<ParallelDofs>(pds->GetCommunicator(), cvp.MoveTable(), MS, false);

	  ParallelVVector<Vec<MS>> diags(block_pds, DISTRIBUTED);
	  diags.FVDouble() = 0.0;
	  // cout << "sizes " << NV << " " << diags.Size() << " " << diags.FVDouble().Size() << endl;
	  // cout << "0-ed diags: " << endl << diags << endl;
	  
	  /**
	     with MPI: 
	     - accumulate diag-blocks
	     - compute kernel of diag-blocks + regularize
	  **/
	  for(auto v : Range(NV)) {
	    for(auto i:Range(rotpv(D)))
	      rdof(i) = NV*(disppv(D)+i) + v;
	    double* mem = &diags.FVDouble()[MS*v];
	    for(auto i:Range(rotpv(D)))
	      for(auto j:Range(rotpv(D)))
		mem[rotpv(D)*i+j] = mat(rdof(i), rdof(j));
	  }
	  diags.Cumulate();
	  for(auto v : Range(NV)) {
	    for(auto i:Range(rotpv(D)))
	      rdof(i) = NV*(disppv(D)+i) + v;
	    Vec<9> mem = diags(v);
	    for(auto i:Range(rotpv(D)))
	      for(auto j:Range(rotpv(D)))
		r_block(i,j) = mem(rotpv(D)*i+j);
	    // cout << "get r block " << v << endl << r_block << endl;
	    LapackEigenValuesSymmetric(r_block, evals, evecs);
	    // cout << "evals " << endl << evals << endl;
	    if(!pds->IsMasterDof(v)) { /*cout << "skip! " << endl;*/ continue; }
	    bool is_sing = (fabs(evals(2))<1e-15) || (evals(0)/evals(2) < 1e-14);
	    if(!is_sing) continue; // rank==3, do nothing
	    // cout << "singular!" << endl;
	    // cout << "evecs " << endl << evecs << endl;
	    if(evals(2)==0.0) {
	      auto val = 1.0/(1+pds->GetDistantProcs(v).Size());
	      for(auto i:Range(rotpv(D)))
		write_mat(rdof(i), rdof(i)) += val;
	    }
	    kv = evecs.Rows(0,1);
	    // double val = evals(2)/(L2Norm(kv)*(1+pds->GetDistantProcs(v).Size()));
	    double val = evals(2)/(L2Norm(kv));
	    kv *= val;
	    // cout << "scaled kv: " << kv << endl;
	    for(auto i:Range(rotpv(D)))
	      for(auto j:Range(rotpv(D)))
		write_mat(rdof(i), rdof(j)) += kv(i)*kv(j);
	  }
	  // cout << "mat done " << endl;
	  pds->GetCommunicator().Barrier();
	}
      }
    }

    
    {
	static Timer t("AMG - setup smoothers");
	RegionTimer rt(t);
      
	for (auto [L,mat] : Enumerate(mats.Part(0, mats.Size()-1))) {
	  // cout << "smoother for: " << L << " " << mats[L] << endl;
	  auto pml = make_shared<ParallelMatrix>(mats[L], dof_map->GetParDofs(L));
	  // smoothers.Append(BuildSmootherOfLevel(L, pml, ((L==0)?finest_freedofs:nullptr)));
	  if(L==0 && cut_blf!=nullptr)
	    smoothers.Append(make_shared<FixedBSFixedOS_HGSS<disppv(D)>>(pml, ((L==0)?finest_freedofs:nullptr)));
	  else
	    smoothers.Append(make_shared<FixedBSFixedOS_HGSS<dofpv(D)>>(pml, ((L==0)?finest_freedofs:nullptr)));
	}
      }

#ifdef SCOREP
      {
	SCOREP_USER_REGION("drop-sync (smoothers)", SCOREP_USER_REGION_TYPE_COMMON);
	MPI_Barrier(ngs_comm);
      }
#endif
    
      amg_matrix = make_shared<BaseAMGMatrix> (dof_map, smoothers, bmats);
      
    
      if (mats.Last()!=nullptr) {
	// cout << "coarse mat: " << endl << *mats.Last() << endl;
	// cout << "diag : " << endl;
	// const auto & cmat = *mats.Last();
	// for(auto k:Range(mats.Last()->Height())) cout << k << ": " << cmat(k,k) << endl;
	// cout << endl;
	if(glob_comm.Size()>1) {
	  auto cpm = make_shared<ParallelMatrix>(mats.Last(), dof_map->GetMappedParDofs());
	  cpm->SetInverseType(amg_options->GetCoarseInvType());
	  // cpm->SetInverseType("mumps");
	  {
#ifdef SCOREP
	    SCOREP_USER_REGION("coarse inv", SCOREP_USER_REGION_TYPE_COMMON);
#endif
	    auto cinv = cpm->InverseMatrix();
	    amg_matrix->AddFinalLevel(cinv);
	  }
	}
	else {
	  mats.Last()->SetInverseType("sparsecholesky");
	  // cout << "final mat: " << endl << *mats.Last() << endl;
	  // cout << "sp-chol inv! " << endl;
	  {
#ifdef SCOREP
	    SCOREP_USER_REGION("coarse inv", SCOREP_USER_REGION_TYPE_COMMON);
#endif
	    auto cinv = mats.Last()->InverseMatrix();
	    amg_matrix->AddFinalLevel(cinv);
	  }
	}
      }
    

      double oc_tot = 0.0;
      Array<double> occ;
      for (auto v:nzes) {
	occ.Append(v/nzes[0]);
	oc_tot += occ.Last();
      }
      double loc_tot = 0.0;
      Array<double> locc;
      if (lnzes[0]!=0)
	for (auto v:lnzes) {
	  locc.Append(v/lnzes[0]);
	  loc_tot += locc.Last();
	}
      auto vc0 = vc[0];
      double vc_tot = 0.0;
      for (auto& v:vc) {
	v = v/vc0;
	vc_tot += v;
      }
    
      double loc_max = glob_comm.AllReduce(loc_tot, MPI_MAX);
    
    if(loc_tot==loc_max) {
      cout << "MAX LOC: " << loc_max << endl;
      cout << "OC: " << oc_tot << endl;
      prow(occ); cout << endl;
      cout << glob_rk << ", LOC: " << loc_tot << endl;
      prow(locc); cout << endl;
      cout << "VC: " << vc_tot << endl;
      prow(vc); cout << endl;
      cout << "perow: " << endl;
      prow(nzepr); cout << endl;
    }
    
#ifdef SCOREP
    {
      SCOREP_USER_REGION("drop-sync (setup)", SCOREP_USER_REGION_TYPE_COMMON);
      MPI_Barrier(ngs_comm);
    }
#endif

  } // end SetupAMGMatrix


  // template<int D>
  // void ElasticityAMGPreconditioner<D> ::
  // CalcReplMatrix(const idedge & edge, FlatMatrix<double> & mat)
  // { CalcReplMatrix(edge_mats[edge.id], p_coords[edge.v[0]], p_coords[edge.v[1]], mat); }

  template<int D>
  void ElasticityAMGPreconditioner<D> ::
  CalcReplMatrix (edge_data & edata, Vec<3,double> cv0, Vec<3,double> cv1, FlatMatrix<double> & mat)
  {
    HeapReset hr(my_lh);
    Vec<D> t = cv1 - cv0;
    static Timer t0("ElasiticityAMG::CalcReplMatrix");
    RegionTimer rt(t0);
    mat = 0.0;
    auto get_ind = [](auto v, auto type, auto comp)
      { return 2*type*disppv(D) + 2*comp + v;};
    FlatMatrix<double> disp_trans(D, 2*dofpv(D), my_lh);
    disp_trans = 0.0;
    FlatMatrix<double> skew_t(D, rotpv(D), my_lh);
    skew_t = 0;
    // auto & edata = edge_mats[edge.id];
    auto edisp = edata.edisp();
    auto erot = edata.erot();
    // cout << "edge: " << edge << " (len: " << edge_mats.Size() << ")" << endl;
    // cout << "disp/rot mats: " << endl << edisp << erot;
    if constexpr(D==3) {
      skew_t(0,1) = -(skew_t(1,0) = -t[2]);
      skew_t(0,2) = -(skew_t(2,0) = t[1]);
      skew_t(1,2) = -(skew_t(2,1) = -t[0]);
    }
    else {
      skew_t(0,0) = -t[1];
      skew_t(1,0) = t[0];
    }
    for(auto i:Range(disppv(D))) {
      disp_trans(i,get_ind(0,0,i)) = -(disp_trans(i,get_ind(1,0,i)) = 1.0);
      for(auto j:Range(rotpv(D))) {
	disp_trans(i, get_ind(0,1,j)) = (disp_trans(i, get_ind(1,1,j)) = 0.5 * skew_t(i,j));
      }
    }
    // cout << "disp trans: " << endl << disp_trans << endl;
    FlatMatrix<double> temp_mat(disppv(D), 2*dofpv(D), my_lh);
    temp_mat = edisp * disp_trans;
    // cout << "temp disp: " << endl << temp_mat << endl;
    mat += Trans(disp_trans) * temp_mat;
    // cout << "only disp: " << endl << mat << endl;
    // FlatMatrix<double> rot_trans(rotpv(D), 2*rotpv(D), my_lh);
    // rot_trans = 0.0;
    // for(auto i:Range(rotpv(D))) {
    //   rot_trans(i,get_ind(0,1,i)) = -(rot_trans(i,get_ind(1,1,i)) = 1.0);
    // }
    // cout << "rot_trans: " << endl << rot_trans << endl;
    // FlatMatrix<double> temp_mat2(2*rotpv(D), rotpv(D), my_lh);
    // temp_mat2 = erot * Trans(rot_trans);
    // mat.Rows(disppv(D), dofpv(D)).Cols(disppv(D), dofpv(D)) += rot_trans * temp_mat2;

    for(auto i:Range(rotpv(D))) {
      auto ii0 = get_ind(0, 1, i);
      auto ii1 = get_ind(1, 1, i);
      for(auto j:Range(rotpv(D))) {
	auto jj0 = get_ind(0, 1, j);
	auto jj1 = get_ind(1, 1, j);
	auto v = erot(i,j);
	mat(ii0, jj0) += v;
	mat(ii1, jj1) += v;
	mat(ii0, jj1) -= v;
	mat(ii1, jj0) -= v;
      }
    }
    // { // check RBMs
    //   FlatMatrix<double> evecs(mat.Height(), mat.Width(), my_lh);
    //   FlatVector<double> evals(mat.Height(), my_lh);
    //   cout << "repl mat for edge " << edge << ": " << endl << mat << endl;
    //   LapackEigenValuesSymmetric(mat, evals, evecs);
    //   cout << "evals: " << endl << evals << endl;
    //   // cout << "evecs: " << endl << Trans(evecs) << endl;
    //   FlatMatrix<double> rbm(mat.Width(), disppv(D)+rotpv(D), my_lh);
    //   CalcRBM(t, rbm);
    //   cout << "rbms: " << endl << rbm << endl;
    //   FlatMatrix<double> mtrbm(mat.Width(), disppv(D)+rotpv(D), my_lh);
    //   mtrbm = mat * rbm;
    //   double sum = 0;
    //   for(auto i:Range(mtrbm.Height()))
    // 	for(auto j:Range(mtrbm.Width()))
    // 	  sum += sqr(mtrbm(i,j));
    //   if(sum>3e-7) {
    // 	cout << "mat*rbm: " << endl << mtrbm << endl;
    //   }
    // } // end check RBMs
  }

  template<int D>
  void ElasticityAMGPreconditioner<D> ::
  CalcRBM (Vec<D> & t, FlatMatrix<double> & rbm) {
    if constexpr(D==3) {
	throw Exception("calcrbm only implemented for 2d!");
      }
    HeapReset hr(my_lh);
    auto get_ind = [](auto v, auto type, auto comp)
      { return 2*type*disppv(D) + 2*comp + v;};
    rbm = 0.0;
    for(auto i:Range(disppv(D))) {
      rbm(get_ind(0, 0, i), i) = rbm(get_ind(1,0,i), i) = 1.0;
    }
    Vec<D> rdir, ei;
    if constexpr(D==2) {
      rdir(0) =  t(1);
      rdir(1) = -t(0);
      }
    // cout << "tang: " << endl << t << endl;
    for(auto i:Range(rotpv(D))) {
      auto col = disppv(D)+i;
      rbm(get_ind(0,1,i), col) = rbm(get_ind(1,1,i), col) = 1.0;
      if constexpr(D==3) {
	ei = 0;
	ei(i) = 1.0;
	rdir = Cross(t, ei);
      }
      // cout << "rot around e_" << i << ": " << endl << rdir << endl;
      for(auto di:Range(disppv(D))) {
	rbm(get_ind(0,0,di), col) = -(rbm(get_ind(1,0,di), col) = 0.5*rdir(di));
      }
    }    
  }

  
  template<int D>
  void ElasticityAMGPreconditioner<D> ::
  AddElementMatrix (FlatArray<int> dnums,
  			 const FlatMatrix<double> & elmat,
  			 ElementId ei,
  			 LocalHeap & lh)
  {
    // cout << "add elmat, dnums: " << endl << dnums << endl;
    static Timer t0("AddElementMatrix");
    RegionTimer rt(t0);
    constexpr size_t n_disp = D;
    constexpr size_t n_rot = D * (D-1) / 2;
    size_t nd = dnums.Size();
    size_t nv = nd / dofpv(D);
    FlatMatrix<double> edge_schur (2 * dofpv(D), lh);
    FlatVector<double> v1(edge_schur.Height(), lh);
    FlatVector<double> v2(edge_schur.Height(), lh);
    BitArray usedij(nd);
    Vec<D> p1,p2;
    Vec<D> t;
    // Array<Vec<D>> n(D-1);
    auto get_dnr = [&](auto vert, auto comp) -> size_t {
      return comp * nv + vert;
    };
    auto get_ednr = [&](auto vert, auto comp) -> size_t {
      return comp * 2 + vert;
    };
    for(auto d1:Range(nv)) {
      for(auto d2:Range(d1+1,nv)) {
	// cout << "get edge data for " << d1 << " " << d2 << " !" << endl;
	auto dn1 = dnums[d1];
	auto dn2 = dnums[d2];
	// edge_mats are independent of edge orientation
	auto & edata = (*hash_edge)[INT<2>(min2(dn1, dn2), max2(dn1, dn2))];
	// cout << "have edge data" << endl;
	auto edisp = edata.edisp();
	auto erot = edata.erot();
  	/** Calc Schur-complement w.r.t a single edge **/
  	usedij.Clear();
  	for(auto l:Range(dofpv(D))) {
  	  usedij.Set(get_dnr(d1, l));
  	  usedij.Set(get_dnr(d2, l));
  	}
	// cout << "calc schur for " << usedij.NumSet() << " of " << usedij.Size() << endl;
  	CalcSchurComplement(elmat, edge_schur, usedij, lh);
	/** M_d = [I|-I|0|0] \cdot S_E \ cdot [transp.] **/
	for(auto i:Range(disppv(D))) {
	  auto ii0 = get_ednr(0,i);
	  auto ii1 = get_ednr(1,i);
	  for(auto j:Range(disppv(D))) {
	    auto jj0 = get_ednr(0,j);
	    auto jj1 = get_ednr(1,j);
	    edisp(i,j) += edge_schur(ii0, jj0) + edge_schur(ii1, jj1) - edge_schur(ii0, jj1) - edge_schur(ii1, jj0);
	  }
	}
	/** M_r = [0|0|I|-I] \cdot S_E \cdot [transp.] **/
	for(auto i:Range(rotpv(D))) {
	  auto ii0 = get_ednr(0,disppv(D)+i);
	  auto ii1 = get_ednr(1,disppv(D)+i);
	  for(auto j:Range(rotpv(D))) {
	    auto jj0 = get_ednr(0,disppv(D)+j);
	    auto jj1 = get_ednr(1,disppv(D)+j);
	    erot(i,j) += edge_schur(ii0, jj0) + edge_schur(ii1, jj1) - edge_schur(ii0, jj1) - edge_schur(ii1, jj0);
	  }
	}

	// { // check RBM
	//   HeapReset hr(my_lh);
	//   Vec<D> t = p_coords[dnums[d2]] - p_coords[dnums[d1]];
	//   FlatMatrix<double> rbm(edge_schur.Width(), disppv(D)+rotpv(D), my_lh);
	//   CalcRBM(t, rbm);
	//   // cout << "RBMS should be: " << endl << rbm << endl;
	//   FlatMatrix<double> mtrbm(edge_schur.Width(), disppv(D)+rotpv(D), my_lh);
	//   mtrbm = edge_schur * rbm;
	//   for(auto i:Range(mtrbm.Height()))
	//     for(auto j:Range(mtrbm.Width()))
	//       if(sqr(mtrbm(i,j)) < sqr(1e-15))
	// 	mtrbm(i,j) = 0.0;
	//   cout << "RBM * ESCHUR: " << endl << mtrbm << endl;
	// } // end check RBM
	
	// cout << "weights: " << endl;
	// CalcEdgeWeights(edata);
      }
    }
  }


  template<int D> shared_ptr<typename ElasticityAMGPreconditioner<D>::ElasticityMesh>
  ElasticityAMGPreconditioner<D>::BuildInitialAlgMesh ()
  {
    static Timer t("ElasticityAMGPreconditioner::BuildInitialAlgMesh");
    RegionTimer rt(t);

    // finest_freedofs->Clear();
    // finest_freedofs->Invert();

    size_t nd = bfa->GetFESpace()->GetNDof();
    size_t nv = nd/dofpv(D);
    
    size_t ne1 = 0;
    for(auto key_val : *hash_edge)
      ne1++;
    Array<edge> edges1(ne1);
    Array<double> ewts1(ne1);
    ne1 = 0;
    ewts1 = 0.0; //just dummy...
    for(auto key_val : *hash_edge) {
      edges1[ne1] = key_val.first;
      auto & e_data = key_val.second;
      ne1++;
    }

    size_t nf = 0;
    Array<face> faces1;
    Array<double> fwts1;
    shared_ptr<ParallelDofs> pd;
    if ( (cut_blf!=nullptr) && (bfa->GetFESpace()->IsParallel()!=cut_blf->GetFESpace()->IsParallel()) ) {
      throw Exception("Elasticity AMG with cutoff and one par/one seq space!");
    }
    if (bfa->GetFESpace()->IsParallel()) {
      pd = cut_blf!=nullptr ? cut_blf->GetFESpace()->GetParallelDofs() : bfa->GetFESpace()->GetParallelDofs();
    }
    else{ // but still compiled with mpi ...
      Table<int>tab(cut_blf==nullptr ? bfa->GetFESpace()->GetNDof() : cut_blf->GetFESpace()->GetNDof(), 0);
      pd = make_shared<ParallelDofs>(MPI_COMM_WORLD, move(tab));
    }
    finest_pds = pd;
    
    auto eqc_h = make_shared<EQCHierarchy>(pd, true);
    Array<size_t> eqcv(nv);
    for(auto k:Range(nv))
      eqcv[k] = eqc_h->FindEQCWithDPs(pd->GetDistantProcs(k));
    Array<double> weights_vertices(nv);
    weights_vertices = 0.0;
    
    BlockedAlgebraicMesh bmesh(nv, ne1, nf,
			      std::move(edges1), std::move(faces1),
			      std::move(weights_vertices), std::move(ewts1),
			      std::move(fwts1),
			      std::move(eqcv), eqc_h);
    
    /** dirichlet-data **/
    bmesh.SetFreeVertices(finest_freedofs);

    /** cumulate edge weight matrices **/
    // cout << " cumulate edge weights!" << endl;
    auto ne = bmesh.NE();
    edge_mats.SetSize(ne);
    int bnr, pos;
    bmesh.BuildMeshBlocks();
    const auto & eqcs = bmesh.GetEQCS();
    for(auto [eqc, mesh_block] : bmesh) {
      auto & bedges = mesh_block->Edges();
      auto bpedges = bmesh.GetPaddingEdges(eqc);
      auto iteratit = [this, &bnr, &pos](const auto & a) {
	for(const auto & edge:a) {
	  if(hash_edge->Used(edge.v, bnr, pos)) {
	    edge_mats[edge.id] = hash_edge->Get(bnr, pos);
	  }
	}
      };
      iteratit(bedges);
      iteratit(bpedges);
    }
    delete hash_edge;
    bmesh.CumulateEdgeData(edge_mats);
    
    // for(auto& c:p_coords) c[2] = 1.0; //check coarsening!
    // cout << "initial coords: " << endl << p_coords << endl;
    
    auto evd = make_unique<EVD>(move(p_coords), CUMULATED);
    auto eed = make_unique<EED>(move(edge_mats), CUMULATED);
    return make_shared<ElasticityAMGPreconditioner<D>::ElasticityMesh>(move(bmesh), move(evd), move(eed));
  }
  


  template<int D>
  PARALLEL_STATUS ElasticityAMGPreconditioner<D>::EVD::
  reduce_func(CoarseMap & cmap, FlatArray<Vec<3,double>> _data, FlatArray<Vec<3,double>> _cdata)
  {
    auto vmap = cmap.GetMap<NT_VERTEX>();
    auto fbmesh = dynamic_pointer_cast<ElasticityMesh>(cmap.GetMesh());
    auto eqc_h = fbmesh->GetEQCHierarchy();
    _cdata = 0.0;
    Array<int> touched(vmap.Size()); //TODO: faster with Parvec?
    touched = 0;
    auto doit = [&](const auto & es) {
      for(auto e : es) {
	auto CV = vmap[e.v[0]];
	if(CV==-1) continue;
	if(CV!=vmap[e.v[1]]) continue;
	_cdata[CV] = 0.5 * (_data[e.v[0]] + _data[e.v[1]]);
	touched[e.v[0]] = touched[e.v[1]] = 1.0;
      }
    };
    for(auto eqc:Range(eqc_h->GetNEQCS())) {
      if(!eqc_h->IsMasterOfEQC(eqc)) continue;
      doit((*fbmesh)[eqc]->template GetNodes<NT_EDGE>());
      doit((*fbmesh)[eqc]->template GetNodes_cross<NT_EDGE>());
    }
    fbmesh->CumulateVertexData(touched);
    auto doit_v = [&](const auto & vs) {
      for(auto V:vs) {
	auto CV = vmap[V];
	if(CV==-1 || touched[V]!=0.0) continue;
	if(_cdata[CV]==0.0) _cdata[CV] = _data[V];
      }
    };
    for(auto eqc:Range(eqc_h->GetNEQCS())) {
      if(!eqc_h->IsMasterOfEQC(eqc)) continue;
      doit_v((*fbmesh)[eqc]->template GetNodes<NT_VERTEX>());
    }
    return DISTRIBUTED;
  }
  
  
  template<int D>
  PARALLEL_STATUS ElasticityAMGPreconditioner<D>::EED::reduce_func(CoarseMap & cmap,
								   FlatArray<ElasticityAMGPreconditioner<D>::edge_data> edge_mats,
								   FlatArray<ElasticityAMGPreconditioner<D>::edge_data> cedge_mats)
  {
    auto mesh = dynamic_pointer_cast<ElasticityAMGPreconditioner<D>::ElasticityMesh>(cmap.GetMesh());
    // cout << "reduce edge data, mesh: " << endl << *mesh << endl;
    // Array<Vec<3,double>> & coords = std::get<0>(mesh->Data())->Data();
    auto & coords = std::get<0>(mesh->Data())->Data();
    cedge_mats = 0.0;
    this->Cumulate();
    auto vmap = cmap.GetMap<NT_VERTEX>();
    auto emap = cmap.GetMap<NT_EDGE>();
    // cout << "vmap: " << endl << vmap << endl;
    // cout << "emap: " << endl << emap << endl;
    Matrix<double> skew_t(disppv(D), rotpv(D));
    Matrix<double> tmp(disppv(D), rotpv(D));
    Matrix<double> add_mat(rotpv(D), rotpv(D));
    Vec<D> t;
    Array<size_t> common_ovs;
    auto eqc_h = mesh->GetEQCHierarchy();
    // jump of coarse rot also has to has a fine disp (stretch/twist/wiggle) energy (i think)
    auto calc_admat = [&](const auto & edge) {
      skew_t = 0;
      if constexpr(D==3) {
	  skew_t(0,1) = -(skew_t(1,0) =  t[2]);
	  skew_t(0,2) = -(skew_t(2,0) = -t[1]);
	  skew_t(1,2) = -(skew_t(2,1) =  t[0]);
	}
      else {
	skew_t(0,0) =  t[1];
	skew_t(1,0) = -t[0];
      }
      auto fdm = edge_mats[edge.id].edisp();
      tmp = fdm * skew_t;
      add_mat = 0.25 * Trans(skew_t) * tmp;
      // add_mat = 0.0;
      // add_mat = Trans(skew_t) * tmp;
    };
    auto econ = mesh->GetEdgeConnectivityMatrix();
    const auto & edges = mesh->Edges();
    auto lam_e = [&](const auto & row) {
      for(auto j:Range(row.Size())) {
	auto edge = row[j];
	auto ec = emap[edge.id];
	if(ec != (size_t)-1) {
	  cedge_mats[ec] += edge_mats[edge.id];
	}
	else if( (vmap[edge.v[0]] == vmap[edge.v[1]]) &&
	 	 (vmap[edge.v[0]] != (size_t)-1) ) {
	  // cout << "edge " << edge << " maps to -1 " << endl;
	  // cout << "d-mat: " << endl << edge_mats[edge.id].edisp();
	  // cout << "r-mat: " << endl << edge_mats[edge.id].erot();
	  // cout << "cv is : " << vmap[edge.v[0]] << endl;
	  t = coords[edge.v[1]] - coords[edge.v[0]];
	  auto ri0 = econ->GetRowIndices(edge.v[0]);
	  auto ri1 = econ->GetRowIndices(edge.v[1]);
	  intersect_arrays(ri0, ri1, common_ovs);
	  for(auto ov:common_ovs) {
	    auto conn_edge_id = (*econ)(edge.v[0], ov);
	    if (emap[conn_edge_id] == -1) continue;
	    auto ce_id = emap[conn_edge_id];
	    auto con_edge = edges[(*econ)(edge.v[0], ov)];
	    auto con_edge2 = edges[(*econ)(edge.v[1], ov)];
	    // cout << "connecting edge " << con_edge << endl;
	    // cout << "there should also be " << con_edge2 << endl;
	    // cout << "edges map to: " << emap[con_edge.id] << " " << emap[con_edge2.id] << endl;
	    // cout << "edge vs map to: " << endl
	    // 	 << vmap[con_edge.v[0]] << " " << vmap[con_edge.v[1]] << endl
	    // 	 << vmap[con_edge2.v[0]] << " " << vmap[con_edge2.v[1]] << endl;
	    auto crm = cedge_mats[ce_id].erot();
	    // cout << "crm before: " << endl << crm;
	    calc_admat(con_edge);
	    crm += add_mat;
	    calc_admat(con_edge2);
	    crm += add_mat;
	    // cout << "crm is now: " << endl << crm;
	  }
	}
      }
    };
    for(auto [eqc, block] : *mesh) {
      if (!eqc_h->IsMasterOfEQC(eqc)) continue;
      // cout << "have " << block->template GetNodes<NT_EDGE>().Size() << " edges" << endl;
      // cout << "edges: " << endl << block->template GetNodes<NT_EDGE>() << endl;
      // cout << "have " << block->template GetNodes_cross<NT_EDGE>().Size() << "cross edges" << endl;
      // cout << "cross edges: " << endl << block->template GetNodes_cross<NT_EDGE>() << endl;
      lam_e(block->template GetNodes<NT_EDGE>());
      lam_e(block->template GetNodes_cross<NT_EDGE>());
    }
    return DISTRIBUTED;
  }


  RegisterPreconditioner<ElasticityAMGPreconditioner<2>> register_elast_2d("amg_elast_2d");
  RegisterPreconditioner<ElasticityAMGPreconditioner<3>> register_elast_3d("amg_elast_3d");
  
} // end namespace amg
