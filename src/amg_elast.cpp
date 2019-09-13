#ifdef ELASTICITY
#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_elast_impl.hpp"
#include "amg_factory_impl.hpp"
#include "amg_pc_impl.hpp"

namespace amg
{

  /** AttachedEED **/


  template<int D> void AttachedEED<D> :: map_data (const BaseCoarseMap & bcmap, AttachedEED<D> & ceed) const
  {
    CoarseMap<ElasticityMesh<D>> * pcmap = dynamic_cast<CoarseMap<ElasticityMesh<D>>*>(const_cast<BaseCoarseMap*>(&bcmap));
    assert(pcmap != nullptr); // "AttachedEED with wrong map, how?!
    auto & cmap(*pcmap);
    // static_assert(std::is_same<TMESH,ElasticityMesh<D>>::value==1, "AttachedEED with wrong map?!");
    // cout << " map edges " << endl;
    static Timer t(string("ElEData<")+to_string(D)+string(">::map_data")); RegionTimer rt(t);
    auto & mesh = static_cast<ElasticityMesh<D>&>(*this->mesh);
    mesh.CumulateData();
    auto sp_eqc_h = mesh.GetEQCHierarchy();
    const auto & eqc_h = *sp_eqc_h;
    // eqc_h.GetCommunicator().Barrier();
    auto neqcs = mesh.GetNEqcs();
    const size_t NE = mesh.template GetNN<NT_EDGE>();
    ElasticityMesh<D> & cmesh = static_cast<ElasticityMesh<D>&>(*ceed.mesh);
    const size_t NCE = cmap.template GetMappedNN<NT_EDGE>();

    // cout << " NE NCE " << NE << " " << NCE << endl;

    ceed.data.SetSize(NCE); ceed.data = 0.0;
    auto e_map = cmap.template GetMap<NT_EDGE>();
    auto v_map = cmap.template GetMap<NT_VERTEX>();
    // if (v_map.Size() < 2000) cout << "e_map: " << endl; prow2(e_map); cout << endl << endl;
    // if (v_map.Size() < 2000) cout << "v_map: " << endl; prow2(v_map); cout << endl << endl;
    auto pecon = mesh.GetEdgeCM();
    // cout << "fine mesh ECM: " << endl << *pecon << endl;
    const auto & econ(*pecon);
    // cout << "cumulate fm vd!" << endl;
    auto fvd = get<0>(mesh.Data())->Data();
    // cout << "cumulate cm vd!" << endl;
    get<0>(cmesh.Data())->Cumulate();
    // cout << "cumulates done!" << endl;
    auto cvd = get<0>(cmesh.Data())->Data();
    auto fed = this->Data();
    auto ced = ceed.Data();
    auto edges = mesh.template GetNodes<NT_EDGE>();

    // cout << "fine edges / mats: " << endl;
    // for (auto & edge : edges) {
    //   cout << "fedge: " << edge << ": " << endl; print_tm(cout, fed[edge.id]); cout << endl;
    // }

    Table<int> c2fe;
    {
      TableCreator<int> cc2fe(NCE);
      for (; !cc2fe.Done(); cc2fe++)
	for (auto k : Range(NE))
	  if (is_valid(e_map[k]))
	    cc2fe.Add(e_map[k], k);
      c2fe = cc2fe.MoveTable();
    }
    Mat<dofpv(D), dofpv(D)> T, TTM;
    T = 0; Iterate<dofpv(D)>([&](auto i) { T(i.value, i.value) = 1.0; });
    Mat<disppv(D), disppv(D), double> W;
    Mat<disppv(D), rotpv(D), double> sktcf0, sktcf1, sktcc;
    Mat<rotpv(D), rotpv(D), double> B, adbm;
    auto calc_trafo = [](auto & T, const auto & tAi, const auto & tBj) {
      Vec<3, double> tang = 0.5 * (tAi + tBj); // t is flipped
      if constexpr(D==3) {
	  T(2,4) = - (T(1,5) = tang(0));
	  T(0,5) = - (T(2,3) = tang(1));
	  T(1,3) = - (T(0,4) = tang(2));
	}
      else {
	T(1,2) =  tang(0);
	T(0,2) = -tang(1);
      }
    };
    auto calc_cemat = [&](const auto & cedge, auto & cemat, auto use_fenr) {
      auto fenrs = c2fe[cedge.id];
      cemat = 0;
      for (auto fenr : fenrs) {
	if (use_fenr(fenr)) {
	  const auto & fedge = edges[fenr];
	  int l = (v_map[fedge.v[0]] == cedge.v[0]) ? 0 : 1;
	  /**
	     I | sk(tcf0+tcf1)
	     0 | I
	  **/
	  Vec<3> tcf0 =  (fvd[fedge.v[0]].pos - cvd[cedge.v[l]].pos);
	  Vec<3> tcf1 =  (fvd[fedge.v[1]].pos - cvd[cedge.v[1-l]].pos);
	  // cout << "t0 " << tcf0 << endl;
	  // cout << "t1 " << tcf1 << endl;
	  calc_trafo(T, tcf0, tcf1);
	  // cout << "fedge " << fedge << endl << " to cedge " << cedge << endl;
	  auto & FM = fed[fedge.id];
	  // cout << "FM: " << endl; print_tm(cout, FM); cout << endl;
	  // cout << "trans: " << endl; print_tm(cout, T); cout << endl;
	  TTM = Trans(T) * FM;
	  cemat += TTM * T;
	  // cout << "cemat now: " << endl; print_tm(cout, cemat); cout << endl;
	}
      }
    };
    typedef Mat<dofpv(D), dofpv(D), double> TM;
    auto calc_cedata = [&](const auto & edge, TM & cfullmat) {
      ced[edge.id] = cfullmat;
    };
    /** ex-edge matrices:  calc & reduce full cmats **/
    if (neqcs>1) {
      Array<int> perow(neqcs);
      for (auto k : Range(neqcs))
	perow[k] = cmesh.template GetENN<NT_EDGE>(k) + cmesh.template GetCNN<NT_EDGE>(k);
      perow[0] = 0;
      Table<TM> tcemats(perow); perow = 0;
      cmesh.template ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  calc_cemat(cedge, tcemats[eqc][perow[eqc]++],
		     [&](auto fenr) { return eqc_h.IsMasterOfEQC(mesh.template GetEqcOfNode<NT_EDGE>(fenr)); } );
	}, false); // def, false!
      Table<TM> cemats = ReduceTable<TM,TM> (tcemats, sp_eqc_h, [&](auto & tab) { return sum_table(tab); });
      perow = 0;
      cmesh.template ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  ced[cedge.id] = cemats[eqc][perow[eqc]++];
	}, false); // def, false!
    }
    /** loc-edge matrices:  all in one **/
    cmesh.template ApplyEQ<NT_EDGE>(Range(min(neqcs, size_t(1))), [&](auto eqc, const auto & cedge){
	calc_cemat(cedge, ced[cedge.id], [&](auto fenr) { return true; });
      }, false); // def, false!
    ceed.SetParallelStatus(CUMULATED);

    // cout << "coarse edges / mats: " << endl;
    // auto cedges = cmesh.template GetNodes<NT_EDGE>();
    // for (auto & edge : cedges) {
    //   cout << "cedge : " << edge << ": " << endl; print_tm(cout, ced[edge.id]); cout << endl;
    // }

    // cout << "wait map edge data " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << "wait map edge data done " << endl;
  } // AttachedEED::map_data


  /** ElasticityAMGFactory **/


  template<int D>
  ElasticityAMGFactory<D> :: ElasticityAMGFactory (shared_ptr<ElasticityAMGFactory<D>::TMESH> mesh, shared_ptr<ElasticityAMGFactory<D>::Options> options,
						   shared_ptr<BaseDOFMapStep> _embed_step)
    : BASE(mesh, options, _embed_step)
  { ; } // ElasticityAMGFactory (..)


  template<int D> void ElasticityAMGFactory<D> :: SetOptionsFromFlags (ElasticityAMGFactory<D>::Options & opts, const Flags & flags, string prefix)
  {
    BASE::SetOptionsFromFlags(opts, flags, prefix);
  } // ElasticityAMGFactory::SetOptionsFromFlags
  

  template<int D> void ElasticityAMGFactory<D> :: SetCoarseningOptions (VWCoarseningData::Options & opts, shared_ptr<ElasticityAMGFactory<D>::TMESH> mesh) const
  {
    static Timer t("SetCoarseningOptions"); RegionTimer rt(t);
    const auto& O = static_cast<const Options&>(*this->options);
    mesh->CumulateData();
    auto NV = mesh->template GetNN<NT_VERTEX>();
    opts.free_verts = this->free_verts;
    opts.min_vcw = O.min_vcw;
    opts.vcw = Array<double>(NV); opts.vcw = 0;
    opts.min_ecw = O.min_ecw;
    if (O.soc_alg == Options::SOC_ALG::SIMPLE)
      { opts.ecw = CalcECWSimple(mesh); }
    else if (O.soc_alg == Options::SOC_ALG::ROBUST)
      { opts.ecw = CalcECWRobust(mesh); }
  } // ElasticityAMGFactory::SetCoarseningOptions


  template<int D> Array<double> ElasticityAMGFactory<D> :: CalcECWSimple (shared_ptr<TMESH> mesh) const
  {
    const TMESH & M(*mesh);
    auto NV = M.template GetNN<NT_VERTEX>();
    auto NE = M.template GetNN<NT_EDGE>();
    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();
    const auto& econ(*M.GetEdgeCM());
    Array<double> ecw(NE);
    Array<double> vcw(NV); vcw = 0;
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);
	// cout << " edge " << edge << ", trace = " << tr << endl; // print_tm(cout, edata[edge.id]); cout << endl;
	vcw[edge.v[0]] += tr;
	vcw[edge.v[1]] += tr;
	// cout << " vcws now " << vcw[edge.v[0]] << " " << vcw[edge.v[1]] << endl;
      }, true);
    // cout << " v-acc wts " << endl; prow2(vcw); cout << endl;
    M.template AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);
	// cout << " edge " << edge << ", trace = " << tr << endl; // print_tm(cout, edata[edge.id]); cout << endl;
	double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
	// cout << " vcws " << vcw[edge.v[0]] << " " << vcw[edge.v[1]] << endl;
	ecw[edge.id] = tr / vw;
	// cout << " ecw " << ecw[edge.id] << endl;
      }, false);
    // cout << " edge-wts for crsening: " << endl; prow2(ecw); cout << endl;
    return move(ecw);
  } // ElasticityAMGFactory::CalcECWSimple


  template<int D> Array<double> ElasticityAMGFactory<D> :: CalcECWRobust (shared_ptr<TMESH> mesh) const
  {
    /**
       This should detect "edge/corner cases", where vertices are mostly strongly connecyed, but
       weakly connected in some way (e.g two stiff blocs, connected along an edge or a corner 
       will be strongly connected in displacements, but weakly in some, or all rotations)
       
       I THINK it works in 2d (but it might not). 3d is, I think, broken.
     **/
    ElasticityMesh<D> & M(*mesh);
    // TODO: only works "sequential" for now ...
    if (mesh->GetEQCHierarchy()->GetCommunicator().Size() > 2)
      { throw Exception("robust ECW only sequential for now"); } // actually, not sure, it might work
    auto NV = M.template GetNN<NT_VERTEX>();
    auto NE = M.template GetNN<NT_EDGE>();
    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();
    const auto& econ(*M.GetEdgeCM());
    Array<double> ecw(NE);
    Matrix<TM> emat(2,2);
    Matrix<double> schur(dofpv(D), dofpv(D)), emoo(dofpv(D), dofpv(D));
    Array<TM> vblocks(NV); vblocks = 0;
    auto edges = M.template GetNodes<NT_EDGE>();
    {
      static Timer t("SetCoarseningOptions - Collect"); RegionTimer rt(t);
      M.template Apply<NT_EDGE> ([&](const auto & edge) LAMBDA_INLINE {
	  CalcRMBlock(M, edge, emat);
	  vblocks[edge.v[0]] += emat(0,0);
	  vblocks[edge.v[1]] += emat(1,1);
	}, true);
    }
    M.template AllreduceNodalData<NT_VERTEX, TM> (vblocks, [](auto & tab) LAMBDA_INLINE { return move(sum_table(tab)); });
    {
      static Timer t("SetCoarseningOptions - Calc"); RegionTimer rt(t);
      M.template Apply<NT_EDGE>([&](const auto & edge) {
	  double cws[2] = {0,0};
	  CalcRMBlock(M, edge, emat);
	  double tr = 0; Iterate<dofpv(D)>([&](auto i) { tr += emat(0,0)(i.value,i.value); } );
	  tr /= dofpv(D);
	  emat /= tr;
	  // cout << "edge mat: " << endl; print_tm_mat(cout, emat); cout << endl;
	  Iterate<2>([&](auto i) {
	      // cout << "edge " << edge << " i " << i.value << endl;
	      constexpr int j = 1-i;
	      emoo = vblocks[edge.v[i.value]];
	      emoo /= tr;
	      // cout << "invert emoo: " << endl << emoo << endl;
	      CalcInverse(emoo);
	      // CalcPseudoInverse<dofpv(D)>(emoo);
	      // cout << "inverted emoo: " << endl << emoo << endl;
	      schur = emat(j, j) - emat(j,i.value) * emoo * emat(i.value, j);
	      emoo = emat(j, j);
	      // cout << "j-block: " << endl << emoo << endl;
	      // cout << "schur: " << endl << schur << endl;
	      cws[i.value] = sqrt(1 - CalcMinGenEV<dofpv(D)>(schur, emoo));
	    });
	  // ecw[edge.id] = sqrt(cws[0]*cws[1]);
	  ecw[edge.id] = cws[0] + cws[1];
	  // cout << "okj, next edge " << endl;
	}, false);
    }
    return move(ecw);
   } // ElasticityAMGFactory::CalcECWRobust


  /** EmbedVAMG<ElasticityAMGFactory> **/

  template<class C> void SetEADO (C& O, int D)
  {
    /** keep vertex positions! **/
    O.keep_vp = true;

    /** Level-control **/
    // O.first_aaf = 1/pow(3, D);
    O.first_aaf = (D == 3) ? 0.025 : 0.05;
    O.aaf = 1/pow(2, D);

    /** Redistribute **/
    O.enable_ctr = true;
    O.ctraf = 0.05;
    O.first_ctraf = O.aaf * O.first_aaf;
    O.ctraf_scale = 1;
    O.ctr_crs_thresh = 0.7;
    O.ctr_min_nv_gl = 5000;
    O.ctr_seq_nv = 5000;
    
    /** Smoothed Prolongation **/
    O.enable_sm = true;
    O.sp_min_frac = (D == 3) ? 0.08 : 0.15;
    O.sp_omega = 1;
    O.sp_max_per_row = 1 + D;

    /** Rebuild Mesh**/
    O.enable_rbm = true;
    O.rbmaf = O.aaf * O.aaf;
    O.first_rbmaf = O.aaf * O.first_aaf;

    /** Embed **/
    // displacement formulations
    O.block_s = { 1 };         // mutlidim, default case
    // O.block_s = { 1, 1, 1 };   // vector-h1 (3 dim)
    // O.block_s = { disppv(D) }; // reordered vector-h1

    // displacement + rot formulations
    // O.block_s = { 1 };                    // multidim
    // O.block_s = { 1, 1, 1, 1, 1, 1 };     // vector-h1 (3 dim)
    // O.block_s = { disppv(D) + rotpv(D) }; // reordered vector-h1
  }

  template<> void EmbedVAMG<ElasticityAMGFactory<2>> :: SetDefaultOptions (EmbedVAMG<ElasticityAMGFactory<2>>::Options& O)
  { SetEADO (O, 2); }
  template<> void EmbedVAMG<ElasticityAMGFactory<3>> :: SetDefaultOptions (EmbedVAMG<ElasticityAMGFactory<3>>::Options& O)
  { SetEADO (O, 3); }


  template<int DIM, class OPT>
  void ModEAO (OPT & O, const Flags & flags, string prefix, shared_ptr<BilinearForm> & bfa)
  {
    auto pfit = [prefix] (string x) { return prefix + x; };

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };

    /** keep vertex positions (should still be set from default-opts, but make sure anyways) **/
    O.keep_vp = true;

    /** block_s **/
    // TODO: this has to fit with buildembedding !
    set_bool(O.with_rots, "rots");

    O.reg_mats = !O.with_rots;
    set_bool(O.reg_mats, "reg_mats");

    O.reg_rmats = !O.with_rots;
    set_bool(O.reg_rmats, "reg_rmats");

    auto & obs = flags.GetNumListFlag(pfit("block_s"));
    if (obs.Size()) { // explicitely given
      O.block_s.SetSize(obs.Size());
      for (auto k : Range(obs))
	{ O.block_s[k] = obs[k]; }
    }
    else { // try to do the best I can
      auto fes_dim = bfa->GetFESpace()->GetDimension();
      if (!O.with_rots) { // displacement formulation
	if (fes_dim == disppv(DIM))
	  { O.block_s = { 1 }; }
	else if (fes_dim == 1) {
	  O.block_s.SetSize(disppv(DIM));
	  O.block_s = 1;
	}
	else {
	  throw Exception(string("Wrong multidim for displacement only formulation, have") + to_string(fes_dim)
			  + string(", expected") + to_string(disppv(DIM)) + string(" (switch to disp+rot with ") + prefix
			  + string("_rots=True) !"));
	}
      }
      else { // displacement + rotation formulation
	if (fes_dim == dofpv(DIM))
	  { O.block_s = { 1 }; }
	else if (fes_dim == 1) {
	  O.block_s.SetSize(dofpv(DIM));
	  O.block_s = 1;
	}
	else {
	  throw Exception(string("Wrong multidim for disp+rot formulation, have") + to_string(fes_dim)
			  + string(", expected") + to_string(dofpv(DIM)) + string(" (switch to disp inly with ") + prefix
			  + string("_rots=False) !"));
	}
      }
    }

    /** rebuild mesh - not implemented (also not sure how I would do this ...) **/

  } // EmbedVAMG::ModifyOptions

  template<> void EmbedVAMG<ElasticityAMGFactory<2>> :: ModifyOptions (EmbedVAMG<ElasticityAMGFactory<2>>::Options & O, const Flags & flags, string prefix)
  { ModEAO<2>(O, flags, prefix, bfa); }
  template<> void EmbedVAMG<ElasticityAMGFactory<3>> :: ModifyOptions (EmbedVAMG<ElasticityAMGFactory<3>>::Options & O, const Flags & flags, string prefix)
  { ModEAO<3>(O, flags, prefix, bfa); }


  template<class C> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh)
  {
    auto a = new AttachedEVD(Array<ElasticityVertexData>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto vdata = a->Data();
    FlatArray<int> vsort = node_sort[NT_VERTEX];
    auto & vp = node_pos[NT_VERTEX];
    for (auto k : Range(vdata)) {
      auto& x = vdata[k];
      x.wt = 0.0;
      x.pos = vp[k];
    }

    auto b = new AttachedEED<C::DIM>(Array<ElasticityEdgeData<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), CUMULATED);
    for (auto & x : b->Data()) { SetIdentity(x); }

    auto mesh = make_shared<typename C::TMESH>(move(*top_mesh), a, b);

    return mesh;
  } // EmbedVAMG::BuildAlgMesh_TRIV


  template<class C> template<class TD2V, class TV2D> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    if (spmat == nullptr)
      { throw Exception("BuildAlgMesh_ALG_scal called with no mat!"); }

    auto a = new AttachedEVD(Array<ElasticityVertexData>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
    const auto & vsort = node_sort[NT_VERTEX];

    // vertex-points
    for (auto k : Range(ma->GetNV())) {
      auto vnum = vsort[k];
      vdata[vnum].wt = 0;
      ma->GetPoint(k, vdata[vnum].pos);
    }

    // edge-mid points, kind of hacky!
    if (top_mesh->GetNN<NT_VERTEX>() > ma->GetNV()) {
      Vec<3> a, b;
      auto ma_nv = ma->GetNV();
      for (auto edge_num : Range(ma->GetNEdges())) {
	auto pnums = ma->GetEdgePNums(edge_num);
	ma->GetPoint(pnums[0], a);
	ma->GetPoint(pnums[1], b);
	auto vnum = vsort[ma_nv + edge_num];
	vdata[vnum].wt = 0;
	vdata[vnum].pos = 0.5 * ( a + b );
      }
    }

    auto b = new AttachedEED<C::DIM>(Array<ElasticityEdgeData<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); // !! has to be distr
    auto edata = b->Data();

    const auto& dof_blocks(options->block_s);
    // auto& fvs = *free_verts;
    const auto& ffds = *finest_freedofs;
    if ( (dof_blocks.Size() == 1) && (dof_blocks[0] == 1) ) { // multidim
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<Mat<disppv(C::DIM),disppv(C::DIM),double>>>(spmat)) { // disp only
	const auto& A(*spm_tm);
	for (auto & e : edges) {
	  auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
	  // cout << "edge " << e << endl << " dofs " << di << " " << dj << endl;
	  // cout << " mat etr " << endl; print_tm(cout, A(di, dj)); cout << endl;
	  // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / disppv(C::DIM) : 1e-4; // after BBDC, diri entries are compressed and mat has no entry 
	  // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
	  double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / sqrt(fabsum(A(di,di)) * fabsum(A(dj,dj))) / disppv(C::DIM) : 1e-4;
	  auto & emat = edata[e.id]; emat = 0;
	  Iterate<disppv(C::DIM)>([&](auto i) LAMBDA_INLINE { emat(i.value, i.value) = fc; });
	  // cout << " emat: " << endl; print_tm(cout, emat); cout << endl;
	}
      }
      else if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<Mat<dofpv(C::DIM),dofpv(C::DIM),double>>>(spmat)) { // disp+rot
	const auto& A(*spm_tm);
	for (auto & e : edges) {
	  auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
	  // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / dofpv(C::DIM) : 1e-4; // after BBDC, diri entries are compressed and mat has no entry 
	  // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
	  double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / sqrt(fabsum(A(di,di)) * fabsum(A(dj,dj))) / dofpv(C::DIM) : 1e-4;
	  auto & emat = edata[e.id]; emat = 0;
	  Iterate<dofpv(C::DIM)>([&](auto i) LAMBDA_INLINE { emat(i.value, i.value) = fc; });
	}
      }
      else
	{ throw Exception(string("not sure how to compute edge weights from mat of type ") + typeid(*spmat).name() + string("!")); }
    }
    else
      { throw Exception("block_s for compound, but called algmesh_alg_scal!"); }


    auto mesh = make_shared<typename C::TMESH>(move(*top_mesh), a, b);

    return mesh;
  } // EmbedVAMG::BuildAlgMesh_ALG_scal


  template<>
  shared_ptr<BaseSparseMatrix> EmbedVAMG<ElasticityAMGFactory<2>> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat,
										       shared_ptr<ParallelDofs> & pardofs) const
  {
    auto& A = static_cast<ElasticityAMGFactory<2>::TSPM_TM&>(*mat);
    if ( (pardofs != nullptr) && (pardofs->GetDistantProcs().Size() != 0) ) {
      Array<int> is_zero(A.Height());
      for(auto k : Range(A.Height()))
	{ is_zero[k] = (A(k,k)(2,2) == 0) ?  1 : 0; }
      AllReduceDofData(is_zero, MPI_SUM, pardofs);
      for(auto k : Range(A.Height()))
	if ( (pardofs->IsMasterDof(k)) && (is_zero[k] != 0) )
	  { A(k,k)(2,2) = 1; }
    }
    else {
      for(auto k : Range(A.Height())) {
	auto & diag_etr = A(k,k);
	if (diag_etr(2,2) == 0)
	  { diag_etr(2,2) = 1; }
      }
    }
    return mat;
  } // EmbedVAMG<ElasticityAMGFactory<2>>::RegularizeMatrix


  template<> shared_ptr<BaseSparseMatrix>
  EmbedVAMG<ElasticityAMGFactory<3>> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat,
							  shared_ptr<ParallelDofs> & pardofs) const
  {
    auto& A = static_cast<ElasticityAMGFactory<3>::TSPM_TM&>(*mat);
    typedef ElasticityAMGFactory<3>::TM TM;

    if ( (pardofs != nullptr) && (pardofs->GetDistantProcs().Size() != 0) ) {
      Array<TM> diags(A.Height());
      for(auto k : Range(A.Height()))
	{ diags[k] = A(k,k); }
      MyAllReduceDofData(*pardofs, diags, [](auto & a, const auto & b) LAMBDA_INLINE { a+= b; });
      for(auto k : Range(A.Height())) {
	if (pardofs->IsMasterDof(k)) {
	  auto & dg_etr = A(k,k);
	  dg_etr = diags[k];
	  RegTM<3,3,6>(dg_etr);
	}
	else
	  { A(k,k) = 0; }
      }
    }
    else {
      for(auto k : Range(A.Height()))
	{ RegTM<3,3,6>(A(k,k)); }
    }
	
    return mat;
  } // EmbedVAMG<ElasticityAMGFactory<3>>::RegularizeMatrix


  template<> template<>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<ElasticityAMGFactory<2>> :: BuildEmbedding_impl<6> (shared_ptr<ElasticityMesh<2>> mesh)
  { return nullptr; }


  template<> template<>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<ElasticityAMGFactory<3>> :: BuildEmbedding_impl<2> (shared_ptr<ElasticityMesh<3>> mesh)
  { return nullptr; }


  template<class C> template<int N>
  shared_ptr<stripped_spm_tm<typename strip_mat<Mat<N, mat_traits<typename C::TM>::HEIGHT, double>>::type>> EmbedVAMG<C> :: BuildED (size_t subset_count, shared_ptr<typename C::TMESH> mesh)
  {
    // static_assert( (N == 1) || (N == disppv(C::DIM)) || (N == dofpv(C::DIM)), "BuildED with nonsensical N.");
    assert( (N == 1) || (N == disppv(C::DIM)) || (N == dofpv(C::DIM)) ); // "BuildED with nonsensical N."

    static_assert( mat_traits<typename C::TM>::HEIGHT == dofpv(C::DIM), "um what??");

    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*this->options);

    typedef stripped_spm_tm<typename strip_mat<Mat<N, dofpv(C::DIM), double>>::type> TED;

    shared_ptr<TED> E_D = nullptr;
    const auto& M(*mesh);

    if constexpr ( N == dofpv(C::DIM) ) {
      // nothing to do right now. usually, woudl flip/permute rots
      assert((O.with_rots) && (O.block_s.Size() == 1) && (O.block_s[0] == 1)); // "elasticity BuildED: disp/disp+rot, block_s mismatch");
      assert(subset_count == M.template GetNN<NT_VERTEX>()); // "elasticity BuildED: subset_count and NV mismatch, what?");
      E_D = nullptr;
    }

    if constexpr ( N == disppv(C::DIM)) { // disp -> disp,rot embedding
      assert((!O.with_rots) && (O.block_s.Size() == 1) && (O.block_s[0] == 1)); // "elasticity BuildED: disp/disp+rot, block_s mismatch");
      assert(subset_count == M.template GetNN<NT_VERTEX>()); // "elasticity BuildED: subset_count and NV mismatch, what?");
      Array<int> perow(M.template GetNN<NT_VERTEX>()); perow = 1;
      E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());
      for (auto k : Range(perow)) {
	E_D->GetRowIndices(k)[0] = k;
	auto & v = E_D->GetRowValues(k)[0];
	v = 0; Iterate<disppv(C::DIM)>([&](auto i) { v(i.value, i.value) = 1; });
      }
    }

    if constexpr ( N == 1 ) {
      /** 2 valid suppoerted block_s: 
	   - disppv x 1
	   - (disppv+rotpv) x 1
	  not supported, but possibly useful:
	   - { dispv } (reordered disp-only)
	   - {disppv, rotpv} (strangely reordered disp+rot)
	   - { disppv x 1, rotpv } (only reordered rot, for some reason)
      **/

      assert( ( O.with_rots && (O.block_s.Size() == dofpv(C::DIM)) ) ||
	      ( (!O.with_rots) && (O.block_s.Size() == disppv(C::DIM)) ) ); // "elasticity BuildED: disp/disp+rot mismatch");
      assert(subset_count == O.block_s.Size() * M.template GetNN<NT_VERTEX>()); // "elasticity BuildED: subset_cnt and block_s mismatch");

      for (auto v : O.block_s)
	{ if (v != 1) { throw Exception("this block_s is not supported"); } }
      Array<int> perow(subset_count); perow = 1;
      E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());
      const auto bss = O.block_s.Size();
      for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
	for (auto j : Range(bss)) {
	  auto row = j * M.template GetNN<NT_VERTEX>() + k;
	  E_D->GetRowIndices(row)[0] = k;
	  E_D->GetRowValues(row)[0](j) = 1;
	}
      }
    }

    // cout << "E_D: " << endl;
    // if (E_D)
    //   print_tm_mat(cout, *E_D);
    // else
    //   cout << " NO E_D!!" << endl;

    return E_D;
  }


  template<class C> shared_ptr<BaseSmoother>
  EmbedVAMG<C> :: BuildSmoother (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> pds,
				 shared_ptr<BitArray> freedofs) const
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*this->options);
    shared_ptr<BaseSmoother> sm = nullptr;
    if (O.sm_ver == BAO::SM_VER::VER1) {
      if (auto spmat = dynamic_pointer_cast<SparseMatrix<typename C::TM>>(mat)) {
	if (O.reg_mats)
	  { sm = make_shared<StabHGSS<dofpv(C::DIM), disppv(C::DIM), dofpv(C::DIM)>> (spmat, pds, freedofs); }
	else
	  { sm = make_shared<HybridGSS<dofpv(C::DIM)>> (spmat, pds, freedofs); }
      }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<Mat<disppv(C::DIM),disppv(C::DIM), double>>>(mat))
	{ sm = make_shared<HybridGSS<disppv(C::DIM)>> (spmat, pds, freedofs); }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<double>>(mat))
	{ sm = make_shared<HybridGSS<1>> (spmat, pds, freedofs); }
    }
    else if (O.sm_ver == BAO::SM_VER::VER2) {
      auto parmat = make_shared<ParallelMatrix>(mat, pds, pds, C2D);
      if (auto spmat = dynamic_pointer_cast<SparseMatrix<typename C::TM>>(mat)) {
	if (O.reg_mats)
	  { throw Exception("V2 regularized diag not implemented"); }
	else {
	  auto v2sm = make_shared<HybridGSS2<Mat<dofpv(C::DIM), dofpv(C::DIM), double>>>(parmat, freedofs);
	  v2sm->SetSymmetric(O.smooth_symmetric);
	  sm = v2sm;
	}
      }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<Mat<disppv(C::DIM),disppv(C::DIM), double>>>(mat)) {
	auto v2sm = make_shared<HybridGSS2<Mat<disppv(C::DIM), disppv(C::DIM), double>>>(parmat, freedofs);
	v2sm->SetSymmetric(O.smooth_symmetric);
	sm = v2sm;
      }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<double>>(mat)) {
	auto v2sm = make_shared<HybridGSS2<double>>(parmat, freedofs);
	v2sm->SetSymmetric(O.smooth_symmetric);
	sm = v2sm;
      }
    }
    else if (O.sm_ver == BAO::SM_VER::VER3) {
      auto parmat = make_shared<ParallelMatrix>(mat, pds, pds, C2D);
      auto eqc_h = make_shared<EQCHierarchy>(pds, false); // TODO: get rid of these!
      if (auto spmat = dynamic_pointer_cast<SparseMatrix<typename C::TM>>(mat)) {
      	if (O.reg_mats) {
	  auto v3sm = make_shared<RegHybridGSS3<typename C::TM, disppv(C::DIM), dofpv(C::DIM)>> (parmat, eqc_h, freedofs, O.mpi_overlap, O.mpi_thread);
	  v3sm->SetSymmetric(options->smooth_symmetric);
	  // v3sm->Finalize();
	  sm = v3sm;
	}
	else {
	  auto v3sm = make_shared<HybridGSS3<Mat<dofpv(C::DIM), dofpv(C::DIM), double>>>(parmat, eqc_h, freedofs, O.mpi_overlap, O.mpi_thread);
	  v3sm->SetSymmetric(options->smooth_symmetric);
	  // v3sm->Finalize();
	  sm = v3sm;
	}
      }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<Mat<disppv(C::DIM),disppv(C::DIM), double>>>(mat)) {
	auto v3sm = make_shared<HybridGSS3<Mat<disppv(C::DIM), disppv(C::DIM), double>>>(parmat, eqc_h, freedofs, O.mpi_overlap, O.mpi_thread);
	v3sm->SetSymmetric(options->smooth_symmetric);
	// v3sm->Finalize();
	sm = v3sm;
      }
      else if (auto spmat = dynamic_pointer_cast<SparseMatrix<double>>(mat)) {
	auto v3sm = make_shared<HybridGSS3<double>>(parmat, eqc_h, freedofs, O.mpi_overlap, O.mpi_thread);
	v3sm->SetSymmetric(options->smooth_symmetric);
	// v3sm->Finalize();
	sm = v3sm;
      }
    }
    return sm;
  }


  /** EmbedWithElmats **/


  template<class C, class D, class E> shared_ptr<typename EmbedWithElmats<C,D,E>::TMESH>
  EmbedWithElmats<C,D,E> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
  {
    return nullptr;
  } // EmbedWithElmats::BuildAlgMesh_ELMAT


  template<class C, class D, class E> void EmbedWithElmats<C,D,E> ::
  AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    ;
  } // EmbedWithElmats::AddElementMatrix

  // i had an undefinde reference to map_data without this
  template class AttachedEED<2>;
  template class AttachedEED<3>;

  RegisterPreconditioner<EmbedWithElmats<ElasticityAMGFactory<2>, double, double>> register_el2d("ngs_amg.elast2d");

  RegisterPreconditioner<EmbedWithElmats<ElasticityAMGFactory<3>, double, double>> register_el3d("ngs_amg.elast3d");

} // namespace amg

#endif
