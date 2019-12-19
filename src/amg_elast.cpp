#ifdef ELASTICITY
#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_elast_impl.hpp"
#include "amg_factory_impl.hpp"
#include "amg_pc_impl.hpp"
#include "amg_bla.hpp"

namespace amg
{

  /** AttachedEED **/


  template<int D> void AttachedEED<D> :: map_data (const BaseCoarseMap & cmap, AttachedEED<D> & ceed) const
  {
    static Timer t(string("ElEData<")+to_string(D)+string(">::map_data")); RegionTimer rt(t);
    auto & M = static_cast<ElasticityMesh<D>&>(*this->mesh); M.CumulateData();
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto& fecon = *M.GetEdgeCM();
    ElasticityMesh<D> & CM = static_cast<ElasticityMesh<D>&>(*ceed.mesh);
    auto cedges = CM.template GetNodes<NT_EDGE>();
    const size_t NV = M.template GetNN<NT_VERTEX>();
    const size_t NE = M.template GetNN<NT_EDGE>();
    const size_t CNV = cmap.template GetMappedNN<NT_VERTEX>();
    const size_t CNE = cmap.template GetMappedNN<NT_EDGE>();
    auto e_map = cmap.template GetMap<NT_EDGE>();
    auto v_map = cmap.template GetMap<NT_VERTEX>();
    get<0>(M.Data())->Cumulate(); // should be cumulated anyways
    auto fvd = get<0>(M.Data())->Data();
    get<0>(CM.Data())->Cumulate(); // we need full vertex positions !
    auto cvd = get<0>(CM.Data())->Data();
    auto fed = this->Data();
    auto ced = ceed.Data();
    typedef Mat<dofpv(D), dofpv(D), double> TM;
    // TODO: we are modifying coarse v-wts here. HACKY!!
    Array<double> add_cvw (CNV); add_cvw = 0;
    Vec<3> posH, posh, tHh;
    TM TEST = 0;
    TM QHh(0), FMQ(0);
    ceed.data.SetSize(CNE); // ced. is a flat-array, so directly access ceed.data
    ced = 0.0;
    M.template ApplyEQ<NT_EDGE>([&] (auto eqc, const auto & fedge) LAMBDA_INLINE {
  	auto cenr = e_map[fedge.id];
	if (cenr != -1) {
  	  const auto & cedge = cedges[cenr];
  	  auto& cemat = ced[cenr];
  	  posH = 0.5 * (cvd[cedge.v[0]].pos + cvd[cedge.v[1]].pos);
  	  posh = 0.5 * (fvd[fedge.v[0]].pos + fvd[fedge.v[1]].pos);
  	  tHh = posh - posH;
  	  ElasticityAMGFactory<D>::CalcQ(tHh, QHh);
  	  FMQ = fed[fedge.id] * QHh;
  	  cemat += Trans(QHh) * FMQ;
  	}
  	else { // add largest eval of connection to ground to the coarse vertex-weight
	  auto tr = calc_trace(fed[fedge.id]) / mat_traits<TM>::HEIGHT; // a guess
  	  Iterate<2>([&](auto l) LAMBDA_INLINE {
  	      if (v_map[fedge.v[1-l.value]] == -1) {
  		auto cvnr = v_map[fedge.v[l.value]];
  		// cout << l.value << ", v " << fedge.v[l.value] << " -> " << cvnr << endl;
  		if (cvnr != -1)
  		  { add_cvw[cvnr] += tr; }
  	      }
  	    });
  	}
      }, true); // master only
    CM.template AllreduceNodalData<NT_VERTEX>(add_cvw, [](auto & in) { return sum_table(in); }, false);
    for (auto k : Range(CNV)) // cvd and add_cvw are both "CUMULATED"
      { cvd[k].wt += add_cvw[k]; }
    ceed.SetParallelStatus(DISTRIBUTED);
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

    if (flags.GetDefineFlagX(prefix+string("stable_soc")).IsTrue())
      { opts.soc_alg = Options::SOC_ALG::ROBUST; }

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

    // cout << " CRSEN, " << NV << " " << NE << endl;
    // cout << "        " << vdata.Size() << " " << edata.Size() << endl;

    // cout << " vwts: " << endl; prow2(vdata); cout << endl << endl;
    // cout << " ewts: " << endl; prow2(edata); cout << endl << endl;

    Array<double> ecw(NE);
    Array<double> vcw(NV); vcw = 0;
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);
	// cout << " edge " << edge << ", trace = " << tr << endl; // print_tm(cout, edata[edge.id]); cout << endl;
	vcw[edge.v[0]] += tr;
	vcw[edge.v[1]] += tr;
	// cout << " vcws now " << vcw[edge.v[0]] << " " << vcw[edge.v[1]] << endl;
      }, true);
    // cout << " v-acc wts " << endl; prow2(vcw); cout << endl << endl;
    M.template AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);

	// double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
	// ecw[edge.id] = tr / vw;

	ecw[edge.id] = tr / sqrt(vcw[edge.v[0]] * vcw[edge.v[1]]);
	// cout << " ecw " << ecw[edge.id] << endl;
      }, false);
    // cout << " edge-wts for crsening: " << endl; prow2(ecw); cout << endl << endl;
    return move(ecw);
  } // ElasticityAMGFactory::CalcECWSimple


  template<int D> Array<double> ElasticityAMGFactory<D> :: CalcECWRobust (shared_ptr<TMESH> mesh) const
  {
    const auto & M(*mesh);

    auto old_rrm = static_cast<Options&>(*this->options).reg_rmats;
    static_cast<Options&>(*this->options).reg_rmats = false;

    M.CumulateData();

    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();

    const auto NV = M.template GetNN<NT_VERTEX>();
    const auto NE = M.template GetNN<NT_EDGE>();

    constexpr int N = dofpv(D);

    Array<double> ecw(NE); ecw = 0;
    
    Array<TM> repl_diag(NV); repl_diag = 0;

    Matrix<double> long_evecs(2*dofpv(D), 2*dofpv(D)), semat(2*dofpv(D), 2*dofpv(D));
    Vector<double> long_evals(2*dofpv(D));

    Matrix<double> evecs(dofpv(D), dofpv(D)), evm(dofpv(D), dofpv(D));
    Vector<double> evals(dofpv(D));

    const auto & fecon = *M.GetEdgeCM();
    cout << "FECON: " << endl << *M.GetEdgeCM() << endl;
    
    // Get diagonal blocks of replacement matrix
    Matrix<TM> edge_mat(2,2);
    // TM Qij, Qji;
    M.template Apply<NT_EDGE> ([&](const auto & edge) LAMBDA_INLINE {
	// CalcQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
	CalcRMBlock(M, edge, edge_mat);
	repl_diag[edge.v[0]] += edge_mat(0,0);
	repl_diag[edge.v[1]] += edge_mat(1,1);
      }, true); // only master, we cumulate this afterwards
    // M.template Apply<NT_VERTEX> ([&](auto v) LAMBDA_INLINE {
    // 	auto& d = repl_diag[v];
    // 	const double val = vdata[v].wt;
    // 	cout << " add. diag for " << v << " " << val << endl;
    // 	Iterate<mat_traits<TM>::HEIGHT>([&](auto i) {
    // 	    d(i.value, i.value) += 0 * val;
    // 	  });
    //   }, true);
    M.template AllreduceNodalData<NT_VERTEX>(repl_diag, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    Matrix<double> prm(N,N), preve(N,N);
    Vector<double> preva(N);
    auto prt_evv = [&](auto & M, string name) {
      prm = M;
      LapackEigenValuesSymmetric(prm, preva, preve);
      cout << " evals " << name << ": "; prow(preva); cout << endl;
      cout << " evecs " << name << ": " << endl << preve << endl;
    };
    
    Array<int> neibs(20);
    TM QNh, QNi, QNj, ENi, ENj, Esum;

    TM QHi, AiQHi, Ai, QHj, AjQHj, Aj, Asum, L, R;
    M.template Apply<NT_EDGE> ([&](const auto & edge) LAMBDA_INLINE {
	// cout << "------" << endl;
	// cout << " edge " << edge << endl;

	// CalcQHls (vdata[edge.v[0]], vdata[edge.v[1]], QHi, QHj);
	CalcQs (vdata[edge.v[0]], vdata[edge.v[1]], QHj, QHi);
	AiQHi = repl_diag[edge.v[0]] * QHi;
	Ai = Trans(QHi) * AiQHi;
	AjQHj = repl_diag[edge.v[1]] * QHj;
	Aj = Trans(QHj) * AjQHj;

	R = edata[edge.id];
	if (true) {
	  auto i_neibs = fecon.GetRowIndices(edge.v[0]);
	  auto j_neibs = fecon.GetRowIndices(edge.v[1]);
	  intersect_sorted_arrays(i_neibs, j_neibs, neibs);
	  Vec<3> posH = 0.5 * (vdata[edge.v[0]].pos + vdata[edge.v[1]].pos);
	  for (auto n : neibs) {
	    Vec<3> tNH = vdata[n].pos - posH; // actually, QHN
	    CalcQ(tNH, QNh);

	    int enri = int(fecon(edge.v[0], n));
	    Vec<3> tNi = 0.5 * (vdata[edge.v[0]].pos - vdata[n].pos);
	    CalcQ(tNi, QNi);
	    ENi = Trans(QNi) * edata[enri] * QNi;

	    int enrj = int(fecon(edge.v[1], n));
	    Vec<3> tNj = 0.5 * (vdata[edge.v[1]].pos - vdata[n].pos);
	    CalcQ(tNj, QNj);
	    ENj = Trans(QNj) * edata[enrj] * QNj;

	    Esum = ENi + ENj;
	    CalcPseudoInverse<N>(Esum);

	    TM t = ENi * Esum * ENj;
	    TM addR = Trans(QNh) * t * QNh;

	    R += addR;
	  }
	}


	double vw0 = vdata[edge.v[0]].wt;
	double vw1 = vdata[edge.v[1]].wt;
	double maxw = max(vw0, vw1);
	double minw = min(vw0, vw1);
	double fac = (maxw == 0) ? 1.0 : minw/maxw;
	auto mmev = MIN_EV_HARM (Ai, Aj, R);

	// cout << " vwts " << vw0 << " " << vw1 << " " << fac << endl;
	// cout << " mmev " << mmev << endl;

	ecw[edge.id] = fac * mmev;
      }, false); // all exec this

    // }, true); // master computes, then reduce
    // M.template AllreduceNodalData<NT_EDGE>(ecw, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    static_cast<Options&>(*this->options).reg_rmats = old_rrm;

    return move(ecw);
  }


    /** EmbedVAMG<ElasticityAMGFactory> **/

  template<class C> void SetEADO (C& O, shared_ptr<BilinearForm> bfa, int D)
  {
    /** keep vertex positions! **/
    O.store_v_nodes = true;

    /** Coarsening Algorithm **/
    O.crs_alg = C::CRS_ALG::AGG;
    O.agg_wt_geom = false;
    O.n_levels_d2_agg = 1;
    O.disc_max_bs = 1;

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
    O.enable_rbm = false; // actually broken ...
    O.rbmaf = O.aaf * O.aaf;
    O.first_rbmaf = O.aaf * O.first_aaf;

    /** Discard **/
    O.enable_disc = false;

    /** Embed **/
    // displacement formulations
    O.block_s = { 1 };         // mutlidim, default case
    // O.block_s = { 1, 1, 1 };   // vector-h1 (3 dim)
    // O.block_s = { disppv(D) }; // reordered vector-h1

    // displacement + rot formulations
    // O.block_s = { 1 };                    // multidim
    // O.block_s = { 1, 1, 1, 1, 1, 1 };     // vector-h1 (3 dim)
    // O.block_s = { disppv(D) + rotpv(D) }; // reordered vector-h1


    /** make a guess wether we have rotations or not, and how the DOFs are ordered **/
    std::function<Array<size_t>(shared_ptr<FESpace>)> check_space = [&](auto fes) -> Array<size_t> {
      auto fes_dim = fes->GetDimension();
      // cout << " fes " << typeid(*fes).name() << " dim " << fes_dim << endl;
      if (auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes)) {
	auto n_spaces = comp_fes->GetNSpaces();
	Array<size_t> comp_bs;
	for (auto k : Range(n_spaces))
	  { comp_bs.Append(check_space((*comp_fes)[k])) ; }
	size_t sum_bs = std::accumulate(comp_bs.begin(), comp_bs.end(), 0);
	if (sum_bs == dofpv(D))
	  { O.with_rots = true; }
	else if (sum_bs == disppv(D))
	  { O.with_rots = false; }
	return comp_bs;
      }
      else if (auto reo_fes = dynamic_pointer_cast<ReorderedFESpace>(fes)) {
	// reordering changes [1,1,1] -> [3]
	auto base_space = reo_fes->GetBaseSpace();
	auto unreo_bs = check_space(base_space);
	size_t sum_bs = std::accumulate(unreo_bs.begin(), unreo_bs.end(), 0);
	if (sum_bs == dofpv(D))
	  { O.with_rots = true; }
	else if (sum_bs == disppv(D))
	  { O.with_rots = false; }
	return Array<size_t>({sum_bs});
      }
      else if (fes_dim == dofpv(D)) { // only works because mdim+compound does not work, so we never land here in the compound case
	O.with_rots = true;
	return Array<size_t>({1});
      }
      else if (fes_dim == disppv(D)) {
	O.with_rots = false;
	return Array<size_t>({1});
      }
      else
	{ return Array<size_t>({1}); }
    };
    O.block_s = check_space(bfa->GetFESpace());

    // cout << " i guiessed " << O.with_rots << " rots, order: " << endl;
    // cout << O.block_s << endl;
    
  } // SetEADO

  template<> void EmbedVAMG<ElasticityAMGFactory<2>> :: SetDefaultOptions (EmbedVAMG<ElasticityAMGFactory<2>>::Options& O)
  { SetEADO (O, bfa, 2); }
  template<> void EmbedVAMG<ElasticityAMGFactory<3>> :: SetDefaultOptions (EmbedVAMG<ElasticityAMGFactory<3>>::Options& O)
  { SetEADO (O, bfa, 3); }


  template<int DIM, class OPT>
  void ModEAO (OPT & O, const Flags & flags, string prefix, shared_ptr<BilinearForm> & bfa)
  {
    auto pfit = [prefix] (string x) { return prefix + x; };

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };

    /** keep vertex positions (should still be set from default-opts, but make sure anyways) **/
    O.store_v_nodes = true;

    /** block_s **/
    // TODO: this has to fit with buildembedding !
    set_bool(O.with_rots, "rots");

    auto sum_bs = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
    if (sum_bs == 1) {
      if (auto reo_fes = dynamic_pointer_cast<ReorderedFESpace>(bfa->GetFESpace()))
	{ sum_bs = reo_fes->GetBaseSpace()->GetDimension(); }
      else
	{ sum_bs = bfa->GetFESpace()->GetDimension(); }
    }
    if (O.with_rots && (sum_bs != dofpv(DIM)))
      { throw Exception( string("wrong number of variables ") + to_string(sum_bs) + string(" for disp+rot formulation!")); }
    else if (!O.with_rots && (sum_bs != disppv(DIM)))
      { throw Exception( string("wrong number of variables ") + to_string(sum_bs) + string(" for disp formulation!")); }

    O.reg_mats = !O.with_rots;
    set_bool(O.reg_mats, "reg_mats");

    O.reg_rmats = !O.with_rots;
    set_bool(O.reg_rmats, "reg_rmats");

    /** rebuild mesh - not implemented (also not sure how I would do this ...) **/
    O.rebuild_mesh = [&](shared_ptr<ElasticityMesh<DIM>> mesh, shared_ptr<BaseSparseMatrix> amat, shared_ptr<ParallelDofs> pardofs) {

      cout << IM(4) << "REBUILD MESH, NV = " << mesh->template GetNNGlobal<NT_VERTEX>() << ", NE = " << mesh->template GetNNGlobal<NT_EDGE>() << endl;

      using TSPM_TM = typename ElasticityAMGFactory<DIM>::TSPM_TM;
      using TM = typename ElasticityAMGFactory<DIM>::TM;
      constexpr int N = mat_traits<TM>::HEIGHT;
      const auto & mat = *static_pointer_cast<TSPM_TM>(amat);
      
      auto n_verts = mesh->template GetNN<NT_VERTEX>();
      auto traverse_graph = [&](const auto& g, auto fun) LAMBDA_INLINE { // vertex->dof,  // dof-> vertex
	for (auto row : Range(n_verts)) {
	  auto ri = g.GetRowIndices(row);
	  auto pos = find_in_sorted_array(int(row), ri); // no duplicates
	  if (pos+1 < ri.Size())
	    for (auto j : ri.Part(pos+1))
	      { fun(row,j); }
	}
      }; // traverse_graph
      size_t n_edges = 0;
      traverse_graph(mat, [&](auto vk, auto vj) LAMBDA_INLINE { n_edges++; });
      Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);
      n_edges = 0;
      traverse_graph(mat, [&](auto vk, auto vj) LAMBDA_INLINE {
	  if (vk < vj) { epairs[n_edges++] = {int(vk), int(vj)}; }
	  else { epairs[n_edges++] = {int(vj), int(vk)}; }
	});
      mesh->template SetNodes<NT_EDGE> (n_edges, [&](auto num) LAMBDA_INLINE { return epairs[num]; }, // (already v-sorted)
			       [](auto node_num, auto id) { /* dont care about edge-sort! */ });
      mesh->ResetEdgeCM();

      n_edges = mesh->template GetNN<NT_EDGE>();
      
      cout << IM(4) << "REBUILT MESH, NV = " << mesh->template GetNNGlobal<NT_VERTEX>() << ", NE = " << mesh->template GetNNGlobal<NT_EDGE>() << endl;


      /** new edge mats **/
      auto avd = get<0>(mesh->Data()); avd->Cumulate();
      auto vdata = avd->Data();
      auto aed = get<1>(mesh->Data()); aed->SetParallelStatus(DISTRIBUTED);
      auto & edata = aed->GetModData(); edata.SetSize(mesh->template GetNN<NT_EDGE>());
      // off-diag entry -> is edge weight
      auto edges = mesh->template GetNodes<NT_EDGE>();
      Matrix<double> evecs(N,N);
      Vector<double> evals(N);
      TM Qij, Qji, EM;
      for (auto & edge : edges) {
	// cout << " re-calc emat for " << edge << endl;
	// prt_evv<N>(edata[edge.id], "OLD EMAT");
	ElasticityAMGFactory<DIM>::CalcQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
	auto metr = mat(edge.v[0], edge.v[1]);
	// cout << " MAT ETR IS : " << endl; print_tm(cout, metr); cout << endl;
	EM = -1 * Trans(Qji) * mat(edge.v[0], edge.v[1]) * Qij;
	// cout << " EM IS : " << endl; print_tm(cout, EM); cout << endl;
	Iterate<N>([&](auto i) LAMBDA_INLINE {
	    Iterate<i.value>([&](auto j) LAMBDA_INLINE {
		EM(i.value, j.value) = (EM(j.value, i.value) = 0.5 * (EM(i.value, j.value) + EM(j.value, i.value)));
	      });
	  });
	// prt_evv<N>(EM, "EM");
	LapackEigenValuesSymmetric(EM, evals, evecs);
	// if (evals(0) < 0) {
	for (auto k : Range(N)) {
	  auto v = evals(k) > 0 ? sqrt(evals(k)) : 0;
	  for (auto j : Range(N))
	    { evecs(k,j) *= v; }
	}
	edata[edge.id] = Trans(evecs) * evecs;
	// }
	// else
	  // { edata[edge.id] = EM; }
	// prt_evv<N>(edata[edge.id], "NEW EMAT");
      }
      aed->Cumulate();

      return mesh;
    };

    
  } // EmbedVAMG::ModifyOptions

  template<> void EmbedVAMG<ElasticityAMGFactory<2>> :: ModifyOptions (EmbedVAMG<ElasticityAMGFactory<2>>::Options & O, const Flags & flags, string prefix)
  { ModEAO<2>(O, flags, prefix, bfa); }
  template<> void EmbedVAMG<ElasticityAMGFactory<3>> :: ModifyOptions (EmbedVAMG<ElasticityAMGFactory<3>>::Options & O, const Flags & flags, string prefix)
  { ModEAO<3>(O, flags, prefix, bfa); }




  template<>
  shared_ptr<BaseSparseMatrix> EmbedVAMG<ElasticityAMGFactory<2>> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat,
										       shared_ptr<ParallelDofs> & pardofs) const
  {
    auto& A = static_cast<ElasticityAMGFactory<2>::TSPM_TM&>(*mat);
    if ( (pardofs != nullptr) && (pardofs->GetDistantProcs().Size() != 0) ) {
      Array<int> is_zero(A.Height());
      for(auto k : Range(A.Height()))
	{ is_zero[k] = (fabs(A(k,k)(2,2)) < 1e-10 ) ?  1 : 0; }
      AllReduceDofData(is_zero, MPI_SUM, pardofs);
      for(auto k : Range(A.Height()))
	if ( (pardofs->IsMasterDof(k)) && (is_zero[k] != 0) )
	  { A(k,k)(2,2) = 1; }
    }
    else {
      for(auto k : Range(A.Height())) {
	auto & diag_etr = A(k,k);
	if (fabs(diag_etr(2,2)) < 1e-8)
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
	  // cout << " REG DIAG " << k << endl;
	  // RegTM<3,3,6>(dg_etr); // might be buggy !?
	  RegTM<0,6,6>(dg_etr);
	  // cout << endl;
	}
	else
	  { A(k,k) = 0; }
      }
    }
    else {
      for(auto k : Range(A.Height())) {
	// cout << " REG DIAG " << k << endl;
	// RegTM<3,3,6>(A(k,k));
	RegTM<0,6,6>(A(k,k));
      }
    }
	
    return mat;
  } // EmbedVAMG<ElasticityAMGFactory<3>>::RegularizeMatrix


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

	const size_t bssum = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	assert( ( O.with_rots && (bssum == dofpv(C::DIM)) ) ||
		( (!O.with_rots) && (bssum == disppv(C::DIM)) ) ); // "elasticity BuildED: disp/disp+rot mismatch");
	assert(subset_count == bssum * M.template GetNN<NT_VERTEX>()); // "elasticity BuildED: subset_cnt and block_s mismatch");

      // for (auto v : O.block_s)
      // 	{ if (v != 1) { throw Exception("this block_s is not supported"); } }
      // Array<int> perow(subset_count); perow = 1;
      // E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());
      // const auto bss = O.block_s.Size();
      // for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
      // 	for (auto j : Range(bss)) {
      // 	  auto row = j * M.template GetNN<NT_VERTEX>() + k;
      // 	  E_D->GetRowIndices(row)[0] = k;
      // 	  E_D->GetRowValues(row)[0] = 0;
      // 	  E_D->GetRowValues(row)[0](j) = 1;
      // 	}
      // }

      Array<int> perow(subset_count); perow = 1;
      E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());
      if (O.dof_ordering == BAO::REGULAR_ORDERING) {
	size_t row = 0, os_ri = 0;
	for (auto bs : O.block_s) {
	  // cout << "bs : " << bs << endl;
	  for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
	    for (auto j : Range(bs)) {
	      E_D->GetRowIndices(row)[0] = k;
	      E_D->GetRowValues(row)[0] = 0;
	      E_D->GetRowValues(row)[0](os_ri + j) = 1;
	      row++;
	    }
	  }
	  // cout << *E_D << endl;
	  os_ri += bs;
	}
      }
      else
	{ throw Exception("var ordering E_D not implemented"); }
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


  // i had an undefinde reference to map_data without this
  template class AttachedEED<2>;
  template class AttachedEED<3>;

  // template class EmbedVAMG<ElasticityAMGFactory<2>>;
  // template class EmbedVAMG<ElasticityAMGFactory<3>>;

  // template class EmbedWithElmats<ElasticityAMGFactory<2>, double, double>;
  // template class EmbedWithElmats<ElasticityAMGFactory<3>, double, double>;

  RegisterPreconditioner<EmbedWithElmats<ElasticityAMGFactory<2>, double, double>> register_el2d("ngs_amg.elast2d");
  RegisterPreconditioner<EmbedWithElmats<ElasticityAMGFactory<3>, double, double>> register_el3d("ngs_amg.elast3d");

} // namespace amg

#endif

#include "amg_tcs.hpp"
