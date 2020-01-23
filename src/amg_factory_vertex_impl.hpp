#ifndef FILE_AMG_FACTORY_VERTEX_IMPL_HPP
#define FILE_AMG_FACTORY_VERTEX_IMPL_HPP

#include "amg_agg.hpp"

namespace amg
{
  /** State **/

  template<class ENERGY, class TMESH, int BS>
  class VertexAMGFactory<ENERGY, TMESH, BS> :: State : public NodalAMGFactory<NT_VERTEX, TMESH, BS>::State
  {
  public:
    shared_ptr<typename HierarchicVWC<TMESH>::Options> crs_opts;
  }; // VertexAMGFactory::State

  /** END State **/


  /** Options **/

  class VertexAMGFactoryOptions : public BaseAMGFactory::Options
  {
  public:

    /** choice of coarsening algorithm **/
    enum CRS_ALG : char { ECOL,                 // edge collapsing
			  AGG };                // aggregaion
    CRS_ALG crs_alg = AGG;

    /** General coarsening **/
    bool ecw_geom = true;                       // use geometric instead of harmonic mean when determining strength of connection
    bool ecw_robust = true;                     // use more expensive, but also more robust edge weights
    double min_ecw = 0.05;
    double min_vcw = 0.3;

    /** Discard **/
    int disc_max_bs = 5;

    /** AGG **/
    int n_levels_d2_agg = 1;                    // do this many levels MIS(2)-like aggregates (afterwards MIS(1)-like)

  public:

    VertexAMGFactoryOptions ()
      : BaseAMGFactory::Options()
    { ; }

    virtual void SetFromFlags (const Flags & flags, string prefix) override
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

      BaseAMGFactory::Options::SetFromFlags(flags, prefix);

      set_enum_opt(crs_alg, "crs_alg", {"ecol", "agg" });

      set_bool(ecw_geom, "ecw_geom");
      set_bool(ecw_robust, "ecw_robust");

      set_num(min_ecw, "edge_thresh");
      set_num(min_vcw, "vert_thresh");
      set_num(min_vcw, "vert_thresh");
      set_num(n_levels_d2_agg, "n_levels_d2_agg");
    } // VertexAMGFactoryOptions::SetFromFlags

  }; // VertexAMGFactoryOptions
    
  /** END Options **/


  /** VertexAMGFactory **/


  template<class ENERGY, class TMESH, int BS>
  VertexAMGFactory<ENERGY, TMESH, BS> :: VertexAMGFactory (shared_ptr<Options> opts)
    : BASE_CLASS(opts)
  {
    ;
  } // VertexAMGFactory(..)


  template<class ENERGY, class TMESH, int BS>
  VertexAMGFactory<ENERGY, TMESH, BS> :: ~VertexAMGFactory ()
  {
    ;
  } // ~VertexAMGFactory


  template<class ENERGY, class TMESH, int BS>
  BaseAMGFactory::State* VertexAMGFactory<ENERGY, TMESH, BS> :: AllocState () const
  {
    return new State();
  } // VertexAMGFactory::AllocState


  template<class ENERGY, class TMESH, int BS>
  void VertexAMGFactory<ENERGY, TMESH, BS> :: InitState (BaseAMGFactory::State & state, BaseAMGFactory::AMGLevel & lev) const
  {
    BASE_CLASS::InitState(state, lev);

    auto & s(static_cast<State&>(state));
    s.crs_opts = nullptr;
  } // VertexAMGFactory::InitState


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildCoarseMap (BaseAMGFactory::State & state)
  {
    auto & O(static_cast<Options&>(*options));

    Options::CRS_ALG calg = O.crs_alg;

    switch(calg) {
    case(Options::CRS_ALG::AGG): { return BuildAggMap(state); break; }
    case(Options::CRS_ALG::ECOL): { return BuildECMap(state); break; }
    default: { throw Exception("Invalid coarsen alg!"); break; }
    }

    return nullptr;
  } // VertexAMGFactory::BuildCoarseMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildAggMap (BaseAMGFactory::State & state)
  {
    auto & O = static_cast<Options&>(*options);
    typedef Agglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;
    typename AGG_CLASS::Options agg_opts;
    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_mesh).name() + string(" for BuildAggMap!")); }
    agg_opts.edge_thresh = O.min_ecw;
    agg_opts.vert_thresh = O.min_vcw;
    agg_opts.cw_geom = O.ecw_geom;
    agg_opts.robust = O.ecw_robust;
    agg_opts.dist2 = ( state.level[1] == 0 ) && ( state.level[0] < O.n_levels_d2_agg );
    // auto agglomerator = make_shared<Agglomerator<FACTORY>>(mesh, state.free_nodes, move(agg_opts));
    auto agglomerator = make_shared<AGG_CLASS>(mesh, state.free_nodes, move(agg_opts));
    return agglomerator;
  } // VertexAMGFactory::BuildCoarseMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildECMap (BaseAMGFactory::State & astate)
  {
    throw Exception("finish this up ...");
    auto & state(static_cast<State&>(astate));
    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_mesh).name() + string(" for BuildECMap!")); }
    // SetCoarseningOptions(*state.crs_opts, cmesh);
    auto calg = make_shared<BlockVWC<TMESH>> (state.crs_opts);
    auto grid_step = calg->Coarsen(mesh);
    return grid_step;
  } // VertexAMGFactory::BuildCoarseMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds)
  {
    static Timer t("PWProlMap"); RegionTimer rt(t);

    const auto & rcmap(*cmap);
    const TMESH & fmesh = static_cast<TMESH&>(*rcmap.GetMesh()); fmesh.CumulateData();
    const TMESH & cmesh = static_cast<TMESH&>(*rcmap.GetMappedMesh()); cmesh.CumulateData();

    size_t NV = fmesh.template GetNN<NT_VERTEX>();
    size_t NCV = cmesh.template GetNN<NT_VERTEX>();

    /** Alloc Matrix **/
    auto vmap = rcmap.template GetMap<NT_VERTEX>();
    Array<int> perow (NV); perow = 0;
    for (auto vnr : Range(NV))
      { if (vmap[vnr] != -1) perow[vnr] = 1; }

    // cout << "vmap: " << endl; prow2(vmap); cout << endl;

    auto prol = make_shared<TSPM_TM>(perow, NCV);

    // Fill Matrix
    auto f_v_data = get<0>(fmesh.Data())->Data();
    auto c_v_data = get<0>(cmesh.Data())->Data();
    for (auto vnr : Range(NV)) {
      auto cvnr = vmap[vnr];
      if (cvnr != -1) {
	prol->GetRowIndices(vnr)[0] = cvnr;
	ENERGY::CalcQHh(c_v_data[cvnr], f_v_data[vnr], prol->GetRowValues(vnr)[0]);
      }
    }

    // cout << "PWPROL: " << endl;
    // print_tm_spmat(cout, *prol); cout << endl;

    return make_shared<ProlMap<TSPM_TM>> (prol, fpds, cpds);
  } // VertexAMGFactory::PWProlMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> tfmesh)
  {
    static Timer t("SmoothedProlMap"); RegionTimer rt(t);

    if (pw_step == nullptr)
      { throw Exception("Need pw-map for SmoothedProlMap!"); }
    auto prol_map =  dynamic_pointer_cast<ProlMap<TSPM_TM>> (pw_step);
    if (prol_map == nullptr)
      { throw Exception(string("Invalid Map type ") + typeid(*pw_step).name() + string(" in SmoothedProlMap!")); }
    auto fmesh = dynamic_pointer_cast<TMESH>(tfmesh);
    if (fmesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*tfmesh).name() + string(" in SmoothedProlMap!")); }

    Options &O (static_cast<Options&>(*options));

    const double MIN_PROL_FRAC = O.sp_min_frac;
    const int MAX_PER_ROW = O.sp_max_per_row;
    const double omega = O.sp_omega;

    const TSPM_TM & pwprol = *prol_map->GetProl();

    const size_t NFV = pwprol.Height(), NCV = pwprol.Width();;

    const auto & FM(*fmesh); FM.CumulateData();
    auto avd = get<0>(FM.Data());
    auto vdata = avd->Data();
    auto aed = get<1>(FM.Data());
    auto edata = aed->Data();
    const auto & eqc_h = *FM.GetEQCHierarchy();
    const auto & fecon = *FM.GetEdgeCM();
    auto all_fedges = FM.template GetNodes<NT_EDGE>();

    auto NV = fmesh->template GetNN<NT_VERTEX>();
    Array<int> vmap(NV); vmap = -1;
    for (auto k : Range(NV)) {
      auto ri = pwprol.GetRowIndices(k);
      if (ri.Size())
	{ vmap[k] = ri[0]; }
    }

    Array<double> vw (NFV); vw = 0;
    auto neqcs = eqc_h.GetNEQCS();
    {
      INT<2, int> cv;
      auto doit = [&](auto the_edges) {
	for (const auto & edge : the_edges) {
	  if ( ((cv[0]=vmap[edge.v[0]]) != -1 ) &&
	       ((cv[1]=vmap[edge.v[1]]) != -1 ) &&
	       (cv[0]==cv[1]) ) {
	    // auto com_wt = self.template GetWeight<NT_EDGE>(fmesh, edge);
	    auto com_wt = ENERGY::GetApproxWeight(edata[edge.id]);
	    vw[edge.v[0]] += com_wt;
	    vw[edge.v[1]] += com_wt;
	  }
	}
      };
      for (auto eqc : Range(neqcs)) {
	if (!eqc_h.IsMasterOfEQC(eqc)) continue;
	doit(FM.template GetENodes<NT_EDGE>(eqc));
	doit(FM.template GetCNodes<NT_EDGE>(eqc));
      }
    }
    FM.template AllreduceNodalData<NT_VERTEX>(vw, [](auto & tab){return move(sum_table(tab)); }, false);


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
	auto EQ = FM.template GetEqcOfNode<NT_VERTEX>(V);
	auto ovs = fecon.GetRowIndices(V);
	auto eis = fecon.GetRowValues(V);
	size_t pos;
	for (auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if (is_invalid(cov) || cov==CV) continue;
	  auto oeq = FM.template GetEqcOfNode<NT_VERTEX>(ov);
	  // cout << V << " " << ov << " " << cov << " " << EQ << " " << oeq << " " << eqc_h.IsLEQ(EQ, oeq) << endl;
	  if (eqc_h.IsLEQ(EQ, oeq)) {
	    // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[eis[j]]);
	    auto wt = ENERGY::GetApproxWeight(edata[all_fedges[eis[j]].id]);
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
	auto EQ = FM.template GetEqcOfNode<NT_VERTEX>(V);
	auto graph_row = graph[V];
	auto all_ov = fecon.GetRowIndices(V);
	auto all_oe = fecon.GetRowValues(V);
	uve.SetSize0();
	for (auto j:Range(all_ov.Size())) {
	  auto ov = all_ov[j];
	  auto cov = vmap[ov];
	  if (is_valid(cov)) {
	    if (graph_row.Contains(cov)) {
	      auto eq = FM.template GetEqcOfNode<NT_VERTEX>(ov);
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
	  auto & fedge = all_fedges[used_edges[l]];
	  ENERGY::CalcRMBlock (block, edata[fedge.id], vdata[fedge.v[0]], vdata[fedge.v[1]]);
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

    // cout << "sprol (no cmesh):: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    return make_shared<ProlMap<TSPM_TM>> (sprol, pw_step->GetParDofs(), pw_step->GetMappedParDofs());
  } // VertexAMGFactory::SmoothedProlMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap)
  {
    static Timer t("SmoothedProlMap"); RegionTimer rt(t);

    if (pw_step == nullptr)
      { throw Exception("Need pw-map for SmoothedProlMap!"); }
    if (cmap == nullptr)
      { throw Exception("Need cmap for SmoothedProlMap!"); }
    auto prol_map =  dynamic_pointer_cast<ProlMap<TSPM_TM>> (pw_step);
    if (prol_map == nullptr)
      { throw Exception(string("Invalid Map type ") + typeid(*pw_step).name() + string(" in SmoothedProlMap!")); }
    
    const TMESH & FM(static_cast<TMESH&>(*cmap->GetMesh())); FM.CumulateData();
    const TMESH & CM(static_cast<TMESH&>(*cmap->GetMappedMesh())); CM.CumulateData();
    const TSPM_TM & pwprol = *prol_map->GetProl();

    const auto & eqc_h(*FM.GetEQCHierarchy()); // coarse eqch == fine eqch !!
    auto neqcs = eqc_h.GetNEQCS();

    auto avd = get<0>(FM.Data());
    auto vdata = avd->Data();
    auto aed = get<1>(FM.Data());
    auto edata = aed->Data();
    const auto & fecon = *FM.GetEdgeCM();
    auto all_fedges = FM.template GetNodes<NT_EDGE>();

    Options &O (static_cast<Options&>(*options));
    const double MIN_PROL_FRAC = O.sp_min_frac;
    const int MAX_PER_ROW = O.sp_max_per_row;
    const double omega = O.sp_omega;

    const size_t NFV = FM.template GetNN<NT_VERTEX>(), NCV = CM.template GetNN<NT_VERTEX>();
    auto vmap = cmap->template GetMap<NT_VERTEX>();

    /** For each fine vertex, find all coarse vertices we can (and should) prolongate from.
	The master of V does this. **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    Array<INT<2,double>> trow;
    Array<int> tcv, fin_row;
    FM.template ApplyEQ<NT_VERTEX>([&](auto EQ, auto V) LAMBDA_INLINE  {
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
	    // in_wt += self.template GetWeight<NT_EDGE>(fmesh, );
	    in_wt += ENERGY::GetApproxWeight(edata[int(eis[j])]);
	    continue;
	  }
	  // auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	  auto oeq = CM.template GetEqcOfNode<NT_VERTEX>(cov);
	  if (eqc_h.IsLEQ(EQ, oeq)) {
	    // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
	    auto wt = ENERGY::GetApproxWeight(edata[int(eis[j])]);
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
    shared_ptr<TSPM_TM> rmat = make_shared<TSPM_TM>(perow, NCV);
    const TSPM_TM & RM = *rmat;

    /** Fill Prolongation **/
    LocalHeap lh(2000000, "hold this", false); // ~2 MB LocalHeap
    Array<INT<2,int>> une(20);
    TM Q, Qij, Qji, diag, rvl, ID; SetIdentity(ID);
    FM.template ApplyEQ<NT_VERTEX>([&](auto EQ, auto V) LAMBDA_INLINE {
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
	    ENERGY::CalcQs(vdata[edge.v[L]], vdata[edge.v[1-L]], Qij, Qji);
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

    shared_ptr<TSPM_TM> sprol = prol_map->GetProl();
    sprol = MatMultAB(RM, *sprol);
    
    /** Now, unfortunately, we have to distribute matrix entries of sprol. We cannot do this for RM.
	(we are also using more local fine edges that map to less local coarse edges) **/
    if (eqc_h.GetCommunicator().Size() > 2) {
      const auto & SP = *sprol;
      Array<int> perow(sprol->Height()); perow = 0;
      FM.template ApplyEQ<NT_VERTEX>( Range(neqcs), [&](auto EQC, auto V) {
	  auto ris = sprol->GetRowIndices(V).Size();
	  perow[V] = ris;
	}, false); // all - also need to alloc loc!
      FM.template ScatterNodalData<NT_VERTEX>(perow);
      auto cumul_sp = make_shared<TSPM_TM>(perow, NCV);
      Array<int> eqc_perow(neqcs); eqc_perow = 0;
      if (neqcs > 1)
	FM.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
	    eqc_perow[EQC] += perow[V];
	  }, false); // all!
      Table<INT<2,int>> ex_ris(eqc_perow);
      Table<TM> ex_rvs(eqc_perow); eqc_perow = 0;
      if (neqcs > 1)
	FM.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
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
	FM.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
	    auto rvs = CSP.GetRowValues(V);
	    auto ris = CSP.GetRowIndices(V);
	    for (auto j : Range(ris)) {
	      auto tup = ex_ris[EQC][eqc_perow[EQC]];
	      ris[j] = CM.template MapENodeFromEQC<NT_VERTEX>(tup[1], eqc_h.GetEQCOfID(tup[0]));
	      rvs[j] = ex_rvs[EQC][eqc_perow[EQC]++];
	    }
	  }, false); // master!
      if (neqcs > 0)
	for (auto V : FM.template GetENodes<NT_VERTEX>(0)) {
	  CSP.GetRowIndices(V) = SP.GetRowIndices(V);
	  CSP.GetRowValues(V) = SP.GetRowValues(V);
	}
      sprol = cumul_sp;
      // cout << "CUMULATED SPROL: " << endl;
      // print_tm_spmat(cout, *sprol); cout << endl;
    }

    // cout << "sprol (with cmesh):: " << endl;
    // print_tm_spmat(cout, *sprol); cout << endl;

    return make_shared<ProlMap<TSPM_TM>> (sprol, pw_step->GetParDofs(), pw_step->GetMappedParDofs());
  } // VertexAMGFactory::SmoothedProlMap


  template<class ENERGY, class TMESH, int BS>
  bool VertexAMGFactory<ENERGY, TMESH, BS> :: TryDiscardStep (BaseAMGFactory::State & state)
  {
    if (!options->enable_disc)
      { return false; }

    if (state.free_nodes != nullptr)
      { throw Exception("discard with dirichlet TODO!!"); }

    shared_ptr<BaseDiscardMap> disc_map = BuildDiscardMap(state);

    if (disc_map == nullptr)
      { return false; }

    auto n_d_v = disc_map->GetNDroppedNodes<NT_VERTEX>();
    auto any_n_d_v = state.curr_mesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_d_v, MPI_SUM);

    cout << " disc dropped " << n_d_v << endl;

    bool map_ok = any_n_d_v != 0; // someone somewhere eliminated some verices

    if (map_ok) { // a non-negligible amount of vertices was eliminated
      auto elim_vs = disc_map->GetMesh()->template GetNNGlobal<NT_VERTEX>() - disc_map->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>();
      auto dv_frac = double(disc_map->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>()) / disc_map->GetMesh()->template GetNNGlobal<NT_VERTEX>();
      map_ok &= (dv_frac < 0.98);
    }

    if (!map_ok)
      { return false; }

    //TODO: disc prol map!!

    state.disc_map = disc_map;
    state.curr_mesh = disc_map->GetMappedMesh();
    state.curr_pds = this->BuildParallelDofs(state.curr_mesh);
    state.free_nodes = nullptr;
    // state.dof_map = disc_prol_map;

    return true;
  } // VertexAMGFactory::TryDiscardStep


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDiscardMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildDiscardMap (BaseAMGFactory::State & state)
  {
    auto & O(static_cast<Options&>(*options));
    auto tm_mesh = dynamic_pointer_cast<TMESH>(state.curr_mesh);
    auto disc_map = make_shared<VDiscardMap<TMESH>> (tm_mesh, O.disc_max_bs);
    return disc_map;
  } // VertexAMGFactory :: BuildDiscardMap

  /** END VertexAMGFactory **/

} // namespace amg

#endif
