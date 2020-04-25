#ifdef STOKES

#ifndef FILE_AMG_FACTORY_STOKES_IMPL_HPP
#define FILE_AMG_FACTORY_STOKES_IMPL_HPP

#include "amg_agg.hpp"
#include "amg_bla.hpp"

namespace amg
{

  /** StokesAMGFactory **/

  template<class TMESH, class ENERGY>
  StokesAMGFactory<TMESH, ENERGY> :: StokesAMGFactory (shared_ptr<StokesAMGFactory<TMESH, ENERGY>::Options> _opts)
    : BASE_CLASS(_opts)
  {
    ;
  } // StokesAMGFactory(..)


  template<class TMESH, class ENERGY>
  BaseAMGFactory::State* StokesAMGFactory<TMESH, ENERGY> :: AllocState () const
  {
    return new BaseAMGFactory::State();
  } // StokesAMGFactory::AllocState


  template<class TMESH, class ENERGY>
  void StokesAMGFactory<TMESH, ENERGY> :: InitState (BaseAMGFactory::State & state, BaseAMGFactory::AMGLevel & lev) const
  {
    BASE_CLASS::InitState(state, lev);
  } // StokesAMGFactory::InitState


  template<class TMESH, class ENERGY>
  shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildCoarseMap (BaseAMGFactory::State & state)
  {
    auto & O(static_cast<Options&>(*options));

    // typename Options::CRS_ALG calg = O.crs_alg;

    return BuildAggMap(state);

  //   switch(calg) {
  //   case(Options::CRS_ALG::AGG): { return BuildAggMap(state); break; }
  //   case(Options::CRS_ALG::ECOL): { return BuildECMap(state); break; }
  //   default: { throw Exception("Invalid coarsen alg!"); break; }
  //   }
  } // StokesAMGFactory::InitState


  template<class TMESH, class ENERGY>
  shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildAggMap (BaseAMGFactory::State & state)
  {
    auto & O = static_cast<Options&>(*options);
    typedef Agglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;
    typename AGG_CLASS::Options agg_opts;

    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_mesh).name() + string(" for BuildAggMap!")); }

    auto edges = mesh->template GetNodes<NT_EDGE>();
    Array<INT<2,double>> olded(edges.Size());
    auto vdata = get<0>(mesh->Data())->Data();
    auto edata = get<1>(mesh->Data())->Data();

    auto fnodes = state.free_nodes;
    int max_surf = 0;
    Array<int> free_surfs;
    for (auto v : Range(vdata.Size()))
      if ( (vdata[v].vol < 0) && ( (!fnodes || fnodes->Test(v)) ) ) {
	int surfnr = -(vdata[v].vol + 1);
	max_surf = max2(max_surf, surfnr);
	auto pos = merge_pos_in_sorted_array(surfnr, free_surfs);
	if ( (pos != -1) && (pos > 0) && (free_surfs[pos-1] == surfnr) )
	  { ; }
	else if (pos >= 0)
	  { free_surfs.Insert(pos, surfnr); }
      }
    Array<int> surf2row(1+max_surf); surf2row = -1;
    for (auto k : Range(free_surfs))
      { surf2row[free_surfs[k]] = k; }
    TableCreator<int> cfas(free_surfs.Size());
    for (; !cfas.Done(); cfas++) {
      for (auto v : Range(vdata.Size()))
	if ( (vdata[v].vol < 0) && ( (!fnodes || fnodes->Test(v)) ) ) {
	  int surfnr = -(vdata[v].vol + 1);
	  cfas.Add(surf2row[surfnr], v);
	}
    }

    // for (const auto & edge : edges) {
    //   if ( (vdata[edge.v[0]].vol < 0) || (vdata[edge.v[1]].vol < 0) ) {
    // 	olded[edge.id][0] = edata[edge.id].edi;//(0,0);
    // 	edata[edge.id].edi = 0;
    // 	olded[edge.id][1] = edata[edge.id].edj;//(0,0);
    // 	edata[edge.id].edj = 0;
    //   }
    // }

    agg_opts.edge_thresh = 0.05;
    agg_opts.vert_thresh = 0;
    agg_opts.cw_geom = false;
    agg_opts.neib_boost = false;
    agg_opts.robust = false;
    agg_opts.dist2 = (state.level[0] == 0) && (state.level[1] == 0);
    // agg_opts.dist2 = state.level[1] == 0;
    // agg_opts.dist2 = false;
    // agg_opts.dist2 = true;

    auto agglomerator = make_shared<AGG_CLASS>(mesh, state.free_nodes, move(agg_opts));

    auto faggs = cfas.MoveTable();

    // cout << " Fixed Aggs:" << endl << faggs << endl;
    agglomerator->SetFixedAggs(move(faggs));

    // for (const auto & edge : edges) {
    //   if ( (vdata[edge.v[0]].vol < 0) || (vdata[edge.v[1]].vol < 0) ) {
    // 	SetScalIdentity(olded[edge.id][0], edata[edge.id].edi);
    // 	SetScalIdentity(olded[edge.id][1], edata[edge.id].edj);
    //   }
    // }

    return agglomerator;
  } // StokesAMGFactory::BuildAggMap


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds)
  {
    auto fmesh = static_pointer_cast<TMESH>(cmap->GetMesh());
    auto cmesh = static_pointer_cast<TMESH>(cmap->GetMappedMesh());
    auto vmap = cmap->GetMap<NT_VERTEX>();
    auto emap = cmap->GetMap<NT_EDGE>();
    TableCreator<int> cva(cmesh->template GetNN<NT_VERTEX>());
    for (; !cva.Done(); cva++) {
      for (auto k : Range(vmap))
	if (vmap[k] != -1)
	  { cva.Add(vmap[k], k); }
    }
    auto v_aggs = cva.MoveTable();
    auto pwprol = BuildPWProl_impl (fpds, cpds, fmesh, cmesh, vmap, emap, v_aggs);
    return make_shared<ProlMap<TSPM_TM>>(pwprol, fpds, cpds);
  } // StokesAMGFactory::PWProlMap


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh)
  {
    throw Exception("Stokes SmoothedProlMap needs coarse map !!");
    return nullptr;
  } // StokesAMGFactory::SmoothedProlMap


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap)
  {
    // throw Exception("Stokes SmoothedProlMap not yet implemented -> TODO !!");
    if (pw_step == nullptr)
      { throw Exception("Need pw-map for SmoothedProlMap!"); }
    auto prol_map =  dynamic_pointer_cast<ProlMap<TSPM_TM>> (pw_step);
    if (prol_map == nullptr)
      { throw Exception(string("Invalid Map type ") + typeid(*pw_step).name() + string(" in SmoothedProlMap!")); }
    auto fmesh = dynamic_pointer_cast<TMESH>(cmap->GetMesh());
    if (fmesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*cmap->GetMesh()).name() + string(" in SmoothedProlMap!")); }
    auto cmesh = dynamic_pointer_cast<TMESH>(cmap->GetMappedMesh());
    if (cmesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*cmap->GetMappedMesh()).name() + string(" in SmoothedProlMap!")); }
    auto vmap = cmap->GetMap<NT_VERTEX>();
    auto emap = cmap->GetMap<NT_EDGE>();
    TableCreator<int> cva(cmesh->template GetNN<NT_VERTEX>());
    for (; !cva.Done(); cva++) {
      for (auto k : Range(vmap))
	if (vmap[k] != -1)
	  { cva.Add(vmap[k], k); }
    }
    auto v_aggs = cva.MoveTable();
    auto sprol = SmoothProlMap_impl(prol_map, fmesh, cmesh, vmap, emap, v_aggs);
    prol_map->SetProl(sprol);
    return prol_map;
  } // StokesAMGFactory::SmoothedProlMap


  template<class TMESH, class ENERGY> shared_ptr<typename StokesAMGFactory<TMESH, ENERGY>::TSPM_TM>
  StokesAMGFactory<TMESH, ENERGY> :: SmoothProlMap_impl (shared_ptr<ProlMap<TSPM_TM>> prol_map,
							 shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
							 FlatArray<int> vmap, FlatArray<int> emap,
							 FlatTable<int> v_aggs)
  {
    cout << " STOKES SMOOTH PROL " << endl;

    auto  &O = static_cast<Options&>(*options);

    auto pwprol = prol_map->GetProl();
    auto fpds = prol_map->GetParDofs();
    auto cpds = prol_map->GetMappedParDofs();

    static constexpr int BS = mat_traits<TM>::HEIGHT;
    const double MIN_PROL_FRAC = O.sp_min_frac;
    const int MAX_PER_ROW = O.sp_max_per_row;
    const int MAX_NEIBS = MAX_PER_ROW - 1;
    const double omega = O.sp_omega;

    /** fine mesh **/
    const auto & FM(*fmesh); FM.CumulateData();
    const auto & fecon(*FM.GetEdgeCM());
    auto fvd = get<0>(FM.Data())->Data();
    auto fed = get<1>(FM.Data())->Data();
    size_t FNV = FM.template GetNN<NT_VERTEX>(), FNE = FM.template GetNN<NT_EDGE>();
    auto fedges = FM.template GetNodes<NT_EDGE>();
    auto free_fes = FM.GetFreeNodes();

    if (free_fes)
      cout << " free_fes: " << endl << *free_fes << endl;
    
    /** coarse mesh **/
    const auto & CM(*cmesh); CM.CumulateData();
    const auto & cecon(*CM.GetEdgeCM());
    auto cvd = get<0>(CM.Data())->Data();
    auto ced = get<1>(CM.Data())->Data();
    size_t CNV = CM.template GetNN<NT_VERTEX>(), CNE = CM.template GetNN<NT_EDGE>();
    auto cedges = CM.template GetNodes<NT_EDGE>();

    cout << "fecon: " << endl << fecon << endl;
    cout << "cecon: " << endl << cecon << endl;
    
    /** Here I also need C->F edge mapping (TODO: for now, fine to coarse edge count would be enough) **/
    TableCreator<int> cc2fe(CNE);
    for (; !cc2fe.Done(); cc2fe++) {
      for (auto fenr : Range(FNE)) {
    	auto cedge = emap[fenr];
    	if (cedge != -1)
    	  { cc2fe.Add(cedge, fenr); }
      }
    }
    Table<int> c2fe = cc2fe.MoveTable();

    /** prol dims **/
    size_t H = FNE, W = CNE;

    LocalHeap lh(2000000, "mooooore memoryyyyyy", false); // ~2 MB LocalHeap

    /** Determine Graph for facet DOFs **/
    Array<tuple<int, double>> cneibs(10);
    Array<int> dummy; Array<int> cols;
    Array<FlatArray<int>> t(3); t[0].Assign(1, lh);
    auto it_f_g = [&](auto lam) {
      for (auto cenr : Range(CNE)) {
	auto & cedge = cedges[cenr];
	cout << " calc strong c neibs for cedge " << cedge << endl;
	int cv0 = cedge.v[0], cv1 = cedge.v[1];
	FlatArray<int> cneibs0 = cecon.GetRowIndices(cv0), cneibs1 = cecon.GetRowIndices(cv1);
	FlatVector<double> ces0 = cecon.GetRowValues(cv0), ces1 = cecon.GetRowValues(cv1);
	int cs0 = cneibs0.Size(), cs1 = cneibs1.Size();
	int maxcns = 0; 
	if (cvd[cv0].vol < 0)
	  { cs0 = 0; }
	else
	  { maxcns--; }
	if (cvd[cv1].vol < 0)
	  { cs1 = 0; }
	else
	  { maxcns--; }
	maxcns += cs0 + cs1;
	if ( (c2fe[cenr].Size() == 1) || (cs0 + cs1 == 0) ) { // case 1 saves complexity, case cannot happen i think ...
	  cols.SetSize(1); cols[0] = cenr;
	  lam(cenr, cols);
	  continue;
	}
	cneibs.SetSize(maxcns);
	int cnt = 0;
	for (auto j : Range(cs0)) { // TODO: here I should check EQCS and set wt to -1 or sth like that
	  auto cvnn = cneibs0[j];
	  if (cvnn != cv1) {
	    int neid(ces0[j]);
	    auto & neibedge = cedges[neid];
	    cneibs[cnt++] = { neid, ENERGY::GetApproxWtDualEdge(ced[cenr], false, ced[neid], (cv0 > cvnn)) };
	  }
	}
	for (auto j : Range(cs1)) {
	  auto cvnn = cneibs1[j];
	  if (cvnn != cv0) {
	    int neid(ces1[j]);
	    auto & neibedge = cedges[neid];
	    cneibs[cnt++] = { neid, ENERGY::GetApproxWtDualEdge(ced[cenr], true, ced[neid], (cv1 > cvnn)) };
	  }
	}
	cneibs.SetSize(cnt); // maybe a couple fewer is BND verts
	cout << "  all cneibs " << endl; prow(cneibs); cout << endl;
	QuickSort(cneibs, [&](auto i, auto j) { return get<1>(i) > get<1>(j); });
	cout << "  sorted cneibs " << endl; prow2(cneibs); cout << endl;
	double sum_wt = get<1>(cneibs[0]); int tnn = 1;
	for (int k : Range(1, min(MAX_NEIBS, int(cneibs.Size())))) {
	  auto wt =  get<1>(cneibs[k]);
	  if (wt > 0) {
	    sum_wt += wt;
	    if (wt < MIN_PROL_FRAC * sum_wt)
	      { break; }
	    else
	      { tnn++; }
	  }
	  else
	    { break; }
	}
	cols.SetSize(1 + tnn);
	for (auto j : Range(tnn))
	  { cols[j] = get<0>(cneibs[j]); }
	cols[tnn] = cenr;
	cout << " graph-row " << cenr << ": "; prow(cols); cout << endl;
	lam(cenr, cols);
      }
    };
    Array<int> perow(H); perow.SetSize(CNE);
    it_f_g([&](auto row, auto& cols)
	   { perow[row] = cols.Size(); });
    Table<int> cfgraph(perow);
    it_f_g([&](auto row, auto& cols) {
	QuickSort(cols);
	cfgraph[row] = cols; 
      });

    cout << " cfgraph: " << endl << cfgraph << endl;

    // /** Check if there are any improvements to the pw-prol to be made at all ? **/
    // bool hntriv = false;
    // for (auto k : Range(CNE))
    //   if (fgraph[k].Size() > 1)
    // 	{ hntriv = true; }
    // if (!hntriv)
    //   { return pwprol; }

    /** Determine final Graph:
	- for agg-int DOFs: these are all coarse DOFs any agg-bnd edges depend on
	- for fine facet DOFs: just those from the coarse edge **/
    Array<FlatArray<int>> mcs(10);
    perow = 0;
    auto it_final_g = [&](auto lam) {
      for (auto fenr : Range(H)) {
	auto & fedge = fedges[fenr];
	if ( free_fes && (!free_fes->Test(fenr)) ) // dirichlet
	  { lam(fenr, dummy); }
	else {
	  auto cenr = emap[fenr];
	  if (cenr != -1) // edge connects two agglomerates
	    { lam(fenr, cfgraph[cenr]); }
	  else {
	    int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
	    if (cv0 == cv1) {
	      if (cv0 == -1) // I don't think this can happen - per definition emap must be -1
		{ throw Exception("Weird case A!"); }
	      else { // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate
		auto cneibs = cecon.GetRowIndices(cv0);
		auto ceids = cecon.GetRowValues(cv0);
		cols.SetSize0();
		mcs.SetSize0(); mcs.SetSize(ceids.Size());
		int maxcols = 0;
		for (auto j : Range(mcs))
		  { mcs[j].Assign(cfgraph[int(ceids[j])]); maxcols += mcs[j].Size(); }
		cols.SetSize0(); cols.SetSize(maxcols);
		merge_arrays(mcs, cols, [&](auto i, auto j) LAMBDA_INLINE { return i < j; });
		lam(fenr, cols);
	      }
	    }
	    else // edge between a Dirichlet BND vertex and an interior one
	      { lam(fenr, dummy); }
	  }
	}
      }
    };
    perow.SetSize(H);
    it_final_g([&](auto row, FlatArray<int> cols)
	       { perow[row] = cols.Size(); });
    Table<int> graph(perow);
    it_final_g([&](auto row, FlatArray<int> cols)
	       { graph[row] = cols; });

    cout << " graph: " << endl << graph << endl;

    /** Alloc Sprol **/
    auto sprol = make_shared<TSPM_TM>(perow, CNE);
    const auto & SP = *sprol;
    const auto & PWP = *pwprol;

    /** Fill facets (and trivial cases) **/
    Array<FlatArray<int>> ag01(2);
    Array<int> agg_vs;
    for (auto cenr : Range(CNE)) {
      auto & cedge = cedges[cenr];
      // FlatArray<int> ffedges = c2fedge[cenr];
      int cv0 = cedge.v[0], cv1 = cedge.v[1];
      FlatArray<int> agg0 = v_aggs[cv0], agg1 = v_aggs[cv1];

      if (cfgraph[cenr].Size() == 1) { // trivial case - copy pwprol!
	cout << " cedge " << cenr << ", triv case " << endl;
	for (auto vk : agg0) {
	  auto vk_neibs = fecon.GetRowIndices(vk);
	  auto vk_fs = fecon.GetRowValues(vk);
	  for (auto j : Range(vk_neibs)) {
	    auto fenr = int(vk_fs[j]);
	    auto vj = vk_neibs[j];
	    if (vmap[vj] == cv1) {
	      SP.GetRowIndices(fenr) = PWP.GetRowIndices(fenr);
	      SP.GetRowValues(fenr) = PWP.GetRowValues(fenr);
	    }
	  }	  
	}
      }
      else { // non-trivial case: coarse edge consists of more than one fine edge

	HeapReset hr(lh);

	cout << " fill agg facet " << cenr << endl;
	cout << " cedge " << cedge << endl;

	auto it_f_facets = [&](auto lam) {
	  auto it_agg_es = [&](auto aggl, auto cva, auto cvb, FlatArray<int> agg_vs, FlatArray<int> n_agg_vs) {
	    if (cvd[cva].vol < 0)
	      { return; }
	    bool surfb = cvd[cvb].vol < 0;
	    for (auto k : Range(agg_vs)) {
	      auto vk = agg_vs[k];
	      auto vk_neibs = fecon.GetRowIndices(vk);
	      auto vk_fs = fecon.GetRowValues(vk);
	      for (auto j : Range(vk_neibs)) {
		auto vj = vk_neibs[j];
		auto cvj = vmap[vj];
		if (cvj == cva) { // neib in same agg - interior facet!
		  auto kj = find_in_sorted_array(vj, agg_vs);
		  if (vj > vk) // do not count interior facets twice!
		    { lam(vk, aggl, k, vj, aggl, kj, int(vk_fs[j])); }
		}
		else if (cvj == cvb) {
		  auto kj = find_in_sorted_array(vj, n_agg_vs);
		  if ( surfb || (vj > vk) ) // do not count facets between aggs twice (if neib is surf, we wont)
		    { lam(vk, aggl, k, vj, (1-aggl), kj, int(vk_fs[j])); }
		}
		else // neib in different agg (or dirichlet)
		  { lam(vk, aggl, k, vj, -1, -1, int(vk_fs[j])); }
	      }
	    }
	  };
	  it_agg_es(0, cv0, cv1, agg0, agg1);
	  it_agg_es(1, cv1, cv0, agg1, agg0);
	};
	// all / interior / facet / facet-neib / facet prim
	int nff = 0, nffi = 0, nfff = 0, nfffn = 0, nfffp = 0;
	it_f_facets([&](int vi, int li, int ki, int vj, int lj, int kj, int eid) LAMBDA_INLINE {
	    nff++;
	    if (lj == -1) // outer facet
	      if ( (vmap[vi] == -1) || (vmap[vj] == -1) ) // dirichlet - assemble but treat as interior? [[does not matter for flow here]]
		{ nffi++; }
	      else
		{ nfff++; nfffn++; }
	    else if (lj == li) // agg-interior facet
	      { nffi++; }
	    else // facet between agg0, agg1
	      { nfff++; nfffp++; }
	  });
	FlatArray<int> ffacets(nff, lh), ffiinds(nffi, lh), fffinds(nfff, lh),
	  fffpinds(nfffp, lh), fffninds(nfffn, lh); // fffpinds, fffninds numbered w.r.t scur-complement numbers
	nff = nffi = nfff = nfffn = nfffp = 0;
	it_f_facets([&](int vi, int li, int ki, int vj, int lj, int kj, int eid) LAMBDA_INLINE {
	    if (lj == -1) // facet neib
	      if ( (vmap[vi] == -1) || (vmap[vj] == -1) ) // dirichlet - assemble but treat as interior? [[does not matter for flow here]]
		{ ffiinds[nfff++] = eid; }
	      else
		{ fffinds[nfff++] = fffninds[nfffn++] = eid; }
	    else if (lj == li)
	      { ffiinds[nffi++] = eid; }
	    else // facet between agg0, agg1
	      { fffinds[nfff++] = fffpinds[nfffp++] = eid; }
	    ffacets[nff++] = eid;
	  });
	auto indify = [&](auto a, auto b) {
	  for (auto k : Range(a))
	    { a[k] = find_in_sorted_array(a[k], b); }
	};
	QuickSort(fffinds);
	indify(fffpinds, fffinds);
	indify(fffninds, fffinds);
	QuickSort(ffacets);
	indify(fffinds, ffacets);
	indify(ffiinds, ffacets);
	
	cout << " nff f i p n " << nff << " " << nffi << " " << nfff << " " << nfffp << " " << nfffn << endl;
	cout << "  ffacets: "; prow2(ffacets); cout << endl;
	cout << "  ffiinds: "; prow(ffiinds); cout << endl;
	cout << "  fffinds: "; prow(fffinds); cout << endl;
	cout << "  fffpinds: "; prow(fffpinds); cout << endl;
	cout << "  fffninds: "; prow(fffninds); cout << endl;

	auto it_f_edges = [&](auto lam) {
	  auto it_agg_f_edges = [&](FlatArray<int> agg_vs, int cv) {
	    for (auto kvi : Range(agg_vs)) {
	      auto vi = agg_vs[kvi];
	      auto vi_neibs = fecon.GetRowIndices(vi);
	      auto vi_fs = fecon.GetRowValues(vi);
	      const int nneibs = vi_fs.Size();
	      for (auto lk : Range(nneibs)) {
		auto vk = vi_neibs[lk];
		auto fik = int(vi_fs[lk]);
		auto kfik = find_in_sorted_array(fik, ffacets);
		for (auto lj : Range(lk)) {
		  auto vj = vi_neibs[lj];
		  auto fij = int(vi_fs[lj]);
		  auto kfij = find_in_sorted_array(fij, ffacets);
		  lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
		}
	      }
	    }
	  };
	  it_agg_f_edges(agg0, cv0);
	  it_agg_f_edges(agg1, cv1);
	};

	int HA = nff * BS, HS = nfff * BS;
	FlatMatrix<double> S(HS, HS, lh);

	/** A block **/
	FlatMatrix<double> A(HA, HA, lh); A = 0;
	FlatMatrix<TM> eblock(2,2,lh);
	it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
	    // if ( (vmap[vj] != -1) && (vmap[vk] != -1) ) { // assemble these in and eliminate with SC
	      ENERGY::CalcRMBlock(eblock, fvd[vi], fvd[vj], fvd[vk], fed[fij], (vi > vj), fed[fik], (vi > vk));
	      Iterate<2>([&](auto i) {
		  int osi = BS * (i.value == 0 ? kfij : kfik);
		  Iterate<2>([&](auto j) {
		      int osj = BS * (j.value == 0 ? kfij : kfik);
		      A.Rows(osi, osi + BS).Cols(osj, osj + BS) += eblock(i.value, j.value);
		    });
		});
	    // }
	  });

	cout << "  A block " << endl << A << endl;

	/** Calc Schur **/
	FlatArray<int> colsi(nffi * BS, lh), colsf(nfff * BS, lh);
	int nci = 0, ncf = 0;
	for (auto k : Range(nffi))
	  for (auto l : Range(BS))
	    { colsi[nci++] = BS * ffiinds[k] + l; }
	for (auto k : Range(nfff))
	  for (auto l : Range(BS))
	    { colsf[ncf++] = BS * fffinds[k] + l; }
	FlatMatrix<double> Aii(nci, nci, lh);
	Aii = A.Rows(colsi).Cols(colsi);
	cout << "   Aii  " << endl << Aii << endl;
	CalcInverse(Aii);
	cout << "   inv Aii  " << endl << Aii << endl;
	FlatMatrix<double> Aii_Aif(nci, ncf, lh);
	Aii_Aif = Aii * A.Rows(colsi).Cols(colsf);
	S = A.Rows(colsf).Cols(colsf) - A.Rows(colsf).Cols(colsi) * Aii_Aif;
	cout << "  S  " << endl << S << endl;
	

	int HSp = BS * nfffp, HSn = BS * nfffn, HB = 1, HM = HSp + HB;
	FlatMatrix<double> M(HM, HM, lh);
	M = 0;
	auto Sp = M.Rows(0, HSp).Cols(0, HSp);
	auto B = M.Rows(HSp, HM).Cols(0, HSp);
	auto BT = M.Rows(0, HSp).Cols(HSp, HM);

	/** (fine) B block **/
	for (auto k : Range(nfffp)) {
	  int ind = fffpinds[k];
	  int col = BS * k;
	  int fenr = ffacets[ind];
	  auto & edge = fedges[fenr];
	  double fac = (vmap[edge.v[0]] == cv0) ? 1.0 : -1.0;
	  auto & flow = fed[fenr].flow;
	  for (auto l : Range(BS))
	    { B(0, col + l) = flow(l); }
	}
	BT = Trans(B);

	cout << "  B: " << endl << B << endl;

	/** Sp **/
	FlatArray<int> colsp(nfffp * BS, lh), colsn(nfffn * BS, lh);
	int ncp = 0, ncn = 0;
	for (auto k : Range(nfffp))
	  for (auto j : Range(BS))
	    { colsp[ncp++] = BS * fffpinds[k] + j; }
	for (auto k : Range(nfffn))
	  for (auto j : Range(BS))
	    { colsn[ncn++] = BS * fffninds[k] + j; }
	cout << "  colsp: "; prow(colsp); cout << endl;
	cout << "  colsn: "; prow(colsn); cout << endl;
	FlatMatrix<double> Spp(nfffp * BS, nfffp * BS, lh);
	Sp = S.Rows(colsp).Cols(colsp);

	cout << "  M: " << endl << M << endl;

	/** Get needed PWP-block **/
	FlatArray<int> prol_cols = cfgraph[cenr];
	int pcs = prol_cols.Size(), ccn = 0, ccp = 0;
	FlatArray<int> ccolsn(BS * (pcs-1), lh), ccolsp(BS, lh);
	FlatMatrix<double> Pblock(BS * nfff, BS * pcs, lh); Pblock = 0;
	for (auto k : Range(nfff)) {
	  int fenr = ffacets[fffinds[k]]; int cenr = emap[fenr];
	  int col = find_in_sorted_array(cenr, prol_cols);
	  Pblock.Rows(k * BS, (k + 1) * BS).Cols(col * BS, (col + 1) * BS) = PWP(fenr, cenr);
	  cout << " add prol entry " << fenr << " x " << cenr << " = " << endl << PWP(fenr, cenr) << endl;
	}
	for (auto k : Range(pcs)) {
	  if (prol_cols[k] == cenr)
	    for (auto l : Range(BS))
	      { ccolsp[ccp++] = BS * k + l; }
	  else
	    for (auto l : Range(BS))
	      { ccolsn[ccn++] = BS * k + l; }
	}

	cout << " pblock: " << endl << Pblock << endl;

	/** RHS **/
	FlatMatrix<double> rhs (HM, pcs*BS, lh);
	rhs = 0;
	rhs.Rows(0, HSp).Cols(ccolsn) = -S.Rows(colsp).Cols(colsn) * Pblock.Rows(colsn).Cols(ccolsn);
	for (auto l : Range(BS))
	  { rhs(HSp, ccolsp[l]) = ced[cenr].flow(l); }

	cout << " rhs: " << endl << rhs << endl;

	/** Solve **/
	FlatMatrix<double> prolvals(HM, pcs*BS, lh);
	CalcInverse(M);
	prolvals = M * rhs;

	cout << " prolvals: " << endl << prolvals << endl;

	cout << " write to rows : ";
	/** write into sprol **/
	for (auto k : Range(fffpinds)) {
	  auto ff = ffacets[fffinds[fffpinds[k]]];
	  cout << ff << " ";
	  auto ris = sprol->GetRowIndices(ff);
	  auto rvs = sprol->GetRowValues(ff);
	  ris = graph[ff];
	  for (auto j : Range(rvs))
	    { rvs[j] = prolvals.Rows(BS * k, BS * (k + 1)).Cols(BS * j, (1 + BS) * j); }
	}
	cout << endl;
      } // non-triv case
    }
    
    cout << "FACET SPROL: " << endl;
    print_tm_spmat(cout, SP);

    /** Fill agg-interior edges, same as for pwprol **/
    Array<int> rcfacets(20);
    for (auto agg_nr : Range(v_aggs)) {
      HeapReset hr(lh);
      auto agg_vs = v_aggs[agg_nr];
      auto cv = vmap[agg_vs[0]];
      
      if (agg_vs.Size() > 1) { // for single verts there are no interior edges
	cout << "cv is " << cv << endl;
	cout << "fill smoothed agg " << agg_nr << ", agg_vs: "; prow(agg_vs); cout << endl;
	cout << "v vols: " << endl;
	for (auto v : agg_vs)
	  { cout << fvd[v].vol << " "; }
	cout << endl;
	auto cv = vmap[agg_vs[0]];
	auto cneibs = cecon.GetRowIndices(cv);
	auto cfacets = cecon.GetRowValues(cv);
	int nfv = agg_vs.Size(), ncv = 1;     // # fine/coarse elements
	/** count fine facets **/
	int nff = 0; int nffi = 0, nfff = 0;  // # fine facets (int/facet)
	auto it_f_facets = [&](auto lam) {
	  for (auto k : Range(agg_vs)) {
	    auto vk = agg_vs[k];
	    cout << k << ", vk " << vk << endl;
    	    auto vk_neibs = fecon.GetRowIndices(vk);
	    cout << "neibs "; prow(vk_neibs); cout << endl;
    	    auto vk_fs = fecon.GetRowValues(vk);
	    for (auto j : Range(vk_neibs)) {
	      auto vj = vk_neibs[j];
	      auto cvj = vmap[vj];
	      cout << j << " vj " << vj << " cvj " << cvj << endl;
	      if (cvj == cv) { // neib in same agg - interior facet!
		auto kj = find_in_sorted_array(vj, agg_vs);
		if (vj > vk) { // do not count interior facets twice!
		  lam(vk, k, vj, kj, int(vk_fs[j]));
		}
	      }
	      else // neib in different agg (or dirichlet)
		{ lam(vk, k, vj, -1, int(vk_fs[j])); }
	    }
	  }
	};
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    nff++;
	    if (kj == -1) // ex-facet
	      { nfff++; }
	    else
	      { nffi++; }
	  });
	cout << "nff/f/i " << nff << " " << nfff << " " << nffi << endl << endl;
	/** All vertices (in this connected component) are in one agglomerate - nothing to do!
	    Sometimes happens on coarsest level. **/
	if (nfff == 0)
	  { continue; }

	/** I think this can only happen for my special test case with virtual vertex aggs ! **/
	if (nffi == 0)
	  { continue; }
	/** fine facet arrays **/
	FlatArray<int> index(nff, lh), buffer(nff, lh);
	FlatArray<int> ffacets(nff, lh), ffiinds(nffi, lh), fffinds(nfff, lh);
	nff = nffi = nfff = 0;
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    index[nff] = nff;
	    if (kj == -1)
	      { fffinds[nfff++] = nff; }
	    else
	      { ffiinds[nffi++] = nff; }
	    ffacets[nff++] = eid;
	  });

	bool ntriv = false;
	for (auto j : Range(cfacets))
	  if (cfgraph[j].Size() > 1)
	    { ntriv = true; break; }
	if (!ntriv) { // trivial case - copy pwprol
	  for (auto fnr : ffacets) {
	    SP.GetRowIndices(fnr) = PWP.GetRowIndices(fnr);
	    SP.GetRowValues(fnr) = PWP.GetRowValues(fnr);
	  }
	  continue;
	}

	cout << "unsorted ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl;
	cout << "unsorted ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl;
	cout << "unsorted fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl;
	QuickSortI(ffacets, index);
	for (auto k : Range(nffi))
	  { ffiinds[k] = index.Pos(ffiinds[k]); }
	for (auto k : Range(nfff)) // TODO...
	  { fffinds[k] = index.Pos(fffinds[k]); }
	cout << "index " << nff << ": "; prow2(index, cout << endl); cout << endl;
	ApplyPermutation(ffacets, index);
	cout << "ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl;
	cout << "ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl;
	cout << "fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl;

	auto it_f_edges = [&](auto lam) {
	  for (auto kvi : Range(agg_vs)) {
	    auto vi = agg_vs[kvi];
    	    auto vi_neibs = fecon.GetRowIndices(vi);
    	    auto vi_fs = fecon.GetRowValues(vi);
	    const int nneibs = vi_fs.Size();
	    for (auto lk : Range(nneibs)) {
	      auto vk = vi_neibs[lk];
	      auto fik = int(vi_fs[lk]);
	      auto kfik = find_in_sorted_array(fik, ffacets);
	      for (auto lj : Range(lk)) {
		auto vj = vi_neibs[lj];
		auto fij = int(vi_fs[lj]);
		auto kfij = find_in_sorted_array(fij, ffacets);
		lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
	      }
	    }
	  }
	};

	int HA = nff * BS, HB = 0;

	auto it_real_vs = [&](auto lam) {
	  int rkvi = 0;
	  for (auto kvi : Range(agg_vs)) {
	    auto vi = agg_vs[kvi];
	    bool is_real = fvd[vi].vol > 0;
	    /** This removes div-constraint on els that have a dirichlet facet **/
	    // for (auto vj : vi_neibs)
	      // if (vmap[vj] == -1)
		// { is_real = false; }
	    lam(is_real, vi, kvi, rkvi);
	    if (is_real)
	      { rkvi++; }
	  }
	};
	bool has_outflow = false;
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { has_outflow = true; }
	    else
	      { HB++; }
	  });
	/** If there is more than one element and no outflow facet, we need to lock constant pressure.
	    If we have only one "real" element, there should always be an outflow, I believe. **/
	bool lock_const_pressure = (!has_outflow) && (HB > 1);
	if ( lock_const_pressure )
	  { HB++; }

	int HM = HA + HB;

	cout << "H A/B/M = " << HA << " / " << HB << " / " << HM << endl;

	FlatMatrix<double> M(HM, HM, lh); M = 0;
	auto A = M.Rows(0, HA).Cols(0, HA);

	/** A block **/
	// FlatMatrix<double> A(nff * BS, nff * BS, lh); A = 0;
	auto a = M.Rows(0, HA).Cols(0, HA);
	FlatMatrix<TM> eblock(2,2,lh);
	it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
	    /** Only an energy-contrib if neither vertex is Dirichlet or grounded. Otherwise kernel vectors are disturbed. **/
	    if ( (vmap[vj] != -1) && (vmap[vk] != -1) ) {
	      ENERGY::CalcRMBlock(eblock, fvd[vi], fvd[vj], fvd[vk], fed[fij], (vi > vj), fed[fik], (vi > vk));
	      // cout << "block verts " << vi << "-" << vj << "-" << vk << endl;
	      // cout << "block faces " << fij << "-" << fik << endl;
	      // cout << "block loc faces " << kfij << "-" << kfik << endl;
	      // cout << "block:" << endl; print_tm_mat(cout, eblock); cout << endl;
	      Iterate<2>([&](auto i) {
		  int osi = BS * (i.value == 0 ? kfij : kfik);
		  Iterate<2>([&](auto j) {
		      int osj = BS * (j.value == 0 ? kfij : kfik);
		      A.Rows(osi, osi + BS).Cols(osj, osj + BS) += eblock(i.value, j.value);
		    });
		});
	    }
	  });

	cout << "A block: " << endl << A << endl;

	/** (fine) B blocks **/
	auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
	auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
	double bfsum = 0;
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { return; }
	    auto BfRow = Bf.Row(rkvi);
	    auto vi_neibs = fecon.GetRowIndices(vi);
	    auto vi_fs = fecon.GetRowValues(vi);
	    cout << "calc b for " << kvi << ", vi " << vi << ", rkvi " << rkvi << endl;
	    cout << "neibs "; prow(vi_neibs); cout << endl;
	    cout << "edgs "; prow(vi_fs); cout << endl;
	    const double fac = 1.0/fvd[vi].vol;
	    for (auto j : Range(vi_fs)) {
	      auto vj = vi_neibs[j];
	      auto fij = int(vi_fs[j]);
	      auto kfij = find_in_sorted_array(fij, ffacets);
	      auto & fijd = fed[fij];
	      int col = BS * kfij;
	      cout << "j " << j << ", vj " << vj << ", fij " << fij << ", kfij " << kfij << ", col " << col << ", fac " << fac << endl;
	      cout << "vol volinv: " << fvd[vi].vol << " " << fac << endl;
	      cout << "flow: " << fijd.flow << endl;
	      for (auto l : Range(BS)) {
		BfRow(col++) = ( (vi < vj) ? 1.0 : -1.0) * fijd.flow(l);
		// BfRow(col++) = ( (vi < vj) ? fac : -fac) * fijd.flow(l);
		bfsum += fac * abs(fijd.flow(l));
	      }
	    }
	  });
	BfT = Trans(Bf);

	/** (coarse) B **/
	int ncf_geom = cfacets.Size();       // # actual coarse facets
	Array<FlatArray<int>> cfgrows(ncf_geom);
	int ncf = 0;
	for (auto k : Range(cfgrows))
	  { cfgrows[k].Assign(cfgraph[cfacets[k]]); ncf += cfgraph[cfacets[k]].Size(); }
	rcfacets.SetSize0();
	merge_arrays(cfgrows, rcfacets, [&](auto i, auto j) { return i < j; });
	ncf = rcfacets.Size();
	
	FlatArray<double> bcbase(BS * ncf, lh);
	FlatMatrix<double> Bc(HB, BS * ncf, lh);
	int bccol = 0;
	for (auto j : Range(cfacets)) {
	  auto cvj = cneibs[j];
	  auto fij = int(cfacets[j]);
	  auto & fijd = ced[fij];
	  bccol = BS * find_in_sorted_array(fij, rcfacets);
	  cout << " c vol " << cvd[cv].vol << " " << 1.0/cvd[cv].vol << endl;
	  cout << " cvj " << cvj << " cfij " << fij << endl;
	  cout << " c flow " << fijd.flow << endl;
	  for (auto l : Range(BS))
	    { bcbase[bccol++] = ( (cv < cvj) ? 1.0 : -1.0) * fijd.flow(l); }
	}

	cout << " bcbase " << endl; prow(bcbase); cout << endl;
	
	/** If we have an outflow, we force 0 divergence, otherwise only constant divergence **/
	const double cvinv = (has_outflow) ? 0.0 : 1.0 / cvd[cv].vol;
	Bc = 0; // last row of Bc is kept 0 for pressure avg. constraint
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { return; }
	    auto bcrow = Bc.Row(rkvi);
	    const double fac = fvd[vi].vol * cvinv;
	    for (auto col : Range(bcbase))
	      { bcrow(col) = fac * bcbase[col]; }
	  });

	/** RHS **/
	int Hi = BS * nffi, Hf = BS * nfff, Hc = BS * ncf;
	FlatArray<int> colsi(Hi + HB, lh), colsf(Hf, lh);
	int ci = 0;
	for (auto j : Range(nffi)) {
	  int base = BS * ffiinds[j];
	  for (auto l : Range(BS))
	    { colsi[ci++] = base++; }
	}
	for (auto kv : Range(HB))
	  { colsi[ci++] = HA + kv; }
	int cf = 0;
	for (auto j : Range(nfff)) {
	  int base = BS * fffinds[j];
	  for (auto l : Range(BS))
	    { colsf[cf++] = base++; }
	}

	cout << "Hi + Hf " << Hi << " " << Hf << " " << Hi+Hf << " " << HA << endl;
	cout << "HA HB HM " << HA << " " << HB << " " << HM << endl;
	cout << "Hc " << Hc << endl;

	cout << "colsi (" << colsi.Size() << ") = "; prow(colsi); cout << endl;
	cout << "colsf (" << colsf.Size() << ") = "; prow(colsf); cout << endl;

	FlatMatrix<double> Pf(Hf, Hc, lh); Pf = 0;
	for (auto j : Range(nfff)) {
	  auto fnr = ffacets[fffinds[j]];
	  auto ris = SP.GetRowIndices(fnr);
	  auto rvs = SP.GetRowIndices(fnr);
	  for (auto j : Range(ris)) {
	    int col = BS * find_in_sorted_array(ris[j], rcfacets);
	    Pf.Rows(j*BS, (j+1)*BS).Cols(col*BS, (col+1)*BS) = rvs[j];
	  }
	}

 	cout << "Pf: " << endl << Pf << endl;

	cout << "Bc: " << endl << Bc << endl;

	/** -A_if * P_f **/
	FlatMatrix<double> rhs(Hi + HB, Hc, lh);
	rhs = -M.Rows(colsi).Cols(colsf) * Pf;

	cout << "Mif " << endl << M.Rows(colsi).Cols(colsf) << endl;
	cout << "rhs only homogen. " << endl << rhs << endl;
	rhs.Rows(Hi, rhs.Height()) += Bc;

	// rhs.Rows(0, Hi) = 0;
	// rhs.Rows(Hi, rhs.Height()) = Bc;
	// cout << "RHS without homogenization" << endl << rhs << endl;
	// rhs -= M.Rows(colsi).Cols(colsf) * Pf;

	cout << "full RHS: " << endl << rhs << endl;

	/** Lock constant pressure **/
	// auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
	// auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
	if ( lock_const_pressure ) {
	  // for (auto k : Range(1)) {
	  for (auto k : Range(HB - 1)) {
	    M(HA + HB - 1, HA + k) = 1.0;
	    M(HA + k, HA + HB - 1) = 1.0;
	  }
	}
	// Bf.Row(HB-1) = bfsum/((HB-1)*(HB-1));
	// BfT.Col(HB-1) = bfsum/((HB-1)*(HB-1));
	cout << "M mat: " << endl << M << endl;

	/** The block to invert **/
	FlatMatrix<double> Mii(Hi + HB, Hi + HB, lh);
	Mii = M.Rows(colsi).Cols(colsi);
	cout << "Mii " << endl << Mii << endl;
	CalcInverse(Mii);
	cout << "Mii-inv " << endl << Mii << endl;

	/** The final prol **/
	FlatMatrix<double> Pext(Hi + HB, Hc, lh);
	Pext = Mii * rhs;

	cout << "Pext: " << endl << Pext << endl;

	/** Write into sprol **/
	for (auto kfi : Range(nffi)) {
	    auto ff = ffacets[ffiinds[kfi]];
	    auto ris = SP.GetRowIndices(ff);
	    auto rvs = SP.GetRowValues(ff);
	    cout << "write row " << kfi << " -> " << ff << endl;
	    cout << "mat space = " << rvs.Size() << " * BS = " << rvs.Size() * BS << endl;
	    cout << "Pext width = " << Pext.Width() << endl;
	    ris = graph[ff];
	    for (auto j : Range(ris)) {
	      int col = BS * find_in_sorted_array(ris[j], rcfacets);
	      rvs[j] = Pext.Rows(BS*kfi, BS*(kfi+1)).Cols(BS*col, BS*(col+1));
	    }
	}
      } // agg_vs.Size() > 1
    } // agglomerate loop

    cout << "FINAL SPROL: " << endl;
    print_tm_spmat(cout, SP);

    cout << " DONE WITH STOKES SMOOTH PROL " << endl;
    return sprol;
  }

    /**
       Stokes prolongation works like this:
           Step  I) Find a prolongation on agglomerate facets
	   Step II) Extend prolongation to agglomerate interiors:
	               Minimize energy. (I) as BC. Additional constraint:
		          \int_a div(u) = |a|/|A| \int_A div(U)    (note: we maintain constant divergence)

       "Piecewise" Stokes Prolongation.
           Step I) Normal PW Prolongation as in standard AMG

       "Smoothed" Stokes Prol takes PW Stokes prol and then:
           Step  I) Take PW prol, then smooth on facets between agglomerates. Additional constraint: 
	              Maintain flow through facet.
    **/

  template<class TMESH, class ENERGY> shared_ptr<typename StokesAMGFactory<TMESH, ENERGY>::TSPM_TM>
  StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl (shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds,
						       shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
						       FlatArray<int> vmap, FlatArray<int> emap,
						       FlatTable<int> v_aggs)//, FlatTable<int> c2f_e)
  {

    static constexpr int BS = mat_traits<TM>::HEIGHT;

    /** fine mesh **/
    const auto & FM(*fmesh); FM.CumulateData();
    const auto & fecon(*FM.GetEdgeCM());
    auto fvd = get<0>(FM.Data())->Data();
    auto fed = get<1>(FM.Data())->Data();
    size_t FNV = FM.template GetNN<NT_VERTEX>(), FNE = FM.template GetNN<NT_EDGE>();
    auto free_fes = FM.GetFreeNodes();

    if (free_fes)
    cout << " free_fes: " << endl << *free_fes << endl;
    
    /** coarse mesh **/
    const auto & CM(*cmesh); CM.CumulateData();
    const auto & cecon(*CM.GetEdgeCM());
    auto cvd = get<0>(CM.Data())->Data();
    auto ced = get<1>(CM.Data())->Data();
    size_t CNV = CM.template GetNN<NT_VERTEX>(), CNE = CM.template GetNN<NT_EDGE>();

    cout << "fecon: " << endl << fecon << endl;
    cout << "cecon: " << endl << cecon << endl;
    
    /** prol dims **/
    size_t H = FNE, W = CNE;

    /** Count entries **/
    Array<int> perow(H); perow = 0;
    auto fedges = FM.template GetNodes<NT_EDGE>();
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      if ( free_fes && (!free_fes->Test(fenr)) ) // dirichlet
    	{ perow[fenr] = 0; }
      else {
	auto cenr = emap[fenr];
	if (cenr != -1) // edge connects two agglomerates
	  { perow[fenr] = 1; }
	else {
	  int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
	  if (cv0 == cv1) {
	    if (cv0 == -1) // I don't think this can happen - per definition emap must be -1
	      { perow[fenr] = 0; throw Exception("Weird case A!"); }
	    else // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate
	      { perow[fenr] = cecon.GetRowIndices(cv0).Size(); }
	  }
	  else { // edge between a Dirichlet BND vertex and an interior one
	    perow[fenr] = 0;
	    // throw Exception("Weird case B!");
	  }
	}
      }
    }

    /** Allocate Matrix  **/
    auto prol = make_shared<TSPM_TM> (perow, W);
    auto & P(*prol);
    const auto & const_P(*prol);
    P.AsVector() = -42;

    /** Fill facets **/
    typename ENERGY::TVD tvf, tvc;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      if (cenr != -1) {
    	int fv0 = fedge.v[0], fv1 = fedge.v[1];
    	tvf = ENERGY::CalcMPData(fvd[fv0], fvd[fv1]);
    	int cv0 = vmap[fv0], cv1 = vmap[fv1];
    	tvc = ENERGY::CalcMPData(cvd[cv0], cvd[cv1]);
    	P.GetRowIndices(fenr)[0] = cenr;
    	ENERGY::CalcQHh(tvc, tvf, P.GetRowValues(fenr)[0]);
      }
    }

    cout << "stage 1 pwp: " << endl;
    print_tm_spmat(cout << endl, P);


    /** Extend the prolongation - Solve:
    	  |A_vv B_v^T|  |u_v|     |-A_vf      0     |  |P_f|
    	  |  ------  |  | - |  =  |  -------------  |  | - | |u_c|
    	  |B_v       |  |lam|     |-B_vf  d(|v|/|A|)|  |B_A| 
     **/
    LocalHeap lh(10 * 1024 * 1024, "Jerry");
    for (auto agg_nr : Range(v_aggs)) {
      HeapReset hr(lh);
      auto agg_vs = v_aggs[agg_nr];
      auto cv = vmap[agg_vs[0]];
      if (agg_vs.Size() > 1) { // for single verts there are no interior edges
	cout << "cv is " << cv << endl;
	cout << "fill agg " << agg_nr << ", agg_vs: "; prow(agg_vs); cout << endl;
	cout << "v vols: " << endl;
	for (auto v : agg_vs)
	  { cout << fvd[v].vol << " "; }
	cout << endl;
	auto cv = vmap[agg_vs[0]];
	auto cneibs = cecon.GetRowIndices(cv);
	auto cfacets = cecon.GetRowValues(cv);
	int nfv = agg_vs.Size(), ncv = 1;     // # fine/coarse elements
	/** count fine facets **/
	int ncf = cfacets.Size();             // # coarse facets
	int nff = 0; int nffi = 0, nfff = 0;  // # fine facets (int/facet)
	auto it_f_facets = [&](auto lam) {
	  for (auto k : Range(agg_vs)) {
	    auto vk = agg_vs[k];
	    cout << k << ", vk " << vk << endl;
    	    auto vk_neibs = fecon.GetRowIndices(vk);
	    cout << "neibs "; prow(vk_neibs); cout << endl;
    	    auto vk_fs = fecon.GetRowValues(vk);
	    for (auto j : Range(vk_neibs)) {
	      auto vj = vk_neibs[j];
	      auto cvj = vmap[vj];
	      cout << j << " vj " << vj << " cvj " << cvj << endl;
	      if (cvj == cv) { // neib in same agg - interior facet!
		auto kj = find_in_sorted_array(vj, agg_vs);
		if (vj > vk) { // do not count interior facets twice!
		  lam(vk, k, vj, kj, int(vk_fs[j]));
		}
	      }
	      else // neib in different agg (or dirichlet)
		{ lam(vk, k, vj, -1, int(vk_fs[j])); }
	    }
	  }
	};
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    nff++;
	    if (kj == -1) // ex-facet
	      { nfff++; }
	    else
	      { nffi++; }
	  });
	cout << "nff/f/i " << nff << " " << nfff << " " << nffi << endl << endl;
	/** All vertices (in this connected component) are in one agglomerate - nothing to do!
	    Sometimes happens on coarsest level. **/
	if (nfff == 0)
	  { continue; }

	/** I think this can only happen for my special test case with virtual vertex aggs ! **/
	if (nffi == 0)
	  { continue; }
	/** fine facet arrays **/
	FlatArray<int> index(nff, lh), buffer(nff, lh);
	FlatArray<int> ffacets(nff, lh), ffiinds(nffi, lh), fffinds(nfff, lh);
	nff = nffi = nfff = 0;
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    index[nff] = nff;
	    if (kj == -1)
	      { fffinds[nfff++] = nff; }
	    else
	      { ffiinds[nffi++] = nff; }
	    ffacets[nff++] = eid;
	  });
	cout << "unsorted ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl;
	cout << "unsorted ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl;
	cout << "unsorted fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl;
	QuickSortI(ffacets, index);
	for (auto k : Range(nffi))
	  { ffiinds[k] = index.Pos(ffiinds[k]); }
	for (auto k : Range(nfff)) // TODO...
	  { fffinds[k] = index.Pos(fffinds[k]); }
	cout << "index " << nff << ": "; prow2(index, cout << endl); cout << endl;
	ApplyPermutation(ffacets, index);
	cout << "ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl;
	cout << "ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl;
	cout << "fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl;

	auto it_f_edges = [&](auto lam) {
	  for (auto kvi : Range(agg_vs)) {
	    auto vi = agg_vs[kvi];
    	    auto vi_neibs = fecon.GetRowIndices(vi);
    	    auto vi_fs = fecon.GetRowValues(vi);
	    const int nneibs = vi_fs.Size();
	    for (auto lk : Range(nneibs)) {
	      auto vk = vi_neibs[lk];
	      auto fik = int(vi_fs[lk]);
	      auto kfik = find_in_sorted_array(fik, ffacets);
	      for (auto lj : Range(lk)) {
		auto vj = vi_neibs[lj];
		auto fij = int(vi_fs[lj]);
		auto kfij = find_in_sorted_array(fij, ffacets);
		lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
	      }
	    }
	  }
	};

	int HA = nff * BS, HB = 0;

	auto it_real_vs = [&](auto lam) {
	  int rkvi = 0;
	  for (auto kvi : Range(agg_vs)) {
	    auto vi = agg_vs[kvi];
	    bool is_real = fvd[vi].vol > 0;
	    /** This removes div-constraint on els that have a dirichlet facet **/
	    // for (auto vj : vi_neibs)
	      // if (vmap[vj] == -1)
		// { is_real = false; }
	    lam(is_real, vi, kvi, rkvi);
	    if (is_real)
	      { rkvi++; }
	  }
	};
	bool has_outflow = false;
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { has_outflow = true; }
	    else
	      { HB++; }
	  });
	/** If there is more than one element and no outflow facet, we need to lock constant pressure.
	    If we have only one "real" element, there should always be an outflow, I believe. **/
	bool lock_const_pressure = (!has_outflow) && (HB > 1);
	if ( lock_const_pressure )
	  { HB++; }

	int HM = HA + HB;

	cout << "H A/B/M = " << HA << " / " << HB << " / " << HM << endl;

	FlatMatrix<double> M(HM, HM, lh); M = 0;
	auto A = M.Rows(0, HA).Cols(0, HA);

	/** A block **/
	// FlatMatrix<double> A(nff * BS, nff * BS, lh); A = 0;
	auto a = M.Rows(0, HA).Cols(0, HA);
	FlatMatrix<TM> eblock(2,2,lh);
	it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
	    /** Only an energy-contrib if neither vertex is Dirichlet or grounded. Otherwise kernel vectors are disturbed. **/
	    if ( (vmap[vj] != -1) && (vmap[vk] != -1) ) {
	      ENERGY::CalcRMBlock(eblock, fvd[vi], fvd[vj], fvd[vk], fed[fij], (vi > vj), fed[fik], (vi > vk));
	      // cout << "block verts " << vi << "-" << vj << "-" << vk << endl;
	      // cout << "block faces " << fij << "-" << fik << endl;
	      // cout << "block loc faces " << kfij << "-" << kfik << endl;
	      // cout << "block:" << endl; print_tm_mat(cout, eblock); cout << endl;
	      Iterate<2>([&](auto i) {
		  int osi = BS * (i.value == 0 ? kfij : kfik);
		  Iterate<2>([&](auto j) {
		      int osj = BS * (j.value == 0 ? kfij : kfik);
		      A.Rows(osi, osi + BS).Cols(osj, osj + BS) += eblock(i.value, j.value);
		    });
		});
	    }
	  });

	cout << "A block: " << endl << A << endl;

	/** (fine) B blocks **/
	auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
	auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
	double bfsum = 0;
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { return; }
	    auto BfRow = Bf.Row(rkvi);
	    auto vi_neibs = fecon.GetRowIndices(vi);
	    auto vi_fs = fecon.GetRowValues(vi);
	    cout << "calc b for " << kvi << ", vi " << vi << ", rkvi " << rkvi << endl;
	    cout << "neibs "; prow(vi_neibs); cout << endl;
	    cout << "edgs "; prow(vi_fs); cout << endl;
	    const double fac = 1.0/fvd[vi].vol;
	    for (auto j : Range(vi_fs)) {
	      auto vj = vi_neibs[j];
	      auto fij = int(vi_fs[j]);
	      auto kfij = find_in_sorted_array(fij, ffacets);
	      auto & fijd = fed[fij];
	      int col = BS * kfij;
	      cout << "j " << j << ", vj " << vj << ", fij " << fij << ", kfij " << kfij << ", col " << col << ", fac " << fac << endl;
	      cout << "vol volinv: " << fvd[vi].vol << " " << fac << endl;
	      cout << "flow: " << fijd.flow << endl;
	      for (auto l : Range(BS)) {
		BfRow(col++) = ( (vi < vj) ? 1.0 : -1.0) * fijd.flow(l);
		// BfRow(col++) = ( (vi < vj) ? fac : -fac) * fijd.flow(l);
		bfsum += fac * abs(fijd.flow(l));
	      }
	    }
	  });
	BfT = Trans(Bf);

	/** (coarse) B **/
	FlatArray<double> bcbase(BS * ncf, lh);
	FlatMatrix<double> Bc(HB, BS * ncf, lh);
	auto cv_neibs = cecon.GetRowIndices(cv);
	auto cv_fs = cecon.GetRowValues(cv);
	int bccol = 0;
	for (auto j : Range(cv_fs)) {
	  auto cvj = cv_neibs[j];
	  auto fij = int(cv_fs[j]);
	  auto & fijd = ced[fij];
	  cout << " c vol " << cvd[cv].vol << " " << 1.0/cvd[cv].vol << endl;
	  cout << " cvj " << cvj << " cfij " << fij << endl;
	  cout << " c flow " << fijd.flow << endl;
	  for (auto l : Range(BS))
	    { bcbase[bccol++] = ( (cv < cvj) ? 1.0 : -1.0) * fijd.flow(l); }
	}

	cout << " bcbase " << endl; prow(bcbase); cout << endl;
	
	/** If we have an outflow, we force 0 divergence, otherwise only constant divergence **/
	const double cvinv = (has_outflow) ? 0.0 : 1.0 / cvd[cv].vol;
	Bc = 0; // last row of Bc is kept 0 for pressure avg. constraint
	it_real_vs([&](auto is_real, auto vi, auto kvi, auto rkvi) {
	    if (!is_real)
	      { return; }
	    auto bcrow = Bc.Row(rkvi);
	    for (auto col : Range(bcbase))
	      { bcrow(col) = cvinv * fvd[vi].vol * bcbase[col]; }
	    // for (auto col : Range(bcbase))
	      // { bcrow(col) = cvinv * bcbase[col]; }
	  });

	/** RHS **/
	int Hi = BS * nffi, Hf = BS * nfff, Hc = BS * ncf;
	FlatArray<int> colsi(Hi + HB, lh), colsf(Hf, lh);
	int ci = 0;
	for (auto j : Range(nffi)) {
	  int base = BS * ffiinds[j];
	  for (auto l : Range(BS))
	    { colsi[ci++] = base++; }
	}
	for (auto kv : Range(HB))
	  { colsi[ci++] = HA + kv; }
	int cf = 0;
	for (auto j : Range(nfff)) {
	  int base = BS * fffinds[j];
	  for (auto l : Range(BS))
	    { colsf[cf++] = base++; }
	}

	cout << "Hi + Hf " << Hi << " " << Hf << " " << Hi+Hf << " " << HA << endl;
	cout << "HA HB HM " << HA << " " << HB << " " << HM << endl;
	cout << "Hc " << Hc << endl;

	cout << "colsi (" << colsi.Size() << ") = "; prow(colsi); cout << endl;
	cout << "colsf (" << colsf.Size() << ") = "; prow(colsf); cout << endl;

	FlatMatrix<double> Pf(Hf, Hc, lh); Pf = 0;
	for (auto j : Range(nfff)) {
	  auto fnr = ffacets[fffinds[j]];
	  for (auto l : Range(cfacets))
	    { Pf.Rows(j*BS, (j+1)*BS).Cols(l*BS, (l+1)*BS) = const_P(fnr, cfacets[l]); }
	}

 	cout << "Pf: " << endl << Pf << endl;

	cout << "Bc: " << endl << Bc << endl;

	/** -A_if * P_f **/
	FlatMatrix<double> rhs(Hi + HB, Hc, lh);
	rhs = -M.Rows(colsi).Cols(colsf) * Pf;

	cout << "Mif " << endl << M.Rows(colsi).Cols(colsf) << endl;
	cout << "rhs only homogen. " << endl << rhs << endl;
	rhs.Rows(Hi, rhs.Height()) += Bc;

	// rhs.Rows(0, Hi) = 0;
	// rhs.Rows(Hi, rhs.Height()) = Bc;
	// cout << "RHS without homogenization" << endl << rhs << endl;
	// rhs -= M.Rows(colsi).Cols(colsf) * Pf;

	cout << "full RHS: " << endl << rhs << endl;

	/** Lock constant pressure **/
	// auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
	// auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
	if ( lock_const_pressure ) {
	  // for (auto k : Range(1)) {
	  for (auto k : Range(HB - 1)) {
	    M(HA + HB - 1, HA + k) = 1.0;
	    M(HA + k, HA + HB - 1) = 1.0;
	  }
	}
	// Bf.Row(HB-1) = bfsum/((HB-1)*(HB-1));
	// BfT.Col(HB-1) = bfsum/((HB-1)*(HB-1));
	cout << "M mat: " << endl << M << endl;


	{
	  FlatMatrix<double> Aii(Hi, Hi, lh), S(HB, HB, lh);
	  auto iii = colsi.Part(0, Hi);
	  auto bbb = colsi.Part(Hi);
	  Aii = M.Rows(iii).Cols(iii);
	  cout << "Aii " << endl << Aii << endl;
	  CalcInverse(Aii);
	  cout << "Aii inv " << endl << Aii << endl;
	  S = M.Rows(bbb).Cols(bbb);
	  S -= M.Rows(bbb).Cols(iii) * Aii * M.Rows(iii).Cols(bbb);
	  cout << "S: " << endl << S << endl;
	  CalcInverse(S);
	  cout << "Sinv: " << endl << S << endl;
	}

	/** The block to invert **/
	FlatMatrix<double> Mii(Hi + HB, Hi + HB, lh);
	Mii = M.Rows(colsi).Cols(colsi);
	cout << "Mii " << endl << Mii << endl;
	CalcInverse(Mii);
	cout << "Mii-inv " << endl << Mii << endl;

	/** The final prol **/
	FlatMatrix<double> Pext(Hi + HB, Hc, lh);
	Pext = Mii * rhs;

	cout << "Pext: " << endl << Pext << endl;

	/** Write into sprol **/
	for (auto kfi : Range(nffi)) {
	    auto ff = ffacets[ffiinds[kfi]];
	    auto ris = P.GetRowIndices(ff);
	    auto rvs = P.GetRowValues(ff);
	    cout << "write row " << kfi << " -> " << ff << endl;
	    cout << "mat space = " << rvs.Size() << " * BS = " << rvs.Size() * BS << endl;
	    cout << "Pext width = " << Pext.Width() << endl;
	    for (auto j : Range(ncf)) // TODO??
	      { ris[j] = cfacets[j]; }
	    QuickSort(ris);
	    for (auto j : Range(ncf)) {
	      // rvs[j] = 0.0;
	      rvs[ris.Pos(cfacets[j])] = Pext.Rows(BS*kfi, BS*(kfi+1)).Cols(BS*j, BS*(j+1));
	    }
	}

      } // agg_vs.Size() > 1
    } // agglomerate loop

    cout << " Final Stokes PWP:" << endl;
    print_tm_spmat(cout << endl, P);

    return prol;
  } // StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl
										   

  /** END StokesAMGFactory **/

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP
#endif // STOKES
