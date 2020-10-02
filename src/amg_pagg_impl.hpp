#ifndef FILE_AMG_PAGG_IMPL_HPP
#define FILE_AMG_PAGG_IMPL_HPP

namespace amg
{

  /** LocCoarseMap **/

  class LocCoarseMap : public BaseCoarseMap
  {
    friend class SPAgglomerator;
  public:
    LocCoarseMap (shared_ptr<TopologicMesh> mesh)
      : BaseCoarseMap(mesh)
    {
      const auto & M = *mesh;
      Iterate<4>([&](auto k) {
	  NN[k] = M.template GetNN<NODE_TYPE(k)>();
	  node_maps[k].SetSize(NN[k]);
	});
    }

    FlatArray<int> GetV2EQ () const { return cv_to_eqc; }
    void SetV2EQ (Array<int> && _cv_to_eqc) { cv_to_eqc = _cv_to_eqc; }

    void FinishUp ()
    {
      /** mapped_mesh **/
      auto eqc_h = mesh->GetQCHierarchy();
      auto cmesh = make_shared<TopologicMesh>(eqc_h, GetMappedNN<NT_VERTEX>(), GetMappedNN<NT_EDGE>(), 0, 0);
      // NOTE: verts are never set, they are unnecessary because we dont build coarse BlockTM!
      /** crs vertex eqcs **/
      auto pbtm = dynamic_pointer_cast<BlockTM>(mesh);
      if (pbtm == nullptr)
	{ throw Exception("No BTM in LocCoarseMap in PAGG!"); }
      const auto & btm = *pbtm;
      cv_to_eqc.SetSize(GetMappedNN<NT_VERTEX>());
      auto c2fv = GetMapC2F<NT_VERTEX>();
      for(auto cenr : Range(cv_to_eqc)) {
	auto fvs = c2fv[cenr];
	if (fvs.Size() == 1)
	  { cv_to_eqc = btm.btm.GetEQCOfNode<NT_VERTEX>(fvs[0]); }
	else {
	  // NOTE: this only works as long as i only go up/down eqc-hierararchy
	  int eqa = btm.GetEQCOfNode<NT_VERTEX>(fvs[0]), eqb = btm.GetEQCOfNode<NT_VERTEX>(fvs[1]);
	  cv_to_eqc = eqc_h->IsLEQ(eqa, eb) ? eqa : eqb;
	}
      }
      /** crs edge connectivity and edges **/
      const auto & fecon = *mesh->GetEdgeCM();
      TableCreator<int> ccg(GetMappedNN<NT_VERTEX>());
      Array<int> neibs;
      for (; !ccg.Done(); ccg++)
	for (auto cvnr : Range(c2fv)) {
	  for (auto fvnr : c2fv[cvnr])
	    for (auto vneib : fecon.GetRowIndices(fvnr))
	      if (vmap[vneib] != -1)
		{ insert_into_sorted_array(vneib, neibs); }
	  ccg.Add(cvnr, vneibs);
	}
      auto graph = ccg.MoveTable();
      auto pcecon = make_shared<SparseMatrix<double>>(grap, GetMappedNN<NT_VERTEX>());
      const auto & cecon = *pcecon;
      size_t cnt = 0;
      auto & cedeges = cmesh->edges;
      cedges.SetSize(GetMappedNN<NT_EDGE>());
      for (auto k : Range(pcecon)) {
	auto ris = cecon.GetRowIndices(k);
	auto rvs = cecon.GetRowValues(k);
	for (auto l : Range(ris)) {
	  if (ris[l] > k)
	    { rvs[l] = cnt++; cedges[int(rvs[l])].v = { k, ris[l] }; cedges[int(rvs[l])].id = int(rvs[l]); }
	  else
	    { rvs[l] = cecon(ris[l], k); }
	}
      }
      cmesh->econ = pcecon;
      /** edge map **/
      auto fedges = mesh->template GetNodes<NT_EDGE>();
      auto & emap = node_maps[NT_EDGE];
      for (auto fenr : Range(emap)) {
	auto & edge = fedges[fenr];
	int cv0 = vmap[edge.v[0]], cv1 = vmap[edge.v[1]];
	if ( (cv0!=-1) && (cv1!=-1) && (cv0 != cv1) )
	  { emap[fenr] = cecon(cv0, cv1); }
	else
	  { emap[fenr] = -1; }
      }
      this->mapped_mesh = cmesh;
    } // FinishUp

    virtual shared_ptr<BaseCoarseMap> Concatenate (shared_ptr<BaseCoarseMap> right_map) override
    {
      /** with concatenated vertex/edge map **/
      auto concmap = BaseCoarseMap::Concatenate(right_map);
      /** concatenated aggs! **/
      int NVF = GetNN<NT_VERTEX>(), NVC = right_map->GetMappedNN<NT_VERTEX>();
      FlatTAble<int> aggs1 = GetMapC2F<NT_VERTEX>(), aggs2 = GetMapC2F<NT_VERTEX>();
      TableCreator<int> ct(NCV);
      for (; !ct.Done(); ct++)
	for (auto k : Range(NCV))
	  for (auto v : aggs2[k])
	    { ct.Add(k, aggs1[v]); }
      concmap->rev_node_maps[NT_VERTEX] = ct.MoveTable();
      /** coarse vertex->eqc mapping is the right one! **/
      concmap->cv_to_eqc = move(right_map->cv_to_eqc);
      return concmap;
    } // Concatenate(map)

    void Concatenate (size_t NCV, FlatArray<int> rvmap)
    {
      /** no mesh on coarse level **/
      this->mapped_mesh = nullptr;
      /** no edges on coarse level **/
      mapped_NN[NT_VERTEX] = NCV;
      mapped_NN[NT_EDGE] = 0;
      /** concatenate vertex map **/
      auto & vmap = node_maps[NT_VERTEX];
      for (auto & val : Range(vmap))
	{ val = (val == -1) ? val : rvmap[val]; }
      /** set up reverse vertex map **/
      auto & aggs = rev_node_maps[NT_VERTEX];
      TableCreator<int> ct(NCV);
      for (; !ct.Done(); ct++)
	for (auto k : Range(aggs))
	  if (rvmap[k] != 1)
	    { ct.Add(rvmap[k], aggs[k]); }
      rev_node_maps[NT_VERTEX] = ct.MoveTable();
      /** no vertex->eqc mapping on coarse level **/
      cv_to_eqc.SetSize(NCV); cv_to_eqc = -1; 
    } // Concatenate(array)

    INLINE int CV2EQ (int v) const { return cv_to_eqc[v]; }

  protected:

    Array<int> cv_to_eqc; /** (coarse) vert->eqc mapping **/
  }; // class LocCoarseMap

  /** END LocCoarseMap **/


  /** SPAgglomerator **/

  template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
  INLINE void Agglomerator<ENERGY, TMESH, ROBUST> :: GetEdgeData (FlatArray<TM> full_data, Array<TMU> data)
  {
    if constexpr(std::is_same<TMU, TM>::value)
      { data.Assign(full_data); }
    else {
      data.SetSize(full_data.Size());
      for (auto k : Range(traces))
	{ traces[k] = ENERGY::GetApproxWeight(full_edata[k]); }
    }
  } // SPAgglomerator::GetEdgeData


  template<class ENERGY, class TMESH, bool ROBUST>
  void SPAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    if constexpr (ROBUST) {
	if (settings.robust) /** cheap, but not robust for some corner cases **/
	  { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
	else /** (much) more expensive, but also more robust **/
	  { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
      }
    else // do not even compile the robust version - saves a lot of ti
      { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
  } // SPAgglomerator::FormAgglomerates


  template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
  INLINE void Agglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    static_assert ( (std::is_same<TMU, TM>::value || std::is_same<TMU, double>::value), "Only 2 options make sense!");

    const int num_rounds = options->num_rounds;

    constexpr int BS = mat_traits<TM>::HEIGHT;
    constexpr int BSU = mat_traits<TMU>::HEIGHT;
    constexpr bool robust = (BS == BSU) && (BSU > 1);

    const bool allrobust = robust && options->allrobust;
    const bool checkbigsoc = options->checkbigsoc;

    auto tm_mesh = dynamic_pointer_cast<TMESH>(mesh);
    const auto & M = *tm_mesh; M.CumulateData();
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    const auto NV = M.template GetNN<NT_VERTEX>();
    const auto NE = M.template GetNN<NT_EDGE>();
    const auto & econ = *M.GetEdgeCM();
    
    
    const Options::CW_TYPE cw_type = options->cw_type;
    const AVG_TYPE minmax_avg_scal = options->minmax_avg;
    const bool minmax_avg_mat_harm = options->minmax_avg_mat == HARM;
    

    shared_ptr<LocCoarseMap> conclocmap;
    
    FlatArray<TVD> base_vdata = get<0>(tm_mesh->Data())->Data();
    FlatArray<TM> base_edata_full = get<1>(tm_mesh->Data())->Data();
    Array<TMU> base_edata; GetEdgeData<TMU>(base_edata_full, base_edata);
    Array<TM> base_diags(M.template GetNN<NT_VERTEX>());
    M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
	ModQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
	const auto & em = edata[edge.id];
	AddQtMQ(1.0, base_diags[edge.v[0]], Qij, em);
	AddQtMQ(1.0, base_diags[edge.v[1]], Qji, em);
      }, true); // only master, we cumulate this afterwards
    M.template AllreduceNodalData<NT_VERTEX>(repl_diag, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    Array<TVD> cvdata;
    Array<TMU> cedata;
    Array<TM> cedata_full, cdiags;
    FlatArray<TVD> fvdata; fvdata.Assign(base_vdata);
    FlatArray<TMU> fedata; fedata.Assign(base_edata);
    FlatArray<TM> fedata_full; fedata_full.Assign(base_edata_full);
    FlatArray<TM> fdiags; fdiags.Assign(base_diags);

    LocalHeap lh(20971520, "cthulu"); // 20 MB

    auto calc_avg_scal = [&](double mtra, double mtrb, AVG_TYPE avg) LAMBDA_INLINE {
      switch(avg) {
      case(MIN): { return min(mtra, mtrb); break; }
      case(GEOM): { return sqrt(mtra * mtrb); break; }
      case(HARM): { return 2 * (mtra * mtrb) / (mtra + mtrb); break; }
      case(ALG): { return (mtra + mtrb) / 2; break; }
      case(MAX): { return max(mtra, mtrb); break; }
      default: { return -1; }
      }
    };

    auto calc_soc_scal = [&](double da, double db, double emt) LAMBDA_INLINE {
      switch(cw_type) {
      case(HARMONIC) : { return emt / calc_avg(da, db, HARM); }
      case(GEOMETRIC) : { return emt / calc_avg(da, db, GEOM); }
      case(MINMAX) : { return emt / calc_avg(da, db, minmax_avg); }
      }
    };

    /** Initial SOC to pick merge candidate. Can be EVP or scalar based. **/
    TMU da, db, dedge;
    auto calc_soc_candidate = [&](auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
      if (allrobust) // is actually robust && alrobust
	{ return calc_soc_check1(); }
      /** calc_trace does nothing for scalar case **/
      dedge = calc_trace(fedata[int(fecon(vi, vj))]);
      switch(cw_type) {
      case(HARMONIC) : { return dedge / calc_avg(calc_trace(fdiags[vi]), calc_trace(fdiags[vj]), HARM); }
      case(GEOMETRIC) : { return dedge / calc_avg(calc_trace(fdiags[vi]), calc_trace(fdiags[vj]), GEOM); }
      case(MINMAX) : {
	da = db = 0;
	for (auto eid : fecon.GetRowValues(vi))
	  { da = max2(da, calc_trace(fedata[int(eid)])); }
	for (auto eid : fecon.GetRowValues(vj))
	  { da = max2(da, calc_trace(fedata[int(eid)])); }
	return dedge / calc_avg(da, db, minmax_avg_scal);
      }
      }
    };

    /** EVP based pairwise SOC. **/
    TM dma, dmb, dmedge, Q;
    auto calc_soc_check1 = [&](auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
      if constexpr(!robust) // dummy - should never be called anyways!
	{ return 1.0; }
      else {
	/** Transform diagonal matrices **/
	TVD H_data = ENERGY::CalcMPData(fvdata[vi], fvdata[vj]);
	ENERGY::ModQHh(H_data, fvdata[vi], Q);
	ENERGY::SetQtMQ(dma, Q, fdiags[vi]);
	ENERGY::ModQHh(H_data, vdata[bj], Q);
	ENERGY::SetQtMQ(dmb, Q, fdiags[vj]);
	/** TODO:: neib bonus **/
	dmedge = fedata_full[int(fecon(vi,vj))];
	double soc = 0;
	switch(cw_type) {
	case(HARMONIC) : { soc = MIN_EV_HARM2(dma, dmb, dmedge); break; }
	case(GEOMETRIC) : { soc = MIN_EV_FG2(dma, dmb, dmedge); break; }
	case(MINMAX) : {
	  double mtra = 0, mtrb = 0;
	  for (auto eid : fecon.GetRowValues(vi))
	    { mtra = max2(mtra, calc_trace(fedata[int(eid)])); }
	  for (auto eid : fecon.GetRowValues(vj))
	    { mtrb = max2(mtrb, calc_trace(fedata[int(eid)])); }
	  double etrace = calc_trace(dmedge);
	  double soc = etrace / calc_avg(mtra, mtrb, minmax_avg_scal);
	  if (soc > MIN_ECW) {
	    dma /= calc_trace(dma);
	    dmb /= calc_trace(dmb);
	    dmedge /= etrace;
	    if (minmax_avg_mat_harm)
	      { soc = min2(soc, MIN_EV_HARM2(dma, dmb, dmedge)); }
	    else
	      { soc = min2(soc, MIN_EV_FG2(dma, dmb, dmedge)); }
	  }
	  break;
	}
	}
	return soc;
    };

    /** SOC for pair of agglomerates w.r.t original matrix **/
    auto calc_soc_check2 = [&](auto memsi, auto memsj, const auto & fecon) LAMBDA_INLINE { };

    /** Finds a neighbor to merge vertex v with. Returns -1 if no suitable ones found **/
    auto find_neib = [&](auto v, const auto & fecon, auto get_mems, bool round_cbs) LAMBDA_INLINE {
      HeapReset hr(lh);
      /** SOC for all neibs **/
      double max_soc = 0, msn = -1;
      FlatArray<INT<2,double>> socs(neibs.Size(), lh);
      for (auto j : Range(neibs))
	{ socs[j] = INT<2, double>(neibs[j], calc_soc_candidate(v, neibs[j], fecon)); }
      QuickSort(socs, [&](const auto & a, const auto & b) LAMBDA_INLINE { return a > b; });
      int candidate = (socs[0][1] > MIN_ECW) ? int(socs[0][0]) : -1;
      if constexpr(robust)
      {
	/** check candidate **/
	for (int j = 1; j < socs.Size(); j++) {
	  if (socs[j][1] < MIN_ECW)
	    { candidate = -1; break; }
	  double stabsoc = socs[j][0];
	  if (robust && (!allrobust) ) /** small EVP soc **/
	    { stabsoc = calc_soc_check1(v, neibs[j], fecon); }
	  if (round_cbs && (stabsoc > MIN_ECW)) /** big EVP soc **/
	    { stabsoc = calc_soc_check2(v, neibs[j], fecon, get_mems); }
	  if (stabsoc > MIN_ECW) /** this neib has strong stable connection **/
	    { candidate = int(socs[j][0]); break; }
	}
      } // robust
      return candidate;
    };

    /** Iterate through unhandled vertices and pair them up **/
    auto pair_vertices = [&](FlatArray<int> vmap, size_t & NCV,
			     int num_verts, auto get_vert, const auto & fecon,
			     BitArray & handled,
			     FlatArray<TMU> diags, FlatArray<TMU> emats,
			     auto get_mems, auto allowed, auto set_pair,
			     bool round_cbs
			     ) LAMBDA_INLINE {
      for (auto k : Range(num_verts)) {
	auto vnr = get_vert(k);
	if (!handled.Test(vnr)) { // try to find a neib
	  int neib = find_neib(vnr, fecon, get_mems, round_cbs);
	  if (neib != -1) {
	    vmap[neib] = NCV;
	    handled.SetBit(neib);
	  }
	  vmap[vnr] = NCV++;
	  handled.SetBit(vnr);
	  set_pair(vnr, neib);
	}
      }
    };

    // pair vertices
    for (int round : Range(num_rounds)) {
      /** Set up a local map to represent this round of pairing vertices **/
      auto locmap = make_shared<LocCoarseMap>((round == 0) ? mesh : conclocmap->GetMappedMesh());
      auto vmap = conclocmap->template GetMap<NT_VERTEX>();
      const TopologicMesh & fmesh = locamap->GetMesh();
      const SparseMatrix<double> & fecon = *fmesh.GetEdgeCM();
      
      fvdata.Assign(cvdata);
      fedata.Assign(fedata);
      fedata_full.Assign(cedata_full);
      fdiags.Assign(cdiags);

      BitArray handled(NV); handled.Clear()
      size_t NCV = 0, NCE = 0;

      if (round == 0) {
	/** CMK ordering **/
	Array<int> cmk;
	CalcCMK(cmk, econ);

	/** Fixed Aggs - set verts to diri, handle afterwards **/

	/** dirichlet vertices **/

	/** collapsed vertices **/

	/** Find pairs for vertices **/
	pair_vertices(vmap, NCV, cmk.Size(), [&](auto k) { return cmk[k]; }, fecon, handled, diags, emats,
		      [&](auto v) { return v; }, // get_mems
		      [&](auto vi, auto vi) {
			int eqi = M.template GetEQCOfNode<NT_VERTEX>(vi), eqj = M.template GetEQCOfNode<NT_VERTEX>(vi);
			return eqc_h.IsLEQ(eqi, eqj) || eqc_h.IsLEQ(eqj, eqi);
		      }, // allowed
		      false); // no big soc necessary
      }
      else {
	/** Find pairs for vertices **/
	auto c2fv = conclocmap->template GetMapC2F<NT_VERTEX>();
	pair_vertices(vmap, NCV,
		      vmap.Size(), [&](auto i) LAMBDA_INLINE { return i; }, // no CMK on later rounds!
		      ECON, handled,
		      DIAGS, EMATS,
		      [&](auto v) LAMBDA_INLINE { return c2f[v]; }, // get_mems
		      [&](auto vi, auto vj) LAMBDA_INLINE { return ; }, // allowed
		      checkbigsoc); // check big soc if turned on
      }

      if (round < num_rounds - 1) { /** proper concatenated map, with coarse mesh **/
	/** Build edge map, C2F vertex map, edge connectivity and coarse mesh **/
	locmap->FinishUp();
	/** Coarse vertex data **/
	Array<TVD> ccvdata(NCV);
	auto c2fv = locmap->template GetMapC2F<NT_VERTEX>();
	for (auto cvnr : Range(ccvdata)) {
	  auto fvs = c2fv[cvnr];
	  if (fvs.Size() == 1)
	    { c2fv[cvnr] = fvdata[fvs[0]]; }
	  else
	    { c2fv[cvnr] = ENERGY::CalcMPData(fvdata[fvs[0]], fvdata[fvs[1]]); }
	}
	/** Coarse edge data **/
	Array<TM> ccedata_full(NCE); cedata_full = 0;
	auto fedges = locmap->GetMesh()->GetNodes<NT_EDGE>();
	auto cedges = locmap->GetMappedMesh()->GetNodes<NT_EDGE>();
	auto emap = locmap->template GetMap<NT_EDGE>();
	TM Q; SetIdentity(Q);
	TVD femp, cemp;
	for (auto fenr : Range(fedges)) {
	  auto & fedge = fedges[fenr];
	  auto cenr = emap[fenr];
	  if (cenr != -1) {
	    auto & cedge = cedges[cenr];
	    femp = ENERGY::CalcMPData(fvdata[fenr[0]], fvdata[fenr[1]]);
	    cemp = ENERGY::CalcMPData(ccvdata[cenr[0]], ccvdata[cenr[1]]);
	    ENERGY::ModQHh(cemp, femp, Q);
	    ENERGY::AddQtMQ(1.0, ccedata_full[cenr], Q, fedata[fenr]);
	  }
	}
	/** Coarse diags, I have to do this here because off-proc entries.
	    Maps are only local on master, so cannot cumulate on coarse level **/
	Array<TM> ccdiags(NCV);
	TM Qij, Qji; SetIdentity(Qij); SetIdentity(Qji);
	for (auto cvnr : Range(ccvdata)) {
	  auto fvs = c2fv[cvnr];
	  if (fvs.Size() == 1)
	    { cdiags[cvnr] = fdiags[fvs[0]]; }
	  else { /** sum up diags, remove contribution of connecting edge **/
	    ENERGY::ModQs(fvdata[fvs[0]], fvdata[fvs[1]], Qij, Qji);
	    cdiags[cvnr] = ENERGY::CalcQtMQ(Qji, fdiags[fvs[0]]);
	    ENERGY::AddQtMQ(1.0, cdiags[cvnr], Qij, fdiags[fvs[1]]);
	    /** note: this should be fine, on coarse level, at least one vert is already in an agg,
		or it would already have been paired in first round **/
	    double fac = ( (round == 0) && use_hack_stab ) ? -1.5 : -2.0;
	    /** note: coarse vert is exactly ad edge-midpoint **/
	    cdiags[cvnr] -= fac * fedata_full[int(fecon(fvs[0], fvs[1]))];
	  }
	}
	/** Concatenate local maps **/
	conclocmap = (round == 0) ? locmap : conclocmap->Concatenate(locmap);

	fvdata.Assign(0, lh); cvdata = move(ccvdata);
	fedata_full.Assign(0, lh); cedata_full = move(ccedata_full);
	fedata.Assign(0, lh); GetEdgeData<TMU>(ccedata_full, cedata);
	fdiags.Assign(0, lh); cdiags = move(ccdiags);
      }
      else {
	/** Only concatenate vertex map, no coarse mesh **/
	conclocmap->Concatenate(vmap);
      }

    } // round-loop

    /** Build final aggregates **/
    size_t n_aggs_p = conclocmap->template GetMappedNN<NT_VERTEX>(), n_aggs_f = fixed_aggs.Size();
    size_t n_aggs_tot = n_aggs_p + n_aggs_f;
    agglomerates.SetSize(n_aggs_tot);
    v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;
    auto set_agg = [&](auto agg_nr, auto vs) {
      auto & agg = agglomerates[id];
      agg.id = agg_nr;
      int ctr_eqc = M.template GetEQCOfNode<NT_VERTEX>(vs[0]), v_eqc = -1;
      agg.ctr = vs[0]; // TODO: somehow mark agg ctrs
      agg.mems.SetSize(vs.Size());
      for (auto l : Range(vs)) {
	v_eqc = M.template GetEQCOfNode<NT_VERTEX>(vs[l]);
	if ( (v_eqc != 0) && (ctr_eqc != v_eqc) && (eqc_h->IsLEQ( v_eqc, ctr_eqc) ) )
	  { agg.ctr = vs[l]; ctr_eqc = v_eqc; }
	agg.mems[l] = vs[l];
	v_to_agg[vs[l]] = agg_nr;
      }
    };
    /** aggs from pairing **/
    auto c2fv = conclocmap->template GetMapC2F<NT_VERTEX>();
    for (auto agg_nr : Range(n_aggs_p)) {
      auto aggvs = c2fv[agg_nr];
      QuickSort(aggvs);
      set_agg(agg_nr, aggvs);
    }
    /** pre-determined fixed aggs **/
    // TODO: should I filter fixed_aggs by master here, or rely on this being OK from outside?
    for (auto k : Range(fixed_aggs))
      { set_agg(n_aggs_p + k, fixed_aggs[k], fixed_aggs[k][0]); }

  } // SPAgglomerator::FormAgglomerates_impl

  /** END SPAgglomerator **/


} // namespace amg
