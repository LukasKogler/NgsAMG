#ifndef FILE_AMG_SPWAGG_IMPL_HPP
#define FILE_AMG_SPWAGG_IMPL_HPP

#ifdef SPWAGG

namespace amg
{


  /** LocCoarseMap **/

  class LocCoarseMap : public BaseCoarseMap
  {
    template<class ATENERGY, class ATMESH, bool AROBUST> friend class SPWAgglomerator;
  public:
    LocCoarseMap (shared_ptr<TopologicMesh> mesh, shared_ptr<TopologicMesh> mapped_mesh = nullptr)
      : BaseCoarseMap(mesh, mapped_mesh)
    {
      const auto & M = *mesh;
      Iterate<4>([&](auto k) {
	  NN[k] = M.template GetNN<NODE_TYPE(k.value)>();
	  node_maps[k].SetSize(NN[k]);
	});
    }

    FlatArray<int> GetV2EQ () const { return cv_to_eqc; }
    void SetV2EQ (Array<int> && _cv_to_eqc) { cv_to_eqc = _cv_to_eqc; }

    void FinishUp ()
    {
      static Timer t("LocCoarseMap::FinishUp"); RegionTimer rt(t);
      /** mapped_mesh **/
      auto eqc_h = mesh->GetEQCHierarchy();
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
	  { cv_to_eqc = btm.GetEqcOfNode<NT_VERTEX>(fvs[0]); }
	else {
	  // NOTE: this only works as long as i only go up/down eqc-hierararchy
	  int eqa = btm.GetEqcOfNode<NT_VERTEX>(fvs[0]), eqb = btm.GetEqcOfNode<NT_VERTEX>(fvs[1]);
	  cv_to_eqc = eqc_h->IsLEQ(eqa, eqb) ? eqb : eqa;
	}
      }
      /** crs edge connectivity and edges **/
      const auto & fecon = *mesh->GetEdgeCM();
      auto vmap = GetMap<NT_VERTEX>();
      TableCreator<int> ccg(GetMappedNN<NT_VERTEX>());
      Array<int> vneibs;
      for (; !ccg.Done(); ccg++)
	for (auto cvnr : Range(c2fv)) {
	  for (auto fvnr : c2fv[cvnr])
	    for (auto vneib : fecon.GetRowIndices(fvnr))
	      if (vmap[vneib] != -1)
		{ insert_into_sorted_array(vneib, vneibs); }
	  ccg.Add(cvnr, vneibs);
	}
      auto graph = ccg.MoveTable();
      Array<int> perow(graph.Size());
      for (auto k : Range(perow))
	{ perow[k] = graph[k].Size(); }
      auto pcecon = make_shared<SparseMatrix<double>>(perow, GetMappedNN<NT_VERTEX>());
      const auto & cecon = *pcecon;
      size_t cnt = 0;
      auto & cedges = cmesh->edges;
      cedges.SetSize(GetMappedNN<NT_EDGE>());
      for (int k : Range(cecon)) {
	auto ris = cecon.GetRowIndices(k);
	ris = graph[k];
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

    virtual shared_ptr<LocCoarseMap> ConcatenateLCM (shared_ptr<LocCoarseMap> right_map)
    {
      static Timer t("LocCoarseMap::ConcatenateLCM(cmap)"); RegionTimer rt(t);
      /** with concatenated vertex/edge map **/
      auto concmap = make_shared<LocCoarseMap>(this->mesh, right_map->mapped_mesh);
      SetConcedMap(right_map, concmap);
      // auto concmap = BaseCoarseMap::Concatenate(right_map);
      /** concatenated aggs! **/
      const size_t NCV = right_map->GetMappedNN<NT_VERTEX>();
      FlatTable<int> aggs1 = GetMapC2F<NT_VERTEX>(), aggs2 = GetMapC2F<NT_VERTEX>();
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
      static Timer t("LocCoarseMap::Concatenate(vmap)"); RegionTimer rt(t);
      /** no mesh on coarse level **/
      this->mapped_mesh = nullptr;
      /** no edges on coarse level **/
      mapped_NN[NT_VERTEX] = NCV;
      mapped_NN[NT_EDGE] = 0;
      /** concatenate vertex map **/
      auto & vmap = node_maps[NT_VERTEX];
      for (auto & val : vmap)
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


  /** SPWAgglomerator **/

  template<class ENERGY, class TMESH, bool ROBUST>
  SPWAgglomerator<ENERGY, TMESH, ROBUST> ::SPWAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts, Options && _settings)
    : BaseCoarseMap(_mesh), AgglomerateCoarseMap<TMESH>(_mesh), free_verts(_free_verts), settings(_settings)
  {
    cout << " SPW C 1 " << endl;
    assert(mesh != nullptr); // obviously this would be bad
  } // SPWAgglomerator(..)


  template<class ENERGY, class TMESH, bool ROBUST>
  SPWAgglomerator<ENERGY, TMESH, ROBUST> ::SPWAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts)
    : BaseCoarseMap(_mesh), AgglomerateCoarseMap<TMESH>(_mesh), free_verts(_free_verts)
  {
    cout << " SPW C 2 " << endl;
    assert(mesh != nullptr); // obviously this would be bad
  } // SPWAgglomerator(..)


  template<class ENERGY, class TMESH, bool ROBUST> template<class ATD, class TMU>
  INLINE void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: GetEdgeData (FlatArray<ATD> in_data, Array<TMU> & out_data)
  {
    if constexpr(std::is_same<ATD, TMU>::value)
      { out_data.FlatArray<TMU>::Assign(in_data); }
    else if constexpr(std::is_same<TMU, double>::value) {
      /** use trace **/
	out_data.SetSize(in_data.Size());
      for (auto k : Range(in_data))
	{ out_data[k] = ENERGY::GetApproxWeight(in_data[k]); }
      }
    else { /** for h1, double -> TM extension is necessary pro forma **/
      /** \lambda * Id **/
      out_data.SetSize(in_data.Size());
      for (auto k : Range(in_data))
	{ SetIdentity(ENERGY::GetApproxWeight(in_data[k]), out_data[k]); }
    }
  } // SPWAgglomerator::GetEdgeData


  template<class ENERGY, class TMESH, bool ROBUST>
  void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    if constexpr (ROBUST) {
	if (settings.robust) /** cheap, but not robust for some corner cases **/
	  { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
	else /** (much) more expensive, but also more robust **/
	  { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
      }
    else // do not even compile the robust version - saves a lot of ti
      { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
  } // SPWAgglomerator::FormAgglomerates


  template<class ENERGY, class TMESH, bool ROBUST> Timer & GetRoundTimer (int round) {
    static Array<Timer> timers ( { Timer("FormAggloemrates - round 0"),
	  Timer("FormAggloemrates - round 1"),
	  Timer("FormAggloemrates - round 2"),
	  Timer("FormAggloemrates - round 3"),
	  Timer("FormAggloemrates - round 4"),
	  Timer("FormAggloemrates - round 5"),
	  Timer("FormAggloemrates - rounds > 5") } );
    return timers[min(5, round)];
  } // GetRoundTimer

  template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
  INLINE void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
  {
    static_assert ( (std::is_same<TMU, TM>::value || std::is_same<TMU, double>::value), "Only 2 options make sense!");

    static Timer t("SPWAgglomerator::FormAgglomerates_impl"); RegionTimer rt(t);
    static Timer tfaggs("FormAgglomerates - finalize aggs");
    static Timer tmap("FormAgglomerates - map loc mesh");
    static Timer tprep("FormAgglomerates - prep");
    static Timer tsv("FormAgglomerates - spec verts");
    static Timer tvp("FormAgglomerates - pair verts");
    tprep.Start();

    cout << " FORM SPW AGGLOMERATES " << endl;

    typedef typename Options::CW_TYPE CW_TYPE;
    const bool print_params = settings.print_aggs;  // parameters for every round
    const bool print_summs = settings.print_aggs;   // summary info
    const bool print_aggs = settings.print_aggs;    // actual aggs

    const int num_rounds = settings.num_rounds;

    constexpr int BS = mat_traits<TM>::HEIGHT;
    constexpr int BSU = mat_traits<TMU>::HEIGHT;
    constexpr bool robust = (BS == BSU) && (BSU > 1);

    auto tm_mesh = dynamic_pointer_cast<TMESH>(mesh);
    const auto & M = *tm_mesh; M.CumulateData();
    const auto & eqc_h = *M.GetEQCHierarchy();
    auto comm = eqc_h.GetCommunicator();
    const auto NV = M.template GetNN<NT_VERTEX>();
    const auto NE = M.template GetNN<NT_EDGE>();
    const auto & econ = *M.GetEdgeCM();
    
    const double MIN_ECW = settings.edge_thresh;
    const double MIN_VCW = settings.vert_thresh;

    if (print_params) {
      cout << "SPWAgglomerator::FormAgglomerates_impl, params are: " << endl;
      cout << " ROBUST = " << robust << endl;
      cout << " num_rounds = " << num_rounds << endl;
      cout << " MIN_ECW = " << MIN_ECW << endl;
      cout << " MIN_VCW = " << MIN_VCW << endl;
    }

    shared_ptr<LocCoarseMap> conclocmap;
    
    FlatArray<TVD> base_vdata = get<0>(tm_mesh->Data())->Data();
    Array<TMU> base_edata_full; GetEdgeData<TED, TMU>(get<1>(tm_mesh->Data())->Data(), base_edata_full);
    Array<double> base_edata; GetEdgeData<TMU, double>(base_edata_full, base_edata);
    Array<TMU> base_diags(M.template GetNN<NT_VERTEX>());
    TM Qij, Qji; SetIdentity(Qij); SetIdentity(Qji);
    M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
	constexpr int rrobust = robust;
	const auto & em = base_edata[edge.id];
	if constexpr(rrobust) {
	  ENERGY::ModQs(base_vdata[edge.v[0]], base_vdata[edge.v[1]], Qij, Qji);
	  ENERGY::AddQtMQ(1.0, base_diags[edge.v[0]], Qij, em);
	  ENERGY::AddQtMQ(1.0, base_diags[edge.v[1]], Qji, em);
	}
	else {
	  base_diags[edge.v[0]] += em;
	  base_diags[edge.v[1]] += em;
	}
      }, true); // only master, we cumulate this afterwards
    M.template AllreduceNodalData<NT_VERTEX>(base_diags, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

    Array<TVD> cvdata;
    Array<TMU> cedata_full, cdiags;
    Array<double> cedata;
    FlatArray<TVD> fvdata; fvdata.Assign(base_vdata);
    FlatArray<TMU> fedata_full; fedata_full.Assign(base_edata_full);
    FlatArray<TMU> fdiags; fdiags.Assign(base_diags);
    FlatArray<double> fedata; fedata.Assign(base_edata);

    LocalHeap lh(20971520, "cthulu"); // 20 MB

    auto calc_avg_scal = [&](AVG_TYPE avg, double mtra, double mtrb) LAMBDA_INLINE {
      switch(avg) {
      case(MIN): { return min(mtra, mtrb); break; }
      case(GEOM): { return sqrt(mtra * mtrb); break; }
      case(HARM): { return 2 * (mtra * mtrb) / (mtra + mtrb); break; }
      case(ALG): { return (mtra + mtrb) / 2; break; }
      case(MAX): { return max(mtra, mtrb); break; }
      default: { return -1.0; }
      }
    };

    /** vertex coll wt. only computed in round 0 **/
    auto calc_vcw = [&](auto v) {
      return 0.0;
    };

    auto allow_merge = [&](auto eqi, auto eqj) LAMBDA_INLINE {
      if (eqi == 0)
	{ return eqc_h.IsMasterOfEQC(eqj); }
      else if (eqj == 0)
	{ return eqc_h.IsMasterOfEQC(eqi); }
      else
	{ return eqc_h.IsMasterOfEQC(eqi) && eqc_h.IsMasterOfEQC(eqj) && ( eqc_h.IsLEQ(eqi, eqj) || eqc_h.IsLEQ(eqj, eqi) ); }
    };

    /** Initial SOC to pick merge candidate. Can be EVP or scalar based. **/
    double da, db, dedge;
    auto calc_soc_scal = [&](CW_TYPE cw_type, AVG_TYPE mm_avg, auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
      /** calc_trace does nothing for scalar case **/
      dedge = calc_trace(fedata[int(fecon(vi, vj))]);
      switch(cw_type) {
      case(Options::CW_TYPE::HARMONIC) : { return dedge / calc_avg_scal(HARM, calc_trace(fdiags[vi]), calc_trace(fdiags[vj])); break; }
      case(Options::CW_TYPE::GEOMETRIC) : { return dedge / calc_avg_scal(GEOM, calc_trace(fdiags[vi]), calc_trace(fdiags[vj])); break; }
      case(Options::CW_TYPE::MINMAX) : {
	da = db = 0;
	for (auto eid : fecon.GetRowValues(vi))
	  { da = max2(da, calc_trace(fedata[int(eid)])); }
	for (auto eid : fecon.GetRowValues(vj))
	  { da = max2(da, calc_trace(fedata[int(eid)])); }
	return dedge / calc_avg_scal(mm_avg, da, db);
	break;
      }
      default : { return 0.0; break; }
      }
    };

    /** EVP based pairwise SOC. **/
    TM dma, dmb, dmedge, Q; SetIdentity(Q);
    auto calc_soc_robust = [&](CW_TYPE cw_type, AVG_TYPE mma_scal, bool mma_mat_harm, auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
      constexpr bool rrobust = robust;
      double soc = 1.0;
      if constexpr(rrobust) { // dummy - should never be called anyways!
	soc = 0;
	/** Transform diagonal matrices **/
	TVD H_data = ENERGY::CalcMPData(fvdata[vi], fvdata[vj]);
	ENERGY::ModQHh(H_data, fvdata[vi], Q);
	ENERGY::SetQtMQ(1.0, dma, Q, fdiags[vi]);
	ENERGY::ModQHh(H_data, fvdata[vj], Q);
	ENERGY::SetQtMQ(1.0, dmb, Q, fdiags[vj]);
	/** TODO:: neib bonus **/
	dmedge = fedata_full[int(fecon(vi,vj))];
	switch(cw_type) {
	case(CW_TYPE::HARMONIC) : { soc = MIN_EV_HARM2(dma, dmb, dmedge); break; }
	case(CW_TYPE::GEOMETRIC) : { soc = MIN_EV_FG2(dma, dmb, dmedge); break; }
	case(CW_TYPE::MINMAX) : {
	  double mtra = 0, mtrb = 0;
	  for (auto eid : fecon.GetRowValues(vi))
	    { mtra = max2(mtra, calc_trace(fedata[int(eid)])); }
	  for (auto eid : fecon.GetRowValues(vj))
	    { mtrb = max2(mtrb, calc_trace(fedata[int(eid)])); }
	  double etrace = calc_trace(dmedge);
	  double soc = etrace / calc_avg_scal(mma_scal, mtra, mtrb);
	  if (soc > MIN_ECW) {
	    dma /= calc_trace(dma);
	    dmb /= calc_trace(dmb);
	    dmedge /= etrace;
	    if (mma_mat_harm)
	      { soc = min2(soc, MIN_EV_HARM2(dma, dmb, dmedge)); }
	    else
	      { soc = min2(soc, MIN_EV_FG2(dma, dmb, dmedge)); }
	  }
	  break;
	}
	} // switch
	} // robust
      return soc;
    };

    auto calc_soc_pair = [&](bool dorobust, CW_TYPE cwt, AVG_TYPE mma_scal, AVG_TYPE mma_mat, auto vi, auto vj, const auto & fecon) { // maybe not force inlining this?
      constexpr bool rrobust = robust;
      if constexpr(rrobust) {
	  if (dorobust)
	    { return calc_soc_robust(cwt, mma_scal, (mma_mat==HARM), vi, vj, fecon); }
	  else
	    { return calc_soc_scal(cwt, mma_scal, vi, vj, fecon); }
	}
      else
	{ return calc_soc_scal(cwt, mma_scal, vi, vj, fecon); }
    };

    /** SOC for pair of agglomerates w.r.t original matrix **/
    auto calc_soc_check2 = [&](auto memsi, auto memsj, const auto & fecon, auto get_mems) LAMBDA_INLINE { return 1.0; };

    /** Finds a neighbor to merge vertex v with. Returns -1 if no suitable ones found **/
    auto find_neib = [&](auto v, const auto & fecon, auto allowed, auto get_mems, bool robust_pick,
			 auto cwt_pick, auto pmmas, auto pmmam, auto cwt_check, auto cmmas, auto cmmam, bool checkbigsoc) LAMBDA_INLINE {
      constexpr bool rrobust = robust;
      HeapReset hr(lh);
      /** SOC for all neibs **/
      double max_soc = 0, msn = -1, c = 0;
      auto neibs = fecon.GetRowIndices(v);
      FlatArray<INT<2,double>> bsocs(neibs.Size(), lh);
      for (auto neib : neibs)
	if (allowed(v, neib))
	  { bsocs[c++] = INT<2, double>(neib, calc_soc_pair(robust_pick, cwt_pick, pmmas, pmmam, v, neib, fecon)); }
      auto socs = bsocs.Part(0, c);
      QuickSort(socs, [&](const auto & a, const auto & b) LAMBDA_INLINE { return a[1] > b[1]; });
      int candidate = ( (c > 0) && (socs[0][1] > MIN_ECW) ) ? int(socs[0][0]) : -1;
      /** check candidate - either small EVP, or large EVP, or both! **/
      bool need_check = (robust && (!robust_pick)) || (checkbigsoc);
      if (need_check) {
	for (int j = 1; j < socs.Size(); j++) {
	  if (socs[j][1] < MIN_ECW)
	    { candidate = -1; break; }
	  double stabsoc = socs[j][0];
	  if constexpr(rrobust) {
	      if (!robust_pick) /** small EVP soc **/
		{ stabsoc = calc_soc_pair(true, cwt_check, cmmas, cmmam, v, neibs[j], fecon); }
	    }
	if (checkbigsoc && (stabsoc > MIN_ECW)) /** big EVP soc **/
	  { stabsoc = calc_soc_check2(v, neibs[j], fecon, get_mems); }
	if (stabsoc > MIN_ECW) /** this neib has strong stable connection **/
	  { candidate = int(socs[j][0]); break; }
	}
      } // need_check
      return candidate;
    };

    /** Iterate through unhandled vertices and pair them up **/
    auto pair_vertices = [&](FlatArray<int> vmap, size_t & NCV,
			     int num_verts, auto get_vert, const auto & fecon,
			     BitArray & handled,
			     auto get_mems, auto allowed,// auto set_pair,
			     bool r_ar,  auto r_cwtp, auto r_pmmas, auto r_pmmam, auto r_cwtc, auto r_cmmas, auto r_cmmam, bool r_cbs
			     ) LAMBDA_INLINE {
      RegionTimer rt(tvp);
      for (auto k : Range(num_verts)) {
	auto vnr = get_vert(k);
	if (!handled.Test(vnr)) { // try to find a neib
	  int neib = find_neib(vnr, fecon, allowed, get_mems, r_ar, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, r_cbs);
	  if (neib != -1) {
	    vmap[neib] = NCV;
	    handled.SetBit(neib);
	  }
	  vmap[vnr] = NCV++;
	  handled.SetBit(vnr);
	  // set_pair(vnr, neib);
	}
      }
      if (print_summs) {
      }
      if (print_aggs) {
      }
    };

    tprep.Stop();

    // pair vertices
    for (int round : Range(num_rounds)) {
      const bool r_ar = robust && settings.allrobust.GetOpt(round);
      const CW_TYPE r_cwtp = settings.pick_cw_type.GetOpt(round);
      const AVG_TYPE r_pmmas = settings.pick_mma_scal.GetOpt(round);
      const AVG_TYPE r_pmmam = settings.pick_mma_mat.GetOpt(round);
      const CW_TYPE r_cwtc = settings.check_cw_type.GetOpt(round);
      const AVG_TYPE r_cmmas = settings.check_mma_scal.GetOpt(round);
      const AVG_TYPE r_cmmam = settings.check_mma_mat.GetOpt(round);
      const bool r_cbs = settings.checkbigsoc;
      const bool use_hack_stab = settings.use_stab_ecw_hack.IsTrue() ||
	( (!settings.use_stab_ecw_hack.IsFalse()) 
	  && ( (   settings.allrobust.GetOpt(round)  && (settings.pick_cw_type.GetOpt(round)  == CW_TYPE::HARMONIC) ) ||
	       ( (!settings.allrobust.GetOpt(round)) && (settings.check_cw_type.GetOpt(round) == CW_TYPE::HARMONIC)) ) ) ;

      if (print_params) {
	cout << " round " << round << " of " << num_rounds << endl;
	cout << " allrobust      = " << r_ar << endl;
	cout << " cwt pick       = " << r_cwtp << endl;
	cout << " mma pick scal  = " << r_pmmas << endl;
	cout << " mma pick mat   = " << r_pmmam << endl;
	cout << " cwt check      = " << r_cwtc << endl;
	cout << " mma check scal = " << r_cmmas << endl;
	cout << " mma check mat  = " << r_cmmam << endl;
	cout << " checkbigsoc    = " << r_cbs << endl;
	cout << " use_hack_stab  = " << use_hack_stab << endl;
      }

      /** Set up a local map to represent this round of pairing vertices **/
      auto locmap = make_shared<LocCoarseMap>((round == 0) ? mesh : conclocmap->GetMappedMesh());
      auto vmap = conclocmap->template GetMap<NT_VERTEX>();
      const TopologicMesh & fmesh = *locmap->GetMesh();
      const SparseMatrix<double> & fecon = *fmesh.GetEdgeCM();
      
      fvdata.Assign(cvdata);
      fedata.Assign(fedata);
      fedata_full.Assign(cedata_full);
      fdiags.Assign(cdiags);

      BitArray handled(NV); handled.Clear();
      size_t NCV = 0, NCE = 0;
      
      if (round == 0) {
	tsv.Start();
	/** dirichlet vertices **/
	if (free_verts != nullptr) {
	  const auto & fvs = *free_verts;
	  for (auto v : Range(vmap))
	    if (!fvs.Test(v))
	      { vmap[v] = -1; handled.SetBit(v); }
	}
	/** non-master vertices **/
	M.template ApplyEQ2<NT_VERTEX>([&](auto eq, auto vs) {
	    if (!eqc_h.IsMasterOfEQC(eq))
	      for (auto v : vs)
		{ vmap[v] = -1; handled.SetBit(v); }
	  }, false); // obviously, not master only
	/** Fixed Aggs - set verts to diri, handle afterwards **/
	for (auto row : fixed_aggs)
	  for (auto v : row)
	    { vmap[v] = -1; handled.SetBit(v); }
	/** collapsed vertices **/
	if (MIN_VCW > 0) { // otherwise, no point
	  for(auto v : Range(vmap)) {
	    double cw = calc_vcw(v);
	    if (cw > MIN_VCW)
	      { vmap[v] = -1; handled.SetBit(v); }
	  }
	}
	tsv.Stop();
	/** CMK ordering **/
	Array<int> cmk;
	CalcCMK(handled, econ, cmk);
	/** Find pairs for vertices **/
	Array<int> dummy(1);
	pair_vertices(vmap, NCV, cmk.Size(), [&](auto k) { return cmk[k]; }, fecon, handled,
		      [&](auto v) { dummy[0] = v; return dummy; }, // get_mems
		      [&](auto vi, auto vj) { return allow_merge(M.template GetEqcOfNode<NT_VERTEX>(vi), M.template GetEqcOfNode<NT_VERTEX>(vj)); }, // allowed
		      r_ar, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, false); // no big soc necessary
      }
      else {
	/** Find pairs for vertices **/
	auto c2fv = conclocmap->template GetMapC2F<NT_VERTEX>();
	auto veqs = conclocmap->GetV2EQ();
	pair_vertices(vmap, NCV,
		      vmap.Size(), [&](auto i) LAMBDA_INLINE { return i; }, // no CMK on later rounds!
		      fecon, handled,
		      [&](auto v) LAMBDA_INLINE { return c2fv[v]; }, // get_mems
		      [&](auto vi, auto vj) LAMBDA_INLINE { return allow_merge(veqs[vi], veqs[vj]); }, // allowed
		      r_ar, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, r_cbs);
      }

      if (round < num_rounds - 1) { /** proper concatenated map, with coarse mesh **/
	RegionTimer art(tmap);
	/** Build edge map, C2F vertex map, edge connectivity and coarse mesh **/
	locmap->FinishUp();
	/** Coarse vertex data **/
	Array<TVD> ccvdata(NCV);
	auto c2fv = locmap->template GetMapC2F<NT_VERTEX>();
	for (auto cvnr : Range(ccvdata)) {
	  auto fvs = c2fv[cvnr];
	  if (fvs.Size() == 1)
	    { ccvdata[cvnr] = fvdata[fvs[0]]; }
	  else
	    { ccvdata[cvnr] = ENERGY::CalcMPData(fvdata[fvs[0]], fvdata[fvs[1]]); }
	}
	/** Coarse edge data **/
	Array<TMU> ccedata_full(NCE); cedata_full = 0;
	auto fedges = locmap->GetMesh()->template GetNodes<NT_EDGE>();
	auto cedges = locmap->GetMappedMesh()->template GetNodes<NT_EDGE>();
	auto emap = locmap->template GetMap<NT_EDGE>();
	TM Q; SetIdentity(Q);
	TVD femp, cemp;
	for (auto fenr : Range(fedges)) {
	  auto & fedge = fedges[fenr];
	  auto cenr = emap[fenr];
	  if (cenr != -1) {
	    auto & cedge = cedges[cenr];
	    if constexpr(robust) {
	      femp = ENERGY::CalcMPData(fvdata[fedge.v[0]], fvdata[fedge.v[1]]);
	      cemp = ENERGY::CalcMPData(ccvdata[cedge.v[0]], ccvdata[cedge.v[1]]);
	      ENERGY::ModQHh(cemp, femp, Q);
	      ENERGY::AddQtMQ(1.0, ccedata_full[cenr], Q, fedata[fenr]);
	    }
	    else
	      { ccedata_full[cenr] += fedata[fenr]; }
	  }
	}
	/** Coarse diags, I have to do this here because off-proc entries.
	    Maps are only local on master, so cannot cumulate on coarse level **/
	Array<TMU> ccdiags(NCV);
	TM Qij, Qji; SetIdentity(Qij); SetIdentity(Qji);
	for (auto cvnr : Range(ccvdata)) {
	  auto fvs = c2fv[cvnr];
	  if (fvs.Size() == 1)
	    { cdiags[cvnr] = fdiags[fvs[0]]; }
	  else { /** sum up diags, remove contribution of connecting edge **/
	    if constexpr(robust) {
	      ENERGY::ModQs(fvdata[fvs[0]], fvdata[fvs[1]], Qij, Qji);
	      ENERGY::SetQtMQ(1.0, cdiags[cvnr], Qji, fdiags[fvs[0]]);
	      ENERGY::AddQtMQ(1.0, cdiags[cvnr], Qij, fdiags[fvs[1]]);
	    }
	    else
	      { cdiags[cvnr] += fdiags[fvs[1]]; }
	    /** note: this should be fine, on coarse level, at least one vert is already in an agg,
		or it would already have been paired in first round **/
	    double fac = ( (round == 0) && use_hack_stab ) ? -1.5 : -2.0;
	    /** note: coarse vert is exactly ad edge-midpoint **/
	    cdiags[cvnr] -= fac * fedata_full[int(fecon(fvs[0], fvs[1]))];
	  }
	}
	/** Concatenate local maps **/
	conclocmap = (round == 0) ? locmap : conclocmap->ConcatenateLCM(locmap);

	fvdata.Assign(0, lh); cvdata = move(ccvdata);
	fedata_full.Assign(0, lh); cedata_full = move(ccedata_full);
	fedata.Assign(0, lh); GetEdgeData<TMU, double>(cedata_full, cedata);
	fdiags.Assign(0, lh); cdiags = move(ccdiags);
      }
      else {
	/** Only concatenate vertex map, no coarse mesh **/
	conclocmap->Concatenate(NCV, vmap);
      }

    } // round-loop

    /** Build final aggregates **/
    tfaggs.Start();
    size_t n_aggs_p = conclocmap->template GetMappedNN<NT_VERTEX>(), n_aggs_f = fixed_aggs.Size();
    size_t n_aggs_tot = n_aggs_p + n_aggs_f;
    agglomerates.SetSize(n_aggs_tot);
    v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;
    auto set_agg = [&](auto agg_nr, auto vs) {
      auto & agg = agglomerates[id];
      agg.id = agg_nr;
      int ctr_eqc = M.template GetEqcOfNode<NT_VERTEX>(vs[0]), v_eqc = -1;
      agg.ctr = vs[0]; // TODO: somehow mark agg ctrs [[ must be in the largest eqc ]] - which one is the best choice?
      agg.mems.SetSize(vs.Size());
      for (auto l : Range(vs)) {
	v_eqc = M.template GetEqcOfNode<NT_VERTEX>(vs[l]);
	if ( (v_eqc != 0) && (ctr_eqc != v_eqc) && (eqc_h.IsLEQ( v_eqc, ctr_eqc) ) )
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
      { set_agg(n_aggs_p + k, fixed_aggs[k]); }
    tfaggs.Stop();
    
  } // SPWAgglomerator::FormAgglomerates_impl


  template<class ENERGY, class TMESH, bool ROBUST>
  void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: CalcCMK (const BitArray & skip, const SparseMatrix<double> & econ, Array<int> & cmk)
  {
    /** Calc a CMK ordering of vertices. Connectivity given by "econ". Only orders those where "skip" is not set **/
    static Timer t("CalcCMK"); RegionTimer rt(t);
    size_t numtake = econ.Height() - skip.NumSet();
    if (numtake == 0)
      { cmk.SetSize0(); return; }
    cmk.SetSize(numtake); cmk = -12;
    size_t cnt = 0, cnt2 = 0;
    BitArray handled(skip);
    Array<int> neibs;
    while (cnt < numtake) {
      /** pick some minimum degree vertex to start with **/
      int mnc = econ.Height(), nextvert = -1; // max # of possible cons should be H - 1
      int ncons;
      for (auto k : Range(econ.Height()))
	if (!handled.Test(k))
	  if ( (ncons = econ.GetRowIndices(k).Size()) < mnc )
	    { mnc = ncons; nextvert = k; }
      if (nextvert == -1) // all verts are fully connected (including to itself, which should be illegal)
	for (auto k : Range(econ.Height()))
	  if (!handled.Test(k))
	    { nextvert = k; break; }
      cmk[cnt++] = nextvert; handled.SetBit(nextvert);
      while((cnt < numtake) && (cnt2 < cnt)) {
	/** add (unadded) neibs of cmk[cnt2] to cmk, ordered by degree, until we run out of vertices
	    to take neibs from - if that happens, we have to pick a new minimum degree vertex to start from! **/
	int c = 0; auto ris = econ.GetRowIndices(cmk[cnt2]);
	neibs.SetSize(ris.Size());
	for(auto k : Range(ris))
	  if (!handled.Test(k))
	    { neibs[c++] = ris[k]; handled.SetBit(nextvert); }
	neibs.SetSize(c);
	QuickSort(neibs, [&](auto vi, auto vj) { return econ.GetRowIndices(vi).Size() < econ.GetRowIndices(vj).Size(); });
	for (auto l : Range(c))
	  { cmk[cnt++] = neibs[l]; handled.SetBit(neibs[l]); }
	cnt2++;
      }
    }
  } // SPWAgglomerator::CalcCMK

  /** END SPWAgglomerator **/


} // namespace amg

#endif // SPWAGG

#endif
