#ifdef STOKES

#ifndef FILE_AMG_PC_STOKES_IMPL_HPP
#define FILE_AMG_PC_STOKES_IMPL_HPP

namespace amg
{

  /** Options **/

  template<class FACTORY, class AUX_SYS>
  class StokesAMGPC<FACTORY, AUX_SYS> :: Options : public FACTORY::Options,
						   public AUXPC::Options
  {
  public:
    bool hiptmair = true;          // Use Hiptmair Smoother
    bool hiptmair_bs = true;       // Inexact Braess-Sarazin/Hiptmair smoother
    bool hiptmair_block = false;   // Use Block-Smoothers in potential space

    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix)
    {
      FACTORY::Options::SetFromFlags(flags, prefix);
      AUXPC::Options::SetFromFlags(fes, flags, prefix);
      hiptmair = !flags.GetDefineFlagX(prefix + "hpt_sm").IsFalse();
      hiptmair_block = flags.GetDefineFlagX(prefix + "hpt_sm_blk").IsTrue();
      hiptmair_bs = !flags.GetDefineFlagX(prefix + "hpt_sm_bs").IsFalse();
    }
  }; // StokesAMGPC::Options

  /** END Options **/


  /** StokesAMGPC **/
  
  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : AUXPC(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : AUXPC(blf, flags, name)
  {
  } // StokesAMGPC(..)


  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: ~StokesAMGPC ()
  {
    ;
  } // ~StokesAMGPC


  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    if (forced_fds)
      { return; }
    
    if (options == nullptr) // should never happen
      { options = this->MakeOptionsFromFlags(this->flags); }

    /** Initialize auxiliary system (filter freedofs / auxiliary freedofs, pardofs / convert-operator / alloc auxiliary matrix ) **/
    AUXPC::InitLevel(freedofs);

    BaseAMGPC::finest_freedofs = aux_sys->GetAuxFreeDofs();
  } // StokesAMGPC::InitLevel


  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitLevelForced (shared_ptr<BitArray> freedofs)
  {
    InitLevel(freedofs);
    forced_fds = true;
  }

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseAMGPC::Options> StokesAMGPC<FACTORY, AUX_SYS> :: NewOpts ()
  {
    return make_shared<Options>();
  } // StokesAMGPC::NewOpts
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: SetDefaultOptions (BaseAMGPC::Options& O)
  {
  } // StokesAMGPC::SetDefaultOptions
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix)
  {
    static_cast<Options&>(O).SetFromFlags(aux_sys->GetCompSpace(), flags, prefix);
  } // StokesAMGPC::SetOptionsFromFlags
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: ModifyOptions (BaseAMGPC::Options & aO, const Flags & flags, string prefix)
  {
    auto & O (static_cast<Options&>(aO));

    AUXPC::__hacky_test = O.do_test;
    O.do_test = false;
  } // StokesAMGPC::ModifyOptions
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<TopologicMesh> StokesAMGPC<FACTORY, AUX_SYS> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh");
    RegionTimer rt(t);
    auto alg_mesh = BuildAlgMesh(BuildTopMesh());
    SetLoops(alg_mesh);
    return alg_mesh;
  } // StokesAMGPC::BuildInitialMesh
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    BaseAMGPC::InitFinestLevel(finest_level);

    /** We set BND-verts to dirichlet if their single facet is Dirichlet. **/
    auto free_facets = aux_sys->GetAuxFreeDofs();
    if (free_facets != nullptr) {
      const auto & ff(*free_facets);
      auto f2a_facet = aux_sys->GetFMapF2A();
      FlatArray<int> esort = node_sort[NT_EDGE];
      const auto & fmesh = static_cast<TMESH&>(*finest_level.cap->mesh);
      auto edges = fmesh.template GetNodes<NT_EDGE>();
      auto free_verts = make_shared<BitArray>(fmesh.template GetNN<NT_VERTEX>());
      free_verts->Clear();
      for (auto ffnr : Range(free_facets->Size())) {
	auto fnr = f2a_facet[ffnr];
	auto enr = esort[ffnr]; // NOT SURE 
	const auto & edge = edges[enr];
	if (ff.Test(ffnr)) {
	  free_verts->SetBit(edge.v[0]);
	  free_verts->SetBit(edge.v[1]);
	}
      }
      {
	// cout << "diri_verts: " << endl;
	// for (auto k : Range(free_verts->Size()))
	  // { if (!free_verts->Test(k)) { cout << k << " "; } }
	// cout << endl << endl;
      }
      finest_level.cap->free_nodes = free_verts;
    }
  } // StokesAMGPC::InitFinestLevel
  

  template<class FACTORY, class AUX_SYS>
  Table<int> StokesAMGPC<FACTORY, AUX_SYS> :: GetGSBlocks2 (const BaseAMGFactory::AMGLevel & amg_level)
  {
    if (amg_level.crs_map == nullptr) {
      throw Exception("Crs Map not saved!!");
      return move(Table<int>());
    }

    /** Blocks: 1 per agg, consisting of all "internal" edges, maybe also 1 per coarse facet **/

    auto & O(static_cast<Options&>(*options));

    const auto & map = *amg_level.crs_map;

    size_t cnv = map.GetMappedNN<NT_VERTEX>(), cne = map.GetMappedNN<NT_EDGE>();
    auto vmap = map.GetMap<NT_VERTEX>();
    auto emap = map.GetMap<NT_EDGE>();

    const auto & fmesh = *static_pointer_cast<TMESH>(map.GetMesh());
    auto loops = fmesh.GetLoops();
    
    TableCreator<int> cblocks(loops.Size());
    for (; !cblocks.Done(); cblocks++) {
      for (auto loop_nr : Range(loops)) {
	auto loop = loops[loop_nr];
	for (auto j : Range(loop))
	  { cblocks.Add(loop_nr, abs(loop[j])-1); }
      }
    }
    
    auto blocks = cblocks.MoveTable();

    return blocks;
  } // StokesAMGPC::GetGSBlocks2


  template<class FACTORY, class AUX_SYS>
  Table<int> StokesAMGPC<FACTORY, AUX_SYS> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    // return GetGSBlocks2(amg_level);

    if (amg_level.crs_map == nullptr) {
      throw Exception("Crs Map not saved!!");
      return move(Table<int>());
    }

    /** TODO: For MPI, sorting not in yet! **/
    /** Blocks: 1 per agg, consisting of all "internal" edges, maybe also 1 per coarse facet **/

    auto & O(static_cast<Options&>(*options));

    const auto & map = *amg_level.crs_map;

    size_t cnv = map.GetMappedNN<NT_VERTEX>(), cne = map.GetMappedNN<NT_EDGE>();
    auto vmap = map.GetMap<NT_VERTEX>();
    auto emap = map.GetMap<NT_EDGE>();

    const auto & fmesh = *map.GetMesh();
    
    Array<int> cnt(cnv);
    cnt = 0;
    for (const auto & edge : fmesh.GetNodes<NT_EDGE>()) {
      if (emap[edge.id] == -1) {
	auto cv = vmap[edge.v[0]];
	if ( (cv != -1) && (vmap[edge.v[1]] == cv) )
	  { cnt[cv]++; }
      }
      // else {
      // 	cnt[vmap[edge.v[0]]]++;
      // 	cnt[vmap[edge.v[1]]]++;
      // }
    }
    Array<int> cv2bnr(cnv); cv2bnr = -1;
    size_t n_blocks = 0;
    for (auto k : Range(cnv))
      if (cnt[k] > 0)
	{ cv2bnr[k] = n_blocks++; }
    size_t ebos = n_blocks; // edge block offset
    n_blocks += cne;
    
    TableCreator<int> cblocks(n_blocks);
    for (; !cblocks.Done(); cblocks++) {
      for (const auto & edge : fmesh.GetNodes<NT_EDGE>()) {
	auto ceid = emap[edge.id];
	if (ceid == -1) {
	  auto cv = vmap[edge.v[0]];
	  if ( (cv != -1) && (vmap[edge.v[1]] == cv) ) {
	    cblocks.Add(cv2bnr[cv], edge.id);
	  }
	}
	else {
	  // cblocks.Add(cv2bnr[vmap[edge.v[0]]], edge.id);
	  // cblocks.Add(cv2bnr[vmap[edge.v[1]]], edge.id);
	  cblocks.Add(ebos + ceid, edge.id);
	}
      }
    }

    auto blocks = cblocks.MoveTable();

    // cout << " BLOCKS ARE: " << endl;
    // cout << blocks << endl;
    
    return move(blocks);
  } // StokesAMGPC::GetGSBlocks
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseAMGFactory> StokesAMGPC<FACTORY, AUX_SYS> :: BuildFactory ()
  {
    return make_shared<FACTORY>(static_pointer_cast<Options>(options));
  } // StokesAMGPC::BuildFactory
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseDOFMapStep> StokesAMGPC<FACTORY, AUX_SYS> :: BuildEmbedding (BaseAMGFactory::AMGLevel & level)
  {
    shared_ptr<TopologicMesh> mesh = level.cap->mesh;

    /** 2-stage embedding: 
	  comp_fes -> aux_fes -> mesh-canonic
	Step 1 is the embedding-matrix from the auxiliary system
	Step 2 needs to consider:
	 i) fine_facets
	ii) edge_sort
       iii) facet-midpoint -> vertex-midpoint-pos
       "BND"-vertices do not matter here because their pos is set s.t facet-MP is axactly vertex-MP
    **/

    /** Step 1 - that was easy! **/
    shared_ptr<typename AUX_SYS::TPMAT_TM> aux_emb = aux_sys->GetPMat();

    /** Step 2 **/
    const TMESH & M = static_cast<const TMESH&>(*mesh); M.CumulateData(); // !!
    auto vdata = get<0>(M.Data())->Data();
    Array<int> perow(M.template GetNN<NT_EDGE>()); perow = 1;
    auto mesh_emb = make_shared<typename FACTORY::TSPM_TM>(perow, ma->GetNFacets());
    auto & edge_sort = node_sort[NT_EDGE];
    auto f2af = aux_sys->GetFMapF2A();
    auto edges = M.template GetNodes<NT_EDGE>();
    typename FACTORY::ENERGY::TM Qij; SetIdentity(Qij);
    typename FACTORY::ENERGY::TVD tvij, tvfacet;
    for (auto enr : Range(M.template GetNN<NT_EDGE>())) {
      auto fnr = f2af[enr];
      auto senr = edge_sort[enr];
      const auto & edge = edges[senr];
      if constexpr(is_same<typename FACTORY::ENERGY::TVD::TVD, double>::value == 0) {
	GetNodePos<DIM>(NodeId(FACET_NT(DIM), fnr), *ma, tvfacet.vd.pos); // cheating ...
	tvij = FACTORY::ENERGY::CalcMPData(vdata[edge.v[0]], vdata[edge.v[1]]);
	FACTORY::ENERGY::ModQHh(tvfacet, tvij, Qij);
      }
      (*mesh_emb)(senr, fnr) = Qij;
    }

    // cout << "PMAT:" << endl;
    // print_tm_spmat(cout << endl, *aux_sys->GetPMat());

    // cout << "mesh_emb:" << endl;
    // print_tm_spmat(cout << endl, *mesh_emb);

    auto mesh_pds = factory->BuildParallelDofs(mesh);

    /** Combine and build DMS **/
    // auto emb = MatMultAB(*aux_emb, *mesh_emb);
    // cout << "embedding:" << endl;
    // print_tm_spmat(cout << endl, *emb);
    // auto emb_dms = make_shared<ProlMap<typename AUX_SYS::TPMAT_TM>>(emb, aux_sys->GetCompParDofs(), mesh_pds);

    auto emb_dms = make_shared<ProlMap<typename AUX_SYS::TAUX_TM>>(mesh_emb, aux_sys->GetAuxParDofs(), mesh_pds);

    // return emb_dms;

    auto sfactory = static_pointer_cast<FACTORY>(factory);
    auto lcc = static_pointer_cast<typename FACTORY::StokesLC>(level.cap);
    sfactory->BuildCurlMat(*lcc);
    sfactory->BuildPotParDofs(*lcc);
    auto emb_prol = emb_dms->GetProl();
    typename FACTORY::TCM_TM & cmat = *lcc->curl_mat;
    auto cep = MatMultAB(*emb_prol, cmat);
    auto emb_pot = make_shared<ProlMap<stripped_spm_tm<Mat<FACTORY::BS, 1, double>>>>(cep, aux_sys->GetAuxParDofs(), lcc->pot_pardofs);

    // Array<shared_ptr<BaseDOFMapStep>> steps(2);
    // steps[0] = emb_dms; steps[1] = emb_pot;
    Array<shared_ptr<BaseDOFMapStep>> steps( { emb_dms, emb_pot } );
    auto multi_emb = make_shared<MultiDofMapStep>(steps);

    return multi_emb;
  } // StokesAMGPC::BuildEmbedding
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
  {
  } // StokesAMGPC::RegularizeMatrix


  template<class FACTORY, class AUX_SYS>
  shared_ptr<BlockTM> StokesAMGPC<FACTORY, AUX_SYS> :: BuildTopMesh ()
  {
    auto & O = static_cast<Options&>(*options);
    
    node_sort.SetSize(4);

    /** --- VERTICES ---
	VOL-Elements are "vertices". VOL-Elements are always local, but we need vertices in interfaces.
	Some VOL-element vertices need to be "published" to (some) neighbours. For every facet on an MPI
	interface, one of it's elements has to be published to the proc of it's other element. The element
	from the lower rank will be published to the higher one. This means we do not change the master of
	a vertex. One vertex can be published TO multiple procs, but only FROM one rank.

	Global enumeration of elements is just local enum on every proc.

	For every facet, we know that either
	  I) we have to publish our appended vertex, and we know to who
	 II) the vertex from the other side will be published to us, but we do not know exactly to who

	 We exchange messages:
	   [ NPV | f2v  | offsets | dpdata ]	  
	     1) # of published verts 
	     2) pub v nrs for shared facets // len = NF, entries in [0..NPV)  [[ -> need this to map edges to correct vertices ]]
	     3) offsets for distproc-table  // len = NPV + 1
	     4) distproc-table data // len = offsets[-1]


	 Additionally, we need "fictitious" vertices for every boundary element, so the boundary facets have an edge to be represented by.
	 SURF-els can have 0, 1 or 2 VOL-elements:
	   0:  -> ?? WTF ?? [[probably something broken in mesh-partition again]]
	   1:   i) local facet -> real boundary, needs fict. vertex
	       ii) ex-facet -> MPI boundary coincides with subdomain boundary
	   2:  subdomain boundary; no fictitious vertex

	 New local vertex numbering is simple:
	  [ pub from kp0 |  pub from kp1 | pub from kp2 | ... | local enum | FICTITIOUS vertices ]
     **/

    const auto & MA(*ma);
    auto aux_pds = aux_sys->GetAuxParDofs();
    auto all_dps = aux_pds->GetDistantProcs();
    auto comp_pds = aux_sys->GetCompParDofs();
    NgsAMG_Comm comm(comp_pds->GetCommunicator());

    // auto& fine_facet = *aux_sys->GetFineFacets();
    auto f2a_facet = aux_sys->GetFMapA2F();
    auto a2f_facet = aux_sys->GetFMapF2A();

    /** We already construct an AlgebraicMesh here, but only set a little data - the rest is set in the next step.
	What we set here:
	   i) vertex position (if needed)
	  ii) vertex vol / vertex surface index
     **/
    typedef typename std::remove_pointer<typename std::tuple_element<0, typename TMESH::TTUPLE>::type>::type ATVD;
    typedef typename ATVD::TDATA TVD;
    Array<TVD> vdata;
    typedef typename std::remove_pointer<typename std::tuple_element<1, typename TMESH::TTUPLE>::type>::type ATED;
    typedef typename ATED::TDATA TED;
    Array<TED> edata;

    TableCreator<int> covdps(ma->GetNE()); // dist-procs for original vertices
    TableCreator<int> cpubels(all_dps.Size()); // published elements
    for (; !covdps.Done(); covdps++) {
      for (auto elnr : Range(ma->GetNE())) {
	  for (auto fnr : ma->GetElFacets(elnr))
	    for (auto dp : aux_pds->GetDistantProcs(a2f_facet[fnr])) {
	      if (dp > comm.Rank()) {
		covdps.Add(elnr, dp);
		auto kp = find_in_sorted_array(dp, all_dps);
		cpubels.Add(kp, elnr);
	      }
	    }
      }
    }
    auto ovdps = covdps.MoveTable();
    for (auto row : ovdps)
      if ( row.Size() )
	{ QuickSort(row); }
    auto pubels = cpubels.MoveTable();

    Array<int> v_dps(20);
    Array<Array<int>> ex_data(all_dps.Size());
    Array<int> el2exel(ma->GetNE());
    for (auto kp : Range(all_dps)) {
      auto dp = all_dps[kp];
      if (dp > comm.Rank()) {
	auto& ex_data_row = ex_data[kp];
	auto exds = aux_pds->GetExchangeDofs(all_dps[kp]);
	int cnt_dps = 0;
	for (auto el : pubels[kp])
	  { cnt_dps += ovdps[el].Size(); }
	// 1 | exdss | (1 + NPV) | TBD
	ex_data_row.SetSize(2 + exds.Size() + pubels[kp].Size() + cnt_dps);
	// 1. NPV
	ex_data[kp][0] = pubels[kp].Size();
	// 2. ex-facet -> ex-el
	auto facels = ex_data_row.Part(1, exds.Size());
	auto pels = pubels[kp];
	for (auto j : Range(pels))
	  { el2exel[pels[j]] = j; }
	for (auto kd : Range(exds)) {
	  auto dnr = exds[kd];
	  auto fnr = f2a_facet[dnr];
	  auto elnr = ma->GetElFacets(fnr)[0];
	  // auto loc_elnr = find_in_sorted_array(elnr, pubels[kp]);
	  // facels[kd] = loc_elnr;
	  facels[kd] = el2exel[elnr];
	}
	// 3. offsets / 4. dps
	auto offsets = ex_data_row.Part(1 + exds.Size(), 1 + pubels[kp].Size());
	auto dpdata = ex_data_row.Part(2 + exds.Size() + pubels[kp].Size());
	offsets[0] = 0;
	int cntdpd = 0;
	for (auto kel : Range(pels)) {
	  auto elnr = pels[kel];
	  auto eldps = ovdps[elnr]; auto eldpss = eldps.Size();
	  offsets[kel+1] = offsets[kel] + eldpss;
	  auto chunk = dpdata.Part(cntdpd, eldpss);
	  for (auto j : Range(eldps))
	    { chunk[j] = (eldps[j] == dp) ? comm.Rank() : eldps[j]; }
	  QuickSort(chunk);
	  cntdpd += eldpss;
	}
      }
    }

    Array<MPI_Request> reqs(all_dps.Size());
    for (auto kp : Range(all_dps)) {
      auto dp = all_dps[kp];
      if (dp > comm.Rank())
	{ reqs[kp] = comm.ISend(ex_data[kp], dp, MPI_TAG_AMG); }
      else
	{ reqs[kp] = comm.IRecv(ex_data[kp], dp, MPI_TAG_AMG); }
    }
    MyMPI_WaitAll(reqs);

    int nsmaller = merge_pos_in_sorted_array(comm.Rank(), all_dps);
    Array<FlatTable<int>*> dptabs(nsmaller);
    Array<int> os_pub(1 + nsmaller);
    os_pub[0] = 0;
    Array<Array<size_t>> ft_st_os(nsmaller); // need offsets as size_t for flattable ...
    for (auto kp : Range(nsmaller)) {
      const int exdss = aux_pds->GetExchangeDofs(all_dps[kp]).Size();
      auto & ftos = ft_st_os[kp];
      auto oos = ex_data[kp].Part(1 + exdss, ex_data[kp][0] + 1);
      ftos.SetSize(ex_data[kp][0] + 1);
      for (auto j : Range(ftos))
	{ ftos[j] = oos[j]; }
      os_pub[kp + 1] = os_pub[kp] + ex_data[kp][0];
      dptabs[kp] = new FlatTable<int>(size_t(ex_data[kp][0]), ftos.Data(), ex_data[kp].Addr(2 + exdss + ex_data[kp][0]));
    }
    
    /** EQCHierarchy **/
    shared_ptr<EQCHierarchy> eqc_h = nullptr;
    {
      Array<INT<2,int>> kjs(100); kjs.SetSize0();
      Array<int> rs(100); rs.SetSize(0);
      if (ma->GetNE() > 0)
	{ rs.Append(0); }
      auto get_kj = [&](auto kj) -> FlatArray<int> {
	if (kj[0] == -1)
	  { return ovdps[kj[1]]; }
	else
	  { return (*dptabs[kj[0]])[kj[1]]; }
      };
      auto add_tab = [&](int k, auto & tab) {
	for (int j : Range(tab)) {
	  auto dps = tab[j];
	  if ( dps.Size() ) {
	    bool isnew = true;
	    for (auto kj : kjs)
	      if (dps == get_kj(kj))
		{ isnew = false; break; }
	    if (isnew)
	      { kjs.Append(INT<2,int>({k, j})); rs.Append(dps.Size()); }
	  }
	}
      };
      add_tab(-1, ovdps);
      for (auto k : Range(nsmaller))
	{ add_tab(k, *dptabs[k]); }
      Table<int> eqdps(rs);
      for (auto k : Range(size_t(1), rs.Size())) {
	eqdps[k] = get_kj(kjs[k-1]);
      }
      eqc_h = make_shared<EQCHierarchy>(move(eqdps), comm, true);
    }

    auto mesh = make_shared<BlockTM>(eqc_h);

    /** fictitious vertices **/
    Array<int> facet_els(2);
    auto it_sels = [&](auto lam) {
      for (auto selnr : Range(ma->GetNSE())) {
	auto sel_facets = ma->GetElFacets(ElementId(BND, selnr));
	if (sel_facets.Size() != 1)
	  { throw Exception("WTF"); }
	auto afnr = sel_facets[0];
	auto ffnr = a2f_facet[afnr];
	ma->GetFacetElements(afnr, facet_els);
	if (facet_els.Size() == 1) { // not an interior facet
	  // no idea why, but this crashes!!
	  // cout << "NFACETS " << ma->GetNFacets() << endl;
	  // cout << "get dps " << selnr << " " << afnr << " " << facet_els[0] << endl;
	  // auto fdps = ma->GetDistantProcs(NodeId(NT_FACET, afnr));
	  // auto fdps = ma->GetDistantProcs(NodeId(FACET_NT(DIM), afnr));
	  auto fdps = aux_pds->GetDistantProcs(afnr);
	  if (fdps.Size() == 0) { // also not an MPI facet, so it must be a "real" surface element
	    lam(selnr, afnr, ffnr);
	  }
	}
      }
    };
    size_t nfict = 0;
    it_sels([&](auto selnr, auto afnr, auto ffnr) {
	nfict++;
      });
    Array<int> facet_to_fict_vertex(f2a_facet.Size()); facet_to_fict_vertex = -1;
    Array<int> fvselnrs(nfict); nfict = 0;
    it_sels([&](auto selnr, auto afnr, auto ffnr) {
	facet_to_fict_vertex[ffnr] = selnr;
	fvselnrs[nfict++] = selnr;
      });

    /** nr of elements + number of new vertices published from other sources ! **/
    const size_t nxpub = os_pub.Last();
    const size_t nvels = ma->GetNE();
    const size_t NV = nxpub + ma->GetNE() + nfict;
    auto & vert_sort = node_sort[0]; vert_sort.SetSize(NV); vert_sort = -1; // not sure about this one
    Array<int> dummy_pds;
    vdata.SetSize(NV); vdata = 0;
    mesh->SetVs (NV, [&](auto vnr) -> FlatArray<int> {
	if (vnr < nxpub) {
	  int kp_orig = merge_pos_in_sorted_array(int(vnr), os_pub);
	  return (*dptabs[kp_orig])[vnr - os_pub[kp_orig]];
	}
	else if ( (vnr - nxpub) < nvels)
	  { return ovdps[vnr - nxpub]; }
	else
	  { return dummy_pds; }
      },
      [&](auto i, auto j) {
	vert_sort[i] = j;
	if (i >= nxpub) {
	  if ( (i-nxpub) < nvels ) {
	    // cout << " calc vol for vertex " << j << " = element " << i-nxpub << " = " << ma->ElementVolume(i-nxpub) << endl;

	    // cout << " calc vol for vertex " << j << ", el+1 " << i-nxpub+1 << " = " << ma->ElementVolume(1+i-nxpub) << endl;
	    vdata[j].vol = ma->ElementVolume(i-nxpub);
	    if constexpr(is_same<typename TVD::TVD, double>::value==0){throw Exception("SET CORRECT POS HERE!!"); }
	  }
	  else {
	    // cout << "vertex " << i << " -> " << j << " is surf index " << ma->GetElIndex(ElementId(BND, fvselnrs[i-nxpub-nvels])) << endl;
	    vdata[j].vol = -1 - ma->GetElIndex(ElementId(BND, fvselnrs[i-nxpub-nvels]));
	    if constexpr(is_same<typename TVD::TVD, double>::value==0){throw Exception("SET CORRECT POS HERE [mirror vol-pos]!!"); }
	  }
	}
      } // not sure
      // [vert_sort](auto i, auto j) { vert_sort[i] = j; } // not sure
      );

    // cout << "vert_sort: " << endl; prow2(vert_sort); cout << endl;

    const size_t osfict = nxpub + nvels;

    /** --- EDGES **/
    f2a_facet.Size();
    size_t n_edges = f2a_facet.Size();
    auto & edge_sort = node_sort[NT_EDGE]; edge_sort.SetSize(n_edges);
    mesh->SetNodes<NT_EDGE> (n_edges, [&](auto edge_num) LAMBDA_INLINE {
	INT<2> pair;
	auto fnr = f2a_facet[edge_num];
	ma->GetFacetElements(fnr, facet_els);
	if (facet_els.Size() == 1) {
	  auto facet_dps = aux_pds->GetDistantProcs(NodeId(NT_FACET, fnr));
	  if (facet_dps.Size() == 0) {
	    pair[0] = facet_els[0];
	    pair[1] = vert_sort[osfict + facet_to_fict_vertex[edge_num]];
	  }
	  else { // MPI-facet, possibly involves published vertex!
	    auto p = facet_dps[0];
	    if (comm.Rank() > p) { // other vertex has been published to me - I have to add this edge
	      auto kp = find_in_sorted_array(p, all_dps);
	      auto loc_fnr = find_in_sorted_array(int(edge_num), aux_pds->GetExchangeDofs(p));
	      pair[0] = vert_sort[os_pub[kp] + ex_data[kp][1 + loc_fnr]];
	      pair[1] = vert_sort[nxpub + facet_els[0]];
	    }
	  }
	}
	else {
	  pair[0] = vert_sort[nxpub + facet_els[0]];
	  pair[1] = vert_sort[nxpub + facet_els[1]];
	}
	if (pair[1] < pair[0])
	  { swap(pair[0], pair[1]); }
	return pair;
      },
      [&](auto edge_num, auto id) LAMBDA_INLINE {
	auto fnum = f2a_facet[edge_num];
	edge_sort[fnum] = id; // VERY unsure if this does something evil with additional edges...
      }
      );

    edata.SetSize(mesh->GetNN<NT_EDGE>()); edata = 0;

    // cout << "final top mesh: " << endl << *mesh << endl;

    auto avd = new ATVD(move(vdata), CUMULATED);
    auto aed = new ATED(move(edata), CUMULATED);
    auto alg_mesh = make_shared<TMESH>(move(*mesh), avd, aed);

    // cout << "alg mesh with partial data: " << endl << *alg_mesh << endl;

    return alg_mesh;
  } // StokesAMGPC::BuildTopMesh


  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: SetLoops (shared_ptr<typename StokesAMGPC<FACTORY, AUX_SYS>::TMESH> mesh)
  {
    /** Set the topological loops for Hiptmair Smoother **/
    auto loops = aux_sys->CalcFacetLoops();
    auto free_facets = aux_sys->GetAuxFreeDofs();
    int cntrl = 0;
    for (auto k : Range(loops)) {
      bool takeloop = true;
      for (auto j : loops[k]) {
	auto fnr = abs(j) - 1;
	if (!free_facets->Test(fnr))
	  { takeloop = false; /** cout << " discard loop " << k << endl; **/ }
      }
      if (takeloop)
	{ cntrl++; }
    }
    auto a2f_facet = aux_sys->GetFMapA2F();
    FlatArray<int> vsort = node_sort[0];
    FlatArray<int> fsort = node_sort[1];
    auto edges = mesh->template GetNodes<NT_EDGE>();
    Array<int> elnums;
    TableCreator<int> crl(cntrl);
    for (; !crl.Done(); crl++) {
      int c = 0;
      for (auto loop_nr : Range(loops)) {
	auto loop = loops[loop_nr];
	bool takeloop = true;
	for(auto j : Range(loop)) {
	  int fnr = abs(loop[j])-1;
	  int enr = fsort[a2f_facet[fnr]];
	  const auto & edge = edges[enr];
	  double fac = 1;
	  ma->GetFacetElements(fnr, elnums);
	  if (vsort[elnums[0]] == edge.v[0])
	    { fac = 1.0; }
	  else if (vsort[elnums[0]] == edge.v[1])
	    { fac = -1; }
	  else
	    { fac = 1; } // doesnt matter - should remove these loops anyways (?)
	  // if (enr != -1) { // let's say this cannot happen ??
	  // actually, I think we should throw out loops that touch the Dirichlet BND
	  loop[j] = (loop[j] > 0) ? fac * (1 + enr) : -fac * (1 + enr);
	  if (!free_facets->Test(fnr))
	    { takeloop = false; /** cout << " discard loop " << loop_nr << endl; **/ break; }
	  // }
	}
	if (takeloop)
	  { crl.Add(c++, loop); }
      }
    }
    auto edata = get<1>(mesh->Data())->Data();
    // cout << " edges/flows: " << endl;
    // for (auto k : Range(edges.Size()))
      // { cout << edges[k] << ", flow " << edata[k].flow << endl; }
    auto mod_loops = crl.MoveTable();
    // cout << " modded loops: " << endl << mod_loops << endl;
    // mesh->SetLoops(move(loops));
    mesh->SetLoops(move(mod_loops));
  } // StokesAMGPC::SetLoops


  template<class FACTORY, class AUX_SYS>
  shared_ptr<typename StokesAMGPC<FACTORY, AUX_SYS>::TMESH> StokesAMGPC<FACTORY, AUX_SYS> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    const auto & O = static_cast<Options&>(*options);

    shared_ptr<TMESH> alg_mesh;

    switch(O.energy) {
    case(Options::TRIV_ENERGY): { alg_mesh = BuildAlgMesh_TRIV(top_mesh); break; }
    case(Options::ALG_ENERGY): { alg_mesh = BuildAlgMesh_ALG(top_mesh); break; }
    case(Options::ELMAT_ENERGY): { throw Exception("Cannot do elmat energy!"); }
    default: { throw Exception("Invalid Energy!"); break; }
    }

    return alg_mesh;
  } // StokesAMGPC::BuildAlgMesh


  template<class FACTORY, class AUX_SYS>
  Array<shared_ptr<BaseSmoother>> StokesAMGPC<FACTORY, AUX_SYS> :: BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels,
										   shared_ptr<DOFMap> dof_map)
  {
    const auto & O = static_cast<Options&>(*options);

    if (O.hiptmair && O.hiptmair_bs)
      { return BuildSmoothersHIBS(amg_levels, dof_map); }
    else {
      Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);
      for (int k = 0; k < amg_levels.Size() - 1; k++) {
	if ( (k > 0) && O.regularize_cmats) // Regularize coarse level matrices
	  { RegularizeMatrix(amg_levels[k]->cap->mat, amg_levels[k]->cap->pardofs); }
	smoothers[k] = BuildSmoother(*amg_levels[k], dof_map);
	cout << "type k sm " << typeid(*smoothers[k]).name() << endl;
      }
      return smoothers;
    }
  } // StokesAMGPC::BuildSmoothers


  template<class FACTORY, class AUX_SYS>
  Array<shared_ptr<BaseSmoother>> StokesAMGPC<FACTORY, AUX_SYS> :: BuildSmoothersHIBS (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
										       shared_ptr<DOFMap> dof_map)
  {
    const auto & O = static_cast<Options&>(*options);

    /** Split potential/range maps from dof_map **/
    auto pot_dof_map = make_shared<DOFMap>();
    auto range_dof_map = make_shared<DOFMap>();
    for (auto k : Range(dof_map->GetNSteps())) {
      dof_map->GetStep(k)->Finalize(); // build transposed prols if we do not have them yet
      if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(dof_map->GetStep(k))) {
	range_dof_map->AddStep(mdms->GetMap(0));
	pot_dof_map->AddStep(mdms->GetMap(1));
      }
      else
	{ throw Exception("Do not have potential dof maps!"); }
    }

    auto build_pot_smoother = [&](int k) {
      const auto & cap = static_cast<typename FACTORY::StokesLC&>(*levels[k]->cap);
      if (O.hiptmair_block) {
	if (levels[k]->crs_map == nullptr)
	  { throw Exception("Crs Map not saved!!"); }
	const auto & M = static_cast<typename FACTORY::TMESH&>(*cap.mesh);
	Table<int> blocks = M.LoopBlocks(*levels[k]->crs_map);
	if (blocks.Size() == 0)
	  { return BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy()); }
	else
	  { return BaseAMGPC::BuildBGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy(), move(blocks)); }
      }
      else
	{ return BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, levels[k]->cap->eqc_h); }
    };

    /** Build Smoothers for the range spaces **/
    Array<shared_ptr<BaseSmoother>> range_smoothers(range_dof_map->GetNSteps());
    for (auto k : Range(range_smoothers)) {
      auto & cap = static_cast<typename FACTORY::StokesLC&>(*levels[k]->cap);
      auto hd_sm = BaseAMGPC::BuildSmoother(*levels[k]);
      /** Potential space maps from level k to max level **/
      shared_ptr<DOFMap> pot_dof_chunk = (k == 0) ? pot_dof_map : pot_dof_map->SubMap(k);
      /** Galerkin project potential space matrices  **/
      Array<shared_ptr<BaseSparseMatrix>> pot_mats_k = pot_dof_chunk->AssembleMatrices(cap.pot_mat);
      Array<shared_ptr<BaseSmoother>> pot_smoothers_k(pot_dof_chunk->GetNSteps());
      for (auto j : Range(pot_smoothers_k)) {
	// auto opm = cap.pot_mat; cap.pot_mat = dynamic_pointer_cast<SparseMatrix<double>>(pot_mats_k[j]);
	pot_smoothers_k[j] = build_pot_smoother(k + j);
	// cap.pot_mat = opm;
      }
      shared_ptr<AMGMatrix> pot_amg_mat = make_shared<AMGMatrix>(pot_dof_chunk, pot_smoothers_k);

      if (pot_amg_mat->GetSmoother(0)->Height() > 0)
	{
	  /** Aux space AMG as a preconditioner for Auxiliary matrix **/
	  auto i1 = printmessage_importance;
	  auto i2 = netgen::printmessage_importance;
	  printmessage_importance = 1;
	  netgen::printmessage_importance = 1;
	  cout << IM(1) << "Test potential space AMG, level " << k << "! " << endl;
	  // EigenSystem eigen(*aux_mat, *amg_mat);
	  EigenSystem eigen(*pot_amg_mat->GetSmoother(0)->GetAMatrix(), *pot_amg_mat); // need parallel mat
	  eigen.SetPrecision(1e-12);
	  eigen.SetMaxSteps(1000); 
	  eigen.Calc();
	  cout << IM(1) << "Results for potential space AMG, V1, level " << k << "! " << endl;
	  cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
	  cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
	  cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
	  printmessage_importance = i1;
	  netgen::printmessage_importance = i2;
	}

      Array<shared_ptr<BaseSmoother>> pot_smoothers_k2(pot_dof_chunk->GetNSteps());
      for (auto j : Range(pot_smoothers_k)) {
	auto & capj = static_cast<typename FACTORY::StokesLC&>(*levels[k + j]->cap);
	cout << k << " " << j << " " << cap.pot_mat << " " << pot_mats_k[j] << endl;
	auto opm = capj.pot_mat; capj.pot_mat = dynamic_pointer_cast<SparseMatrix<double>>(pot_mats_k[j]);
	pot_smoothers_k2[j] = build_pot_smoother(k + j);
	capj.pot_mat = opm;
      }
      shared_ptr<AMGMatrix> pot_amg_mat2 = make_shared<AMGMatrix>(pot_dof_chunk, pot_smoothers_k2);

      if (pot_amg_mat->GetSmoother(0)->Height() > 0)
	{
	  /** Aux space AMG as a preconditioner for Auxiliary matrix **/
	  auto i1 = printmessage_importance;
	  auto i2 = netgen::printmessage_importance;
	  printmessage_importance = 1;
	  netgen::printmessage_importance = 1;
	  cout << IM(1) << "Test potential space AMG, level " << k << "! " << endl;
	  // EigenSystem eigen(*aux_mat, *amg_mat);
	  EigenSystem eigen(*pot_amg_mat2->GetSmoother(0)->GetAMatrix(), *pot_amg_mat2); // need parallel mat
	  eigen.SetPrecision(1e-12);
	  eigen.SetMaxSteps(1000); 
	  eigen.Calc();
	  cout << IM(1) << "Results for potential space AMG, V2, level " << k << "! " << endl;
	  cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
	  cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
	  cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
	  printmessage_importance = i1;
	  netgen::printmessage_importance = i2;
	}

      auto hc_sm = make_shared<AMGSmoother>(pot_amg_mat2, 0);
      hc_sm->Finalize();
      shared_ptr<BaseMatrix> cmat, cmat_T;
      if (k == 0) {
	cout << " l0 emb " << levels[0]->embed_map << endl;
	cout << " l0 emb tp " << typeid(*levels[0]->embed_map).name() << endl;
	if ( auto emb_mdms = dynamic_pointer_cast<MultiDofMapStep>(levels[0]->embed_map) ) {
	  if ( auto emb_pot = dynamic_pointer_cast<ProlMap<stripped_spm_tm<Mat<2, 1, double>>>>(emb_mdms->GetMap(1)) ) {
	    cmat = dynamic_pointer_cast<SparseMatrix<Mat<FACTORY::BS, 1, double>>>(emb_pot->GetProl());
	    cmat_T = dynamic_pointer_cast<SparseMatrix<Mat<1, FACTORY::BS, double>>>(emb_pot->GetProlTrans());
	    if ( (cmat == nullptr) || (cmat_T == nullptr) )
	      { throw Exception("wow i really do need a lot of casts here ..."); }
	  }
	  else
	    { throw Exception("pot emb not correct"); }
	}
	else
	  { throw Exception("level 0 should be mdms!"); }
      }
      else {
	cmat = cap.curl_mat;
	cmat_T = cap.curl_mat_T;
      }
      auto hsm = make_shared<HiptMairSmoother>(hc_sm, hd_sm, cap.pot_mat, cap.mat, cmat, cmat_T);
      hsm->Finalize();
      range_smoothers[k] = hsm;
    }

    return range_smoothers;
  } // StokesAMGPC::BuildSmoothersHIBS


  template<class FACTORY, class AUX_SYS>
  Array<shared_ptr<BaseSmoother>> StokesAMGPC<FACTORY, AUX_SYS> :: BuildSmoothersHIBS2 (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
										       shared_ptr<DOFMap> dof_map)
  {
    const auto & O = static_cast<Options&>(*options);

    /** Split potential/range maps from dof_map **/
    auto pot_dof_map = make_shared<DOFMap>();
    auto range_dof_map = make_shared<DOFMap>();
    for (auto k : Range(dof_map->GetNSteps())) {
      dof_map->GetStep(k)->Finalize(); // build transposed prols if we do not have them yet
      if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(dof_map->GetStep(k))) {
	range_dof_map->AddStep(mdms->GetMap(0));
	pot_dof_map->AddStep(mdms->GetMap(1));
      }
      else
	{ throw Exception("Do not have potential dof maps!"); }
    }

    /** Build Smoothers for the potential spaces. **/
    Array<shared_ptr<BaseSmoother>> pot_smoothers(pot_dof_map->GetNSteps());
    for (auto k : Range(pot_smoothers)) {
      const auto & cap = static_cast<typename FACTORY::StokesLC&>(*levels[k]->cap);
      if (O.hiptmair_block) {
	if (levels[k]->crs_map == nullptr)
	  { throw Exception("Crs Map not saved!!"); }
	const auto & M = static_cast<typename FACTORY::TMESH&>(*cap.mesh);
	Table<int> blocks = M.LoopBlocks(*levels[k]->crs_map);
	if (blocks.Size() == 0)
	  { pot_smoothers[k] = BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy()); }
	else
	  { pot_smoothers[k] = BaseAMGPC::BuildBGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy(), move(blocks)); }
      }
      else
	{ pot_smoothers[k] = BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, levels[k]->cap->eqc_h); }
    }

    /** TODO: should I invert on the coarsest level potential space (if small enough??)
	Also, I think that would be a singular problem. **/

    /** Build an AMGMatrix for potential spaces. **/
    auto pot_amg_mat = make_shared<AMGMatrix>(pot_dof_map, pot_smoothers);
    {
      /** Aux space AMG as a preconditioner for Auxiliary matrix **/
      auto i1 = printmessage_importance;
      auto i2 = netgen::printmessage_importance;
      printmessage_importance = 1;
      netgen::printmessage_importance = 1;
      cout << IM(1) << "Test potential space AMG! " << endl;
      // EigenSystem eigen(*aux_mat, *amg_mat);
      EigenSystem eigen(*pot_amg_mat->GetSmoother(0)->GetAMatrix(), *pot_amg_mat); // need parallel mat
      eigen.SetPrecision(1e-12);
      eigen.SetMaxSteps(1000); 
      eigen.Calc();
      cout << IM(1) << "Results for potential space AMG! " << endl;
      cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
      cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      printmessage_importance = i1;
      netgen::printmessage_importance = i2;
    }

    /** Build Smoothers for the range spaces **/
    Array<shared_ptr<BaseSmoother>> range_smoothers(range_dof_map->GetNSteps());
    for (auto k : Range(range_smoothers)) {
      const auto & cap = static_cast<typename FACTORY::StokesLC&>(*levels[k]->cap);
      auto hd_sm = BaseAMGPC::BuildSmoother(*levels[k]);
      auto hc_sm = make_shared<AMGSmoother>(pot_amg_mat, k);
      hc_sm->Finalize();
      shared_ptr<BaseMatrix> cmat, cmat_T;
      if (k == 0) {
	cout << " l0 emb " << levels[0]->embed_map << endl;
	cout << " l0 emb tp " << typeid(*levels[0]->embed_map).name() << endl;
	if ( auto emb_mdms = dynamic_pointer_cast<MultiDofMapStep>(levels[0]->embed_map) ) {
	  if ( auto emb_pot = dynamic_pointer_cast<ProlMap<stripped_spm_tm<Mat<2, 1, double>>>>(emb_mdms->GetMap(1)) ) {
	    cmat = dynamic_pointer_cast<SparseMatrix<Mat<FACTORY::BS, 1, double>>>(emb_pot->GetProl());
	    cmat_T = dynamic_pointer_cast<SparseMatrix<Mat<1, FACTORY::BS, double>>>(emb_pot->GetProlTrans());
	    if ( (cmat == nullptr) || (cmat_T == nullptr) )
	      { throw Exception("wow i really do need a lot of casts here ..."); }
	  }
	  else
	    { throw Exception("pot emb not correct"); }
	}
	else
	  { throw Exception("level 0 should be mdms!"); }
      }
      else {
	cmat = cap.curl_mat;
	cmat_T = cap.curl_mat_T;
      }
      auto hsm = make_shared<HiptMairSmoother>(hc_sm, hd_sm, cap.pot_mat, cap.mat, cmat, cmat_T);
      hsm->Finalize();
      range_smoothers[k] = hsm;
    }

    return range_smoothers;
  } // StokesAMGPC::BuildSmoothersHIBS2


  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseSmoother> StokesAMGPC<FACTORY, AUX_SYS> :: BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level, shared_ptr<DOFMap> dof_map)
  {
    const auto & O = static_cast<Options&>(*options);

    /** Smoother in the HDiv-like space **/
    auto sm = BaseAMGPC::BuildSmoother(amg_level);

    if (O.hiptmair)
	{ sm = BuildHiptMairSmoother(amg_level, sm); }

    return sm;
  } // StokesAMGPC::BuildSmoother


  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseSmoother> StokesAMGPC<FACTORY, AUX_SYS> :: BuildHiptMairSmoother (const BaseAMGFactory::AMGLevel & amg_level, shared_ptr<BaseSmoother> sm)
  {
    static Timer t("BuildHiptMairSmoother");
    RegionTimer rt(t);

    const auto & O = static_cast<Options&>(*options);

    // const auto & cap = *static_pointer_cast<typename FACTORY::StokesLC>(amg_level.cap);
    const auto & cap = static_cast<typename FACTORY::StokesLC&>(*amg_level.cap);

    const auto & M = *static_pointer_cast<TMESH>(cap.mesh);
    M.CumulateData();

    /** Smoother in HCurl-like space (not sure about EQCH!!) **/
    shared_ptr<BaseSmoother> csm;

    bool hblocks = O.hiptmair_block;
    if (hblocks) {
      Table<int> blocks;
      if (amg_level.crs_map == nullptr)
	{ throw Exception("Crs Map not saved!!"); }
      blocks = move(M.LoopBlocks(*amg_level.crs_map));
      if (blocks.Size() == 0)
	{ csm = BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy()); }
      else
	{ csm = BaseAMGPC::BuildBGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy(), move(blocks)); }
    }
    else
      { csm = BaseAMGPC::BuildGSSmoother(cap.pot_mat, cap.pot_pardofs, M.GetEQCHierarchy()); }

    /** Wrap parallel matrices **/
    auto A_p = make_shared<ParallelMatrix>(cap.mat, cap.pardofs, PARALLEL_OP::C2D);
    auto CTAC_p = make_shared<ParallelMatrix>(cap.pot_mat, cap.pot_pardofs, PARALLEL_OP::C2D);
    auto C_p = make_shared<ParallelMatrix>(cap.curl_mat, cap.pot_pardofs, cap.pardofs, PARALLEL_OP::C2C);
    auto CT_p = make_shared<ParallelMatrix>(cap.curl_mat_T, cap.pardofs, cap.pot_pardofs, PARALLEL_OP::D2D);

    /** Construct Hiptmair Smother **/
    auto hsm = make_shared<HiptMairSmoother>(csm, sm, CTAC_p, A_p, C_p, CT_p);

    return hsm;
  } // StokesAMGPC<FACTORY, AUX_SYS>::BuildHiptMairSmoother


  /** Older version - also sets up curl-mat and pot pardofs**/
  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseSmoother> StokesAMGPC<FACTORY, AUX_SYS> :: BuildHiptMairSmoother1 (const BaseAMGFactory::AMGLevel & amg_level, shared_ptr<BaseSmoother> sm)
  {
    static Timer t("BuildHiptMairSmoother");
    RegionTimer rt(t);

    const auto & O = static_cast<Options&>(*options);

    const auto & M = *static_pointer_cast<TMESH>(amg_level.cap->mesh);
    M.CumulateData();

    const auto & eqc_h = *M.GetEQCHierarchy();
    auto loops = M.GetLoops();
    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();

    /** ParallelDofs for (fake) HCurl-like space **/
    Array<int> dps(50);
    TableCreator<int> cdps(loops.Size());
    for (; !cdps.Done(); cdps++) {
      for (auto loop_nr : Range(loops)) {
	auto loop = loops[loop_nr];
	dps.SetSize0();
	for (auto etr : loop) {
	  int enr = abs(etr) - 1;
	  auto eqc = M.template GetEqcOfNode<NT_EDGE>(enr);
	  auto edps = eqc_h.GetDistantProcs(eqc);
	  for (auto p : edps) {
	    auto pos = merge_pos_in_sorted_array(p, dps);
	    if ( (pos == 0) || (dps[pos] < p) )
	      { dps.Insert(pos, p); }
	  }
	}
	cdps.Add(loop_nr, dps);
      }
    }
    auto loop_pds = cdps.MoveTable();

    /** Discrete curl matrix **/
    typedef stripped_spm_tm<Mat<1, FACTORY::BS, double>> TM_CT;
    Array<int> perow(loops.Size()); perow = 0;
    for (auto k : Range(loops.Size())) {
      perow[k] = loops[k].Size();
    }
    auto curlT_mat = make_shared<TM_CT>(perow, M.template GetNN<NT_EDGE>());
    for (auto k : Range(loops.Size())) {
      auto loop = loops[k];
      auto ris = curlT_mat->GetRowIndices(k);
      auto rvs = curlT_mat->GetRowValues(k);
      for (auto j : Range(ris))
	{ ris[j] = abs(loop[j]) - 1; }
      QuickSort(ris);
      for (auto j : Range(ris)) {
	int enr = abs(loop[j]) - 1;
	int col = ris.Pos(enr);
	int fac = (loop[j] < 0) ? -1 : 1;
	auto flow = edata[enr].flow;
	double fsum = 0;
	for (auto l : Range(FACTORY::BS))
	  { fsum += sqr(flow[l]); }
	for (auto l : Range(FACTORY::BS)) {
	  rvs[col](0, l) = fac * flow[l]/fsum;
	}
      }
    }

    // cout << " A DIMS " << amg_level.mat->Height() << " x " << amg_level.mat->Width() << endl;
    // cout << " curlT_mat dims " << curlT_mat->Height() << " x " << curlT_mat->Width() << endl;    
    // cout << " discrete curl mat" << endl;
    // print_tm_spmat(cout, *curlT_mat);
    // cout << endl;
    
	    
    /** Project matrix to HCurl-like space **/
    auto & CT_TM = curlT_mat;
    auto A_TM = dynamic_pointer_cast<typename FACTORY::TSPM_TM>(amg_level.cap->mat);
    auto C_TM = TransposeSPM(*CT_TM);
    // cout << " CT_TM " << endl;
    // print_tm_spmat(cout, *CT_TM); cout << endl;
    auto AC_TM = MatMultAB(*A_TM, *C_TM);
    // cout << " AC_TM " << endl;
    // print_tm_spmat(cout, *AC_TM); cout << endl;
    auto CTAC_TM = MatMultAB(*CT_TM, *AC_TM);

    // cout << " curl space mat " << endl;
    // print_tm_spmat(cout, *CTAC_TM);
    // cout << endl;

    typedef SparseMatrix<Mat<FACTORY::BS, 1, double>> T_C;
    typedef SparseMatrix<Mat<1, FACTORY::BS, double>> T_CT;
    typedef SparseMatrix<Mat<FACTORY::BS, FACTORY::BS, double>> T_A;

    auto A = dynamic_pointer_cast<T_A>(amg_level.cap->mat);
    auto C = make_shared<T_C>(move(*C_TM));
    auto CT = make_shared<T_CT>(move(*CT_TM));
    auto CTAC = make_shared<SparseMatrix<double>>(move(*CTAC_TM));    

    /** HC-like space is scalar ! **/
    auto hc_pds = make_shared<ParallelDofs>(eqc_h.GetCommunicator(), move(loop_pds), 1, false);

    /** Smoother in HCurl-like space (not sure about EQCH!!) **/
    Table<int> blocks;
    bool hblocks = O.hiptmair_block;
    if (hblocks) {
      if (amg_level.crs_map == nullptr)
	{ throw Exception("Crs Map not saved!!"); }
      blocks = move(M.LoopBlocks(*amg_level.crs_map));
      if (blocks.Size() == 0)
	{ hblocks = false; }
    }
    
    // cout << " use hiptmair blocks: " << hblocks << endl;

    shared_ptr<BaseSmoother> csm;
    if (hblocks)
      { csm = BaseAMGPC::BuildBGSSmoother(CTAC, hc_pds, M.GetEQCHierarchy(), move(blocks)); }
    else
      { csm = BaseAMGPC::BuildGSSmoother(CTAC, hc_pds, M.GetEQCHierarchy()); }

    /** Wrap parallel matrices **/
    auto A_p = make_shared<ParallelMatrix>(A, amg_level.cap->pardofs, PARALLEL_OP::C2D);
    auto CTAC_p = make_shared<ParallelMatrix>(CTAC, hc_pds, PARALLEL_OP::C2D);
    auto C_p = make_shared<ParallelMatrix>(C, hc_pds, amg_level.cap->pardofs, PARALLEL_OP::C2C);
    auto CT_p = make_shared<ParallelMatrix>(CT, amg_level.cap->pardofs, hc_pds, PARALLEL_OP::D2D);

    /** Construct Hiptmair Smother **/
    auto hsm = make_shared<HiptMairSmoother>(csm, sm, CTAC_p, A_p, C_p, CT_p);

    return hsm;
  } // StokesAMGPC<FACTORY, AUX_SYS>::BuildHiptMairSmoother1


  /** END StokesAMGPC **/

} // namespace amg

#endif // FILE_AMG_PC_STOKES_IMPL_HPP
#endif // STOKES
