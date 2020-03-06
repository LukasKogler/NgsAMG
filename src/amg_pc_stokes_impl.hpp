#ifdef STOKES

#ifndef FILE_AMG_PC_STOKES_IMPL_HPP
#define FILE_AMG_PC_STOKES_IMPL_HPP

namespace amg
{

  /** Options **/

  template<class FACTORY, class AUX_SYS>
  class StokesAMGPC<FACTORY, AUX_SYS> :: Options : public FACTORY::Options,
						   public BaseAMGPC::Options
  {
  public:
    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix)
    {
      FACTORY::Options::SetFromFlags(flags, prefix);
      BaseAMGPC::Options::SetFromFlags(flags, prefix);
    }
  }; // StokesAMGPC::Options

  /** END Options **/


  /** StokesAMGPC **/
  
  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : BaseAMGPC(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }


  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : BaseAMGPC(blf, flags, name)
  {
    aux_sys = make_shared<AUX_SYS>(bfa);
  } // StokesAMGPC(..)


  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: ~StokesAMGPC ()
  {
    ;
  } // ~StokesAMGPC


  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    if (options == nullptr) // should never happen
      { options = MakeOptionsFromFlags(flags); }

    /** Initialize auxiliary system (filter freedofs / auxiliary freedofs, pardofs / convert-operator / alloc auxiliary matrix ) **/
    aux_sys->Initialize(finest_freedofs);

    BaseAMGPC::finest_freedofs = aux_sys->GetAuxFreeDofs();

  } // StokesAMGPC::InitLevel
  

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
  } // StokesAMGPC::SetOptionsFromFlags
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix)
  {
  } // StokesAMGPC::ModifyOptions
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<TopologicMesh> StokesAMGPC<FACTORY, AUX_SYS> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh");
    RegionTimer rt(t);
    return BuildAlgMesh(BuildTopMesh());
  } // StokesAMGPC::BuildInitialMesh
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    BaseAMGPC::InitFinestLevel(finest_level);
  } // StokesAMGPC::InitFinestLevel
  

  template<class FACTORY, class AUX_SYS>
  Table<int> StokesAMGPC<FACTORY, AUX_SYS> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
    return Table<int>();
  } // StokesAMGPC::GetGSBlocks
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseAMGFactory> StokesAMGPC<FACTORY, AUX_SYS> :: BuildFactory ()
  {
    return make_shared<FACTORY>(static_pointer_cast<Options>(options));
  } // StokesAMGPC::BuildFactory
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseDOFMapStep> StokesAMGPC<FACTORY, AUX_SYS> :: BuildEmbedding (shared_ptr<TopologicMesh> mesh)
  {
    // consider fine_facet AND edge_sort
    throw Exception("Embedding TODO!!");
    return nullptr;
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

	 New local vertex numbering is simple:
	  [ pub from kp0 |  pub from kp1 | pub from kp2 | ... | local enum ]
     **/

    const auto & MA(*ma);
    auto aux_pds = aux_sys->GetAuxParDofs();
    auto all_dps = aux_pds->GetDistantProcs();
    auto comp_pds = aux_sys->GetCompParDofs();
    NgsAMG_Comm comm(comp_pds->GetCommunicator());

    // auto& fine_facet = *aux_sys->GetFineFacets();
    auto f2a_facet = aux_sys->GetFMapA2F();
    auto a2f_facet = aux_sys->GetFMapF2A();

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

    /** nr of elements + number of new vertices published from other sources ! **/
    const size_t nxpub = os_pub.Last();
    const size_t NV = nxpub + ma->GetNE();
    auto & vert_sort = node_sort[0]; vert_sort.SetSize(ma->GetNE()); // not sure about this one
    mesh->SetVs (NV, [&](auto vnr) -> FlatArray<int> {
	if (vnr < nxpub) {
	  int kp_orig = merge_pos_in_sorted_array(int(vnr), os_pub);
	  return (*dptabs[kp_orig])[vnr - os_pub[kp_orig]];
	}
	else
	  { return ovdps[vnr - nxpub]; }
      },
      [vert_sort, &nxpub](auto i, auto j) { vert_sort[i - nxpub] = j; } // not sure
      // [vert_sort](auto i, auto j) { vert_sort[i] = j; } // not sure
      );


    /** --- EDGES **/
    size_t n_edges = aux_pds->GetNDofLocal();
    Array<int> facet_els(2);
    auto & edge_sort = node_sort[1]; edge_sort.SetSize(ma->GetNFacets());
    mesh->SetNodes<NT_EDGE> (n_edges, [&](auto edge_num) LAMBDA_INLINE {
	INT<2> pair;
	auto fnr = f2a_facet[edge_num];
	ma->GetFacetElements(fnr, facet_els);
	if (facet_els.Size() == 1) {
	  auto p = aux_pds->GetDistantProcs(edge_num)[0];
	  if (comm.Rank() > p) { // other vertex has been published to me - I have to add this edge
	    auto kp = find_in_sorted_array(p, all_dps);
	    auto loc_fnr = find_in_sorted_array(int(edge_num), aux_pds->GetExchangeDofs(p));
	    pair[0] = os_pub[kp] + ex_data[kp][1 + loc_fnr];
	    pair[1] = nxpub + facet_els[0];
	  }
	}
	else {
	  if (facet_els[0] < facet_els[1]) {
	    pair[0] = nxpub + facet_els[0];
	    pair[1] = nxpub + facet_els[1];
	  } else {
	    pair[0] = nxpub + facet_els[1];
	    pair[1] = nxpub + facet_els[0];
	  }
	}
	return pair;
      },
      [&](auto edge_num, auto id) LAMBDA_INLINE {
	auto fnum = f2a_facet[edge_num];
	edge_sort[fnum] = id; // VERY unsure if this does something evil with additional edges...
      }
      );

    return mesh;
  } // StokesAMGPC::BuildTopMesh


  template<class FACTORY, class AUX_SYS>
  shared_ptr<typename StokesAMGPC<FACTORY, AUX_SYS>::TMESH> StokesAMGPC<FACTORY, AUX_SYS> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    return BuildAlgMesh_TRIV(top_mesh);
  } // StokesAMGPC::BuildAlgMesh


  /** END StokesAMGPC **/

} // namespace amg

#endif // FILE_AMG_PC_STOKES_IMPL_HPP
#endif // STOKES
