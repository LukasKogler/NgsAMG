#ifndef FILE_AMG_FACTORY_STOKES_IMPL_HPP
#define FILE_AMG_FACTORY_STOKES_IMPL_HPP

#include <utils_io.hpp>
#include <alg_mesh_nodes.hpp>

#include "stokes_map.hpp"
#include "stokes_factory.hpp"

#include "stokes_map_impl.hpp"
#include "utils.hpp"
namespace amg
{

INLINE void print_rank2(string name, FlatMatrix<double> A, LocalHeap & lh, ostream & os)
{
  HeapReset hr(lh);
  int N = A.Height();
  FlatMatrix<double> a(N, N, lh), evecs(N, N, lh); a = A;
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(a, evals, evecs);
  int neg = 0, pos = 0, rk = 0;
  double eps = 1e-10 * fabs(evals(N-1));
  double minpos = fabs(evals(N-1));
  for (auto k :Range(evals))
    if (evals(k) > eps)
{ pos++; rk++; minpos = min(evals(k), minpos); }
    else if (evals(k) < -eps)
{ neg++; rk++; }
  // os << " trace of " << name << " = " << calc_trace(A) << endl;
  // os << " rank of " << name << " = " << rk << " of " << N << ", pos = " << pos << ", neg = " << neg << ", min = " << evals(0) << ", max = " << evals(N-1) << ", all evals = "; prow(evals); cout << endl;
  os << " rank of " << name << " = " << rk << " of " << N << ", pos = " << pos << ", neg = " << neg << ", min = " << evals(0) << ", minpos = " << minpos << ", max = " << evals(N-1) << ", all evals = " << endl;
  prow2(evals);
  cout << endl << " evecs: " << endl << evecs << endl;
  // cout << " all evals are = "; prow(evals); cout << endl;
}

INLINE void print_rank2(string name, FlatMatrix<double> A, LocalHeap & lh)
{
  print_rank2(name, A, lh, cout);
}


/** Options **/

template<class TMESH, class ENERGY>
class StokesAMGFactory<TMESH, ENERGY> :: Options : public BaseAMGFactory::Options
                                                 , public AggOptions
#ifdef SPW_AGG
                                                 , public SPWAggOptions
#endif
#ifdef MIS_AGG
                                                 , public MISAggOptions
#endif
{
public:
  /** choice of supported coarsening algorithm **/
  enum CRS_ALG : int { // int instead of char for readability
#ifdef MIS_AGG
  #ifdef SPW_AGG
    SPW = 1,         // successive pairwise aggregaion
    MIS = 2          // MIS-based aggregaion
  #else // SPW_AGG
    MIS = 2
  #endif // SPW_AGG
#else // MIS_AGG
    SPW = 1 // SPW is guaranteed to be enabled in this case
#endif
  };
  SpecOpt<CRS_ALG> crs_alg = CRS_ALG::MIS;

  bool build_div_mats = false;
  bool check_loop_divs = false;

public:
  Options ()
    : BaseAMGFactory::Options()
  { ; }

  virtual void SetFromFlags (const Flags & flags, string prefix) override
  {
    BaseAMGFactory::Options::SetFromFlags(flags, prefix);

    SetAggFromFlags(flags, prefix);

    Array<CRS_ALG> crs_algs;
    Array<string> crs_alg_names;
#ifdef MIS_AGG
    SetMISFromFlags(flags, prefix);
    crs_algs.Append(MIS);
    crs_alg_names.Append("mis");
#endif
#ifdef SPW_AGG
    SetSPWFromFlags(flags, prefix);
    crs_algs.Append(SPW);
    crs_alg_names.Append("spw");
#endif
    crs_alg.SetFromFlagsEnum(flags, prefix + "crs_alg", crs_alg_names, crs_algs);

    auto & specfa = flags.GetStringListFlag(prefix + "crs_alg" + "_spec");

    build_div_mats = flags.GetDefineFlagX(prefix + "build_div_mats").IsTrue();
    check_loop_divs = flags.GetDefineFlagX(prefix + "check_loop_divs").IsTrue();

    build_div_mats |= check_loop_divs;

    this->keep_grid_maps |= check_loop_divs;
  }
}; // class StokesAMGFactory::Options

/** END Options **/


/** StokesAMGFactory **/

template<class TMESH, class ENERGY>
StokesAMGFactory<TMESH, ENERGY> :: StokesAMGFactory (shared_ptr<StokesAMGFactory<TMESH, ENERGY>::Options> _opts)
  : BASE_CLASS(_opts)
  , SecondaryAMGSequenceFactory()
{
  ;
} // StokesAMGFactory(..)


template<class TMESH, class ENERGY>
UniversalDofs
StokesAMGFactory<TMESH, ENERGY> :: BuildUDofs (BaseAMGFactory::LevelCapsule const &cap) const
{
  auto pmesh = my_dynamic_pointer_cast<TMESH>(cap.mesh,
    "StokesAMGFactory::BuildParallelDofs");

  // DWARAI/TODO: THIS only works as long as we have only one DOF per coarse edge
  // that is, as soon as I start adding actual preserved vectors, we need to overload this
  // for HDIV!!
  return pmesh->GetDofedEdgeUDofs(BS);
  // const auto & M = *pmesh;
  // TODO: add a GetDEParDofs that gets a vector "dofs_per_de" or sth I think
  // auto [dofed_edges, dof2e, e2dof] = M.GetDOFedEdges();
  // return M.GetDofedEdgeUDofs(BS);
  // auto de_pds = M.GetDEParDofs(BS);
  // return UniversalDofs(de_pds, e2dof.Size(), BS);
} // StokesAMGFactory::BuildUDofs


template<class TMESH, class ENERGY>
BaseAMGFactory::State* StokesAMGFactory<TMESH, ENERGY> :: AllocState () const
{
  return new BaseAMGFactory::State();
} // StokesAMGFactory::AllocState


template<class TMESH, class ENERGY>
shared_ptr<BaseAMGFactory::LevelCapsule> StokesAMGFactory<TMESH, ENERGY> :: AllocCap () const
{
  return make_shared<StokesLevelCapsule>();
} // StokesAMGFactory::AllocCap


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
StokesAMGFactory<TMESH, ENERGY>::
MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps,
          shared_ptr<BaseAMGFactory::AMGLevel> & f_lev,
          shared_ptr<BaseAMGFactory::AMGLevel> & c_lev)
{
  static Timer t("StokesAMGFactory::MapLevel");
  RegionTimer rt(t);

  if (c_lev->cap->mesh == f_lev->cap->mesh) // coarsening stuck ...
    { return nullptr; }

  auto & O(static_cast<Options&>(*options));

  auto & fcap = static_cast<StokesLevelCapsule&>(*f_lev->cap);
  auto & ccap = static_cast<StokesLevelCapsule&>(*c_lev->cap);

  shared_ptr<BaseDOFMapStep> final_step;

  if (fcap.baselevel == 0)
  {
    /**
     * The first map in the step can be the embedding, which can be a multi-dof step
     * To enable this to get properly concatenated with the coarse-map, we only give in the
     * primary space component of that here.
     * Note: That meanst that the original multi-dof step we have here gets lost and only
     *       the first component goes into the concatenated map.
     *       When we set up the Hiptmair smoother on the first level later, we still have
     *       access to the embed_map in f_lev and use the second component that way.
     */
    Array<shared_ptr<BaseDOFMapStep>> single_steps(dof_steps.Size());
    for (auto k : Range(single_steps))
    {
      auto asMulti = dynamic_pointer_cast<MultiDofMapStep>(dof_steps[k]);
      single_steps[k] = (asMulti == nullptr) ? dof_steps[k] : asMulti->GetMap(0);
    }

    final_step = MakeSingleStep2(single_steps);
    final_step->Finalize();

    fcap.savedDOFMaps = std::move(single_steps);
  }
  else
  {
    final_step = MakeSingleStep2(dof_steps);
    final_step->Finalize();

    fcap.savedDOFMaps.SetSize(dof_steps.Size());
    fcap.savedDOFMaps = dof_steps;
  }

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_single_dmap_rk_" + to_string(fcap.uDofs.GetCommunicator().Rank()) + "_l_" + to_string(fcap.baselevel) + ".out");
    out << *final_step << endl;
  }

  auto embed_map = f_lev->embed_map;

  if (embed_map != nullptr)
  {
    my_dynamic_pointer_cast<MultiDofMapStep>(embed_map, "Stokes PC must embed with multi-emb");
  }


  if (f_lev->level == 0)
  {
    if (f_lev->embed_map != nullptr)
    {
      if (auto multi_emb = dynamic_pointer_cast<MultiDofMapStep>(f_lev->embed_map))
        {
          auto cpm = static_pointer_cast<SparseMatrixTM<double>>(multi_emb->GetMap(1)->AssembleMatrix(fcap.mat));
          if (O.log_level >= Options::LOG_LEVEL::DBG) {
            ofstream out ("stokes_pmat_rk_" + to_string(f_lev->cap->eqc_h->GetCommunicator().Rank()) + "_l_0.out");
            print_tm_spmat(out, *cpm);
          }
          fcap.pot_mat = make_shared<SparseMatrix<double>>(std::move(*cpm));
        }
      else
        { throw Exception("Cannot get potential space mat on level 0!"); }
    }
    else
      { ProjectToPotSpace(fcap); }
  }

  ccap.mat = final_step->AssembleMatrix(fcap.mat);

  /**
   * Here is the question - there are two ways to get the coarse curl matrix.
   * These are not equivalent because pot/range prols and curl mats do not quite commute.
   * The second one is probably the better one. I could still Galerkin project in the potential space
   * later when I build the smoothers.
   */
  if (ccap.uDofs.IsValid()) // check if contracted out
  {
    ProjectToPotSpace(ccap);
  }

  if (O.build_div_mats)
  {
    // note: also sets up range-range-pardofs
    if (fcap.div_mat == nullptr)
      { BuildDivMat(fcap); }

    if ( ccap.mat != nullptr ) // check if contracted out
    {
      if (ccap.div_mat == nullptr)
        { BuildDivMat(ccap); }
    }
  }

  return final_step;
} // StokesAMGFactory::MapLevel



template<class TMESH, class ENERGY>
shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildCoarseMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
{
  // auto & O(static_cast<Options&>(*options));
  auto &O = *my_dynamic_pointer_cast<Options>(options,
    "StokesAMGFactory::BuildCoarseMap - options");

  auto slc = static_pointer_cast<StokesLevelCapsule>(mapped_cap);

  // typedef typename Options::CRS_ALG CA;
  // CA calg = O.crs_alg.GetOpt(state.level[0]);

  typename Options::CRS_ALG calg = O.crs_alg.GetOpt(state.level[0]);
  // return BuildAggMap(state, slc);

  switch(calg) {
    case(Options::CRS_ALG::MIS): { return BuildAggMap<MISAgglomerator<ENERGY, typename TMESH::T_MESH_W_DATA, ENERGY::NEED_ROBUST>>(state, slc); break; }
    case(Options::CRS_ALG::SPW): { return BuildAggMap<SPWAgglomerator<ENERGY, typename TMESH::T_MESH_W_DATA, ENERGY::NEED_ROBUST>>(state, slc); break; }
    default: { throw Exception("Invalid coarsen alg!"); break; }
  }

} // StokesAMGFactory::InitState


template<class TMESH, class ENERGY> template<class AGG_CLASS>
INLINE shared_ptr<StokesCoarseMap<TMESH>> StokesAMGFactory<TMESH, ENERGY> :: BuildAggMap (BaseAMGFactory::State & state, shared_ptr<StokesLevelCapsule> & mapped_cap)
{
  static Timer t("StokesAMGFactory::BuildAggMap");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);
  // typedef MISAgglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;

  const int level = state.level[0];

  auto mesh = my_dynamic_pointer_cast<TMESH>(state.curr_cap->mesh, "BuildAggMap - mesh");

  mesh->CumulateData();

  auto edges = mesh->template GetNodes<NT_EDGE>();
  Array<IVec<2,double>> olded(edges.Size());
  auto vdata = get<0>(mesh->Data())->Data();
  auto edata = get<1>(mesh->Data())->Data();
  auto ghost_verts = mesh->GetGhostVerts();

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_mesh_rk_" + to_string(mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
      + to_string(state.curr_cap->baselevel) + ".out");
    out << *mesh << endl;
  }
  // TODO: OFC I can only do fully solid agglomerates here!

  // for (auto v : Range(vdata.Size()))
  //   if ( (vdata[v].vol < 0) && ( (!fnodes || fnodes->Test(v)) ) ) {
  // 	int surfnr = -(vdata[v].vol + 1);
  // 	max_surf = max2(max_surf, surfnr);
  // 	auto pos = merge_pos_in_sorted_array(surfnr, free_surfs);
  // 	if ( (pos != -1) && (pos > 0) && (free_surfs[pos-1] == surfnr) )
  // 	  { ; }
  // 	else if (pos >= 0)
  // 	  { free_surfs.Insert(pos, surfnr); }
  //   }
  // Array<int> surf2row(1+max_surf); surf2row = -1;
  // for (auto k : Range(free_surfs))
  //   { surf2row[free_surfs[k]] = k; }
  // TableCreator<int> cfas(free_surfs.Size());
  // const int fss = free_surfs.Size();
  // for (; !cfas.Done(); cfas++) {
  //   for (auto v : Range(vdata.Size()))
  // 	if ( (vdata[v].vol < 0) && ( (!fnodes || fnodes->Test(v)) ) ) {
  // 	  int surfnr = -(vdata[v].vol + 1);
  // 	  cfas.Add(surf2row[surfnr], v);
  // 	}
  // }
  // auto faggs = cfas.MoveTable();


  // cout << " COARSENING LEVEL " << state.level[0] << endl;
  auto coarseMap = make_shared<DiscreteStokesCoarseMap<TMESH, AGG_CLASS>>(mesh);

  // set options
  coarseMap->Initialize(O, level);

  // set ghost vertices
  if ( (ghost_verts != nullptr) && (mesh->GetEQCHierarchy()->GetCommunicator().Size() > 2) ) {
    auto sverts = make_shared<BitArray>(*ghost_verts);
    sverts->Invert();
    // cout << " solid_verts: " << endl;
    // for (auto k : Range(sverts->Size())) {
    // 	cout << "(" << k << "::" << sverts->Test(k) << ") ";
    // }
    // cout << endl;
    coarseMap->SetSolidVerts(sverts);
  }

  // free vertices
  coarseMap->SetFreeVerts(state.curr_cap->free_nodes);


  /**
   * We need to set all edges with 0 flow to forbidden - otherwise
   * local Stokes problems can be unsolvable in prolongation
   * Happens when agglomerates are "not connected", i.e subsets are only connected by 0 flow edge(s)
   */
  int ne_zf = 0;
  shared_ptr<BitArray> nzf_edges = make_shared<BitArray>(mesh->template GetNN<NT_EDGE>());
  nzf_edges->Clear();
  for (auto k : Range(mesh->template GetNN<NT_EDGE>())) {
    if (is_zero(edata[k].flow)) {
      nzf_edges->SetBit(k);
      ne_zf++;
      cout << "ZF edge " << k << "/" << mesh->template GetNN<NT_EDGE>() << endl;
    }
  }
  if (ne_zf) {
    cout << endl << "NZF EDGES = " << ne_zf << endl;
    nzf_edges->Invert();
    coarseMap->SetAllowedEdges(nzf_edges);
  }

  /** set up special, fixed agglomerates for fictitious vertices (group by surf nr + EQC) **/
  auto fnodes = state.curr_cap->free_nodes;
  Table<int> faggs;
  {
    int max_surf = 0, neqcs = mesh->GetEQCHierarchy()->GetNEQCS();
    Array<int> free_surfs;
    auto it_vs = [&](auto lam) {
      mesh->template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto v) {
        if (vdata[v].vol < 0) {
          if (eqc == 0 || (ghost_verts == nullptr) || (ghost_verts->Test(v)) ) {
            if ( (fnodes == nullptr) || (fnodes->Test(v)) ) {
              int surfnr = -(vdata[v].vol + 1);
              lam(neqcs*surfnr+eqc, v);
            }
          }
        }
      }, false); // not master only
    };
    int max_bid = 0;
    it_vs([&](int bnd_id, auto vnum) {
      max_bid = max(max_bid, bnd_id);
    });
    Array<int> cnt_per_bid(1+max_bid); cnt_per_bid = 0;
    it_vs([&](int bnd_id, auto vnum) {
      cnt_per_bid[bnd_id]++;
    });
    Array<int> perow(1+max_bid);
    Array<int> compress_bid(1+max_bid); compress_bid = -1;
    int cnt = 0;
    for (auto k : Range(cnt_per_bid)) {
      if (cnt_per_bid[k] > 0) {
        perow[cnt] = cnt_per_bid[k];
        compress_bid[k] = cnt++;
      }
    }
    perow.SetSize(cnt);
    faggs = std::move(Table<int>(perow)); perow = 0;
    it_vs([&](int bnd_id, auto vnum) {
      auto rnr = compress_bid[bnd_id];
      faggs[rnr][perow[rnr]++] = vnum;
    });
  }

  coarseMap->SetFixedAggs(std::move(faggs));

  // for (const auto & edge : edges) {
  //   if ( (vdata[edge.v[0]].vol < 0) || (vdata[edge.v[1]].vol < 0) ) {
  // 	SetScalIdentity(olded[edge.id][0], edata[edge.id].edi);
  // 	SetScalIdentity(olded[edge.id][1], edata[edge.id].edj);
  //   }
  // }

  auto cmesh = coarseMap->GetMappedMesh();
  mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
  mapped_cap->mesh  = cmesh;
  // mapped_cap->pardofs = this->BuildParallelDofs(cmesh);

  // NOTE: I moved some finishing-up of the coarse mesh (loops, UDofs, etc. ) to buildDOFmap
  //       because the HDIV can only do it at that point!

  // TODO: OFC I can only do fully solid agglomerates here!
  // cout << " crs mesh (on level " << state.level[0] + 1 << ") = " << endl;
  // cout << *ctm << endl;
  // cout << endl;

  return coarseMap;
} // StokesAMGFactory::BuildAggMap


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
StokesAMGFactory<TMESH, ENERGY> :: BuildCoarseDOFMap (shared_ptr<BaseCoarseMap> cmap,
                                                      shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
                                                      shared_ptr<BaseAMGFactory::LevelCapsule> ccap,
                                                      shared_ptr<BaseDOFMapStep> embMap)
{

  static Timer t("StokesAMGFactory::BuildCoarseDOFMap");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  shared_ptr<StokesLevelCapsule> fc = dynamic_pointer_cast<StokesLevelCapsule>(fcap);

  if (fc == nullptr)
    { throw Exception("Wrong fine Cap!"); }

  shared_ptr<StokesLevelCapsule> cc = dynamic_pointer_cast<StokesLevelCapsule>(ccap);

  if (cc == nullptr)
    { throw Exception("Wrong crs  Cap!"); }

  if (cmap == nullptr)
  {
    std::cout << " HAVE NO CMAP!" << std::endl;
  }

  auto stokes_cmap = my_dynamic_pointer_cast<StokesCoarseMap<TMESH>>(cmap, "StokesAMGFactory::BuildCoarseDOFMap - camp");

  // Array<shared_ptr<BaseDOFMapStep>> step_comps(2);
  // step_comps[0] = RangePWProl(stokes_cmap, fc, cc);
  // step_comps[1] = PotPWProl(stokes_cmap, fc, cc, static_pointer_cast<ProlMap<TSPM_TM>>(step_comps[0]));
  // auto multi_step = make_shared<MultiDofMapStep>(step_comps);

  auto step = RangeProlMap(stokes_cmap, fc, cc);

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_crs_dmap_rk_" + to_string(fc->uDofs.GetCommunicator().Rank()) + "_l_" + to_string(fcap->baselevel) + ".out");
    out << *step << endl;
  }

  return step;
} // StokesAMGFactory::BuildCoarseDOFMap


template<class TMESH, class ENERGY>
void
StokesAMGFactory<TMESH, ENERGY> :: FinalizeCoarseMap (StokesLevelCapsule     const &fCap,
                                                      StokesLevelCapsule           &cCap,
                                                      StokesCoarseMap<TMESH>       &cMap)
{
  static Timer t("StokesAMGFactory :: FinalizeCoarseMap");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  auto cMesh = my_dynamic_pointer_cast<TMESH>(cCap.mesh, "FinalizeCoarseMap - CMESH");

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    auto const rk = cMesh->GetEQCHierarchy()->GetCommunicator().Rank();

    ofstream out ("stokes_cmap_rk_" + to_string(rk) + "_l_" + to_string(fCap.baselevel) + "_preFinal.out");
    cMap.PrintTo(out, "");

    ofstream meshout ("stokes_mesh_rk_" + to_string(rk) + "_l_" + to_string(fCap.baselevel + 1) + "_preFinal.out");
    cMesh->BlockTM::printTo(meshout);
  }

  // loops
  cMap.MapAdditionalDataB();

  // DAWARI/TODO: uargh, HDIV can only do this AFTER the coarse DOFMap is complete
  // TEMPORARILY (!!) just re-do it in hdiv
  cCap.uDofs = this->BuildUDofs(cCap);
  BuildPotUDofs(cCap);

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    auto const rk = cMesh->GetEQCHierarchy()->GetCommunicator().Rank();
    ofstream out ("stokes_cmap_rk_" + to_string(rk) + "_l_" + to_string(fCap.baselevel) + ".out");
    cMap.PrintTo(out, "");
  }
} // StokesAMGFactory::FinalizeCoarseMap


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
StokesAMGFactory<TMESH, ENERGY> :: RangeProlMap (shared_ptr<StokesCoarseMap<TMESH>> cmap,
                                                 shared_ptr<StokesLevelCapsule> fcap,
                                                 shared_ptr<StokesLevelCapsule> ccap)
{
  static Timer t("StokesAMGFactory::RangeProlMap");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  /** Prolongation for HDiv-like space **/
  auto fmesh = static_pointer_cast<TMESH>(cmap->GetMesh());
  auto cmesh = static_pointer_cast<TMESH>(cmap->GetMappedMesh());

  auto vmap = cmap->template GetMap<NT_VERTEX>();
  auto emap = cmap->template GetMap<NT_EDGE>();

  // // why not C2F<NT_VERTEX>??
  // TableCreator<int> cva(cmesh->template GetNN<NT_VERTEX>());
  // for (; !cva.Done(); cva++) {
  //   for (auto k : Range(vmap))
  //     if (vmap[k] != -1)
  //       { cva.Add(vmap[k], k); }
  // }
  // auto v_aggs = cva.MoveTable();

  auto v_aggs = cmap->template GetMapC2F<NT_VERTEX>();

  auto spA = dynamic_pointer_cast<TSPM>(fcap->mat);

  auto pwprol = BuildPrimarySpaceProlongation(*fcap, *ccap, *cmap,
                                              *fmesh, *cmesh,
                                              vmap, emap, v_aggs,
                                              spA.get());

  FinalizeCoarseMap(*fcap, *ccap, *cmap);

  return make_shared<ProlMap<TM>>(pwprol, fcap->uDofs, ccap->uDofs);
} // StokesAMGFactory::RangeProlMap




template<class TMESH, class ENERGY>
void StokesAMGFactory<TMESH, ENERGY> :: ProjectToPotSpace (StokesLevelCapsule& cap) const
{
  static Timer t("StokesAMGFactory::ProjectToPotSpace");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  const auto & M = *static_pointer_cast<TMESH>(cap.mesh);
  // shared_ptr<TSPM_TM> range_mat = static_pointer_cast<TSPM_TM>();
  // shared_ptr<TCM_TM> curl_mat = static_pointer_cast<TCM_TM>(cap.curl_mat);
  // shared_ptr<TCTM_TM> curl_mat_T = static_pointer_cast<TCTM_TM>(cap.curl_mat_T);
  if (cap.curl_mat == nullptr)
    { BuildCurlMat(cap); }


  if (O.log_level >= Options::LOG_LEVEL::DBG)
  {
    ofstream out ("stokes_rmat_rk_" + to_string(M.GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(cap.baselevel) + ".out");

    auto spm = my_dynamic_pointer_cast<TSPM>(cap.mat, "ProjectToPotSpace - mat");

    print_tm_spmat(out, *spm);
  }

  // cout << "L " << cap.baselevel << ",  range_mat " << endl; print_tm_spmat(cout, *cmtm); cout << endl;

  // auto RC = MatMultAB(*cmtm, (TCM_TM&)(*cap.curl_mat));
  // shared_ptr<SparseMatrix<double>> pot_mat = MatMultAB((TCTM_TM&)(*cap.curl_mat_T), *RC);

  auto [ AC, pot_mat_t ] = RestrictMatrixKeepFactor(*cap.mat, *cap.curl_mat, *cap.curl_mat_T);
  // auto AC = MatMultABGeneric(cap.mat, cap.curl_mat);

  // auto pot_mat_t = MatMultABGeneric(cap.curl_mat_T, AC);

  auto pot_mat = my_dynamic_pointer_cast<SparseMatrix<double>>(pot_mat_t);

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_pmat_rk_" + to_string(M.GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(cap.baselevel) + ".out");
    print_tm_spmat(out, *pot_mat);
  }

  // cout << "L " << cap.baselevel << ", pot_mat " << endl; print_tm_spmat(cout, *pot_mat); cout << endl;

  // cap.pot_mat = make_shared<TPM>(std::move(*pot_mat));
  cap.pot_mat = pot_mat;
  cap.AC      = AC;
  cap.pot_freedofs = M.GetActiveLoops();

  {
    auto loops = M.GetLoops();
    auto ne = M.template GetNN<NT_EDGE>();
    auto nv = M.template GetNN<NT_VERTEX>();
    auto nl = M.GetLoops().Size();
    // cout << " NE, NL, NL/NE " << ne << " " << nl << " " << double(nl)/ne << endl;
    // cout << " NE/NV " << ne/double(nv) << endl;
    double all = 0.0;
    for (auto k : Range(loops)) {
all += loops[k].Size();
    }
    all /= loops.Size();
    // cout << " avg e per loop: " << all << endl;
    double eprr = GetScalNZE(cap.mat.get())/(BS*double(ne)), eprp = GetScalNZE(cap.pot_mat.get())/double(nl);
    // cout << " epr range, pot " << eprr << " " << eprp << endl << endl << endl;
    // cout << " LOOPS " << endl << loops << endl;
  }

} // StokesAMGFactory::ProjectToPotSpace


template<class TMESH, class ENERGY>
void StokesAMGFactory<TMESH, ENERGY> :: BuildPotUDofs (StokesLevelCapsule& cap) const
{
  const auto & M = *static_pointer_cast<TMESH>(cap.mesh);
  // cap.pot_uDofs = UniversalDofs(M.GetLoopParDofs(), M.GetLoops().Size(), 1);
  cap.pot_uDofs = M.GetLoopUDofs();
} // StokesAMGFactory::BuildPotUDofs


/** Contract **/
template<class TMESH, class ENERGY> shared_ptr<BaseGridMapStep>
StokesAMGFactory<TMESH, ENERGY> :: BuildContractMap (double factor,
                                                     shared_ptr<TopologicMesh> mesh,
                                                     shared_ptr<BaseAMGFactory::LevelCapsule> & b_mapped_cap) const
{
  static Timer t("BuildContractMap"); RegionTimer rt(t);
  if (mesh == nullptr)
    { throw Exception("BuildContractMap needs a mesh!"); }

  auto m = my_dynamic_pointer_cast<TMESH>(mesh, "StokesAMGFactory::BuildContractMap - mesh");
  auto mapped_cap = my_dynamic_pointer_cast<StokesLevelCapsule>(b_mapped_cap, "StokesAMGFactory::BuildContractMap - cap");

  // at least 2 groups - dont send everything from 1 to 0 for no reason
  int n_groups = (factor == -1) ? 2 : max2(int(2), int(1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * factor)));
  Table<int> groups = PartitionProcsMETIS (*m, n_groups);

  auto cm = make_shared<StokesContractMap<TMESH>>(std::move(groups), m);

  mapped_cap->mesh = cm->GetMappedMesh();
  mapped_cap->free_nodes = nullptr;
  mapped_cap->eqc_h = cm->IsMaster() ? mapped_cap->mesh->GetEQCHierarchy() : nullptr;
  mapped_cap->uDofs = UniversalDofs(); // is set in BuildDOFContractMap

  return cm;
} // StokesAMGFactory::BuildContractMap


template<class TMESH, class ENERGY>
shared_ptr<GridContractMap> StokesAMGFactory<TMESH, ENERGY> :: AllocateContractMap (Table<int> && groups, shared_ptr<TMESH> mesh) const
{
  return make_shared<StokesContractMap<TMESH>>(std::move(groups), mesh);
} // StokesAMGFactory::AllocateContractMap

template<class TMESH, class ENERGY>
void
StokesAMGFactory<TMESH, ENERGY> ::
DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  auto &O = static_cast<Options&>(*options);
  // if (O.check_loop_divs)
    // { CheckLoopDivs(amg_levels, map); }
} // StokesAMGFactory::CheckKVecs


template<class TMESH, class ENERGY> void
StokesAMGFactory<TMESH, ENERGY> :: CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  throw Exception("StokesAMGFactory::CheckLoopDivs called!");
} // StokesAMGFactory::ChekLoopDivs


/** END StokesAMGFactory **/

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP


