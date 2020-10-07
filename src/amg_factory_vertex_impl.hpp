#ifndef FILE_AMG_FACTORY_VERTEX_IMPL_HPP
#define FILE_AMG_FACTORY_VERTEX_IMPL_HPP

#include "amg_agg.hpp"
#include "amg_spwagg.hpp"
#include "amg_bla.hpp"

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
    enum CRS_ALG : char {
      ECOL = 0,             // edge collapsing
      AGG = 1               // MIS-based aggregaion
#ifdef SPWAGG
      , SPW = 2             // successive pairwise aggregaion
#endif
    };
    SpecOpt<CRS_ALG> crs_alg = AGG;

    /** General coarsening **/
    SpecOpt<bool> ecw_geom = true;              // use geometric instead of harmonic mean when determining strength of connection
    SpecOpt<bool> ecw_robust = true;            // use more expensive, but also more robust edge weights
    SpecOpt<xbool> ecw_minmax = xbool(maybe);
    SpecOpt<xbool> ecw_stab_hack = xbool(maybe);
    SpecOpt<double> min_ecw = 0.05;
    SpecOpt<double> min_vcw = 0.3;

    /** Smoothed Prolongation **/
    SpecOpt<bool> sp_aux_only = false;           // smooth prolongation using only auxiliary matrix
    SpecOpt<bool> newsp = true;
    SpecOpt<int> sp_max_per_row_classic = 5;     // maximum entries per row (should be >= 2!) where " newst" uses classic

    /** Discard **/
    int disc_max_bs = 5;

    /** AGG **/
    SpecOpt<bool> agg_neib_boost = false;
    SpecOpt<bool> lazy_neib_boost = false;
    SpecOpt<bool> print_aggs = false;            // print agglomerates (for debugging purposes)
    SpecOpt<AVG_TYPE> agg_minmax_avg;

#ifdef SPWAGG
    /** SPW-AGG **/
    SpecOpt<int> spw_rounds = 3;
    SpecOpt<bool> spw_allrobust = true;
    SpecOpt<bool> spw_checkbigsoc = true;
    SpecOpt<SPW_CW_TYPE> spw_pick_cwt  = MINMAX;
    SpecOpt<AVG_TYPE> spw_pick_mma_scal  = GEOM;
    SpecOpt<AVG_TYPE> spw_pick_mma_mat  = GEOM;
    SpecOpt<SPW_CW_TYPE> spw_check_cwt = HARMONIC;
    SpecOpt<AVG_TYPE> spw_check_mma_scal  = GEOM;
    SpecOpt<AVG_TYPE> spw_check_mma_mat  = GEOM;
#endif // SPWAGG

  public:

    VertexAMGFactoryOptions ()
      : BaseAMGFactory::Options()
    { ; }

    virtual void SetFromFlags (const Flags & flags, string prefix) override
    {

      auto pfit = [&](string x) LAMBDA_INLINE { return prefix + x; };

      BaseAMGFactory::Options::SetFromFlags(flags, prefix);

#ifdef SPWAGG
      SetEnumOpt(flags, crs_alg, pfit("crs_alg"), { "ecol", "agg", "spw" }, { ECOL, AGG, SPW });
#else // SPWAGG
      SetEnumOpt(flags, crs_alg, pfit("crs_alg"), { "ecol", "agg" }, { ECOL, AGG });
#endif // SPWAGG

      ecw_geom.SetFromFlags(flags, prefix + "ecw_geom");
      ecw_robust.SetFromFlags(flags, prefix + "ecw_robust");
      ecw_minmax.SetFromFlags(flags, prefix + "ecw_minmax");
      ecw_stab_hack.SetFromFlags(flags, prefix + "ecw_stab_hack");
      agg_minmax_avg.SetFromFlagsEnum(flags, prefix + "agg_minmax_avg", {"min", "geom", "harm", "alg", "max"});

      min_ecw.SetFromFlags(flags, prefix + "edge_thresh");
      min_vcw.SetFromFlags(flags, prefix + "vert_thresh");
      min_vcw.SetFromFlags(flags, prefix + "vert_thresh");
      agg_neib_boost.SetFromFlags(flags, prefix + "agg_neib_boost");
      lazy_neib_boost.SetFromFlags(flags, prefix + "lazy_neib_boost");
      print_aggs.SetFromFlags(flags, prefix + "print_aggs");

      sp_aux_only.SetFromFlags(flags, prefix + "sp_aux_only");
      newsp.SetFromFlags(flags, prefix + "newsp");
      sp_max_per_row_classic.SetFromFlags(flags, prefix + "sp_max_per_row_classic");

#ifdef SPWAGG
      spw_rounds.SetFromFlags(flags, prefix + "spw_rounds");
      spw_allrobust.SetFromFlags(flags, prefix +  "spw_check_robust");
      spw_checkbigsoc.SetFromFlags(flags, prefix + "spw_checkbigsoc");
      spw_pick_cwt.SetFromFlagsEnum(flags, prefix + "", {"harm", "geom", "mmx"});
      spw_pick_mma_scal.SetFromFlagsEnum(flags, prefix + "", {"min", "geom", "harm", "alg", "max"});
      spw_pick_mma_mat.SetFromFlagsEnum(flags, prefix + "", {"min", "geom", "harm", "alg", "max"});
      spw_check_cwt.SetFromFlagsEnum(flags, prefix + "", {"harm", "geom", "mmx"});
      spw_check_mma_scal.SetFromFlagsEnum(flags, prefix + "", {"min", "geom", "harm", "alg", "max"});
      spw_check_mma_mat.SetFromFlagsEnum(flags, prefix + "", {"min", "geom", "harm", "alg", "max"});
#endif // SPWAGG

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
  void VertexAMGFactory<ENERGY, TMESH, BS> :: InitState (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::AMGLevel> & lev) const
  {
    BASE_CLASS::InitState(state, lev);

    auto & s(static_cast<State&>(state));
    s.crs_opts = nullptr;
  } // VertexAMGFactory::InitState


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildCoarseMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
  {
    auto & O(static_cast<Options&>(*options));

    Options::CRS_ALG calg = O.crs_alg.GetOpt(state.level[0]);

    switch(calg) {
    case(Options::CRS_ALG::AGG): { return BuildAggMap(state, mapped_cap); break; }
    case(Options::CRS_ALG::ECOL): { return BuildECMap(state, mapped_cap); break; }
#ifdef SPWAGG
    case(Options::CRS_ALG::SPW): { return BuildSPWAggMap(state, mapped_cap); break; }
#endif
    default: { throw Exception("Invalid coarsen alg!"); break; }
    }

    return nullptr;
  } // VertexAMGFactory::BuildCoarseMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
  {
    auto & O = static_cast<Options&>(*options);
    typedef Agglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;
    typename AGG_CLASS::Options agg_opts;
    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_cap->mesh).name() + string(" for BuildAggMap!")); }

    const int level = state.level[0];

    agg_opts.edge_thresh = O.min_ecw.GetOpt(level);
    agg_opts.vert_thresh = O.min_vcw.GetOpt(level);
    agg_opts.cw_geom = O.ecw_geom.GetOpt(level);
    agg_opts.neib_boost = O.agg_neib_boost.GetOpt(level);
    agg_opts.lazy_neib_boost = O.lazy_neib_boost.GetOpt(level);
    agg_opts.robust = O.ecw_robust.GetOpt(level);
    agg_opts.use_stab_ecw_hack = O.ecw_stab_hack.GetOpt(level);
    agg_opts.use_minmax_soc = O.ecw_minmax.GetOpt(level);
    agg_opts.dist2 = O.d2_agg.GetOpt(level);
    agg_opts.print_aggs = O.print_aggs.GetOpt(level);
    agg_opts.minmax_avg = O.agg_minmax_avg.GetOpt(level);
    // auto agglomerator = make_shared<Agglomerator<FACTORY>>(mesh, state.free_nodes, move(agg_opts));

    auto agglomerator = make_shared<AGG_CLASS>(mesh, state.curr_cap->free_nodes, move(agg_opts));

    /** Set mapped Capsule **/
    auto cmesh = agglomerator->GetMappedMesh();
    mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
    mapped_cap->mesh = cmesh;
    mapped_cap->pardofs = this->BuildParallelDofs(cmesh);

    return agglomerator;
  } // VertexAMGFactory::BuildCoarseMap


#ifdef SPWAGG
  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildSPWAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
  {
    cout << " BuildSPWAggMap " << endl;
    auto & O = static_cast<Options&>(*options);
    typedef SPWAgglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;
    typename AGG_CLASS::Options agg_opts;
    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_cap->mesh).name() + string(" for BuildAggMap!")); }

    const int level = state.level[0];

    cout << " BuildSPWAggMap2 " << endl;
    agg_opts.edge_thresh = O.min_ecw.GetOpt(level);
    agg_opts.vert_thresh = O.min_vcw.GetOpt(level);
    agg_opts.robust = O.ecw_robust.GetOpt(level);
    agg_opts.num_rounds = O.spw_rounds.GetOpt(level);
    agg_opts.allrobust = O.spw_allrobust.GetOpt(level);
    agg_opts.pick_cw_type = O.spw_pick_cwt.GetOpt(level);
    agg_opts.pick_mma_scal = O.spw_pick_mma_scal.GetOpt(level);
    agg_opts.pick_mma_mat = O.spw_pick_mma_mat.GetOpt(level);
    agg_opts.check_cw_type = O.spw_check_cwt.GetOpt(level);
    agg_opts.check_mma_scal = O.spw_check_mma_scal.GetOpt(level);
    agg_opts.check_mma_mat = O.spw_check_mma_mat.GetOpt(level);
    agg_opts.print_aggs = O.print_aggs.GetOpt(level);
    agg_opts.checkbigsoc = O.spw_checkbigsoc.GetOpt(level);
    agg_opts.use_stab_ecw_hack = O.ecw_stab_hack.GetOpt(level);
    
    cout << " BuildSPWAggMap3 " << endl;
    auto agglomerator = make_shared<AGG_CLASS>(mesh, state.curr_cap->free_nodes, move(agg_opts));

    cout << " BuildSPWAggMap4 " << endl;
    /** Set mapped Capsule **/
    auto cmesh = agglomerator->GetMappedMesh();
    mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
    mapped_cap->mesh = cmesh;
    mapped_cap->pardofs = this->BuildParallelDofs(cmesh);
    cout << " BuildSPWAggMap5 " << endl;

    return agglomerator;
  } // VertexAMGFactory::BuildSPWAggMap
#endif


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseCoarseMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildECMap (BaseAMGFactory::State & astate, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
  {
    // throw Exception("finish this up ...");
    auto & O = static_cast<Options&>(*options);
    auto & state(static_cast<State&>(astate));
    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_cap->mesh).name() + string(" for BuildECMap!")); }

    shared_ptr<typename HierarchicVWC<TMESH>::Options> coarsen_opts;

    coarsen_opts = make_shared<typename HierarchicVWC<TMESH>::Options>();
    coarsen_opts->free_verts = state.curr_cap->free_nodes;
    coarsen_opts->min_vcw = O.min_vcw.GetOpt(100); // TODO: placeholder
    coarsen_opts->min_ecw = O.min_vcw.GetOpt(100);

    if (O.ecw_robust.GetOpt(100)) // TODO: placeholder
      { CalcECOLWeightsRobust (state, coarsen_opts->vcw, coarsen_opts->ecw); }
    else
      { CalcECOLWeightsSimple (state, coarsen_opts->vcw, coarsen_opts->ecw); }

    auto calg = make_shared<BlockVWC<TMESH>>(coarsen_opts);

    auto grid_step = calg->Coarsen(mesh);

    /** Set mapped Capsule **/
    auto cmesh = grid_step->GetMappedMesh();
    mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
    mapped_cap->mesh = cmesh;
    mapped_cap->pardofs = this->BuildParallelDofs(cmesh);

    return grid_step;
  } // VertexAMGFactory::BuildCoarseMap


  template<class ENERGY, class TMESH, int BS>
  void VertexAMGFactory<ENERGY, TMESH, BS> :: CalcECOLWeightsSimple (BaseAMGFactory::State & state, Array<double> & vcw, Array<double> & ecw)
  {
    const auto & O = static_cast<Options&>(*options);

    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
    const auto & M(*mesh); M.CumulateData();

    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();

    vcw.SetSize(M.template GetNN<NT_VERTEX>());
    ecw.SetSize(M.template GetNN<NT_EDGE>());

    // vcw = 0;
    M.template Apply<NT_VERTEX>([&](auto v) { vcw[v] = ENERGY::GetApproxVWeight(vdata[v]); }, true);
    M.template Apply<NT_EDGE>([&](const auto & edge) {
	auto awt = ENERGY::GetApproxWeight(edata[edge.id]);
	vcw[edge.v[0]] += awt;
	vcw[edge.v[1]] += awt;
      }, true);
    M.template AllreduceNodalData<NT_VERTEX>(vcw, [&](auto & in) { return sum_table(in); });

    M.template Apply<NT_VERTEX>([&](auto v) { vcw[v] = (vcw[v] == 0) ? 1.0 : ENERGY::GetApproxVWeight(vdata[v]) / vcw[v]; }, false);
    if (O.ecw_geom.GetOpt(100)) { // TODO::placeholder
      M.template Apply<NT_EDGE>([&](const auto & edge) {
	  double vw0 = vcw[edge.v[0]], vw1 = vcw[edge.v[1]];
	  ecw[edge.id] = ENERGY::GetApproxWeight(edata[edge.id]) / sqrt(vw0 * vw1);
	}, false);
    }
    else {
      M.template Apply<NT_EDGE>([&](const auto & edge) {
	  double vw0 = vcw[edge.v[0]], vw1 = vcw[edge.v[1]];
	  ecw[edge.id] = ENERGY::GetApproxWeight(edata[edge.id]) * (vw0 + vw1) / (2 * vw0 * vw1);
	}, false);
    }
  } // VertexAMGFactory::CalcECOLWeightsSimple


  template<class ENERGY, class TMESH, int BS>
  void VertexAMGFactory<ENERGY, TMESH, BS> :: CalcECOLWeightsRobust (BaseAMGFactory::State & state, Array<double> & vcw, Array<double> & ecw)
  {
    if constexpr(ENERGY::NEED_ROBUST == false) {
      CalcECOLWeightsSimple(state, vcw, ecw);
    }
    else {
      // #ifndef ELASTICITY_ROBUST_ECW
      // CalcECOLWeightsSimple(state, vcw, ecw);
      // #else
      const auto & O = static_cast<Options&>(*options);

      auto mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
      const auto & M(*mesh); M.CumulateData();

      auto vdata = get<0>(M.Data())->Data();
      auto edata = get<1>(M.Data())->Data();

      vcw.SetSize(M.template GetNN<NT_VERTEX>());
      ecw.SetSize(M.template GetNN<NT_EDGE>());

      Array<typename ENERGY::TM> diag_mats(M.template GetNN<NT_VERTEX>());

      M.template Apply<NT_VERTEX>([&](auto v) { diag_mats[v] = ENERGY::GetVMatrix(vdata[v]); });

      TM Qij, Qji;
      SetIdentity(Qij); SetIdentity(Qji);
      M.template Apply<NT_EDGE>([&](const auto & edge) {
	  typename ENERGY::TVD &vdi = vdata[edge.v[0]], &vdj = vdata[edge.v[1]];
	  ENERGY::ModQs(vdi, vdj, Qij, Qji);
	  Add_AT_B_A(1.0, diag_mats[edge.v[0]], Qij, ENERGY::GetEMatrix(edata[edge.id]));
	  Add_AT_B_A(1.0, diag_mats[edge.v[1]], Qji, ENERGY::GetEMatrix(edata[edge.id]));
	}, true);
      M.template AllreduceNodalData<NT_VERTEX>(diag_mats, [&](auto & in) { return sum_table(in); });

      TM A, B;
      if (O.ecw_geom.GetOpt(100)) { // TODO: placeholder
	M.template Apply<NT_EDGE>([&](const auto & edge) {
	    typename ENERGY::TVD &vdi = vdata[edge.v[0]], &vdj = vdata[edge.v[1]];
	    ENERGY::ModQs(vdi, vdj, Qij, Qji);
	    A = AT_B_A(Qji, diag_mats[edge.v[0]]);
	    B = AT_B_A(Qij, diag_mats[edge.v[1]]);
	    ecw[edge.id] = MIN_EV_FG ( A, B, Qij, Qji, ENERGY::GetEMatrix(edata[edge.id]) );
	  }, false);
      }
      else {
	M.template Apply<NT_EDGE>([&](const auto & edge) {
	    typename ENERGY::TVD &vdi = vdata[edge.v[0]], &vdj = vdata[edge.v[1]];
	    ENERGY::ModQs(vdi, vdj, Qij, Qji);
	    A = AT_B_A(Qji, diag_mats[edge.v[0]]);
	    B = AT_B_A(Qij, diag_mats[edge.v[1]]);
	    ecw[edge.id] = MIN_EV_HARM ( A, B, ENERGY::GetEMatrix(edata[edge.id]) );
	  }, false);
      }

      M.template Apply<NT_VERTEX>([&](auto v) {
	  auto tr = calc_trace(diag_mats[v]);
	  vcw[v] = (tr == 0) ? 1.0 : BS * ENERGY::GetApproxVWeight(vdata[v]) / tr;
	}, false);
      // #endif
    } // VertexAMGFactory::CalcECOLWeightsSimple
  }

  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: PWProlMap (shared_ptr<BaseCoarseMap> cmap,
									       shared_ptr<BaseAMGFactory::LevelCapsule> fcap, shared_ptr<BaseAMGFactory::LevelCapsule> ccap)
  {
    static Timer t("PWProlMap"); RegionTimer rt(t);

    shared_ptr<ParallelDofs> fpds = fcap->pardofs, cpds = ccap->pardofs;

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
    // cout << "vmap: " << endl; prow(vmap); cout << endl;

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
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap, shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
  {
    Options &O (static_cast<Options&>(*options));
    if (O.newsp.GetOpt(100)) // TODO: placeholder
      return SmoothedProlMap_impl_v2(static_pointer_cast<ProlMap<TSPM_TM>>(pw_step), cmap, fcap);
    else
      return SmoothedProlMap_impl(pw_step, cmap, fcap);
  } // VertexAMGFactory::SmoothedProlMap


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap_impl_v2 (shared_ptr<ProlMap<TSPM_TM>> pw_step, shared_ptr<BaseCoarseMap> cmap,
											     shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
  {
    static Timer t("SmoothedProlMap_hyb"); RegionTimer rt(t);
    /** Use fine level matrix to smooth prolongation ("classic prol") where we can, that is whenever:
	   I) We would not break MAX_PER_ROW, so if all algebraic neibs map to <= MAX_PER_ROW coarse verts.
	  II) We would not break the hierarchy, so if all algebraic neibs map to coarse verts in the same, or higher EQCs
	Where we cannot, smooth using the replacement matrix ("aux prol"). **/
    Options &O (static_cast<Options&>(*options));

    const int baselevel = fcap->baselevel;
    const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(baselevel);
    const int MAX_PER_ROW = O.sp_max_per_row.GetOpt(baselevel);
    const int MAX_PER_ROW_CLASSIC = O.sp_max_per_row_classic.GetOpt(baselevel);
    const double omega = O.sp_omega.GetOpt(baselevel);
    const bool aux_only = O.sp_aux_only.GetOpt(baselevel); // TODO:placeholder

    // NOTE: something is funky with the meshes here ... 
    const auto & FM = *static_pointer_cast<TMESH>(fcap->mesh);
    const auto & CM = *static_pointer_cast<TMESH>(cmap->GetMappedMesh());
    const auto & eqc_h = *FM.GetEQCHierarchy();
    const int neqcs = eqc_h.GetNEQCS();
    const auto & fecon = *FM.GetEdgeCM();
    auto fpds = pw_step->GetParDofs();
    auto cpds = pw_step->GetMappedParDofs();
    
    FM.CumulateData();
    CM.CumulateData();

    /** Because of embedding, this can be nullptr for level 0!
	I think using pure aux on level 0 should not be an issue. **/
    auto fmat = dynamic_pointer_cast<TSPM_TM>(fcap->mat);
    /** "fmat" can be the pre-embedded matrix. In that case we can't use it for smoothing. **/
    bool have_fmat = (fmat != nullptr);
    /** if "fmat" has no pardofs, it is not the original finest level matrix, so fine to use! !**/
    if ( have_fmat && (fmat->GetParallelDofs() != nullptr) )
      { have_fmat &= (fmat->GetParallelDofs() == fpds); }

    // cout << " have fmat " << have_fmat << (fmat != nullptr) << (fmat->GetParallelDofs() == fpds) << endl;
    // cout << " pds " << fmat->GetParallelDofs() << fpds << endl;
    
    const TSPM_TM & pwprol = *pw_step->GetProl();

    auto vmap = cmap->GetMap<NT_VERTEX>();

    const size_t FNV = FM.template GetNN<NT_VERTEX>(), CNV = CM.template GetNN<NT_VERTEX>();

    LocalHeap lh(2000000, "muchmemory", false); // ~2 MB LocalHeap

    /** Find Graph, decide if classic or aux prol. **/
    Array<int> cols(20); cols.SetSize0();
    BitArray use_classic(FNV); use_classic.Clear();
    Array<int> proltype(FNV); proltype = 0;

    auto fvdata = get<0>(FM.Data())->Data();
    auto fedata = get<1>(FM.Data())->Data();

    auto fedges = FM.template GetNodes<NT_EDGE>();

    // cout << "FM: " << endl << FM << endl;
    // cout << "fecon: " << endl << fecon << endl;

    auto get_cols_classic = [&](auto eqc, auto fvnr) {
      if (eqc != 0) // might still have non-hierarchic neibs on other side!
	{ return false; }
      auto ovs = fecon.GetRowIndices(fvnr);
      int nniscv = 0; // number neibs in same cv
      auto cvnr = vmap[fvnr];
      for (auto v : ovs)
	if (vmap[v] == cvnr)
	  { nniscv++; }
      if (nniscv == 0) { // no neib that maps to same coarse vertex - use pwprol!
	cols.SetSize(1);
	cols[0] = cvnr;
	return true;
      }
      auto ris = fmat->GetRowIndices(fvnr);
      cols.SetSize0();
      bool is_ok = true;
      for (auto j : Range(ris)) {
	auto fvj = ris[j];
	auto cvj = vmap[fvj];
	if (cvj != -1) {
	  auto eqcj = CM.template GetEqcOfNode<NT_VERTEX>(cvj);
	  if (!eqc_h.IsLEQ(eqc, eqcj))
	    { is_ok = false; break; }
	  insert_into_sorted_array_nodups(cvj, cols);
	}
	else
	  { is_ok = false; }
      }
      // is_ok &= (cols.Size() <= MAX_PER_ROW);
      is_ok &= (cols.Size() <= MAX_PER_ROW_CLASSIC);
      return is_ok;
    }; // get_cols_classic

    /** Judge connection to coarse neibs by sum of fine connections. **/
    Array<double> dg_wt(FNV); dg_wt = 0;
    FM.template Apply<NT_EDGE>([&](auto & edge) LAMBDA_INLINE {
	auto approx_wt = ENERGY::GetApproxWeight(fedata[edge.id]);
	dg_wt[edge.v[0]] = max2(dg_wt[edge.v[0]], approx_wt);
	dg_wt[edge.v[1]] = max2(dg_wt[edge.v[1]], approx_wt);
      }, false );
    FM.template AllreduceNodalData<NT_VERTEX>(dg_wt, [](auto & tab){return move(max_table(tab)); }, false);

    Array<INT<2,double>> trow;
    Array<int> tcv;
    auto get_cols_aux = [&](auto EQ, auto V) {
      cols.SetSize0();
      auto CV = vmap[V];
      if ( is_invalid(CV) )
	{ return; }
      trow.SetSize0(); tcv.SetSize0();
      auto ovs = fecon.GetRowIndices(V);
      int nniscv = 0; // number neibs in same cv
      for (auto v : ovs)
	if (vmap[v] == CV)
	  { nniscv++; }
      if (nniscv == 0) { // no neib that maps to same coarse vertex - use pwprol!
	cols.SetSize(1);
	cols[0] = CV;
	return;
      }
      auto eis = fecon.GetRowValues(V);
      size_t pos; double in_wt = 0;
      for (auto j : Range(ovs.Size())) {
	auto ov = ovs[j];
	auto cov = vmap[ov];
	if ( is_invalid(cov) )
	  { continue; }
	if (cov == CV) {
	  // in_wt += self.template GetWeight<NT_EDGE>(fmesh, );
	  in_wt += ENERGY::GetApproxWeight(fedata[int(eis[j])]);
	  continue;
	}
	// auto oeq = fmesh.template GetEqcOfNode<NT_VERTEX>(ov);
	auto oeq = CM.template GetEqcOfNode<NT_VERTEX>(cov);
	if (eqc_h.IsLEQ(EQ, oeq)) {
	  // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
	  auto wt = ENERGY::GetApproxWeight(fedata[int(eis[j])]);
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
      double dgwt = dg_wt[V];
      cols.Append(CV);
      size_t max_adds = min2(MAX_PER_ROW-1, int(trow.Size()));
      for (auto j : Range(max_adds)) {
	cw_sum += trow[j][1];
	if ( ( !(trow[j][1] > MIN_PROL_FRAC * cw_sum) ) ||
	     ( trow[j][1] < MIN_PROL_FRAC * dgwt ) )
	  { break; }
	cols.Append(trow[j][0]);
      }
      // NOTE: these cols are unsorted - they are sorted later!
    }; // get_cols_aux

    // cout << " vmap: " << endl; prow2(vmap); cout << endl;

    auto itg = [&](auto lam) {
      // AHHM das is bloedsinn, er nimmt da einfach die finest_mat her, die is mit MPI auf die pre-embedded mat gesetzt !
      FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
	  for (auto fvnr : nodes) {
	    if (vmap[fvnr] == -1)
	      { continue; }
	    bool classic_ok = (have_fmat && (!aux_only)) ? get_cols_classic(eqc, fvnr) : false;
	    if (!classic_ok)
	      { get_cols_aux(eqc, fvnr); }
	    lam(fvnr, classic_ok);
	  }
	}, true);
    };
    Array<int> perow(FNV); perow = 0;
    itg([&](auto fvnr, bool cok) {
	if (cok)
	  { use_classic.SetBit(fvnr); }
	perow[fvnr] = cols.Size();
      });
    /** Scatter Graph per row (col-vals are garbage for non-masters here! ) **/
    // cout << " perow1: "; prow2(perow); cout << endl;
    FM.template ScatterNodalData<NT_VERTEX>(perow);
    // cout << " scattered perow: "; prow2(perow); cout << endl;
    Table<int> graph(perow);
    itg([&](auto fvnr, bool cok) {
	if (!cok)
	  { QuickSort(cols); }
	graph[fvnr] = cols;
      });

    // cout << " graph: " << endl << graph << endl;

    /** Alloc sprol **/
    auto sprol = make_shared<TSPM_TM>(perow, CNV);
    const auto & CSP = *sprol;

    /** #classic, #aux, #triv **/
    int nc = 0, na = 0, nt = 0;
    double fc = 0, fa = 0, ft = 0;

    const double omo = 1.0 - omega;
    TM d;
    auto fill_sprol_classic = [&](auto fvnr) {
      // cout << " fill " << fvnr;
      auto ris = CSP.GetRowIndices(fvnr);
      if (ris.Size() == 0)
	{ return; }
      auto rvs = CSP.GetRowValues(fvnr);
      ris = graph[fvnr];
      if (ris.Size() == 1) {
	// cout << " c-triv " << endl;
	// rvs[0] = pwprol->GetRowIndices(fvnr)[0];
	// SetIdentity(rvs[0]);
	rvs[0] = pwprol(fvnr, ris[0]);
	nt++;
	return;
      }
      // cout << " classic " << endl;
      // cout << " ris "; prow(ris); cout << endl;
      nc++;
      rvs = 0;
      auto fmris = fmat->GetRowIndices(fvnr);
      auto fmrvs = fmat->GetRowValues(fvnr);
      // cout << " fmris "; prow(fmris); cout << endl;
      // cout << " vmap fmris "; prow(vmap[fmris]); cout << endl;
      // cout << " fmrvs "; prow(fmrvs); cout << endl;
      d = fmrvs[find_in_sorted_array(fvnr, fmris)];
      // cout << " diag " << endl;
      // print_tm(cout, d);
      if (BS == 1)
	{ CalcInverse(d); }
      else {
	/** Normalize to trace of diagonal mat. More stable Pseudo inverse (?) **/
	double trinv = double(BS) / calc_trace(d);
	d *= trinv;
	CalcStabPseudoInverse(d, lh);
	d *= trinv;
      }
      // cout << " inv diag " << endl;
      // print_tm(cout, d);
      TM od_pwp;
      for (auto j : Range(fmris)) {
	auto fvj = fmris[j];
	int col = vmap[fmris[j]];
	int colind = find_in_sorted_array(col, ris);
	if (colind == -1) // Dirichlet - TODO: cleaner would be finding an aux-ext to diri DOFs first
	  { continue; }
	// cout << j << " " << col << " " << colind << endl;
	// if (fvj == fvnr)
	  // { rvs[colind] += omo * pwprol(fvj, col); }
	// else {
	  // od_pwp = fmrvs[j] * pwprol(fvj, col);
	  // rvs[colind] -= omega * d * od_pwp;
	// }
	if (fvj == fvnr)
	  { rvs[colind] += pwprol(fvj, col); }
	od_pwp = fmrvs[j] * pwprol(fvj, col);
	rvs[colind] -= omega * d * od_pwp;
      }
    }; // fill_sprol_classic

    TM Qij, Qji, QM;
    auto fill_sprol_aux = [&](auto fvnr) {
      // cout << " fill " << fvnr;
      auto ris = CSP.GetRowIndices(fvnr);
      if ( ris.Size() == 0)
	{ return; }
      ris = graph[fvnr];
      auto rvs = CSP.GetRowValues(fvnr);
      auto cvnr = vmap[fvnr];
      if ( ris.Size() == 1) {
	// cout << " a-triv " << endl;
	// rvs[0] = pwprol.GetRowIndices(fvnr)[0];
	// SetIdentity(rvs[0]);
	rvs[0] = pwprol(fvnr, ris[0]);
	nt++;
	return;
      }
      // cout << " aux " << endl;
      na++;
      rvs = 0;
      auto fneibs = fecon.GetRowIndices(fvnr);
      auto fenrs = fecon.GetRowValues(fvnr);
      int nufneibs = 0, pos = 0, cvj = 0;
      /** Here we use any neibs that map to used coarse vertices. This means we can also unse non-hierarchic
	  FINE edges (!). This does still give us a hierarchic prolongation in the end! **/
      for (auto vj : fneibs)
	if ( ( (cvj = vmap[vj]) != -1 ) &&
	     ( (pos = find_in_sorted_array(vmap[vj], ris)) != -1 ) )
	  { nufneibs++; }
      nufneibs++; // the vertex itself
      FlatArray<int> ufneibs(nufneibs, lh), ufenrs(nufneibs, lh);
      nufneibs = 0;
      int dcol = -1;
      for (auto j : Range(fneibs)) {
	auto vj = fneibs[j];
	if ( ( (cvj = vmap[vj]) != -1 ) &&
	     ( (pos = find_in_sorted_array(vmap[vj], ris)) != -1 ) ) {
	  if ( (dcol == -1) && (vj > fvnr) ) {
	    dcol = nufneibs;
	    ufenrs[nufneibs] = -1;
	    ufneibs[nufneibs++] = fvnr;
	  }
	  ufenrs[nufneibs] = fenrs[j];
	  ufneibs[nufneibs++] = vj;
	}
      }
      if ( (dcol == -1) ) {
	dcol = nufneibs;
	ufenrs[nufneibs] = -1;
	ufneibs[nufneibs++] = fvnr;
      }
      // cout << " fneibs "; prow(fneibs); cout << endl;
      // cout << "ufneibs "; prow(ufneibs); cout << endl;
      FlatMatrix<TM> rmrow(1, nufneibs, lh); rmrow = 0;
      for (auto j : Range(ufneibs)) {
	if (j == dcol)
	  { continue; }
	const auto & edge = fedges[ufenrs[j]];
	int l = (fvnr == edge.v[0]) ? 0 : 1;
	ENERGY::CalcQs(fvdata[edge.v[l]], fvdata[edge.v[1-l]], Qij, Qji);
	TM EM = ENERGY::GetEMatrix(fedata[ufenrs[j]]);
	if constexpr(mat_traits<TM>::HEIGHT!=1) {
	    if (vmap[edge.v[1-l]] == cvnr) {
	      /** We regularize edge-matrices to other vertices in the same agglomerate.
		  Yes, this is cheating. No, I do not care. **/
	      RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EM);
	    }
	  }
	QM = Trans(Qij) * EM;
	rmrow(0, j) -= QM * Qji;
	rmrow(0, dcol) += QM * Qij;
      }
      // cout << " rmrow: " << endl; print_tm_mat(cout, rmrow); cout << endl;
      TM d = rmrow(0, dcol);
      // cout << " diag " << endl;
      // print_tm(cout, d);
      if (BS == 1)
	{ CalcInverse(d); }
      else {
	/** Normalize to trace of diagonal mat. More stable Pseudo inverse (?) **/
	double trinv = double(BS) / calc_trace(d);
	rmrow *= trinv;
	d *= trinv;
	// CalcStabPseudoInverse(d, lh);
	CalcInverse(d);
      }
      // cout << " diag inv " << endl;
      // print_tm(cout, d);
      TM od_pwp;
      for (auto j : Range(nufneibs)) {
	auto fvj = ufneibs[j];
	int col = vmap[fvj];
	int colind = find_in_sorted_array(col, ris);
	if (fvj == fvnr)
	  { rvs[colind] += omo * pwprol(fvj, col); }
	else {
	  od_pwp = rmrow(0, j) * pwprol(fvj, col);
	  rvs[colind] -= omega * d * od_pwp;
	}
      }
    }; // fill_sprol_aux


    /** Fill sprol **/
    FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
	for (auto fvnr : nodes) {
	  HeapReset hr(lh);
	  if (use_classic.Test(fvnr))
	    { fill_sprol_classic(fvnr); }
	  else
	    { fill_sprol_aux(fvnr); }
	}
      }, true);


    /** Scatter sprol colnrs & vals **/
    if (neqcs > 1) {
      Array<int> eqc_perow(neqcs); eqc_perow = 0;
      FM.template ApplyEQ<NT_VERTEX>( Range(1, neqcs), [&](auto EQC, auto V) {
	  eqc_perow[EQC] += perow[V];
	}, false); // all!
      Table<INT<2,int>> ex_ris(eqc_perow);
      Table<TM> ex_rvs(eqc_perow); eqc_perow = 0;
      FM.template ApplyEQ<NT_VERTEX>( Range(1, neqcs), [&](auto EQC, auto V) {
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
      eqc_perow = 0;
      FM.template ApplyEQ<NT_VERTEX>( Range(1, neqcs), [&](auto EQC, auto V) {
	  auto rvs = CSP.GetRowValues(V);
	  auto ris = CSP.GetRowIndices(V);
	  for (auto j : Range(ris)) {
	    auto tup = ex_ris[EQC][eqc_perow[EQC]];
	    ris[j] = CM.template MapENodeFromEQC<NT_VERTEX>(tup[1], eqc_h.GetEQCOfID(tup[0]));
	    rvs[j] = ex_rvs[EQC][eqc_perow[EQC]++];
	  }
	}, false); // master!
    }

    if ( ( (eqc_h.GetCommunicator().Size() == 1) || (eqc_h.GetCommunicator().Rank() == 1) ) &&
	 ( FNV > 0 ) ) {
      cout << "NV,   nc/na/nt " << FNV << ", " << nc << " " << na << " " << nt << endl;
      cout << "fracs c/a/t    " << double(nc)/FNV << " " << double(na)/FNV << " " << double(nt)/FNV << endl;
    }

    // cout << " sprol: " << endl;
    // print_tm_spmat(cout, *sprol);
    // cout << endl;

    return make_shared<ProlMap<TSPM_TM>>(sprol, fpds, cpds);
  } // VertexAMGFactory::SmoothedProlMap_impl_v2


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
  {
    static Timer t("SmoothedProlMap_cap"); RegionTimer rt(t);

    shared_ptr<TopologicMesh> tfmesh = fcap->mesh;

    if (pw_step == nullptr)
      { throw Exception("Need pw-map for SmoothedProlMap!"); }
    auto prol_map =  dynamic_pointer_cast<ProlMap<TSPM_TM>> (pw_step);
    if (prol_map == nullptr)
      { throw Exception(string("Invalid Map type ") + typeid(*pw_step).name() + string(" in SmoothedProlMap!")); }
    auto fmesh = dynamic_pointer_cast<TMESH>(tfmesh);
    if (fmesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*tfmesh).name() + string(" in SmoothedProlMap!")); }

    Options &O (static_cast<Options&>(*options));

    const int baselevel = fcap->baselevel;

    const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(baselevel);
    const int MAX_PER_ROW = O.sp_max_per_row.GetOpt(baselevel);
    const double omega = O.sp_omega.GetOpt(baselevel);

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
  shared_ptr<BaseDOFMapStep> VertexAMGFactory<ENERGY, TMESH, BS> :: SmoothedProlMap_impl (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap, shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
  {
    static Timer t("SmoothedProlMap_map"); RegionTimer rt(t);

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

    // cout << " FM: " << endl << FM << endl;
    // cout << " VDATA: " << endl; prow2(vdata); cout << endl << endl;
    // cout << " EDATA: " << endl; prow2(edata); cout << endl << endl;

    const int baselevel = fcap->baselevel;

    Options &O (static_cast<Options&>(*options));
    const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(baselevel);
    const int MAX_PER_ROW = O.sp_max_per_row.GetOpt(baselevel);
    const double omega = O.sp_omega.GetOpt(baselevel);

    const size_t NFV = FM.template GetNN<NT_VERTEX>(), NCV = CM.template GetNN<NT_VERTEX>();
    auto vmap = cmap->template GetMap<NT_VERTEX>();

    /** For each fine vertex, find all coarse vertices we can (and should) prolongate from.
	The master of V does this. **/
    Table<int> graph(NFV, MAX_PER_ROW); graph.AsArray() = -1; // has to stay
    Array<int> perow(NFV); perow = 0; // 
    Array<INT<2,double>> trow;
    Array<int> tcv, fin_row;
    Array<double> dg_wt(NFV); dg_wt = 0;
    FM.template Apply<NT_EDGE>([&](auto & edge) LAMBDA_INLINE {
	auto approx_wt = ENERGY::GetApproxWeight(edata[edge.id]);
	dg_wt[edge.v[0]] = max2(dg_wt[edge.v[0]], approx_wt);
	dg_wt[edge.v[1]] = max2(dg_wt[edge.v[1]], approx_wt);
      }, false );
    FM.template AllreduceNodalData<NT_VERTEX>(dg_wt, [](auto & tab){return move(max_table(tab)); }, false);
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
	double dgwt = dg_wt[V];
	fin_row.Append(CV);
	size_t max_adds = min2(MAX_PER_ROW-1, int(trow.Size()));
	for (auto j : Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if ( ( !(trow[j][1] > MIN_PROL_FRAC * cw_sum) ) ||
	       ( trow[j][1] < MIN_PROL_FRAC * dgwt ) )
	    { break; }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	for (auto j:Range(fin_row.Size()))
	  { graph[V][j] = fin_row[j]; }
	if (fin_row.Size() == 1)
	  { perow[V] = 1; }
	else {
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
	if (ris.Size() == 1)
	  { SetIdentity(rvs[0]); ris[0] = V; return; }
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
	    // TM EMAT = edata[une[l][1]];
	    TM EMAT = ENERGY::GetEMatrix(edata[une[l][1]]);
	    if constexpr(mat_traits<TM>::HEIGHT!=1) {
		// RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT);
		// RegTM<0, FACTORY_CLASS::DIM, mat_traits<TM>::HEIGHT>(EMAT);
		// if (vmap[une[l][0]] == CV)
		// { RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT, maxtr); }
		if (vmap[une[l][0]] == CV) {
		  RegTM<0, mat_traits<TM>::HEIGHT, mat_traits<TM>::HEIGHT>(EMAT);
		}
	      }
	    else
	      { EMAT = max2(EMAT, 1e-8); }
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
  

    // cout << endl << "assembled RM: " << endl;
    // print_tm_spmat(cout, RM); cout << endl;

    shared_ptr<TSPM_TM> sprol = prol_map->GetProl();
    sprol = MatMultAB(RM, *sprol);
    
    /** Now, unfortunately, we have to distribute matrix entries of sprol. We cannot do this for RM.
	(we are also using more local fine edges that map to less local coarse edges) **/
    if (eqc_h.GetCommunicator().Size() > 2) {
      // cout << endl << "un-cumualted sprol: " << endl;
      // print_tm_spmat(cout, *sprol); cout << endl;
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
  } // VertexAMGFactory::SmoothedProlMap_impl


  template<class ENERGY, class TMESH, int BS>
  bool VertexAMGFactory<ENERGY, TMESH, BS> :: TryDiscardStep (BaseAMGFactory::State & state)
  {
    if (!options->enable_disc.GetOpt(state.level[0]))
      { return false; }

    if (state.curr_cap->free_nodes != nullptr)
      { throw Exception("discard with dirichlet TODO!!"); }

    shared_ptr<BaseAMGFactory::LevelCapsule> c_cap = this->AllocCap();
    c_cap->baselevel = state.level[0];

    shared_ptr<BaseDiscardMap> disc_map = BuildDiscardMap(state, c_cap);

    if (disc_map == nullptr)
      { return false; }

    auto n_d_v = disc_map->GetNDroppedNodes<NT_VERTEX>();
    auto any_n_d_v = state.curr_cap->mesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_d_v, MPI_SUM);

    bool map_ok = any_n_d_v != 0; // someone somewhere eliminated some verices

    if (map_ok) { // a non-negligible amount of vertices was eliminated
      auto elim_vs = disc_map->GetMesh()->template GetNNGlobal<NT_VERTEX>() - disc_map->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>();
      auto dv_frac = double(disc_map->GetMappedMesh()->template GetNNGlobal<NT_VERTEX>()) / disc_map->GetMesh()->template GetNNGlobal<NT_VERTEX>();
      map_ok &= (dv_frac < 0.98);
    }

    if (!map_ok)
      { return false; }

    //TODO: disc prol map!!

    auto disc_prol_map = PWProlMap(disc_map, state.curr_cap, c_cap);

    state.disc_map = disc_map;
    state.curr_cap = c_cap;
    state.dof_map = disc_prol_map;

    return true;
  } // VertexAMGFactory::TryDiscardStep


  template<class ENERGY, class TMESH, int BS>
  shared_ptr<BaseDiscardMap> VertexAMGFactory<ENERGY, TMESH, BS> :: BuildDiscardMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & c_cap)
  {
    auto & O(static_cast<Options&>(*options));
    auto tm_mesh = dynamic_pointer_cast<TMESH>(state.curr_cap->mesh);
    auto disc_map = make_shared<VDiscardMap<TMESH>> (tm_mesh, O.disc_max_bs);

    c_cap->mesh = disc_map->GetMappedMesh();
    c_cap->pardofs = this->BuildParallelDofs(c_cap->mesh);
    c_cap->free_nodes = nullptr;

    return disc_map;
  } // VertexAMGFactory :: BuildDiscardMap

  /** END VertexAMGFactory **/

} // namespace amg

#endif
