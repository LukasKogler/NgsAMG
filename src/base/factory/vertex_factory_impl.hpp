#ifndef FILE_VERTEX_FACTORY_IMPL_HPP
#define FILE_VERTEX_FACTORY_IMPL_HPP

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ngs_stdcpp_include.hpp>
#include <utils_denseLA.hpp>
#include <utils_io.hpp>
#include <utils_arrays_tables.hpp>
#include <utils_buffering.hpp>

#include "base_smoother.hpp"
#include "dof_map.hpp"
#include "reducetable.hpp"
#include "utils.hpp"
#include "utils_sparseLA.hpp"
#include "utils_sparseMM.hpp"
#include "vertex_factory.hpp"
#include "nodal_factory.hpp"

#include "nodal_factory_impl.hpp"

#include <agglomerate_map.hpp>
#include <spw_agg_map.hpp>
#include <mis_agg_map.hpp>
#include <plate_test_agg_map.hpp>

namespace amg
{

/** Options **/

class VertexAMGFactoryOptions : public BaseAMGFactory::Options
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
    MIS = 2,         // MIS-based aggregaion
    PLATE_TEST = 3
  #else // SPW_AGG
    MIS = 2
  #endif // SPW_AGG
#else // MIS_AGG
    SPW = 1 // SPW is guaranteed to be enabled in this case
#endif
  };
  SpecOpt<CRS_ALG> crs_alg = CRS_ALG::SPW;

  /** Prolongation **/
  enum PROL_TYPE : int {
    PIECEWISE         = 0, // unsmoothed, piecewise aggregation prolongation
    AUX_SMOOTHED      = 1, // smoothed prolongation with auxiliary matrix
    SEMI_AUX_SMOOTHED = 2, // smoothed with auxiliary/real matrix
  };
  SpecOpt<PROL_TYPE> prol_type = SEMI_AUX_SMOOTHED;
  // SpecOpt<bool> sp_aux_only = false;           // smooth prolongation using only auxiliary matrix
  SpecOpt<int> sp_max_per_row_classic = 5;     // maximum entries per row (should be >= 2!) where " newst" uses classic
  SpecOpt<bool> improve_avgs = false;

  // /** Discard **/
  // int disc_max_bs = 5;

  /** for elasticity - scaling of rotations on level 0 **/
  SpecOpt<double> rot_scale = 1.0;

public:

  VertexAMGFactoryOptions ()
    : BaseAMGFactory::Options()
  { ; }

  virtual void SetFromFlags (const Flags & flags, string prefix) override
  {
    BaseAMGFactory::Options::SetFromFlags(flags, prefix);

    Array<CRS_ALG> crs_algs({SPW});
    Array<string> crs_alg_names({"spw"});

    SetAggFromFlags(flags, prefix);

#ifdef SPW_AGG
    SetSPWFromFlags(flags, prefix);
    crs_algs.Append(SPW);
    crs_alg_names.Append("spw");
#endif
#ifdef MIS_AGG
    SetMISFromFlags(flags, prefix);
    crs_algs.Append(MIS);
    crs_alg_names.Append("mis");
#endif
    crs_algs.Append(PLATE_TEST);
    crs_alg_names.Append("plate_test");

    crs_alg.SetFromFlagsEnum(flags, prefix + "crs_alg", crs_alg_names, crs_algs);

    // sp_aux_only.SetFromFlags(flags, prefix + "sp_aux_only");

    Array<PROL_TYPE> prol_types({PIECEWISE, AUX_SMOOTHED, SEMI_AUX_SMOOTHED});
    Array<string> prol_type_names({"piecewise", "aux_smoothed", "semi_aux_smoothed"});
    prol_type.SetFromFlagsEnum(flags, prefix + "prol_type", prol_type_names, prol_types);

    sp_max_per_row_classic.SetFromFlags(flags, prefix + "sp_max_per_row_classic");

    improve_avgs.SetFromFlags(flags, prefix + "improve_avgs");

    ecw_geom.SetFromFlags(flags, prefix + "ecw_geom");

    rot_scale.SetFromFlags(flags, prefix + "rot_scale");
  } // VertexAMGFactoryOptions::SetFromFlags

}; // VertexAMGFactoryOptions

/** END Options **/


/** VertexAMGFactory **/


template<class ENERGY, class TMESH, int BS>
VertexAMGFactory<ENERGY, TMESH, BS>::
VertexAMGFactory (shared_ptr<Options> opts)
  : BASE_CLASS(opts)
{
  ;
} // VertexAMGFactory(..)


template<class ENERGY, class TMESH, int BS>
VertexAMGFactory<ENERGY, TMESH, BS>::
~VertexAMGFactory ()
{
  ;
} // ~VertexAMGFactory


template<class ENERGY, class TMESH, int BS>
BaseAMGFactory::State*
VertexAMGFactory<ENERGY, TMESH, BS>::
AllocState () const
{
  return new State();
} // VertexAMGFactory::AllocState


template<class ENERGY, class TMESH, int BS>
void
VertexAMGFactory<ENERGY, TMESH, BS>::
InitState (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::AMGLevel> & lev) const
{
  BASE_CLASS::InitState(state, lev);
  // auto & s(static_cast<State&>(state));
  // s.crs_opts = nullptr;
} // VertexAMGFactory::InitState




template<int BSF, int BS, class ENERGY, class TMESH, class Options>
shared_ptr<SparseMat<BSF, BS>>
EmbeddedSProl (Options const &O,
               SparseMat<BSF, BS>  const &emb,
               SparseMat<BSF, BSF> const &fMat,
               ParallelDofs              *fParDofs,
               TMESH               const &fMesh,
               TMESH               const &cMesh,
               BaseCoarseMap       const &cMap)
{
  /**
   *  Creates a smoothed prolongation using an actual fine-matrix,
   *  which is connected to the fine mesh by the embedding.
   *  The "embedding" can e.g. have elements of
   *      - dist+rot->dist embedding,
   *      - the reordering we do in parallel
   *      - embedding from low to high order
   *      - the elasticity p2-embeddong
   *  The embedding must be a C2C operator.
   *
   *  The output DOF-step goes from fine matrix to AMG-level 1
   *  (that is, it already includes the embedding)!
   *
   *   A row of the prolongation is something like
   *     (I - \omega d^{-1} rowA) E Phat cEXT
   *   where cEXT is a coarse extension that maps from used-cols to
   *   all cols appearing in "AP := rowA E Phat"
   *
   *   cEXT is done implicitly by simply adding all off-diag contribs
   *   belonging to non-used cols to the used ones instead.
   *   [[ for the p2-emb, these contribs are divided equally by the two parent-verts ]]
   *
   * TODO: could compute PTAP easier here since we already have AP
   */
  const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(0);
  const double MIN_SUM_FRAC  = 1.0 - sqrt(MIN_PROL_FRAC); // MPF 0.15 -> 0.61
  const int MAX_PER_ROW = O.sp_max_per_row.GetOpt(0);
  const int MAX_PER_ROW_CLASSIC = O.sp_max_per_row_classic.GetOpt(0);
  const double omega = O.sp_omega.GetOpt(0);


  fMesh.CumulateData();
  cMesh.CumulateData();

  auto const &eqc_h = *fMesh.GetEQCHierarchy();
  auto const &fECon = *fMesh.GetEdgeCM();

  auto fVData = get<0>(fMesh.Data())->Data();
  auto fEData = get<1>(fMesh.Data())->Data();

  auto cVData = get<0>(cMesh.Data())->Data();

  auto vMap = cMap.template GetMap<NT_VERTEX>();

  auto const FNV = fMesh.template GetNN<NT_VERTEX>();
  auto const CNV = cMesh.template GetNN<NT_VERTEX>();

  Array<double> dg_wt(FNV);

  dg_wt = 0;

  fMesh.template Apply<NT_EDGE>([&](auto & edge) LAMBDA_INLINE
  {
    auto approx_wt = ENERGY::GetApproxWeight(fEData[edge.id]);
    dg_wt[edge.v[0]] = max2(dg_wt[edge.v[0]], approx_wt);
    dg_wt[edge.v[1]] = max2(dg_wt[edge.v[1]], approx_wt);
  }, false );

  fMesh.template AllreduceNodalData<NT_VERTEX>(dg_wt, [](auto & tab){return std::move(max_table(tab)); }, false);

  // create prol-graph  [[ L0-mesh <- L1-mesh ]]
  Array<int> cols;
  Array<IVec<2,double>> trow;
  Array<int> tcv;

  auto getCols = [&](auto EQ, auto V)
  {
    cols.SetSize0();

    auto CV = vMap[V];

    if ( is_invalid(CV) )
      { return; }

    trow.SetSize0(); tcv.SetSize0();

    auto ovs = fECon.GetRowIndices(V);
    auto edgeNums = fECon.GetRowValues(V);

    size_t pos;
    double in_wt = 0;

    for (auto j : Range(ovs.Size()))
    {
      auto ov = ovs[j];
      auto cov = vMap[ov];

      if ( is_invalid(cov) )
        { continue; }

      if (cov == CV)
      {
        in_wt += ENERGY::GetApproxWeight(fEData[int(edgeNums[j])]);
        continue;
      }

      auto oeq = cMesh.template GetEQCOfNode<NT_VERTEX>(cov);

      if (eqc_h.IsLEQ(EQ, oeq))
      {
        // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
        auto wt = ENERGY::GetApproxWeight(fEData[int(edgeNums[j])]);

        if ( (pos = tcv.Pos(cov)) == size_t(-1))
        {
          trow.Append(IVec<2,double>(cov, wt));
          tcv.Append(cov);
        }
        else
          { trow[pos][1] += wt; }
      }
    }

    QuickSort(trow, [](const auto & a, const auto & b) LAMBDA_INLINE { return a[1]>b[1]; });

    int max_adds = min2(MAX_PER_ROW-1, int(trow.Size()));

    double cw_sum = 0.0; // all edges in the same agg are automatically assembled (penalize so we dont pw-ize too many)
    double dgwt   = dg_wt[V];

    cols.Append(CV);

    for (auto j : Range(max_adds))
    {
      auto const newWt = trow[j][1];

      if ( ( newWt < MIN_PROL_FRAC * cw_sum ) || // contrib weak compared to what we already have
           ( newWt < MIN_PROL_FRAC * dgwt ) )    // weak compared to diagonal
      {
        break;
      }
      cols.Append(trow[j][0]);
    }
    // NOTE: these cols are unsorted - they are sorted later!
  }; // getCols

  Array<int> perowM2M(FNV);
  perowM2M = 0;

  fMesh.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes)
  {
    for (auto vNr : nodes)
    {
      getCols(eqc, vNr);
      perowM2M[vNr] = cols.Size();
    }
  },
  true);

  fMesh.template ScatterNodalData<NT_VERTEX>(perowM2M);

  auto meshProl = make_shared<SparseMat<BS, BS>>(perowM2M, CNV);

  // set cols, initialize with pw-prol vals
  fMesh.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes)
  {
    for (auto vNr : nodes)
    {
      auto ris = meshProl->GetRowIndices(vNr);
      auto rvs = meshProl->GetRowValues(vNr);

      getCols(eqc, vNr);
      QuickSort(cols);
      ris = cols;

      if ( rvs.Size() )
      {
        auto const cVNr = vMap[vNr];

        rvs = 0;

        auto &v = rvs[find_in_sorted_array(cVNr, cols)];

        SetIdentity(v); ENERGY::GetQiToj(cVData[cVNr], fVData[vNr]).MQ(v);
      }
    }
  },
  true);

  // TODO: exchange prol-vals, exchange A_embP row-vals (somehow?)

  // embProl = emb * prol  [[ L0 <- L1-mesh ]]
  auto embProl = MatMultAB(emb, *meshProl); // vals will be updated

  // AP = A * embProl
  auto A_embP = MatMultAB(fMat, *embProl);

  // {std::ofstream of("justEmb.out"); print_tm_spmat(of, emb);}
  // {std::ofstream of("meshProl.out"); print_tm_spmat(of, *meshProl);}
  // {std::ofstream of("initEmbProl.out"); print_tm_spmat(of, *embProl);}
  // {std::ofstream of("A_embP.out"); print_tm_spmat(of, *A_embP);}

  // smooth prol-rows
  Array<int> replaceColPos;
  for (auto k : Range(fMat.Height()))
  {
    auto prolCols = embProl->GetRowIndices(k);
    auto prolVals = embProl->GetRowValues(k);

    if ( prolCols.Size() < 2 )
    {
      // not in range of emb, Dirichlet, or prols from single CV -> nothing to do
      continue;
    }

    auto allCols = A_embP->GetRowIndices(k);
    auto aepVals = A_embP->GetRowValues(k);


    // cout << " update row " << k << endl;

    Mat<BSF, BSF, double> dInv = fMat(k, k);
    // cout << " diag: " << endl; print_tm(cout, dInv); cout << endl;
    CalcInverse(dInv); // no issues with inverse here!
    // cout << " dInv: " << endl; print_tm(cout, dInv); cout << endl;

    double repFac = 1.0;

    if( prolCols.Size() < allCols.Size() )
    {
      // find CVs to use for contribs for unused cols
      //    just divide equally between all cvs of cols appearin in emb
      auto embCols = emb.GetRowIndices(k);

      replaceColPos.SetSize0();
      for (auto fv : embCols)
      {
        auto cv = vMap[fv];

        int pos;
        if ( (cv != -1) &&
             (pos = find_in_sorted_array(cv, prolCols)) != -1 )
        {
          replaceColPos.Append(pos);
          // break;
        }
      }
      repFac = 1.0 / replaceColPos.Size();
    }

    iterate_AC(allCols, prolCols, [&](auto where, auto idxA, auto idxP)
    {
      Mat<BSF, BS, double> upVal = dInv * aepVals[idxA];

      // cout << idxA << ", col " << allCols[idxA] << endl;
      // cout << " dinv x offd: " << endl; print_tm(cout, upVal); cout << endl;

      if ( where == INTERSECTION ) // used col
      {
        // cout << " USED @ " << idxP << endl;
        prolVals[idxP] -= omega * upVal;
      }
      else // unused col
      {
        // instead add as contrib to reCols (i.e. cvs of fvs from initial emb!)
        for (auto idx : replaceColPos)
        {
          auto const repCol = prolCols[idx];

          // cout << " REPLACE w. " << repCol << " @ " << idx << endl;
          prolVals[idx] += ENERGY::GetQiToj(cVData[repCol], cVData[allCols[idxA]]).GetMQ(-omega * repFac, upVal);
        }
      }
    });
  }

  // TOOD: re-smooth?

  // std::ofstream of("embProl.out");
  // print_tm_spmat(of, *embProl);

  return embProl;
} // EmbeddedSProl

template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseCoarseMap>
VertexAMGFactory<ENERGY, TMESH, BS>::
BuildCoarseMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
{
  static Timer t("BuildCoarseMap"); RegionTimer rt(t);

  auto & O(static_cast<Options&>(*options));

  Options::CRS_ALG calg = O.crs_alg.GetOpt(state.level[0]);

  shared_ptr<BaseCoarseMap> cmap = nullptr;

  if (O.log_level >= Options::LOG_LEVEL::DBG)
  {
    ofstream out ("alg_mesh_rk_" + to_string(state.curr_cap->mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
      + to_string(state.curr_cap->baselevel) + ".out");
    out << *state.curr_cap->mesh << endl;
  }

  switch(calg) {
#ifdef MIS_AGG
  case(Options::CRS_ALG::MIS): { cmap = BuildMISAggMap(state, mapped_cap); break; }
#endif // MIS_AGG
#ifdef SPW_AGG
  case(Options::CRS_ALG::SPW): { cmap = BuildSPWAggMap(state, mapped_cap); break; }
#endif // SPW_AGG
  case(Options::CRS_ALG::PLATE_TEST): { cmap = BuildPlateTestAggMap(state, mapped_cap); break; }
  default: { throw Exception("Invalid coarsen alg!"); break; }
  }

  if (O.log_level >= Options::LOG_LEVEL::DBG)
  {
    ofstream out ("alg_cMesh_rk_" + to_string(state.curr_cap->mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
      + to_string(state.curr_cap->baselevel) + ".out");
    out << *cmap->GetMappedMesh() << endl;
  }

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("cmap_rk_" + to_string(state.curr_cap->mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
      + to_string(state.curr_cap->baselevel) + ".out");
    out << *cmap << endl;
  }

  return cmap;
} // VertexAMGFactory::BuildCoarseMap


template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dofSteps,
          shared_ptr<AMGLevel> &fCap,
          shared_ptr<AMGLevel> &cCap)
{
  auto & O = static_cast<Options&>(*options);

  size_t off = (fCap->level == 0 && O.use_emb_sp) ? 1 : 0;

  return BaseAMGFactory::MapLevel(dofSteps.Range(off, dofSteps.Size()), fCap, cCap);
}


template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseCoarseMap>
VertexAMGFactory<ENERGY, TMESH, BS>::
BuildPlateTestAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
{
  static Timer t("BuildMISAggMap"); RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);
  typedef PlateTestAgglomerateCoarseMap<TMESH> AGG_CLASS;
  auto mesh = my_dynamic_pointer_cast<TMESH>(state.curr_cap->mesh, "BuildPlateTestAggMap::BuildPlateTestAggMap, mesh");

  const int level = state.level[0];

  auto aggMap = make_shared<AGG_CLASS>(mesh);
  aggMap->Initialize(O, level);
  aggMap->SetFreeVerts(state.curr_cap->free_nodes);

  /** Set mapped Capsule **/
  auto cmesh = aggMap->GetMappedMesh();
  mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
  mapped_cap->mesh = cmesh;
  mapped_cap->uDofs = this->BuildUDofs(*mapped_cap);

  return aggMap;
} // VertexAMGFactory::BuildCoarseMap

#ifdef MIS_AGG
template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseCoarseMap>
VertexAMGFactory<ENERGY, TMESH, BS>::
BuildMISAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
{
  static Timer t("BuildMISAggMap"); RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);
  typedef MISAgglomerateCoarseMap<TMESH, ENERGY> AGG_CLASS;
  auto mesh = my_dynamic_pointer_cast<TMESH>(state.curr_cap->mesh, "BuildMISAggMap - mesh");

  const int level = state.level[0];

  auto aggMap = make_shared<AGG_CLASS>(mesh);
  aggMap->Initialize(O, level);
  aggMap->SetFreeVerts(state.curr_cap->free_nodes);

  /** Set mapped Capsule **/
  auto cmesh = aggMap->GetMappedMesh();
  mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
  mapped_cap->mesh = cmesh;
  mapped_cap->uDofs = this->BuildUDofs(*mapped_cap);

  return aggMap;
} // VertexAMGFactory::BuildCoarseMap
#endif // MIS_AGG


#ifdef SPW_AGG
template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseCoarseMap>
VertexAMGFactory<ENERGY, TMESH, BS>::
BuildSPWAggMap (BaseAMGFactory::State & state, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap)
{
  static Timer t("BuildSPWAggMap"); RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);
  typedef SPWAgglomerateCoarseMap<TMESH, ENERGY> AGG_CLASS;

  auto mesh = my_dynamic_pointer_cast<TMESH>(state.curr_cap->mesh, " BuildSPWAggMap - mesh");

  const int level = state.level[0];

  auto aggMap = make_shared<AGG_CLASS>(mesh);
  aggMap->Initialize(O, level);
  aggMap->SetFreeVerts(state.curr_cap->free_nodes);
  // agglomeartor->Finalize();

  /** Set mapped Capsule **/
  auto cmesh = aggMap->GetMappedMesh();
  mapped_cap->eqc_h = cmesh->GetEQCHierarchy();
  mapped_cap->mesh = cmesh;
  mapped_cap->uDofs = this->BuildUDofs(*mapped_cap);

  return aggMap;
} // VertexAMGFactory::BuildSPWAggMap
#endif // SPW_AGG


template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
BuildCoarseDOFMap (shared_ptr<BaseCoarseMap>                cmap,
                   shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
                   shared_ptr<BaseAMGFactory::LevelCapsule> ccap,
                   shared_ptr<BaseDOFMapStep> embMap)
{
  Options &O (static_cast<Options&>(*options));

  Options::PROL_TYPE prol_type = O.prol_type.GetOpt(fcap->baselevel);

  auto cap = my_dynamic_pointer_cast<LevelCapsule>(fcap, "VertexAMGFactory::BuildCoarseDOFMap - cap");

  shared_ptr<BaseDOFMapStep> step = nullptr;

  if( fcap->baselevel == 0 && O.use_emb_sp )
  {
    auto fMat = cap->mat;

    DispatchSquareMatrix(*fMat, [&](auto const &fineA, auto BSSF)
    {
      constexpr int BSF = BSSF.value;

      // prevent some weird/irrelevant cases from compiling
      if constexpr( IsProlMapCompiled<BSF, BS>() && BSF < BS && (BSF > 1 == BS > 1) )
      {
        auto embProlMap = my_dynamic_pointer_cast<ProlMap<StripTM<BSF, BS>>>(embMap, "emb-prol");
        auto embProl    = embProlMap->GetProl();

        auto const &fMesh = *my_dynamic_pointer_cast<TMESH>(fcap->mesh, "TMESH");
        auto const &cMesh = *my_dynamic_pointer_cast<TMESH>(ccap->mesh, "TMESH");

        auto fUDofs = embMap->GetUDofs();

        auto sprol = EmbeddedSProl<BSF, BS, ENERGY>(
                        O,
                        *embProl,
                        fineA,
                        nullptr, // fParDofs,
                        fMesh,
                        cMesh,
                        *cmap);

        step = make_shared<ProlMap<StripTM<BSF, BS>>>(sprol, fUDofs, ccap->uDofs);
      }
    });
  }
  else
  {
    switch(prol_type)
    {
      case(Options::PROL_TYPE::PIECEWISE)         : { step = PWProlMap(*cmap, *fcap, *ccap); break; }
      case(Options::PROL_TYPE::AUX_SMOOTHED)      : { step = AuxSProlMap(PWProlMap(*cmap, *fcap, *ccap), cmap, fcap); break; }
      case(Options::PROL_TYPE::SEMI_AUX_SMOOTHED) : { step = SemiAuxSProlMap(PWProlMap(*cmap, *fcap, *ccap), cmap, fcap); break; }
    }
  }

  return step;
} // BuildCoarseDOFMap


template<class ENERGY, class TMESH, int BS>
shared_ptr<ProlMap<typename VertexAMGFactory<ENERGY, TMESH, BS>::TM>>
VertexAMGFactory<ENERGY, TMESH, BS>::
PWProlMap (BaseCoarseMap                const &cmap,
           BaseAMGFactory::LevelCapsule const &fcap,
           BaseAMGFactory::LevelCapsule const &ccap)
{
  auto & O(static_cast<Options&>(*options));

  static Timer t("PWProlMap"); RegionTimer rt(t);

  // shared_ptr<ParallelDofs> fpds = fcap->uDofs.GetParallelDofs(), cpds = ccap->uDofs.GetParallelDofs();

  const TMESH & fmesh = static_cast<TMESH&>(*cmap.GetMesh()); fmesh.CumulateData();
  const TMESH & cmesh = static_cast<TMESH&>(*cmap.GetMappedMesh()); cmesh.CumulateData();

  size_t NV = fmesh.template GetNN<NT_VERTEX>();
  size_t NCV = cmesh.template GetNN<NT_VERTEX>();

  const double rscale = O.rot_scale.GetOpt(fcap.baselevel);

  // cout << " rscale on level " << fcap->baselevel << " = " << rscale << endl;

  /** Alloc Matrix **/
  auto vmap = cmap.template GetMap<NT_VERTEX>();
  Array<int> perow (NV); perow = 0;
  for (auto vnr : Range(NV))
    { if (vmap[vnr] != -1) perow[vnr] = 1; }

  // cout << "vmap: " << endl; prow2(vmap); cout << endl;
  // cout << "vmap: " << endl; prow(vmap); cout << endl;

  auto prol = make_shared<TSPM>(perow, NCV);

  // Fill Matrix
  auto f_v_data = get<0>(fmesh.Data())->Data();
  auto c_v_data = get<0>(cmesh.Data())->Data();
  for (auto vnr : Range(NV)) {
    auto cvnr = vmap[vnr];
    if (cvnr != -1) {
      prol->GetRowIndices(vnr)[0] = cvnr;
      // ENERGY::CalcQHh(c_v_data[cvnr][0], f_v_data[vnr][0], prol->GetRowValues(vnr)[0], rscale);
      ENERGY::CalcQHh(c_v_data[cvnr], f_v_data[vnr], prol->GetRowValues(vnr)[0], rscale);
    }
  }

  auto pwprol = make_shared<ProlMap<TM>> (prol, fcap.uDofs, ccap.uDofs);

  // cout << "PWPROL: " << endl;
  if ( O.log_level == Options::LOG_LEVEL::DBG )
  {
    std::ofstream of("ngs_amg_pwprol_r_" + std::to_string(fcap.uDofs.GetCommunicator().Rank()) +
                                   "_l_" + std::to_string(fcap.baselevel) + ".out");

    of << *pwprol << endl;
    // print_tm_spmat(of, *prol); of << endl;
  }

  // return make_shared<ProlMap<TSPM_TM>> (prol, fpds, cpds);
  return pwprol;
} // VertexAMGFactory::PWProlMap


// template<int N, int M, class T>
// void
// operator += (SliceMatrix<T> a, const Mat<N, M, double> & b)
// {
//   Iterate<N>([&](auto I) {
//     Iterate<M>([&](auto J) {
//       a(N, M) += b(N, M);
//     });
//   });
// }

// // scalar case - keep this only here because it is ugly
// template<class T>
// void
// operator += (SliceMatrix<T> a, T & b)
// {
//   a(0) += b;
// }


template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
SemiAuxSProlMap (shared_ptr<ProlMap<TM>> pw_step,
                 shared_ptr<BaseCoarseMap> cmap,
                 shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
{
  static Timer t("SemiAuxSProlMap");
  RegionTimer rt(t);

  /** Use fine level matrix to smooth prolongation ("classic prol") where we can, that is whenever:
       I) We would not break MAX_PER_ROW, so if all algebraic neibs map to <= MAX_PER_ROW coarse verts.
      II) We would not break the hierarchy, so if all algebraic neibs map to coarse verts in the same, or higher EQCs
      Where we cannot, smooth using the replacement matrix ("aux prol"). **/
  Options &O (static_cast<Options&>(*options));

  const int baselevel = fcap->baselevel;
  const double MIN_PROL_FRAC = O.sp_min_frac.GetOpt(baselevel);
  const double MIN_SUM_FRAC  = 1.0 - sqrt(MIN_PROL_FRAC); // MPF 0.15 -> 0.61
  const int MAX_PER_ROW = O.sp_max_per_row.GetOpt(baselevel);
  const int MAX_PER_ROW_CLASSIC = O.sp_max_per_row_classic.GetOpt(baselevel);
  const double omega = O.sp_omega.GetOpt(baselevel);
  // const bool aux_only = O.sp_aux_only.GetOpt(baselevel); // TODO:placeholder
  const bool aux_only = O.prol_type.GetOpt(baselevel) == Options::PROL_TYPE::AUX_SMOOTHED;

  // NOTE: something is funky with the meshes here ...
  const auto & FM = *static_pointer_cast<TMESH>(fcap->mesh);
  const auto & CM = *static_pointer_cast<TMESH>(cmap->GetMappedMesh());
  const auto & eqc_h = *FM.GetEQCHierarchy();
  const int neqcs = eqc_h.GetNEQCS();
  const auto & fecon = *FM.GetEdgeCM();
  auto fpds = pw_step->GetUDofs().GetParallelDofs();
  auto cpds = pw_step->GetMappedUDofs().GetParallelDofs();

  FM.CumulateData();
  CM.CumulateData();

  /** Because of embedding, this can be nullptr for level 0!
      I think using pure aux on level 0 should not be an issue. **/
  auto fmat = dynamic_pointer_cast<TSPM>(fcap->mat);
  /** "fmat" can be the pre-embedded matrix. In that case we can't use it for smoothing. **/
  bool have_fmat = (fmat != nullptr);
  /** if "fmat" has no pardofs, it is not the original finest level matrix, so fine to use! !**/
  if ( have_fmat && (fmat->GetParallelDofs() != nullptr) )
    { have_fmat &= (fmat->GetParallelDofs() == fpds); }

  // cout << " have fmat " << have_fmat << " " << (fmat != nullptr) << endl; //" " << (fmat->GetParallelDofs() == fpds) << endl;
  // cout << " pds " << fmat->GetParallelDofs() << fpds << endl;

  const TSPM & pwprol = *pw_step->GetProl();

  auto vmap = cmap->GetMap<NT_VERTEX>();

  const size_t FNV = FM.template GetNN<NT_VERTEX>(), CNV = CM.template GetNN<NT_VERTEX>();

  // 16 MB = 16777216 bytes
  LocalHeap lh(64 * 1024 * 1024, "muchmemory", false);

  /** Find Graph, decide if classic or aux prol. **/
  Array<int> cols(20); cols.SetSize0();
  BitArray use_classic(FNV); use_classic.Clear();
  Array<int> proltype(FNV); proltype = 0;

  auto fvdata = get<0>(FM.Data())->Data();
  auto fedata = get<1>(FM.Data())->Data();

  auto cvdata = get<0>(CM.Data())->Data();

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
        auto eqcj = CM.template GetEQCOfNode<NT_VERTEX>(cvj);
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
  FM.template AllreduceNodalData<NT_VERTEX>(dg_wt, [](auto & tab){return std::move(max_table(tab)); }, false);

  Array<IVec<2,double>> trow;
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
      // auto oeq = fmesh.template GetEQCOfNode<NT_VERTEX>(ov);
      auto oeq = CM.template GetEQCOfNode<NT_VERTEX>(cov);
      if (eqc_h.IsLEQ(EQ, oeq)) {
        // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
        auto wt = ENERGY::GetApproxWeight(fedata[int(eis[j])]);
        if ( (pos = tcv.Pos(cov)) == size_t(-1)) {
          trow.Append(IVec<2,double>(cov, wt));
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
    FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
      for (auto fvnr : nodes) {
        if (vmap[fvnr] == -1)
          { continue; }
        bool classic_ok = false;
        classic_ok = (have_fmat && (!aux_only)) ? get_cols_classic(eqc, fvnr) : false;
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
  auto sprol = make_shared<TSPM>(perow, CNV);
  const auto & CSP = *sprol;

  /** #classic, #aux, #triv **/
  int nc = 0, na = 0, nt = 0;
  double fc = 0, fa = 0, ft = 0;

  const double omo = 1.0 - omega;
  TM d(0);
  auto fill_sprol_classic = [&](auto fvnr) {
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
    // bool const doPrint = (fvnr == 47) && (vmap[fvnr] == 58);
    constexpr bool doPrint = false;

    nc++;
    rvs = 0;
    auto fmris = fmat->GetRowIndices(fvnr);
    auto fmrvs = fmat->GetRowValues(fvnr);
    d = fmrvs[find_in_sorted_array(fvnr, fmris)];

    if (BS == 1)
      { CalcInverse(d); }
    else {
      /** Normalize to trace of diagonal mat. More stable Pseudo inverse (?) **/
      double trinv = double(BS) / calc_trace(d);
      d *= trinv;
      // CalcStabPseudoInverse(d, lh);
      CalcPseudoInverseNew(d, lh);
      // CalcPseudoInverseTryNormal(d, lh);
      d *= trinv;
    }

    TM od_pwp(0);
    for (auto j : Range(fmris))
    {
      auto fvj = fmris[j];
      int col = vmap[fmris[j]];
      int colind = find_in_sorted_array(col, ris);
      if (colind == -1)
      {
        // Dirichlet - skip, give up kernel-perservation next to boundary
        continue;
        // Dirichlet - prol from CV, preserve kernels next to BND
        auto Q = ENERGY::GetQiToj(cvdata[vmap[fvnr]], fvdata[fvnr]);
        od_pwp = Q.GetMQ(1.0, fmrvs[j]);
        rvs[colind] -= omega * d * od_pwp;
      }
      else
      {
        if (fvj == fvnr)
          { rvs[colind] += pwprol(fvj, col); }

        od_pwp = fmrvs[j] * pwprol(fvj, col);

        rvs[colind] -= omega * d * od_pwp;
      }
    }

  }; // fill_sprol_classic

  TM Qij(0), Qji(0), QM(0);
  auto fill_sprol_aux = [&](auto fvnr) {
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
    // bool const doPrint = (fvnr == 47) && (vmap[fvnr] == 58);
    constexpr bool doPrint = false;
    if ( doPrint )
    {
      cout << " fill " << fvnr << " -> " << vmap[fvnr] << " AUX " << endl;;
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
    if ( dcol == -1 )
    {
      dcol = nufneibs;
      ufenrs[nufneibs] = -1;
      ufneibs[nufneibs++] = fvnr;
    }
    // cout << " fneibs "; prow(fneibs); cout << endl;
    // cout << "ufneibs "; prow(ufneibs); cout << endl;
    FlatMatrix<TM> rmrow(1, nufneibs, lh); rmrow = 0;

    for (auto j : Range(ufneibs))
    {
      if (j == dcol)
        { continue; }

      const auto & edge = fedges[ufenrs[j]];
      int l = (fvnr == edge.v[0]) ? 0 : 1;

      ENERGY::CalcQs(fvdata[edge.v[l]], fvdata[edge.v[1-l]], Qij, Qji);

      TM EM = ENERGY::GetEMatrix(fedata[ufenrs[j]]);

      QM = Trans(Qij) * EM;
      rmrow(0, j) -= QM * Qji;
      rmrow(0, dcol) += QM * Qij;
    }
    // cout << " rmrow: " << endl; print_tm_mat(cout, rmrow); cout << endl;
    TM d = rmrow(0, dcol);
    // cout << " diag " << endl;
    // print_tm(cout, d);
    if constexpr (BS == 1)
      { CalcInverse(d); }
    else
    {
      /** Normalize to trace of diagonal mat. More stable Pseudo inverse (?) **/
      double trinv = double(BS) / calc_trace(d);
      rmrow *= trinv;
      d *= trinv;
      // CalcStabPseudoInverse(d, lh);
      // if ( )
      // CalcInverse(d);
      FlatMatrix<double> flatD(BS, BS, &d(0,0));
      CalcPseudoInverseWithTol(flatD, lh);
    }
    // cout << " diag inv " << endl;
    // print_tm(cout, d);
    TM od_pwp(0);
    for (auto j : Range(nufneibs))
    {
      auto fvj = ufneibs[j];
      int col = vmap[fvj];
      int colind = find_in_sorted_array(col, ris);
      // if (fvj == fvnr)
      //   { rvs[colind] += omo * pwprol(fvj, col); }
      // else {
      //   od_pwp = rmrow(0, j) * pwprol(fvj, col);
      //   rvs[colind] -= omega * d * od_pwp;
      // }
      if ( fvj == fvnr )
      {
        rvs[colind] += pwprol(fvj, col);
      }
      od_pwp = rmrow(0, j) * pwprol(fvj, col);
      rvs[colind] -= omega * d * od_pwp;
    }

    // if constexpr(Height<TM>() == 6)
    // {
    //   // if ( fvnr == 2241 || fvnr == 1840)
    //   {
    //     Iterate<6>([&](auto l)
    //     {
    //       double sum = 0;

    //       for (auto j : Range(rvs.Size()))
    //       {
    //         sum += rvs[j](l.value, l.value);
    //       }
    //       sum = abs(1.0 - sum);
    //       if ( ( sum > 1e-4 ) || ( fvnr == 2241 || fvnr == 1840) )
    //       cout << "  SA-SP row " << fvnr << "." << int(l.value) << " diagSum = " << sum << endl;
    //     });
    //   }
    // }

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
  if ( neqcs > 1 )
  {
    Array<int> eqc_perow(neqcs); eqc_perow = 0;
    FM.template ApplyEQ<NT_VERTEX>( Range(1, neqcs), [&](auto EQC, auto V) {
      eqc_perow[EQC] += perow[V];
    }, false); // all!
    Table<IVec<2,int>> ex_ris(eqc_perow);
    Table<TM> ex_rvs(eqc_perow); eqc_perow = 0;
    FM.template ApplyEQ<NT_VERTEX>( Range(1, neqcs), [&](auto EQC, auto V) {
      auto rvs = sprol->GetRowValues(V);
      auto ris = sprol->GetRowIndices(V);
      for (auto j : Range(ris)) {
        int jeq = CM.template GetEQCOfNode<NT_VERTEX>(ris[j]);
        int jeq_id = eqc_h.GetEQCID(jeq);
        int jlc = CM.template MapENodeToEQC<NT_VERTEX>(jeq, ris[j]);
        ex_ris[EQC][eqc_perow[EQC]] = IVec<2,int>({ jeq_id, jlc });
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

  if ( options->log_level != Options::LOG_LEVEL::NONE ) {
    nc = eqc_h.GetCommunicator().Reduce(nc, NG_MPI_SUM);
    na = eqc_h.GetCommunicator().Reduce(na, NG_MPI_SUM);
    nt = eqc_h.GetCommunicator().Reduce(nt, NG_MPI_SUM);
    if ( eqc_h.GetCommunicator().Rank() == 0 ) {
      size_t FNV = FM.template GetNNGlobal<NT_VERTEX>();
      cout << "NV,   nc/na/nt " << FNV << ", " << nc << " " << na << " " << nt << endl;
      cout << "fracs c/a/t    " << double(nc)/FNV << " " << double(na)/FNV << " " << double(nt)/FNV << endl;
    }
  }

  if ( options->log_level == Options::LOG_LEVEL::DBG )
  {
    std::ofstream of("SP_semi_aux_rk_" + std::to_string(FM.GetEQCHierarchy()->GetCommunicator().Rank()) +
                                 "_l_" + std::to_string(fcap->baselevel) + ".out");

    print_tm_spmat(of, *sprol);
  }

  auto prolMap = make_shared<ProlMap<TM>>(sprol, pw_step->GetUDofs(), pw_step->GetMappedUDofs());

  return prolMap;
} // VertexAMGFactory::SemiAuxSProlMap




template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
AuxSProlMap (shared_ptr<BaseDOFMapStep> pw_step,
                      shared_ptr<BaseCoarseMap> cmap,
                      shared_ptr<BaseAMGFactory::LevelCapsule> fcap)
{
  static Timer t("AuxSProlMap");
  RegionTimer rt(t);

  if (pw_step == nullptr)
    { throw Exception("Need pw-map for SmoothedProlMap!"); }
  if (cmap == nullptr)
    { throw Exception("Need cmap for SmoothedProlMap!"); }

  auto prol_map =  my_dynamic_pointer_cast<ProlMap<TM>> (pw_step, "AuxSProlMap, ProlMap");

  const TMESH & FM(static_cast<TMESH&>(*cmap->GetMesh())); FM.CumulateData();
  const TMESH & CM(static_cast<TMESH&>(*cmap->GetMappedMesh())); CM.CumulateData();
  const TSPM & pwprol = *prol_map->GetProl();

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
  Array<IVec<2,double>> trow;
  Array<int> tcv, fin_row;
  Array<double> dg_wt(NFV); dg_wt = 0;
  FM.template Apply<NT_EDGE>([&](auto & edge) LAMBDA_INLINE {
auto approx_wt = ENERGY::GetApproxWeight(edata[edge.id]);
dg_wt[edge.v[0]] = max2(dg_wt[edge.v[0]], approx_wt);
dg_wt[edge.v[1]] = max2(dg_wt[edge.v[1]], approx_wt);
    }, false );
  FM.template AllreduceNodalData<NT_VERTEX>(dg_wt, [](auto & tab){return std::move(max_table(tab)); }, false);
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
  // auto oeq = fmesh.template GetEQCOfNode<NT_VERTEX>(ov);
  auto oeq = CM.template GetEQCOfNode<NT_VERTEX>(cov);
  if (eqc_h.IsLEQ(EQ, oeq)) {
    // auto wt = self.template GetWeight<NT_EDGE>(fmesh, all_fedges[int(eis[j])]);
    auto wt = ENERGY::GetApproxWeight(edata[int(eis[j])]);
    if ( (pos = tcv.Pos(cov)) == size_t(-1)) {
      trow.Append(IVec<2,double>(cov, wt));
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
  shared_ptr<TSPM> rmat = make_shared<TSPM>(perow, NCV);
  const TSPM & RM = *rmat;

  /** Fill Prolongation **/
  LocalHeap lh(2000000, "hold this", false); // ~2 MB LocalHeap
  Array<IVec<2,int>> une(20);
  TM Q(0), Qij(0), Qji(0), diag(0), rvl(0), ID(0); SetIdentity(ID);
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
IVec<2,int> ME ({ V, -1 });
une.Append(ME);
for (auto jn : Range(neibs)) {
  int n = neibs[jn];
  int CN = vmap[n];
  auto pos = find_in_sorted_array(CN, grow);
  if (pos != -1)
    { une.Append(IVec<2,int>({n, int(neibeids[jn])})); }
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
maxtr /= ngbla::Height<TM>();
for (auto l : Range(une)) {
  if (l != MEpos) {
    const auto & edge = all_fedges[une[l][1]];
    int L = (V == edge.v[0]) ? 0 : 1;
    // cout << " l " << l << " L " << L << " edge " << edge << ", un " << une[l][0] << " " << une[l][1] << endl;
    ENERGY::CalcQs(vdata[edge.v[L]], vdata[edge.v[1-L]], Qij, Qji);
    // Q = Trans(Qij) * s_emats[used_edges[l]];
    // TM EMAT = edata[une[l][1]];
    TM EMAT = ENERGY::GetEMatrix(edata[une[l][1]]);
    if constexpr(ngbla::Height<TM>()!=1) {
  // RegTM<0, ngbla::Height<TM>(), ngbla::Height<TM>()>(EMAT);
  // RegTM<0, FACTORY_CLASS::DIM, ngbla::Height<TM>()>(EMAT);
  // if (vmap[une[l][0]] == CV)
  // { RegTM<0, ngbla::Height<TM>(), ngbla::Height<TM>()>(EMAT, maxtr); }
  if (vmap[une[l][0]] == CV) {
    RegTM<0, ngbla::Height<TM>(), ngbla::Height<TM>()>(EMAT);
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

// prt_evv<ngbla::Height<TM>()> (diag, "diag", false);
// if constexpr(ngbla::Height<TM>()!=1) {
//     RegTM<0, ngbla::Height<TM>(), ngbla::Height<TM>()>(diag);
//   }
// CalcPseudoInverse2<ngbla::Height<TM>()>(diag, lh);
// prt_evv<ngbla::Height<TM>()> (diag, "inv diag", false);

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

  shared_ptr<TSPM> sprol = prol_map->GetProl();
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
    auto cumul_sp = make_shared<TSPM>(perow, NCV);
    Array<int> eqc_perow(neqcs); eqc_perow = 0;
    if (neqcs > 1)
FM.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
    eqc_perow[EQC] += perow[V];
  }, false); // all!
    Table<IVec<2,int>> ex_ris(eqc_perow);
    Table<TM> ex_rvs(eqc_perow); eqc_perow = 0;
    if (neqcs > 1)
FM.template ApplyEQ<NT_VERTEX>( Range(size_t(1), neqcs), [&](auto EQC, auto V) {
    auto rvs = sprol->GetRowValues(V);
    auto ris = sprol->GetRowIndices(V);
    for (auto j : Range(ris)) {
      int jeq = CM.template GetEQCOfNode<NT_VERTEX>(ris[j]);
      int jeq_id = eqc_h.GetEQCID(jeq);
      int jlc = CM.template MapENodeToEQC<NT_VERTEX>(jeq, ris[j]);
      ex_ris[EQC][eqc_perow[EQC]] = IVec<2,int>({ jeq_id, jlc });
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

  // return make_shared<ProlMap<TSPM_TM>> (sprol, pw_step->GetParDofs(), pw_step->GetMappedParDofs());
  return make_shared<ProlMap<TM>> (sprol, pw_step->GetUDofs(), pw_step->GetMappedUDofs());
} // VertexAMGFactory::AuxSProlMap

/** END VertexAMGFactory **/

} // namespace amg

#endif
