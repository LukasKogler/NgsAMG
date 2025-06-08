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

#include <aux_mat.hpp>

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

  SpecOpt<int> sp_improve_its = 0;
  SpecOpt<bool> sp_use_roots = true;
  SpecOpt<bool> sp_use_asb   = false;

  SpecOpt<bool> improve_c_aux_mats = false;

  // /** Discard **/
  // int disc_max_bs = 5;

  /** for elasticity - scaling of rotations on level 0 **/
  SpecOpt<double> rot_scale = 1.0;

  /** mostly for debugging **/
  bool use_aux_mat = false;
  SpecOpt<bool> check_aux_mats = false;


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

    sp_improve_its.SetFromFlags(flags, prefix + "sp_improve_its");
    sp_use_roots.SetFromFlags(flags, prefix + "sp_use_roots");
    sp_use_asb.SetFromFlags(flags, prefix + "sp_use_asb");
    improve_c_aux_mats.SetFromFlags(flags, prefix + "improve_c_aux_mats");

    // TURNED OFF because it does not (yet?) work
    improve_c_aux_mats = false;

    rot_scale.SetFromFlags(flags, prefix + "rot_scale");

    use_aux_mat = flags.GetDefineFlagX(prefix + "use_aux_mat").IsTrue();
    check_aux_mats.SetFromFlags(flags, prefix + "check_aux_mats");
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
  int const improveIts = O.sp_improve_its.GetOpt(0);

  int const totalSteps = improveIts + 1;

  double const baseOmega = O.sp_omega.GetOpt(0);
  double const omega     = 1 - pow(1 - baseOmega, 1./totalSteps);
  // double const omega     = baseOmega;

  fMesh.CumulateData();
  cMesh.CumulateData();

  auto const &eqc_h = *fMesh.GetEQCHierarchy();
  auto const &fECon = *fMesh.GetEdgeCM();

  if ( eqc_h.IsTrulyParallel() )
  {
    std::cerr << " WARNING!! sp_improve_its option is not fully parallelized!" << std::endl;
  }

  auto fVData = get<0>(fMesh.Data())->Data();
  auto fEData = get<1>(fMesh.Data())->Data();

  auto cVData = get<0>(cMesh.Data())->Data();

  auto vMap = cMap.template GetMap<NT_VERTEX>();
  auto aggs = cMap.template GetMapC2F<NT_VERTEX>();

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

    if ( aggs[CV][0] == V )
    {
      cols.Append(CV);
      return;
    }

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

  Array<int> replaceColPos;

  // smooth prol-rows
  for (auto k : Range(totalSteps))
  {
    if ( k > 0 )
    {
      MatMultABUpdateVals(fMat, *embProl, *A_embP);
    }

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

      Mat<BSF, BSF, double> dInv = fMat(k, k);
      CalcInverse(dInv); // no issues with inverse here!

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

        if ( where == INTERSECTION ) // used col
        {
          prolVals[idxP] -= omega * upVal;
        }
        else // unused col
        {
          // instead add as contrib to reCols (i.e. cvs of fvs from initial emb!)
          for (auto idx : replaceColPos)
          {
            auto const repCol = prolCols[idx];

            // TQ::MQ/GetMQ/etc. are hard-coded to BS x BS -> BS x BS, we need BSF x BS -> BSF x BS
            // prolVals[idx] += ENERGY::GetQiToj(cVData[repCol], cVData[allCols[idxA]]).GetMQ(-omega * repFac, upVal);
            Mat<BS,BS,double> Q;
            SetIdentity(Q);
            ENERGY::GetQiToj(cVData[repCol], cVData[allCols[idxA]]).MQ(Q);
            
            prolVals[idx] -= omega * repFac * upVal * Q;
          }
        }
      });
    }
  }


  // TOOD: re-smooth?
  if ( O.log_level == Options::LOG_LEVEL::DBG )
  {
    std::ofstream of("embProl.out");
    print_tm_spmat(of, *embProl);
  }

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


template<int BS>
constexpr int FINEBS()
{
  if constexpr(BS == 6)
  {
    return 3;
  }
  else if constexpr(BS == 3)
  {
    return 2;
  }
  else {
    return BS;
  }
} // FINEBS


template<int IMIN, int N, int NN> INLINE void RegTMXX (Mat<NN,NN,double> & m, double maxadd = -1)
{
  // static Timer t(string("RegTM<") + to_string(IMIN) + string(",") + to_string(3) + string(",") + to_string(6) + string(">")); RegionTimer rt(t);
  static_assert( (IMIN + N <= NN) , "ILLEGAL RegTM!!");
  static Matrix<double> M(N,N), evecs(N,N);
  static Vector<double> evals(N);
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      M(i.value, j.value) = m(IMIN+i.value, IMIN+j.value);
    });
  });
  LapackEigenValuesSymmetric(M, evals, evecs);
  const double eps = max2(1e-15, 1e-8 * evals(N-1));
  double min_nzev = 0; int nzero = 0;
  for (auto k : Range(N))
    if (evals(k) > eps)
      { min_nzev = evals(k); break; }
    else
      { nzero++; }
  if (maxadd >= 0)
    { min_nzev = min(maxadd, min_nzev); }
  if (nzero < N) {
    for (auto l : Range(nzero)) {
      Iterate<N>([&](auto i) {
          Iterate<N>([&](auto j) {
            m(IMIN+i.value, IMIN+j.value) += min_nzev * evecs(l, i.value) * evecs(l, j.value);
          });
      });
    }
  }
  else {
    SetIdentity(m);
    if (maxadd >= 0)
      { m *= maxadd; }
  }
} // RegTM

template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dofSteps,
          shared_ptr<AMGLevel> &fLev,
          shared_ptr<AMGLevel> &cLev)
{
  auto & O = static_cast<Options&>(*options);

  if ( O.check_aux_mats.GetOpt(0) )
  {
    shared_ptr<BaseMatrix> A    = nullptr;
    shared_ptr<BaseMatrix> Ahat = nullptr;

    auto fM = my_dynamic_pointer_cast<TMESH>(fLev->cap->mesh, "Vertex MapLevel - for CheckAux F-MESH");

    if ( fLev->level == 0 )
    {
      static constexpr int FBS = FINEBS<BS>();
      typedef stripped_spm<Mat<FBS,FBS,double>> TSPM_F;

      if ( auto fA = dynamic_pointer_cast<TSPM_F>(fLev->cap->mat) )
      {
        A = fA;

        Ahat = AssembleAhatSparse<ENERGY, TMESH, TSPM_F>(*fM, true, [&](auto &a, auto const &b) {
          if constexpr(BS  != 1) // so it compiles
          {
            Iterate<FBS>([&](auto j) {
              Iterate<FBS>([&](auto i) {
                a(i.value, j.value) += b(i.value, j.value);
              });
            });
          }
          else
          {
            a += b;
          }
        });
      }
    }

    if ( A == nullptr )
    {
      typedef stripped_spm<Mat<BS,BS,double>> TSPM_F;

      auto fA = my_dynamic_pointer_cast<TSPM_F>(fLev->cap->mat, "Vertex MapLevel - for CheckAux F-MAT");

      A = fA;
      auto Ah = AssembleAhatSparse<ENERGY, TMESH>(*fM, true);
      Ahat = Ah;

      if ( fLev->level > 0 )
      {
        if constexpr( BS > 1 ) // hacks, hacks everywhere ...
        {
          for (auto k : Range(A->Height()))
          {
            RegTMXX<0,BS,BS>((*fA)(k,k));
            RegTMXX<0,BS,BS>((*Ah)(k,k));
          }
        }
      }

      {
        std::ofstream out("ngs_amg_Ahat_l_" + std::to_string(fLev->level) + ".out");
        // out << *Ahat << std::endl;
        print_tm_spmat(out, *Ah); out << endl;
      }
    }

    std::string msg0 = "Aux-Matrix equivalence on level " + std::to_string(fLev->level) + " (lam_lin Ahat <= A <=lam_max Ahat)";

    Ahat->SetInverseType(SPARSECHOLESKY);

    auto AhatInv = Ahat->InverseMatrix(fLev->cap->free_nodes);
    // cout << " AhatInv = " << endl << *AhatInv << endl;

    auto comm = fM->GetEQCHierarchy()->GetCommunicator();

    cout << endl << endl;
    DoTest(*A, *AhatInv, comm, msg0);
    cout << endl << endl;
  }

  // if (O.check_aux_mats.GetOpt(cLev->level))
  // {
  //   auto cA = my_dynamic_pointer_cast<TSPM>(cLev->cap->mat, "Vertex MapLevel - for CheckAux C-MAT");
  //   auto cM = my_dynamic_pointer_cast<TMESH>(cLev->cap->mesh, "Vertex MapLevel - for CheckAux C-MESH");

  //   std::string msg = "Aux-Matrix equivalence on level " + std::to_string(cLev->level);

  //   cout << endl << endl;
  //   CheckAuxMatEquivalence<ENERGY, TMESH, TSPM>(*cM, cLev->cap->free_nodes, *cA, msg);
  //   cout << endl << endl;
  // }

  // With use_emb_sp, the first step (EmbeddedSProl) already includes the embedding so
  // we don't need it here!
  size_t off = (fLev->level == 0 && O.use_emb_sp) ? 1 : 0;

  return BaseAMGFactory::MapLevel(dofSteps.Range(off, dofSteps.Size()), fLev, cLev);
} // VertexAMGFactory::MapLevel


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

  if( fcap->baselevel == 0 &&
      ( prol_type != Options::PROL_TYPE::PIECEWISE ) &&
      O.use_emb_sp &&
      ( !cap->mesh->GetEQCHierarchy()->IsTrulyParallel() ) ) // not implemented in parallel ATM!
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
      case(Options::PROL_TYPE::SEMI_AUX_SMOOTHED) : { step = SemiAuxSProlMap(PWProlMap(*cmap, *fcap, *ccap), cmap, fcap, embMap); break; }
    }
  }

  // if ( ( O.improve_c_aux_mats.GetOpt( fcap->baselevel ) ) &&
  //      ( prol_type != Options::PROL_TYPE::PIECEWISE ) )
  if ( O.improve_c_aux_mats.GetOpt( fcap->baselevel ) )
  {
    ImproveCoarseEnergy(*fcap, *ccap, *step);
  }

  return step;
} // BuildCoarseDOFMap


template<class ENERGY, class TMESH, int BS>
void
VertexAMGFactory<ENERGY, TMESH, BS>::
ImproveCoarseEnergy(LevelCapsule         &fCap,
                    LevelCapsule         &cCap,
                    BaseDOFMapStep const &dofStep)
{
  TMESH const &fMesh = *my_dynamic_pointer_cast<TMESH const>(fCap.mesh, " ImproveCoarseEnergy - FMESH");
  TMESH const &cMesh = *my_dynamic_pointer_cast<TMESH const>(cCap.mesh, " ImproveCoarseEnergy - CMESH");

  ProlMap<TM> const &prolMap = *my_dynamic_cast<ProlMap<TM> const>(&dofStep, " ImproveCoarseEnergy - map");

  fMesh.CumulateData();
  cMesh.CumulateData();

  auto const fNV = fMesh.template GetNN<NT_VERTEX>();
  auto const fNE = fMesh.template GetNN<NT_EDGE>();

  auto const cNV = cMesh.template GetNN<NT_VERTEX>();

  // coarserst level
  if ( cMesh.template GetNN<NT_EDGE>() == 0 )
  {
    return;
  }

  auto fVData = get<0>(fMesh.Data())->Data();
  auto fEData = get<1>(fMesh.Data())->Data();

  auto cVData = get<0>(cMesh.Data())->Data();
  auto cEData = get<1>(cMesh.Data())->Data();

  auto const &P  = *prolMap.GetProl();
  auto const &PT = *prolMap.GetProlTrans();

  auto fEdges = fMesh.template GetNodes<NT_EDGE>();

  auto AhatF = AssembleAhatSparse<ENERGY>(fMesh, true);
  auto AhatC = AssembleAhatSparse<ENERGY>(cMesh, true);

if constexpr(BS == 1) // makes it easier for init. version
{
  cout << " ImproveCoarseEnergy, NV " << fNV << " -> " << cNV << endl;
  cout << " ImproveCoarseEnergy, NE " << fNE << " -> " <<  cMesh.template GetNN<NT_EDGE>() << endl;

  // A ~ Ahat = GT Alpha G
  Array<int> perow(fNE);
  perow = 1;
  // auto Alpha = make_shared<DiagonalMatrix<TM>>(fNV);

  auto Alpha = make_shared<SparseMat<BS,BS>>(perow, fNE);

  for (auto k : Range(fNE))
  {
    Alpha->GetRowIndices(k)[0] = k;
    // Alpha(k, k) = ENERGY::GetEMatrix(fEdata[k]);
    Alpha->GetRowValues(k)[0] = ENERGY::GetEMatrix(fEData[k]);
  }

  // We have
  //    PT Ahat P = (PG)^T Alpha PG = G_H^T S^T Alpha S G_H
  // Where "S" is the |fine edge| x |coarse edge| matrix with entries
  //   S_{ij,IJ} = P_iI P_jJ
  // We write M := S^T Alpha S and try to find diagonal Alpha_H
  // such that AlphaH ~ M and therefore A_H = PT A P ~ G_H^T Alpha_H G_H = Ahat_H

  perow.SetSize(fNE);
  perow = 2;

  auto pG_h = make_shared<SparseMat<BS,BS>>(perow, fNV);
  auto const &G_h = *pG_h;

  for (auto k : Range(fNE))
  {
    auto fEdge = fEdges[k];

    G_h.GetRowIndices(k)[0] = fEdge.v[0];
    G_h.GetRowIndices(k)[1] = fEdge.v[1];

    auto [Qij, Qji] = ENERGY::GetQijQji(fVData[fEdge.v[0]], fVData[fEdge.v[1]]);
    
    Qij.SetQ(-1.0, G_h.GetRowValues(k)[0]);
    Qji.SetQ(G_h.GetRowValues(k)[1]);
  }

  auto pG_h_T = TransposeSPM(G_h);
  auto const &G_h_T = *pG_h_T;

  bool upCEs = true;

  size_t cNE = cMesh.template GetNN<NT_EDGE>();
  FlatArray<AMG_Node<NT_EDGE>> cEdges = cMesh.template GetNodes<NT_EDGE>();

  shared_ptr<SparseMatrix<double>> pNewCEcon;
  Array<AMG_Node<NT_EDGE>> newCEdges;

  if ( upCEs )
  {
    std::set<std::tuple<int, int>> setCE;

    auto ItCreateEdges = [&](auto lam)
    {
      auto baseCEdges = cMesh.template GetNodes<NT_EDGE>();

      for (auto cENr : Range(cMesh.template GetNN<NT_EDGE>()))
      {
        lam(baseCEdges[cENr].v[0], baseCEdges[cENr].v[1]);
      }

      for (auto fENr : Range(fNE))
      {
        auto const &fEdge = fEdges[fENr];

        auto colsi = P.GetRowIndices(fEdge.v[0]);    
        auto colsj = P.GetRowIndices(fEdge.v[1]);    

        // an entry for every edge I-J, I in colsi, j in colsj
        // cout << " itGraph, fENr = " << fENr << endl;
        // cout << "    colsi = "; prow(colsi); cout << endl;
        // cout << "    colsj = "; prow(colsj); cout << endl;

        for (auto I : colsi)
        {
          for (auto J : colsj)
          {
            if ( I != J )
            {
              lam(I, J);
            }
          }
        }
      }
    };
    
    ItCreateEdges([&](int const &I, int const &J)
    {
      if ( I < J )
      {
        setCE.insert(std::make_tuple(I, J));
      }
      else
      {
        setCE.insert(std::make_tuple(J, I));
      }
    });

    cNE = setCE.size();

    cout << " #CE " << cMesh.template GetNN<NT_EDGE>() << " -> " << cNE << endl;

    perow.SetSize(cNV);

    newCEdges.SetSize(setCE.size());
    cEdges.Assign(newCEdges);

    int cnt = 0;
    for (auto const &tup : setCE)
    {
      int const eID = cnt++;

      auto [ v0, v1 ] = tup;

      cEdges[eID].id = eID;
      cEdges[eID].v  = { v0, v1 };

      perow[v0]++;
      perow[v1]++;
    }

    pNewCEcon = make_shared<SparseMatrix<double>>(perow, cNV);
    auto const &cEcon = *pNewCEcon;

    perow = 0;

    for (auto const &cEdge : cEdges)
    {
      for (auto l : Range(2))
      {
        cEcon.GetRowIndices(cEdge.v[l])[perow[cEdge.v[l]]] = cEdge.v[1-l];
        cEcon.GetRowValues(cEdge.v[l])[perow[cEdge.v[l]]]  = int(cEdge.id);
        perow[cEdge.v[l]]++;
      }
    }
  }

  SparseMatrix<double> const &cEcon = upCEs ? *pNewCEcon: *cMesh.GetEdgeCM();

  perow.SetSize(cNE);
  perow = 2;

  auto pG_H = make_shared<SparseMat<BS,BS>>(perow, cNV);
  auto const &G_H = *pG_H;

  for (auto k : Range(cNE))
  {
    auto cEdge = cEdges[k];

    G_H.GetRowIndices(k)[0] = cEdge.v[0];
    G_H.GetRowIndices(k)[1] = cEdge.v[1];

    auto [Qij, Qji] = ENERGY::GetQijQji(cVData[cEdge.v[0]], cVData[cEdge.v[1]]);
    
    Qij.SetQ(-1.0, G_H.GetRowValues(k)[0]);
    Qji.SetQ(G_H.GetRowValues(k)[1]);
  }

  auto pG_H_T = TransposeSPM(G_H);
  auto const &G_H_T = *pG_H_T;

  // there are better ways to do this, but for now, for testing, explicitly set up
  // S as a sparse-matrix
  perow.SetSize(fNE);
  perow = 0;

  std::set<int> cols; // sorted!

  auto itGraph = [&](auto lam)
  {
    for (auto fENr : Range(fNE))
    {
      auto const &fEdge = fEdges[fENr];

      auto colsi = P.GetRowIndices(fEdge.v[0]);    
      auto colsj = P.GetRowIndices(fEdge.v[1]);    

      // an entry for every edge I-J, I in colsi, j in colsj
      cols.clear();

      // cout << " itGraph, fEdge = " << fEdge << endl;
      // cout << "    colsi = "; prow(colsi); cout << endl;
      // cout << "    colsj = "; prow(colsj); cout << endl;

      for (auto I : colsi)
      {
        for (auto J : colsj)
        {
          if ( I != J )
          {
            int pos = find_in_sorted_array(J, cEcon.GetRowIndices(I));

            if ( pos != -1 )
            {
              int const cENum(cEcon.GetRowValues(I)[pos]);

              // cout << " add " << I << " x " << J << " = cE " << cENum << endl;

              cols.insert(cENum);
            }
            else
            {
              // // find connection I <-> K <-> J to take the weight
              // auto commonNeibs = intersect_sorted_arrays(cEcon.GetRowIndices(I),
              //                                            cEcon.GetRowIndices(J),
              //                                            lh);

              auto neibsI = cEcon.GetRowIndices(I);
              auto eidsI  = cEcon.GetRowValues(I);
              auto neibsJ = cEcon.GetRowIndices(J);
              auto eidsJ  = cEcon.GetRowValues(J);

              int c = 0;
              iterate_intersection(
                neibsI, neibsJ,
                [&](auto idxNI, auto idxNJ)
                {
                  auto const neib = neibsI[idxNI];

                  int const eIdIN = int(eidsI[idxNI]);
                  int const eIdJN = int(eidsI[idxNJ]);

                  cols.insert(eIdIN);
                  cols.insert(eIdJN);

                  // cout << " redirecting " << I << " - " << J << " over " << neib << endl;

                  c++;
                }
              );

              if ( c == 0 )
              {
                cout << " connection " << I << " - " << J << " NO COMMON NEIB FOUND! " << endl;
              }


            }
          }
        }
      }

      lam(fENr, cols);
    }
  };

  itGraph([&](auto row, auto const &cols) { perow[row] = cols.size(); });

  auto spS = make_shared<SparseMat<BS, BS>>(perow, cNE);

  auto const &S = *spS;

  // better would be second iteration with std::map or sth
  itGraph(
    [&](auto row, auto const &cols)
    {
      // cout << " itG2, row " << row << " f " << fNE << ", cols " << cols.size() << " into " << S.GetRowIndices(row).Size() << endl;
      std::copy(cols.begin(), cols.end(), S.GetRowIndices(row).begin());
      // cout << "     OK " << endl;
    }
  );

  for (auto fENr : Range(fNE))
  {
    auto rCols = S.GetRowIndices(fENr);
    auto rVals = S.GetRowValues(fENr);

    auto const &fEdge = fEdges[fENr];

    auto colsi = P.GetRowIndices(fEdge.v[0]);    
    auto valsi = P.GetRowValues(fEdge.v[0]);    

    auto colsj = P.GetRowIndices(fEdge.v[1]);    
    auto valsj = P.GetRowValues(fEdge.v[1]);    

    rVals = 0.0;

    // cout << " FILL " << fENr << " f " << fNE << endl;
    // cout << "    colsi = "; prow(colsi); cout << endl;
    // cout << "    colsj = "; prow(colsj); cout << endl;
    // cout << " rCols: "; prow(rCols); cout << endl;
    // cout << " rVals: "; prow(rVals); cout << endl;

    for (auto idxI : Range(colsi))
    {
      auto const I = colsi[idxI];

      for (auto idxJ : Range(colsj))
      {
        auto const J = colsj[idxJ];

        if ( I != J )
        {
          int pos = find_in_sorted_array(J, cEcon.GetRowIndices(I));

          if ( pos != -1 )
          {
            int const cENum(cEcon.GetRowValues(I)[pos]);

            int const rPos = find_in_sorted_array(cENum, rCols);

            // cout << " add " << I << " x " << J << " = cE " << cENum << " -> " << rPos << endl;

            // fEdge.v[0] < fEdge.v[1] is given
            double const orient = ( I < J ) ? 1.0 : -1.0;

            auto const piI = valsi[idxI];
            auto const pjJ = valsj[idxJ];

            rVals[rPos] += orient * piI * pjJ;
            // rVals[rPos] += piI * pjJ;

            // cout << " vals are " << piI << " " << pjJ << " -> rVal now " << rVals[rPos] << endl;
          }
          else
          {

            auto neibsI = cEcon.GetRowIndices(I);
            auto eidsI  = cEcon.GetRowValues(I);
            auto neibsJ = cEcon.GetRowIndices(J);
            auto eidsJ  = cEcon.GetRowValues(J);

            double totStrength = 0;

            // move weight from I-J to I-N, N-J proportionally
            // to I-N,N-J connection strength
            iterate_intersection(
              neibsI, neibsJ,
              [&](auto idxNI, auto idxNJ)
              {
                auto const neib = neibsI[idxNI];

                int const eIdIN = int(eidsI[idxNI]);
                int const eIdJN = int(eidsI[idxNJ]);

                double sIN = ENERGY::GetApproxWeight(cEData[eIdIN]);
                double sJN = ENERGY::GetApproxWeight(cEData[eIdJN]);

                double s = 2 * sIN * sJN / ( sIN + sJN );

                totStrength += s;
              }
            );

            auto const piI = valsi[idxI];
            auto const pjJ = valsj[idxJ];

            iterate_intersection(
              neibsI, neibsJ,
              [&](auto idxNI, auto idxNJ)
              {
                auto const neib = neibsI[idxNI];

                int const eIdIN = int(eidsI[idxNI]);
                int const eIdJN = int(eidsI[idxNJ]);

                double sIN = ENERGY::GetApproxWeight(cEData[eIdIN]);
                double sJN = ENERGY::GetApproxWeight(cEData[eIdJN]);

                double s = 2 * sIN * sJN / ( sIN + sJN );

                double frac = s / totStrength;

                int const rPosI = find_in_sorted_array(eIdIN, rCols);
                int const rPosJ = find_in_sorted_array(eIdJN, rCols);

                double const orientI = ( I < neib ) ? 1.0 : -1.0;
                double const orientJ = ( J < neib ) ? 1.0 : -1.0;

                // cols IN, jN
                rVals[rPosI] += orientI * frac * piI * pjJ; // * QN->J or sth
                rVals[rPosJ] += orientJ * frac * piI * pjJ;
              }
            );
          }
        }
      }
    }
  }
  
  auto spST = TransposeSPM(S);
  auto const &ST = *spST;

  auto Alpha_S = MatMultAB(*Alpha, S);

  auto spM = MatMultAB(ST, *Alpha_S);
  auto const &M = *spM;

  auto spDiagM = make_shared<DiagonalMatrix<TM>>(cNE);
  auto &diagM = *spDiagM;

  auto spDiagMInv = make_shared<DiagonalMatrix<TM>>(cNE);
  auto &diagMInv = *spDiagMInv;


  perow.SetSize(cNE);
  perow = 1;
  auto spDiagMS = make_shared<SparseMat<BS,BS>>(perow, cNE);
  auto &diagMS = *spDiagMS;

  auto spCAlpha = make_shared<DiagonalMatrix<TM>>(cNE);
  auto &cAlpha = *spCAlpha;

  auto spCAlphaSPM = make_shared<SparseMat<BS,BS>>(perow, cNE);
  auto &cAlphaSPM = *spCAlphaSPM;

  // auto spCAlphaInv = make_shared<DiagonalMatrix<TM>>(cNE);
  // auto &cAlphaInv = *spCAlphaInv;

  for (auto k : Range(cNE))
  {
    // cout << " diag " << k << "/" << cNE << endl;
    TM d = M(k,k);

    // TM d = 0;
    // for (auto v : M.GetRowValues(k))
    // {
    //   d += v;
    // }

    diagM(k) = d;
    diagMS.GetRowIndices(k)[0] = k;
    diagMS.GetRowValues(k)[0]  = d;

    CalcInverse(d);
    diagMInv(k) = d;

    if ( upCEs )
    {
      d = 1;
    }
    else
    {
      d = ENERGY::GetEMatrix(cEData[k]);
    }
    cAlpha(k) = d;
    cAlphaSPM(k,k) = d;
    // CalcInverse(d);
    // cAlphaInv(k) = d;
  }

  auto dM_G    = MatMultAB(diagMS, G_H);
  auto GT_dM_G = MatMultAB(G_H_T, *dM_G);

  auto M_G = MatMultAB(M, G_H);
  auto GT_M_G = MatMultAB(G_H_T, *M_G);

  auto Ahat_P = MatMultAB(*AhatF, P);
  auto PT_Ahat_P = MatMultAB(PT, *Ahat_P);

  auto Alpha_G = MatMultAB(cAlphaSPM, G_H);
  auto GT_Alpha_G = MatMultAB(G_H_T, *Alpha_G);

  if ( !upCEs ) // only do that if we make cEdges complete
  { // check that PT Ahat P = G_HT M G_H
    cout << " CHECK PT_Ahat_P - M :" << endl;
    for (auto k : Range(cNV))
    {
      auto risPAP = PT_Ahat_P->GetRowIndices(k);
      auto rvsPAP = PT_Ahat_P->GetRowValues(k);

      auto risGMG = GT_M_G->GetRowIndices(k);
      auto rvsGMG = GT_M_G->GetRowValues(k);

      cout << " row " << k << "/" << cNE << ":" << endl;

      bool isOK = true;
      if ( risPAP.Size() != risGMG.Size() )
      {
        cout << " SIZE mismatch!" << endl;
        isOK = false;
      }

      if ( isOK )
      {
        for (auto j : Range(risPAP))
        {
          auto diff = abs(rvsPAP[j] - rvsGMG[j]);

          if ( diff > 1e-12 * min(abs(rvsPAP[j]), abs(rvsGMG[j])))
          {
            cout << " VAL mismatch " << j << " diff " << diff << ", rel = " << diff / min(abs(rvsPAP[j]), abs(rvsGMG[j])) << endl;
            isOK = false;
          }
        }
      }

      double msum = 0.0;

      for ( auto j : Range(risGMG))
      {
        msum += rvsGMG[j];
      }

      if ( msum > 1e-12 )
      {
        isOK = false;
        cout << " RSUM M is non-zero, rs = " << msum << endl;
      }

      if ( !isOK )
      {
        cout << "    ris PAP "; prow2(risPAP); cout << endl;
        cout << "    ris GMG "; prow2(risGMG); cout << endl;
        cout << "    rvs PAP "; prow2(rvsPAP); cout << endl;
        cout << "    rvs GMG "; prow2(rvsGMG); cout << endl;
      }

    }

  }

  shared_ptr<BitArray> fn = cCap.free_nodes;

  if ( fn != nullptr )
  {
    fn = make_shared<BitArray>(*fn);
  }
  else
  {
    fn = make_shared<BitArray>(cNV);
    fn->Clear();
    fn->Invert();
  }

  for (auto k : Range(cNV))
  {
    if ( GT_dM_G->GetRowIndices(k).Size() == 0 )
    {
      // isolated vertex - set Dirichlet
      //   GT_X_G has no entries in these rows, instead of adding a way to force
      //   an entry into MatMultAB result, add it DIRI, that is easier
      fn->SetBit(k);
    }
    else
    {
      double const vW = cVData[k][0];

      // if (vW > 0)
      // {
      //   cout << " add C v-ctrb " << vW << " to " << k << "/" << cNV << endl;
      // }

      // (*PT_Ahat_P)(k,k)  += vW;
      (*GT_dM_G)(k,k)    += vW;
      (*GT_M_G)(k,k)     += vW;
      (*GT_Alpha_G)(k,k) += vW;
    }
  }

  {
    std::ofstream of("ngs_amg_G_H_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, G_H);
  }
  {
    std::ofstream of("ngs_amg_G_H_T_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, G_H_T);
  }
  {
    std::ofstream of("ngs_amg_Alpha_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, *Alpha);
  }
  {
    std::ofstream of("ngs_amg_PT_Ahat_P_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, *PT_Ahat_P);
  }
  {
    std::ofstream of("ngs_amg_GT_dM_G_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, *GT_dM_G);
  }
  {
    std::ofstream of("ngs_amg_S_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, S);
  }
  {
    std::ofstream of("ngs_amg_ST_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, ST);
  }
  {
    std::ofstream of("ngs_amg_M_l_" + std::to_string(fCap.baselevel) + ".out");
    print_tm_spmat(of, M);
  }
  // auto fn = make_shared<BitArray>(cNV);
  // fn->Clear();
  // fn->Invert();
  // fn->Clear(0); // set one row do DIRI
  // if ( fn->Size() > 1 )
  //   fn->Clear(1); // set one row do DIRI
  // if ( fn->Size() > 2 )
  //   fn->Clear(2); // set one row do DIRI
  // if ( fn->Size() > 1 )
  //   fn->Clear(fn->Size()-1); // set one row do DIRI
  // if ( fn->Size() > 2 )
  //   fn->Clear(fn->Size()-2); // set one row do DIRI

  // auto fn = cCap.free_nodes;

  cout << " fn: " << fn << flush; if ( fn ) cout << " s " << fn->Size() << endl << *fn; cout << endl;

  GT_dM_G->SetInverseType(SPARSECHOLESKY);
  auto GT_dM_G_inv = GT_dM_G->InverseMatrix(fn);

  PT_Ahat_P->SetInverseType(SPARSECHOLESKY);
  auto PT_Ahat_P_inv = PT_Ahat_P->InverseMatrix(fn);
  // auto PT_Ahat_P_inv = MatMultAB(PT, *Ahat_P);

  if ( false ) // not super useful
  {
    std::ofstream of("ngs_amg_diagsCMP_l_" + std::to_string(fCap.baselevel) + ".out");

    // auto cEdges = cMesh.template GetNodes<NT_EDGE>();
  
    int mindod_idx = -1;
    double mindod = 1e12;

    for (auto k : Range(cNE))
    {
      double rSum = 0;
      double odSum = 0;

      auto ris = M.GetRowIndices(k);
      auto rvs = M.GetRowValues(k);
      for (auto j : Range(rvs))
      {
        auto v = rvs[j];
        rSum += abs(v);
        if ( ris[j] != k )
          odSum += abs(v);
      }

      of << " cEdge " << cEdges[k] << ": " << endl;
      // of << "       Alpha = " << cAlpha(k) << endl;
      of << "       dM = "    << diagM(k) << endl;
      // of << "       dM/Alpha = " << diagM(k)/cAlpha(k) << endl;
      of << "       M abs-rsum = " << rSum << endl;
      of << "       M abs-od-sum = " << odSum << endl;
      of << "       dM / rsum = " << diagM(k) / rSum << endl;
      of << "       dM / od-sum = " << diagM(k) / odSum << endl;

      auto dod = diagM(k) / odSum;

      if ( dod < mindod)
      {
        mindod = dod;
        mindod_idx = k;
      }
    }

    of << endl;
    of << " MIN DIAG / OFF-DIAG RATIO = " << mindod << " @ " << mindod_idx << endl;

    cout << " MIN DIAG / OFF-DIAG RATIO = " << mindod << " @ " << mindod_idx << endl;

    of << endl;
  }

  DoTest(M, diagMInv,  "EV-Test diagMInv  x M level " + std::to_string(fCap.baselevel));
  DoTest(*GT_Alpha_G, *PT_Ahat_P_inv, "EV-Test PT_Ahat_P-inv x GT_Alpha_GT level " + std::to_string(fCap.baselevel));
  DoTest(*GT_M_G, *PT_Ahat_P_inv, "EV-Test PT_Ahat_P-inv x GT_M_GT level " + std::to_string(fCap.baselevel));
  DoTest(*GT_M_G, *GT_dM_G_inv, "EV-Test GT_diagM_GT-inv x GT_M_GT level " + std::to_string(fCap.baselevel));
  DoTest(*PT_Ahat_P, *GT_dM_G_inv, "EV-Test GT_diagM_GT-inv x PT_Ahat_P level " + std::to_string(fCap.baselevel));
  DoTest(*GT_dM_G, *PT_Ahat_P_inv, "EV-Test PT_Ahat_P-inv x GT_diagM_GT level " + std::to_string(fCap.baselevel));

  // DoTest(M, cAlphaInv, "EV-Test cAlphaInv x M level " + std::to_string(fCap.baselevel));
  // DoTest(M, diagMInv, "EV-Test cAhatInv x M level " + std::to_string(fCap.baselevel));

  cout << " cEData " << cEData.Size() << " cNE " << cNE << " diagM " << diagM.Height() << " " << diagM.Width() << endl;

  if ( !upCEs )
  {
    for (auto k : Range(cNE))
    {
      cEData[k] = diagM(k);
    }
  }

  cout << " cEData " << cEData.Size() << " cNE " << cNE << " diagM " << diagM.Height() << " " << diagM.Width() << endl;
}


} // VertexAMGFactory::ImproveCoarseEnergy


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


template<int N, int M, class T>
void
operator += (SliceMatrix<T> a, const Mat<N, M, double> & b)
{
  Iterate<N>([&](auto I) {
    Iterate<M>([&](auto J) {
      a(N, M) += b(N, M);
    });
  });
}

// scalar case - keep this only here because it is ugly
template<class T>
void
operator += (SliceMatrix<T> a, T & b)
{
  a(0) += b;
}



INLINE void
RegularizeMatrix (BaseSparseMatrix& mat)
{
  auto spA = dynamic_cast<SparseMatrix<Mat<6, 6, double>>*>(&mat);

  if (spA == nullptr)
  {
    throw Exception("RegularizeMatrix called on garbate!");
    return;
  }

  auto &A = *spA;

  for (auto k : Range(A.Height()))
    { RegTM<0,6,6>(A(k,k)); }
} // RegularizeMatrix


template<class ENERGY, class TMESH, class TSPM>
void
CheckAuxMatEquivalence (TMESH const &FM,
                        shared_ptr<BitArray> freeVerts,
                        TSPM &A,
                        std::string message)
{
  auto Ahat = AssembleAhatSparse<ENERGY, TMESH>(FM, true);

  shared_ptr<BitArray> freeRows = freeVerts;

  // // force-set last row to Dirichlet so we at least have SOMETHING
  // if (freeRows == nullptr)
  // {

  // }

  // cout << " CheckAuxMatEquivalence, A = " << endl;
  // print_tm_spmat(cout, A); cout << endl;

  // cout << " CheckAuxMatEquivalence, Ahat = " << endl;
  // print_tm_spmat(cout, *Ahat); cout << endl;

  // auto AInv = A.InverseMatrix(freeVerts);

  // RegularizeMatrix(A);
  if constexpr(std::is_same<typename TSPM::TENTRY, double>::value == false)
  {
    RegularizeMatrix(*Ahat);
  }

  Ahat->SetInverseType(SPARSECHOLESKY);

  auto AhatInv = Ahat->InverseMatrix(freeVerts);
  // cout << " AhatInv = " << endl << *AhatInv << endl;

  auto comm = FM.GetEQCHierarchy()->GetCommunicator();

  DoTest(A, *AhatInv, comm, message);
  // DoTest(*Ahat, *AInv, comm, message);
} // CheckAuxMatEquivalence


template<class ENERGY, class TMESH, class TAMAT, class TAPMAT, class UPVALS>
INLINE
void
ImproveSProlRow (int            const &fvnr,
                 FlatArray<int>        prolCols,
                 FlatArray<int>        extCols,
                 double         const &omega,
                 TMESH          const &CM,
                 TAMAT          const &A,  // sys-mat   // TPMAT          const &P,  // prol
                 TAPMAT         const &AP, // A * P
                 UPVALS                updateValues,
                 LocalHeap            &lh)
{
  typedef typename ENERGY::TM TM;

  /**
    * Improve an sprol-row without changing the graph.
    *
    * We do something like normal prol-smoothing of the form
    *      P -> (I-omega Dinv A)P = P - omega Dinv AP,
    * with the exception that we add a coarse extension E that maps
    *      E: extCols -> all-AP-cols
    * and the prol-update becomes
    *      P -> P - omega Dinv APE
    * That is, this does NOT increase the graph of "P"!
    *
    * The extension E is simple weighted Q-prol with equal weights.
    * Most of the time, extCols only contains vmap[fvnr].
    */

  static Timer t("ImproveSProlRow");
  RegionTimer rt(t);

  FlatArray<int> extColIdx(extCols.Size(), lh);

  for (auto l : Range(extColIdx))
  {
    extColIdx[l] = find_in_sorted_array(extCols[l], prolCols);
  }

  double const ecFactor = - omega / extCols.Size();

  auto cVData = get<0>(CM.Data())->Data();

  auto aCols = A.GetRowIndices(fvnr);
  auto aVals = A.GetRowValues(fvnr);

  TM dInv = aVals[find_in_sorted_array(fvnr, aCols)];

  if constexpr( Height<TM>() > 1 )
  {
    FlatMatrix<double> flatDI(Height<TM>(), Height<TM>(), &dInv(0, 0));
    CalcPseudoInverseWithTol(flatDI, lh);
  }
  else
  {
    CalcInverse(dInv);
  }

  auto apCols = AP.GetRowIndices(fvnr);
  auto apVals = AP.GetRowValues(fvnr);

  FlatArray<TM> upVals(prolCols.Size(), lh);

  upVals = 0.0;

  iterate_AC(apCols, prolCols, [&](auto where, auto const &idxAP, auto const &idxP)
  {
    if ( where == INTERSECTION )
    {
      upVals[idxP] -= omega * dInv * apVals[idxAP];
    }
    else
    {
      auto const apCol = apCols[idxAP];

      auto const &apColD = cVData[apCol];

      for (auto l : Range(extCols))
      {
        TM upVal = dInv * apVals[idxAP];
        upVals[extColIdx[l]] += ENERGY::GetQiToj(cVData[extCols[l]], cVData[apCol]).GetMQ(ecFactor, upVal);
      }
    }
  });

  updateValues(upVals);
} // ImproveSProlRow


template<class ENERGY, class TMESH, int BS>
shared_ptr<BaseDOFMapStep>
VertexAMGFactory<ENERGY, TMESH, BS>::
SemiAuxSProlMap (shared_ptr<ProlMap<TM>> pw_step,
                 shared_ptr<BaseCoarseMap> cmap,
                 shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
                 shared_ptr<BaseDOFMapStep>   const &embMap)
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
  const bool aux_only = O.prol_type.GetOpt(baselevel) == Options::PROL_TYPE::AUX_SMOOTHED;
  int const improveIts = O.sp_improve_its.GetOpt(baselevel);

  const auto & FM = *static_pointer_cast<TMESH>(fcap->mesh);
  const auto & CM = *static_pointer_cast<TMESH>(cmap->GetMappedMesh());
  const auto & eqc_h = *FM.GetEQCHierarchy();
  const int neqcs = eqc_h.GetNEQCS();
  const auto & fecon = *FM.GetEdgeCM();
  auto fpds = pw_step->GetUDofs().GetParallelDofs();
  auto cpds = pw_step->GetMappedUDofs().GetParallelDofs();

  FM.CumulateData();
  CM.CumulateData();

  if ( (improveIts > 0) && eqc_h.IsTrulyParallel() && (baselevel == 0) )
  {
    std::cerr << " WARNING!! sp_improve_its option is not fully parallelized!" << std::endl;
  }

  /** Because of embedding, this can be nullptr for level 0!
      I think using pure aux on level 0 should not be an issue. **/
  auto actualFmat = dynamic_pointer_cast<TSPM>(fcap->mat);
  shared_ptr<TSPM> fmat = actualFmat;

  // if ( baselevel == 0 )
  // {
  //   fmat = nullptr;
  // }

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
    bool const doPrint = (fvnr == 297) && (vmap[fvnr] == 10);
    // constexpr bool doPrint = false;

    nc++;
    rvs = 0;
    auto fmris = fmat->GetRowIndices(fvnr);
    auto fmrvs = fmat->GetRowValues(fvnr);
    d = fmrvs[find_in_sorted_array(fvnr, fmris)];

    if ( doPrint ) { cout << " CLASS. d for row " << fvnr << ": " << endl; print_tm(cout, d); cout << endl; }

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

    if ( doPrint ) { cout << " d inv row " << fvnr << ": " << endl; print_tm(cout, d); cout << endl; }

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

        if ( doPrint )
        {
          cout << " od-val for " << fvj << " -> cv " << col << " -> col-idx " << colind << ": " << endl;
          print_tm(cout, fmrvs[j]); cout << endl;
          cout << " od_pwp = " << endl;
          print_tm(cout, od_pwp);cout << endl;
          TM up = -1 * d * od_pwp;
          cout << " up = " << endl; print_tm(cout, up); cout << endl;
        }

        od_pwp = fmrvs[j] * pwprol(fvj, col);

        rvs[colind] -= omega * d * od_pwp;
      }
    }

  }; // fill_sprol_classic

  TM Qij(0), Qji(0), QM(0);
  auto fill_sprol_aux = [&](auto fvnr)
  {
    bool const doPrint = (fvnr == 297) && (vmap[fvnr] == 10);

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
    if ( doPrint ) { cout << " AUX. d for row " << fvnr << ": " << endl; print_tm(cout, d); cout << endl; }
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
    if ( doPrint ) { cout << " AUX. inv d for row " << fvnr << ": " << endl; print_tm(cout, d); cout << endl; }
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

      if ( doPrint )
      {
        cout << " od-val for " << fvj << " -> cv " << col << " -> col-idx " << colind << ": " << endl;
        print_tm(cout, rmrow(0, j)); cout << endl;
        cout << " od_pwp = " << endl;
        print_tm(cout, od_pwp);cout << endl;
        TM up = -1 * d * od_pwp;
        cout << " up = " << endl; print_tm(cout, up); cout << endl;
      }

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

  if ( options->log_level > Options::LOG_LEVEL::NORMAL ) {
    nc = eqc_h.GetCommunicator().Reduce(nc, NG_MPI_SUM);
    na = eqc_h.GetCommunicator().Reduce(na, NG_MPI_SUM);
    nt = eqc_h.GetCommunicator().Reduce(nt, NG_MPI_SUM);
    if ( eqc_h.GetCommunicator().Rank() == 0 ) {
      size_t FNV = FM.template GetNNGlobal<NT_VERTEX>();
      cout << "NV,   #class/aux/triv " << FNV << ", " << nc << " " << na << " " << nt << endl;
      cout << "fracs  class/aux/triv    " << double(nc)/FNV << " " << double(na)/FNV << " " << double(nt)/FNV << endl;
    }
  }

  if (improveIts > 0) //  && (neqcs == 1)) // MPI is TODO
  {
    static Timer tImp("SemiAuxSProlMap 0 improve-its");
    auto const &CSP = *sprol;

    fmat = actualFmat;

    if ( fmat == nullptr )
    {
      shared_ptr<BaseMatrix> mat = embMap->AssembleMatrix(fcap->mat);

      fmat = my_dynamic_pointer_cast<TSPM>(mat, "SemiAuxSProlMap - emb-A!");
    }

    shared_ptr<SparseMat<BS,BS>> AP;


    RegionTimer rt(tImp);

    for (auto improveIt : Range(improveIts))
    {
      if ( improveIt == 0 )
      {
        AP = MatMultAB(*fmat, CSP);
      }
      else
      {
        MatMultABUpdateVals(*fmat, CSP, *AP);
      }

      /**
       * Only updates local rows! MPI is TODO, we need "full" rows of AP.
       */
      if ( neqcs > 0 )
      FM.template ApplyEQ2<NT_VERTEX>(Range(0, 1), [&](auto eqc, auto nodes)
      {
        for (auto fvnr : nodes)
        {
          auto ris = CSP.GetRowIndices(fvnr);

          if (ris.Size() > 1)
          {
            auto rvs = CSP.GetRowValues(fvnr);

            HeapReset hr(lh);

            FlatArray<int> extCols(1, lh);
            extCols[0] = vmap[fvnr];

            ImproveSProlRow<ENERGY>(
              fvnr,
              ris,
              extCols,
              omega,
              CM,
              *fmat,
              *AP,
              [&](auto update)
              {
                for (auto j : Range(ris))
                {
                  rvs[j] += update[j];
                }
              },
              lh
            );
          }
        }
      }, true);
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
