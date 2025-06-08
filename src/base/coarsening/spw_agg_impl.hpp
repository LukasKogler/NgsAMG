#ifndef FILE_SPW_AGG_IMPL_HPP
#define FILE_SPW_AGG_IMPL_HPP

#include "agglomerator.hpp"
#include "alg_mesh_nodes.hpp"
#include "vertex_factory_impl.hpp"
#include <type_traits>
#ifdef SPW_AGG

#include <utils.hpp>
#include <utils_denseLA.hpp>
#include <utils_arrays_tables.hpp>
#include <utils_io.hpp>

#include "spw_agg.hpp"

#include "agglomerator_utils.hpp"
#include "agglomerator_impl.hpp"

namespace amg
{


/** LocCoarseMap **/

class LocCoarseMap : public BaseCoarseMap
{
  template<class ATENERGY, class ATMESH, bool AROBUST> friend class SPWAgglomerator;

public:

  LocCoarseMap (shared_ptr<TopologicMesh> mesh,
                shared_ptr<TopologicMesh> mapped_mesh = nullptr)
    : BaseCoarseMap(mesh, mapped_mesh)
  {
    const auto & M = *mesh;

    Iterate<4>([&](auto k)
    {
      NN[k] = M.template GetNN<NODE_TYPE(k.value)>();
      node_maps[k].SetSize(NN[k]);
    });
  }

  ~LocCoarseMap() = default;

  FlatArray<int> GetV2EQ () const { return cv_to_eqc; }

  size_t & GetNCV () { return mapped_NN[NT_VERTEX]; }
  INLINE int CV2EQ (int v) const { return cv_to_eqc[v]; }

  void
  FinishUp (LocalHeap &lh)
  {
    static Timer t("LocCoarseMap::FinishUp");
    RegionTimer rt(t);

    auto const &eqc_h = mesh->GetEQCHierarchy();

    /** mapped_mesh **/
    this->mapped_mesh = make_shared<TopologicMesh>(eqc_h, GetMappedNN<NT_VERTEX>(), 0, 0, 0);;
    auto &cmesh = this->mapped_mesh;

    // NOTE: verts are never set, they are unnecessary because we dont build coarse BlockTM!
    /** crs vertex eqcs **/
    auto c2fv = GetMapC2F<NT_VERTEX>();

    // cout << " LocCoarseMap FinishUp, c2fv: " << endl << c2fv << endl;

    if ( auto pbtm = dynamic_pointer_cast<BlockTM>(mesh) )
    {
      // we are the first map - construct cv_to_eqc. for other maps, concatenate it!
      auto vMap = GetMap<NT_VERTEX>();

      if ( eqc_h->GetNEQCS() > 1 )
      {
        cv_to_eqc.SetSize(GetMappedNN<NT_VERTEX>());
        cv_to_eqc = 0;

        // works as long as we only go up/down in hierarchy
        pbtm->template ApplyEQ<NT_VERTEX>(Range(1ul, eqc_h->GetNEQCS()), [&](auto eqc, auto fVNr)
        {
          auto cVNr = vMap[fVNr];

          if ( cVNr != -1 )
          {
            auto cVEQ = cv_to_eqc[cVNr];

            cv_to_eqc[cVNr] = ( cVEQ == 0 ) ? eqc
                                            : ( eqc_h->IsLEQ(eqc, cVEQ) ? cVEQ : eqc );
          }
        }, false);
      }
      else
      {
        cv_to_eqc.SetSize0();
      }
    }

    /** crs edge connectivity and edges **/
    const auto & fecon = *mesh->GetEdgeCM();

    auto vmap = GetMap<NT_VERTEX>();

    std::set<int> cNeibSet;

    // c-graph by merging
    auto itCoarseGraph = [&](auto lam)
    {
      // !!int!! - comparison otherwise promotes -1 to numeric_limit!
      for (int cVNr : Range(c2fv))
      {
        HeapReset hr(lh);

        auto fVerts = c2fv[cVNr];

        if ( fVerts.Size() == 1 )
        {
          auto const fV0 = fVerts[0];
          auto fNeibs = fecon.GetRowIndices(fV0);

          FlatArray<int> extCNeibs(fNeibs.Size(), lh);
          int c = 0;

          for (auto j : Range(fNeibs))
          {
            auto cJ = vmap[fNeibs[j]];

            if ( cJ > cVNr ) // no duplicates, no edges to DIRI/collpased
            {
              extCNeibs[c++] = cJ;
            }
          }

          auto cNeibs = extCNeibs.Range(0, c);
          QuickSort(cNeibs);

          lam(cVNr, cNeibs);
        }
        else
        {
          cNeibSet.clear();

          for (auto j : Range(fVerts))
          {
            auto fvJ    = fVerts[j];
            auto fNeibs = fecon.GetRowIndices(fvJ);

            for (auto j : Range(fNeibs))
            {
              int const cJ = vmap[fNeibs[j]];

              if ( cJ > cVNr ) // no duplicates, no edges to DIRI/collapsed
              {
                cNeibSet.insert(cJ);
              }
            }
          }

          FlatArray<int> cNeibs(cNeibSet.size(), lh);
          int c = 0;
          for (auto it : cNeibSet)
          {
            cNeibs[c++] = it;
          }

          lam(cVNr, cNeibs);
        }
      }
    };

    Array<int> cEconPR(GetMappedNN<NT_VERTEX>());
    cEconPR = 0;

    auto &numCEdges = this->mapped_NN[NT_EDGE];
    numCEdges = 0;

    itCoarseGraph([&](auto cv, auto cols)
    {
      // edges are only counted once!
      numCEdges += cols.Size();

      // cv -> col entries
      cEconPR[cv] += cols.Size();

      // col -> cv  entry for each edge
      for (auto col : cols)
      {
        cEconPR[col]++;
      }
    });

    auto &cEdges = cmesh->edges;
    cEdges.SetSize(GetMappedNN<NT_EDGE>());

    auto pcecon = make_shared<SparseMatrix<double>>(cEconPR, GetMappedNN<NT_VERTEX>());

    cEconPR = 0;
    numCEdges = 0;

    itCoarseGraph([&](int const cv, auto cols)
    {
      auto colsI = pcecon->GetRowIndices(cv);
      auto valsI = pcecon->GetRowValues(cv);

      for (auto col : cols)
      {
        int const cEId = numCEdges++;

        cEdges[cEId].id = cEId;
        cEdges[cEId].v  = { cv, col }; // col > cv per construction

        int const offi = cEconPR[cv]++;
        int const offj = cEconPR[col]++;

        colsI[offi] = col;
        valsI[offi] = cEId;

        pcecon->GetRowIndices(col)[offj] = cv;
        pcecon->GetRowValues(col)[offj]  = cEId;
      }
    });

    auto const &cecon = *pcecon;

    cmesh->econ = pcecon;
    cmesh->nnodes[NT_EDGE] = GetMappedNN<NT_EDGE>();

    /** edge map **/
    auto fedges = mesh->template GetNodes<NT_EDGE>();
    auto & emap = node_maps[NT_EDGE]; emap.SetSize(GetNN<NT_EDGE>());

    // cout << " pcecon: " << endl << *pcecon << endl << endl;

    for (auto fenr : Range(emap))
    {
      auto & edge = fedges[fenr];

      int cv0 = vmap[edge.v[0]], cv1 = vmap[edge.v[1]];

      if ( (cv0 != -1) && (cv1 != -1) && (cv0 != cv1) )
        { emap[fenr] = cecon(cv0, cv1); }
      else
        { emap[fenr] = -1; }
    }

    if (f_allowed_edges)
    {
      c_allowed_edges = make_shared<BitArray>(cmesh->template GetNN<NT_EDGE>());
      c_allowed_edges->Clear();

      for (auto k : Range(fedges))
      {
        if ( (emap[k] != -1) && (f_allowed_edges->Test(k)) )
          { c_allowed_edges->SetBit(emap[k]); }
      }
    }
  } // FinishUp

  shared_ptr<LocCoarseMap>
  ConcatenateProper (shared_ptr<LocCoarseMap> rightMap)
  {
    static Timer t("LocCoarseMap::ConcatenateProper");
    RegionTimer rt(t);

    auto concMap = make_shared<LocCoarseMap>(this->mesh, rightMap->mapped_mesh);

    /** sets vertex/edge map for concmap **/
    SetConcedMap(rightMap, concMap);

    /** concatenated aggs! **/
    const size_t NCV = rightMap->GetMappedNN<NT_VERTEX>();
    FlatTable<int> aggs1 = GetMapC2F<NT_VERTEX>();
    FlatTable<int> aggs2 = rightMap->GetMapC2F<NT_VERTEX>();

    TableCreator<int> ct(NCV);

    for (; !ct.Done(); ct++)
    {
      for (auto k : Range(NCV))
      {
        for (auto v : aggs2[k])
          { ct.Add(k, aggs1[v]); }
      }
    }
    concMap->rev_node_maps[NT_VERTEX] = ct.MoveTable();

    /** coarse vertex->eqc mapping - right map does not have it yet (no BTM to construct it from) **/
    auto eqc_h = mesh->GetEQCHierarchy();

    auto & cv2eq = concMap->cv_to_eqc;

    if (eqc_h->IsTrulyParallel())
    {
      cv2eq.SetSize(NCV);
      for (auto k : Range(NCV))
      {
        auto agg = aggs2[k];

        if ( agg.Size() == 1 )
          { cv2eq[k] = cv_to_eqc[agg[0]]; }
        else
        {
          int cEQ = cv_to_eqc[agg[0]];

          for (auto l : Range(1ul, agg.Size()))
          {
            auto lEQ = cv_to_eqc[agg[l]];

            cEQ = eqc_h->IsLEQ(cEQ, lEQ) ? lEQ : cEQ;
          }

          cv2eq[k] = cEQ;
        }
      }
    }
    else
    {
      cv2eq.SetSize0();
    }

    concMap->f_allowed_edges = f_allowed_edges;
    concMap->c_allowed_edges = rightMap->c_allowed_edges;

    return concMap;
  } // ConcatenateLCM

  void
  ConcatenateVMap (size_t NCV, FlatArray<int> rvmap)
  {
    static Timer t("LocCoarseMap::ConcatenateVMap");
    RegionTimer rt(t);

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
    cout << " ConcatenateVMap rev-map " << endl;


    auto & aggs = rev_node_maps[NT_VERTEX];

    // cout << "  LEFT V-MAP: " << endl << vmap << endl;
    // cout << "  RIGHT V-MAP: " << endl << rvmap << endl;

    // cout << "  Laggs: " << endl << aggs << endl;

    TableCreator<int> ct(NCV);
    for (; !ct.Done(); ct++)
    {
      for (auto k : Range(aggs))
      {
        if (rvmap[k] != -1)
          { ct.Add(rvmap[k], aggs[k]); }
      }
    }
    rev_node_maps[NT_VERTEX] = ct.MoveTable();

    /** no vertex->eqc mapping on coarse level **/
    cv_to_eqc.SetSize0();
  } // Concatenate(array)

protected:

  /**
   * vert->eqc mapping for the coarse mesh,
   * this is saved here, in the map, instead of the coarse mesh itself
   * because the meshes jave that info in form of offsets, not direct map
   */
  Array<int> cv_to_eqc;

  /** generates coarse level allowed edes IF FINE LEVEL is set before FinishUp **/
  shared_ptr<BitArray> f_solid_verts, c_solid_verts;
  /** non-solid verts are set to "handled" below anyways! **/
  shared_ptr<BitArray> f_allowed_edges = nullptr, c_allowed_edges = nullptr;


public:
  void SetAllowedFEdges (shared_ptr<BitArray> _f_allowed_edges) { f_allowed_edges = _f_allowed_edges; }
  shared_ptr<BitArray> GetAllowedFEdges () const { return f_allowed_edges; }
  shared_ptr<BitArray> GetAllowedCEdges () const { return c_allowed_edges; }

  void SetSolidFVerts (shared_ptr<BitArray> _f_solid_verts) { f_solid_verts = _f_solid_verts; }
  shared_ptr<BitArray> GetSolidFVerts () const { return f_solid_verts; }
  shared_ptr<BitArray> GetSolidCVerts () const { return c_solid_verts; }
}; // class LocCoarseMap

/** END LocCoarseMap **/



/** AggData **/

template<class ATMESH, class TENERGY, class ATMU>
class SPWAggData : public AgglomerationData<ATMESH, TENERGY, ATMU, TWEIGHT>
{
public: // all public, for my sanity
  using TMESH  = ATMESH;
  using ENERGY = TENERGY;

  using BASE         = AgglomerationData<TMESH, ENERGY, ATMU, TWEIGHT>;
  using MAPPED_CLASS = SPWAggData<TopologicMesh, ENERGY, ATMU> ;

  using TVD = typename BASE::TVD;
  using TM  = typename BASE::TM;
  using TMU = typename BASE::TMU;

  static constexpr bool ROBUST = Height<TMU>() > 1;

  SPWAggData (TMESH const &aMesh)
    : AgglomerationData<TMESH, TENERGY, ATMU, TWEIGHT>(aMesh)
  {}

  ~SPWAggData () {};

  std::unique_ptr<MAPPED_CLASS>
  Map (LocCoarseMap const &locMap,
       TWEIGHT      const &diagStabBoost)
  {
    static Timer t("SPWAggData::Map");
    RegionTimer rt(t);

    auto pCAggData = std::make_unique<MAPPED_CLASS>(*locMap.GetMappedMesh());
    auto &cAggData = *pCAggData;

    auto const NCV  = locMap.GetMappedNN<NT_VERTEX>();
    auto       vMap = locMap.GetMap<NT_VERTEX>();

    /** Coarse vertex data, diags **/
    auto &cVData    = cAggData.vData;
    auto &cAuxDiags = cAggData.auxDiags;

    cVData.SetSize(NCV);
    cAuxDiags.SetSize(NCV);

    auto c2fv = locMap.template GetMapC2F<NT_VERTEX>();

    for (auto cVNr : Range(NCV))
    {
      auto fvs = c2fv[cVNr];

      if (fvs.Size() == 1)
      {
        cVData[cVNr]    = this->vData[fvs[0]];
        cAuxDiags[cVNr] = this->auxDiags[fvs[0]];
      }
      else
      {
        auto &cData = cVData[cVNr];

        // does not matter, mostly sets position, just use mid-point between
        // first 2 members
        cData = ENERGY::CalcMPData(this->vData[fvs[0]], this->vData[fvs[1]]);

        // weight
        TMU cWeight = TMU(0);
        TMU cDiag   = TMU(0);

        for (auto l : Range(fvs))
        {
          auto const fData = this->vData[fvs[l]];

          if constexpr(ROBUST)
          {
            TM const &wt = ENERGY::GetVMatrix(fData);

            auto const Q = ENERGY::GetQiToj(cData, fData);

            cWeight += Q.GetQTMQ(1.0, wt);
            cDiag   += Q.GetQTMQ(1.0, this->auxDiags[fvs[l]]);
          }
          else
          {
            cWeight += ENERGY::GetApproxVWeight(fData);
            cDiag   += this->auxDiags[fvs[l]];
          }
        }

        ENERGY::SetVMatrix(cData, cWeight);

        cAuxDiags[cVNr] = cDiag;
      }
    }

    /** Coarse edge data **/
    auto const NCE  = locMap.GetMappedNN<NT_EDGE>();
    auto       eMap = locMap.GetMap<NT_EDGE>();

    auto &cEMats     = cAggData.edgeMats;
    auto &cEdgeTrace = cAggData.edgeTrace;

    cEMats.SetSize(NCE);
    cEMats = 0;

    if constexpr(ROBUST)
    {
      cEdgeTrace.SetSize(NCE);
      cEdgeTrace = 0;
    }
    else
    {
      cEdgeTrace.FlatArray<TMU>::Assign(cEMats);
    }

    auto fEdges = this->GetMesh().template GetNodes<NT_EDGE>();
    auto cEdges = locMap.GetMappedMesh()->template GetNodes<NT_EDGE>();

    // max stability -> do not remove, min stability -> remove full!
    TWEIGHT const inAggEdgeFactor = -2.0 * ( 1.0 - diagStabBoost );

    for ( auto fENr : Range(fEdges))
    {
      auto const &fEdge = fEdges[fENr];
      auto const  cENr  = eMap[fENr];

      /**
       * Note: if only one vertex drops, no need to do anything,
       *       that contrib. is included in diags and will be
       *       summed into coarse diags
       */

      if ( cENr != -1 )
      {
        if constexpr(ROBUST)
        {
          // order of cverts does not matter
          auto const &cEdge = cEdges[cENr];

          TVD fMid = ENERGY::CalcMPData(this->vData[fEdge.v[0]],  this->vData[fEdge.v[1]]);
          TVD cMid = ENERGY::CalcMPData(cVData[cEdge.v[0]], cVData[cEdge.v[1]]);

          cEMats[cENr] += ENERGY::GetQiToj(cMid, fMid).GetQTMQ(1.0, this->edgeMats[fEdge.id]);
        }

        cEdgeTrace[cENr] += this->edgeTrace[fENr];
      }
      else if ( inAggEdgeFactor != 0.0 )
      {
        // remove (most of) contribs of in-agg edges from coarse diags
        auto const cv0 = vMap[fEdge.v[0]];
        auto const cv1 = vMap[fEdge.v[1]];

        if ( ( cv0 != -1 ) && ( cv0 == cv1 ) )
        {
          if constexpr(ROBUST)
          {
            TVD fMid = ENERGY::CalcMPData(this->vData[fEdge.v[0]], this->vData[fEdge.v[1]]);

            auto const Q = ENERGY::GetQiToj(cVData[cv0], fMid);

            cAuxDiags[cv0] += Q.GetQTMQ(inAggEdgeFactor, this->edgeMats[fEdge.id]);
          }
          else
          {
            cAuxDiags[cv0] += this->edgeMats[fEdge.id];
          }
        }
      }
    }

    /** Coarse off-proc-trace **/
    cAggData.hasOffProcTrace = this->hasOffProcTrace;

    auto &cOffProcTrace = cAggData.offProcTrace;

    if ( this->hasOffProcTrace )
    {
      cOffProcTrace.SetSize(NCV);

      for (auto cVNr : Range(NCV))
      {
        auto fVNrs = c2fv[cVNr];

        TWEIGHT opc = 0;

        for (auto fVNr : fVNrs)
        {
          opc += this->offProcTrace[fVNr];
        }

        cOffProcTrace[cVNr] = opc;
      }
    }
    else
    {
      cOffProcTrace.SetSize0();
    }

    /** Coarse max. off-diag trace **/
    auto &cMaxTrOD  = cAggData.maxTrOD;

    cMaxTrOD.SetSize(NCV);

    auto const &cEcon = *locMap.GetMappedMesh()->GetEdgeCM();

    auto aggs = locMap.template GetMapC2F<NT_VERTEX>();

    // calc max-trod based on coarse mesh
    for ( auto cVNr : Range(NCV))
    {
      auto cNeibs = cEcon.GetRowIndices(cVNr);
      auto cENrs  = cEcon.GetRowValues(cVNr);

      // off-proc trace
      TWEIGHT mTrOD = this->hasOffProcTrace ? cOffProcTrace[cVNr] : 0.0;

      // max-trod computed from coarse mesh
      for (auto j : Range(cNeibs))
      {
        mTrOD = std::max(mTrOD, cEdgeTrace[int(cENrs[j])]);
      }

      // take the max with all fine trods for stability
      auto agg = aggs[cVNr];
      for (auto l : Range(agg))
      {
        mTrOD = std::max(mTrOD, this->maxTrOD[agg[l]]);
      }

      cMaxTrOD[cVNr] = mTrOD;
    }
    return pCAggData;
  }
}; // class SPWAggData

/** END AggData **/

/** Coarsening Logic **/

template<class ALLOWED_NEIB,
         class TCALC_INI_SOC,
         class TCALC_STABLE_SOC,
         class TCHECK_SOC_PW,
         class TCHECK_SOC_AGG>
INLINE int
FindNeib3Step (int const &vi,
               FlatArray<int> neibs,
               bool    const &robustPick,
               bool    const &doAggWideCheck, // absolute tol
               TWEIGHT const &scalRelThresh,  // relative tol
               TWEIGHT const &robAbsThresh,   // absolute tol
               TWEIGHT const &bigAbsThresh,
               TCALC_INI_SOC     calcInitSOC,
               TCALC_STABLE_SOC  calcRobSOC,
               TCHECK_SOC_PW     checkRobSOC,
               TCHECK_SOC_AGG    checkBigSOC,
               ALLOWED_NEIB      allowed,        //
               LocalHeap        &lh)
{
  /**
   * Find a "good" match for vertex vi
   *     (a). compute (assumed cheap) initial SOC, filter with REL. thresh
   *     (b).  i) optionally, compute stable SOC, re-filter & re-order
   *          ii) re-check with stable SOC/big-SOC
   *
   * Since we are using a RELATIVE thresh in step I, we should go over ALL
   * neibs here in order to get the correct maximum to compare to.
   *
   * Steps 2 and 3 are using an absolute threshold.
   */
  auto const numNeibs = neibs.Size();

  // cout << " FindNeib3Step " << vi << ", #neibs = " << numNeibs << endl;

  /** I. Scalar SOC - filter out weak neibs **/
  FlatArray<TWEIGHT> weights(numNeibs, lh);

  TWEIGHT maxScalWt = 0;

  for (auto j : Range(neibs))
  {
    // calc scalar-SOC for ALL (incl. forbidden) to get
    // correct maximum (affects thresholds!)
    TWEIGHT const soc       = calcInitSOC(j);
    auto    const isAllowed = allowed(vi, neibs[j]);

    weights[j] = isAllowed ? soc : -1;

    maxScalWt = max(maxScalWt, soc);
  }

  // cout << "    (scalar) weights: "; prow2(weights); cout << endl;

  TWEIGHT const scalWtThresh = scalRelThresh * maxScalWt;

  if (robustPick)
  {
    // TODO: try computing ROBUST SOC for non-allowed/taken to get true max-SOC (first implement BIG-SOC)
    for (auto j : Range(neibs))
    {
      auto const scalSOC = weights[j];

      if ( scalSOC >= scalWtThresh )
      {
        weights[j] = calcRobSOC(j);
      }
      else
      {
        weights[j] = -1;
      }
    }

    // cout << "    (robust) weights: "; prow2(weights); cout << endl;
  }

  FlatArray<int> neibIdx = makeSequence(numNeibs, lh);

  QuickSortI(weights, neibIdx, [&](auto wi, auto wj) { return wj < wi; });

  // cout << "    sorted idx "; prow(neibIdx); cout << endl;

  TWEIGHT const robWtThresh = min(scalWtThresh, robAbsThresh);
  TWEIGHT const bigWtThresh = min(robWtThresh, bigAbsThresh);

  TWEIGHT const weightThresh = robustPick ? robWtThresh : scalWtThresh;

  // cout << " scalWtThresh = " << scalWtThresh << endl;
  // cout << " robWtThresh = " << robWtThresh << endl;
  // cout << " bigWtThresh = " << bigWtThresh << endl;
  // cout << " weightThresh = " << weightThresh << endl;

  /** Iterate through remaining neibs, strongest first, take the first good one */
  for (auto l : Range(neibIdx))
  {
    auto const idx    = neibIdx[l];
    auto const neib   = neibs[idx];
    auto const scalWt = weights[idx];

    // cout << "      neib-idx " << l << "=" << idx << ", neib =" << neib << " -> scalWt  " << scalWt << endl;

    // further ones are already weak with scalar weight
    if ( scalWt < weightThresh )
    {
      // cout << scalWt << " < " << weightThresh << " -> ABORT, NO NEIB! " << endl;
      break;
    }

    bool isViable = true;

    if (!robustPick)
    {
      // check whether rob-soc (vi, neib) > robWtThresh
      isViable = checkRobSOC(idx, robWtThresh);
    }

    // cout << "      neib-idx " << l << "=" << idx << ", neib =" << neib << " -> viable I  " << isViable << endl;

    if ( isViable && doAggWideCheck )
    {
      // check whether agg-wide SOC (agg(vi), agg(neib)) > robWtThresh
      isViable = checkBigSOC(idx, bigWtThresh);

      // if (!isViable)
      // {
        // cout << "      neib-idx " << l << "=" << idx << ", neib =" << neib << " -> viable II " << isViable << endl;
      // }
    }


    if ( isViable )
    {
      // cout << "   -> FOUND NEIB " << neib << endl << endl;
      return neib;
    }
  }

  return -1;
} // FindNeib3Step


template<class ENERGY, bool FIRST_ROUND, class AGG_DATA, class AGG_DATA_F, class TALLOW, class TGET_AGG>
INLINE int
FindNeighborToMatch (int        const &vi,
                     AGG_DATA   const &aggData,
                     AGG_DATA_F const &fAggData,
                     SPWConfig  const &cfg,
                     TALLOW            allowedNeighbor,
                     TGET_AGG          getFullAgg,
                     LocalHeap        &lh)
{
  auto neibs    = aggData.GetEdgeCM().GetRowIndices(vi);
  auto edgeNums = aggData.GetEdgeCM().GetRowValues(vi);

  auto calcScalSOC = [&](auto idxN) -> TWEIGHT
  {
    // cout << " Approx SOC for neib " << idxN << "=" << neibs[idxN] << endl;
    // cout << "    max-trod = " << aggData.GetMaxTrOD(vi) << " " << aggData.GetMaxTrOD(neibs[idxN]) << endl;
    // cout << "    approx-e-wt = " << aggData.GetApproxEdgeWeight(int(edgeNums[idxN])) << endl;

    return CalcApproxSOC<ENERGY, TWEIGHT>(
            cfg.avgTypeScal,
            aggData.GetVData(vi),
            aggData.GetVData(neibs[idxN]),
            aggData.GetApproxEdgeWeight(int(edgeNums[idxN])),
            aggData.GetMaxTrOD(vi),
            aggData.GetMaxTrOD(neibs[idxN]),
            false); // l2Boost
  };

  auto calcRobSOC = [&](auto idxN) -> TWEIGHT
  {
    if constexpr(AGG_DATA::ROBUST)
    {
      return CalcRobSOC<ENERGY>(vi, neibs[idxN], int(edgeNums[idxN]), aggData, cfg, lh);
    }
    else
    {
      return TWEIGHT(1);
    }
  };

  auto checkRobSOC = [&](auto idxN, auto const &rho)
  {
    if constexpr( AGG_DATA::ROBUST )
    {
      return CheckRobSOC<ENERGY>(rho, vi, neibs[idxN], int(edgeNums[idxN]), aggData, cfg, lh);
    }
    else
    {
      return std::integral_constant<bool, true>();
    }
  };

  auto checkBigSOC = [&](auto idxN, auto const &rho)
  {
    if constexpr( !FIRST_ROUND )
    {
      auto const neib = neibs[idxN];

      FlatArray<int> aggI = getFullAgg(vi);
      FlatArray<int> aggJ = getFullAgg(neib);

      return AggregateWideStabilityCheck<ENERGY>(rho,
                                                fAggData,
                                                aggI,
                                                aggJ,
                                                false,
                                                false,
                                                lh);
    }
    else
    {
      return std::integral_constant<bool, true>();
    }
  };

  return FindNeib3Step (vi,
                        neibs,
                        cfg.robustPick,
                        cfg.checkBigSOC,
                        cfg.scalRelThresh,
                        cfg.absRobThresh,
                        cfg.absBigThresh,
                        calcScalSOC,
                        calcRobSOC,
                        checkRobSOC,
                        checkBigSOC,
                        allowedNeighbor,
                        lh);
} // FindNeighborToMatch


template<class ENERGY, class AGG_DATA, class TALLOW>
INLINE int
FindNeighborToJoin (int       const &vi,
                    AGG_DATA  const &aggData,
                    SPWConfig const &cfg,
                    TALLOW           allowedNeighbor,
                    LocalHeap       &lh)
{
  auto neibs    = aggData.GetEdgeCM().GetRowIndices(vi);
  auto edgeNums = aggData.GetEdgeCM().GetRowValues(vi);

  auto calcScalSOC = [&](auto idxN) -> TWEIGHT
  {
    return CalcApproxJoinSOC<ENERGY>(aggData.GetVData(vi),
                                     aggData.GetVData(neibs[idxN]),
                                     aggData.GetApproxEdgeWeight(int(edgeNums[idxN])),
                                     aggData.GetMaxTrOD(vi),
                                     aggData.GetMaxTrOD(neibs[idxN]),
                                     false); // l2Boost
  };

  auto calcRobSOC = [&](auto idxN) -> TWEIGHT
  {
    if constexpr(AGG_DATA::ROBUST)
    {
      return CalcRobJoinSOC<ENERGY>(vi, neibs[idxN], int(edgeNums[idxN]), aggData, cfg, lh);
    }
    else
    {
      return TWEIGHT(1);
    }
  };

  auto checkRobSOC = [&](auto idxN, auto const &rho)
  {
    if constexpr(AGG_DATA::ROBUST)
    {
      return CheckRobJoinSOC<ENERGY>(rho, vi, neibs[idxN], int(edgeNums[idxN]), aggData, cfg, lh);
    }
    else
    {
      return std::integral_constant<bool, true>();
    }
  };

  auto checkBigSOC = [&](auto idxN, auto const &rho)
  {
    if constexpr(AGG_DATA::ROBUST)
    {
      return true;
    }
    else
    {
      return std::integral_constant<bool, true>();
    }
  };

  return FindNeib3Step (vi,
                        neibs,
                        cfg.robustPick,
                        false, // checkBigSOC,
                        cfg.scalRelThresh,
                        cfg.absRobThresh,
                        cfg.absBigThresh,
                        calcScalSOC,
                        calcRobSOC,
                        checkRobSOC,
                        checkBigSOC,
                        allowedNeighbor,
                        lh);
} // FindNeighborToJoin


template<bool PREFER_LDEG, class TGET_V, class TLAM>
INLINE void
IterateVertsRev (int         const &numVs,
                 TGET_V             getV,
                 MatrixGraph const &connectivity,
                 BitArray          &handled,
                 LocalHeap         &lh,
                 TLAM               lam)
{
  /** NOTE: if PREFER_LDEG, this can also call lam on verts outsite getV! **/
  int c = numVs - 1 ;

  for (auto l : Range(numVs))
  {
    auto const v = getV(c--);

    if (!handled.Test(v))
    {
      if constexpr( PREFER_LDEG )
      {
        HeapReset hr(lh);
        // handle neighbors with lower degree first, sorted by their degree
        auto vNeibs = connectivity.GetRowIndices(v);

        int const deg = vNeibs.Size();

        FlatArray<int> allLDIdx(deg, lh);
        int cntLDG = 0;

        for (auto j : Range(vNeibs))
        {
          auto const neib = vNeibs[j];

          if (!handled.Test(neib))
          {
            int const jDeg = connectivity.GetRowIndices(neib).Size();

            if ( jDeg < deg )
            {
              allLDIdx[cntLDG++] = j;
            }
          }
        }

        auto lDIdx = allLDIdx.Range(0, cntLDG);

        QuickSort(lDIdx, [&](auto i, auto j) {
          auto const degi = connectivity.GetRowIndices(vNeibs[i]).Size();
          auto const degj = connectivity.GetRowIndices(vNeibs[j]).Size();
          return degi < degj;
        });

        for (auto l : lDIdx)
        {
          auto neibV = vNeibs[l];

          if (!handled.Test(neibV)) //  re-check handled, can change inside "lam"!
          {
            lam(std::false_type(), neibV);
          }
        }
      }

      // now handle vertex itself
      if (!handled.Test(v)) //  re-check handled, can change inside "lam"!
      {
        lam(std::true_type(), v);
      }
    }
  }
} // IterateVertsRev


template<bool PREFER_LDEG, class TLAM>
INLINE void
IterateVertsReverseParOrder (BlockTM     const &FM,
                             BitArray          &handled,
                             LocalHeap         &lh,
                             TLAM               lam)
{
  auto const eqc_h = *FM.GetEQCHierarchy();

  auto const neqcs = eqc_h.GetNEQCS();

  auto const econ = *FM.GetEdgeCM();

  for (int eqc = neqcs - 1; eqc >= 0; eqc--)
  {
    auto eqVs = FM.template GetENodes<NT_VERTEX>(eqc);

    // if PREFER_LDEG, will also handle verts from other EQCs!
    IterateVertsRev<PREFER_LDEG>(eqVs.Size(),
                                 [&](auto k) { return eqVs[k]; },
                                 econ,
                                 handled,
                                 lh,
                                 [&](auto isEQcVertex, auto v)
                                 {
                                    if constexpr(isEQcVertex) // true_type -> in the same EQC for sure
                                    {
                                      lam(eqc, v);
                                    }
                                    else // potentially in different EQC
                                    {
                                      lam(isEQcVertex, v); // called with integral_constant<bool, false>
                                    }
                                  });
  }
} // IterateVertsReverseParOrder


template<bool PREFER_LDEG, class TLAM>
INLINE void
IterateVertsSimpleReverse (TopologicMesh const &FM,
                           BitArray            &handled,
                           LocalHeap           &lh,
                           TLAM                 lam)
{
  auto const econ = *FM.GetEdgeCM();

  IterateVertsRev<PREFER_LDEG>(FM.template GetNN<NT_VERTEX>(),
                               [&](auto k) { return k; },
                               econ,
                               handled,
                               lh,
                               [&](auto isEQcVertex, auto v) { lam(std::integral_constant<bool, false>(), v); });
} // void


template<class ENERGY, bool FIRST_ROUND, class TMESH, class AGG_DATA, class AGG_DATA_F>
INLINE int
PairingIteration (TMESH           const &FM,
                  AGG_DATA        const &aggData,
                  AGG_DATA_F      const &fAggData,
                  FlatTable<int>         currAggs,
                  SPWConfig       const &cfg,
                  FlatArray<int>         vmap,
                  FlatArray<int>         v2eq,
                  BitArray              &handled,
                  shared_ptr<BitArray>   solidVerts,
                  LocalHeap             &lh,
                  int tag = 0,
                  int                   cvOff = 0)
{
  static Timer t("CalcRobustPairSOC");

  auto t0 = t.GetTime();
  t.Start();

  auto const eqc_h = *FM.GetEQCHierarchy();

  int cntVerts = 0;

  auto getFullAgg = [&](auto vi) -> FlatArray<int> { return currAggs[vi]; };

  auto itVerts = [&](auto getNeib)
  {
    auto createPair = [&](auto eqcIfKnown, auto v)
    {
      HeapReset hr(lh);

      auto neib = getNeib(eqcIfKnown, v);

      auto const cV = cvOff + (cntVerts++);

      if ( neib != -1 )
      {
        // cout << " PairingIteration " << tag << ", PAIR " << v << ", " << neib << " -> " << cV << endl;
        vmap[neib] = cV;
        handled.SetBit(neib);
      }
      // else
      // {
        // cout << " PairingIteration " << tag << ", SINGLE " << v << ", " << neib << " -> " << cV << endl;
      // }

      vmap[v] = cV;
      handled.SetBit(v);
    };

    if constexpr(std::is_base_of_v<BlockTM, TMESH>)
    {
      // prefer lower-degree in first step, should help to reduce orphans
      IterateVertsReverseParOrder<true>(FM, handled, lh, createPair);
    }
    else
    {
      // just do inverse order later (mostly hits unmatched from prev level first)
      IterateVertsSimpleReverse<false>(FM, handled, lh, createPair);
    }
  };


  if ( eqc_h.IsTrulyParallel() )
  {
    auto getVEQC = [&](auto v)
    {
      int eqc;
      if constexpr(std::is_base_of_v<BlockTM, TMESH>)
      {
        // base-level, get eq FROM BlockTM
        eqc = FM.template GetEQCOfNode<NT_VERTEX>(v);
      }
      else
      {
        // later levels, have explicit v->eq mapping
        eqc = v2eq[v];
      }
      return eqc;
    };

    auto getActualEQC = [&](auto eqcIfKnown, int const &v) -> int
    {
      if constexpr(is_same_v<std::remove_reference_t<decltype(eqcIfKnown)>, int>)
      {
        return eqcIfKnown;
      }
      else
      {
        static_assert(is_same_v<decltype(eqcIfKnown), std::integral_constant<bool, false>>, "getActualEQC");

        return getVEQC(v);
      }
    };

    if ( solidVerts != nullptr )
    {
      auto const &sVerts = *solidVerts;

      itVerts([&](auto eqcIfKnown, int const &v)
      {
        return FindNeighborToMatch<ENERGY, FIRST_ROUND>(
                v,
                aggData,
                fAggData,
                cfg,
                [&](auto v, auto neib)
                {
                  if ( ( !handled.Test(neib) ) && sVerts.Test(neib) )
                  {
                    int const eqi = getActualEQC(eqcIfKnown, v);
                    int const eqj = getVEQC(neib);

                    return ( eqi == 0 )          ||
                           ( eqj == 0 )          ||
                           eqc_h.IsLEQ(eqi, eqj) ||
                           eqc_h.IsLEQ(eqj, eqi);
                  }
                  else
                  {
                    return false;
                  }
                },
                getFullAgg,
                lh
              );
      });
    }
    else
    {
      itVerts([&](auto eqcIfKnown, int const &v)
      {
        return FindNeighborToMatch<ENERGY, FIRST_ROUND>(
                  v,
                  aggData,
                  fAggData,
                  cfg,
                  [&](auto v, auto neib)
                  {
                    if ( handled.Test(neib) )
                      { return false; }

                    int const eqc    = getActualEQC(eqcIfKnown, v);
                    int const eqNeib = getVEQC(neib);

                    if ( eqNeib == 0 )
                      { return true; }
                    else if ( eqc == 0 )
                      { return eqc_h.IsMasterOfEQC(eqNeib); }
                    else
                    {
                      // bool allowed = eqc_h.IsMasterOfEQC(eqNeib) &&
                      //        ( eqc_h.IsLEQ(eqc, eqNeib) || eqc_h.IsLEQ(eqNeib, eqc) );
                      // cout << " allow " << v << " x " << neib << ", eqs " << eqc << " " << eqNeib << "? " << allowed << endl;
                      // cout << "    eqc_h.IsMasterOfEQC(eqNeib) = " << eqc_h.IsMasterOfEQC(eqNeib) << endl;
                      // cout << "    eqc_h.IsLEQ(eqc, eqNeib) = " << eqc_h.IsLEQ(eqc, eqNeib) << endl;
                      // cout << "    eqc_h.IsLEQ(eqNeib, eqc) = " << eqc_h.IsLEQ(eqNeib, eqc) << endl;
                      return eqc_h.IsMasterOfEQC(eqNeib) &&
                             ( eqc_h.IsLEQ(eqc, eqNeib) || eqc_h.IsLEQ(eqNeib, eqc) );
                    }
                  },
                  getFullAgg,
                  lh);
      });
    }
  }
  else
  {
    // actually serial, no EQ-checks needed at all
    itVerts([&](auto eqcIfKnown, auto v)
    {
      // eqc is irrelevant
      return FindNeighborToMatch<ENERGY, FIRST_ROUND>(
                v,
                aggData,
                fAggData,
                cfg,
                [&](auto v, auto neib) { return !handled.Test(neib); },
                getFullAgg,
                lh);
    });
  }

  t.Stop();

  auto t1 = t.GetTime();

  std::cout << " PairingIteration, time = " << t1 - t0 << std::endl;

  return cntVerts;
} // PairingIteration


template<class ENERGY, class AGG_DATA>
INLINE int
JoiningIteration (AGG_DATA       const &aggData,
                  SPWConfig      const &cfg,
                  FlatArray<int>        vmap,
                  FlatArray<int>        v2eq,
                  BitArray              &isAggregated,
                  LocalHeap             &lh)
{
  auto const &FM    = aggData.GetMesh();
  auto const &eqc_h = *FM.GetEQCHierarchy();

  BitArray isJoinable(isAggregated);

  // int numJoined = 0;
  int NCV = 0;

  // map aggregated to 0..#real aggs
  for (auto k : Range(FM.template GetNN<NT_VERTEX>()))
  {
    if (isAggregated.Test(k))
    {
      vmap[k] = NCV++;
    }
  }

  auto itVerts = [&](auto getAgg)
  {
    IterateVertsSimpleReverse<false>(
      FM,
      isAggregated,
      lh,
      [&](auto eqcIfKnown, auto v)
      {
        HeapReset hr(lh);

        // cout << " joiningIteration, find AGG for cv " << v << endl;

        auto agg = getAgg(eqcIfKnown, v);

        // cout << " joiningIteration, AGG for cv " << v << " = " << agg << endl;

        if ( agg != -1 )
        {
          // found an agg to join
          vmap[v] = vmap[agg];
        }
        else
        {
          // stays single
          vmap[v] = NCV++;
        }

        isAggregated.SetBit(v);
      }
    );
  };

  if ( eqc_h.IsTrulyParallel() )
  {
    itVerts([&](auto eqcIfKnown, auto v)
    {
      auto const eqV = v2eq[v];

      return FindNeighborToJoin<ENERGY>(
              v,
              aggData,
              cfg,
              [&](auto v, auto neib)
              {
                if ( !isJoinable.Test(neib) )
                {
                  return false;
                }

                auto const eqNeib = v2eq[neib];

                return eqc_h.IsMasterOfEQC(eqNeib) &&
                       ( eqc_h.IsLEQ(eqV, eqNeib) || eqc_h.IsLEQ(eqNeib, eqV) );
              },
              lh);
    });
  }
  else
  {
    itVerts([&](auto dummyArgument, auto v)
    {
      return FindNeighborToJoin<ENERGY>(v,
                                        aggData,
                                        cfg,
                                        [&](auto v, auto neib) { return isJoinable.Test(neib) && std::integral_constant<bool, true>(); },
                                        lh);
    });
  }

  return NCV;
} // JoiningIteration

/** END Coarsening Logic **/


/** SPWAgglomerator **/

template<class ENERGY, class TMESH, bool COMPILE_EV_BASED>
SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::
SPWAgglomerator (shared_ptr<TMESH> mesh)
  :	VertexAgglomerator<ENERGY, TMESH>(mesh)
{
  assert(mesh != nullptr); // obviously this would be bad
} // SPWAgglomerator(..)


template<class ENERGY, class TMESH, bool COMPILE_EV_BASED>
void
SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::
Initialize (const SPWAggOptions & opts, int level)
{
  // general control
  cfg.numRounds   = max(1, opts.numRounds.GetOpt(level));
  cfg.dropDDVerts = ( this->vert_thresh > 0 );
  cfg.orphanRound = opts.orphanRound.GetOpt(level);
  cfg.diagStabBoost = min(1.0, max(0.0,
                                   opts.diagStabBoost.GetOpt(level)));

  // initial neib-filtering
  cfg.avgTypeScal = opts.scalAvg.GetOpt(level);

  // neib-picking
  cfg.robustPick = opts.robustPick.GetOpt(level);
  cfg.neibBoost  = opts.neibBoost.GetOpt(level);

  // big-SOC settings
  cfg.checkBigSOC      = opts.checkBigSOC.GetOpt(level);
  cfg.robBigSOC        = opts.bigSOCRobust.GetOpt(level);
  cfg.bigSOCUseBDG     = opts.bigSOCBlockDiagSM.GetOpt(level);
  cfg.bigSOCHackThresh = opts.bigSOCCheckHackThresh.GetOpt(level);

  // thresholds
  cfg.vertThresh    = this->vert_thresh;
  cfg.scalRelThresh = 0.25;
  cfg.absRobThresh  = this->edge_thresh;
  cfg.absBigThresh  = this->edge_thresh;

  // others
	cfg.printParams = this->print_aggs || opts.printParams.GetOpt(level);
	cfg.printSumms  = this->print_aggs || opts.printSummary.GetOpt(level);
} // SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::SetLevelSPWOptions





template<class ENERGY, class TMESH, bool COMPILE_EV_BASED>
void
SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::
FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  if constexpr ( COMPILE_EV_BASED && !std::is_same_v<TM, TWEIGHT> )
  {
    if ( this->robust_crs )
      { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
    else /** (much) more expensive, but also more robust **/
      { FormAgglomerates_impl<TWEIGHT> (agglomerates, v_to_agg); }
  }
  else // when not necessary, do not even compile the robust version
  {
    FormAgglomerates_impl<TWEIGHT> (agglomerates, v_to_agg);
  }
} // SPWAgglomerator::FormAgglomerates


template<class ENERGY, class TMESH, bool COMPILE_EV_BASED>
template<class TMU>
INLINE void
SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::
FormAgglomerates_impl (Array<Agglomerate> &agglomerates,
                       Array<int>         &v_to_agg)
{
  static Timer t("SPWAgglomerator::FormAgglomerates_impl");
  RegionTimer rt(t);

  constexpr bool ROBUST = Height<TMU>() > 1;

  auto const &cfg = this->cfg;

  const bool printAggs   = this->print_aggs;    // actual aggs

  TMESH       &ncFM = this->GetMesh();
  TMESH const &FM   = ncFM;
  FM.CumulateData();

  auto const &eqc_h = *FM.GetEQCHierarchy();

  const auto FNV = FM.template GetNN<NT_VERTEX>();
  const auto FNE = FM.template GetNN<NT_EDGE>();

  constexpr int BS = ngbla::Height<TM>();
  constexpr int BSU = ngbla::Height<TMU>();

  auto baseAggData = std::make_unique<SPWAggData<TMESH, ENERGY, TMU>>(FM);

  auto baseVData = get<0>(ncFM.AttachedData())->Data();
  auto baseEData = get<1>(ncFM.AttachedData())->Data();

  auto &baseDiags    = baseAggData->auxDiags;
  auto &maxTrOD      = baseAggData->maxTrOD;
  auto &offProcTrace = baseAggData->offProcTrace;

  VertexAgglomerator<ENERGY, TMESH>::InitializeAggData(*baseAggData);

  int maxAggSize = pow(2, cfg.numRounds) + 20; // probably, at least ...

  // at least 64 MB, at most 256 MB, and as much as we probably
  // at most need for bigSOC check
  size_t const locHeapSize = std::min(size_t(256 * 1024 * 1024),
                                      std::max(size_t(64 * 1024 * 1024),
                                               10 * sqr(maxAggSize) * sizeof(TWEIGHT)));

  LocalHeap lh(locHeapSize, "Cthulhu");

  shared_ptr<LocCoarseMap> concLocMap; // baseLevel -> round 1
  shared_ptr<LocCoarseMap> locMap;     // round k -> round k+1

  locMap = make_shared<LocCoarseMap>( this->_mesh );
  locMap->SetAllowedFEdges( this->GetAllowedEdges() );

  concLocMap = locMap;

  { // Round 0
    auto vmap = locMap->template GetMap<NT_VERTEX>();
    vmap = -1;

    BitArray handled(FNV);
    handled.Clear();

    // Dirichlet
    if ( auto free_verts = this->GetFreeVerts() )
    {
      const auto & fvs = *free_verts;

      for (auto v : Range(vmap))
      {
        if ( !fvs.Test(v) )
          { vmap[v] = -1; handled.SetBit(v); }
      }
    }

    if ( eqc_h.IsTrulyParallel() )
    {
      // leave vertices that are coarsened by someone else out of intermed. levels
      if ( auto solidVerts = this->GetSolidVerts() )
      {
        // eliminate ghost-verts from intermediate levels - do not need to
        // take solid/ghost into account after base level !
        FM.template ApplyEQ2<NT_VERTEX>(Range(1ul, eqc_h.GetNEQCS()),
          [&](auto eq, auto vs)
        {
          for (auto v : vs)
          {
            if (!solidVerts->Test(v))
              { vmap[v] = -1; handled.SetBit(v); }
          }
        }, false); // obviously, not master only
      }
      else
      {
        // Non-master, set to Dirichlet, coarsened by someone else
        FM.template ApplyEQ2<NT_VERTEX>([&](auto eq, auto vs)
        {
          if ( !eqc_h.IsMasterOfEQC(eq) )
          {
            for (auto v : vs)
              { vmap[v] = -1; handled.SetBit(v); }
          }
        }, false); // obviously, not master only
      }
    }

    // Fixed aggs, set members to Dirichlet, deal with them in the end
    auto fixedAggs = this->GetFixedAggs();
    for (auto agg : fixedAggs)
    {
      for (auto v : agg)
      {
        vmap[v] = -1;
        handled.SetBit(v);
      }
    }

    // collapsing vertices with dominant L2-weight
    if ( cfg.vertThresh > 0 )
    {
      for(auto v : Range(vmap))
      {
        if (!handled.Test(v))
        {
          bool isL2Dominant = false;

          if constexpr(ROBUST)
          {
            isL2Dominant =
              CheckVertexWeight<ENERGY, TMU>(
                cfg.scalRelThresh,
                cfg.vertThresh,
                baseAggData->GetMaxTrOD(v),
                ENERGY::template GetVMatrix<TWEIGHT>(baseVData[v]),
                baseDiags[v],
                lh
              );
          }
          else
          {
            isL2Dominant =
              CheckVertexWeight<ENERGY, TMU>(
                cfg.scalRelThresh,
                cfg.vertThresh,
                baseAggData->GetMaxTrOD(v),
                ENERGY::template GetApproxVWeight<TWEIGHT>(baseVData[v]),
                baseDiags[v],
                lh
              );
          }

          if ( isL2Dominant )
          {
            vmap[v] = -1;
            handled.SetBit(v);
          }
        }
      }
    }

    // drop isolated vertices from the coarse level
    //   ( only do loc vertices so we are sure they are isolated!)
    if ( eqc_h.GetNEQCS() > 0 ) // idle master
    {
      auto const &fecon = *FM.GetEdgeCM();

      for (auto v : FM.template GetENodes<NT_VERTEX>(0))
      {
        if (!handled.Test(v))
        {
          if ( fecon.GetRowIndices(v).Size() == 0 )
          {
            vmap[v] = -1;
            handled.SetBit(v);
          }
        }
      }
    }

    // round 0 matching iteration (iterates in reverse, mostly from parallel to local)

    Table<int> dummyAggs;
    locMap->GetNCV()
      = PairingIteration<ENERGY, true>(
          FM,
          *baseAggData,
          *baseAggData,
          dummyAggs, // first matching - no big-SOC
          cfg,
          vmap,
          FlatArray<int>(0, nullptr), // v -> eqc map
          handled,
          this->GetSolidVerts(),
          lh, 0);

    if ( cfg.printSumms )
    {
      cout << " INITIAL PAIRING, NV " << FM.template GetNN<NT_VERTEX>() << " -> " << locMap->GetNCV() << endl;
    }
  }

  std::unique_ptr<SPWAggData<TopologicMesh, ENERGY, TMU>> currAggData = nullptr;

  if ( ( cfg.numRounds > 1 ) ||
       ( cfg.orphanRound ) )
  {
    locMap->FinishUp(lh);
    currAggData = baseAggData->Map(*locMap, cfg.diagStabBoost);
  }

  int const totalRounds = cfg.numRounds + ( cfg.orphanRound ? 1 : 0 );

  // subsequent matching iterations, potential extra round for orphan joining
  for (auto round : Range(1, totalRounds))
  {
    /** Build edge map, C2F vertex map, coarse edge connectivity, coarse V->EQ map **/
    if ( round > 1 )
    {
      locMap->FinishUp(lh);

      currAggData = currAggData->Map(*locMap, cfg.diagStabBoost);

      concLocMap = concLocMap->ConcatenateProper(locMap);
    }

    locMap = make_shared<LocCoarseMap>( concLocMap->GetMappedMesh());
    locMap->SetAllowedFEdges( concLocMap->GetAllowedCEdges() );

    auto const &currMesh = *concLocMap->GetMappedMesh();

    auto vmap = locMap->template GetMap<NT_VERTEX>();

    // is actually CV-to-eq of "concLocMap", i.e. current  v->eq!
    auto v2eq = concLocMap->GetV2EQ();

    // on intermed. levels, always handle ALL vertices!
    BitArray handled(currMesh.template GetNN<NT_VERTEX>());
    handled.Clear();

    if ( round < cfg.numRounds )
    {
      auto currAggs = concLocMap->GetMapC2F<NT_VERTEX>();

      // cout << " PAIRING ROUND " << round << ", currAggs = " << endl << currAggs << endl;

      // matching
      locMap->GetNCV()
        = PairingIteration<ENERGY, false>(
            currMesh,
            *currAggData,
            *baseAggData,
            currAggs,
            cfg,
            vmap,
            v2eq,
            handled,
            this->GetSolidVerts(),
            lh,
            round);

      if ( cfg.printSumms )
      {
        cout << " PAIRING ROUND " << round << ", NV " << currMesh.template GetNN<NT_VERTEX>() << " -> " << locMap->GetNCV() << endl;
      }

      if ( round == totalRounds - 1) // last round, no more joining!
      {
        concLocMap->ConcatenateVMap(locMap->template GetMappedNN<NT_VERTEX>(),
                                    locMap->template GetMap<NT_VERTEX>());
      }
    }
    else if ( round == cfg.numRounds )
    {
      // orphan treatment

      int numOrphans = 0;
      auto &isAggregated = handled;

      auto currAggs = concLocMap->template GetMapC2F<NT_VERTEX>();

      // cout << " combined AGGS before orphan round: " << endl << currAggs << endl;

      // cout << " ORPHAN round, look for orphans: " << endl;
      for (auto cVNr : Range(currAggs))
      {
        auto agg = currAggs[cVNr];

        if (agg.Size() > 1)
        {
          isAggregated.SetBit(cVNr);
        }
        else
        {
          // cout << " agg  #" << cVNr << " is ORPHAN, agg = "; prow(agg); cout << endl;
          numOrphans++;
        }
      }

      if ( cfg.printSumms )
      {
        cout << " ORPHAN round, numOrphans = " << numOrphans << endl;
      }

      if ( numOrphans > 0 )
      {
        // always on at least intermediate level 1 -> no solid vs!
        int const newNCV
          = JoiningIteration<ENERGY>(*currAggData,
                                     cfg,
                                     vmap,
                                     v2eq,
                                     isAggregated,
                                     lh);

        if ( cfg.printSumms )
        {
          cout << " ORPHAN round, #V " << currMesh.template GetNN<NT_VERTEX>()
              << " -> " << newNCV << endl;
        }

        if ( newNCV < currMesh.template GetNN<NT_VERTEX>() )
        {
          concLocMap->ConcatenateVMap(newNCV,
                                      locMap->template GetMap<NT_VERTEX>());
        }
      }
    }
  }

  // set up "Agglomerates"-vector, TODO: refactor this, just the "vmap" should be enough!

  /** Build final aggregates **/
  auto fixedAggs = this->GetFixedAggs();

  size_t const n_aggs_spec = 0; // spec_aggs.Size()
  size_t const n_aggs_p    = concLocMap->template GetMappedNN<NT_VERTEX>();
  size_t const n_aggs_f    = fixedAggs.Size();

  size_t const n_aggs_tot = n_aggs_spec + n_aggs_p + n_aggs_f;

  agglomerates.SetSize(n_aggs_tot);
  v_to_agg.SetSize(FM.template GetNN<NT_VERTEX>());
  v_to_agg = -1;

  auto set_agg = [&](int agg_nr, auto vs)
  {
    auto & agg = agglomerates[agg_nr];
    agg.id = agg_nr;
    int ctr_eqc = FM.template GetEQCOfNode<NT_VERTEX>(vs[0]);
    int v_eqc   = -1;
    agg.ctr = vs[0];
    agg.mems.SetSize(vs.Size());
    for (auto l : Range(vs))
    {
      v_eqc = FM.template GetEQCOfNode<NT_VERTEX>(vs[l]);
      if ( (v_eqc != 0) && (ctr_eqc != v_eqc) && (eqc_h.IsLEQ( ctr_eqc, v_eqc) ) )
        { agg.ctr = vs[l]; ctr_eqc = v_eqc; }
      agg.mems[l] = vs[l];
      v_to_agg[vs[l]] = agg_nr;
    }

    // TODO: remove
    //   for now, leave in, VERY useful for debugging !
    for (auto l : Range(vs))
    {
      v_eqc = FM.template GetEQCOfNode<NT_VERTEX>(vs[l]);
      if ( !eqc_h.IsLEQ(v_eqc, ctr_eqc) )
      {
        cout << " set_aggs ERROR, AGG " << agg << ", ctr-EQC " << ctr_eqc << ", EQ " << l << " = " << v_eqc << endl;
      }
    }
  };

  // for (auto k : Range(n_aggs_spec))
  //   { agglomerates[k] = std::move(specAggs[k]); }

  size_t agg_id = n_aggs_spec;

  /** actual agglomerates from coarsening **/
  auto c2fv = concLocMap->template GetMapC2F<NT_VERTEX>();

  for (auto agg_nr : Range(n_aggs_p))
  {
    auto aggvs = c2fv[agg_nr];
    QuickSort(aggvs);
    set_agg(agg_id++, aggvs);
  }

  /** pre-determined fixed aggs **/
  for (auto k : Range(fixedAggs))
    { set_agg(agg_id++, fixedAggs[k]); }

  MapVertsTest (agglomerates, v_to_agg);

} // SPWAgglomerator::FormAgglomerates_impl


template<class ENERGY, class TMESH, bool COMPILE_EV_BASED>
void
SPWAgglomerator<ENERGY, TMESH, COMPILE_EV_BASED>::
MapVertsTest (FlatArray<Agglomerate> agglomerates,
              FlatArray<int> v_to_agg)
{
  /**
   * TODO: remove this again eventually, but for now keep it so I
   * can overload it in one of the cpp files and do some checks...
   */
} // SPWAgglomerator::MapVertsTest

/** END SPWAgglomerator **/

} // namespace amg

#endif // SPW_AGG

#endif // FILE_SPW_AGG_IMPL_HPP
