#ifndef FILE_HDIV_STOKES_FACTORY_IMPL_HPP
#define FILE_HDIV_STOKES_FACTORY_IMPL_HPP


#include "dof_map.hpp"
#include "dyn_block.hpp"
#include "preserved_vectors_impl.hpp"

#include "hdiv_stokes_factory.hpp"
#include "utils.hpp"
#include "utils_sparseLA.hpp"

namespace amg
{

/** HDivStokesAMGFactory **/

template<class TMESH, class ENERGY>
HDivStokesAMGFactory<TMESH, ENERGY> ::
HDivStokesAMGFactory (shared_ptr<Options> _opts)
  : StokesAMGFactory<TMESH, ENERGY>(_opts)
{
  ;
} // HDivStokesAMGFactory(..)


template<class TMESH, class ENERGY>
shared_ptr<BaseAMGFactory::LevelCapsule>
HDivStokesAMGFactory<TMESH, ENERGY> ::
AllocCap () const
{
  return make_shared<HDivStokesLevelCapsule>();
} // HDivStokesAMGFactory::AllocCap

template<class TSCAL>
void
CheckEvals(FlatMatrix<TSCAL> A, LocalHeap &lh)
{
  HeapReset hr(lh);

  int const N = A.Height();

  FlatMatrix<TSCAL> evecs(N, N, lh);
  FlatVector<TSCAL> evals(N, lh);
  LapackEigenValuesSymmetric(A, evals, evecs);

  std::cout << " eval-check of " << N << " x " << N << " matrix, evals = " << std::endl;

  for (auto k : Range(N))
  {
    cout << k << ": " << evals(k) << std::endl;

    if ( abs(evals(k)) < 1e-14 )
    {
      std::cout << "     -> evec = ";
      for (auto j : Range(N))
      {
        cout << evecs(k, j) << " ";
      }
      cout << endl;
    }
  }
}


INLINE void
StructuredPrint2(int BS, FlatMatrix<double> A, int const &prec = 8, std::ostream &os = std::cout)
{
  auto H = A.Height();
  auto h = H / BS;

  auto W = A.Width();
  auto w = W / BS;

  for (auto K : Range(h))
  {
    for (auto lK : Range(BS))
    {
      auto k = BS * K + lK;

      for (auto J : Range(w))
      {
        for (auto lJ : Range(BS))
        {
          auto j = BS * J + lJ;

          auto val = A(k, j);

          if ( val > 0 )
          {
            os << " ";
          }

          os << std::scientific << std::setprecision(prec) << setw(prec + 6) << val << " ";
        }
        os << " | ";
      }
      os << endl;
    }
    os << " --------- " << endl;
  }
  os << std::defaultfloat;
}

template<class TMESH, class ENERGY>
shared_ptr<typename StokesAMGFactory<TMESH, ENERGY>::TSPM>
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildPrimarySpaceProlongation (BaseAMGFactory::LevelCapsule const &baseFCap,
                               BaseAMGFactory::LevelCapsule       &baseCCap,
                               StokesCoarseMap<TMESH>       const &cmap,
                               TMESH                        const &fmesh,
                               TMESH                              &cmesh,
                               FlatArray<int>                      vmap,
                               FlatArray<int>                      emap,
                               FlatTable<int>                      v_aggs,
                               TSPM_TM                      const *pA)
{
  /**
    *   Stokes prolongation works like this:
    *        Step  I) Find a prolongation on agglomerate facets
    *  Step II) Extend prolongation to agglomerate interiors:
    *              Minimize energy. (I) as BC. Additional constraint:
    *          \int_a div(u) = |a|/|A| \int_A div(U)    (note: we maintain constant divergence)
    *
    *   "Piecewise" Stokes Prolongation.
    *       Step I) Normal PW Prolongation as in standard AMG
    *
    *   "Smoothed" Stokes Prol takes PW Stokes prol and then:
    *       Step  I) Take PW prol, then smooth on facets between agglomerates. Additional constraint:
    *           Maintain flow through facet.
    */
  static Timer t("StokesAMGFactory::BuildPWProl_impl");
  RegionTimer rt(t);

  auto       &O    = GetOptions();
  auto const &fCap = static_cast<HDivStokesLevelCapsule const&>(baseFCap);
  auto       &cCap = static_cast<HDivStokesLevelCapsule&>      (baseCCap);

  // auto fmesh = my_dynamic_pointer_cast<TMESH>(fCap.mesh, "BuildPrimarySpaceProlongation FMESH");
  // auto cmesh = my_dynamic_pointer_cast<TMESH>(cCap.mesh, "BuildPrimarySpaceProlongation FMESH");

  /** fine mesh **/
  // auto const & FM(*fmesh); FM.CumulateData();
  auto const &FM       = fmesh; FM.CumulateData();
  auto const &fECon    = *FM.GetEdgeCM();
  auto        fEdges   = FM.template GetNodes<NT_EDGE>();
  auto        fVData   = get<0>(FM.Data())->Data();
  auto        fEData   = get<1>(FM.Data())->Data();
  auto const  FNV      = FM.template GetNN<NT_VERTEX>();
  auto const  FNE      = FM.template GetNN<NT_EDGE>();
  auto        fgv      = FM.GetGhostVerts();
  auto        free_fes = FM.GetFreeNodes();

  auto [fDOFedEdges_SB, fdofe2e_SB, fe2dofe_SB] = FM.GetDOFedEdges();
  auto &fDOFedEdges = fDOFedEdges_SB;
  auto &fdofe2e = fdofe2e_SB;
  auto &fe2dofe = fe2dofe_SB;

  /** fine matrix **/
  bool const have_spm = true;

  TSPM_TM const *fSPM = pA; // TODO

  if (baseFCap.baselevel == 0)
  {
    fSPM = fCap.embeddedRangeMatrix.get();
  }

  /** coarse mesh **/
  // auto const & CM(*cmesh); CM.CumulateData();
  auto const &CM     = cmesh; CM.CumulateData();
  auto const &cecon  = *CM.GetEdgeCM();
  auto        cEdges = CM.template GetNodes<NT_EDGE>();
  auto        cVData = get<0>(CM.Data())->Data();
  auto        cEData = get<1>(CM.Data())->Data();
  auto        cgv    = FM.GetGhostVerts();
  auto const  CNV    = CM.template GetNN<NT_VERTEX>();
  auto const  CNE    = CM.template GetNN<NT_EDGE>();

  auto [cDOFedEdges_SB, cdofe2e_SB, ce2dofe_SB] = CM.GetDOFedEdges();
  auto &cDOFedEdges = cDOFedEdges_SB;
  auto &cdofe2e = cdofe2e_SB;
  auto &ce2dofe = ce2dofe_SB;
  auto const & eqc_h = *FM.GetEQCHierarchy();

  auto aggs     = cmap.template GetMapC2F<NT_VERTEX>();
  auto c2f_edge = cmap.template GetMapC2F<NT_EDGE>();

  // TODO/DAWARI: can this be easier? I don't think so..

  /** DOF <-> edge mapping **/
  auto const &fMeshDOFs     = *fCap.meshDOFs;
  // auto const maxDOFsPerEdge =  fMeshDOFs.GetmaxDOFsPerEdge();
  auto const totalFine      =  fMeshDOFs.GetNDOF();

  auto const &finePresVecs = *fCap.preservedVectors;

  size_t const lhSize = 64 * 1024 * 1024;
  LocalHeap lh(lhSize, "HinigerSteinadler"); // 64 MB

  // std::cout << " BuildPrimarySpaceProlongation " << std::endl;

  // std::cout << " #V " << FNV << " -> " << CNV << std::endl;
  // std::cout << " #AGGS = " << v_aggs.Size() << std::endl;
  // std::cout << " #E " << FNE << " -> " << CNE << std::endl;

  // std::cout << " vmap: "; prow2(vmap); cout << endl;
  // cout << " aggs : " << endl << aggs<< endl;
  // std::cout << " emap: "; prow2(emap); cout << endl;
  // cout << " c2f_edge " << endl << c2f_edge << endl;

  /** Utility for I. **/

  // computes the first basis function
  auto computeSpecialVecs = [&](auto               const &cEdge,
                                FlatMatrix<double>        V)
  {
    /**
     * We have all fine edges and all their DOFs and data making up a coarse ss or sg-edge,
     * so every proc sharing a caorse sg edge computes a consistent V here.
     * We skip gg-edges where we have no DOFs.
     * If cEdge is DOFed (i.e. ss or sg), we must have all fine edges it is made up of because all
     * aggregates are solid one a single proc.
    */

    if (!cDOFedEdges->Test(cEdge.id)) // gg-edge!
      { return;  }

    auto fENrs = c2f_edge[cEdge.id];

    /**
     * The special BFs is "n", i.e. scaled such that it has constant 1 normal component,
     * that is the flow of the special CF is just the surface area of the physical face.
    */
    // double cSurf = 0.0;
    // for (auto k : Range(fENrs))
    // {
    //   double const fSurf = abs(fEData[fENr].flow[0]);
    //   cSurf += fSurf;
    // }

    // std::cout << " computeSpecialVeccs for CE " << cEdge << std::endl;

    int rowOff = 0;

    V.Col(0) = 0;

    for (auto k : Range(fENrs))
    {
      auto const  fENr     = fENrs[k];
      auto const &fEdge    = fEdges[fENr];
      auto const nEdgeDOFs = fMeshDOFs.GetNDOF(fEdge);

      if (nEdgeDOFs)
      {

        // double const fSurf             = abs(fEData[fENr].flow[0]);
        bool const fineFlip      = fEData[fENr].flow[0] < 0;        // does FINE DOF already flow in reverse direction?
        bool const flipForCoarse = vmap[fEdge.v[0]] != cEdge.v[0]; // do we flip between fine and coarse level

        // cout << "  fEnr " << fENrs[k] << ": " << fEdges[fENrs[k]] << ", vmaps to " << vmap[fEdges[fENrs[k]].v[0]] << " " << vmap[fEdges[fENrs[k]].v[1]]
        //      << ", data " << fEData[fENrs[k]] << " -> fips " << fineFlip << " " << flipForCoarse << std::endl;

        double const orientationFactor = (fineFlip == flipForCoarse) ? 1.0 : -1.0;
        V(rowOff, 0) = orientationFactor;

        rowOff += nEdgeDOFs;
      }
      else
      {
        // TODO: remove once I am sure it never triggers
        throw Exception("Fine Edge without DOFs for a coarse SS/SG-EDGE ???");
      }
    }
  };

  shared_ptr<TSPM> prol;

  /** I: coarse edge -> cross-agg edges **/
  {
    PreservedVectorsMap<TMESH> vecMap(cmap,
                                      fMeshDOFs,
                                      finePresVecs);

    Array<int> bufferOffsets(cmesh.template GetNN<NT_EDGE>());

    // buffer for coarse facet prol blocks
    size_t const cFBufferSize = vecMap.computeCFBufferSize(bufferOffsets);

    Array<double> facetBuffer(cFBufferSize);

    /**
     * For each coarse edge, compute the number of coarse DOFs, their coordinates wrt. the
     * fine basis (stashed in buffer) and the coordinates of the coarse preserved vectors
     * in the coarse bases (also stashed in buffer).
    */
    vecMap.computeCFProlBlocks(bufferOffsets,
                               facetBuffer,
                               computeSpecialVecs,
                               lh);

    auto [ cDOFs, cPreserved ] = vecMap.Finalize(bufferOffsets,
                                                 facetBuffer);

    // cCap.meshDOFs         = ( cMeshDOFsPtr = cDOFs );
    cCap.meshDOFs         = cDOFs;
    cCap.preservedVectors = cPreserved;

    auto &cMeshDOFs   = *cDOFs;

    // cout << " fine   Mesh-DOFS: " << endl << fMeshDOFs << endl;
    // cout << " coarse Mesh-DOFS: " << endl << cMeshDOFs << endl;
    // cout << " cPreserved = " << cPreserved << endl;

    /** create graph **/

    Array<int> perow(fMeshDOFs.GetNDOF());
    perow = 0.0;

    fmesh.template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &fEdge)
    {
      auto fineDOFs = fMeshDOFs.GetDOFNrs(fEdge);

      if ( free_fes && (!free_fes->Test(fEdge.id)) ) // Dirichlet
      {
        perow[fineDOFs] = 0;
      }
      else
      {
        auto const cEnr = emap[fEdge.id];

        if (cEnr != -1) // edge connects two coarse vertices
        {
          // cout << " perow " << fEdge << " " << cMeshDOFs.GetDOFNrs(cEnr).Size() << " or " << cMeshDOFs.GetNDOF(cEnr) << endl;
          // cout << "   -> set@ " << "[ " << fineDOFs.First() << ", " << fineDOFs.Next() << ")" << " from cEdge " << cEdges[cEnr] << endl;
          perow[fineDOFs] = cMeshDOFs.GetNDOF(cEnr);
        }
        else
        {
          /**
           *  Agglomerates are formed only of "S" verts one the same proc, so from our point of view,
           *  all agglomerates consist of ONLY S or ONLY G vertices. Therefore, all SG edges are cross-agglomerate
           *  and since we only have SS and SG edges, the edge here MUST be an SS one!
           */
          auto const cv0 = vmap[fEdge.v[0]];
          auto const cv1 = vmap[fEdge.v[1]];

          if (cv0 == cv1)
          {
            if (cv0 == -1)
            {
              /**
               * Must be due to vertex-collapse. Since I don't use that ATM, throw an exception since
               * that case is untested with the Stokes prols.
               */
              perow[fineDOFs] = 0;
              throw Exception("Weird case B!");
            }
            else
            {
              // auto coarseENrs = cecon.GetRowIndices(cv0);
              auto coarseENrs = cecon.GetRowValues(cv0);

              int const totalDOFs = std::accumulate(coarseENrs.begin(), coarseENrs.end(), 0,
                [&](auto const &partialSum, auto const &cENum) { return partialSum + cMeshDOFs.GetNDOF(int(cENum)); });

              perow[fineDOFs] = totalDOFs;
            }
          }
          else {
            /**
             * Edge between Dirichlet BND vertex and interior one.
             * I THINK this case should be caught above and we should ever get here, so throw an
             * exception.
            */
            perow[fineDOFs] = 0;
            throw Exception("Weird case B!");
          }
        }
      }

    }, false); // also handle sg-edges in non-master eqcs

    // cout << endl << " perow: " << endl << perow << endl;


    /** allocate prolongation **/
    prol = make_shared<SparseMatrix<double>> (perow, cMeshDOFs.GetNDOF());
    auto & P(*prol);
    const auto & const_P(*prol);
    P.AsVector() = -42; // debug vals

    /**
     * Fill cross-agg prol-blocks from buffer
     * Compute & set coarse flow for all ss and sg edges. This means that the c-edge data is
     * in an inconsistent state - the status is CUMULATED, and the weights are correct, but the
     * flow is zero everywhere where the edge is gg. We fix this later in FinalizeCoarseMap!
    **/
    cmesh.template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &cEdge)
    {
      HeapReset hr(lh);

      if (!cDOFedEdges->Test(cEdge.id)) // gg-edge
      {
        // std::cout << " SKIP CROSS " << cEdge << std::endl;
        return;
      }

      auto fENRs = c2f_edge[cEdge.id];

      auto const nFDOFs = std::accumulate(fENRs.begin(), fENRs.end(), 0,
                                          [&](auto const &partialSum, auto const &fEnr) { return partialSum + fMeshDOFs.GetNDOF(fEnr); });

      auto const nCDOFs = cMeshDOFs.GetNDOF(cEdge.id);

      // cout << endl << "FILL CROSS " << cEdge << ", nCoarse = " << nCDOFs << std::endl;
      // cout << "    fENRs: "; prow2(fENRs); cout << " -> total nFine = " << nFDOFs << std::endl;

      FlatMatrix<double> prolBlock = vecMap.ReadProlBlockFromBuffer(bufferOffsets,
                                                                    facetBuffer,
                                                                    cEdge.id);

      // cout << " prolBlock: " << prolBlock.Height() << " x " << prolBlock.Width() << endl << prolBlock << endl;

      auto cDOFNrs = cMeshDOFs.GetDOFNrs(cEdge);

      // prol block, fine flow
      FlatVector<double> fFlow(nFDOFs, lh);

      int countRow = 0;

      for (auto k : Range(fENRs))
      {
        auto const  fENr  = fENRs[k];
        auto const &fData = fEData[fENr];

        // cout << " fe " << k << " / " << fENRs.Size() << ": " << fENr << ", #dofs " << fMeshDOFs.GetNDOF(fENr) << endl;

        // the flow is given from v0->v1, therefore, the flow in direction of the coarse
        // edge is the reverse of the fine flow when the edge flips
        double const orientationFactor = ( vmap[fEdges[fENr].v[0]] == cEdge.v[0] ) ? 1.0 : -1.0;

        for (auto l : Range(fMeshDOFs.GetNDOF(fENr)))
        {
          auto const fineDOF = fMeshDOFs.EdgeToDOF(fENr, l);

          // cout << "   fDOF " << fineDOF << endl;

          // prol-block
          auto ris = P.GetRowIndices(fineDOF);
          auto rvs = P.GetRowValues(fineDOF);

          // cout << " ris/rvs size " << ris.Size() << " " << rvs.Size() << endl;

          ris = cDOFNrs;

          // cout << " -> set RIS "; prow(ris); cout << endl;

          for (auto ll : Range(rvs))
          {
            // cout << " rv " << ll << endl;
            rvs[ll] = prolBlock(countRow, ll);
          }

          // cout << " set flow " << endl;

          fFlow(countRow) = orientationFactor * ( l == 0 ? fData.flow[l] : 0.0 );
          // fFlow(countRow) = fData.flow[l];

          countRow++;
        }
      }

      /**
        * Coarse DOF flow is just fineFlow * prol, with fineFlow as a 1xN matrix.
        * This is just as Trans(prol) * fineFlow as a vector
        */

      // cout << " calc cFlow " << endl;

      // cout << " fFlow: "; prow(fFlow); cout << endl;
      // cout << " prolBlock: " << endl << prolBlock << endl;

      FlatVector<double> cFlow(nCDOFs, lh);
      cFlow = Trans(prolBlock) * fFlow;
      // cout << " calced cFlow " << endl << cFlow << endl;

      // TODO: could check here that the special vec flow matches what it, per construction, should
      // cout << " start se tit " << endl;
      auto &cData = cEData[cEdge.id];
      cData.flow = .0;
      cData.flow[0] = cFlow[0];
      // for (auto l : Range(nCDOFs))
      //   { cData.flow(l) = cFlow(l); }
        // cout << " se tit " << endl;

    }, false); // Need to fill sg-edges of eqcs we are not master of (TODO: exchange instead??)
  }

  auto       &cMeshDOFs = *cCap.meshDOFs;
  auto const totalCoarse = cMeshDOFs.GetNDOF();

  auto       &P         = *prol;
  const auto &const_P   = *prol;


  /** II: extension to in-agg edges! **/

  /**
   * Iterates over interior I-edges and facet F-edges of an aggregate edges
   * of an aggregate and call a lambda on:
   *   - vnr "i"
   *   - position of edge in neibs of "i"
   *   - vnr "j"
   *   - position of edge in neibs of "j"  (-1 if facet-edge!)
   *   - edge-nr
  */
  auto iterateFineEdges = [&](auto agg_vs, auto lam)
  {
    for (auto k : Range(agg_vs)) {
      auto const vk  = agg_vs[k];
      auto const cVk = vmap[vk];
      auto vkNeibs   = fECon.GetRowIndices(vk);
      auto vkEnrs    = fECon.GetRowValues(vk);

      for (auto j : Range(vkNeibs))
      {
        auto const vj  = vkNeibs[j];
        auto const cVj = vmap[vj];
        int  const fEnr(vkEnrs[j]);
        // if (doco) cout << j << " vj " << vj << " cvj " << cvj << endl;
        if (cVj == cVk)
        { // neib in same agg -> interior
          auto kj = find_in_sorted_array(vj, agg_vs);
          if (vj > vk) // count interior edges only once
            { lam(vk, k, vj, kj, fEnr); }
        }
        else // neib in different agg or dirichlet -> facet
          { lam(vk, k, vj, -1, fEnr); }
      }
    }
  };

  /**
   * Iterate over fine energy contributions
   *  (vi<->vj) x (vi<->vk)
   * for vi in aggregate (vj, vk can be in agg but are not necessarily)
   * and call a lambda with:
   *  - vnr "i"
   *  - position of vnr "i" in aggregate
   *  - vnr "k" (in agg if edge i<->k is I)
   *  - vnr "j" (in agg if edge i<->j is I)
   *  - enr edge i<->j
   *  - position of edge i<->j in enrs
   *  - enr edge i<->k
   *  - position of edge i<->k in enrs
  */
  auto iterateFineEnergyContribs = [&](FlatArray<int> agg_vs,
                                       FlatArray<int> fEnrs,
                                       auto lam)
  {
    for (auto kvi : Range(agg_vs))
    {
      int  const vi       = agg_vs[kvi];
      auto       vi_neibs = fECon.GetRowIndices(vi);
      auto       vi_fs    = fECon.GetRowValues(vi);
      int  const nneibs   = vi_fs.Size();

      for (auto lk : Range(nneibs))
      {
        auto const vk   = vi_neibs[lk];
        int  const fik  = vi_fs[lk];
        auto const kfik = find_in_sorted_array(fik, fEnrs);

        for (auto lj : Range(lk))
        {
          auto const vj   = vi_neibs[lj];
          int  const fij  = vi_fs[lj];
          auto const kfij = find_in_sorted_array(fij, fEnrs);

          lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
        }
      }
    }
  };

  auto getABlocksFromMatrix = [&](auto A_II, // slice-matrix
                                  FlatArray<int> fENRsI, FlatArray<int> offsetsI, FlatArray<int> dofsI,
                                  auto  A_IF,  // slice-matrix
                                  FlatArray<int> fENRsF, FlatArray<int> offsetsF, FlatArray<int> dofsF)
  {
    HeapReset hr(lh);

    const auto &A = *fSPM;

    int countRow = 0;
    int posI, posF;

    for (auto k : Range(fENRsI))
    {
      auto const fENr = fENRsI[k];
      auto const nDK  = fMeshDOFs.GetNDOF(fENr);

      for (auto kk : Range(nDK))
      {
        // this inner loop might be slightly efficient if we go by edge again, not sure
        auto const dK  = fMeshDOFs.EdgeToDOF(fENr, kk);
        auto       ris = A.GetRowIndices(dK);
        auto       rvs = A.GetRowValues(dK);
        auto const row = countRow++;
        // cout << " RIS for row " << countRow << ", " << dK << " = "; prow(ris); cout << endl;
        // cout << " RVS  = "; prow(rvs); cout << endl;

        for (auto l : Range(ris))
        {
          auto const dL   = ris[l];

          if ( dL == dK )
          {
            // cout << " l " << dL << " II @ " << row << endl;
            A_II(row, row) = rvs[l];
          }
          else if ( ( posI = find_in_sorted_array(dL, dofsI) ) != -1)
          {
            // cout << " l " << dL << " II @ " << posI << endl;
            A_II(row, posI) = rvs[l];
          }
          else if ( ( posF  = find_in_sorted_array(dL, dofsF) ) != -1)
          {
            // cout << " l " << dL << " IF @ " << posF << endl;
            A_IF(row, posF) = rvs[l];
          }
        }
      }
    }
  };

  // V: extend fine facet -> agg-interior
  cmesh.template ApplyEQ<NT_VERTEX>([&](auto const &eqc, auto const &CV)
  {
    HeapReset hr(lh);

    auto const agg_nr = CV; // I think?

    auto agg_vs = v_aggs[agg_nr];

    /**
     * ghost-agg: has no interior DOFs
     *   Therefore, after this, all interior edges must be ss, and all exterior ss or sg,
     *   that is they all have DOFs!
    */
    if ( (fgv != nullptr) && fgv->Test(agg_vs[0]))
      { return; }

    /** aggregates made up of a single vertex have no interior edges **/
    if (agg_vs.Size() <= 1)
      { return; }

    // bool doco = ( (agg_vs.Pos(117) != -1) || agg_vs.Contains(243) );
    static constexpr bool doco = false;

    /** If we get here, the agg must be S, so all interior edges are SS **/


    // const bool doco = bdoco && ( (agg_vs.Contains(84) || agg_vs.Contains(96) || agg_vs.Contains(1246) || agg_vs.Contains(1245) || (cv == 146) ) );
    // const bool doco = (cv == 486);
    // const bool doco = bdoco;
    if (doco) {
      cout << endl << "FILL AGG FOR CV " << CV << "/" << v_aggs.Size() << endl;
      cout << endl << "FILL AGG FOR CV " << CV << "/" << v_aggs.Size() << endl;
      cout << "fill agg " << agg_nr << ", agg_vs: "; prow(agg_vs); cout << endl;
      cout << "v vols: " << endl;
      for (auto v : agg_vs)
        { cout << fVData[v].vol << " "; }
      cout << endl;
    }

    auto cNeibs = cecon.GetRowIndices(CV);
    auto cEnrs = cecon.GetRowValues(CV);

    // # fine/coarse elements
    int const nfv = agg_vs.Size();
    int const ncv = 1;

    /** count fine edges **/
    int const nCEdges = cEnrs.Size(); // # coarse edges
    int       nFE     = 0;            // # fine edges (total)
    int       nFEI    = 0;            // # fine edges (int)
    int       nFEF    = 0;            // # fine edges (facet)

    iterateFineEdges(agg_vs, [&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
      nFE++;
      if (kj == -1) // ex-facet
        { nFEF++; }
      else
        { nFEI++; }
    });

    /**
     * There are no facet-edges, which sometimes happens on the coarsest level when
     * all vertices are in a single agglomerate. There is nothing to do in that case!
    */
    if (nFEF == 0)
      { return; }

    /**
     * There are no interior edges, in which case there is nothing to do.
     * I think this only happens for fictitious vertex aggs.
    */
    if (nFEI == 0)
      { return; }


    /** fine edge arrays **/

    FlatArray<int> fENRsI (nFEI, lh); // fine edge-nrs I
    FlatArray<int> fENRsF (nFEF, lh); // fine edge-nrs F

    nFEI = nFEF = 0;
    // cout << " iterateFineEdges " << endl;
    iterateFineEdges(agg_vs, [&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
      // cout << " vi " << vi << " @ " << ki
      //      << " vj " << vj << " @ " << kj
      //      << " conn. by eid " << eid << endl;
      // cout <<    " maps V " << vmap[vi] << " " << vmap[vj] << " E " << emap[eid] << endl;
      if (kj == -1)
        { fENRsF[nFEF++] = eid; }
      else
        { fENRsI[nFEI++] = eid; }
    });

    QuickSort(fENRsI);
    QuickSort(fENRsF);

    /**
     * We extend to aggregate-interior DOFs by solving the local problem
     *   | A_II A_IF B_I^T | | u_I | = 0
     *   | A_FI A_FF B_F^T | | u_F | = 0
     *   | B_I  B_F        | |  p  | = f
     * where:
     *    i) A_** are the row/cols of the sparse matrix
     *   ii) Dirichlet conditions are: u_F = P_F U
     *  iii) Bu = f enforces div(u) = const = c with
     *       a) c = 1/|A| \int_{\partial A} u_F\cdot n   (without outflow)
     *       b) c = 0                                    (with    outflow)
     * That is, we solve
     *   | A_II B_I^T | | u_I | = -A_IF u_F
     *   | B_I        | |  p  | = f - B_F u_F
     * where
     *   (Bu)_i = 1/|v_i| \int_{v_i} div(u)
     *     f_i  = (a) 1/|A| \int_{A} div(u)
     *            (b) 0
     *   That is, f_i = B_C U
     *
     * So we have
     *   | A_II B_I^T | | u_I | = | -A_IF |  P_F U + |  0  | U
     *   | B_I        | |  p  | = | - B_F |          | B_C |
     * where P_F is the facet-prol and B_C is the same in every row.
     * Or:
     *   | u_I |  = M_I^{-1} (| -A_IF |  P_F + |  0  |) U = M_I^{-1} M_{IC} U
     *   |  p  |             (| -B_F  |        | B_C |)
     *
     * Notes:
     *   i) The # of conditions enforced by "B" is the # of "real"
     *      (i.e. non-fictitious) vertices.
     *        - fict. Dirichlet vertices get dropped and are not in any agg,
     *          edges connecting to those are F,
     *        - fict. Non-Dirichlet vertices represent outflow boundaries,
     *          they SHOULD not be in any agglomerate either?
     *      ACTUALlY, I DONT !!THINK!! non-real vertices may ever occur here,
     *      so check for them and throw an exception if encountered!
     *  ii) If there is more than one "real" vertex AND NO OUTFLOW,
     *      we need one more condition to lock the constant pressure.
     *      ACTUALLY, THIS SHOULD NOW ALWAYS BE THE CASE
     * iii) I DONT THINK I EVER HAVE OUTFLOW THE WAY I DO THIS NOW !?
    */

    for (auto vi : agg_vs) {
      auto v_vol = fVData[vi].vol;
      // cout << " vi " << vi << " with vol " << v_vol << endl;
      if (v_vol < 0.0)
        { throw Exception("FICT. VERTEX WHERE NOT EXPECTED!!"); }
    }

    // bool hasOutflow = false;

    auto unsortedCENrs = cecon.GetRowValues(CV); // NOT sorted!

    FlatArray<int> cENRs(unsortedCENrs.Size(), lh); // SORTED

    convertCopy(unsortedCENrs, cENRs);
    QuickSort(cENRs);

    /** If there is more than one element and no outflow facet, we need to lock constant pressure.
        If we have only one "real" element, there should always be an outflow, I believe. **/
    bool const lockConstantPressure = true;

    // int const nI = std::accumulate(fENRsI.begin(), fENRsI.end(),
    //   [&](auto const &partialSum, auto const &fENum) { return partialSum + fMeshDOFs.GetNDOF(fENum); });
    // int const nF = std::accumulate(fENRsF.begin(), fENRsF.end(),
    //   [&](auto const &partialSum, auto const &fENum) { return partialSum + fMeshDOFs.GetNDOF(fENum); });

    auto [ nI, localIDOFOffsets, iDOFNrs ] = fMeshDOFs.SubDOFs(fENRsI, lh);
    auto [ nF, localFDOFOffsets, fDOFNrs ] = fMeshDOFs.SubDOFs(fENRsF, lh);
    auto [ nC, localCDOFOffsets, cDOFNrs ] = cMeshDOFs.SubDOFs(cENRs, lh);

    // int const nC = std::accumulate(cENrs.begin(), cENrs.end(), 0,
    //   [&](auto const &partialSum, auto const &cENum) { return partialSum + cMeshDOFs.GetNDOF(int(cENum)); });

    int const nRV            = agg_vs.Size();
    int const nCPConditions  = lockConstantPressure ? 1 : 0;
    int const nConditions    = nRV + nCPConditions;

    FlatMatrix<double> M_II(nI + nConditions, nI + nConditions, lh); // the problem we invert
    FlatMatrix<double> M_IF(nI + nConditions, nF,               lh); // Dirichlet-cols of the problem we solve
    FlatMatrix<double> M_IC(nI + nConditions, nC,               lh); // M_IF * P_FC
    FlatMatrix<double> P_FC(nF              , nC,               lh); // prolongation coarse -> F
    FlatMatrix<double> P_IC(nI + nConditions, nC,               lh); // prolongation caorse -> I (THIS IS WHAT WE COMPUTE!)

    M_II = 0.0;
    M_IF = 0.0;
    M_IC = 0.0;
    P_FC = 0.0;

    auto A_II  = M_II.Rows(0, nI).Cols(0, nI);
    auto B_I   = M_II.Rows(nI, nI + nRV).Cols(0, nI);
    auto B_I_T = M_II.Rows(0, nI).Cols(nI, nI + nRV);
    auto A_IF  = M_IF.Rows(0, nI);
    auto B_F   = M_IF.Rows(nI, nI + nRV);

    // A
    if (doco)
    {
      cout << " fENRsI = "; prow(fENRsI); cout << endl;
      cout << " localIDOFOffsets = "; prow(localIDOFOffsets); cout << endl;
      cout << " iDOFNrs = "; prow(iDOFNrs); cout << endl;

      cout << " fENRsF = "; prow(fENRsF); cout << endl;
      cout << " localFDOFOffsets = "; prow(localFDOFOffsets); cout << endl;
      cout << " fDOFNrs = "; prow(fDOFNrs); cout << endl;

      cout << " cENRs = "; prow(cENRs); cout << endl;
      cout << " localCDOFOffsets = "; prow(localCDOFOffsets); cout << endl;
      cout << " cDOFNrs = "; prow(cDOFNrs); cout << endl;
    }

    getABlocksFromMatrix(A_II,
                         fENRsI, localIDOFOffsets, iDOFNrs,
                         A_IF,
                         fENRsF, localFDOFOffsets, fDOFNrs);

    // (fine) B_I, interior edges - both vertices take part
    for (auto k : Range(fENRsI))
    {
      auto const fEnr     = fENRsI[k];
      auto const nEDOFs   = fMeshDOFs.GetNDOF(fEnr);
      auto const localCol = localIDOFOffsets[k];
      auto const &fEdge   = fEdges[fEnr];

      auto const locVNum0 = find_in_sorted_array(fEdge.v[0], agg_vs);
      for (auto j : Range(nEDOFs)) {
        // B_I  (locVNum0,     localCol + j) = fEData[fEnr].flow[j];
        // B_I_T(localCol + j, locVNum0)     = fEData[fEnr].flow[j];
        B_I  (locVNum0,     localCol + j) = j == 0 ? fEData[fEnr].flow[j] : 0.0;
        B_I_T(localCol + j, locVNum0)     = j == 0 ? fEData[fEnr].flow[j] : 0.0;
      }

      auto const locVNum1 = find_in_sorted_array(fEdge.v[1], agg_vs);
      for (auto j : Range(nEDOFs))
      {
        B_I  (locVNum1,     localCol + j) = j == 0 ? -fEData[fEnr].flow[j] : 0.0;
        B_I_T(localCol + j, locVNum1)     = j == 0 ? -fEData[fEnr].flow[j] : 0.0;
      }
    }

    // (fine) B_F, F edges - only one vertex takes part (note the reversed sign)
    for (auto k : Range(fENRsF))
    {
      auto const fEnr     = fENRsF[k];
      auto const nEDOFs   = fMeshDOFs.GetNDOF(fEnr);
      auto const localCol = localFDOFOffsets[k]; // cols 0..nF
      auto const &fEdge   = fEdges[fEnr];

      auto const locVNum0 = find_in_sorted_array(fEdge.v[0], agg_vs);

      if (locVNum0 != -1)
      {
        // for (auto j : Range(nEDOFs))
        //   { B_F(locVNum0, localCol + j) = fEData[fEnr].flow[j]; }
        for (auto j : Range(nEDOFs))
          { B_F(locVNum0, localCol + j) = j == 0 ? fEData[fEnr].flow[j] : 0.0; }
      }
      else
      {
        auto const locVNum1 = find_in_sorted_array(fEdge.v[1], agg_vs);
        // for (auto j : Range(nEDOFs))
        //   { B_F(locVNum1, localCol + j) = -fEData[fEnr].flow[j]; }
        for (auto j : Range(nEDOFs))
          { B_F(locVNum1, localCol + j) = j == 0 ? -fEData[fEnr].flow[j] : 0.0; }
      }
    }

    // (fine) B_I, lock constant pressure
    if (lockConstantPressure)
    {
      int const nOtherConditions = agg_vs.Size();

      /**
       *  M_II looks like this:
       *    | A_II B_I_T       | nI
       *    | B_I        BCP_T | nOtherConditions
       *    |      BCP         | 1
      */
      auto BCP   = M_II.Rows(nI + nRV, nI + nConditions).Cols(nI, nI + nRV);
      auto BCP_T = M_II.Rows(nI, nI + nRV).Cols(nI + nRV, nI + nConditions);

      for (auto l : Range(nRV))
      {
        BCP  (0, l) = 1.0;
        BCP_T(l, 0) = 1.0;
      }
    }

    // (coarse) B_C, coarse F edges - only one vertex takes part
    auto const cVolInv = 1.0 / cVData[CV].vol;
    FlatVector<double> BcRow(nC, lh);

    // cout << " cVOl " << cVData[CV].vol << ", cVolInv " << cVolInv << endl;

    int  cntCDOFs    = 0;
    for (auto j : Range(cENRs))
    {
      int   const  cEnum             = cENRs[j];
      auto  const &cEdge             = cEdges[cEnum];
      auto  const  nCEDOFs           = cMeshDOFs.GetNDOF(cEdge);
      float const  orientationFactor = (cEdges[cEnum].v[0] == CV) ? 1.0 : -1.0;
      // cout << " flow of " << j << "-th cEdge = " << cEdge << ", dofs = " << cMeshDOFs.GetDOFNrs(cEdge)
          //  << ", "; prow(cEData[cEnum].flow); cout << endl;
      for (auto l : Range(nCEDOFs))
      {
        // BcRow(cntCDOFs++) = cVolInv * orientationFactor * cEData[cEnum].flow(l); // !! coarse flow must be set here!
        double const cDFlow = (l == 0) ? cEData[cEnum].flow(l) : 0.0;
        // cout << "   l = " << l << ", flow " << cDFlow << " -> " << cVolInv * orientationFactor * cDFlow << endl;
        BcRow(cntCDOFs++) = cVolInv * orientationFactor * cDFlow; // !! coarse flow must be set here!
      }
    }

    for (auto j : Range(agg_vs))
    {
      auto const fVnr = agg_vs[j];
      auto const fVol = fVData[fVnr].vol;

      // cout << " fv " << j << " = " << fVnr << " vol = " << fVol << endl;

      // |v| / |A| \int \partial A
      // M_IC.Row(nI + j) = fVol * BcRow; // <-- THIS IS BROKEN?
      for (auto l : Range(nC))
      {
        M_IC(nI + j, l) = fVol * BcRow(l);
      }
    }

    // P_FC
    for (auto k : Range(fENRsF))
    {
      auto const fENr = fENRsF[k];
      for (auto l : Range(localFDOFOffsets[k], localFDOFOffsets[k+1]))
      {
        auto const fDNr = fDOFNrs[l];
        auto       ris  = P.GetRowIndices(fDNr);
        auto       rvs  = P.GetRowValues(fDNr);

        for (auto ll : Range(ris))
        {
          auto const cDNr = ris[ll];
          auto const pos = find_in_sorted_array(cDNr, cDOFNrs);

          P_FC(l, pos) = rvs[ll];
        }
      }
    }

    /** Solve problem **/

    if (doco)
    {
      cout << " ----- " << endl;
      cout << " Fill interior agg " << agg_nr << ": "; prow2(agg_vs); cout << endl;
      cout << "   lockConstantPressure = " << lockConstantPressure << endl;

      cout << " nCEdges = " << nCEdges << endl;
      cout << " nFE = " << nFE << endl;
      cout << " nFEI = " << nFEI << endl;
      cout << " nFEF = " << nFEF << endl;

      cout << " fENRsI = "; prow(fENRsI); cout << endl;
      cout << " fENRsF = "; prow(fENRsF); cout << endl;

      cout << " nI = " << nI << ", iDOFNrs = "; prow2(iDOFNrs); cout << endl;
      cout << " nF = " << nF << ", fDOFNrs = "; prow2(fDOFNrs); cout << endl;
      cout << " nC = " << nC << ", cDOFNrs = "; prow2(cDOFNrs); cout << endl;

      cout << " A_II = " << endl << A_II << endl;
      cout << " A_IF = " << endl << A_IF << endl;
      cout << " B_I = " << endl << B_I << endl;
      cout << " B_F = " << endl << B_F << endl;
      cout << " M_IC = " << endl << M_IC << endl;
      cout << " P_FC = " << endl << P_FC << endl;

      cout << " BcRow = ";
      for (auto l : Range(nC))
      {
        cout << BcRow(l) << " ";
      } cout << endl;

      cout << " all of M_II = " << endl << M_II << endl;
    }

    if (O.log_level == Options::LOG_LEVEL::DBG)
    {
      cout << " check that pVecs are in kernel of A/M: " << endl;

      // A_{I x (I,F)}
      FlatMatrix<double> AI(nI, nI + nF, lh);
      AI.Cols(0, nI) = A_II;
      AI.Cols(nI, nI + nF) = A_IF;

      // "Full" B
      FlatMatrix<double> fB(nRV, nI + nF, lh);
      fB.Cols(0,  nI)      = B_I;
      fB.Cols(nI, nI + nF) = B_F;

      // cout << " full A_I rows: " << endl << AI << endl;
      // cout << " full (fine) B: " << endl << fB << endl;

      auto const &fPres = *fCap.preservedVectors;
      auto const &cPres = *cCap.preservedVectors;

      for (unsigned numP = 0; numP < fPres.GetNPreserved(); numP++)
      {
        cout << "  check AGG " << agg_nr << ", pres-vec " << numP << ", energy + diff(pres-vec,facet-prol*c-pres-vec) " << endl;

        auto pV  = fPres.GetVector(numP).FVDouble();
        auto cPV = cPres.GetVector(numP).FVDouble();

        FlatVector<double> v(nI + nF, lh);
        FlatVector<double> pcv(nI + nF, lh);

        for (auto l : Range(nI))
        {
          v[l] = pV[iDOFNrs[l]];
        }

        for (auto l : Range(nF))
        {
          v[nI + l] = pV[fDOFNrs[l]];
        }

        FlatVector<double> cv(nC, lh);

        for (auto l : Range(nC))
        {
          cv[l] = cPV[cDOFNrs[l]];
        }

        FlatVector<double> Av(nI, lh);
        FlatVector<double> Bv(nRV, lh);

        pcv.Range(0, nI)       = 0;
        pcv.Range(nI, nI + nF) = P_FC * cv;

        Av = AI * v;
        Bv = fB * v;

        // cout << " cv: " << endl << cv << endl;
        // cout << " v: " << endl << v << endl;
        // cout << " P_FC cv: " << endl << pcv << endl;

        bool hasDiff = false;

        cout << " vF, P_FC cv, diff: " << endl;
        for (int l = 0; l < nF; l++)
        {
          auto const d = v[nI + l] - pcv[nI + l];
          if (abs(d) > 1e-12)
          {
            cout << "   " << l << ": " << v[nI + l] << " | " << pcv[nI + l]
                 << "  | " << d << endl;
            hasDiff = true;
          }
        }
        if (hasDiff == false)
        {
          continue;
        }
        cout << endl;

        double const cFlow = InnerProduct(BcRow, cv);

        cout << " BcRow * cv  = " << cFlow << endl;
        cout << "   -> cFLow = " << cVData[CV].vol * cFlow << endl;

        cout << " Av: " << endl << Av << endl;
        cout << " Bv: " << endl << Bv << endl;
      }

      // cout << " TODO: check that P_FC c-pVec == f-pVec on F-DOFs" << endl;
    }


    M_IC -= M_IF * P_FC;

    if ( doco )
    {
      std::cout << " M_II: " << endl;
      StructuredPrint2(1, M_II);
      cout << endl;

      CheckEvals(M_II, lh);
    }

    CalcInverse(M_II);
    P_IC = M_II * M_IC;

    if (O.log_level == Options::LOG_LEVEL::DBG)
    {
      cout << " TODO: check that C pVec ext is f pVec " << endl;
    }


    if (doco)
    {
      cout << " final M_IC rhs = " << endl << M_IC << endl;
      CheckEvals(M_II, lh);
      cout << " M_II inv = " << endl << M_II << endl;
      cout <<  " -> PROL-BLOCk P_IC = " << endl << P_IC << endl << endl;
     }

    /** Fill prolongation **/

    int pRow = 0;
    for (auto j : Range(fENRsI))
    {
      int const fENr      = fENRsI[j];
      int const nEdgeDOFs = fMeshDOFs.GetNDOF(fENr);

      for (auto l : Range(nEdgeDOFs))
      {
        int const fDNr = fMeshDOFs.EdgeToDOF(fENr, l);
        auto ris       = P.GetRowIndices(fDNr);
        auto rvs       = P.GetRowValues(fDNr);

        ris = cDOFNrs;

        for (auto ll : Range(nC))
          { rvs[ll] = P_IC(pRow, ll); }

        pRow++;
      }
    }

    // if (doco) {
    //   // check div of all crs BFs
    //   cout << " fEnrs "; prow2(fEnrs); cout << endl;
    //   cout << " fEIndsI "; prow2(fEIndsI); cout << endl;
    //   cout << " fEIndsF "; prow2(fEIndsF); cout << endl;
    //   Array<double> divs(agg_vs.Size());
    //   for (auto j : Range(nCEdges)) {
    //     for (auto l : Range(BS)) {
    //       divs = 0.0;
    //       int cenr = cfacets[j], cdnr = c_e2dofe[cenr];
    //       cout << endl << endl << " ====== " << endl << "check div of cdof " << j << "/" << nCEdges << " = e" << cenr << "/d" << cdnr << ", component " << l << endl;
    //       // cout << "c flow " << ced[cenr].flow << endl;
    //       auto Pcol = Pext.Col(BS*j+l);
    //       for (auto kv : Range(agg_vs)) {
    // 	auto vnr = agg_vs[kv];
    // 	// cout << " @vnr " << kv << " " << vnr << endl;
    // 	if (fvd[vnr].vol <= 0)
    // 	  { divs[kv] = -1.0; continue; }
    // 	divs[kv] = 0.0;
    // 	auto ecri = fECon.GetRowIndices(vnr);
    // 	auto ecrv = fECon.GetRowValues(vnr);
    // 	for (auto kj : Range(ecri)) {
    // 	  int vj = ecri[kj], posj = find_in_sorted_array(vj, agg_vs);
    // 	  int eij = int(ecrv[kj]), dij = f_e2dofe[eij];
    // 	  int kij = find_in_sorted_array(eij, fEnrs);
    // 	  int ki_fij = find_in_sorted_array(kij, fEIndsI);
    // 	  int kf_fij = find_in_sorted_array(kij, fEIndsF);
    // 	  // cout << "  " << vj << "  " << posj << " " << eij << " " << dij << " " << kij << " " << ki_fij << " " << kf_fij << endl;
    // 	  double addval = 0.0;
    // 	  // cout << "  f flow " << fed[eij].flow << endl;
    // 	  if (ki_fij != -1) { // "free" interior edge
    // 	    for (auto kl : Range(BS))
    // 	      { addval += 1.0/fvd[vnr].vol * ( (fedges[eij].v[0]==vnr) ? 1.0 : -1.0) * (fed[eij].flow(kl) * Pcol(ki_fij*BS+kl)); }
    // 	  } else if (kf_fij != -1) { // "bnd" edge
    // 	    // cout << "  Pf block " << endl;
    // 	    // for (auto kl : Range(BS))
    // 	      // cout << "  Pf(" << BS*kf_fij+kl << " " << BS*j+l << ") = " << Pf(BS*kf_fij+kl, BS*j+l) << endl;
    // 	    for (auto kl : Range(BS))
    // 	      { addval += 1.0/fvd[vnr].vol * ( (fedges[eij].v[0]==vnr) ? 1.0 : -1.0) * (fed[eij].flow(kl) * Pf(BS*kf_fij+kl, BS*j+l)); }
    // 	  }
    // 	  if (addval) {
    // 	    // cout << "  -> add val " << addval << " to " << kv << ", old = " << divs[kv] << endl;
    // 	    divs[kv] += addval;
    // 	    // cout << "  is now " << divs[kv] << endl;
    // 	  }
    // 	} // ecri
    //       } // agg_vs
    //       cout << " CHECKED DIVS "; prow2(divs); cout << endl;
    //       for (auto j : Range(divs))
    // 	{ divs[j] *= fvd[agg_vs[j]].vol; }
    //       cout << " CHECKED int(DIVS) "; prow2(divs); cout << endl;
    //       double tot_int_div = 0.0, totvol = 0.0;
    //       for (auto j : Range(divs))
    // 	{ tot_int_div += divs[j]; totvol += fvd[agg_vs[j]].vol; }
    //       cout << " CHECKED td_int " << tot_int_div << " td_avgval " << tot_int_div/totvol << " totvol " << totvol << endl;
    //       cout << " div-avg_div = " << endl;
    //       for (auto j : Range(divs))
    // 	{ cout << divs[j]/fvd[agg_vs[j]].vol - tot_int_div/totvol << " "; }
    //       cout << " int(div-avg_div) = " << endl;
    //       for (auto j : Range(divs))
    // 	{ cout << divs[j] - fvd[agg_vs[j]].vol * tot_int_div/totvol << " "; }
    //     } // BS
    //   } // cdof
    // } // doco
  }, false); // // also handle sg-edges in non-master eqcs


  if (O.log_level == Options::LOG_LEVEL::DBG)
  {
    auto const rk = eqc_h.GetCommunicator().Rank();
    auto const fn = "stokes_mesh_dofs_rk_" + std::to_string(rk) + "_l_" + std::to_string(fCap.baselevel + 1) + ".out";
    std::ofstream of(fn);
    cCap.meshDOFs->template PrintAs<TMESH>(of);
  }


  if (O.log_level == Options::LOG_LEVEL::DBG)
  {

    auto const &fPres = *fCap.preservedVectors;
    auto const &cPres = *cCap.preservedVectors;

    auto pPCV  = fCap.uDofs.CreateVector();
    auto pDiff = fCap.uDofs.CreateVector();
    auto pAV   = fCap.uDofs.CreateVector();

    auto &pCV  = *pPCV;
    auto &diff = *pDiff;
    auto &aV   = *pAV;

    cout << " HDIV-STOKES-FACTORY, check pres-vec ENERGY " << fCap.baselevel << endl;

    for (unsigned numP = 0; numP < min(DIM, fPres.GetNPreserved()); numP++)
    {

      const auto &A = *fSPM;

      auto const &fV = fPres.GetVector(numP);

      aV = A * fV;

      auto fVV   = fV.FVDouble();
      auto aVV = aV.FVDouble();

      cout << "   check ENERGY A pV " << numP << endl;

      FM.template Apply<NT_EDGE>([&](auto const &fEdge)
      {
        auto const nEdgeDOFs = fMeshDOFs.GetNDOF(fEdge);

        if (nEdgeDOFs == 0)
          { return; }

        double n = 0.0;
        double d = 0.0;

        for (auto dof : fMeshDOFs.GetDOFNrs(fEdge))
        {
          d += aVV[dof] * aVV[dof];
          n += fVV[dof] * fVV[dof];
        }

        n = sqrt(n);
        d = sqrt(d);

        if (d < 1e-12 * n || d < 1e-10)
          { return; }

        auto const cENum = emap[fEdge.id];

        cout << "     ENERGY @ fEdge " << fEdge << " -> cEdgeNum " << cENum << endl;

        if (cENum != -1)
        {
          cout << "          cEdge: " << cEdges[cENum] << endl;
        }

        for (auto dof : fMeshDOFs.GetDOFNrs(fEdge))
        {
          cout << "     dof " << dof << ", vec: " << fVV[dof] << ", Avec: " << aVV[dof] << endl;
        }
        cout << endl;

      });
    }


    cout << " HDIV-STOKES-FACTORY, check pres-vecs " << fCap.baselevel << " -> " << fCap.baselevel + 1 << endl;

    for (unsigned numP = 0; numP < fPres.GetNPreserved(); numP++)
    {
      auto const &fV = fPres.GetVector(numP);
      auto const &cV = cPres.GetVector(numP);

      cout << "   check PV " << numP << endl;

      pCV = (*prol) * cV;
      diff = fV - pCV;

      auto fVV   = fV.FVDouble();
      auto diffV = diff.FVDouble();
      auto pCVV  = pCV.FVDouble();

      auto const &fMeshDOFs = *fCap.meshDOFs;

      FM.template Apply<NT_EDGE>([&](auto const &fEdge)
      {
        auto const nEdgeDOFs = fMeshDOFs.GetNDOF(fEdge);

        if (nEdgeDOFs == 0)
          { return; }

        double n = 0.0;
        double d = 0.0;

        for (auto dof : fMeshDOFs.GetDOFNrs(fEdge))
        {
          d += diffV[dof] * diffV[dof];
          n += fVV[dof] * fVV[dof];
        }

        n = sqrt(n);
        d = sqrt(d);

        if (d < 1e-8 * n || d < 1e-12)
          { return; }

        auto const cENum = emap[fEdge.id];

        // pres-vecs DIM..end are not zero-energy and therefore not
        // preserved exactly on interior facets
        if (numP >= DIM && cENum == -1)
        {
          return;
        }

        cout << "     DIFF @ fEdge " << fEdge << " -> cEdgeNum " << cENum << endl;

        if (cENum != -1)
        {
          cout << "          cEdge: " << cEdges[cENum] << endl;
        }

        for (auto dof : fMeshDOFs.GetDOFNrs(fEdge))
        {
          cout << "     dof " << dof << ", vals " << fVV[dof] << " " << pCVV[dof] << " -> DIFF " << diffV[dof] << endl;
        }
        cout << endl;

      });


    }

  }

  return prol;
} // StokesAMGFactory::BuildPrimarySpaceProlongation


template<class TMESH, class ENERGY>
void
HDivStokesAMGFactory<TMESH, ENERGY> ::
FinalizeCoarseMap (StokesLevelCapsule     const &aFCap,
                   StokesLevelCapsule           &aCCap,
                   StokesCoarseMap<TMESH>       &cMap)
{
  auto &fCap = *my_dynamic_cast<HDivStokesLevelCapsule const>(&aFCap, "FinalizeCoarseMap - cap");
  auto &cCap = *my_dynamic_cast<HDivStokesLevelCapsule>(&aCCap, "FinalizeCoarseMap - cap");

  /**
   * Coarse edge-data is now CUMULATED, but the edge-flow is only computed where an edge is sg.
   * To make this consistent again, we zero out all coarse edge-data where we are not the master
   * OF THE DOFED EDGE. (NOT just the edge because we CAN be master of a gg edge) and set the
   * status to DISTRIBUTED
  */
  auto  cMesh = my_dynamic_pointer_cast<TMESH>(cCap.mesh, "FinalizeCoarseMap - mesh");
  auto &CM    = *cMesh;

  auto  fMesh = my_dynamic_pointer_cast<TMESH>(fCap.mesh, "FinalizeCoarseMap - mesh");
  auto &FM    = *fMesh;

  // TODO: should this coarse flow stuff be in BuildPrimarySpaceProlongation ?
  if (CM.GetEQCHierarchy()->GetCommunicator().Size() > 2)
  {
    auto [dofed_edges_SB, dofe2e_SB, e2dofe_SB] = CM.GetDOFedEdges();
    auto &dofed_edges = dofed_edges_SB;
    auto &dofe2e = dofe2e_SB;
    auto &e2dofe = e2dofe_SB;

    // cout << " CM.GetDofedEdgeUDofs(1) = " << CM.GetDofedEdgeUDofs(1) << endl;
    // cout << " CM.GetDofedEdgeUDofs(1).GetParallelDofs() = " << CM.GetDofedEdgeUDofs(1).GetParallelDofs() << endl;

    auto cDOFedEdgePds = *CM.GetDofedEdgeUDofs(1).GetParallelDofs();

    auto const &cMeshDOFs     = *static_cast<HDivStokesLevelCapsule&>(cCap).meshDOFs;

    // we do not have uDofs yet!
    // auto const &cDOFedEdgePds = *cCap.uDofs.GetParallelDofs();

    auto parCEData = get<1>(CM.Data());
    auto cEData = parCEData->Data();

    CM.template ApplyEQ<NT_EDGE>(Range(1ul, CM.GetEQCHierarchy()->GetNEQCS()), // only non-local eqcs
                                 [&](auto cEQ, auto const &cEdge)
    {
      auto const dofENr = e2dofe[cEdge.id];

      if ( ( dofENr == -1 ) || // gg
           ( !cDOFedEdgePds.IsMasterDof(dofENr) ) ) // ss or sg
      {
        cEData[cEdge.id] = 0.0;
      }
    }, false); // !! everyone!

    parCEData->SetParallelStatus(DISTRIBUTED);
  }

  BASE::FinalizeCoarseMap(fCap, cCap, cMap);

  // cout << " DOF-BLOCKINGs level " << fCap.baselevel << ": " << endl;
  const_cast<HDivStokesLevelCapsule&>(fCap).dOFBlocking        = BuildDOFBlocking(FM, *fCap.meshDOFs);
  const_cast<HDivStokesLevelCapsule&>(fCap).preCtrCDOFBlocking = BuildDOFBlocking(CM, *cCap.meshDOFs);

  cCap.dOFBlocking        = fCap.preCtrCDOFBlocking;
} // HDivStokesAMGFactory::FinalizeCoarseMap


template<class TMESH, class ENERGY>
DynVectorBlocking<>
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildDOFBlocking(TMESH    const &mesh,
                 MeshDOFs const &meshDOFs)
{
  auto [dofed_edges, dofe2e, e2dofe] = mesh.GetDOFedEdges();

  Array<unsigned> cDOFOffsets(dofe2e.Size() + 1);
  cDOFOffsets[0] = 0;

  for (auto k : Range(dofe2e.Size()))
  {
    auto eNum = dofe2e[k];

    cDOFOffsets[k+1] = cDOFOffsets[k] + meshDOFs.GetNDOF(eNum);
  }

  // cout << " dofe2e: " << dofe2e.Data() << endl << dofe2e << endl;
  // cout << " e2dofe: " << e2dofe.Data() << endl << e2dofe << endl;
  // cout << " dofed_edges S " << dofed_edges->Size() << endl;

  // cout << " DOF-blocking " << dofe2e.Size() << " -> " << meshDOFs.GetNDOF() << endl;
  // cout << cDOFOffsets << endl;

  return DynVectorBlocking<>(dofe2e.Size(),
                             meshDOFs.GetNDOF(),
                             std::move(cDOFOffsets));
} // HDivStokesAMGFactory::BuildDOFBlocking


template<class TMESH, class ENERGY>
void
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildDivMat (StokesLevelCapsule& stokesCap) const
{
  static Timer t("HDivStokesAMGFactory::BuildDivMat");
  RegionTimer rt(t);

  auto &cap = static_cast<HDivStokesLevelCapsule&>(stokesCap);

  auto const &M = *my_dynamic_pointer_cast<TMESH>(cap.mesh, "HDivStokesAMGFactory::BuildDivMat");
  M.CumulateData();

  auto const &eqc_h = *M.GetEQCHierarchy();

  auto const &meshDOFs = *cap.meshDOFs;

  /**
   *  Use range DOFs for ALL vertices, also ghost ones.
   *  Reason is that dofed edges can also have one ghost vertex.
   *  TODO: Rows without entries (can this happen?)
   *  TODO: would it not be better to have only solid verts??
   */
  shared_ptr<ParallelDofs> l2_pds = nullptr;

  if (cap.uDofs.IsParallel())
  {
    TableCreator<int> c_l2_dps(M.template GetNN<NT_VERTEX>());
    for (; !c_l2_dps.Done(); c_l2_dps++) {
      M.template ApplyEQ<NT_VERTEX>([&](auto eqc, auto v) {
        c_l2_dps.Add(v, eqc_h.GetDistantProcs(eqc));
      }, false);
    }
    auto l2_dps = c_l2_dps.MoveTable();
    l2_pds = make_shared<ParallelDofs>(eqc_h.GetCommunicator(), std::move(l2_dps), 1, false);
  }

  shared_ptr<TDM> div_mat;
  {
    auto vData = get<0>(M.Data())->Data();
    auto eData = get<1>(M.Data())->Data();
    auto edges = M.template GetNodes<NT_EDGE>();

    // does that even make sense for HDiv??
    // auto [dofed_edges, dofe2e, e2dofe] = M.GetDOFedEdges();
    auto [dofed_edges_SB, dofe2e_SB, e2dofe_SB] = M.GetDOFedEdges();
    auto &dofed_edges = dofed_edges_SB;
    auto &dofe2e = dofe2e_SB;
    auto &e2dofe = e2dofe_SB;
    auto NE = M.template GetNN<NT_EDGE>(), NE_dofed = dofe2e.Size();
    auto r_pds = cap.uDofs.GetParallelDofs();

    Array<int> perow(M.template GetNN<NT_VERTEX>());
    perow = 0.0;

    M.template ApplyEQ2<NT_EDGE>([&](auto eqc, auto edges)
    {
      for (auto const &edge : edges)
      {
        auto const nEdgeDOFs = meshDOFs.GetNDOF(edge);

        perow[edge.v[0]] += nEdgeDOFs;
        perow[edge.v[1]] += nEdgeDOFs;
      }
    }, false); // TODO: MPI

    // cout << " BuildDivMat on " << cap.baselevel << " #dofed-e = " << NE_dofed << ", #mesh-DOFS " << meshDOFs.GetNDOF() << endl;

    // cout << endl << " LEVEL " << cap.baselevel << " perow " << endl; prow2(perow); cout << endl << endl;
    div_mat = make_shared<TDM>(perow, meshDOFs.GetNDOF());

    perow = 0;

    M.template ApplyEQ2<NT_EDGE>([&](auto eqc, auto edges)
    {
      for (auto const &edge : edges)
      {
        auto const nEdgeDOFs = meshDOFs.GetNDOF(edge);
        auto       ris0      = div_mat->GetRowIndices(edge.v[0]);
        auto       ris1      = div_mat->GetRowIndices(edge.v[1]);
        for (auto dof : meshDOFs.GetDOFNrs(edge))
        {
          ris0[perow[edge.v[0]]++] = dof;
          ris1[perow[edge.v[1]]++] = dof;
        }
      }
    }, false); // TODO: MPI

    for (auto k : Range(perow))
      { QuickSort(div_mat->GetRowIndices(k)); }

    // M.template ApplyEQ2<NT_EDGE>([&](auto eqc, auto edges)
    // {
    //   for (auto const &edge : edges)
    //   {
    //     auto const nEdgeDOFs = meshDOFs.GetNDOF(edge);
    //     auto const firstDOF  = meshDOFs.EdgeToDOF(edge, 0);
    //     auto       ris0      = div_mat->GetRowIndices(edge.v[0]);
    //     auto       rvs0      = div_mat->GetRowValues(edge.v[0]);
    //     auto const pos0      = find_in_sorted_array(firstDOF, ris0);
    //     auto       ris1      = div_mat->GetRowIndices(edge.v[1]);
    //     auto       rvs1      = div_mat->GetRowValues(edge.v[1]);
    //     auto const pos1      = find_in_sorted_array(firstDOF, ris1);
    //     auto const &flow     = eData[edge.id].flow;
    //     for (auto l : Range(nEdgeDOFs))
    //     {
    //       rvs0[pos0 + l] =  flow(l);
    //       rvs1[pos1 + l] = -flow(l);
    //     }
    //   }
    // }, false); // TODO: MPI

    M.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto verts)
    {
      for (auto const &v : verts)
      {
        auto       ris  = div_mat->GetRowIndices(v);
        auto       rvs  = div_mat->GetRowValues(v);
        int const nCols = ris.Size();
        int       c     = 0;
        while (c < nCols)
        {
          auto          eNum      = meshDOFs.DOFToEdge(ris[c]);
          auto const    nEdgeDOFs = meshDOFs.GetNDOF(eNum);
          auto const   &flow      = eData[eNum].flow;
          double const  orient    = (edges[eNum].v[0] == v) ? 1.0 : -1.0;
          // cout << "row " << v << ", pos " << c << "/" << rvs.Size() << ", col " << ris[c] << ", add " << nEdgeDOFs << " from edge " << eNum << endl;
          // for (auto l : Range(nEdgeDOFs))
          //   { rvs[c++] = orient * flow(l); }
          for (auto l : Range(nEdgeDOFs))
            { rvs[c++] = orient * (l == 0 ? flow(l) : 0.0); }
        }

      }
    }, false); // TODO: MPI
  }

  cap.rr_uDofs = UniversalDofs(l2_pds, M.template GetNN<NT_VERTEX>(), 1);
  cap.div_mat = div_mat;
} // HDivStokesAMGFactory :: BuildDivMat


template<class TMESH, class ENERGY>
void
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildCurlMat (StokesLevelCapsule& stokesCap) const
{
  static Timer t("NCStokesAMGFactory::HDivStokesAMGFactory");
  RegionTimer rt(t);

  /**
   * For loop, we find some function that has constant flow along each edge in the loop.
   * Since we can have multiple DOFs per edge that can have non-zero flow, that means we
   * have many ways to construct such a function. We ignore all DOFs 1..nEdgeDOFs and just
   * use the first, special, "n"-DOF.
   */

  auto &O = GetOptions();

  auto &cap = static_cast<HDivStokesLevelCapsule&>(stokesCap);

  auto const &M = *my_dynamic_pointer_cast<TMESH>(cap.mesh, "HDivStokesAMGFactory::BuildDivMat");
  M.CumulateData();

  auto const &eqc_h = *M.GetEQCHierarchy();

  auto const &meshDOFs = *cap.meshDOFs;

  auto loops = M.GetLoops();
  auto active_loops = M.GetActiveLoops();
  auto edata = get<1>(M.Data())->Data();
  // auto [dofed_edges, dofe2e, e2dofe] = M.GetDOFedEdges();
  auto [dofed_edges_SB, dofe2e_SB, e2dofe_SB] = M.GetDOFedEdges();
  auto &dofed_edges = dofed_edges_SB;
  auto &dofe2e = dofe2e_SB;
  auto &e2dofe = e2dofe_SB;

  size_t const ND = meshDOFs.GetNDOF();

  auto edges = M.template GetNodes<NT_EDGE>();

  // cout << " BuildCurlMat " << endl;
  // cout << " EDGES " << endl;
  // for (auto k : Range(edata)) {
  //   cout << edges[k] << " /// " << edata[k] << endl;
  // }
  // cout << endl;
  // cout << " dofed_edges " << endl << dofed_edges << endl;
  // cout << " dof -> edge " << endl; prow2(dofe2e); cout << endl;
  // cout << " edge -> dof " << endl; prow2(e2dofe); cout << endl;
  // cout << " datas " << dofe2e.Data() << " " << e2dofe.Data() << endl;

  Array<int> perow(loops.Size()); perow = 0;
  for (auto k : Range(loops.Size()))
    if ( (!active_loops) || (active_loops->Test(k)) )
  for (auto ore : loops[k])
    if (dofed_edges->Test(abs(ore) - 1))
      { perow[k]++; } // only 1 DOF per loop edge!

  // cout << " loops " << endl << loops << endl;
  // cout << endl << endl;
  // cout << " perow " << endl << perow << endl;
  // cout << endl << endl;

  auto curlT_mat = make_shared<TCTM_TM>(perow, ND);

  for (auto k : Range(loops.Size())) {
    if ( active_loops && (!active_loops->Test(k)) )
      { continue; }
    auto loop = loops[k];
    auto ris = curlT_mat->GetRowIndices(k);
    auto rvs = curlT_mat->GetRowValues(k);
    int c = 0;
    for (auto j : Range(loop)) {
      int enr = abs(loop[j]) - 1;
      if (dofed_edges->Test(enr))
        { ris[c++] = meshDOFs.EdgeToDOF(enr, 0); }
    }
    QuickSort(ris);
    for (auto j : Range(loop)) {
      int enr = abs(loop[j]) - 1;
      if (dofed_edges->Test(enr)) { // YES, loops include non-dofed, (gg-) edges
        int dnr = meshDOFs.EdgeToDOF(enr, 0);
        int col = ris.Pos(dnr);
        int fac = (loop[j] < 0) ? -1 : 1;
        auto flow = edata[enr].flow;
        // 1./flow has flow 1 v0->v1 // therefore, fac/flow has flow 1 in correct direction
        rvs[col] = fac / edata[enr].flow(0); // scaling always needed for orientation!
      }
    }
  }

  auto curl_mat = TransposeSPM(*curlT_mat);

  // cout << " curl_mat " << curl_mat->Height() << " x " << curl_mat->Width() << endl;
  // print_tm_spmat(cout, *curl_mat);
  // cout << endl << endl;

  // cout << " curlT_mat " << curlT_mat->Height() << " x " << curlT_mat->Width() << endl;
  // print_tm_spmat(cout, *curlT_mat);
  // cout << endl << endl;

  cap.curl_mat_T = make_shared<TCTM>(std::move(*curlT_mat));
  cap.curl_mat = make_shared<TCM>(std::move(*curl_mat));

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_CT_rk_" + to_string(M.GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(cap.baselevel) + ".out");
    // out << "MESH " << endl;
    // out << M << endl;
    // out << endl << endl;
    out << " EDATA " << endl;
    out << edata << endl << endl;
    out << " ECON " << endl;
    if (active_loops)  {
      out << " ACTIVE_LOOPS " << endl;
      for (auto j : Range(active_loops->Size()))
        { out << j << ": " << ( active_loops->Test(j) ? 1 : 0 ) << endl; }
      out << endl;
    }
    out << *(M.GetEdgeCM()) << endl << endl;
    out << "CT-MAT " << endl;
    print_spmat(out, *cap.curl_mat_T);
  }

} // HDivStokesAMGFactory :: BuildCurlMat


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildContractDOFMap (shared_ptr<BaseGridMapStep> baseCMap,
                     shared_ptr<BaseAMGFactory::LevelCapsule> & b_cap,
                     shared_ptr<BaseAMGFactory::LevelCapsule> & b_mapped_cap) const
{
  static Timer t("BuildDOFContractMap");
  RegionTimer rt(t);

  auto &O = GetOptions();

  auto ctrMap    = my_dynamic_pointer_cast<StokesContractMap<TMESH>>(baseCMap,
                   "BuildDOFContractMap - map");
  auto fCap      = my_dynamic_pointer_cast<HDivStokesLevelCapsule>(b_cap,
                   "BuildDOFContractMap - F capsule");
  auto mappedCap = my_dynamic_pointer_cast<HDivStokesLevelCapsule>(b_mapped_cap,
                   "BuildDOFContractMap - C capsule");
  auto fMesh     = my_dynamic_pointer_cast<TMESH>(ctrMap->GetMesh(),
                   "NCStokesAMGFactory::BuildContractDOFMap FMESH");

  fCap->savedCtrMap = ctrMap;

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    auto const rk = ctrMap->GetMesh()->GetEQCHierarchy()->GetCommunicator().Rank();
    int const lev = fCap->baselevel + 1;

    ofstream outfm ("stokes_mesh_prectr_rk_" + to_string(rk) + "_l_" + to_string(lev) + ".out");
    outfm << *fMesh << endl;

    ofstream out ("stokes_ctrmap_rk_" + to_string(rk) + "_l_" + to_string(lev) + ".out");
    out << *ctrMap << endl;
  }

  // contracted meshDOFs
  auto [ contractedMeshDOFs, dofMaps ] = ContractMeshDOFs(*ctrMap, *fCap->meshDOFs);

  mappedCap->meshDOFs = contractedMeshDOFs;
  mappedCap->uDofs    = (ctrMap->IsMaster()) ? this->BuildUDofs(*mappedCap) : UniversalDofs();

  auto dofCtrMapRange = make_shared<CtrMap<double>> (fCap->uDofs.GetParallelDofs(),
                                                     mappedCap->uDofs,
                                                     Array<int>(ctrMap->GetGroup()),
                                                     std::move(dofMaps));
  if (ctrMap->IsMaster())
    { dofCtrMapRange->_comm_keepalive_hack = ctrMap->GetMappedEQCHierarchy()->GetCommunicator(); }

  // auto pot_fpd = fmesh->GetLoopUDofs().GetParallelDofs();
  // group.SetSize(fg.Size()); group = fg;
  // auto loop_maps = CopyTable(ctrMap->GetLoopMaps());
  // shared_ptr<ParallelDofs> pot_cpd = nullptr;

  if (ctrMap->IsMaster()) {
    this->BuildPotUDofs(*mappedCap);
    // pot_cpd = mappedCap.pot_uDofs.GetParallelDofs();
  }
  else
  {
    // dummy UniversalDofs
    mappedCap->pot_uDofs = UniversalDofs();
  }

  // preserved vecs
  auto const &finePreserved = *fCap->preservedVectors;

  Array<shared_ptr<BaseVector>> ctrPresVecs(ctrMap->IsMaster() ? finePreserved.GetNPreserved() : 0);

  for (auto k : Range(finePreserved.GetNPreserved()))
  {
    shared_ptr<BaseVector> ctrVec = ctrMap->IsMaster() ? dofCtrMapRange->CreateMappedVector() : nullptr;
    dofCtrMapRange->TransferF2C( &finePreserved.GetVector(k), ctrVec.get());
    if (ctrMap->IsMaster())
      { ctrPresVecs[k] = ctrVec; }
  }

  mappedCap->preservedVectors = ctrMap->IsMaster() ? make_shared<PreservedVectors>(finePreserved.GetNSpecial(), std::move(ctrPresVecs)) : nullptr;

  return dofCtrMapRange;

  // if (false)
  // {
  //   /**
  //    * I don't think we ever need to redistribute potential space vectors,
  //    * and that the Multi-step here was only used for the potential-space
  //    * AMG, which we don't/can't do anymore because we don't have commuting
  //    * potential space prolongations.
  //   */
  //   // pot DOF map

  //   auto ctr_map_pot = make_shared<CtrMap<typename strip_vec<double>::type>> (pot_fpd, pot_cpd, std::move(group), std::move(loop_maps));
  //   if (ctrMap->IsMaster())
  //     { ctr_map_pot->_comm_keepalive_hack = ctrMap->GetMappedEQCHierarchy()->GetCommunicator(); }

  //   Array<shared_ptr<BaseDOFMapStep>> step_comps(2);
  //   step_comps[0] = ctr_map_range;
  //   step_comps[1] = ctr_map_pot;
  //   auto multi_step = make_shared<MultiDofMapStep>(step_comps);

  //   return multi_step;
  // }
  // else
  // {
  //   return ctr_map_range;
  // }
} // NCStokesAMGFactory::BuildDOFContractMap


template<class TMESH, class ENERGY>
tuple<shared_ptr<MeshDOFs>, Table<int>>
HDivStokesAMGFactory<TMESH, ENERGY> ::
ContractMeshDOFs (StokesContractMap<TMESH> const &ctrMap,
                  MeshDOFs                 const &fMeshDOFs) const
{
  static Timer t("ContractCapsule"); RegionTimer rt(t);

  auto fMesh   = ctrMap.GetMesh();

  auto fComm   = fMesh->GetEQCHierarchy()->GetCommunicator();
  auto myGroup = ctrMap.GetGroup();

  shared_ptr<MeshDOFs> cMeshDOFs = nullptr;
  Table<int> dofMaps;

  if (ctrMap.IsMaster())
  {
    /** map the Mesh-DOFs **/
    auto cMesh = my_dynamic_pointer_cast<BlockTM>(ctrMap.GetMappedMesh(), "ContractMeshDOFs - cMesh");

    auto const CNE = cMesh->template GetNN<NT_EDGE>();

    Array<int> cMeshOffsets(1 + CNE);
    cMeshOffsets = 0;

    Array<Array<size_t>> allRecvOffsets(myGroup.Size());

    Array<int> dofMapSizes(myGroup.Size());

    for (auto kg : Range(myGroup))
    {
      auto const mem  = myGroup[kg];
      auto const isMe = mem == fComm.Rank();
      auto emap = ctrMap.template GetNodeMap<NT_EDGE>(kg);

      Array<size_t> &recvOffsets = allRecvOffsets[kg];

      if (!isMe)
      {
        recvOffsets.SetSize(emap.Size() + 1);
        fComm.Recv(recvOffsets, mem, NG_MPI_TAG_AMG);
      }

      FlatArray<size_t> fOffsets = isMe ? fMeshDOFs.GetOffsets() : recvOffsets;

      for (auto l : Range(emap))
      {
        auto const cENr   = emap[l];
        auto const nEDOFs = fOffsets[l + 1] - fOffsets[l];

        // #of dofs per fine edge is 0 where the edge is gg-edges, and consistent for all other
        // procs (where it is ss or sg), so we set to the max here
        cMeshOffsets[cENr + 1] = max(cMeshOffsets[cENr + 1], int(nEDOFs));
      }

      dofMapSizes[kg] = fOffsets[fOffsets.Size() -1];
    }

    cMeshDOFs = make_shared<MeshDOFs>(cMesh);

    for (auto k : Range(CNE))
    {
      cMeshOffsets[k + 1] += cMeshOffsets[k];
    }
    cMeshDOFs->SetOffsets(std::move(cMeshOffsets));

    dofMaps = Table<int>(dofMapSizes);

    auto const &cMD = *cMeshDOFs;

    for (auto kg : Range(myGroup))
    {
      auto              const mem      = myGroup[kg];
      auto              const isMe     = mem == fComm.Rank();
      auto                    emap     = ctrMap.template GetNodeMap<NT_EDGE>(kg);
      auto                    dofMap   = dofMaps[kg];
      FlatArray<size_t>       fOffsets = isMe ? fMeshDOFs.GetOffsets() : allRecvOffsets[kg];

      for (auto l : Range(emap))
      {
        auto const cENr  = emap[l];
        auto       cDOFs = cMD.GetDOFNrs(cENr);
        // std::cout << " dofs from mem " << kg << " rk " << mem << ", edge " << l << " -> " << cENr
        //           << "[ " << fOffsets[l] << ", " << fOffsets[l+1] << ")"
        //           << " -> [" << cDOFs.First() << ", " << cDOFs.Next() << ")" << endl;
        dofMap.Range(fOffsets[l], fOffsets[l + 1]) = cDOFs;
      }
    }
  }
  else
  {
    auto const master = myGroup[0];
    fComm.Send(fMeshDOFs.GetOffsets(), master, NG_MPI_TAG_AMG);
  }

  return make_tuple(cMeshDOFs, dofMaps);
} // HDivStokesAMGFactory :: ContractMeshDOFs


template<class TMESH, class ENERGY>
void
HDivStokesAMGFactory<TMESH, ENERGY> ::
DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  auto &O = GetOptions();
  if (O.check_loop_divs)
    { CheckLoopDivs(amg_levels, map); }
} // NCStokesAMGFactory::CheckKVecs


// TODO: inlined for linking, move implementation to a cpp file, definition to utils.hpp or sth
INLINE void SetUnitVec (shared_ptr<BaseVector> vec, int rank, int dof, double scale = 1.0, int bs = 1, int comp = 0)
{
  auto fv = vec->FVDouble(); fv = 0.0;
  if (auto parvec = dynamic_pointer_cast<ParallelBaseVector>(vec)) {
    auto comm = parvec->GetParallelDofs()->GetCommunicator();
    if (rank != 0) {
      if (rank == comm.Rank())
        { fv(bs*dof+comp) = scale; }
      vec->SetParallelStatus(DISTRIBUTED);
      vec->Cumulate();
    }
    else if (comm.Size() == 2) {
    if (rank == 1)
      { fv(bs*dof+comp) = scale; }
    }
    else
    {
      vec->SetParallelStatus(CUMULATED);
      auto pds = parvec->GetParallelDofs();
      auto all = make_shared<BitArray>(pds->GetNDofLocal()); all->Set();
      Array<int> gdn; int gn;
      pds->EnumerateGlobally(all, gdn, gn);
      for (auto k : Range(gdn))
        if (gdn[k] == dof)
          { fv(bs*k+comp) = 1.0; }
    }
  }
  else
    { fv(bs*dof+comp) = scale; }
}

template<class TMESH, class ENERGY>
void
HDivStokesAMGFactory<TMESH, ENERGY> ::
CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  constexpr int BS = 1;

  auto gcomm = amg_levels[0]->cap->mesh->GetEQCHierarchy()->GetCommunicator();
  auto prtv = [&](auto & vec, string title, int bs) {
    auto fv = vec.FVDouble();
    cout << title << " = " << endl;
    cout << "  size = " << vec.Size() << " " << fv.Size() << endl;
    cout << "  bs = " << bs << endl;
    cout << "  stat = " << vec.GetParallelStatus() << endl;
    cout << "  vals = " << endl;
    for (auto k : Range(vec.Size())) {
      bool nz = false;
      for (auto l : Range(bs))
        if (abs(fv(k*bs+l)) > 1e-12)
          { nz = true; break; }
      if (nz) {
        cout << k << ": ";
        for (auto l : Range(bs))
          { cout << fv(k*bs+l) << " "; }
        cout << endl;
      }
    }
    cout << endl;
  };

  auto check_div = [&](int level, string name, shared_ptr<BaseVector> lvec)
  {
    cout << " check div of vec " << name << " from level " << level << endl;
    cout << " levels " << amg_levels << endl;
    gcomm.Barrier();
    int level_loc = int(amg_levels.Size())-1;
    if (amg_levels.Last()->cap->mat == nullptr) // contracted!
      { level_loc--; }
    level_loc = min(level, level_loc);
    cout << " levels : " << level << " " << amg_levels.Size()-1 << ", use " << level_loc << endl;
    // cout << " last mat " << amg_levels.Last()->cap->mat << endl;
    Array<shared_ptr<BaseVector>> r_vecs(level+1);
    for (auto k : Range(min(level, int(amg_levels.Size()))))
      { r_vecs[k] = map->CreateVector(k); }
    // cout << " got vecs " << endl;
    if (level < amg_levels.Size())
      { r_vecs.Last() = lvec; }
    // cout << " got crst vec " << endl;
    // cout << " r_vecs " << endl << r_vecs << endl;
    gcomm.Barrier();
    /** get [0..level) vecs **/
    // need to count "empty" level from contr for vec transfers
    int level_loc_tr = min(int(amg_levels.Size())-1, level);
    for (int k = level_loc_tr-1; k >= 0; k--) // A->B, vec_A, vec_B
      { map->TransferAtoB(k+1, k, r_vecs[k+1].get(), r_vecs[k].get()); }
    cout << " transferred vecs ( " << level_loc_tr << " -> " << level_loc << ")" << endl;
    // prtv(*r_vecs.Last(), "AT r_vec", BS);
    /** check divs **/
    gcomm.Barrier();
    for (int k = level_loc; k >= 0; k--)
    {
      cout << endl << " check div of vec " << name << " from level " << level << " on level " << k << ": " << endl;
      auto slc_cap = static_pointer_cast<StokesLevelCapsule>(amg_levels[k]->cap);
      const auto & cap = static_cast<const StokesLevelCapsule&>(*amg_levels[k]->cap);
      auto pds = map->GetParDofs(k);
      // cout << " baselev " << cap.baselevel << endl;
      // cout << " mat " << cap.mat << endl;
      // cout << " pds "  << pds << endl;
      // cout << " pds2 " << cap.uDofs.GetParallelDofs() << endl;
      // cout << " pds3 " << cap.pot_uDofs.GetParallelDofs() << endl;
      // cout << " pds4 " << cap.rr_uDofs.GetParallelDofs() << endl;
      // cout << " mat name " << typeid(cap.mat).name() << endl;
      pds->GetCommunicator().Barrier();
      auto div_mat = cap.div_mat;
      cout << " div_mat " << div_mat << endl;
      shared_ptr<BaseMatrix> p_div_mat = div_mat;
      if (cap.rr_uDofs.IsParallel()) // TODO: should this not be rr_uDofs and uDofs instea dof rr_uDofs twice??
       p_div_mat = make_shared<ParallelMatrix>(p_div_mat, cap.rr_uDofs.GetParallelDofs(), cap.rr_uDofs.GetParallelDofs(), PARALLEL_OP::C2D);
      r_vecs[k]->Cumulate();
      auto rr_vec = p_div_mat->CreateColVector();
      (*rr_vec) = (*p_div_mat) * (*r_vecs[k]);
      (*rr_vec).Cumulate(); // autovector
      auto fv_r = r_vecs[k]->FV<Vec<BS, double>>();
      auto fv = (*rr_vec).FVDouble();
      double eps = 1e-8;
      const auto & M = static_cast<const TMESH&>(*cap.mesh);
      auto [dofed_edges, dofe2e, e2dofe] = M.GetDOFedEdges();
      auto edges = M.template GetNodes<NT_EDGE>();
      auto vdata = get<0>(M.Data())->Data();
      auto edata = get<1>(M.Data())->Data();
      cout << " DIV-ENTRIES > EPS = " << eps << endl;
      // cout << " ran vec sz " << r_vecs.Size() << endl;
      // cout << " rvk " << r_vecs[k] << endl;
      // prtv(*r_vecs[k], "ran vec "+to_string(k), BS);
      for (auto vnr : Range(fv))
      {
        if (abs(fv(vnr)) > eps)
        {
          cout << endl << " div on level " << k << ", vert " << vnr << " = " << fv(vnr) << ", vol = " << vdata[vnr].vol << endl;
          cout << " vert shared with = "; prow(cap.rr_uDofs.GetDistantProcs(vnr)); cout << endl;
          cout << " dnr/enr/r_val/div_val/div_val*r_val/edge_flow/edge_flow*r_val/dof-dps: " << endl;
          auto div_cols = div_mat->GetRowIndices(vnr);
          // auto div_vals = div_mat->GetRowValues(vnr);
          for (auto j : Range(div_cols))
          {
            auto dnr = div_cols[j], enr = dofe2e[dnr];
            throw Exception("Need to Dispatch over div-mat width here!");
            // double div_vf = (div_vals[j] * fv_r(dnr))(0); // result is Vec<1, double>
            // if (abs(div_vf) < 1e-8)
              // { cout << " SKIP edge " << edges[enr] << endl; continue; }
            double flow_vf = 0.0;
            for (auto l : Range(BS))
              { flow_vf += edata[enr].flow(l) * fv_r(dnr)(l); }
            if ( edges[enr].v[0] != vnr)
              { flow_vf = -flow_vf; }
            cout << " edge " << edges[enr] << ",";
            cout << " pds:: "; prow(pds->GetDistantProcs(dnr)); cout << endl;
            cout << j << " of " << div_cols.Size() << ": " << dnr << " / " << enr << " / " << fv_r(dnr) << endl;
            // cout << " divval::" << div_vals[j] << " / div::" << div_vf << endl;
            cout << " eflow:: " << edata[enr].flow << " / flow::" << flow_vf << " / ";
            cout << endl;
          }
        }
      }
      cout << "   those were all!" << endl;
    }
  };

  auto check_loop_div = [&](int rank, int level, int lnr) {
    /** get "level" range vector **/
    cout << " check_loop_div " << rank << "L " << level << " loop " << lnr << endl;
    shared_ptr<BaseVector> r_vec;
    if (level < amg_levels.Size())
    {
      auto scap = my_dynamic_pointer_cast<StokesLevelCapsule>(amg_levels[level]->cap, "CheckLoopDivs - cap");
      // cout << " SCAP = " << scap << endl;
      auto c_mat       = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).curl_mat;
      auto pot_uDofs   = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).pot_uDofs;
      auto range_uDofs = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).uDofs;
      // cout << " mat p r pds " << c_mat << " " << pot_uDofs << " " << range_uDofs << endl;
      // shared_ptr<BaseMatrix> par_c_mat = c_mat;
      // if (pot_uDofs.IsParallel())
      //  { par_c_mat = make_shared<ParallelMatrix>(c_mat, pot_uDofs.GetParallelDofs(), range_uDofs.GetParallelDofs(), PARALLEL_OP::C2C); }
      auto par_c_mat = WrapParallelMatrix(c_mat, pot_uDofs, range_uDofs, PARALLEL_OP::C2C);
      /** Get pot. space vec on correct level with one entry at correct loop # **/
      auto pot_vec = par_c_mat->CreateRowVector();
      SetUnitVec(pot_vec, rank, lnr, 1.0, 1, 0);
      /** range vector **/
      r_vec = par_c_mat->CreateColVector();
      *r_vec = 0.0;
      (*pot_vec).Cumulate();
      c_mat->Mult(*pot_vec, *r_vec);
      r_vec->Cumulate();
      cout << " rvl " << r_vec << endl;
      // prtv(*pot_vec, "pot_vec", 1);
      // prtv(*r_vec, "r_vec", BS);
    }
    check_div(level, "loop rk "+to_string(rank)+", lev "+to_string(level)+", lnr " + to_string(lnr), r_vec);
  };

  check_loop_div(0, 2, 3);
  check_loop_div(0, 2, 4);
  check_loop_div(0, 2, 5);
  check_loop_div(0, 2, 6);

} // NCStokesAMGFactory::ChekLoopDivs


template<class TMESH, class ENERGY>
UniversalDofs
HDivStokesAMGFactory<TMESH, ENERGY> ::
BuildUDofs (BaseAMGFactory::LevelCapsule const &baseCap) const
{
  auto &cap = *my_dynamic_cast<HDivStokesLevelCapsule>(&baseCap,
    "HDivStokesAMGFactory::BuildUDofs - cap");

  auto const &mesh = *my_dynamic_pointer_cast<TMESH>(cap.mesh,
    "HDivStokesAMGFactory::BuildUDofs - mesh");

  auto const &eqc_h = *mesh.GetEQCHierarchy();

  auto [fDOFedEdges_SB, fdofe2e_SB, fe2dofe_SB] = mesh.GetDOFedEdges();
  
  // lambda capute of SB
  auto &fDOFedEdges = fDOFedEdges_SB;
  auto &fdofe2e     = fdofe2e_SB;
  auto &fe2dofe     = fe2dofe_SB;
  
  auto const &meshDOFs = *cap.meshDOFs;

  if ( mesh.GetEQCHierarchy()->IsDummy() )
  {
    // contracted out
    return UniversalDofs(meshDOFs.GetNDOF(), 1);
  }
  else if ( ( cap.preservedVectors != nullptr ) &&
            ( cap.preservedVectors->GetNPreserved() == 0 ) )
  {
    /**
     * No preserved vectors, so exactly 1 DOF per edge. This method is called from
     * the PC for the finest level, and there the preservedVectors are not computed
     * yet, so we skip this optimization in that case and build a pardofs object in
     * parallel.
     */
    return mesh.GetDofedEdgeUDofs(1);
  }
  else if (eqc_h.GetCommunicator().Size() > 1)
  {
    /**
     * Not all edges that are DOFed are also shared with all procs in their EQC,
     * so it is easiest to use the static-BS-1 pardofs from stokes-mesh and create
     * new pardofs based on that!
     */
    auto const &basePDs = *mesh.GetDofedEdgeUDofs(1).GetParallelDofs();
    auto const nD = meshDOFs.GetNDOF();

    TableCreator<int> createDPs(nD);
    int totShared = 0;
    for (; !createDPs.Done(); createDPs++)
    {
      mesh.template ApplyEQ2<NT_EDGE>(Range(1ul, eqc_h.GetNEQCS()), [&](auto eqc, auto nodes)
      {
        for (auto const &edge : nodes)
        {
          auto const dofedENum = fe2dofe[edge.id];

          if (dofedENum != -1)
          {
            auto distProcs = basePDs.GetDistantProcs(dofedENum);

            for (auto dof : meshDOFs.GetDOFNrs(edge))
            {
              createDPs.Add(dof, distProcs);
            }
          }
        }
      });
    }

    cout << " tot shared: " << endl << totShared << endl;

    auto parDOFs = make_shared<ParallelDofs>(eqc_h.GetCommunicator(), createDPs.MoveTable(), 1, false);

    cout << " -> PDFS " << endl << *parDOFs << endl;

    return UniversalDofs(parDOFs);
  }
  else
  {
    return UniversalDofs(meshDOFs.GetNDOF(), 1);
  }
}


template<class TMESH, class ENERGY>
std::tuple<shared_ptr<BaseProlMap>,
           shared_ptr<BaseStokesLevelCapsule>>
HDivStokesAMGFactory<TMESH, ENERGY> ::
CreateRTZLevel (BaseStokesLevelCapsule const &primCap) const
{
  /**
   * Right now, used freedofs on level 0 comes from the SPACE and
   * on levels > 0 from free_nodes. For Stokes, the free_nodes on level > 0
   * are ALWAYS nullptr because we do not keep any Dirichlet edges in the
   * mesh (the introduced dirichlet-boundary fict. vertices are eliminated).
   *
   * If that ever changes, HDIV would need to construct freedofs from free_nodes
   * and the meshDOFs since (a) there are edges without DOFs and (b) the # of DOFs
   * per edge can vary.
   *
   * THIS ALSO MEANS THAT WE DO NOT HAVE VALID FREEDOFS FOR RTZ LEVEL ZERO!!!
   *    -> THIS LEVEL IS NEVER USED ANYWAYS!!!
   *
   * So, on secundary level 0 we ONLY NEED THE EMBEDDING AND NOTHING ELSE!
   *
  */

  auto pCap = make_shared<HDivStokesLevelCapsule>();
  auto &cap = *pCap;

  if (primCap.baselevel == 0)
  {
    throw Exception("Called into CreateRTZLevel on AMG-level 0!");
  }

  auto embStep = CreateRTZEmbeddingImpl(primCap);

  // mesh - do we need it??
  cap.baselevel = primCap.baselevel;

  // primCap.mesh = cap.mesh;

  // embedding, fine:mesh<-coarse:RTZ, so use mapped UDofs
  cap.uDofs = embStep->GetMappedUDofs();

  // range-matrix
  cap.mat = embStep->AssembleMatrix(primCap.mat);

  // cout << " RTZ-matrix = " << cap.mat << ": " << cap.mat->Height() << " x " << cap.mat->Width() << endl;

  if (primCap.free_nodes)
  {
    throw Exception("cap.free_nodes != nullptr in HDIV CreateRTZLevel!");
  }

  cap.free_nodes = primCap.free_nodes; // probably always NULLPTR ATM !?

  // if (primCap.freedofs != nullptr)
  // {
  //   auto const &TM = *my_dynamic_pointer_cast<TMESH>(cap.mesh, "CreateRTZLevel");

  //   auto [dOFedEdges, dofe2e, e2dofe] = TM.GetDOFedEdges();

  //   cap.freedofs = make_shared<BitArray>(cap.uDofs.GetND());
  //   cap.freedofs->Clear();

  //   for (auto dofE : Range(dofe2e))
  //   {
  //     auto const eNr = dofe2e[dofE];

  //     if (primCap.freedofs->Test(eNr))
  //     {
  //       cap.freedofs->SetBit(dofE);
  //     }
  //   }
  // }

  // cout << " RTZ-embedding: " << embStep->GetUDofs().GetND() << " -> " << embStep->GetMappedUDofs().GetND() << endl;

  /**
   *  We have the same potential space, and the curl-matrix
   *   pot->sec is just pot->prim->sec, using the
   *  prim->sec restriction  (i.e. inverse of sec->prim prol)
   */
  cap.pot_mat      = primCap.pot_mat;
  cap.pot_freedofs = primCap.pot_freedofs;
  cap.pot_uDofs    = primCap.pot_uDofs;

  // MatMultABGeneric returns a BaseMatrix-ptr because it can be a paralell-matrix
  // cout << "CREATE CURL-MAT " << endl;
  auto curlMat  = MatMultABGeneric(embStep->GetProlTrans(), primCap.curl_mat);
  // cout << " -> CURL-mat DIMS: " << curlMat->Height() << " x " << curlMat->Width() << endl;
  auto curlMatT = TransposeAGeneric(curlMat);

  cap.curl_mat   = GetLocalSparseMat(curlMat);
  cap.curl_mat_T = GetLocalSparseMat(curlMatT);

  return std::make_tuple(embStep, pCap);
} // HDivStokesAMGFactory::CreateRTZLevel


template<class TMESH, class ENERGY>
shared_ptr<ProlMap<double>>
HDivStokesAMGFactory<TMESH, ENERGY> ::
CreateRTZEmbeddingImpl (BaseStokesLevelCapsule const &baseCap) const
{
  auto const &cap = *my_dynamic_cast<HDivStokesLevelCapsule const>(&baseCap, "CreateRTZEmbedding - cap");

  auto const &TM = *my_dynamic_pointer_cast<TMESH>(cap.mesh, "CreateRTZEmbedding");

  // TODO: should there be a "restriction/permutation" dof-step type CLASS?

  auto const &meshDOFs = *cap.meshDOFs;

  auto [dOFedEdges, dofe2e_SB, e2dofe] = TM.GetDOFedEdges();
  auto &dofe2e     = dofe2e_SB;

  auto const nEdges = TM.template GetNN<NT_EDGE>();
  auto const nRTZ   = dofe2e.Size();

  // cout << " CreateRTZEmbedding!" << endl;

  // cout << " nDOFs  = " << meshDOFs.GetNDOF() << endl;
  // cout << " nEdges = " << nEdges << endl;
  // cout << " nRTZ   = " << nRTZ << endl;

  auto itG = [&](auto lam)
  {
    for (auto k : Range(nRTZ))
    {
      auto const eNr = dofe2e[k];

      lam(meshDOFs.EdgeToDOF(eNr, 0), k);
    }
  };

  Array<int> perow(meshDOFs.GetNDOF());
  perow = 0;

  itG([&](auto row, auto col) { perow[row]++; });

  // cout << " perow: " << endl << perow << endl;

  auto embProl = make_shared<SparseMatrix<double>>(perow, nRTZ);

  itG([&](auto row, auto col)
  {
    embProl->GetRowIndices(row)[0] = col;
    embProl->GetRowValues(row)[0]  = 1.0;
  });

  // cout << " embProl: " << endl << *embProl << endl;

  UniversalDofs meshUDofs;

  if (cap.baselevel > 0)
  {
    meshUDofs = cap.uDofs;
  }
  else
  {
    // level 0 first saved map is the embedding, the mapped UDofs of that are the
    // ones associated to the mesh!
    meshUDofs = cap.savedDOFMaps[0]->GetMappedUDofs();
  }

  // cout << " MESH-UDofs : " << endl << meshUDofs << endl;
  // cout << " orig-UD    : " << endl << TM.GetDofedEdgeUDofs(1) << endl;

  auto embStep = make_shared<ProlMap<double>>(embProl,
                                              meshUDofs,                // "fine"
                                              TM.GetDofedEdgeUDofs(1)); // "coarse"

  embStep->Finalize();

  // if (GetOptions().log_level == Options::LOG_LEVEL::DBG)
  // {
  //   cout << " MMAB PTP for RTZ level " << cap.baselevel << endl;
  //   auto ptp = MatMultABGeneric(*embStep->GetProlTrans(), *embProl);

  //   cout << " TEST PTP for RTZ level " << cap.baselevel << endl;
  //   cout <<endl << *ptp << endl << endl;
  // }

  if (GetOptions().log_level == Options::LOG_LEVEL::DBG)
  {
    int const level = cap.baselevel;
    std::string const fileName = "ngs_amg_RTZ_emb_l_" + std::to_string(level) + ".out";
    std::ofstream of(fileName);
    embStep->PrintTo(of);
  }

  return embStep;
} // HDivStokesAMGFactory::CreateRTZEmbeddingImpl


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
HDivStokesAMGFactory<TMESH, ENERGY> ::
CreateRTZEmbedding (BaseStokesLevelCapsule const &cap) const
{
  return CreateRTZEmbeddingImpl(cap);
} // HDivStokesAMGFactory::CreateRTZEmbedding


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
HDivStokesAMGFactory<TMESH, ENERGY> ::
CreateRTZDOFMap (BaseStokesLevelCapsule const &fCap,
                 BaseDOFMapStep         const &fEmb,
                 BaseStokesLevelCapsule const &cCap,
                 BaseDOFMapStep         const &cEmb,
                 BaseDOFMapStep         const &dOFStep) const
{
  // std::cout << "inDOFStep: " << typeid(inDOFStep).name() << endl;

  // BaseDOFMapStep const *dofStepPtr = &inDOFStep;

  // if (auto multiStep = dynamic_cast<MultiDofMapStep const*>(dofStepPtr))
  // {
  //   dofStepPtr = multiStep->GetMap(0).get();
  // }

  // std::cout << "dofStepPtr: " << typeid(*dofStepPtr).name() << endl;

  BaseDOFMapStep const *meshDOFStep;

  if (fCap.baselevel == 0)
  {
    // on level 0, "dOFStep" is concatenation of embedding, the first prolongation
    // (and potentially first contract-map, but that does not work yet anyways!)
    auto const &savedMaps = fCap.savedDOFMaps;

    meshDOFStep = savedMaps.Last().get();
  }
  else
  {
    meshDOFStep = &dOFStep;
  }

  auto const &prolStep = *my_dynamic_cast<ProlMap<double> const>(meshDOFStep,
                    "HDivStokesAMGFactory::CreateRTZDOFMap - DOFStep I");

  // std::cout << "fEmb: " << typeid(fEmb).name() << endl;

  auto const &fEmbTM = *my_dynamic_cast<ProlMap<double> const>(&fEmb,
                    "HDivStokesAMGFactory::CreateRTZDOFMap - DOFStep II");

  // std::cout << "cEmb: " << typeid(cEmb).name() << endl;

  auto const &cEmbTM = *my_dynamic_cast<ProlMap<double> const>(&cEmb,
                    "HDivStokesAMGFactory::CreateRTZDOFMap - DOFStep III");

  /**
   * We are lucky here, the prolongation is merely an embedding of the first
   * DOF of every "edge". That means that it is an orthogonal matrix, i.e. the
   * transpose prolongation/restriction is its inverse. That is, the DOF-step
   * in the secondary sequence can be expressed as:
   *     secProl = fineRestriction * primProl * coarseEmbedding
   *  ( that is: crs-sec. -> crs-prim -> f-prim -> f-sec )

   *  (Note: This is not the case for the NC-like AMG sequence!)
   */

  /*
   * TODO: handle ctr-map here. This is a bit of an issue because I don't have the
   *       grid-contract map anymore. It would be easy to construct the restricted
   *       dof-contract-map from that, but also should not be too big of an issue to
   *       implement restriction of the ctr-map.
  */
  // shared_ptr<TSPM> BC = MatMultAB(*prolStep.GetProl(), *cEmbTM.GetProl());

  // cout << " FEMBT = " << fEmbTM.GetProlTrans()->Height() << " x " << fEmbTM.GetProlTrans()->Width() << endl;
  // cout << " PROL  = " << prolStep.GetProl()->Height() << " x " << prolStep.GetProl()->Width() << endl;
  // cout << " CEMB  = " << cEmbTM.GetProl()->Height() << " x " << cEmbTM.GetProl()->Width() << endl;

  auto secProl = MatMultAB(*fEmbTM.GetProlTrans(),
                           *MatMultAB(*prolStep.GetProl(), *cEmbTM.GetProl()));

  auto secProlMap = make_shared<ProlMap<double>>(secProl,
                                                 fEmb.GetMappedUDofs(),  // "fine":   the mapped UD are sec-space
                                                 cEmb.GetMappedUDofs());  // "coarse": the mapped UD are sec-space
  secProlMap->Finalize();

  return secProlMap;
} // HDivStokesAMGFactory::CreateRTZDOFMap

/** END HDivStokesAMGFactory **/


} // namespace amg

#endif //  FILE_HDIV_STOKES_FACTORY_IMPL_HPP
