#include "dof_map.hpp"
#include "stokes_factory.hpp"

#include <utils_sparseLA.hpp>

#include "stokes_pc.hpp"
#include "loop_utils.hpp"

#include "stokes_pc_impl.hpp"
#include "universal_dofs.hpp"
#include "utils.hpp"

namespace amg
{

extern void DoTest (BaseMatrix &mat, BaseMatrix &pc, NgMPI_Comm & gcomm, string message);

/** BaseStokesAMGPrecond **/


BaseStokesAMGPrecond :: BaseStokesAMGPrecond (shared_ptr<BilinearForm>        blf,
                                              Flags                    const &flags,
                                              string                   const &name,
                                              shared_ptr<Options>             opts)
  : BaseAMGPC(blf, flags, name, opts)
  , _fes(blf->GetFESpace())
{
  ;
} // BaseStokesAMGPrecond(..)


// set up from assembled matrix
BaseStokesAMGPrecond :: BaseStokesAMGPrecond (shared_ptr<FESpace>           fes,
                                              Flags                        &flags,
                                              string                 const &name,
                                              shared_ptr<Options>           opts)
  : BaseAMGPC(shared_ptr<BaseMatrix>(nullptr), flags, name, opts)
  , _fes(fes)
{
  ;
} // BaseStokesAMGPrecond(..)


// shared_ptr<BaseAMGPC::Options> BaseStokesAMGPrecond :: NewOpts ()
// {
//   return make_shared<Options>();
// } // BaseStokesAMGPrecond::NewOpts


void BaseStokesAMGPrecond :: SetDefaultOptions (BaseAMGPC::Options& baseO)
{
  auto & O = static_cast<Options&>(baseO);

  O.sm_type       = Options::SM_TYPE::HIPTMAIR;
  O.sm_type_range = Options::SM_TYPE::GS;
  O.sm_type_pot   = Options::SM_TYPE::GS;
} // BaseStokesAMGPrecond::SetDefaultOptions


// void BaseStokesAMGPrecond :: SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix)
// {
//   static_cast<Options&>(O).SetFromFlags(this->GetFESpace(), finest_mat, flags, prefix);
// } // BaseStokesAMGPrecond::SetOptionsFromFlags


void BaseStokesAMGPrecond :: ModifyOptions (BaseAMGPC::Options & aO, const Flags & flags, string prefix)
{
  if (aO.force_ass_flmat)
  { throw Exception("Force assemble FLM not yet done for stokes (PC-MapLevel)"); }
} // BaseStokesAMGPrecond::ModifyOptions


shared_ptr<TopologicMesh> BaseStokesAMGPrecond :: BuildInitialMesh ()
{
  const auto & O = static_cast<Options&>(*options);

  static Timer t("BaseStokesAMGPrecond::BuildInitialMesh");
  RegionTimer rt(t);

  auto [ top_mesh, fvd ] = BuildTopMesh();
  auto alg_mesh = BuildAlgMesh(std::move(top_mesh), fvd);
  return alg_mesh;
} // BaseStokesAMGPrecond::BuildInitialMesh


tuple<shared_ptr<BlockTM>, Array<FVDescriptor>> BaseStokesAMGPrecond :: BuildTopMesh ()
{
  auto & O = static_cast<Options&>(*options);

  auto &auxInfo = GetFacetAuxInfo();

  // TODO: OVERLAP CHANGES DIRICHLET!!!! <-- is this really still TODO?? I thought it worked already!!


  /**
   *  Note: facet_to_ext_fv is not currently used from here on put, I could remove it.
   *        I used to need it loop-sorting on facet-loops where we don't have 2 elements for ext-facets,
   *        but I am now sorting loops of EDGES directly, where we always have two vertices!
   */
  auto [ mesh, fVDescriptors, aE2VMap, aF2EMap, facet_to_ext_fv] = BuildStokesMesh(GetMA(), auxInfo);

  this->EL2VMap = std::move(aE2VMap);
  this->F2EMap = std::move(aF2EMap);

  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    auto rk = rankInMatComm(*finest_mat);
    std::ofstream of("stokes_FE_2_amg_mesh_rk_" + std::to_string(rk) + ".out");
    of << " Facet-Aux info: " << endl;
    of << GetFacetAuxInfo() << endl;
    of << " -------- " << endl;

    of << " Fictitious Vertices: size = " << fVDescriptors.Size() << endl;
    of << "  ";
    prow3(fVDescriptors, of, "  ", 1);
    of << endl << endl;

    of << " EL -> V map: size = " << this->EL2VMap.Size() << endl;
    of << "  ";
    prow3(this->EL2VMap, of, "  ", 20);
    of << endl << endl;

    of << " Facet -> Edge map: size = " << this->F2EMap.Size() << endl;
    of << "  ";
    prow3(this->F2EMap, of, "  ", 20);
    of << endl << endl;

  }

  return make_tuple(mesh, fVDescriptors);
} // BaseStokesAMGPrecond::BuildTopMesh


shared_ptr<BitArray> BaseStokesAMGPrecond :: BuildGhostVerts (BlockTM &fineMesh, FlatArray<FVDescriptor> fvd)
{
  static Timer t("BaseStokesAMGPrecond::BuildGhostVerts");
  RegionTimer rt(t);

  /** Set ghost/solid verts **/
  auto ghost_verts = make_shared<BitArray>(fineMesh.template GetNN<NT_VERTEX>());
  ghost_verts->Clear();
  ghost_verts->Invert();

  // (local) fict verts are no ghosts
  for (auto k : Range(fvd))
    { ghost_verts->Clear(k); }

  // otherwise, everything touched by EL -> VERTEX map is solid
  for (auto elnr : Range(GetMA().GetNE())) {
    // auto delnr = nc_fes.E2DE(elnr);
    // if (delnr != size_t(-1))
    // 	{ ghost_verts->Clear(vert_sort[delnr]); }
    auto vnr = EL2V(elnr);
    if (vnr != -1)
      { ghost_verts->Clear(vnr); }
  }

  // cout << "BuildGhostVerts" << endl;
  // cout << " ghost_verts= " << ghost_verts <<endl;
  // cout << *ghost_verts << endl;

  return ghost_verts;
} // BaseStokesAMGPrecond::BuildGhostVerts


INLINE bool masterOfFacet(int const &fnr, MeshAccess const &ma)
{
  auto dps = ma.GetDistantProcs(NodeId(NT_FACET, fnr));
  return (dps.Size() == 0) || (ma.GetCommunicator().Rank() < dps[0]);
}


template<class TLAM>
INLINE FlatArray<int>
makeFacetLoop2D(int const &vnr,
                FacetAuxiliaryInformation const &auxInfo,
                MeshAccess const &ma,
                TLAM isFree,
                Array<int> &scratch,
                Array<int> &loop_facets)
{
  loop_facets.SetSize0();

  auto dps = ma.GetDistantProcs(NodeId(NT_VERTEX, vnr));

  if (dps.Size()) cout << "  makeFacetLoop2D " << vnr << endl;

  ma.GetVertexSurfaceElements(vnr, scratch);

  Array<int> facetEls;
  for (auto vsel : scratch) {
    auto selfacets = ma.GetElFacets(ElementId(BND, vsel));
    if (dps.Size()) { cout << " sel " << vsel << ", facets: "; prow(selfacets); cout << endl; }
    for (auto selfacet : selfacets)
    {
      if (dps.Size())
      {
        ma.GetFacetElements(selfacet, facetEls);
        cout << "   sel-facet-els: "; prow(facetEls); cout << endl;
      }
      if (auxInfo.IsFacetRel(selfacet))
      {
        if (isFree(selfacet) && masterOfFacet(selfacet, ma))
          { insert_into_sorted_array_nodups(selfacet, loop_facets); }
      }
    }
  }

  ma.GetVertexElements(vnr, scratch);

  for (auto elnr : scratch) {
    auto el_facets = ma.GetElFacets(ElementId(VOL, elnr));
    if (dps.Size())  { cout << " el " << elnr << ", facets: "; prow(el_facets); cout << endl; }
    for (auto fnr : el_facets) {
      if (dps.Size())
      {
        ma.GetFacetElements(fnr, facetEls);
        cout << "   el-facet-els: "; prow(facetEls); cout << endl;
      }
      if (auxInfo.IsFacetRel(fnr)) {
        auto everts = ma.GetEdgePNums(fnr);
        if ( (everts[0] == vnr) || (everts[1] == vnr) ) {
          if (isFree(fnr) && masterOfFacet(fnr, ma))
            { insert_into_sorted_array_nodups(fnr, loop_facets); }
        }
      }
    }
  }

  return ma.GetDistantProcs(NodeId(NT_VERTEX, vnr));
} // makeFacetLoop2D



template<class TLAM>
INLINE FlatArray<int>
makeFacetLoop3D(int const &mesh_enr,
                FacetAuxiliaryInformation const &auxInfo,
                MeshAccess const &ma,
                TLAM isFree,
                Array<int> &scratch,
                Array<int> &loop_facets)
{
  loop_facets.SetSize0();

  ma.GetEdgeSurfaceElements(mesh_enr, scratch);
  for (auto mesel : scratch) {
    auto selfacets = ma.GetElFacets(ElementId(BND, mesel));
    for (auto selfacet : selfacets) {
      if (auxInfo.IsFacetRel(selfacet)) {
        if (isFree(selfacet) && masterOfFacet(selfacet, ma))
          { insert_into_sorted_array_nodups(selfacet, loop_facets); }
      }
    }
  }


  ma.GetEdgeElements(mesh_enr, scratch);

  for (auto volnr : scratch) {
    auto el_facets = ma.GetElFacets(ElementId(VOL, volnr));
    for (auto fnr : el_facets) {
      if (auxInfo.IsFacetRel(fnr)) {
        if (isFree(fnr) && masterOfFacet(fnr, ma)) {
          auto facet_edges = ma.GetFaceEdges(fnr);
          for (auto e : facet_edges) {
            if (e == mesh_enr)
              { insert_into_sorted_array_nodups(fnr, loop_facets); break; }
          }
        }
      }
    }
  }

    // cout << " it_loops_3d, mesh-enr " << mesh_enr << " -> DEF " << lnr << endl;
    // cout << "   DPS "; prow2(ma.GetDistantProcs(NodeId(NT_EDGE, mesh_enr))); cout << endl;
    // cout << "   SELS "; prow2(sels); cout << endl;
    // cout << "   ELS "; prow2(els); cout << endl;
    // cout << "   facets "; prow2(loop_facets); cout << endl;
    // cout << endl;

  return ma.GetDistantProcs(NodeId(NT_EDGE, mesh_enr));
} // makeFacetLoop3D


template<class TLAM>
INLINE void iterateFacetLoops(FacetAuxiliaryInformation const &auxInfo,
                              FESpace const &fes,
                              TLAM lam)
{
  MeshAccess const &ma = *fes.GetMeshAccess();

  // Note: elint does not matter for facet-dofs
  auto fes_free = fes.GetFreeDofs();
  bool const allFree = (fes_free == nullptr);

  Array<int> facet_dofs(30);
  auto isFree = [&](auto fnr) {
    if (!allFree) {
      fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);
      return fes_free->Test(facet_dofs[0]);
    }
    return true;
  };

  Array<int> scratch(50);
  Array<int> loop_facets(50);

  if (ma.GetDimension() == 2)
  {
    for (auto r_psnum : Range(auxInfo.GetNPSN_R())) {
      auto fnr = auxInfo.R2A_PSN(r_psnum);
      auto dps = makeFacetLoop2D(fnr, auxInfo, ma, isFree, scratch, loop_facets);
      lam(r_psnum, loop_facets, dps);
    }
  }
  else
  {
    for (auto r_psnum : Range(auxInfo.GetNPSN_R())) {
      auto fnr = auxInfo.R2A_PSN(r_psnum);
      auto dps = makeFacetLoop3D(fnr, auxInfo, ma, isFree, scratch, loop_facets);
      lam(r_psnum, loop_facets, dps);
    }
  }
} // iterateFacetLoops2D


INLINE int sortTup4Loop(FlatArray<IVec<4, int>> in_loop, LocalHeap &lh)
{
  // auto flip = [](auto &tup) { tup = IVec<4, int>({ tup[2], tup[3], tup[0], tup[1] }) };
  auto flip = [](auto &tup) {
    swap(tup[0], tup[2]);
    swap(tup[1], tup[3]);
  };

  // does tupB fit to the end of (flipped) A? (Yes, Flipped, NO)
  auto orient = [](auto & tupA, bool flipA, auto & tupB) {
    int const offA = flipA ? 0 : 2;
    if ( ( tupA[offA]   == tupB[0] ) && // fits
         ( tupA[offA+1] == tupB[1] ) )
      { return 1; }
    else if ( ( tupA[offA]   == tupB[2] ) && // fits flipped
              ( tupA[offA+1] == tupB[3] ) )
      { return -1; }
    else
      { return 0; } // no fit
  };

  return SortAndOrientLoop(in_loop, lh, flip, orient);
}


template<class TIN, class T>
INLINE bool linearizeReductionData(TIN &in_data, Array<T> &out)
{
  auto nRows = in_data.Size();

  if (nRows == 0)
    { return false; }

  auto s = in_data[0].Size();
  for (auto k : Range(size_t(1), nRows))
    { s += in_data[k].Size(); }

  if (s == 0)
    { return false; }

  out.SetSize(s);
  s = 0;

  for (auto k : Range(in_data)) {
    out.Range(s, s + in_data[k].Size()) = in_data[k];
    s += in_data[k].Size();
  }

  // cout << "linearizeReductionData, in data " << endl;
  // for (auto k : Range(in_data)) {
  //   cout << k << ", s = " << in_data[k].Size() << ", data = "; prow2(in_data[k]); cout << endl;
  // }
  // cout << endl;
  // cout << "   -> linearized data: " << endl;
  // prow2(out);
  // cout << endl;

  return true;
}

template<class T, class TEX, class TOUT, class TGEN, class TENCODE, class TREDUCE, class TDECODE, class TLOCTRANS>
INLINE tuple<Table<TOUT>, Table<int>>
ProduceReducedTable(int nRows,
                    NgsAMG_Comm comm,
                    TGEN generateData,
                    TENCODE encodeData,
                    TREDUCE reduceData,
                    TDECODE decodeData,
                    TLOCTRANS localTransformData)
{
  size_t n_loc = 0;
  size_t n_ex  = 0;

  Array<int> perow(nRows);
  Array<int> perow_ex(nRows);
  Array<int> perow_dps(nRows);
  Array<int> exRows(nRows);

  perow     = 0;
  perow_ex  = 0;
  perow_dps = 0;

  for (auto row : Range(nRows)) {
    auto [data, dps] = generateData(row);
    if (dps.Size()) {
      perow_ex[n_ex]  = data.Size();
      perow_dps[row] = dps.Size();
      exRows[n_ex] = row;
      n_ex++;
    }
    else
      { perow[n_loc++] = data.Size(); }
  }

  perow_ex.SetSize(n_ex);
  exRows.SetSize(n_ex);

  // cout << " exRows = "; prow2(exRows); cout << endl;
  // cout << " perow_ex = "; prow2(perow_ex); cout << endl;

  Table<int> dps_table;
  Table<TEX> red_ex;

  size_t ex_offset = n_loc;

  if (n_ex)
  {
    // final DP-table: last n_ex entries have the collected dps
    Array<int> tmp(nRows);
    tmp = perow_dps;
    perow_dps.Range(0, nRows - n_ex)     = 0;
    perow_dps.Range(nRows - n_ex, nRows) = tmp[exRows];

    dps_table = Table<int>(perow_dps);
    Table<TEX> table_ex(perow_ex);

    for (auto exRow : Range(n_ex)) {
      auto fullRow = exRows[exRow];
      auto [data, dps] = generateData(fullRow);
      dps_table[ex_offset + exRow] = dps;
      // cout << endl << " encode " << fullRow << " -> ex " << exRow << endl;
      encodeData(data, table_ex[exRow]);
    }

    // cout << " ProduceReducedTable, dps_table = " << endl << dps_table << endl;
    // cout << " ProduceReducedTable, table_ex = " << endl << table_ex << endl;

    /**
     *  TODO: Two things that could make this better:
     *          1) overlap the localTransforms below with the exchanges
     *             Note: if size changes, does not really work when size changes because we don't have
     *                   the per-row of the recv data
     *          2) make ReduceTable call the reduce-lambda with a buffer Array<TOUT>, then we don't need
     *             to return the output and keep allocating new Array<TOU> in the reduceData lamdba
     *        The question is whether that really matters.
    */
    red_ex = ReduceTable<TEX, TEX>(table_ex, comm, [&](auto exRow) { return dps_table[ex_offset + exRow]; }, reduceData);

    // cout << " ProduceReducedTable, red_ex = " << endl << red_ex << endl;

    // reduced size can be different than put-in size, so call decode once to correctly size output!
    Array<TOUT> scratch(50);
    for (auto k : Range(n_ex)) {
      scratch.SetSize(red_ex[k].Size());
      // cout << " decode ex " << k << " -> " << ex_offset + k << endl;
      perow[ex_offset + k] = decodeData(red_ex[k], scratch);
      // cout << " -> size " << perow[ex_offset + k] << endl;
    }

    // for (auto k : Range(n_ex)) {
    //   { perow[ex_offset + k] = red_ex[k].Size(); }
  }
  else {
    dps_table = Table<int>(perow_dps);
  }

  Table<TOUT> out(perow);

  int const n_ex_total = n_ex;

  n_ex = 0;
  n_loc = 0;

  for (auto row : Range(nRows)) {
    if ( (n_ex < n_ex_total) &&
         (exRows[n_ex] == row) ) {
      // cout << endl << " decode ex " << n_ex << " -> " << ex_offset + n_ex << endl;
      // cout << "   sizes " << red_ex[n_ex].Size() << " " << out[ex_offset + n_ex].Size() << endl;
      int cnt = decodeData(red_ex[n_ex], out[ex_offset + n_ex]);
      // cout << " -> decoded " << cnt << endl;
      // cout << "    ENCODED "; prow(red_ex[n_ex]); cout << endl;
      // cout << "    DECODED "; prow(out[ex_offset + n_ex]); cout << endl;
      n_ex++;
    }
    else {
      auto [data, dps] = generateData(row);
      localTransformData(data, out[n_loc++]);
    }
  }

  return make_tuple(out, dps_table);
}


std::tuple<Table<int>, Table<int>>
BaseStokesAMGPrecond :: CalcFacetLoops (BlockTM const &blockTM, BitArray const &ghostVerts) const
{
  static Timer t("BaseStokesAMGPrecond::CalcFacetLoops");
  RegionTimer rt(t);

  LocalHeap lh(10*1024*1024, "Frodo");

  /**
   *  TODO:  In principle (esp with definedon), certain potential space
   *         nodes could have multiple "loops". 2D example:
   *           D | 0     D == defined el, 0 == not defined
   *          ---v---    vertex "v" should be 2 seperate loops (BL-TL-TR and TR-BR-BL)
   *           0 | D     BUT it probably happens very rarely...
   *          Should be pretty easy to fix if it becomes relevant I think. I just need to
   *          post-process the loops and split them if it comes to it.
   */
  std::string const weirdDefonException = "Something strange must have happened with definedon, formed loops are unable to guarantee smoother robustness!";

  auto &auxInfo = GetFacetAuxInfo();

  const auto &eqc_h = *blockTM.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();

  auto edges = blockTM.template GetNodes<NT_EDGE>();

  const auto &econ = *blockTM.GetEdgeCM();


  const auto &fes = this->GetFESpace();
  const auto &ma  = *fes.GetMeshAccess();

  if (options->log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    int const rk = eqc_h.GetCommunicator().Rank();
    std::string ofn = "ngs_amg_FL_BTM_rk_" + std::to_string(rk) + ".out";
    std::ofstream ofs(ofn);
    blockTM.printTo(ofs);
  }

  // auto produceLoops = [&](auto lam) {
  //   /**
  //    *  Gives loops as lists of mesh-facets, only adding facets
  //    *     i) that are "R"-facets
  //    *    ii) that are "free" in the FESpace
  //    *   iii) where we are the master
  //    *  That is, every added facet here has an (gg or sg) edge, and once
  //    *  gathered there are NO DUPLICATES!
  //    */
  //   if (GetMA().GetDimension() == 2)
  //     { iterateFacetLoops2D(auxInfo, this->GetFESpace(), lam); }
  //   else
  //     { iterateFacetLoops3D(auxInfo, this->GetFESpace(), lam); }
  // };

  // working arrays that need to be able to be resized dynamically
  Array<int> facet_dofs(30);
  Array<int> scratch_facets(50);
  Array<int> loop_facets(50);

  // Note: elint does not matter for facet-dofs
  int const meshDim = ma.GetDimension();

  auto fes_free = fes.GetFreeDofs();
  bool const allFree = (fes_free == nullptr);

  auto isFacetFree = [&](auto fnr) {
    if (!allFree) {
      fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);
      return fes_free->Test(facet_dofs[0]);
    }
    return true;
  };


  auto produceLoop = [&](auto PSNum) {
    auto fnr = auxInfo.R2A_PSN(PSNum);
    FlatArray<int> dps = (meshDim == 2) ?
                            makeFacetLoop2D(fnr, auxInfo, ma, isFacetFree, scratch_facets, loop_facets) :
                            makeFacetLoop3D(fnr, auxInfo, ma, isFacetFree, scratch_facets, loop_facets);

    if (dps.Size())
    {
      cout << " produce EX-Loop, PSNum " << PSNum << endl;
      cout << "   loop_facets: "; prow2(loop_facets); cout << endl;
    }

    return make_tuple(loop_facets, dps);
  };

  auto encodeLoop = [&](FlatArray<int> facetLoop, FlatArray<IVec<4, int>> tupLoop)
  {
    cout << " encodeLoop sizes " << facetLoop.Size() << " " << tupLoop.Size() << endl;
    cout << "  encode f-loop "; prow(facetLoop); cout << endl;
    for (auto j : Range(facetLoop))
    {
      auto fnr = facetLoop[j];
      auto enr = F2E(fnr); // "this" captured

      if (enr == -1)
        { throw Exception("OOPSIE, enr bad!"); }

      auto & edge = edges[enr];

      auto [eq0, lv0] = blockTM.template MapENodeToEQLNR<NT_VERTEX>(edge.v[0]);
      auto [eq1, lv1] = blockTM.template MapENodeToEQLNR<NT_VERTEX>(edge.v[1]);

      cout << "    facet " << fnr << " -> EDGE " << edge << " -> CODED "
           << eqc_h.GetEQCID(eq0) << " " << lv0 << " " << eqc_h.GetEQCID(eq1) << " " << lv1 << endl;

      tupLoop[j] = IVec<4, int>({ eqc_h.GetEQCID(eq0), lv0, eqc_h.GetEQCID(eq1), lv1 });
    }
  };

  auto reduceLoops = [&](auto &in_data)
  {
    Array<IVec<4, int>> out;

    cout << " reduceLoops, in_data: " << endl << in_data << endl;

    bool nonEmpty = linearizeReductionData(in_data, out);

    cout << " -> linearized: "; prow(out); cout << endl;

    if (nonEmpty) {
      auto nChunks = sortTup4Loop(out, lh);

      cout << " nChunks = " << nChunks << endl;
      cout << " -> SORTED: "; prow(out); cout << endl;

      if (nChunks > 1)
      {
        throw Exception(weirdDefonException);
      }
    }

    return out;
  };

  auto decodeLoop = [&](FlatArray<IVec<4, int>> tupLoop, FlatArray<int> edgeLoop) -> int
  {
    // cout << "   decodeLoop, in size = " << tupLoop.Size() << " space in out = " << edgeLoop.Size() << endl;
    int cnt = 0;
    for (auto j : Range(tupLoop))
    {
      auto &[eqid0, lv0, eqid1, lv1] = tupLoop[j];
      // auto &tup = tupLoop[j];

      // cout << "    decode " << eqid0 << " " << lv0 << " " << eqid1 << " " << lv1 << endl;

      // int eqid0 = tup[0];
      int eq0 = eqc_h.GetEQCOfID(eqid0);
      if (eq0 == -1)
        { continue; }

      // int eqid1 = tup[2];
      int eq1 = eqc_h.GetEQCOfID(eqid1);
      if (eq1 == -1)
        { continue; }

      // int lv0 = tup[1];
      int v0 = blockTM.template MapENodeFromEQC<NT_VERTEX>(lv0, eq0);

      // int lv1 =tup[3];
      int v1 = blockTM.template MapENodeFromEQC<NT_VERTEX>(lv1, eq1);

      // cout << "      eqs " << eq0 << " " << eq1 << " -> vnrs " << v0 << " " << v1 << endl;
      // cout << "      NV " << blockTM.template GetNN<NT_VERTEX>() << " " << ghostVerts.Size() << endl;

      if ( !( ghostVerts.Test(v0) && ghostVerts.Test(v1) ) ) { // do not add gg-edges. they exist somewhere else!
        // cout << "       non-ghost ! -> cnt = " << cnt+1 << endl;
        // cout << "            -> ENCODE EDGE " << int(econ(v0, v1)) << flush;
        // cout << ": " << edges[int(econ(v0, v1))] << " to " << loopEncode(v0, v1, econ(v0, v1) ) << endl;
        edgeLoop[cnt++] = loopEncode(v0, v1, econ(v0, v1));
      }
    }
    return cnt;
  };

  Array<IVec<2, int>> scratch;
  auto localTransformLoop = [&](FlatArray<int> facetLoop, FlatArray<int> edgeLoop) {
    // fnr -> enr, sort
    for (auto j : Range(facetLoop))
    {
      auto fnr = facetLoop[j];
      auto enr = F2E(fnr);

      if (enr == -1)
        { throw Exception("OOPSIE loc"); }

      edgeLoop[j] = 1 + enr;
    }

    auto flip = [](auto &code) {
      code = -code;
    };

    // does B fit to the end of (flipped) A? (Yes, Flipped, NO)
    auto orient = [&](auto & codeA, bool flipA, auto & codeB) {
      bool aFlipped = codeA < 0;
      int  enrA     = abs(codeA) - 1;
      auto &edgeA   = edges[enrA]; // edges captured

      bool bFlipped = codeB < 0;
      int  enrB     = abs(codeB) - 1;
      auto &edgeB   = edges[enrB]; // edges captured

      int endA = edgeA.v[(aFlipped^flipA) ? 0 : 1];

      // cout << " orient LOC, flipA = " << flipA << endl
      //      << "   " << codeA << " -> " << edgeA << ", flipped " << aFlipped << endl
      //      << "   " << codeB << " -> " << edgeB << endl;

      if (endA == edgeB.v[0]) // B fits
        { return 1; }
      else if (endA == edgeB.v[1]) // B fits flipped
        { return -1; }
      else // no fit
        { return 0; }
    };

    int nChunks = SortAndOrientLoop(edgeLoop, lh, flip, orient);

    if (nChunks > 1)
      { throw Exception(weirdDefonException); }
  };


  auto [loops, dps] =
    ProduceReducedTable<int, IVec<4, int>, int>
      (auxInfo.GetNPSN_R(),
       comm,
       produceLoop,
       encodeLoop,
       reduceLoops,
       decodeLoop,
       localTransformLoop);

  // cout << " final loops: " << endl << loops << endl;
  // cout << " final dps: " << endl << dps << endl;

  return make_tuple(std::move(loops), std::move(dps));
} // BaseStokesAMGPrecond::CalcFacetLoops


std::tuple<shared_ptr<BitArray>, shared_ptr<BitArray>, shared_ptr<BitArray>>
BaseStokesAMGPrecond :: BuildFreeNodes(BlockTM const &fmesh,
                                       FlatTable<int> loops,
                                       UniversalDofs const &loop_uDofs)
{

  auto comm = fmesh.GetEQCHierarchy()->GetCommunicator();

  auto &auxInfo = GetFacetAuxInfo();

  /** FreeDofs for the potential space **/

  auto fpf = make_shared<BitArray>(loops.Size());

  fpf->Clear();

  // just the empty ones are Dirichlet
  for (auto k : Range(loops))
    if (loops[k].Size()==0)
      { fpf->SetBit(k); }

  // reduce Dirichlet info
  if (comm.Size() > 2) {
    Array<int> isset(fpf->Size());
    for (auto k : Range(isset))
      { isset[k] = fpf->Test(k) ? 1 : 0; } // isset is diri, not free
    MyAllReduceDofData(*loop_uDofs.GetParallelDofs(), isset, [](auto a, auto b) { return max(a, b); }); // TODO: should this be "min" ??
    for (auto k : Range(isset)) {
      if (isset[k] != 0)
        { fpf->SetBit(k); }
    }
  }

  fpf->Invert();

  /**
   * FreeDofs for the range space (vertices):
   *   We set BND-verts to Dirichlet if their single facet is Dirichlet.
   */

  shared_ptr<BitArray> free_verts = nullptr;

  auto const &fes      = this->GetFESpace();
  auto fes_free = fes.GetFreeDofs();

  if (fes_free != nullptr) {

    // cout << " BuildFreeNodes, free_facets = " << *free_facets << endl;

    free_verts = make_shared<BitArray>(fmesh.template GetNN<NT_VERTEX>());
    free_verts->Clear();

    auto edges = fmesh.template GetNodes<NT_EDGE>();

    Array<int> facet_els(50), facet_dofs(50);

    for (auto fnr : Range(GetMA().GetNFacets())) {

      auto enr = F2E(fnr);

      if (enr != -1) {

        fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);

        if (!fes_free->Test(facet_dofs[0])) {

          GetMA().GetFacetElements(fnr, facet_els);

          const auto & edge = edges[enr];

          if (facet_els.Size()) {
            // the vertex that HAS a VOL-el, so the one that is NOT Dirichlet, but the other one is!
            auto el0vert = EL2V(facet_els[0]);
            if (edge.v[0] == el0vert)
              { free_verts->SetBit(edge.v[1]); }
            if (edge.v[1] == el0vert)
              { free_verts->SetBit(edge.v[0]); }
          }
        }

      }
    }

    // With definedon, can have shared DIRI verts...
    const auto & eqc_h = *fmesh.GetEQCHierarchy();
    if (eqc_h.GetCommunicator().Size() > 2) { // do not bother for NP 2...
      Array<int> dvs(fmesh.template GetNN<NT_VERTEX>());
      for (auto k : Range(dvs))
        { dvs[k] = free_verts->Test(k) ? 1 : 0; } // so far, its actually diri_verts, not free_verts
      fmesh.template AllreduceNodalData<NT_VERTEX>(dvs, [](auto & tab){ return std::move(max_table(tab)); }, false);
      for (auto k : Range(dvs))
        if (dvs[k] != 0)
          { free_verts->SetBit(k); }
    }

    free_verts->Invert();

    // cout << " BuildFreeNodes, free_verts = " << *free_verts << endl;
  }

  /**
   *  free edges:
   *     (i) CURRENTLY they are used only for the FIRST prolongation
   *    (ii) POTENTIALLY, they could also be used for subsequent ones, if we keep the "Dirichlet"
   *         fictitious vertices around. I am not sure whether that could be beneficial, but it
   *         could make coarse grid base functions fulfill the Dirichlet conditions in a "smoother"
   *         manner because the connection to the Dirichlet facet would then play a role in the
   *         energy minimization.
   */
  shared_ptr<BitArray> free_edges = nullptr;

  if (fes_free != nullptr)
  {
    Array<int> facet_dofs(40);

    free_edges = make_shared<BitArray>(fmesh.template GetNN<NT_EDGE>());

    free_edges->Clear(); // Clear is not split into atomic/non-atomic yet, so only use SetBit for now

    for (auto ffnr : Range(auxInfo.GetNFacets_R())) {
      auto fnr = auxInfo.R2A_Facet(ffnr);
      fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);
      if (!fes_free->Test(facet_dofs[0])) {
        auto enr = F2E(fnr);
        if (enr != -1)
          { free_edges->SetBit(enr); }
      }
    }

    // TODO: Should I not reduce this info? The free_facets ARE consistent,
    //       but the overlap should mean that I have to reduce nonetheless??

    free_edges->Invert();
  }

  return make_tuple(free_verts, free_edges, fpf);
} // BaseStokesAMGPrecond::BuildFreeNodes


void BaseStokesAMGPrecond :: FinalizeLevel (shared_ptr<BaseMatrix> mat)
{

  BaseAMGPC::FinalizeLevel(mat);

  // debugging code
  // auto ff = finest_freedofs;
  // if (ff != nullptr) {
  //   auto map = amg_mat->GetMap();
  //   auto step0 = map->GetStep(0);
  //   cout << " step0 = " << step0 << endl;
  //   auto m0 = dynamic_pointer_cast<MultiDofMapStep>(step0);
  //   cout << " m0 = " << m0 << endl;
  //   auto m00 = m0->GetMap(0);
  //   cout << " m00 = " << m00 << endl;
  //   auto ps00 = dynamic_pointer_cast<ProlMap<SparseMatrixTM<Mat<3, 3, double>>>>(m00);
  //   cout << " ps00 = " << ps00 << endl;
  //   auto prol = ps00->GetProl();
  //   cout << " prol = " << prol << endl;
  //   for (auto k : Range(prol->Height())) {
  //     if (!ff->Test(k)) {
  //       auto ris = prol->GetRowIndices(k);
  //       if (ris.Size()) {
  //         auto rvs = prol->GetRowValues(k);
  //         cout << " DOF " << k << " IS DIRI, VALS = " << endl;
  //         for (auto j : Range(ris)) {
  //           cout << "(" << ris[j] << "::" << rvs[j] << ")  ";
  //         }
  //         cout << endl;
  //       }
  //     }
  //   }
  // }

} // BaseStokesAMGPrecond::FinalizeLevel


// TODO: add to header!
extern shared_ptr<BaseDOFMapStep> MakeSingleStep3 (FlatArray<shared_ptr<BaseDOFMapStep>> init_steps);


Array<shared_ptr<BaseSmoother>>
BaseStokesAMGPrecond::
BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> aMGLevels,
                shared_ptr<DOFMap> dOFMap)
{
  const auto &O = static_cast<Options&>(*options);

  auto gcomm = dOFMap->GetUDofs().GetCommunicator();

  // return BaseAMGPC::BuildSmoothers(amg_levels, dof_map);
  if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
    { cout << " set up smoothers " << endl; }

  // cout << endl << "BUILD SMOOTHERS!" << endl;

  auto const numGlobalLevels = aMGLevels.Size(); // TODO: MPI!

  Array<shared_ptr<BaseSmoother>> primarySmoothers(aMGLevels.Size() - 1);

  SmootherContext globOuter({ SmootherContext::GLOBAL, SmootherContext::OUTER });

  for (int k = 0; k + 1 < aMGLevels.Size(); k++)
  {
    // cout << endl << " BuildSmoothers, primary " << k << endl;
    primarySmoothers[k] = BuildSmoother(*aMGLevels[k], { SmootherContext::PRIMARY, SmootherContext::OUTER });
    // cout << " BuildSmoothers, primary " << k
    //      << " -> " << typeid(*primarySmoothers[k]).name() << endl;

    if (O.test_smoothers)
    {
      NgMPI_Comm comm = aMGLevels[k]->cap->uDofs.GetCommunicator();

      TestSmoother(primarySmoothers[k]->GetAMatrix(),
                   primarySmoothers[k],
                   comm,
                   string("\n Test primary space smoother on level " + to_string(k)));
    }
  }

  // this has to happen here, AFTER we have all the matrices
  OptimizeDOFMap(aMGLevels, dOFMap);

  bool needSecSequence = false;

  for (auto const &level : aMGLevels.Range(0, aMGLevels.Size() - 1))
  {
    // UniversalDofs uDofs = dof_map->GetUDofs(k);
    // if ( (k > 0) && O.regularize_cmats) // Regularize coarse level matrices
    //   { RegularizeMatrix(amg_levels[k]->cap->mat, uDofs.GetParallelDofs()); }

    if (SelectSmoother(*level, globOuter) == Options::SM_TYPE::AMGSM)
    {
      needSecSequence = true;
      break;
    }
  }

  // cout << " needSecSequence = " << needSecSequence << endl;
  // cout << endl << endl;

  Array<shared_ptr<BaseAMGFactory::AMGLevel>> secLevels;
  Array<shared_ptr<BaseDOFMapStep>>           secEmbs;
  shared_ptr<DOFMap>                          secMap;

  Array<shared_ptr<BaseSmoother>> secSmoothers(aMGLevels.Size());

  shared_ptr<BaseMatrix> secCMat;
  shared_ptr<BaseMatrix> secCInv;

  if (needSecSequence)
  {
    // TODO: build secondary sequence starting from level "k" where k is the first level we are using AMG-as-smoother
    auto [aSecLevels, aSecEmbs, aSecMap] = CreateSecondaryAMGSequence(aMGLevels, *dOFMap);
    secLevels = aSecLevels;
    secEmbs   = aSecEmbs;
    secMap    = aSecMap;

    auto [aSecCMat, aSecCInv] = this->CoarseLevelInv(*secLevels.Last());

    secCMat = aSecCMat;
    secCInv = aSecCInv;
  }

  Array<shared_ptr<BaseSmoother>> finalSmoothers(aMGLevels.Size() - 1);

  for (int k = 0; k + 1 < aMGLevels.Size(); k++)
  {
    auto const &aMGLevel = *aMGLevels[k];

    auto smType = SelectSmoother(aMGLevel, globOuter);

    if ( ( smType != Options::SM_TYPE::AMGSM ) ||
         ( k == numGlobalLevels - 2          ) ) // only 1 more AMG-level, no point in sec-sequence!
    {
      // cout << " level " << k << " -> use primarySmoother!" << endl;
      finalSmoothers[k] = primarySmoothers[k];
    }
    else
    {
      /**
       *  Note: The "secondary" matrix on level "k+j" is the Galerkin-projection of the
       *        level "k+j" "primary" matrix to the low order space. This is NOT necessarily the same
       *        as the Galerkin projection of the level "k" secondary matrix to secondary level
       *        "k+j"!
       *        It IS the same for HDiv, where E_k^T E_k = Id, and the low order space is EXACTLY
       *        the subspace of functions that have a divergence (i.e. all besides the first base function
       *        for each edge has zero flow).
       */
      // cout << " BUILD HDIV AMG-SMOOTHER for level " << k << "! " << endl;

      // cout << " make sure we have secondary smoothers [" << k+1 << ", " << aMGLevels.Size() - 1 << ")" << endl;
      for (int j = k + 1; j + 1 < aMGLevels.Size(); j++)
      {
        if (secSmoothers[j] == nullptr)
        {
          // cout << " BuildSmoother sec outer j = " << j << endl;
          secSmoothers[j] = BuildSmoother(*secLevels[j], { SmootherContext::SECONDARY, SmootherContext::OUTER });
          // cout << " BuildSmoother sec outer j = " << j << " - OK!" << endl;

          if (O.test_smoothers)
          {
            NgMPI_Comm comm = secLevels[j]->cap->uDofs.GetCommunicator();

            TestSmoother(secSmoothers[j]->GetAMatrix(),
                         secSmoothers[j],
                         comm,
                         string("\n Test secondary space smoother on level " + to_string(j)));

          }
        }
      }

      Array<shared_ptr<BaseSmoother>> subSmoothers;
      auto smootherDM = make_shared<DOFMap>();

      // "full" primary smoother on level "k"
      subSmoothers.Append(primarySmoothers[k]);

      // map from primary level "k" -> secondary level "k+1"
      Array<shared_ptr<BaseDOFMapStep>> steps;
      if ( (k == 0) && (aMGLevels[0]->embed_map != nullptr) )
      {
        steps.Append(aMGLevels[0]->embed_map);
      }
      steps.Append(secEmbs[k]);
      steps.Append(secMap->GetStep(k));

      // if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
      // {
      //   cout << " FIRST AMG-SM step sub-steps:" << endl;
      //   for (auto l : Range(steps))
      //   {
      //     auto step = steps[l];
      //     cout << " step " << l << "/" << steps.Size() << ": ";
      //     cout << step->GetUDofs().GetND() << " -> " << step->GetMappedUDofs().GetND() << endl;
      //   }
      // }

      auto embPlusCrs = MakeSingleStep3(steps);

      embPlusCrs->Finalize();

      // if ( k == 0 )
      // {
      //   cout << " embPlusCrs SUB-STEPS: " << endl;

      //   for (int l = 0; l < steps.Size(); l++)
      //   {
      //     cout << " SUB-STEP " << l << "/" << steps.Size() << ": " << endl;
      //     cout << *steps[l] << endl << endl;
      //   }
      //   cout << " embPlusCrs : " << endl;
      //   cout << *embPlusCrs << endl;
      //   cout << endl;
      // }

      smootherDM->AddStep(embPlusCrs);

      for (auto j = k + 1; j + 1 < aMGLevels.Size(); j++)
      {
        // secondary smoothing on level "j"
        subSmoothers.Append(secSmoothers[j]);
        // secondary map j -> j + 1
        smootherDM->AddStep(secMap->GetStep(j));
      }

      smootherDM->Finalize();

      // if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
      // {
      //   cout << " MATRICES FROM AMG-SMOOTHER, L " << k << endl;

      //   shared_ptr<BaseMatrix> projMat = primarySmoothers[k]->GetAMatrix();

      //   int c = 0;
      //   for (auto j = k + 1; j + 1 < aMGLevels.Size(); j++)
      //   {
      //     cout << "    SEC-MAT  " << j << " = " << endl << *secSmoothers[j]->GetAMatrix() << endl;
      //     cout << endl << endl;

      //     auto step = smootherDM->GetStep(c);
      //     c++;

      //     projMat = step->AssembleMatrix(GetLocalSparseMat(projMat));

      //     cout << "    PROJ-MAT " << k << " -> " << j << " = " << endl << *projMat << endl;
      //   }
      // }

      // if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
      // {
      //   cout << " SMOOTHER-STEPS AMG-SM level " << k << endl;
      //   for (auto l : Range(smootherDM->GetNSteps()))
      //   {
      //     auto step = smootherDM->GetStep(l);
      //     cout << "   step " << l << "/" << smootherDM->GetNSteps() << ": ";
      //     cout << step->GetUDofs().GetND() << " -> " << step->GetMappedUDofs().GetND() << endl;
      //   }
      // }

      auto smootherAMGMat = make_shared<AMGMatrix>(smootherDM, subSmoothers);

      // {
      //   cout << endl << endl;
      //   DoTest(*primarySmoothers[k]->GetAMatrix(), *smootherAMGMat, gcomm, string("\n test AMG-SM as PC (NO INV) on level " + to_string(k)));
      //   cout << endl << endl;
      // }

      smootherAMGMat->SetCoarseInv(secCInv, secLevels.Last()->cap->mat);

      // {
      //   cout << endl << endl;
      //   DoTest(*primarySmoothers[k]->GetAMatrix(), *smootherAMGMat, gcomm, string("\n test AMG-SM as PC (W. INV) on level " + to_string(k)));
      //   cout << endl << endl;
      // }

      // only 1 fine level smoothing step for AMGSmoother
      auto const singleFLS = O.amg_sm_single_fls.GetOpt(k);

      finalSmoothers[k] = make_shared<AMGSmoother>(smootherAMGMat, 0, singleFLS);

      finalSmoothers[k]->Finalize();
    }
  }

  if (gcomm.Rank() == 0 && O.log_level_pc > Options::LOG_LEVEL_PC::NONE)
    { cout << " smoothers built" << endl; }

  if ( O.test_smoothers )
  {
    for (int k = 0; k < aMGLevels.Size() - 1; k++)
    {
      UniversalDofs uDofs = (k == 0) ? MatToUniversalDofs(*finest_mat, DOF_SPACE::COLS)
                                     : aMGLevels[k]->cap->uDofs;

      auto smt = WrapParallelMatrix( ( k == 0 ) ? GetLocalMat(finest_mat)
                                                : aMGLevels[k]->cap->mat,
                                    uDofs,
                                    uDofs,
                                    C2D);

      // cout << " TYPE FINAL " << k << " = " << typeid(*finalSmoothers[k]).name() << endl;

      TestSmoother(finalSmoothers[k]->GetAMatrix(),
                   finalSmoothers[k],
                   gcomm,
                   string("\n Test full, final smoother (I)  on level " + to_string(k)));
      // TestSmoother(smoothers[k]->GetAMatrix(), smoothers[k], gcomm, string("\n Test full, final smoother (II) on level " + to_string(k)));
    }
  }

  return finalSmoothers;
} // BaseStokesAMGPrecond::BuildSmoothers


void
BaseStokesAMGPrecond::
OptimizeDOFMap (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> aMGLevels,
                shared_ptr<DOFMap> dOFMap)
{
  ;
}


shared_ptr<BaseSmoother>
BaseStokesAMGPrecond::
BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level)
{
  throw Exception("called BaseStokesAMGPrecond::BuildSmoother - should never get here!");
  return nullptr;
} // BaseStokesAMGPrecond::BuildSmoother


shared_ptr<BaseSmoother>
BaseStokesAMGPrecond::
BuildSmoother (BaseAMGFactory::AMGLevel const &aMGLevel,
               SmootherContext          const &smootherCtx)
{
  auto const &O = static_cast<Options&>(*options);

  auto const level = aMGLevel.level;

  auto const smType  = SelectSmoother(aMGLevel, smootherCtx);

  std::cout << "  BuildSmoother " << smType << " @" << smootherCtx << ", l = " << aMGLevel.level << endl;

  bool const hiptmair = smType == Options::SM_TYPE::HIPTMAIR;

  bool const isFinest = ( level == 0 ) && ( smootherCtx.scope == SmootherContext::PRIMARY );

  cout << " for level " << level << endl;
  cout << " isFinest = " << isFinest << endl;

  /** Range Smoother */
  auto const smTypeR = hiptmair ?
                          SelectSmoother(aMGLevel, { smootherCtx.scope, SmootherContext::RANGE } ) :
                          smType;

  std::cout << "    BuildSmoother, RAN: " << smTypeR << " @" << smootherCtx << ", l = " << aMGLevel.level << endl;

  shared_ptr<BaseMatrix> rangeMat;
  shared_ptr<BitArray>   rangeFreeDOFs;

  if (isFinest)
  {
    rangeMat      = finest_mat;
    rangeFreeDOFs = finest_freedofs;
  }
  else
  {
    rangeMat = WrapParallelMatrix(aMGLevel.cap->mat,
                                  aMGLevel.cap->uDofs,
                                  aMGLevel.cap->uDofs,
                                  PARALLEL_OP::C2D);

    rangeFreeDOFs = GetFreeDofs(aMGLevel);
  }

  auto rangeSmoother = BuildSmootherConcrete(smTypeR,
                                             rangeMat,
                                             rangeFreeDOFs,
                                             [&](){ return GetGSBlocks(aMGLevel); });

  if (hiptmair)
  {

    if ( O.test_smoothers )
    {
      NgMPI_Comm comm = aMGLevel.cap->uDofs.GetCommunicator();

      TestSmoother(rangeMat, rangeSmoother, comm, string("\n Test partial smoother on level " + to_string(level)));

      // if (comm.Size() > 1) // get a multadd error sequentially because wrong vec type somewhere
      //   { DoTest(*rangeMat, *rangeSmoother, comm, string("\n Test (add) partial smoother on level " + to_string(level))); }
    }

    if (O.log_level_pc >= Options::LOG_LEVEL_PC::DBG)
    {
      NgMPI_Comm comm = aMGLevel.cap->uDofs.GetCommunicator();

      std::string const fileName = "stokes_rsm_rk_" + to_string(comm.Rank())
                                                    + "_l_" + to_string(aMGLevel.level)
                                                    + ".out";

      ofstream out(fileName);

      rangeSmoother->PrintTo(out);
    }

    auto const &cap = static_cast<BaseStokesLevelCapsule const &>(*aMGLevel.cap);

    auto potMat = WrapParallelMatrix(cap.pot_mat,
                                     cap.pot_uDofs,
                                     cap.pot_uDofs,
                                     PARALLEL_OP::C2D);

    shared_ptr<BaseMatrix> curlMat;
    shared_ptr<BaseMatrix> curlTMat;

    if (isFinest)
    {
      auto embMultiStep = my_dynamic_pointer_cast<MultiDofMapStep>(aMGLevel.embed_map,
                                                                   "BaseStokesAMGPrecond::BuildSmoother - finset Multi-Step!");

      // DOF-step that goes potSpace->meshCanonic->finestLevel
      auto embCurlStep = my_dynamic_pointer_cast<BaseProlMap>(embMultiStep->GetMap(1),
                                                              "BaseStokesAMGPrecond::BuildSmoother - finset Multi-Step!");

      curlMat = WrapParallelMatrix(embCurlStep->GetBaseProl(),
                                   cap.pot_uDofs,
                                   embMultiStep->GetUDofs(),
                                   PARALLEL_OP::C2C);

      curlTMat = WrapParallelMatrix(embCurlStep->GetBaseProlTrans(),
                                    embMultiStep->GetUDofs(),
                                    cap.pot_uDofs,
                                    PARALLEL_OP::D2D);
    }
    else
    {
      curlMat = WrapParallelMatrix(cap.curl_mat,
                                   cap.pot_uDofs,
                                   cap.uDofs,
                                   PARALLEL_OP::C2C);

      curlTMat = WrapParallelMatrix(cap.curl_mat_T,
                                    cap.uDofs,
                                    cap.pot_uDofs,
                                    PARALLEL_OP::D2D);
    }

    auto const smTypeP = SelectSmoother(aMGLevel, { smootherCtx.scope, SmootherContext::POT });

    // std::cout << "    BuildSmoother, POT: " << smTypeP << " @" << smootherCtx << ", l = " << aMGLevel.level << endl;

    auto potSmoother = BuildSmootherConcrete(smTypeP,
                                             potMat,
                                             cap.pot_freedofs,
                                             [&](){ return GetPotentialSpaceGSBlocks(aMGLevel); });

    if ( O.test_smoothers )
    {
      NgMPI_Comm comm = aMGLevel.cap->uDofs.GetCommunicator();

      TestSmoother(potMat, potSmoother, comm, string("\n Test potential space smoother on level " + to_string(level)));

      // if (comm.Size() > 1) // get a multadd error sequentially because wrong vec type somewhere
      //   { DoTest(*rangeMat, *rangeSmoother, comm, string("\n Test (add) partial smoother on level " + to_string(level))); }

    }

    if (O.log_level_pc >= Options::LOG_LEVEL_PC::DBG)
    {
      NgMPI_Comm comm = aMGLevel.cap->uDofs.GetCommunicator();

      std::string const fileName = "stokes_psm_rk_" + to_string(comm.Rank())
                                                    + "_l_" + to_string(aMGLevel.level)
                                                    + ".out";

      ofstream out(fileName);

      potSmoother->PrintTo(out);
    }

    // whether Hiptmair Smoother builds range_mat*curl_mat explicitly (for performance, especially HDiv!)
    bool const adOpt = true;

    // Note: the rangeMat is given only for adOpt, it actually uses rangeSmoother->GetAMatrix() as sys-mat!
    auto hSmoother = make_shared<HiptMairSmoother>(potSmoother,
                                                   rangeSmoother,
                                                   potSmoother->GetAMatrix(),
                                                   rangeMat,
                                                   curlMat,
                                                   curlTMat,
                                                   cap.AC,   // A * C
                                                   nullptr); // CT * A -> transpose dyn in smoother

    cout << " hSmoother A-matrix type: " << typeid(*hSmoother->GetAMatrix()).name() << endl;

    if (O.log_level_pc >= Options::LOG_LEVEL_PC::DBG)
    {
      NgMPI_Comm comm = aMGLevel.cap->uDofs.GetCommunicator();

      std::string const fileName = "stokes_hpt_sm_rk_" + to_string(comm.Rank())
                                                       + "_l_" + to_string(aMGLevel.level)
                                                       + ".out";

      ofstream out(fileName);

      hSmoother->PrintTo(out);
    }

    return hSmoother;
  }
  else
  {
    return rangeSmoother;
  }
} // BaseStokesAMGPrecond::BuildSmoother


shared_ptr<BaseSmoother>
BaseStokesAMGPrecond::
BuildSmootherConcrete (BaseAMGPC::Options::SM_TYPE const &smType,
                       shared_ptr<BaseMatrix>             A,
                       shared_ptr<BitArray>               freeDofs,
                       std::function<Table<int>()>        getBlocks)
{
  switch(smType)
  {
    case(Options::SM_TYPE::BGS):    { return BuildBGSSmoother(A, getBlocks()); }
    case(Options::SM_TYPE::JACOBI): { return BuildJacobiSmoother(A, freeDofs); }
    case(Options::SM_TYPE::DYNBGS): { return BuildDynamicBlockGSSmoother(A, freeDofs); }
    default:                        { return BuildGSSmoother(A, freeDofs); }
  }
} // BaseStokesAMGPrecond::BuildSmootherConcrete

BaseAMGPC::Options::SM_TYPE
BaseStokesAMGPrecond::
SelectSmoother(BaseAMGFactory::AMGLevel const &amgLevel) const
{
  throw Exception("Called BaseStokesAMGPrecond::SelectSmoother - should not get here!");

  return Options::SM_TYPE::GS;
} // BaseStokesAMGPrecond::SelectSmoother


BaseAMGPC::Options::SM_TYPE
BaseStokesAMGPrecond::
SelectSmoother(BaseAMGFactory::AMGLevel const &aMGLevel,
               SmootherContext          const &smootherCtx) const
{
  auto const &O = static_cast<Options&>(*options);

  int const level = aMGLevel.level;

  // cout << "      SelectSmoother, level " << aMGLevel.level << ", ctx " << smootherCtx << std::endl;

  // TODO: block-smoother gating for NC/HDIV separately !
  // TODO: pot-space block-smoother might be broken?

  switch(smootherCtx.space)
  {
    case(SmootherContext::RANGE):
    {
      if (smootherCtx.scope == SmootherContext::SECONDARY)
      {
        return Options::SM_TYPE::GS;
      }
      else
      {
        return O.sm_type_range.GetOpt(level);
      }
      break;
    }
    case(SmootherContext::POT):
    {
      if (smootherCtx.scope == SmootherContext::SECONDARY)
      {
        return Options::SM_TYPE::GS;
      }
      else
      {
        return O.sm_type_pot.GetOpt(level);
      }
      break;
    }
    case(SmootherContext::OUTER):
    {
      Options::SM_TYPE globalSMType = O.sm_type.GetOpt(level);

      if (smootherCtx.scope == SmootherContext::GLOBAL)
      {
        return globalSMType;
      }
      else if ( ( smootherCtx.scope == SmootherContext::PRIMARY ) &&
                ( globalSMType != Options::SM_TYPE::AMGSM ) )
      {
        // when not inside AMG-as-smoother, the "primary" smoother is the global one
        return globalSMType;
      }
      else
      {
        // primary/secondary smoother inside AMG-as-smoother -> hard-coded to hiptmair
        return Options::SM_TYPE::HIPTMAIR;
      }

      break;
    }
  }
} // BaseStokesAMGPrecond::SelectSmoother


std::tuple<Array<shared_ptr<BaseAMGFactory::AMGLevel>>,
           Array<shared_ptr<BaseDOFMapStep>>,
           shared_ptr<DOFMap>>
BaseStokesAMGPrecond::
CreateSecondaryAMGSequence(FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>>        aMGLevels,
                           DOFMap                                          const &dOFMap) const
{
  static Timer t("CreateSecondaryAMGSequence");
  RegionTimer rt(t);

  auto secDOFMap = make_shared<DOFMap>();

  Array<shared_ptr<BaseAMGFactory::AMGLevel>> secLevels(aMGLevels.Size());
  Array<shared_ptr<BaseDOFMapStep>>           secEmbs(aMGLevels.Size());

  // cout << endl << "CreateSecondaryAMGSequence!" << endl;

  auto castCap = [&](auto const &cap) -> BaseStokesLevelCapsule const &
  {
    return *my_dynamic_cast<BaseStokesLevelCapsule const>(&cap,
              "CreateSecondaryAMGSequence - cap");
  };


  if (aMGLevels.Size())
  {
    // cout << "  L0 CreateRTZEmbedding!" << endl;
    secLevels[0]   = nullptr;
    secEmbs[0]     = GetSecondaryAMGSequenceFactory().CreateRTZEmbedding(castCap(*aMGLevels[0]->cap));
    // cout << "  L0 CreateRTZEmbedding - OK!" << endl;
  }

  for (int k = 1; k < aMGLevels.Size(); k++)
  {
    auto const &cap = static_cast<BaseStokesLevelCapsule&>(*aMGLevels[k]->cap);

    // cout << "  L" << k << " CreateRTZLevel!" << endl;
    auto [embStep, secCap] = GetSecondaryAMGSequenceFactory().CreateRTZLevel(cap);
    // cout << "  L" << k << " CreateRTZLevel - OK!" << endl;

    auto secLevel = make_shared<BaseAMGFactory::AMGLevel>();
    secLevel->cap   = secCap;
    secLevel->level = secCap->baselevel;

    secLevels[k]   = secLevel;
    secEmbs[k]     = embStep;
  }

  for (int k = 0; k + 1 < aMGLevels.Size(); k++)
  {
    // cout << "  L" << k << " CreateRTZDOFMap!" << endl;
    // cout << "     dOFMap.GetStep(k) = " << dOFMap.GetStep(k) << endl;
    auto step = GetSecondaryAMGSequenceFactory().CreateRTZDOFMap(castCap(*aMGLevels[k]->cap),
                                                                 *secEmbs[k],
                                                                 castCap(*aMGLevels[k+1]->cap),
                                                                 *secEmbs[k+1],
                                                                 *dOFMap.GetStep(k));
    // cout << "  L" << k << " CreateRTZDOFMap - OK!" << endl;
    secDOFMap->AddStep(step);
  }

  // cout << endl << "CreateSecondaryAMGSequence - OK!" << endl;
  return std::make_tuple(secLevels, secEmbs, secDOFMap);
} // StokesAMGPrecond::CreateSecondaryAMGSequence

/** END BaseStokesAMGPrecond **/


} // namespace amg
