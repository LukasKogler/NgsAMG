#ifndef FILE_STOKES_PC_IMPL_HPP
#define FILE_STOKES_PC_IMPL_HPP

#include "stokes_pc.hpp"

namespace amg
{

/** BaseStokesAMGPrecond **/

template<class TLAM>
void BaseStokesAMGPrecond :: CalcVertexVols (BlockTM &fineMesh, FlatArray<FVDescriptor> fvd, TLAM set_vol)
{
  auto comm = GetMA().GetCommunicator();
  auto &auxInfo = GetFacetAuxInfo();

  // fictitious vertices; set volume to -(1 + bnd_id)
  for (auto k : Range(fvd))
    { set_vol(k, -1 - fvd[k].code); }

  // Compute an ID for UNDEF els such that it does not conflict with any bnd_ids anywhere
  // For that I take the max. over all appearing codes and add 1
  int MAX_CODE = 0; // MINIMAL index used globally for exterior fict vertices
  for (auto k : Range(fvd))
    { MAX_CODE = max2(MAX_CODE, fvd[k].code); }
  MAX_CODE = comm.AllReduce(MAX_CODE, MPI_MAX);

  int UNDEF_VOL = -2 - MAX_CODE;

  // active/inactive vertices
  for (auto elnr : Range(GetMA().GetNE())) {
    auto vnum = EL2V(elnr);
    if (vnum != -1) {
      auto r_elnr = auxInfo.A2R_EL(elnr);
      if (r_elnr == -1) { // inactive
        set_vol(vnum, UNDEF_VOL);
      }
      else { // active
        set_vol(vnum, GetMA().ElementVolume(elnr));
      }
    }
  }

} // BaseStokesAMGPrecond::CalcVertexVols


void BaseStokesAMGPrecond :: GetDefinedFacetElements (int facnr, Array<int> & elnums) const
{
  auto const &fes = this->GetFESpace();
  int c0 = 0;

  fes.GetMeshAccess()->GetFacetElements(facnr, elnums);
  for (auto j : Range(elnums))
    if (fes.DefinedOn(ElementId(VOL, elnums[j])))
      { elnums[c0++] = elnums[j]; }
  elnums.SetSize(c0);
} // BaseStokesAMGPrecond::GetDefinedFacetElements

/** END BaseStokesAMGPrecond **/


/** StokesAMGPrecond **/

template<class TMESH>
StokesAMGPrecond<TMESH> :: StokesAMGPrecond (shared_ptr<BilinearForm>        blf,
                                             Flags                    const &flags,
                                             string                   const &name,
                                             shared_ptr<Options>             opts)
  : BaseStokesAMGPrecond(blf, flags, name, opts)
{
  ;
} // StokesAMGPrecond(..)


template<class TMESH>
StokesAMGPrecond<TMESH> :: StokesAMGPrecond (shared_ptr<FESpace>           fes,
                                             Flags                        &flags,
                                             string                 const &name,
                                             shared_ptr<Options>           opts)
  : BaseStokesAMGPrecond(fes, flags, name, opts)
{
  ;
} // StokesAMGPrecond(..)


template<class TMESH>
shared_ptr<TMESH> StokesAMGPrecond<TMESH> :: BuildEmptyAlgMesh (shared_ptr<BlockTM> &&aBlockTM, FlatArray<FVDescriptor> fvd)
{
  static Timer t("StokesAMGPrecond::BuildEmptyAlgMesh");
  RegionTimer rt(t);

  auto blockTM = aBlockTM;

  /** ghost vertices **/
  auto ghost_verts = this->BuildGhostVerts(*blockTM, fvd);

  // this is how I got the types before
  // typedef typename std::remove_pointer<typename std::tuple_element<0, typename TMESH::TTUPLE>::type>::type ATVD;
  // typedef typename ATVD::TDATA TVD;
  // Array<TVD> vdata;
  // typedef typename std::remove_pointer<typename std::tuple_element<1, typename TMESH::TTUPLE>::type>::type ATED;
  // typedef typename ATED::TDATA TED;
  // Array<TED> edata;

  /** vertex data w. volumes **/
  typedef typename std::remove_pointer<typename std::tuple_element<0, typename TMESH::TTUPLE>::type>::type TATTATCHED_V;
  typedef typename TATTATCHED_V::TDATA TVD;

  // Array<TVD> vdata(blockTM->template GetNN<NT_VERTEX>());
  // vdata = 0.0;

  TATTATCHED_V* attached_vdata = new TATTATCHED_V(Array<TVD>(blockTM->template GetNN<NT_VERTEX>()), DISTRIBUTED);

  auto vdata = attached_vdata->Data();

  vdata = 0.0;

  this->CalcVertexVols(*blockTM, fvd, [&](auto k, auto vol) { vdata[k].vol = vol; });

  /** edge data w. flows **/
  typedef typename std::remove_pointer<typename std::tuple_element<1, typename TMESH::TTUPLE>::type>::type TATTACHED_E;
  typedef typename TATTACHED_E::TDATA TED;

  // Array<TED> edata(blockTM->template GetNN<NT_EDGE>());
  // edata = 0.0;

  TATTACHED_E* attached_edata = new TATTACHED_E (Array<TED>(blockTM->template GetNN<NT_EDGE>()), DISTRIBUTED);

  auto edata = attached_edata->Data();

  edata = 0.0;

  /** construct final mesh! **/
  // auto algMesh = make_shared<TMESH>(std::move(*blockTM), make_tuple(attached_vdata, attached_edata));
  auto algMesh = make_shared<TMESH>(std::move(*blockTM), make_tuple(attached_vdata, attached_edata));

  algMesh->SetGhostVerts(ghost_verts);

  /** set up loops **/
  this->CalcLoops(*algMesh);

  return algMesh;
} // StokesAMGPrecond::BuildEmptyAlgMesh


template<class TMESH>
Table<int> StokesAMGPrecond<TMESH> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
{
  // const auto & FM = static_cast<TMESH&>(*amg_level.cap->mesh);
  const auto & FM = static_cast<TMESH&>(*amg_level.crs_map->GetMesh());
  auto loops = FM.GetLoops();

  auto const &fes = this->GetFESpace();

  Table<int> gs_blocks;

  if (amg_level.level > 0) {
    auto [dofed_edges, dof2e, e2dof] = FM.GetDOFedEdges();
    Array<int> perow(loops.Size());
    for (auto k : Range(perow))
      { perow[k] = loops[k].Size(); }
    Table<int> gs_blocks(perow);
    for (auto k : Range(perow)) {
      auto lk = loops[k];
      for (auto j : Range(lk)) // I think only dofed edges ?
        { gs_blocks[k][j] = e2dof[abs(lk[j])-1]; }
    }
  }
  else {
    TableCreator<int> cBlocks(loops.Size());
    Array<int> facet_dofs(50);
    Array<int> loop_dofs(50);
    for (; !cBlocks.Done(); cBlocks++) {
      for (auto kloop : Range(loops)) {
        auto loop = loops[kloop];
        for (auto j : Range(loop)) {
          auto enr = abs(loop[j]) - 1;
          throw Exception(" EDGE -> FACET MAP NEEDED!");
          // auto fnr = EDGE_TO_FACET(enr); // TODO:
          int fnr = -1;
          fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);
          for (auto l : Range(facet_dofs)) {
            if ( ( finest_freedofs == nullptr ) || ( finest_freedofs->Test(facet_dofs[l]) ) ) {
              insert_into_sorted_array_nodups(facet_dofs[l], loop_dofs);
            }
          }
        }
        cBlocks.Add(kloop, loop_dofs);
      }
    }
  }

  return std::move(gs_blocks);
} // StokesAMGPrecond::GetGSBlocks

template<class TMESH>
Table<int> StokesAMGPrecond<TMESH> :: GetPotentialSpaceGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
{
  const auto & FM = static_cast<TMESH&>(*amg_level.crs_map->GetMesh());

  return FM.LoopBlocks(*amg_level.crs_map);
} // StokesAMGPrecond::GetPotentialSpaceGSBlocks

template<class TMESH>
void StokesAMGPrecond<TMESH> :: CalcLoops (TMESH &fMesh) const
{
  auto &auxInfo = GetFacetAuxInfo();

  const auto &ghost_verts = *fMesh.GetGhostVerts();

  /**
   *  Loops as sorted & oriented list of (active) facets (all-facet numbering).
   *  For shared loops, can contain its local part as either one or multiple chunks.
   *  (I think) shared facets are duplicated.
   */
  auto [edge_loops, loop_dps] = this->CalcFacetLoops(fMesh, ghost_verts);

  // cout << " edge_loops = " << endl << edge_loops << endl;
  // cout << " loop_dps = " << endl << loop_dps << endl;

  fMesh.SetLoops(std::move(edge_loops));
  fMesh.SetLoopDPs(std::move(loop_dps));

} // StokesAMGPrecond::CalcLoops


/** END StokesAMGPrecond **/

} // namespace amg

#endif // FILE_STOKES_PC_IMPL_HPP