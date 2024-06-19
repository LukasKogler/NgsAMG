#ifndef FILE_STOKES_MESH_IMPL_HPP
#define FILE_STOKES_MESH_IMPL_HPP

#include <base_coarse.hpp>
#include <utils_io.hpp>

#include "stokes_mesh.hpp"
#include "loop_utils.hpp"

namespace amg
{


template<class... T> template<class TLAM>
INLINE void StokesMesh<T...> :: AddOrientedLoops (FlatTable<int> init_loops, FlatTable<int> dist_loopers, TLAM set_lsort)
{
  // cout << " AddOrientedLoops, init_loops = " << endl << init_loops << endl;
  // cout << " AddOrientedLoops, dist_loopers = " << endl << dist_loopers << endl;

  const auto & eqc_h = *this->GetEQCHierarchy(); // in RANGE space !!
  auto comm = eqc_h.GetCommunicator();
  const auto & econ = *this->GetEdgeCM();
  auto edges = this->template GetNodes<NT_EDGE>();

  /** We can locally say whether a loop will be removed **/

  BitArray remove_loop(init_loops.Size());

  remove_loop.Clear();
  for (auto k : Range(init_loops)) {
    if ( init_loops[k].Size() == 0 )
      { remove_loop.SetBit(k); }
  }

  // ?? not sure about this tbh, probably something about overlap??
  Array<int> all_loop_dps(eqc_h.GetDistantProcs().Size());

  all_loop_dps = eqc_h.GetDistantProcs();
  for (auto k : Range(dist_loopers)) {
    for (auto p : dist_loopers[k])
      { insert_into_sorted_array_nodups(p, all_loop_dps); }
  }

  // cout << " remove_loop " << endl << remove_loop << endl;

  /** Remove loopers where loops are locally disabled from the loop communication tables **/

  auto [ filt_loopers, filt_map ] = ShrinkDPTable(remove_loop, dist_loopers, all_loop_dps, eqc_h.GetCommunicator());

  // cout << " filtered loopers " << endl << filt_loopers << endl;

  Array<int> perow(filt_loopers.Size());
  for (auto k : Range(filt_map)) {
    set_lsort(k, filt_map[k]);
    if ( filt_map[k] != -1 )
      { perow[filt_map[k]] = init_loops[k].Size(); }
  }
  Table<int> filt_loops(perow);
  for (auto k : Range(filt_map))
    if ( filt_map[k] != -1 )
      { filt_loops[filt_map[k]] = init_loops[k]; }

  SetLoops(std::move(filt_loops));
  SetLoopDPs(std::move(filt_loopers));

  // TestParDofsConsistency(*GetLoopParDofs(), "AddOrientedLoops, CLOOP_DPS");
} // StokesMesh::AddOrientedLoops


template<class... T>
Table<int> StokesMesh<T...> :: LoopBlocks (const BaseCoarseMap & cmap) const
{
  // TODO: should check c2f edge, if those are simple, dont do anything
  auto & fmesh = *this;
  fmesh.CumulateData();

  auto & cmesh = *static_pointer_cast<StokesMesh<T...>>(cmap.GetMappedMesh());

  auto fedges = fmesh.template GetNodes<NT_EDGE>();
  auto cedges = cmesh.template GetNodes<NT_EDGE>();

  auto vmap = cmap.GetMap<NT_VERTEX>();
  auto emap = cmap.GetMap<NT_EDGE>();

  TableCreator<int> cblocks;
  Array<int> ce2block(cedges.Size());
  Array<int> ucfacets(30); // unique coarse facets

  // for (; !cblocks.Done(); cblocks++) {
  // 	for (auto flnr : Range(loops.Size())) {
  // 	  cblocks.Add(0, flnr);
  // 	}
  // }
  // return cblocks.MoveTable();

  // cout << " mak loop blocks, loops are: " << endl << loops << endl;
  for (; !cblocks.Done(); cblocks++) {
    int cntblocks = 0; ce2block = -1;
    for (auto flnr : Range(loops.Size())) {
      auto floop = loops[flnr];
      // cout << " floop nr " << flnr << ", loop = "; prow(floop); cout << endl;
      ucfacets.SetSize0();
      for (auto j : Range(floop.Size())) {
        auto sfenr = floop[j];
        int fenr = abs(sfenr) - 1;
        auto cenr = emap[fenr];
        if (cenr != -1) {
          auto pos = merge_pos_in_sorted_array(cenr, ucfacets);
          if ( (pos != -1) && (pos > 0) && (ucfacets[pos-1] == cenr) )
            { continue; }
          else if (pos >= 0)
            { ucfacets.Insert(pos, cenr); }
        }
      }
      // cout << " cfacets: "; prow(ucfacets); cout << endl;
      if (ucfacets.Size() == 1) {
        auto cenr = ucfacets[0];
        if (ce2block[cenr] == -1)
          { ce2block[cenr] = cntblocks++; }
        cblocks.Add(ce2block[cenr], flnr);
      }
      else if (ucfacets.Size() > 1) {
        for (auto cenr : ucfacets) {
          if (ce2block[cenr] == -1)
            { ce2block[cenr] = cntblocks++; }
          cblocks.Add(ce2block[cenr], flnr);
        }
      }
      else
        { cblocks.Add(cntblocks++, flnr); }
    }
  }

  auto blocks = cblocks.MoveTable();

  // cout << " hiptmair blocks: " << endl << blocks << endl;

  bool need_blocks = false;
  for (auto k : Range(blocks))
    if (blocks[k].Size() > 1)
      { need_blocks = true; break; }

  // cout << " need blocsk: " << need_blocks << endl;

  if (need_blocks)
    { return std::move(blocks); }
  else
    { return Table<int>(); }
} // StokesMesh::LoopBlocks


template<class... T>
void StokesMesh<T...> :: SetUpDOFedEdges ()
{
  auto gvs = GetGhostVerts();
  dofed_edges = make_shared<BitArray>(this->template GetNN<NT_EDGE>());
  e2dof.SetSize(this->template GetNN<NT_EDGE>());
  if (gvs == nullptr)
  {
    // TODO: in serial, we could get rid of the dofed_edges, dof2e, e2dof entirely
    if (!this->GetEQCHierarchy()->IsDummy())
      { throw Exception("No ghost-verts in non-serial StokesMesh!"); }
    dofed_edges->Set();
    dof2e.SetSize(this->template GetNN<NT_EDGE>());
    for (auto k : Range(dof2e))
      { dof2e[k] = (e2dof[k] = k); }
  }
  else
  {
    dofed_edges->Clear();
    int cnt = 0;
    for (const auto & edge : this->template GetNodes<NT_EDGE>()) {
      // cout << " ITE " << edge;
      bool notgg = !(gvs->Test(edge.v[0]) && gvs->Test(edge.v[1]));
      if (notgg) {
        dofed_edges->SetBit(edge.id);
        e2dof[edge.id] = cnt++;
      }
      else {
        e2dof[edge.id] = -1;
      }
      // cout << " -> " << edge.id << " " << e2dof[edge.id] << " gsts " << gvs->Test(edge.v[0]) << " " << gvs->Test(edge.v[1]) << endl;
    }
    dof2e.SetSize(cnt);
    for (auto k : Range(e2dof)) {
      if (dofed_edges->Test(k))
        { dof2e[e2dof[k]] = k; }
    }
    // cout << " dof -> edge " << endl; prow2(dof2e); cout << endl;
    // cout << " edge -> dof " << endl; prow2(e2dof); cout << endl;
    // cout << " datas " << dof2e.Data() << " " << e2dof.Data() << endl;
  }
} // StokesMesh::SetUpDOFedEdges


template<class... T>
tuple<shared_ptr<BitArray>, FlatArray<int>, FlatArray<int>>
StokesMesh<T...> :: GetDOFedEdges () const
{
  if (dofed_edges == nullptr)
    { const_cast<StokesMesh<T...>&>(*this).SetUpDOFedEdges(); }
  // When i dont do FlatArray(..) here, the data-ptr is advanced by one !
  return make_tuple(dofed_edges, FlatArray<int>(dof2e), FlatArray<int>(e2dof));
} // StokesMesh::GetDOFedEdges

template<class... T>
void StokesMesh<T...> :: SetLoopDPs (Table<int> && _loop_dps)
{
  if (this->GetEQCHierarchy()->IsDummy())
  {
    loopUDofs = UniversalDofs(GetLoops().Size(), 1);
  }
  else
  {
    auto parDofs = make_shared<ParallelDofs>(this->GetEQCHierarchy()->GetCommunicator(), std::move(_loop_dps));
    loopUDofs = UniversalDofs(parDofs);
  }
} // StokesMesh::SetLoopDPs

template<class... T>
UniversalDofs const& StokesMesh<T...> :: GetDofedEdgeUDofs (int BS) const
{
  if ( (dofedEdgeUDofs.Size() < BS + 1) || (dofedEdgeUDofs[BS] == nullptr) )
  {
    auto & ncArray = const_cast<Array<unique_ptr<UniversalDofs>>&>(dofedEdgeUDofs);

    if (ncArray.Size() < BS + 1) {
      int olds = ncArray.Size();
      ncArray.SetSize(BS + 1);
      for (auto k : Range(olds, BS + 1)) // PROBABLY not needed
        { ncArray[k]= nullptr; }
    }

    // auto [dofed_edges, dof2e, e2dof] = GetDOFedEdges();
    auto [dofed_edges_SB, dof2e_SB, e2dof_SB] = GetDOFedEdges();
    auto &dofed_edges = dofed_edges_SB;
    auto &dof2e = dof2e_SB;
    auto &e2dof = e2dof_SB;

    size_t const nDE = dof2e.Size();

    // cout << " this " << this << endl;
    // cout << " GetDofedEdgeUDofs, comm size = " << this->GetEQCHierarchy()->GetCommunicator().Size() << endl;
    // cout << " EQCH " << *this->GetEQCHierarchy() << endl;

    if (this->GetEQCHierarchy()->GetCommunicator().Size() < 2)
    {
      ncArray[BS] = make_unique<UniversalDofs>(nDE, BS);
    }
    else
    {
      const auto & eqc_h = *this->GetEQCHierarchy();

      Array<int> perow(eqc_h.GetDistantProcs().Size()); perow = 0;

      this->template ApplyEQ2<NT_EDGE>(Range(size_t(1), eqc_h.GetNEQCS()), [&](auto eqc, auto edges) {
        for (auto p : eqc_h.GetDistantProcs(eqc)) {
          auto kp = find_in_sorted_array(p, eqc_h.GetDistantProcs());
          perow[kp] += edges.Size();
        }
      }, false);

      Table<int> send_data(perow), recv_data(perow);
      perow = 0;

      this->template ApplyEQ2<NT_EDGE>(Range(size_t(1), eqc_h.GetNEQCS()), [&](auto eqc, auto edges) {
        for (auto p : eqc_h.GetDistantProcs(eqc)) {
          auto kp = find_in_sorted_array(p, eqc_h.GetDistantProcs());
          for (const auto & edge : edges)
            { send_data[kp][perow[kp]++] = dofed_edges->Test(edge.id) ? 1 : 0; }
        }
      }, false);

      // cout << " send_data " << endl << send_data << endl;
      // RANGE pardofs, so use eqc_h-distprocs
      ExchangePairWise(eqc_h.GetCommunicator(), eqc_h.GetDistantProcs(), send_data, recv_data);
      // cout << " recv_data " << endl << recv_data << endl;

      TableCreator<int> cdps(dof2e.Size());

      for (; !cdps.Done(); cdps++) {
        perow = 0;
        this->template ApplyEQ2<NT_EDGE>(Range(size_t(1), eqc_h.GetNEQCS()), [&](auto eqc, auto edges) {
          for (auto p : eqc_h.GetDistantProcs(eqc)) {
            auto kp = find_in_sorted_array(p, eqc_h.GetDistantProcs());
            auto rd = recv_data[kp]; auto & pr = perow[kp];
            for (const auto & edge : edges) {
              if ( (rd[pr++] == 1) && dofed_edges->Test(edge.id)) // ORDER!! need to always increase pr!!
                { cdps.Add(e2dof[edge.id], p); }
            }
          }
        }, false);
      }

      auto tab = cdps.MoveTable();

      ncArray[BS]  = make_unique<UniversalDofs>(
        make_shared<ParallelDofs> (eqc_h.GetCommunicator(), std::move(tab) /* cdps.MoveTable() */, BS, false)
      );

    }
  }
  return *(dofedEdgeUDofs[BS]);
} // StokesMesh::GetDEParDofs



template<class... T>
void StokesMesh<T...> :: printTo(std::ostream &os) const
{
  BlockAlgMesh<T...>::printTo(os);

  os << " STokesMesh Data: " << endl;

  os << "  ghost_verts: " << endl;
  if (GetGhostVerts())
  {
    prowBA(*GetGhostVerts(), os, "   ");
    os << endl;
  }
  else
  {
    os << "  NO GHOST VERTS! " << endl;
  }

  // auto [dofed_edges, dof2e, e2dof] = GetDOFedEdges();
  auto [dofed_edges_SB, dofe2e_SB, e2dofe_SB] = GetDOFedEdges();
  auto &dofed_edges = dofed_edges_SB;
  auto &dofe2e = dofe2e_SB;
  auto &e2dofe = e2dofe_SB;

  os << "DOFED-EDGE UDofs " << endl;
  os << GetDofedEdgeUDofs(1) << endl;

  os << " EDGE <-> DOFED-EDGE maps: " << endl;
  os << "  dofed_edges: " << endl;
  if (dofed_edges)
  {
    prowBA(*dofed_edges, os, "   ");
    os << endl;
  }
  else
  {
    os << " not dofed_edges BA" << endl;
  }
  os << endl << " DOF-E -> E " << endl;
  prow3(dofe2e, os, "   ");
  os << endl;
  os << endl << " E -> DOF-E " << endl;
  prow3(e2dofe, os, "   ");
  os << endl << endl;

  os << "LOOP UDofs " << endl;
  os << GetLoopUDofs() << endl;

  if(GetLoopUDofs().IsParallel())
  {
    os << " LOOP Pardofs: " << endl;
    os << *GetLoopUDofs().GetParallelDofs() << endl;
    os << endl;
  }

  os << " LOOPS: " << endl;
  printTable(GetLoops(), os, "  ");
  os << endl;

  os << " active_loops: " << endl;
  if (GetActiveLoops())
  {
    prowBA(*GetActiveLoops(), os, "   ");
  }
  else
  {
    os << " NO ACTIVE LOOP BA (i.e. all loops are active)" << endl;
  }
}


template<class TVD>
void
AttachedSVD<TVD> :: map_data(const BaseCoarseMap & cmap, AttachedSVD<TVD> *cevd) const
{
  auto mesh = dynamic_pointer_cast<BlockTM>(cmap.GetMesh());
  if (mesh == nullptr)
  {
    throw Exception("Need a BlockTM in AttachedSVD::map_data!");
  }
  this->Cumulate();

  auto &cdata = cevd->data;
  cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0.0;
  cevd->SetParallelStatus(DISTRIBUTED);

  auto vmap = cmap.template GetMap<NT_VERTEX>();
  const auto fecon = *mesh->GetEdgeCM();

  mesh->template Apply<NT_VERTEX>([&](auto v) {
    auto cv = vmap[v];
    if (cv != -1) {
      if (cdata[cv].vol >= 0) { // ?? Not sure about this one
        if (data[v].vol > 0)
          { cdata[cv].vol += data[v].vol; }
        else
          { cdata[cv].vol = data[v].vol; /*-1*/; }
      }
      cdata[cv].vd += data[v].vd;
    }
  }, true);

} // AttachedSVD<C> :: map_data

} // namespace amg

#endif // FILE_STOKES_MESH_IMPL_HPP
