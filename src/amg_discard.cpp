#ifndef FILE_AMG_DISCARD_CPP
#define FILE_AMG_DISCARD_CPP

#include "amg.hpp"

namespace amg
{

  template<class TMESH>
  VDiscardMap<TMESH> :: VDiscardMap (shared_ptr<TMESH> _mesh, size_t _max_bs)
    : BaseDiscardMap(_mesh, nullptr), GridMapStep<TMESH>(_mesh), max_bs(_max_bs)
  {
    CalcDiscard();
    // SetUpMM();
  } // VDiscardMap(..)


  template<class TMESH>
  shared_ptr<TopologicMesh> VDiscardMap<TMESH> :: GetMappedMesh () const
  {
    if (mapped_mesh == nullptr)
      { const_cast<VDiscardMap<TMESH>&>(*this).SetUpMM(); }
    return mapped_mesh;
  }


  template<class TMESH>
  void VDiscardMap<TMESH> :: CalcDiscard ()
  {
    /**
       Mark "hanging vertices": a vertex, where:
           - for each par of neighbors N1 and N2 
	     there exists an edge connecting them.
	   - eqc(vertex) <= eqc(N) for all neighbours
	     (if we accept non-exact SC for repl mat, we can relax this condition
	      and instead require that there is at lest one STRONGLY(!) connected neighbour
	      in a larger eqc)
       -
       Assume v1 and v2 are two neighbouring, hanging vertices.
         (I)  all neibs of v1 are connected ==> v2 is connected to all neibs of v1
	 (II) all neibs of v2 are connected ==> v2 neibs are connected to v1 neibs
	==> We can drop both without losing any additional coarse edges.
       -
       BUT: we can have many connected hanging nodes:
        all 1 DOF    --------------------
                     --------------------
                      |   |   |   |   |          | |
                      H - H - H - H - H ...... - H |  <-- large block of H verts !!
                      |   |   |   |   |          | |
        all 1 DOF    --------------------
       -              --------------------
       -
       So group these into blocks:
                - block sizes must be bounded
		- No vertices in different blocks can be connected
		- blocks must be within an EQC (!!!)
		(-> some "hanging" vertices might not be assigned a block, 
		    we leave these in the mesh)
     **/

    /** mark hanging vertices **/

    const auto& M (*mesh);
    const auto NV = M.template GetNN<NT_VERTEX>();
    const auto& ECM = *M.GetEdgeCM();
    const auto& eqc_h = *M.GetEQCHierarchy();

    size_t max_elim = 0, min_elim = NV;
    Array<int> keep_v(NV); keep_v = 0;

    /** mark all non-hanging vertices - we HAVE to keeep these **/

    M.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto vnr) LAMBDA_INLINE { // mark vertices I need to keep
	auto neibs = ECM.GetRowIndices(vnr);
	const auto nn = neibs.Size();
	for (auto N : neibs) // can only eliminate the vertex locally if all neighbours are "more global"
	  if (!eqc_h.IsLEQ(eqc, M.template GetEqcOfNode<NT_VERTEX>(N)))
	    { keep_v[vnr] = 1; return; }
	for (auto k : Range(neibs)) { // if there are 2 neighbours that are not connected, keep vertex
	  for (auto j : Range(k)) {
	    auto pos = find_in_sorted_array(neibs[k], ECM.GetRowIndices(neibs[j]));
	    if (pos == decltype(pos)(-1))
	      { keep_v[vnr] = 1; return; }
	  }
	}
	min_elim = min2(size_t(vnr), min_elim);
	max_elim = max2(size_t(vnr), max_elim);
      }, false ); // everyone has to look through all eqcs!
    M.template AllreduceNodalData<NT_VERTEX>(keep_v,  [](auto & in) { return sum_table(in); }, false);

    /** form blocks of hanging vertices **/

    const size_t MAX_BS = max2(size_t(1), max_bs);
    TableCreator<size_t> cb;
    dropped_vert = make_shared<BitArray>(NV);
    auto & dropv (*dropped_vert);
    size_t block_num = 0;
    Array<int> block_vs;
    for (; !cb.Done(); cb++) {
      dropv.Clear(); block_num = 0;
      if (max_elim > min_elim)
	for (auto k : Range(min_elim, max_elim)) { // buggy I think...
	  // for (auto k = min_elim; k < max_elim; k++) {
	  if ( (keep_v[k] == 0) && !dropv.Test(k) ) {
	    block_vs.SetSize(1); block_vs[0] = k;
	    auto neibs = ECM.GetRowIndices(k);
	    // guarantee that there is at least one "remaining" vertex adjacent to every block, otherwise we lose information
	    // detect cases where there is a fully connected set of verts without an edge leading outside the cluster
	    bool all_hang = true;
	    // cout << " k " << k << endl;
	    // cout << " neibs "; prow2(neibs); cout << endl;
	    for (size_t j = 1; j < neibs.Size(); j++)
	      { all_hang &= (keep_v[neibs[j]] == 0); } // all neibs i can drop are more local than the "center" verteix
	    if (all_hang)
	      { keep_v[neibs.Last()] = 1; } // in that case, keep one vertex in the group
	    const auto nn = neibs.Size();
	    const auto this_block_num = block_num++;
	    cb.Add(this_block_num, k); dropv.SetBit(k);
	    size_t bcnt = 1;
	    for (size_t j = 0; (j < nn) && (bcnt < MAX_BS); j++) {
	      // cout << " j = " << j << endl;
	      const auto neib = neibs[j];
	      if ( (keep_v[neib] == 0) && !dropv.Test(neib) && (neib != k) ) {
		// this is a hanging neighbour which is not in a block yet - add it
		cb.Add(this_block_num, neib); bcnt++; dropv.SetBit(neib); block_vs.Append(neib);
	      }
	    }
	    // set all neibs of neibs to KEEP (cannot connect dropped vertices!)
	    // can let that slide for "pw-" prol only.
	    for (auto mem : block_vs) {
	      for (auto neib : ECM.GetRowIndices(mem)) {
		if (!block_vs.Contains(neib))
		  { keep_v[neib] = 1; }
	      }
	    }
	  }
	}
    }
    vertex_blocks = make_shared<Table<size_t>>();
    *vertex_blocks = cb.MoveTable();
    
    // cout << " DISCARD SUMMARY: " << endl;
    // cout << " econ: " << endl << ECM << endl;
    // cout << " blocks: " << endl << *vertex_blocks << endl;
    // cout << " discard map drops " << dropv.NumSet() << " of " << dropv.Size() << " verts in " << vertex_blocks->Size() << " blocks " << endl;

  } // VDiscardMap::CalcDiscard


  template<class TMESH>
  void VDiscardMap<TMESH> :: SetUpMM ()
  {
    auto& fmesh (*mesh);
    const auto& eqc_h = *fmesh.GetEQCHierarchy();

    auto p_cmesh = make_shared<BlockTM>(fmesh.GetEQCHierarchy());
    auto& cmesh (*p_cmesh);

    auto neqcs = eqc_h.GetNEQCS();
    
    node_maps.SetSize(4);
    mapped_nnodes.SetSize(4); mapped_nnodes = 0;

    for (NODE_TYPE NT : {NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL} ) {
      mapped_nnodes[NT] = 0;
      cmesh.has_nodes[NT] = fmesh.has_nodes[NT];
      cmesh.nnodes_glob[NT] = 0;
      cmesh.nnodes[NT] = 0;
    }

    /** vertices **/
    const auto NV = fmesh.template GetNN<NT_VERTEX>();
    auto& vblocks (*vertex_blocks);
    auto& dvert (*dropped_vert);
    const size_t n_dropped_v = vblocks.AsArray().Size();
    size_t mapped_nv = fmesh.template GetNN<NT_VERTEX>() - n_dropped_v;
    mapped_nnodes[NT_VERTEX] = cmesh.nnodes[NT_VERTEX] = mapped_nv;
    cmesh.verts.SetSize(mapped_nv); auto & btm_verts = cmesh.verts;
    cmesh.disp_cross[NT_VERTEX].SetSize(neqcs+1); cmesh.disp_cross[NT_VERTEX] = 0;
    cmesh.disp_eqc[NT_VERTEX].SetSize(neqcs+1); auto & vdeq = cmesh.disp_eqc[NT_VERTEX];
    size_t nvg = 0;
    auto & vmap = node_maps[NT_VERTEX]; vmap.SetSize(NV); vmap = -1;
    mapped_nv = 0; // !
    fmesh.template ApplyEQ<NT_VERTEX> ( [&] (auto eqc, auto vnr) LAMBDA_INLINE {
	  if (!dvert.Test(vnr)) {
	    btm_verts[mapped_nv] = mapped_nv;
	    vmap[vnr] = mapped_nv++;
	    vdeq[1+eqc]++;
	    if (eqc_h.IsMasterOfEQC(eqc))
	      { nvg++; }
	  }
      }, false);
    cmesh.nnodes_cross[NT_VERTEX].SetSize(neqcs); cmesh.nnodes_cross[NT_VERTEX] = 0;
    auto& nv_eq = cmesh.nnodes_eqc[NT_VERTEX]; nv_eq.SetSize(neqcs); nv_eq = 0;
    for (auto k : Range(neqcs)) {
      nv_eq[k] = vdeq[k+1];
      vdeq[k+1] += vdeq[k];
    }
    cmesh.eqc_verts = MakeFT<AMG_Node<NT_VERTEX>> (neqcs, vdeq, btm_verts, 0);
    cmesh.nnodes_glob[NT_VERTEX] = eqc_h.GetCommunicator().AllReduce(nvg, MPI_SUM);    

    /** edges **/
    const auto NE = fmesh.template GetNN<NT_EDGE>();
    size_t mapped_ne = 0;
    auto & emap = node_maps[NT_EDGE]; emap.SetSize(NE); emap = -1;
    auto is_valid = [] (auto x) LAMBDA_INLINE { return x != decltype(x)(-1); };
    auto & disp_eqc_e = cmesh.disp_eqc[NT_EDGE]; disp_eqc_e.SetSize(neqcs + 1); disp_eqc_e = 0;
    auto & disp_cross_e = cmesh.disp_cross[NT_EDGE]; disp_cross_e.SetSize(neqcs + 1); disp_cross_e = 0;
    const auto n_f_eqc_e = fmesh.disp_eqc[NT_EDGE].Last();
    size_t cnt_eq = 0, cnt_cross = 0;
    size_t neg = 0;
    fmesh.template ApplyEQ<NT_EDGE> ( [&] (auto eqc, const auto & fedge) LAMBDA_INLINE {
	if ( is_valid(vmap[fedge.v[0]]) && is_valid(vmap[fedge.v[1]])) {
	  if (fedge.id >= n_f_eqc_e) { // this is a cross-edge
	    disp_cross_e[1+eqc]++;
	    emap[fedge.id] = cnt_cross++;
	  }
	  else { // eqc-edge
	    disp_eqc_e[1+eqc]++;
	    emap[fedge.id] = cnt_eq++;
	  }
	  if (eqc_h.IsMasterOfEQC(eqc))
	    { neg++; }
	}
      }, false);
    // for (auto k : Range(n_f_eqc_e, NE))
      // if (is_valid(emap[k]))
	// { emap[k] += cnt_eq; }

    mapped_nnodes[NT_EDGE] = cmesh.nnodes[NT_EDGE] = cnt_eq + cnt_cross;
    cmesh.nnodes_glob[NT_EDGE] = eqc_h.GetCommunicator().AllReduce(neg, MPI_SUM);
    cmesh.nnodes_eqc[NT_EDGE].SetSize(neqcs);
    cmesh.nnodes_cross[NT_EDGE].SetSize(neqcs);

    // cout << " disp_cross_e " << endl; prow2(disp_cross_e); cout << endl;
    // cout << " disp_eqc_e " << endl; prow2(disp_eqc_e); cout << endl;

    for (auto k : Range(neqcs)) {
      cmesh.nnodes_cross[NT_EDGE][k] = disp_cross_e[k+1];
      disp_cross_e[k+1] += disp_cross_e[k];
      cmesh.nnodes_eqc[NT_EDGE][k] = disp_eqc_e[k+1];
      disp_eqc_e[k+1] += disp_eqc_e[k];
    }
    auto & cme = cmesh.edges;
    cme.SetSize(cmesh.nnodes[NT_EDGE]);
    for (const auto & e : fmesh.template GetNodes<NT_EDGE>()) {
      auto ce_id = emap[e.id];
      if (is_valid(ce_id)) {
	if (e.id >= n_f_eqc_e)
	  { emap[e.id] += cnt_eq; ce_id += cnt_eq; }
	cme[ce_id].id = ce_id;
	cme[ce_id].v = { vmap[e.v[0]], vmap[e.v[1]] };
      }
    }

    // cout << "cmesh edges: " << endl; prow2(cme); cout << endl;
    // cout << " cmesh disp_eqc " << endl; prow2(cmesh.disp_eqc[NT_EDGE]); cout << endl;
    // cout << " cmesh disp_cross " << endl; prow2(cmesh.disp_cross[NT_EDGE]); cout << endl;

    cmesh.eqc_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_eqc[NT_EDGE], cmesh.edges, 0);
    cmesh.cross_edges = MakeFT<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_cross[NT_EDGE], cmesh.edges, cnt_eq);

    if constexpr(std::is_same<TMESH, BlockTM>::value == 1) {
	mapped_mesh = p_cmesh;
      }
    else {
      auto mapped_data = mesh->MapData(*this);
      mapped_mesh = make_shared<TMESH> ( move(*p_cmesh), mapped_data );
    }

    // cout << " coarse mesh: " << endl;
    // cout << *static_pointer_cast<TMESH>(mapped_mesh) << endl;

  } // VDiscardMap::SetUpMM


} // namespace amg

#include "amg_tcs.hpp"

#endif
