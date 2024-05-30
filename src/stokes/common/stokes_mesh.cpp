#include "stokes_mesh.hpp"

#include <base_mesh_impl.hpp>

namespace amg
{

/**
 * Returns:
 *   - Nr of elements rep. as vertex
 *   - is el RAV?
 *   - EL        -> VERTEX-EL
 *   - VERTEX-EL -> EL
*/
tuple<size_t, shared_ptr<BitArray>, Array<int>, Array<int>>
ElementsRepresentedAsVertices(MeshAccess const &MA, FacetAuxiliaryInformation const &auxInfo);


tuple<shared_ptr<BlockTM>, Array<FVDescriptor>, Array<int>, Array<int>, Array<int>>
BuildStokesMesh(MeshAccess const &MA, FacetAuxiliaryInformation const &auxInfo)
{

  static Timer t("BuildStokesMesh");
  RegionTimer rt(t);

  const auto &fes       = *auxInfo.GetFESpace();
  auto comm             = fes.GetMeshAccess()->GetCommunicator();
  auto fes_pds          = fes.GetParallelDofs();
  bool const isParallel = (fes_pds != nullptr);

  Array<int> emptyDummy;
  FlatArray<int> all_dps(isParallel ? fes_pds->GetDistantProcs() : emptyDummy);

  Array<int> facetDOFs;
  auto getLOFacetDOF = [&](auto fnr) {
    fes.GetDofNrs(NodeId(NT_FACET, fnr), facetDOFs);
    return facetDOFs[0];
  };
      
  /**
   *  --- VERTICES ---
	 * VOL-Elements are "vertices". VOL-Elements are always local, but we need vertices in interfaces.
	 * So, we do 1 layer of overlap, leading to "ghost" vertices.
	 * Verts are "published" to any proc there is a connecting edge to.
	 * This means there are not only loc-ghost but also ghost-ghost edges.
	 *     !!-> "non-ghost" == "solid" <-!!
	 * Additionally, fictitious vertices have to be introduced at domain boundaries to hold boundary facets:
	 *     1)  X LOCAL fict. vertex for every BND nr.
	 *                [ [ EQC = 0 ] [ VOL = -1-bnd_index ] ]
	 *     2)  X LOCAL fict vertex for boundary facets without surface elements (-> no bnd index!)
	 *                [ [ EQC = 0 ] [ VOL = -2-n_bnd_indices ] ]
	 *     3)  1 fict vertex for every undef. element WITH AT LEAST ONE DEFED FACET
	 *                [ [ EQC = varies ] [ VOL = -2-n_bnd_indices-material_ind ] ]
	 * 	 these get agglomerated into eqc/material blocks which is probably the least i can do
	 *    "X" is the max. nr of facets a that connect single element to the same BND-index
	 *        (or max. nr of facets of a single element on mesh boundary without surf el)
	 *   EXTERNAL FICT V == 1 and 2
	 *   INTERNAL FICT V == 3
	 *   That does not conflict with solid/ghost verts. A vertex is solid iff. it was mine before, therefore
	 *   I also have the facet that connects to the BND, therefore the fict vert can be local, the sf-edge is cross (in EQC 0) and
	 *   as I still have all edges connected to the vertex, it stays solid.
	 *  !!! FICT VERTS CAN BE SOLID OR GHOST !!
	 *     [[ second kind can come from defindeon, when el A is on one proc (DEF) and el B on another one (UNDEF) ]]
   */


  /** which elements need to be represented as vertices ?*/

  auto [nelsasv, elasv, E2VE, VE2E] = ElementsRepresentedAsVertices(MA, auxInfo);


  /** dist-procs for solid verts, and list of verts to publish to neibs **/
  Table<int> sv_dps; // dist-procs for solid verts
  Table<int> pub_vs; // verts to publish to neibs
  {
    TableCreator<int> c_sv_dps(nelsasv);
    TableCreator<int> c_pub_vs(all_dps.Size());
    Array<int> tmp(50); tmp.SetSize0();
    for (; !c_sv_dps.Done(); c_sv_dps++, c_pub_vs++) {
      if (isParallel) {
        for (auto velnr : Range(nelsasv)) {
          auto elnr = VE2E[velnr];
          tmp.SetSize0();
          for (auto fnr : MA.GetElFacets(elnr))  {
            // if facet is not active both els must be undef -> no need for EX!!
            if (auxInfo.IsFacetRel(fnr)) {
              // check directly with MA because of UNDEF els ->
              // for (auto dp : fes_pds->GetDistantProcs(a2f_facet[fnr]))
              // for (auto dp : fes_pds->GetDistantProcs(auxInfo.A2R_Facet(fnr)))
              for (auto dp : fes_pds->GetDistantProcs(getLOFacetDOF(fnr)))
                { insert_into_sorted_array_nodups(dp, tmp); }
            }
          }
          c_sv_dps.Add(velnr, tmp);
          for (auto dp : tmp) {
            auto kp = find_in_sorted_array(dp, all_dps);
            c_pub_vs.Add(kp, velnr);
          }
        }
      }
    }
    sv_dps = c_sv_dps.MoveTable();
    pub_vs = c_pub_vs.MoveTable();
  }

  /** Set up EQCHierarchy **/
  Table<int> eq_dps;
  {
    /** set up list of facets with dist-procs representing EQCs **/
    Array<int> reps_range(100); reps_range.SetSize0();

    auto check_dps_range = [&](FlatArray<int> dps)->bool {
      if (dps.Size() < 2) // add these manually
        { return true; }
      // cout << " find "; prow(dps); cout << endl;
      for (auto l : Range(reps_range))
        if (dps == sv_dps[reps_range[l]])
          { return true; }
      // cout << " not found!" << endl;
      return false;
    }; // check_dps_range

    for (auto k : Range(sv_dps)) {
      if (!check_dps_range(sv_dps[k]))
        { reps_range.Append(k); }
    }

    /** Set up dp-table from reps **/
    const int adps = all_dps.Size();
    TableCreator<int> c_eq_dps(1 + adps + reps_range.Size());
    for (; !c_eq_dps.Done(); c_eq_dps++) {
      for (auto k : Range(adps))
        { c_eq_dps.Add(1 + k, all_dps[k]); }
      for (auto k : Range(reps_range))
        { c_eq_dps.Add(1 + adps + k, sv_dps[reps_range[k]]); }
    }
    eq_dps = c_eq_dps.MoveTable();
  }

  // cout << " eq_dps: " << endl << eq_dps << endl;

  // auto p_eqc_h = make_shared<EQCHierarchy>(std::move(eq_dps), comm, true); // not sure if I need true here
  // need the other constructor, not guaranteed to have all eqcs everywhere !
  shared_ptr<EQCHierarchy> p_eqc_h;
  if (isParallel)
    { p_eqc_h = make_shared<EQCHierarchy>(all_dps, eq_dps, comm, true); } // not sure if I need true here
  else
    { p_eqc_h = make_shared<EQCHierarchy>(); }

  const auto & eqc_h = *p_eqc_h;
  size_t neqcs = eqc_h.GetNEQCS();

  auto mesh = make_shared<BlockTM>(p_eqc_h);


  /** Get EQCs for solid verts, exchange lists of eqcs of published vertices **/

  Array<int> sv_eqcs(nelsasv);
  // Array<int> sv_eq_ids(ma->GetNE());    // eqc id for vertex
  Array<int> sv_p_cnt(all_dps.Size()); sv_p_cnt = 0; // # of solid verts in each eqc
  Array<int> sv_eq_cnt(neqcs); sv_eq_cnt = 0;
  for (auto k : Range(sv_eqcs)) {
    size_t eq = (sv_dps[k].Size()) ? eqc_h.FindEQCWithDPs(sv_dps[k]) : 0;
    // sv_eq_ids[k] = eqc_h.GetEQCID(eq);
    sv_eqcs[k] = eq;
    sv_eq_cnt[eq]++;
    for (auto p : sv_dps[k])
      { sv_p_cnt[find_in_sorted_array(p, all_dps)]++; }
  }

  Table<int> send_data(sv_p_cnt); // each row: [ list of eqc-ids ]
  sv_p_cnt = 0;
  for (auto k : Range(sv_eqcs)) {
    if (sv_dps[k].Size()) {
      size_t eqc = eqc_h.FindEQCWithDPs(sv_dps[k]);
      auto eqid = eqc_h.GetEQCID(eqc);
      for (auto p : sv_dps[k]) {
        auto kp = find_in_sorted_array(p, all_dps);
        send_data[kp][sv_p_cnt[kp]++] = eqid;
      }
    }
  }
  Array<Array<int>> recv_data(all_dps.Size());

  // cout << " all_dps " << endl; prow(all_dps); cout << endl;
  // cout << " send_data (vert) " << endl << send_data << endl << endl;

  // we don't know how much we will be receiving
  ExchangePairWise_norecbuffer(comm, all_dps, send_data, recv_data);


  /**
   * New vertex order is: for every eqc, [ verts p0, verts p1, ... ]
   */
  Array<int> allprocs;
  allprocs.Append(all_dps);
  allprocs.Append(comm.Rank());

  /** !!! Possibly need more than 1 fict vertex per BNDNR !!!
	 * elements can have more than 1 facet on a specific BNDNR
	 * so we count N = max. nr of facets of single elements connected to a single bnd nr
	 * and we use N * n_bnds fict verts.
	 * all "BND" facets that have no surface element and are not MPI facets
	 * are assigned an additional BNDNR
   */

  /**
   * facets can have multiple surface elements at interior surfaces;
   * these are ignored!
   */
  auto n_bnds = MA.GetNBoundaries();

  /**
   * Note: I am not sure whether boundaries are 0 or 1 based
   *       I am also also not 100% sure that they are guaranteed to have numbers
   *       0/1 .. n_bnds/n_bnds+1; but I really think so.
   */
  int DEF_UNDEF_CODE = n_bnds + 1;


  /** facet -> fictitious vertex map **/
  Array<int> facet_to_ext_fv(MA.GetNFacets()); // facet -> ext. fict. vetex
  facet_to_ext_fv = -1;

  /** descriptions of added fictitious vertices **/
  Array<FVDescriptor> fv_descr;

  auto surf_facet_to_vert = [&](int facnr) -> int
  { 
    // cout << " surf_facet_to_vert " << facnr << ", s = " << facet_to_ext_fv.Size() << " -> " << facet_to_ext_fv[facnr] << endl;
    return facet_to_ext_fv[facnr];
  };

  MA.SelectMesh(); // TODO: I hope this will not be necessary anymore at some point

  // bnd-indices for fictitious vertices
  Array<int> ext_fv_bnd_inds;

  {
    Array<int> cnt_per_bndi(n_bnds + 2); // +1 in case bnd-inds are 1 based (not sure); +1 for no surfel

    // for MPI-BND facets we need same bnd-id once for DEF-UNDEF and once for UNDEF-DEF
    Array<int> mfv_per_bndi(n_bnds + 2); mfv_per_bndi = 0;
    Array<int> facet_els, facet_surfels, facet_dofs;

    /** Iterate over DEF-OUT facets. NO DEF-UNDEF and no DEF-NEIB facets!! **/
    auto it_bnd_facets = [&](auto pre_el, auto alam, auto post_el) {
      for (auto delnr : Range(auxInfo.GetNE_R())) { // only over defed elements
        ElementId veid(VOL, auxInfo.R2A_EL(delnr));
        pre_el();
        // cout << endl << " EL " << auxInfo.R2A_EL(delnr) << " -> R-el " << delnr << endl;
        // ma->GetElFacets(ElementId(VOL, elnr), el_facets);
        auto el_facets = MA.GetElFacets(veid);
        for (auto j : Range(el_facets)) { // NC space defined on element -> facet has a DOF
          auto facnr = el_facets[j];

          // facet with 2 elements, DEF-DEF or DEF-UNDEF
          //  either way, not a boundary facet
          MA.GetFacetElements(facnr, facet_els); // can be undefed
          if (facet_els.Size() == 2) // DEF-DEF or DEF-UNDEF
            { continue; }

          // facet with a DOF that has a neighbor, DEF-NEIB
          if (isParallel) {
            fes.GetDofNrs(NodeId(NT_FACET, facnr), facet_dofs);

            if (facet_dofs.Size() == 0)
              { throw Exception("AuxInfo has a facet without DOFs as relevant!"); }

            if (fes_pds->GetDistantProcs(facet_dofs[0]).Size())
              { continue; }
          }

          // if we get here, the facet must be DEF-OUT
          FVDescriptor desc;
          // int bnd_id = -1;

          MA.GetFacetSurfaceElements(facnr, facet_surfels);

          // if we have surface elements, we have a proper boundary-ID
          // otherwise, use max_bnd_id + 1
          if (facet_surfels.Size()) {
            desc.type = FVType::BOUNDARY;
            desc.code = MA.GetElIndex(ElementId(BND, facet_surfels[0]));
            // cout << "facet " << facnr << " w. SURF-EL " << facet_surfels[0] << " ON BND " << MA.GetElIndex(ElementId(BND, facet_surfels[0])) << endl;
            // cout << "  desc " << desc << endl;
            // { bnd_id = ma->GetElIndex(ElementId(BND, facet_surfels[0])); }
          }
          else { // DEF-OUT, take first surf el
            desc.type = FVType::NO_BOUNDARY;
            desc.code = DEF_UNDEF_CODE;
            // cout << " facet " << facnr << ", NO-SURF-EL, code = " << DEF_UNDEF_CODE << endl;
            // { bnd_id = DEF_UNDEF_CODE; }
          }

          // okay, apply lambda to this facet + facet-id!
          // alam(facnr, bnd_id);
          alam(facnr, desc);
          // ma->GetFacetElements(facnr, facet_els);
        }
        post_el();
      } // elnr loop
    }; // it_bnd_facets

    // count facets per boundary-id
    it_bnd_facets([&] () { cnt_per_bndi = 0; },
                  [&](auto fnr, auto fvdesc) { cnt_per_bndi[fvdesc.code]++; },
                  [&] () {
                    for (auto l : Range(mfv_per_bndi))
                      { mfv_per_bndi[l] = max(mfv_per_bndi[l], cnt_per_bndi[l]); }
                  });

    // compress to remove empty boundaries
    int totmax = 0;
    for (auto l : Range(mfv_per_bndi))
      { totmax = max(totmax, mfv_per_bndi[l]); }

    int cnt_fv = 0;
    Array<int> bnd_compress(totmax * (n_bnds + 2)); bnd_compress = -1;

    ext_fv_bnd_inds.SetSize(totmax * (n_bnds + 2));  // do this AFTER reduce

    for (auto k : Range(n_bnds + 2)) {
      for (auto l : Range(mfv_per_bndi[k])) {
        ext_fv_bnd_inds[cnt_fv] = k;  // do this AFTER reduce
        bnd_compress[totmax * k + l] = cnt_fv++;
      }
    }

    ext_fv_bnd_inds.SetSize(cnt_fv); // do this AFTER reduce

    // write into facet -> fictitious vertex map, fv-description
    fv_descr.SetSize(cnt_fv);
    it_bnd_facets([&](){ cnt_per_bndi = 0; },
                  [&](auto fnr, auto fvdesc) {
                    auto bndi = fvdesc.code;
                    int fv_nr = bnd_compress[totmax * bndi + cnt_per_bndi[bndi]++];
                    facet_to_ext_fv[fnr] = fv_nr;
                    // cout << "fnr " << fnr << ", desc " << fvdesc << ", compress " << (totmax * bndi + cnt_per_bndi[bndi]-1) << " -> FV " << fv_nr << "/" << cnt_fv << endl;
                    fv_descr[fv_nr] = fvdesc;
                  },
                  [&](){ ; });
  }

    /** Elements we are not defed on **/


  /**
   * vertex order:
   *   EQ 0:   [fict_vs, loc solid_vs]
   *   EQ k:   [ghost_v p0, ghost_v p1, ...., solid_vs, ghost_v pi, .... ] (with sorted mems of EQC: p0 < p1 < p2.... < rank < pi < ....)
  **/
  size_t NV_ext_fict = ext_fv_bnd_inds.Size();
  size_t NV_solid    = nelsasv; // includes fict. verts from non-def els
  size_t NV_ghost    = std::accumulate(recv_data.begin(), recv_data.end(), size_t(0), [](size_t a, FlatArray<int> b) ->size_t { return a + b.Size(); });

  size_t NV          = NV_ext_fict + NV_solid + NV_ghost;
  size_t NV_glob     = comm.AllReduce(NV_ext_fict + NV_solid, MPI_SUM);

  Array<int> perow(neqcs);
  for (auto eqc : Range(perow))
    { perow[eqc] = 1 + eqc_h.GetDistantProcs(eqc).Size(); }

  // each row: [ cnt_vs_from_p0, cnt_vs_from_p1, ... ] (includes myself!)
  Table<int> eqc_vpcnts(perow);
  if (perow.Size())
    { eqc_vpcnts.AsArray() = 0; }

  /** count solid verts **/
  // cout << endl << " eqc_vpcnts: " << endl << eqc_vpcnts << endl;
  for (auto eqc : Range(eqc_vpcnts)) {
    int lme = merge_pos_in_sorted_array(comm.Rank(), eqc_h.GetDistantProcs(eqc));
    // cout << "eqc " << eqc << ", lme " << lme << endl;
    eqc_vpcnts[eqc][lme] = sv_eq_cnt[eqc];
  }

  /** count ghost verts **/
  for (auto kp : Range(recv_data)) {
    int p = all_dps[kp];
    for (auto eqid : recv_data[kp]) {
      // cout << " eqid " << eqid << endl;
      auto eqc = eqc_h.GetEQCOfID(eqid);
      // cout << " eqc " << eqc << endl;
      int lp = find_in_sorted_array(p, eqc_h.GetDistantProcs(eqc));
      if (p > comm.Rank())
        { lp++; }
      eqc_vpcnts[eqc][lp]++;
    }
  }

  /** vertex-sort given in terms of ALL EL-NRS!! (TODO: does this still make sense? should I go back to elnrs?) **/
  Array<int> VE2V(nelsasv);    // vertex-el -> vertex
  Array<int> E2V(MA.GetNE());  // element   -> vertex  ( == concatenated el -> v-el -> v )
  E2V = -1;
  {
    /** eqc_vert offsets **/
    Array<size_t> disp_eq(1 + neqcs);   // offstes for eqc_verts FlatTable
    Array<int> my_offset(1 + neqcs); // offsets where to put solid verts
    disp_eq[0] = NV_ext_fict; // have to set this back to 0 later!
    my_offset[0] = NV_ext_fict;
    for (auto eqc : Range(neqcs)) {
      int nvk = std::accumulate(eqc_vpcnts[eqc].begin(), eqc_vpcnts[eqc].end(), double(0.0)); // # of verts in eqc (excluding fict verts)
      // cout << " eqc " << eqc << ", nvk " << nvk << endl;
      disp_eq[1+eqc] = disp_eq[eqc] + nvk;
      int myl = merge_pos_in_sorted_array(comm.Rank(), eqc_h.GetDistantProcs(eqc));
      // cout << " myl " << myl << endl;
      auto lower_range = eqc_vpcnts[eqc].Range(0, myl);
      // cout << "lower_range "; prow(lower_range); cout << endl;
      int nvk_sp = std::accumulate(lower_range.begin(), lower_range.end(), double(0.0)); // # verts in eqc from smaller proc
      // cout << " eqc " << eqc << ", nvk_sp: " << nvk_sp << endl;
      my_offset[1+eqc] = disp_eq[1+eqc];
      my_offset[eqc] += nvk_sp;
    }
    disp_eq[0] = 0; // set back to 0 !!
    // cout << " disp_eq " << endl << disp_eq << endl;
    // cout << " my_disp " << endl << my_offset << endl;
    /** ok, now vert_sort is easy - just count from my_offsets! **/
    for (auto velnr : Range(sv_eqcs)) {
      int vnr = my_offset[sv_eqcs[velnr]]++;
      auto elnr = VE2E[velnr];
      // auto delnr = nc_fes->E2DE(elnr);
      // if (doco && (vnr == spec_vnum))
      // cout << "ID " << "vel" << velnr << " el" << elnr << " vrt" << vnr << endl;
      // E2V[velnr] = vnr;
      // vert_sort[elnr] = vnr;
      VE2V[velnr] = vnr;
      E2V[elnr] = vnr;
    }

    /** ok, have displacement, now set up vertex table **/
    Array<AMG_Node<NT_VERTEX>> verts(NV);
    for (auto k : Range(verts))
      { verts[k] = k; }
    mesh->SetNodeArray<NT_VERTEX>(std::move(verts)); // not implemented yet
    mesh->SetNN<NT_VERTEX>(NV, NV_glob);
    mesh->SetEQOS<NT_VERTEX>(std::move(disp_eq)); // not implemented yet
    // for (auto eqc : Range(eqc_h.GetNEQCS())) {
    //   auto eqvs = mesh->GetENodes<NT_VERTEX>(eqc);
    //   cout << " verts eqc " << eqc << ": " << mesh->GetENN<NT_VERTEX>(eqc) << endl;
    //   prow2(eqvs); cout << endl;
    // }
  }

  /**
   *  --- EDGES ---
   * solid - solid:  add locally, resorted by SetNodes
   * ghost - ghost:  added by SetNodes reduce (are solid-solid edges on a neib)
   * solid - ghost:  these are the problem...
   *    i) Every facet is shared by 2 procs
   *   ii) Ex-facets are ordered consistently
   * Therefore, exchange data: list of (eqid, locnr) for local vertex of solid-ghost edge
   * And [(x;y) for x,y in zip(send, recv)] are the solid-ghost edges
   */
  Table<INT<2, int>> send_data_edge;
  Array<int> elnums(20); elnums.SetSize0();
  {
    Array<int> facet_dofs(20);
    Array<int> sg_cnt(all_dps.Size());
    TableCreator<INT<2, int>> csde(all_dps.Size());
    TableCreator<INT<2, int>> csg_ffnrs(all_dps.Size()); // (fine) facet nrs for sg edges
    if (isParallel) {
      for (; !csde.Done(); csde++) {
        sg_cnt = 0;
        for (auto ffnr : Range(auxInfo.GetNFacets_R())) {
          auto facnr = auxInfo.R2A_Facet(ffnr);
          fes.GetDofNrs(NodeId(NT_FACET, facnr), facet_dofs);
          if (!facet_dofs.Size())
            { throw Exception("R-facet without DOFS??"); }
          auto dps = fes_pds->GetDistantProcs(facet_dofs[0]);
          if (!dps.Size())
            { continue; }
          MA.GetFacetElements(facnr, elnums);
          // GetDefinedFacetElements(facnr, elnums);
          auto kp = find_in_sorted_array(dps[0], all_dps);
          // ma->GetFacetElements(facnr, elnums);
          auto venum = E2VE[elnums[0]];
          int vnr = VE2V[venum];
          int locnr = mesh->MapNodeToEQC<NT_VERTEX>(vnr);
          int eqc = mesh->GetEQCOfNode<NT_VERTEX>(vnr);
          INT<2, int> tup ({ eqc_h.GetEQCID(eqc), locnr });
          csde.Add(kp, tup);
          sg_cnt[kp]++;
          if (dps.Size() > 1)
            { throw Exception("HOW????"); }
        }
      }
    }
    send_data_edge = csde.MoveTable();
  }

  // cout << " AM RANK " << comm.Rank() << " of " << comm.Size() << endl;

  // Array<Array<INT<2, int>>> recv_data_edge(all_dps.Size());
  perow.SetSize(send_data_edge.Size());
  for (auto k : Range(perow))
    { perow[k] = send_data_edge[k].Size(); }
  Table<INT<2, int>> recv_data_edge(perow);
  // cout << " send_data_edge : " << endl << send_data_edge << endl;
  ExchangePairWise(comm, all_dps, send_data_edge, recv_data_edge);
  // cout << " recv_data_edge : " << endl << recv_data_edge << endl;


  // ???? I think debugging
  auto fake_lam_neq = [&](auto node_num, const auto & vs) LAMBDA_INLINE {
    // cout << " lam_neq with " << vs << endl;
    constexpr int NODE_SIZE = sizeof(AMG_CNode<NT_EDGE>::v)/sizeof(AMG_Node<NT_VERTEX>);
    INT<NODE_SIZE,int> eqcs;
    auto eq_in = mesh->GetEQCOfNode<NT_VERTEX>(vs[0]);
    auto eq_cut = eq_in;
    for (auto i:Range(NODE_SIZE)) {
      auto eq_v = mesh->GetEQCOfNode<NT_VERTEX>(vs[i]);
      eqcs[i] = eqc_h.GetEQCID(eq_v);
      eq_in = (eq_in==eq_v) ? eq_in : -1;
      eq_cut = (eq_cut==eq_v) ? eq_cut : eqc_h.GetCommonEQC(eq_cut, eq_v);
    }
    // cout << " eq in cut " << eq_in << " " << eq_cut << endl;
    // cout << " eqcs " << eqcs << endl;
    AMG_CNode<NT_EDGE> node = {{vs}, eqcs};
    // if (eq_in!=size_t(-1)) cout << " call eqc " << node_num << " " << node << " " << eq_in << endl;
    // else cout << " call cross " << node_num << " " << node << " " << eq_cut << endl;
  };

  Array<int> F2E(MA.GetNFacets());
  F2E = -1;

  size_t n_edges = auxInfo.GetNFacets_R(); // f2a_facet also includes MPI bnd facets (so sg edges)
  // cout << " n_edges = " << n_edges << endl;

  Array<int> sg_cnt(all_dps.Size());
  sg_cnt = 0;

  Array<int> facet_dofs(200);
  facet_dofs.SetSize0();

  {
    Array<int> vertexEls;
    // cout << " EX-verts! " << endl;
    int cnt = 0;
    for (auto k : Range(MA.GetNV()))
    {
      auto dps = MA.GetDistantProcs(NodeId(NT_VERTEX, k));
      if (dps.Size())
      {
        // cout << "mesh-vertex " << k << " is ex # " << cnt << ", shared with "; prow(dps); cout << endl;
        cnt++;

        // cout << "  els of mesh-vertex: " << endl;
        MA.GetVertexElements(k, vertexEls);

        // for (auto j : Range(vertexEls))
        // {
        //   cout << "    " << vertexEls[j] << " -> V " << E2V[vertexEls[j]] << endl;
        // }
      }
    }
  }

  mesh->SetNodes<NT_EDGE> (n_edges, [&](auto edge_num) LAMBDA_INLINE {
      // cout << endl << " get edge " << edge_num << " of " << n_edges << endl;
      decltype(AMG_Node<NT_EDGE>::v) pair;
      // int fnr = f2a_facet[edge_num], dnr = edge_num;
      int fnr = auxInfo.R2A_Facet(edge_num);

      fes.GetDofNrs(NodeId(NT_FACET, fnr), facet_dofs);

      if (facet_dofs.Size() == 0)
      {
        // cout << " n_edges = " << n_edges << endl;
        // cout << " edge_num " << edge_num << endl;
        // cout << " fnr " << fnr << endl;
        throw Exception("Stokes-Mesh SetNodes<NT_EDGE> R-facet without DOFs!");
      }

      auto dps = isParallel ? fes_pds->GetDistantProcs(facet_dofs[0]) : emptyDummy;

      // cout << "  fes-dof " << facet_dofs[0] << endl;
      // cout << " fnr " << fnr << ", dps "; prow2(dps); cout << endl;
      if (!dps.Size()) { // SS
        MA.GetFacetElements(fnr, elnums);
        // cout << " elnums : "; prow(elnums); cout << endl;
        if (elnums.Size() == 2) { // DEF-DEF or DEF-UNDEF
          // cout << " loc vol" << endl;
          pair[0] = E2V[elnums[0]];
          pair[1] = E2V[elnums[1]];
        }
        else { // DEF-FICT
          // cout << " sf edge " << endl;
          pair[0] = E2V[elnums[0]]; // vert_sort[denr] == VE2V[venr] for defed els
          // cout << " pair[0] = " << pair[0] << endl;
          pair[1] = surf_facet_to_vert(fnr);
          // cout << " pair[1] = " << pair[1] << endl;
        }
        // if (doco && ( (pair[0] == spec_vnum) || (pair[0] == spec_vnum) ) ) {
          // cout << "(V1) SetNodes; facet " << fnr << " ffnr " << edge_num << " dnum " << dnr << " -> pair " << pair << endl;
        // }
      }
      else { // SG edge
        // cout << " sg edge " << endl;
        int kp = find_in_sorted_array(dps[0], all_dps); // <- only a single dp!
        int l = sg_cnt[kp]++;
        // cout << " kp " << kp << ", l " << l << endl;
        FlatArray<INT<2, int>> rowa = send_data_edge[kp];
        FlatArray<INT<2, int>> rowb = recv_data_edge[kp];
        // cout << " sizes " << rowa.Size() << " " << rowb.Size() << endl;
        // cout << " rowa " << rowa[l] << ", rowb " << rowb[l] << endl;
        int eqa = eqc_h.GetEQCOfID(rowa[l][0]);
        int eqb = eqc_h.GetEQCOfID(rowb[l][0]);
        // cout << " eqa " << eqa << ", eqb " << eqb << endl;
        pair[0] = mesh->MapENodeFromEQC<NT_VERTEX>(rowa[l][1], eqa);
        pair[1] = mesh->MapENodeFromEQC<NT_VERTEX>(rowb[l][1], eqb);
        // if (doco && ( (pair[0] == spec_vnum) || (pair[0] == spec_vnum) ) ) {
          // cout << "(V2) SetNodes; facet " << fnr << " ffnr " << edge_num << " dnum " << dnr << " -> pair " << pair << endl;
          // cout << " dps "; prow(dps); cout << endl;
          // cout << " dp " << dps[0] << ", kp " << kp << endl;
          // cout << " eqs " << eqa << " " << eqb << endl;
          // cout << " rowa/b entries " << rowa[l] << " " << rowb[l] << endl;
        // }
      }
      // cout << " unsorted pair = " << pair[0] << " , " << pair[1] << endl;
      if (pair[0] > pair[1])
        { swap(pair[0], pair[1]); }
      // cout << " sorted pair   = " << pair[0] << " , " << pair[1] << endl;
      if (edge_num == n_edges - 1) /** reset iteration **/
        { sg_cnt = 0; }
      // fake_lam_neq(edge_num, pair); // ??? I think DEBUGGING ???
      return pair;
    },
    [&](auto ffnr, auto edge_id) LAMBDA_INLINE {
      // auto fnr = f2a_facet[ffnr];
      // cout << " facet-nr " << fnr << ", ffnr " << ffnr << " sorted to " << id << endl;
      // is it fnr or fnr??
      int fnr = auxInfo.R2A_Facet(ffnr);
      F2E[fnr] = edge_id; // unsure if this does something evil with additional edges (apparently not?)
    } );

  return make_tuple(mesh, fv_descr, E2V, F2E, facet_to_ext_fv);
} // BuildStokesMesh


tuple<size_t, shared_ptr<BitArray>, Array<int>, Array<int>>
ElementsRepresentedAsVertices(MeshAccess const &MA, FacetAuxiliaryInformation const &auxInfo)
{
  size_t nelsasv = 0;
  shared_ptr<BitArray> elasv; // element as vertex ?
  Array<int> E2VE, VE2E;      // vertex-el <-> el mapping

  elasv = make_shared<BitArray>(MA.GetNE());
  elasv->Clear();

  for (auto elnr : Range(MA.GetNE())) {
    // NOTE: We are adding elements that have SOME active facet, and not active ELEMENTS!
    //       Therefore, we also add inactive elements with an active neighboring element
    //       on a different proc.
    bool need_vert = auxInfo.IsElementRel(elnr);
    if (!need_vert) {
      for (auto fnr : MA.GetElFacets(ElementId(VOL, elnr)))
        if (auxInfo.IsFacetRel(fnr))
          { need_vert = true; break; }
    }
    if (need_vert)
      { elasv->SetBit(elnr); nelsasv++; }
  }

  E2VE.SetSize(MA.GetNE()); E2VE = -1;
  VE2E.SetSize(nelsasv); nelsasv = 0;

  for (auto elnr : Range(MA.GetNE())) {
    if (elasv->Test(elnr)) {
      VE2E[nelsasv] = elnr;
      E2VE[elnr] = nelsasv++;
    }
  }

  return make_tuple(nelsasv, elasv, E2VE, VE2E);
} // ElementsRepresentedAsVertices

} // namespace amg