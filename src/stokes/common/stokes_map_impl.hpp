#ifndef FILE_STOKES_MAP_IMPL_HPP
#define FILE_STOKES_MAP_IMPL_HPP

#include "grid_contract.hpp"
#include "stokes_map.hpp"
#include "loop_utils.hpp"

namespace amg
{

/** StokesCoarseMap **/

template<class T>
Table<T> RemoveDuplicates (FlatTable<T> tin, bool issorted = false)
{
  int n = tin.Size();
  Array<int> perow(n); perow = 0;
  for (auto k : Range(n)) {
    if (tin[k].Size() < 2) {
      perow[k] = tin[k].Size();
      continue;
    }
    if (!issorted)
      { QuickSort(tin[k]); }
    int ndup = 0;
    for (auto j : Range(1, int(tin[k].Size()))) {
      if (tin[k][j] == tin[k][j-1])
        { ndup++; }
    }
    perow[k] = tin[k].Size() - ndup;
  }
  Table<int> tout(perow);
  for (auto k : Range(n)) {
    int nk = tin[k].Size(), c = 0;
    if ( nk == 0)
      { continue; }
    tout[k][0] = tin[k][0];
    if (nk < 2)
      { continue; }
    for (auto j : Range(1, int(tin[k].Size()))) {
    if (tin[k][j] != tout[k][c])
      { c++; tout[k][c] = tin[k][j]; }
    }
  }
  return std::move(tout);
}

template<class TMESH>
shared_ptr<BaseCoarseMap> StokesCoarseMap<TMESH> :: Concatenate (shared_ptr<BaseCoarseMap> right_map)
{
  // TODO: I think we can remove this now?
  // normal concatenate
  auto cmap = make_shared<BaseCoarseMap>(this->mesh, right_map->GetMappedMesh());
  BaseCoarseMap::SetConcedMap(right_map, cmap);

  if (auto rsm = dynamic_pointer_cast<StokesCoarseMap<TMESH>>(right_map)) {
    // WE NO MORE HAVE A LOOP_MAP!
    // auto & right_map = rsm->loop_map;
    // Array<int> nlm(loop_map.Size());
    // auto & nlm = cmap->loop_map; nlm.SetSize(loop_map.Size());
    // for (auto k : Range(loop_map)) {
    //   nlm[k] = 0;
    //   auto mk = loop_map[k];
    //   if (mk != 0) {
    //       auto mid_loop_nr = abs(mk)-1;
    //       double fac = (mk < 0) ? -1.0 : 1.0;
    //       auto mmk = right_map[mid_loop_nr];
    //       if (mmk != 0)
    //         { nlm[k] = fac * mmk; }
    //   }
    // }
  }
  else
      { throw Exception("This should probably not happen ..."); }

  return cmap;
} // StokesCoarseMap<TMESH>::Concatenate

template<class TMESH>
void StokesCoarseMap<TMESH> :: MapAdditionalDataA ()
{
  static Timer t("StokesCoarseMap::MapAdditionalData - A"); RegionTimer rt(t);

  auto & fmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMesh(),       "StokesCoarseMap::MapAdditionalData F-Mesh");
  auto & cmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMappedMesh(), "StokesCoarseMap::MapAdditionalData C-Mesh");

  /** ghost vertices **/

  auto vmap = this->template GetMap<NT_VERTEX>();

  auto fgv = fmesh.GetGhostVerts();
  auto cgv = make_shared<BitArray>(cmesh.template GetNN<NT_VERTEX>());

  cgv->Clear();
  for (auto k : Range(fgv->Size())) {
    if ( fgv->Test(k) && (vmap[k] != -1) )
      { cgv->SetBit(vmap[k]); }
  }
  cmesh.SetGhostVerts(cgv);

}

template<class TMESH>
void StokesCoarseMap<TMESH> :: MapAdditionalDataB ()
{
  static Timer t("StokesCoarseMap::MapAdditionalData - B"); RegionTimer rt(t);

  LocalHeap lh(24*1024*1024, "Tom_Turbo");

  auto & fmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMesh(),       "StokesCoarseMap::MapAdditionalData F-Mesh");
  auto & cmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMappedMesh(), "StokesCoarseMap::MapAdditionalData C-Mesh");

  // cout << " START StokesCoarseMap::MapAdditionalData" << endl;

  const auto & eqc_h = *fmesh.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();

  fmesh.CumulateData();
  cmesh.CumulateData();

  auto fedata = get<1>(fmesh.Data())->Data();
  auto fvdata = get<0>(fmesh.Data())->Data();

  auto cedata = get<1>(cmesh.Data())->Data();
  auto cvdata = get<0>(cmesh.Data())->Data();

  auto fedges = fmesh.template GetNodes<NT_EDGE>();
  auto cedges = cmesh.template GetNodes<NT_EDGE>();

  const auto & fecon = *fmesh.GetEdgeCM();
  const auto & cecon = *cmesh.GetEdgeCM();

  auto vmap = this->template GetMap<NT_VERTEX>();
  auto emap = this->template GetMap<NT_EDGE>();

  // TODO: simpler in serial ?!

  auto floop_uDofs = fmesh.GetLoopUDofs();
  auto fde_uDofs   = fmesh.GetDofedEdgeUDofs(1);
  // auto & fde_pds = *fmesh.GetDEParDofs(1);


  // cout << " StokesCoarseMap::MapAdditionalData" << endl;
  // cout << " meshes: " << this->GetMesh() << " -> " << this->GetMappedMesh() << endl;
  // cout << "  CGV = " <<cgv << endl << *cgv << endl;
  // cout << " eqc_h = " << endl << eqc_h << endl;

  // NOTE: coarse mesh has ghost verts now, so we can build coarse mesh dof <-> edge maps

  auto [ f_des, f_dof2e, f_e2dof ] = fmesh.GetDOFedEdges();
  auto [ c_des, c_dof2e, c_e2dof ] = cmesh.GetDOFedEdges();

  Array<int> ucfacets(30); // unique coarse facets
  Array<int> cloop(30);

  /**
   *  I cannot locally determine whether a loop is
   *     i) "deflated": consists of only one edge, traversed forward and backward
   *                    For a deflated loop at least one looper knows it is deflated.
   *    ii) "empty":    all edges vanish (it lies completely within an agglomerate)
   *                    ALL loopers must see it locally empty.
   *  LOOP CODING:
   *      0 .. empty
   *      1 .. OK
   *      2 .. deflated
   *      3 .. includes 0 flow edge
   *      <0 .. ided with loop #[abs(x)-1]
   *  NOTE: I think this comment is out of date:
   *          - At least one looper SHOULD be able to tell whether a loop is empty locally
   *            since aggregates exist on solid vertices
   *          - A loop can locally look deflated on one looper but still be a proper loop
   *            on others
   */

  // cout << " CRS MAP ADD DATA " << endl;
  // // cout << " FMESH " << endl << fmesh << endl;
  // // cout << " CMESH " << endl << cmesh << endl;
  // cout << " FMESH ECON " << *fmesh.GetEdgeCM() << endl;
  // cout << " CMESH ECON " << *cmesh.GetEdgeCM() << endl;
  // cout << " VMAP " << endl; prow2(vmap); cout << endl;
  // cout << " EMAP " << endl; prow2(emap); cout << endl;
  // cout << endl << endl; TestParDofsConsistency(floop_pds, "MAD_impl, FLOOP_DPS"); cout << endl << endl;
  // cout << " floops " << endl << loops << endl << endl;


  // { // I think here we just check that every vertex really has exactly one partition where it is solid
  //   auto [ c_des, c_dof2e, c_e2dof ] = fmesh.GetDOFedEdges();
  //   // cout << endl << " CRS f_ghost_vs " << endl; print_ba_as_a(*fmesh.GetGhostVerts()); cout << endl;
  //   // cout << endl << " CRS f_dof2e " << endl; prow2(c_dof2e); cout << endl << endl;
  //   // cout << endl << " CRS f_e2dof " << endl; prow2(c_e2dof); cout << endl << endl;
  //   auto gvs = fmesh.GetGhostVerts();
  //   const auto & eqc_h = *fmesh.GetEQCHierarchy();
  //   Array<int> check_gv(fmesh.template GetNN<NT_VERTEX>());
  //   for (auto k : Range(check_gv))
  //     { check_gv[k] = gvs->Test(k) ? 0 : 1; }
  //   fmesh.template AllreduceNodalData<NT_VERTEX>(check_gv, [&](const auto & tin) { return sum_table(tin); }, false);
  //   for (auto vnr : Range(check_gv)) {
  //     if (check_gv[vnr] != 1) {
  //       auto [eqc, lnr] = fmesh.template MapENodeToEQLNR<NT_VERTEX>(vnr);
  //       cout << " vertex " << vnr << " eqc " << eqc << " loc-nr " << lnr << ", shared with "; prow(eqc_h.GetDistantProcs(eqc));
  //       cout << " has GV = " << check_gv[vnr] << ", vol = " << fvdata[vnr].vol << endl;
  //     }
  //   }
  // }

  /**
   *    I) c loops as tuple list with (eq0,vnr0, eq1,v1) entries
   *        - deflated loops: [vi, vj, vi]
   *        - empty: []
   *        - ided:
   *        - loops can "split", e.g these examples
   *            a - a - b - b       a - b  ||  a - x - y - b      a         b
   *            |           |        \ /   ||  |           |      | \     / |
   *            y           y  ->     y    ||  c           d  ->  c  x - y  d
   *            |           |        / \   ||  |           |      | /     \ |
   *            c - c - d - d       c - d  ||  e - x - y - f      e         f
   *            this should split into 2   ||  this should split into 2 loops
   *            coarse loops               ||  [and almost looks deflated, because edge x-y is traversed twice in opposite directions]
   *          NOTE: only master of every fine DOF(!!) adds the coarse EDGE(!!) -> gathered loops have NO OVERLAP
   *   II) gather c loops on loopers, sort chunks together, split up loops and output them s.t.:
   *        - they are (eqid, lv) lists
   *        - they are rotated/flipped to a pre-determinded "start" (see below for details)
   *        - every looper of the fine loop has the coarse loop
   *  III) remove duplicates and deflated and empty loops
   *        -> output loops as (oriented) !! EDGE !! list
   *       NOTE: I am not 100% sure this catches ALL duplicates, but I do think so.
   *   IV) unify loops: call AddOrientedLoops: (not 100% sure this is needed, but pretty sure overlap makes it necessary)
   *
   *  NOTE: loop_map is not a useful concept anymore because of these splits
   *  TODO: Can PHASE I+II be combined such that we do not create the INT<4>-table for completely local loops?
   *        Need to measure how time critical that would be.
   */

  /** PHASE I **/
  // cout << " /** PHASE I **/ " << endl;

  /**
   * Convert FINE loops to COARSE (eq0, lv0, eq1, lv1) tuples
   *   - table has exactly one entry per fine loop
   *   - includes duplicates
   *   - includes empty ones
   */
  auto loops = fmesh.GetLoops();

  Table<INT<4, int>> init_cl_vs;
  {
    TableCreator<INT<4, int>> c_cl_vs(loops.Size());
    for (; !c_cl_vs.Done(); c_cl_vs++ ) {
      for (auto k : Range(loops)) {
        auto loop = loops[k];
        // cout << k << "/" << loops.Size() << endl;
        if (!loop.Size())
          { continue; }
        int last_cv = -1;
        for (auto j : Range(loop)) {
          // cout << "  " << j << "/" << loop.Size() << " " << loop[j] << flush;
          // int fenr = abs(loop[j])-1, fdnr = f_e2dof[fenr];
          auto [orient, fenr] = loopDecode(loop[j]);
          // cout << " -> decoded " << orient << " " << fenr << flush;
          int fdnr = f_e2dof[fenr];
          // cout << " -> fdnr " << fdnr << endl;
          // if (fdnr == -1)
          // {
          //   cout << " edge = " << fedges[fenr] << endl;
          //   cout << "   is dofed ? " << f_des->Test(fenr) << endl;
          //   cout << "   are vertices ghosts? " <<  fgv->Test(fedges[fenr].v[0]) << " " <<  fgv->Test(fedges[fenr].v[1]) << endl;
          // }
          /** TODO: if local loop, call a simplified split-function here
                    instead of setting up a superfluous table **/
          if ( (fdnr == -1) || // loops include gg-edges, someone else (the master of the DOF) adds them
               (!fde_uDofs.IsMasterDof(fdnr)) )
            { continue; }
          int cv0 = vmap[fedges[fenr].v[0]], cv1 = vmap[fedges[fenr].v[1]];
          if (cv0 == cv1) // edge -> -1
            { continue; }
          if (orient == NEGATIVE)
            { swap(cv0, cv1); }
          auto [eq0, lv0] = cmesh.template MapENodeToEQLNR<NT_VERTEX>(cv0);
          auto [eq1, lv1] = cmesh.template MapENodeToEQLNR<NT_VERTEX>(cv1);
          if ( (eq0 >= eqc_h.GetNEQCS()) || (eq1 >= eqc_h.GetNEQCS()) || (eq0 == -1) || (eq1 == -1) ) {
            cout << " fenr " << k << " -> icvs, loop =  "; prow(loop); cout <<  endl;
            cout << " check " << fenr << " d " << fdnr << " of " << fde_uDofs.GetNDofLocal() << " glob " << fde_uDofs.GetNDofGlobal() << endl;
            cout << fedges[fenr] << "/d" << fdnr << " cvs " << cv0 << " " << cv1 << endl;
            cout << " eqv0 " << eq0 << " " << lv0 << " eqv1 " << eq1 << " " << lv1 << endl;
            cout << " eqi0 " << eqc_h.GetEQCID(eq0) << flush << " eqi1 " << eqc_h.GetEQCID(eq1) << endl;
          }
          c_cl_vs.Add(k, { eqc_h.GetEQCID(eq0), lv0, eqc_h.GetEQCID(eq1), lv1 });
        }
      }
    }
    init_cl_vs = c_cl_vs.MoveTable();
  }

  // cout << endl << " init_cl_vs " << endl << init_cl_vs << endl;

  /** PHASE  II **/
  Table<INT<2, int>> all_c_loops;
  Table<int> all_c_loopers;
  // Table<int> c_loop_buckets;
  {
    /**
     *  PHASE II.I: set up and exchange clv-tables
     *    The data from neighbor k is:
     *        [ (size_shared_loop_0, -1, -1, -1), tup0, ..., tupLast,
     *          (size_shared_loop_1, -1, -1, -1), tup1, ..., tupLast, etc. ]
     *    This way, we cna iterate through init_cl_vs table in order and get the chunk of the
     *    received data belonging to that loop by just incrementing counts properly.
     */
    // cout << " /** PHASE II.I **/ " << endl;

    auto all_dps = floop_uDofs.GetDistantProcs(); // != eqc_h.GetDistantProcs() !? not sure why/when this happens...

    // Exchange: [(size, -1, -1, -1), tup0, tup1, ..., tuplast]
    Table<INT<4, int>> send_data;
    Array<Array<INT<4, int>>> recv_data(all_dps.Size());
    {
      TableCreator<INT<4, int>> csd(all_dps.Size());
      for (; !csd.Done(); csd++) {
        for (auto k : Range(loops)) {
          auto dlers = floop_uDofs.GetDistantProcs(k);
          // cout << " fl " << k << ", dps = "; prow(dlers); cout << endl;
          for (auto j : Range(dlers)) {
            auto pj = dlers[j]; auto kp = find_in_sorted_array(pj, all_dps);
            // cout << "fl " << k << " pj " << pj << " is " << j << "'th ex" << ", kp " << kp << endl;
            csd.Add( kp, { int(init_cl_vs[k].Size()), -1, -1, -1 } );
            // csd.Add( kp, init_cl_vs[k] ); // does not build
            for (auto j : Range(init_cl_vs[k])) // so do it entry for entry
              { csd.Add( kp, init_cl_vs[k][j] ); }
          }
        }
      }
      send_data = csd.MoveTable();
      // cout << endl << "send_data " << endl << send_data << endl;
      ExchangePairWise2(eqc_h.GetCommunicator(), all_dps, send_data, recv_data);
      // cout << endl << "recv_data " << endl;
      // for (auto k : Range(recv_data))
        // { cout << k << ": "; prow(recv_data[k]); cout << endl; }
    }

    /** PHASE II.II: compute "coordindates"
     *      (kp, p, os, len)
     *    of loops in received data
     */
    // cout << " /** PHASE II.II **/ " << endl;

    Table<INT<4, int>> chunk_ijks;
    {
      auto all_dps = floop_uDofs.GetDistantProcs(); // != eqc_h.GetDistantProcs()??
      Array<int> recv_cnt(all_dps.Size());
      TableCreator<INT<4, int>> c_ijks(loops.Size());
      for (; !c_ijks.Done(); c_ijks++) {
        recv_cnt = 0;
        for (auto lnr : Range(loops)) {
          auto dps = floop_uDofs.GetDistantProcs(lnr);
          for (auto j : Range(dps))
          {
            auto pj  = dps[j];
            auto kpj = find_in_sorted_array(pj, all_dps);
            int  os  = recv_cnt[kpj] + 1;
            int  len = recv_data[kpj][recv_cnt[kpj]][0];

            recv_cnt[kpj] += 1 + len;

            c_ijks.Add(lnr, INT<4, int>({ int(kpj), pj, os, len }));
          }
          c_ijks.Add(lnr, INT<4, int>({ -1, comm.Rank(), 0, int(loops[lnr].Size()) }));
        }
      }
      chunk_ijks = c_ijks.MoveTable();
      for (auto k : Range(chunk_ijks)) {
        QuickSort(chunk_ijks[k], [&](auto & ta, auto & tb) {
            return ta[1] < tb[1]; // sort by rank
          });
      }
    }
    // cout << endl << "chunk_ijks" << endl << chunk_ijks << endl;

    /**
     *  PHASE II.III: set up tables of valid coarse loops
     *     - they can still contain duplicates
     *     - "closed" generated coarse loops are "rotated" such that they "start"
     *       at their "smallest" vertex and oriented such that the second traversed
     *       vertex is the smaller of the neighbors of the initial one
     *     - "open" generated coarse loops start at the smaller of their ends
     *     - In PHASE III, this loop-format allows us to sort the loops into
     *       buckets by their first vertex.
     *     -
     */
    // cout << " /** PHASE II.III **/ " << endl;

    // auto rot_loop = [&](FlatArray<INT<2, int>> slk) { rotateVLoop(slk, rot_buffer); };


    Array<INT<2, int>> rot_buffer;
    Array<Array<INT<2, int>>> simple_loops;
    Array<FlatArray<INT<4, int>>> the_data;

    /**
     * iterate through coarse loops by merging the local with
     * the received chunks with "SimpleLoopsFromChunks"
     *
     * Remove empty and deflated loops, then convert to oriented-edge-list
     * and call given lambda on that.
     */
    auto it_loops = [&](bool rot_loops, auto alam) {
      for (auto flnr : Range(loops)) {
        auto dps = floop_uDofs.GetDistantProcs(flnr);
        the_data.SetSize(chunk_ijks[flnr].Size());
        for (auto j :  Range(the_data)) {
          auto [ kp, p, os, len ] = chunk_ijks[flnr][j];
          if (kp < 0)
            { the_data[j].Assign(init_cl_vs[flnr]); }
          else
            { the_data[j].Assign(recv_data[kp].Part(os, len)); }
        }
        simple_loops.SetSize0();
        SimpleLoopsFromChunks(the_data, simple_loops, lh);
        // if (simple_loops.Size() > 1) {
        //   cout << " flnr " << flnr << "/" << loops.Size() << endl;
        //   cout << " sort chunks for loop " << flnr << ": " << the_data.Size() << endl;
        //   for (auto j : Range(the_data)) {
        //     cout << "  " << j << ": "; prow(the_data[j]); cout << endl;
        //   }
        //   cout << " sorted, split loop chunks for loop " << flnr << ": " << endl;
        //   for (auto j : Range(simple_loops))
        //     { cout << j << ": "; prow(simple_loops[j]); cout << endl; }
        //   cout << endl;
        // }
        for (auto k : Range(simple_loops)) {
          FlatArray<INT<2, int>> slk = simple_loops[k];
          if (slk.Size() < 2)
            { continue; }
          bool closed = slk[0] == slk.Last();
          if ( (slk.Size() <= 3) && closed ) // [A,A] or [A,B,A]
            { continue; }
          if (rot_loops) {
            // cout << " rotate loop "; prow(slk); cout << endl;
            rotateVLoop(slk, rot_buffer);
            // { rot_loop(slk); }
            // cout << " rotated loop "; prow(slk); cout << endl;
          }
          alam(flnr, simple_loops[k]);

          // this did the conversion from INT<2> -> int
          // scl.SetSize0(); // dont know size a priori (off-proc edges)
          // for (auto k : Range(slk.Size()-1))
          //   { add_scl(slk[k], slk[k+1]); }
          // // cout << " for flnr " << flnr << ", e-loop is "; prow(scl); cout << endl;
          // // if (closed) // NO ... closed loop is [A,B,....,X,A] so no need
          //   // { add_scl(slk.Last(), slk[0]); }
          // alam(flnr, scl);
        }
      }
    };

    // count loops, no need for bucketable loops
    int ccl = 0;
    it_loops(false, [&](auto flnr, FlatArray<INT<2, int>> cl) {
      // TODO: I don't think need the add_scl calls here!
      //       would it be better to have it_loops call lambda on the simple_loops
      //       and convert to the scl-loops only in rounds 2+3??
      // cout << " f loop " << flnr << " -> " << ccl << ", zflow = " << zflow << ", cl = "; prow(cl); cout << endl;
      ccl++;
    });

    TableCreator<INT<2, int>> c_c_loops(ccl);
    TableCreator<int>         c_c_loopers(ccl);
    // TableCreator<int>         c_c_loop_buckets(cmesh.template GetNN<NT_EDGE>());

    // for the rest of the
    bool rot_loops = false;
    for (; !c_c_loops.Done(); c_c_loops++, c_c_loopers++) {//, c_c_loop_buckets++) {
      ccl = 0;
      it_loops(true, [&](auto flnr, FlatArray<INT<2, int>> cl) { // TODO: we are also rotating on IT 1, is this needed??
          // if (cl.Size()) // when local chunk is empty
          //   { c_c_loop_buckets.Add(abs(cl[0])-1, ccl); } // bucket == first enr
          // does not compile for some reason:
          // c_c_loops.Add(ccl, cl);
          for (auto j : Range(cl))
            { c_c_loops.Add(ccl, cl[j]); }
          c_c_loopers.Add(ccl, floop_uDofs.GetDistantProcs(flnr));
          ccl++;
        });
      rot_loops = true; // only need rotated loops in the very end
    }
    // c_loop_buckets = c_c_loop_buckets.MoveTable();
    all_c_loops = c_c_loops.MoveTable();
    all_c_loopers = c_c_loopers.MoveTable();
  } // II

  // cout << endl << "bucketable all_c_loops" << endl << all_c_loops << endl;
  // cout << endl << "all_c_loopers" << endl << all_c_loopers << endl;

  // cout << " /** PHASE III (remove duplicates) **/ " << endl;

  Table<int> init_cloops, init_loopers;
  {
    /**
     * Put the INT<2>-loops into buckets, buckets are determined by the first vertex of each loop,
     * which is given as (eqid, lnr). Since we can also have eqids of eqcs we don't know about,
     * we don't know a-priory how many buckets we can have. So, we go through all loops, collect the
     * appearing eqc-ids and the minimum and maximum appearing local vertex numbers.
     * That allows us to compute a maximum amount of needed buckets, 1 + (max_v - min_v) for every appearing eq-id.
     */

    // all eqids, min_v and max_v appearing in all_c_loops
    Array<int> all_eqids(200);
    Array<int> eq_next_vs(200);
    Array<int> eq_min_vs(200);
    all_eqids.SetSize(0);
    eq_next_vs.SetSize(0);
    eq_min_vs.SetSize(0);
    for (auto k : Range(all_c_loops))
    {
      int eqid = all_c_loops[k][0][0];
      int lv   = all_c_loops[k][0][1];
      auto pos = merge_pos_in_sorted_array(eqid, all_eqids);
      if ( (pos > 0) && (all_eqids[pos - 1] == eqid) ) {
        pos = pos - 1;
        eq_min_vs[pos]  = min(eq_min_vs[pos], lv);
        eq_next_vs[pos] = max(eq_next_vs[pos], lv + 1);
      }
      else {
        all_eqids.Insert(pos, eqid);
        eq_min_vs.Insert(pos, lv);
        eq_next_vs.Insert(pos, lv + 1);
      }
    }

    int n_all_eq = all_eqids.Size();

    // TODO: WHY AM I SORTING THIS??
    auto inds = makeSequence(n_all_eq, lh);
    QuickSortI(all_eqids, inds);
    ApplyPermutation(all_eqids, inds);
    ApplyPermutation(eq_next_vs, inds);
    ApplyPermutation(eq_min_vs, inds);

    // we need buckets for all_eqids, [eq_min_vs ... eq_next_vs)
    Array<int> eq_bucket_offsets(n_all_eq + 1);
    eq_bucket_offsets[0] = 0;
    for (auto k : Range(all_eqids)) {
      int max_buckets_eq = eq_next_vs[k] - eq_min_vs[k];
      eq_bucket_offsets[k + 1] = eq_bucket_offsets[k] + max_buckets_eq;
    }
    int max_buckets = eq_bucket_offsets[n_all_eq];

    auto tup2bucket = [&](auto &tup) {
      int eq_nr = find_in_sorted_array(tup[0], all_eqids);
      int compressed_vnr = tup[1] - eq_min_vs[eq_nr];
      // offset for bucktes with the same id, + (vert - min_vert)
      int offset = eq_bucket_offsets[eq_nr] + compressed_vnr;
      return offset;
    };

    TableCreator<int> c_c_loop_buckets(max_buckets);
    for (; !c_c_loop_buckets.Done(); c_c_loop_buckets++ ) {
      for (auto k : Range(all_c_loops))
        { c_c_loop_buckets.Add(tup2bucket(all_c_loops[k][0]), k); }
    }

    // ok, we have buckets !
    Table<int> c_loop_buckets = c_c_loop_buckets.MoveTable();

    /**
     *  Find duplicates;
     *   NOTE: I don't think this finds ALL duplicates after MPI unify??
     */
    Array<int> ce_dups(all_c_loops.Size());
    ce_dups = -1;

    Array<int> dupof(20);

    size_t tcl  = all_c_loops.Size();
    size_t tclu = tcl;

    for (auto bucket_nr : Range(c_loop_buckets)) {
      auto bucket = c_loop_buckets[bucket_nr];
      auto tots = bucket.Size();
      dupof.SetSize(tots); dupof = -1;
      int cnt_dup = 0;
      for (auto j : Range(bucket)) {
        for (auto l : Range(j)) // check if dup of l
          // TODO: if critical, maybe better: with QuicksortI?
          // all loops start at lowest # edge which is oriented positively
          if (all_c_loops[bucket[j]] == all_c_loops[bucket[l]])
            { cnt_dup++; dupof[j] = l; break; }
        if (dupof[j] != -1) {
          // cout << " CL " << bucket[j] << " is dup of " << bucket[dupof[j]] << endl;
          ce_dups[bucket[j]] = bucket[dupof[j]];
        }
      }
      tclu -= cnt_dup;
    }

    // cout << endl << " all_c_loops " << endl << all_c_loops << endl;
    // cout << endl << " stokes coarse map, loops locally reduced " << tcl << " -> " << tclu << endl;


    // converts (eq0id, lv0), (eq1id, lv1) -> oriented enr
    auto convert_loop = [&](auto loop, Array<int> &scl) {
      int const N = loop.Size();
      int eqid0, lv0, eq0, eqid1, lv1, eq1, cv0, cv1, cenr;

      eqid0 = loop[0][0];
      lv0   = loop[0][1];
      eq0   = eqc_h.GetEQCOfID(eqid0);

      scl.SetSize(0);
      for (int k = 1; k < N; k++) {
        eqid1 = loop[k][0];
        lv1   = loop[k][1];
        eq1   = eqc_h.GetEQCOfID(eqid1);

        if ( (eq0 != -1) && (eq1 != -1) ) {
          cv0 = cmesh.template MapENodeFromEQC<NT_VERTEX>(lv0, eq0);
          cv1 = cmesh.template MapENodeFromEQC<NT_VERTEX>(lv1, eq1);
          cenr = int(cecon(cv0, cv1));
          if (cenr != -1)
            { scl.Append( loopEncode(cv0, cv1, cenr) ); }
        }

        eqid0 = eqid1;
        lv0   = lv1;
        eq0   = eq1;
      }
    };

    /**
     * Construct duplicate-free table and converts (eq0id, lv0), (eq1id, lv1) -> oriented enr
     *  !! cinversion also picks out only the LOCAL part of a given loop !!
     */
    TableCreator<int> c_init_cloops (tclu);
    TableCreator<int> c_init_loopers(tclu);
    Array<int> scl(200);
    for (; !c_init_cloops.Done(); c_init_cloops++, c_init_loopers++) {
      int cnt = 0;
      for (auto loop_nr : Range(all_c_loops)) {
        if (ce_dups[loop_nr] == -1) { // no duplicate!
          // cout << endl << " convert " << loop_nr << ": "; prow(all_c_loops[loop_nr]); cout << endl;
          convert_loop(all_c_loops[loop_nr], scl);
          // cout << "    -> COVNERTED: "; prow(scl); cout << endl;
          c_init_cloops.Add(cnt, scl);
          c_init_loopers.Add(cnt, all_c_loopers[loop_nr]);
          cnt++;
        }
      }
    }
    init_cloops = c_init_cloops.MoveTable();
    init_loopers = c_init_loopers.MoveTable();


    // Convert

    // { /** reduce duplicate loop info **/
    //   auto p_acl_dps = make_shared<ParallelDofs>(floop_pds.GetCommunicator(), std::move(CopyTable(all_c_loopers)), 1, false);
    //   const auto & acl_dps = *p_acl_dps;
    //   auto all_dps = acl_dps.GetDistantProcs();
    //   Array<int> perow(all_dps.Size());
    //   for (auto k : Range(perow))
    //     { perow[k] = acl_dps.GetExchangeDofs(all_dps[k]).Size(); }
    //   Table<int> send_data(perow), recv_data(perow);
    //   // send lists of ided loops (in ex-dof enum)
    //   for (auto k : Range(perow)) {
    //     auto ex_loops = acl_dps.GetExchangeDofs(all_dps[k]);
    //     for (auto j : Range(perow[k])) {
    //       int lnr = ex_loops[j], ided_lnr = ce_dups[lnr], ided_ex_lnr = -1;
    //       if (ided_lnr != -1)
    //         { ided_ex_lnr = find_in_sorted_array(ce_dups[lnr], ex_loops); }
    //       send_data[k][j] = ided_ex_lnr;
    //     }
    //   }
    //   // exchange data
    //   ExchangePairWise(acl_dps.GetCommunicator(), all_dps, send_data, recv_data);
    //   // check if me+neib id same loops with one another
    //   for (auto k : Range(perow)) {
    //     auto ex_loops = acl_dps.GetExchangeDofs(all_dps[k]);
    //     for (auto j : Range(perow[k])) {
    //       int lnr = ex_loops[j], ided_lnr = ce_dups[lnr], ided_ex_lnr = recv_data[k][j];
    //       if ( (ided_ex_lnr == -1) || (ex_loops[ided_ex_lnr] != ided_lnr) ) { // me + neib id loops differently
    //         // cout << " DO NOT DUP " << lnr << " and " << ided_lnr << " (ex-proc "
    //       // << all_dps[k] << " ids with " << ided_ex_lnr;
    //         // if (ided_ex_lnr != -1)
    //     // { cout << " -> " << ex_loops[ided_ex_lnr]; }
    //         // cout << ")" << endl;
    //         { ce_dups[lnr] = -1; }
    //       }
    //     }
    //   }
    // }

    // cout << " REDUCED ce_dups " << endl; prow2(ce_dups); cout << endl;

    // tclu = 0;
    // for (auto k : Range(ce_dups)) {
    //   if (ce_dups[k] == -1)
    //     { tclu++; }
    // }
    // cout << endl << " unified reduced " << tcl << " -> " << tclu << endl;

    // TableCreator<int> c_init_cloops(tclu), c_init_loopers(tclu);
    // for (; !c_init_cloops.Done(); c_init_cloops++, c_init_loopers++) {
    //   int cnt = 0;
    //   for (auto k : Range(all_c_loops)) {
    //     if (ce_dups[k] == -1) {
    //       c_init_cloops.Add(cnt, all_c_loops[k]);
    //       c_init_loopers.Add(cnt, all_c_loopers[k]);
    //       cnt++;
    //     }
    //   }
    // }
    // init_cloops = c_init_cloops.MoveTable();
    // init_loopers = c_init_loopers.MoveTable();
  } // PHASE III

  // cout << " /** PHASE IV (unify) **/ " << endl;

  // auto & loop_map = dynamic_cast<StokesCoarseMap<TMESH>&>(const_cast<BaseCoarseMap&>(cmap)).GetLoopMapMod();
  // loop_map.SetSize(loops.Size()); loop_map = -1;

  // Array<int> unif_sort(init_cloops.Size());

  // cout << " init_cloops = " << endl << init_cloops << endl;
  // cout << endl << " map loops, call AddOrientedLoops " << endl;

  cmesh.AddOrientedLoops(init_cloops, init_loopers, [&](auto init_cnum, auto scnum) {
    // loop_map does not fit with splitting loops
    // unif_sort[init_cnum] = scnum;
  });

  // cout << endl << "loop_map I  "; prow2(loop_map); cout << endl << endl;
  // cout << endl << "unif_sort "; prow2(unif_sort); cout << endl << endl;
  // // Note: loops that have become locally empty (by merging all locally traversed vertices)
  // // are removed, then unif_sort[loop_map[k] == -1
  // for (auto k : Range(loop_map))
  //   if (loop_map[k] != -1)
  // 	{ loop_map[k] = unif_sort[loop_map[k]]; }
  // cout << endl << "loop_map II "; prow2(loop_map); cout << endl << endl;

  // cout << " /** LOOP STUFF DONE, now just some bookkeeping and DONE **/ " << endl;

  /**
   *  When a coarse vertx has only a single neib and no (collapsed) DIRICHLET connection,
   *  all BFs have divergence 0 (they are constant!). I think then the coarse vertex should have volume 0.
   *  That way, constant divergence is automatically not enforced on those elements!
   */

  Array<int> zero_vol(cmesh.template GetNN<NT_VERTEX>()); zero_vol = 0;

  cmesh.template Apply<NT_VERTEX>([&](auto cv) {
    if (cecon.GetRowIndices(cv).Size() == 1) {
      int cenr = int(cecon.GetRowValues(cv)[0]);
      auto ceflow = L2Norm(cedata[cenr].flow);
      if (ceflow < 1e-12 * cvdata[cv].vol)
        { zero_vol[cv] = 1.0; }
    }
  }, false); // need G verts

  cmesh.template AllreduceNodalData<NT_VERTEX>(zero_vol, [&](const auto & in){ return std::move(sum_table(in)); }, false);

  for (auto cv : Range(zero_vol)) {
    if (zero_vol[cv] != 0)
      { cout << " CV " << cv << " HAS ZERO VOL!!" << endl; }
  }

  for (auto cv : Range(zero_vol)) {
    if (zero_vol[cv] != 0)
      { cvdata[cv].vol = 0.0; }
  }

  // {
  //   auto [ c_des, c_dof2e, c_e2dof ] = cmesh.GetDOFedEdges();
  //   cout << endl << " CRS c_ghost_vs " << endl; print_ba_as_a(*cmesh.GetGhostVerts()); cout << endl;
  //   cout << endl << " CRS c_dof2e " << endl; prow2(c_dof2e); cout << endl << endl;
  //   cout << endl << " CRS c_e2dof " << endl; prow2(c_e2dof); cout << endl << endl;
  // }

  /**
   *  Loops are marked as "inactive" when they contain an edge with 0 flow
   *   !! an inactive loop CAN become active again due to coarsening!!
   */
  {
    // std::cout << " CHECK ZERO FLOW LOOPS !" << std::endl;
    // TODO: should be simpler in serial!
    // auto [ c_des, c_dof2e, c_e2dof ] = cmesh.GetDOFedEdges();

    auto [ c_des_SB, c_dof2e_SB, c_e2dof_SB ] = cmesh.GetDOFedEdges();

    auto &c_des = c_des_SB;
    auto &c_dof2e = c_dof2e_SB;
    auto &c_e2dof = c_e2dof_SB;

    auto cloop_uDofs = cmesh.GetLoopUDofs();
    // auto cloop_pds = cmesh.GetLoopParDofs();
    auto cloops = cmesh.GetLoops();
    Array<int> zf(cloops.Size()); zf = 0;
    for (auto k : Range(cloops)) {
      for (auto j : Range(cloops[k]))
        if ( is_zero(cedata[abs(cloops[k][j])-1].flow) )
          {
            cout << " cloop " << k << " ZERO-F because of edge " << j << "/" << cloops[k].Size() << " = " << abs(cloops[k][j])-1 << " w. ED " << cedata[abs(cloops[k][j])-1] << endl;
            zf[k] = 1;
            break;
          }
    }
    if (cloop_uDofs.IsParallel())
      { MyAllReduceDofData(*cloop_uDofs.GetParallelDofs(), zf,  [&](auto & a, const auto & b) { a += b; }); }
    auto cl_active = make_shared<BitArray>(cloops.Size());
    cl_active->Clear();
    for (auto k : Range(cloops)) {
      if (zf[k] == 0)
        { cl_active->SetBit(k); }
    }
    cmesh.SetActiveLoops(cl_active);
  }

  // cout << endl << "StokesCoarseMap::MapAdditionalData DONE " << endl << endl;

} // StokesCoarseMap::MapAdditionalData

/** END StokesCoarseMap **/


/** StokesContractMap **/

template<class TMESH>
void StokesContractMap<TMESH> :: MapAdditionalData ()
{
  auto & fmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMesh(), "StokesContractMap::MapAdditionalData F-Mesh");

  const auto & f_eqc_h = *fmesh.GetEQCHierarchy();
  auto f_comm = f_eqc_h.GetCommunicator();
  auto loop_pardofs = fmesh.GetLoopUDofs().GetParallelDofs();

  // cant get this table out of pardofs ATM...
  auto loops       = fmesh.GetLoops();
  auto loop_dps    = BuildDPTable(*loop_pardofs);
  auto ghost_verts = fmesh.GetGhostVerts();
  // auto [ dofed_edges, f_dof2e, f_e2dof ] = fmesh.GetDOFedEdges();

  auto [ dofed_edges_SB, f_dof2e_SB, f_e2dof_SB ] = fmesh.GetDOFedEdges();

  auto &dofed_edges = dofed_edges_SB;
  auto &f_dof2e = f_dof2e_SB;
  auto &f_e2dof = f_e2dof_SB;

  auto group = this->GetGroup();

  // cout << " MAD contr, group " ; prow(group); cout << endl;

  // {
  //   auto & fmesh = *static_pointer_cast<THIS_CLASS>(cmap.GetMesh());
  //   auto [ c_des, c_dof2e, c_e2dof ] = fmesh.GetDOFedEdges();
  //   cout << endl << " CONTR f_ghost_vs " << endl; print_ba_as_a(*fmesh.GetGhostVerts()); cout << endl;
  //   cout << endl << " CONTR f_dof2e " << endl; prow2(c_dof2e); cout << endl << endl;
  //   cout << endl << " CONTR f_e2dof " << endl; prow2(c_e2dof); cout << endl << endl;
  // }

  /**
   * Global enum of loops makes everything so much easier.
   * I guess if i have time and want to do it i could get rid of this.
   * But for now I don't care.
   */
  Array<int> f_loop_globnums; int nl_glob;
  loop_pardofs->EnumerateGlobally(nullptr, f_loop_globnums, nl_glob);

  // cout << "f_loop_globnums " << endl; prow2(f_loop_globnums); cout << endl << endl;
  // cout << "f_loops " << endl << loops << endl;

  auto pack_table = [&](bool pack, auto & table, FlatArray<int> array) -> int {
    size_t tot_size = 1 + table.IndexArray().Size() + table.AsArray().Size();
    if (pack) {
      array[0] = table.Size();
      array.Part(1, table.IndexArray().Size()) = table.IndexArray();
      array.Part(1 + table.IndexArray().Size(), table.AsArray().Size()) = table.AsArray();
    }
    return tot_size;
  };

  auto unpack_table = [&](FlatArray<int> msg, FlatArray<size_t> fi_buffer) -> tuple<size_t, size_t, unique_ptr<FlatTable<int>>>  {
    // cout << " UNPACK TABLE " << endl;
    int cnt = 0, nrows = msg[cnt++];
    // cout << " NROWS " << nrows << " FIBS " << fi_buffer.Size() << endl;
    fi_buffer.Range(0, nrows+1) = msg.Range(cnt, cnt+nrows+1); cnt += nrows+1;
    // cout << " FI BF "; prow(fi_buffer.Range(0, nrows+1)); cout << endl;
    auto ft = make_unique<FlatTable<int>>(nrows, fi_buffer.Data(), msg.Part(cnt, 0).Data()); cnt += fi_buffer[nrows];
    // cout << " UNPACKED TABLE " << endl << *ft << endl;
    return make_tuple(cnt, nrows+1, std::move(ft));
  };

  // pack mesage:
  auto pack_message = [&](auto & msg) {
    // count
    size_t cnt = 0;
    cnt += 1 + f_loop_globnums.Size(); // len, globnums
    cnt += 1; // fi-buffer size
    cnt += pack_table(false, loops, msg); // loops
    cnt += pack_table(false, loop_dps, msg); // loop_dps
    cnt += 1 + fmesh.template GetNN<NT_VERTEX>(); // len, ghost_verts
    cnt += 1 + fmesh.template GetNN<NT_EDGE>();   // len, dofed_edges
    // pack
    msg.SetSize(cnt); cnt = 0;
    msg[cnt++] = f_loop_globnums.Size(); // globnums
    msg.Part(cnt, f_loop_globnums.Size()) = f_loop_globnums; cnt += f_loop_globnums.Size();
    msg[cnt++] = loops.IndexArray().Size() + loop_dps.IndexArray().Size(); // fi-buffer
    cnt += pack_table(true, loops, msg.Range(cnt, msg.Size())); // loops
    cnt += pack_table(true, loop_dps, msg.Range(cnt, msg.Size())); // loop_dps
    if (ghost_verts != nullptr) { // ghost_verts
      msg[cnt++] = fmesh.template GetNN<NT_VERTEX>();
      for (auto v : fmesh.template GetNodes<NT_VERTEX>())
        { msg[cnt++] = ghost_verts->Test(v) ? 1 : 0; }
    }
    else
      { msg[cnt++] = 0; }
    msg[cnt++] = fmesh.template GetNN<NT_EDGE>();
    for (auto k : Range(fmesh.template GetNN<NT_EDGE>())) // dofed_edges
      { msg[cnt++] = (dofed_edges->Test(k)) ? 1 : 0; }
  };

  auto unpack_message = [&](auto & msg, Array<size_t> & fib) -> tuple<FlatArray<int>, unique_ptr<FlatTable<int>>,
                unique_ptr<FlatTable<int>>, FlatArray<int>, FlatArray<int>> {
    size_t cnt = 0;
    size_t lgn = msg[cnt++];
    auto ftgns = msg.Part(cnt, lgn); cnt += lgn; // globnums
    // cout << " LGN " << lgn << "; frgns "; prow2(ftgns); cout << endl;
    size_t fib_size = msg[cnt++];
    // cout << " FIB SIZE " << fib_size << endl;
    fib.SetSize(fib_size);
    auto [ cl, cfi, ft_loops ] = unpack_table(msg.Part(cnt), fib); cnt += cl; // loops
    // cout << " CL CFI " << cl << " " << cfi << endl;
    auto [ cl_dps, cfi2, ft_dps ] = unpack_table(msg.Part(cnt), fib.Range(cfi, fib.Size())); cnt += cl_dps; // loop_dps
    int len_gv = msg[cnt++];
    auto gv = msg.Part(cnt, len_gv); cnt += len_gv; // ghost_verts
    int len_de = msg[cnt++];
    auto de = msg.Part(cnt, len_de); cnt += len_de; // dofed_edges
    return make_tuple(std::move(ftgns), std::move(ft_loops), std::move(ft_dps), std::move(gv), std::move(de));
  };

  // everyone sends
  Array<int> send_msg; pack_message(send_msg);

  // cout << endl << " send_msg to " << group[0] << endl; prow(send_msg); cout << endl << endl;

  auto req_send = f_comm.ISend(send_msg, group[0], MPI_TAG_AMG);
  if (!this->IsMaster()) // non group masters return
  {
    MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    return;
  }

  Array<Array<int>> recv_msgs(group.Size());
  for (auto k : Range(recv_msgs))
    { f_comm.Recv(recv_msgs[k], group[k], MPI_TAG_AMG); }
  MPI_Wait(&req_send, MPI_STATUS_IGNORE);  // wait for self-msg

  // cout << " recv_msgs " << endl;
  // for (auto k : Range(recv_msgs)) {
  //   cout << " from "  << group[k] << ": "; prow(recv_msgs[k]); cout << endl << endl;
  // }

  auto & cmesh = *my_dynamic_pointer_cast<TMESH>(this->GetMappedMesh(), "StokesContractMap::MapAdditionalData C-Mesh");
  const auto & c_eqc_h = *cmesh.GetEQCHierarchy();
  auto c_comm = c_eqc_h.GetCommunicator();

  /** unpack messages **/
  Array<Array<size_t>> fi_buffers(group.Size());
  Array<FlatArray<int>> k_globnums(group.Size()), k_gvs(group.Size()), k_des(group.Size());
  Array<unique_ptr<FlatTable<int>>> k_loops(group.Size()), k_dps(group.Size()); // Array<FT> does not work (deleted constructor)
  for (auto k : Range(group)) {
    auto [ klgn, kloops, kdps, kgvs, kdes ] = unpack_message(recv_msgs[k], fi_buffers[k]);
    // cout << " -> klgn "; prow2(klgn); cout << endl;
    k_globnums[k].Assign(klgn);
    k_loops[k] = std::move(kloops);
    k_dps[k] = std::move(kdps);
    k_gvs[k].Assign(kgvs);
    k_des[k].Assign(kdes);
    // cout << " unpacked from " << group[k] << ": ";
    // cout << " globnums "; prow2(k_globnums[k]); cout << endl;
    // cout << " loops " << endl << *k_loops[k] << endl;
    // cout << " dps " << endl << *k_dps[k] << endl;
    // cout << " gvs "; prow2(k_gvs[k]); cout << endl;
    // cout << " des "; prow2(k_des[k]); cout << endl << endl;
  }

  auto proc_map = this->GetProcMap();

  /** loop maps via glob nums **/
  Array<int> allgnums(loops.Size()*group.Size()); allgnums.SetSize0();
  for (auto k : Range(group)) {
    for (auto gn : k_globnums[k])
      { insert_into_sorted_array_nodups(gn, allgnums); }
  }
  Array<int> perow(group.Size()), perow2(group.Size());
  perow = 0; perow2 = 0;
  for (auto k : Range(perow)) {
    perow[k] = k_globnums[k].Size();
    for (auto v : k_des[k]) {
      if (v == 1)
        { perow2[k]++; }
    }
  }

  // set up BA of non-ghosts. if non-ghost anywhere -> non-ghost in contracted
  auto cgv = make_shared<BitArray>(cmesh.template GetNN<NT_VERTEX>());
  cgv->Clear();
  for (auto k : Range(group)) {
    auto v_map = this->template GetNodeMap<NT_VERTEX>(k);
    for (auto j : Range(k_gvs[k])) {
      if (k_gvs[k][j] == 0) // not ghost!
        { cgv->SetBit(v_map[j]); }
    }
  }
  cgv->Invert();

  // cout << " contr set GV " << cgv << std::endl;

  cmesh.SetGhostVerts(cgv);

  // let the mesh set up dof <-> edge maps
  // auto [ c_des, c_dof2e, c_e2dof ] = cmesh.GetDOFedEdges();

  auto [ c_des_SB, c_dof2e_SB, c_e2dof_SB ] = cmesh.GetDOFedEdges();

  auto & c_des = c_des_SB;
  auto & c_dof2e = c_dof2e_SB;
  auto & c_e2dof = c_e2dof_SB;

  // cout << endl << " CONTR c_ghost_vs " << endl; print_ba_as_a(*cmesh.GetGhostVerts()); cout << endl;
  // cout << endl << " CONTR c_dof2e " << endl; prow2(c_dof2e); cout << endl << endl;
  // cout << endl << " CONTR c_e2dof " << endl; prow2(c_e2dof); cout << endl << endl;

  // {
  //   cout << " ALL CONTR vert_map " << endl;
  //   for (auto k : Range(group))
  // 	{ cout << " CONTR vert_map " << endl; cout << k << ": "; prow2(cmap.template GetNodeMap<NT_VERTEX>(k)); cout << endl; }
  //   cout << " ALL CONTR edge_map " << endl;
  //   for (auto k : Range(group))
  // 	{ cout << " CONTR edge_map " << endl; cout << k << ": "; prow2(cmap.template GetNodeMap<NT_EDGE>(k)); cout << endl; }
  // }

  { // loop_maps gets moved out!
    Table<int> loop_maps(perow); perow = 0.0;
    Table<int> dofed_edge_maps(perow2); perow2 = 0.0;
    for (auto k : Range(group)) {
      // cout << " k " << k << " " << k_globnums[k].Size() << " " << loop_maps[k].Size() << endl;
      for (auto j : Range(k_globnums[k]))
        { loop_maps[k][j] = find_in_sorted_array(k_globnums[k][j], allgnums); }
      auto edge_map = this->template GetNodeMap<NT_EDGE>(k);
      for (auto j : Range(k_des[k])) {
        if (k_des[k][j] == 1) {
          // cout << k << " " << j << " pr " << perow2[k] << "/" << dofed_edge_maps[k].Size() << " mp " << edge_map.Size() << endl;
          dofed_edge_maps[k][perow2[k]++] = c_e2dof[edge_map[j]]; }
      }
    }
    // cout << endl << " ALL CONTR loop_maps " << endl;
    // for (auto k : Range(loop_maps)) {
    // 	cout << endl << " CONTR loop_maps " << endl;
    // 	cout << k << ": "; prow2(loop_maps[k]); cout << endl;
    // }
    // 	cout << endl << " ALL CONTR de_maps " << endl;
    // for (auto k : Range(dofed_edge_maps)) {
    // 	cout << endl << " CONTR de_maps " << endl;
    // 	cout << k << ": "; prow2(dofed_edge_maps[k]); cout << endl;
    // }
    SetLoopMaps(std::move(loop_maps));
    SetDofedEdgeMaps(std::move(dofed_edge_maps));
  }

  auto loop_maps = GetLoopMaps();

  /** loops (w.o duplicates needs removeduplicates)
   *  TAKE INTO ACCOUNT THAT EDGES CAN BE FLIPPED DURING REDIST (see below)
   */
  const size_t ncl = allgnums.Size();
  Table<int> cloops, cldps;
  {
    TableCreator<int> ccl(ncl), ccdps(ncl);
    for (; !ccl.Done(); ccl++, ccdps++) {
      for (auto k : Range(group)) {
        const BitArray& eflips = this->template GetFlipNodes<NT_EDGE>(k);
        auto e_map = this->template GetNodeMap<NT_EDGE>(k);
        for (auto j : Range(k_globnums[k])) {
          int c_loop_num = loop_maps[k][j];
          auto f_loop = (*k_loops[k])[j];
          // if (c_loop_num == 538) cout << " MAPS TO " << 358 << endl;
          // if (c_loop_num == 538) cout << " proc " << group[k] << endl;
          // if (c_loop_num == 538) { cout << k << " " << j << " fl " << f_loop.Size() << ": "; prow(f_loop); cout << endl; }
          // if (c_loop_num == 538)cout << "clnr " << c_loop_num << endl;
          for (auto l : Range(f_loop)) {
            // if (c_loop_num == 538)cout << k << " " << j << " " << l << "  ";
            int ort = (f_loop[l] > 0) ? 1 : -1, enr = ort * f_loop[l] - 1;
            // if (c_loop_num == 538)cout << ort << " " << enr << endl;
            int cenr = e_map[enr];
            if (eflips.Test(enr))
              { ort = -ort; }
            // if (c_loop_num == 538)cout << " -> " << ort << " " << cenr << endl;
          // if (c_loop_num == 538) cout << " DONE MAP TO " << 358 << endl;
            ccl.Add(c_loop_num, ort*(cenr+1));
          }
          auto f_dps = (*k_dps[k])[j];
          // dont need to add the proc itself - we know it maps TO ME
          for (auto p : f_dps) {
            if (proc_map[p] != c_comm.Rank())
              { ccdps.Add(c_loop_num, proc_map[p]); }
          }
        }
      }
    }
    auto dupl_cls = ccl.MoveTable();
    // cout << " cloops + dups = " << endl << dupl_cls << endl;
    cloops = RemoveDuplicates(std::move(dupl_cls));
    cldps = RemoveDuplicates(ccdps.MoveTable());
  }

  // cout << " contr loops " << endl << cloops << endl;
  // cout << " contr dps " << endl << cldps << endl;

  // auto clpds = make_shared<ParallelDofs>(c_eqc_h.GetCommunicator(), std::move(cldps), 1, false);

  cmesh.SetLoops(std::move(cloops));
  cmesh.SetLoopDPs(std::move(cldps));

  /**
   * Have coarse edge/vertex data - BUT FLOW ORIENTATION CAN BE WRONG
   * (whenever vertex map flips larger/smaller v-num).
   * Vertices are numbered consistently, so an edge is flipped iff. all procs flip it.
   * So we can just iterate through edge maps and check flips.
   * BUT!! there can also be edges which WE HAVE on the contracted level, but we have
   * no previous edge that maps to it!! -> NEED TO CUMULATE THIS
   */
  BitArray isflipped(cmesh.template GetNN<NT_EDGE>()); isflipped.Clear(); // do not flip multiple times
  cmesh.CumulateData(); // not sure if this is strictly needed, but will have to do it at some point anyways
  auto cedata = get<1>(cmesh.Data())->Data();
  for (auto k : Range(group)) {
    const auto& flips = this->template GetFlipNodes<NT_EDGE>(k);
    auto emap         = this->template GetNodeMap<NT_EDGE>(k);
    // cout << endl << " from " << k << " " << group[k] << ", EDGE FLIPS =  " << endl;
    // for (auto k :Range(flips.Size()))
    // { cout << "(" << k << "/" << emap[k] << "::" << (flips.Test(k)?-1:1) << ") "; }
    // cout << endl;
    for (auto j : Range(emap)) {
      if (flips.Test(j) && (!isflipped.Test(emap[j])))
        { isflipped.SetBit(emap[j]); }
      }
    }

  Array<int> flip_edge(isflipped.Size());
  for (auto k : Range(flip_edge))
    { flip_edge[k] = isflipped.Test(k) ? 1 : 0; }
  cmesh.template AllreduceNodalData<NT_EDGE>(flip_edge, [&](const auto & tin){ return sum_table(tin); }, false);
  for (auto k : Range(flip_edge)) {
    if (flip_edge[k] != 0)
      { isflipped.SetBit(k); }
  }

  // cout << endl << " coarse edge flip : " << endl;
  // for (auto k : Range(isflipped.Size())) {
  //   cout << "(" << k << "::" << (isflipped.Test(k)?-1:+1) << ") ";
  // }
  // cout << endl << endl;

  for (auto k : Range(isflipped.Size())) {
    if (isflipped.Test(k))
      { cedata[k].flow *= -1; }
  }

  /**
   * Loops are marked as "inactive" when they contain an edge with 0 flow
   * [[ an inactive loop CAN become active again due to coarsening! ]]
   */
  {
    // auto [ c_des, c_dof2e, c_e2dof ] = cmesh.GetDOFedEdges();
    auto [ c_des_SB, c_dof2e_SB, c_e2dof_SB ] = cmesh.GetDOFedEdges();
    auto & c_des = c_des_SB;
    auto & c_dof2e = c_dof2e_SB;
    auto & c_e2dof = c_e2dof_SB;

    auto cloop_pds = cmesh.GetLoopUDofs().GetParallelDofs();
    auto cloops = cmesh.GetLoops();
    Array<int> zf(cloops.Size()); zf = 0;
    for (auto k : Range(cloops)) {
      for (auto j : Range(cloops[k]))
        if ( is_zero(cedata[abs(cloops[k][j])-1].flow) )
          { zf[k] = 1; break; }
    }
    MyAllReduceDofData(*cloop_pds, zf,  [&](auto & a, const auto & b) { a += b; });
    // just .. WHY?? this is so wrong I am keeping it as a comment
    // cmesh.template AllreduceNodalData<NT_EDGE>(zf,  [&](const auto & in){ return std::move(sum_table(in)); }, false);
    auto cl_active = make_shared<BitArray>(cloops.Size());
    cl_active->Clear();
    for (auto k : Range(cloops)) {
      if (zf[k] == 0)
        { cl_active->SetBit(k); }
    }
    cmesh.SetActiveLoops(cl_active);
  }

  return;
} // StokesContractMap::MapAdditionalData


template<class TMESH>
void
StokesContractMap<TMESH> :: PrintTo (std::ostream & os, std::string prefix) const
{
  std::string const prefix2 = prefix + "  ";
  std::string const prefix3 = prefix2 + "  ";

  os << prefix << " StokesContractMap, base-ctr-map: " << endl;
  this->GridContractMap::PrintTo(os, prefix);


  if (this->IsMaster())
  {
    os << prefix << "Loop maps: " << endl;
    for (auto k : Range(GetGroup()))
    {
      os << prefix2 << "Loop-map for member #" << k << " = " << GetGroup()[k] << ": ";
      prow3(GetLoopMaps()[k], os, prefix3, 10);
      os << endl;
    }
    os << endl;

    os << prefix2 << " DOFed edge maps: " << endl;
    for (auto k : Range(GetGroup()))
    {
      os << prefix2 << "DOFed-edge-map for member #" << k << " = " << GetGroup()[k] << ": ";
      prow3(GetDofedEdgeMaps()[k], os, prefix3, 10);
      os << endl;
    }
    os << endl;
  }

}

/** StokesContractMap **/


} // namespace amg

#endif // FILE_STOKES_MAP_IMPL_HPP