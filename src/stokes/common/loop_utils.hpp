#ifndef FILE_LOOP_UTILS_HPP
#define FILE_LOOP_UTILS_HPP

#include <base.hpp>
#include <utils_arrays_tables.hpp>

/**
 *  Confine some of this loop-sorting garbage to this header so the
 *  rest of the Stokes code becomes a bit more readable
 */

namespace amg
{

enum ORIENTATION : bool
{
  NEGATIVE = false,
  POSITIVE = true
};


INLINE tuple<ORIENTATION, int> loopDecode(int k)
{
  return make_tuple(ORIENTATION(k > 0), abs(k) - 1);
}


INLINE int loopEncode(int vStart, int vStop, int enr)
{
  // return ( vStart < vStop ? 1 : -1 ) * ( enr + 1 );
  return vStart < vStop ? ( enr + 1 ) : ( -enr - 1 );
}

/** Orients and sorts (partial) loop of some kind, **/
template<class TLD, class TFLIP, class TORIENT>
INLINE int SortAndOrientLoop (FlatArray<TLD> loop, LocalHeap & lh, TFLIP flip, TORIENT orient)
{
  HeapReset hr(lh);

  // cout << " SortAndOrientLoop, IN = "; prow2(loop); cout << endl;

  int ls = loop.Size(), cnt = 0;
  FlatArray<int> inds(ls, lh), used(ls, lh);
  used = 0; inds = -1;
  auto add_ind = [&](auto ort, auto k) {
    // cout << " add_ind " << ort << " " << k << endl;
    inds[cnt++] = k;
    used[k] = ort;
  };

  auto extend_chunk = [&](TLD start, bool flipstart) {
    TLD curr = start;
    bool flipcurr = flipstart;
    bool foundone = false;
    int start_ort = flipstart ? -1 : 1;
    do {
      foundone = false;
      for (auto k : Range(loop)) {
        if (used[k] == 0) {
          // 1..fits, -1..flipped fits, 0..does not fit
          int ort = orient(curr, flipcurr, loop[k]);
          // cout << " orient " << curr << " (flip " << flipcurr << ")" << flipstart << " " << loop[k] << ", k " << k << " -> " << ort << endl;
          if (ort != 0) {
            // if start is flipped (extend front), flip an additional time!
            add_ind(ort*start_ort, k);
            curr = loop[k];
            flipcurr = ort < 0;
            foundone = true;
            break;
          }
        }
      }
    } while (foundone);
  };

  int n_chunks = 0;
  while(cnt < ls) {
    /** look for a place to start a chunk (loop can be disconnected...) **/
    // cout << "   chunk @" << cnt << endl;
    TLD start, curr;
    for (auto k : Range(loop))
      if (used[k] == 0) {
        start = loop[k];
        add_ind(1, k);
        break;
      }
    n_chunks++;
    /** extend in back **/
    extend_chunk(start, false);
    /** extend in front **/
    extend_chunk(start, true);
  }
  for (auto k : Range(used))
    if (used[k] == -1)
      { flip(loop[k]); }
  // cout << " inds "; prow(inds); cout << endl;
  ApplyPermutation(loop, inds);
  return n_chunks;
} // SortLoop
  
  
/**
 * Sort a list of chunks such that they fit together
 * 
 * A "chunk" (oriented part of a loop) is given as [eqid_i, num_i, eqid_j, num_j]
 * where eqid_i and num_i are the eqc-id and eqc-local number of the "start" of the chunk
 * and eqid_j/num_j that of the "end".
 * 
 * A "chunk" can have duplicate entries!
 */
template<class TDATA>
INLINE Array<INT<4, int>> sortChunks(TDATA &indata)
{
  Array<INT<4, int>> out;

  // cout << endl << "LSC " << endl;
  // for (auto k : Range(indata)) {
  // cout << k << ": "; prow(indata[k]); cout << endl;
  // }
  // cout << endl;
  /** out-size probably smaller than sum of in sizes (loop chunks overlap) **/

  /** Start with first, non-empty junk **/
  int nchunks = indata.Size();

  if (nchunks == 0) // no chunks ?? - should not happen
    { throw Exception("AHMM I think there should always be 1+ chunks...."); }

  if (nchunks == 1) { // only 1 chunk - can absolutely happen (fully solid loop that is shared b/c overlap)
    out.SetSize(indata[0].Size());
    out = indata[0];
    return out;
  }

  int maxsize = std::accumulate(indata.begin(), indata.end(), int(0),
        [&](int s, FlatArray<INT<4, int>> ar) { return s + int(ar.Size()); });


  INT<4, int> curr({ -1, -1, -1, -1 }), first({ -1, -1, -1, -1 });

  /** find an edge to start the loop with **/

  Array<INT<3, int>> iios(maxsize); // (row,col,orient) for the output
  int cnt = 0;
  bool foundone = false;

  // cout << " sizes " << endl;
  // for (int k : Range(indata))
  // cout << indata[k].Size() << " ";
  // cout << "-> maxsize = " << maxsize << endl;

  for (int k : Range(indata))
    if (indata[k].Size()) {
      curr = indata[k][0];
      first = curr;
      iios[cnt++] = INT<3, int>({ k, 0, 1 });
      indata[k][0][0] *= -1;
      // cout << " start w. " << indata[k][0] << ", +1" << endl;
      foundone = true;
      break;
    }
  if (!foundone)
    { return out; }

  // check if "osi" part of chunk tupi fits with "osj" part of chunk tup j
  auto issame = [&](const auto & tupi, int osi, const auto & tupj, int osj) {
    return ( tupi[osi] == tupj[osj] ) && ( tupi[osi + 1] == tupj[osj + 1] );
  };
  
  auto flip_chunk = [&](auto &chunk) {
      swap(chunk[0], chunk[2]);
      swap(chunk[1], chunk[3]);
  };

  auto add_chunk = [&](int k, int j, bool flip) {
    iios[cnt++] = INT<3, int>({ k, j, flip ? 1 : -1 });
    int osi = flip ? 2 : 0;
    int osj = flip ? 2 : 0;
    curr = indata[k][j];
    indata[k][j][0] *= -1;
    if (flip)
      { flip_chunk(curr); }
  };

  /** extend loop in back, [..., curr] -> [..., curr, new] **/

  foundone = true;
  while(foundone) {
    // look for next edge
    foundone = false;
    for (int k : Range(indata)) {
      for (int j : Range(indata[k])) {
        auto &datakj = indata[k][j];
        if (datakj[0] > 0) { // we negate eqid0 for "used" ones
          // cout << k << " I/" << j << " " << datakj << " look for  " << curr;
          if ( issame(datakj, 0, curr, 2) ) { // new-fit or dup-flip
            if ( issame(datakj, 2, curr, 0) ) // dup-flip
              { datakj[0] *= -1; /*cout << " dup-flip " << endl;*/ }
            else { // new-fit
              add_chunk(k, j, false);
              foundone = true;
              // cout << " -> ort +1!" << endl;
              break; // still need to break out of k loop
            }
          }
          else if ( issame(datakj, 2, curr, 2) ) { // new-flip or dup-fit
            if ( issame(datakj, 0, curr, 0) ) // dup-fit
              { datakj[0] *= -1; /* cout << " dup-fit " << endl; */ }
            else { // new-flip
              add_chunk(k, j, true);
              foundone = true;
              // cout << " -> ort -1!" << endl;
              break; // still need to break out of k loop
            }
          }
          // cout << " no fit " << endl;
        }
      }
      if (foundone)
        { /* cout << " break k " << endl; */ break; }
    }
  }

  /** extend loop in front, [curr, ...] -> [new, curr, ...] **/

  int in_front_from_here = cnt; // [0..bfh) are in order, [bfh..end) are in reverse order
  curr = first; // start with first again
  foundone = true;
  while(foundone) {
    foundone = false;
    for (int k : Range(indata)) {
      for (int j : Range(indata[k])) {
        auto &datakj = indata[k][j];
        if (datakj[0] > 0) { // we negate eqid0 for "used" ones
          // cout << "II/" << k << " " << j << " " << datakj << " look for  " << curr;
          if ( issame(datakj, 2, curr, 0) ) { // new-fit or dup-flip
            if ( issame(datakj, 0, curr, 2) ) //dup-flip
              { datakj[0] *= -1; /* cout << " dup-flip " << endl; */ }
            else { // new-fit
              add_chunk(k, j, false);
              foundone = true;
              // cout << " -> ort II/+1!" << endl;
              break; // still need to break out of k loop
            }
          }
          else if ( issame(datakj, 0, curr, 0) ) { // new-flip or dup-fit
            if ( issame(datakj, 2, curr, 2) ) // dup-fit
              { datakj[0] *= -1; /* cout << " dup-fit " << endl; */ }
            else { // new-flip
              add_chunk(k, j, true);
              foundone = true;
              // cout << " -> ort II/-1!" << endl;
              break; // still need to break out of k loop
            }
          }
          // cout << " no fit " << endl;
        }
      }
      if (foundone)
        { /* cout << " break k " << endl; */ break; }
    }
  }

  out.SetSize(cnt);

  int n_in_front = cnt - in_front_from_here;

  auto set_out = [&](int k, const auto & iio) {
    out[k] = indata[iio[0]][iio[1]];
    out[k][0] *= -1;
    if (iio[2] < 0)
      { flip_chunk(out[k]); }
  };

  for (auto k : Range(n_in_front)) // [front, start)
    { set_out(k, iios[cnt - 1 - k]); }

  for (auto k : Range(in_front_from_here)) // [start, back]
    { set_out(n_in_front + k, iios[k]); }

  // for closed loops, first and last are still the same, re-use curr/first
  if ( (cnt > 1) && (out[0] == out.Last()) )
    { out.SetSize(out.Size() - 1); }

  return out;
} // sortChunks


/** 
 * This might fit better into one of the parallel headers, but we only need it in
 * Stokes for the loops, so we leave It here
 */
INLINE tuple<Table<int>, Array<int>> ShrinkDPTable (BitArray & remove_me,
                                                    FlatTable<int> dist_procs,
                                                    FlatArray<int> all_dps,
                                                    NgsAMG_Comm comm)
{

  // cout << " ShrinkDPTable by " << remove_me << endl;
  // cout << " dist-procs " << endl << dist_procs << endl;
  // cout << " all_dps "; prow2(all_dps); cout << endl;

  /**
   *  removes proc from dp-tables where bitarray is set.
   * output: < new_dp_table, row_map >
   */
  int N = dist_procs.Size(), n = 0;
  Array<int> row_map(N), perow(all_dps.Size());
  row_map = -1; perow = 0;
  for (auto k : Range(N)) {
    if (!remove_me.Test(k))
      { row_map[k] = n++; }
    for (auto p : dist_procs[k])
      { perow[find_in_sorted_array(p, all_dps)]++; }
  }

  // cout << " rm "; prow2(row_map); cout << endl;

  Table<int> send_data(perow), recv_data(perow);
  perow = 0;
  for (auto k : Range(N)) {
    int istay = remove_me.Test(k) ? 0 : 1;
    for (auto p : dist_procs[k]) {
      auto kp = find_in_sorted_array(p, all_dps);
      send_data[kp][perow[kp]++] = istay;
    }
  }

  // cout << " send_data " << endl << send_data << endl;
  ExchangePairWise(comm, all_dps, send_data, recv_data);
  // cout << " recv_data " << endl << recv_data << endl;

  TableCreator<int> cndps(n);
  for (; !cndps.Done(); cndps++) {
    perow = 0;
    for (auto k : Range(N)) {
      for (auto p : dist_procs[k]) {
        auto kp = find_in_sorted_array(p, all_dps);
        auto ikp = perow[kp]++;
        if ( (recv_data[kp][ikp] == 1) && (!remove_me.Test(k)) )
          { cndps.Add(row_map[k], p); }
      }
    }
  }

  auto new_dps = cndps.MoveTable();

  // cout << " -> new_dps " << endl << new_dps << endl;

  return make_tuple(std::move(new_dps), std::move(row_map));
} // ShrinkDPTable


/** 
 *  Rotate vertex-loop (eqid, vloc) such that it starts at:
 *       - closed loop: the smallest vertex
 *       - open   loop: the smallest END-vertex
 *  CLOSED loops are also oriented such that the second vertex is the smaller
 *  one of the neighbors of the first.
 * 
 *  The purpose is that it makes loops bucket-able by their first vertex,
 *  which makes identifying duplicates relatively. 
 */
INLINE void rotateVLoop(FlatArray<INT<2, int>> slk, Array<INT<2, int>> &rot_buffer)
{
  // auto smaller = [&](const auto & ta, const auto & tb) {
    // if (ta[0] == tb[0]) { return ta[1] < tb[1]; }
    // else { return ta[0] < tb[0]; }
  // };
  bool closed = slk[0] == slk.Last();
  int sslk = slk.Size();
  if (closed) {
    // new start index: start at "smallest" vnr
    int nsi = 0; INT<2, int> vstart = slk[0];
    for (auto j : Range(slk))
      if (slk[j] < vstart)
        { nsi = j; vstart = slk[j]; }
    // cout << "  rotateVLoop, start with " << vstart << " @ " << nsi << std::endl;
    // new direction: flip towards "smaller" neib
    bool flip = false;
    // Note/TODO: I think flip is wrong this way?? (">" does not exist) Should not matter, only needs to be bucketable!
    if ( (0 < nsi) && (nsi + 1 < sslk) ) // start in the middle
      { flip = slk[nsi+1] < slk[nsi-1]; }
    else // closed (so len of array >= 4) + at an end [A,B,...,C,A] -> flip determined by B,C
      { flip = slk[sslk-2] < slk[1]; }
    // cout << "  FLIP ? " << flip << endl;
    if ( flip || (nsi != 0) ) {
      rot_buffer.SetSize(sslk);
      int c = 0;
      if (flip) {
        for (int j = nsi; j >0; j--)
         { rot_buffer[c++] = slk[j]; }
        for (int j = sslk-1; j >= nsi; j-- )
          { rot_buffer[c++] = slk[j]; }
      } else {
        for (auto j : Range(nsi, sslk))
          { rot_buffer[c++] = slk[j]; }
        for (auto j : Range(1, nsi+1))
          { rot_buffer[c++] = slk[j]; }
      }
      slk = rot_buffer;
    }
  }
  else { // open loop
    if (slk.Last() < slk[0]) {
      rot_buffer.SetSize(slk.Size());
      for (auto k : Range(slk))
        { rot_buffer[k] = slk[sslk-1-k]; }
      slk = rot_buffer;
    }
  }
} // rotateVLoop


/**
 *  Gets a loop (or loop-chunk) as list of (eqid, lv), splits it into parts that do not contain cycles
 *  and adds the resulting loops to the output
 * 
 *  NOTE: This calls itself, so it cannot be inlined!
 */
void SplitChunk (FlatArray<INT<2, int>> chunk, Array<Array<INT<2, int>>> & split_out, LocalHeap & lh);


/**
 *  Gets a table, every row is a loop-chunk as a list of (eqid0, lv0, eqid1, lv1) tuples
 *      !! the input junks must NOT HAVE OVERLAP !!
 *  Merges the chunks together, if necessary splits them into sub-loops that contain no
 *  cycles, and writes into the output AS "simple" loops with (eqid, lv) entries.
*/
void SimpleLoopsFromChunks (FlatArray<FlatArray<INT<4, int>>> the_data, Array<Array<INT<2, int>>> & simple_loops, LocalHeap & lh);
  
} // namespace amg


#endif //FILE_LOOP_UTILS_HPP
