#include "loop_utils.hpp"

namespace amg
{

// TODO: there must be a better way to do this, right?
//       should measure how critical this code even is
void SplitChunk (FlatArray<INT<2, int>> chunk, Array<Array<INT<2, int>>> & split_out, LocalHeap & lh)
{
  HeapReset hr(lh);

  int const S = chunk.Size();

  if (S < 2)
    { return; }

  if ( (S == 2) && (chunk[0] == chunk[1]) ) // [A, ... A, ... A] leads to bnd-array [A,A] because of overlap
    { return; }

  auto smaller = [&](auto a, auto b) {
    if (a[0] == b[0]) { return a[1] < b[1]; }
    else { return a[0] < b[0]; }
  };

  bool closed = (chunk[0] == chunk.Last());

  // sort chunk so we can find duplicate entries
  // FlatArray<int> inds(S, lh);
  // for (auto k : Range(inds))
  //   { inds[k] = k; }
  auto inds = makeSequence(S, lh);

  QuickSortI(chunk, inds, smaller);

  for (int k = 0; k + 1 < S; k++ ) {
    int j = k + 1; // j is "next"
    while ( (j < S) && (chunk[inds[j]] == chunk[inds[k]]) )
      { j++; }
    int sss = j - k;
    if (sss < 2) // value chunk[k] only occurs once
      { continue; }
    // cout << " handle INDS [" << k << ", " << j << ")" << endl;
    FlatArray<int> specinds(sss, lh);
    for (auto l : Range(specinds))
      { specinds[l] = inds[k+l]; }
    QuickSort(specinds);
    // cout << " specinds "; prow(specinds); cout << endl;
    if ( (sss == 2) && (specinds[0] == 0) && (specinds[1] == S-1) ) // first/last of closed loop -> ignore duplicate
      { /** cout << " just first/last of closed loop -> ignore duplicate!" << endl; **/ continue; }
    // the split that goes over array "end" - copy to a help array
    // [spi[-1] .. S-1] \cup [0..spi[0]]
    int s1 = S - specinds.Last(), s2 = specinds[0] + (closed ? 0 : 1);
    FlatArray<INT<2, int>> bnd_arr(s1 + s2, lh);
    bnd_arr.Part(0 , s1)    = chunk.Part(specinds.Last(), s1);
    bnd_arr.Part(s1, s1+s2) = chunk.Part(closed ? 1 : 0, s2);
    // cout << " s1 s2 " << s1 << " " << s2 << endl;
    // cout << " bnd_arr = "; prow(bnd_arr); cout << endl;
    SplitChunk(bnd_arr, split_out, lh);
    // rest of splits are "normal" [specinds[i], specinds[i+1]]
    for (int l : Range(sss-1)) // j-k >= 2
      { SplitChunk(chunk.Range(specinds[l], specinds[l+1]+1), split_out, lh); }
    return;
  }
  // no contained cycles found - append array!
  // cout << " no contained cycles found - append array! "; prow(chunk); cout << endl;
  split_out.SetSize(split_out.Size() + 1);
  split_out.Last().SetSize(chunk.Size());
  split_out.Last() = chunk;
} // SplitChunk

void SimpleLoopsFromChunks (FlatArray<FlatArray<INT<4, int>>> the_data, Array<Array<INT<2, int>>> & simple_loops, LocalHeap & lh)
{
  // TODO: can this be simplified for only single chunks??
  // cout << "SimpleLoopsFromChunks, data (t4) =  " << the_data.Size() << endl;
  // for (auto k : Range(the_data))
    // {  cout << k << ": "; prow(the_data[k]); cout << endl; }
  Array<INT<2, int>> sub_chunk; // need dynamically resizable memory, this is the lazy way to do it
  sub_chunk.SetSize0();
  bool foundany = false;
  int ndone = 0, totsize = std::accumulate(the_data.begin(), the_data.end(), int(0), [&](auto a, auto b) -> int { return a + b.Size(); });
  // cout << " totsize " << totsize << endl;
  auto tick_kj = [&](auto k, auto j) {
    ndone++;
    the_data[k][j][0] *= -1; // !! flip first entry to keep track of which entries are done
    foundany = true;
  };
  auto fits = [&](const auto & ta, int ia, const auto & tb) {
    // cout << " fit " << ta << " i" << ia << " and " << tb << " -> " << ( (ta[2*ia] == tb[0]) && (ta[2*ia+1] == tb[1]) ) << endl;
    return (ta[2*ia] == tb[0]) && (ta[2*ia+1] == tb[1]);
  };
  while (ndone < totsize) {
    // cout << " wl " << ndone << " " << totsize << endl;
    if (sub_chunk.Size() == 0) { // look for a new edge to start from!
      foundany = false;
      for (auto k : Range(the_data)) {
        for (auto j : Range(the_data[k])) {
          if (the_data[k][j][0] > 0) { // flipped if entry
            sub_chunk.Append(INT<2, int>({ the_data[k][j][0], the_data[k][j][1] }));
            sub_chunk.Append(INT<2, int>({ the_data[k][j][2], the_data[k][j][3] }));
            tick_kj(k, j);
            break;
          }
        }
        if (foundany)
          { break; }
      }
    }
    if (ndone == totsize) // no need to interate through again
      { break; }
    foundany = false;
    for (auto k : Range(the_data)) {
      for (auto j : Range(the_data[k])) {
        // cout <<" check " << k << " " << j << " " << (the_data[k][j][0] > 0) << endl;
        if (the_data[k][j][0] > 0) { // flipped if entry
          if (fits(the_data[k][j], 0, sub_chunk.Last())) {
            // cout << " append " << k << " " << j << " data " << the_data[k][j] << endl;
            sub_chunk.Append(INT<2, int>({ the_data[k][j][2], the_data[k][j][3]}));
            tick_kj(k, j);
            // cout << " sub_chunk now "; prow(sub_chunk); cout << endl;
          }
          else if (fits(the_data[k][j], 1, sub_chunk[0])) {
            // cout << " prepend " << k << " " << j << " data " << the_data[k][j] << endl;
            sub_chunk.Insert(0, INT<2, int>({ the_data[k][j][0], the_data[k][j][1]}));
            tick_kj(k, j);
            // cout << " sub_chunk now "; prow(sub_chunk); cout << endl;
          }
        }
      }
    }
    if (!foundany) // lucky breakdown: chunk is already a (sub-)loop, but not necessarily a simple one
      { SplitChunk(sub_chunk, simple_loops, lh); sub_chunk.SetSize0(); }
  }
  if (sub_chunk.Size())
    { SplitChunk(sub_chunk, simple_loops, lh); }
  // flip first entry back
  for (auto k : Range(the_data)) {
    for (auto j : Range(the_data[k]))
      { the_data[k][j][0] *= -1; }
  }
  // cout << "   -> OUT: " << endl;
  // for (auto k : Range(simple_loops))
  // {
  //   cout << "      " << k << "/" << simple_loops.Size() << "(s = " << simple_loops[k].Size()<< "): ";
  //   prow2(simple_loops[k]);
  //   cout << endl;
  // }
} // SimpleLoopsFromChunks


// /**
//  *  Orients and sorts a loop, taking into account that the input loop may
//  *  split up into multile smaller loops.
//  *  This can, for example, happen due to "definedon" of the FESpace:
//  *           D | 0     D == defined el, 0 == not defined
//  *          ---v---    vertex "v" should be 2 seperate loops (BL-TL-TR and TR-BR-BL)
//  *           0 | D
//  *  The input loop is sorted into one chunk after another, and the offset-array pointing
//  *  to the chunk starts is returned.
//  * 
//  *  I AM NOT USING THIS ATM because
//  *    - I have no case to test that this actually works
//  *    - it makes other code more complicated
//  */
// template<class TLD, class TFLIP, class TORIENT>
// INLINE FlatArray<int> SortAndOrientLoopV2 (FlatArray<TLD> loop, LocalHeap & lh, TFLIP flip, TORIENT orient)
// {
//   FlatArray<int> chunkPtr(loop.Size() + 1, lh); // !! before HeapReset since we return this!

//   HeapReset hr(lh);

//   int ls = loop.Size(), cnt = 0;
//   FlatArray<int> inds(ls, lh), used(ls, lh);
//   used = 0; inds = -1;
//   auto add_ind = [&](auto ort, auto k) {
//     // cout << " add_ind " << ort << " " << k << endl;
//     inds[cnt++] = k;
//     used[k] = ort;
//   };

//   auto extend_chunk = [&](TLD start, bool flipstart) {
//     TLD curr = start;
//     bool flipcurr = flipstart;
//     bool foundone = false;
//     int start_ort = flipstart ? -1 : 1;
//     do {
//       foundone = false;
//       for (auto k : Range(loop)) {
//         if (used[k] == 0) {
//           // 1..fits, -1..flipped fits, 0..does not fit
//           int ort = orient(curr, flipcurr, loop[k]);
//           // cout << " orient " << curr << " " << flipstart << " " << loop[k] << ", k " << k << " -> " << ort << endl;
//           if (ort != 0) {
//             // if start is flipped (extend front), flip an additional time!
//             add_ind(ort*start_ort, k);
//             curr = loop[k];
//             flipcurr = ort < 0;
//             foundone = true;
//             break;
//           }
//         }
//       }
//     } while (foundone);
//   };

//   int n_chunks = 0;
//   while(cnt < ls) {
//     chunkPtr[n_chunks] = cnt;
//     /** look for a place to start a chunk (loop can be disconnected...) **/
//     TLD start, curr;
//     for (auto k : Range(loop))
//       if (used[k] == 0) {
//         start = loop[k];
//         add_ind(1, k);
//         break;
//       }
//     n_chunks++;
//     /** extend in back **/
//     extend_chunk(start, false);
//     /** extend in front **/
//     extend_chunk(start, true);
//   }

//   chunkPtr[n_chunks] = ls;
//   chunkPtr.Assign(chunkPtr.Data(), n_chunks + 1);

//   for (auto k : Range(used))
//     if (used[k] == -1)
//       { flip(loop[k]); }

//   // cout << " inds "; prow(inds); cout << endl;

//   ApplyPermutation(loop, inds);

//   return chunkPtr;
// } // SortLoopV2


} // namespace amg