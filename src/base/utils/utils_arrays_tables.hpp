#ifndef FILE_UTILS_ARRAYS_TABLES_HPP
#define FILE_UTILS_ARRAYS_TABLES_HPP

#include <base.hpp>

namespace amg
{

template<typename T>
INLINE bool lex_smaller (FlatArray<T> ara, FlatArray<T> arb)
{
  const int sa = ara.Size();
  if (sa == arb.Size())
    for (auto j : Range(sa)) {
  if (ara[j] < arb[j])
    { return true; }
  else if (ara[j] != arb[j])
    { return false; }
  }
  return sa < arb.Size();
}

// like GetPositionTest of NGSolve-SparseMatrix
template<typename T>
INLINE size_t find_in_sorted_array (const T & elem, FlatArray<T> a)
{
  size_t first(0), last(a.Size());
  while (last > first+5) {
    size_t mid = (last+first)/2;
    if (a[mid] > elem)
      { last = mid; }
    else {
      if (a[mid] == elem)
        { return mid; }
      first = mid+1;
    }
  }
  for (size_t i = first; i < last; i++)
    if (a[i] == elem)
      { return i; }
  // return numeric_limits<typename remove_reference<decltype(a[0])>::type>::max();
  return (typename remove_reference<decltype(a[0])>::type)(-1);
};

// like GetPositionTest of NGSolve-SparseMatrix
template<typename T, class TS>
INLINE size_t find_in_sorted_array (const T & elem, FlatArray<T> a, TS smaller)
{
  size_t first(0), last(a.Size());
  while (smaller(first+5, last)) {
    size_t mid = (last+first)/2;
    if (smaller(elem, a[mid]))
      { last = mid; }
    else {
      if (a[mid] == elem)
        { return mid; }
      first = mid+1;
    }
  }
  for (size_t i = first; i < last; i++)
    if (a[i] == elem)
      { return i; }
  // return numeric_limits<typename remove_reference<decltype(a[0])>::type>::max();
  return (typename remove_reference<decltype(a[0])>::type)(-1);
};


// find position such thata a[0..pos) <= elem < a[pos,a.Size())
template<typename T>
INLINE size_t merge_pos_in_sorted_array (const T & elem, FlatArray<T> a)
{
  size_t first(0), last(a.Size());
  if (last == 0)
    { return 0; } // handle special cases so we still get -1 if garbage
  else if (elem < a[0])
    { return 0; }
  else if (elem >= a.Last())
    { return last; }
  while (last > first+5) {
    size_t mid = (last + first)/2; // mid>0!
    if ( a[mid] <= elem ) // search right
      { first = mid; }
    else if ( a[mid-1] > elem ) // search left
      { last = mid; }
    else // a[mid-1] <= elem < a[mid]!!
      { return mid; }
  }
  for (size_t i = first; i < last; i++)
    if (a[i] > elem)
      { return i; }
  // return (typename remove_reference<decltype(a[0])>::type)(-1);
  return size_t(-1);
};

template<typename T>
INLINE bool insert_into_sorted_array_nodups (T elem, Array<T> & a)
{
  int pos = merge_pos_in_sorted_array(elem, a);
  if ( (pos == -1) || ((pos > 0) && (a[pos-1] == elem)) )
    {  return false; }
  else
    {  a.Insert(pos, elem); return true; }
}

template<typename T>
INLINE void insert_into_sorted_array (T elem, Array<T> & a)
{
  a.Insert(merge_pos_in_sorted_array(elem, a), elem);
}

template<typename T>
INLINE std::tuple<bool, int> sorted_insert_unique (T elem, Array<T> & a)
{
  int pos = merge_pos_in_sorted_array(elem, a);
  if ( (pos == -1) || ((pos > 0) && (a[pos-1] == elem)) )
    {  return std::make_tuple(false, pos); }
  else
    {  a.Insert(pos, elem); return std::make_tuple(true, pos); }
}

template<typename T>
INLINE int sorted_pos (T elem, Array<T> & a)
{
  int pos = merge_pos_in_sorted_array(elem, a);
  a.Insert(pos, elem);
  return pos;
}

template<class T>
INLINE int removeEntryFlat (int pos, FlatArray<T> a)
{
  int const oldSize = a.Size();
  int const newSize = a.Size() - 1;

  a.Range(pos, newSize) = a.Range(pos + 1, oldSize);

  return newSize;
}

template<class T>
INLINE int removeEntry (int pos, Array<T> & a)
{
  int newSize = removeEntryFlat(pos, a);
  a.SetSize(newSize);
  return newSize;
}

template<typename T> INLINE int ind_of_max (FlatArray<T> a)
{
  if (!a.Size())
    { return -1; }
  int ind = 0; T max = a[ind]; auto as = a.Size();
  for (int k = 1; k < as; k++)
    if (a[k] > max)
      { max = a[k]; ind = k; }
  return ind;
}

/**
 * Reorder array with permutation inline:
 *   sorted[k] = orig[perm[k]]
*/
template<typename T> INLINE void ApplyPermutation (FlatArray<T> array, FlatArray<int> perm)
{
  for (auto i : Range(array)) {
    // swap through one circle; use target as buffer as buffer
    if (perm[i] == -1)
      { continue; }
    int check = i; int from_here;
    while ( ( from_here = perm[check]) != i ) { // stop when we are back
      swap(array[check], array[from_here]);
      perm[check] = -1; // have right ntry for pos. check
      check = from_here; // check that position next
    }
    perm[check] = -1;
  }
} // ApplyPermutation

template<typename T>
INLINE void
ApplyAndKeepPermutation (FlatArray<T> array, FlatArray<int> perm)
{
  for (auto i : Range(array)) {
    // swap through one circle; use target as buffer as buffer
    if (perm[i] < 0)
      { continue; }
    int check = i; int from_here;
    while ( ( from_here = perm[check]) != i ) { // stop when we are back
      swap(array[check], array[from_here]);
      perm[check] = -1 - perm[check]; // have right ntry for pos. check
      check = from_here; // check that position next
    }
    perm[check] = -1 - perm[check];
  }
  for (auto k : Range(array))
  {
    perm[k] = -perm[k] - 1;
  }
} // ApplyAndKeepPermutation

/**
 * Invert permutation:
 *   perm[invPerm[k]] == k -> sorted[invPerm[k]] = orig[perm[invPerm]] = orig
 * 
 */
template<typename T> INLINE void InvertPermutation(FlatArray<T> perm, FlatArray<T> invPerm)
{
  for (auto l : Range(perm))
  {
    invPerm[perm[l]] = l;
  }
}


/**
 * Reorder array with permutation, use buffer:
 *   buffer = array
 *   array[k] = buffer[perm[k]]
*/
template<typename T> INLINE void ApplyPermutation (FlatArray<T> array, FlatArray<int> perm, FlatArray<T> buffer)
{
  for (auto k : Range(array))
    { buffer[k] = array[k]; }
  for (auto k : Range(array))
    { array[k] = buffer[perm[k]]; }
} // ApplyPermutation


template<class T1, class T2, class T3>
INLINE void merge_arrays (T1& tab_in, Array<T2> & out, T3 lam_comp)
{
  const size_t nrows = tab_in.Size();
  size_t max_size = 0;
  for (size_t k = 0; k < nrows; k++ )
    if (tab_in[k].Size()>max_size)
  max_size = tab_in[k].Size();
  //probably not gonna add more than 5%??
  max_size += (105*max_size)/100; // why +=, not =??
  out.SetSize(max_size); out.SetSize0();
  Array<size_t> count(nrows);
  count = 0;
  size_t ndone = 0;
  BitArray hasmin(nrows);
  hasmin.Clear();
  int rofmin = -1;
  for (size_t k = 0;((k<nrows)&&(rofmin==-1));k++)
    { if (tab_in[k].Size()>0) rofmin = k; }
  if (rofmin==-1) { return; }
  // cerr << "rofmin so far: " << rofmin << endl;
  for (size_t k = 0; k < nrows; k++ )
    if (tab_in[k].Size()==0)
      { ndone++; }
  auto min_datum = tab_in[rofmin][0];
  while (ndone < nrows) {
    for (size_t k = 0; k < nrows; k++ ) {
      if (count[k]<tab_in[k].Size()) {
        if (tab_in[k][count[k]] == min_datum)
          { hasmin.SetBit(k); }
        else if (lam_comp(tab_in[k][count[k]], min_datum)) {
          hasmin.Clear();
          hasmin.SetBit(k);
          min_datum = tab_in[k][count[k]];
          rofmin = k;
        }
      }
    }
    for (size_t k = 0; k < nrows; k++) {
      if (hasmin.Test(k)) {
        count[k]++;
        if (count[k]==tab_in[k].Size())
          { ndone++; }
      }
    }
    out.Append(min_datum);
    rofmin = -1;
    for (size_t k=0; (k < nrows) && (rofmin == -1); k++) {
      if (count[k]<tab_in[k].Size()) {
        rofmin = k;
        min_datum = tab_in[k][count[k]];
      }
    }
    hasmin.Clear();
  }
  return;
};

template<class T> struct tab_scal_trait {
  typedef typename std::remove_pointer<typename std::result_of<decltype(&T::Data)(T)>::type>::type T0;
  typedef typename std::remove_pointer<typename std::result_of<decltype(&T0::Data)(T0)>::type>::type type;
};

template<class T1, class T3>
INLINE Array<typename tab_scal_trait<T1>::type> merge_arrays (T1& tab_in, T3 lam_comp)
{
  Array<typename tab_scal_trait<T1>::type> out;
  merge_arrays(tab_in, out, lam_comp);
  return out;
}

template<class T>
INLINE Array<typename tab_scal_trait<T>::type> merge_arrays (T& tab_in)
{
  return merge_arrays(tab_in, [&](const auto & i, const auto & j) LAMBDA_INLINE { return i<j; });
}

template<class T>
INLINE FlatArray<T> merge_arrays_lh (FlatArray<T> a, FlatArray<T> b, LocalHeap & lh)
{
  const int sa = a.Size(), sb = b.Size();
  FlatArray<T> out(sa+sb, lh);
  int ia = 0, ib = 0, io = 0;
  while ( (ia < sa) && (ib < sb) ) {
    if (a[ia] == b[ib])
      { out[io++] = a[ia++]; ib++; }
    else if (a[ia] < b[ib])
      { out[io++] = a[ia++]; }
    else
      { out[io++] = b[ib++]; }
  }
  while (ia < sa)
    { out[io++] = a[ia++]; }
  while (ib < sb)
    { out[io++] = b[ib++]; }
  out.Assign(out.Part(0, io));
  return out;
} // merge_arrays_lh


template<class T>
INLINE void merge_arrays (FlatArray<T> a, FlatArray<T> b, Array<T> &out)
{
  const int sa = a.Size(), sb = b.Size();
  out.SetSize(sa+sb);
  int ia = 0, ib = 0, io = 0;
  while ( (ia < sa) && (ib < sb) ) {
    if (a[ia] == b[ib])
      { out[io++] = a[ia++]; ib++; }
    else if (a[ia] < b[ib])
      { out[io++] = a[ia++]; }
    else
      { out[io++] = b[ib++]; }
  }
  while (ia < sa)
    { out[io++] = a[ia++]; }
  while (ib < sb)
    { out[io++] = b[ib++]; }
  out.SetSize(io);
} // merge_arrays_lh

template<class T>
INLINE void merge_a_into_b (FlatArray<T> a, Array<T> &b, LocalHeap &lh)
{
  FlatArray<int> bcopy(b.Size(), lh);
  bcopy = b;
  merge_arrays(a, bcopy, b);
} // merge_arrays_lh


template<class T>
FlatArray<T> posSortedInSorted(FlatArray<T> a, FlatArray<T> b, LocalHeap &lh)
{
  auto const sa = a.Size();
  auto const sb = b.Size();
  FlatArray<T> inds(sa, lh);
  int ia = 0, ib = 0;
  while ( (ia < sa) && (ib < sb) ) {
    if (a[ia] == b[ib])
      { inds[ia++] = ib; }
    ib++;
  }
  while (ia < sa)
    { inds[ia++] = -1; }
  return inds;
}


template<class T> INLINE Array<typename tab_scal_trait<T>::type> sum_table (T & tab)
{
  Array<typename tab_scal_trait<T>::type> out;
  auto nrows = tab.Size();
  if (nrows == 0)
    { return out; }
  auto row_s = tab[0].Size();
  if (row_s == 0)
    { return out; }
  out.SetSize(row_s); out = tab[0];
  if (nrows == 1)
    { return out; }
  for (size_t k = 1; k < tab.Size(); k++) {
    auto row = tab[k];
    for (auto l : Range(row_s))
      { out[l] += row[l]; }
  }
  return out;
} // sum_table

template<class T> INLINE Array<typename tab_scal_trait<T>::type> min_table (T & tab)
{
  Array<typename tab_scal_trait<T>::type> out;
  auto nrows = tab.Size();
  if (nrows == 0)
    { return out; }
  auto row_s = tab[0].Size();
  if (row_s == 0)
    { return out; }
  out.SetSize(row_s); out = tab[0];
  if (nrows == 1)
    { return out; }
  for (size_t k = 1; k < tab.Size(); k++) {
    auto row = tab[k];
    for (auto l : Range(row_s))
      { out[l] = min(out[l], row[l]); }
  }
  return out;
} // min_table


template<class T> INLINE Array<typename tab_scal_trait<T>::type> max_table (T & tab)
{
  Array<typename tab_scal_trait<T>::type> out;
  auto nrows = tab.Size();
  if (nrows == 0)
    { return out; }
  auto row_s = tab[0].Size();
  if (row_s == 0)
    { return out; }
  out.SetSize(row_s); out = tab[0];
  if (nrows == 1)
    { return out; }
  for (size_t k = 1; k < tab.Size(); k++) {
    auto row = tab[k];
    for (auto l : Range(row_s))
      { out[l] = max2(out[l], row[l]); }
  }
  return out;
} // max_table


template<class T, class TLAM>
INLINE Array<typename tab_scal_trait<T>::type> reduce_table (T & tab, TLAM lam)
{
  Array<typename tab_scal_trait<T>::type> out;
  auto nrows = tab.Size();
  if (nrows == 0)
    { return out; }
  auto row_s = tab[0].Size();
  if (row_s == 0)
    { return out; }
  out.SetSize(row_s); out = tab[0];
  if (nrows == 1)
    { return out; }
  for (size_t k = 1; k < tab.Size(); k++) {
    auto row = tab[k];
    for (auto l : Range(row_s))
      { out[l] = lam(out[l], row[l]); }
  }
  return out;
} // reduce_table

template<typename T> FlatTable<T> MakeFT (size_t nrows, FlatArray<size_t> firstis, FlatArray<T> data, size_t offset)
{
  if (nrows == 0)
    { return FlatTable<T> (nrows, nullptr, nullptr); }
  else if (firstis.Last() == firstis[0])
    { return FlatTable<T> (nrows, firstis.Data(), nullptr); }
  else
    { return FlatTable<T> (nrows, firstis.Data(), &data[offset]); }
} // MakeFT

template<class A, class B, class TLAM>
INLINE void iterate_intersection (const A & a, const B & b, TLAM lam)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2]) {
      lam(i1, i2);
      i1++; i2++;
    }
    else if (a[i1] < b[i2])
      { i1++; }
    else
      { i2++; }
  }
} // iterate_intersection

template<class A, class B, class TLAM_A, class TLAM_B, class TLAM_C>
INLINE void iterate_ABC (const A & a, const B & b, TLAM_A lam_a, TLAM_B lam_b, TLAM_C lam_c)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2]) {
      lam_c(i1, i2);
      i1++; i2++;
    }
    else if (a[i1] < b[i2])
      { lam_a(i1); i1++; }
    else
      { lam_b(i2); i2++; }
  }
  for (; i1 < sa; i1++)
    { lam_a(i1); }
  for (; i2 < sb; i2++)
    { lam_b(i2); }
} // iterate_intersection

enum ABC_KIND {
  INTERSECTION = 0,
  A_NOT_B = 1,
  B_NOT_A = 2,
};

template<class A, class B, class TLAM>
INLINE void iterate_ABC (const A & a, const B & b, TLAM lam)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2]) {
      lam(INTERSECTION, i1, i2);
      i1++; i2++;
    }
    else if (a[i1] < b[i2])
      { lam(A_NOT_B, i1, -1); i1++; }
    else
      { lam(B_NOT_A, -1, i2); i2++; }
  }
  for (; i1 < sa; i1++)
    { lam(A_NOT_B, i1, -1); }
  for (; i2 < sb; i2++)
    { lam(B_NOT_A, -1, i2); }
} // iterate_intersection


template<class A, class B, class TLAM>
INLINE void iterate_AC (const A & a, const B & b, TLAM lam)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2]) {
      lam(INTERSECTION, i1, i2);
      i1++; i2++;
    }
    else if (a[i1] < b[i2])
      { lam(A_NOT_B, i1, -1); i1++; }
    else
      { i2++; }
  }
  for (; i1 < sa; i1++)
    { lam(A_NOT_B, i1, -1); }
} // iterate_intersection

INLINE
std::tuple<FlatArray<int>, FlatArray<int>>
partitionAB (FlatArray<int> subA, FlatArray<int> bigSet, LocalHeap &lh, int BS = 1)
{
  int const n  = bigSet.Size();
  int nA = subA.Size();
  int nB = n - nA;

  FlatArray<int> idxA(nA * BS, lh);
  FlatArray<int> idxB(nB * BS, lh);

  nA = 0;
  nB = 0;

  iterate_AC(bigSet, subA, [&](ABC_KIND const &where, auto const idxS, auto const idxSubA)
  {
    if (where == INTERSECTION)
    {
      for (auto l : Range(BS))
      {
        idxA[nA++] = BS * idxS + l;
      }
    }
    else
    {
      for (auto l : Range(BS))
      {
        idxB[nB++] = BS * idxS + l;
      }
    }
  });

  return std::make_tuple(idxA, idxB);
}

template<class A, class B>
INLINE bool is_intersect_empty (const A & a, const B & b)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2])
      { return false; }
    else if (a[i1] < b[i2])
      { i1++; }
    else
      { i2++; }
  }
  return true;
} // is_intersect_empty

template<class A, class B, class C>
INLINE void intersect_sorted_arrays (const A & a, const B & b, C & c)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0; c.SetSize0();
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] == b[i2]) {
      c.Append(a[i1]);
      i1++; i2++;
    }
    else if (a[i1] < b[i2])
      { i1++; }
    else
    { i2++; }
  }
} // intersect_sorted_arrays

template<class A, class B, class TLAM>
INLINE void iterate_anotb (const A & a, const B & b, TLAM lam)
{
  const int sa = a.Size(), sb = b.Size();
  int i1 = 0, i2 = 0;
  while ( (i1 < sa) && (i2 < sb) ) {
    if (a[i1] < b[i2])
      { lam(i1); i1++; }
    else if (a[i1] == b[i2])
      { i1++; i2++; }
    else
      { i2++; }
  }
  while (i1 < sa)
    { lam(i1); i1++; }
} // iterate_anotb

template<class T>
INLINE FlatArray<T> setMinus(FlatArray<T> a, FlatArray<T> b, LocalHeap &lh)
{
  FlatArray<int> c(a.Size(), lh);
  int cntC = 0;
  iterate_anotb(a, b, [&](auto idxA) {
    c[cntC++] = a[idxA];
  });
  return c.Range(0, cntC);
}

inline FlatArray<int> makeSequence(size_t const &N, LocalHeap &lh)
{
  // TODO: this is kinda suspect
  FlatArray<int> inds(N, lh);
  for (auto k : Range(inds))
    { inds[k] = k; }
  return inds;
}

/** Taken from ngsolve/linalg/sparsematrix.cpp (was not in any ngsolve-header) **/
template <typename FUNC>
INLINE void MergeArrays1 (FlatArray<int*> ptrs,
        FlatArray<int> sizes,
        // FlatArray<int> minvals,
        FUNC f)
{
  STACK_ARRAY(int, minvals, sizes.Size());
  int nactive = 0;
  for (auto i : sizes.Range())
    if (sizes[i])
      {
        nactive++;
        minvals[i] = *ptrs[i];
      }
    else
      minvals[i] = numeric_limits<int>::max();
  while (nactive)
    {
      int minval = minvals[0];
      for (size_t i = 1; i < sizes.Size(); i++)
        minval = min2(minval, minvals[i]);
      f(minval);
      for (int i : sizes.Range())
        if (minvals[i] == minval)
          {
            ptrs[i]++;
            sizes[i]--;
            if (sizes[i] == 0)
              {
                nactive--;
                minvals[i] = numeric_limits<int>::max();
              }
            else
              minvals[i] = *ptrs[i];
          }
    }
} // MergeArrays1

/** Taken from ngsolve/linalg/sparsematrix.cpp (was not in any ngsolve-header) **/
template <typename FUNC>
INLINE void MergeArrays (FlatArray<int*> ptrs,
                          FlatArray<int> sizes,
                          FUNC f)
{
  if (ptrs.Size() <= 16)
    {
      MergeArrays1 (ptrs, sizes, f);
      return;
    }
  struct valsrc { int val, src; };
  struct trange { int idx, val;  };
  int nactive = 0;
  int nrange = 0;
  ArrayMem<valsrc,1024> minvals(sizes.Size());
  ArrayMem<trange,1024> ranges(sizes.Size()+1);
  constexpr int nhash = 1024; // power of 2
  int hashes[nhash];
  for (int i = 0; i < nhash; i++)
    hashes[i] = -1;
  for (auto i : sizes.Range())
    while (sizes[i])
      {
        auto val = *ptrs[i];
        sizes[i]--;
        ptrs[i]++;
        if (hashes[val&(nhash-1)] == val) continue;
        minvals[nactive].val = val;
        hashes[val&(nhash-1)] = val;
        minvals[nactive].src = i;
        nactive++;
        break;
      }
  while (nactive > 0)
    {
      int lower = 0;
      if (nrange > 0) lower = ranges[nrange-1].idx;
      while (true)
        {
          int firstval = minvals[lower].val;
          int otherval = firstval;
          for (int i = lower+1; i < nactive; i++)
            {
              if (minvals[i].val != firstval)
                {
                  otherval = minvals[i].val;
                  break;
                }
            }
          if (firstval == otherval)
            { // all values in last range are the same -> presorting commplete
              if (nrange == 0)
                {
                  ranges[0].idx = 0;
                  ranges[0].val = firstval;
                  nrange = 1;
                }
              break;
            }
          int midval = (firstval+otherval)/2;
          int l = lower, r = nactive-1;
          while (l <= r)
            {
              while (minvals[l].val > midval) l++;
              while (minvals[r].val <= midval) r--;
              if (l < r)
                Swap (minvals[l++], minvals[r--]);
            }
          ranges[nrange].idx = l;
          ranges[nrange].val = midval;
          nrange++;
          lower = l;
        }
      nrange--;
      int last = ranges[nrange].idx;
      f(minvals[last].val);
      // insert new values
      FlatArray<valsrc> tmp(nactive-last, &minvals[last]);
      nactive = last;
      for (valsrc vs : tmp)
        while (sizes[vs.src])
          {
            vs.val = *ptrs[vs.src];
            sizes[vs.src]--;
            ptrs[vs.src]++;
            // take next value if already in queue
            if (hashes[vs.val&(nhash-1)] == vs.val) continue;
            int prevpos = nactive;
            for (int i = nrange-1; i >= 0; i--)
              {
                if (vs.val <= ranges[i].val)
                  break;
                int pos = ranges[i].idx;
                minvals[prevpos] = minvals[pos];
                prevpos = pos;
                ranges[i].idx++;
              }
            minvals[prevpos] = vs;
            hashes[vs.val&(nhash-1)] = vs.val;
            nactive++;
            break;
          }
    }
} // MergeArrays


template<class TGET>
INLINE
FlatArray<int>
mergeFlatArrays (size_t    const  nRows,
                 LocalHeap       &lh,
                 TGET             get)
{
  int maxSize = 0;
  FlatArray<int> sizes(nRows, lh);
  FlatArray<int*> ptrs(nRows, lh);
  for (auto k : Range(nRows))
  {
    FlatArray<int> rowK = get(k);
    sizes[k] = rowK.Size();
    ptrs[k] = rowK.Data();
    maxSize += sizes[k];
  }
  
  FlatArray<int> out(maxSize, lh);
  int cnt = 0;

  MergeArrays(ptrs, sizes, [&](auto val) { out[cnt++] = val; });

  return out.Range(0, cnt);
}

template<class T, class TGET>
INLINE
void MergeArrays (Array<T>      &out,
                  LocalHeap       &lh,
                  size_t    const  nRows,
                  TGET             get)
{
  int maxSize = 0;
  FlatArray<int> sizes(nRows, lh);
  FlatArray<T*> ptrs(nRows, lh);

  for (auto k : Range(nRows))
  {
    FlatArray<T> rowK = get(k);
    sizes[k] = rowK.Size();
    ptrs[k] = rowK.Data();
    maxSize += sizes[k];
  }
  
  out.SetSize(maxSize);

  int cnt = 0;

  MergeArrays(ptrs, sizes, [&](auto val) { out[cnt++] = val; });

  out.SetSize(cnt);
}


template<class T, class TLAM>
INLINE FlatArray<T>
CreateFlatArray(int s, LocalHeap &lh, TLAM lam)
{
  FlatArray<T> a(s, lh);
  for (auto k : Range(s))
  {
    a[k] = lam(k);
  }

  return a;
}

} // namespace amg

#endif // FILE_UTILS_ARRAYS_TABLES_HPP
