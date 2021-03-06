#ifndef FILE_AMGUTILS
#define FILE_AMGUTILS

namespace amg
{

  enum AVG_TYPE : int { MIN,    // min(a,b)
			GEOM,   // sqrt(ab)
			HARM,   // 2 (ainv + binv)^{-1}
			ALG,    // (a+b)/2
			MAX     // max(a,b)
  };

  template<int D> INLINE void GetNodePos (NodeId id, const MeshAccess & ma, Vec<D> & pos, Vec<D> & t) {
    auto set_pts = [&](auto pnums) LAMBDA_INLINE {
      pos = 0;
      for (auto k : Range(pnums)) {
	ma.GetPoint(pnums[k], t);
	pos += t;
      }
      pos *= 1.0/pnums.Size();
    };
    switch(id.GetType()) {
    case(NT_VERTEX) : { ma.GetPoint(id.GetNr(), pos); break; }
    case(NT_EDGE)   : {
      auto pnums = ma.GetEdgePNums(id.GetNr());
      pos = 0;
      ma.GetPoint(pnums[0], t);
      pos += t;
      ma.GetPoint(pnums[1], t);
      pos += t;
      pos *= 0.5;
      break;
    }
    case(NT_FACE)   : { set_pts(ma.GetFacePNums(id.GetNr())); break; }
    case(NT_CELL)   : { set_pts(ma.GetElPNums(ElementId(VOL, id.GetNr()))); break; }
    }
  }

  INLINE void TimedLapackEigenValuesSymmetric (ngbla::FlatMatrix<double> a, ngbla::FlatVector<double> lami,
					       ngbla::FlatMatrix<double> evecs)
  {
    static Timer t("LapackEigenValuesSymmetric"); RegionTimer rt(t);
    LapackEigenValuesSymmetric (a, lami, evecs);
  }
  
  template<class T>
  INLINE void prow (const T & ar, std::ostream &os = cout) {
    for (auto v:ar) os << v << " ";
  };

  template<class T>
  INLINE void prow2 (const T & ar, std::ostream &os = cout) {
    for (auto k : Range(ar.Size())) os << "(" << k << "::" << ar[k] << ") ";
  };

  // like GetPositionTest of NGSolve-SparseMatrix
  template<typename T>
  INLINE size_t find_in_sorted_array (const T & elem, FlatArray<T> a)
  {
    size_t first(0), last(a.Size());
    while (last > first+5) {
      size_t mid = (last+first)/2;
      if (a[mid] > elem) last = mid;
      else { if (a[mid] == elem) { return mid; } first = mid+1; }
    }
    for (size_t i = first; i < last; i++)
      if (a[i] == elem) { return i; }
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
      if (smaller(elem, a[mid])) last = mid;
      else { if (a[mid] == elem) { return mid; } first = mid+1; }
    }
    for (size_t i = first; i < last; i++)
      if (a[i] == elem) { return i; }
    // return numeric_limits<typename remove_reference<decltype(a[0])>::type>::max();
    return (typename remove_reference<decltype(a[0])>::type)(-1);
  };


  // find position such thata a[0..pos) <= elem < a[pos,a.Size())
  template<typename T>
  INLINE size_t merge_pos_in_sorted_array (const T & elem, FlatArray<T> a)
  {
    size_t first(0), last(a.Size());
    if (last==0) return 0; // handle special cases so we still get -1 if garbage
    else if (elem<a[0]) return 0;
    else if (elem>=a.Last()) return last; 
    while (last > first+5) {
      size_t mid = (last+first)/2; // mid>0!
      if ( a[mid] <= elem ) { first = mid; } // search right
      else if ( a[mid-1] > elem ) { last = mid; } // search left
      else { return mid; } // a[mid-1] <= elem < a[mid]!!
    }
    for (size_t i = first; i < last; i++)
      if (a[i] > elem) { return i; }
    return (typename remove_reference<decltype(a[0])>::type)(-1);
  };

  template<typename T>
  INLINE bool insert_into_sorted_array_nodups (T elem, Array<T> & a)
  {
    int pos = merge_pos_in_sorted_array(elem, a);
    if ( (pos == -1) || ((pos > 0) && (a[pos-1] == elem) ))
      {  return false; }
    else
      {  a.Insert(pos, elem); return true; }
  }

  template<typename T>
  INLINE void insert_into_sorted_array (T elem, Array<T> & a)
  { a.Insert(merge_pos_in_sorted_array(elem, a), elem); }

  template<typename T> INLINE int ind_of_max (FlatArray<T> a) {
    if (!a.Size())
      { return -1; }
    int ind = 0; T max = a[ind]; auto as = a.Size();
    for (int k = 1; k < as; k++)
      if (a[k] > max) { max = a[k]; ind = k; }
    return ind;
  }


  template<typename T> INLINE void ApplyPermutation (FlatArray<T> array, FlatArray<int> perm)
  {
    for (auto i : Range(array)) {
      // swap through one circle; use target as buffer as buffer
      if (perm[i] == -1) continue;
      int check = i; int from_here;
      while ( ( from_here = perm[check]) != i ) { // stop when we are back
	swap(array[check], array[from_here]);
	perm[check] = -1; // have right ntry for pos. check
	check = from_here; // check that position next
      }
      perm[check] = -1;
    }
  } // ApplyPermutation


  template<typename T> INLINE void ApplyPermutation (FlatArray<T> array, FlatArray<int> perm, FlatArray<T> buffer)
  {
    for (auto k : Range(array))
      { buffer[k] = array[k]; }
    for (auto k : Range(array))
      { array[k] = buffer[perm[k]]; }
  } // ApplyPermutation


  template<class T>
  INLINE void print_ft(std::ostream &os, const T& t) {
    if (!t.Size()) {
      ( os << "empty flattable!!" << endl );
    }
    cout << "t.s: " << t.Size() << endl;
    for (auto k:Range(t.Size())) {
      os << "t[ " << k << "].s: " << t[k].Size();
      os << " || "; prow(t[k], os); os << endl;
    }
  }

  template<int D> INLINE double TVNorm (const Vec<D,double> & v) {
    double n = 0; for (auto k : Range(D)) n += v(k)*v(k); n = sqrt(n); return n;
  }
  INLINE double TVNorm (double v) { return abs(v); }
  
  template<typename T> bool operator < (const INT<2,T> & a, const INT<2,T> & b) {
    if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
    else return a[1]<b[1];
  }
  template<typename T> bool operator < (const INT<3,T> & a, const INT<3,T> & b) {
    if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
    else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
    else return a[2]<b[2];
  }
  template<typename T> bool operator < (const INT<4,T> & a, const INT<3,T> & b) {
    if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
    else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
    else if (a[2]<b[2]) return true; else if (a[2]>b[2]) return false;
    else return a[3]<b[3];
  }
  template<typename T> bool operator < (const INT<5,T> & a, const INT<5,T> & b) {
    if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
    else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
    else if (a[2]<b[2]) return true; else if (a[2]>b[2]) return false;
    else if (a[3]<b[3]) return true; else if (a[2]>b[2]) return false;
    else return a[4]<b[4];
  }

  template<int N, class T> INLINE INT<N,T> & operator += (INT<N,T> & a, const INT<N,T> & b)
  { Iterate<N>([&](auto i) { a[i.value] += b[i.value]; }); return a; }

  /** size of a parameter pack **/
  template<typename... T> struct count_ppack;
  template<> struct count_ppack<> { static constexpr int value = 0; };
  template<class T, typename... V> struct count_ppack<T, V...> { static constexpr int value = 1 + count_ppack<V...>::value;};
    
  
  template<class T1, class T2, class T3>
  INLINE void merge_arrays (T1& tab_in, Array<T2> & out, T3 lam_comp)
  {
    const size_t nrows = tab_in.Size();
    size_t max_size = 0;
    for (size_t k = 0; k < nrows; k++ )
      if (tab_in[k].Size()>max_size)
	max_size = tab_in[k].Size();
    //probably not gonna add more than 5%??
    // cerr << "tab in nrows " << nrows << endl;
    // for (auto k:Range(nrows)) {
    //   cerr << "row " << k << " ( " << tab_in[k].Size() << "): "; prow(tab_in[k], cerr); cerr << endl;
    // }
    // cerr << endl;
    max_size += (105*max_size)/100; 
    // Array<typename tab_scal_trait<T1>::type> out(max_size);
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
    for (size_t k = 0; k < nrows; k++ ) {
      // cerr << "check " << k << endl; cerr << "ndone " << ndone <<endl;
      // cerr << "row: "; prow(tab_in[k],cerr); cerr << endl; cerr << "len " << tab_in[k].Size() << endl;
      // ndone++;
      // cout << "still here ";
      // cout << ndone << endl;
      // ndone--;
      // cout << "still here ";
      // cout << ndone << endl;
      if (tab_in[k].Size()==0) ndone++;
      // cerr << "check ok " << endl;
    }
    // cerr << "suspicious loop done " << endl;
    auto min_datum = tab_in[rofmin][0];
    while (ndone<nrows) {
      for (size_t k = 0; k < nrows; k++ ) {
	if (count[k]<tab_in[k].Size()) {
	  if (tab_in[k][count[k]]==min_datum)
	    { hasmin.SetBit(k); }
	  else if (lam_comp(tab_in[k][count[k]],min_datum)) {
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
	    ndone++;
	}
      }
      out.Append(min_datum);
      rofmin = -1;
      for (size_t k=0;((k<nrows)&&(rofmin==-1));k++) {
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
    // typedef typename std::result_of<decltype(&T::operator[int])(T, int)>::type T0;
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
  }


  /** binary_op table **/
  template<class T, class TLAM> INLINE Array<typename tab_scal_trait<T>::type> bop_table (T & tab, TLAM lam)
  {
    Array<typename tab_scal_trait<T>::type> out;
    auto nrows = tab.Size();
    if (nrows == 0) return out;
    auto row_s = tab[0].Size();
    if (row_s == 0) return out;
    out.SetSize(row_s); out = tab[0];
    if (nrows == 1) { return out; }
    for (size_t k = 1; k < tab.Size(); k++) {
      auto row = tab[k];
      for (auto l : Range(row_s))
	{ lam(out[l], row[l]); }
    }
    return out;
  }


  template<class T> INLINE Array<typename tab_scal_trait<T>::type> sum_table (T & tab)
  {
    Array<typename tab_scal_trait<T>::type> out;
    auto nrows = tab.Size();
    if (nrows == 0) return out;
    auto row_s = tab[0].Size();
    if (row_s == 0) return out;
    out.SetSize(row_s); out = tab[0];
    if (nrows == 1) { return out; }
    for (size_t k = 1; k < tab.Size(); k++) {
      auto row = tab[k];
      for (auto l : Range(row_s)) out[l] += row[l];
    }
    return out;
  }

  template<class T> INLINE Array<typename tab_scal_trait<T>::type> min_table (T & tab)
  {
    Array<typename tab_scal_trait<T>::type> out;
    auto nrows = tab.Size();
    if (nrows == 0) return out;
    auto row_s = tab[0].Size();
    if (row_s == 0) return out;
    out.SetSize(row_s); out = tab[0];
    if (nrows == 1) { return out; }
    for (size_t k = 1; k < tab.Size(); k++) {
      auto row = tab[k];
      for (auto l : Range(row_s)) out[l] = min(out[l], row[l]);
    }
    return out;
  }


  template<class T> INLINE Array<typename tab_scal_trait<T>::type> max_table (T & tab)
  {
    Array<typename tab_scal_trait<T>::type> out;
    auto nrows = tab.Size();
    if (nrows == 0) return out;
    auto row_s = tab[0].Size();
    if (row_s == 0) return out;
    out.SetSize(row_s); out = tab[0];
    if (nrows == 1) { return out; }
    for (size_t k = 1; k < tab.Size(); k++) {
      auto row = tab[k];
      for (auto l : Range(row_s))
	{ out[l] = max2(out[l], row[l]); }
    }
    return out;
  }

  template<typename TA, typename TB> INLINE ostream & operator << (ostream &os, const tuple<TA, TB>& t)
  { return os << "t<" << get<0>(t) << ", " << get<1>(t) << ">" ; }
    

  template<typename T> ostream & operator << (ostream &os, const FlatTable<T>& t) {
    if (!t.Size()) return ( os << "empty flattable!!" << endl );
    os << "t.s: " << t.Size() << endl;
    for (auto k:Range(t.Size())) {
      os << "t[ " << k << "].s: " << t[k].Size();
      os << " || "; prow(t[k], os); os << endl;
    }
    return os;
  }


  template<class T>
  INLINE void print_tm (ostream &os, const T & mat) {
    constexpr int H = mat_traits<T>::HEIGHT;
    constexpr int W = mat_traits<T>::WIDTH;
    for (int kH : Range(H)) {
      for (int jW : Range(W)) { os << mat(kH,jW) << " "; }
      os << endl;
    }
  }
  template<> INLINE void print_tm (ostream &os, const double & mat) { os << mat << endl; }
  template<class T>
  INLINE void print_tm_mat (ostream &os, const T & mat) {
    constexpr int H = mat_traits<typename std::remove_reference<typename std::result_of<T(int, int)>::type>::type>::HEIGHT;
    constexpr int W = mat_traits<typename std::remove_reference<typename std::result_of<T(int, int)>::type>::type>::WIDTH;
    for (auto k : Range(mat.Height())) {
      for (int kH : Range(H)) {
	for (auto j : Range(mat.Width())) {
	  auto& etr = mat(k,j);
	  for (int jW : Range(W)) { os << etr(kH,jW) << " "; }
	  os << " | ";
	}
	os << endl;
      }
      os << "----" << endl;
    }
  }
  template<> INLINE void print_tm_mat (ostream &os, const Matrix<double> & mat) { os << mat << endl; }
  template<> INLINE void print_tm_mat (ostream &os, const FlatMatrix<double> & mat) { os << mat << endl; }
  
  // copied from ngsolve/comp/h1amg.cpp
  // and removed all shm-parallelization.
  // (the function is not in any ngsolve-header)
  template <typename TFUNC>
  void RunParallelDependency (FlatTable<int> dag,
                              TFUNC func)
  {
    Array<int> cnt_dep(dag.Size());
    cnt_dep = 0;
    for (auto i:Range(dag))
      for (int j : dag[i])
	cnt_dep[j]++;
    // cerr << "cnt_dep: " << endl << cnt_dep << endl;
    size_t num_ready(0), num_final(0);
    // cerr << "count " << cnt_dep.Size() << endl;
    for (auto k:Range(cnt_dep.Size())) {
      // cerr << " now row " << k << endl;
      // cerr << "if (cnt_dep[" << k << "] == 0) " << num_ready << "++;" << endl;
      // if (cnt_dep[k] == 0) cerr << "(is true)" << endl;
      // else cerr << "(is false)" << endl;
      if (cnt_dep[k] == 0) num_ready++;
      if (dag[k].Size() == 0) num_final++;
    }
    Array<int> ready(num_ready);
    ready.SetSize0();
    for (int j : Range(cnt_dep))
      if (cnt_dep[j] == 0) ready.Append(j);
    while (ready.Size())
      {
	int size = ready.Size();
	int nr = ready[size-1];
	ready.SetSize(size-1);
	func(nr);
	for (int j : dag[nr])
	  {
	    cnt_dep[j]--;
	    if (cnt_dep[j] == 0)
	      ready.Append(j);
	  }
      }
    return;
  }


  INLINE std::ostream & operator<<(std::ostream &os, const ParallelDofs& p)
  {
    auto comm = p.GetCommunicator();
    os << "Pardofs, rank " << comm.Rank() << " of " << comm.Size() << endl;
    os << "ndof = " << p.GetNDofLocal() << ", glob " << p.GetNDofGlobal() << endl;
    os << "dist-procs: " << endl;
    for (auto k : Range(p.GetNDofLocal())) {
      auto dps = p.GetDistantProcs(k);
      os << k << ": "; prow(dps); os << endl;
    }
    os << endl;
    return os;
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

  INLINE void SetIdentity (double & x) { x = 1.0; }
  template<int H, int W> INLINE void SetIdentity (Mat<H,W,double> & x) { x = 0.0; for (auto i:Range(min(H,W))) x(i,i) = 1.0; }
  template<class TM> INLINE void SetIdentity (FlatMatrix<TM> mat)
  { for (auto k : Range(mat.Height())) SetIdentity(mat(k,k)); }
  template<int D, class TM> INLINE void SetIdentity (FlatMatrixFixWidth<D, TM> mat)
  { mat = 0; for (auto k : Range(D)) SetIdentity(mat(k,k)); }

  INLINE void SetScalIdentity (double scal, double & x) { x = scal; }
  template<int H, int W> INLINE void SetScalIdentity (double scal, Mat<H,W,double> & x) { x = 0.0; for (auto i:Range(min(H,W))) x(i,i) = scal; }
  template<class TM> INLINE void SetScalIdentity (double scal, FlatMatrix<TM> mat)
  { for (auto k : Range(mat.Height())) SetScalIdentity(scal, mat(k,k)); }
  template<int D, class TM> INLINE void SetScalIdentity (double scal, FlatMatrixFixWidth<D, TM> mat)
  { mat = 0; for (auto k : Range(D)) SetScalIdentity(scal, mat(k,k)); }
  
  template<class TM>
  shared_ptr<stripped_spm_tm<TM>> BuildPermutationMatrix (FlatArray<int> sort) {
    size_t N = sort.Size();
    Array<int> epr(N); epr = 1.0;
    auto embed_mat = make_shared<stripped_spm_tm<TM>>(epr, N);
    const auto & em = *embed_mat;
    for (auto k : Range(N)) {
      em.GetRowIndices(k)[0] = sort[k];
      SetIdentity(em.GetRowValues(k)[0]);
    }
    return embed_mat;
  }

  template<class T>
  INLINE void prow_tm (ostream &os, const T & ar) {
    constexpr int H = mat_traits<typename std::remove_reference<typename std::result_of<T(int)>::type>::type>::HEIGHT;
    constexpr int W = mat_traits<typename std::remove_reference<typename std::result_of<T(int)>::type>::type>::WIDTH;
    for (int kH : Range(H)) {
      for (auto k : Range(ar.Size())) {
	if (kH==0) os << "" << setw(4) << k << "::";
      	else os << "      ";
	auto& etr = ar(k);
	if constexpr(H==W) os << etr;
	else for (int jW : Range(W)) { os << etr(kH,jW) << " "; }
	os << " | ";
      }
      os << endl;
    }
  }

  // INLINE double calc_pow_det (FlatMatrixFixWidth<1,double> x) {
  //   return x(0,0);
  // }
  // INLINE double calc_pow_det (FlatMatrixFixWidth<2,double> x) {
  //   return sqrt(x(0,0)*x(1,1)-x(0,1)*x(1,0));
  // }
  template<int A, class B> INLINE double calc_trace (FlatMatrixFixWidth<A,B> x) {
    double sum = 0; for (auto k : Range(A)) sum += x(k,k); return sum;
  }
  template<int A, int B, class C> INLINE double calc_trace (const Mat<A,B,C> & x) {
    double sum = 0; for (auto k : Range(A)) sum += x(k,k); return sum;
  }
  INLINE double calc_trace (double x) { return x; }
  INLINE Complex calc_trace (Complex x) { return x; }
  template<int A, int B, class C> INLINE double fabsum (const Mat<A,B,C> & x) {
    double sum = 0;
    for (auto k : Range(A))
      for (auto j : Range(B))
	sum += fabs(x(k,j));
    return sum;
  }
  INLINE double calc_trace (FlatMatrix<double> x) {
    double sum = 0;
    for (auto k : Range(x.Height()))
      { sum += x(k,k); }
    return sum;
  }

  template<class A, class B, class TLAM>
  INLINE void iterate_intersection (const A & a, const B & b, TLAM lam)
  {
    const int sa = a.Size(), sb = b.Size();
    int i1 = 0, i2 = 0;
    while ( (i1<sa) && (i2<sb) ) {
      if (a[i1]==b[i2]) {
	lam(i1, i2);
	i1++; i2++;
      }
      else if (a[i1]<b[i2]) { i1++; }
      else { i2++; }
    }
  };

  template<class A, class B>
  INLINE bool is_intersect_empty (const A & a, const B & b)
  {
    const int sa = a.Size(), sb = b.Size();
    int i1 = 0, i2 = 0;
    while ( (i1<sa) && (i2<sb) ) {
      if (a[i1]==b[i2]) { return false; }
      else if (a[i1]<b[i2]) { i1++; }
      else { i2++; }
    }
    return true;
  }

  template<class A, class B, class C>
  INLINE void intersect_sorted_arrays (const A & a, const B & b, C & c)
  {
    const int sa = a.Size(), sb = b.Size();
    // iterate_intersect([&](auto ia, auto ib) LAMBDA_INLINE { c.Append(a[ia]); });
    int i1 = 0, i2 = 0; c.SetSize0();
    while ( (i1<sa) && (i2<sb) ) {
      if (a[i1]==b[i2]) {
	c.Append(a[i1]);
	i1++; i2++;
      }
      else if (a[i1]<b[i2]) { i1++; }
      else { i2++; }
    }
  };

  template<class A, class B, class TLAM>
  INLINE void iterate_anotb (const A & a, const B & b, TLAM lam)
  {
    const int sa = a.Size(), sb = b.Size();
    int i1 = 0, i2 = 0;
    while ( (i1<sa) && (i2<sb) ) {
      if (a[i1] < b[i2])
	{ lam(i1); i1++; }
      else if (a[i1]==b[i2])
	{ i1++; i2++; }
      else { i2++; }
    }
    while (i1<sa)
      { lam(i1); i1++; }
  };

  template<class T> INLINE bool is_invalid (T val) { return val == T(-1); };
  template<class T> INLINE bool is_valid (T val) { return val != T(-1); };

  template<int IMINI, int IMINJ, int A, int B, int C, int D>
  void GetTMBlock (Mat<A,B> & a, const Mat<C,D> & b) {
    static_assert( ((IMINI+A<=C) && (IMINJ+B<=D)), "GET ILLEGAL BLOCK!!");
    Iterate<A>([&](auto i) {
	Iterate<B> ([&](auto j) {
	    a(i.value, j.value) = b(IMINI+i.value, IMINJ+j.value);
	  });
      });
  }

  template<int IMINI, int IMINJ, int A, int B, int C, int D>
  void AddTMBlock (Mat<C,D> & a, const Mat<A,B> & b) {
    static_assert( ((IMINI+A<=C) && (IMINJ+B<=D)), "ADD ILLEGAL BLOCK!!");
    Iterate<A>([&](auto i) {
	Iterate<B> ([&](auto j) {
	    a(IMINI+i.value, IMINJ+j.value) += b(i.value, j.value);
	  });
      });
  }

  template<int IMINI, int IMINJ, int A, int B, int C, int D>
  void AddTMBlock (double val, Mat<C,D> & a, const Mat<A,B> & b) {
    static_assert( ((IMINI+A<=C) && (IMINJ+B<=D)), "ADD ILLEGAL BLOCK!!");
    Iterate<A>([&](auto i) {
	Iterate<B> ([&](auto j) {
	    a(IMINI+i.value, IMINJ+j.value) += val * b(i.value, j.value);
	  });
      });
  }

  template<int N, class T> INLINE void CalcPseudoInverse2 (T & mat, LocalHeap & lh) // needs evals >= -eps
  {
    if constexpr(N==1) { mat = (fabs(mat) > 1e-12) ? 1.0/mat : 0; return; }
    else {
      HeapReset hr(lh);
      double tr = calc_trace(mat) / N;
      double eps = max2(1e-8 * tr, 1e-15);
      int M = 0;
      for (auto k : Range(N))
	if (mat(k,k) > eps)
	  { M++; }
      FlatArray<int> nzeros(M, lh);
      M = 0;
      for (auto k : Range(N))
	if (mat(k,k) > eps)
	  { nzeros[M] = k; M++; }
      if (M == 0)
	{ mat = 0; return; }
      cout << " mat " << endl << mat << endl;
      FlatMatrix<double> smallD(M,M,lh);
      double rcs = 0;
      for (auto i : Range(M))
	for (auto j : Range(M)) {
	  smallD(i,j) = mat(nzeros[i], nzeros[j]);
	  rcs += fabs(mat(nzeros[i], nzeros[j]));
	}
      rcs = M*M / rcs; cout << " rcs " << rcs << endl; rcs = 1; smallD *= rcs;
      cout << "smallD: " << endl << smallD << endl;
      FlatMatrix<double> evecs(M, M, lh);
      FlatVector<double> evals(M, lh);
      TimedLapackEigenValuesSymmetric(smallD, evals, evecs);
      cout << " small D evals (of " << M << "): "; prow(evals); cout << endl;
      int nnz = 0; double max_ev = evals(M-1);
      for (auto k : Range(M)) {
	bool is_nz = (evals(k) > 1e-6 * max_ev);
	double f = is_nz ? 1/sqrt(evals(k)) : 0;
	if (is_nz) { nnz++; }
	for (auto j : Range(M))
	  { evecs(k,j) *= f; }
      }
      mat = 0;
      if (nnz > 0) {
	cout << " scaled evecs " << endl << evecs << endl;
	auto nzrows = evecs.Rows(M-nnz, M);
	cout << " scaled nzrows " << endl << nzrows << endl;
	smallD = Trans(nzrows) * nzrows;
	cout << " psinv smallD " << endl << smallD << endl;
	for (auto i : Range(M))
	  for (auto j : Range(M))
	    { mat(nzeros[i],nzeros[j]) = rcs * smallD(i,j); }
	cout << " mat: " << endl << mat << endl;
      }
    }
  }


  template<int N, class T> INLINE void CalcSqrtInv (T & m)
  {
    static Timer t("CalcPseudoInverse_neg"); RegionTimer rt(t);
    static Matrix<double> M(N,N), evecs(N,N);
    static Vector<double> evals(N);
    M = m;
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    double tol = 0;
    for (auto v : evals)
      { tol += fabs(v); }
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    int nneg = 0, npos = 0;
    for (auto & v : evals) {
      if (fabs(v) > tol) {
	if (v < 0) { nneg++; }
	else { npos++; }
	v = 1/sqrt(sqrt(fabs(v)));
      }
      else
	{ v = 0; }
    }
    Iterate<N>([&](auto i) {
	Iterate<N>([&](auto j) {
	    evecs(i.value,j.value) *= evals(i.value);
	  });
      });
    if (npos > 0) {
      m = Trans(evecs.Rows(N-npos, N)) * evecs.Rows(N-npos, N);
      if (nneg > 0)
	{ m -= Trans(evecs.Rows(0, nneg)) * evecs.Rows(0, nneg); }
    }
    else if (nneg > 0)
      { m = -Trans(evecs.Rows(0, nneg)) * evecs.Rows(0, nneg); }
    else
      { m = 0; }
  }

  template<int N, class T> INLINE void CalcPseudoInverse_neg (T & m)
  {
    static Timer t("CalcPseudoInverse_neg"); RegionTimer rt(t);
    static Matrix<double> M(N,N), evecs(N,N);
    static Vector<double> evals(N);
    M = m;
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    double tol = 0;
    for (auto v : evals)
      { tol += fabs(v); }
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    int nneg = 0, npos = 0;
    for (auto & v : evals) {
      if (fabs(v) > tol) {
	if (v < 0) { nneg++; }
	else { npos++; }
	v = 1/sqrt(fabs(v));
      }
      else
	{ v = 0; }
    }
    Iterate<N>([&](auto i) {
	Iterate<N>([&](auto j) {
	    evecs(i.value,j.value) *= evals(i.value);
	  });
      });
    if (npos > 0) {
      m = Trans(evecs.Rows(N-npos, N)) * evecs.Rows(N-npos, N);
      if (nneg > 0)
	{ m -= Trans(evecs.Rows(0, nneg)) * evecs.Rows(0, nneg); }
    }
    else if (nneg > 0)
      { m = -Trans(evecs.Rows(0, nneg)) * evecs.Rows(0, nneg); }
    else
      { m = 0; }
  }


  INLINE void CalcStabPseudoInverse (double & M, LocalHeap & lh)
  { M = (M == 0) ? 0 : 1.0/M; }

  INLINE void CalcPseudoInverse2 (double & M, LocalHeap & lh)
  { M = (M == 0) ? 0 : 1.0/M; }

  INLINE void CalcPseudoInverse2 (FlatMatrix<double> M, LocalHeap & lh)
  {
    static Timer t("CalcPseudoInverse2"); RegionTimer rt(t);
    const int N = M.Height();
    FlatMatrix<double> evecs(N, N, lh);
    FlatVector<double> evals(N, lh);
    LapackEigenValuesSymmetric(M, evals, evecs);
    double tol = 0; for (auto v : evals) tol += v;
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    for (auto & v : evals)
      { v = (v > tol) ? 1/sqrt(v) : 0; }
    for (auto i : Range(N))
      for (auto j : Range(N))
	{ evecs(i, j) *= evals(i); }
    M = Trans(evecs) * evecs;
  }


  INLINE void CalcStabPseudoInverse (FlatMatrix<double> mat, LocalHeap & lh)
  {
    static Timer t("CalcStabPseudoInverse2"); RegionTimer rt(t);
    int N = mat.Height(), M = 0;
    double tr = calc_trace(mat) / N;
    double eps = 1e-8 * tr;
    for (auto k : Range(N))
      if (mat(k,k) > eps)
	{ M++; }
    FlatArray<double> mat_diags(M, lh);
    FlatArray<double> mat_diag_invs(M, lh);
    FlatArray<int> nzeros(M, lh);
    M = 0;
    for (auto k : Range(N)) {
      if (mat(k,k) > eps) {
	auto rt = sqrt(mat(k,k));
	mat_diags[M] = rt;
	mat_diag_invs[M] = 1.0/rt;
	nzeros[M] = k;
	M++;
      }
    }
    FlatMatrix<double> small_mat(M, M, lh);
    for (auto i : Range(M))
      for (auto j : Range(M))
	{ small_mat(i,j) = mat(nzeros[i], nzeros[j]) * mat_diag_invs[i] * mat_diag_invs[j]; }
    CalcPseudoInverse2(small_mat, lh);
    mat = 0;
    for (auto i : Range(M))
      for (auto j : Range(M))
	{ mat(nzeros[i],nzeros[j]) = small_mat(i,j) * mat_diag_invs[i] * mat_diag_invs[j]; }
  }

  template<int N> INLINE void CalcStabPseudoInverse (Mat<N, N, double> & TM, LocalHeap & lh)
  {
    FlatMatrix<double> mat(N, N, lh);
    mat = TM;
    CalcStabPseudoInverse(mat, lh);
    TM = mat;
  }

  template<int N, class T> INLINE void CalcPseudoInverse (T & m)
  {
    // static Timer t("CalcPseudoInverse"); RegionTimer rt(t);
    // static Timer tl("CalcPseudoInverse - Lapck");

    static Matrix<double> M(N,N), evecs(N,N);
    static Vector<double> evals(N);
    M = m;
    // cout << "pseudo inv M: " << endl << M << endl;
    // tl.Start();
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    // tl.Stop();
    // cout << "pseudo inv evals: "; prow(evals); cout << endl;
    double tol = 0; for (auto v : evals) tol += v;
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    // cout << "tol: " << tol << endl;
    for (auto & v : evals)
      v = (v > tol) ? 1/sqrt(v) : 0;
    // cout << "rescaled evals: "; prow(evals); cout << endl;
    Iterate<N>([&](auto i) {
	Iterate<N>([&](auto j) {
	    evecs(i.value,j.value) *= evals(i.value);
	  });
      });
    // cout << "rescaled evecs: " << endl << evecs << endl;
    m = Trans(evecs) * evecs;
  }

  INLINE void CalcPseudoInverseFM (FlatMatrix<double> & M, LocalHeap & lh)
  {
    // static Timer t("CalcPseudoInverseFM"); RegionTimer rt(t);
    HeapReset hr(lh);
    // static Timer tl("CalcPseudoInverse - Lapck");
    const int N = M.Height();
    FlatMatrix<double> evecs(N, N, lh);
    FlatVector<double> evals(N, lh);
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    double tol = 0; for (auto v : evals) tol += v;
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    int DK = 0; // dim kernel
    for (auto & v : evals) {
      if (v > tol)
	{ v = 1/sqrt(v); }
      else {
	DK++;
	v = 0;
      }
    }
    int NS = N-DK;
    for (auto i : Range(N))
      for (auto j : Range(N))
	evecs(i,j) *= evals(i);
    if (DK > 0)
      { M = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
    else
      { M = Trans(evecs) * evecs; }
  }


  template<class T>
  INLINE void CalcPseudoInverse_impl (FlatMatrix<double> & M, T & out, LocalHeap & lh)
  {
    // static Timer t("CalcPseudoInverse_impl"); RegionTimer rt(t);
    // static Timer tl("CalcPseudoInverse - Lapck");
    const int N = M.Height();
    FlatMatrix<double> evecs(N, N, lh);
    FlatVector<double> evals(N, lh);
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    double tol = 0; for (auto v : evals) tol += v;
    tol = 1e-12 * tol; tol = max2(tol, 1e-15);
    int DK = 0; // dim kernel
    for (auto & v : evals) {
      if (v > tol)
	{ v = 1/sqrt(v); }
      else {
	DK++;
	v = 0;
      }
    }
    int NS = N-DK;
    for (auto i : Range(N))
      for (auto j : Range(N))
	evecs(i,j) *= evals(i);
    if (DK > 0)
      { out = Trans(evecs.Rows(DK, N)) * evecs.Rows(DK, N); }
    else
      { out = Trans(evecs) * evecs; }
  }


  template<int N>
  INLINE void CalcPseudoInverse (Mat<N, N, double> & M, LocalHeap & lh)
  {
    // static Timer t("CalcPseudoInverse expr");
    HeapReset hr(lh);
    FlatMatrix<double> Mf(N,N,lh);
    Mf = M;
    CalcPseudoInverse_impl(Mf, M, lh);
  }


  template<int IMIN, int N, int NN> INLINE void RegTM (Mat<NN,NN,double> & m, double maxadd = -1)
  {
    // static Timer t(string("RegTM<") + to_string(IMIN) + string(",") + to_string(3) + string(",") + to_string(6) + string(">")); RegionTimer rt(t);
    static_assert( (IMIN + N <= NN) , "ILLEGAL RegTM!!");
    static Matrix<double> M(N,N), evecs(N,N);
    static Vector<double> evals(N);
    Iterate<N>([&](auto i) {
	Iterate<N>([&](auto j) {
	    M(i.value, j.value) = m(IMIN+i.value, IMIN+j.value);
	  });
      });
    TimedLapackEigenValuesSymmetric(M, evals, evecs);
    const double eps = max2(1e-15, 1e-12 * evals(N-1));
    double min_nzev = 0; int nzero = 0;
    for (auto k : Range(N))
      if (evals(k) > eps)
	{ min_nzev = evals(k); break; }
      else
	{ nzero++; }
    if (maxadd >= 0)
      { min_nzev = min(maxadd, min_nzev); }
    if (nzero < N) {
      for (auto l : Range(nzero)) {
	Iterate<N>([&](auto i) {
	    Iterate<N>([&](auto j) {
		m(IMIN+i.value, IMIN+j.value) += min_nzev * evecs(l, i.value) * evecs(l, j.value);
	      });
	  });
      }
    }
    else {
      SetIdentity(m);
      if (maxadd >= 0)
	{ m *= maxadd; }
    }
  }

  template<> INLINE void RegTM<2,1,3> (Mat<3,3,double> & m, double maxadd)
  {
    double tr = 0.5 * (m(0,0) + m(1,1));
    if (maxadd >= 0)
      { tr = min2(maxadd, tr); }
    if ( m(2,2) / tr < 1e-8 )
      { m(2,2) = tr; }
  }

  /**
       assume rank 0, 2 or 3

       rank 1 never happens for 3d elasticity (I think?)

       rank 0: mat is 0, add to diags..
       rank 3: do nothing

       rank 2: if any diagonal value is 0, add to diag

       rank 2, otherwise:
       m = (w1, w2, w3)
       m * w_i !=0 (because diag values !=0)
       z := w1 x w2
       if z==0:
          w1 = alpha w2
	  kernel-candidate = (1, -alpha, 0)
       else:
          w1, w2 LU and they span the range of m (if really rank 2)
	  z is orthogonal to w1, w2
          kernel-candidate = z
       then check if candidate is really kernel
     **/
  template<> INLINE void RegTM<3,3,6> (Mat<6,6,double> & m, double maxadd)
  {
    Vec<3, double> mkv, kv, vi, vj;

    int cnt0; cnt0 = 0;
    double atr = 0.33 * (m(3,3) + m(4,4) + m(5,5));
    double utr = 0.33 * (m(0,0) + m(1,1) + m(2,2));
    // if basically 0, scale to u-part, else scale to r-part
    if (atr < 1e-8 * utr) atr = utr;
    Iterate<3>([&](auto i) {
  	if (m(3+i.value, 3+i.value) < 1e-6 * atr) {
  	  cnt0++; m(3+i.value, 3+i.value) = atr;
  	}
      });

    if (cnt0 > 0)
      { return; }

    // t_cps.Start();
    double norm = 0;
    int I = -1, J = -1, L = -1;
    auto reg = [&](auto i, auto j, auto l) {
      Iterate<3>([&](auto k) {
  	  vi(k.value) = m(3+i, 3+k.value);
  	  vj(k.value) = m(3+j, 3+k.value);
  	});
      vi /= L2Norm(vi); vj /= L2Norm(vj);
      kv = Cross(vi, vj);
      norm = L2Norm(kv);
      I = i; J = j; L = l;
    };

    reg(0, 1, 2);
    if (norm < 0.1)
      reg(1, 2, 0);
    if (norm < 0.1)
      reg(2, 0, 1);

    if (norm < 1e-3) {
      int JJ = 0; double mxe = 0;
      Iterate<3>([&](auto i) {
  	  if (fabs(vj(i.value)) > mxe) {
  	    mxe = vj(i.value);
  	    JJ = i.value;
  	  }
  	});
      double alpha = vi(JJ) / vj(JJ);
      kv(I) = 1;
      kv(J) = - alpha;
      kv(L) = 0;
      kv /= L2Norm(kv);
    }
    else {
      kv /= norm;
    }

    mkv = m.Rows(3,6).Cols(3,6) * kv;
    double ncross = L2Norm(mkv);

    // {
    //   cout << "m: " << endl << m.Rows(3,6).Cols(3,6) << endl;
    //   cout << "kv: " << kv << endl;
    //   cout << "atr: " << atr << endl;
    //   cout << "ncross: " << ncross << endl;
    // }

    if (ncross > 1e-12 * atr)
      { return; }

    // cout << " update: " << endl;
    // Iterate<3>([&](auto i) {
    // 	Iterate<3>([&](auto j) {
    // 	    cout << kv(i.value) * kv(j.value) << " ";
    // 	  });
    // 	cout << endl;
    //   });

    Iterate<3>([&](auto i) {
  	Iterate<3>([&](auto j) {
  	    // m(3+i.value, 3+j.value) += atr * kv(i.value) * kv(j.value);
  	    m(3+i.value, 3+j.value) += kv(i.value) * kv(j.value);
  	  });
      });

    // cout << "m now: " << endl << m.Rows(3,6).Cols(3,6) << endl;
  }
  
  template<int N> INLINE double CalcDet (Mat<N,N,double> & m)
  { throw Exception(string("CalcDet<")+to_string(N)+string(",")+to_string(N)+string("> not implemented")); }
  template<> INLINE double CalcDet (Mat<2,2,double> & m)
  { return m(0,0)*m(1,1)-m(1,0)*m(0,1); }

  template<int N> INLINE void CalcPseudoInv (Mat<N,N,double> & m)
  { throw Exception(string("CalcPseudoInv<")+to_string(N)+string(",")+to_string(N)+string("> not implemented")); }

  template<> INLINE void CalcPseudoInv (Mat<2,2,double> & m)
  {
    // cout << "m " << endl; print_tm(cout, m);
    double avg = 0.5 * calc_trace(m); double tol = 1e-14 * avg;
    if ( CalcDet(m) > tol ) {
      // cout << "case 1 " << endl;
      CalcInverse(m);
    }
    else if ( fabs(m(0,0)) < tol ) {
      if ( fabs(m(1,1)) < tol )
	{ /*cout << "case 2 " << endl;*/ m = 0; }
      else
	{ /*cout << "case 3 " << endl; */ m(0,0) = m(1,0) = m(0,1) = 0; m(1,1) = 1/m(1,1); }
    }
    else if ( fabs(m(1,1)) < tol )
      { /*cout << "case 4 " << endl;*/ m(1,1) = m(1,0) = m(0,1) = 0; m(0,0) = 1/m(0,0); }
    else {
      /*cout << "case 5 " << endl;*/
      // add kv-projector, invert, substract it again
      static Vec<2,double> kv; kv(0) = m(0,1); kv(1) = -m(0,0);
      double kvn = L2Norm(kv);
      Iterate<2>([&](auto i) {
	  Iterate<2>([&](auto j) {
	      m(i.value,j.value) += kv(i.value)*kv(j.value);
	    });
	});
      CalcInverse(m);
      kv /= kvn; kv /= kvn;
      Iterate<2>([&](auto i) {
	  Iterate<2>([&](auto j) {
	      m(i.value,j.value) -= kv(i.value)*kv(j.value);
	    });
	});
    }
    // cout << "m pseudo inv: " << endl; print_tm(cout, m); cout << endl;
  } // CalcPseudoInv



  INLINE bool is_zero (const double & x) { return x == 0; }
  INLINE bool is_zero (const int & x) { return x == 0; }
  INLINE bool is_zero (const size_t & x) { return x == 0; }
  template<int N, int M, typename TSCAL>
  INLINE bool is_zero (const Mat<N, M,  TSCAL> & m)
  {
    for (auto k : Range(N))
      for (auto j : Range(M))
	if (!is_zero(m(k,j))) return false;
    return true;
  }
  template<int N, typename TSCAL>
  INLINE bool is_zero (const INT<N, TSCAL> & m)
  {
    for (auto k : Range(N))
      if (!is_zero(m[k])) return false;
    return true;
  }
  template<int N, typename TSCAL>
  INLINE bool is_zero (const Vec<N, TSCAL> & m)
  {
    for (auto k : Range(N))
      if (!is_zero(m(k))) return false;
    return true;
  }

  template<typename T> FlatTable<T> MakeFT (size_t nrows, FlatArray<size_t> firstis, FlatArray<T> data, size_t offset)
  {
    if (nrows == 0)
      { return FlatTable<T> (nrows, nullptr, nullptr); }
    else if (firstis.Last() == firstis[0])
      { return FlatTable<T> (nrows, firstis.Data(), nullptr); }
    else
      { return FlatTable<T> (nrows, firstis.Data(), &data[offset]); }
  }


  template<int N, class T> INLINE void prt_evv (T& M, string name, bool vecs = true) {
    static Matrix<double> prm(N,N), preve(N,N);
    static Vector<double> preva(N);
    prm = M;
    TimedLapackEigenValuesSymmetric(prm, preva, preve);
    cout << name << " evals: "; prow(preva); cout << endl;
    if (vecs)
      { cout << name << " evecs: " << endl << preve << endl; }
  };

  template<> INLINE void prt_evv<1,double> (double& M, string name, bool vecs) {
    cout << name << " evals: " << M << endl;
  }


  template<class OC>
  struct SpecOpt
  {
  public:
    OC default_opt;
    Array<OC> spec_opt;
    SpecOpt () : spec_opt(0) { ; }
    SpecOpt (OC _default_opt) : default_opt(_default_opt), spec_opt(0) { ; }
    SpecOpt (OC _default_opt, Array<OC> & _spec_opt) : default_opt(_default_opt), spec_opt(_spec_opt) { ; }
    SpecOpt (OC _default_opt, Array<OC> _spec_opt) : default_opt(_default_opt), spec_opt(_spec_opt) { ; }
    SpecOpt (const SpecOpt<OC> & other) : default_opt(default_opt), spec_opt(other.spec_opt) { ; }
    SpecOpt (SpecOpt<OC> && other) : default_opt(other.default_opt), spec_opt(move(other.spec_opt)) { ; }
    SpecOpt (const Flags & flags, OC defopt, string defkey, Array<string> optkeys, Array<OC> enum_vals)
      : SpecOpt(defopt)
    { SetFromFlagsEnum(flags, defkey, move(optkeys), move(enum_vals)); }
    SpecOpt<OC> & operator = (OC defoc) {
      default_opt = defoc;
      spec_opt.SetSize0();
      return *this;
    }
    SpecOpt<OC> & operator = (const SpecOpt<OC> & other) {
      default_opt = other.default_opt;
      spec_opt.SetSize(other.spec_opt.Size());
      spec_opt = other.spec_opt;
      return *this;
    }
    void SetFromFlagsEnum (const Flags & flags, string defkey, Array<string> optkeys, Array<OC> enum_vals)
    {
      auto setoc = [&](auto & oc, string val) {
	int index;
	if ( (index = optkeys.Pos(val)) != -1)
	  { oc = enum_vals[index]; }
      };
      setoc(default_opt, flags.GetStringFlag(defkey, ""));
      auto & specfa = flags.GetStringListFlag(defkey + "_spec");
      spec_opt.SetSize(specfa.Size()); spec_opt = default_opt;
      for (auto k : Range(spec_opt))
	{ setoc(spec_opt[k], specfa[k]); }
    }
    INLINE void SetFromFlags (const Flags & flags, string key);
    OC GetOpt (int level) const { return (level < spec_opt.Size()) ? spec_opt[level] : default_opt; }

    template<class X>
    friend INLINE std::ostream & operator<<(std::ostream &os, const SpecOpt<OC>& so);
  }; // struct SpecOpt

  template<class OC>
  INLINE void SpecOpt<OC> :: SetFromFlags (const Flags & flags, string key)
  {
    throw Exception("SFF not overloaded!!");
  } // SpecOpt::SetFromFlags

  template<>
  INLINE void SpecOpt<bool> :: SetFromFlags (const Flags & flags, string key)
  {
    if (default_opt) { default_opt = !flags.GetDefineFlagX(key).IsFalse(); }
    else { default_opt = flags.GetDefineFlagX(key).IsTrue(); }
    auto & arr = flags.GetNumListFlag(key+"_spec");
    spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
    for (auto k : Range(spec_opt))
      { spec_opt[k] = (arr[k] != 0); }
  } // SpecOpt<bool>::SetFromFlags

  template<>
  INLINE void SpecOpt<xbool> :: SetFromFlags (const Flags & flags, string key)
  {
    default_opt = flags.GetDefineFlagX(key);
    auto & arr = flags.GetNumListFlag(key+"_spec");
    spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
    for (auto k : Range(spec_opt))
      { spec_opt[k] = (arr[k] == 0) ? false : true; }
  } // SpecOpt<bool>::SetFromFlags

  template<>
  INLINE void SpecOpt<double> :: SetFromFlags (const Flags & flags, string key)
  {
    default_opt = flags.GetNumFlag(key, default_opt);
    auto & arr = flags.GetNumListFlag(key+"_spec");
    spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
    for (auto k : Range(spec_opt))
      { spec_opt[k] = arr[k]; }
  } // SpecOpt<bool>::SetFromFlags


  template<>
  INLINE void SpecOpt<int> :: SetFromFlags (const Flags & flags, string key)
  {
    default_opt = int(flags.GetNumFlag(key, double(default_opt)));
    auto & arr = flags.GetNumListFlag(key+"_spec");
    spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
    for (auto k : Range(spec_opt))
      { spec_opt[k] = arr[k]; }
  } // SpecOpt<bool>::SetFromFlags


  template<typename X>
  void SetEnumOpt (const Flags & flags, X & opt, string key, Array<string> vals, Array<X> evals) {
    string val = flags.GetStringFlag(key, "");
    for (auto k : Range(vals)) {
      if (val == vals[k])
	{ opt = evals[k]; return; }
    }
  }

  template<typename X>
  void SetEnumOpt (const Flags & flags, X & opt, string key, Array<string> vals, Array<X> evals, X default_val) {
    string val = flags.GetStringFlag(key, "");
    for (auto k : Range(vals)) {
      if (val == vals[k])
	{ opt = evals[k]; return; }
    }
    opt = default_val;
  }

  template<class OC>
  INLINE std::ostream & operator<<(std::ostream &os, const SpecOpt<OC>& so)
  {
    os << "SpecOpts<" << typeid(OC).name() << ">, default val = " << so.default_opt << ", spec opts (" << so.spec_opt.Size() << ") = ";
    prow2(so.spec_opt, os);
    return os;
  }


} // namespace amg

#endif
