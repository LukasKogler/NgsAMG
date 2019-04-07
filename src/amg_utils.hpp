#ifndef FILE_AMGUTILS
#define FILE_AMGUTILS

namespace amg
{
  
  auto prow = [](const auto & ar, std::ostream &os = cout){ for(auto v:ar) os << v << " "; };
  auto prow2 = [](const auto & ar, std::ostream &os = cout) {
    for(auto k : Range(ar.Size())) os << "(" << k << "::" << ar[k] << ") ";
  };

  // like GetPositionTest of ngsovle-SparseMatrix
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

  // find position such thata a[0..pos) <= elem < a[pos,a.Size())
  template<typename T>
  INLINE size_t merge_pos_in_sorted_array (const T & elem, FlatArray<T> a)
  {
    size_t first(0), last(a.Size());
    if (last==0) return 0; // handle special cases so we still get -1 if garbage
    else if (elem<a[0]) return 0;
    else if (elem>a.Last()) return last; 
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

  template<class T>
  INLINE void print_ft(std::ostream &os, const T& t) {
    if(!t.Size()) {
      ( os << "empty flattable!!" << endl );
    }
    cout << "t.s: " << t.Size() << endl;
    for(auto k:Range(t.Size())) {
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

  auto hack_eval_tab = [](const auto &x) { return x[0][0]; };
  template<class T> struct tab_scal_trait {
    typedef typename std::remove_reference<typename std::result_of<decltype(hack_eval_tab)(T)>::type>::type type;
  };
  template<class T1, class T2>
  INLINE Array<typename tab_scal_trait<T1>::type> merge_arrays (T1& tab_in, T2 lam_comp)
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
    Array<typename tab_scal_trait<T1>::type> out(max_size);
    out.SetSize(0);
    Array<size_t> count(nrows);
    count = 0;
    size_t ndone = 0;
    BitArray hasmin(nrows);
    hasmin.Clear();
    int rofmin = -1;
    for (size_t k = 0;((k<nrows)&&(rofmin==-1));k++)
      { if (tab_in[k].Size()>0) rofmin = k; }
    if (rofmin==-1) { return out; }
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
	    { hasmin.Set(k); }
	  else if (lam_comp(tab_in[k][count[k]],min_datum)) {
	    hasmin.Clear();
	    hasmin.Set(k);
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
    return out;
  };

  // TODO: coutl this be binary_op_table?
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

  template<typename T> ostream & operator << (ostream &os, const FlatTable<T>& t) {
    if(!t.Size()) return ( os << "empty flattable!!" << endl );
    os << "t.s: " << t.Size() << endl;
    for(auto k:Range(t.Size())) {
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
    for(auto k:Range(cnt_dep.Size())) {
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
  shared_ptr<stripped_spm<TM>> BuildPermutationMatrix (FlatArray<int> sort) {
    size_t N = sort.Size();
    Array<int> epr(N); epr = 1.0;
    auto embed_mat = make_shared<stripped_spm<TM>>(epr, N);
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

  template<int A, class B> INLINE double calc_trace (FlatMatrixFixWidth<A,B> x) {
    double sum = 0; for (auto k : Range(A)) sum += x(k,k); return sum;
  }
  
} // namespace amg

#endif
