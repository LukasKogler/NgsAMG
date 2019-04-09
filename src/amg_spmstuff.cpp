#define FILE_AMG_SPMSTUFF_CPP

#include "amg.hpp"

namespace amg
{

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
        for (int i = 1; i < sizes.Size(); i++)
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

  
  // template <typename TM_RES, typename TMA, typename TMB,
  // 	    typename TSCAL_MATCH = typename std::enable_if<std::is_same<mat_traits<TMA>::T,mat_tratis<TMB>::T>::value>::type,
  // 	    typename TSIZE_MATCH = typename std::enable_if<mat_traits<TMA>::W==mat_traits<TMB>::H>::type>
  // shared_ptr<SparseMatrixTM<Mat<mat_traits<TMA>::H,mat_traits<TMB>::W,mat_traits<TMA>::T>>>
  // MatMultAB (const SparseMatrixTM<TMA> & mata, const SparseMatrixTM<TMB> & matb);
  // {
  //   static Timer t ("sparse matrix multiplication");
  //   static Timer t1a ("sparse matrix multiplication - setup a");
  //   static Timer t1b ("sparse matrix multiplication - setup b");
  //   static Timer t1b1 ("sparse matrix multiplication - setup b1");
  //   static Timer t2 ("sparse matrix multiplication - mult"); 
  //   RegionTimer reg(t);
  //   t1a.Start();
  //   Array<int> cnt(mata.Height());
  //   cnt = 0;
  //   ParallelForRange
  //     (mata.Height(), [&] (IntRange r)
  //      {
  //        Array<int*> ptrs;
  //        Array<int> sizes;
  //        for (int i : r)
  //          {
  //            auto mata_ci = mata.GetRowIndices(i);
  //            ptrs.SetSize(mata_ci.Size());
  //            sizes.SetSize(mata_ci.Size());
  //            for (int j : Range(mata_ci))
  //              {
  //                ptrs[j] = matb.GetRowIndices(mata_ci[j]).Addr(0);
  //                sizes[j] = matb.GetRowIndices(mata_ci[j]).Size();
  //              }
  //            int cnti = 0;
  //            MergeArrays(ptrs, sizes, [&cnti] (int col) { cnti++; } );
  //            cnt[i] = cnti;
  //          }
  //      },
  //      TasksPerThread(10));
  //   t1a.Stop();
  //   t1b.Start();
  //   t1b1.Start();
  //   auto prod = make_shared<SparseMatrix<TM_RES>>(cnt, matb.Width());
  //   prod->AsVector() = 0.0;
  //   t1b1.Stop();
  //   // fill col-indices
  //   ParallelForRange
  //     (mata.Height(), [&] (IntRange r)
  //      {
  //        Array<int*> ptrs;
  //        Array<int> sizes;
  //        for (int i : r)
  //          {
  //            auto mata_ci = mata.GetRowIndices(i);
  //            ptrs.SetSize(mata_ci.Size());
  //            sizes.SetSize(mata_ci.Size());
  //            for (int j : Range(mata_ci))
  //              {
  //                ptrs[j] = matb.GetRowIndices(mata_ci[j]).Addr(0);
  //                sizes[j] = matb.GetRowIndices(mata_ci[j]).Size();
  //              }
  //            int * ptr = prod->GetRowIndices(i).Addr(0);
  //            MergeArrays(ptrs, sizes, [&ptr] (int col)
  //                        {
  //                          *ptr = col;
  //                          ptr++;
  //                        } );
  //          }
  //      },
  //      TasksPerThread(10));
  //   t1b.Stop();
  //   t2.Start();
  //   ParallelForRange
  //     (mata.Height(), [&] (IntRange r)
  //      {
  //        struct thash { int idx; int pos; };
  //        size_t maxci = 0;
  //        for (auto i : r)
  //          maxci = max2(maxci, size_t (prod->GetRowIndices(i).Size()));
  //        size_t nhash = 2048;
  //        while (nhash < 2*maxci) nhash *= 2;
  //        ArrayMem<thash,2048> hash(nhash);
  //        size_t nhashm1 = nhash-1;
  //        for (auto i : r)
  //          {
  //            auto mata_ci = mata.GetRowIndices(i);
  //            auto matc_ci = prod->GetRowIndices(i);
  //            auto matc_vals = prod->GetRowValues(i);
  //            for (int k = 0; k < matc_ci.Size(); k++)
  //              {
  //                size_t hashval = size_t(matc_ci[k]) & nhashm1; // % nhash;
  //                hash[hashval].pos = k;
  //                hash[hashval].idx = matc_ci[k];
  //              }
  //            for (int j : Range(mata_ci))
  //              {
  //                auto vala = mata.GetRowValues(i)[j];
  //                int rowb = mata.GetRowIndices(i)[j];
  //                auto matb_ci = matb.GetRowIndices(rowb);
  //                auto matb_vals = matb.GetRowValues(rowb);
  //                for (int k = 0; k < matb_ci.Size(); k++)
  //                  {
  //                    auto colb = matb_ci[k];
  //                    unsigned hashval = unsigned(colb) & nhashm1; // % nhash;
  //                    if (hash[hashval].idx == colb)
  //                      { // lucky fast branch
  //                       matc_vals[hash[hashval].pos] += vala * matb_vals[k]; 
  //                      }
  //                    else
  //                     { // do the binary search
  //                       (*prod)(i,colb) += vala * matb_vals[k];
  //                     }
  //                  }
  //              }
  //          }
  //      },
  //      TasksPerThread(10));
  //   t2.Stop();
  //   return prod;
  // }

  template<> shared_ptr<SparseMatrix<double>>
  MatMultAB<SparseMatrix<double>,SparseMatrix<double>> (const SparseMatrix<double> & mata, const SparseMatrix<double> & matb)
  { return dynamic_pointer_cast<SparseMatrix<double>>(MatMult(mata,matb)); }
	     
  template<> shared_ptr<SparseMatrix<double>> TransposeSPM<SparseMatrix<double>> (const SparseMatrix<double> & mat)
  { return dynamic_pointer_cast<SparseMatrix<double>>(TransposeMatrix(mat)); } 
  
} // namespace amg
