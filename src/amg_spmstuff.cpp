#define FILE_AMG_SPMSTUFF_CPP

#include "amg.hpp"

#include "amg_tcs.hpp"

namespace amg
{
  
  // A = B.T
  // template<int N, int M> INLINE void TM_Set_A_BT (Mat<N,M,double> & a, const Mat<M,N,double> & b)
  // { for (auto i : Range(N)) for (auto j : Range(M)) a(i,j) = b(j,i); }
  // INLINE void TM_Set_A_BT (double & a, const double & b) { a = b; }

  INLINE Timer & timer_hack_TransposeSPM (int nr) {
    switch(nr) {
    case(0): { static Timer t("AMG - TransposeMatrix 1"); return t; }
    case(1): { static Timer t("AMG - TransposeMatrix 2"); return t; }
    default: { static Timer t("TIMERERR"); return t; }
    }
  };
  template <typename TMA> shared_ptr<typename trans_spm<TMA>::type> TransposeSPM (const TMA & mat)
  {
    Timer & t1 = timer_hack_TransposeSPM(0);
    Timer & t2 = timer_hack_TransposeSPM(1);
    t1.Start();
    Array<int> cnt(mat.Width()); cnt = 0;
    for (int i : Range(mat.Height()))
      for (int c : mat.GetRowIndices(i))
	cnt[c]++;
    t1.Stop();
    t2.Start();
    auto trans = make_shared<typename trans_spm<TMA>::type>(cnt, mat.Height());
    cnt = 0;
    for (int i : Range(mat.Height())) {
      for (int ci : Range(mat.GetRowIndices(i))) {
	int c = mat.GetRowIndices(i)[ci];
	int pos = cnt[c]++;
	trans -> GetRowIndices(c)[pos] = i;
	trans -> GetRowValues(c)[pos] = Trans(mat.GetRowValues(i)[ci]);
      }
    }
    for (int r : Range(trans->Height())) {
      auto rowvals = trans->GetRowValues(r);
      BubbleSort (trans->GetRowIndices(r),
		  FlatArray<typename TM_OF_SPM<typename trans_spm<TMA>::type>::type> (rowvals.Size(), &rowvals(0)));
    }
    t2.Stop();
    return trans;
  } // TransposeSPM (..)


  INLINE Timer & timer_hack_MatMultAB (int nr) {
    switch(nr) {
    case(0): { static Timer t("AMG - sparse matrix multiplication"); return t; }
    case(1): { static Timer t("AMG - sparse matrix multiplication - setup a"); return t; }
    case(2): { static Timer t("AMG - sparse matrix multiplication - setup b"); return t; }
    case(3): { static Timer t("AMG - sparse matrix multiplication - setup b1"); return t; }
    case(4): { static Timer t("AMG - sparse matrix multiplication - mult"); return t; }
    default: { static Timer t("TIMERERR"); return t; }
    }
  }
  template <typename TSPMA, typename TSPMB>
  shared_ptr<typename mult_spm<TSPMA,TSPMB>::type> MatMultAB (const TSPMA & mata, const TSPMB & matb)
  {
    Timer & t = timer_hack_MatMultAB(0);
    Timer & t1a = timer_hack_MatMultAB(1);
    Timer & t1b = timer_hack_MatMultAB(2);
    Timer & t1b1 = timer_hack_MatMultAB(3);
    Timer & t2 = timer_hack_MatMultAB(4);

    // C = A * B
    typedef typename mult_spm<TSPMA,TSPMB>::type TSPMC;
    // typedef typename TM_OF_SPM<TSPMA>::type TMA;
    // typedef typename TM_OF_SPM<TSPMB>::type TMB;
    // typedef typename TM_OF_SPM<TSPMC>::type TMC;
    
    RegionTimer reg(t);
    t1a.Start();
    Array<int> cnt(mata.Height());
    cnt = 0;
    ParallelForRange
      (mata.Height(), [&] (IntRange r)
       {
         Array<int*> ptrs;
         Array<int> sizes;
         for (int i : r)
           {
             auto mata_ci = mata.GetRowIndices(i);
             ptrs.SetSize(mata_ci.Size());
             sizes.SetSize(mata_ci.Size());
             for (int j : Range(mata_ci))
               {
                 ptrs[j] = matb.GetRowIndices(mata_ci[j]).Addr(0);
                 sizes[j] = matb.GetRowIndices(mata_ci[j]).Size();
               }
             int cnti = 0;
             MergeArrays(ptrs, sizes, [&cnti] (int col) { cnti++; } );
             cnt[i] = cnti;
           }
       },
       TasksPerThread(10));
    t1a.Stop();
    t1b.Start();
    t1b1.Start();
    auto prod = make_shared<TSPMC>(cnt, matb.Width());
    prod->AsVector() = 0.0;
    t1b1.Stop();
    // fill col-indices
    ParallelForRange
      (mata.Height(), [&] (IntRange r)
       {
         Array<int*> ptrs;
         Array<int> sizes;
         for (int i : r)
           {
             auto mata_ci = mata.GetRowIndices(i);
             ptrs.SetSize(mata_ci.Size());
             sizes.SetSize(mata_ci.Size());
             for (int j : Range(mata_ci))
               {
                 ptrs[j] = matb.GetRowIndices(mata_ci[j]).Addr(0);
                 sizes[j] = matb.GetRowIndices(mata_ci[j]).Size();
               }
             int * ptr = prod->GetRowIndices(i).Addr(0);
             MergeArrays(ptrs, sizes, [&ptr] (int col)
                         {
                           *ptr = col;
                           ptr++;
                         } );
           }
       },
       TasksPerThread(10));
    t1b.Stop();
    t2.Start();
    ParallelForRange
      (mata.Height(), [&] (IntRange r)
       {
         struct thash { int idx; int pos; };
         size_t maxci = 0;
         for (auto i : r)
           maxci = max2(maxci, size_t (prod->GetRowIndices(i).Size()));
         size_t nhash = 2048;
         while (nhash < 2*maxci) nhash *= 2;
         ArrayMem<thash,2048> hash(nhash);
         size_t nhashm1 = nhash-1;
         for (auto i : r)
           {
             auto mata_ci = mata.GetRowIndices(i);
             auto matc_ci = prod->GetRowIndices(i);
             auto matc_vals = prod->GetRowValues(i);
	     const int mccis = matc_ci.Size();
             for (int k = 0; k < mccis; k++)
               {
                 size_t hashval = size_t(matc_ci[k]) & nhashm1; // % nhash;
                 hash[hashval].pos = k;
                 hash[hashval].idx = matc_ci[k];
               }
             for (int j : Range(mata_ci))
               {
                 auto vala = mata.GetRowValues(i)[j];
                 int rowb = mata.GetRowIndices(i)[j];
                 auto matb_ci = matb.GetRowIndices(rowb);
                 auto matb_vals = matb.GetRowValues(rowb);
		 const int mbcs = matb_ci.Size();
                 for (int k = 0; k < mbcs; k++)
                   {
                     auto colb = matb_ci[k];
                     unsigned hashval = unsigned(colb) & nhashm1; // % nhash;
                     if (hash[hashval].idx == colb)
                       { // lucky fast branch
                        matc_vals[hash[hashval].pos] += vala * matb_vals[k]; 
                       }
                     else
                      { // do the binary search
                        (*prod)(i,colb) += vala * matb_vals[k];
                      }
                   }
               }
           }
       },
       TasksPerThread(10));
    t2.Stop();
    return prod;
  }

  
} // namespace amg

