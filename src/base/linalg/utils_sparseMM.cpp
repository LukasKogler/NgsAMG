
#include <base.hpp>
#include <universal_dofs.hpp>

#include "dyn_block.hpp"
#include "utils.hpp"
#include "utils_arrays_tables.hpp"

#define FILE_UTILS_SPARSEMM_CPP
#include "utils_sparseMM.hpp"
#undef FILE_UTILS_SPARSEMM_CPP

// #ifdef USE_MUMPS
// #include <mumpsinverse.hpp>
// #endif

// #ifdef USE_UMFPACK
// // #include <umfpackinverse.hpp>
// #endif

// undef USE_<INVERSE> such that we don't get problems with missing headers
#undef USE_UMFPACK
#undef USE_MUMPS
#include <sparsematrix_impl.hpp>

namespace ngbla
{
  /** So we can have SparseMatrix<Mat<1,N,double>, Vec<N,double>, double> ! **/
  template<class TB>
  INLINE double & operator+= (double & a, const Expr<TB> & b) { a += b.Spec()(0,0); return a; }

  template<int H>
  INLINE Mat<H,1,double> operator* (const Mat<H,1,double> & mat, const double & x)
  {
    Mat<H,1,double> res; for (int i = 0; i < H; i++) res(i,0) = mat(i,0) * x;
    return res;
  }
}

namespace amg
{
  // A = B.T
  // template<int N, int M> INLINE void TM_Set_A_BT (Mat<N,M,double> & a, const Mat<M,N,double> & b)
  // { for (auto i : Range(N)) for (auto j : Range(M)) a(i,j) = b(j,i); }
  // INLINE void TM_Set_A_BT (double & a, const double & b) { a = b; }

  INLINE Timer<TTracing, TTiming>& timer_hack_TransposeSPMImpl ()
  {
    static Timer t("AMG - TransposeMatrix");
    return t;
  };

  template<int H, int W>
  shared_ptr<SparseMat<W, H>>
  TransposeSPMImpl (SparseMatTM<H, W> const &mat)
  {
    RegionTimer rt(timer_hack_TransposeSPMImpl());

    Array<int> cnt(mat.Width());
    cnt = 0;

    for (int i : Range(mat.Height())) {
      for (int c : mat.GetRowIndices(i))
      	{ cnt[c]++; }
    }

    auto trans = make_shared<SparseMat<W, H>>(cnt, mat.Height());

    cnt = 0;

    for (int i : Range(mat.Height()))
    {
      for (int ci : Range(mat.GetRowIndices(i)))
      {
        int c = mat.GetRowIndices(i)[ci];
        int pos = cnt[c]++;
        trans -> GetRowIndices(c)[pos] = i;
        trans -> GetRowValues(c)[pos] = Trans(mat.GetRowValues(i)[ci]);
      }
    }

    // is this needed??
    for (int r : Range(trans->Height()))
    {
      auto rowvals = trans->GetRowValues(r);

      BubbleSort (trans->GetRowIndices(r),
 		              FlatArray<typename TM_OF_SPM<SparseMatTM<W, H>>::type> (rowvals.Size(), rowvals.Data()));
    }

    return trans;
  } // TransposeSPMImpl (..)


INLINE Timer<TTracing, TTiming>& timer_hack_MatMultABImpl (int nr) {
  switch(nr) {
  case(0): { static Timer t("AMG - sparse matrix multiplication"); return t; }
  case(1): { static Timer t("AMG - sparse matrix multiplication - setup a"); return t; }
  case(2): { static Timer t("AMG - sparse matrix multiplication - setup b"); return t; }
  case(3): { static Timer t("AMG - sparse matrix multiplication - setup b1"); return t; }
  case(4): { static Timer t("AMG - sparse matrix multiplication - mult"); return t; }
  default: { static Timer t("TIMERERR"); return t; }
  }
}

template <int A, int B, int C>
shared_ptr<SparseMat<A, C>>
MatMultABImpl (SparseMatTM<A, B> const &mata,
               SparseMatTM<B, C> const &matb)
{
  Timer<TTracing, TTiming>& t = timer_hack_MatMultABImpl(0);
  Timer<TTracing, TTiming>& t1a = timer_hack_MatMultABImpl(1);
  Timer<TTracing, TTiming>& t1b = timer_hack_MatMultABImpl(2);
  Timer<TTracing, TTiming>& t1b1 = timer_hack_MatMultABImpl(3);
  Timer<TTracing, TTiming>& t2 = timer_hack_MatMultABImpl(4);

  double t2_time = t2.GetTime();

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
  auto prod = make_shared<SparseMat<A, C>>(cnt, matb.Width());
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
  t2_time = t2.GetTime() - t2_time;

  // auto gcomm = NgMPI_Comm(NG_MPI_COMM_WORLD);
  // if ( (gcomm.Size() == 1) || (gcomm.Rank() == 1) ) {
  //   cout << "(" << mata.Height() << ":" << EntryHeight<TMA>() << ")x("
  //         << mata.Width() << ":" << EntryWidth<TMA>() << ") X "
  //         << "(" << matb.Height() << ":" << EntryHeight<TMB>() << ")x("
  //         << matb.Width() << ":" << EntryWidth<TMB>() << "), "
  //         << " SPMM T2 = " << t2_time << endl;
  // }

  return prod;
}

INLINE Timer<TTracing, TTiming>& timer_hack_MatMultABUpdateValsImpl () {
  static Timer t("AMG - sparse matrix multiplication value update");
  return t;
}

template <int A, int B, int C>
void
MatMultABUpdateValsImpl (SparseMatTM<A, B> const &mata,
                         SparseMatTM<B, C> const &matb,
                         SparseMatTM<A, C>       &prod)
{
  Timer<TTracing, TTiming>& t = timer_hack_MatMultABUpdateValsImpl();

  double t_time = t.GetTime();

  t.Start();

  prod.AsVector() = 0.0;

  ParallelForRange(
    mata.Height(),
    [&] (IntRange r)
      {
        struct thash { int idx; int pos; };
        size_t maxci = 0;
        for (auto i : r)
          maxci = max2(maxci, size_t (prod.GetRowIndices(i).Size()));
        size_t nhash = 2048;
        while (nhash < 2*maxci) nhash *= 2;
        ArrayMem<thash,2048> hash(nhash);
        size_t nhashm1 = nhash-1;
        for (auto i : r)
        {
          auto mata_ci = mata.GetRowIndices(i);
          auto matc_ci = prod.GetRowIndices(i);
          auto matc_vals = prod.GetRowValues(i);
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
                prod(i,colb) += vala * matb_vals[k];
              }
            }
          }
        }
    },
    TasksPerThread(10)
  );

  t.Stop();
  t_time = t.GetTime() - t_time;

  // auto gcomm = NgMPI_Comm(NG_MPI_COMM_WORLD);
  // if ( (gcomm.Size() == 1) || (gcomm.Rank() == 1) ) {
  //   cout << "(" << mata.Height() << ":" << EntryHeight<TMA>() << ")x("
  //         << mata.Width() << ":" << EntryWidth<TMA>() << ") X "
  //         << "(" << matb.Height() << ":" << EntryHeight<TMB>() << ")x("
  //         << matb.Width() << ":" << EntryWidth<TMB>() << "), "
  //         << " SPMM CAL_UP = " << t_time << endl;
  // }
}

// fix some missing symbols
#define InstTransMat(H,W)						\
  SPARSE_MM_EXTERN template shared_ptr<SparseMat<W, H>>	\
  TransposeSPMImpl<H,W>(SparseMatTM<H, W> const &A); \

  InstTransMat(2,6);
  InstTransMat(6,1);
  InstTransMat(6,3);

#undef InstTransMat


} // namespace amg


namespace ngla
{
#define InstSPM(N,M)				  \
  template class SparseMatrix<Mat<N,M,double>>; \

// template class SparseMatrix<Mat<5,6,double>>;

}

  // /**
//  * Now the explicit instantiations
//  */
// namespace ngla
// {
// #define InstSPMS(N,M)				  \
//   template class SparseMatrixTM<Mat<N,M,double>>; \
//   template class SparseMatrix<Mat<N,M,double>>;

//   // this does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
//   // EXTERN template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;

// #if MAX_SYS_DIM < 2
//   InstSPMS(2,2);
//   InstSPMS(1,2);
//   InstSPMS(2,1);
// #endif // MAX_SYS_DIM < 2
// #if MAX_SYS_DIM < 3
//   InstSPMS(3,3);
//   InstSPMS(1,3);
//   InstSPMS(3,1);
// #endif // MAX_SYS_DIM < 3
//   InstSPMS(2,3);
//   InstSPMS(3,2);
// #ifdef ELASTICITY
// #if MAX_SYS_DIM < 6
//   InstSPMS(6,6);
//   InstSPMS(1,6);
//   InstSPMS(6,1);
// #endif // MAX_SYS_DIM < 6
//   InstSPMS(3,6);
//   InstSPMS(6,3);
// #endif // ELASTICITY

// /**
//  * Even with large enough block-size, NGSolve instantiates SparseMatrix<Mat<1, N>> with Vec<1> col-vecs
//  * we want these matrices with just "double" col-vecs, so instantiate that here.
//  * The SparseMatrixTM are already instantiated, and the <1,1> case is just a normal sparse-mat which is
//  * handled correctly.
//  *
//  *
//  * ACTUALLY, cannot instantiate these!, there are some "*" operators that are not defined for
//  * Mat<1,N> and double
// */
// // #define InstStrippedSPM(N) \
// //   template class SparseMatrix<Mat<1, N, double>, double,          Vec<N, double>>; \
// //   template class SparseMatrix<Mat<N, 1, double>, Vec<N, double>, double>;

// // InstStrippedSPM(2);
// // InstStrippedSPM(3);

// // // Do I even need that??
// // // #ifdef ELASTICITY
// // //   InstStrippedSPM(6);
// // // #endif

// // STOKES workaround again...
// InstSPMS(2,6);
// InstSPMS(6,2);

// // elasticity workaround - no idea why we need those symbols suddenly
// InstSPMS(5,6);
// InstSPMS(6,5);


// #undef InstSPMS
// } // namespace ngla

// namespace amg
// {
// #define InstTransMat(N,M)						\
//   template shared_ptr<trans_spm_tm<stripped_spm_tm<Mat<N,M,double>>>>	\
//   TransposeSPMImpl<stripped_spm_tm<Mat<N,M,double>>> (const stripped_spm_tm<Mat<N,M,double>> & mat);

//   /** [A \times B] Transpose **/
//   InstTransMat(1,1);
//   InstTransMat(1,2);
//   InstTransMat(2,1);
//   InstTransMat(2,2);
//   InstTransMat(3,3);
//   InstTransMat(1,3);
//   InstTransMat(3,1);
//   InstTransMat(2,3);
// #ifdef ELASTICITY
//   InstTransMat(1,6);
//   InstTransMat(3,6);
//   InstTransMat(6,6);
// #endif //ELASTICITY

// // STOKES workaround, instantiate some extra stuff
// InstTransMat(1,4);
// InstTransMat(4,1);
// InstTransMat(1,5);
// InstTransMat(5,1);
// InstTransMat(3,2);
// InstTransMat(6,2);

// // elasticity workaround
// InstTransMat(6,5);
// InstTransMat(5,6);

// #undef InstTransMat

// #define InstMultMat(A,B,C)						\
//   template shared_ptr<stripped_spm_tm<Mat<A,C,double>>>			\
//   MatMultABImpl<stripped_spm_tm<Mat<A,B,double>>> (const stripped_spm_tm<Mat<A,B,double>> & mata, const stripped_spm_tm<Mat<B,C,double>> & matb);

// #define InstEmbedMults(N,M) /* embedding NxN to MxM */	\
//   InstMultMat(N,M,M); /* conctenate prols */		\
//   InstMultMat(N,N,M); /* A * P */			\
//   InstMultMat(M,N,M); /* PT * [A*P] */

//   /** [A \times B] * [B \times C] **/
//   InstMultMat(1,1,1);
//   InstMultMat(2,2,2);
//   InstMultMat(3,3,3);
//   InstEmbedMults(1,2);
//   InstEmbedMults(2,1);
//   InstEmbedMults(1,3);
//   InstEmbedMults(3,1);
//   InstEmbedMults(2,3);
// #ifdef ELASTICITY
//   InstMultMat(6,6,6);
//   InstEmbedMults(1,6);
//   InstEmbedMults(2,6);
//   InstEmbedMults(3,6);
// #endif // ELASTICITY

// // // STOKES workaround, instantiate some extra stuff
// InstMultMat(1,4,1);
// InstMultMat(1,5,1);
// InstMultMat(1,6,1);
// InstMultMat(2,3,2);
// InstMultMat(2,6,2);
// InstMultMat(3,2,1);
// InstMultMat(3,2,2);
// InstMultMat(3,3,2);
// InstMultMat(4,1,1);
// InstMultMat(4,4,1);
// InstMultMat(5,1,1);
// InstMultMat(5,5,1);
// InstMultMat(6,1,1);
// InstMultMat(6,2,1);
// InstMultMat(6,2,2);
// InstMultMat(6,6,2);
// InstEmbedMults(1,4);
// InstEmbedMults(1,5);

// // elasticity workaround - no idea why we need those symbols suddenly
// InstEmbedMults(5,6);
// InstMultMat(6,5,5);
// InstMultMat(5,6,5);
// InstMultMat(6,6,5);


// #undef InstMultMat
// #undef InstEmbedMults

// } // namespace amg


