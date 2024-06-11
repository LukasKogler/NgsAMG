#define FILE_UTILS_SPARSEMM_CPP

#include <base.hpp>
#include <universal_dofs.hpp>

#include "dyn_block.hpp"
#include "utils.hpp"
#include "utils_arrays_tables.hpp"

#include "utils_sparseMM.hpp"


// undef USE_<INVERSE> such that we don't get problems with missing headers
// #if MAX_SYS_DIM < AMG_MAX_SYS_DIM
// #undef USE_UMFPACK
// #undef USE_MUMPS
// #else
//   #ifdef USE_MUMPS
//     #include <mumpsinverse.hpp>
//   #endif
//   #ifdef USE_UMFPACK
//     #include <umfpackinverse.hpp>
//   #endif
// #endif
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

} // namespace amg


/**
 * template instantiations
 *     do not use instantiations in header but have these here explicitly
 *     to hopefully fix missing symbols on apple
 */
 
namespace ngla
{
/**
 * NGSolve instantiates NxN, 1xN and Nx1 up to N=MAX_SYS_DIM.
 * The rest of the needed sparse matrices are compiled into the AMG library.
 */

#define InstSPMS(N,M)				  \
  template class SparseMatrixTM<Mat<N,M,double>>; \
  template class SparseMatrix<Mat<N,M,double>>; \

#define InstSPM2(N,M)\
  InstSPMS(N, M); \
  InstSPMS(M, N);\

// does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
// template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;

#if MAX_SYS_DIM < 2
  InstSPM2(1,2);
  InstSPMS(2,2);
#endif

#if MAX_SYS_DIM < 3
  InstSPM2(1,3);
  InstSPMS(3,3);
#endif
InstSPM2(2,3);

#ifdef ELASTICITY

// 1x4, 4x4, 1x5, 5x5 would be compiled into NGSolve with MAX_SYS_DIM large enough
// so to keep it simple instantiate them too for smaller MAX_SYS_DIM
#if MAX_SYS_DIM < 4
  InstSPM2(1,4);
  InstSPMS(4,4);
#endif

#if MAX_SYS_DIM < 5
  InstSPM2(1,5);
  InstSPMS(5,5);
#endif

#if MAX_SYS_DIM < 6
  InstSPM2(1,6);
  InstSPMS(6,6);
#endif

  InstSPM2(3,6);

#endif // ELASTICITY

} // namespace ngla


/** Sparse-Matrix transpose */

namespace amg
{
#define InstTransMat(H,W) \
  template shared_ptr<SparseMat<W, H>>	\
  TransposeSPMImpl<H,W>(SparseMatTM<H, W> const &A); \

#define InstTransMat2(N,M) \
  InstTransMat(M,N); \
  InstTransMat(N,M);

  InstTransMat(1,1);

  InstTransMat2(1,2);
  InstTransMat(2,2);

  InstTransMat2(1,3);
  InstTransMat2(2,3);
  InstTransMat(3,3);


#ifdef ELASTICITY
InstTransMat2(1,4);
InstTransMat(4,4);

InstTransMat2(1,5);
InstTransMat(5,5);

InstTransMat2(1,6);
InstTransMat2(3,6);
InstTransMat(6,6);
#endif //ELASTICITY

#undef InstTransMat
#undef InstTransMat2
}


/** Sparse-Matrix multiplication */

namespace amg
{

#define InstMultMat(A,B,C)						\
  template shared_ptr<SparseMat<A,C>>			\
  MatMultABImpl<A,B,C> (SparseMatTM<A, B> const &matAB, SparseMatTM<B, C> const &matBC); \
  InstMultMatUpdate(A,B,C); \

#define InstMultMatUpdate(A, B, C) \
  template void \
  MatMultABUpdateValsImpl<A,B,C> (SparseMatTM<A, B> const &mata, SparseMatTM<B, C> const &matb, SparseMatTM<A, C> &prod); \

#define InstProlMults(N,M) /* embedding NxN to MxM */	\
  InstMultMat(N,M,M); /* conctenate prols */		\
  InstMultMat(N,N,M); /* A * P */			\
  InstMultMat(M,N,M); /* PT * [A*P] */

#define InstProlMults2(N,M) \
  InstProlMults(N,M); \
  InstProlMults(M,N); \
  
InstMultMat(1,1,1);

InstMultMat(2,2,2);
InstProlMults2(1,2);

InstProlMults2(1,3);
InstProlMults2(2,3);
InstMultMat(3,3,3);

#ifdef ELASTICITY
InstProlMults2(1,4);
InstMultMat(4,4,4);

InstProlMults2(1,5);
InstMultMat(5,5,5);

InstProlMults2(1,6);
InstProlMults2(3,6);
InstMultMat(6,6,6);
#endif // ELASTICITY

#undef InstProlMults2
#undef InstProlMults
#undef InstMultMatUpdate
#undef InstMultMat

} // namespace amg


