#ifndef FILE_UTILSE_SPARSELA_HPP
#define FILE_UTILSE_SPARSELA_HPP

#include <base.hpp>
#include <utils.hpp>

#include "utils_sparseMM.hpp"
#include "dyn_block.hpp"

/**
 * Utilities for sparse LA
 */
namespace amg
{

template<int H, int W>
constexpr bool isSparseMatrixCompiled()
{
  return IsMatrixCompiledTrait<H, W>::value;
}

shared_ptr<BaseMatrix> TransposeAGeneric (shared_ptr<BaseMatrix> A);

// mult mats. A~NxN/C->D, B~NxM/C->C
shared_ptr<BaseMatrix> MatMultABGeneric (shared_ptr<BaseMatrix> A, shared_ptr<BaseMatrix> B);
shared_ptr<BaseMatrix> MatMultABGeneric (BaseMatrix const &A, BaseMatrix const &B);

// attempts to make a sparse matrix out of a symbolic concatenation of base-matrices
shared_ptr<BaseMatrix> BaseMatrixToSparse (shared_ptr<BaseMatrix> A);

// returns a sparse-matrix even if parallel matrix is put in
[[deprecated("Use TransposeAGeneric instead!")]]
shared_ptr<BaseMatrix> TransposeSPMGeneric (shared_ptr<BaseMatrix> A);

// remove all entrise < thresh from matrix A (computes sum of vals for block-entries)
shared_ptr<BaseMatrix> CompressAGeneric(shared_ptr<BaseMatrix> A, double const &thresh = 1e-20);

INLINE BaseMatrix const * GetLocalMat(BaseMatrix const *A)
{
  if (auto par = dynamic_cast<ParallelMatrix const *>(A))
    { return par->GetMatrix().get(); }
  else
    { return A; }
}

INLINE BaseMatrix       * GetLocalMat(BaseMatrix       *A)
{
  if (auto par = dynamic_cast<ParallelMatrix const *>(A))
    { return par->GetMatrix().get(); }
  else
    { return A; }
}

INLINE BaseMatrix const & GetLocalMat(BaseMatrix const &A)
{
  if (auto par = dynamic_cast<ParallelMatrix const *>(&A))
    { return *par->GetMatrix(); }
  else
    { return A; }
}

INLINE BaseMatrix       & GetLocalMat(BaseMatrix       &A)
{
  if (auto par = dynamic_cast<ParallelMatrix const *>(&A))
    { return *par->GetMatrix(); }
  else
    { return A; }
}

INLINE shared_ptr<BaseMatrix> GetLocalMat(shared_ptr<BaseMatrix> A)
{
  if (auto par = dynamic_pointer_cast<ParallelMatrix>(A))
    { return par->GetMatrix(); }
  else
    { return A; }
}

INLINE BaseSparseMatrix const * GetLocalSparseMat(BaseMatrix const *A)
{
  return my_dynamic_cast<BaseSparseMatrix>(GetLocalMat(A), "GetLocalSparseMat");
}

INLINE BaseSparseMatrix const & GetLocalSparseMat(BaseMatrix const &A)
{
  return *my_dynamic_cast<BaseSparseMatrix>(GetLocalMat(&A), "GetLocalSparseMat");
}

INLINE BaseSparseMatrix       & GetLocalSparseMat(BaseMatrix       &A)
{
  return *my_dynamic_cast<BaseSparseMatrix>(GetLocalMat(&A), "GetLocalSparseMat");
}

INLINE shared_ptr<BaseSparseMatrix> GetLocalSparseMat(shared_ptr<BaseMatrix> A)
{
  return my_dynamic_pointer_cast<BaseSparseMatrix>(GetLocalMat(A), "GetLocalSparseMat");
}

template<class TM>
INLINE shared_ptr<stripped_spm_tm<TM>> GetLocalTMM(shared_ptr<BaseMatrix> A)
{
  return dynamic_pointer_cast<stripped_spm_tm<TM>>(GetLocalMat(A));
}


template<class TLAM>
INLINE void DispatchOverMatrixDimensions(BaseMatrix const &mat, TLAM lam, bool const &mustFind = true)
{
  auto const &locMat = GetLocalSparseMat(mat);

  bool found = false;

  Iterate<MAX_SYS_DIM> ([&](auto HM) {
    Iterate<MAX_SYS_DIM> ([&](auto WM) {
      constexpr int H = HM + 1;
      constexpr int W = WM + 1;
      if constexpr(isSparseMatrixCompiled<H, W>())
      {
        if (auto p = dynamic_cast<stripped_spm<Mat<H, W, double>> const *>(&locMat))
        {
          lam(*p, IC<H>(), IC<W>());
          found = true;
        }
      }
    });
  });

  if ( mustFind && (!found) )
  {
    throw Exception("DispatchOverMatrixDimensions - could not identify matrix!");
  }
}

template<class TLAM>
INLINE void DispatchOverMatrixDimensions(BaseMatrix &mat, TLAM lam, bool const &mustFind = true)
{
  auto &locMat = GetLocalSparseMat(mat);

  bool found = false;

  Iterate<MAX_SYS_DIM> ([&](auto HM) {
    Iterate<MAX_SYS_DIM> ([&](auto WM) {
      constexpr int H = HM + 1;
      constexpr int W = WM + 1;
      if constexpr(isSparseMatrixCompiled<H, W>())
      {
        if (auto p = dynamic_cast<stripped_spm<Mat<H, W, double>> *>(&locMat))
        {
          lam(*p, IC<H>(), IC<W>());
          found = true;
        }
      }
    });
  });

  if ( mustFind && (!found) )
  {
    throw Exception("DispatchOverMatrixDimensions - could not identify matrix!");
  }
}

template<class TLAM>
INLINE void DispatchOverMatrixDimensions(shared_ptr<BaseMatrix> mat, TLAM lam, bool const &mustFind = true)
{
  auto locMat = GetLocalSparseMat(mat);

  bool found = false;

  Iterate<MAX_SYS_DIM> ([&](auto HM) {
    Iterate<MAX_SYS_DIM> ([&](auto WM) {
      constexpr int H = HM + 1;
      constexpr int W = WM + 1;
      if constexpr(isSparseMatrixCompiled<H, W>())
      {
        if (auto p = dynamic_pointer_cast<stripped_spm<Mat<H, W, double>>>(locMat))
        {
          lam(p, IC<H>(), IC<W>());
          found = true;
        }
      }
    });
  });

  if ( mustFind && (!found) )
  {
    throw Exception("DispatchOverMatrixDimensions - could not identify matrix!");
  }
}

// workaround for sparse-matrix memory usage
// cast to TSPM_TM because "error: member 'GetMemoryUsage' found in multiple base classes of different types"
INLINE Array<MemoryUsage> GetMUHack (const BaseSparseMatrix & mat)
{
  Array<MemoryUsage> memory_use;

  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) { memory_use = A.GetMemoryUsage(); });

  return memory_use;
}

template<class TLAM>
INLINE void DispatchMatrixHeight(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    lam(IC<H>());  
  });
}

template<class TLAM>
INLINE void DispatchMatrixWidth(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    lam(IC<W>());  
  });
}

template<class TLAM>
INLINE void DispatchRectangularMatrix(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, lam);
}

template<class TLAM>
INLINE void DispatchRectangularMatrixBS(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    lam(IC<H>(), IC<W>());  
  });
}

template<class TLAM>
INLINE void DispatchSquareMatrix(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    if constexpr(H != W)
    {
      throw Exception("DispatchSquareMatrix - non-square matrix!");
    }
    else
    {
      lam(A, IC<H>());
    }
  });
}

template<class TLAM>
INLINE void DispatchSquareMatrix(shared_ptr<BaseMatrix> mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    if constexpr(H != W)
    {
      throw Exception("DispatchSquareMatrix - non-square matrix!");
    }
    else
    {
      lam(A, IC<H>());
    }
  });
}

template<class TLAM>
INLINE void DispatchSquareMatrixBS(BaseMatrix const &mat, TLAM lam)
{
  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) {
    if constexpr(H != W)
    {
      throw Exception("DispatchSquareMatrixBS - non-square matrix!");
    }
    else
    {
      lam(IC<H>());  
    }
  });
}

  
INLINE int GetEntryHeight (BaseMatrix const &mat)
{
  int AH = -1;

  if (auto ptr = dynamic_cast<DynBlockSparseMatrix<double> const *>(&mat))
  {
    return 1;
  }

  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) { AH = H; });

  return AH;
}

INLINE int GetEntryWidth (BaseMatrix const &mat)
{
  int AW = -1;

  if (auto ptr = dynamic_cast<DynBlockSparseMatrix<double> const *>(&mat))
  {
    return 1;
  }

  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) { AW = W; });

  return AW;
}

INLINE std::tuple<int, int> GetEntryDims (BaseMatrix const &mat)
{
  int AH = -1;
  int AW = -1;

  if (auto ptr = dynamic_cast<DynBlockSparseMatrix<double> const *>(&mat))
  {
    return std::make_tuple(1, 1);
  }

  DispatchOverMatrixDimensions(mat, [&](auto const &A, auto H, auto W) { AH = H; AW = W; });

  return std::make_tuple(AH, AW);
}

INLINE int GetEntrySize (BaseMatrix const &mat)
{
  int h = GetEntryHeight(mat), w = GetEntryWidth(mat);
  return (h>0 && w>0) ? h*w : -1;
}

INLINE int GetEntryDim (BaseMatrix const &mat)
{
  int h = GetEntryHeight(mat), w = GetEntryWidth(mat);
  return h == w ? h : -1;
}

size_t GetScalNZE (BaseMatrix const *pm);

INLINE int GetEntrySize (BaseMatrix const *mat) { return GetEntrySize(*mat); }
INLINE int GetEntryDim (BaseMatrix const *mat) { return GetEntryDim(*mat); }
INLINE std::tuple<int, int> GetEntryDims (BaseMatrix const *mat) { return GetEntryDims(*mat); }
INLINE int GetEntryHeight (BaseMatrix const *mat) { return GetEntryHeight(*mat); }
INLINE int GetEntryWidth (BaseMatrix const *mat) { return GetEntryWidth(*mat); }


INLINE size_t GetScalNZE (BaseMatrix const &pm) { return GetScalNZE(&pm); }


template<int D> struct SSSTrait { static constexpr bool val = false; };

template<> struct SSSTrait<1> { static constexpr bool val = true; };
template<> struct SSSTrait<2> { static constexpr bool val = true; };
template<> struct SSSTrait<3> { static constexpr bool val = true; };
#ifdef ELASTICITY
template<> struct SSSTrait<6> { static constexpr bool val = true; };
#endif

template<int BS> constexpr int isSmootherSupported()
{
  return SSSTrait<BS>::val;
}

template<int H>
shared_ptr<SparseMat<H, H>>
BuildPermutationMatrix (FlatArray<int> sort)
{
  size_t N = sort.Size();
  Array<int> epr(N); epr = 1.0;
  auto embed_mat = make_shared<SparseMat<H, H>>(epr, N);
  const auto & em = *embed_mat;
  for (auto k : Range(N)) {
    em.GetRowIndices(k)[0] = sort[k];
    SetIdentity(em.GetRowValues(k)[0]);
  }
  return embed_mat;
}

INLINE unique_ptr<BaseVector> CreateSuitableVector(size_t N, size_t BS, shared_ptr<ParallelDofs> pardofs, PARALLEL_STATUS stat = DISTRIBUTED)
{
  unique_ptr<BaseVector> vec = nullptr;
  if (N != size_t(-1))
  {
    Switch<MAX_SYS_DIM> (BS - 1, [&] (auto BSM) {
      constexpr int cxpr_BS = BSM + 1;
      if (pardofs == nullptr)
      {
        vec = make_unique<VVector<StripVec<cxpr_BS>>>(N);
      }
      else
      {
        vec = make_unique<ParallelVVector<StripVec<cxpr_BS>>>(pardofs, stat);
      }
    });
  }
  return vec;
} // CreateSuitableVector

INLINE shared_ptr<BaseVector> CreateSuitableSPVector(size_t N, size_t BS, shared_ptr<ParallelDofs> pardofs, PARALLEL_STATUS stat = DISTRIBUTED)
{
  shared_ptr<BaseVector> vec = nullptr;
  if (N != size_t(-1))
  {
    Switch<MAX_SYS_DIM> (BS - 1, [&] (auto BSM) {
      constexpr int cxpr_BS = BSM + 1;
      if (pardofs == nullptr)
      {
        vec = make_shared<VVector<StripVec<cxpr_BS>>>(N);
      }
      else
      {
        vec = make_shared<ParallelVVector<StripVec<cxpr_BS>>>(pardofs, stat);
      }
    });
  }
  return vec;
} // CreateSuitableVector


std::tuple<shared_ptr<BaseMatrix>, // AP
           shared_ptr<BaseMatrix>> // PT A P
RestrictMatrixKeepFactor(BaseMatrix const &A,
                         BaseMatrix const &P,
                         BaseMatrix const &PT);

} // namespace amg

#endif //  FILE_UTILSE_SPARSELA_HPP