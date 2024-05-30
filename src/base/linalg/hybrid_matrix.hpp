#ifndef FILE_AMG_HYBRID_MATRIX_HPP
#define FILE_AMG_HYBRID_MATRIX_HPP

#include <base.hpp>
#include <utils_sparseLA.hpp>

#include "dcc_map.hpp"
#include "dyn_block.hpp"

namespace amg
{

/**
 *   A hybrid matrix is a splitting of a normal NGSolve-stype ParallelMatrix into: (M .. master, G .. ghost)
 *
 *       A         =       M        +         G
 *
 *   A_MM  A_MG          M_MM  0           0    A_MG
 *   A_GM  A_GG           0    0          A_GM  G_GG
 *
 *   m = A_MM + Adist_GG
 *
 *   The MM-block of M is an actual diagonal block of the global matrix!
 *
 *   G_GG has entries ij where master(i) != master(j)
 */
template<class TSCAL>
class HybridBaseMatrix : public BaseMatrix
{
public:
  HybridBaseMatrix(shared_ptr<ParallelDofs>  parDOFs,
                   shared_ptr<DCCMap<TSCAL>> dCCMap,
                   shared_ptr<BaseMatrix>    aM = nullptr,
                   shared_ptr<BaseMatrix>    aG = nullptr);

  virtual ~HybridBaseMatrix() = default;

  shared_ptr<DCCMap<TSCAL>> GetDCCMapPtr ();

  DCCMap<TSCAL>       &GetDCCMap ();
  DCCMap<TSCAL> const &GetDCCMap () const;

  shared_ptr<BaseMatrix> GetLocalOp () const { return _MPG; }
  shared_ptr<BaseMatrix> GetM () const { return _M; };
  shared_ptr<BaseMatrix> GetG () const { return _G; };

  bool HasGLocal  () const { return _G != nullptr; }
  bool HasGGlobal () const { return !g_zero; }

  virtual int VHeight () const override { return GetM()->Height(); }
  virtual int VWidth  () const override { return GetM()->Width(); }

  virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const override;
  virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
  virtual void Mult (const BaseVector & x, BaseVector & y) const override;
  virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override;
  virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const override;
  virtual void MultTrans (const BaseVector & x, BaseVector & y) const override;

  virtual AutoVector CreateVector () const override;
  virtual AutoVector CreateRowVector () const override;
  virtual AutoVector CreateColVector () const override;

  bool IsComplex() const override;
  size_t NZE () const override;
  virtual size_t EntrySize () const { return 1; }

protected:
  void SetMG(shared_ptr<BaseMatrix> aM, shared_ptr<BaseMatrix> aG);

private:
  shared_ptr<DCCMap<TSCAL>> _dCCMap;

  shared_ptr<BaseMatrix> _M;
  shared_ptr<BaseMatrix> _G;
  shared_ptr<BaseMatrix> _MPG;

  bool dummy;  // actually just a local matrix
  bool g_zero; // G is zero/nullptr on all ranks
}; // HybridBaseMatrix


/**
 * Hybrid Matrix implementation for regular sparse matrices
 */
template<class TM>
class HybridMatrix : public HybridBaseMatrix<typename mat_traits<TM>::TSCAL>
{
public:
  using TSCAL = typename mat_traits<TM>::TSCAL;
  static constexpr int BS () { return ngbla::Height<TM>(); }
  using TV = StripVec<BS()>;

  // from a normal (sparse) matrix
  HybridMatrix (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map);

  // creates a simple DCC-map
  HybridMatrix (shared_ptr<BaseMatrix> mat);

  virtual ~HybridMatrix() = default;

  shared_ptr<SparseMatrix<TM>> GetSpM () const { return spM; }
  shared_ptr<SparseMatrix<TM>> GetSpG () const { return spG; }

  size_t EntrySize () const override;

  /**
   * This is purely for block-smoother where we replace spM by something differenct.
   * That is a hack, but I don't care for now.
   */
  void ReplaceM (shared_ptr<BaseMatrix> M);

protected:

  shared_ptr<SparseMatrix<TM>> spM, spG;

  void SetUpMats (shared_ptr<SparseMatrix<TM>> A);
}; // class HybridMatrix


/**
 * Hybrid Matrix implementation for dynamic block sparse matrices.
 */
template<class TSCAL>
class DynamicBlockHybridMatrix : public HybridBaseMatrix<TSCAL>
{
public:
  // from a normal (sparse) matrix
  DynamicBlockHybridMatrix (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map);

  // creates a simple DCC-map
  DynamicBlockHybridMatrix (shared_ptr<BaseMatrix> mat);

  virtual ~DynamicBlockHybridMatrix() = default;


  shared_ptr<DynBlockSparseMatrix<TSCAL>> GetDynSpM() const { return dynSpM; }
  shared_ptr<DynBlockSparseMatrix<TSCAL>> GetDynSpG() const { return dynSpG; }

private:

  shared_ptr<DynBlockSparseMatrix<TSCAL>> dynSpM;
  shared_ptr<DynBlockSparseMatrix<TSCAL>> dynSpG;
}; // class DynamicBlockHybridMatrix

} // namespace amg

#endif // FILE_AMG_HYBRID_MATRIX_HPP