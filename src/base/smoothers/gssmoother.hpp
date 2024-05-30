#ifndef FILE_GSSMOOTHER_HPP
#define FILE_GSSMOOTHER_HPP

#include <condition_variable>

#include "base_smoother.hpp"
#include "hybrid_smoother.hpp"

namespace amg
{


/** 
 *  Sequential Gauss-Seidel. Can update residual during computation.
 *  If add_diag is given, adds uses (D + add_D)^-1 instead if D^-1.
 */
template<class TM>
class GSS3 : public BaseSmoother
{
protected:
  size_t H;
  shared_ptr<SparseMatrix<TM>> spmat;
  shared_ptr<BitArray> freedofs;
  bool pinv = false;
  Array<TM> dinv;
  size_t first_free, next_free;

  virtual void SmoothRESInternal (size_t first, size_t next, BaseVector &x, BaseVector &res, bool backwards) const;
  virtual void SmoothRHSInternal (size_t first, size_t next, BaseVector &x, const BaseVector &b, bool backwards) const;

public:
  using TSCAL = typename mat_traits<TM>::TSCAL;
  // using BS = mat_traits<TM>::HEIGHT;
  static constexpr int BS () { return ngbla::Height<TM>(); }
  using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

  GSS3 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset = nullptr, bool _pinv = false);

  /** take elements form repl_dinv as diagonal inverses instead of computing them from mat  **/
  GSS3 (shared_ptr<SparseMatrix<TM>> mat, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset = nullptr, bool _pinv = false);

protected:
  void SetUp (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset);

public:
  // smooth with RHS
  virtual void Smooth (size_t first, size_t next, BaseVector &x, const BaseVector &b) const
  { SmoothRHSInternal(first, next, x, b, false); }
  virtual void SmoothBack (size_t first, size_t next, BaseVector &x, const BaseVector &b) const
  { SmoothRHSInternal(first, next, x, b, true); }

  // smooth with residual and keep it up do date
  virtual void SmoothRES (size_t first, size_t next, BaseVector &x, BaseVector &res) const
  { SmoothRESInternal(first, next, x, res, false); }
  virtual void SmoothBackRES (size_t first, size_t next, BaseVector &x, BaseVector &res) const
  { SmoothRESInternal(first, next, x, res, true); }

  virtual void Smooth (BaseVector  &x, const BaseVector &b,
                        BaseVector  &res, bool res_updated,
                        bool update_res, bool x_zero) const override;

  virtual void SmoothBack (BaseVector  &x, const BaseVector &b,
                            BaseVector &res, bool res_updated,
                            bool update_res, bool x_zero) const override;

  shared_ptr<BitArray> GetFreeDofs () const { return freedofs; }

  virtual void CalcDiags (FlatArray<TM> replDiags);
  INLINE void CalcDiags () { this->CalcDiags(Array<TM>()); }

  virtual int VHeight () const override { return H; }
  virtual int VWidth () const override { return H; }
  virtual AutoVector CreateVector () const override
  { return make_unique<VVector<typename strip_vec<Vec<BS(), double>>::type>>(H); };
  virtual AutoVector CreateRowVector () const override { return CreateVector(); }
  virtual AutoVector CreateColVector () const override { return CreateVector(); }

  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

  virtual void PrintTo (ostream & os, string prefix = "") const override;


protected:
  void SmoothInternal (int type, BaseVector  &x, const BaseVector &b, BaseVector &res,
      bool res_updated = false, bool update_res = true, bool x_zero = false) const;

}; // class GSS3

/** END GSS3 **/


/** 
 *  Sequential Gauss-Seidel. Can update residual during computation.
 *  If add_diag is given, adds uses (D + add_D)^-1 instead if D^-1
 *  Compresses rows/cols from orig. sparse mat that it needs.
 *  Meant to be used only for "small" subsets of rows.
 */
template<class TM>
class GSS4
{
protected:
  Array<int> xdofs/*, resdofs*/;        // dofnrs we upadte, dofnrs we need (so also all neibs)
  shared_ptr<SparseMatrix<TM>> cA;  // compressed A
  Array<TM> dinv;
  bool pinv = false;

public:
  using TSCAL = typename mat_traits<TM>::TSCAL;
  static constexpr int BS () { return ngbla::Height<TM>(); }
  using TV = typename strip_vec<Vec<BS(),TSCAL>> :: type;

  GSS4 (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset = nullptr, bool _pinv = false);

  GSS4 (shared_ptr<SparseMatrix<TM>> A, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset = nullptr, bool _pinv = false);

  void MultAdd (double s, const BaseVector & b, BaseVector & x) const;

protected:

  void SetUp (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset);

  void CalcDiags ();

  template<class TLAM> INLINE void iterate_rows (TLAM lam, bool bw) const;

  virtual void SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const;
  virtual void SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const;

public:
  // smooth with RHS
  INLINE void Smooth (BaseVector &x, const BaseVector &b) const
  { SmoothRHSInternal(x, b, false); }
  INLINE void SmoothBack (BaseVector &x, const BaseVector &b) const
  { SmoothRHSInternal(x, b, true); }

  // smooth with residual and keep it up do date
  INLINE void SmoothRES (BaseVector &x, BaseVector &res) const
  { SmoothRESInternal(x, res, false); }
  INLINE void SmoothBackRES (BaseVector &x, BaseVector &res) const
  { SmoothRESInternal(x, res, true); }

  virtual void PrintTo (ostream & os, string prefix = "") const;
}; // class GSS4

/** end GSS4 **/


/** 
 * Hybrid smoother with local Gauss-Seidel smoothers
 */
template<class TM>
class HybridGSSmoother : public HybridSmoother<TM>
{
public:
  using TSCAL = typename mat_traits<TM>::TSCAL;
  using TV    = typename strip_vec<Vec<ngbla::Height<TM>(), typename mat_traits<TM>::TSCAL>> :: type;

  HybridGSSmoother (shared_ptr<BaseMatrix> _A,
                    shared_ptr<BitArray> _subset,
                    bool _pinv,
                    bool _overlap,
                    bool _in_thread,
                    bool _symm_loc = false,
                    int _nsteps_loc = 1);

  virtual ~HybridGSSmoother () = default;

  virtual void Finalize () override;

  void PrintTo (ostream & os, string prefix = "") const override;

protected:
  using SMOOTH_STAGE = typename HybridBaseSmoother<TSCAL>::SMOOTH_STAGE;

  void
  SmoothStageRHS (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b,
                  BaseVector                &res,
                  bool                const &x_zero) const override;

  void
  SmoothStageRes (SMOOTH_STAGE        const &stage,
                  SMOOTHING_DIRECTION const &direction,
                  BaseVector                &x,
                  BaseVector          const &b,
                  BaseVector                &res,
                  bool                const &x_zero) const override;

protected:

  shared_ptr<BitArray> subset;
  bool pinv = false;
  bool symm_loc = false;

  size_t split_ind;

  shared_ptr<GSS3<TM>> jac_loc, jac_exo;
  shared_ptr<GSS4<TM>> jac_ex;
}; // HybridGSSmoother

/** HybridGSSmoother **/

} // namespace amg

#endif // FILE_GSSMOOTHER_HPP
