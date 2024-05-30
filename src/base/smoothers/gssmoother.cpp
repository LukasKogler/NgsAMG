
#include <utils_denseLA.hpp>

#include "gssmoother.hpp"
#include "hybrid_smoother.hpp"

#ifdef USE_TAU
#include <Profile/Profiler.h>
// #include "TAU.h"
#endif

namespace amg
{
template<class T>
INLINE void print_tmPF (ostream &os, const T & mat, std::string const &prefix)
{
  constexpr int H = mat_traits<T>::HEIGHT;
  constexpr int W = mat_traits<T>::WIDTH;
  for (int kH : Range(H)) {
    if (kH > 0)
      { os << prefix; }
    for (int jW : Range(W)) { os << mat(kH,jW) << " "; }
    os << endl;
  }
} // print_tm


template<>
INLINE void print_tmPF<FlatMatrix<double>> (ostream &os, const FlatMatrix<double> & mat, std::string const &prefix)
{
  const int H = mat.Height();
  const int W = mat.Width();
  for (int kH : Range(H)) {
    if (kH > 0)
      { os << prefix; }
    for (int jW : Range(W)) { os << mat(kH,jW) << " "; }
    os << endl;
  }
} // print_tm

INLINE void print_tmPF (ostream &os, double const &mat, std::string const &prefix)
{
  os << mat << endl;
}

INLINE void printEvals (std::ostream &os, double const &M, LocalHeap & lh)
{
  os << " scalar, no evals! " << endl;
} // CalcPseudoInverseWithTol

INLINE void printEvals (std::ostream &os, FlatMatrix<double> & M, LocalHeap & lh)
{
  // cout << " CPI FB for " << endl << M << endl;
  // static Timer t("CalcPseudoInverseFB"); RegionTimer rt(t);
  const int N = M.Height();
  FlatMatrix<double> evecs(N, N, lh);
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(M, evals, evecs);

  os << " evals "; prow(evals, os); os << endl;

} // CalcPseudoInverseWithTol

template<int N>
void printEvals (std::ostream &os, FlatMatrix<Mat<N, N, double>> mat, LocalHeap & lh)
{
  auto const H = mat.Height();
  auto const W = mat.Width();

  FlatMatrix<double> B(mat.Height() * N, mat.Width() * N, lh);

  ToFlat(mat, B);

  printEvals(os, B, lh);
}

template<int N>
void printEvals (std::ostream &os, Mat<N, N, double> const &mat, LocalHeap & lh)
{
  FlatMatrix<double> B(N, N, lh);

  Iterate<N>([&](auto const &k){
    Iterate<N>([&](auto const &j) {
      B(k.value, j.value) = mat(k.value, j.value);
    });
  });

  printEvals(os, B, lh);
}

/** GSS3 **/

template<class TM>
GSS3<TM> :: GSS3 (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset, bool _pinv)
  : BaseSmoother(mat), pinv(_pinv)
{
  SetUp(mat, subset);
  CalcDiags();
}


template<class TM>
GSS3<TM> :: GSS3 (shared_ptr<SparseMatrix<TM>> mat, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset, bool _pinv)
  : BaseSmoother(mat), pinv(_pinv)
{
  SetUp(mat, subset);
  CalcDiags(repl_diag);
}

template<class TM>
void GSS3<TM> :: SetUp (shared_ptr<SparseMatrix<TM>> mat, shared_ptr<BitArray> subset)
{
  spmat = mat;
  freedofs = subset;
  H = spmat->Height();
  const auto & A(*spmat);
  auto numset = freedofs ? freedofs->NumSet() : H;
  first_free = 0; next_free = A.Height();
  if (freedofs != nullptr) {
    if (freedofs->NumSet() == 0)
      { next_free = 0; }
    else if (freedofs->NumSet() != freedofs->Size()) {
      int c = 0;
      while(c < freedofs->Size()) {
        if (freedofs->Test(c))
          { first_free = c; break; }
        c++;
      }
      if (freedofs->Size()) {
        c = freedofs->Size() - 1;
        while( c != size_t(-1) ) {
          if (freedofs->Test(c))
            { next_free = c + 1; break; }
          c--;
        }
      }
    }
  }
} // GSS3::SetUp


template<class TM>
void GSS3<TM> :: CalcDiags (FlatArray<TM> repl_diag)
{
  dinv.SetSize (H);
  const auto& A(*spmat);
  LocalHeap glh(10 * 1024 * 1024, "for_pinv");
  ParallelForRange (IntRange(dinv.Size()), [&] ( IntRange r ) {
    LocalHeap lh = glh.Split();
    for (auto i : r) {
      if (!freedofs || freedofs->Test(i))
        {
          if (repl_diag.Size())
          {
            dinv[i] = repl_diag[i];
          }
          else
          {
            dinv[i] = A(i,i);
          }
          if (pinv)
            { HeapReset hr(lh); CalcPseudoInverseTryNormal(dinv[i], lh); }
          else
            { CalcInverse(dinv[i]); }
        }
      else
        { dinv[i] = TM(0.0); }
    }
  });
}


template<class TM>
void GSS3<TM> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  auto fvx = x.FV<TV>();
  auto fvb = b.FV<TV>();
  if (freedofs) {
    ParallelForRange (IntRange(dinv.Size()), [&] ( IntRange r ) {
  const auto & fds = *freedofs;
  for (auto rownr : r)
    if (fds.Test(rownr))
      { fvx(rownr) += s * dinv[rownr] * fvb(rownr); }
});
  }
  else {
    ParallelForRange (IntRange(dinv.Size()), [&] ( IntRange r ) {
  for (auto rownr : r)
    { fvx(rownr) += s * dinv[rownr] * fvb(rownr); }
});
  }
} // GSS3::MultAdd


template<class TM>
void GSS3<TM> :: SmoothRHSInternal (size_t first, size_t next, BaseVector &x, const BaseVector &b, bool backwards) const
{
#ifdef USE_TAU
  TAU_PROFILE("SmoothRHSInternal", TAU_CT(*this), TAU_DEFAULT);
#endif
  static Timer t(string("GSS3<bs=")+to_string(BS())+">::SmoothRHS");
  RegionTimer rt(t);

  const auto& A(*spmat);
  auto fds = freedofs.get();
  auto fvx = x.FV<TV>();
  auto fvb = b.FV<TV>();

  auto updateRow = [&](auto rownr) {
    TV r = A.RowTimesVector(rownr, fvx);
    fvx(rownr) += dinv[rownr] * (fvb(rownr) - r);
  };

  if (!backwards) {
    const int use_first = max2(first, first_free);
    const int use_next = min2(next, next_free);
    size_t rownr = use_first;
    if (fds) {
      if (use_next > use_first) // split_ind can be weird
        for ( int rownr : Range(use_first, use_next) )
          if (fds->Test(rownr)) {
            updateRow(rownr);
          }
    }
    else {
      if (use_next > use_first) // split_ind can be weird
        for ( int rownr : Range(use_first, use_next) ) {
          updateRow(rownr);
        }
    }
  }
  else {
    const int use_first = max2(first, first_free);
    const int use_next = min2(next, next_free);
    const int upf = use_first + 1;
    if (fds) {
      for (int rownr = use_next - 1; rownr >= upf; rownr--) {
        A.PrefetchRow(rownr-1);
        if (fds->Test(rownr)) {
          updateRow(rownr);
        }
      }
      if (use_next > use_first)
        if (fds->Test(use_first)) {
          updateRow(use_first);
        }
    } else {
      for (int rownr = use_next - 1; rownr >= upf; rownr--) {
        A.PrefetchRow(rownr-1);
        updateRow(rownr);
      }
      if (use_next > use_first) {
        updateRow(use_first);
      }
    }
  }
} // SmoothRHSInternal


template<class TM>
void GSS3<TM> :: SmoothRESInternal (size_t first, size_t next, BaseVector &x, BaseVector &res, bool backwards) const
{
#ifdef USE_TAU
  TAU_PROFILE("SmoothRESInternal", TAU_CT(*this), TAU_DEFAULT);
#endif
  static Timer t(string("GSS3<bs=")+to_string(BS())+">::SmoothRES");
  RegionTimer rt(t);

  const auto& A(*spmat);
  auto fds = freedofs.get();
  auto fvx = x.FV<TV>();
  auto fvr = res.FV<TV>();

  auto up_row = [&](auto rownr) LAMBDA_INLINE {
    TV w = -dinv[rownr] * fvr(rownr);
    A.AddRowTransToVector(rownr, w, fvr);
    fvx(rownr) -= w;
  };

  if (!backwards) {
    const size_t use_first = max2(first, first_free);
    const size_t use_next = min2(next, next_free);
    if (fds) {
      if (use_next > use_first) // split_ind can be weird
        for (auto rownr : Range(use_first, use_next))
          if (fds->Test(rownr))
            { up_row(rownr); }
    } else {
      if (use_next > use_first) // split_ind can be weird
        for (auto rownr : Range(use_first, use_next))
          { up_row(rownr); }
    }
  }
  else {
    const int use_first = max2(first, first_free);
    const int use_next = min2(next, next_free);
    const int upf = use_first + 1;
    if (fds) {
      for (int rownr = use_next - 1; rownr >= upf; rownr--) {
        A.PrefetchRow(rownr-1);
        if (fds->Test(rownr))
          { up_row(rownr); }
      }
      if ( (use_next > use_first) && (fds->Test(use_first)) )
        { up_row(use_first); }
    } else {
      for (int rownr = use_next - 1; rownr >= upf; rownr--) {
        A.PrefetchRow(rownr-1);
        up_row(rownr);
      }
      if (use_next > use_first)
        { up_row(use_first); }
    }
  }
} // GSS3::SmoothRESInternal

  template<class TM>
  void GSS3<TM> :: PrintTo (ostream & os, string prefix) const
  {
    os << prefix << "GSS3, BS = " << ngbla::Height<TM>() << ", H = " << H << endl;
    os << prefix << "  using pinv = " << pinv << endl;
    int nfree = (freedofs == nullptr) ? next_free - first_free : freedofs->NumSet();
    os << prefix << "  " << nfree << " free dofs in range [" << first_free << " " << next_free << ")" << endl;
    os << prefix << "free dinvs = " << endl;
    string spaces(prefix.size()+2, ' ');
    auto pA = my_dynamic_pointer_cast<stripped_spm_tm<TM>>(this->GetAMatrix(), "GSS3::PrintTo cast A!");
    auto const &A = *pA;
    std::string const prefix2 = prefix + "    ";
    LocalHeap lh(15*1024*1024, "whatever");
    for (auto k : Range(H)) {
       bool free = (freedofs == nullptr) ? true : freedofs->Test(k);
       if (free)
      {
        HeapReset hr(lh);
        os << prefix << "  " << k << ": " << endl;
        os << prefix << "    diag: " << endl;
        print_tmPF(os << prefix2, A(k,k), prefix2);
        printEvals(os, A(k,k), lh);
        os << prefix << "    diag inv : " << endl;
        print_tmPF(os << prefix2, dinv[k], prefix2);
        printEvals(os, dinv[k], lh);
        os << endl;
      }
    }

} // GSS3::PrintTo


template<class TM>
void GSS3<TM> :: Smooth (BaseVector  &x, const BaseVector &b,
                         BaseVector  &res, bool res_updated,
                         bool update_res, bool x_zero) const
{
  if (res_updated)
  {
    if (update_res) // keep res up to date
      { SmoothRESInternal(size_t(0), H, x, res, false); }
    else // forget about res and use RHS instead
      { SmoothRHSInternal(size_t(0), H, x, b, false); }
  }
  else
  {
    if (update_res)
    { // update residual with spmv, then keep it up-to date
      // TODO: benchmark whether residuum update before or after is faster!
      this->CalcResiduum(x, b, res, x_zero);
      SmoothRESInternal(size_t(0), H, x, res, false);
    }
    else
      { SmoothRHSInternal(size_t(0), H, x, b, false); }
  }
} // GSS3::Smooth


template<class TM>
void GSS3<TM> :: SmoothBack (BaseVector  &x, const BaseVector &b,
                             BaseVector &res, bool res_updated,
                             bool update_res, bool x_zero) const
{
  if (res_updated)
  {
    if (update_res) // keep res up to date
      { SmoothRESInternal(size_t(0), H, x, res, true); }
    else // forget about res and use RHS instead
      { SmoothRHSInternal(size_t(0), H, x, b, true); }
  }
  else
  {
    if (update_res) { // update residual with spmv, then keep it up-to date
      res = b;
      if (!x_zero)
      	{ res -= (*this->GetAMatrix()) * x; }
      SmoothRESInternal(size_t(0), H, x, res, true);
    }
    else
      { SmoothRHSInternal(size_t(0), H, x, b, true); }
  }
} // GSS3::SmoothBack


/** END GSS3 **/


/** GSS4 **/

template<class TM>
GSS4<TM> :: GSS4 (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset, bool _pinv)
  : pinv(_pinv)
{
  // cout << " GSS4, self diag " << endl;
  SetUp(A, subset);
  CalcDiags();
} // GSS4(..)


template<class TM>
GSS4<TM> :: GSS4 (shared_ptr<SparseMatrix<TM>> A, FlatArray<TM> repl_diag, shared_ptr<BitArray> subset, bool _pinv)
  : pinv(_pinv)
{
  // cout << " GSS4, repl diag " << endl;
  SetUp(A, subset);
  const auto& ncA(*cA);
  dinv.SetSize(xdofs.Size());
  LocalHeap glh(10 * 1024 * 1024, "for_pinv");
  ParallelForRange (IntRange(dinv.Size()), [&] ( IntRange r ) {
    LocalHeap lh = glh.Split();
    for (auto i : r) {
      if (repl_diag.Size())
        { dinv[i] = repl_diag[xdofs[i]]; }
      else
        { dinv[i] = ncA(i, i); } // yeah, that is correct
      if (pinv)
        { HeapReset hr(lh); CalcPseudoInverseTryNormal(dinv[i], lh); }
      else
        { CalcInverse(dinv[i]); }
    }
  });
} // GSS4(..)


template<class TM>
template<class TLAM>
INLINE void GSS4<TM> :: iterate_rows (TLAM lam, bool bw) const
{
  if (bw) {
    for (int rownr = int(cA->Height()) - 1; rownr >= 0; rownr--)
      { lam(rownr); }
  }
  else {
    for (auto k : Range(cA->Height()))
      { lam(k); }
  }
} // GSS4::iterate_rows

template<class TM>
void GSS4<TM> :: SetUp (shared_ptr<SparseMatrix<TM>> A, shared_ptr<BitArray> subset)
{
  if (subset && (subset->NumSet() != A->Height()) ) {
    /** xdofs / resdofs **/
    // Array<int> resdofs;
    size_t cntx = 0, cntres = 0;
    BitArray res_subset(subset->Size()); res_subset.Clear();
    for (auto k : Range(A->Height())) {
if (subset->Test(k)) {
  cntx++; res_subset.SetBit(k);
  for (auto j : A->GetRowIndices(k))
    { res_subset.SetBit(j); }
}
    }
    xdofs.SetSize(cntx); cntx = 0;
    // resdofs.SetSize(res_subset->NumSet()); cntres = 0;
    for (auto k : Range(A->Height())) {
if (subset->Test(k))
  { xdofs[cntx++] = k; }
// if (res_subset->Test(k))
//   { resdofs[cntres++] = k; }
    }
    /** compress A **/
    Array<int> perow(xdofs.Size()); perow = 0;
    for (auto k : Range(xdofs))
for (auto col : A->GetRowIndices(xdofs[k]))
  { if (res_subset.Test(col)) { perow[k]++; } }
    cA = make_shared<SparseMatrix<TM>>(perow); perow = 0;
    for (auto k : Range(xdofs)) {
auto ri = cA->GetRowIndices(k);
auto rv = cA->GetRowValues(k);
auto Ari = A->GetRowIndices(xdofs[k]);
auto Arv = A->GetRowValues(xdofs[k]);
int c = 0;
for (auto j : Range(Ari)) {
  auto col = Ari[j];
  if (res_subset.Test(col)) {
    ri[c] = col;
    rv[c++] = Arv[j];
  }
}
    }
  } // if (subset)
  else {
    xdofs.SetSize(A->Height()); //resdofs.SetSize(A->Height());
    for (auto k : Range(A->Height()))
{ xdofs[k] = /*resdofs[k] = */ k; }
    cA = A;
  }
  // cout << " compressed A " << endl;
  // print_tm_spmat(cout, *cA); cout << endl;
} // GSS4::SetUp


template<class TM>
void GSS4<TM> :: CalcDiags ()
{
  /** invert diag **/
  const auto& ncA(*cA);
  dinv.SetSize(xdofs.Size());
  LocalHeap glh(10 * 1024 * 1024, "for_pinv");
  ParallelForRange (IntRange(dinv.Size()), [&] ( IntRange r ) {
LocalHeap lh = glh.Split();
for (auto i : r) {
  dinv[i] = ncA(i, i);
  if (pinv)
    { HeapReset hr(lh); CalcPseudoInverseTryNormal(dinv[i], lh); }
  else
    { CalcInverse(dinv[i]); }
}
    });
} // GSS4::CalcDiags


template<class TM>
void GSS4<TM> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  auto fvx = x.FV<TV>();
  auto fvb = b.FV<TV>();
  ParallelForRange (IntRange(xdofs.Size()), [&] ( IntRange r ) {
for (auto rownr : r)
  { fvx(xdofs[rownr]) += s * dinv[rownr] * fvb(xdofs[rownr]); }
    });
} // GSS4::MultAdd


template<class TM>
void GSS4<TM> :: SmoothRESInternal (BaseVector &x, BaseVector &res, bool backwards) const
{
#ifdef USE_TAU
  TAU_PROFILE("GSS4::RES", "", TAU_DEFAULT);
#endif
  static Timer t("GSS4::RES"); RegionTimer rt(t);
  const auto& A(*cA);
  auto fvx = x.FV<TV>();
  auto fvr = res.FV<TV>();

  // cout << " SRI, res = " << res << endl;

  iterate_rows( [&](auto rownr) LAMBDA_INLINE {
auto w = -dinv[rownr] * fvr(xdofs[rownr]);
A.AddRowTransToVector(rownr, w, fvr);
fvx(xdofs[rownr]) -= w;
    }, backwards);

} // GSS4::SmoothRESInternal


template<class TM>
void GSS4<TM> :: SmoothRHSInternal (BaseVector &x, const BaseVector &b, bool backwards) const
{
#ifdef USE_TAU
  TAU_PROFILE("GSS4::RHS", "", TAU_DEFAULT);
#endif
  static Timer t("GSS4::RHS"); RegionTimer rt(t);

  // cout << " SRHI, rhs = " << b << endl;

  const auto& A(*cA);
  auto fvx = x.FV<TV>();
  auto fvb = b.FV<TV>();
  iterate_rows([&](auto rownr) LAMBDA_INLINE {
const TV r = fvb(xdofs[rownr]) - A.RowTimesVector(rownr, fvx);
fvx(xdofs[rownr]) += dinv[rownr] * r;
const TV w = dinv[rownr] * r;
    }, backwards);

} // GSS4::SmoothRHSInternal


template<class TM>
void GSS4<TM> :: PrintTo (ostream & os, string prefix) const
{
  os << prefix << "GSS4, BS = " << ngbla::Height<TM>() << ", compressed mat dims = " << cA->Height() << " x " << cA->Width() << endl;
  os << prefix << "  using pinv = " << pinv << endl;
  os << prefix << "  xdofs/dinvs = " << endl;
  string spaces(prefix.size()+2, ' ');
  for (auto k : Range(dinv))
    { os << spaces << k << ", dof = " << xdofs[k] << ": " << dinv[k] << endl; }
} // GSS4::PrintTo


/** END GSS4 **/


/** HybridGSSmoother **/

template<class TM>
HybridGSSmoother<TM>::
HybridGSSmoother (shared_ptr<BaseMatrix> _A,
                  shared_ptr<BitArray> _subset,
                  bool _pinv,
                  bool _overlap,
                  bool _in_thread,
                  bool _symm_loc,
                  int _nsteps_loc)
  : HybridSmoother<TM>(_A, _overlap, _in_thread, _nsteps_loc)
  , subset(_subset)
  , pinv(_pinv)
  , symm_loc(_symm_loc)
{
} // HybridGSSmoother (..)


template<class TM>
void
HybridGSSmoother<TM>::
Finalize ()
{
  auto &A = this->GetHybSparseA();

  auto& M = *A.GetSpM();

  auto pardofs = A.GetParallelDofs();
  auto m_dofs  = A.GetDCCMap().GetMasterDOFs();

  // if (pardofs != nullptr)
  //   for (auto k : Range(add_diag.Size()))
  // 	if ( ((!subset) || (subset->Test(k))) && ((!m_dofs) || (m_dofs->Test(k))) )
  // 	  { M(k,k) += add_diag[k]; }

  shared_ptr<BitArray> loc = subset, ex = nullptr;

  if (pardofs != nullptr)
  {
    loc = make_shared<BitArray>(M.Height()); loc->Clear();
    ex = make_shared<BitArray>(M.Height()); ex->Clear();

    for (auto k : Range(M.Height()))
    {
      if (m_dofs->Test(k))
      {
        auto dps = pardofs->GetDistantProcs(k);
        if (dps.Size())
          { ex->SetBit(k); }
        else
          { loc->SetBit(k); }
      }
    }

    if ( subset != nullptr )
    {
      loc->And(*subset);
      ex->And(*subset);
      // loc->Or(*ex);
    }
  }

  if (subset == nullptr) // all DOFs are free - probably good enough
  {
    split_ind = M.Height() / 2;
  }
  else
  { // probably not hugely important, but split loc dofs in actual halfs
    size_t cnt = 0, numset_half = loc->NumSet() / 2;
    split_ind = 0; // better not leave it at -1 for size 0 matrices
    for (auto k : Range(loc->Size()))
    {
      if ( ( loc->Test(k) ) &&
            ( cnt++ == numset_half ) )
        { split_ind = k; break; }
    }
  }

  // split_ind = 0; // A->Height();

  // if (subset) {
    // cout << "rank " << A->GetParallelDofs()->GetCommunicator().Rank() << " numsets " << loc->NumSet() << " " << ex->NumSet() << " " << loc->Size() << " "
    // 	   << double(loc->NumSet()) / loc->Size() << " " << double(ex->NumSet()) / ex->Size() << endl;
    // cout << "split and index " << split_ind << endl;
    // cout << "subset: " << endl << *subset << endl;
    // cout << "loc: " << endl << *loc << endl;
    // cout << "ex: " << endl << *ex << endl;
  // }

  auto mod_diag = this->CalcModDiag(subset);

  jac_loc = make_shared<GSS3<TM>>(A.GetSpM(), mod_diag, loc, pinv);

  if ( (pardofs != nullptr) && (ex->NumSet() != 0) )
    { jac_ex = make_shared<GSS4<TM>>(A.GetSpM(), mod_diag, ex, pinv); }

  // cout << " jac_loc: " << endl;
  // jac_loc->PrintTo(cout, "   ");
  // if (jac_ex) {
    // cout << " jac_ex: " << endl;
    // jac_ex->PrintTo(cout, "   ");
  // }
} // HybridGSSmoother::Finalize


template<class TM>
void
HybridGSSmoother<TM>::
SmoothStageRHS (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  auto const N = this->GetHybSparseA().Height();

  // cout << " GSS, smooth RHS " << direction << " on " << int(char(stage)) << endl;

  if ( symm_loc )
  {
    // SYMM:      F,-,F / -,FB,- / B,-,B

    switch(stage)
    {
      case(SMOOTH_STAGE::LOC_PART_1):
      {
        jac_loc->Smooth(0, N, x, b);
        break;
      }
      case(SMOOTH_STAGE::EX_PART):
      {
        if ( jac_ex != nullptr )
        {
          jac_ex->Smooth(x, b);
          jac_ex->SmoothBack(x, b);
        }
        break;
      }
      case(SMOOTH_STAGE::LOC_PART_2):
      {
        jac_loc->SmoothBack(0, N, x, b);
        break;
      }
    }
  }
  else
  {
    // NON SYMM:  F,-,- / -,F,- / -,-,F

    if (stage == SMOOTH_STAGE::EX_PART)
    {
      if ( jac_ex != nullptr )
      {
        if ( direction == FORWARD )
        {
          jac_ex->Smooth(x, b);
        }
        else
        {
          jac_ex->SmoothBack(x, b);
        }
      }
    }
    else
    {
      auto const useFirstPart = ( stage == SMOOTH_STAGE::LOC_PART_1 );

      auto const first = ( useFirstPart ) ? 0         : split_ind;
      auto const next  = ( useFirstPart ) ? split_ind : N;

      if ( direction == FORWARD )
      {
        jac_loc->Smooth(first, next, x, b);
      }
      else
      {
        jac_loc->SmoothBack(first, next, x, b);
      }
    }
  }
} // HybridGSSmoother<TM>::SmoothStageRHS


template<class TM>
void
HybridGSSmoother<TM>::
SmoothStageRes (SMOOTH_STAGE        const &stage,
                SMOOTHING_DIRECTION const &direction,
                BaseVector                &x,
                BaseVector          const &b,
                BaseVector                &res,
                bool                const &x_zero) const
{
  auto const N = this->GetHybSparseA().Height();

  // cout << " GSS, smooth RES " << direction << " on " << int(char(stage)) << endl;

  if ( symm_loc )
  {
    // SYMM:      F,-,F / -,FB,- / B,-,B

    switch(stage)
    {
      case(SMOOTH_STAGE::LOC_PART_1):
      {
        jac_loc->SmoothRES(0, N, x, res);
        break;
      }
      case(SMOOTH_STAGE::EX_PART):
      {
        if ( jac_ex != nullptr )
        {
          jac_ex->SmoothRES(x, res);
          jac_ex->SmoothBackRES(x, res);
        }
        break;
      }
      case(SMOOTH_STAGE::LOC_PART_2):
      {
        jac_loc->SmoothBackRES(0, N, x, res);
        break;
      }
    }
  }
  else
  {
    // NON SYMM:  F,-,- / -,F,- / -,-,F
    if (stage == SMOOTH_STAGE::EX_PART)
    {
      if ( jac_ex != nullptr )
      {
        if ( direction == FORWARD )
        {
          jac_ex->SmoothRES(x, res);
        }
        else
        {
          jac_ex->SmoothBackRES(x, res);
        }
      }
    }
    else
    {
      auto const useFirstPart = ( stage == SMOOTH_STAGE::LOC_PART_1 );

      auto const first = ( useFirstPart ) ? 0         : split_ind;
      auto const next  = ( useFirstPart ) ? split_ind : N;

      if ( direction == FORWARD )
      {
        jac_loc->SmoothRES(first, next, x, res);
      }
      else
      {
        jac_loc->SmoothBackRES(first, next, x, res);
      }
    }
  }
} // HybridGSSmoother<TM>::SmoothStageRes


template<class TM>
void
HybridGSSmoother<TM>::
PrintTo (ostream & os, string prefixA) const
{
  string prefix = prefixA + "  ";

  HybridSmoother<TM>::PrintTo(os, prefixA);

  os << prefix << "HybridGSSmoother, BS = " << ngbla::Height<TM>() << endl;
  os << prefix << "  use pinv = " << pinv << ", split_ind = " << split_ind << endl;
  if (subset == nullptr)
    { os << prefix << "  no subset (all free)! " << endl; }
  else
    { os << prefix << "  subset hat " << subset->NumSet() << " of " << subset->Size() << endl; }
  if (jac_loc == nullptr)
    { os << prefix << "  no local DOF smoother" << endl; }
  else
    { jac_loc->PrintTo(os, prefix + "  local DOF smoother "); }
  if (jac_ex == nullptr)
    { os << prefix << "  no ex DOF smoother" << endl; }
  else
    { jac_ex->PrintTo(os, prefix + "  ex DOF smoother "); }
} // HybridGSSmoother::PrintTo

/** END HybridGSSmoother **/

template class GSS3<double>;
template class GSS3<Mat<2, 2 , double>>;
template class GSS3<Mat<3, 3 , double>>;
#ifdef ELASTICITY
template class GSS3<Mat<6, 6 , double>>;
#endif

template class GSS4<double>;
template class GSS4<Mat<2, 2 , double>>;
template class GSS4<Mat<3, 3 , double>>;
#ifdef ELASTICITY
template class GSS4<Mat<6, 6 , double>>;
#endif

template class HybridGSSmoother<double>;
template class HybridGSSmoother<Mat<2, 2 , double>>;
template class HybridGSSmoother<Mat<3, 3 , double>>;
#ifdef ELASTICITY
template class HybridGSSmoother<Mat<6, 6 , double>>;
#endif
} // namespace amg
