
#include "LinearSolver.hpp"

namespace amg
{

/** LinearSolver **/

LinearSolver :: LinearSolver (shared_ptr<BaseMatrix> _A, int _max_steps, double _rtol, bool _print, string _prefix)
  : BaseMatrix(_A->GetParallelDofs())
  , A(_A)
  , max_steps(_max_steps)
  , rtol(_rtol)
  , print(_print)
  , prefix(_prefix)
{
  ;
} // LinearSolver(..)


void LinearSolver :: Solve (BaseVector &x, const BaseVector &b, bool initialize, bool is_x_zero)
{
  if (work_res == nullptr)    
    { work_res = CreateColVector(); }
  Solve(x, b, *work_res, initialize, is_x_zero, false);
} // LinearSolver::Solve


void LinearSolver :: Solve (BaseVector &x, const BaseVector &b, BaseVector &res, bool initialize, bool is_x_zero, bool is_res_updated)
{
  bool check_err = this->check_error && (this->rtol > 0);
  bool compute_err = this->print || check_err || save_errs;

  b.Distribute();

  bool xzero = is_x_zero;
  if (initialize)
  {
    x = 0.0;
    x.SetParallelStatus(CUMULATED);
  }
  if (!is_res_updated)
  {
    res = b;
    if (!xzero)
    {
      A->MultAdd(-1.0, x, res);
    }
  }

  curr_iter = -1;
  init_err = compute_err ? ComputeError(x, b, res) : -1.0;
  curr_err = init_err;
  converged = false;

  if (save_errs)
  {
    errors.SetSize(1 + max_steps);
    errors[0] = init_err;
  }

  double prev_err = init_err;

  std::string use_prefix = prefix;
  if (use_prefix.size() > 0)
    { use_prefix += std::string(", "); }

  auto do_output = [&]() {
    std::cout << use_prefix
              << "iter = " << curr_iter + 1
              << ", err = " << curr_err
              << ", rel err = " << curr_err / init_err
              << ", reduction = " << curr_err / prev_err
              << std::endl;
    prev_err = curr_err;
  };


  bool master = (A->GetParallelDofs() == nullptr) || (A->GetParallelDofs()->GetCommunicator().Rank() == 0);

  if (master && print)
    { do_output(); }

  InitializeSolve(x, b, res, is_x_zero);

  for (curr_iter = 0; (curr_iter < max_steps) && (!converged); curr_iter++)
  {
    DoIteration(x, b, res);
    
    if (compute_err)
      { curr_err = ComputeError(x, b, res); }

    if (save_errs)
      { errors[1 + curr_iter] = curr_err; }

    if (master && print)
      { do_output(); }

    if (check_error)
      { converged = ( curr_err < rtol * init_err ); }
  }

  needed_steps = converged ? (curr_iter + 1) : -1;

} // LinearSolver::Solve


double LinearSolver :: ComputeError (BaseVector &x, const BaseVector &b, BaseVector &res)
{
  // I guess per default use L2 norm
  return L2Norm(res);
} // LinearSolver::ComputeError


void LinearSolver :: InitializeSolve(BaseVector &x, const BaseVector &b, BaseVector &res, bool is_x_zero)
{
  ;
} // LinearSolver::InitializeSolve


void LinearSolver :: Mult (const BaseVector & b, BaseVector & x) const
{
  // bahhh I hate it
  const_cast<LinearSolver&>(*this).Solve(x, b, true, false);
} // LinearSolver::Mult


void LinearSolver :: MultTrans (const BaseVector & b, BaseVector & x) const
{
  Mult(b, x);
} // LinearSolver::MultTrans


void LinearSolver :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
{
  if (work_x == nullptr) // should we make work_x mutable?
    { const_cast<shared_ptr<BaseVector>&>(work_x) = shared_ptr<BaseVector>(CreateColVector()); }
  Mult(b, *work_x);
  x += s * (*work_x);
} // LinearSolver::MultAdd


void LinearSolver :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
{
  MultAdd(s, b, x);
} // LinearSolver::MultTransAdd


int LinearSolver :: VHeight () const
{
  return A->VHeight();
} // LinearSolver::VHeight


int LinearSolver :: VWidth () const
{
  return A->VWidth();
} // LinearSolver::VWidth


/** END LinearSolver **/


/** AMGAsLinearSolver **/


AMGAsLinearSolver :: AMGAsLinearSolver (shared_ptr<AMGMatrix> _amg_mat, int max_steps, double rtol, bool print, string prefix)
  : LinearSolver(_amg_mat->GetSmoother(0)->GetAMatrix(), max_steps, rtol, print, prefix)
  , amg_mat(_amg_mat)
{
  amg_mat = _amg_mat;
} // AMGAsLinearSolver(..)


void AMGAsLinearSolver :: DoIteration (BaseVector &x, const BaseVector &b, BaseVector &res)
{
  // res_up, up_res, x_zero
  amg_mat->SmoothVFromLevel(0, x, b, res, true, true, false);
} // AMGAsLinearSolver::DoIteration


AutoVector AMGAsLinearSolver :: CreateVector () const
{
  auto vec = amg_mat->CreateColVector();
  cout << " AMGAsLinearSolver create VEC, type = " << typeid(*vec).name() << " PSTAT = " << vec.GetParallelStatus() << std::endl;
  return vec;
} // AMGAsLinearSolver::CreateVector


AutoVector AMGAsLinearSolver :: CreateColVector () const
{
  return CreateVector();
} // AMGAsLinearSolver::CreateColVector


AutoVector AMGAsLinearSolver :: CreateRowVector () const
{
  return CreateVector();
} // AMGAsLinearSolver::CreateRowVector

/** END AMGAsLinearSolver **/


} // namespace amg