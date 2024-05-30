#ifndef FILE_LINEARSOLVER_HPP
#define FILE_LINEARSOLVER_HPP

#include <base.hpp>
#include <amg_matrix.hpp>

namespace amg
{

class LinearSolver : public BaseMatrix
{
public:

  LinearSolver(shared_ptr<BaseMatrix> A, int max_steps, double rtol, bool print = false, string prefix = "");

  virtual ~LinearSolver() = default;

  virtual void Solve(BaseVector &x, const BaseVector &b, bool initialize = false, bool is_x_zero = false);
  virtual void Solve(BaseVector &x, const BaseVector &b, BaseVector &res, bool initialize = false, bool is_x_zero = false, bool is_res_updated = false);

  void SetSaveErrors(bool _save_errs)   { save_errs = _save_errs; }
  void SetMaxIterations(int _max_steps) { max_steps = _max_steps; }
  void SetRelTolerance(double _rtol)    { rtol = _rtol; }

  bool GetConverged() const           { return converged; }
  int GetNeededSteps() const          { return needed_steps; }
  FlatArray<double> GetErrors() const { return errors; }

protected:
  virtual void InitializeSolve(BaseVector &x, const BaseVector &b, BaseVector &res, bool is_x_zero);

  virtual void DoIteration(BaseVector &x, const BaseVector &b, BaseVector &res) = 0;

  virtual double ComputeError(BaseVector &x, const BaseVector &b, BaseVector &res);

protected:
  shared_ptr<BaseMatrix> A;

  int max_steps;    // max. number of steps to do on this level
  bool check_error; // do we even check for error
  double rtol;      // (relative) error tolerance
  bool print;       // write output
  bool save_errs;   // do we save the errors
  string prefix;    // output prefix to make it more readable

  double init_err;  // initial error
  double curr_err;  // current error
  int curr_iter;    // current iteration

  shared_ptr<BaseVector> work_res, work_x;

  bool converged;
  int needed_steps;
  Array<double> errors;

public:
  /** BaseMatrix overloads **/
  virtual void Mult (const BaseVector & b, BaseVector & x) const override;
  virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
  virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
  // don't overload this here, we have no precond and "A" can create the wrong, non-parallle vectors
  // when running in serial
  // virtual AutoVector CreateVector () const override;
  // virtual AutoVector CreateColVector () const override;
  // virtual AutoVector CreateRowVector () const override;
  virtual int VHeight () const override;
  virtual int VWidth () const override;
  virtual bool IsComplex () const override { return false; }
}; // class LinearSolver


class AMGAsLinearSolver : public LinearSolver
{
public:
  AMGAsLinearSolver(shared_ptr<AMGMatrix> amg_mat, int max_steps, double rtol, bool print = false, string prefix = "");

  virtual AutoVector CreateVector () const override;
  virtual AutoVector CreateColVector () const override;
  virtual AutoVector CreateRowVector () const override;

protected:
  virtual void DoIteration(BaseVector &x, const BaseVector &b, BaseVector &res) override;
  
protected:
  shared_ptr<AMGMatrix> amg_mat;
}; // class AMGAsLinearSolver

// class SmootherAsLinearSolver ??

} // namespace amg

#endif // FILE_LINEARSOLVER_HPP