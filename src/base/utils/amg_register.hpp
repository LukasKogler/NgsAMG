#ifndef FILE_AMG_REGISTER_HPP
#define FILE_AMG_REGISTER_HPP

#include <base.hpp>

#include "amg_solver_settings.hpp"

namespace amg
{

class BaseAMGPC;

class NGsAMGMatrixHandle
{
public:
  NGsAMGMatrixHandle () = default;
  ~NGsAMGMatrixHandle () = default;

  virtual shared_ptr<BaseMatrix> getMatrix()       const = 0;
  virtual shared_ptr<BitArray>   getFreeDofs()     const = 0;
  virtual shared_ptr<BitArray>   getFreeScalRows() const = 0;
};

class AMGRegister
{
public:

  using CreatorFunction = std::function<shared_ptr<BaseAMGPC>(NGsAMGMatrixHandle const&, AMGSolverSettings const&)>;

  static
  AMGRegister&
  getAMGRegister ();

  static
  bool
  isRegistered (std::string const &amgType)
  {
    return getAMGRegister().findType(amgType) != -1;
  }

  static
  shared_ptr<BaseAMGPC>
  createAMGSolver (std::string        const &amgType,
                   NGsAMGMatrixHandle const &A,
                   AMGSolverSettings  const &settings)
  {
    return getAMGRegister().createAMGSolverImpl(amgType, A, settings);
  }

  static
  void
  addAMGSolver (std::string     const &name,
                CreatorFunction        func)
  {
    getAMGRegister().addAMGSolverImpl(name, func);
  }

private:

  AMGRegister() = default;
  ~AMGRegister() = default;

  void
  addAMGSolverImpl (std::string     const &name,
                    CreatorFunction        func);

  shared_ptr<BaseAMGPC>
  createAMGSolverImpl (std::string        const &amgType,
                       NGsAMGMatrixHandle const &A,
                       AMGSolverSettings  const &settings);

  int findType (std::string const &amgType);

  Array<std::string>     _amgTypes;
  Array<CreatorFunction> _amgCreators;
}; // AMGRegister


template<class AMG_CLASS>
class RegisterAMGSolver
{
public:
  RegisterAMGSolver(std::string const &name)
    : _name(name)
    , _regPC("NgsAMG." + name)
  {
    AMGRegister::getAMGRegister().addAMGSolver(
      name,
      [&](NGsAMGMatrixHandle const &A,
          AMGSolverSettings  const &settings) {
      return make_shared<AMG_CLASS>(A.getMatrix(), settings.getFlags(), _name + "FromRegisterAMGSolver");
    });
  }

private:
  std::string _name;
  RegisterPreconditioner<AMG_CLASS> _regPC;
};

} // namespace amg

#endif //FILE_AMG_REGISTER_HPP
