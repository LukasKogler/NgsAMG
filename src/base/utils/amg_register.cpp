
#include "amg_register.hpp"

namespace amg
{

AMGRegister&
AMGRegister::
getAMGRegister()
{
  static AMGRegister reg;
  return reg;
}


void
AMGRegister::
addAMGSolverImpl (std::string     const &name,
                  CreatorFunction        func)
{
  _amgTypes.Append(name);
  _amgCreators.Append(func);
} // AMGRegister::addAMGSolverImpl


shared_ptr<BaseAMGPC>
AMGRegister::
createAMGSolverImpl (std::string        const &amgType,
                     NGsAMGMatrixHandle const &A,
                     AMGSolverSettings  const &settings)
{
  int idx = findType(amgType);

  if (idx != -1)
  {
    auto pc = _amgCreators[idx](A, settings);

    return pc;
  }

  throw Exception("Could not find solver type!");

  return nullptr;
} // AMGRegister::createAMGSolverImpl

int AMGRegister :: findType(std::string const &amgType)
{
  for (auto k : Range(_amgTypes))
  {
    if (_amgTypes[k] == amgType)
    {
      return k;
    }
  }
  return -1;
} // AMGRegister::findType           


} // namespace amg