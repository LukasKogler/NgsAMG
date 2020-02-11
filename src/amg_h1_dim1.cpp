#define FILE_AMGH1_CPP
#define FILE_AMGH1_CPP_DIM1

#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES


namespace amg
{

  template class H1AMGFactory<1>;
  template class VertexAMGPC<H1AMGFactory<1>>;
  template class ElmatVAMG<H1AMGFactory<1>, double, double>;

  using PCC = ElmatVAMG<H1AMGFactory<1>, double, double>;

  RegisterPreconditioner<PCC> register_h1amg_1d ("ngs_amg.h1_scal");

} // namespace amg


#include "python_amg.hpp"

namespace amg
{
  void ExportH1Scal (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<H1AMGFactory<1>, double, double>>(m, "h1_scal", "", [&](auto & m) { ; } );
  };
}
