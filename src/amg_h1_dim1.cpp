#define FILE_AMGH1_CPP
#define FILE_AMGH1_CPP_DIM1

#include "amg.hpp"

#include "amg_factory.hpp"
#include "amg_factory_impl.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_factory_vertex_impl.hpp"
#include "amg_pc.hpp"
#include "amg_pc_impl.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_pc_vertex_impl.hpp"
#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES


namespace amg
{

  // extern template class Agglomerator<H1AMGFactory>;

  template class H1Energy<1, double, double>;
  template class H1AMGFactory<1>;

  // using PCC = VertexAMGPC<H1AMGFactory<1>>;
  // using PCC = ElmatVAMG<H1AMGFactory<1>, double, double>;

  // template class PCC;

  // RegisterPreconditioner<PCC> register_h1amg_1d ("ngs_amg.h1_scal");

} // namespace amg


#include "python_amg.hpp"

namespace amg
{
  void ExportH1Scal (py::module & m)
  {
    // ExportAMGClass<ElmatVAMG<H1AMGFactory<1>, double, double>>(m, "ngs_amg.h1_scal", "", [&](auto & m) { ; } );
  };
}
