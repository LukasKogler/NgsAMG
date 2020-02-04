#ifdef ELASTICITY

#define FILE_AMG_ELAST_CPP
#define FILE_AMG_ELAST_3D_CPP

#include "amg.hpp"

#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_nodal_impl.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_factory_vertex_impl.hpp"
#include "amg_pc.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_pc_vertex_impl.hpp"
#include "amg_elast.hpp"
#include "amg_elast_impl.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

namespace amg
{
  template class ElasticityAMGFactory<3>;
  template class VertexAMGPC<ElasticityAMGFactory<3>>;

  using PCC = ElmatVAMG<ElasticityAMGFactory<3>, double, double>;

  RegisterPreconditioner<PCC> register_elast_3d ("ngs_amg.elast_3d");
} // namespace amg

#include "python_amg.hpp"

namespace amg
{
  void ExportElast3d (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<ElasticityAMGFactory<3>, double, double>>(m, "elast_3d", "", [&](auto & m) { ; } );
  };
} // namespace amg

#endif // ELASTICITY
