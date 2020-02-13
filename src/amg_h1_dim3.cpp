#define FILE_AMGH1_CPP
#define FILE_AMGH1_CPP_DIM3

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
#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES


namespace amg
{

  using ENC = H1Energy<3, double, double>;

  // extern template class Agglomerator<H1AMGFactory>;
  extern template class SeqVWC<FlatTM>;
  extern template class BlockVWC<H1Mesh>;
  extern template class HierarchicVWC<H1Mesh>;
  extern template class CoarseMap<H1Mesh>;
  extern template class Agglomerator<ENC, H1Mesh, H1Energy<3, double, double>::NEED_ROBUST>;
  extern template class CtrMap<Vec<3, double>>;
  extern template class GridContractMap<H1Mesh>;
  extern template class VDiscardMap<H1Mesh>;

  // template class H1Energy<1, double, double>;
  template class H1AMGFactory<3>;
  template class VertexAMGPC<H1AMGFactory<3>>;
  template class ElmatVAMG<H1AMGFactory<3>, double, double>;

  // using PCC = VertexAMGPC<H1AMGFactory<3>>;
  using PCC = ElmatVAMG<H1AMGFactory<3>, double, double>;

  // template class PCC;

  RegisterPreconditioner<PCC> register_h1amg_3d ("ngs_amg.h1_3d");

} // namespace amg


#include "python_amg.hpp"

namespace amg
{
  void ExportH1Dim3 (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<H1AMGFactory<3>, double, double>>(m, "h1_3d", "", [&](auto & m) { ; } );
  }
}
