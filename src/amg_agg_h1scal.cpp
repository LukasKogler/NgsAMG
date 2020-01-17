#define FILE_AGG_H1SCAL_CPP

#include "amg.hpp"
#include "amg_bla.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_agg.hpp"
#include "amg_agg_impl.hpp"
#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_pc.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"

namespace amg
{
  template class Agglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  // template class Agglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  template class Agglomerator<H1Energy<3, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
}
