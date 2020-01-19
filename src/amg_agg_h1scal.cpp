#define FILE_AGG_H1SCAL_CPP

#include "amg_agg.hpp"
#include "amg_bla.hpp"
#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"
#include "amg_agg_impl.hpp"

namespace amg
{
  template class AgglomerateCoarseMap<H1Mesh>;
  template class Agglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  template class Agglomerator<H1Energy<2, double, double>, H1Mesh, H1Energy<2, double, double>::NEED_ROBUST>;
  template class Agglomerator<H1Energy<3, double, double>, H1Mesh, H1Energy<3, double, double>::NEED_ROBUST>;
}
