#ifdef SPWAGG

#define FILE_SPWAGG_H1SCAL_CPP

#include "amg_agg.hpp"
#include "amg_spwagg.hpp"
#include "amg_bla.hpp"
#include "amg_h1.hpp"
#include "amg_h1_impl.hpp"
#include "amg_spwagg_impl.hpp"

namespace amg
{
  template class SPWAgglomerateCoarseMap<H1Mesh>;
  template class SPWAgglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  template class SPWAgglomerator<H1Energy<2, double, double>, H1Mesh, H1Energy<2, double, double>::NEED_ROBUST>;
  template class SPWAgglomerator<H1Energy<3, double, double>, H1Mesh, H1Energy<3, double, double>::NEED_ROBUST>;
}

#endif // SPWAGG
