#ifdef ELASTICITY

#define FILE_AGG_EL2D_CPP

#include "amg.hpp"
#include "amg_bla.hpp"
#include "amg_agg.hpp"
#include "amg_elast.hpp"
#include "amg_elast_impl.hpp"
#include "amg_agg_impl.hpp"


namespace amg
{
  template class Agglomerator<ElasticityAMGFactory<2>>;
}

#endif
