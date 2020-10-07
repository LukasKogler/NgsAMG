#ifdef ELASTICITY
#ifdef SPWAGG

#define FILE_SPWAGG_EL2D_CPP

#include "amg.hpp"
#include "amg_bla.hpp"
#include "amg_agg.hpp"
#include "amg_spwagg.hpp"
#include "amg_elast_impl.hpp"
#include "amg_spwagg_impl.hpp"


namespace amg
{
  using FCLASS = ElasticityAMGFactory<2>;
  template class SPWAgglomerator<FCLASS::ENERGY, FCLASS::TMESH, FCLASS::ENERGY::NEED_ROBUST>;
}

#endif // SPWAGG
#endif // ELASTICITY
