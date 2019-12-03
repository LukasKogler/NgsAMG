#define FILE_AGG_H1SCAL_CPP

#include "amg.hpp"
#include "amg_bla.hpp"
#include "amg_agg.hpp"
#include "amg_agg_impl.hpp"

namespace amg
{
  template class Agglomerator<H1AMGFactory>;
}
