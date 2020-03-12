#ifndef FILE_AMG
#define FILE_AMG

#include <comp.hpp>

namespace amg {
  using namespace std;
  using namespace ngcomp;
}

#ifdef USE_TAU
#include "TAU.h"
#endif

#include "mpiwrap_extension.hpp"
#include "amg_typedefs.hpp"  
#include "amg_spmstuff.hpp"  
#include "amg_utils.hpp"  // needs to come after spmstuff
#include "amg_bla.hpp" // by now, this is unfortunately used in a bunch of places
#include "eqchierarchy.hpp"  
#include "reducetable.hpp"
#include "amg_mesh.hpp"

// #include "amg_map.hpp"
// #include "amg_agg.hpp"
// #include "amg_contract.hpp"
// #include "amg_coarsen.hpp"
// #include "amg_discard.hpp"
// #include "amg_smoother.hpp"
// #include "amg_smoother2.hpp"
// #include "amg_smoother3.hpp"
// #include "amg_blocksmoother.hpp"
// #include "amg_matrix.hpp"

#endif
