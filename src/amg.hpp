#ifndef FILE_AMG
#define FILE_AMG

#include <comp.hpp>

namespace amg {
  using namespace std;
  using namespace ngcomp;
}

#include "mpiwrap_extension.hpp"
#include "amg_typedefs.hpp"  
#include "amg_spmstuff.hpp"  
#include "amg_utils.hpp"  
#include "eqchierarchy.hpp"  
#include "reducetable.hpp"
#include "amg_mesh.hpp"
#include "amg_map.hpp"
#include "amg_coarsen.hpp"
#include "amg_contract.hpp"
#include "amg_smoother.hpp"
#include "amg_smoother2.hpp"
#include "amg_matrix.hpp"
#include "amg_precond.hpp"
#include "amg_h1.hpp"
#include "amg_elast.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

#endif
