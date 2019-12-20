
/** 2D, laplace + div div **/

#include "amg.hpp"
#include "amg_elast_impl.hpp"
#include "amg_factory_impl.hpp"
#include "amg_pc_impl.hpp"
#include "amg_facet_aux.hpp"
#include "amg_hdiv_templates.hpp"
#include "amg_vfacet_templates.hpp"
#include "amg_facet_aux_impl.hpp"
#include "amg_stokes.hpp"
#include "amg_stokes_impl.hpp"

namespace amg
{

  using STOKES_FACTORY = StokesAMGFactory<2,
					  H1StokesMesh<2>,
					  H1Energy<2, H1StokesVData<2>, H1StokesEData<2>>>;
  template class STOKES_FACTORY;

  using STOKES_PC = StokesAMGPC<STOKES_FACTORY>;
  template class STOKES_PC;

  using MCS_STOKES_AMG_PC = FacetWiseAuxiliarySpaceAMG<2,
						       HDivHighOrderFESpace,
						       VectorFacetFESpace,
						       FacetH1FE<2>,
						       STOKES_PC>;
  template class MCS_TOKES_AMG_PC;

  RegisterPreconditioner<MCS_STOKES_AMG_PC> reg_mcs_stokes_h1 ("ngs_amg.mcs_h1_stokes_2d");

} // namespace amg
