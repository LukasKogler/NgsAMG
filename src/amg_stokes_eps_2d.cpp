
/** 2D, laplace + div div **/

namespace amg
{

  using MCS_STOKES_AMG_PC = FacetWiseAuxiliarySpaceAMG<2,
						       HDivHighOrderFESpace,
						       VectorFacetFESpace,
						       EmbedWithElmats<StokesAMGFactory<2, ElasticityAMGFactory<2>>>>;

  RegisterPreconditioner<MCS_STOKES_AMG_PC> reg_mcs_stokes_h1 ("ngs_amg.mcs_epseps_stokes_2d");

} // namespace amg
