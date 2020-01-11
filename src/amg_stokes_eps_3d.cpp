/** 2D, laplace + div div **/

namespace amg
{

  using MCS_STOKES_AMG_PC = FacetWiseAuxiliarySpaceAMG<3,
						       HDivHighOrderFESpace,
						       VectorFacetFESpace,
						       EmbedWithElmats<StokesAMGFactory<3, ElasticityAMGFactory<3>>>>;

  RegisterPreconditioner<MCS_STOKES_AMG_PC> reg_mcs_stokes_h1 ("ngs_amg.mcs_epseps_stokes_3d");

} // namespace amg
