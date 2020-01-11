
/** 3D, laplace + div div **/

namespace amg
{

  using MCS_STOKES_AMG_PC = FacetWiseAuxiliarySpaceAMG<3,
						       HDivHighOrderFESpace,
						       VectorFacetFESpace,
						       EmbedWithElmats<StokesAMGFactory<3, H1AMGFactoryV2<3>>>>;

  RegisterPreconditioner<MCS_STOKES_AMG_PC> reg_mcs_stokes_h1 ("ngs_amg.mcs_h1_stokes_3d");
} // namespace amg
