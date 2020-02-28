#if defined(AUX_AMG) && defined(ELASTICITY)

#ifndef FILE_AMG_MCS_EPSEPS_2D_CPP
#define FILE_AMG_MCS_EPSEPS_2D_CPP

#ifdef ELASTICITY

#include "amg.hpp"

#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_nodal_impl.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_factory_vertex_impl.hpp"

#include "amg_blocksmoother.hpp"

#include "amg_pc.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_pc_vertex_impl.hpp"

#include "amg_elast.hpp"

#define FILE_AMG_ELAST_CPP // uargh!
#include "amg_elast_impl.hpp"
#undef FILE_AMG_ELAST_CPP

#include "amg_facet_aux.hpp"
#include "amg_hdiv_templates.hpp"
#include "amg_vfacet_templates.hpp"
#include "amg_facet_aux_impl.hpp"


#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

namespace amg
{

  extern template class ElmatVAMG<ElasticityAMGFactory<2>, double, double>;

  template class FacetAuxSystem<2, HDivHighOrderFESpace, VectorFacetFESpace, FacetRBModeFE<2>>;
  using MCS_AUX_SYS = FacetAuxSystem<2, HDivHighOrderFESpace, VectorFacetFESpace, FacetRBModeFE<2>>;

  using MCS_AMG_PC = FacetAuxVertexAMGPC<2, MCS_AUX_SYS, ElmatVAMG<ElasticityAMGFactory<2>, double, double>>;

  template<> INLINE void MCS_AUX_SYS :: Add_Vol (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
						 ElementId ei, LocalHeap & lh)
  {
    Add_Vol_simple(dnums, elmat, ei, lh);
  }


  /** 3d HDiv low order face DOFs are:
      0              .. p0 DOF
      1              .. p1 DOF
      1 + order      .. p1 DOF (??)
      1 + p*(1+p)/2  .. p1 ODF (??)
  **/
  template<> template<class TLAM> INLINE
  void MCS_AUX_SYS :: ItLO_A (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    const FESpace& F(*spacea); F.GetDofNrs(node_id, dnums);
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) { // for unused (true for interface/wirebasket)
	lam(0);
	int p = spacea->GetOrder(node_id);
	if (p >= 1) // why ?? 
	  { lam(1); }
      }
    }
  } // MCS_AMG_PC::ItLO_A


  /** 2d VectorFacetFESpace (in python: TangentialFacetFESpace) low order face DOFs are:
       0                  .. constant2
       1                  .. P1
       2*(p+1), 2*(p+1)+1   .. second p1
  **/
  template<> template<class TLAM> INLINE
  void MCS_AUX_SYS :: ItLO_B (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    const FESpace& F(*spaceb); F.GetDofNrs(node_id, dnums);
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) // for unused (true for interface/wirebasket)
	{ lam(0); }
    }
  } // MCS_AMG_PC::ItLO_B


  template class FacetAuxVertexAMGPC<2, MCS_AUX_SYS, ElmatVAMG<ElasticityAMGFactory<2>, double, double>>;

  RegisterPreconditioner<MCS_AMG_PC> register_mcs_epseps_2d("ngs_amg.mcs_epseps_2d");

} // namespace amg


namespace amg
{
  void ExportMCS_epseps_2d (py::module & m) {
    ExportFacetAux<MCS_AMG_PC> (m, "mcs_epseps_2d", "2d MCS elasticity auxiliary space AMG", [&](auto & x) { ; } );
  }
} // namespace amg

#endif

#endif

#endif // defined(AUX_AMG) && defined(ELASTICITY)
