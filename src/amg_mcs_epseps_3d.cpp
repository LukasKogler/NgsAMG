#if defined(AUX_AMG) && defined(ELASTICITY)

#ifndef FILE_AMG_MCS_EPSEPS_3D_CPP
#define FILE_AMG_MCS_EPSEPS_3D_CPP

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

  extern template class ElmatVAMG<ElasticityAMGFactory<3>, double, double>;

  using MCS_AUX_SYS = FacetAuxSystem<3, HDivHighOrderFESpace, VectorFacetFESpace, FacetRBModeFE<3>>;

  template class FacetAuxSystem<3, HDivHighOrderFESpace, VectorFacetFESpace, FacetRBModeFE<3>>;

  using MCS_AMG_PC = FacetAuxVertexAMGPC<3, MCS_AUX_SYS, ElmatVAMG<ElasticityAMGFactory<3>, double, double>>;

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
    if (node_id.GetType() == NT_EDGE)
      { return; }
    const FESpace& F(*spacea); F.GetDofNrs(node_id, dnums);
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) { // for unused (true for interface/wirebasket)
	lam(0);
	int p = spacea->GetOrder(node_id);
	if (p >= 1) {
	  // cout << dnums[1] << " " << dnums[1+(p*(1+p))/2];
	  // cout << 1 << " " << dnums[(1+p)] << endl;
	  lam(1);
	  // lam(1+p);
	  lam(1+(p*(1+p))/2);
	}
	// cout << endl;
      }
    }
  } // MCS_AMG_PC::ItLO_A

  /** 3d VectorFacetFESpace (in python: TangentialFacetFESpace) low order face DOFs are:
       0,1                  .. constant2
       1,2                  .. P1
       2*(p+1), 2*(p+1)+1   .. second p1
  **/
  template<> template<class TLAM> INLINE
  void MCS_AUX_SYS :: ItLO_B (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    if (node_id.GetType() == NT_EDGE)
      { return; }
    // spaceb->FESpace::GetDofNrs(node_id, dnums);
    const FESpace& F(*spaceb); F.GetDofNrs(node_id, dnums);
    // cout << " select B for id " << node_id << ", dnums are "; prow(dnums); cout << endl;
    // cout << " select " << dnums[0] << " " << dnums[1] << " ";
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) { // for unused (true for interface/wirebasket)
	lam(0); lam(1);
	int p = spaceb->GetOrder(node_id);
	if (p >= 1) {
	  // cout << dnums[2] << " " << dnums[3] << " " << dnums[2*(1+p)] << " " << dnums[2*(1+p)+1];
	  lam(2); lam(3);
	  lam(2*(1+p)); lam(2*(1+p)+1);
	}
	// cout << endl;
      }
    }
  } // MCS_AMG_PC::ItLO_B

  template class FacetAuxVertexAMGPC<3, MCS_AUX_SYS, ElmatVAMG<ElasticityAMGFactory<3>, double, double>>;

  RegisterPreconditioner<MCS_AMG_PC> register_mcs_epseps_3d("ngs_amg.mcs_epseps_3d");

} // namespace amg


namespace amg
{
  void ExportMCS_epseps_3d (py::module & m) {
    ExportAuxiliaryAMG<MCS_AMG_PC> (m, "mcs_epseps_3d", "3d MCS elasticity auxiliary space AMG", [&](auto & x) { ; } );
  }
} // namespace amg

#endif

#endif

#endif // defined(AUX_AMG) && defined(ELASTICITY)
