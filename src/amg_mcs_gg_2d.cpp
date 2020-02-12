#ifdef AUX_AMG

#ifndef FILE_AMG_MCS_GG_2D_CPP
#define FILE_AMG_MCS_GG_2D_CPP


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

#include "amg_h1.hpp"

#define FILE_AMGH1_CPP // uargh!
#include "amg_h1_impl.hpp"
#undef FILE_AMGH1_CPP

#include "amg_facet_aux.hpp"
#include "amg_hdiv_templates.hpp"
#include "amg_vfacet_templates.hpp"
#include "amg_facet_aux_impl.hpp"


#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

namespace amg
{

  extern template class ElmatVAMG<H1AMGFactory<2>, double, double>;

  using MCS_AMG_PC = FacetWiseAuxiliarySpaceAMG<2,
						HDivHighOrderFESpace,
						VectorFacetFESpace,
						FacetH1FE<2>,
						ElmatVAMG<H1AMGFactory<2>, double, double>>;


  template<> INLINE void MCS_AMG_PC :: Add_Vol (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
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
  void MCS_AMG_PC :: ItLO_A (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    // spacea->FESpace::GetDofNrs(node_id, dnums); // might this do the wrong thing in some cases ??
    const FESpace& F(*spacea); F.GetDofNrs(node_id, dnums);
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) // for unused (true for interface/wirebasket)
	{ lam(0); }
    }
  }


  /** 3d VectorFacetFESpace (in python: TangentialFacetFESpace) low order face DOFs are:
       0,1                  .. constant2
       1,2                  .. P1
       2*(p+1), 2*(p+1)+1   .. second p1
  **/
  template<> template<class TLAM> INLINE
  void MCS_AMG_PC :: ItLO_B (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    // spaceb->FESpace::GetDofNrs(node_id, dnums);
    const FESpace& F(*spaceb); F.GetDofNrs(node_id, dnums);
    if (dnums.Size() > 0) { // for unused + compressed (definedon, refined)
      auto ct0 = F.GetDofCouplingType(dnums[0]);
      if ( ct0 & EXTERNAL_DOF ) // for unused (true for interface/wirebasket)
	{ lam(0); }
    }
  }


  // template<> shared_ptr<BaseSmoother> MCS_AMG_PC :: BuildFLS () const
  // {
    // return nullptr;
  // } // FacetWiseAuxiliarySpaceAMG::BuildFLS


  RegisterPreconditioner<MCS_AMG_PC> register_mcs_gg_2d("ngs_amg.mcs_gg_2d");

} // namespace amg


namespace amg
{
  void ExportMCS_gg_2d (py::module & m) {
    ExportFacetAux<MCS_AMG_PC> (m, "mcs_gg_2d", "2d MCS H1 auxiliary space AMG", [&](auto & x) { ; } );
  }
} // namespace amg

#endif


#endif // AUX_AMG
