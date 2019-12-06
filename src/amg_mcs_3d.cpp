#ifndef FILE_AMG_MCS_3D_CPP
#define FILE_AMG_MCS_3D_CPP

#include "amg.hpp"
#include "amg_facet_aux.hpp"
#include "amg_hdiv_templates.hpp"
#include "amg_vfacet_templates.hpp"
#include "amg_facet_aux_impl.hpp"

namespace amg
{

  using MCS_AMG_PC = FacetWiseAuxiliarySpaceAMG<3, HDivHighOrderFESpace, VectorFacetFESpace>;


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
    if (node_id.GetType() == NT_EDGE)
      { return; }
    // spacea->FESpace::GetDofNrs(node_id, dnums); // might this do the wrong thing in some cases ??
    const FESpace& F(*spacea); F.GetDofNrs(node_id, dnums);
    cout << " select A for id " << node_id << ", dnums are "; prow(dnums); cout << endl;
    cout << " select " << dnums[0] << " ";
    lam(0);
    int p = spacea->GetOrder(node_id);
    if (p >= 1) {
      cout << dnums[1] << " " << dnums[1+(p*(1+p))/2];
      // cout << 1 << " " << dnums[(1+p)] << endl;
      lam(1);
      // lam(1+p);
      lam(1+(p*(1+p))/2);
    }
    cout << endl;
  }


  /** 3d VectorFacetFESpace (in python: TangentialFacetFESpace) low order face DOFs are:
       0,1                  .. constant2
       1,2                  .. P1
       2*(p+1), 2*(p+1)+1   .. second p1
  **/
  template<> template<class TLAM> INLINE
  void MCS_AMG_PC :: ItLO_B (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    if (node_id.GetType() == NT_EDGE)
      { return; }
    // spaceb->FESpace::GetDofNrs(node_id, dnums);
    const FESpace& F(*spaceb); F.GetDofNrs(node_id, dnums);
    cout << " select B for id " << node_id << ", dnums are "; prow(dnums); cout << endl;
    cout << " select " << dnums[0] << " " << dnums[1] << " ";
    lam(0); lam(1);
    int p = spaceb->GetOrder(node_id);
    if (p >= 1) {
      /** Actually, I only need (0,x) and (y,0). (x,0) and (0,y) can be omitted.
       ?? This probably means 3, 2*(1+p) ?? **/
      // lam(3); lam(2*(1+p));
      cout << dnums[2] << " " << dnums[3] << " " << dnums[2*(1+p)] << " " << dnums[2*(1+p)+1];
      lam(2); lam(3);
      lam(2*(1+p)); lam(2*(1+p)+1);
    }
    cout << endl;
  }

  RegisterPreconditioner<FacetWiseAuxiliarySpaceAMG<3, HDivHighOrderFESpace, VectorFacetFESpace>> register_mcs_3d("ngs_amg.mcs3d");


  void ExportMCS3D (py::module & m) {
    ExportFacetAux<FacetWiseAuxiliarySpaceAMG<3, HDivHighOrderFESpace, VectorFacetFESpace>>
      (m, "mcs3d", "3d MCS elasticity auxiliary space AMG", [&](auto & x) { ; } );
  }

} // namespace amg

#endif
