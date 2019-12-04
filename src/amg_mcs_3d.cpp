#ifndef FILE_AMG_MCS_3D_CPP
#define FILE_AMG_MCS_3D_CPP

#include "amg.hpp"
#include "amg_facet_aux.hpp"
#include "amg_facet_aux_impl.hpp"

namespace amg
{

  using MCS_AMG_PC = FacetWiseAuxiliarySpaceAMG<3, HDivHighOrderFESpace, VectorFacetFESpace>;



  template<> template<class TELEM, class TMIP> INLINE
  void MCS_AMG_PC :: CSDS_A (const TELEM & fel, const TMIP & mip, FlatMatrix<double> s, FlatMatrix<double> sd)
  {
    fel.CalcShape(mip, s);
    fel.CalcDualShape(mip, sd);
  }


  /** VectorFacet has (and needs) no dual shapes **/
  template<> template<class TELEM, class TMIP> INLINE
  void MCS_AMG_PC :: CSDS_B (const TELEM & fel, const TMIP & mip, FlatMatrix<double> s, FlatMatrix<double> sd)
  {
    fel.CalcShape(mip, s);
    sd = s;
  }


  /** 3d HDiv low order face DOFs are:
      0          .. p0 DOF
      1          .. p1 DOF
      1 + order  .. p1 DOF (??)
      p*(1+p)/2  .. p1 ODF (??)
  **/
  template<> template<class TLAM> INLINE
  void MCS_AMG_PC :: ItLO_A (NodeId node_id, Array<int> & dnums, TLAM lam)
  {
    if (node_id.GetType() == NT_EDGE)
      { return; }
    // spacea->FESpace::GetDofNrs(node_id, dnums); // might this do the wrong thing in some cases ??
    const FESpace& F(*spacea); F.GetDofNrs(node_id, dnums);
    lam(0);
    int p = spacea->GetOrder(node_id);
    if (p >= 1) {
      lam(1);
      // lam(1+p);
      lam((p*(1+p))/2);
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
    if (node_id.GetType() == NT_EDGE)
      { return; }
    // spaceb->FESpace::GetDofNrs(node_id, dnums);
    const FESpace& F(*spaceb); F.GetDofNrs(node_id, dnums);
    lam(0); lam(1);
    int p = spaceb->GetOrder(node_id);
    if (p >= 1) {
      lam(2); lam(3);
      lam(2*(1+p)); lam(p*(1+p)+1);
    }
  }

  RegisterPreconditioner<FacetWiseAuxiliarySpaceAMG<3, HDivHighOrderFESpace, VectorFacetFESpace>> register_mcs_3d("ngs_amg.mcs3d");

} // namespace amg

#endif
