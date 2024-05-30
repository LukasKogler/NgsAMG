#ifndef FILE_NC2D_HPP
#define FILE_NC2D_HPP

#include <base.hpp>

#include "facet_aux_info.hpp"

namespace amg
{

/**
 * Class for the facet-trace of a NC element - this is just super super simple!
 */
template<int DIM>
class NCH1FacetTrace
{
public:
  NCH1FacetTrace () { ; }
  NCH1FacetTrace (NodeId facet_id, MeshAccess & ma) { ; }
  static constexpr int ND = DIM;
  INLINE void CalcMappedShape (const BaseMappedIntegrationPoint & mip,
        SliceMatrix<double> shapes) const
  {
    shapes = 0;
    Iterate<DIM>([&](auto k) { shapes(k.value, k.value) = 1.0; });
    // for (auto k : Range(DIM))
    //   { shapes(k,k) = 1; }
  }
}; // class NCH1FacetTrace

template <ELEMENT_TYPE ET> class NoCoH1Shape;

template <ELEMENT_TYPE ET,
          class SHAPES = NoCoH1Shape<ET>,
          class BASE = T_ScalarFiniteElement< SHAPES, ET> >
class NoCoH1Element : public BASE, public ET_trait<ET>
{
  enum { DIM = ET_trait<ET>::DIM };

  // using ScalarFiniteElement<DIM>::ndof;
  // using ScalarFiniteElement<DIM>::order;

  using FiniteElement::ndof;
  using FiniteElement::order;

  using ET_trait<ET>::N_VERTEX;
  using ET_trait<ET>::N_EDGE;
  using ET_trait<ET>::N_FACE;
  using ET_trait<ET>::N_CELL;
  using ET_trait<ET>::FaceType;
  using ET_trait<ET>::GetEdgeSort;
  using ET_trait<ET>::GetFaceSort;
  using ET_trait<ET>::PolDimension;
  using ET_trait<ET>::PolBubbleDimension;

public:

  INLINE NoCoH1Element ()
  {
    order = 1;
    ndof = ElementTopology::GetNFacets(ET);
    // cout << " NCH EL, ET = " << ET << " -> ND = " << ndof << endl;
  }

  INLINE ~NoCoH1Element () { ; }

}; // class NoCoH1Element


template <ELEMENT_TYPE ET>
class NoCoH1Shape : public NoCoH1Element<ET, NoCoH1Shape<ET>>
{
  static constexpr int DIM = ngfem::Dim(ET);

public:
  template<typename Tx, typename TFA>
  INLINE void T_CalcShape (TIP<DIM,Tx> ip, TFA & shape) const;

  void CalcDualShape2 (const BaseMappedIntegrationPoint & mip, SliceVector<> shape) const
  { throw Exception ("dual shape not implemented, NC H1Ho"); }
}; // class NoCoH1Shape


template<> template<typename Tx, typename TFA>
INLINE void NoCoH1Shape<ET_TRIG> :: T_CalcShape (TIP<DIM,Tx> ip, TFA & shape) const
{
  /**
    static double trig_points [][3] =
      { { 1, 0 },
    { 0, 1 },
    { 0, 0 } };
    static const int trig_edges[3][2] =
{ { 2, 0 },
  { 1, 2 },
  { 0, 1 }};
    **/
  Tx lam[3] = { ip.x, ip.y, 1-ip.x-ip.y };
  shape[0] = lam[2] + lam[0] - lam[1]; // (0,0) -> (1,0), 1-2y
  shape[1] = lam[1] + lam[2] - lam[0]; // (0,1) -> (0,0), 1-2x
  shape[2] = lam[0] + lam[1] - lam[2]; // (1,0) -> (0,0), 2x+2y-1
} // NoCoH1Shape<ET_TRIG>::T_CalcShape


template<> template<typename Tx, typename TFA>
INLINE void NoCoH1Shape<ET_TET> :: T_CalcShape (TIP<DIM,Tx> ip, TFA & shape) const
{
  /** TET
    static double tet_points [][3] =
      { { 1, 0, 0 },
  { 0, 1, 0 },
  { 0, 0, 1 },
  { 0, 0, 0 } };
    static int tet_faces[4][4] =
{ { 3, 1, 2, -1 },
  { 3, 2, 0, -1 },
  { 3, 0, 1, -1 },
  { 0, 2, 1, -1 } }; // all faces point into interior!

    **/
  Tx lam[4] = { ip.x, ip.y, ip.z, 1-ip.x-ip.y-ip.z };
  shape[0] = lam[1] + lam[2] + lam[3] - 2*lam[0];
  shape[1] = lam[0] + lam[2] + lam[3] - 2*lam[1];
  shape[2] = lam[0] + lam[1] + lam[3] - 2*lam[2];
  shape[3] = lam[0] + lam[1] + lam[2] - 2*lam[3];
} // NoCoH1Shape<ET_TRIG>::T_CalcShape


template <ELEMENT_TYPE ET> class NoCoH1TraceShape;

template <ELEMENT_TYPE ET,
          class SHAPES = NoCoH1TraceShape<ET>,
          class BASE = T_ScalarFiniteElement< SHAPES, ET> >
class NoCoH1TraceElement : public BASE, public ET_trait<ET>
{
  enum { DIM = ET_trait<ET>::DIM };
  using FiniteElement::ndof;
  using FiniteElement::order;
public:
  INLINE NoCoH1TraceElement ()
  {
    order = 1;
    ndof = 1;
  }
  INLINE ~NoCoH1TraceElement () { ; }
}; // class NoCoH1TraceElement


template <ELEMENT_TYPE ET>
class NoCoH1TraceShape : public NoCoH1Element<ET, NoCoH1TraceShape<ET>>
{
  static constexpr int DIM = ngfem::Dim(ET);
public:
  template<typename Tx, typename TFA>
  INLINE void T_CalcShape (TIP<DIM,Tx> ip, TFA & shape) const
  { shape[0] = Tx(1.0); }
  void CalcDualShape2 (const BaseMappedIntegrationPoint & mip, SliceVector<> shape) const
  { throw Exception ("dual shape not implemented, NC H1Ho"); }
}; // class NoCoH1TraceShape


class NoCoH1FESpace : public FESpace
{
protected:

  // TODO: get rid of this bookkeeping, or use stokes_aux_info in some way

  /**
   *  Note: The free_dofs of this space are not the same as the free_facets of auxInfo!
   *        free_dofs: 
   *           - # DOF-sized, i.e. #R-facet sized
   *           - says whether the DOF for that facet is free
   *           - is nullptr if there are no Dirichlet conditions
   *        auxInfo.free_facets:
   *           - # A-facets sized
   *           - says whether MESH-FACET is R AND FREE
   *           - is nullptr if there are no Dirichlet conditions AND all facets are R
   */

  unique_ptr<FacetAuxiliaryInformation> auxInfo;

  // // TODO: should be able to remove these!
  // size_t nve_defon;                       /** # of els the space is defined on **/
  // Array<size_t> e2de, de2e;               /** elements <-> defon elements **/

  // Array<int> a2f_facet, f2a_facet;        /** all facets <-> fine active facets **/
  // shared_ptr<BitArray> fine_facet;        /** is facet an "active" facet ? [[ definedon and refined can mess with this ]]**/

  // size_t npsn_defon;                      /** # of active potential space nodes (2d:verts, 3d:edges) **/
  // Array<size_t> psn2dpsn, dpsn2psn;       /** psnodes <-> active psnodes **/

public:

  NoCoH1FESpace (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags = false);

  ~NoCoH1FESpace () { ; }

  virtual string GetClassName () const override
  { return "NoCoH1FESpace"; }

  virtual void Update () override;

  virtual void UpdateDofTables () override;

  virtual void UpdateCouplingDofArray () override;

  virtual FiniteElement & GetFE (ElementId ei, Allocator & alloc) const override;

  // TODO: These should be implemented to work with the aux-info instead
  virtual void GetDofNrs (ElementId ei, Array<DofId> & dnums) const override;

  virtual void GetDofNrs (NodeId ni, Array<DofId> & dnums) const override;

  virtual void GetFaceDofNrs (int fanr, Array<DofId> & dnums) const override;
  virtual void GetEdgeDofNrs (int ednr, Array<DofId> & dnums) const override;

  // // TODO: I should be able to remove all these
  // shared_ptr<BitArray> GetFineFacets () const { return fine_facet; }
  // FlatArray<int> GetFMapA2F () const { return a2f_facet; }
  // FlatArray<int> GetFMapF2A () const { return f2a_facet; }

  // size_t GetNVEDO () const { return nve_defon; }
  // size_t E2DE (size_t elnr) const { return e2de.Size() ? e2de[elnr] : elnr; }
  // size_t DE2E (size_t delnr) const { return de2e.Size() ? de2e[delnr] : delnr; }

  // size_t GetNPSNDO () const { return npsn_defon; }
  // size_t PSN2DPSN (size_t psnr) const { return psn2dpsn.Size() ? psn2dpsn[psnr] : psnr; }
  // size_t DPSN2PSN (size_t dpsnr) const { return dpsn2psn.Size() ? dpsn2psn[dpsnr] : dpsnr; }

  INLINE FacetAuxiliaryInformation const & GetFacetAuxInfo() const { return *auxInfo; }
  INLINE FacetAuxiliaryInformation       & GetFacetAuxInfo()       { return *auxInfo; }

}; // class NoCoH1FESpace


} // namespace amg

#endif // FILE_NC2D_HPP
