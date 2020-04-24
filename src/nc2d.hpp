#ifndef FILE_NC2D_HPP
#define FILE_NC2D_HPP

namespace amg
{

  template <ELEMENT_TYPE ET> class NoCoH1Shape;

  template <ELEMENT_TYPE ET, 
            class SHAPES = NoCoH1Shape<ET>,
            class BASE = T_ScalarFiniteElement< SHAPES, ET> >
  class NoCoH1Element : public BASE, public ET_trait<ET>
  {
    enum { DIM = ET_trait<ET>::DIM };

    using ScalarFiniteElement<DIM>::ndof;
    using ScalarFiniteElement<DIM>::order;

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
    Tx lam[3] = { ip.x, ip.y, 1-ip.x-ip.y };
    shape[0] = lam[2] + lam[0] - lam[1];
    shape[1] = lam[1] + lam[2] - lam[0];
    shape[2] = lam[0] + lam[1] - lam[2];
  } // NoCoH1Shape<ET_TRIG>::T_CalcShape
  
  
  class NoCoH1FESpace : public FESpace
  {
  protected:

    Array<int> a2f_facet, f2a_facet;        /** all facets <-> fine facets mappings **/
    shared_ptr<BitArray> fine_facet;        /** is facet an "active" facet ? [[ definedon and refined can mess with this ]]**/

  public:

    NoCoH1FESpace (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags = false);

    ~NoCoH1FESpace () { ; }

    virtual string GetClassName () const override
    { return "NoCoH1FESpace"; }

    virtual void Update () override;

    virtual void UpdateDofTables () override;

    virtual void UpdateCouplingDofArray () override;    

    virtual FiniteElement & GetFE (ElementId ei, Allocator & alloc) const override;

    virtual void GetDofNrs (ElementId ei, Array<DofId> & dnums) const override;

    virtual void GetDofNrs (NodeId ni, Array<DofId> & dnums) const;

  }; // class NoCoH1FESpace

} // namespace amg

#endif // FILE_NC2D_HPP
