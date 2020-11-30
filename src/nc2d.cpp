#include "amg.hpp"
#include "nc2d.hpp"

#include <tscalarfe_impl.hpp>

namespace amg
{
  template class NoCoH1Element<ET_TRIG>;

  /** NoCoH1FESpace **/


  NoCoH1FESpace :: NoCoH1FESpace (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags)
    : FESpace(ama, flags)
  {
    switch (ma->GetDimension())
      {
      case 1:
	{
	  break;
	}
      case 2:
	{
	  evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<2>>>();
	  break;
	}
      case 3:
	{
	  evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<3>>>();
	  break;
	}
      }
    if (dimension > 1)
        for (auto vb : { VOL,BND, BBND, BBBND })
          {
            if (evaluator[vb])
              evaluator[vb] = make_shared<BlockDifferentialOperator> (evaluator[vb], dimension);
	  }
  } // NoCoH1FESpace(..)


  void NoCoH1FESpace :: Update ()
  {
    FESpace::Update();

    // TODO: would this be cleaner with definedon?
    size_t nfacets = ma->GetNFacets(), nff = 0;
    fine_facet = make_shared<BitArray>(nfacets);
    a2f_facet.SetSize(nfacets); a2f_facet = -1;
    Array<int> elnums;
    for (auto k : Range(nfacets)) {
      ma->GetFacetElements(k, elnums);
      if ( !elnums.Size() )
	{ ma->GetFacetSurfaceElements(k, elnums); }
      if ( elnums.Size() )
	{ fine_facet->SetBit(k); a2f_facet[k] = nff++; }
    }
    f2a_facet.SetSize(nff);
    nff = 0;
    for (auto k : Range(nfacets))
      if (a2f_facet[k] != -1)
	{ f2a_facet[nff++] = k; }
    // cout << " fine facets " << nff << endl; prow2(f2a_facet); cout << endl;
    // cout << " update done " << endl;

    UpdateDofTables ();
    UpdateCouplingDofArray ();
  } // NoCoH1FESpace::Update


  void NoCoH1FESpace :: UpdateDofTables ()
  {
    // cout << " UDT " << endl;
    SetNDof(f2a_facet.Size());
    // cout << " UDT done! " << endl;
  } // NoCoH1FESpace::Update


  void NoCoH1FESpace :: UpdateCouplingDofArray ()
  {
    // cout << " cda " << endl;
    ctofdof.SetSize(GetNDof());
    ctofdof = WIREBASKET_DOF;
    // cout << " cda done " << endl;
  } // NoCoH1FESpace::Update


  FiniteElement & NoCoH1FESpace :: GetFE (ElementId ei, Allocator & alloc) const
  {
    // cout << " get fel, ei " << ei << endl;
    Ngs_Element ngel = ma->GetElement(ei);
    ELEMENT_TYPE eltype = ngel.GetType();

    if (eltype == ET_TRIG) {
      auto el = new (alloc) NoCoH1Element<ET_TRIG>;
      return *el;
      }
    else if (eltype == ET_TET) {
      auto el = new (alloc) NoCoH1Element<ET_TET>;
      return *el;
    }

    return SwitchET (eltype, [&] (auto et) -> FiniteElement& {
	return *new (alloc) ScalarDummyFE<et.ElementType()> ();
      });
  } // NoCoH1FESpace::GetFE


  void NoCoH1FESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
  {
    // cout << " el dnrs " << ei << endl;
    auto el_facets = ma->GetElFacets(ei);
    dnums.SetSize(el_facets.Size());
    for (auto k : Range(el_facets))
      { dnums[k] = a2f_facet[el_facets[k]]; }
    // cout << " ndofs " << GetNDof() << endl; prow2(dnums); cout << endl;
    // cout << " el dnrs OK " << ei << endl;
  } // NoCoH1FESpace::GetDofNrs


  void NoCoH1FESpace :: GetDofNrs (NodeId ni, Array<DofId> & dnums) const
  {
    // cout << " n dnrs " << ni << endl;
    if ( ( (ma->GetDimension()==2) && (ni.GetType() == NT_EDGE) ) ||
	 ( (ma->GetDimension()==3) && (ni.GetType() == NT_FACE) ) ) {
      dnums.SetSize(1);
      dnums[0] = f2a_facet[ni.GetNr()];
    }
    else
      { dnums.SetSize0(); }
    // cout << " n dnrs OK " << ni << endl;
  } // NoCoH1FESpace::GetDofNrs


  /** END NoCoH1FESpace **/

} // namespace amg


#include "python_comp.hpp"

namespace amg
{
  void ExportNCSpace (py::module & m)
  {
    ExportFESpace<NoCoH1FESpace> (m, "NoCoH1");
  }
} // namespace amg
