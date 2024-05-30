#include "ncfespace.hpp"

#include <tscalarfe_impl.hpp>

namespace amg
{
template class NoCoH1Element<ET_TRIG>;

/** NoCoH1FESpace **/


NoCoH1FESpace :: NoCoH1FESpace (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags)
  : FESpace(ama, flags)
{
  switch (ma->GetDimension()) {
    case 1: {
        break;
      }
    case 2: {
        evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<2>>>();
        evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpId<2>>>();
        flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpGradient<2>>>();
        break;
      }
    case 3: {
        evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<3>>>();
        evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpId<3>>>();
              flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpGradient<3>>>();
        break;
      }
    }
  if (dimension > 1) {
    additional_evaluators.Set ("Grad", make_shared<BlockDifferentialOperatorTrans>(flux_evaluator[VOL], dimension));
    for (auto vb : { VOL,BND, BBND, BBBND }) {
      if (evaluator[vb])
        { evaluator[vb] = make_shared<BlockDifferentialOperator> (evaluator[vb], dimension); }
      if (flux_evaluator[vb])
        { flux_evaluator[vb] = make_shared<BlockDifferentialOperator> (evaluator[vb], dimension); }
    }
  }
} // NoCoH1FESpace(..)


void NoCoH1FESpace :: Update ()
{
  FESpace::Update();

  auxInfo = make_unique<FacetAuxiliaryInformation>(static_pointer_cast<FESpace>(shared_from_this()));

  size_t nfacets = ma->GetNFacets(), nff = 0;
  // for (auto VB : { VOL, BND } ) {

  auto ma = GetMeshAccess();

  // if (FESpace::definedon[VOL].Size()  == 0) { // defined everywhere
  //   nve_defon = ma->GetNE();
  //   npsn_defon = (ma->GetDimension() == 2) ? ma->GetNV() : ma->GetNEdges();
  // }
  // else { // defined on part of mesh
  //   size_t dim = ma->GetDimension();
  //   size_t npsn = (dim == 2) ? ma->GetNV() : ma->GetNEdges();
  //   psn2dpsn.SetSize(npsn); psn2dpsn = 0;
  //   e2de.SetSize(ma->GetNE()); e2de = 0;
  //   Array<int> el_psns;

  //   // MARK all -> defined
  //   for (auto elnr : Range(ma->GetNE())) {
  //     ElementId eid(VOL, elnr);
  //     if (DefinedOn(eid)) {
  //       e2de[elnr] = 1;
  //       if (dim == 2)
  //         { ma->GetElVertices(eid, el_psns); }
  //       else
  //         { ma->GetElEdges(eid, el_psns); }
  //       for (auto j : Range(el_psns))
  //         { psn2dpsn[el_psns[j]] = 1; }
  //     }
  //   } // element loop

  //   // cout << " reduce EL all <-> defined" << endl;
  //   ma->AllReduceNodalData ( (ma->GetDimension() == 2) ? NT_FACE : NT_CELL,
  //           e2de, NG_MPI_LOR);
  //   // cout << " reduce PSN all <-> defined" << endl;
  //   ma->AllReduceNodalData ( (ma->GetDimension() == 2) ? NT_VERTEX : NT_EDGE,
  //           psn2dpsn, NG_MPI_LOR);

  //   // COUNT defined ->  all
  //   nve_defon = 0; npsn_defon = 0;
  //   for (auto j : Range(e2de))
  //     if (e2de[j])
  //       { nve_defon++; }
  //   for (auto j : Range(psn2dpsn))
  //     if (psn2dpsn[j])
  //       { npsn_defon++; }

  //   // SET defined ->  all
  //   dpsn2psn.SetSize(npsn_defon); de2e.SetSize(nve_defon);
  //   npsn_defon = 0; nve_defon = 0;

  //   for (auto j : Range(e2de))
  //     if (e2de[j] == 0)
  //       { e2de[j] = -1; }
  //     else {
  //       de2e[nve_defon] = j;
  //       e2de[j] = nve_defon++;
  //     }
  //   for (auto j : Range(psn2dpsn)) {
  //     if (psn2dpsn[j] == 0)
  //       { psn2dpsn[j] = -1; }
  //     else {
  //       dpsn2psn[npsn_defon] = j;
  //       psn2dpsn[j] = npsn_defon++;
  //     }
  //   }
  // } // defined on part of mesh

  // /**
  //  * Not sure what would happen if i iterated over BND els too.
  //  * (facet of a bnd trig is a trig not an edge, etc)
  //  * Should not matter I think ...
  //  */
  // fine_facet = make_shared<BitArray>(nfacets); fine_facet->Clear();
  // ma->IterateElements(VOL, [&](auto ei) {
  //   if (DefinedOn(ei)) // <- !!
  //     for (auto facet : ma->GetElFacets(ei))
  //       { fine_facet->SetBitAtomic(facet); }
  // });

  // // }
  // a2f_facet.SetSize(nfacets);

  // for (auto k : Range(nfacets))
  //   { a2f_facet[k] = fine_facet->Test(k) ? 1.0 : 0.0; }

  // ma->AllReduceNodalData ((ma->GetDimension()==2) ? NT_EDGE : NT_FACE,
  //       a2f_facet, NG_MPI_LOR);

  // nff = 0;

  // for (auto k : Range(nfacets)) {
  //   bool ff = a2f_facet[k] != 0.0;
  //   a2f_facet[k] = ff ? nff++ : -1;
  //   if (ff)
  //     { fine_facet->SetBit(k); }
  // }

  // f2a_facet.SetSize(nff); nff = 0;

  // for (auto k : Range(nfacets))
  //   if (a2f_facet[k] != -1)
  //     { f2a_facet[nff++] = k; }

  // // cout << " NF NFF " << nfacets << " " << nff << endl;
  // // cout << endl << " fine -> all facets " << f2a_facet.Size() << endl; prow2(f2a_facet); cout << endl;
  // // cout << endl << " all -> fine facets " << a2f_facet.Size() << endl; prow2(a2f_facet); cout << endl;
  // // cout << " update done " << endl;

  UpdateDofTables ();
  UpdateCouplingDofArray ();
} // NoCoH1FESpace::Update


void NoCoH1FESpace :: UpdateDofTables ()
{
  // SetNDof(f2a_facet.Size());
  SetNDof(GetFacetAuxInfo().GetNFacets_R());
} // NoCoH1FESpace::Update


void NoCoH1FESpace :: UpdateCouplingDofArray ()
{
  ctofdof.SetSize(GetNDof());
  ctofdof = WIREBASKET_DOF;
} // NoCoH1FESpace::Update


FiniteElement & NoCoH1FESpace :: GetFE (ElementId ei, Allocator & alloc) const
{
  Ngs_Element ngel = ma->GetElement(ei);
  ELEMENT_TYPE eltype = ngel.GetType();

  switch(ei.VB()) {
    case(VOL) : {
      if (eltype == ET_TRIG) {
        auto el = new (alloc) NoCoH1Element<ET_TRIG>;
        return *el;
      }
      else if (eltype == ET_TET) {
        auto el = new (alloc) NoCoH1Element<ET_TET>;
        return *el;
      }
      break;
    }
    case(BND) : {
      if (eltype == ET_SEGM) {
        auto el = new (alloc) NoCoH1TraceElement<ET_SEGM>;
        return *el;
      }
      else if (eltype == ET_TRIG) {
        auto el = new (alloc) NoCoH1TraceElement<ET_TRIG>;
        return *el;
      }
      break;
    }
    default : {
      break;
    }
  }

  return SwitchET (eltype, [&] (auto et) -> FiniteElement& {
    return *new (alloc) ScalarDummyFE<et.ElementType()> ();
  });
} // NoCoH1FESpace::GetFE


void NoCoH1FESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
{
  // if (!DefinedOn (ei))
    // { dnums.SetSize0(); return; }
  auto ma = GetMeshAccess();
  auto el_facets = ma->GetElFacets(ei);

  int cnt = 0;

  // cout << " NC::GetDofNrs for " << ei << endl;

  dnums.SetSize(el_facets.Size());

  // cout << " el_facets: "; prow(el_facets); cout << endl;

  for (auto k : Range(el_facets)) {
    // auto dof = a2f_facet[el_facets[k]];
    // cout << " map " << el_facets[k];
    auto dof = GetFacetAuxInfo().A2R_Facet(el_facets[k]);
    // cout << " -> " << dof << endl;
    if (dof != -1)
      { dnums[cnt++] = dof; }
  }
  // for (auto k : Range(el_facets)) {
    // { dnums[k] = a2f_facet[el_facets[k]]; }
  dnums.SetSize(cnt);
} // NoCoH1FESpace::GetDofNrs


void NoCoH1FESpace :: GetDofNrs (NodeId ni, Array<DofId> & dnums) const
{
  auto ma = GetMeshAccess();
  if ( ( ni.GetType() == NT_FACET )                               ||
       ( (ma->GetDimension() == 2) && (ni.GetType() == NT_EDGE) ) ||
       ( (ma->GetDimension() == 3) && (ni.GetType() == NT_FACE) ) )
  {
    dnums.SetSize(1);
    // cout << " NC Node-DNrs " << ni << endl;
    // cout << " map " << ni.GetNr();
    // dnums[0] = a2f_facet[ni.GetNr()]; // allowed to be -1 !
    dnums[0] = GetFacetAuxInfo().A2R_Facet(ni.GetNr());
    // cout << " -> " << dnums[0] << endl;;
  }
  else {
    dnums.SetSize0();
  }
} // NoCoH1FESpace::GetDofNrs


void NoCoH1FESpace :: GetFaceDofNrs (int fanr, Array<DofId> & dnums) const
{
  GetDofNrs(NodeId(NT_FACE, fanr), dnums);
} // NoCoH1FESpace::GetFaceDofNrs


void NoCoH1FESpace :: GetEdgeDofNrs (int ednr, Array<DofId> & dnums) const
{
  GetDofNrs(NodeId(NT_EDGE, ednr), dnums);
} // NoCoH1FESpace::GetEdgeDofNrs

/** END NoCoH1FESpace **/

} // namespace amg
