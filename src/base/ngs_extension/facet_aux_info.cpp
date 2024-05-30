#include "facet_aux_info.hpp"

namespace amg
{

template<class T>
void countSubsetMarks(FlatArray<T> all2Sub, size_t &nSub)
{
  nSub = 0;

  for (auto j : Range(all2Sub)) {
    if (all2Sub[j])
      { nSub++; }
  }
}

template<class T>
void markToSubsetMappings(FlatArray<T> all2Sub, size_t &nSub, Array<T> &subToAll)
{
  countSubsetMarks(all2Sub, nSub);

  subToAll.SetSize(nSub);
  nSub = 0;

  for (auto j : Range(all2Sub)) {
    bool fine = (all2Sub[j] != 0);
    if (fine) {
      all2Sub[j]       = nSub;
      subToAll[nSub++] = j;
    }
    else
     { all2Sub[j] = -1; }    
  }
} // subsetToAllMapping


/** FacetAuxiliaryInformation **/

FacetAuxiliaryInformation :: FacetAuxiliaryInformation(shared_ptr<FESpace> afes)
  : fes(afes)
{
  setup();
} // FacetAuxiliaryInformation(..)

void FacetAuxiliaryInformation :: setup()
{
  const auto &myFes = *fes;
  const auto &ma    = *myFes.GetMeshAccess();

  have_maps = myFes.DefinedOn(VOL).Size() != 0;

  /**  all <-> relevant maps for elements and pot-space nodes  **/

  if (!have_maps) { // defined everywhere -> trivial case
    n_el_all  = n_el_rel  = ma.GetNE();
    n_psn_all = n_psn_rel = (ma.GetDimension() == 2) ? ma.GetNV() : ma.GetNEdges();
  }
  else { // defined on part of mesh
    auto dim = ma.GetDimension();

    n_el_all  = ma.GetNE();
    n_psn_all = (ma.GetDimension() == 2) ? ma.GetNV() : ma.GetNEdges();

    size_t npsn = (dim == 2) ? ma.GetNV() : ma.GetNEdges();

    a2r_el.SetSize(ma.GetNE());
    a2r_psn.SetSize(npsn);
    a2r_el = 0;
    a2r_psn = 0;
    
    // mark relevant
    Array<int> el_psns;
    for (auto elnr : Range(ma.GetNE())) {
      ElementId eid(VOL, elnr);
      if (myFes.DefinedOn(eid)) {
        a2r_el[elnr] = 1;
        // non-deprecated calls don't work because they return different
        // things for vertices and edges (edges have a number + a ptr to vnums)
        // auto el_psns = (dim == 2) ? ma.GetElVertices(eid) : ma.GetElEdges(eid);
        // deprecated calls:
        if (dim == 2)
          { ma.GetElVertices(eid, el_psns); }
        else
          { ma.GetElEdges(eid, el_psns); }
        for (auto j : Range(el_psns))
          { a2r_psn[el_psns[j]] = 1; }
      }
    } // element loop

    // reduce relevant
    ma.AllReduceNodalData((ma.GetDimension() == 2) ? NT_FACE : NT_CELL,
                          a2r_el,
                          NG_MPI_LOR);
    ma.AllReduceNodalData((ma.GetDimension() == 2) ? NT_VERTEX : NT_EDGE,
                          a2r_psn,
                          NG_MPI_LOR);

    // mark -> [ all <-> relevant maps]
    markToSubsetMappings(a2r_el,  n_el_rel,  r2a_el );
    markToSubsetMappings(a2r_psn, n_psn_rel, r2a_psn);
  } // defined on part of mesh


  /**  all <-> relevant maps for facets  (NOTE: not sure why I am doing this separately from the above) **/

  n_facet_all = ma.GetNFacets();

  /**
   * mark relevant
   * 
   * NOTE: Not sure what would happen if i iterated over BND els too.
   *       (facet of a bnd trig is a trig not an edge, etc)
   *       Should not matter I think ...
   */
  rel_facets = make_shared<BitArray>(n_facet_all);
  rel_facets->Clear();
  ma.IterateElements(VOL, [&](auto ei) {
    if (myFes.DefinedOn(ei)) // <- !!
      for (auto facet : ma.GetElFacets(ei))
        { rel_facets->SetBitAtomic(facet); }
    });

  // reduce relevant
  a2r_facet.SetSize(n_facet_all);
  for (auto k : Range(n_facet_all))
    { a2r_facet[k] = rel_facets->Test(k) ? 1.0 : 0.0; }

  ma.AllReduceNodalData((ma.GetDimension() == 2) ? NT_EDGE : NT_FACE,
                        a2r_facet,
                        NG_MPI_LOR);

  // mark -> [ all <-> relevant maps ]
  markToSubsetMappings(a2r_facet, n_facet_rel, r2a_facet);

  // update BitArray w. reduced data
  for (auto k : Range(n_facet_rel))
    { rel_facets->SetBit(k); }


  /** free facets MOVED OUT! **/
  // Array<int> facet_dofs(50);
  // auto fes_free = fes->GetFreeDofs();

  // if ( ( fes_free != nullptr ) ||        // Dirichlet conditions
  //      ( n_facet_all != n_facet_rel ) )  // not defined on all facets (refinement/definedon)
  // {
  //   // Note: elint does not matter here since we only check for DOFs on facets
  //   free_facets = make_shared<BitArray>(*rel_facets);

  //   for (auto k : Range(ma.GetNFacets())) {
  //     if (free_facets->Test(k)) {
  //       fes->GetDofNrs(NodeId(NT_FACET, k), facet_dofs);
  //       if ( facet_dofs.Size()    &&
  //           (!fes_free->Test(facet_dofs[0])) )
  //         { free_facets->Clear(k); }
  //     }
  //   }
  // }
  // else {
  //   free_facets = nullptr;
  // }

  //   cout << " FacetAuxiliaryInformation::setup DONE: " << std::endl;
  //   printTo(cout, "   ");

} // FacetAuxiliaryInformation::setup


// TODO: next compile: move to central header
template<class T>
INLINE std::string my_typename(T const *ptr)
{
  if (ptr == nullptr)
  {
    return std::string(" IS NULLPTR!");
  }
  else
  {
    return typeid(*ptr).name();
  }
}

template<class T>
INLINE std::string my_typename(T *ptr)
{
  if (ptr == nullptr)
  {
    return std::string(" IS NULLPTR!");
  }
  else
  {
    return typeid(*ptr).name();
  }
}

template<class T>
INLINE std::string my_typename(shared_ptr<T> const &ptr)
{
  return my_typename(ptr.get());
}

template<class T>
INLINE void prow3 (const T & ar, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  for (auto k : Range(ar.Size())) {
    if (k > 0 && (k%per_row == 0))
    {
      os << endl << off;
    }
    os << "(" << k << "::" << ar[k] << ") ";
  }
}

INLINE void prowBA (const BitArray & ba, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  for (auto k : Range(ba.Size())) {
    if ( (k > 0) &&
         ( ( k % per_row ) == 0 ) )
    {
      os << endl << off;
    }
    os << "(" << k << "=" << int( ba.Test(k) ? 1 : 0 ) << ") ";
  }
}

INLINE void prowBA (const BitArray *ba, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  if (ba == nullptr)
  {
    os << off << " BitArray is nullptr !";
  }
  else
  {
    prow3(*ba, os, off, per_row);
  }
}


void FacetAuxiliaryInformation :: printTo (ostream &ost, std::string const &prefix) const
{
  std::string offset2 = prefix + "  ";
  std::string offset3 = offset2 + "  ";
  ost << prefix  << "FacetAuxiliaryInformation @ " << this << endl;
  ost << offset2 << "FESpace = " << fes << ", type = " << my_typename(fes) << endl;
  ost << offset2 << "#Elements:  all = " << n_el_all << ", rel = " << n_el_rel << endl;
  ost << offset2 << "#Pot-Space: all = " << n_psn_all << ", rel = " << n_psn_rel << endl;
  ost << offset2 << "#Facets:    all = " << n_facet_all << ", rel = " << n_facet_rel << endl;
  ost << offset2 << " ---------------- " << endl;
  ost << offset2 << "Maps:" << endl;
  ost << offset2 << "  have_maps = " << have_maps << endl;
  if (have_maps)
  {
    ost << offset3 << "Elements A -> R: " << endl;
    ost << offset3 << "  "; prow3(a2r_el, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
    ost << offset3 << "Elements R -> A: " << endl;
    ost << offset3 << "  "; prow3(r2a_el, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
  }
  else
  {
    ost << offset3 << " No Element Maps!" << endl;
  }
  if (have_maps)
  {
    ost << offset3 << "Pot-Space A -> R: " << endl;
    ost << offset3 << "  "; prow3(a2r_psn, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
    ost << offset3 << "Pot-Space R -> A: " << endl;
    ost << offset3 << "  "; prow3(r2a_psn, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
  }
  else
  {
    ost << offset3 << " No Pot-Space Maps!" << endl;
  }
  if (a2r_facet.Size() != 0)
  {
    ost << offset3 << "Facets A -> R: " << endl;
    ost << offset3 << "  "; prow3(a2r_facet, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
    ost << offset3 << "Facets R -> A: " << endl;
    ost << offset3 << "  "; prow3(r2a_facet, ost, offset3, 20); ost << endl << offset3 << " -------------- " << endl;
  }
  else
  {
    ost << offset3 << " F map sizes " << r2a_facet.Size() << " " << a2r_facet.Size() << endl;
    ost << offset3 << " No Facet Maps!" << endl;
  }
  // ost << offset3 << "Free Facets: " << endl; prowBA(free_facets.get(), ost, offset3); ost << endl;
} // FacetAuxiliaryInformation::printTo

/** END FacetAuxiliaryInformation **/

} // namespace amg