#ifndef FILE_STOKES_AUX_INFO_HPP
#define FILE_STOKES_AUX_INFO_HPP

#include <base.hpp>

namespace amg
{

/** FacetAuxiliaryInformation **/

/**
 * Auxiliary information needed to set up Stokes preconditioners,
 * takes care of "definedon" spaces and "inactive" nodes due to mesh refinement.
 *  [[ "active" or "relevant" := definedon + fine ]]
 *     i) elements  <-> active elements
 *    ii) facets    <-> active facets mapping
 *   iii) pot-nodes <-> active pot-nodes
*/
class FacetAuxiliaryInformation
{
public:
  FacetAuxiliaryInformation(shared_ptr<FESpace> afes);

  FacetAuxiliaryInformation(FacetAuxiliaryInformation const &other) = default;

  FacetAuxiliaryInformation() = default; // dummy constructor
 
  ~FacetAuxiliaryInformation() = default;

protected:

  void setup();

  shared_ptr<FESpace> fes;

  /**
   *  Note: Element and Potential-space maps only are needed when we have some definedon,
   *        but facet-maps are also needed when we have refinement, so "have_maps" only
   *        concerns the former two!
   *        We ALWAYS set up facet-maps (TODO: when all facets are R, do not use facet-maps?!)
  */

  bool have_maps;                        /** whether I have maps (non-trivial) or none (trivial) **/
  size_t n_el_all, n_el_rel;             /** # of els the space is defined on **/
  Array<size_t> a2r_el, r2a_el;          /** elements <-> relevant elements **/

  size_t n_psn_all, n_psn_rel;           /** # of active potential space nodes (2d:verts, 3d:edges) **/
  Array<size_t> a2r_psn, r2a_psn;        /** psnodes <-> relevant psnodes **/

  size_t n_facet_all, n_facet_rel;
  Array<int> a2r_facet, r2a_facet;       /** all facets <-> relevant facets **/
  shared_ptr<BitArray> rel_facets;       /** is facet an "active" facet ? [[ definedon and refined can mess with this ]]**/
  // shared_ptr<BitArray> free_facets;      /** is facet "active" AND FREE? */

public:

  shared_ptr<const FESpace> GetFESpace () const { return fes; }

  INLINE size_t GetNE_A () const { return n_el_all; }
  INLINE size_t GetNE_R () const { return n_el_rel; }
  INLINE bool   IsElementRel (size_t k) const { return have_maps ? (a2r_el[k] != -1) : true; }
  INLINE size_t A2R_EL (size_t k) const { return have_maps ? a2r_el[k] : k; }
  INLINE size_t R2A_EL (size_t k) const { return have_maps ? r2a_el[k] : k; }
  INLINE FlatArray<size_t> GetElMapA2R () const { return a2r_el; }
  INLINE FlatArray<size_t> GetElMapR2A () const { return r2a_el; }

  INLINE size_t GetNPSN_A() const { return n_psn_all; }
  INLINE size_t GetNPSN_R() const { return n_psn_rel; }
  INLINE bool   IsPSNRel (size_t k) const { return have_maps ? (a2r_psn[k] != -1) : true; }
  INLINE size_t A2R_PSN (size_t k) const { return have_maps ? a2r_psn[k] : k; }
  INLINE size_t R2A_PSN (size_t k) const { return have_maps ? r2a_psn[k] : k; }
  INLINE FlatArray<size_t> GetPSNMapA2R () const { return a2r_psn; }
  INLINE FlatArray<size_t> GetPSNMapR2A () const { return r2a_psn; }

  shared_ptr<BitArray> GetActiveFacets () const { return rel_facets; }
  // shared_ptr<BitArray> GetFreeFacets () const { return free_facets; }
  INLINE size_t GetNFacets_A() const { return n_facet_all; }
  INLINE size_t GetNFacets_R() const { return n_facet_rel; }
  INLINE bool   IsFacetRel (int k) const { return rel_facets->Test(k); }
  // INLINE bool   IsFacetFree (int k) const { return free_facets->Test(k); }
  INLINE int A2R_Facet (int k) const { return a2r_facet[k]; }
  INLINE int R2A_Facet (int k) const { return r2a_facet[k]; }
  INLINE FlatArray<int> GetFacetMapA2R () const { return a2r_facet; }
  INLINE FlatArray<int> GetFacetMapR2A () const { return r2a_facet; }

  void printTo (ostream &ost, std::string const &prefix = "") const;

}; // class FacetAuxiliaryInformation


INLINE std::ostream & operator<< (std::ostream &os, FacetAuxiliaryInformation const &faux)
{
  faux.printTo(os);
  return os;
}

/** END FacetAuxiliaryInformation **/

} // namespace amg

#endif // FILE_STOKES_AUX_INFO_HPP