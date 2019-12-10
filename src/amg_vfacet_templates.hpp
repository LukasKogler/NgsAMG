#ifndef FILE_AMG_VFACET_TEMPLATES_HPP
#define FILE_AMG_VFACET_TEMPLATES_HPP

namespace amg
{

  template<ELEMENT_TYPE ET> struct STRUCT_SPACE_EL<VectorFacetFESpace,ET> { typedef VectorFacetVolumeFE<ET> fe_type; };
  template struct STRUCT_SPACE_EL<VectorFacetFESpace,ET_TRIG>;
  template struct STRUCT_SPACE_EL<VectorFacetFESpace,ET_QUAD>;
  template struct STRUCT_SPACE_EL<VectorFacetFESpace,ET_TET>;
  template struct STRUCT_SPACE_EL<VectorFacetFESpace,ET_HEX>;

  template<> struct SPACE_DS_TRAIT<VectorFacetFESpace> : std::false_type
  {
    static constexpr bool take_tang   = true;
    static constexpr bool take_normal = false;
  };

} // namespace amg

#endif
