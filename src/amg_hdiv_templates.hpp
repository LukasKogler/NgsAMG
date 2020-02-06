#ifndef FILE_AMG_HDIV_TEMPLATES_HPP
#define FILE_AMG_HDIV_TEMPLATES_HPP

#include <hdivhofe.hpp>

namespace amg
{

  template<ELEMENT_TYPE ET> struct STRUCT_SPACE_EL<HDivHighOrderFESpace,ET> { typedef HDivHighOrderFE<ET> fe_type; };
  template struct STRUCT_SPACE_EL<HDivHighOrderFESpace,ET_TRIG>;
  template struct STRUCT_SPACE_EL<HDivHighOrderFESpace,ET_QUAD>;
  template struct STRUCT_SPACE_EL<HDivHighOrderFESpace,ET_TET>;
  template struct STRUCT_SPACE_EL<HDivHighOrderFESpace,ET_HEX>;

} // namespace amg

#endif
