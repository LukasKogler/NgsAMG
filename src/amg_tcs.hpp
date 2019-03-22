
/** 
    This file is included twice - once with AMG_EXTERN_TEMPLATES, and once without:
       The first include happens from "amg.hpp", to get "extern template class ..."
       The second includ happens at the end of the ".cpp" files to get "template class ..."
**/
#if (defined(AMG_EXTERN_TEMPLATES) && !defined(FILE_AMGTCS_ONE)) || (!defined(AMG_EXTERN_TEMPLATES) && !defined(FILE_AMGTCS_TWO))

#ifdef AMG_EXTERN_TEMPLATES
#define FILE_AMGTCS_ONE
#define EXTERN extern
#else
#define FILE_AMGTCS_TWO
#define EXTERN
#endif

namespace amg
{

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM_CPP)
  EXTERN template class HybridGSS<1>;
  EXTERN template class HybridGSS<2>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCRS_CPP)
  EXTERN template class SeqVWC<FlatTM>;
  EXTERN template class BlockVWC<H1Mesh>;
  EXTERN template class HierarchicVWC<H1Mesh>;
  EXTERN template class CoarseMap<H1Mesh>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCTR_CPP)
  EXTERN template class CtrMap<double>;
  EXTERN template class GridContractMap<H1Mesh>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGH1_CPP)
  EXTERN template class EmbedVAMG<H1AMG>;
#endif

} // namespace amg

#undef EXTERN

#endif
