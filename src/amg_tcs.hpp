
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
  EXTERN template class HybridGSS<3>;
  EXTERN template class HybridGSS<4>;
  EXTERN template class HybridGSS<5>;
  EXTERN template class HybridGSS<6>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCRS_CPP)
  EXTERN template class SeqVWC<FlatTM>;
  EXTERN template class BlockVWC<H1Mesh>;
  EXTERN template class HierarchicVWC<H1Mesh>;
  EXTERN template class CoarseMap<H1Mesh>;
  EXTERN template class BlockVWC<ElasticityMesh<2>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<2>>;
  EXTERN template class CoarseMap<ElasticityMesh<2>>;
  EXTERN template class BlockVWC<ElasticityMesh<3>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<3>>;
  EXTERN template class CoarseMap<ElasticityMesh<3>>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCTR_CPP)
  EXTERN template class CtrMap<double>;
  EXTERN template class CtrMap<Vec<2,double>>;
  EXTERN template class CtrMap<Vec<3,double>>;
  EXTERN template class CtrMap<Vec<4,double>>;
  EXTERN template class CtrMap<Vec<5,double>>;
  EXTERN template class CtrMap<Vec<6,double>>;
  EXTERN template class GridContractMap<H1Mesh>;
  EXTERN template class GridContractMap<ElasticityMesh<2>>;
  EXTERN template class GridContractMap<ElasticityMesh<3>>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGH1_CPP)
  EXTERN template class EmbedVAMG<H1AMG>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGELAST_CPP)
#ifndef FILE_AMGELAST_CPP
  EXTERN template class ElasticityAMG<2>;
  EXTERN template class ElasticityAMG<3>;
#endif
  EXTERN template class EmbedVAMG<ElasticityAMG<2>>;
  EXTERN template class EmbedVAMG<ElasticityAMG<3>>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_SPMSTUFF_CPP)
  /** Transpose **/
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<double>>::type> TransposeSPM (const SparseMatrix<double> & mat);
  // EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<2,1,double>, double, Vec<2,double>>>::type> TransposeSPM
  // (const SparseMatrix<Mat<2,1,double>, double, Vec<2,double>> & mat); // TODO: NGSolve cant hande this for now ...
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<2,2,double>, Vec<2,double>, Vec<2,double>>>::type> TransposeSPM (const SparseMatrix<Mat<2,2,double>, Vec<2,double>, Vec<2,double>> & mat);
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>>>::type> TransposeSPM (const SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>> & mat);
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<4,4,double>, Vec<4,double>, Vec<4,double>>>::type> TransposeSPM (const SparseMatrix<Mat<4,4,double>, Vec<4,double>, Vec<4,double>> & mat);
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<5,5,double>, Vec<5,double>, Vec<5,double>>>::type> TransposeSPM (const SparseMatrix<Mat<5,5,double>, Vec<5,double>, Vec<5,double>> & mat);
  EXTERN template shared_ptr<typename trans_spm<SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>>>::type> TransposeSPM (const SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>> & mat);


  /** A * B **/
  EXTERN template shared_ptr<SparseMatrix<double>> MatMultAB (const SparseMatrix<double> & mata, const SparseMatrix<double> & matb);
  EXTERN template shared_ptr<SparseMatrix<Mat<2,2,double>, Vec<2,double>, Vec<2,double>>> MatMultAB (const SparseMatrix<Mat<2,2,double>, Vec<2,double>, Vec<2,double>> & mata, const SparseMatrix<Mat<2,2,double>, Vec<2,double>, Vec<2,double>> & matb);
  EXTERN template shared_ptr<SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>>> MatMultAB (const SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>> & mata, const SparseMatrix<Mat<3,3,double>, Vec<3,double>, Vec<3,double>> & matb);
  EXTERN template shared_ptr<SparseMatrix<Mat<4,4,double>, Vec<4,double>, Vec<4,double>>> MatMultAB (const SparseMatrix<Mat<4,4,double>, Vec<4,double>, Vec<4,double>> & mata, const SparseMatrix<Mat<4,4,double>, Vec<4,double>, Vec<4,double>> & matb);
  EXTERN template shared_ptr<SparseMatrix<Mat<5,5,double>, Vec<5,double>, Vec<5,double>>> MatMultAB (const SparseMatrix<Mat<5,5,double>, Vec<5,double>, Vec<5,double>> & mata, const SparseMatrix<Mat<5,5,double>, Vec<5,double>, Vec<5,double>> & matb);
  EXTERN template shared_ptr<SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>>> MatMultAB (const SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>> & mata, const SparseMatrix<Mat<6,6,double>, Vec<6,double>, Vec<6,double>> & matb);
    
#endif
  
} // namespace amg

#undef EXTERN

#endif
