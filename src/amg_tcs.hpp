
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

namespace ngla
{

  /**
     depending on which MAX_SYS_DIM NGSolve has been compiled with, 
     we might not already have SparseMatrix<Mat<K,J>>
   **/
#ifdef ELASTICITY
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_SPMSTUFF_CPP)
#if MAX_SYS_DIM < 3
  EXTERN template class SparseMatrixTM<Mat<3,3,double>>;
  EXTERN template class SparseMatrix<Mat<3,3,double>>;
#endif
  EXTERN template class SparseMatrixTM<Mat<1,3,double>>;
  EXTERN template class SparseMatrix<Mat<1,3,double>>;
  EXTERN template class SparseMatrixTM<Mat<2,3,double>>;
  EXTERN template class SparseMatrix<Mat<2,3,double>>;
#if MAX_SYS_DIM < 6
  EXTERN template class SparseMatrixTM<Mat<6,6,double>>;
  EXTERN template class SparseMatrix<Mat<6,6,double>>;
#endif
  EXTERN template class SparseMatrixTM<Mat<1,6,double>>;
  EXTERN template class SparseMatrix<Mat<1,6,double>>;
  EXTERN template class SparseMatrixTM<Mat<3,6,double>>;
  EXTERN template class SparseMatrix<Mat<3,6,double>>;
#endif
#endif
  
} // namespace ngla


namespace amg
{
  
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM_CPP)
  EXTERN template class HybridGSS<1>;
#ifdef ELASTICITY
  EXTERN template class HybridGSS<2>;
  EXTERN template class HybridGSS<3>;
  EXTERN template class StabHGSS<3,2,3>;
  // EXTERN template class HybridGSS<4>;
  // EXTERN template class HybridGSS<5>;
  EXTERN template class HybridGSS<6>;
  EXTERN template class StabHGSS<6,3,6>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCRS_CPP)
  EXTERN template class SeqVWC<FlatTM>;
  EXTERN template class BlockVWC<H1Mesh>;
  EXTERN template class HierarchicVWC<H1Mesh>;
  EXTERN template class CoarseMap<H1Mesh>;
#ifdef ELASTICITY
  EXTERN template class BlockVWC<ElasticityMesh<2>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<2>>;
  EXTERN template class CoarseMap<ElasticityMesh<2>>;
  EXTERN template class BlockVWC<ElasticityMesh<3>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<3>>;
  EXTERN template class CoarseMap<ElasticityMesh<3>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCTR_CPP)
  EXTERN template class CtrMap<double>;
#ifdef ELASTICITY
  EXTERN template class CtrMap<Vec<2,double>>;
  EXTERN template class CtrMap<Vec<3,double>>;
  EXTERN template class CtrMap<Vec<6,double>>;
#endif
  EXTERN template class GridContractMap<H1Mesh>;
#ifdef ELASTICITY
  EXTERN template class GridContractMap<ElasticityMesh<2>>;
  EXTERN template class GridContractMap<ElasticityMesh<3>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGH1_CPP)
  EXTERN template class EmbedVAMG<H1AMG>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGELAST_CPP)
#ifdef ELASTICITY
  //#ifndef FILE_AMGELAST_CPP
  EXTERN template class ElasticityAMG<2>;
  EXTERN template class ElasticityAMG<3>;
  //#endif
  EXTERN template class EmbedVAMG<ElasticityAMG<2>, double, STABEW<2>>;
  EXTERN template class EmbedVAMG<ElasticityAMG<3>, double, STABEW<3>>;
#endif
#endif

#define InstTransMat(N,M) \
  template shared_ptr<trans_spm<stripped_spm<Mat<N,M,double>>>>	\
  TransposeSPM<stripped_spm<Mat<N,M,double>>> (const stripped_spm<Mat<N,M,double>> & mat);
  // template shared_ptr<trans_spm<SparseMatrix<Mat<N,M,double>>>>	\
  // TransposeSPM<SparseMatrix<typename strip_mat<Mat<N,M,double>>::type>> (const SparseMatrix<typename strip_mat<Mat<N,M,double>>::type> & mat);

#define InstMultMat(A,B,C) \
  template shared_ptr<stripped_spm<Mat<A,C,double>>>			\
  MatMultAB<stripped_spm<Mat<A,B,double>>, stripped_spm<Mat<B,C,double>>> (const stripped_spm<Mat<A,B,double>> & mata, const stripped_spm<Mat<B,C,double>> & matb);
  
#define InstEmbedMults(N,M) /* embedding NxN to MxM */	\
  InstMultMat(N,M,M); /* conctenate prols */		\
  InstMultMat(N,N,M); /* A * P */			\
  InstMultMat(M,N,M); /* PT * [A*P] */ 
  
#if !defined(AMG_EXTERN_TEMPLATES) && defined(FILE_AMG_SPMSTUFF_CPP)
  /** Transpose **/
  InstTransMat(1,1);
#ifdef ELASTICITY
  InstTransMat(1,3);
  InstTransMat(2,3);
  InstTransMat(3,3);
  InstTransMat(1,6);
  InstTransMat(3,6);
  InstTransMat(6,6);
#endif

  /** A * B **/
  InstMultMat(1,1,1);
#ifdef ELASTICITY
  InstEmbedMults(1,3);
  InstEmbedMults(2,3);
  InstMultMat(3,3,3);
  InstMultMat(6,6,6);
  InstEmbedMults(1,6);
  InstEmbedMults(3,6);
#endif

#endif
  
} // namespace amg

#undef EXTERN
#undef InstTransMat
#undef InstMultMat
#undef InstEmbedMults

#endif
