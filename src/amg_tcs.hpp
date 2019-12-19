
/** 
    This file is included twice - once with AMG_EXTERN_TEMPLATES, and once without:
       The first include happens from "amg.hpp", to get "extern template class ..."
       The second include happens at the end of the ".cpp" files to get "template class ..."
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

#define InstSPMS(N,M)				  \
  EXTERN template class SparseMatrixTM<Mat<N,M,double>>; \
  EXTERN template class SparseMatrix<Mat<N,M,double>>;

  // this does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
  // EXTERN template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;
  
#if MAX_SYS_DIM < 3
  InstSPMS(3,3);
#endif
  InstSPMS(1,3);
  InstSPMS(3,1);
  InstSPMS(2,3);
  InstSPMS(3,2);
#if MAX_SYS_DIM < 6
  InstSPMS(6,6);
#endif
  InstSPMS(1,6);
  InstSPMS(6,1);
  InstSPMS(3,6);
  InstSPMS(6,3);
#undef InstSPMS
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

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM2_CPP)
  EXTERN template class HybridGSS2<double>;
#ifdef ELASTICITY
  EXTERN template class HybridGSS2<Mat<2,2,double>>;
  EXTERN template class HybridGSS2<Mat<3,3,double>>;
  EXTERN template class HybridGSS2<Mat<6,6,double>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM3_CPP)
  EXTERN template class HybridGSS3<double>;
#ifdef ELASTICITY
  EXTERN template class HybridGSS3<Mat<2,2,double>>;
  EXTERN template class HybridGSS3<Mat<3,3,double>>;
  EXTERN template class HybridGSS3<Mat<6,6,double>>;
  EXTERN template class RegHybridGSS3<Mat<3,3,double>, 2, 3>;
  EXTERN template class RegHybridGSS3<Mat<6,6,double>, 3, 6>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_BS_CPP)
  EXTERN template class HybridBS<double>;
// #ifdef ELASTICITY
//   EXTERN template class HybridBS<Mat<2,2,double>>;
//   EXTERN template class HybridBS<Mat<3,3,double>>;
//   EXTERN template class HybridBS<Mat<6,6,double>>;
// #endif
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

#if defined(AMG_EXTERN_TEMPLATES)
  EXTERN template class Agglomerator<H1AMGFactory>;
#ifdef ELASTICITY
  EXTERN template class Agglomerator<ElasticityAMGFactory<2>>;
  EXTERN template class Agglomerator<ElasticityAMGFactory<3>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCTR_CPP)
  EXTERN template class CtrMap<double>;
#ifdef ELASTICITY
  // EXTERN template class CtrMap<Vec<2,double>>; // why??
  EXTERN template class CtrMap<Vec<3,double>>;
  EXTERN template class CtrMap<Vec<6,double>>;
#endif
  EXTERN template class GridContractMap<H1Mesh>;
#ifdef ELASTICITY
  EXTERN template class GridContractMap<ElasticityMesh<2>>;
  EXTERN template class GridContractMap<ElasticityMesh<3>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_DISCARD_CPP)
  EXTERN template class VDiscardMap<H1Mesh>;
#ifdef ELASTICITY
  EXTERN template class VDiscardMap<ElasticityMesh<2>>;
  EXTERN template class VDiscardMap<ElasticityMesh<3>>;
#endif
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGH1_CPP)
  EXTERN template class EmbedVAMG<H1AMGFactory>;
  EXTERN template class EmbedWithElmats<H1AMGFactory, double, double>;
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGMAP_CPP)
#define InstProlMap(A,B) \
  EXTERN template class ProlMap<stripped_spm_tm<Mat<A,B,double>>>;

  InstProlMap(1,1);
#ifdef ELASTICITY
  InstProlMap(1,3);
  InstProlMap(2,3);
  InstProlMap(3,3);
  InstProlMap(1,6);
  InstProlMap(3,6);
  InstProlMap(6,6);
#endif
#undef InstProLMap
#endif

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGELAST_CPP)
#ifdef ELASTICITY
  //#ifndef FILE_AMGELAST_CPP
  EXTERN template class ElasticityAMGFactory<2>;
  EXTERN template class ElasticityAMGFactory<3>;
  //#endif
  EXTERN template class EmbedVAMG<ElasticityAMGFactory<2>>;
  EXTERN template class EmbedVAMG<ElasticityAMGFactory<3>>;
  // EXTERN template class EmbedWithElmats<ElasticityAMGFactory<2>, double, ElasticityEdgeData<2>>;
  // EXTERN template class EmbedWithElmats<ElasticityAMGFactory<3>, double, ElasticityEdgeData<3>>;
  EXTERN template class EmbedWithElmats<ElasticityAMGFactory<2>, double, double>;
  EXTERN template class EmbedWithElmats<ElasticityAMGFactory<3>, double, double>;
#endif
#endif

#define InstTransMat(N,M) \
  template shared_ptr<trans_spm_tm<stripped_spm_tm<Mat<N,M,double>>>>	\
  TransposeSPM<stripped_spm_tm<Mat<N,M,double>>> (const stripped_spm_tm<Mat<N,M,double>> & mat);

#define InstMultMat(A,B,C)						\
  template shared_ptr<stripped_spm_tm<Mat<A,C,double>>>			\
  MatMultAB<stripped_spm_tm<Mat<A,B,double>>> (const stripped_spm_tm<Mat<A,B,double>> & mata, const stripped_spm_tm<Mat<B,C,double>> & matb);
  
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
