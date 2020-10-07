
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
#ifdef FILE_AMG_SPMSTUFF_HPP // only need these if we include the spmstuff-header (which we almost always do)
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_SPMSTUFF_CPP)

#define InstSPMS(N,M)				  \
  EXTERN template class SparseMatrixTM<Mat<N,M,double>>; \
  EXTERN template class SparseMatrix<Mat<N,M,double>>;

  // this does not work because of Conj(Trans(Mat<1,3>)) * double does not work for some reason...
  // EXTERN template class SparseMatrix<Mat<N,M,double>, typename amg::strip_vec<Vec<M,double>>::type, typename amg::strip_vec<Vec<N,double>>::type>;
  
#if MAX_SYS_DIM < 2
  InstSPMS(2,2);
  // if MAX_SYS_DIM>=2, this is now in NGSolve
  InstSPMS(1,2);
  InstSPMS(2,1);
#endif // MAX_SYS_DIM < 2

#if MAX_SYS_DIM < 3
  InstSPMS(3,3);
  // if MAX_SYS_DIM>=3, this is now in NGSolve
  InstSPMS(1,3);
  InstSPMS(3,1);
#endif // MAX_SYS_DIM < 3
  InstSPMS(2,3);
  InstSPMS(3,2);

#ifdef ELASTICITY

#if MAX_SYS_DIM < 6
  InstSPMS(6,6);
  // if MAX_SYS_DIM>=6, this is now in NGSolve
  InstSPMS(1,6);
  InstSPMS(6,1);
#endif // MAX_SYS_DIM < 6
  InstSPMS(3,6);
  InstSPMS(6,3);

#endif // ELASTICITY

#undef InstSPMS

#endif
#endif // FILE_AMG_SPMSTUFF_HPP
  
} // namespace ngla


namespace amg
{
#ifdef FILE_AMG_SMOOTHER_HPP // TODO: split this into seperate headers

#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM_CPP)
  EXTERN template class HybridGSS<1>;
  EXTERN template class HybridGSS<2>;
  EXTERN template class HybridGSS<3>;
#ifdef ELASTICITY
  EXTERN template class StabHGSS<3,2,3>;
  // EXTERN template class HybridGSS<4>;
  // EXTERN template class HybridGSS<5>;
  EXTERN template class HybridGSS<6>;
  EXTERN template class StabHGSS<6,3,6>;
#endif
#endif

#ifdef FILE_AMGSM2
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM2_CPP)
  EXTERN template class HybridGSS2<double>;
  EXTERN template class HybridGSS2<Mat<2,2,double>>;
  EXTERN template class HybridGSS2<Mat<3,3,double>>;
#ifdef ELASTICITY
  EXTERN template class HybridGSS2<Mat<6,6,double>>;
#endif
#endif
#endif // FILE_AMGSM2

#ifdef FILE_AMGSM3
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGSM3_CPP)
  EXTERN template class GSS3<double>;
  EXTERN template class HybridGSS3<double>;
  EXTERN template class HybridSmoother2<double>;
  EXTERN template class GSS3<Mat<2,2,double>>;
  EXTERN template class HybridGSS3<Mat<2,2,double>>;
  EXTERN template class HybridSmoother2<Mat<2,2,double>>;
  EXTERN template class GSS3<Mat<3,3,double>>;
  EXTERN template class HybridGSS3<Mat<3,3,double>>;
  EXTERN template class HybridSmoother2<Mat<3,3,double>>;
#ifdef ELASTICITY
  EXTERN template class GSS3<Mat<6,6,double>>;
  EXTERN template class HybridGSS3<Mat<6,6,double>>;
  EXTERN template class HybridSmoother2<Mat<6,6,double>>;
  EXTERN template class RegHybridGSS3<Mat<3,3,double>, 2, 3>;
  EXTERN template class RegHybridGSS3<Mat<6,6,double>, 3, 6>;
#endif // ELASTICITY
#endif
#endif // FILE_AMGSM3

#ifdef FILE_AMG_BS_HPP
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_BS_CPP)
  EXTERN template class BSmoother<double>;
  EXTERN template class HybridBS<double>;
  EXTERN template class BSmoother<Mat<2,2,double>>;
  EXTERN template class HybridBS<Mat<2,2,double>>;
  EXTERN template class BSmoother<Mat<3,3,double>>;
  EXTERN template class HybridBS<Mat<3,3,double>>;
#ifdef ELASTICITY
  EXTERN template class BSmoother<Mat<6,6,double>>;
  EXTERN template class HybridBS<Mat<6,6,double>>;
#endif // ELASTICITY
#endif
#endif // FILE_AMG_BS_HPP

#endif // FILE_AMG_SMOOTHER_HPP

#ifdef FILE_AMGCRS
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCRS_CPP)

  EXTERN template class SeqVWC<FlatTM>;

#ifdef FILE_AMGH1_HPP
  EXTERN template class BlockVWC<H1Mesh>;
  EXTERN template class HierarchicVWC<H1Mesh>;
  EXTERN template class CoarseMap<H1Mesh>;
#endif //  FILE_AMGH1_HPP

#if defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
  EXTERN template class BlockVWC<ElasticityMesh<2>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<2>>;
  EXTERN template class CoarseMap<ElasticityMesh<2>>;
  EXTERN template class BlockVWC<ElasticityMesh<3>>;
  EXTERN template class HierarchicVWC<ElasticityMesh<3>>;
  EXTERN template class CoarseMap<ElasticityMesh<3>>;
#endif // ELASTICITY && FILE_AMG_ELAST_HPP

#endif // defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCRS_CPP)
#endif // FILE_AMGCRS

#ifdef FILE_AMGCRS2_HPP
#ifdef AMG_EXTERN_TEMPLATES
#ifdef FILE_AMGH1_HPP
  EXTERN template class Agglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  EXTERN template class Agglomerator<H1Energy<2, double, double>, H1Mesh, H1Energy<2, double, double>::NEED_ROBUST>;
  EXTERN template class Agglomerator<H1Energy<3, double, double>, H1Mesh, H1Energy<3, double, double>::NEED_ROBUST>;
#ifdef SPWAGG
  EXTERN template class SPWAgglomerator<H1Energy<1, double, double>, H1Mesh, H1Energy<1, double, double>::NEED_ROBUST>;
  EXTERN template class SPWAgglomerator<H1Energy<2, double, double>, H1Mesh, H1Energy<2, double, double>::NEED_ROBUST>;
  EXTERN template class SPWAgglomerator<H1Energy<3, double, double>, H1Mesh, H1Energy<3, double, double>::NEED_ROBUST>;
#endif // SPWAGG
#endif // FILE_AMGH1_HPP

#define InstAgg(FCLASS) \
  EXTERN template class Agglomerator<FCLASS::ENERGY, FCLASS::TMESH, FCLASS::ENERGY::NEED_ROBUST>;

#if defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
  EXTERN template class Agglomerator<ElasticityAMGFactory<2>::ENERGY, ElasticityAMGFactory<2>::TMESH, ElasticityAMGFactory<2>::ENERGY::NEED_ROBUST>;
  EXTERN template class Agglomerator<ElasticityAMGFactory<3>::ENERGY, ElasticityAMGFactory<3>::TMESH, ElasticityAMGFactory<3>::ENERGY::NEED_ROBUST>;
#ifdef SPWAGG
  EXTERN template class SPWAgglomerator<ElasticityAMGFactory<2>::ENERGY, ElasticityAMGFactory<2>::TMESH, ElasticityAMGFactory<2>::ENERGY::NEED_ROBUST>;
  EXTERN template class SPWAgglomerator<ElasticityAMGFactory<3>::ENERGY, ElasticityAMGFactory<3>::TMESH, ElasticityAMGFactory<3>::ENERGY::NEED_ROBUST>;
#endif // SPWAGG
#endif // defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
#endif // AMG_EXTERN_TEMPLATES
#endif // FILE_AMGCRS2_HPP

#ifdef FILE_AMGCTR
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGCTR_CPP)
  EXTERN template class CtrMap<double>;
  EXTERN template class CtrMap<Vec<2,double>>;
  EXTERN template class CtrMap<Vec<3,double>>;
#ifdef ELASTICITY
  EXTERN template class CtrMap<Vec<6,double>>;
#endif // ELASTICITY

#ifdef FILE_AMGH1_HPP
  EXTERN template class GridContractMap<H1Mesh>;
#endif // FILE_AMGH1_HPP
#if defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
  EXTERN template class GridContractMap<ElasticityMesh<2>>;
  EXTERN template class GridContractMap<ElasticityMesh<3>>;
#endif // defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
#endif
#endif // FILE_AMGCTR

#if defined (STOKES) && defined(FILE_STOKES_GG_HPP)
  EXTERN template class GridContractMap<GGStokesMesh<2>>;
#endif // defined (STOKES) && defined(FILE_STOKES_GG_HPP)

#ifdef FILE_AMG_DISCARD_HPP
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMG_DISCARD_CPP)
#ifdef FILE_AMGH1_HPP
  EXTERN template class VDiscardMap<H1Mesh>;
#endif // FILE_AMGH1_HPP
#if defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
  EXTERN template class VDiscardMap<ElasticityMesh<2>>;
  EXTERN template class VDiscardMap<ElasticityMesh<3>>;
#endif // defined(ELASTICITY) && defined(FILE_AMG_ELAST_HPP)
#endif
#endif // FILE_AMG_DISCARD_HPP

// what was this for??
// #ifdef FILE_AMGH1_HPP
// #if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGH1_CPP)
//   EXTERN template class EmbedVAMG<H1AMGFactory>;
//   EXTERN template class EmbedWithElmats<H1AMGFactory, double, double>;
// #endif
// #endif //  FILE_AMGPC_HPP

#ifdef FILE_AMG_MAP
#if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGMAP_CPP)
#define InstProlMap(A,B) \
  EXTERN template class ProlMap<stripped_spm_tm<Mat<A,B,double>>>;

  InstProlMap(1,1);
  InstProlMap(1,2);
  InstProlMap(2,2);
  InstProlMap(1,3);
  InstProlMap(2,3);
  InstProlMap(3,3);
#ifdef ELASTICITY
  InstProlMap(1,6);
  InstProlMap(3,6);
  InstProlMap(6,6);
#endif
#ifdef STOKES
  /** Need these for embed-prol to potential space! **/
  InstProlMap(2,1);
  InstProlMap(3,1);
#endif
#undef InstProLMap
#endif
#endif //  FILE_AMG_MAP

  // what was this for??
// #if defined(AMG_EXTERN_TEMPLATES) ^ defined(FILE_AMGELAST_CPP)
// #ifdef ELASTICITY
//   //#ifndef FILE_AMGELAST_CPP
//   EXTERN template class ElasticityAMGFactory<2>;
//   EXTERN template class ElasticityAMGFactory<3>;
//   //#endif
//   EXTERN template class EmbedVAMG<ElasticityAMGFactory<2>>;
//   EXTERN template class EmbedVAMG<ElasticityAMGFactory<3>>;
//   // EXTERN template class EmbedWithElmats<ElasticityAMGFactory<2>, double, ElasticityEdgeData<2>>;
//   // EXTERN template class EmbedWithElmats<ElasticityAMGFactory<3>, double, ElasticityEdgeData<3>>;
//   EXTERN template class EmbedWithElmats<ElasticityAMGFactory<2>, double, double>;
//   EXTERN template class EmbedWithElmats<ElasticityAMGFactory<3>, double, double>;
// #endif
// #endif


#ifdef FILE_AMG_SPMSTUFF_HPP
#define InstTransMat(N,M)						\
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
  /** [A \times B] Transpose **/
  InstTransMat(1,1);
  InstTransMat(1,2);
  InstTransMat(2,1);
  InstTransMat(2,2);
  InstTransMat(3,3);
  InstTransMat(1,3);
  InstTransMat(3,1);
  InstTransMat(2,3);
#ifdef ELASTICITY
  InstTransMat(1,6);
  InstTransMat(3,6);
  InstTransMat(6,6);
#endif

  /** [A \times B] * [B \times C] **/
  InstMultMat(1,1,1);
  InstMultMat(2,2,2);
  InstMultMat(3,3,3);
  InstEmbedMults(1,2);
  InstEmbedMults(2,1);
  InstEmbedMults(1,3);
  InstEmbedMults(3,1);
  InstEmbedMults(2,3);
#ifdef ELASTICITY
  InstMultMat(6,6,6);
  InstEmbedMults(1,6);
  InstEmbedMults(2,6);
  InstEmbedMults(3,6);
#endif

#endif
#endif // FILE_AMG_SPMSTUFF_HPP
} // namespace amg

#undef EXTERN
#undef InstTransMat
#undef InstMultMat
#undef InstEmbedMults

#endif
