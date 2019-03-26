#ifndef FILE_AMG_SPMSTUFF
#define FILE_AMG_SPMSTUFF

/**
   SparseMatrix - Multiplication taken from NGSolve and templated here.
   Remove this file as soon as this stuff comes into NGSolve!!
**/
namespace amg
{
  // Strip Mat<1,1,double> and Vec<1,double> -> double
  template<class T> struct strip_mat { typedef T type; };
  template<> struct strip_mat<Mat<1,1,double>> { typedef double type; };
  template<class T> struct strip_vec { typedef T type; };
  template<> struct strip_vec<Vec<1, double>> { typedef double type; };
  // Strip SparseMatrix<Mat<...>,..> -> SparseMatrix<double>
  template<class TM, class TVX, class TVY> struct strip_spmat {
    typedef SparseMatrix<typename strip_mat<TM>::type,
			 typename strip_vec<TVX>::type,
			 typename strip_vec<TVY>::type > type;
  };
  template<class TSA, class TSB> struct mult_scal { typedef double type; };
  template<class TMA, class TMB> struct mult_spm {
    typedef typename strip_spmat<Mat<mat_traits<typename TMA::TVY>::HEIGHT,
				     mat_traits<typename TMB::TVX>::HEIGHT,
				     typename mult_scal<typename TMA::TSCAL, typename TMB::TSCAL>::type>,
				 typename TMA::TVY,
				 typename TMB::TVX>::type type;
  };
  template<class TMA> struct trans_spm {
    typedef typename strip_spmat<Mat<mat_traits<typename TMA::TVX>::HEIGHT,
				     mat_traits<typename TMA::TVY>::HEIGHT,
				     typename TMA::TSCAL>,
				 typename TMA::TVX,
				 typename TMA::TVY>::type type;
  };
  // Get TM/TVX/TVY back from SparseMatrix<....>
  template<class TM> struct TM_OF_SPM { typedef typename std::remove_reference<typename std::result_of<TM(int, int)>::type >::type type; };
  template<> struct TM_OF_SPM<SparseMatrix<double>> { typedef double type; };

  // template<class TM> struct TVX_OF_SPM { typedef typename TM::TVX type; };
  // template<> struct TVX_OF_SPM<SparseMatrix<double>> { typedef double type; };
  // template<class TM> struct TVY_OF_SPM { typedef typename TM::TVY type; };
  // template<> struct TVY_OF_SPM<SparseMatrix<double>> { typedef double type; };

  
  // A.T
  template <typename TMA> shared_ptr<typename trans_spm<TMA>::type> TransposeSPM (const TMA & mat);
  // template<> shared_ptr<SparseMatrix<double>> TransposeSPM<SparseMatrix<double>> (const SparseMatrix<double> & mat);
  
  // A * B
  // typename TSIZE_MATCH = typename std::enable_if<mat_traits<typename TMA::TVX>::HEIGHT==mat_traits<typename TMB::TVY>::HEIGHT>::type >
  template <typename TMA, typename TMB>
  shared_ptr<typename mult_spm<TMA,TMB>::type>
  MatMultAB (const TMA & mata, const TMB & matb);

  // B.T * A * B
  template <typename TMA, typename TMB,
	    typename TSIZE_MATCH = typename std::enable_if<mat_traits<TMA>::WIDTH==mat_traits<TMB>::HEIGHT>::type >
  // shared_ptr<typename mult_spm<typename trans_spm<TMB>::type,TMB>::type>
  INLINE shared_ptr<typename mult_spm<typename mult_spm<typename trans_spm<TMB>::type,TMA>::type, TMB>::type>
  RestrictMatrix (const TMA & A, const TMB & P)
  {
    cout << "restrict A: " << A.Height() << " x " << A.Width() << endl;
    cout << "with P: " << P.Height() << " x " << P.Width() << endl;
    auto PT = TransposeSPM(P);
    cout << "PT: " << PT->Height() << " x " << PT->Width() << endl;
    auto AP = MatMultAB(A, P);
    cout << "AP: " << AP->Height() << " x " << AP->Width() << endl;
    auto PTAP = MatMultAB(*PT, *AP);
    cout << "result PTAP: " << PTAP->Height() << " x " << PTAP->Width() << endl;
    return PTAP;
  }

// #ifndef FILE_AMG_SPMSTUFF_CPP
// #define SPMSEXT extern
// #else
// #define SPMSEXT
// #endif
// #undef SPMSEXT

} // namespace amg


#endif
