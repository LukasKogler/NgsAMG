#ifndef FILE_AMG_SPMSTUFF
#define FILE_AMG_SPMSTUFF

/**
   SparseMatrix - Multiplication taken from NGSolve and templated here.
   Remove this file as soon as this stuff comes into NGSolve!!
**/
namespace amg
{
  template<class T>
  INLINE void print_tm_spmat (ostream &os, const T & mat) {
    constexpr int H = mat_traits<typename std::remove_reference<typename std::result_of<T(int, int)>::type>::type>::HEIGHT;
    constexpr int W = mat_traits<typename std::remove_reference<typename std::result_of<T(int, int)>::type>::type>::WIDTH;
    for (auto k : Range(mat.Height())) {
      auto ri = mat.GetRowIndices(k);
      auto rv = mat.GetRowValues(k);
      if (ri.Size()) {
	for (int kH : Range(H)) {
	  if (kH==0) os << "Row " << setw(6) << k  << ": ";
	  else os << "          : ";
      	  for (auto j : Range(ri.Size())) {
	    if (kH==0) os << setw(4) << ri[j] << ": ";
	    else os << "    : ";
	    auto& etr = rv[j];
	    for (int jW : Range(W)) { os << setw(4) << etr(kH,jW) << " "; }
	    os << " | ";
	  }
	  os << endl;
	}
      }
      else { os << "Row " << setw(6) << k << ": (empty)" << endl; }
    }
  }
  template<> INLINE void print_tm_spmat (ostream &os, const SparseMatrix<double> & mat) { os << mat << endl; }

  INLINE int GetEntrySize (BaseSparseMatrix* mat)
  {
    if (auto m = dynamic_cast<SparseMatrixTM<double>*>(mat)) { return 1; }
    else if (auto m = dynamic_cast<SparseMatrixTM<Mat<2,2,double>>*>(mat)) { return 4; }
    else if (auto m = dynamic_cast<SparseMatrixTM<Mat<3,3,double>>*>(mat)) { return 9; }
    else if (auto m = dynamic_cast<SparseMatrixTM<Mat<4,4,double>>*>(mat)) { return 16; }
    else if (auto m = dynamic_cast<SparseMatrixTM<Mat<5,5,double>>*>(mat)) { return 25; }
    else if (auto m = dynamic_cast<SparseMatrixTM<Mat<6,6,double>>*>(mat)) { return 36; }
    return -1;
  }
	
  
  // Strip Mat<1,1,double> and Vec<1,double> -> double
  template<class T> struct strip_mat { typedef T type; };
  template<> struct strip_mat<Mat<1,1,double>> { typedef double type; };
  template<class T> struct strip_vec { typedef T type; };
  template<> struct strip_vec<Vec<1, double>> { typedef double type; };
  // Strip SparseMatrix<Mat<...>,..> -> SparseMatrix<double>
  template<class TM,
	   class TVX = typename mat_traits<TM>::TV_ROW,
	   class TVY = typename mat_traits<TM>::TV_COL>
  using stripped_spm =  SparseMatrix<typename strip_mat<TM>::type,
				  typename strip_vec<TVX>::type,
				  typename strip_vec<TVY>::type >;
  template<class TM> struct TM_OF_SPM { typedef typename std::remove_reference<typename std::result_of<TM(int, int)>::type >::type type; };
  template<> struct TM_OF_SPM<SparseMatrix<double>> { typedef double type; };
  template<class TSPM>
  using strip_spm =  SparseMatrix<typename strip_mat<typename TM_OF_SPM<TSPM>::type>::type,
				  typename strip_vec<typename TSPM::TVX>::type,
				  typename strip_vec<typename TSPM::TVY>::type >;
  // Sparse Matrix Transpose
  template<class TMA>
  using trans_spm = stripped_spm<Mat<mat_traits<typename TMA::TVX>::HEIGHT,
				     mat_traits<typename TMA::TVY>::HEIGHT,
				     typename TMA::TSCAL>,
				 typename TMA::TVY,
				 typename TMA::TVX>;
  template <typename TMA> shared_ptr<trans_spm<TMA>> TransposeSPM (const TMA & mat);

  // Sparse Matrix Multiplication
  template<class TSA, class TSB> struct mult_scal { typedef double type; };
  template<class TMA, class TMB>
  using mult_spm = stripped_spm<Mat<mat_traits<typename TMA::TVY>::HEIGHT,
				    mat_traits<typename TMB::TVX>::HEIGHT,
				    typename mult_scal<typename TMA::TSCAL, typename TMB::TSCAL>::type>,
				typename TMB::TVX,
				typename TMA::TVY >;
  template <typename TMA, typename TMB> shared_ptr<mult_spm<TMA,TMB>>
  MatMultAB (const TMA & mata, const TMB & matb);

  // Get TM/TVX/TVY back from SparseMatrix<....>

  template<class TSPMAT> struct amg_spm_traits {
    typedef stripped_spm<Mat<mat_traits<typename TSPMAT::TVX>::HEIGHT, mat_traits<typename TSPMAT::TVX>::HEIGHT, typename TSPMAT::TSCAL>,
			 typename TSPMAT::TVX, typename TSPMAT::TVX> T_RIGHT; // TVX - row_vector
    typedef stripped_spm<Mat<mat_traits<typename TSPMAT::TVY>::HEIGHT, mat_traits<typename TSPMAT::TVY>::HEIGHT, typename TSPMAT::TSCAL>,
			 typename TSPMAT::TVY, typename TSPMAT::TVY> T_LEFT;  // TVY - col_vector
  };
  
  

  // B.T * A * B
  INLINE Timer & timer_hack_restrictspm () { static Timer t("RestrictMatrix"); return t; }
  template <typename TMA, typename TMB>
  INLINE shared_ptr<mult_spm<mult_spm<trans_spm<TMB>,TMA>, TMB>>
  RestrictMatrix (const TMA & A, const TMB & P)
  {
    RegionTimer rt(timer_hack_restrictspm());
    // cout << "restrict A: " << A.Height() << " x " << A.Width() << endl;
    // cout << "with P: " << P.Height() << " x " << P.Width() << endl;
    // cout << "A: " << endl; print_tm_spmat(cout, A); cout << endl;
    // cout << "P: " << endl; print_tm_spmat(cout, P); cout << endl;

    auto PT = TransposeSPM(P);
    // cout << "PT: " << PT->Height() << " x " << PT->Width() << endl;
    // cout << "PT: " << endl; print_tm_spmat(cout, *PT); cout << endl;
    auto AP = MatMultAB(A, P);
    // cout << "AP: " << AP->Height() << " x " << AP->Width() << endl;
    // cout << "AP: " << endl; print_tm_spmat(cout, *AP); cout << endl;
    auto PTAP = MatMultAB(*PT, *AP);
    // cout << "result PTAP: " << PTAP->Height() << " x " << PTAP->Width() << endl;
    return PTAP;
  }

} // namespace amg


#endif
