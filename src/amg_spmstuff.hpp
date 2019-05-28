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
  template<class TM> using stripped_spm_tm =  SparseMatrixTM<typename strip_mat<TM>::type>;
  template<class TM,
	   class TVX = typename mat_traits<TM>::TV_ROW,
	   class TVY = typename mat_traits<TM>::TV_COL>
  using stripped_spm =  SparseMatrix<typename strip_mat<TM>::type,
				  typename strip_vec<TVX>::type,
				  typename strip_vec<TVY>::type >;
  template<class TM> struct TM_OF_SPM { typedef typename std::remove_reference<typename std::result_of<TM(int, int)>::type >::type type; };
  template<> struct TM_OF_SPM<SparseMatrix<double>> { typedef double type; };
  //  Matrix Transpose
  template<class TM>
  using trans_mat = typename strip_mat<Mat<mat_traits<TM>::WIDTH, mat_traits<TM>::HEIGHT, typename mat_traits<TM>::TSCAL>>::type;
  template<class TM>
  using trans_spm_tm = SparseMatrixTM<trans_mat<typename TM::TENTRY>>;
  template<class TM>
  using trans_spm = SparseMatrix<trans_mat<typename TM::TENTRY>>;
  template <class TM> shared_ptr<trans_spm_tm<TM>> TransposeSPM (const TM & mat);

  // Matrix Multiplication
  template<class TSA, class TSB> struct mult_scal { typedef void type; };
  template<> struct mult_scal<double, double> { typedef double type; };
  template<class TMA, class TMB>
  using mult_mat = typename strip_mat<Mat<mat_traits<TMA>::HEIGHT, mat_traits<TMB>::WIDTH,
					  typename mult_scal<typename mat_traits<TMA>::TSCAL, typename mat_traits<TMB>::TSCAL>::type>>::type;
  template<class TMA, class TMB>
  using mult_spm_tm = stripped_spm_tm<mult_mat<typename TMA::TENTRY, typename TMB::TENTRY>>;
  template<class TMA, class TMB>
  shared_ptr<mult_spm_tm<TMA, TMB>> MatMultAB (const TMA & mata, const TMB & matb);
    
  
  INLINE Timer & timer_hack_restrictspm2 () { static Timer t("RestrictMatrix2"); return t; }
  template <class TMA, class TMB>
  INLINE shared_ptr<mult_spm_tm<mult_spm_tm<trans_spm_tm<TMB>,TMA>, TMB>>
  RestrictMatrixTM (const trans_spm_tm<TMB> & PT, const TMA & A, const TMB & P)
  {
    RegionTimer rt(timer_hack_restrictspm2());
    auto AP = MatMultAB(A, P);
    auto PTAP = MatMultAB(PT, *AP);
    return PTAP;
  }
  
  INLINE Timer & timer_hack_restrictspm1 () { static Timer t("RestrictMatrix1"); return t; }
  template <class TMA, class TMB>
  INLINE shared_ptr<mult_spm_tm<mult_spm_tm<trans_spm_tm<TMB>,TMA>, TMB>>
  RestrictMatrixTM (const TMA & A, const TMB & P)
  {
    RegionTimer rt(timer_hack_restrictspm1());
    auto PT = TransposeSPM(P);
    return RestrictMatrixTM(*PT, A, P);
  }


} // namespace amg


#endif
