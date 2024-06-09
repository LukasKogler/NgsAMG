
#define FILE_BLA_EXTENSION_CPP

#include "bla_extension.hpp"


#ifdef ELASTICITY
#if MAX_SYS_DIM < 6

// we need CalcInverse for 6x6 entries, which is not compiled into NGSolve with MAX_SYS_DIM<6
// so add that here, code copied out of NGSolve calcinverse.cpp

#include "calcinverse.hpp"

namespace ngbla
{
  inline double abs_IMPL (double a)
  {
    return std::fabs(a);
  }

  inline double abs_IMPL (Complex a)
  {
    return std::abs(a);
  }

  template <int N, int N2, typename SCAL>
  inline double abs_IMPL (Mat<N,N2,SCAL> & m)
  {
    double sum = 0;
    for (int i = 0; i < N; i++)
      sum += abs_IMPL(m(i,i));
    return sum;
  }
  
  template <class T2>
  void T_CalcInverse_IMPL (FlatMatrix<T2> inv)
  {
    // static Timer t("CalcInverse");
    // RegionTimer reg(t);

    // Gauss - Jordan - algorithm
    // Algorithm of Stoer, Einf. i. d. Num. Math, S 145
    // int n = m.Height();

    int n = inv.Height();

    ngstd::ArrayMem<int,100> p(n);   // pivot-permutation
    for (int j = 0; j < n; j++) p[j] = j;
    
    for (int j = 0; j < n; j++)
      {
	// pivot search
	double maxval = abs_IMPL(inv(j,j));
	int r = j;

	for (int i = j+1; i < n; i++)
	  if (abs_IMPL (inv(j, i)) > maxval)
	    {
	      r = i;
	      maxval = abs_IMPL (inv(j, i));
	    }
      
        double rest = 0.0;
        for (int i = j+1; i < n; i++)
          rest += abs_IMPL(inv(r, i));
	if (maxval < 1e-20*rest)
	  {
	    throw Exception ("Inverse matrix: Matrix singular");
	  }

	// exchange rows
	if (r > j)
	  {
	    for (int k = 0; k < n; k++)
	      swap (inv(k, j), inv(k, r));
	    swap (p[j], p[r]);
	  }
      

	// transformation
	
	T2 hr;
	CalcInverse (inv(j,j), hr);
	for (int i = 0; i < n; i++)
	  {
	    T2 h = hr * inv(j, i);
	    inv(j, i) = h;
	  }
	inv(j,j) = hr;

	for (int k = 0; k < n; k++)
	  if (k != j)
	    {
	      T2 help = inv(n*k+j);
	      T2 h = help * hr;   

	      for (int i = 0; i < n; i++)
		{
		  T2 h = help * inv(n*j+i); 
		  inv(n*k+i) -= h;
		}

	      inv(k,j) = -h;
	    }
      }

    // row exchange
  
    VectorMem<100,T2> hv(n);
    for (int i = 0; i < n; i++)
      {
	for (int k = 0; k < n; k++) hv(p[k]) = inv(k, i);
	for (int k = 0; k < n; k++) inv(k, i) = hv(k);
      }
  }

  template<> void CalcInverse<Mat<6,6,double>> (FlatMatrix<Mat<6,6,double> > inv, INVERSE_LIB il)
  {
    T_CalcInverse_IMPL (inv);
  }

} // namespace ngbla

#endif
#endif