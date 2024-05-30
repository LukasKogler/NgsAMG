#ifndef FILE_BLA_EXTENSION_HPP
#define FILE_BLA_EXTENSION_HPP

#include <bla.hpp>

namespace ngbla
{
// template<class TM> constexpr int Height () { return mat_traits<TM>::HEIGHT; }
// template<class TM> constexpr int Width () { return mat_traits<TM>::WIDTH; }

// template<> constexpr int Height<double> () { return 1; }
// template<> constexpr int Width<double> () { return 1; }
} // namespace ngbla

/**
 *  Enable some expr-stuff for float. size_t I probably don't need, but put it here
 *  anyways
 */
namespace ngbla
{

template<> struct is_scalar_type<float>  { static constexpr bool value = true; };
template<> struct is_scalar_type<size_t> { static constexpr bool value = true; };

template <> inline constexpr auto Height<float> () { return 1; }
template <> inline constexpr auto Width<float>  () { return 1; }

template<class TM>
using TScal = typename mat_traits<TM>::TSCAL;

INLINE void CalcInverse (float x, float &inv) { inv = 1.0/x; }

INLINE void CalcInverse (float &x) { CalcInverse(x, x); }

template <typename TM, typename FUNC>
void T_NgGEMV (double s, BareSliceMatrix<TM,RowMajor> a, FlatVector<const double> x, FlatVector<double> y, FUNC func) NETGEN_NOEXCEPT  
{
  for (size_t i = 0; i < y.Size(); i++)
  {
    double sum = 0;
    for (size_t j = 0; j < x.Size(); j++)
      sum += a(i,j) * x(j);
    func(y(i), s*sum);
  }
}

template <typename TM, typename FUNC>
void T_NgGEMV (double s, BareSliceMatrix<TM,ColMajor> a, FlatVector<const double> x, FlatVector<double> y, FUNC func) NETGEN_NOEXCEPT  
{
  for (size_t i = 0; i < y.Size(); i++)
  {
    double sum = 0;
    for (size_t j = 0; j < x.Size(); j++)
      sum += a(i,j) * x(j);
    func(y(i), s*sum);
  }
}
  
template <bool ADD, ORDERING ord, int N, int M>
NGS_DLL_HEADER  
void NgGEMV (double s, BareSliceMatrix<Mat<N, M, double>,ord> a, FlatVector<Vec<M, double>> x, FlatVector<Vec<N, double>> y) NETGEN_NOEXCEPT
{
  if constexpr(ADD)
  {
    T_NgGEMV (s, a, x, y, [](double & y, double sum) { y+=sum; });
  }
  else
  {
    T_NgGEMV (s, a, x, y, [](double & y, double sum) { y=sum; });
  }
}
  

} // namespace ngbla

#endif // FILE_BLA_EXTENSION_HPP