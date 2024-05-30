#ifndef FILE_UTILS_NUMERIC_TYPES_HPP
#define FILE_UTILS_NUMERIC_TYPES_HPP

namespace amg
{

enum AVG_TYPE : int {
  MIN,    // min(a,b)
  GEOM,   // sqrt(ab)
  HARM,   // 2 (ainv + binv)^{-1}
  ALG,    // (a+b)/2
  MAX     // max(a,b)
};

template<AVG_TYPE TAVG, class TV>
INLINE TV average(const TV & a, const TV & b)
{
  if constexpr(TAVG == MIN)
    { return min(a, b); }
  else if constexpr(TAVG == GEOM)
    { return sqrt(a * b); }
  else if constexpr(TAVG == HARM)
    { return 2. / (1. / a + 1. / b); }
  else if constexpr(TAVG == ALG)
    { return 0.5 * (a + b); }
  else if constexpr(TAVG == MAX)
    { return max(a, b); }
} // average

INLINE bool is_zero (const double & x)
  { return x == 0; }

INLINE bool is_zero (const int & x)
  { return x == 0; }

INLINE bool is_zero (const size_t & x)
  { return x == 0; }

template<int N, int M, typename TSCAL>
INLINE bool is_zero (const Mat<N, M,  TSCAL> & m)
{
  for (auto k : Range(N))
    for (auto j : Range(M))
      if (!is_zero(m(k,j)))
        { return false; }
      return true;
} // is_zero

template<int N, typename TSCAL>
INLINE bool is_zero (const IVec<N, TSCAL> & m)
{
  for (auto k : Range(N))
    if (!is_zero(m[k]))
      { return false; }
  return true;
} // is_zero

template<int N, typename TSCAL>
INLINE bool is_zero (const Vec<N, TSCAL> & m)
{
  for (auto k : Range(N))
    if (!is_zero(m(k)))
      { return false; }
  return true;
} // is_zero


  template<class T> INLINE bool is_invalid (T val) { return val == T(-1); };
  template<class T> INLINE bool is_valid (T val) { return val != T(-1); };

 } // namespace amg


#endif // FILE_UTILS_NUMERIC_TYPES_HPP