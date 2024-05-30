#ifndef FILE_UTILS_BUFFERING_HPP
#define FILE_UTILS_BUFFERING_HPP

#include <base.hpp>

namespace amg
{

template<class T>
constexpr int SIZE_IN_BUFFER() { return SIZE_IN_BUFFER_TRAIT<T>::value; }

template<> struct SIZE_IN_BUFFER_TRAIT<double> { static constexpr int value = 1; };

INLINE int
PackIntoBuffer(double const &v, double *buf)
{
  *buf = v;
  return SIZE_IN_BUFFER<double>();
}

INLINE int
UnpackFromBuffer(double &v, double const *buf)
{
  v = *buf;
  return SIZE_IN_BUFFER<double>();
}

template<int N> struct SIZE_IN_BUFFER_TRAIT<Vec<N, double>> { static constexpr int value = N; };

template<int N>
INLINE int
PackIntoBuffer(Vec<N, double> const &v, double *buf)
{
  Iterate<N>([&](auto i) { buf[i] = v[i]; });
  return SIZE_IN_BUFFER<Vec<N,double>>();
}

template<int N>
INLINE int
UnpackFromBuffer(Vec<N, double> &v, double const *buf)
{
  Iterate<N>([&](auto i) { v[i] = buf[i]; });
  return SIZE_IN_BUFFER<Vec<N,double>>();
}

template<int N> struct SIZE_IN_BUFFER_TRAIT<Mat<N, N, double>> { static constexpr int value = N * N; };

template<int N>
INLINE int
PackIntoBuffer(Mat<N, N, double> const &m, double *buf)
{
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      buf[i.value * N + j.value] = m(i.value, j.value);
    });
  });
  return SIZE_IN_BUFFER<Mat<N,N,double>>();
}

template<int N>
INLINE int
UnpackFromBuffer(Mat<N, N, double> &m, double const *buf)
{
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      m(i.value, j.value) = buf[i.value * N + j.value];
    });
  });
  return SIZE_IN_BUFFER<Mat<N,N,double>>();
}

template<int N> struct SIZE_IN_BUFFER_TRAIT<INT<N, double>> { static constexpr int value = N; };

template<int N>
INLINE int
PackIntoBuffer(INT<N, double> const &t, double *buf)
{
  Iterate<N>([&](auto i) {
    buf[i.value] = t[i.value];
  });
  return SIZE_IN_BUFFER<INT<N,double>>();
}

template<int N>
INLINE int
UnpackFromBuffer(INT<N, double> &t, double const *buf)
{
  Iterate<N>([&](auto i) {
    t[i.value] = buf[i.value];
  });
  return SIZE_IN_BUFFER<INT<N,double>>();
}

} // namespace amg

#endif // FILE_UTILS_BUFFERING_HPP