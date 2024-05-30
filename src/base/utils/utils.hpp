#ifndef FILE_AMGUTILS
#define FILE_AMGUTILS

#include <base.hpp>

namespace amg
{

template<int D> INLINE void GetNodePos (NodeId id, const MeshAccess & ma, Vec<D> & pos, Vec<D> & t)
{
  auto set_pts = [&](auto pnums) LAMBDA_INLINE {
    pos = 0;
    for (auto k : Range(pnums)) {
      ma.GetPoint(pnums[k], t);
      pos += t;
    }
    pos *= 1.0/pnums.Size();
  };
  switch(id.GetType()) {
    case(NT_VERTEX) : { ma.GetPoint(id.GetNr(), pos); break; }
    case(NT_EDGE)   : {
      auto pnums = ma.GetEdgePNums(id.GetNr());
      pos = 0;
      ma.GetPoint(pnums[0], t);
      pos += t;
      ma.GetPoint(pnums[1], t);
      pos += t;
      pos *= 0.5;
      break;
    }
    case(NT_FACE)   : { set_pts(ma.GetFacePNums(id.GetNr())); break; }
    case(NT_CELL)   : { set_pts(ma.GetElPNums(ElementId(VOL, id.GetNr()))); break; }
    default         : { pos = 1e30; break; }
  }
}


template<typename T> bool operator < (const IVec<2,T> & a, const IVec<2,T> & b) {
  if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
  else return a[1]<b[1];
}
template<typename T> bool operator < (const IVec<3,T> & a, const IVec<3,T> & b) {
  if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
  else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
  else return a[2]<b[2];
}
template<typename T> bool operator < (const IVec<4,T> & a, const IVec<3,T> & b) {
  if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
  else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
  else if (a[2]<b[2]) return true; else if (a[2]>b[2]) return false;
  else return a[3]<b[3];
}
template<typename T> bool operator < (const IVec<5,T> & a, const IVec<5,T> & b) {
  if (a[0]<b[0]) return true; else if (a[0]>b[0]) return false;
  else if (a[1]<b[1]) return true; else if (a[1]>b[1]) return false;
  else if (a[2]<b[2]) return true; else if (a[2]>b[2]) return false;
  else if (a[3]<b[3]) return true; else if (a[2]>b[2]) return false;
  else return a[4]<b[4];
}

template<int N, class T> INLINE IVec<N,T> & operator += (IVec<N,T> & a, const IVec<N,T> & b)
{ Iterate<N>([&](auto i) { a[i.value] += b[i.value]; }); return a; }

/** size of a parameter pack **/
template<typename... T> struct count_ppack;
template<> struct count_ppack<> { static constexpr int value = 0; };
template<class T, typename... V> struct count_ppack<T, V...> { static constexpr int value = 1 + count_ppack<V...>::value;};

template<class SCAL> Matrix<SCAL> MakeDense (BaseMatrix & mat, BitArray* free = NULL);
// #ifndef FILE_AMG_UTILS_CPP
extern template Matrix<Complex> MakeDense (BaseMatrix & mat, BitArray* free);
extern template Matrix<double> MakeDense (BaseMatrix & mat, BitArray* free);
// #endif // FILE_AMG_UTILS_CPP

template<class T>
Table<T> CopyTable (const FlatTable<T> & tab_in);

extern template Table<int> CopyTable (const FlatTable<int> & tab_in);
extern template Table<double> CopyTable (const FlatTable<double> & tab_in);
extern template Table<size_t> CopyTable (const FlatTable<size_t> & tab_in);

// NGSolve is now more strict with "=" of arrays containing different datatypes,
// this convenience function lets us do it nonetheless
template<class TA, class TB>
INLINE void convertCopy(TA const &a, TB &b)
{
  for (auto k : Range(a))
  {
    b[k] = a[k];
  }
}

template<class T>
INLINE void prow (const T & ar, std::ostream &os = cout)
{
  for (auto v : ar)
    { os << v << " "; }
}

template<class T>
INLINE void prow2 (const T & ar, std::ostream &os = cout)
{
  for (auto k : Range(size_t(ar.Size())))
    { os << "(" << k << "::" << ar[k] << ") "; }
}

INLINE void SetIdentity (double & x)
  { x = 1.0; }

template<int H, int W, class TSCAL>
INLINE void
SetIdentity (Mat<H, W, TSCAL> &x)
{
  Iterate<H>([&](auto k)
  {
    Iterate<W>([&](auto j)
    {
      if constexpr(k == j)
      {
        x(k, j) = 1.0;
      }
      else
      {
        x(k, j) = 0.0;
      }
    });
  });
} // SetIdentity

template<class TM>
INLINE void
SetIdentity (FlatMatrix<TM> mat)
{
  for (auto k : Range(mat.Height()))
    { SetIdentity(mat(k,k)); }
} // SetIdentity

template<int D, class TM>
INLINE void
SetIdentity (FlatMatrixFixWidth<D, TM> mat)
{
  mat = 0;
  for (auto k : Range(D))
    { SetIdentity(mat(k,k)); }
} // SetIdentity

template<class TA, class TB, class ENABLE=std::enable_if_t<is_scalar_type<TB>::value>>
INLINE void
SetScalIdentity (TA const &scal, TB &x)
{
  x = scal;
}

template<int H, int W, class TSCAL_A, class TSCAL_B>
INLINE void
SetScalIdentity (TSCAL_A const &scal, Mat<H,W,TSCAL_B> &x)
{
  x = 0.0;
  for (auto i:Range(min(H,W)))
    { x(i,i) = scal;}
} // SetScalIdentity

template<class TSCAL, class TM>
INLINE void
SetScalIdentity (TSCAL const &scal, FlatMatrix<TM> mat)
{
  for (auto k : Range(mat.Height()))
    { SetScalIdentity(scal, mat(k,k)); }
} // SetScalIdentity

template<int D, class TM, class TSCAL> INLINE void
SetScalIdentity (TSCAL const &scal, FlatMatrixFixWidth<D, TM> mat)
{
  mat = 0;
  for (auto k : Range(D))
    { SetScalIdentity(scal, mat(k,k)); }
} // SetScalIdentity

template<int N, class TSCAL, class TALPHA>
INLINE void
addIdentity(Mat<N, N, TSCAL> &x,
            TALPHA const &alpha = 1.0)
{
  Iterate<N>([&](auto i) {
    x(i.value, i.value) += alpha;
  });
}

template<class TSCAL, class TALPHA, class ENABLE=std::enable_if_t<is_scalar_type<TSCAL>::value>>
INLINE void
addIdentity(TSCAL &x, TALPHA const &alpha = 1.0)
{
  x += alpha;
}


INLINE double kappaOf(double lammin, double lammax)
{
  return (lammin == 0.0) ? std::numeric_limits<double>::infinity() :  lammin / lammax;
} // kappaOf


template<class T, class TBASE>
INLINE shared_ptr<T> my_dynamic_pointer_cast(shared_ptr<TBASE> const &sp, std::string onFail = std::string("failed my_dynamic_pointer_cast"))
{
  if (sp == nullptr)
    { throw Exception(onFail + " - no input!"); }

  auto spt = dynamic_pointer_cast<T>(sp);

  if (spt == nullptr)
    { throw Exception(onFail + " - cast failed!"); }

  return spt;
} // _dynamic_cast

template<class T, class TBASE>
INLINE T const* my_dynamic_cast(TBASE const *sp, std::string onFail = std::string("failed my_dynamic_pointer_cast"))
{
  if (sp == nullptr)
    { throw Exception(onFail + " - no input!"); }

  auto spt = dynamic_cast<T const *>(sp);

  if (spt == nullptr)
    { throw Exception(onFail + " - cast failed!"); }

  return spt;
} // _dynamic_cast

template<class T, class TBASE>
INLINE T* my_dynamic_cast(TBASE *sp, std::string onFail = std::string("failed my_dynamic_pointer_cast"))
{
  if (sp == nullptr)
    { throw Exception(onFail + " - no input!"); }

  auto spt = dynamic_cast<T*>(sp);

  if (spt == nullptr)
    { throw Exception(onFail + " - cast failed!"); }

  return spt;
} // _dynamic_cast


// not sure this works, but it would be a point to start from
// template<class TARRAY> struct array_trait {
//   // typedef typename std::result_of<decltype(&T::operator[int])(T, int)>::type T0;
//   typedef typename std::remove_pointer<typename std::result_of<decltype(&TARRAY::Data)(TARRAY)>::type>::type T;
// };

template<class T>
INLINE T& checked_dereference(shared_ptr<T> const &sp)
{
  if (sp == nullptr)
  {
    throw Exception("Failed in checked_dereference!");
  }
  return *sp;
}

template<class T>
INLINE T const& checked_dereference(shared_ptr<const T> const &sp)
{
  if (sp == nullptr)
  {
    throw Exception("Failed in checked_dereference!");
  }
  return *sp;
}


} // namespace amg

#endif
