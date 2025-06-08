#ifndef FILE_UTILS_IO_HPP
#define FILE_UTILS_IO_HPP

#include "utils_sparseLA.hpp"
#include "utils_sparseMM.hpp"

namespace amg
{

template<class T>
INLINE void print_ft(std::ostream &os, const T& t)
{
  if (!t.Size())
    { os << "empty flattable!!" << endl; }
  else {
    cout << "t.s: " << t.Size() << endl;
    for (auto k : Range(t.Size())) {
      os << "t[ " << k << "].s: " << t[k].Size();
      os << " || "; prow(t[k], os); os << endl;
    }
  }
} // print_ft

template<typename TA, typename TB> INLINE ostream & operator << (ostream &os, const tuple<TA, TB>& t)
  { return os << "t<" << get<0>(t) << ", " << get<1>(t) << ">" ; }


template<typename T> ostream & operator << (ostream &os, const FlatTable<T>& t)
{
  if (!t.Size())
    { return ( os << "empty flattable!!" << endl ); }
  os << "t.s: " << t.Size() << endl;
  for (auto k:Range(t.Size())) {
    os << "t[ " << k << "].s: " << t[k].Size();
    os << " || "; prow(t[k], os); os << endl;
  }
  return os;
}

template<class T>
INLINE void print_tm (ostream &os, const T & mat)
{
  constexpr int H = mat_traits<T>::HEIGHT;
  constexpr int W = mat_traits<T>::WIDTH;
  for (int kH : Range(H)) {
    for (int jW : Range(W)) { os << mat(kH,jW) << " "; }
    os << endl;
  }
} // print_tm

template<> INLINE void print_tm (ostream &os, const double & mat)
  { os << mat << endl; }

template<class T>
INLINE void print_tm_mat (ostream &os, const T & mat)
{
  constexpr int H = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int, int>::type>::type>::HEIGHT;
  constexpr int W = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int, int>::type>::type>::WIDTH;
  for (auto k : Range(mat.Height())) {
    for (int kH : Range(H)) {
      for (auto j : Range(mat.Width())) {
        auto& etr = mat(k,j);
        for (int jW : Range(W))
          { os << etr(kH,jW) << " "; }
        os << " | ";
      }
      os << endl;
    }
    os << "----" << endl;
  }
} // print_tm_mat

template<> INLINE void print_tm_mat (ostream &os, const Matrix<double> & mat)
  { os << mat << endl; }

template<> INLINE void print_tm_mat (ostream &os, const FlatMatrix<double> & mat)
  { os << mat << endl; }


template<class T>
INLINE void prow_tm (ostream &os, const T & ar)
{
  constexpr int H = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int>::type>::type>::HEIGHT;
  constexpr int W = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int>::type>::type>::WIDTH;
  for (int kH : Range(H)) {
    for (auto k : Range(ar.Size())) {
      if (kH==0)
        { os << "" << setw(4) << k << "::"; }
      else
        { os << "      "; }
      auto& etr = ar(k);
      if constexpr(H == W)
        { os << etr; }
      else for (int jW : Range(W))
        { os << etr(kH,jW) << " "; }
      os << " | ";
    }
    os << endl;
  }
} // prow_tm

template<class T>
INLINE void print_tm_spmat (ostream &os, const T & mat) {
  constexpr int H = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int, int>::type>::type>::HEIGHT;
  constexpr int W = mat_traits<typename std::remove_reference<typename std::invoke_result<T,int, int>::type>::type>::WIDTH;
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
    else
      { os << "Row " << setw(6) << k << ": (empty)" << endl; }
  }
} // print_tm_spmat

template<> INLINE void print_tm_spmat (ostream &os, const SparseMatrixTM<double> & mat)
  { os << mat << endl; }

template<> INLINE void print_tm_spmat (ostream &os, const SparseMatrix<double> & mat)
  { os << mat << endl; }

INLINE void print_spmat (ostream &os, const BaseSparseMatrix &mat)
{
  DispatchRectangularMatrixBS(mat, [&](auto H, auto W) {
    const auto &tmMat = static_cast<const stripped_spm_tm<Mat<H.value,W.value,double>>&>(mat);
    print_tm_spmat(os, tmMat);
  });
}

INLINE std::ostream & operator<<(std::ostream &os, const ParallelDofs& p)
{
  auto comm = p.GetCommunicator();
  os << "Pardofs, rank " << comm.Rank() << " of " << comm.Size() << endl;
  os << "ndof = " << p.GetNDofLocal() << ", glob " << p.GetNDofGlobal() << endl;
  os << "dist-procs: " << endl;
  for (auto k : Range(p.GetNDofLocal())) {
    auto dps = p.GetDistantProcs(k);
    os << k << ": "; prow(dps, os); os << endl;
  }
  os << endl;
  return os;
}


template<class T>
INLINE std::string my_typename(T const *ptr)
{
  if (ptr == nullptr)
  {
    return std::string(" IS NULLPTR!");
  }
  else
  {
    return typeid(*ptr).name();
  }
}

template<class T>
INLINE std::string my_typename(T *ptr)
{
  if (ptr == nullptr)
  {
    return std::string(" IS NULLPTR!");
  }
  else
  {
    return typeid(*ptr).name();
  }
}

template<class T>
INLINE std::string my_typename(shared_ptr<T> const &ptr)
{
  return my_typename(ptr.get());
}

template<class T>
INLINE void prow3 (const T & ar, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  for (auto k : Range(ar.Size())) {
    if (k > 0 && (k%per_row == 0))
    {
      os << endl << off;
    }
    os << "(" << k << "::" << ar[k] << ") ";
  }
}

INLINE void prowBA (const BitArray & ba, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  for (auto k : Range(ba.Size())) {
    if ( (k > 0) &&
         ( ( k % per_row ) == 0 ) )
    {
      os << endl << off;
    }
    os << "(" << k << "=" << int( ba.Test(k) ? 1 : 0 ) << ") ";
  }
}

INLINE void prowBA (const BitArray *ba, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  if (ba == nullptr)
  {
    os << off << " BitArray is nullptr !";
  }
  else
  {
    prow3(*ba, os, off, per_row);
  }
}

template<class T>
INLINE void printTable (const T & tab, std::ostream &os = cout, std::string const &off = "", int per_row = 30)
{
  std::string off2 = off + "  ";
  os << off << "Table w. " << tab.Size() << " rows:" << endl;
  for (auto k : Range(tab)) {
    os << off << "row " << k << "/" << tab.Size() << ", s = " << tab[k].Size() << ": "; 
    prow3(tab[k], os, off2, per_row);
    os << endl;
  }
}


template<class T, int D>
INLINE void prowtup (FlatArray<IVec<D, T>> ar, std::ostream &os = cout) {
  for (auto v : ar) {
    os << "[";
    for (auto l : Range(D))
      { os << v[l] << ", "; }
    os << "] ";
  }
};

template<class T, int D>
INLINE void prow2tup (FlatArray<IVec<D, T>> ar, std::ostream &os = cout) {
  for (auto k : Range(ar)) {
    auto v = ar[k];
    os << "[" << k << ": ";
    for (auto l : Range(D))
      { os << v[l] << ", "; }
    os << "] ";
  }
};


INLINE void
StructuredPrint(int BS, FlatMatrix<double> A, std::ostream &os = std::cout)
{
  auto H = A.Height();
  auto h = H / BS;

  auto W = A.Width();
  auto w = W / BS;

  for (auto K : Range(h))
  {
    for (auto lK : Range(BS))
    {
      auto k = BS * K + lK;

      for (auto J : Range(w))
      {
        for (auto lJ : Range(BS))
        {
          auto j = BS * J + lJ;

          auto val = A(k, j);

          if ( val > 0 )
          {
            os << " ";
          }

          os << std::scientific << std::setprecision(3) << setw(12) << val << " ";
        }
        os << " | ";
      }
      os << endl;
    }
    os << " --------- " << endl;
  }
  os << std::defaultfloat;
}

INLINE void
StructuredPrint(double const &A, std::ostream &os = std::cout)
{
  os << A;
}

INLINE void
StructuredPrint(float const &A, std::ostream &os = std::cout)
{
  os << A;
}

INLINE void
StructuredPrint(FlatMatrix<double> A, std::ostream &os = std::cout)
{
  os << A;
}

INLINE void
StructuredPrint(FlatMatrix<float> A, std::ostream &os = std::cout)
{
  os << A;
}

template<int BS, class T>
INLINE void
StructuredPrint(FlatMatrix<Mat<BS, BS, T>> A, std::ostream &os = std::cout)
{
  auto h = A.Height();
  auto w = A.Width();

  for (auto K : Range(h))
  {
    for (auto lK : Range(BS))
    {
      for (auto J : Range(w))
      {
        for (auto lJ : Range(BS))
        {
          auto val = A(K, J)(lK, lJ);

          if ( val > 0 )
          {
            os << " ";
          }

          os << std::scientific << std::setprecision(3) << setw(12) << val << " ";
        }
        os << " | ";
      }
      os << endl;
    }
    os << " --------- " << endl;
  }
  os << std::defaultfloat;
}



INLINE void
PrintComponent(int comp, int BS, FlatMatrix<double> mtx)
{
  int h = mtx.Height() / BS;
  int w = mtx.Width() / BS;

  for (auto k : Range(h))
  {
    for (auto j : Range(w))
    {
      cout << mtx(k*BS+comp, j*BS+comp) << " ";
    }
    cout << endl;
  }
}

INLINE void
PrintComponentTM(int comp, FlatMatrix<double> mtx)
{
  cout << mtx;
}

template<int BS>
INLINE void
PrintComponentTM(int comp, FlatMatrix<Mat<BS, BS, double>> mtx)
{
  int h = mtx.Height();
  int w = mtx.Width();

  for (auto k : Range(h))
  {
    for (auto j : Range(w))
    {
      cout << mtx(k, j)(comp, comp) << " ";
    }
    cout << endl;
  }
}

template<class TI, class TJ>
INLINE void
CheckRange (std::string const &tag, TI off, TJ cnt, IntRange bufRange)
{
  cout << "checkRange " << tag << ": [" << off << ", " << off + cnt << ") in "
        << "[" << bufRange.First() << ", " << bufRange.Next() << ")" << endl;

  if ( off < bufRange.First() )
  {
    cout << " ERR - off UNDER!" << endl;
  }
  if (cnt > 0)
  {
    if ( off >= bufRange.Next() )
    {
      cout << " ERR - off OVER!" << endl;
    }
  }
  if ( off + cnt > bufRange.Next() )
  {
    cout << " ERR - off+cnt OVER!" << endl;
  }
}

} // namespace amg

#endif // FILE_UTILS_IO_HPP
