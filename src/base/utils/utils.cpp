#define FILE_AMG_UTILS_CPP

#include <base.hpp>
#include <utils.hpp>

namespace amg
{

template<class SCAL>
Matrix<SCAL> MakeDense (BaseMatrix & mat, BitArray* free)
{
  int N = mat.Height(), n = (free == NULL) ? N : free->NumSet();

  Matrix<SCAL> dmat(n);
  auto v = mat.CreateRowVector();
  auto fv = v.FV<double>();
  auto mv = mat.CreateColVector();
  auto  fmv = mv.FV<double>();

  int crow = 0, ccol = 0;
  for (auto k : Range(N)) {
    if (free && !free->Test(k))
      { continue; }
    fv = 0.0; fv(k) = 1.0;
    *mv = mat * (*v);
    crow = 0;
    for (auto j : Range(N)) {
      if (free && !free->Test(j))
        { continue; }
      dmat(crow++, ccol) = fmv(j);
    }
    ccol++;
  }

  return dmat;
}

template Matrix<Complex> MakeDense (BaseMatrix & mat, BitArray* free);
template Matrix<double> MakeDense (BaseMatrix & mat, BitArray* free);

template<class T>
Table<T> CopyTable (const FlatTable<T> & tab_in)
{
  Array<int> perow(tab_in.Size());
  for (auto k : Range(perow))
    { perow[k] = tab_in[k].Size(); }
  Table<T> tab_out(perow);
  for (auto k : Range(tab_out))
    { tab_out[k] = tab_in[k]; }
  return std::move(tab_out);
} // CopyTable

template Table<int> CopyTable (const FlatTable<int> & tab_in);
template Table<double> CopyTable (const FlatTable<double> & tab_in);
template Table<size_t> CopyTable (const FlatTable<size_t> & tab_in);



} // namespace amg
