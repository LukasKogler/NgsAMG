#ifndef FILE_PLATE_TEST_AGG_IMPL_HPP
#define FILE_PLATE_TEST_AGG_IMPL_HPP

#include "plate_test_agg.hpp"

#include <utils.hpp>
#include <utils_arrays_tables.hpp>

namespace amg
{

template<class T>
INLINE IVec<2, double> getVDPos(int const &v, T const &vd)
{
  return IVec<2, double>({vd.pos[0], vd.pos[1]});
}

template<> INLINE IVec<2, double> getVDPos<double>(int const &v, double const &vd)                 { return IVec<2, double>({ double(v), double(v) }); }
template<> INLINE IVec<2, double> getVDPos<IVec<2, double>>(int const &v, IVec<2, double> const &vd) { return IVec<2, double>({ double(v), double(v) }); }

template<class TMESH>
void PlateTestAgglomerator<TMESH> ::
FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  const auto & M = this->GetMesh();

  M.CumulateData();

  auto vdata = get<0>(M.AttachedData())->Data();
  auto free_verts = this->GetFreeVerts();

  v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;

  auto const &econ = *M.GetEdgeCM();

  auto isFree = [&](auto v) {
    return ( ( free_verts == nullptr ) || ( free_verts->Test(v) ) ) &&
           ( v_to_agg[v] == -1 );
  };

  auto getPos = [&](auto v) {
    return getVDPos(v, vdata[v]);
  };

  agglomerates.SetSize(v_to_agg.Size() / 2);
  agglomerates.SetSize0();

  Array<int> newMems(100);

  int cntAggs = 0;
  M.template ApplyEQ2<NT_VERTEX>([&](auto eq, auto eqVs)
  {
    for (auto vs : eqVs)
    {
      if ( isFree(vs) )
      {
        int aggId = cntAggs++;
        bool foundNew = true;

        agglomerates.Append(Agglomerate(vs, aggId));
        Agglomerate &agg(agglomerates.Last());

        auto const posZ = getPos(vs);

        while(foundNew)
        {
          newMems.SetSize0();
          for (auto mem : agg.members())
          {
            for (auto neib : econ.GetRowIndices(mem))
            {
              if ( isFree(neib) && (getPos(neib) == posZ) )
              {
                v_to_agg[neib] = aggId;
                insert_into_sorted_array_nodups(neib, newMems);            }
              }
          }
          for (auto newMem : newMems)
          {
            agg.AddSort(newMem);
          }
          foundNew = newMems.Size() > 0;
        }
      }
    }
  }, false);

}

} // namespace amg

#endif // FILE_PLATE_TEST_AGG_IMPL_HPP