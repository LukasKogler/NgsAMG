#define FILE_AMGH1_CPP
#define FILE_AMGH1_CPP_DIM1

#include "h1.hpp"
#include <amg_pc_vertex.hpp>

#include "h1_impl.hpp"
#include "h1_energy_impl.hpp"
#include <amg_pc_vertex_impl.hpp>

#include <amg_register.hpp>
// #define AMG_EXTERN_TEMPLATES
// #include "amg_tcs.hpp"
// #undef AMG_EXTERN_TEMPLATES

#include "plate_test_agg_impl.hpp"

namespace amg
{

template<> shared_ptr<H1Mesh> ElmatVAMG<H1AMGFactory<1>, double, double> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
{
  auto & O(static_cast<Options&>(*options));

  // if ( (this->ht_vertex == nullptr) || (this->ht_edge == nullptr) )
    // { throw Exception("elmat-energy, but have to HTs! (HOW)"); }
  if ( (ht_vertex == nullptr) || (ht_edge == nullptr) )
    { throw Exception("elmat-energy, but have to HTs! (HOW)"); }

  auto a = new H1VData(Array<IVec<2,double>>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED);
  auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);

  FlatArray<int> vsort = node_sort[NT_VERTEX];
  Array<int> rvsort(vsort.Size());
  for (auto k : Range(vsort.Size()))
    rvsort[vsort[k]] = k;
  auto ad = a->Data();
  for (auto key_val : *ht_vertex) {
    ad[rvsort[get<0>(key_val)]] = IVec<2, double>(get<1>(key_val), 1.0);
  }
  auto bd = b->Data();
  auto edges = top_mesh->GetNodes<NT_EDGE>();
  for (auto & e : edges) {
    bd[e.id] = (*ht_edge)[IVec<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
  }

  auto mesh = make_shared<H1Mesh>(std::move(*top_mesh), a, b);

  // probably can delete the hash-tables now
  ht_vertex.reset(nullptr);
  ht_edge.reset(nullptr);
  return mesh;
} // VertexAMGPC<H1AMGFactory, double, double>::BuildAlgMesh

using T_MESH           = H1AMGFactory<1>::TMESH;
using T_ENERGY         = H1AMGFactory<1>::ENERGY;
using T_MESH_WITH_DATA = typename T_MESH::T_MESH_W_DATA;

extern template class SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;
extern template class MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;

extern template class DiscreteAgglomerateCoarseMap<T_MESH, SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;
extern template class DiscreteAgglomerateCoarseMap<T_MESH, MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;

template class H1AMGFactory<1>;
template class VertexAMGPC<H1AMGFactory<1>>;
template class ElmatVAMG<H1AMGFactory<1>, double, double>;

template class PlateTestAgglomerator<T_MESH_WITH_DATA>;
template class DiscreteAgglomerateCoarseMap<T_MESH, PlateTestAgglomerator<T_MESH_WITH_DATA>>;

using PCCBASE = VertexAMGPC<H1AMGFactory<1>>;
using PCC = ElmatVAMG<H1AMGFactory<1>, double, double>;

// RegisterPreconditioner<PCC> register_h1amg_1d ("ATAmg.h1_scal");
RegisterAMGSolver<PCC> register_h1amg_1d ("h1_scal");

} // namespace amg

