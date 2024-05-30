#include <base.hpp>
#include <base_coarse.hpp>

#include <utils_io.hpp>

namespace amg
{

// grid_map.hpp header has no cpp file
std::ostream & operator<<(std::ostream &os, const BaseGridMapStep& p)
{
  p.PrintTo(os);
  return os;
}

template<NODE_TYPE NT> FlatTable<int> BaseCoarseMap :: GetMapC2F () const
{
  auto & t = rev_node_maps[NT];
  /** Set up table only once. **/
  if (t.Size() != mapped_NN[NT]) {
    FlatArray<int> map = node_maps[NT];
    TableCreator<int> ct(mapped_NN[NT]);
    for (; !ct.Done(); ct++)
        for (auto k : Range(map))
        if (map[k] != -1)
            { ct.Add(map[k], k); }
    t = ct.MoveTable();
  }
  return t;
}

template FlatTable<int> BaseCoarseMap::GetMapC2F<NT_VERTEX> () const;
template FlatTable<int> BaseCoarseMap::GetMapC2F<NT_EDGE> () const;
template FlatTable<int> BaseCoarseMap::GetMapC2F<NT_FACE> () const;
template FlatTable<int> BaseCoarseMap::GetMapC2F<NT_CELL> () const;


shared_ptr<BaseCoarseMap> BaseCoarseMap :: Concatenate (shared_ptr<BaseCoarseMap> right_map)
{
  auto cmap = make_shared<BaseCoarseMap>(this->mesh, right_map->mapped_mesh);
  SetConcedMap(right_map, cmap);
  return cmap;
} // BaseCoarseMap::Concatenate


void BaseCoarseMap :: PrintTo (std::ostream & os, string prefix) const
{
  os << prefix << "BaseCoarseMap, F NN = " << NN[0] << " " << NN[1] << " " << NN[2] << " " << NN[3] << endl;
  os << prefix << "BaseCoarseMap, C NN = " << mapped_NN[0] << " " << mapped_NN[1] << " " << mapped_NN[2] << " " << mapped_NN[3] << endl;

  os << endl << prefix << " F->C NODE_MAP FOR NT_VERTEX = " << NT_VERTEX << endl;
  prow3(node_maps[NT_VERTEX], os, "   ", 20);
  os << endl;
  auto cfv = GetMapC2F<NT_VERTEX>();
  os << endl << prefix << " C->F NODE_MAP FOR NT_VERTEX = " << NT_VERTEX << endl << cfv << endl;

  os << endl << prefix << " F->C NODE_MAP FOR NT_EDGE = " << NT_EDGE << endl;
  prow3(node_maps[NT_EDGE], os, "   ", 20);
  os << endl;
  auto cfe = GetMapC2F<NT_EDGE>();
  os << endl << prefix << " C->F NODE_MAP FOR NT_EDGE = " << NT_EDGE << endl << cfe << endl;
} // BaseCoarseMap::PrintTo


void BaseCoarseMap :: SetConcedMap (shared_ptr<BaseCoarseMap> right_map, shared_ptr<BaseCoarseMap> cmap)
{
  for ( NODE_TYPE NT : { NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL } ) {
    cmap->NN[NT] = this->NN[NT];
    cmap->mapped_NN[NT] = right_map->mapped_NN[NT];
    FlatArray<int> lmap = this->node_maps[NT], rmap = right_map->node_maps[NT];
    Array<int> & cnm = cmap->node_maps[NT];
    cnm.SetSize(this->NN[NT]);
    // if (NT == NT_VERTEX) {
      // cout << "conc, lmap = "; prow2(lmap); cout << endl;
      // cout << "conc, rmap = "; prow2(rmap); cout << endl;
    // }
    for (auto k : Range(this->NN[NT])) {
      auto midnum = lmap[k];
      cnm[k] = (midnum == -1) ? -1 : rmap[midnum];
      // if (NT == NT_VERTEX)
      // cout << k << "->" << midnum << "->" << cnm[k] << endl;
    }
  }
} // BaseCoarseMap::SetConecMap

} // namespace amg