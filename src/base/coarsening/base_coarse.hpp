#ifndef FILE_AMGCRS
#define FILE_AMGCRS

#include <base.hpp>
#include "grid_map.hpp"
#include <SpecOpt.hpp>

namespace amg
{

class BaseCoarseMap : public BaseGridMapStep
{
public:
  BaseCoarseMap (shared_ptr<TopologicMesh> mesh, shared_ptr<TopologicMesh> mapped_mesh = nullptr)
    : BaseGridMapStep(mesh, mapped_mesh)
  { ; } //{ NN = 0; mapped_NN = 0; }

  virtual ~BaseCoarseMap () { ; }

  template<NODE_TYPE NT>
  INLINE size_t GetNN () const { return NN[NT]; }
  
  template<NODE_TYPE NT>
  INLINE size_t GetMappedNN () const { return mapped_NN[NT]; }
  
  template<NODE_TYPE NT>
  INLINE FlatArray<int> GetMap () const { return node_maps[NT]; }
  
  template<NODE_TYPE NT>
  FlatTable<int> GetMapC2F () const;
  
  virtual shared_ptr<BaseCoarseMap> Concatenate (shared_ptr<BaseCoarseMap> right_map);

  virtual void PrintTo (std::ostream & os, string prefix) const override;

protected:
  void SetConcedMap (shared_ptr<BaseCoarseMap> right_map, shared_ptr<BaseCoarseMap> cmap);

  Array<Array<int>> node_maps = Array<Array<int>> (4);
  Array<Table<int>> rev_node_maps = Array<Table<int>>(4);
  size_t NN[4] = {0,0,0,0};
  size_t mapped_NN[4] = {0,0,0,0};
}; // class BaseCoarseMap

} // namespace amg

#endif
