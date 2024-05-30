#ifndef FILE_GRID_MAP_HPP
#define FILE_GRID_MAP_HPP

#include <base.hpp>

#include <base_mesh.hpp>

namespace amg
{

class BaseGridMapStep
{
protected:
  shared_ptr<TopologicMesh> mesh, mapped_mesh;
public:
  BaseGridMapStep (shared_ptr<TopologicMesh> _mesh, shared_ptr<TopologicMesh> _mapped_mesh = nullptr)
    : mesh(_mesh), mapped_mesh(_mapped_mesh)
  { ; }
  virtual ~BaseGridMapStep () { ; }
  virtual shared_ptr<TopologicMesh> GetMesh () const { return mesh; }
  virtual shared_ptr<TopologicMesh> GetMappedMesh () const { return mapped_mesh; }
  virtual void CleanupMeshes ()
  {
    mesh = nullptr;
    mapped_mesh = nullptr;
  }

  virtual void PrintTo (std::ostream & os, string prefix = "") const { os << prefix << "BaseGridMapStep!" << endl; }

}; // class BaseGridMapStep

std::ostream & operator<<(std::ostream &os, const BaseGridMapStep& p);

/** This maps meshes and their NODES between levels. **/
class GridMap
{
public:
  void AddStep (shared_ptr<BaseGridMapStep> step) { steps.Append(step); }
  shared_ptr<BaseGridMapStep> GetStep (size_t nr) { return steps[nr]; }
  void CleanupStep (int level) { steps[level] = nullptr; }
  auto begin() const { return steps.begin(); }
  auto end() const { return steps.end(); }
private:
  Array<shared_ptr<BaseGridMapStep>> steps;
}; // class GridMap
  
} // namespace amg

#endif // FILE_GRID_MAP_HPP
