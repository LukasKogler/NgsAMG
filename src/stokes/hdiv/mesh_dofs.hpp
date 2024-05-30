#ifndef FILE_MESH_DOFS_HPP
#define FILE_MESH_DOFS_HPP

#include <base.hpp>

namespace amg
{


/**
 *  Provides mapping between DOFs and edges 
 */
class MeshDOFs
{
public:
  MeshDOFs(std::shared_ptr<BlockTM> blockMesh)
    : _blockMesh(blockMesh)
  {
  }

  /** Setup variant 1: provide finished offsets from outside **/
  void SetOffsets (Array<int> &&offsets)
  {
    _dofOffsets = std::move(offsets);
  }

  FlatArray<size_t> GetOffsets () const { return _dofOffsets; }

  /** Setup variant 2: initialize empty offsets, increments dofs per edge, call finalize **/

  void Initialize ()
  {
    _dofOffsets.SetSize(GetMesh().template GetNN<NT_EDGE>() + 1);
    _dofOffsets = 0;
  }

  void AddDOF (size_t const &nodeId)
  {
    _dofOffsets[nodeId + 1]++;
  }
 
  void AddDOF (AMG_Node<NT_EDGE> const &node)
  {
    AddDOF(node.id);
  }

  void SetNDOF (size_t const &nodeId, int const &nDOFs)
  {
    _dofOffsets[nodeId + 1] = nDOFs;
  }

  void SetNDOF (AMG_Node<NT_EDGE> const &node, int const &nDOFs)
  {
    SetNDOF(node.id, nDOFs);
  }

  void Finalize ()
  {
    for (auto k : Range(GetMesh().template GetNN<NT_EDGE>()))
    {
      _dofOffsets[k + 1] += _dofOffsets[k];
    }
  }

  /** After either of the setup paths, these here can be called **/

  INLINE size_t GetNDOF () const
  {
    return _dofOffsets.Last();
  }

  INLINE IntRange GetDOFNrs (size_t const &nodeId) const
  {
    return Range(_dofOffsets[nodeId], _dofOffsets[nodeId + 1]);
  }

  INLINE IntRange GetDOFNrs (AMG_Node<NT_EDGE> const &node) const
  {
    return GetDOFNrs(node.id);
  }

  INLINE size_t GetNDOF (size_t const &nodeId) const
  {
    return _dofOffsets[nodeId + 1] - _dofOffsets[nodeId];
  }

  INLINE size_t GetNDOF (AMG_Node<NT_EDGE> const &node) const
  {
    return GetNDOF(node.id);
  }

  INLINE size_t EdgeToDOF (size_t const &nodeId, int const &locNum) const
  {
    return _dofOffsets[nodeId] + locNum;
  }

  INLINE size_t EdgeToDOF (AMG_Node<NT_EDGE> const &node, int const &locNum) const
  {
    return EdgeToDOF(node.id, locNum);
  }
  
  INLINE size_t DOFToEdge (size_t globNum) const
  {
    return merge_pos_in_sorted_array(globNum, _dofOffsets) - 1;
  }

  // returns count, offsets, dof-numbers for subset of edges
  INLINE tuple<int, FlatArray<int>, FlatArray<int>>
  SubDOFs(FlatArray<int> edgeNums,
          LocalHeap &lh) const
  {
    FlatArray<int> offsets(edgeNums.Size() + 1, lh);
    offsets[0] = 0;
    for (auto k : Range(edgeNums))
      { offsets[k + 1] = offsets[k] + GetNDOF(edgeNums[k]); }
    FlatArray<int> dOFNums(offsets.Last(), lh);
    for (auto k : Range(edgeNums))
      { dOFNums.Range(offsets[k], offsets[k + 1]) = GetDOFNrs(edgeNums[k]); }
    return make_tuple(offsets.Last(), offsets, dOFNums);    
  }

  template<class TMESH>
  INLINE
  void
  PrintAs(std::ostream &os,
          std::string const &offset = "") const
  {
    auto const &M = static_cast<TMESH const&>(GetMesh());
    std::string const of2 = offset + "  ";
    os << offset << " MeshDOFs on mesh " << _blockMesh << endl;
    os << of2 << " Mesh has " << GetMesh().template GetNN<NT_VERTEX>() << " vertices and " << GetMesh().template GetNN<NT_EDGE>() << " edges. " << endl;
    os << of2 << " Total #dofs = " << GetNDOF() << endl;
    // auto [dOFEdges, DOFE2E, E2DOFE] = M.GetDOFedEdges();
    auto [dOFEdges_SB, DOFE2E_SB, E2DOFE_SB] = M.GetDOFedEdges();
    auto &dOFEdges = dOFEdges_SB;
    auto &DOFE2E = DOFE2E_SB;
    auto &E2DOFE = E2DOFE_SB;
    GetMesh().template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &edge) {
      os << of2 << " " << GetNDOF(edge) << " dofs for edge edge " << edge
                << ", DOFed ? " << dOFEdges->Test(edge.id) << ", dofed-nr " << E2DOFE[edge.id]
                << ": [" << _dofOffsets[edge.id] << ", " << _dofOffsets[edge.id+1] << ")" << endl; 
    }, false);
    os << of2 << " " << endl;
  }

  void
  PrintTo(std::ostream &os,
          std::string const &offset = "") const
  {
    std::string const of2 = offset + "  ";
    os << offset << " MeshDOFs on mesh " << _blockMesh << endl;
    os << of2 << " Mesh has " << GetMesh().template GetNN<NT_VERTEX>() << " vertices and " << GetMesh().template GetNN<NT_EDGE>() << " edges. " << endl;
    os << of2 << " Total #dofs = " << GetNDOF() << endl;
    GetMesh().template ApplyEQ<NT_EDGE>([&](auto const &eqc, auto const &edge) {
      os << of2 << " " << GetNDOF(edge) << " dofs for edge edge " << edge
                << ": [" << _dofOffsets[edge.id] << ", " << _dofOffsets[edge.id+1] << ")" << endl; 
    }, false);
    os << of2 << " " << endl;
  }

private:
  INLINE BlockTM const & GetMesh () const { return *_blockMesh; }

  shared_ptr<BlockTM> _blockMesh;
  Array<size_t>       _dofOffsets;
}; // class MeshDOFs

INLINE std::ostream& operator<<(std::ostream &os, MeshDOFs const &meshDOFs)
{
  meshDOFs.PrintTo(os);
  return os;
}


} // namespace amg

#endif // FILE_MESH_DOFS_HPP