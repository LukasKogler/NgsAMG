#ifndef FILE_HDIV_HDG_EMBEDDING
#define FILE_HDIV_HDG_EMBEDDING

#include <stokes_pc.hpp>
#include <base_mesh.hpp>

#include <dof_map.hpp>
#include <hdivhofespace.hpp>
#include <tangentialfacetfespace.hpp>

// #include "preserved_vectors.hpp"
#include "mesh_dofs.hpp"

namespace amg
{

class HDivHDGEmbedding
{
public:
  enum AUX_SPACE
  {
    RTZ = 0,
    P0  = 1,
    P1  = 2
  };

  HDivHDGEmbedding(BaseStokesAMGPrecond const &stokesPre,
                   AUX_SPACE            const &auxSpace);

  // HDivHighOrderFESpace const &GetHDivFESpace () const { return *_hDivSpace; }
  FESpace const &GetFESpace          () const { return *_fes; }
  FESpace const &GetHDivFESpace      () const { return *_hDivSpace; }
  FESpace const *GetTangFacetFESpace () const { return _vFSpace.get(); }

  shared_ptr<MeshDOFs>
  CreateMeshDOFs(shared_ptr<BlockTM> fineMesh);
  
  std::tuple<shared_ptr<SparseMatrix<double>>,
             Array<double>, // facetSurf
             Array<double>, // facetFlow
             Array<shared_ptr<BaseVector>>> // vectors to preserve
  CreateDOFEmbedding(BlockTM const &fMesh,
                     MeshDOFs const &meshDOFs);

  
  Array<shared_ptr<BaseVector>>
  CreateVectorsToPreserve(BlockTM const &fMesh,
                          MeshDOFs const &meshDOFs);

  INLINE int GetLOFacetDOF (int facetNr) const
  {
    Array<int> &fD = hDivFacetDOFs;

    GetHDivFESpace().GetDofNrs(NodeId(NT_FACET, facetNr), fD);

    return _hDivRange.First() + hDivFacetDOFs[0];
  }

protected:

  template<class FACETFE>
  std::tuple<shared_ptr<SparseMatrix<double>>,
             Array<double>, // facetSurf
             Array<double>, // facetFlow
             Array<shared_ptr<BaseVector>>> // vectors to preserve
  CreateDOFEmbeddingImpl(BlockTM const &fMesh,
                         MeshDOFs const &meshDOFs);

  void FindSpaces();

  // to circumvent unimplemented HDiv dual shapes we implement GetFE here
  // and always return an order 1 volume element without hodivfree
  template <ELEMENT_TYPE ET, int FACET_ORDER>
  HDivFiniteElement<ET_trait<ET>::DIM>&
  T_GetHDivFE (int elnr, LocalHeap & lh) const;

  template<int DIM, int FACET_ORDER = 0>
  HDivFiniteElement<DIM>&
  GetHDivFE (ElementId ei, LocalHeap & lh) const;

protected:
  BaseStokesAMGPrecond const & getStokesPre() const { return _stokesPre; }

  AUX_SPACE _auxSpace;

  shared_ptr<FESpace>         _fes;
  BaseStokesAMGPrecond const &_stokesPre;

  // FlatArray<int> element2Vertex;
  // FlatArray<int> facetToEdge;

  // INLINE int EL2V(int k) const { return element2Vertex[k]; }
  // INLINE int F2E(int k)  const { return facetToEdge[k]; }

  int                              _hDivIdx   = -1;
  // shared_ptr<HDivHighOrderFESpace> _hDivSpace = nullptr;
  shared_ptr<FESpace> _hDivSpace = nullptr;
  IntRange                         _hDivRange = IntRange(0,0);

  //
  int                              _vFIdx   = -1;
  // shared_ptr<TangentialFacetFESpace>   _vFSpace = nullptr;
  shared_ptr<FESpace> _vFSpace = nullptr;
  IntRange                         _vFRange = IntRange(0,0);

  mutable Array<int> hDivFacetDOFs; // for use in GetLOFacetDOF
}; // HDivHDGEmbedding


INLINE std::ostream&
operator<<(std::ostream &os, HDivHDGEmbedding::AUX_SPACE const &auxSpace)
{
  switch(auxSpace)
  {
    case(HDivHDGEmbedding::AUX_SPACE::RTZ): { os << "RTZ"; break; }
    case(HDivHDGEmbedding::AUX_SPACE::P0):  { os << "P0";  break; }
    case(HDivHDGEmbedding::AUX_SPACE::P1):  { os << "P1";  break; }
  }
  return os;
}

} // namespace amg

#endif // FILE_HDIV_HDG_EMBEDDING
