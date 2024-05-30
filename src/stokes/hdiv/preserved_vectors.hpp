#ifndef FILE_PRESERVED_VECTORS_HPP
#define FILE_PRESERVED_VECTORS_HPP

#include "mesh_dofs.hpp"

namespace amg
{

/**
 * This class does nothing now, We could (or should??) remove it.
 * Its only use is in the verbosity of having it called what it is.
 */
class PreservedVectors
{
public:
  PreservedVectors(int const &nSpecial,
                   Array<shared_ptr<BaseVector>> &&vectors)
    : _nSpecial(nSpecial)
    , _vectors(vectors)
  {
    ;
  }

  INLINE int GetNSpecial()   const { return _nSpecial; }
  INLINE int GetNPreserved() const { return _vectors.Size(); }

  INLINE BaseVector const& GetVector(int k) const { return *_vectors[k]; }
  INLINE BaseVector      & GetVector(int k)       { return *_vectors[k]; }

  INLINE shared_ptr<BaseVector> GetVectorPtr(int k) const { return _vectors[k]; }

private:
  int const                     _nSpecial;
  Array<shared_ptr<BaseVector>> _vectors;
}; // class PreservedVectors


template<class TMESH>
class PreservedVectorsMap
{
public:
  PreservedVectorsMap (AgglomerateCoarseMap<TMESH> const &cmap,
                       MeshDOFs                    const &meshDofs,
                       PreservedVectors            const &preservedVectors);

  int computeCFBufferSize (FlatArray<int> bufferOffsets);

  template<class TLAM>
  INLINE void computeCFProlBlocks (FlatArray<int>     bufferOffsets,
                                   FlatArray<double>  buffer,
                                   TLAM               computeSpecialVecs,
                                   LocalHeap         &lh);

  INLINE FlatMatrix<double> ReadProlBlockFromBuffer (FlatArray<int>            bufferOffsets,
                                                     FlatArray<double>         buffer,
                                                     int                const &cENr);
                        
  tuple<shared_ptr<MeshDOFs>, shared_ptr<PreservedVectors>>
    Finalize (FlatArray<int>    bufferOffsets,
              FlatArray<double> buffer);

private:
  INLINE PreservedVectors const &GetFinePreserved () const { return _fineVecs; }
  INLINE MeshDOFs         const &GetFineMeshDOFs  () const { return _fineMeshDOFs; }

  int computeCoarseBasis(int enr, FlatMatrix<double> prol);

  Array<unique_ptr<BaseVector>> finalizeCoarseVecs ();

  AgglomerateCoarseMap<TMESH> const &_cmap;
  TMESH                       const &_fMesh;
  TMESH                       const &_cMesh;
  MeshDOFs                    const &_fineMeshDOFs;
  PreservedVectors            const &_fineVecs;

  shared_ptr<MeshDOFs>         _coarseMeshDOFs;
  shared_ptr<PreservedVectors> _coarseVecs;

  Array<int> _dofsPerEdge; // # of DOFs per edge on the coarse level
  Array<unique_ptr<BaseVector>> _cVecs; // un-finalized coarse coords of preserved vectors,
}; // class PreservedVectorsMap

} // namespace amg

#endif // FILE_PRESERVED_VECTORS_HPP