#ifndef FILE_STOKES_MESH_HPP
#define FILE_STOKES_MESH_HPP

#include <base.hpp>
#include <universal_dofs.hpp>
#include <alg_mesh.hpp>

#include "facet_aux_info.hpp"

namespace amg
{

class BaseCoarseMap;

/**
 *  Extends a Mesh with:
 *      I) facet-loops needed for Hiptmair smoother.
 *     II) info on wether vertices are "ghosts" (mirrors of verts on neib procs)
 *         ghost_vert guarantees:
 *          -) for every vertex, there is exactly one proc where ghost_vert[v]==false
 *          -) if ghost_vert[v]==false, all edges connected to v are available locally
 */
template<class... T>
class StokesMesh : public BlockAlgMesh<T...>
{
  // using THIS_CLASS = StokesMesh<T...>;
public:

  StokesMesh (shared_ptr<EQCHierarchy> &eqc_h)
    : BlockAlgMesh<T...>(eqc_h)
  { ; }

  StokesMesh (BlockTM && _mesh, T*... _data)
    : BlockAlgMesh<T...>(std::move(_mesh), _data...)
  { ; }

  StokesMesh (BlockTM && _mesh, tuple<T*...> _data)
    : BlockAlgMesh<T...>(std::move(_mesh), _data)
  { ; }

  ~StokesMesh () = default;

  virtual void printTo(std::ostream &os) const override;

  /** loops **/

  /**
   * Adds loops that are locally sorted and oriented lists of edges to the mesh.
   * Only procs that have a DOFed edge in a loop have any loop.
   *     -> loopers == loop_procs!!
   * Input requirements:
   *   I) loops are ordered AND ORIENTED consistently among procs that have them
   *  II) every loop is present on EXACTLY the loopers
   * 
   * Also tracks mapping (input loop) -> (final loop)
   * 
   * Notes:
   *    i) The input may contain empty/deflated loops, these are removed, set_lsort(loop_num, -1)
   *       is called.
   *   ii) Globally non-deflated loops can be LOCALLY EMPTY on one or more loopers.
   *            e.g.: Say that, after coarsening, all LOCAL vertices a loop traverses are merged,
   *                  the connecting edges vanish. That loop is deflated LOCALLY, but could still
   *                  exist on one or more of the other loopers!
   *       In such cases, these loops ARE REMOVED from procs where they
   *       are deflated and set_lsort(loop_num, -1) is called
   */
  template<class TLAM>
  INLINE void AddOrientedLoops (FlatTable<int> loops, FlatTable<int> dist_loopers, TLAM set_lsort);

  /**
   *  For use with block-smoothers:
   *    i) One block per COARSE edge, containing all FINE loops touching that edge
   *   ii) One block per fine loop that vanishes on the coarse level
   *             (Should this better be one block per coarse vertex??) 
   * 
   *  TODO: this should get vmap,emap, etc. as input so we don't need to include the
   *        BaseCoarseMap header (in the _impl header)
   */
  Table<int> LoopBlocks (const BaseCoarseMap & cmap) const;

  /** DOFed edges **/

  void SetUpDOFedEdges ();

  /**
   * TODO: This dofed-edge stuff (NC) and MeshDOFs (HDiv) have essentially the same role,
   *       they should be unified somehow, or made into a template parameter so we don't
   *       have to set up all this duplicate info everywhere...
   * Returns:
   *     i) dofed_edges BitArray
   *    ii) dofed_edge -> edge
   *   iii) edge       -> dofed_edge
  */
  tuple<shared_ptr<BitArray>, FlatArray<int>, FlatArray<int>> GetDOFedEdges () const;
  
  /** setters/getters/utils **/

  void SetGhostVerts  (shared_ptr<BitArray> _ghost_verts)  { ghost_verts = _ghost_verts; }
  void SetLoops       (Table<int> && _loops)               { loops = std::move(_loops); }
  void SetActiveLoops (shared_ptr<BitArray> _active_loops) { active_loops = _active_loops; }
  void SetLoopDPs     (Table<int> && _loop_dps);

  shared_ptr<BitArray>        GetGhostVerts     ()       const { return ghost_verts; }
  shared_ptr<BitArray>        GetActiveLoops    ()       const { return active_loops; }
  FlatTable<int>              GetLoops          ()       const { return loops; }
  UniversalDofs        const& GetLoopUDofs      ()       const { return loopUDofs; }
  UniversalDofs        const& GetDofedEdgeUDofs (int BS) const;

  INLINE bool IsGhostVert (int vnr) { return (ghost_verts == nullptr) ? false : ghost_verts->Test(vnr); }


  // virtual void MapAdditionalData (const BaseGridMapStep & amap) override
  // {
  //   // TERRIBLE (!!), but I really don't feel like thinking of something better ...
  //   if (auto ctr_map = dynamic_cast<const GridContractMap<THIS_CLASS>*>(&amap))
  //     { MapAdditionalData_impl(*ctr_map); }
  //   else if (auto crs_map = dynamic_cast<const BaseCoarseMap*>(&amap))
  //     { MapAdditionalData_impl(*crs_map); }
  //   else
  //     { throw Exception(string("Not Map for ") + typeid(amap).name()); }
  // }

  // void MapAdditionalData_impl (const GridContractMap<THIS_CLASS> & cmap);

  // void MapAdditionalData_impl (const BaseCoarseMap & cmap);

protected:
  /** solid/ghost vertices **/
  shared_ptr<BitArray> ghost_verts;

  /** loops **/
  shared_ptr<BitArray> active_loops; // if nullptr, all loops are active

  /**
   * Entries in this table are either (1 + enr) or -(1 + enr),
   * depending on orientation of the edge in the loop. 
   */
  Table<int> loops;
  UniversalDofs loopUDofs;

  /**
   *  DOFed edge == edge with locally associated DOF
   *  I.E. edges with at least one solid vertex 
   */
  Array<unique_ptr<UniversalDofs>> dofedEdgeUDofs; // we want this to be able to be a nullptr
  shared_ptr<BitArray> dofed_edges;
  Array<int> dof2e, e2dof;
}; // class StokesMesh


enum FVType : char {
  BOUNDARY    = 0, // introduced on boundary
  NO_BOUNDARY = 1, // introduced on a mesh-boundary without surface elements (-> no bnd index!)
};

/**
 *  Description of fictitious vertices, the "code" is:
 *    i) BOUNDARY:    bnd_index
 *    2) NO_BOUNDARY: max. bnd_index of the mesh + 1
*/
struct FVDescriptor
{
  FVType type;
  int code;
};

INLINE ostream & operator<< (ostream &os, FVDescriptor const &fvd)
{
  if (fvd.type == BOUNDARY)
    { os << "[BND, code<" << fvd.code << "> -> bnd_idx " << fvd.code << "]"; }
  else
    { os << "[NO_BND, code<" << fvd.code << ">]"; }
  return os;
}


/**
 * Returns:
 *     i) the BlockTM
 *    ii) ghost-vertex BitArray
 *   iii) element -> vertex map
 *    iv) facet   -> edge map
 *     V) facet   -> fict. vertex map
*/
tuple<shared_ptr<BlockTM>, Array<FVDescriptor>, Array<int>, Array<int>, Array<int>>
BuildStokesMesh(MeshAccess const &ma, FacetAuxiliaryInformation const &auxInfo);


/** This may be better put somewhere else, but whatever... **/
template<int ADIM, class ATVD>
struct StokesVData
{
  static constexpr int DIM = ADIM;
  using TVD = ATVD;
  TVD vd;
  double vol;                  // if positive, the volume. if negative, the vertex is fictitious [[added for non-diri boundary facets]]
  INLINE bool IsReal  () const { return vol > 0;  }     // a regular vertex that stands for a volume
  INLINE bool IsImag  () const { return vol < 0;  }     // an imaginary vertex, appended for boundary facets
  INLINE bool IsWeird () const { return vol == 0; }   // a temporary vertex, usually from CalcMPData
  StokesVData (double val) : vd(val), vol(val) { ; }
  StokesVData () : StokesVData (0) { ; }
  StokesVData (TVD _vd, double _vol) : vd(_vd), vol(_vol) { ; }
  StokesVData (TVD && _vd, double && _vol) : vd(std::move(_vd)), vol(std::move(_vol)) { ; }
  StokesVData (StokesVData<DIM, TVD> && other) : vd(std::move(other.vd)), vol(std::move(other.vol)) { ; }
  StokesVData (const StokesVData<DIM, TVD> & other) : vd(other.vd), vol(other.vol) { ; }
  INLINE void operator = (double x) { vd = x; vol = x; }
  INLINE void operator = (const StokesVData<DIM, TVD> & other) { vd = other.vd; vol = other.vol; }
  INLINE void operator += (const StokesVData<DIM, TVD> & other) { vd += other.vd; vol += other.vol; }
  INLINE bool operator == (const StokesVData<DIM, TVD> & other) { return (vd == other.vd) && (vol == other.vol); }
  // workaround because the H1-V-data is actually a INT<2, double> currently because I was experimenting with
  // absorbed-vertex-counter
  template<class TINDEX> INLINE double operator[](const TINDEX &i) const { return vd; }
}; // struct StokesVData

template<int DIM, class TVD> INLINE std::ostream & operator << (std::ostream & os, StokesVData<DIM, TVD> & v)
{ os << "[vol:" << v.vol << " | wt:" << v.vd << "]"; return os; }

template<int DIM, class TVD> INLINE bool is_zero (const StokesVData<DIM, TVD> & vd) { return is_zero(vd.vd) && is_zero(vd.vol); }


template<class ATVD>
class AttachedSVD : public AttachedNodeData<NT_VERTEX, ATVD>
{
public:
  using TVD = ATVD;
  static constexpr int DIM = TVD::DIM;
  using BASE = AttachedNodeData<NT_VERTEX, ATVD>;
  using BASE::mesh;
  using BASE::data;

  AttachedSVD (Array<ATVD> && _data, PARALLEL_STATUS stat)
    : BASE(std::move(_data), stat)
  { ; }

  void map_data (const BaseCoarseMap & cmap, AttachedSVD<TVD> *cevd) const;
}; // class AttachedSVD

} // namespace amg



namespace ngcore
{

  // ATM, only using TVD=double, if I need anything else I just have to specialize the template there

  template<class SVD>
  struct MPI_typetrait_StokesVData {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
      {
        int block_len[2] = { 1, 1 };
        MPI_Aint displs[2] = { 0, sizeof(typename SVD::TVD) };
        MPI_Datatype types[2] = { GetMPIType<typename SVD::TVD>(), GetMPIType<double>() };
        MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
        MPI_Type_commit ( &MPI_T );
      }
      return MPI_T;
    }
  }; // struct MPI_typetrait_StokesVData

  template<> struct MPI_typetrait<amg::StokesVData<2, double>> {
    static MPI_Datatype MPIType () {
      return MPI_typetrait_StokesVData<amg::StokesVData<2, double>>::MPIType();
    }
  }; // struct MPI_typetrait

  template<> struct MPI_typetrait<amg::StokesVData<3, double>> {
    static MPI_Datatype MPIType () {
      return MPI_typetrait_StokesVData<amg::StokesVData<3, double>>::MPIType();
    }
  }; // struct MPI_typetrait

} // namespace ngcore

#endif // FILE_STOKES_MESH_HPP