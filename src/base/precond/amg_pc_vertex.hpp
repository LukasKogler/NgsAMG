#ifndef FILE_AMGPC_VERTEX_HPP
#define FILE_AMGPC_VERTEX_HPP

#include "amg_pc.hpp"

namespace amg
{

/**
 *  Vertex based AMG Preconditioner
 */
template<class AFACTORY>
class VertexAMGPC : public BaseAMGPC
{
public:
  class Options;

  using FACTORY = AFACTORY;

  static constexpr int DIM = FACTORY::DIM;

  using TMESH = typename FACTORY::TMESH;

  template<int BSA>
  static constexpr int BSC() { return (FACTORY::BS > BSA) ? FACTORY::BS - BSA : 1; }

protected:

  using BaseAMGPC::options;
  using BaseAMGPC::strict_alg_mode;

  Array<Array<int>> node_sort;

  bool use_v2d_tab = false;
  Array<int> d2v_array, v2d_array;
  Table<int> v2d_table;
  Array<Array<Vec<DIM,double>>> node_pos;

  shared_ptr<BitArray> free_verts;

public:

  VertexAMGPC (shared_ptr<BilinearForm> blf, Flags const &flags, const string name, shared_ptr<Options> opts = nullptr);

  // strict_alg_mode constructor
  VertexAMGPC (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts = nullptr);

  ~VertexAMGPC ();

  /** In strict alg mode, if vertex positions are required, they must be set after InitializeOptions and before InitLevel **/
  virtual void SetVertexCoordinates(FlatArray<double> coords);

  virtual void SetEmbProjScalRows(shared_ptr<BitArray> scalFree);

  /** BaseAMGPC overloads **/
  virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
  virtual void Update () override { ; } // TODO: what should this do??

protected:

  /** BaseAMGPC overloads **/

  virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
  virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
  virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
  virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

  virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;
  virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;
  virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) override;
  virtual BaseAMGFactory& GetBaseFactory() const override { return GetFactory(); };

  // virtual shared_ptr<BaseAMGFactory> BuildFactory () override;
  virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (BaseAMGFactory::AMGLevel & mesh) override;

  BaseAMGPC::Options::SM_TYPE
  SelectSmoother(BaseAMGFactory::AMGLevel const &amgLevel) const override;

public:
  virtual void RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const override;

protected:
  /** New Methods **/

  FACTORY& GetFactory() const;
  virtual void SetUpMaps ();
  virtual shared_ptr<EQCHierarchy> BuildEQCH ();
  virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h);
  virtual shared_ptr<BlockTM> BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h);
  virtual shared_ptr<BlockTM> BTM_Alg (shared_ptr<EQCHierarchy> eqc_h);

  virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
  virtual shared_ptr<TMESH> BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh);
  template<class TD2V, class TV2D> // for 1 DOF per vertex (also multidim-DOF)
  shared_ptr<TMESH> BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
              TD2V D2V, TV2D V2D) const; // implemented seperately for all AMG_CLASS
  template<class TD2V, class TV2D> // multiple DOFs per vertex (compound spaces)
  shared_ptr<TMESH> BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
                TD2V D2V, TV2D V2D) const; // implemented seperately for all AMG_CLASS
  virtual shared_ptr<TMESH> BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const; // implement seperately (but easy)
  virtual shared_ptr<TMESH> BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh);


  template<int BSA> shared_ptr<BaseDOFMapStep> BuildEmbedding_impl (BaseAMGFactory::LevelCapsule const &cap);
  template<int BSA> shared_ptr<SparseMat<BSA, BSA>> BuildES ();
  template<int BSA> shared_ptr<SparseMat<BSA, FACTORY::BS>> BuildED (size_t height, shared_ptr<TopologicMesh> mesh);

  template<int BSA>
  shared_ptr<SparseMat<BSC<BSA>(), FACTORY::BS>>
  BuildEDC (size_t height, shared_ptr<TopologicMesh> mesh)
  {
    return nullptr;
  }

public:
  mutable shared_ptr<FACTORY> _factory;
  /** Utility for better python wrapping **/
  virtual IVec<3, double> GetElmatEVs() const;
}; // class VertexAMGPC


/**
 * Vertex based AMG Preconditioner
 *  - is aware of Mesh, FESpace, BLF, so it can figure out certain things in many cases:
 *     * embedding from FES to AT canonical
 *     * if needed, vertex positions
 *  - has access to element matrices
 */
template<class FACTORY, class HTVD = double, class HTED = double>
class ElmatVAMG : public VertexAMGPC<FACTORY>
{
public:
  using BASE = VertexAMGPC<FACTORY>;
  using TMESH = typename BASE::TMESH;
  using Options = typename BASE::Options;

  ElmatVAMG (shared_ptr<BilinearForm> blf, Flags const &flags, const string name, shared_ptr<Options> opts = nullptr);

  // strict_alg_mode constructor
  ElmatVAMG (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts = nullptr);

  ~ElmatVAMG ();

  virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h) override;
  using BASE::BTM_Mesh, BASE::BTM_Alg;
  virtual shared_ptr<BlockTM> BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h);
  virtual shared_ptr<TMESH> BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh) override;

  virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
          ElementId ei, LocalHeap & lh) override;

  // Okay, this is nasty, ultimately we should derive this class for all PCs separately,
  // but for now ... whatever
  virtual void CalcAuxWeightsSC (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
          ElementId ei, LocalHeap & lh);
  virtual void CalcAuxWeightsALG (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
          ElementId ei, LocalHeap & lh);
  virtual void CalcAuxWeightsLSQ (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
          ElementId ei, LocalHeap & lh);


  virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
  virtual void FinalizeLevel (shared_ptr<BaseMatrix> mat) override;

  virtual IVec<3, double> GetElmatEVs() const override;

protected:
  using BASE::options;
  unique_ptr<HashTable<int, HTVD>> ht_vertex = nullptr;
  unique_ptr<HashTable<IVec<2,int>, HTED>> ht_edge = nullptr;

  IVec<3, double> elmat_evs;

}; // class EmbedVAMG


/** VertexAMGPCOptions **/

class VertexAMGPCOptions : public BaseAMGPC::Options
{
public:
  /** Which subset of DOFs to perform the COARSENING on?
   *      << This is SEPARATE from which DOFs to SOLVE on! >>
   *      <<   e.g. LO AMG + HO block-smoother             >>
   *  Per default, in "normal" mode, we use ALL dofs as subset,
   *  including DIRICHLET entries which are NOT free.
   **/

  /** "Regular" subsets **/
  enum DOF_SUBSET : unsigned {
    RANGE_SUBSET = 0,        // use Union { [ranges[i][0], ranges[i][1]) }
    SELECTED_SUBSET = 1      // given by a BitArray defining a special subset (e.g. nodalp2/low order)
  };
  DOF_SUBSET subset = RANGE_SUBSET;
  Array<IVec<2, size_t>> ss_ranges; // ranges must be non-overlapping and incresing

  /** If "SELECTED_SUBSET", this **/
  enum SPECIAL_SUBSET : unsigned {
    SPECSS_NONE = 0,         // no special subset
    SPECSS_FREE = 1,         // take free-dofs as subset
    SPECSS_NODALP2 = 2,      // 0..nv, and then first DOF of each edge
    // SPECSS_BA = 3            // a provided custom BitArray [[not implemented]]
  };
  SPECIAL_SUBSET spec_ss = SPECSS_NONE;
  shared_ptr<BitArray> ss_select;

  /** How the DOFs in the subset are mapped to vertices **/
  enum DOF_ORDERING : unsigned {
    REGULAR_ORDERING = 0,
    VARIABLE_ORDERING = 1     // provided assignment [[not implemented]]
  };
  /**	REGULAR: sum(block_s) DOFs per "vertex", defined by block_s and ss_ranges/ss_select
        e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
        each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks)
        subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13],
        but DOF 13 not in subset

      VARIABLE_ORDERING: PLACEHOLDER
  **/
  DOF_ORDERING dof_ordering = REGULAR_ORDERING;
  Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
  Table<int> v_blocks;

  /** How "extra" vertex information (currently: hard-coded position) is specified **/
  enum VERT_SPEC : unsigned {
    NOT_NEEDED      = 0, // no extra info is needed (default!)
    FROM_MESH_NODES = 1, // AMG-vertices are associated to Mesh-Nodes, computed from vertex<->DOF and DOF<->Node
    ALGEBRAIC       = 2  // Vertex-data passed from outside!
  };
  VERT_SPEC vertexSpecification = NOT_NEEDED;

  /** AMG-Vertex <-> Mesh-Node Identification **/
  bool has_node_dofs[4] = { false, false, false, false };
  Array<NodeId> v_nodes;

  /** vertex position given from the outside **/
  Array<Vec<3, double>> algVPositions;

  /** How do we define the topology ? **/
  enum TOPO : unsigned {
    ALG_TOPO = 0,        // by entries of the finest level sparse matrix
    MESH_TOPO = 1,       // via the mesh [[deprecated/non-functional]]
    ELMAT_TOPO = 2       // via element matrices [[deprecated/non-functional]]
  };
  TOPO topo = ALG_TOPO;

  /** How do we compute vertex positions (if we need them) (outdated..) **/
  enum POSITION : unsigned {
    VERTEX_POS = 0,      // take from mesh vertex-positions
    GIVEN_POS = 1        // supplied from outside [[not implemented]]
  };
  POSITION v_pos = VERTEX_POS;
  FlatArray<Vec<3>> v_pos_array;


  /** For projecting out components in case of partial Dirichlet conditions **/
  shared_ptr<BitArray> scalFreeRows = nullptr;

  // compute equivalence between real and auxiliary
  // element matrices
  bool calc_elmat_evs = false;
  int aux_elmat_version = 0;

public:

  VertexAMGPCOptions ()
    : BaseAMGPC::Options()
  { ; }

  virtual void SetFromFlags (shared_ptr<FESpace> fes, shared_ptr<BaseMatrix> finest_mat, const Flags & flags, string prefix);


}; // class VertexAMGPCOptions

/** END VertexAMGPCOptions **/

} // namespace amg

#endif
