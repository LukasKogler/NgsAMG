#ifndef FILE_AMGPC_VERTEX_HPP
#define FILE_AMGPC_VERTEX_HPP

#include "amg_pc.hpp"

namespace amg
{

  /** **/
  template<class AFACTORY>
  class VertexAMGPC : public BaseAMGPC
  {
  public:
    class Options;

    using FACTORY = AFACTORY;

    static constexpr int DIM = FACTORY::DIM;

    using TMESH = typename FACTORY::TMESH;

  protected:

    using BaseAMGPC::options;

    Array<Array<int>> node_sort;

    bool use_v2d_tab = false;
    Array<int> d2v_array, v2d_array;
    Table<int> v2d_table;
    Array<Array<Vec<DIM,double>>> node_pos;

    shared_ptr<BitArray> free_verts;

  public:

    VertexAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond");
    VertexAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~VertexAMGPC ();

  protected:

    /** BaseAMGPC overloads **/
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
    virtual void Update () override { ; } // TODO: what should this do??

    virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
    virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
    virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
    virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;
    virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;
    virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) override;
    virtual shared_ptr<BaseAMGFactory> BuildFactory () override;
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) override;

    virtual void RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const override;

    /** New Methods **/

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

    template<int BSA> shared_ptr<BaseDOFMapStep> BuildEmbedding_impl (shared_ptr<TopologicMesh> mesh);
    template<int BSA> shared_ptr<stripped_spm_tm<Mat<BSA, BSA, double>>> BuildES ();
    template<int BSA> shared_ptr<stripped_spm_tm<Mat<BSA, FACTORY::BS, double>>> BuildED (size_t height, shared_ptr<TopologicMesh> mesh);

  }; // class VertexAMGPC


  /** VertexAMGPC + element-matrices **/
  template<class FACTORY, class HTVD = double, class HTED = double>
  class ElmatVAMG : public VertexAMGPC<FACTORY>
  {
  public:
    using BASE = VertexAMGPC<FACTORY>;
    class Options;

    ElmatVAMG (const PDE & apde, const Flags & aflags, const string aname = "precond");
    ElmatVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~ElmatVAMG ();

    virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h) override;
    using BASE::BTM_Mesh, BASE::BTM_Alg;
    virtual shared_ptr<BlockTM> BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h);

    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh) override;

  protected:
    using BASE::options;
    HashTable<int, HTVD> * ht_vertex = nullptr;
    HashTable<INT<2,int>, HTED> * ht_edge = nullptr;
  }; // class EmbedVAMG


  /** VertexAMGPCOptions **/

  class VertexAMGPCOptions : public BaseAMGPC::Options
  {
  public:
    /** Which subset of DOFs to perform the coarsening on **/
    enum DOF_SUBSET : char { RANGE_SUBSET = 0,        // use Union { [ranges[i][0], ranges[i][1]) }
			     SELECTED_SUBSET = 1 };   // given by bitarray
    DOF_SUBSET subset = RANGE_SUBSET;
    Array<INT<2, size_t>> ss_ranges; // ranges must be non-overlapping and incresing
    /** special subsets **/
    enum SPECIAL_SUBSET : char { SPECSS_NONE = 0,
				 SPECSS_FREE = 1,            // take free-dofs as subset
				 SPECSS_NODALP2 = 2 };       // 0..nv, and then first DOF of each edge
    SPECIAL_SUBSET spec_ss = SPECSS_NONE;
    shared_ptr<BitArray> ss_select;
    
    /** How the DOFs in the subset are mapped to vertices **/
    enum DOF_ORDERING : char { REGULAR_ORDERING = 0,
			       VARIABLE_ORDERING = 1 };
    /**	REGULAR: sum(block_s) DOFs per "vertex", defined by block_s and ss_ranges/ss_select
	   e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
	   each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks) 
	subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13], but DOF 13 not in subet
	   
	VARIABLE: PLACEHOLDER !! || DOFs for vertex k: v_blocks[k] || ignores subset
    **/
    DOF_ORDERING dof_ordering = REGULAR_ORDERING;
    Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
    Table<int> v_blocks;

    /** AMG-Vertex <-> Mesh-Node Identification **/
    bool store_v_nodes = false;
    bool has_node_dofs[4] = { false, false, false, false };
    Array<NodeId> v_nodes;

    /** How do we define the topology ? **/
    enum TOPO : char { ALG_TOPO = 0,        // by entries of the finest level sparse matrix
		       MESH_TOPO = 1,       // via the mesh
		       ELMAT_TOPO = 2 };    // via element matrices
    TOPO topo = ALG_TOPO;

    /** How do we compute vertex positions (if we need them) (outdated..) **/
    enum POSITION : char { VERTEX_POS = 0,    // take from mesh vertex-positions
			   GIVEN_POS = 1 };   // supplied from outside
    POSITION v_pos = VERTEX_POS;
    FlatArray<Vec<3>> v_pos_array;

    /** How do we compute the replacement matrix **/
    enum ENERGY : char { TRIV_ENERGY = 0,     // uniform weights
			 ALG_ENERGY = 1,      // from the sparse matrix
			 ELMAT_ENERGY = 2 };  // from element matrices
    ENERGY energy = ALG_ENERGY;

  public:
    
    VertexAMGPCOptions ()
      : BaseAMGPC::Options()
    { ; }

    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix);

      
  }; // class VertexAMGPCOptions

  /** END VertexAMGPCOptions **/

} // namespace amg

#endif
