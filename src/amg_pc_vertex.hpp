#ifndef FILE_AMGPC_VERTEX_HPP
#define FILE_AMGPC_VERTEX_HPP

namespace amg
{

  /** **/
  template<class FACTORY>
  class VertexAMGPC : public BaseAMGPC
  {
  public:
    struct Options;

    using FACTORY::DIM;
    using TMESH = FACTORY::TMESH;

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

    virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
    virtual void SetOptionsFromFlags (Options& O, const Flags & flags, string prefix = "ngs_amg_") override;

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;

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

    virtual shared_ptr<BaseAMGFactory> BuildFactory () override;
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) override;

    virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;
  }; // class VertexAMGPC


  /** VertexAMGPC + element-matrices **/
  template<class FACTORY, class HTVD = double, class HTED = double>
  class ElmatVAMG : public VertexAMGPC<FACTORY>
  {
  public:
    using BASE = VertexAMGPC<FACTORY>;
    struct Options;

    ElmatVAMG (const PDE & apde, const Flags & aflags, const string aname = "precond");
    ElmatVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~VertexAMGPC ();

    virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h) override;
    virtual shared_ptr<BlockTM> BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h);

    void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
			   ElementId ei, LocalHeap & lh) override;

  protected:
    using BASE::options;
    HashTable<int, HTVD> * ht_vertex;
    HashTable<INT<2,int>, HTED> * ht_edge;
  }; // class EmbedVAMG


} // namespace amg

#endif
