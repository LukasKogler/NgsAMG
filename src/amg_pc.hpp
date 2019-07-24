#ifndef FILE_AMGPC_HPP
#define FILE_AMGPC_HPP

namespace amg
{
  /**
     This class handles the conversion from the original FESpace
     to the form that the Vertex-wise AMG-PC needs.

     Virtual class. 

     Needed for different numbering of DOFs or non-nodal base functions, 
     for example:
  	  - compound FESpaces [numbering by component, then node]
  	  - rotational DOFs [more DOFs, the "original" case]

     Implements:
       - Construct topological part of the "Mesh" (BlockTM)
       - Reordering of DOFs and embedding of a VAMG-style vector (0..NV-1, vertex-major blocks)
         into the FESpace.
     Implement in specialization:
       - Construct attached data for finest mesh
   **/
  template<class FACTORY>
  class EmbedVAMG : public Preconditioner
  {
  public:
    using TMESH = typename FACTORY::TMESH;

    struct Options;

    EmbedVAMG (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name = "precond");

    EmbedVAMG (const PDE & apde, const Flags & aflags, const string aname = "precond")
      : Preconditioner(&apde, aflags, aname)
    { throw Exception("PDE-constructor not implemented!"); }

    virtual ~EmbedVAMG () { ; }

    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;

    virtual void FinalizeLevel (const BaseMatrix * mat) override;

    virtual void Update () override { ; };

    virtual const BaseMatrix & GetAMatrix () const override
    { if (finest_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } return *finest_mat; }
    virtual const BaseMatrix & GetMatrix () const override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } return *amg_mat; }
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } return amg_mat; }
    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } amg_mat->Mult(b, x); }
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } amg_mat->MultTrans(b, x); }
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } amg_mat->MultAdd(s, b, x); }
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override
    { if (amg_mat == nullptr) { throw Exception("NGsAMG Preconditioner not ready!"); } amg_mat->MultTransAdd(s, b, x); }

  protected:
    shared_ptr<Options> options;

    shared_ptr<BilinearForm> bfa;

    shared_ptr<BitArray> finest_freedofs, free_verts;
    shared_ptr<BaseMatrix> finest_mat;

    Array<Array<int>> node_sort;
    Array<Array<Vec<3,double>>> node_pos;

    shared_ptr<AMGMatrix> amg_mat;

    shared_ptr<FACTORY> factory;


    /** Configuration **/

    // I) new options II) call SetDefault III) call SetFromFlags IV) call Modify  (modify and default differ by being called before/after setfromflags)
    virtual shared_ptr<Options> MakeOptionsFromFlags (const Flags & flags, string prefix = "ngs_amg_");
    virtual void SetOptionsFromFlags (Options& O, const Flags & flags, string prefix = "ngs_amg_");

    // I do not want to derive from this class, instead I hook up specific options here. dummy-implementation in impl-header, specialize in cpp
    virtual void SetDefaultOptions (Options& O);
    virtual void ModifyOptions (Options & O, const Flags & flags, string prefix = "ngs_amg_");
    

    /** AMG-Mesh; includes topologic and algebraic components **/

    virtual shared_ptr<TMESH> BuildInitialMesh ();


    /** AMG-Mesh; topologic components **/

    virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h); // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h); // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BTM_Alg (shared_ptr<EQCHierarchy> eqc_h); // implemented once for all AMG_CLASS


    /** AMG-Mesh; algebraic components **/

    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);

    // mappings implemented once - hands maps to impl-method which must be implemented for each AMG_CLASS 
    virtual shared_ptr<TMESH> BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh);
    template<class TD2V, class TV2D> // for 1 DOF per vertex (also multidim-DOF)
    shared_ptr<TMESH> BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
					     TD2V D2V, TV2D V2D) const; // implemented seperately for all AMG_CLASS
    // template<class TD2V, class TV2D> // multiple DOFs per vertex (compound spaces)
    // shared_ptr<TMESH> BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
    // 					    TD2V D2V, TV2D V2D) const; // implemented seperately for all AMG_CLASS

    virtual shared_ptr<TMESH> BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh); // implement seperately (but easy)

    // template<class TD2V, class TV2D>
    // shared_ptr<TMESH> BuildAlgMesh_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
    // 					TD2V D2V, TV2D V2D) const; // implemented seperately for all AMG_CLASS


    /** Smoothers **/

    virtual shared_ptr<BaseSmoother> BuildSmoother (shared_ptr<BaseSparseMatrix> m, shared_ptr<ParallelDofs> pds,
						    shared_ptr<BitArray> freedofs = nullptr) const;


    /** other stuff  **/

    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding ();

    virtual shared_ptr<BaseSparseMatrix> RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat,
							   shared_ptr<ParallelDofs> & pardofs) const;

    shared_ptr<FACTORY> BuildFactory (shared_ptr<TMESH> mesh);
    virtual void Finalize ();
    virtual void BuildAMGMat ();

  };


  /**
     basically just overloads addelementmatrix
   **/
  template<class FACTORY, class HTVD = double, class HTED = double>
  class EmbedWithElmats: public EmbedVAMG<FACTORY>
  {
  public:
    using BASE = EmbedVAMG<FACTORY>;
    using TMESH = typename BASE::TMESH;
    
    EmbedWithElmats (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name = "precond");

    EmbedWithElmats (const PDE & apde, const Flags & aflags, const string aname = "precond")
      : BASE(apde, aflags, aname)
    { throw Exception("PDE-constructor not implemented!"); }

    ~EmbedWithElmats ();

    virtual shared_ptr<BlockTM> BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h) override;

    virtual shared_ptr<BlockTM> BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h);

    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh) override;

    virtual shared_ptr<TMESH> BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh);

    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh) override;
  protected:
    using BASE::options;

    HashTable<int, HTVD> * ht_vertex;
    HashTable<INT<2,int>, HTED> * ht_edge;
  };

};

#endif // FILE_AMGPC_HPP
