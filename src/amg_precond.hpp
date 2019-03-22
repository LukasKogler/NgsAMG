#ifndef FILE_AMGPC
#define FILE_AMGPC

namespace amg
{

  /**
     Vertex-wise AMG preconditioner. Abstract base-class.

     This class is used when DOF-numbering is 0..ndof-1, 
     with one DOF per "vertex" (so multidim-fespace style).
     
     This is per construction always the case for coarse levels. For the finest
     level we have to make a difference. See EmbeddedAMGPC for that!

     Implements:
       - coarse-level-loop
       - assembly loop for tentative prolongations
     CRTP for:
       - mesh-type to be used (for attached data)
       - scalar/block-scalar types
       - prolongation-kernels
       - replacement-matrix kernels
     Pure virtual methods:
       - Collapse
   **/
  template<class AMG_CLASS, class ATMESH, class TMAT>
  class VWiseAMG : public BaseMatrix
  {
  public:
    struct Options
    {
      int max_n_levels = 20;
      size_t max_n_verts = 1;
      /** Dirichlet conditions for finest level **/
      shared_ptr<BitArray> free_verts = nullptr;
      shared_ptr<BitArray> finest_free_dofs = nullptr;
    };

    using TMESH = ATMESH;
    using TV = typename mat_traits<TMAT>::TV_ROW;
    using TSCAL = typename mat_traits<TMAT>::TSCAL;
    using TSPMAT = SparseMatrix<TMAT, TV, TV>;
    // typedef SparseMatrix<TMAT, TV, TV> TSPMAT;

    VWiseAMG (shared_ptr<TMESH> finest_mesh, shared_ptr<Options> opts) : options(opts), mesh(finest_mesh) { ;  };
    /** the first prolongation is concatenated with embed_step (it should be provided by EmbedAMGPC) **/
    void Finalize (shared_ptr<BaseMatrix> fine_mat, shared_ptr<BaseDOFMapStep> embed_step = nullptr);

    INLINE void Mult (const BaseVector & b, BaseVector & x) const { amg_mat->Mult(b,x); }

    // CRTP FOR THIS!!
    // INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap & map,
    // 			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, TMAT & mat);

    size_t GetNLevels(int rank) const
    {return this->amg_mat->GetNLevels(rank); }
    void GetBF(size_t level, int rank, size_t dof, BaseVector & vec) const
    {this->amg_mat->GetBF(level, rank, dof, vec); }
    size_t GetNDof(size_t level, int rank) const
    { return this->amg_mat->GetNDof(level, rank); }
    
  protected:
    string name = "VWiseAMG";
    shared_ptr<Options> options;
    shared_ptr<AMGMatrix> amg_mat;
    shared_ptr<TMESH> mesh;
    shared_ptr<BaseMatrix> finest_mat;
    shared_ptr<BaseDOFMapStep> embed_step;

    virtual void SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts, INT<3> level, shared_ptr<TMESH> mesh) = 0;

    virtual shared_ptr<CoarseMap<TMESH>> TryCoarsen  (INT<3> level, shared_ptr<TMESH> mesh);
    virtual shared_ptr<GridContractMap<TMESH>> TryContract (INT<3> level, shared_ptr<TMESH> mesh);
    virtual shared_ptr<BaseGridMapStep> TryDiscard  (INT<3> level, shared_ptr<TMESH> mesh) { return nullptr; }
    
    void Setup ();

    INLINE void SetIdentity (double & x) const { x = 1.0; }
    template<int D> INLINE void SetIdentity (Mat<D,D,double> & x) const { x = 0.0; for (auto i:Range(D)) x(i,i) = 1.0; }
    
    shared_ptr<ParallelDofs> BuildParDofs (shared_ptr<TMESH> amesh);
    shared_ptr<ProlMap<TSPMAT>> BuildDOFMapStep (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd);
    shared_ptr<CtrMap<TV>> BuildDOFMapStep (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd);
  };

  // /**
  //    AMG for Elasticity.

  //    Works with Vec<3, double> as scalar.
  // **/
  // template<int D>
  // class ElastAMG : public VWiseAMG<ElastAMG<D>>
  // {
  // public:
  //   static constexpr int disppv (int dim)
  //   { return dim; }
  //   static constexpr int rotpv (int dim)
  //   { return dim*(dim-1)/2; }
  //   static constexpr int dofpv (int n)
  //   { return disppv(n)+rotpv(n); }
  //   using double = TSCAL;
  //   using Vec<dofpv(D), TSCAL> = TV;
  //   using Mat<dofpv(D), dofpv(D), TSCAL> = TMAT;
  //   using BlockTM = TMESH;
  // protected:
    
  // };



  /**
     This class handles the conversion from the original FESpace
     to the form that the Vertex-wise AMG-PC needs.

     Virtual class. 

     Needed for different numbering of DOFs or non-nodal base functions, 
     for example:
          - TDNNS [non-nodal; "vertices" from faces]
  	  - compound FESpaces [numbering by component, then node]
  	  - rotational DOFs [more DOFs, the "original" case]

     Implements:
       - Construct topological part of the "Mesh" (BlockTM)
       - Reordering of DOFs and embedding of a VAMG-style vector (0..NV-1, vertex-major blocks)
         into the FESpace.
     Pure virtual:
       - Construct attached data for finest mesh
  	  - Construct 2-level matrix
  	  - AddElementMatrix
   **/
  template<class AMG_CLASS>
  class EmbedVAMG : public Preconditioner
  {
  public:
    struct Options : AMG_CLASS::Options
    {
      /** nr of vertices **/
      size_t n_verts = 0;
      /** v_dofs:
	    "NODAL" -> sum(block_s) dofs per "vertex", determined by on_dofs+block_s
	    "VARIABLE" -> dofs for vertex k: v_blocks[k] (need for 3d TDNNS)
       **/
      string v_dofs = "NODAL";
      shared_ptr<BitArray> on_dofs = nullptr; FlatArray<int> block_s;
      FlatTable<int> v_blocks;
      /** v_pos: 
	    "VERTEX", "FACE" -> use node pos
	    "GIVEN" -> positions in v_pos_array **/
      string v_pos = "VERTEX"; FlatArray<Vec<3>> v_pos_array;
      /** energy: 
	    "ELMAT" -> calc from elmats, use ext_blf if given, else blf
	    "ALGEB" -> determine algebraically (not implemented properly)
	    "TRIV" -> use 1 weights everywhere **/
      string energy = "TRIV"; shared_ptr<BilinearForm> ext_blf = nullptr; shared_ptr<BitArray> elmat_dofs = nullptr;
      /** kvecs: 
	    "TRIV" -> dofs in each block have to stand for: have to stand for: trans_x/y/z(+ rot_x/y/z if rot-dofs)
	    "VEC" -> kernel_vecs have to be trans_x/y/z, rot_x/y/z **/
      string kvecs = "TRIV"; FlatArray<shared_ptr<BaseVector>> kernel_vecs;
      /** edges: 
	    "ELMAT", "ELMAT_FULL" -> calc from elmats, FULL->all-to-all
	    "MESH" -> take from Mesh
	    "ALG" -> calc from FEM-Matrix **/
      string edges = "MESH";
    };
    using TMESH = typename AMG_CLASS::TMESH;

    EmbedVAMG (shared_ptr<BilinearForm> blf, shared_ptr<Options> opts);
    ~EmbedVAMG () { ; }

    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    { amg_pc->Mult(b, x); }
    virtual const BaseMatrix & GetAMatrix() const override
    { return *finest_mat; }
    virtual int VHeight() const override { return finest_mat->VHeight(); }
    virtual int VWidth() const override { return finest_mat->VWidth();}

    virtual void FinalizeLevel (const BaseMatrix * mat) override;
    virtual void Update () override { ; };
    virtual void Setup ();

    void MyTest () const
    {
      cout << IM(1) << "Compute eigenvalues" << endl;
      const BaseMatrix & amat = GetAMatrix();
      const BaseMatrix & pre = GetMatrix();

      auto v = amat.CreateVector();
      int eigenretval;

      EigenSystem eigen (amat, pre);
      eigen.SetPrecision(1e-30);
      eigen.SetMaxSteps(100); 
        
      eigen.SetPrecision(1e-15);
      eigenretval = eigen.Calc();
      eigen.PrintEigenValues (*testout);
      cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
      cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      (*testout) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      (*testout) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
        
      if(testresult_ok) *testresult_ok = eigenretval;
      if(testresult_min) *testresult_min = eigen.EigenValue(1);
      if(testresult_max) *testresult_max = eigen.MaxEigenValue();
        
        
      //    (*testout) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      //    for (int i = 1; i < min2 (10, eigen.NumEigenValues()); i++)
      //      cout << "cond(i) = " << eigen.MaxEigenValue() / eigen.EigenValue(i) << endl;
      (*testout) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl;
        
    }

    size_t GetNLevels(int rank) const {return this->amg_pc->GetNLevels(rank); }
    void GetBF(size_t level, int rank, size_t dof, BaseVector & vec) const {this->amg_pc->GetBF(level, rank, dof, vec); }
    size_t GetNDof(size_t level, int rank) const { return this->amg_pc->GetNDof(level, rank); }
    
    shared_ptr<Options> options;
  protected:
    shared_ptr<BilinearForm> bfa;
    shared_ptr<FESpace> fes;
    shared_ptr<AMG_CLASS> amg_pc;
    shared_ptr<BaseMatrix> finest_mat = nullptr;
    // shared_ptr<BaseMatrix> embed_mat;

    Array<Array<int>> node_sort;
    
    // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BuildTopMesh ();
    // implemented seperately for all AMG_CLASS
    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
    virtual shared_ptr<TMESH> BuildInitialMesh () { return BuildAlgMesh(BuildTopMesh()); }
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding ();
  };

  
} // namespace amg

#endif
