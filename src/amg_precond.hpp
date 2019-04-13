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
       - assembly loop for tentative & smoothed prolongations
     CRTP for:
       - scalar/block-scalar types
       - prolongation-kernels
       - replacement-matrix kernels
     Virtual methods:
       - set coarsening options
       - regularize matrix
   **/
  template<class AMG_CLASS, class ATMESH, class TMAT>
  class VWiseAMG : public BaseMatrix
  {
    // TODO: do I need ATMESH? could take that from AMG_CLASS
  public:

    using TMESH = ATMESH;
    using TV = typename mat_traits<TMAT>::TV_ROW;
    using TSCAL = typename mat_traits<TMAT>::TSCAL;
    using TSPMAT = SparseMatrix<TMAT, TV, TV>;

    struct Options
    {
      /** Dirichlet conditions for finest level **/
      shared_ptr<BitArray> free_verts = nullptr;
      shared_ptr<BitArray> finest_free_dofs = nullptr;
      /** Coarsening **/
      double min_ecw = 0.05, min_vcw = 0.5;
      /** Level-control **/
      int max_n_levels = 20;               // maximum number of coarsening steps
      size_t max_n_verts = 1;              // stop coarsening when the coarsest mesh has this few vertices
      int skip_ass_first = 2;              // skip this many levels in the beginning
      Array<int> ass_levels;               // force matrix assembly on these levels
      Array<int> ass_skip_levels;          // forbid matrix assembly on these levels
      bool force_ass = false;              // force assembly ONLY on ass_levels
      double ass_after_frac = 0.15;        // assemble a level after reducing NV by this factor
      /** Discard - only dummies for not!!  **/
      bool enable_disc = false;            // enable node-discarding
      double disc_crs_thresh = 0.7;        // try discard if coarsening becomes worse than this
      double disc_fac_ok     = 0.95;       // accept discard map if we reduce by at least this
      /** Contract (Re-Distribute) **/
      bool enable_ctr = true;              // enable re-distributing
      double ctr_after_frac = 0.05;        // re-distribute after we have reduced the NV by this factor
      double ctr_crs_thresh = 0.7;         // if coarsening slows down more than this, ctract
      double ctr_pfac = 0.25;              // contract proc-factor (overruled by min NV per proc)
      size_t ctr_min_nv = 500;             // re-distribute such that at least this many NV per proc remain
      size_t ctr_seq_nv = 500;             // re-distribute to sequential once NV reached this threshhold
      /** Prolongation smoothing **/
      bool enable_sm = true;               // emable prolongation-smoothing
      double min_prol_frac = 0.1;          // min. (relative) wt to include an edge
      int max_per_row = 3;                 // maximum entries per row (should be >= 2!)
      double sp_omega = 0.5;               // relaxation parameter for prol-smoothing
      int skip_smooth_first = 3;           // do this many piecewise prols in the beginning
      double smooth_after_frac = 0.5;      // smooth a prol after reducing NV by this factor
      Array<int> sm_levels;                // force prol-smoothing on these levels
      Array<int> sm_skip_levels;           // forbid prol-smoothing on these levels
      bool force_sm = false;               // force smoothing on exactyle sm_levels
      /** Smoothers - haha, you have no choice  **/
      /** Coarsest level opts **/
      string clev_type = "INV"; // available: "INV", "NOTHING"
      string clev_inv_type = "masterinverse";
      /** Wether we keep track of info **/
      INFO_LEVEL info_level = NONE;
    };

    
    struct Info {
      /** Only values on rank 0 are true, others can be garbage !! **/
      INFO_LEVEL ilev;
      // BASIC+ // summary values
      Array<INT<3>> lvs;                   // levels [CRS,CTR,DISC]
      Array<int> isass;                    // is level assembled?
      Array<size_t> NVs;                   // NR of vertices per level
      double v_comp;                       // vertex-complexity: \frac{sum_l NV_l}{NV_0}
      Array<double> vcc;                   // vertex-complexity components
      double op_comp;                      // operator-complexity: \frac{sum_l NZE_l}{NZE_0}
      Array<double> occ;                   // oeprator-complexity components
      // DETAILED+ // per-level info
      Array<size_t> NEs;                   // NR of edges per level
      Array<size_t> NPs;                   // #active procs per level
      double mem_comp1;                    // memory-complexity, V1:  \frac{sum_l MMAT_l}{ MMAT_0 }
      Array<double> mcc1;                  // ...
      double mem_comp2;                    // memory-complexity, V1:  \frac{sum_l MSM_l}{ MMAT_0 }
      Array<double> mcc2;                  // ...
      // EXTRA+ // per-level local info
      double v_comp_l;                     // max. local v-comp
      Array<double> vcc_l;                 // ...
      double op_comp_l;                    // max. local op-comp
      Array<double> occ_l;                 // ...
      double mem_comp1_l;                  // ...
      Array<double> mcc1_l;                // memory-complexity components
      double mem_comp2_l;                  // ...
      Array<double> mcc2_l;                // memory-complexity components
      Array<double> fvloc;                 // fraction of local vertices

      bool has_comm;
      NgsAMG_Comm glob_comm;
    public:
      Info (INFO_LEVEL ailev, size_t asize) : ilev(ailev) {
	has_comm = false;
	auto alloc = [asize](auto & array) {
	  array.SetSize(asize); array.SetSize0();
	};
	if (ilev >= BASIC)
	  { alloc(lvs); alloc(isass); alloc(NVs); alloc(vcc); alloc(occ); }
	if (ilev >= DETAILED)
	  { alloc(mcc1); alloc(mcc2); alloc(NEs); alloc(NPs); }
	if (ilev >= EXTRA)
	  { alloc(vcc_l); alloc(occ_l); alloc(mcc1_l); alloc(mcc2_l); }
      }

      void LogMesh (INT<3> level, shared_ptr<TMESH> & amesh, bool assit) {
	if (ilev == NONE) return;
	auto comm = amesh->GetEQCHierarchy()->GetCommunicator();
	if(!has_comm) { has_comm = true; glob_comm = comm; }
	if (comm.Rank() == 0) {
	  lvs.Append(level);
	  isass.Append(assit?1:0);
	  NVs.Append(amesh->template GetNNGlobal<NT_VERTEX>());
	}
	if (ilev <= BASIC) return;
	if (comm.Rank() == 0) {
	  NEs.Append(amesh->template GetNNGlobal<NT_EDGE>());
	  NPs.Append(comm.Size());
	}
	if (ilev <= DETAILED) return;
	vcc_l.Append(amesh->template GetNN<NT_EDGE>());
	size_t locnv_l = amesh->template GetENN<NT_VERTEX>(0);
	size_t locnv_g = comm.Reduce(locnv_l, MPI_SUM, 0);
	if (comm.Rank()==0) fvloc.Append( locnv_g/double(NVs.Last()) );
      }
      
      void LogMatSm (shared_ptr<BaseSparseMatrix> & amat, shared_ptr<BaseSmoother> & sm) {
	if (ilev == NONE) return;
	auto comm = sm->GetParallelDofs()->GetCommunicator();
	size_t ocmat_l = amat->NZE()*mat_traits<TMAT>::HEIGHT*mat_traits<TMAT>::WIDTH;
	size_t ocmat_g = comm.Reduce(ocmat_l, MPI_SUM, 0);
	if (comm.Rank()==0) occ.Append(ocmat_g);
	if (ilev <= BASIC) return;
	// cast to TSPMAT because "error: member 'GetMemoryUsage' found in multiple base classes of different types"
	size_t nbts_mat_l = 0; for (auto & mu : static_cast<TSPMAT*>(amat.get())->GetMemoryUsage()) nbts_mat_l += mu.NBytes();
	size_t nbts_mat_g = comm.Reduce(nbts_mat_l, MPI_SUM, 0);
	size_t nbts_sm_l = 0; for (auto & mu : sm->GetMemoryUsage()) nbts_sm_l += mu.NBytes();
	size_t nbts_sm_g = comm.Reduce(nbts_sm_l, MPI_SUM, 0);
	if (comm.Rank()==0) { mcc1.Append(nbts_mat_g); mcc2.Append(nbts_sm_g); }
	if (ilev <= DETAILED ) return;
	occ_l.Append(ocmat_l);
	mcc1_l.Append(nbts_mat_l);
	mcc2_l.Append(nbts_sm_l);
      }
      
      void Finalize() {
	if (ilev == NONE) return;
	static Timer t("Info::Finalize"); RegionTimer rt(t);
	int n_meshes = NVs.Size(), n_mats = occ.Size(); // == 0 if not master
	auto lam_max = [&](auto & val, auto & arr) {
	  auto val2 = glob_comm.AllReduce(val, MPI_MAX);
	  int mrk = (val==val2) ? glob_comm.Rank() : glob_comm.Size()+1;
	  int sender = glob_comm.AllReduce(mrk, MPI_MIN);
	  if (sender != 0) {
	    if (glob_comm.Rank()==sender) { glob_comm.Send(arr, 0, MPI_TAG_AMG); }
	    else if (glob_comm.Rank()==0) { val = val2; glob_comm.Recv(arr, sender, MPI_TAG_AMG); }
	  }
	};
	// VC
	v_comp = 0;
	for (auto k : Range(n_meshes))
	  if (isass[k]==1) { double val = double(NVs[k])/NVs[0]; v_comp += val; vcc.Append(val); }
	// OC
	op_comp = 0; double oc0 = occ[0];
	for (auto k : Range(n_mats))
	  { auto v = occ[k]/oc0; op_comp += v; occ[k] = v; }
	if (ilev <= BASIC) return;
	// MC-1 & MC-2
	mem_comp1 = 0; mem_comp2 = 0; double mm0 = mcc1[0];
	for (auto k : Range(n_mats))
	  {
	    auto v1 = mcc1[k]/mm0; mem_comp1 += v1; mcc1[k] = v1;
	    auto v2 = mcc2[k]/mm0; mem_comp2 += v2; mcc2[k] = v2;
	  }
	if (ilev <= DETAILED ) return;
	n_meshes = vcc_l.Size(); n_mats = occ_l.Size();
	// VC-L
	v_comp_l = 0; double vcl0 = max2(1.0, vcc_l[0]);
	for (auto k : Range(n_meshes))
	  if (isass[k]==1) { double val = vcc_l[k]/vcl0; v_comp_l += val; vcc_l[k] = val; }
	lam_max(v_comp_l, vcc_l);
	// OC-L
	op_comp_l = 0; double ocl0 = max2(1.0, occ_l[0]);
	for (auto k : Range(n_mats))
	  { auto v = occ_l[k]/ocl0; op_comp_l += v; occ_l[k] = v; }
	lam_max(op_comp_l, occ_l);
	// MC-1-L & MC-2-L
	mem_comp1_l = 0; mem_comp2_l = 0; double mml0 = max2(1.0, mcc1_l[0]);
	for (auto k : Range(n_mats))
	  {
	    auto v1 = mcc1_l[k]/mml0; mem_comp1_l += v1; mcc1_l[k] = v1;
	    auto v2 = mcc2_l[k]/mml0; mem_comp2_l += v2; mcc2_l[k] = v2;
	  }
	lam_max(mem_comp1_l, mcc1_l);
	lam_max(mem_comp2_l, mcc2_l);
      }

    };

    VWiseAMG (shared_ptr<TMESH> finest_mesh, shared_ptr<Options> opts) : options(opts), mesh(finest_mesh) { ; };
    /** the first prolongation is concatenated with embed_step (it should be provided by EmbedAMGPC) **/
    void Finalize (shared_ptr<BaseMatrix> fine_mat, shared_ptr<BaseDOFMapStep> embed_step = nullptr);

    INLINE void Mult (const BaseVector & b, BaseVector & x) const { amg_mat->Mult(b,x); }

    // CRTP FOR THIS //
    // INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap & map,
    // 			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, TMAT & mat);
    // CRTP FOR THIS // calculate an edge-contribution to the replacement matrix
    // INLINE void CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<double> mat) const { mat = -1; }
    // CRTP FOR THIS // get weight for edge (used for s-prol)
    // template<NODE_TYPE NT> INLINE double GetWeight (const TMESH & mesh, const AMG_Node<NT> * edge) const

    size_t GetNLevels(int rank) const
    {return this->amg_mat->GetNLevels(rank); }
    void GetBF(size_t level, int rank, size_t dof, BaseVector & vec) const
    {this->amg_mat->GetBF(level, rank, dof, vec); }
    size_t GetNDof(size_t level, int rank) const
    { return this->amg_mat->GetNDof(level, rank); }
    shared_ptr<Info> GetInfo () const { return infos; }

  protected:
    string name = "VWiseAMG";
    shared_ptr<Options> options;
    shared_ptr<AMGMatrix> amg_mat;
    shared_ptr<TMESH> mesh;
    shared_ptr<BaseMatrix> finest_mat;
    shared_ptr<BaseDOFMapStep> embed_step;
    double ctr_factor = -1;
    shared_ptr<Info> infos = nullptr;
    
    virtual void SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts, INT<3> level, shared_ptr<TMESH> mesh);
    virtual shared_ptr<CoarseMap<TMESH>> TryCoarsen  (INT<3> level, shared_ptr<TMESH> mesh);
    virtual shared_ptr<GridContractMap<TMESH>> TryContract (INT<3> level, shared_ptr<TMESH> mesh);
    virtual shared_ptr<BaseGridMapStep> TryDiscard  (INT<3> level, shared_ptr<TMESH> mesh) { return nullptr; }
    virtual shared_ptr<BaseSmoother> BuildSmoother  (INT<3> level, shared_ptr<BaseSparseMatrix> mat,
						     shared_ptr<ParallelDofs> par_dofs,
						     shared_ptr<BitArray> free_dofs) = 0;
    virtual void SmoothProlongation (shared_ptr<ProlMap<TSPMAT>> pmap, shared_ptr<TMESH> mesh) const;
    virtual shared_ptr<TSPMAT> RegularizeMatrix (shared_ptr<TSPMAT> mat, shared_ptr<ParallelDofs> & pardofs) { return mat; }
    void Setup ();
    
    shared_ptr<ParallelDofs> BuildParDofs (shared_ptr<TMESH> amesh);
    shared_ptr<ProlMap<TSPMAT>> BuildDOFMapStep (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd);
    shared_ptr<CtrMap<TV>> BuildDOFMapStep (shared_ptr<GridContractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd);

  };



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
     Implement in specialization:
       - Construct attached data for finest mesh
  	  - Construct 2-level matrix (?)
  	  - AddElementMatrix
   **/
  template<class AMG_CLASS, class HTVD = double, class HTED = double>
  class EmbedVAMG : public Preconditioner
  {
  public:
    struct Options : AMG_CLASS::Options
    {
      /** nr of vertices **/
      size_t n_verts = 0;
      /** v_dofs:
	    "NODAL" -> sum(block_s) dofs per "vertex", determined by on_dofs+block_s
	    "VARIABLE" -> dofs for vertex k: v_blocks[k] (need for 3d TDNNS) **/
      string v_dofs = "NODAL";
      shared_ptr<BitArray> on_dofs = nullptr; Array<int> block_s;
      Table<int> v_blocks;
      /** v_pos: 
	    "VERTEX", "FACE" -> use node pos
	    "GIVEN" -> positions in v_pos_array **/
      string v_pos = "VERTEX"; FlatArray<Vec<3>> v_pos_array;
      bool keep_vp = false; // save vertex position
      /** energy: 
	    "ELMAT" -> calc from elmats, use ext_blf if given, else blf (not back yet)
	    "ALGEB" -> determine algebraically (not implemented)
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
    // virtual ~EmbedVAMG ();
    ~EmbedVAMG ();

    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    { amg_pc->Mult(b, x); }
    virtual const BaseMatrix & GetAMatrix() const override
    { return *finest_mat; }
    virtual int VHeight() const override { return finest_mat->VHeight(); }
    virtual int VWidth() const override { return finest_mat->VWidth();}

    virtual void FinalizeLevel (const BaseMatrix * mat) override;
    virtual void Update () override { ; };
    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh) override;

    shared_ptr<typename AMG_CLASS::Info> GetInfo () const { return amg_pc->GetInfo(); }

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
      (*testout) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl;
    }

    size_t GetNLevels(int rank) const {return this->amg_pc->GetNLevels(rank); }
    void GetBF(size_t level, int rank, size_t dof, BaseVector & vec) const {this->amg_pc->GetBF(level, rank, dof, vec); }
    size_t GetNDof(size_t level, int rank) const { return this->amg_pc->GetNDof(level, rank); }
    
    shared_ptr<Options> options;
  protected:
    string name = "EmbedVAMG";
    shared_ptr<BilinearForm> bfa;
    shared_ptr<FESpace> fes;
    shared_ptr<AMG_CLASS> amg_pc;
    shared_ptr<BaseMatrix> finest_mat = nullptr;
    // shared_ptr<BaseMatrix> embed_mat;

    Array<Array<int>> node_sort;
    Array<Array<Vec<3,double>>> node_pos;

    HashTable<int, HTVD> * ht_vertex;
    HashTable<INT<2,int>, HTED> * ht_edge;
    
    // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BuildTopMesh ();
    // implemented seperately for all AMG_CLASS
    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
    virtual shared_ptr<TMESH> BuildInitialMesh () { return BuildAlgMesh(BuildTopMesh()); }
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding ();
  };

  
} // namespace amg

#endif
