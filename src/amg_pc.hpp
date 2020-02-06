#ifndef FILE_AMGPC_HPP
#define FILE_AMGPC_HPP

#include "amg.hpp"
#include "amg_map.hpp"
#include "amg_factory.hpp"
#include "amg_smoother.hpp"
#include "amg_matrix.hpp"

namespace amg
{
  /**
     Base class AMG Preconditioner
   **/
  class BaseAMGPC : public Preconditioner
  {
  public:
    
    class Options;

  protected:
    shared_ptr<Options> options;

    shared_ptr<BilinearForm> bfa;

    shared_ptr<BaseAMGFactory> factory;
    shared_ptr<BitArray> finest_freedofs;
    shared_ptr<BaseMatrix> finest_mat;
    shared_ptr<AMGMatrix> amg_mat;

  public:

    /** Constructors **/
    BaseAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond");
    BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~BaseAMGPC ();

    /** Preconditioner overloads **/
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
    virtual void FinalizeLevel (const BaseMatrix * mat) override;

    /** BaseMatrix overloads **/
    virtual const BaseMatrix & GetAMatrix () const override;
    virtual const BaseMatrix & GetMatrix () const override;
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override;
    virtual shared_ptr<AMGMatrix> GetAMGMatrix () const;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;

  protected:
    
    /** Options: construct, set default, set from flags, modify **/
    shared_ptr<Options> MakeOptionsFromFlags (const Flags & flags, string prefix = "ngs_amg_");
    virtual shared_ptr<Options> NewOpts () = 0;
    virtual void SetDefaultOptions (Options& O);
    virtual void SetOptionsFromFlags (Options& O, const Flags & flags, string prefix = "ngs_amg_");
    virtual void ModifyOptions (Options & O, const Flags & flags, string prefix = "ngs_amg_");

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () = 0;
    virtual shared_ptr<BaseAMGFactory> BuildFactory () = 0;
    
    virtual void Finalize ();
    virtual void BuildAMGMat ();

    virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level);
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) = 0;

    virtual shared_ptr<BaseSmoother> BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level);

    virtual shared_ptr<BaseSmoother> BuildGSSmoother (shared_ptr<BaseSparseMatrix> spm, shared_ptr<ParallelDofs> pardofs,
						      shared_ptr<EQCHierarchy> eqc_h, shared_ptr<BitArray> freedofs = nullptr);
    virtual shared_ptr<BitArray> GetFreeDofs (const BaseAMGFactory::AMGLevel & amg_level);

    virtual shared_ptr<BaseSmoother> BuildBGSSmoother (shared_ptr<BaseSparseMatrix> spm, shared_ptr<ParallelDofs> pardofs,
						       shared_ptr<EQCHierarchy> eqc_h, Table<int> && blocks);
    virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level);

    virtual void RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const;

  }; // BaseAMGPC


  /** Options **/

  class BaseAMGPC :: Options
  {
  public:

    /** What we do on the coarsest level **/
    enum CLEVEL : char { INV_CLEV = 0,       // invert coarsest level
			 SMOOTH_CLEV = 1,    // smooth coarsest level
			 NO_CLEV = 2 };
    CLEVEL clev = INV_CLEV;
    INVERSETYPE cinv_type = MASTERINVERSE;
    INVERSETYPE cinv_type_loc = SPARSECHOLESKY;
    size_t clev_nsteps = 1;                  // if smoothing, how many steps do we do?
    
    /** Smoothers **/

    enum SM_TYPE : char /** available smoothers **/
      { GS = 0,     // (l1/hybrid - ) Gauss-Seidel
	BGS = 1 };  // Block - (l1/hybrid - ) Gauss-Seidel 
    SM_TYPE sm_type = SM_TYPE::GS;       // the default smoother type
    Array<SM_TYPE> spec_sm_types;        // specific smoothers for levels

    enum GS_VER : char /** different hybrid GS versions (mostly for testing) **/
      { VER1 = 0,    // old version
	VER2 = 1,    // newer (maybe a bit faster than ver3 without overlap)
	VER3 = 2 };  // newest, optional overlap
    GS_VER gs_ver = GS_VER::VER3;

    bool sm_symm = false;                // smooth symmetrically
    bool sm_mpi_overlap = true;          // overlap communication/computation (only VER3)
    bool sm_mpi_thread = false;          // do MPI-comm in seperate thread (only VER3)
    bool sm_shm = true;                  // shared memory parallelization for (block-)smoothers ?
    bool sm_sl2 = false;                 // use SharedLoop2 instead of ParallelFor for (block-)smoothers ?

    /** Misc **/
    bool sync = false;                   // synchronize via MPI-Barrier in places
    bool do_test = false;                // perform PC-test for amg_mat
    bool smooth_lo_only = false;         // smooth only on low order part -> AMG-PC is for the LO part only
    bool regularize_cmats = false;       // do we need to regularize coarse level matrices ?

  public:

    Options () { ; }

    virtual void SetFromFlags (const Flags & flags, string prefix);
  }; //BaseAMGPC::Options

  /** END Options **/

} // namespace amg

#endif
