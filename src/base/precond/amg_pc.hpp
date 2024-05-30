#ifndef FILE_AMGPC_HPP
#define FILE_AMGPC_HPP

#include <base.hpp>
#include <base_factory.hpp>
#include <base_smoother.hpp>
#include <amg_matrix.hpp>
#include <dof_map.hpp>

namespace amg
{
  /**
    * Base class AMG Preconditioner;
    *  This is more or less the layer between NGSolve and the AMG code.
    *    (a) sets up the topology + attached data needed for AMG setup
    *    (b) embeds from the AMG "canonical" DOFs to the FESpace DOFs
    *        - in parallel, this involves a renumbering
    *        - e.g. for elasticity, displacement gets embedded into disp+rotations
    *        - if I ever get around to AUX space, it is also more
    *    (c) sets up the smoothers for every level
    *  Ideally, I would like to seperate (a) and (b) from (c), with (a) and (b) staying in the preconditioener
    *  and (c) moving to the factory, because:
    *    - it would allow easier coupling from the outside
    *    - easier Auxiliary space AMG
    */
  class BaseAMGPC : public Preconditioner
  {
  public:

    class Options
    {
    public:

      /** Which AMG cycle to use **/
      enum MG_CYCLE : unsigned {
        V_CYCLE = 0,       // V cycle
        W_CYCLE = 1,       // W cycle
        BS_CYCLE = 2       // (hacky) Braess-Sarazin
      };
      MG_CYCLE mg_cycle = V_CYCLE;

      /** What we do on the coarsest level **/
      enum CLEVEL : unsigned {
        INV_CLEV = 0,       // invert coarsest level
        SMOOTH_CLEV = 1,    // smooth coarsest level
        NO_CLEV = 2
      };
      CLEVEL clev = INV_CLEV;
      INVERSETYPE cinv_type = MASTERINVERSE;
      INVERSETYPE cinv_type_loc = SPARSECHOLESKY;
      size_t clev_nsteps = 1;                  // if smoothing, how many steps do we do?

      /** Smoothers **/

      enum SM_TYPE : unsigned { /** available smoothers **/
        GS          = 0, // (l1/hybrid - ) Gauss-Seidel
        BGS         = 1, // Block - (l1/hybrid - ) Gauss-Seidel
        JACOBI      = 2, // jacobi, ONLY LOCAL
        HIPTMAIR    = 3, // potential space smoother
        AMGSM       = 4, // an AMG cycle used as smoother
        DYNBGS      = 5  // dynamic block Gauss-Seidel
      };
      SpecOpt<SM_TYPE> sm_type = SM_TYPE::GS; // smoother type

      SpecOpt<int> sm_steps = 1;           // # of smoothing steps
      SpecOpt<int> sm_steps_loc = 1;       // # of smoothing steps
      SpecOpt<bool> sm_symm = false;       // smooth symmetrically
      SpecOpt<bool> sm_symm_loc = false;   // smooth symmetrically
      bool sm_NG_MPI_overlap = true;          // overlap communication/computation (only VER3)
      bool sm_NG_MPI_thread = false;          // do MPI-comm in seperate thread (only VER3)
      bool sm_shm = true;                  // shared memory parallelization for (block-)smoothers ?
      bool sm_sl2 = false;                 // use SharedLoop2 instead of ParallelFor for (block-)smoothers ?

      /** Misc **/
      bool sync = false;                   // synchronize via MPI-Barrier in places
      bool do_test = false;                // perform PC-test for amg_mat
      bool test_levels = false;            // perform PC-tests on every level
      SpecOpt<bool> test_2level = false;            // perform PC-tests on every level
      bool test_smoothers = false;         // perform PC-tests for smoothers
      bool smooth_lo_only = false;         // smooth only on low order part -> AMG-PC is for the LO part only
      bool regularize_cmats = false;       // do we need to regularize coarse level matrices ?
      bool force_ass_flmat = false;        // force assembling of matrix belonging to finest level (embedding)

      /** How do we compute the replacement matrix **/
      enum ENERGY : unsigned {
        TRIV_ENERGY = 0,     // uniform weights
        ALG_ENERGY = 1,      // from the sparse matrix
        ELMAT_ENERGY = 2 };  // from element matrices
      ENERGY energy = ALG_ENERGY;

      /** Logging **/
      enum LOG_LEVEL_PC : unsigned {
        NONE   = 0,              // nothing
        BASIC  = 1,              // summary info
        NORMAL = 2,              // global level-wise info
        EXTRA  = 3,              // local level-wise info
        DBG    = 4               // extra debug info
      };
      LOG_LEVEL_PC log_level_pc = LOG_LEVEL_PC::NONE;   // how much info do we collect
      bool print_log_pc = true;                           // print log to shell
      string log_file_pc = "";                            // which file to print log to (none if empty)

    public:

      Options () { ; }

      virtual void SetFromFlags (shared_ptr<FESpace> fes, shared_ptr<BaseMatrix> finest_mat, const Flags & flags, string prefix);
    }; //BaseAMGPC::Options

  protected:
    shared_ptr<Options> options;

    shared_ptr<BilinearForm> bfa;

    // shared_ptr<BaseAMGFactory> factory;
    shared_ptr<BitArray> finest_freedofs;
    shared_ptr<BaseMatrix> finest_mat;
    shared_ptr<AMGMatrix> amg_mat;

    // are we running in "strict algebraic" mode,
    // i.e. without acces to mesh,fes,blf
    bool strict_alg_mode;

    INLINE bool inStrictAlgMode () const { return strict_alg_mode; }

  public:

    /** Constructors **/
    BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    // strict_alg_mode constructor
    BaseAMGPC (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts = nullptr);

    ~BaseAMGPC ();

    /** Preconditioner overloads **/
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;

    // raw-ptr FinalzieLevel only hands over to shared_ptr FinalizeLevel
    virtual void FinalizeLevel (const BaseMatrix * mat) override final;
    virtual void FinalizeLevel (shared_ptr<BaseMatrix> mat);

    /** BaseMatrix overloads **/
    virtual const BaseMatrix & GetAMatrix () const override;
    virtual const BaseMatrix & GetMatrix () const override;
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override;
    virtual shared_ptr<AMGMatrix> GetAMGMatrix () const;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual AutoVector CreateRowVector () const override // sequentially, GetAMatrix().Create gives a VVector, GetMatrix.Create a (dummy) ParVec
      { return GetAMGMatrix()->CreateColVector(); }
    virtual AutoVector CreateColVector () const override
      { return GetAMGMatrix()->CreateRowVector(); }

    /** called from InitLevel*/
    void InitializeOptions ();

  protected:

    /** Options: construct, set default, set from flags, modify **/
    shared_ptr<Options> MakeOptionsFromFlags (const Flags & flags, string prefix = "ngs_amg_");
    virtual shared_ptr<Options> NewOpts () = 0;
    virtual void SetDefaultOptions (Options& O);
    virtual void SetOptionsFromFlags (Options& O, const Flags & flags, string prefix = "ngs_amg_");
    virtual void ModifyOptions (Options & O, const Flags & flags, string prefix = "ngs_amg_");

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () = 0;
    // virtual shared_ptr<BaseAMGFactory> BuildFactory () = 0;
    virtual BaseAMGFactory& GetBaseFactory() const = 0;

    virtual void Finalize ();
    virtual void BuildAMGMat ();

    virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level);
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (BaseAMGFactory::AMGLevel & finest_level) = 0;

    virtual
    Array<shared_ptr<BaseSmoother>>
    BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
							      shared_ptr<DOFMap> dof_map);

    virtual shared_ptr<BaseSmoother>
    BuildSmoother (BaseAMGFactory::AMGLevel const &amg_level);

    virtual shared_ptr<BaseSmoother>
    BuildGSSmoother (shared_ptr<BaseMatrix> spm,
						         shared_ptr<BitArray> freedofs = nullptr);

    virtual shared_ptr<BaseSmoother>
    BuildBGSSmoother (shared_ptr<BaseMatrix> spm,
                      Table<int> && blocks);

    virtual shared_ptr<BaseSmoother>
    BuildJacobiSmoother (shared_ptr<BaseMatrix> spm,
						             shared_ptr<BitArray> freedofs = nullptr);

    virtual shared_ptr<BaseSmoother>
    BuildDynamicBlockGSSmoother (shared_ptr<BaseMatrix> spm,
						                     shared_ptr<BitArray> freedofs = nullptr);

    virtual shared_ptr<BitArray> GetFreeDofs (const BaseAMGFactory::AMGLevel & amg_level);

    virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level);

    virtual Options::SM_TYPE
    SelectSmoother(BaseAMGFactory::AMGLevel const &amgLevel) const = 0;

    // return (cmat, cinv)
    virtual std::tuple<std::shared_ptr<BaseMatrix>, shared_ptr<BaseMatrix>>
    CoarseLevelInv(BaseAMGFactory::AMGLevel const &coarseLevel);

  public: // python-export!
    virtual void RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const;

  protected:
    BilinearForm const & GetBFA() const { return checked_dereference(bfa); }
    BilinearForm       & GetBFA()       { return checked_dereference(bfa); }

  }; // BaseAMGPC


  /** Options **/

  INLINE std::ostream& operator<<(std::ostream &os, BaseAMGPC::Options::SM_TYPE const &sMType)
  {
    switch(sMType)
    {
      case(BaseAMGPC::Options::SM_TYPE::GS)       : { os << "GSS"; break; }
      case(BaseAMGPC::Options::SM_TYPE::BGS)      : { os << "BGS"; break; }
      case(BaseAMGPC::Options::SM_TYPE::JACOBI)   : { os << "JAC"; break; }
      case(BaseAMGPC::Options::SM_TYPE::HIPTMAIR) : { os << "HIP"; break; }
      case(BaseAMGPC::Options::SM_TYPE::AMGSM)    : { os << "AAS"; break; }      
      case(BaseAMGPC::Options::SM_TYPE::DYNBGS)   : { os << "DYNBGS"; break; }      
    }
    return os;
  }

  /** END Options **/

  /** utility **/
  void TestSmoother (shared_ptr<BaseMatrix> mt, shared_ptr<BaseSmoother> sm, NgMPI_Comm & gcomm, string message = "");
  void TestSmoother (shared_ptr<BaseSmoother> sm, NgMPI_Comm & gcomm, string message = "");
  void TestSmootherLAPACK (shared_ptr<BaseMatrix> mt, shared_ptr<BaseSmoother> sm, shared_ptr<BitArray> free, NgMPI_Comm & gcomm, string message = "");
  void TestAMGMat (shared_ptr<BaseMatrix> mt, shared_ptr<AMGMatrix> amt, int start_lev, NgMPI_Comm & gcomm, string message = "");

} // namespace amg

#endif
