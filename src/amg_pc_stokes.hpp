#ifdef STOKES
#ifndef FILE_AMG_PC_STOKES_HPP
#define FILE_AMG_PC_STOKES_HPP


namespace amg
{

  /** StokesAMGPC **/

  template<class AFACTORY, class AAUX_SYS>
  class StokesAMGPC : public AuxiliarySpacePreconditioner<AAUX_SYS, BaseAMGPC>
  {
  public:
    using FACTORY = AFACTORY;
    static constexpr int DIM = FACTORY::DIM;
    using TMESH = typename FACTORY::TMESH;
    using AUX_SYS = AAUX_SYS;
    using AUXPC = AuxiliarySpacePreconditioner<AAUX_SYS, BaseAMGPC>;
    class Options;
    
  protected:

    using Preconditioner::ma;
    using BaseAMGPC::options, BaseAMGPC::factory;
    using AUXPC::aux_sys, AUXPC::emb_amg_mat;

    Array<Array<int>> node_sort;
    bool forced_fds = false;

  public:

    StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond");

    StokesAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~StokesAMGPC ();

    virtual void InitLevelForced (shared_ptr<BitArray> freedofs = nullptr);

  protected:

    /** BaseAMGPC overloads **/
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
    // virtual void FinalizeLevel (const BaseMatrix * mat) override;
    virtual void Update () override { ; } // TODO: what should this do??
    virtual shared_ptr<AMGLevel> AllocLevel () override;

    virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
    virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
    virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
    virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;
    virtual void InitFinestLevel (shared_ptr<BaseAMGFactory::AMGLevel> & finest_level) override;
    virtual Table<int> GetGSBlocks (shared_ptr<const BaseAMGFactory::AMGLevel> & amg_level) override;
    virtual Table<int> GetGSBlocks2 (shared_ptr<const BaseAMGFactory::AMGLevel> & amg_level);
    virtual shared_ptr<BaseAMGFactory> BuildFactory () override;
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) override;

    virtual void RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const override;

    /** For BuildInitialMesh **/
    virtual shared_ptr<BlockTM> BuildTopMesh ();
    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
    virtual void SetLoops (shared_ptr<TMESH> alg_mesh);
    virtual shared_ptr<TMESH> BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const; // implement seperately (but easy)
    virtual shared_ptr<TMESH> BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh) const; // implement seperately (but easy)

    /** Hiptmair Smoother for Stokes **/
    virtual shared_ptr<BaseSmoother> BuildSmoother (const shared_ptr<BaseAMGFactory::AMGLevel> & amg_level) override;
    virtual shared_ptr<BaseSmoother> BuildHiptMairSmoother (const shared_ptr<BaseAMGFactory::AMGLevel> & amg_level, shared_ptr<BaseSmoother> smoother);

  }; // class StokesAMGPC

  /** END StokesAMGPC **/

} // namespace amg


#endif // FILE_AMG_PC_STOKES_HPP
#endif // STOKES
