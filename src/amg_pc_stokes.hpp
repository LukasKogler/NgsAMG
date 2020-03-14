#ifdef STOKES
#ifndef FILE_AMG_PC_STOKES_HPP
#define FILE_AMG_PC_STOKES_HPP


namespace amg
{

  /** StokesAMGPC **/

  template<class AFACTORY, class AAUX_SYS>
  class StokesAMGPC : public BaseAMGPC
  {
  public:
    using FACTORY = AFACTORY;
    static constexpr int DIM = FACTORY::DIM;
    using TMESH = typename FACTORY::TMESH;
    using AUX_SYS = AAUX_SYS;

    class Options;
    
  protected:

    using BaseAMGPC::options;

    Array<Array<int>> node_sort;

    shared_ptr<AUX_SYS> aux_sys;

    shared_ptr<BitArray> free_facets; // re-sorted aux_sys->GetAuxFreeDofs()

    shared_ptr<EmbeddedAMGMatrix> emb_amg_mat; /** as a preconditioner for the compound BLF **/

  public:

    StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond");

    StokesAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~StokesAMGPC ();

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

    /** New methods **/
    shared_ptr<AUX_SYS> GetAuxSys () const { return aux_sys; }
    shared_ptr<typename AUX_SYS::TPMAT> GetPMat () const { return aux_sys->GetPMat(); }
    shared_ptr<typename AUX_SYS::TAUX> GetAuxMat () const { return aux_sys->GetAuxMat(); }
    shared_ptr<BitArray> GetAuxFreeDofs () const { return aux_sys->GetAuxFreeDofs(); } // free_verts are after sorting
    Array<Array<shared_ptr<BaseVector>>> GetRBModes () const { return aux_sys->GetRBModes(); }
    virtual shared_ptr<BaseVector> CreateAuxVector () const { return aux_sys->CreateAuxVector(); }
    shared_ptr<EmbeddedAMGMatrix> GetEmbAMGMat () const;

    /** For BuildInitialMesh **/
    virtual shared_ptr<BlockTM> BuildTopMesh ();
    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
    virtual shared_ptr<TMESH> BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const; // implement seperately (but easy)
  }; // class StokesAMGPC

  /** END StokesAMGPC **/

} // namespace amg


#endif // FILE_AMG_PC_STOKES_HPP
#endif // STOKES
