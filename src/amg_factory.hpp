#ifndef FILE_AMG_FACTORY_HPP
#define FILE_AMG_FACTORY_HPP

namespace amg
{
  /**
     Factories build grid-transfer operators (DofMaps) and coarse level matrices.
     
     They assume canonical ordering of DOFs and need a finished finest mesh to start from.
   **/


  /**
     (abstract)
     Sets up prolongation- and coarse level matrices.
   **/
  template<class TMESH, class TM>
  class AMGFactory
  {
  public:
    using TSPM_TM = stripped_spm_tm<TM>;

    class Options; // configurable from the outside

    class State; // some internal book-keeping

  protected:
    shared_ptr<Options> opts;
    shared_ptr<TMESH> finest_mesh;
    shared_ptr<BaseDOFMapStep> embed_step;
    State state;

  public:

    AMGFactory (shared_ptr<TMESH> _finest_mesh, shared_ptr<BaseDOFMapStep> _embed_step,
		shared_ptr<Options> _opts);

    void SetupLevels (Array<BaseSparseMatrix> & mats, shared_ptr<DOFMap> & dmap);

  protected:

    // e.g NV for H1 amg
    size_t GetMeshMeasure (TMESH & m) const = 0;

    struct Capsule
    {
      int level;
      shared_ptr<TopologicMesh> mesh;
      shared_ptr<ParallelDofs> pardofs;
      shared_ptr<BaseSparseMatrix> mat;
    };

    // recursive setup method - sets up one more level and calls itself
    Array<shared_ptr<BaseSparseMatrix>> RSU (Capsule cap, shared_ptr<DOFMap> dof_map);
    
    virtual shared_ptr<CoarseMap<TMESH>> BuildCoarseMap  (shared_ptr<TMESH> mesh) const = 0;
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<GridConctractMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const = 0;

    virtual shared_ptr<GridContractMap<TMESH>> BuildContractMap (double factor, shared_ptr<TMESH> mesh) const;


  };

  /**
     DPN dofs per node of the specified kind
   **/
  template<NODE_TYPE NT, class TMESH, class TM>
  class NodalAMGFactory : public AMGFactory<TMESH, TM>
  {
  public:
    struct Options;

  };


  template<class TMESH, class TM>
  class VertexBasedAMGFactory : public NodalAMGFactory<NT_VERTEX, TMESH, TM>
  {
  public:
    struct Options;

  protected:
    shared_ptr<BitArray> free_vertices;

    virtual shared_ptr<CoarseMap<TMESH>> BuildCoarseMap  (shared_ptr<TMESH> mesh) const;
  };

} // namespace amg

#endif
