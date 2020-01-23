#ifndef FILE_AMG_FACTORY_NODAL_HPP
#define FILE_AMG_FACTORY_NODAL_HPP

namespace amg
{

  /** BS DOFs per node of type NT **/
  template<NODE_TYPE ANT, class ATMESH, int ABS>
  class NodalAMGFactory : public BaseAMGFactory
  {
  public:
    static constexpr NODE_TYPE NT = ANT;
    static constexpr int BS = ABS;
    using TMESH = ATMESH;
    using OPTIONS = BaseAMGFactory::Options;

  public:    

    NodalAMGFactory (shared_ptr<Options> _opts);

    virtual shared_ptr<ParallelDofs> BuildParallelDofs (shared_ptr<TopologicMesh> amesh) const override;

  protected:    

    virtual size_t ComputeMeshMeasure (const TopologicMesh & m) const override;
    virtual double ComputeLocFrac (const TopologicMesh & m) const override;

    virtual size_t ComputeGoal (const AMGLevel & f_lev, State & state) override;

    virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh) const override;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFContractMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<ParallelDofs> fpd) const override;
  }; // NodalAMGFactory

} // namespace amg

#endif
