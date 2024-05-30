#ifndef FILE_AMG_FACTORY_NODAL_HPP
#define FILE_AMG_FACTORY_NODAL_HPP

#include "base_factory.hpp"

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

    virtual UniversalDofs BuildUDofs (LevelCapsule const &cap) const override;

  protected:

    virtual size_t ComputeMeshMeasure (const TopologicMesh & m) const override;
    virtual double ComputeLocFrac (const TopologicMesh & m) const override;

    virtual size_t ComputeGoal (const shared_ptr<AMGLevel> & f_lev, State & state) override;

    virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh, shared_ptr<LevelCapsule> & mapped_cap) const override;
    virtual shared_ptr<GridContractMap> AllocateContractMap (Table<int> && groups, shared_ptr<TMESH> mesh) const = 0;
    virtual shared_ptr<BaseDOFMapStep> BuildContractDOFMap (shared_ptr<BaseGridMapStep> cmap, shared_ptr<LevelCapsule> &fCap, shared_ptr<LevelCapsule> & mapped_cap) const override;

  }; // NodalAMGFactory

} // namespace amg

#endif
