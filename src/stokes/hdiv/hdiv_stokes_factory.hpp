#ifndef FILE_HDIV_STOKES_FACTORY_HPP
#define FILE_HDIV_STOKES_FACTORY_HPP

#include "dyn_block.hpp"
#include "mesh_dofs.hpp"
#include "preserved_vectors.hpp"

#include "stokes_factory.hpp"
#include "stokes_map.hpp"
#include "universal_dofs.hpp"

namespace amg
{

template<class ATMESH, class AENERGY>
class HDivStokesAMGFactory : public StokesAMGFactory<ATMESH, AENERGY>
{
public:
  using BASE = StokesAMGFactory<ATMESH, AENERGY>;

  static_assert(BASE::BS == 1, "BLOCK-SIZE MISMATCH!");

  using TSPM_TM            = typename BASE::TSPM_TM;
  using TSPM               = typename BASE::TSPM;

  // div-mat
  using TDM                = typename BASE::TDM;

  // curl-mat
  using TCM_TM             = typename BASE::TCM_TM;
  using TCM                = typename BASE::TCM;
  using TCTM_TM            = typename BASE::TCTM_TM;
  using TCTM               = typename BASE::TCTM;

  using TMESH              = typename BASE::TMESH;
  using ENERGY             = typename BASE::ENERGY;
  using Options            = typename BASE::Options;
  using StokesLevelCapsule = typename BASE::StokesLevelCapsule;

  static constexpr int DIM = ENERGY::DIM;

protected:
  using BASE::options;

public:
  HDivStokesAMGFactory (shared_ptr<Options> _opts);

  ~HDivStokesAMGFactory () = default;

  struct HDivStokesLevelCapsule : public StokesLevelCapsule
  {
    shared_ptr<TSPM>             embeddedRangeMatrix; // for level 0
    shared_ptr<MeshDOFs>         meshDOFs;          // DOF <-> edge association
    shared_ptr<PreservedVectors> preservedVectors;  // vectors preserved on every level (usually: constants plus "n")

    DynVectorBlocking<>          dOFBlocking;        // DOF-blocking for the mesh-canonic space
    DynVectorBlocking<>          preCtrCDOFBlocking; // DOF-blocking of the pre-ctr (coarse) mesh
  };

  virtual shared_ptr<BaseAMGFactory::LevelCapsule> AllocCap () const override;

protected:

  shared_ptr<TSPM>
  BuildPrimarySpaceProlongation (BaseAMGFactory::LevelCapsule const &fcap,
                                 BaseAMGFactory::LevelCapsule       &ccap,
                                 StokesCoarseMap<TMESH>       const &cmap,
                                 TMESH                        const &fmesh,
                                 TMESH                              &cmesh,
                                 FlatArray<int>                      vmap,
                                 FlatArray<int>                      emap,
                                 FlatTable<int>                      v_aggs,
                                 TSPM_TM                      const *pA) override;

  void
  FinalizeCoarseMap (StokesLevelCapsule     const &fCap,
                     StokesLevelCapsule           &cCap,
                     StokesCoarseMap<TMESH>       &cMap) override;

public:
  virtual void BuildDivMat  (StokesLevelCapsule& cap) const override;
  virtual void BuildCurlMat (StokesLevelCapsule& cap) const override;

  virtual UniversalDofs BuildUDofs (BaseAMGFactory::LevelCapsule const &amesh) const override;

  /** lowest order Raviart Thomas 0 **/

  std::tuple<shared_ptr<BaseProlMap>,
             shared_ptr<BaseStokesLevelCapsule>>
  CreateRTZLevel (BaseStokesLevelCapsule const &cap) const override;

  shared_ptr<ProlMap<double>>
  CreateRTZEmbeddingImpl (BaseStokesLevelCapsule const &cap) const;

  shared_ptr<BaseDOFMapStep>
  CreateRTZEmbedding (BaseStokesLevelCapsule const &cap) const override;

  shared_ptr<BaseDOFMapStep>
  CreateRTZDOFMap (BaseStokesLevelCapsule const &fCap,
                   BaseDOFMapStep         const &fEmb,
                   BaseStokesLevelCapsule const &cCap,
                   BaseDOFMapStep         const &cEmb,
                   BaseDOFMapStep         const &dOFStep) const override;

  DynVectorBlocking<>
  BuildDOFBlocking(TMESH    const &mesh,
                   MeshDOFs const &meshDOFs);

protected:
  virtual shared_ptr<BaseDOFMapStep> BuildContractDOFMap (shared_ptr<BaseGridMapStep> cmap,
                                                          shared_ptr<BaseAMGFactory::LevelCapsule> &fCap,
                                                          shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) const override;

  /**
   * Returns:
   *  - contracted MeshDOFs (nullptr if not master master)
   *  - DOF-maps            (empty if not master)
  */
  tuple<shared_ptr<MeshDOFs>, Table<int>>
  ContractMeshDOFs (StokesContractMap<TMESH> const &ctrMap,
                    MeshDOFs                 const &fMeshDOFs) const;

  /** DEBUGGING **/
  virtual void DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map) override;

  virtual void CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map) override;

private:
  Options const &GetOptions() const { return static_cast<Options&>(*options); }
}; // class HDivStokesAMGFactory

} // namespace amg

#endif // FILE_HDIV_STOKES_FACTORY_HPP