#ifndef FILE_AMG_FACTORY_STOKES_NC_HPP
#define FILE_AMG_FACTORY_STOKES_NC_HPP

#include <stokes_factory.hpp>

namespace amg
{


/**
 * Stokes AMG, for ENERGY + div-div penalty:
 *   We assume that we have DIM DOFs per facet of the mesh. Divergence-The divergence
 *   - DOFs are assigned to edges of the dual mesh.
 */

template<class ATMESH, class AENERGY>
class NCStokesAMGFactory : public StokesAMGFactory<ATMESH, AENERGY>
{
public:
  using TMESH = ATMESH;
  using ENERGY = AENERGY;

  static constexpr int BS = ENERGY::DPV;
  static constexpr int DIM = ENERGY::DIM;

  using BASE_CLASS = StokesAMGFactory<ATMESH, AENERGY>;

  using TM                 = typename BASE_CLASS::TM;
  using TSPM               = typename BASE_CLASS::TSPM;
  using TSPM_TM            = typename BASE_CLASS::TSPM_TM;
  using TCM_TM             = typename BASE_CLASS::TCM_TM;
  using TCM                = typename BASE_CLASS::TCM;
  using TCTM_TM            = typename BASE_CLASS::TCTM_TM;
  using TCTM               = typename BASE_CLASS::TCTM;
  using TPM                = typename BASE_CLASS::TPM;
  using TPM_TM             = typename BASE_CLASS::TPM_TM;
  using TDM                = typename BASE_CLASS::TDM;
  using TDM_TM             = typename BASE_CLASS::TDM_TM;
  using Options            = typename BASE_CLASS::Options;
  using StokesLevelCapsule = typename BASE_CLASS::StokesLevelCapsule;

protected:
  using BASE_CLASS::options;

public:

  NCStokesAMGFactory (shared_ptr<Options> _opts);

  ~NCStokesAMGFactory() { ; }

protected:

  // virtual shared_ptr<BaseDOFMapStep> BuildCoarseDOFMap (shared_ptr<BaseCoarseMap> cmap,
  //                                                       shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
  //                                                       shared_ptr<BaseAMGFactory::LevelCapsule> ccap) override;

  // virtual shared_ptr<BaseDOFMapStep> RangeProlMap (shared_ptr<StokesCoarseMap<TMESH>> cmap,
  //                                                  shared_ptr<StokesLevelCapsule> fcap,
  //                                                  shared_ptr<StokesLevelCapsule> ccap);

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
public:
  /**
   * Mostly used for setting up coarse levels, but also needs to be called from
   * outside when setting up the finest level.
   */
  virtual void BuildDivMat  (StokesLevelCapsule& cap) const override;
  virtual void BuildCurlMat (StokesLevelCapsule& cap) const override;


protected:

  virtual shared_ptr<BaseDOFMapStep> BuildContractDOFMap (shared_ptr<BaseGridMapStep> cmap,
                                                          shared_ptr<BaseAMGFactory::LevelCapsule> &fCap,
                                                          shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) const override;

  /** DEBUGGING **/
  virtual void DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map) override;

  virtual void CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map) override;


  /** lowest order Raviart Thomas 0 **/

  class RTZToBlockEmbedding : public ProlMap<StripTM<BS, 1>>
  {
  public:
    using BASE = ProlMap<StripTM<BS, 1>>;
    using TM   = StripTM<BS, 1>;

    RTZToBlockEmbedding (UniversalDofs uDofs);

    virtual ~RTZToBlockEmbedding () = default;

    void Finalize () override;

    shared_ptr<BitArray> FinalizeWithFD ();

    void
    SetPreservedComponent (size_t          const &k,
                           Vec<BS, double> const &vec);

    template<int BSA, int BSB>
    shared_ptr<SparseMatrix<double>>
    ProjectMatrix (SparseMatTM<BSA, BSB> const &A) const;

    shared_ptr<SparseMatrix<double>>
    ProjectProl (SparseMatTM<BS, BS> const &P,
                 RTZToBlockEmbedding const &rightEmb) const;

  protected:
    template<bool IS_RIGHT, int BSA, int BSB>
    INLINE shared_ptr<SparseMatrix<double>>
    ProjectMatrixImpl (SparseMatTM<BSA, BSB> const &A,
                       RTZToBlockEmbedding   const *rightEmb = nullptr) const;

    void
    SetUpProl ();

  protected:
    Vec<BS, double> const &GetPresComp (int k) const { return _presComp[k]; }
    Vec<BS, double>       &GetPresComp (int k)       { return _presComp[k]; }

    Array<Vec<BS, double>> _presComp;
  }; // class RTZToBlockEmbedding

  std::tuple<shared_ptr<BaseProlMap>,
             shared_ptr<BaseStokesLevelCapsule>>
  CreateRTZLevel (BaseStokesLevelCapsule const &cap) const override;

  shared_ptr<RTZToBlockEmbedding>
  CreateRTZEmbeddingImpl (BaseStokesLevelCapsule const &cap) const;

  shared_ptr<BaseDOFMapStep>
  CreateRTZEmbedding (BaseStokesLevelCapsule const &cap) const override;

  shared_ptr<BaseDOFMapStep>
  CreateRTZDOFMap (BaseStokesLevelCapsule const &fCap,
                   BaseDOFMapStep         const &fEmb,
                   BaseStokesLevelCapsule const &cCap,
                   BaseDOFMapStep         const &cEmb,
                   BaseDOFMapStep         const &dOFStep) const override;
}; // class StokesAMGFactory




} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_NC_HPP
