#ifndef FILE_AMG_FACTORY_STOKES_HPP
#define FILE_AMG_FACTORY_STOKES_HPP

#include <base.hpp>
#include <base_factory.hpp>
#include <nodal_factory.hpp>

#include "grid_contract.hpp"
#include "stokes_map.hpp"

namespace amg
{


/**
 * Ugly, but this is outside the StokesAMGFactory class because it is needed
 * from BaseStokesAMGPrecond, which we don't want to depend on the discrete
 * StokesAMGFactory type.
*/
struct BaseStokesLevelCapsule: public BaseAMGFactory::LevelCapsule
{
  shared_ptr<SparseMatrix<double>> pot_mat;           // matrix in the potential space
  shared_ptr<BitArray>             pot_freedofs;      // can have empty loops which must be Dirichlet
  UniversalDofs                    pot_uDofs;         // udofs in the potential space
  shared_ptr<BaseSparseMatrix>     curl_mat;          // discrete curl matrix on this level
  shared_ptr<BaseSparseMatrix>     curl_mat_T;        // discrete curl matrix transposed on this level
  UniversalDofs                    rr_uDofs;          // udofs for range of range (so the l2-like space)
  shared_ptr<BaseSparseMatrix>     divSparseMat;      // discrete divergence matrix (atm. for debugging purposes)
  shared_ptr<BaseSparseMatrix>     divSparseMatT;     // discrete (trans) divergence matrix (atm. for debugging purposes)
  shared_ptr<BaseMatrix>           AC;                // range-mat * curl-mat (opt. for hiptmair)

  /**
   * In div-div penalty mode, we work with only velocities, in that case
   *      - BaseAMGFactory::LevelCapsule::mat == primaryMat
   *      - divMat == nulptr.
   * In pressure-mode, we work with velocity + pressure and a saddle-point
   * block-matrix instead of div-div penalty. Then
   *      - primaryMat is the velocity-matrix
   *      - divMat/divMatT are the divergence-matrix + its transpose
   *      - BaseAMGFactory::LevelCapsule::mat is
   *             primaryMat  divMatT
   *             divMat       None
   */
  shared_ptr<BaseMatrix>           primaryMat;
  shared_ptr<BaseMatrix>           divMat;
  shared_ptr<BaseMatrix>           divMatT;

  /**
  * For Stokes, we keep around
  *   - the grid-contract map, if it exists
  *   - the dof-maps leading to the next (primary) level
  * Both of these are needed in order to set up the secondary AMG-sequence
  * when required. This leaves some of the maps in scope a little longer,
  * but they still go out of scope at the very end of the AMG setup.
  * The main difference should be that they now live until AFTER the smoothers
  * are created, so we are probably increasing the max used memory a bit.
  */
  shared_ptr<GridContractMap>       savedCtrMap;  // (nasty: set in BuildContractDOFMap, TODO: clean up)
  Array<shared_ptr<BaseDOFMapStep>> savedDOFMaps; // (set in MapLevel)
}; // struct StokesLevelCapsule


class SecondaryAMGSequenceFactory
{
public:
  SecondaryAMGSequenceFactory() = default;

  virtual ~SecondaryAMGSequenceFactory() = default;

  virtual std::tuple<shared_ptr<BaseProlMap>,
                     shared_ptr<BaseStokesLevelCapsule>>
  CreateRTZLevel (BaseStokesLevelCapsule const &cap) const = 0;

  virtual shared_ptr<BaseDOFMapStep>
  CreateRTZEmbedding (BaseStokesLevelCapsule const &cap) const = 0;

  virtual shared_ptr<BaseDOFMapStep>
  CreateRTZDOFMap (BaseStokesLevelCapsule const &fCap,
                   BaseDOFMapStep         const &fEmb,
                   BaseStokesLevelCapsule const &cCap,
                   BaseDOFMapStep         const &cEmb,
                   BaseDOFMapStep         const &dOFStep) const = 0;
}; // class SecondaryAMGSequenceFactory

/**
 * Stokes AMG, for ENERGY + div-div penalty:
 *   We assume that we have DIM DOFs per facet of the mesh. Divergence-The divergence
 *   - DOFs are assigned to edges of the dual mesh.
 */

template<class ATMESH, class AENERGY>
class StokesAMGFactory : public NodalAMGFactory<NT_EDGE, ATMESH, AENERGY::DPV>
                       , public SecondaryAMGSequenceFactory
{
public:
  using TMESH = ATMESH;
  using ENERGY = AENERGY;
  static constexpr int BS = ENERGY::DPV;
  static constexpr int DIM = ENERGY::DIM;
  using BASE_CLASS = NodalAMGFactory<NT_EDGE, ATMESH, AENERGY::DPV>;
  // using Options = typename BASE_CLASS::Options;
  class Options;
  // Note: NGSolve instantiates SparseMatrix<Mat<1, N>> with Vec<1> col-vecs, stripped_spm would strip that out!
  using TM                 = typename ENERGY::TM;
  using TSPM_TM            = stripped_spm_tm<TM>;
  using TSPM               = stripped_spm<TM>;
  using TCM_TM             = stripped_spm_tm<Mat<BS, 1, double>>;
  using TCM                = stripped_spm<Mat<BS, 1, double>>;
  using TCTM_TM            = stripped_spm_tm<Mat<1, BS, double>>;
  using TCTM               = stripped_spm<Mat<1, BS, double>>;
  using TPM_TM             = SparseMatrixTM<double>;
  using TPM                = SparseMatrix<double>;
  using TDM_TM             = stripped_spm_tm<Mat<1, BS>>;
  using TDM                = stripped_spm<Mat<1, BS>>;
  using StokesLevelCapsule = BaseStokesLevelCapsule;


protected:
  using BASE_CLASS::options;

public:

  StokesAMGFactory (shared_ptr<Options> _opts);

  ~StokesAMGFactory() { ; }

  virtual UniversalDofs BuildUDofs (BaseAMGFactory::LevelCapsule const &cap) const override;

protected:
  /** Misc overloads **/
  virtual BaseAMGFactory::State* AllocState () const override;
  virtual shared_ptr<BaseAMGFactory::LevelCapsule> AllocCap () const override;
  virtual shared_ptr<BaseDOFMapStep> MapLevel (FlatArray<shared_ptr<BaseDOFMapStep>> dof_steps,
                                               shared_ptr<BaseAMGFactory::AMGLevel> &f_cap,
                                               shared_ptr<BaseAMGFactory::AMGLevel> &c_cap) override;

  /** Coarse **/

  virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (BaseAMGFactory::State & state,
                                                    shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) override;

  /**
   * moved some finishing-up of the coarse mesh (loops, UDofs, etc. ) out of BuildCoarseMap
   * into FinalizeCoarseMap because the HDIV can only do it at that point once it knows
   * the coarse DOFs!
  */
  virtual void FinalizeCoarseMap (StokesLevelCapsule     const &fCap,
                                  StokesLevelCapsule           &cCap,
                                  StokesCoarseMap<TMESH>       &cMap);

  template<class AGG_CLASS>
  INLINE shared_ptr<StokesCoarseMap<TMESH>> BuildAggMap (BaseAMGFactory::State & state,
                                                         shared_ptr<StokesLevelCapsule> & mapped_cap);

  virtual shared_ptr<BaseDOFMapStep> BuildCoarseDOFMap (shared_ptr<BaseCoarseMap> cmap,
                                                        shared_ptr<BaseAMGFactory::LevelCapsule> fcap,
                                                        shared_ptr<BaseAMGFactory::LevelCapsule> ccap,
                                                        shared_ptr<BaseDOFMapStep> embMap = nullptr) override;

  virtual shared_ptr<BaseDOFMapStep> RangeProlMap (shared_ptr<StokesCoarseMap<TMESH>> cmap,
                                                   shared_ptr<StokesLevelCapsule> fcap,
                                                   shared_ptr<StokesLevelCapsule> ccap);

  virtual shared_ptr<TSPM>
  BuildPrimarySpaceProlongation (BaseAMGFactory::LevelCapsule const &fcap,
                                 BaseAMGFactory::LevelCapsule       &ccap,
                                 StokesCoarseMap<TMESH>       const &cmap,
                                 TMESH                        const &fmesh,
                                 TMESH                              &cmesh,
                                 FlatArray<int>                      vmap,
                                 FlatArray<int>                      emap,
                                 FlatTable<int>                      v_aggs,
                                 TSPM_TM                      const *pA) = 0;

  shared_ptr<SparseMatrix<double>>
  BuildPressureProlongation(shared_ptr<StokesCoarseMap<TMESH>> cmap,
                            shared_ptr<StokesLevelCapsule> fcap,
                            shared_ptr<StokesLevelCapsule> ccap);

public:
  /**
   * Mostly used for setting up coarse levels, but also needs to be called from
   * outside when setting up the finest level.
   */
  virtual void BuildDivMat  (StokesLevelCapsule& cap) const = 0;
  virtual void BuildCurlMat (StokesLevelCapsule& cap) const = 0;

  virtual void ProjectToPotSpace (StokesLevelCapsule& cap) const;
  virtual void BuildPotUDofs (StokesLevelCapsule& cap) const;

protected:

  /** Contract **/
  virtual shared_ptr<BaseGridMapStep> BuildContractMap (double factor,
                                                        shared_ptr<TopologicMesh> mesh,
                                                        shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) const override;

  virtual shared_ptr<GridContractMap> AllocateContractMap (Table<int> && groups, shared_ptr<TMESH> mesh) const override;

  /** DEBUGGING **/
  virtual void DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map) override;

  virtual void CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels, shared_ptr<DOFMap> dof_map);


}; // class StokesAMGFactory

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP
