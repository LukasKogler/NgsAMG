#ifndef FILE_NC_STOKES_PC_HPP
#define FILE_NC_STOKES_PC_HPP

#include "stokes_pc.hpp"
#include "ncfespace.hpp"
#include "mesh_dofs.hpp"
#include "preserved_vectors.hpp"

#include "hdiv_hdg_embedding.hpp"

namespace amg
{

/** HDivStokesAMGPC **/

/** Stokes-AMG Preconditioner for non-conforming H1 space **/
template<class ATFACTORY>
class HDivStokesAMGPC : public StokesAMGPrecond<typename ATFACTORY::TMESH>
{
public:
  using TFACTORY = ATFACTORY;
  using TMESH = typename TFACTORY::TMESH;

  static constexpr int DIM = TFACTORY::DIM;
  static constexpr int BS  = 1;

  class Options : public TFACTORY::Options,
                  public BaseStokesAMGPrecond::Options
  {
  public:
    HDivHDGEmbedding::AUX_SPACE auxSpace;
    bool use_dynbs_prols = true;

    virtual void SetFromFlags (shared_ptr<FESpace> fes,
                               shared_ptr<BaseMatrix> finest_mat,
                               const Flags & flags,
                               string prefix) override
    {
      TFACTORY::Options::SetFromFlags(flags, prefix);
      BaseStokesAMGPrecond::Options::SetFromFlags(fes, finest_mat, flags, prefix);

      auto auxSpaceStr = flags.GetStringFlag("ngs_amg_pres_vecs", "P0");

      if (auxSpaceStr == "RTZ")
      {
        auxSpace = HDivHDGEmbedding::AUX_SPACE::RTZ;
      }
      else if (auxSpaceStr == "P1")
      {
        auxSpace = HDivHDGEmbedding::AUX_SPACE::P1;
      }
      else if (auxSpaceStr == "FULL_P1")
      {
        auxSpace = HDivHDGEmbedding::AUX_SPACE::FULL_P1;
      }
      else
      {
        auxSpace = HDivHDGEmbedding::AUX_SPACE::P0;
      }

      use_dynbs_prols = !flags.GetDefineFlagX(prefix + "use_dynbs_prols").IsFalse();
    }
  }; // class BaseStokesAMGPrecond::Options

public:

  HDivStokesAMGPC (shared_ptr<BilinearForm>      blf,
                   Flags                  const &flags,
                   string                 const &name,
                   shared_ptr<Options>           opts = nullptr,
                   shared_ptr<BaseMatrix>        weightMat = nullptr);

  // set up from assembled matrix
  HDivStokesAMGPC (shared_ptr<FESpace>           fes,
                   Flags                        &flags,
                   string                 const &name,
                   shared_ptr<Options>           opts = nullptr,
                   shared_ptr<BaseMatrix>        weightMat = nullptr);

  ~HDivStokesAMGPC () = default;

  // void SetVectorsToPreserve (FlatArray<shared_ptr<BaseVector>> vectorsToPreserve);

  virtual void FinalizeLevel (shared_ptr<BaseMatrix> mat) override;

  virtual FacetAuxiliaryInformation const & GetFacetAuxInfo() const override;

  using BaseStokesAMGPrecond::options;
  using BaseStokesAMGPrecond::F2E;
  using BaseStokesAMGPrecond::EL2V;

protected:

  /** Options **/
  virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
  virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
  virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
  virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

  /** Mesh initialization  **/

  // For HDIV, we can only fill the ALG-mesh with facet-flows and weights after we have the (range-) prolongation
  // Therefore, BuildInitialMesh is overloaded here again and does not fill weights for flows
  virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;

  virtual shared_ptr<TopologicMesh> BuildAlgMesh (shared_ptr<BlockTM> &&top_mesh, FlatArray<FVDescriptor> fvd) override;
  // virtual shared_ptr<TMESH> FillAlgMesh_TRIV (TMESH & alg_mesh) const; // compute edge/vertex weights
  // virtual shared_ptr<TMESH> FillAlgMesh_ALG  (TMESH & alg_mesh) const; // compute edge/vertex weights

  template<class TLAM>
  void FillAlgMesh(TMESH const &alg_mesh, TLAM weightLambda);

  /**
   * Facet flow (divergence constraint):
   *   - flow of facet-BF
   *   - facet surf area
   */
  Array<IVec<2, double>> CalcFacetFlows () const; // TODO: has to change for eps-eps // TODO: give this a lambda that sets the val directly?

  /** Embedding **/
  virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (BaseAMGFactory::AMGLevel& finest_level) override;

  /** Finest level initializing **/
  shared_ptr<MeshDOFs> BuildMeshDOFs (shared_ptr<TMESH> const &fMesh);
  // shared_ptr<PreservedVectors> BuildPreservedVectors (BaseDOFMapStep const *embedding);

  BaseAMGFactory&
  GetBaseFactory () const override;

  SecondaryAMGSequenceFactory const&
  GetSecondaryAMGSequenceFactory () const override;

  TFACTORY &GetFactory () const;

  virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;

  UniversalDofs const &GetHDivUDofs() const { return hDivUDofs; }

  // virtual bool SupportsBlockSmoother(const BaseAMGFactory::AMGLevel &aMGLevel) override;
  virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) override;

  virtual Table<int> GetFinestLevelGSBlocks ();

  // std::tuple<Array<shared_ptr<BaseAMGFactory::AMGLevel>>,
  //            Array<shared_ptr<BaseDOFMapStep>>,
  //            shared_ptr<DOFMap>>
  // CreateSecondaryAMGSequence(FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>>        amgLevels,
  //                            DOFMap                                          const &dOFMap) const override;

  virtual Array<shared_ptr<BaseSmoother>>
  BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
                  shared_ptr<DOFMap> dof_map) override;

  void
  OptimizeDOFMap (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> aMGLevels,
                  shared_ptr<DOFMap>                              dOFMap) override;

  // TODO: OVERLOAD GetFreeDofs! I don't necessarily need it RIGHT NOW because free_nodes on coarse levels is always nullptr!

protected:

  mutable shared_ptr<TFACTORY> _factory;
  // HDivHighOrderFESpace const &GetHDivSpace() const { return _hDivHDGEmbedding->GetHDivFESpace(); }

  shared_ptr<FacetAuxiliaryInformation> _facetAuxInfo;
  unique_ptr<HDivHDGEmbedding>          _hDivHDGEmbedding;

  // shared_ptr<FESpace>                   hDivFES;
  UniversalDofs                         hDivUDofs;
  shared_ptr<PreservedVectors>          preservedVectors;

  // Array<shared_ptr<BaseVector>> preservedVecs; // vectors in FESpace!

  using BaseStokesAMGPrecond::finest_mat;
  shared_ptr<BaseMatrix> finest_weight_mat;

public:
  // for pros-vecs python-binding (debugging feature)
  Array<Array<shared_ptr<BaseVector>>> stashedPresVecs;
  shared_ptr<BaseDOFMapStep>           stashedEmb;

  // I need the surfs from BuildEmbedding, so I am saving that stuff here FOR NOW! TODO: clean up somehow, I dont want that stuff left over!
  // Array<IVec<2, double>> savedFacetFlows; // indexed by ffnrs!
}; // class StokesAMGPC


/** END HDivStokesAMGPC **/


} // namespace amg

#endif // FILE_NC_STOKES_PC_HPP
