#ifndef FILE_NC_STOKES_PC_HPP
#define FILE_NC_STOKES_PC_HPP

#include "stokes_pc.hpp"
#include "ncfespace.hpp"

namespace amg
{

/** NCStokesAMGPC **/

/** Stokes-AMG Preconditioner for non-conforming H1 space **/
template<class ATFACTORY>
class NCStokesAMGPC : public StokesAMGPrecond<typename ATFACTORY::TMESH>
{
public:
  using TFACTORY = ATFACTORY;
  using TMESH = typename TFACTORY::TMESH;
  static constexpr int BS = TFACTORY::BS;

  class Options : public TFACTORY::Options,
                  public BaseStokesAMGPrecond::Options
  {
  public:

    virtual void SetFromFlags (shared_ptr<FESpace> fes,
                               shared_ptr<BaseMatrix> finest_mat,
                               const Flags & flags,
                               string prefix) override
    {
      TFACTORY::Options::SetFromFlags(flags, prefix);
      BaseStokesAMGPrecond::Options::SetFromFlags(fes, finest_mat, flags, prefix);
    }
  }; // class BaseStokesAMGPrecond::Options

public:

  NCStokesAMGPC (shared_ptr<BilinearForm>      blf,
                 Flags                  const &flags,
                 string                 const &name,
                 shared_ptr<Options>           opts = nullptr,
                 shared_ptr<BaseMatrix>        weight_mat  = nullptr);

  // set up from assembled matrix
  NCStokesAMGPC (shared_ptr<FESpace>           fes,
                 Flags                        &flags,
                 string                 const &name,
                 shared_ptr<Options>           opts = nullptr,
                 shared_ptr<BaseMatrix>        weight_mat  = nullptr);

  ~NCStokesAMGPC () = default;

  virtual void FinalizeLevel (shared_ptr<BaseMatrix> mat) override;

protected:

  virtual BaseAMGFactory&
  GetBaseFactory () const override;

  SecondaryAMGSequenceFactory const&
  GetSecondaryAMGSequenceFactory () const override;

  virtual TFACTORY &GetFactory () const;

  /** Options **/
  virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
  virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
  virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
  virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

  /** Mesh initialization  **/
  virtual shared_ptr<TopologicMesh> BuildAlgMesh (shared_ptr<BlockTM> &&top_mesh, FlatArray<FVDescriptor> fvd) override;
  // virtual shared_ptr<TMESH> FillAlgMesh_TRIV (TMESH & alg_mesh) const; // compute edge/vertex weights
  // virtual shared_ptr<TMESH> FillAlgMesh_ALG  (TMESH & alg_mesh) const; // compute edge/vertex weights

  template<class TLAM>
  void FillAlgMesh(TMESH const &alg_mesh, TLAM weightLambda);

  /** Facet flow (divergence constraint) **/
  Array<Vec<BS, double>> CalcFacetFlows () const; // TODO: has to change for eps-eps // TODO: give this a lambda that sets the val directly?

  /** Embedding **/
  virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (BaseAMGFactory::AMGLevel& finest_level) override;

  /** Finest level initializing **/
  virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;

  virtual FacetAuxiliaryInformation const & GetFacetAuxInfo() const override;

  UniversalDofs const &GetNCUDofs() const { return nc_uDofs; }

  // std::tuple<Array<shared_ptr<BaseAMGFactory::AMGLevel>>,
  //            Array<shared_ptr<BaseDOFMapStep>>,
  //            shared_ptr<DOFMap>>
  // CreateSecondaryAMGSequence(FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>>        amgLevels,
  //                            DOFMap                                          const &dOFMap) const override;

protected:
  using BaseStokesAMGPrecond::options;
  using BaseStokesAMGPrecond::F2E;
  using BaseStokesAMGPrecond::EL2V;

  mutable shared_ptr<TFACTORY> _factory;

  shared_ptr<NoCoH1FESpace> nc_fes;
  UniversalDofs nc_uDofs;

  using BaseStokesAMGPrecond::finest_mat;
  shared_ptr<BaseMatrix> finest_weight_mat;
}; // class StokesAMGPC


/** END NCStokesAMGPC **/


} // namespace amg

#endif // FILE_NC_STOKES_PC_HPP
