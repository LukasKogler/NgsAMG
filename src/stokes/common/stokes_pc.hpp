#ifndef FILE_STOKES_PC_HPP
#define FILE_STOKES_PC_HPP

#include <base.hpp>
#include <amg_pc.hpp>

#include "stokes_mesh.hpp"
#include "stokes_factory.hpp"

namespace amg
{

/** BaseStokesAMGPrecond **/

/**
 * Base class for "stokes-type" preconditioners, it can set up smoothers
 * and convert from MeshAccess to a "dual" algebraic mesh.
 * The class does not know about the concrete mesh-class or type of FESpace
 * and defers computing anything that needs knowledge of that to derived classes.
 * It nonetheless provides utility methods for setting up facet-loops,
 * computing ghost-vertices and computing vertex-volumes.
 */
class BaseStokesAMGPrecond : public BaseAMGPC
{
public:
  struct SmootherContext
  {
    enum Scope : char
    {
      GLOBAL    = 0,
      PRIMARY   = 1,
      SECONDARY = 2,
    };

    enum Space : char
    {
      OUTER = 0,
      RANGE = 1, // range-space
      POT   = 2  // potential
    };

    Scope scope;
    Space space;
  };

public:
  class Options;

  BaseStokesAMGPrecond (shared_ptr<BilinearForm>        blf,
                        Flags                    const &flags,
                        string                   const &name,
                        shared_ptr<Options>             opts = nullptr);

  // set up from assembled matrix
  BaseStokesAMGPrecond (shared_ptr<FESpace>           fes,
                        Flags                        &flags,
                        string                 const &name,
                        shared_ptr<Options>           opts = nullptr);

  virtual ~BaseStokesAMGPrecond () = default;

  // virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level) override;
  virtual void FinalizeLevel (shared_ptr<BaseMatrix> mat) override;

  virtual FacetAuxiliaryInformation const &GetFacetAuxInfo() const = 0;

  INLINE int EL2V(int k) const { return EL2VMap[k]; }
  INLINE int F2E(int k)  const { return F2EMap[k]; }

protected:

  using BaseAMGPC::options;

  /** has to be overloaded **/
  virtual void Update () override { ; } // TODO: what should this do/would it be useful ??

  /** Options **/
  // virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
  virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
  // virtual void SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix = "ngs_amg_") override;
  virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

  /** Mesh initialization  **/
  virtual shared_ptr<TopologicMesh> BuildInitialMesh () override;
  virtual tuple<shared_ptr<BlockTM>, Array<FVDescriptor>> BuildTopMesh ();
  virtual shared_ptr<TopologicMesh> BuildAlgMesh (shared_ptr<BlockTM> &&top_mesh, FlatArray<FVDescriptor> fvd) = 0;

  shared_ptr<BitArray> BuildGhostVerts (BlockTM &fineMesh, FlatArray<FVDescriptor> fvd);

  /**
   * Computes Facet-loops from the Netgen mesh and returns them
   * as lists of edges. These lists are not sorted in any way.
   */
  std::tuple<Table<int>, Table<int>> CalcFacetLoops (BlockTM const &topMesh, BitArray const &ghostVerts) const;

  /**
   * For the finest level, computes:
   *    (i) free vertices:
   *          Fictitious vertices added on Dirichlet boundaries are dropped by the coarsening.
   *   (ii) free edges:
   *          CURRENTLY they are used only for the FIRST prolongation
   *          POTENTIALLY, could also be used for subsequent ones, if we keep the "Dirichlet"
   *            fictitious vertices around. I am not sure whether that would be beneficial, but it
   *            could make coarse grid base functions fulfill the Dirichlet conditions in a "smoother"
   *            manner because the connection to the Dirichlet facet would then play a role in the
   *            local energy minimization.
   *  (iii) free pot-space nodes:
   *          In parallel, we don't necessarily know which LOCAL loops might touch a Dirichlet
   *          boundary on a neighbor. We are adding all loops, including those that touch a
   *          Dirichlet boundary to the mesh and need need to set those to Dirichlet in
   *          the potential space smoother.
   */
  std::tuple<shared_ptr<BitArray>, shared_ptr<BitArray>, shared_ptr<BitArray>>
  BuildFreeNodes(BlockTM const &fmesh,
                 FlatTable<int> loops,
                 UniversalDofs const &loop_uDofs);

  /** computes volumes of solid vertices, does nothing to ghosts */
  template<class TLAM> void CalcVertexVols (BlockTM &fineMesh, FlatArray<FVDescriptor> fvd, TLAM set_vol);

  /** Smoothers **/
  virtual Array<shared_ptr<BaseSmoother>>
  BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
                  shared_ptr<DOFMap> dof_map) override;

  /** For HDIV dyn-block**/
  virtual void
  OptimizeDOFMap (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> aMGLevels,
                  shared_ptr<DOFMap> dOFMap);

  virtual shared_ptr<BaseSmoother>
  BuildSmoother (BaseAMGFactory::AMGLevel const &amg_level) override;


  /**
   * Build a smoother for either
   *     - the "primary"   space: RT0+optionally pres.constants, or NC
   *     - the "secondary" space: RT0
   */
  virtual shared_ptr<BaseSmoother>
  BuildSmoother (BaseAMGFactory::AMGLevel const &aMGLevel,
                 SmootherContext          const &smootherCtx);

  virtual shared_ptr<BaseSmoother>
  BuildSmootherConcrete (BaseAMGPC::Options::SM_TYPE const &smType,
                         shared_ptr<BaseMatrix>             A,
                         shared_ptr<BitArray>               freeDofs,
                         std::function<Table<int>()>        getBlocks = [](){ return Table<int>(); });


  virtual BaseAMGPC::Options::SM_TYPE
  SelectSmoother(BaseAMGFactory::AMGLevel const &amgLevel) const override;

  virtual BaseAMGPC::Options::SM_TYPE
  SelectSmoother(BaseAMGFactory::AMGLevel const &amgLevel,
                 SmootherContext          const &smootherCtx) const;

  /** Hiptmair Smoother **/
  // virtual shared_ptr<BaseSmoother> BuildHiptMairSmoother (const BaseAMGFactory::AMGLevel & amg_level, shared_ptr<BaseSmoother> smoother);

  /** V-cycle of **/

  virtual Table<int> GetPotentialSpaceGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) = 0;

  /** Utility **/
  INLINE void GetDefinedFacetElements (int facnr, Array<int> & elnums) const;

protected:

  virtual SecondaryAMGSequenceFactory const&
  GetSecondaryAMGSequenceFactory () const = 0;

  virtual
  std::tuple<Array<shared_ptr<BaseAMGFactory::AMGLevel>>,
             Array<shared_ptr<BaseDOFMapStep>>,
             shared_ptr<DOFMap>>
  CreateSecondaryAMGSequence(FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>>        amgLevels,
                             DOFMap                                          const &dOFMap) const;


  FESpace const &GetFESpace() const { return *_fes; }
  FESpace       &GetFESpace()       { return *_fes; }

  // MeshAccess const &getMA () const { return *GetFESpace().GetMeshAccess(); }
  // MeshAccess       &getMA ()       { return *GetFESpace().GetMeshAccess(); }

  // this cannot return MeshAccess const& because MeshAccess has a couple of methods that
  // should be marked const but are not, so we could not call them otherwise
  MeshAccess &GetMA () const { return *GetFESpace().GetMeshAccess(); }

  shared_ptr<FESpace> const &GetFESpacePtr() const { return _fes; }
  shared_ptr<FESpace>       &GetFESpacePtr()       { return _fes; }

  Array<int> EL2VMap;  // MESH-VOL-ELMNT  -> VERTEX
  Array<int> F2EMap;   // MESH-FACET      -> EDGE

  shared_ptr<BaseVector> presVecs;

  shared_ptr<FESpace> _fes;
}; // class BaseStokesAMGPrecond

/** END BaseStokesAMGPrecond **/


/** StokesAMGPrecond **/

/**
 * This class brings in the concrete mesh. It implements setup of the
 * (algebraic) mesh without computing energy-related node-data!
 *
 * This leaves derived classes to implement:
 *     i) computing the energy-related node-data
 *    ii) the embedding
 *   iii) facet flows (depends on the FESpace)
 */
template<class TMESH>
class StokesAMGPrecond : public BaseStokesAMGPrecond
{
public:

  StokesAMGPrecond (shared_ptr<BilinearForm>      blf,
                    Flags                  const &flags,
                    string                 const &name,
                    shared_ptr<Options>           opts = nullptr);

  // set up from assembled matrix
  StokesAMGPrecond (shared_ptr<FESpace>           fes,
                    Flags                        &flags,
                    string                 const &name,
                    shared_ptr<Options>           opts = nullptr);

  virtual Table<int> GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) override;

  virtual Table<int> GetPotentialSpaceGSBlocks (const BaseAMGFactory::AMGLevel & amg_level) override;

protected:
  shared_ptr<TMESH> BuildEmptyAlgMesh(shared_ptr<BlockTM> &&blockTM, FlatArray<FVDescriptor> fvd);

  void CalcLoops (TMESH &fineMesh) const;
}; // class StokesAMGPrecond

/** END StokesAMGPrecond **/


/** Options **/

class BaseStokesAMGPrecond :: Options : public BaseAMGPC::Options
{
public:
  using BaseAMGPC::Options::SM_TYPE;

  // bool hiptmair = true;    // Use Hiptmair Smoother
  SpecOpt<SM_TYPE> sm_type_pot   = SM_TYPE::GS;
  SpecOpt<SM_TYPE> sm_type_range = SM_TYPE::GS;
  SpecOpt<bool>    amg_sm_single_fls = true;

  virtual void SetFromFlags (shared_ptr<FESpace> fes,
                             shared_ptr<BaseMatrix> finest_mat,
                             const Flags & flags,
                             string prefix) override
  {
    // TFACTORY::Options::SetFromFlags(flags, prefix);
    BaseAMGPC::Options::SetFromFlags(fes, finest_mat, flags, prefix);

    // sm_type -> global scope

    // sm_type_primary   (used if sm_type is AMG)
    // sm_type_secondary (used is sm_type is AMG)

    sm_type_range.SetFromFlagsEnum(flags,
                                   prefix + "sm_type_range",
                                   { "gs", "bgs", "jacobi", "hiptmair", "amg_smoother", "dyn_block_gs" },
                                   { GS, BGS, JACOBI, HIPTMAIR, AMGSM, DYNBGS });

    sm_type_pot.SetFromFlagsEnum(flags,
                                 prefix + "sm_type_pot",
                                 { "gs", "bgs", "jacobi", "hiptmair", "amg_smoother", "dyn_block_gs" },
                                 { GS, BGS, JACOBI, HIPTMAIR, AMGSM, DYNBGS });

    amg_sm_single_fls.SetFromFlags(flags, prefix + "amg_sm_single_fls");

    // hiptmair = !flags.GetDefineFlagX(prefix + "hpt_sm").IsFalse();

    // hiptmair_blk = flags.GetDefineFlagX(prefix + "hpt_sm_blk").IsTrue();
    // hiptmair_ex = flags.GetDefineFlagX(prefix + "hpt_sm_ex").IsTrue();
    // hiptmair_steps = flags.GetNumFlag(prefix + "hpt_sm_symm", 1);
    // hiptmair_symm = flags.GetDefineFlagX(prefix + "hpt_sm_symm").IsTrue();
  }
}; // class BaseStokesAMGPrecond::Options

INLINE std::ostream &operator<<(std::ostream &os, BaseStokesAMGPrecond::SmootherContext const &ctx)
{
  os << "(";
  switch(ctx.scope)
  {
    case(BaseStokesAMGPrecond::SmootherContext::GLOBAL):    { os << "GLOB";    break; }
    case(BaseStokesAMGPrecond::SmootherContext::PRIMARY):   { os << "PRIM";   break; }
    case(BaseStokesAMGPrecond::SmootherContext::SECONDARY): { os << "SECO"; break; }
  }
  os << "-";
  switch(ctx.space)
  {
    case(BaseStokesAMGPrecond::SmootherContext::OUTER): { os << "OUT"; break; }
    case(BaseStokesAMGPrecond::SmootherContext::RANGE): { os << "RAN"; break; }
    case(BaseStokesAMGPrecond::SmootherContext::POT):   { os << "POT"; break; }
  }
  os << ")";
  return os;
}

/** END Options **/

} // namespace amg

#endif // FILE_STOKES_PC_HPP