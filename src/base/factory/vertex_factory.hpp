#ifndef FILE_AMG_FACTORY_VERTEX_HPP
#define FILE_AMG_FACTORY_VERTEX_HPP

#include "nodal_factory.hpp"

namespace amg
{

/** DOFs attached to vertices, edge-wise energy. **/
class VertexAMGFactoryOptions;

template<class AENERGY, class ATMESH, int ABS>
class VertexAMGFactory : public NodalAMGFactory<NT_VERTEX, ATMESH, ABS>
{
public:
  using ENERGY = AENERGY;
  using TMESH = ATMESH;
  static constexpr int BS = ABS;

  using BASE_CLASS   = NodalAMGFactory<NT_VERTEX, TMESH, BS>;
  using TM           = typename ENERGY::TM;
  using TSPM_TM      = SparseMat<BS, BS>;
  using TSPM         = SparseMat<BS, BS>;
  using AMGLevel     = typename BaseAMGFactory::AMGLevel;
  using Options      = VertexAMGFactoryOptions;
  using LevelCapsule = BaseAMGFactory::LevelCapsule;
  using State        = BaseAMGFactory::State;

protected:
  using BaseAMGFactory::options;

public:
  VertexAMGFactory (shared_ptr<Options> opts);

  ~VertexAMGFactory ();

protected:

  /** State **/
  virtual BaseAMGFactory::State* AllocState () const override;
  virtual void InitState (State & state, shared_ptr<AMGLevel> & lev) const override;

  /** Coarse **/
  virtual shared_ptr<BaseCoarseMap> BuildCoarseMap (State & state, shared_ptr<LevelCapsule> & mapped_cap) override;
  virtual shared_ptr<BaseCoarseMap> BuildPlateTestAggMap (State & state, shared_ptr<LevelCapsule> & mapped_cap);
#ifdef MIS_AGG
  virtual shared_ptr<BaseCoarseMap> BuildMISAggMap (State & state, shared_ptr<LevelCapsule> & mapped_cap);
#endif // MIS_AGG
#ifdef SPW_AGG
  virtual shared_ptr<BaseCoarseMap> BuildSPWAggMap (State & state, shared_ptr<LevelCapsule> & mapped_cap);
#endif // SPW_AGG
  virtual shared_ptr<BaseDOFMapStep> BuildCoarseDOFMap (shared_ptr<BaseCoarseMap> cmap,
                                                        shared_ptr<LevelCapsule> fcap,
                                                        shared_ptr<LevelCapsule> ccap) override;

  shared_ptr<ProlMap<TM>>
  PWProlMap (BaseCoarseMap const &cmap,
             LevelCapsule  const &fcap,
             LevelCapsule  const &ccap);

  shared_ptr<BaseDOFMapStep>
  AuxSProlMap (shared_ptr<BaseDOFMapStep> pw_step,
               shared_ptr<BaseCoarseMap>  cmap,
               shared_ptr<LevelCapsule>   fcap);

  shared_ptr<BaseDOFMapStep>
  SemiAuxSProlMap (shared_ptr<ProlMap<TM>>   pw_step,
                   shared_ptr<BaseCoarseMap> cmap,
                   shared_ptr<LevelCapsule>  fcap);

  shared_ptr<ProlMap<TM>>
  GroupWiseSProl (BaseCoarseMap &cmap,
                  LevelCapsule  &fcap,
                  LevelCapsule  &ccap);


}; // VertexAMGFactory

} // namespace amg

#endif
