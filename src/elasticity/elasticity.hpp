#ifdef ELASTICITY

#ifndef FILE_AMG_ELAST_HPP
#define FILE_AMG_ELAST_HPP

#include <alg_mesh.hpp>
#include <base_coarse.hpp>
#include <agglomerate_map.hpp>

#include <vertex_factory.hpp>
#include <vertex_factory_impl.hpp>

#include "elasticity_mesh.hpp"

#include <amg_register.hpp>

namespace amg
{

template<int DIM>
class AttachedEVD : public AttachedNodeData<NT_VERTEX, ElastVData<DIM>>
{
public:
  static constexpr NODE_TYPE TNODE = NT_VERTEX;
  using BASE = AttachedNodeData<NT_VERTEX, ElastVData<DIM>>;
  // using BASE::map_data, BASE::Cumulate, BASE::mesh, BASE::data;
  using BASE::Cumulate, BASE::mesh, BASE::data;

  AttachedEVD (Array<ElastVData<DIM>> && _data, PARALLEL_STATUS stat)
    : BASE(std::move(_data), stat)
  { ; }

  // I think this was pairwise
  // template<class TMESH> INLINE void map_data (const CoarseMap<TMESH> & cmap, AttachedEVD<DIM> *cevd) const;

  template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedEVD<DIM> *cevd) const;

}; // class AttachedEVD

template<int DIM>
class AttachedEED : public AttachedNodeData<NT_EDGE, ElasticityEdgeData<DIM>>
{
public:
  static constexpr NODE_TYPE TNODE = NT_EDGE;
  using BASE = AttachedNodeData<NT_EDGE, ElasticityEdgeData<DIM>>;
  // using BASE::map_data, BASE::Cumulate, BASE::mesh;
  using BASE::Cumulate, BASE::mesh;
  static constexpr int BS = mat_traits<ElasticityEdgeData<DIM>>::HEIGHT;

  AttachedEED (Array<ElasticityEdgeData<DIM>> && _data, PARALLEL_STATUS stat)
    : BASE(std::move(_data), stat)
  { ; }

  // in impl header beacuse I static_cast to elasticity-mesh
  void map_data (const BaseCoarseMap & cmap, AttachedEED<DIM> *ceed) const;
}; // class AttachedEED

template<int DIM> using ElasticityMesh = BlockAlgMesh<AttachedEVD<DIM>, AttachedEED<DIM>>;

/** Factory **/

template<int ADIM>
class ElasticityAMGFactory : public VertexAMGFactory<EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>,
                  ElasticityMesh<ADIM>,
                  mat_traits<ElasticityEdgeData<ADIM>>::HEIGHT>
{
public:
  static constexpr int DIM = ADIM;
  using ENERGY = EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>;
  using TMESH = ElasticityMesh<DIM>;
  static constexpr int BS = ENERGY::DPV;
  using BASE = VertexAMGFactory<EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>,
        ElasticityMesh<ADIM>, mat_traits<ElasticityEdgeData<ADIM>>::HEIGHT>;

  class Options : public BASE::Options
  {
  public:
    bool with_rots = false;
  }; // class ElasticityAMGFactory::Options

protected:
  using BASE::options;

  virtual shared_ptr<GridContractMap> AllocateContractMap (Table<int> && groups, shared_ptr<TMESH> mesh) const override
  {
    return make_shared<AlgContractMap<TMESH>>(std::move(groups), mesh);
    // return make_shared<H1GridContractMap>(std::move(groups), mesh);
  }

public:
  ElasticityAMGFactory (shared_ptr<Options> _opts)
    : BASE(_opts)
  { ; }

  /** Misc **/
  void CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map) override;

}; // class ElasticityAMGFactory

/** END Factory **/



template<class AMG_CLASS>
class RegisterElasticityAMGSolver
{
public:
  RegisterElasticityAMGSolver(std::string const &name)
    : _name(name)
    , _regPC(name)
  {
    // use rigid bodies here!
    AMGRegister::getAMGRegister().addAMGSolver(
      name,
      [&](NGsAMGMatrixHandle const &aHandle,
          AMGSolverSettings  const &settings) {

      auto pc = make_shared<AMG_CLASS>(aHandle.getMatrix(),
                                       settings.getFlags(),
                                       _name + "FromRegisterElasticityAMGSolver");

      pc->InitializeOptions();

      std::vector<double> const &stdVec = settings.getVertexCoordinates();
      FlatArray<double> fa(stdVec.size(), const_cast<double*>(stdVec.data())); // TODO: clean up
      pc->SetVertexCoordinates(fa);

      if (auto freeScalRows = aHandle.getFreeScalRows())
      {
        pc->SetEmbProjScalRows(freeScalRows);
      }

      return pc;
    });
  }

private:
  std::string _name;
  RegisterPreconditioner<AMG_CLASS> _regPC;
};

} // namespace amg


#endif
#endif
