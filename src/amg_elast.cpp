#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{
  template class ElasticityAMG<2>;
  template class ElasticityAMG<3>;

  template<int D> shared_ptr<ElasticityMesh<D>>
  Hack_BuildAlgMesh (const EmbedVAMG<ElasticityAMG<D>>* amg, shared_ptr<BlockTM> top_mesh)
  { return nullptr; }

  template<> shared_ptr<ElasticityMesh<2>>
  EmbedVAMG<ElasticityAMG<2>> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  { return Hack_BuildAlgMesh(this, top_mesh); }

  template<> shared_ptr<ElasticityMesh<3>>
  EmbedVAMG<ElasticityAMG<3>> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  { return Hack_BuildAlgMesh(this, top_mesh); }

  template<int D>
  void ElasticityAMG<D> :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts,
						 INT<3> level, shared_ptr<ElasticityMesh<D>> mesh)
  { ; }

  template<> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<ElasticityAMG<2>> :: BuildEmbedding ()
  { return nullptr; }

  template<> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<ElasticityAMG<3>> :: BuildEmbedding ()
  { return nullptr; }



} // namespace amg

#include "amg_tcs.hpp"
