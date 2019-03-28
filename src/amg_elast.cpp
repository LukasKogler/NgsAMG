#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{
  template class ElasticityAMG<2>;
  template class ElasticityAMG<3>;

  template<int D> shared_ptr<ElasticityMesh<D>>
  Hack_BuildAlgMesh (Array<Vec<3,double>> && vp, shared_ptr<BlockTM> top_mesh)
  {
    // Array<Vec<3,double>> vp = move(amg->node_pos[NT_VERTEX]); // dont keep this
    cout << "v-pos: " << vp.Size() << " " << top_mesh->GetNN<NT_VERTEX>() << endl << vp << endl;
    Array<PosWV> pwv(top_mesh->GetNN<NT_VERTEX>());
    for (auto k : Range(pwv.Size())) {
      pwv[k].wt = 0.0;
      pwv[k].pos = vp[k];
    }
    auto a = new ElVData(move(pwv), CUMULATED);
    Array<ElEW<D>> we(top_mesh->GetNN<NT_EDGE>()); we = 1.0;
    auto b = new ElEData<D>(move(we), CUMULATED);
    auto mesh = make_shared<ElasticityMesh<D>>(move(*top_mesh), a, b);
    return mesh;
  }

  template<> shared_ptr<ElasticityMesh<2>>
  EmbedVAMG<ElasticityAMG<2>> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  { return Hack_BuildAlgMesh<2>(move(node_pos[NT_VERTEX]), top_mesh); }

  template<> shared_ptr<ElasticityMesh<3>>
  EmbedVAMG<ElasticityAMG<3>> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  { return Hack_BuildAlgMesh<3>(move(node_pos[NT_VERTEX]), top_mesh); }

  template<> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<ElasticityAMG<2>> :: BuildEmbedding ()
  {
    auto & vsort = node_sort[NT_VERTEX];
    auto permat = BuildPermutationMatrix<Mat<3,3,double>>(vsort);
    return make_shared<ProlMap<SparseMatrix<Mat<3,3,double>>>>(permat, fes->GetParallelDofs(), nullptr);
  }

  template<> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<ElasticityAMG<3>> :: BuildEmbedding ()
  {
    auto & vsort = node_sort[NT_VERTEX];
    auto permat = BuildPermutationMatrix<Mat<6,6,double>>(vsort);
    return make_shared<ProlMap<SparseMatrix<Mat<6,6,double>>>>(permat, fes->GetParallelDofs(), nullptr);
  }



} // namespace amg

#include "amg_tcs.hpp"
