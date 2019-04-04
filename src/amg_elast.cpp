#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{

  INLINE Timer & timer_hack_Hack_BuildAlgMesh () { static Timer t("ElasticityAMG::BuildAlgMesh"); return t; }
  template<class C> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    Timer & t(timer_hack_Hack_BuildAlgMesh()); RegionTimer rt(t);
    FlatArray<Vec<3,double>> vp = node_pos[NT_VERTEX];
    Array<PosWV> pwv(top_mesh->GetNN<NT_VERTEX>());
    for (auto k : Range(pwv.Size())) {
      pwv[k].wt = 0.0;
      pwv[k].pos = vp[k];
    }
    auto a = new ElVData(move(pwv), CUMULATED);
    Array<ElEW<C::DIM>> we(top_mesh->GetNN<NT_EDGE>());
    for (auto & x : we) { SetIdentity(x.bend_mat()); SetIdentity(x.wigg_mat()); }
    auto b = new ElEData<C::DIM>(move(we), CUMULATED);
    auto mesh = make_shared<ElasticityMesh<C::DIM>>(move(*top_mesh), a, b);
    return mesh;
  }

  template<class C> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<C> :: BuildEmbedding ()
  {
    static Timer t(this->name+string("::BuildEmbedding")); RegionTimer rt(t);
    auto & vsort = node_sort[NT_VERTEX];
    bool need_mat = false;
    for (int k : Range(vsort.Size()))
      if (vsort[k]!=k) { need_mat = true; break; }
    if (need_mat == false) return nullptr;
    if (options->v_dofs == "NODAL") {
      if (options->block_s.Size() == 1 ) { // ndof/vertex != #kernel vecs
	if (options->block_s[0] != disppv(C::DIM)) {
	  // there is really only multidim=dofpv(D) and multidim=disppv(D) that make sense here...
	  throw Exception("This should not happen ... !");
	}
	using TESM = Mat<disppv(C::DIM), dofpv(C::DIM)>;
	auto pmap = make_shared<ProlMap<SparseMatrix<TESM>>> (fes->GetParallelDofs(), nullptr);
	pmap->SetProl(BuildPermutationMatrix<TESM>(vsort));
	return pmap;
      }
      else if (options->block_s.Size() > 1) {
	using TESM = Mat<1, dofpv(C::DIM)>;
	auto pmap = make_shared<ProlMap<stripped_spm<TESM>>> (fes->GetParallelDofs(), nullptr);
	pmap->SetProl(BuildPermutationMatrix<TESM>(vsort));
	throw Exception("Compound FES embedding not implemented, sorry!");
	return nullptr;
      }
      else { // ndof/vertex != #kernel vecs (so we have rotational DOFs)
	auto pmap = make_shared<ProlMap<SparseMatrix<typename C::TMAT>>>(fes->GetParallelDofs(), nullptr);
	pmap->SetProl(BuildPermutationMatrix<typename C::TMAT>(vsort));
	return pmap;
      }
    }
    else {
      throw Exception("variable dofs embedding not implemented, sorry!");
      return nullptr;
    }
  }

  // template class ElasticityAMG<2>;
  // template class ElasticityAMG<3>;

} // namespace amg

#include "amg_tcs.hpp"
