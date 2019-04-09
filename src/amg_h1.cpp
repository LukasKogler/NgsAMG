#define FILE_AMGH1_CPP
#include "amg.hpp"

#include "amg_coarsen_impl.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{

  H1AMG :: H1AMG (shared_ptr<H1AMG::TMESH> mesh,  shared_ptr<H1AMG::Options> opts)
    : VWiseAMG<H1AMG, H1AMG::TMESH, double>(mesh, opts)
  { name = "H1AMG"; }

  void H1AMG :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts, INT<3> level, shared_ptr<H1AMG::TMESH> _mesh)
  {
    const TMESH & mesh(*_mesh);
    auto NV = mesh.GetNN<NT_VERTEX>();
    auto vdata = get<0>(mesh.Data()); vdata->Cumulate();
    auto vwts = vdata->Data();

    auto NE = mesh.GetNN<NT_EDGE>();
    auto edata = get<1>(mesh.Data()); edata->Cumulate();
    auto ewts = edata->Data();

    Array<double> vcw(NV); vcw = 0;
    auto econ = mesh.GetEdgeCM();
    auto & eqc_h = *mesh.GetEQCHierarchy();
    auto neqcs = eqc_h.GetNEQCS();
    for (auto eqc : Range(neqcs)) {
      if (!eqc_h.IsMasterOfEQC(eqc)) continue;
      auto lam_es = [&](auto the_edges) {
	for (const auto & edge : the_edges) {
	  auto ew = ewts[edge.id];
	  vcw[edge.v[0]] += ew;
	  vcw[edge.v[1]] += ew;
	}
      };
      lam_es(mesh.GetENodes<NT_EDGE>(eqc));
      lam_es(mesh.GetCNodes<NT_EDGE>(eqc));
    }
    // cout << "unreduced vwts: " << endl; prow2(vwts); cout << endl;
    mesh.AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
    // cout << "reduced vwts: " << endl; prow2(vwts); cout << endl;
    vcw += vwts;
    Array<double> ecw(NE);
    for (const auto & edge : mesh.GetNodes<NT_EDGE>()) {
      double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
      ecw[edge.id] = ewts[edge.id] / vw;
      // double vw0 = vcw[edge.v[0]];
      // double vw1 = vcw[edge.v[1]];
      // ecw[edge.id] = ewts[edge.id] * (vw0 + vw1) / (vw0 * vw1);
    }
    for (auto k : Range(NV)) vcw[k] = vwts[k] / vcw[k];
    // opts->vcw = Array<double>(NV, &vcw[0]); vcw.NothingToDelete();
    // opts->ecw = Array<double>(NE, &ecw[0]); ecw.NothingToDelete();

    // cout << "ecw: " << endl; prow(ecw); cout << endl;
    // cout << "vcw: " << endl; prow(vcw); cout << endl;

    opts->vcw = move(vcw);
    opts->ecw = move(ecw);

    // opts->vcw.SetSize(mesh->template GetNN<NT_VERTEX>()); opts->vcw = 0.0;
    // opts->ecw.SetSize(mesh->template GetNN<NT_EDGE>()); opts->ecw = 1.0;
  }

  template<> shared_ptr<EmbedVAMG<H1AMG>::TMESH>
  EmbedVAMG<H1AMG> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED); a->Data() = 0.0;
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); b->Data() = 1.0;
    auto mesh = make_shared<H1AMG::TMESH>(move(*top_mesh), a, b);
    // cout << "finest mesh: " << endl << *mesh << endl;
    return mesh;
  }

  template<> shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMG> :: BuildEmbedding ()
  {
    auto & vsort = node_sort[NT_VERTEX];
    size_t NV = vsort.Size();
    Array<int> epr(NV); epr = 1.0;
    auto embed_mat = make_shared<SparseMatrix<double>>(epr, NV);
    const auto & em = *embed_mat;
    for (auto k : Range(NV)) {
      em.GetRowIndices(k)[0] = vsort[k];
      em.GetRowValues(k)[0] = 1.0;
    }
    // cout << "embedding mat: " << endl << *embed_mat << endl;
    return make_shared<ProlMap<SparseMatrix<double>>> (embed_mat, fes->GetParallelDofs(), nullptr);
  }


  template class EmbedVAMG<H1AMG>;
} // namespace amg
