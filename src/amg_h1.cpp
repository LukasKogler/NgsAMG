#define FILE_AMGH1_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{

  H1AMG :: H1AMG (shared_ptr<H1AMG::TMESH> mesh,  shared_ptr<H1AMG::Options> opts)
    : VWiseAMG<H1AMG, H1AMG::TMESH, double>(mesh, opts)
  { name = "H1AMG"; }

  shared_ptr<BaseSmoother> H1AMG :: BuildSmoother  (INT<3> level, shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> par_dofs,
						    shared_ptr<BitArray> free_dofs)
  {
    shared_ptr<const TSPMAT> spmat = dynamic_pointer_cast<TSPMAT> (mat);
    return make_shared<HybridGSS<1>> (spmat, par_dofs, free_dofs);
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


  // void H1AMG :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts, INT<3> level, shared_ptr<H1AMG::TMESH> _mesh)
  // {
  //   const TMESH & mesh(*_mesh);
  //   auto NV = mesh.GetNN<NT_VERTEX>();
  //   auto vdata = get<0>(mesh.Data()); vdata->Cumulate();
  //   auto vwts = vdata->Data();
  //   auto NE = mesh.GetNN<NT_EDGE>();
  //   auto edata = get<1>(mesh.Data()); edata->Cumulate();
  //   auto ewts = edata->Data();
  //   Array<double> vcw(NV);
  //   auto econ = mesh.GetEdgeCM();
  //   auto & eqc_h = *mesh.GetEQCHierarchy();
  //   auto neqcs = eqc_h.GetNEQCS();
  //   for (auto eqc : Range(neqcs)) {
  //     if (!eqc_h.IsMasterOfEQC(eqc)) continue;
  //     auto lam_es = [&](auto the_edges) {
  // 	for (const auto & edge : the_edges) {
  // 	  auto ew = ewts[edge.id];
  // 	  vcw[edge.v[0]] += ew;
  // 	  vcw[edge.v[1]] += ew;
  // 	}
  //     };
  //     lam_es(mesh.GetENodes<NT_EDGE>(eqc));
  //     lam_es(mesh.GetCNodes<NT_EDGE>(eqc));
  //   }
  //   mesh.AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
  //   vcw += vwts;
  //   Array<double> ecw(NE);
  //   for (const auto & edge : mesh.GetNodes<NT_EDGE>()) {
  //     double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
  //     ecw[edge.id] = ewts[edge.id] / vw;
  //   }
  //   for (auto k : Range(NV)) vcw[k] = vwts[k] / vcw[k];
  //   opts->vcw = move(vcw);
  //   opts->ecw = move(ecw);
  // }

  template<> shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMG> :: BuildEmbedding ()
  {
    auto & vsort = node_sort[NT_VERTEX];
    auto pmap = make_shared<ProlMap<SparseMatrix<double>>>(fes->GetParallelDofs(), nullptr);
    pmap->SetProl(BuildPermutationMatrix<double>(vsort));
    return pmap;
  }

} // namespace amg

#include "amg_tcs.hpp"

