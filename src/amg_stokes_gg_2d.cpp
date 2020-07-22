#ifdef STOKES

/** 2D, laplace + div div **/

#include "amg.hpp"

#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_nodal_impl.hpp"

#include "amg_stokes.hpp" // need special StokesBCM for the factory
#include "amg_factory_stokes.hpp"

// need to include these before facet_aux
#include "amg_matrix.hpp"
#include "amg_pc.hpp"
#include "amg_blocksmoother.hpp"

#include "amg_facet_aux.hpp"
#include "amg_hdiv_templates.hpp"
#include "amg_vfacet_templates.hpp"
#include "amg_facet_aux_impl.hpp"

#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"

#include "amg_bla.hpp"
#include "amg_agg.hpp"
#include "amg_agg_impl.hpp"

#include "amg_pc_stokes.hpp"
#include "amg_stokes_gg.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_factory_stokes_impl.hpp"
#include "amg_pc_stokes_impl.hpp"


namespace amg
{

  // extern template class SeqVWC<GGStokesMesh<2>>;
  // extern template class BlockVWC<GGStokesMesh<2>>;
  // extern template class HierarchicVWC<GGStokesMesh<2>>;
  extern template class CoarseMap<GGStokesMesh<2>>;
  template class Agglomerator<GGStokesEnergy<2>, GGStokesMesh<2>, GGStokesEnergy<2>::NEED_ROBUST>;
  extern template class CtrMap<Vec<2,double>>;
  extern template class GridContractMap<GGStokesMesh<2>>;
  extern template class VDiscardMap<GGStokesMesh<2>>;

  template class StokesAMGFactory<GGStokesMesh<2>, GGStokesEnergy<2>>;

  using STOKES_FACTORY = StokesAMGFactory<GGStokesMesh<2>, GGStokesEnergy<2>>;

  extern template class FacetAuxSystem<2, HDivHighOrderFESpace, VectorFacetFESpace, FacetH1FE<2>>;

  using AUX_SYS = FacetAuxSystem<2, HDivHighOrderFESpace, VectorFacetFESpace, FacetH1FE<2>>;

  template class StokesAMGPC<STOKES_FACTORY, AUX_SYS>;

  using STOKES_PC = StokesAMGPC<STOKES_FACTORY, AUX_SYS>;

  RegisterPreconditioner<STOKES_PC> reg_stokes_pc ("ngs_amg.stokes_gg_2d");

} // namespace amg


namespace amg
{

  void ExportStokes_gg_2d (py::module & m)
  {
    ExportAuxiliaryAMG<STOKES_PC> (m, "stokes_gg_2d", "Stokes Preconditioner, grad-grad + div-div penalty.",
				   [&](auto & pyclass) {
				     pyclass.def("PoC", [&](shared_ptr<STOKES_PC> spc, int level, int comp, shared_ptr<BaseVector> comp_vec) {
					 auto eam = spc->GetEmbAMGMat();
					 auto am = eam->GetAMGMatrix();
					 auto facet_emb = eam->GetEmbedding();
					 auto map = am->GetMap();
					 auto facet_vec = map->CreateVector(0);
					 int os = (comp == 0) ? 0 : 1;
					 if (level == 1) {
					   auto lvec = map->CreateVector(1);
					   lvec->FVDouble() = 0;
					   for (int k = os; k < lvec->FVDouble().Size(); k += 2)
					     { lvec->FVDouble()[k] = 1; }
					   map->TransferC2F(0, facet_vec.get(), lvec.get());
					 }
					 else {
					   facet_vec->FVDouble() = 0;
					   for (int k = os; k < facet_vec->FVDouble().Size(); k += 2)
					     { facet_vec->FVDouble()[k] = 1; }
					 }
					 facet_emb->TransferC2F(comp_vec.get(), facet_vec.get());
				       });
				     pyclass.def("GetLoop", [&](shared_ptr<STOKES_PC> spc, int level, int comp, shared_ptr<BaseVector> comp_vec) {
					 auto eam = spc->GetEmbAMGMat();
					 auto am = eam->GetAMGMatrix();
					 auto facet_emb = eam->GetEmbedding();
					 auto map = am->GetMap();
					 auto facet_vec = map->CreateVector(0);
					 auto smoothers = am->GetSmoothers();
					 cout << " smoothers level " << smoothers.Size() << " " << level << " " << comp << endl;
					 if (level < smoothers.Size()) {
					   auto hsm = dynamic_pointer_cast<const HiptMairSmoother>(smoothers[level]);
					   if (hsm == nullptr)
					     { cout << " GARBAGE1!" << endl; return; }
					   auto & sm = const_cast<HiptMairSmoother&>(*hsm);
					   auto C = sm.GetD();
					   auto v = C->CreateRowVector();
					   if (comp >= v.FVDouble().Size())
					     { cout << " GARBAGE2!" << endl; return; }
					   v.FVDouble() = 0; v.FVDouble()[comp] = 1;
					   cout << " v: " << endl << *v << endl;
					   if (level > 0)  {
					     auto w = map->CreateVector(level);
					     C->Mult(*v, *w);
					     cout << " w: " << endl << *w << endl;
					     map->TransferAtoB(level, 0, w.get(), facet_vec.get());
					     cout << " facet_vec " << endl << *facet_vec << endl;
					   }
					   else
					     { C->Mult(*v, *facet_vec); }
					   facet_emb->TransferC2F(comp_vec.get(), facet_vec.get());
					 }
					 else
					   { cout << " GARBAGE3!" << endl; return; }
				       });
				     pyclass.def("SetFreeDofs", [&](shared_ptr<STOKES_PC> pc, shared_ptr<BitArray> fds) {
					 pc->InitLevelForced(fds);
				       });
				     pyclass.def("set_hacked_emb", [&](shared_ptr<STOKES_PC> pc, shared_ptr<BaseMatrix> emb1, shared_ptr<BaseMatrix> emb2) {
					 pc->GetAuxSys()->__hacky__set__Pmat(emb1, emb2);
				       }, py::arg("emb1") = nullptr, py::arg("emb2") = nullptr);

				   } );
  } // ExportStokes_gg_2d

} // namespace amg

#endif // STOKES
