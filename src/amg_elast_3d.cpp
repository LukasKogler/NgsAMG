#ifdef ELASTICITY

#define FILE_AMG_ELAST_CPP
#define FILE_AMG_ELAST_3D_CPP

#include "amg.hpp"

#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_nodal_impl.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_factory_vertex_impl.hpp"
#include "amg_pc.hpp"
#include "amg_energy.hpp"
#include "amg_energy_impl.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_pc_vertex_impl.hpp"
#include "amg_elast.hpp"
#include "amg_elast_impl.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

namespace amg
{

  template<>
  void ElasticityAMGFactory<3> :: CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
  {
    auto intr = [&](auto tmmm) {
      cout << endl;
      auto mvd = get<0>(tmmm->Data())->Data();
      cout << " eqc vs: " << endl;
      tmmm->template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
	  if (nodes.Size() > 0)
	    cout << " eqc " << eqc << " = [" << nodes[0] << " ... " << nodes.Last() << "]" << endl;
	  else
	    cout << " eqc " << eqc << " = []" << endl;
	}, false);
      cout << " v data: " << endl;
      for (auto k : Range(mvd))
	{ cout << k << " " << mvd[k] << endl; }
      cout << endl;
    };
    cout << " Check KVECS, meshes" << endl;
    for (auto l : Range(amg_levels)) {
      cout << " level " << l << endl;
      if (auto cap = amg_levels[l]->cap) {
	if (auto mesh = cap->mesh) {
	  intr(static_pointer_cast<TMESH>(mesh));
	}
	else
	  { cout << " no mesh " << endl; }
      }
      else
	{ cout << " no cap " << endl; }
    }
    auto chkab = [&](auto fva, auto fvb, int n, auto fnodes, string title) {
      if ( n == 0)
	{ return; }
      cout << " check " << title << endl;
      int bs = fva.Size() / n;
      int numdf = 0;
      for (auto k : Range(n)) {
	if ( (fnodes==nullptr) || (fnodes->Test(k)) ) {
	  double df = 0;
	  for (auto l : Range(k*bs, (k+1)*bs))
	    { df += sqr(fva(l)-fvb(l)); }
	  df = sqrt(df);
	  if (df > 1e-14) {
	    numdf++;
	    cout << " DIFF " << k << " norm = " << df << ", diff = ";
	    for (auto l : Range(k*bs, (k+1)*bs))
	      { cout << "(" << fva(l) << "/" << fvb(l) << "/" << fva(l)-fvb(l) << ") "; }
	    cout << endl;
	  }
	}
      }
      cout << " done " << title << ", mismates = " << numdf << endl;
      if (fnodes != nullptr)
	{ cout << " fnodes non-set = " << fnodes->Size() - fnodes->NumSet() << endl; }
    };
    auto set_kvec = [&](auto & vec, int kvnr, BaseAMGFactory::AMGLevel & alev) {
      typename ENERGY::TVD opt(0); /** just an arbitrary point ... **/
      typename ENERGY::TM Q; SetIdentity(Q);
      Vec<BS, double> vcos, ovec;
      ovec = 0; ovec(kvnr) = 1;
      auto mesh = static_pointer_cast<TMESH>(alev.cap->mesh);
      auto vdata = get<0>(mesh->Data())->Data();
      vec.SetParallelStatus(CUMULATED);
      vec.FVDouble() = 0;
      cout << " set kvec : " << endl;
      if (mesh->template GetNN<NT_VERTEX>() > 0) {
	// opt = vdata[0]; /// DONT DO THAT !!! compares different kvecs then !!!
	int vbs = vec.FVDouble().Size()/mesh->template GetNN<NT_VERTEX>();
	auto fv = vec.FVDouble();
	for (auto vnr : Range(mesh->template GetNN<NT_VERTEX>())) {
	  ENERGY::CalcQHh(opt, vdata[vnr], Q);
	  vcos = Q * ovec;
	  for (int l = 0; l < vbs; l++)
	    { fv(vbs * vnr + l) = vcos(l); }
	  cout << vnr << " = ";
	  for (int l = 0; l < vbs; l++)
	    cout << fv(l) << " ";
	  cout << endl;
	}
	cout << endl;
      }
    };
    for (int kvnr = 0; kvnr < BS; kvnr++) {
      auto gcomm = amg_levels[0]->cap->eqc_h->GetCommunicator();
      int nlevsglob = gcomm.AllReduce(amg_levels.Size(), MPI_MAX);
      int nlevsloc = map->GetNLevels();
      unique_ptr<BaseVector> cvec = move(map->CreateVector(nlevsloc-1));
      if ( (nlevsloc == nlevsglob) && (cvec != nullptr) )
	{ set_kvec(*cvec, kvnr, *amg_levels[nlevsloc-1]); }
      for (int lev = nlevsloc-2; lev >= 0; lev--) {
	unique_ptr<BaseVector> fvec1 = map->CreateVector(lev), fvec2 = map->CreateVector(lev);
	map->TransferC2F(lev, fvec1.get(), cvec.get());
	set_kvec(*fvec2, kvnr, *amg_levels[lev]);
	chkab(fvec1->FVDouble(), fvec2->FVDouble(), amg_levels[lev]->cap->mesh->template GetNN<NT_VERTEX>(), amg_levels[lev]->cap->free_nodes,
	      string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(lev));
	cvec = move(fvec2);
      }
    }
  }


  template class ElasticityAMGFactory<3>;
  template class VertexAMGPC<ElasticityAMGFactory<3>>;
  template class ElmatVAMG<ElasticityAMGFactory<3>, double, double>;

  using PCC = ElmatVAMG<ElasticityAMGFactory<3>, double, double>;

  RegisterPreconditioner<PCC> register_elast_3d ("ngs_amg.elast_3d");
} // namespace amg

#include "python_amg.hpp"

namespace amg
{
  void ExportElast3d (py::module & m)
  {
    ExportAMGClass<ElmatVAMG<ElasticityAMGFactory<3>, double, double>>(m, "elast_3d", "", [&](auto & m) { ; } );
  };
} // namespace amg

#endif // ELASTICITY
