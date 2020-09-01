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
    auto gcomm = amg_levels[0]->cap->eqc_h->GetCommunicator();
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
    auto chkab = [&](auto fva, auto fvb, int n, shared_ptr<BitArray> fnodes, string title) {
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
      if (numdf > 0)
	{ cout << " done " << title << ", mismatches = " << numdf << endl; }
      if (fnodes != nullptr)
	{ cout << " fnodes non-set = " << fnodes->Size() - fnodes->NumSet() << endl; }
    };
    auto prtv = [&](auto & vec, int nv, string title) {
      if (nv == 0)
	{ return; }
      auto fv = vec.FVDouble();
      int vbs = fv.Size()/nv;
      cout << title << " = " << endl;
      cout << "  stat = " << vec.GetParallelStatus() << endl;
      cout << "  vals = " << endl;
      for (auto vnr : Range(nv)) {
	cout << "  " << vnr << " = ";
	for (int l = 0; l < vbs; l++)
	  cout << fv(vbs * vnr + l) << " ";
	cout << endl;
      }
      cout << endl;
    };
    auto set_kvec = [&](auto & vec, int kvnr, BaseAMGFactory::AMGLevel & alev, shared_ptr<BitArray> free_nodes) {
      typename ENERGY::TVD opt(0); /** just an arbitrary point ... **/
      // for (auto l : Range(3))
	// { opt.pos(0) = l * gcomm.Size() + gcomm.Rank(); }
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
	  if ( (free_nodes == nullptr) || (free_nodes->Test(vnr)) ) {
	    ENERGY::CalcQHh(opt, vdata[vnr], Q);
	    vcos = Q * ovec;
	    for (int l = 0; l < vbs; l++)
	      { fv(vbs * vnr + l) = vcos(l); }
	    cout << vnr << " = ";
	    for (int l = 0; l < vbs; l++)
	      cout << fv(vbs * vnr + l) << " ";
	    cout << endl;
	  }
	  else {
	    for (int l = 0; l < vbs; l++)
	      { fv(vbs * vnr + l) = 0.0; }
	  }
	}
	cout << endl;
      }
    };
    auto clcenrg = [&](auto & lev, auto & v, string title) {
      cout << " mat type " << typeid(*lev.cap->mat).name() << endl;
      auto pds = lev.embed_map != nullptr ? lev.embed_map->GetParDofs() : lev.cap->pardofs;
      auto A = make_shared<ParallelMatrix>(lev.cap->mat, pds, pds, C2D);
      prtv(v, lev.cap->mesh->template GetNN<NT_VERTEX>(), "vec v");
      unique_ptr<BaseVector> Av = A->CreateColVector();
      A->Mult(v, *Av);
      prtv(*Av, lev.cap->mesh->template GetNN<NT_VERTEX>(), "vec Av");
      double enrg = sqrt(fabs(InnerProduct(*Av, v)));
      cout << title << ", energy = " << enrg << ", vv = " << InnerProduct(v,v) << ", relative = " << enrg/sqrt(InnerProduct(v, v)) << endl;
    };
    for (int kvnr = 0; kvnr < BS; kvnr++) {
      int nlevsglob = gcomm.AllReduce(amg_levels.Size(), MPI_MAX);
      int nlevsloc = map->GetNLevels();
      unique_ptr<BaseVector> cvec = move(map->CreateVector(nlevsloc-1));
      if ( (nlevsloc == nlevsglob) && (cvec != nullptr) ) {
	set_kvec(*cvec, kvnr, *amg_levels[nlevsloc-1], nullptr);
	clcenrg(*amg_levels[nlevsloc-1], *cvec,
		string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(nlevsloc-1));
      }
      for (int lev = nlevsloc-2; lev >= 0; lev--) {
	bool havemb = (lev == 0) && (amg_levels[0]->embed_map != nullptr);
	unique_ptr<BaseVector> fvec1 = map->CreateVector(lev), fvec2 = havemb ? amg_levels[0]->embed_map->CreateMappedVector() : map->CreateVector(lev);
	map->TransferC2F(lev, fvec1.get(), cvec.get());
	set_kvec(*fvec2, kvnr, *amg_levels[lev], amg_levels[lev]->cap->free_nodes);
	if ( havemb ) {
	  unique_ptr<BaseVector> fvec3 = amg_levels[0]->embed_map->CreateVector();
	  amg_levels[0]->embed_map->Finalize(); // this can be  concatenated before crs grid projection!
	  amg_levels[0]->embed_map->TransferC2F(fvec3.get(), fvec2.get());
	  fvec2 = move(fvec3);
	}
	chkab(fvec1->FVDouble(), fvec2->FVDouble(), amg_levels[lev]->cap->mesh->template GetNN<NT_VERTEX>(), nullptr,
	      string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(lev));
	clcenrg(*amg_levels[lev], *fvec1,
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
