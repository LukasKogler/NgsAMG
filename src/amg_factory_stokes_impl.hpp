#ifdef STOKES

#ifndef FILE_AMG_FACTORY_STOKES_IMPL_HPP
#define FILE_AMG_FACTORY_STOKES_IMPL_HPP

#include "amg_agg.hpp"
#include "amg_bla.hpp"

namespace amg
{

  /** StokesAMGFactory **/

  template<class TMESH, class ENERGY>
  StokesAMGFactory<TMESH, ENERGY> :: StokesAMGFactory (shared_ptr<StokesAMGFactory<TMESH, ENERGY>::Options> _opts)
    : BASE_CLASS(_opts)
  {
    ;
  } // StokesAMGFactory(..)


  template<class TMESH, class ENERGY>
  BaseAMGFactory::State* StokesAMGFactory<TMESH, ENERGY> :: AllocState () const
  {
    return new BaseAMGFactory::State();
  } // StokesAMGFactory::AllocState


  template<class TMESH, class ENERGY>
  void StokesAMGFactory<TMESH, ENERGY> :: InitState (BaseAMGFactory::State & state, BaseAMGFactory::AMGLevel & lev) const
  {
    BASE_CLASS::InitState(state, lev);
  } // StokesAMGFactory::InitState


  template<class TMESH, class ENERGY>
  shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildCoarseMap (BaseAMGFactory::State & state)
  {
    auto & O(static_cast<Options&>(*options));

    // typename Options::CRS_ALG calg = O.crs_alg;

    return BuildAggMap(state);

  //   switch(calg) {
  //   case(Options::CRS_ALG::AGG): { return BuildAggMap(state); break; }
  //   case(Options::CRS_ALG::ECOL): { return BuildECMap(state); break; }
  //   default: { throw Exception("Invalid coarsen alg!"); break; }
  //   }
  } // StokesAMGFactory::InitState


  template<class TMESH, class ENERGY>
  shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildAggMap (BaseAMGFactory::State & state)
  {
    auto & O = static_cast<Options&>(*options);
    typedef Agglomerator<ENERGY, TMESH, ENERGY::NEED_ROBUST> AGG_CLASS;
    typename AGG_CLASS::Options agg_opts;

    auto mesh = dynamic_pointer_cast<TMESH>(state.curr_mesh);
    if (mesh == nullptr)
      { throw Exception(string("Invalid mesh type ") + typeid(*state.curr_mesh).name() + string(" for BuildAggMap!")); }

    agg_opts.edge_thresh = 0.05;
    agg_opts.vert_thresh = 0;
    agg_opts.cw_geom = false;
    agg_opts.neib_boost = false;
    agg_opts.robust = false;
    agg_opts.dist2 = state.level[1] == 0;

    auto agglomerator = make_shared<AGG_CLASS>(mesh, state.free_nodes, move(agg_opts));

    return agglomerator;
  } // StokesAMGFactory::InitState


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: PWProlMap (shared_ptr<BaseCoarseMap> cmap, shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds)
  {
    auto fmesh = static_pointer_cast<TMESH>(cmap->GetMesh());
    auto cmesh = static_pointer_cast<TMESH>(cmap->GetMappedMesh());
    auto vmap = cmap->GetMap<NT_VERTEX>();
    auto emap = cmap->GetMap<NT_EDGE>();
    TableCreator<int> cva(cmesh->template GetNN<NT_VERTEX>());
    for (; !cva.Done(); cva++) {
      for (auto k : Range(vmap))
	if (vmap[k] != -1)
	  { cva.Add(vmap[k], k); }
    }
    auto v_aggs = cva.MoveTable();
    auto pwprol = BuildPWProl_impl (fpds, cpds, fmesh, cmesh, vmap, emap, v_aggs);
    return make_shared<ProlMap<TSPM_TM>>(pwprol, fpds, cpds);
  } // StokesAMGFactory::PWProlMap


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<TopologicMesh> fmesh)
  {
    throw Exception("Stokes SmoothedProlMap not yet implemented -> TODO !!");
    return nullptr;
  } // StokesAMGFactory::SmoothedProlMap


  template<class TMESH, class ENERGY> shared_ptr<BaseDOFMapStep>
  StokesAMGFactory<TMESH, ENERGY> :: SmoothedProlMap (shared_ptr<BaseDOFMapStep> pw_step, shared_ptr<BaseCoarseMap> cmap)
  {
    throw Exception("Stokes SmoothedProlMap not yet implemented -> TODO !!");
    return nullptr;
  } // StokesAMGFactory::SmoothedProlMap

    /**
       Stokes prolongation works like this:
           Step  I) Find a prolongation on agglomerate facets
	   Step II) Extend prolongation to agglomerate interiors:
	               Minimize energy. (I) as BC. Additional constraint:
		          \int_a div(u) = |a|/|A| \int_A div(U)    (note: we maintain constant divergence)

       "Piecewise" Stokes Prolongation.
           Step I) Normal PW Prolongation as in standard AMG

       "Smoothed" Stokes Prol takes PW Stokes prol and then:
           Step  I) Take PW prol, then smooth on facets between agglomerates. Additional constraint: 
	              Maintain flow through facet.
    **/

  template<class TMESH, class ENERGY> shared_ptr<typename StokesAMGFactory<TMESH, ENERGY>::TSPM_TM>
  StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl (shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds,
						       shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
						       FlatArray<int> vmap, FlatArray<int> emap,
						       FlatTable<int> v_aggs)//, FlatTable<int> c2f_e)
  {

    static constexpr int BS = mat_traits<TM>::HEIGHT:

    /** fine mesh **/
    const auto & FM(*fmesh); FM.CumulateData();
    const auto & fecon(*FM.GetEdgeCM());
    auto fvd = get<0>(FM.Data())->Data();
    auto fed = get<1>(FM.Data())->Data();
    size_t FNV = FM.template GetNN<NT_VERTEX>(), FNE = FM.template GetNN<NT_EDGE>();
    auto free_fes = CM.GetFreeNodes();

    /** coarse mesh **/
    const auto & CM(*cmesh); CM.CumulateData();
    const auto & cecon(*CM.GetEdgeCM());
    auto cvd = get<0>(CM.Data())->Data();
    auto ced = get<1>(CM.Data())->Data();
    size_t CNV = CM.template GetNN<NT_VERTEX>(), CNE = CM.template GetNN<NT_EDGE>();

    /** prol dims **/
    size_t H = FNE, W = CNE;

    /** Count entries **/
    Array<int> perow(H); perow = 0;
    auto fedges = FM.template GetNodes<NT_EDGE>();
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      if ( free_fes && (!free_fes->Test(fenr)) ) // dirichlet
    	{ perow[fenr] = 0; }
      else {
	auto cenr = emap[fenr];
	if (cenr != -1) // edge connects two agglomerates
	  { perow[fenr] = 1; }
	else {
	  int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
	  if (cv0 == cv1) {
	    if (cv0 == -1) // I don't think this can happen - per definition emap must be -1
	      { perow[fenr] = 0; throw Exception("Weird case A!"); }
	    else // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate
	      { perow[fenr] = cecon.GetRowIndices(cv0).Size(); }
	  }
	  else // I don't think this can happen
	    { perow[fenr] = 0; throw Exception("Weird case B!"); }
	}
      }
    }

    /** Allocate Matrix  **/
    auto prol = make_shared<TSPM_TM> (perow, W);
    auto & P(*prol);
    P.AsVector() = -42;

    /** Fill facets **/
    typename ENERGY::TVD tvf, tvc;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      if (cenr != -1) {
    	int fv0 = fedge.v[0], fv1 = fedge.v[1];
    	ENERGY::CalcMPData(fvd[fv0], fvd[fv1], tvf);
    	int cv0 = vmap[fv0], cv1 = vmap[fv1];
    	ENERGY::CalcMPData(cvd[cv0], cvd[cv1], tvc);
    	P.GetRowIndices(fenr)[0] = cenr;
    	ENERGY::CalcQHh(tvc, tvf, P.GetRowValues(fenr)[0]);
      }
    }

    cout << "stage 1 pwp: " << endl;
    print_tm_spmat(P, cout << endl);


    /** Extend the prolongation - Solve:
    	  |A_vv B_v^T|  |u_v|     |-A_vf      0     |  |P_f|
    	  |  ------  |  | - |  =  |  -------------  |  | - | |u_c|
    	  |B_v       |  |lam|     |-B_vf  d(|v|/|A|)|  |B_A| 
     **/
    LocalHeap lh("Jerry", 10 * 1024 * 1024);
    for (auto agg_nr : Range(v_aggs)) {
      auto agg_vs = v_aggs[agg_nr];
      auto cv = vmap[agg_vs[0]];
      if (agg_vs.Size() > 1) { // for single verts there are no interior edges
	HeapReset hr(lh);
	auto cv = vmap[agg_vs[0]];
	auto cneibs = cecon.GetRowindices(cv);
	auto cfacets = cecon.GetRowValues(cv);
	/** count fine facets **/
	int ncff = cfacets.Size();    // # coarse facets
	int nff = 0; int nffi = 0, nfff = 0;       // # fine facets (int/facet)
	auto it_f_facets = [&](auto lam) {
	  for (auto kv : Range(agg_vs)) {
	    auto vk = agg_vs[kv];
    	    auto vk_neibs = fecon.GetRowIndices(v);
    	    auto vk_fs = fecon.GetRowValues(v);
	    for (auto j : Range(v_neibs)) {
	      auto vj = v_neibs[j];
	      auto cvj = vmap[vj];
	      if (cvj != -1) {
		if (cvj == cv) { // neib in same agg
		  auto kj = find_in_sorted_array(vj, agg_vs);
		  lam(vk, k, vj, kj, int(vk_fs[j]));
		}
		else // neib in different agg
		  { lam(vk, k, vj, -1, int(vk_fs[j])); }
	      }
	    }
	  }
	};
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    nff++;
	    if (kj == -1) // ex-facet
	      { nfff++; }
	    else
	      { nffi++; }
	  });
	/** fine facet arrays **/
	FlatArray<int> index(nff, lh), buffer(nff, lh);
	FlatArray<int> ffacets(nff, lh), ffiinds(nffi, lh), fffinds(nfff, lh);
	it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
	    index[nff] = nff;
	    if (kj == -1)
	      { fffinds[nfff++] = nff; }
	    else
	      { ffiinds[nffi++] = nff; }
	    ffacets[nff++] = eid;
	  });
	QuickSortI(ffacets, index);
	for (auto k : Range(nffi))
	  { ffiinds[k] = index[ffiinds[k]]; }
	for (auto k : Range(nfff))
	  { fffinds[k] = index[fffinds[k]]; }
	ApplyPermutation(ffacets, index);

	auto it_f_edges = [&](auto lam) {
	  for (auto kvi : Range(agg_vs)) {
	    auto vi = agg_vs[kv];
    	    auto vi_neibs = fecon.GetRowIndices(vi);
    	    auto vi_fs = fecon.GetRowValues(vi);
	    const int nneibs = vk_fs.Size();
	    for (auto lk : Range(nneibs)) {
	      auto vk = vi_neibs[lk];
	      auto fik = int(vk_fs[lk]);
	      auto kfik = find_in_sorted_array(fik, ffacets);
	      for (auto lj : Range(kvj)) {
		auto vj = vi_neibs[lj];
		auto fij = int(vk_fs[lj]);
		auto kfij = find_in_sorted_array(fij, ffacets);
		lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
	      }
	    }
	  }
	};

	int HA = nff * BS, HB = nfv, HM = HA + HB;

	FlatMatrix<double> M(HM, HM, lh); M = 0;

	/** A block **/
	// FlatMatrix<double> A(nff * BS, nff * BS, lh); A = 0;
	auto a = M.Rows(0, HA).Cols(0, HA);
	FlatMatrix<TM> eblock(2,2,lh);
	it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
	    ENERGY::CalcRMBlock(eblock, fvd[vi], fvd[vj], fvd[vk], fed[fij], (vi > vj), fed[fik], (vi > vk));
	    Iterate<2>([&](auto i) {
		int osi = BS * (i.value == 0 ? fij, fik);
		Iterate<2>([&](auto j) {
		    int osj = BS * (j.value == 0 ? fij, fik);
		    A.Rows(osi, osi + BS).Cols(osj, osj + BS) += eblock(i.value, j.value);
		  });
	      });
	  });

	/** (fine) B blocks **/
	auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
	auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
	for (auto kvi : Range(nfv)) {
	  auto BfRow = Bf.Row(kvi);
	  auto vi = agg_vs[kvi];
	  auto vi_neibs = fecon.GetRowIndices(vi);
	  auto vi_fs = fecon.GetRowValues(vi);
	  for (auto j : Range(vi_fs)) {
	    auto vj = vi_neibs[j];
	    auto fij = int(vi_fs[j]);
	    auto kfij = find_in_sorted_array(fij, ffacets);
	    auto & fijd = fed[fij];
	    int col = BS * kvj;
	    for (auto l : Range(BS))
	      { BfRow(col++) = fed[fij].flow(l); }
	  }
	}

	/** (coarse) B **/
	FlatArray<double> bcbase(BS * ncf);
	FlatMatrix<double> Bc(nfv, BS * ncf);
	auto cv_neibs = cecon.GetRowIndices(cv);
	auto cv_fs = cecon.GetRowValues(cv);
	int bccol = 0;
	for (auto j : Range(cv_fs)) {
	  auto cvj = cv_neibs[j];
	  auto fij = int(cv_fs[j]);
	  auto & fijd = ced[fij];
	  for (auto l : Range(BS))
	    { bcbase(col++) = fijd.flow(l); }
	}
	double cvol = cvd[cv].vol;
	for (auto kvi : Range(nfv)) {
	  auto vi = agg_vs[kvi];
	  auto bcrow = Bc.Row(kvi);
	  for (auto col : Range(bcbase))
	    { bcrow(col) = fvd[vi].vol / cvol * bcbcase[col]; }
	}

	/** RHS **/
	int Hi = BS * nffi, Hf = BS * nffc, Hc = BS * ncf;
	FlatArray<int> colsi(Hi + nfv, lh), colsf(Hf, lh);
	int ci = 0;
	for (auto j : Range(nffi)) {
	  int base = BS * ffiinds[j];
	  for (auto l : Range(BS))
	    { colsi[ci++] = base++; }
	}
	for (auto kv : Range(nfv))
	  { colsi[ci++] = HA + kv; }
	int cf = 0;
	for (auto j : Range(nfff)) {
	  int base = BS * fffinds[j];
	  for (auto l : Range(BS))
	    { colsf[cf++] = base++; }
	}

	FlatMatrix<double> Pf(Hf, Hc, lh);

	/** -A_if * P_f **/
	FlatMatrix<double> rhs(Hi + nfv, Hc, lh);
	rhs.Rows(0, Hi) = 0;
	rhs.Rows(Hi, Rhs.Height()) = Bc;
	rhs -= M.Rows(colsi).Cols(colsf);
	
	/** The block to invert **/
	FlatMatrix<double> Mii(Hi + nfv, Hi + nfv, lh);
	Mii = M.Rows(colsi).Cols(colsi);
	CalcInverse(mii);

	/** The final prol **/
	FlatMatrix<double> Pext(Hi + nfv, Hc, lh);
	Pext = Mii * rhs;

	/** Write into sprol **/
	for (auto kfi : Range(nffi)) {
	    auto ff = ffacets[ffiinds[kfi]];
	    auto ris = P.GetRowIndices(ff);
	    auto rvs = P.GetRowValues(ff);
	    for (auto j : Range(ncf)) {
	      ris[j] = cfacets[j];
	      rvs[j] = Pext.Rows(BS*kfi, BS*(kfi+1)).Cols(BS*j, BS*(j+1));
	    }
	}

      } // agg_vs.Size() > 1
    } // agglomerate loop

    cout << " Final Stokes PWP:" << endl;
    print_tm_spmat(P, cout << endl);

    // for (auto agg_nr : Range(v_aggs)) {
    //   auto agg_vs = v_aggs[agg_nr];
    //   if (agg_vs.Size() > 1) {
    // 	auto cv = vmap[agg_vs[0]];
    // 	auto cneibs = cecon.GetRowIndices(cv);
    // 	auto cfacets = cecon.GetRowValues(cv);
    // 	auto it_edges = [&](auto lam) {
    // 	  for (auto kv : Range(agg_vs)) {
    // 	    auto v = agg_vs[kv];
    // 	    auto v_neibs = fecon.GetRowIndices(v);
    // 	    auto v_es = fecon.GetRowValues(v);
    // 	    for (auto j : Range(v_neibs)) {
    // 	      auto neib = v_neibs[j];
    // 	      auto cneib = vmap[neib];
    // 	      if (cneib != -1) {
    // 		if (cneib != cv)
    // 		  { lam_ex(v, kv, neib, -1, int(v_es[j])); }
    // 		else if (neib > v) // do not count interior edges twice!
    // 		  { lam_in(v, kv, neib, agg_vs.Pos(neib), int(v_es[j])); }
    // 	      }
    // 	    }
    // 	  }
    // 	};
    // 	/** get int/ext fine edges **/
    // 	int nff_f = 0, nfv_f = 0, nf_f = 0;
    // 	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
    // 	    nf_f++;
    // 	    if (kn == -1)
    // 	      { nff_f++; }
    // 	    else
    // 	      { nfv_f++; }
    // 	  });
    // 	FlatArray<int> fenrs(nf_f, lh); nf_f = 0;
    // 	FlatArray<int> frows(BS*nff_f, lh); nff_f = 0;
    // 	FlatArray<int> vrows(BS*nfv_f, lh); nfv_f = 0;
    // 	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
    // 	    const int os = BS*nff_f;
    // 	    if (kn == -1)
    // 	      { Iterate<BS>([&](auto i) LAMBDA_INLINE { frows[nff_f++] = os + i.value; }); }
    // 	    else
    // 	      { Iterate<BS>([&](auto i) LAMBDA_INLINE { vrows[nfv_f++] = os + i.value; }); }
    // 	    fenrs[nf_f++] = eid;
    // 	  });
    // 	nff_f /= BS; nfv_f /= BS;
    // 	/** Fine A **/
    // 	// int na = BS * nf_f, nb = nv_f; // ??
    // 	int na = BS * nff_f, nb = nfv_f; // ??
    // 	FlatMatrix<TM> eblock(2, 2, lh);
    // 	FlatMatrix<double> A(na, na, lh); A = 0;
    // 	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
    // 	    ENERGY::CalcRMBlock(eblock, fed[eid], fvd[v], fvd[n]);
    // 	    A.Rows(kv*BS, (1+kv)*BS, kn*BS, (1+kn)*BS) += eblock;
    // 	  });
    // 	/** Fine B **/
    // 	FlatMatrix<double> B(nb, na, lh); B = 0;
    // 	for (auto kv : Range(agg_vs)) {
    // 	  auto v = agg_vs[kv];
    // 	  auto neibs = fecon.GetRowIndices(v);
    // 	  auto enrs = fecon.GetRowValues(v);
    // 	  for (auto l : Range(neibs)) {
    // 	    int enr(enrs[l]);
    // 	    auto oscol = fenrs.Pos(enr);
    // 	    const auto & flow = fed[enr].flow;
    // 	    Iterate<BS>([&](auto i) LAMBDA_INLINE { B(kv, oscol + i.value) = flow(i.value); });
    // 	  }
    // 	}
    // 	/** P **/
    // 	// FlatMatrix<double> Pf (BS*nff_f, BS*nf_c, lh); Pf = 0; // ??
    // 	FlatMatrix<double> Pf (BS*nff_f, BS*nf_f, lh); Pf = 0; // !!! thats probably very wrong !!!
    // 	for (auto r : Range(nff_f)) {
    // 	  int enr = fenrs[frows[BS*r]/BS];
    // 	  int cenr = emap[enr];
    // 	  if (cenr != -1) {
    // 	    int col = cfacets.Pos(double(cenr));
    // 	    Pf.Rows(r*BS, (1+r)*BS).Cols(col*BS, (1+col)*BS) = P(enr, cenr);
    // 	  }
    // 	}
    // 	/** B coarse **/
    // 	// FlatMatrix<double> Bc(1, BS*nf_c, lh); Bc = 0; // ??
    // 	FlatMatrix<double> Bc(1, BS*nf_f, lh); Bc = 0; // !!! same here !!!
    // 	for (auto j : Range(cfacets))
    // 	  { Bc.Row(0).Cols(j*BS, (1+j)*BS) = cedata[cfacets[j]].flow; }
    // 	FlatMatrix<double> Bc_stack(nv_f, BS*nf_c, lh); Bc_stack = 0;
    // 	for (auto k : Range(agg_vs))
    // 	  { Bc_stack.Row(k) = fvd[agg_vs[k]].vol / cvol * Bc; }

    // 	/** Invert u_v/lam part **/
    // 	int nfreef = BS*nfv_f, nfreev = nv_f, nfree = nfreef + nfreev;
    // 	FlatMatrix<double> inv(nfree, nfree, lh);
    // 	inv.Rows(0, nfreef).Cols(0, nfreef) = A.Rows(vrows).Cols(vrows);
    // 	inv.Rows(0, nfreef).Cols(nfreef, nfree) = Trans(B.Cols(vrows));
    // 	inv.Rows(nfreef, nfree).Cols(0, nfreef) = B.Cols(vrows);
    // 	inv.Rows(nfreef, nfree).Cols(0, nfree) = 0;
    // 	CalcInverse(inv);

    // 	/** multiply mats **/
    // 	FlatMatrix<double> rhs(nfree, BS*nf_c, lh); rhs = 0;
    // 	rhs.Rows(0, nfreef) = -A.Rows(vrows).Cols(frows) * Pf;
    // 	rhs.Rows(nfreef, nfree).Cols(0, nfreef) = -B.Cols(frows) * Pf;
    // 	rhs.Rows(nfreef, nfree).Cols(nfreef, nfree) = Bc_stack;
    // 	FlatMatrix<double> Pv(nfree, BS*nf_c, lh); Pv = inv * rhs;
	
    // 	/** fill sprol **/
    // 	for (auto ki : Range(nfv_f)) {
    // 	  auto fenr = fenrs[ki];
    // 	  auto ris = P.GetRowIndices(fenr);
    // 	  auto rvs = P.GetRowValues(fenr);
    // 	  for (auto j : Range(cfacets)) {
    // 	    ris[k] = cfacets[j];
    // 	    rvs[j] = Pv.Rows(ki*BS, (1+k)*BS).Cols(j*BS, (1+j)*BS);
    // 	  }
    // 	}
    //   }
    // }

    return nullptr;
  } // StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl
										   

  /** END StokesAMGFactory **/

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_HPP
#endif // STOKES
