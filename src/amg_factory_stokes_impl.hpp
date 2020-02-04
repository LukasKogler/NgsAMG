#ifdef STOKES

#ifndef FILE_AMG_FACTORY_STOKES_HPP
#define FILE_AMG_FACTORY_STOKES_HPP

namespace amg
{

  /** StokesAMGFactory **/

  template<class TMESH, class ENERGY>
  shared_ptr<BaseCoarseMap> StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl (shared_ptr<ParallelDofs> fpds, shared_ptr<ParallelDofs> cpds,
										 shared_ptr<TMESH> fmesh, shared_ptr<TMESH> cmesh,
										 FlatArray<int> vmap, FlatArray<int> emap,
										 FlatTable<int> v_aggs, FlatTable<int> c2f_e)
  {
    const auto & FM(*fmesh); FM.CumulateData();
    const auto & fecon(*FM.GetEdgeCM());
    const auto & CM(*cmesh); CM.CumulateData();
    const auto & cecon(*CM.GetEdgeCM());

    size_t NV = FM.template GetNN<NT_VERTEX>(), CNV = CM.template GetNN<NT_VEERTEX>(),
      NE = FM.template GetNN<NT_EDGE>(), CNE = CM.template GetNN<NT_EDGE>();

    size_t H = NE, W = CNE;

    auto fedges = FM.template GetNodes<NT_EDGE>();

    /** Count entries **/
    Array<int> perow(H); perow = 0;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      if (cenr == -1) {
	int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
	if (cv0 == cv1) { // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate
	  if (cv0 == -1)
	    { perow[fenr] = 0; }
	  else
	    { perow[fenr] = cecon.GetRowIndices(cv0).Size(); }
	}
	else // probably some dirichlet-fuckery involved
	  { perow[fenr] = 0; }
      }
      else // an edge connecting two agglomerates
	{ perow[fenr] = 1; }
    }

    /** Allocate Matrix  **/
    auto prol = make_shared<TSPM_TM> (perow, W);
    auto & P(*prol);

    /** Fill facets **/
    ENERGY::TVD fvd, cvd;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      if (cenr != -1) {
	int fv0 = fedge.v[0], fv1 = fedge.v[1];
	ENERGY::CalcMPData(vdata[fv0], vdata[fv1], fvd);
	int cv0 = vmap[fv0], cv1 = vmap[fv1];
	ENERGY::CalcMPData(vdata[cv0], vdata[cv1], cvd);
	P.GetRowIndices(fenr)[0] = cenr;
	ENERGY::CalcPWPBlock(fvd, cvd, P.GetRowValues(fenr)[0]);
      }
    }


    /** Solve:
	  |A_vv B_v^T|  |u_v|     |-A_vf      0     |  |P_f|
	  |  ------  |  | - |  =  |  -------------  |  | - | |u_c|
	  |B_v       |  |lam|     |-B_vf  d(|v|/|A|)|  |B_A| 
     **/
    LocalHeap lh("Jerry", 10 * 1024 * 1024);
    for (auto agg_nr : Range(v_aggs)) {
      auto agg_vs = v_aggs[agg_nr];
      if (agg_vs.Size() > 1) {
	auto cv = vmap[agg_vs[0]];
	auto cneibs = cecon.GetRowIndices(cv);
	auto cfacets = cecon.GetRowValues(cv);
	auto it_edges = [&](auto lam) {
	  for (auto kv : Range(agg_vs)) {
	    auto v = agg_vs[kv];
	    auto v_neibs = fecon.GetRowIndices(v);
	    auto v_es = fecon.GetRowValues(v);
	    for (auto j : Range(v_neibs)) {
	      auto neib = v_neibs[j];
	      auto cneib = vmap[neib]l
	      if (cneib != -1) {
		if (cneib != cv)
		  { lam_ex(v, kv, n, -1, int(v_es[j])); }
		else if (n > v) // do not count interior edges twice!
		  { lam_in(v, kv, n, agg_vs.Pos(neib), int(v_es[j])); }
	      }
	    }
	  }
	};
	/** get int/ext fine edges **/
	int nff_f = 0, nfv_f = 0, nf_f = 0
	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
	    nf_f++;
	    if (kn == -1)
	      { nff_f++; }
	    else
	      { nfv_f++; }
	  });
	FlatArray<int> fenrs(nf_f, lh); nf_f = 0;
	FlatArray<int> frows(BS*nff_f, lh); nff_f = 0;
	FlatArray<int> vrows(BS*nfv_f, lh); nfv_f = 0;
	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
	    const int os = BS*nff_f;
	    if (kn == -1)
	      { Iterate<BS>([&](auto i) LAMBDA_INLINE { frows[nff_f++] = os + i.value; }) }
	    else
	      { Iterate<BS>([&](auto i) LAMBDA_INLINE { vrows[nfv_f++] = os + i.value; }) }
	    fenrs[nf_f++] = eid;
	  });
	nff_f /= BS; nfv_f /= BS;
	/** Fine A **/
	int na = BS * nf_f, nb = nv_f;
	FlatMatrix<TM> eblock(2, 2, lh);
	FlatMatrix<double> A(na, na, lh); A = 0;
	it_edges([&](auto v, auto kv, auto n, auto kn, int eid) {
	    ENERGY::CalcRMBlock(eblock, fed[eid], fvd[v], fvd[n]);
	    A.Rows(kv*BS, (1+kv)*BS, kn*BS, (1+kn)*BS) += eblock;
	  });
	/** Fine B **/
	FlatMatrix<double> B(nb, na, lh); B = 0;
	for (auto kv : Range(agg_vs)) {
	  auto v = agg_vs[kv];
	  auto neibs = fecon.GetRowIndices(v);
	  auto enrs = fecon.GetRowValues(v);
	  for (auto l : Range(neibs)) {
	    int enr(enrs[l]);
	    auto oscol = fenrs.Pos(enr);
	    const auto & flow = fedata[enr].flow;
	    Iterate<BS>([&](auto i) LAMBDA_INLINE { B(kv, oscol + i.value) = flow(i.value); });
	  }
	}
	/** P **/
	FlatMatrix<double> Pf (BS*nff_f, BS*nf_c, lh); Pf = 0;
	for (auto r : Range(nff_f)) {
	  int enr = fenrs[frows[BS*r]/BS];
	  int cenr = emap[enr];
	  if (cenr != -1) {
	    int col = cfacets.Pos(double(cenr));
	    Pf.Rows(r*BS, (1+r)*BS).Cols(col*BS, (1+col)*BS) = P(enr, cenr);
	  }
	}
	/** B coarse **/
	FlatMatrix<double> Bc(1, BS*nf_c, lh); Bc = 0;
	for (auto j : Range(cfacets))
	  { Bc.Row(0).Cols(j*BS, (1+j)*BS) = cedata[cfacets[j]].flow; }
	FlatMatrix<double> Bc_stack(nv_f, BS*nf_c, lh); Bc_stack = 0;
	for (auto k : Range(agg_vs))
	  { Bc_stack.Row(k) = fvdata[agg_vs[k]].vol / cvol * Bc; }

	/** Invert u_v/lam part **/
	int nfreef = BS*nfv_f, nfreev = nv_f, nfree = nfreef + nfreev;
	FlatMatrix<double> inv(nfree, nfree, lh);
	inv.Rows(0, nfreef).Cols(0, nfreef) = A.Rows(vrows).Cols(vrows);
	inv.Rows(0, nfreef).Cols(nfreef, nfree) = Trans(B.Cols(vrows));
	inv.Rows(nfreef, nfree).Cols(0, nfreef) = B.Cols(vrows);
	inv.Rows(nfreef, nfree).Cols(0, nfree) = 0;
	CalcInverse(inv);

	/** multiply mats **/
	FlatMatrix<double> rhs(nfree, BS*nf_c, lh); rhs = 0;
	rhs.Rows(0, nfreef) = -A.Rows(vrows).Cols(frows) * Pf;
	rhs.Rows(nfreef, nfree).Cols(0, nfreef) = -B.Cols(frows) * Pf;
	rhs.Rows(nfreef, nfree).Cols(nfreef, nfree) = Bc_stack;
	FlatMatrix<double> Pv(nfree, BS*nf_c, lh); Pv = inv * rhs;
	
	/** fill sprol **/
	for (auto ki : Range(nfv_f)) {
	  auto fenr = fenrs[ki];
	  auto ris = P.GetRowIndices(fenr);
	  auto rvs = P.GetRowValues(fenr);
	  for (auto j : Range(cfacets)) {
	    ris[k] = cfacets[j];
	    rvs[j] = Pv.Rows(ki*BS, (1+k)*BS).Cols(j*BS, (1+j)*BS);
	  }
	}
      }
    }

  } // StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl
										   

  /** END StokesAMGFactory **/

} // namespace amg

#endif
