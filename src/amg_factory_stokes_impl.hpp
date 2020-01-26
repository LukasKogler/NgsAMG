#ifdef STOKES

#ifndef FILE_AMG_FACTORY_STOKES_HPP
#define FILE_AMG_FACTORY_STOKES_HPP

namespace amg
{

  /** StokesAMGFactory **/

  template<class TMESH, class ENERGY>
  void StokesAMGFactory<TMESH, ENERGY> :: AssA (const TMESH & mesh, FlatArray<ENERGY::TED> edata, FlatArray<ENERGY::TVD> vdata,
						FlatArray<int> vmems, FlatArray<int> emems, FlatMatrix<double> A, LocalHeap & lh)
  {
    /** Assemble replacement matrix diagonal for vertices vmems and edges ememes.
	emems/vmems are assumed to be sorted. **/
    const auto & cecon = *mesh.GetEdgeCM();

    for (auto kv : Range(vmems)) {

    }
  } // StokesAMGFactoryAssA

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


    /** Fill Agg interiors by imposing
	    u_f = P_f U
	as "Dirichlet" condition on it's boundary. For the interior minimize energy norm,
	under the restriction that we preserve divergence:
	\int_A_f \div(u) = |A_f|/|A| \int_A \div(U)    [\forall A_f]
	Solve the saddle point problem (for u_i, lam):
	        AA B^T         u_f      0
	        AA B^T  \cdot  u_i  =   0
	        BB 0           lam     Cu_c
     **/
    LocalHeap lh("Jerry", 10 * 1024 * 1024);
    for (auto agg_nr : Range(v_aggs)) {
      auto agg_vs = v_aggs[agg_nr];
      if (agg_vs.Size() > 1) {
	auto cv = vmap[agg_vs[0]];
	auto cneibs = cecon.GetRowIndices(cv);
	auto cfacets = cecon.GetRowValues(cv);
	int nfef = 0, ncef = 0;     // # fine/crs bnd-facets
	int nfei = 0;               // # fine int-facets
	int nfcells = 0;            // # fine cells
	int H = BS * nfef + BS * nfei + nfc;
	int Hi = BS * nfei + nfc;
	/** nfef |  A_BB A_BI B_B^T
	    nfei |  A_IB A_II B_F^T
	    nfce |  B_B  B_F   0    **/
	FlatMatrix<double> M (H, H, lh); M = 0;
	/** crs -> fine facet pw-prol (from above) **/
	FlatMatrix<double> P (BS * nfef, BS * ncef, lh); P = 0;
	/** nfef | 0
	    nfei | 0
	    nfce | |A_fi| / |A_c| \int_A_c div(u_c) **/
	FlatMatrix<double> C(H, BS * ncef, lh); C = 0;
	/** \int_A_c div(u_c) **/
	FlatMatrix<double> BC(H, BS * ncef, lh); BC = 0;
	FlatMatrix<double> inv (hi, hi, lh);
	inv = M.Rows(BS * nfef, H).Cols(BS * nfef, h);
	CalcInverse(inv);
	FlatMatrix<double> rhs (hi, BS * nced);
	rhs = C.Rows(BS * nfef, H) - M.Rows(BS * nfef, H).Cols(0, BS * nfef) * P;
	FlatMatrix<double> pblock (hi, BS * nced);
	pblock = inv * rhs;
      }
    }

  } // StokesAMGFactory<TMESH, ENERGY> :: BuildPWProl_impl
										   

  /** END StokesAMGFactory **/

} // namespace amg

#endif
