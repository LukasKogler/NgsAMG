#ifdef STOKES

#define FILE_AMG_STOKES_CPP

#include "amg.hpp"
#include "amg_stokes.hpp"


namespace amg
{


  /** AttachedSED **/


  template<int D> template<class TMESH>
  INLINE void AttachedSED<D> :: map_data (const BaseCoarseMap & map, AttachedSED<D> & csed) const
  {
    auto & M = static_cast<StokesMesh<D>&>(*this->mesh); M.CumulateData();
  } // AttachedSED::map_data


  /** StokesAMGFactory **/




  template<int D>
  void StokesAMGFactory<D> :: SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const
  {
    /**  I) Smooth normally for agg boundaries
        II) Fix agg interiors **/
    /**
       I need agglomerates of vertices, and maps of edges. Can I get those from the prol itself??
       I guess I can get an implicit map of edge-nrs from prol - remaining edges have only one entry
       in row -> probably can get facets. 
         -) what about MPI??
	 -) is this the only way that there is only one entry?? What about coarse level dangling vertex?? <- actually, a problem!

       I definitely need a coarse-map here ...
     **/
    FillAggs(prol, ...);
  } // StokesAMGFactory::SmoothProlongation


  template<int D>
  void StokesAMGFactory<D> :: FillAggs (shared_ptr<StokesAMGFactory<D>::TSPM_TM> prol, ..) const
  {
    /**
       For a prolongation that already sets DOFs on the boundary of every agglomerate, fill the interior
       by energy minimization while ensuring divergence compliance by imposing pw constant divergence on
       each cell (=vertex) in the aggregate. 
       The prolongation has to already have allocated entries for the agg-interiors
     **/

    LocalHeap lh(30*1024*1024, "FillAggs");

    constexpr int H = mat_traits<TM>::HEIGHT;
    TVD mpi, mpj;
    TM Qij, Qji, M, X;
    SetIdentity(Qij); SetIdentity(Qji);
    int posi, posj;
    for (auto agg_nr : Range(n_aggs)) {
      auto nav = agg_verts.Size();
      auto nae = agg_edges.Size();
      /** Alloc saddle point mat - one row per edge + **/
      int sizea = nae * H, hb = nav;
      FlatMatrix<double> sp_mat ( sizea + hb, sizea + hb, lh); sp_mat = 0;
      auto A = sp_mat.Rows(0, sizea).Cols(0, sizea);
      auto B = sp_mat.Rows(sizea, sizea + hb).Cols(0, sizea);
      auto BT = sp_mat.Rows(0, sizea).Cols(sizea, sizea + hb);
      BitArray dirie(nae, lh); diri_e.Clear();
      double area_agg = 0;
      /** Set up A/B **/
      for (auto vi : Range(agg_verts)) {
	auto vert_nr = agg_verts[i];
	auto ri = econ.GetRowIndices(vert_nr);
	auto rv = econ.GetRowValues(vert_nr);
	area_agg += vdata[vert_nr].area;
	const auto area_inv = 1.0/vdata[vert_nr].area;
	for (auto i : Range(ri)) {
	  auto vnri = ri[i];
	  auto enri = int(rv[i]);
	  if ( (posi = agg_edges.Pos(enri)) != -1) {
	    if ( (posj = agg_verts.Pos(vnri)) == -1 ) // edge is in, but other vert is out
	      { dirie.SetBit(posi); }
	    /** A contrib **/
	    int ilo = H * posi, ihi = H * (1 + posi);
	    for (auto j : Range(i)) {
	      auto vnrj = ri[j];
	      auto enri = int(rv[j]);
	      if ( (posj = agg_edges.Pos(enrj)) ) {
		GetEdgeMat(enri, enrj, M); // something like that..
		CalcMPData(vdata[vert_nr], vdata[vnri], mpi);
		CalcMPData(vdata[vert_nr], vdata[vnrj], mpj);
		ModQs(mpi, mpj, Qij, Qji);
		int jlo = H * posj, jhi = H * (1 + posj);
		X = M * Qij;
		A.Rows(ilo, ihi).Cols(ilo, ihi) += Trans(Qij) * X;
		A.Rows(jlo, jhi).Cols(ilo, ihi) -= Trans(Qji) * X;
		X = M * Qji;
		A.Rows(ilo, ihi).Cols(jlo, jhi) += Trans(Qij) * X;
		A.Rows(ilo, ihi).Cols(jlo, jhi) -= Trans(Qji) * X;
	      }
	    }
	    /** B contrib **/
	    for (auto j : Range(D)) // plusminus
	      { B(vi, ilo+j) = area_inv * edata[enri].flow(j); }
	  }
	}
	
	/** Compute C = (..\int_\partial\Agg u_bnd*n..) **/
	area = 1.0/area;
	for (auto i : Range(agg_edges))
	  if (dirie.Test(i)) {
	    auto enri = agg_edges[i];
	    int l = (agg_verts.Pos(edges[enri].v[0]) != -1) ? 0 : 1;
	    bool flip = (  ) ? false : true;
	    for (auto j : Range(H * posi, H * (1 + posi) ) )
	      { B(vin, j) -= area * edata[enri].flow(j); }
	  }

	BT = Trans(B);

	/** Solve   A B.T  u  = 0
	    B      p  = (c...c)
	    with BC u=u_c on agg boundary **/
	int nediri = dirie.NumSet(), nefree = nae - nediri;
	int ndfree = nefree * H + hb, nddiri = M.Height() - ndfree;
	FlatArray<int> indf (ndfree, lh), indd (nddiri, lh);
	ndfree = nddiri = 0;
	for (auto i : Range(agg_edges)) {
	  const auto di = H * i;
	  if (dirie.Test(i)) {
	    for (auto ii : Range(H))
	      { indd[nddiri++] = di + ii; }
	  }
	  else {
	    for (auto ii : Range(H))
	      { indf[ndfree++] = di + ii; }
	  }
	}
	for (auto i : Range(H*nae, H*nae + hb))
	  { indf[ndfree++] = i; }
	FlatMatrix<double> MI (ndfree, ndfree);
	MI = M.Rows(indf).Cols(indf);
	CalcInverse(MI);
	FlatMatrix<double> P (ndfree, nddiri);
	P = -1.0 * MI * M.Rows(indf).Cols(indd);

	/** Fill mat **/
      }
      
      return;
    } // StokesAMGFactory<D> :: FillAggs


  /** END StokesAMGFactory **/

} // namespace amg

#endif
