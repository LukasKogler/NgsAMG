namespace amg
{


  template<int D> template<class TMAP> shared_ptr<StokesAMGFactory<D>::TSPM_TM>
  StokesAMGFactory<D> :: BuildPWProl_impl (shared_ptr<TMAP> cmap, shared_ptr<ParallelDofs> fpd) const
  {
    /**
       I) Any fine edge that maps to a coarse edge is prolongated default PW
       II) Fill the interior of all agglomerates
    **/
    const auto & M (static_cast<TMESH&>(*cmap->GetMesh()));
    const auto & fecon(*M.GetEdgeCM());
    const auto & CM (static_cast<TMESH&>(*cmap->GetMappedMesh()));
    const auto & cecon(*CM.GetEdgeCM());
    auto vmap = cmap->template GetMap<NT_VERTEX>();
    auto emap = cmap->template GetMap<NT_EDGE>();
    size_t NV = M.template GetNN<NT_VERTEX>(), CNV = cmap->template GetMappedNN<NT_VEERTEX>(),
      NE = M.template GetNN<NT_EDGE>(), CNE = cmap->template GetMappedNN<NT_EDGE>();

    size_t H = NE, W = CNE;

    auto fedges = M.template GetNodes<NT_EDGE>();

    /** count entries **/
    Array<int> perow(H); perow = 0;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      if (cenr == -1) {
	int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
	if (cv0 == cv1) { // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate
	  perow[fenr] = cecon.GetRowIndices(cv0).Size();
	}
	else // probably some dirichlet-fuckery involved
	  { perow[fenr] = 0; }
      }
      else // an edge connecting two agglomerates
	{ perow[fenr] = 1; }
    }

    /** alloc mat  **/
    auto prol = make_shared<TSPM_TM> (perow, W);
    auto & P(*prol);

    /** fill agg boundaries **/
    ENERGY::TVD fvd, cvd;
    for (auto fenr : Range(H)) {
      auto & fedge = fedges[fenr];
      auto cenr = emap[fenr];
      int fv0 = fedge.v[0], fv1 = fedge.v[1];
      ENERGY::CalcMPData(vdata[fv0], vdata[fv1], cvd);
      int cv0 = vmap[fv0], cv1 = vmap[fv1];
      if (cenr != -1) {
	P.GetRowIndices(fenr)[0] = cenr;
	ENERGY::CalcMPData(vdata[cv0], vdata[cv1], cvd);
	ENERGY::CalcPWPBlock(fvd, cvd, P.GetRowValues(fenr)[0]);
      }
      else if (cv0 == cv1)
	{ P.GetRowIndices(fenr) = cecon.GetRowIndices(cenr); }
    }

    /** fill agg interiors **/


  } // StokesAMGFactory::BuildPWProl_impl


} // namespace amg
