namespace amg
{


  /** --- StokesAMGFactory --- **/


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

    /** Fill agg interiors **/


  } // StokesAMGFactory::BuildPWProl_impl


  template<int D>
  void StokesAMGFactory<D> :: SmoothProlongation (shared_ptr<ProlMap<TSPM_TM>> pmap, shared_ptr<TMESH> mesh) const
  {
    /**  I) Smooth normally for agg boundaries
        II) Fix agg interiors **/
  } // StokesAMGFactory::SmoothProlongation


  template<int D>
  void StokesAMGFactory<D> :: FillAggs (shared_ptr<TSPM_TM> prol, shared_ptr<TMESH> mesh, Table<int> & aggs) const
  {
  } // StokesAMGFactory::FillAggs


  /** --- END StokesAMGFactory --- **/


  /** --- StokesAMGPC --- **/


  template<class FACTORY>
  struct StokesAMGPC<FACTORY> :: Options : public FACTORY::Options,
					   public BaseAMGOptions
  {
    ;
  }; // StokesAMGPC::Options


  template<class FACTORY>
  StokesAMGPC<FACTORY> :: StokesAMGPC (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : Preconditioner(bfa, aflags, name)
  {
    ;
  } // StokesAMGPC::StokesAMGPC


  template<class FACTORY>
  StokesAMGPC<FACTORY> :: StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname)
  {
    throw Exception("StokesAMGPC PDE-constructor not implemented!!");
  } // StokesAMGPC::StokesAMGPC


  template<class FACTORY>
  shared_ptr<TMESH> StokesAMGPC<FACTORY> :: BuildInitialMesh ()
  {
    return nullptr;
  } // StokesAMGPC::BuildInitialMesh


  template<class FACTORY>
  shared_ptr<TMESH> StokesAMGPC<FACTORY> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    return nullptr;
  } // StokesAMGPC::BuildInitialMesh


  template<class FACTORY>
  shared_ptr<FACTORY> StokesAMGPC<FACTORY> :: BuildFactory (shared_ptr<TMESH> mesh)
  {
    return nullptr;
  } // StokesAMGPC::BuildFactory


  template<class FACTORY>
  void StokesAMGPC<FACTORY> :: BuildAMGMat ()
  {
    ;
  } // StokesAMGPC::BuildAMGMat


  template<class FACTORY>
  void StokesAMGPC<Factory> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    // I think this should never be called
  } // StokesAMGPC::InitLevel


  template<class FACTORY>
  void StokesAMGPC<Factory> :: FinalizeLevel (const BaseMatrix * mat)
  {
    // I think this should never be called
  } // StokesAMGPC::FinalizeLevel


  template<class FACTORY>
  void StokesAMGPC<Factory> :: Update ()
  {
    // I think this should never be called
  } // StokesAMGPC::Update


  /** --- END StokesAMGPC --- **/


} // namespace amg
