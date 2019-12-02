#include "amg_tdnns.hpp"
#include "amg_elast_impl.hpp"

namespace amg
{


  template<int DIM>
  TDNNS_AUX_AMG_Preconditioner<DIM> :: TDNNS_AUX_AMG_Preconditioner (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : Preconditioner (bfa, aflags, name)
  {
    blf = bfa;
    auto fes = bfa->GetFESpace();
    comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
    if (comp_fes == nullptr)
      { throw Exception(string("Need a Compound space, got ") + typeid(*fes).name() + string("!")); }
    ind_hc = ind_hd = -1;
    for (auto space_nr : Range(comp_fes->GetNSpaces())) {
      auto s = (*comp_fes)[space_nr];
      if (auto hc = dynamic_pointer_cast<HCurlHighOrderFESpace>(s))
	{ hcurl = hc; ind_hc = space_nr; }
      else if (auto hd = dynamic_pointer_cast<HDivHighOrderFESpace>(s))
	{ hdiv = hd; ind_hd = space_nr; }
    }
    if (ind_hc == -1)
      { throw Exception("No HCurl space in Compound space!"); }
    if (ind_hc == -1)
      { throw Exception("No HDiv space in Compound space!"); }

    auto hc_range = comp_fes->GetRange(ind_hc); os_hc = hc_range.First();
    auto hd_range = comp_fes->GetRange(ind_hd); os_hd = hd_range.First();

    hc_per_facetfacet  = int(aflags.GetNumFlag("hc_per_facetfacet",  ( (DIM==3) ? 1 : 0 ) ));
    hd_per_facet = int(aflags.GetNumFlag("hd_per_facet", DIM));
    hc_per_facet = int(aflags.GetNumFlag("hc_per_facet", (DIM == 3) ? 0 : 1));

    if (DIM == 2)
      { hc_per_facetfacet = 0; }
  } // TDNNS_AUX_AMG_Preconditioner(..)


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    /** Any edge on a dirichlet-boundary can be averaged only from dirichlet facets. **/
    aux_freedofs = make_shared<BitArray>(ma->GetNFacets());
    aux_freedofs->Clear();
    if (freedofs && false) {
      for (auto facet_nr : Range(aux_freedofs->Size())) {
	auto lodofs = facet_lo_dofs[facet_nr].Range(0, lors[facet_nr][0]); // check only hdiv-DOFs
	bool diri_facet = true;
	for (auto d : lodofs) // if any hd-dof is free, it is a free facet (HO dofs are dirichlet after BBDC)
	  if (freedofs->Test(d))
	    { diri_facet = false; break; }
	if (diri_facet)
	  { aux_freedofs->SetBit(facet_nr); }
      }
    }
    aux_freedofs->Invert();

    AllocAuxMat();
    SetUpFacetMats();
    SetUpAuxParDofs();
    BuildEmbedding();
  }


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: FinalizeLevel (const BaseMatrix * mat)
  {

    cout << "aux_mat: " << endl;
    print_tm_spmat(cout, *aux_mat); cout << endl;
  }
    

  template<int DIM> AutoVector TDNNS_AUX_AMG_Preconditioner<DIM> :: CreateColVector () const
  {
    auto blf_mat = blf->GetMatrixPtr();
    if (blf_mat == nullptr)
      { throw Exception("mat not ready"); }
    return blf_mat->CreateColVector();
  }


  template<int DIM> AutoVector TDNNS_AUX_AMG_Preconditioner<DIM> :: CreateRowVector () const
  {
    auto blf_mat = blf->GetMatrixPtr();
    if (blf_mat == nullptr)
      { throw Exception("mat not ready"); }
    return blf_mat->CreateRowVector();
  }


  template<int DIM> shared_ptr<BaseVector> TDNNS_AUX_AMG_Preconditioner<DIM> :: CreateAuxVector () const
  {
    if (aux_pardofs != nullptr)
      { return make_shared<ParallelVVector<TV>> (pmat->Width(), aux_pardofs, DISTRIBUTED); }
    else
      { return make_shared<VVector<TV>> (pmat->Width()); }
  } // TDNNS_AUX_AMG_Preconditioner::CreateAuxVector


  template<int DIM> template<ELEMENT_TYPE ET> INLINE
  void TDNNS_AUX_AMG_Preconditioner<DIM> :: CalcElP (ElementId vol_elid, FlatArray<int> udm, FlatMatrix<double> Pmat, LocalHeap & lh)
  {
    HeapReset hr(lh);
    
    // cout << "CalcElP!" << endl;

    auto facetnrs = ma->GetElFacets(vol_elid);
    auto nfacets = facetnrs.Size();

    // cout << " facetnrs: "; prow(facetnrs); cout << endl;
    
    int nudm = udm.Size();

    FlatMatrix<double> M(nudm, nudm, lh); M = 0;
    FlatMatrix<double> Mmix(nudm, nfacets * DPV(), lh); Mmix = 0;

    ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
    ELEMENT_TYPE et_vol = ET;
    auto & hcfe = static_cast<HCurlHighOrderFE<ET>&>(hcurl->GetFE(vol_elid, lh));
    auto & hdfe = static_cast<HDivHighOrderFE<ET>&>(hdiv->GetFE(vol_elid, lh));
    int hc_nd_vol = hcfe.GetNDof(), hd_nd_vol = hdfe.GetNDof();

    int ir_order = 2 * max(1, max(hdfe.Order(), hcfe.Order()));

    FlatMatrix<double> all_shapes_hd(hd_nd_vol, DIM, lh), all_shapes_hdd (hd_nd_vol, DIM, lh), all_shapes_hc(hc_nd_vol, DIM, lh),
      all_shapes_hcd(hc_nd_vol, DIM, lh), shapes_nc(DPV(), DIM, lh);

    Array<int> hc_dnums(hcfe.GetNDof(), lh);
    hcurl->GetDofNrs(vol_elid, hc_dnums);
    Array<int> hd_dnums(hdfe.GetNDof(), lh);
    hdiv->GetDofNrs(vol_elid, hd_dnums);

    Array<int> hce2u(hcfe.GetNDof(), lh), hce2v(hcfe.GetNDof(), lh);
    hce2u.SetSize(0); hce2v.SetSize0();

    Array<int> hcenrs(nudm, lh); hcenrs.SetSize0();

    if constexpr(DIM == 3) {
      /** HCurl edge contribs - each edge couples to two faces **/
      Facet2ElementTrafo seg_2_vol(et_vol, BBND);
      const IntegrationRule & ir_seg = SelectIntegrationRule (ET_SEGM, ir_order); // reference segment
      auto edgenrs = ma->GetElEdges(vol_elid);
      Array<int> allefaces(10, lh), efaces(2,lh);
      Array<int> ednrs(nudm, lh);
      Array<int> e2used(nudm, lh), e2vol(nudm, lh);
      for (auto loc_enr : Range(edgenrs)) {
	/** get VOL-intrule on this edge **/
	IntegrationRule & ir_vol = seg_2_vol(loc_enr, ir_seg, lh); // reference VOL
	auto & basemir = eltrans(ir_vol, lh); // mapped VOL
	auto & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
	auto enr = edgenrs[loc_enr];
	/** local numering of hcurl-dofs in this edge **/
	hcurl->GetEdgeDofNrs(enr, ednrs);
	e2used.SetSize0(); e2vol.SetSize0();
	for (auto i : Range(ednrs)) {
	  auto hc_dof = ednrs[i];
	  auto comp_dof = os_hc + ednrs[i];
	  auto pos = udm.Pos(comp_dof);
	  if (pos != -1) {
	    e2used.Append(pos);
	    e2vol.Append(hc_dnums.Pos(hc_dof));
	    hcenrs.Append(pos);
	  }
	}
	hce2u.Append(e2used);
	hce2v.Append(e2vol);
	/** get faces common to this edge and the vol-el **/
	ma->GetEdgeFaces(enr, allefaces);
	// cout << " all faces "; prow(allefaces); cout << endl;
	efaces.SetSize0();
	for (auto k : Range(allefaces)) {
	  auto pos = facetnrs.Pos(allefaces[k]);
	  if (pos != -1)
	    { efaces.Append(pos); }
	}
	/** construct NC-elements and get local NC dofs **/
	Array<FacetRBModeFE<DIM>> ncfes(efaces.Size(), lh);
	// Array<int> loc_nc_dnrs(efaces.Size()*DPV(), lh); // should be 2 * DPV()
	// int c = 0;
	for (auto I : Range(efaces)) {
	  ncfes[I] = FacetRBModeFE<DIM>(ma, facetnrs[efaces[I]]);
	  // for (auto II : Range(efaces[I]*DPV(), (efaces[I]+1)*DPV()))
	    // { loc_nc_dnrs[c++] = II; }
	}
	// cout << " loc enr " << loc_enr << ", enr " << enr << endl;
	// cout << " M dims " << M.Height() << " x " << M.Width() << endl;
	// cout << " Mmix dims " << Mmix.Height() << " x " << Mmix.Width() << endl;
	// cout << " use faces "; prow(efaces); cout << endl;
	// cout << " e2used "; prow(e2used); cout << endl;
	// cout << " e2vol  "; prow(e2vol); cout << endl;
	// cout << " nc2vol  "; prow(loc_nc_dnrs); cout << endl;
	/** compute the integrals **/
	for (auto ip_nr : Range(mir_vol)) {
	  auto mip = mir_vol[ip_nr];
	  hcfe.CalcMappedShape(mip, all_shapes_hc); auto hce = all_shapes_hc.Rows(e2vol);
	  hcfe.CalcDualShape(mip, all_shapes_hcd); auto hcde = all_shapes_hcd.Rows(e2vol);
	  // cout << " hc shapes " << endl << hce << endl;
	  // cout << " hc dual shapes " << endl << hcde << endl;
	  M.Rows(e2used).Cols(e2used) += mip.GetWeight() * hcde * Trans(hce);
	  // cout << " go for faces " << endl;
	  for (auto I : Range(efaces)) {
	    auto facenr = efaces[I];
	    auto & ncfe = ncfes[I];
	    ncfe.CalcMappedShape(mip, shapes_nc);
	    // cout << " nc shapes " << endl << shapes_nc << endl;
	    Mmix.Rows(e2used).Cols(efaces[I]*DPV(), (efaces[I]+1)*DPV()) += 0.5 * mip.GetWeight() * hcde * Trans(shapes_nc); // 0.5 fixed
	  }
	}
      }
    }

    /** HDiv (and possibly HCurl) facet contribs **/
    {
      auto get_f_dnums = [&] (const auto & space, auto facet_nr, auto & dnums) {
	if constexpr (DIM == 2)
	{ space->GetEdgeDofNrs(facet_nr, dnums); }
	else
	  { space->GetFaceDofNrs(facet_nr, dnums); }
      };
      Array<int> hdf2u(nudm, lh), hdf2v(nudm, lh), hcf2u(nudm, lh), hcf2v(nudm, lh);
      Array<int> hdfnrs(nudm, lh), hcfnrs(nudm, lh);
      for (auto I : Range(facetnrs)) {
	auto facetnr = facetnrs[I];
	ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (ET, I);
	/** VOL-intrule on facet **/
	const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
	Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
	IntegrationRule & ir_vol = facet_2_el(I, ir_facet, lh); // reference VOL
	BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
	MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
	/** NC finite element **/
	int nclow = I*DPV(), nchi = (I+1)*DPV();
	FacetRBModeFE<DIM> ncfe(ma, facetnr);
	/** loc. HD/HC dnrs**/
	auto get_loc_dnrs = [&](auto & space, auto & d2used, auto & d2vol, auto & f_dnrs, auto & vol_dnums, auto os) {
	  get_f_dnums(space, facetnr, f_dnrs);
	  d2used.SetSize0(); d2vol.SetSize0(); 
	  for (auto k : Range(f_dnrs)) {
	    auto dof = f_dnrs[k];
	    auto comp_dof = os + dof;
	    auto pos = udm.Pos(comp_dof);
	    if (pos != -1) {
	      d2used.Append(pos);
	      d2vol.Append(vol_dnums.Pos(dof));
	    }
	  }
	};
	get_loc_dnrs(hcurl, hcf2u, hcf2v, hcfnrs, hc_dnums, os_hc);
	get_loc_dnrs(hdiv , hdf2u, hdf2v, hdfnrs, hd_dnums, os_hd);
	// cout << " facet contrib " << I << ", facetnr " << facetnr << endl;
	/** do the integrals **/
	for (auto ip_nr : Range(mir_vol)) {
	  auto mip = mir_vol[ip_nr];
	  ncfe.CalcMappedShape(mip, shapes_nc);
	  hdfe.CalcMappedShape(mip, all_shapes_hd); auto shd = all_shapes_hd.Rows(hdf2v);
	  hdfe.CalcDualShape(mip, all_shapes_hdd); auto shdd = all_shapes_hdd.Rows(hdf2v);
	  M.Rows(hdf2u).Cols(hdf2u) += shdd * Trans(shd);
	  Mmix.Rows(hdf2u).Cols(nclow, nchi) += shdd * Trans(shapes_nc);
	  if (hcfnrs.Size() > 1) {
	    hcfe.CalcMappedShape(mip, all_shapes_hc);
	    hcfe.CalcDualShape(mip, all_shapes_hcd);
	    auto hcdf = all_shapes_hc.Rows(hcf2v);
	    /** F - E (facet-functionals applied to edge-dofs) **/
	    auto shce = all_shapes_hc.Rows(hce2v);
	    M.Rows(hcf2u).Cols(hce2u) += hcdf * Trans(shce);
	    /** F - F (facet-functionals applied to facet dofs) **/
	    auto shcf = all_shapes_hc.Rows(hcf2v);
	    M.Rows(hcf2u).Cols(hcf2u) += hcdf * Trans(shcf);
	    Mmix.Rows(hcf2u).Cols(nclow, nchi) += hcdf * Trans(shapes_nc);
	  }
	}
      }
    }

    // cout << "Mmix: " << endl << Mmix << endl;
    // cout << "M: " << endl << M << endl;
    CalcInverse(M);
    // cout << "inv M: " << endl << M << endl;

    Pmat = M * Mmix;

    // cout << "Pmat: " << endl << Pmat << endl;
  } // TDNNS_AUX_AMG_Preconditioner::CalcElP

  template<int DIM> template<ELEMENT_TYPE ET> INLINE
  void TDNNS_AUX_AMG_Preconditioner<DIM> :: CalcFacetMat (ElementId vol_elid, int facet_nr, LocalHeap & lh,
							  FlatArray<int> hd_f_in_vol, FlatArray<int> hc_f_in_vol,
							  FlatArray<Array<int>> hc_e_in_vol, FlatArray<Array<int>> hc_e_in_f)
  {
    HeapReset hr(lh);

    // cout << " calc facet mat for " << facet_nr << ", vol el " << vol_elid << endl;
    // cout << " HD f->vol "; prow(hd_f_in_vol); cout << endl;
    // cout << " HC f->vol "; prow(hc_f_in_vol); cout << endl;
    // if (DIM == 3) {
    //   cout << " HC e->vol " << endl;
    //   for (auto k : Range(hc_e_in_vol)) {
    // 	cout << "           " << k << ": "; prow(hc_e_in_vol[k]); cout << endl;
    //   }
    //   cout << " HC e->f " << endl;
    //   for (auto k : Range(hc_e_in_f))
    // 	{ cout << "         " << k << ": "; prow(hc_e_in_f[k]); cout << endl; }
    // }

    ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);

    auto el_facet_nrs = ma->GetElFacets(vol_elid);
    int loc_facet_nr = el_facet_nrs.Pos(facet_nr);
    ELEMENT_TYPE et_vol = ET;
    ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
    auto & hcfe = static_cast<HCurlHighOrderFE<ET>&>(hcurl->GetFE(vol_elid, lh));
    auto & hdfe = static_cast<HDivHighOrderFE<ET>&>(hdiv->GetFE(vol_elid, lh));
    Vec<DIM> mid = 0;
    Array<int> pnums;
    ma->GetFacetPNums(facet_nr, pnums);
    for (auto pnum : pnums)
      { mid += 1.0/pnums.Size() * ma->GetPoint<DIM>(pnum); }
    facet_cos[facet_nr] = mid;
    FacetRBModeFE<DIM> ncfe (mid);
    
    int hc_nd_vol = hcfe.GetNDof(), hc_nd_f = hc_f_in_vol.Size();
    int hc_nd_e = 0; for (auto & ar : hc_e_in_vol) { hc_nd_e += ar.Size(); }
    int hc_nd_fullf = hc_nd_e + hc_nd_f;
    int hd_nd_vol = hdfe.GetNDof(), hd_nd_f = hd_f_in_vol.Size();
    int nc_nd = DPV();
    int r0 = 0, r1 = hd_nd_f, r2 = hd_nd_f + hc_nd_e, r3 = hd_nd_f + hc_nd_e + hc_nd_f;

    // cout << " ranges " << r0 << " " << r1 << " " << r2 << " " << r3 << endl;

    FlatMatrix<double> all_shapes_hd(hd_nd_vol, DIM, lh), all_shapes_hdd (hd_nd_vol, DIM, lh), all_shapes_hc(hc_nd_vol, DIM, lh),
      all_shapes_hcd(hc_nd_vol, DIM, lh), shapes_nc(nc_nd, DIM, lh);

    FlatMatrix<double> hcd_hc(hc_nd_fullf, hc_nd_fullf, lh), hcd_nc(hc_nd_fullf,  nc_nd, lh);
    hcd_hc = 0; hcd_nc = 0;

    FlatMatrix<double> hdd_hd(hd_nd_f, hd_nd_f, lh), hdd_nc(hd_nd_f,  nc_nd, lh);
    hdd_hd = 0; hdd_nc = 0;
    
    // auto phw = [&](auto & x, string name) {
    //   cout << name << " dims " << x.Height() << " " << x.Width() << endl;
    // };
    // phw(hcd_hc, "hcd_hc");
    // phw(hcd_nc, "hcd_nc");
    // phw(hdd_hd, "hdd_hd");
    // phw(hdd_nc, "hdd_nc");

    /** BBND contribs **/
    if constexpr(DIM == 3) {
      if (r2 > r1) {
	int ir_order = 2 * max(1, hcfe.Order());
	const IntegrationRule & ir_seg = SelectIntegrationRule (ET_SEGM, ir_order); // reference segment
	Facet2ElementTrafo seg_2_vol(et_vol, BBND);
	auto vol_el_edges = ma->GetElEdges(vol_elid);
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto loc_enr : Range(ElementTopology::GetNEdges(et_facet))) {
	  HeapReset hr(lh);
	  auto volel_enr = vol_el_edges.Pos(facet_edges[loc_enr]);
	  IntegrationRule & ir_vol = seg_2_vol(volel_enr, ir_seg, lh); // reference VOL
	  auto & basemir = eltrans(ir_vol, lh); // mapped VOL
	  auto & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
	  auto & e2v = hc_e_in_vol[loc_enr];
	  auto & e2f = hc_e_in_f[loc_enr];
	  auto n = e2v.Size();
	  for (auto ip_nr : Range(mir_vol)) {
	    auto mip = mir_vol[ip_nr];
	    ncfe.CalcMappedShape(mip, shapes_nc); 
	    hcfe.CalcMappedShape(mip, all_shapes_hc); auto shapes_hc = all_shapes_hc.Rows(e2v);
	    hcfe.CalcDualShape(mip, all_shapes_hcd); auto shapes_hcd = all_shapes_hcd.Rows(e2v);
	    hcd_hc.Rows(e2f).Cols(e2f) += mip.GetWeight() * shapes_hcd * Trans(shapes_hc);
	    hcd_nc.Rows(e2f) += mip.GetWeight() * shapes_hcd * Trans(shapes_nc);
	  }
	  // cout << " hcd_hc after " << loc_enr << endl;
	  // cout << hcd_hc << endl;
	}
      }
    }
    
    /** BND contribs **/
    {
      HeapReset hr(lh);
      int ir_order = 2 * max(1, max(hcfe.Order(), hdfe.Order()));
      const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
      Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
      IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
      BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
      MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
      mir_vol.ComputeNormalsAndMeasure(et_vol, loc_facet_nr); // ?? I THINK ??
      int lo = hc_nd_e, hi = hc_nd_e + hc_nd_f;
      for (auto ip_nr : Range(mir_vol)) {
	auto mip = mir_vol[ip_nr];
	ncfe.CalcMappedShape(mip, shapes_nc);
	// {
	//   Vec<DIM> n = mip.GetNV();
	//   for (auto k : Range(shapes_nc.Height())) {
	//     double ip = InnerProduct(shapes_nc.Row(k), n);
	//     shapes_nc.Row(k) = ip * n;
	//   }
	// }
	hdfe.CalcMappedShape(mip, all_shapes_hd); auto shapes_hd = all_shapes_hd.Rows(hd_f_in_vol);
	hdfe.CalcDualShape(mip, all_shapes_hdd); auto shapes_hdd = all_shapes_hdd.Rows(hd_f_in_vol);
	hdd_hd += mip.GetWeight() * shapes_hdd * Trans(shapes_hd);
	hdd_nc += mip.GetWeight() * shapes_hdd * Trans(shapes_nc);
	if (hi > lo) { // THATS WRONG !! I CANT JUST INVERT THE FACE-FACE block, there is also an edge-face in this row!!
	  // {
	  //   ncfe.CalcMappedShape(mip, shapes_nc);
	  //   Vec<DIM> n = mip.GetNV();
	  //   for (auto k : Range(shapes_nc.Height())) {
	  //     double ip = InnerProduct(shapes_nc.Row(k), n);
	  //     shapes_nc.Row(k) -= ip * n;
	  //   }
	  // }
	  hcfe.CalcMappedShape(mip, all_shapes_hc); auto shapes_hc = all_shapes_hc.Rows(hc_f_in_vol);
	  hcfe.CalcDualShape(mip, all_shapes_hcd); auto shapes_hcd = all_shapes_hcd.Rows(hc_f_in_vol);
	  for (auto loc_enr : Range(hc_e_in_vol)) {
	    auto & e2v = hc_e_in_vol[loc_enr];
	    auto & e2f = hc_e_in_f[loc_enr];
	    hcd_hc.Rows(lo,hi).Cols(e2f) += mip.GetWeight() * shapes_hcd * Trans(all_shapes_hc.Rows(e2v));
	  }
	  hcd_hc.Rows(lo,hi).Cols(lo,hi) += mip.GetWeight() * shapes_hcd * Trans(shapes_hc);
	  hcd_nc.Rows(lo,hi) += mip.GetWeight() * shapes_hcd * Trans(shapes_nc);
	}
      }
    }
    // cout << " hcd_hc " << endl << hcd_hc << endl;
    // cout << " hdd_hd " << endl << hdd_hd << endl;
    CalcInverse(hcd_hc); CalcInverse(hdd_hd);
    // cout << " inv hcd_hc " << endl << hcd_hc << endl;
    // cout << " inv hdd_hd " << endl << hdd_hd << endl;

    auto & fmat = facet_mat[facet_nr]; fmat = 0;

    fmat.Rows(r0,r1) = hdd_hd * hdd_nc;
    // cout << " facet mat + HD" << endl << fmat << endl;
    fmat.Rows(r1,r3) = hcd_hc * hcd_nc;
    // cout << " final facet mat " << endl << fmat << endl;
  } // TDNNS_AUX_AMG_Preconditioner::CalcFacetMat


  template<int DIM> template<ELEMENT_TYPE ET> INLINE
  void TDNNS_AUX_AMG_Preconditioner<DIM> :: CalcFacetMat2 (ElementId vol_elid, int facet_nr, LocalHeap & lh,
							   FlatArray<int> hd_f_in_vol, FlatArray<int> hc_f_in_vol,
							   FlatArray<Array<int>> hc_e_in_vol, FlatArray<Array<int>> hc_e_in_f)
  {
    HeapReset hr(lh);
    // cout << " calc facet mat for " << facet_nr << ", vol el " << vol_elid << endl;
    // cout << " HD f->vol "; prow(hd_f_in_vol); cout << endl;
    // cout << " HC f->vol "; prow(hc_f_in_vol); cout << endl;
    // if (DIM == 3) {
    //   cout << " HC e->vol " << endl;
    //   for (auto k : Range(hc_e_in_vol)) {
    // 	cout << "           " << k << ": "; prow(hc_e_in_vol[k]); cout << endl;
    //   }
    //   cout << " HC e->f " << endl;
    //   for (auto k : Range(hc_e_in_f))
    // 	{ cout << "         " << k << ": "; prow(hc_e_in_f[k]); cout << endl; }
    // }
    ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
    auto el_facet_nrs = ma->GetElFacets(vol_elid);
    int loc_facet_nr = el_facet_nrs.Pos(facet_nr);
    ELEMENT_TYPE et_vol = ET;
    ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
    auto & hcfe = static_cast<HCurlHighOrderFE<ET>&>(hcurl->GetFE(vol_elid, lh));
    auto & hdfe = static_cast<HDivHighOrderFE<ET>&>(hdiv->GetFE(vol_elid, lh));
    Vec<DIM> mid = 0;
    Array<int> pnums;
    ma->GetFacetPNums(facet_nr, pnums);
    for (auto pnum : pnums)
      { mid += 1.0/pnums.Size() * ma->GetPoint<DIM>(pnum); }
    facet_cos[facet_nr] = mid;
    FacetRBModeFE<DIM> ncfe (mid);
    
    int hc_nd_vol = hcfe.GetNDof(), hc_nd_f   = hc_f_in_vol.Size();
    int hc_nd_e = 0; for (auto & ar : hc_e_in_vol) { hc_nd_e += ar.Size(); }
    int hd_nd_vol = hdfe.GetNDof(), hd_nd_f = hd_f_in_vol.Size();
    int nc_nd = DPV();
    int r0 = 0, r1 = hd_nd_f, r2 = hd_nd_f + hc_nd_e, r3 = hd_nd_f + hc_nd_e + hc_nd_f;

    auto & fmat = facet_mat[facet_nr]; fmat = 0;

    FlatMatrix<double> all_shapes_hd(hd_nd_vol, DIM, lh), all_shapes_hdd (hd_nd_vol, DIM, lh), all_shapes_hc(hc_nd_vol, DIM, lh),
      all_shapes_hcd(hc_nd_vol, DIM, lh), shapes_nc(nc_nd, DIM, lh);

    /** Facet contributions **/
    {
      HeapReset hr(lh);
      int ir_order = 2 * max(1, max(hcfe.Order(), hdfe.Order()));
      const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
      Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
      IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
      BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
      MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
      mir_vol.ComputeNormalsAndMeasure(et_vol, loc_facet_nr); // ?? I THINK ??
      FlatMatrix<double> hdd_hd (hd_nd_f, hd_nd_f, lh), hdd_nc(hd_nd_f, nc_nd, lh),
	hcd_hc(hc_nd_f, hc_nd_f, lh), hcd_nc(hc_nd_f, nc_nd, lh);
      FlatMatrix<double> X(hc_nd_f, hc_nd_f, lh);
      hcd_hc = 0; hcd_nc = 0;
      hdd_hd = 0; hdd_nc = 0;
      for (auto ip_nr : Range(mir_vol)) {
	auto mip = mir_vol[ip_nr];
	ncfe.CalcMappedShape(mip, shapes_nc);
	{
	  Vec<DIM> n = mip.GetNV();
	  for (auto k : Range(shapes_nc.Height())) {
	    double ip = InnerProduct(shapes_nc.Row(k), n);
	    shapes_nc.Row(k) = ip * n;
	  }
	}
	hdfe.CalcMappedShape(mip, all_shapes_hd); auto shapes_hd = all_shapes_hd.Rows(hd_f_in_vol);
	hdfe.CalcDualShape(mip, all_shapes_hdd); auto shapes_hdd = all_shapes_hdd.Rows(hd_f_in_vol);
	hdd_hd += mip.GetWeight() * shapes_hdd * Trans(shapes_hd);
	hdd_nc += mip.GetWeight() * shapes_hdd * Trans(shapes_nc);
	if (r3 > r2) { // THATS WRONG !! I CANT JUST INVERT THE FACE-FACE block, there is also an edge-face in this row!!
	  {
	    ncfe.CalcMappedShape(mip, shapes_nc);
	    Vec<DIM> n = mip.GetNV();
	    for (auto k : Range(shapes_nc.Height())) {
	      double ip = InnerProduct(shapes_nc.Row(k), n);
	      shapes_nc.Row(k) -= ip * n;
	    }
	  }
	  hcfe.CalcMappedShape(mip, all_shapes_hc); auto shapes_hc = all_shapes_hc.Rows(hc_f_in_vol);
	  hcfe.CalcDualShape(mip, all_shapes_hcd); auto shapes_hcd = all_shapes_hcd.Rows(hc_f_in_vol);
	  // cout << " mip " << mip << endl;
	  // cout << "ALL hc shapes " << endl << all_shapes_hc << endl;
	  // cout << "ALL hc shapes dual " << endl << all_shapes_hcd << endl;
	  // cout << " hc shapes " << endl << shapes_hc << endl;
	  // cout << " hc shapes dual " << endl << shapes_hcd << endl;
	  // cout << "weight " << mip.GetWeight() << endl;
	  X = mip.GetWeight() * shapes_hcd * Trans(shapes_hc);
	  // cout << " contrib: " << endl << X << endl;
	  hcd_hc += mip.GetWeight() * shapes_hcd * Trans(shapes_hc);
	  // cout << " partial hcd_hc " << endl << hcd_hc << endl;
	  hcd_nc += mip.GetWeight() * shapes_hcd * Trans(shapes_nc);
	}
      }
      // cout << " facet hdd_hd " << endl << hdd_hd << endl;
      // cout << " facet hdd_nc " << endl << hdd_nc << endl;
      CalcInverse(hdd_hd);
      fmat.Rows(r0, r1) = hdd_hd * hdd_nc;
      if (r3 > r2) {
	// cout << " facet hcd_hc " << endl << hcd_hc << endl;
	// cout << " facet hcd_nc " << endl << hcd_nc << endl;
	CalcInverse(hcd_hc);
	fmat.Rows(r2, r3) = hcd_hc * hcd_nc;
      }
    }
    // cout << " facet facet mat: " << endl << fmat << endl;

    /** Edge contributions **/
    if constexpr(DIM==3) {
      if (r2 > r1)
	{
	  int ir_order = 2 * max(1, hcfe.Order());
	  const IntegrationRule & ir_seg = SelectIntegrationRule (ET_SEGM, ir_order); // reference segment
	  Facet2ElementTrafo seg_2_vol(et_vol, BBND);
	  auto vol_el_edges = ma->GetElEdges(vol_elid);
	  auto facet_edges = ma->GetFaceEdges(facet_nr);
	  for (auto loc_enr : Range(ElementTopology::GetNEdges(et_facet))) {
	    HeapReset hr(lh);
	    auto volel_enr = vol_el_edges.Pos(facet_edges[loc_enr]);
	    IntegrationRule & ir_vol = seg_2_vol(volel_enr, ir_seg, lh); // reference VOL
	    auto & basemir = eltrans(ir_vol, lh); // mapped VOL
	    auto & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
	    auto & loc_edofs = hc_e_in_vol[loc_enr];
	    auto n = loc_edofs.Size();
	    FlatMatrix<double> hcd_hc(n, n, lh), hcd_nc(n,  nc_nd, lh);
	    hcd_hc = 0; hcd_nc = 0;
	    for (auto ip_nr : Range(mir_vol)) {
	      auto mip = mir_vol[ip_nr];
	      ncfe.CalcMappedShape(mip, shapes_nc); 
	      hcfe.CalcMappedShape(mip, all_shapes_hc); auto shapes_hc = all_shapes_hc.Rows(loc_edofs);
	      hcfe.CalcDualShape(mip, all_shapes_hcd); auto shapes_hcd = all_shapes_hcd.Rows(loc_edofs);
	      hcd_hc += mip.GetWeight() * shapes_hcd * Trans(shapes_hc);
	      hcd_nc += mip.GetWeight() * shapes_hcd * Trans(shapes_nc);
	    }
	    // cout << " edge hcd_hc " << endl << hcd_hc << endl;
	    // cout << " edge hcd_nc " << endl << hcd_nc << endl;
	    CalcInverse(hcd_hc);
	    fmat.Rows(r1, r2).Rows(hc_e_in_f[loc_enr]) = hcd_hc * hcd_nc;
	    // cout << " facet + edges [0.." << loc_enr << "] facet mat: " << endl << fmat << endl;
	  }
	}
      }
    // cout << " final facet mat: " << endl << fmat << endl;
  } // CalcFacetMat


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: SetUpFacetMats ()
  {
    /** allocate facet matrix data **/
    auto nfacets = ma->GetNFacets(); // ...
    TableCreator<int> ctab(nfacets);
    lors.SetSize(nfacets); lors = 0;
    Array<int> dnums(100); Array<int> lodnums(15*hc_per_facetfacet);
    auto get_f_dnums = [&] (const auto & space, auto facet_nr) {
      if constexpr (DIM == 2)
      { space->GetEdgeDofNrs(facet_nr, dnums); }
      else
	{ space->GetFaceDofNrs(facet_nr, dnums); }
    };
    for (; !ctab.Done(); ctab++) {
      lors = 0;
      for (auto facet_nr : Range(nfacets)) {
	auto & lor = lors[facet_nr];
	int & n_hd = lor[0], n_hc_e = lor[1], n_hc_f = lor[2];
	auto facet_type = ma->GetFacetType(facet_nr);
	get_f_dnums(hdiv, facet_nr);
	for (auto j : Range(min2(hd_per_facet, int(dnums.Size()))))
	  { ctab.Add(facet_nr, os_hd + dnums[j]); n_hd++; }
	if constexpr(DIM==3) {
	  if (hc_per_facetfacet > 0) {
	    lodnums.SetSize0();
	    for (auto enr : ma->GetFaceEdges(facet_nr)) {
	      hcurl->GetEdgeDofNrs(enr, dnums);
	      for (auto j : Range(min2(hc_per_facetfacet, int(dnums.Size()))))
		{ lodnums.Append(os_hc + dnums[j]); }
	    }
	    QuickSort(lodnums);
	    ctab.Add(facet_nr, lodnums);
	    n_hc_e = lodnums.Size();
	  }
	}
	get_f_dnums(hcurl, facet_nr);
	for (auto j : Range(min2(hc_per_facet, int(dnums.Size()))))
	  { ctab.Add(facet_nr, os_hc + dnums[j]); n_hc_f++; }
      }
    }
    facet_lo_dofs = ctab.MoveTable();
    // cout << "facet_lo_dofs: " << endl;
    // cout << facet_lo_dofs << endl;
    size_t s = 0, nd = 0;
    for (auto facet_nr : Range(nfacets))
      { s += facet_lo_dofs[facet_nr].Size() * DPV(); }
    facet_mat_data.SetSize(s); s = 0;
    facet_mat.SetSize(nfacets);
    facet_cos.SetSize(nfacets);
    for (auto facet_nr : Range(nfacets)) {
      auto nd = facet_lo_dofs[facet_nr].Size();
      facet_mat[facet_nr].AssignMemory(nd, DPV(), facet_mat_data.Addr(s));
      s += nd * DPV();
    }
    // cout << " have facet mat mem" << endl;
    LocalHeap clh (20*1024*1024, "facet-lh");
    SharedLoop2 sl_facets(nfacets);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
	 Array<int> elnums_of_facet, facet_nrs, dnums;
	 Array<int> hc_dofs, hc_f_in_vol;
	 Array<int> hd_dofs, hd_f_in_vol;
	 Array<Array<int>> hc_e_in_vol, hc_e_in_f;
	 auto get_f_dnums = [&] (const auto & space, auto facet_nr) {
	   if constexpr (DIM == 2)
	   { space->GetEdgeDofNrs(facet_nr, dnums); }
	   else
	     { space->GetFaceDofNrs(facet_nr, dnums); }
	 };
	 for (int facet_nr : sl_facets) {
	   HeapReset hr(lh);
	   /** pick one vol-el **/
	   // cout << " facet nr " << facet_nr << endl;
	   ma->GetFacetElements(facet_nr, elnums_of_facet);
	   // cout << "vol elnums: "; prow(elnums_of_facet); cout << endl;
	   ElementId vol_elid(VOL, elnums_of_facet[0]);
	   ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
	   ELEMENT_TYPE et_vol = eltrans.GetElementType();
	   hcurl->GetDofNrs(vol_elid, hc_dofs);
	   hdiv->GetDofNrs(vol_elid, hd_dofs);

	   // cout << " hcurl dofs: "; prow2(hc_dofs); cout << endl;
	   // cout << " hdiv  dofs: "; prow2(hd_dofs); cout << endl;

	   auto all_lo_dofs = facet_lo_dofs[facet_nr];
	   get_f_dnums(hdiv, facet_nr);
	   hd_f_in_vol.SetSize0();
	   for (auto j : Range(dnums)) {
	     auto hd_dof = os_hd + dnums[j];
	     auto lopos = all_lo_dofs.Pos(hd_dof);
	     if (lopos != -1) {
	       auto volpos = hd_dofs.Pos(dnums[j]);
	       hd_f_in_vol.Append(volpos);
	     }
	   }
	   auto nhdds = hd_f_in_vol.Size();
	   auto hc_lo_dofs = all_lo_dofs.Part(nhdds);
	   get_f_dnums(hcurl, facet_nr);
	   hc_f_in_vol.SetSize0();
	   for (auto j : Range(dnums)) {
	     auto hc_dof = os_hc + dnums[j];
	     auto lopos = hc_lo_dofs.Pos(hc_dof);
	     if (lopos != -1) {
	       auto volpos = hc_dofs.Pos(dnums[j]);
	       hc_f_in_vol.Append(volpos);
	     }
	   }

	   /** Call templated facet mat method **/
	   if constexpr(DIM == 2) {
	     hc_e_in_vol.SetSize0();
	     hc_e_in_f.SetSize0();
	     switch(et_vol) {
	     case(ET_TRIG): { CalcFacetMat<ET_TRIG> (vol_elid, facet_nr, lh, hd_f_in_vol, hc_f_in_vol, hc_e_in_vol, hc_e_in_f); break; }
	     case(ET_QUAD): { CalcFacetMat<ET_QUAD> (vol_elid, facet_nr, lh, hd_f_in_vol, hc_f_in_vol, hc_e_in_vol, hc_e_in_f); break; }
	     default : { break; }
	     }
	   }
	   else {
	     auto facet_edges = ma->GetFaceEdges(facet_nr);
	     hc_e_in_vol.SetSize(facet_edges.Size());
	     hc_e_in_f.SetSize(facet_edges.Size());
	     // cout << " all lo "; prow(all_lo_dofs); cout << endl;
	     // cout << " part " << nhdds << " + " << all_lo_dofs.Size() - hc_f_in_vol.Size() << endl;
	     auto hc_lo_e_dofs = all_lo_dofs.Range(nhdds, all_lo_dofs.Size() - hc_f_in_vol.Size());
	     // cout << " hc lo "; prow(hc_lo_e_dofs); cout << endl;
	     for (auto loc_enr : Range(facet_edges)) {
	       auto enr = facet_edges[loc_enr];
	       hcurl->GetEdgeDofNrs(enr, dnums);
	       auto & e2vol = hc_e_in_vol[loc_enr]; e2vol.SetSize0();
	       auto & e2f   = hc_e_in_f[loc_enr]; e2f.SetSize0();
	       for (auto j : Range(dnums)) {
		 auto hc_dof = os_hc + dnums[j];
		 auto lopos = hc_lo_e_dofs.Pos(hc_dof);
		 if (lopos != -1) {
		   auto volpos = hc_dofs.Pos(dnums[j]);
		   // cout << "loc_enr " << loc_enr << " dnum " << j << ", hc_dof " << hc_dof << " lopos " << lopos << " volpos " << volpos << endl;
		   e2vol.Append(volpos);
		   e2f.Append(lopos);
		 }
	       }
	     }
	     switch(et_vol) {
	     case(ET_TET): { CalcFacetMat<ET_TET> (vol_elid, facet_nr, lh, hd_f_in_vol, hc_f_in_vol, hc_e_in_vol, hc_e_in_f); break; }
	     case(ET_HEX): { CalcFacetMat<ET_HEX> (vol_elid, facet_nr, lh, hd_f_in_vol, hc_f_in_vol, hc_e_in_vol, hc_e_in_f); break; }
	     default : { break; }
	     }
	   }

	 }
       });
  } // TDNNS_AUX_AMG_Preconditioner::SetUpFacetMats


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: BuildEmbedding ()
  {
    auto H = comp_fes->GetNDof();
    auto nfacets = ma->GetNFacets(); // ...
    auto W = nfacets;
    Array<int> perow_f(H), perow_d(H);
    perow_f = 0; perow_d = 0;
    const auto & free_facets = *aux_freedofs;
    /** For any edge, if there is a connected dirichlet facet, we have to average only
	facet matrices from Dirichlet facets. Otherwise, Dirichlet BCs are violated in the auxiliary space.
	TODO: Probably leads to problems with elements in "Dirichlet-Corners" **/
    for (auto facet_nr : Range(nfacets)) {
      auto lodofs = facet_lo_dofs[facet_nr];
      if (free_facets.Test(facet_nr)) {
	for (auto l : Range(lodofs))
	  { perow_f[lodofs[l]]++; }
      }
      else {
	for (auto l : Range(lodofs))
	  { perow_d[lodofs[l]]++; }
      }
    }
    for (auto dof : Range(H))
      { perow_f[dof] += perow_d[dof]; }
      // { perow_f[dof] = (perow_d[dof] == 0) ? perow_f[dof] : perow_d[dof]; }
    // for (auto dof : Range(H))
    pmat = make_shared<TPMAT> (perow_f, W);
    perow_f = 0; perow_d = 0;
    for (auto facet_nr : Range(nfacets)) {
      auto lodofs = facet_lo_dofs[facet_nr];
      // cout << " write facet mat " << facet_nr << " of " << facet_mat.Size() << endl;
      // cout << " lodofs are "; prow(lodofs); cout << endl;
      // cout << " addr " << &facet_mat[facet_nr] << endl;
      // cout << " dims " << facet_mat[facet_nr].Height() << " x " << facet_mat[facet_nr].Width() << endl;
      auto& fmat = facet_mat[facet_nr];
      // cout << fmat << endl;
      for (auto l : Range(lodofs)) {
	auto lodof = lodofs[l];
	auto ris = pmat->GetRowIndices(lodof);
	auto rvs = pmat->GetRowValues(lodof);
	auto j = perow_f[lodof]; perow_f[lodof]++;
	// cout << facet_nr << ", " << l << " " << lodof << endl;
	ris[j] = facet_nr;
	double fac = 1.0/ris.Size();
	// cout << " write row " << l << " to " << lodof << ", colnr " << j << endl; 
	for (auto k : Range(DPV()))
	  { rvs[j](0, k) = fac * fmat(l, k); }
      }
    }
    cout << "pmat: " << endl;
    print_tm_spmat(cout, *pmat); cout << endl;


    auto pcol = pmat->CreateColVector();
    auto prow = pmat->CreateRowVector();
    // cout << "PCOL TYPE "; cout << typeid(*pcol).name() << endl;
    // cout << "PROW TYPE "; cout << typeid(*prow).name() << endl;
  }


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: AllocAuxMat ()
  {
    auto nfacets = ma->GetNFacets();
    Array<int> elnums;
    auto itit = [&](auto lam) LAMBDA_INLINE {
      for (auto fnr : Range(nfacets)) {
	lam(fnr, fnr);
	ma->GetFacetElements(fnr, elnums);
	for (auto elnr : elnums) {
	  for (auto ofnum : ma->GetElFacets(ElementId(VOL, elnr)))
	    { if (ofnum != fnr) { lam(fnr, ofnum); } }
	}
      }
    };
    Array<int> perow(nfacets); perow = 0;
    itit([&](auto i, auto j) LAMBDA_INLINE { perow[i]++; });
    aux_mat = make_shared<SparseMatrix<Mat<DPV(), DPV(), double>>>(perow, nfacets);
    aux_mat->AsVector() = 0;
    perow = 0;
    itit([&](auto i, auto j) LAMBDA_INLINE {
	aux_mat->GetRowIndices(i)[perow[i]++] = j;
      });
    for (auto k : Range(nfacets))
      { QuickSort(aux_mat->GetRowIndices(k)); }

    auto auxrow = aux_mat->CreateRowVector();
    // cout << "AUX TYPE "; cout << typeid(*auxrow).name() << endl;
  }


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> :: SetUpAuxParDofs ()
  {
    if (!blf->GetFESpace()->IsParallel())
      { return; }
    auto nfacets = ma->GetNFacets(); // ...
    TableCreator<int> ct(nfacets);
    for (; !ct.Done(); ct++) {
      for (auto facet_nr : Range(nfacets)) {
	auto dps = ma->GetDistantProcs(NodeId(NT_FACET, facet_nr));
	for (auto p : dps)
	  { ct.Add(facet_nr, p); }
      }
    }
    aux_pardofs = make_shared<ParallelDofs> ( blf->GetFESpace()->GetParallelDofs()->GetCommunicator() , ct.MoveTable(), DPV(), false);
  }


  template<int DIM>
  Array<Array<shared_ptr<BaseVector>>> TDNNS_AUX_AMG_Preconditioner<DIM> :: GetRBModes () const
  {
    Array<Array<shared_ptr<BaseVector>>> rbms(2);

    auto blf_mat = blf->GetMatrixPtr();

    if (blf_mat == nullptr)
      { throw Exception("mat not ready"); }
    if (pmat == nullptr)
      { throw Exception("pmat not ready"); }
      
    const auto & P = *pmat;

    /** displacements: (1,0,0), (0,1,0), (0,0,1) **/
    for (auto comp : Range(DIM)) {
      auto w = CreateAuxVector();
      w->SetParallelStatus(CUMULATED);
      auto fw = w->template FV<TV>();
      auto v = blf_mat->CreateRowVector();
      v.SetParallelStatus(CUMULATED);
      auto fv = v.FV<double>();
      for (auto k : Range(fw.Size()))
	{ fw(k) = 0; fw(k)(comp) = 1; }
      // cout << " disp " << comp << ", w: " << endl << fw << endl;
      P.Mult(*w, *v);
      // cout << " disp " << comp << ", v: " << endl << fv << endl;
      rbms[0].Append(v);
      rbms[1].Append(w);
    }

    /** rotations **/
    if constexpr(DIM == 3)
      {
	/** (1,0,0) \cross x = (0, z, -y)
	    (0,1,0) \cross x = (-z, 0, y)
	    (0,0,1) \cross x = (y, -x, 0) **/
	Vec<3> r, cross;
	for (auto rnr : Range(3)) {
	  r = 0; r(rnr) = 1;
	  auto w = CreateAuxVector();
	  w->SetParallelStatus(CUMULATED);
	  auto fw = w->template FV<TV>();
	  // cout << " fw size " << fw.Size() << " nfacets " << ma->GetNFacets() << " ncos " << facet_cos.Size()
	       // << " pmw " << P.Width() << endl;
	  auto v = blf_mat->CreateRowVector();
	  v.SetParallelStatus(CUMULATED);
	  auto fv = v.FV<double>();
	  for (auto k : Range(fw.Size())) {
	    fw(k) = 0;
	    cross = Cross(r, facet_cos[k]);
	    for (auto l : Range(DIM))
	      { fw(k)(l) = cross(l); }
	    fw(k)(DIM + rnr) = 1;
	  }
	  P.Mult(*w, *v);
	  rbms[0].Append(v);
	  rbms[1].Append(w);
	  // cout << " rot " << rnr << ", w: " << endl << fw << endl;
	  // cout << " rot " << rnr << ", v: " << endl << fv << endl;
	}
      }
    else
      {
	throw Exception("not implemented...");
      }
    return rbms;
  } // TDNNS_AUX_AMG_Preconditioner :: GetRBModes


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> ::
  AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    AddElementMatrix4(dnums, elmat, ei, lh);
  }


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> ::
  AddElementMatrix4 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    static Timer t("TDNNS_AMG_PC_AddElementMatrix");
    RegionTimer rt(t);
    HeapReset hr(lh);

    auto facetnrs = ma->GetElFacets(ei);
    auto nfacets = facetnrs.Size();
    ElementTransformation & eltrans = ma->GetTrafo (ei, lh);
    ELEMENT_TYPE et_vol = eltrans.GetElementType();

    // cout << "add ET " << et_vol << endl;
	       
    Array<int> udnums(dnums.Size(), lh);
    Array<int> loc_used(dnums.Size(), lh);
    udnums.SetSize0(); loc_used.SetSize0();
    for (auto k : Range(dnums)) {
      if (dnums[k] != -1) {
	udnums.Append(dnums[k]);
      	loc_used.Append(k);
      }
    }

    int ncdofs = nfacets * DPV();
    FlatMatrix<double> Pmat(udnums.Size(), ncdofs, lh); Pmat = 0; // we return this

    auto get_facet_mat = [&](){
      auto fnum = facetnrs[0];
      auto lonums = facet_lo_dofs[fnum];
      // cout << " udnums "; prow(udnums); cout << endl;
      // cout << "lo dofs "; prow(lonums); cout << endl;
      FlatArray<int> locnums (lonums.Size(), lh);
      for (auto k : Range(lonums)) {
	auto pos = udnums.Pos(lonums[k]);
	locnums[k] = pos;
      }
      // cout << " facet mat: " << endl << facet_mat[fnum] << endl;
      // cout << " locnums "; prow(locnums); cout << endl;
      // cout << " pmat dims " << Pmat.Height() << " x " << Pmat.Width() << endl;
      Pmat.Rows(locnums) = facet_mat[fnum];
    };

    if constexpr(DIM == 2) {
      switch(et_vol) {
      case(ET_SEGM): { get_facet_mat(); break; }
      case(ET_TRIG): { CalcElP<ET_TRIG>(ei, udnums, Pmat, lh); break; }
      case(ET_QUAD): { CalcElP<ET_QUAD>(ei, udnums, Pmat, lh); break; }
      default : { break; }
      }
    }
    else {
      switch(et_vol) {
      case(ET_TRIG): { get_facet_mat(); break; }
      case(ET_QUAD): { get_facet_mat(); break; }
      case(ET_TET): { CalcElP<ET_TET>(ei, udnums, Pmat, lh); break; }
      case(ET_HEX): { CalcElP<ET_HEX>(ei, udnums, Pmat, lh); break; }
      default : { break; }
      }
    }

    // if (et_vol == ET_TRIG) {
      // cout << " dnums: "; prow(dnums); cout << endl;
      // cout << " udnums: "; prow(udnums); cout << endl;
      // cout << "Pmat: " << endl << Pmat << endl;
    // }

    FlatMatrix<double> facet_elmat (ncdofs, ncdofs, lh);
    facet_elmat = Trans(Pmat) * elmat.Rows(loc_used).Cols(loc_used) * Pmat;

    // {
    //   FlatMatrix<double> evecs (ncdofs, ncdofs, lh);
    //   FlatVector<double> evals (ncdofs, lh);
    //   LapackEigenValuesSymmetric(facet_elmat, evals, evecs);
    //   cout << " facet_elmat evals: " << endl; prow2(evals); cout << endl;
    // }

    // cout << " facet_mat: " << endl << facet_elmat << endl;
  
    auto & aux = *aux_mat;
    for (auto I : Range(facetnrs)) {
      auto fI = facetnrs[I];
      int ilo = I*DPV(), ihi = (I+1)*DPV();
      for (auto J : Range(facetnrs)) {
	auto fJ = facetnrs[J];
	int jlo = J*DPV(), jhi = (J+1)*DPV();
	aux(fI, fJ) += facet_elmat.Rows(ilo, ihi).Cols(jlo,jhi);
      }
    }

    // auto & P = *pmat;
    // for (auto i : Range(udnums)) {
    //   cout << "get " << i << " " << udnums.Size() << endl;
    //   auto di = udnums[i];
    //   cout << " di " << di << endl;
    //   auto ri = P.GetRowIndices(di);
    //   cout << " ri: "; prow(ri); cout << endl;
    //   auto rv = P.GetRowValues(di);
    //   for (auto J : Range(facetnrs)) {
    // 	auto fJ = facetnrs[J];
    // 	auto pos = ri.Pos(fJ);
    // 	if (pos != -1) {
    // 	  int jlo = J*DPV(), jhi = (J+1)*DPV();
    // 	  // P(di, fJ) = Pmat.Rows(i,i+1).Cols(jlo,jhi);
    // 	  auto x = P(di,fJ);
    // 	  cout << " PMAT " << di << " " << fJ << endl;
    // 	  print_tm(cout, x); cout << endl;
    // 	  cout << "el-PMAT " << di << " " << fJ << endl;
    // 	  cout << Pmat.Rows(i,i+1).Cols(jlo,jhi) << endl;
    // 	  x -= Pmat.Rows(i,i+1).Cols(jlo,jhi);
    // 	  cout << "DIFF " << di << " " << fJ << endl;
    // 	  print_tm(cout, x); cout << endl;
    // 	}
    //   }
    // }
    // cout << "pmat2: " << endl;
    // print_tm_spmat(cout, *pmat); cout << endl;

  }


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> ::
  AddElementMatrix2 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    /** Regularize facet-elmat with null space of P^TP **/
    static Timer t("TDNNS_AMG_PC_AddElementMatrix");
    RegionTimer rt(t);
    HeapReset hr(lh);

    // if (ei.Nr() != 0)
      // { return; }

    int ndof = dnums.Size();
    int ndof_used = 0;
    for (auto d : dnums)
      { if (d != -1) { ndof_used++; } }
    FlatArray<int> dnums_used (ndof_used, lh), dnums_used_loc(ndof_used, lh);
    ndof_used = 0;
    for (auto k : Range(dnums)) {
      auto d = dnums[k];
      if (d != -1) {
	dnums_used[ndof_used] = d;
	dnums_used_loc[ndof_used] = k;
	ndof_used++;
      }
    }

    cout << " dnums: "; prow2(dnums); cout << endl;
    cout << " dnums_used: "; prow2(dnums_used); cout << endl;
    // cout << " elmat: " << endl << elmat << endl;
    
    {
      FlatMatrix<double> used_elmat(ndof_used, ndof_used, lh), evecs(ndof_used, ndof_used, lh);
      FlatVector<double> evals(ndof_used, lh);
      used_elmat = elmat.Rows(dnums_used_loc).Cols(dnums_used_loc);
      LapackEigenValuesSymmetric(used_elmat, evals, evecs);
      cout << " orig USED elmat evals: "; prow(evals); cout << endl;
      // cout << " orig USED elmat evecs: " << endl << evecs << endl;
    }

    auto el_facet_nrs = ma->GetElFacets(ei);
    auto n_el_facets = el_facet_nrs.Size();

    Array<int> loc_lo_dofs(ndof_used); loc_lo_dofs.SetSize0();
    for (auto J : Range(el_facet_nrs)) {
      auto facet_J = el_facet_nrs[J];
      auto facet_J_lodofs = facet_lo_dofs[facet_J];
      for (auto jdof : facet_J_lodofs) {
	int pos = dnums_used.Pos(jdof);
	int pos2 = merge_pos_in_sorted_array(pos, loc_lo_dofs); // a[..pos2-1) <= pos < a[pos2..)
	if (pos2 == 0) 
	  { loc_lo_dofs.Insert(pos2, pos); }
	else if (loc_lo_dofs.Size() > pos2) {
	  if (loc_lo_dofs[pos2-1] != pos)
	    { loc_lo_dofs.Insert(pos2, pos); }
	}
	else
	  { loc_lo_dofs.Append(pos); }
      }
    }
    cout << " loc_lo_dofs: "; prow(loc_lo_dofs); cout << endl;
    FlatArray<int> lo_dofs(loc_lo_dofs.Size(), lh);
    for (auto j : Range(lo_dofs))
      { lo_dofs[j] = dnums_used[loc_lo_dofs[j]]; }
    cout << " lo_dofs: "; prow(lo_dofs); cout << endl;

    int ndof_used_lo = lo_dofs.Size();
    FlatArray<int> dnums_used_loc_lo(lo_dofs.Size(), lh);
    for (auto j : Range(ndof_used_lo))
      { dnums_used_loc_lo[j] = dnums_used_loc[loc_lo_dofs[j]]; }
    
    FlatMatrix<double> P_el (ndof_used_lo, DPV() * n_el_facets, lh); P_el = 0;
    for (auto I : Range(el_facet_nrs)) {
      auto facet_I = el_facet_nrs[I];
      auto facet_I_lodofs = facet_lo_dofs[facet_I];
      auto & facet_mat_I = facet_mat[facet_I];
      cout << "loc facet nr " << I << ", is " << facet_I << endl;
      cout << "facet dnrs :"; prow(facet_I_lodofs); cout << endl;
      cout << "loc   dnrs :";
      for (auto i : Range(facet_I_lodofs)) {
	auto idof = facet_I_lodofs[i];
	auto loc_idof = lo_dofs.Pos(idof);
	cout << loc_idof << " ";
	if (loc_idof == -1)
	  { throw Exception("i feared this might happen (maybe dirichlet/bddc or a combination thereof"); }
	/** strictly speaking, we have to differentiate between facet and edge hc dofs
	    but the hc facet dof rows are 0 anyways. **/
	bool is_hc = (ind_hc < ind_hd) ? (idof < os_hd) : (os_hc <= idof);
	double fac = (is_hc) ? 0.5 : 1;
	P_el.Rows(loc_idof, loc_idof+1).Cols(I*DPV(), (I+1)*DPV()) = fac * facet_mat_I.Row(i);
      }
      cout << endl;
    }
    
    cout << "P_el: " << endl << P_el << endl;

    FlatMatrix<double> PTP (P_el.Width(), P_el.Width(), lh);
    PTP = Trans(P_el) * P_el;
    FlatMatrix<double> REG (P_el.Width(), P_el.Width(), lh);

    FlatMatrix<double> evecs (P_el.Width(), P_el.Width(), lh);
    FlatVector<double> evals (P_el.Width(), lh);
    LapackEigenValuesSymmetric(PTP, evals, evecs);
    cout << " EVALS PTP "; prow(evals); cout << endl;

    PTP = Trans(P_el) * elmat.Rows(dnums_used_loc_lo).Cols(dnums_used_loc_lo) * P_el;

    {
      FlatMatrix<double> evecs (P_el.Width(), P_el.Width(), lh);
      FlatVector<double> evals (P_el.Width(), lh);
      LapackEigenValuesSymmetric(PTP, evals, evecs);
      cout << " EVALS facet_elmat "; prow(evals); cout << endl;
    }

    int nzero = P_el.Width();
    for (auto k : Range(evals))
      { if (fabs(evals[k]) > 1e-12) { nzero = k; break; } }
    cout << " PTP has " << nzero << " null space vecs " << endl;

    double fac = 0;
    for (auto k : Range(dnums))
      { fac += elmat(k,k); }
    fac /= dnums.Size();



    REG = Trans(evecs.Rows(0, nzero)) * evecs.Rows(0, nzero);
    FlatMatrix<double> kvs (nzero, P_el.Width(), lh);
    kvs = Trans(evecs.Rows(0, nzero));
    
    if constexpr(DIM == 3)
      {
	auto nfacets = el_facet_nrs.Size();
	cout << " test W elmat rbms" << endl;
	FlatMatrix<double> rbms (DPV(), DPV() * nfacets, lh); rbms = 0;
	for (auto rbm_nr : Range(6)) {
	  for (auto I : Range(el_facet_nrs))
	    { rbms(rbm_nr, I*DPV() + rbm_nr) = 1; }
	}
	Vec<3> ctr = {0,1,2};
	for (auto rot_nr : Range(3)) {
	  for (auto I : Range(nfacets)) {
	    Vec<3> r, cross;
	    r = 0; r(rot_nr) = 1;
	    Vec<3> x = facet_cos[el_facet_nrs[I]] - ctr;
	    cross = Cross(r, x);
	    for (auto l : Range(DIM))
	      { rbms(3+rot_nr, I*DPV() + l) = cross(l); }
	  }
	}
	for (auto rbm_nr : Range(6)) {
	  for (auto j : Range(rbm_nr))
	    { rbms.Row(rbm_nr) -= InnerProduct(rbms.Row(rbm_nr), rbms.Row(j)) * rbms.Row(j); }
	  rbms.Row(rbm_nr) /= L2Norm(rbms.Row(rbm_nr));
	}
	cout << " rbm cos: " << endl << rbms << endl;

	cout << "P_el w " << P_el.Width() << " = " << DPV() << " * " << nfacets << endl;

	FlatMatrix<double> kvup (nzero, DPV() * nfacets, lh);
	FlatMatrix<double> kv_dot_rbm (nzero, DPV(), lh);
	kv_dot_rbm = kvs * Trans(rbms);
	cout << "INIT kv_dot_rbm :" << endl; cout << kv_dot_rbm << endl;

	for (auto k : Range(6)) {
	  FlatMatrix<double> kv_x_rb (nzero, 1, lh);
	  kv_x_rb = kvs * Trans(rbms).Col(k);
	  cout << "kv_x_rb 0 = " << kv_x_rb << endl;
					       kvup = kv_x_rb * rbms.Row(k);
					       cout << "kvup" << endl << kvup;
	for (auto j : Range(6)) {
	  kvup.Row(j) = kv_x_rb(j) * rbms.Row(k);
	}
					       cout << "kvup" << endl << kvup;

									 kvs -= kvup;
	  kv_dot_rbm = kvup * Trans(rbms);
									 cout << "kv dow rbms " << endl << kv_dot_rbm << endl;
									 kv_x_rb = kvs * Trans(rbms).Col(k);
	  cout << "kv_x_rb 1 = " << kv_x_rb << endl;

	  kv_dot_rbm = kvs * Trans(rbms);
	  cout << "kv_dot_rbm " << k << " :" << endl; cout << kv_dot_rbm << endl;
	}
	
	    REG = Trans(kvs) * kvs;


	  for (auto rbm_nr : Range(6)) {
	    for (auto I : Range(el_facet_nrs))
	      { rbms(rbm_nr, I*DPV() + rbm_nr) = 1; }
	  }
	  for (auto rot_nr : Range(3)) {
	    for (auto I : Range(nfacets)) {
	      Vec<3> r, cross;
	      r = 0; r(rot_nr) = 1;
	      Vec<3> x = facet_cos[el_facet_nrs[I]] - ctr;
	      cross = Cross(r, x);
	      for (auto l : Range(DIM))
		{ rbms(3+rot_nr, I*DPV() + l) = cross(l); }
	    }
	  }


	  FlatMatrix<double> evecs_reg (P_el.Width(), P_el.Width(), lh);
	FlatVector<double> evals_reg (P_el.Width(), lh);
	LapackEigenValuesSymmetric(REG, evals_reg, evecs_reg);
	cout << "evals of REG "; prow(evals_reg); cout << endl;

	PTP += fac * REG;

	FlatMatrix<double> fm_rbms (PTP.Height(), DPV(), lh); fm_rbms = -1;
	fm_rbms = PTP * Trans(rbms);
	cout << "fm_rbms: " << endl;
	cout << fm_rbms << endl;
	fm_rbms = REG * Trans(rbms);
	cout << "REG * rbms: " << endl;
	cout << fm_rbms << endl;

	
	FlatMatrix<double> Pr (ndof_used_lo, DPV(), lh); Pr = -1;
	FlatMatrix<double> Pr_M_Pr (DPV(), DPV(), lh); Pr_M_Pr = -1;
	Pr = P_el * Trans(rbms);
	Pr_M_Pr = Trans(Pr) * elmat.Rows(dnums_used_loc_lo).Cols(dnums_used_loc_lo) * Pr;
	
	cout << "PrMPr" << endl << Pr_M_Pr << endl;
      }

    {
      FlatMatrix<double> evecs_reg (P_el.Width(), P_el.Width(), lh);
      FlatVector<double> evals_reg (P_el.Width(), lh);
      LapackEigenValuesSymmetric(PTP, evals_reg, evecs_reg);
      cout << " EVALS_REG reged facet_elmat "; prow(evals_reg); cout << endl;
      int nzero2 = P_el.Width();
      for (auto k : Range(evals_reg))
	{ if (fabs(evals_reg[k]) > 1e-12) { nzero2 = k; break; } }
      cout << "reged  elmat has " << nzero2 << " null evals " << endl;
      // cout << " EVECS_REG reged facet_elmat " << endl << evecs_reg << endl;
      FlatMatrix<double> Pker (ndof_used_lo, nzero2, lh);
      Pker = P_el * Trans(evecs_reg.Rows(0, nzero2));

      // FlatMatrix<double> PkP (Pker.Height(), Pker.Height(), lh);
      // PkP = Pker * Trans(Pker);
      // FlatMatrix<double> PkPevecs (Pker.Height(), Pker.Height(), lh);
      // FlatVector<double> PkPevals (Pker.Height(), lh);
      // LapackEigenValuesSymmetric(PkP, PkPevals, PkPevecs);
      // cout << "PkP evals; "; prow(PkPevals); cout << endl;
	
      FlatMatrix<double> elmat_Pker (ndof_used_lo, nzero2, lh);
      elmat_Pker = elmat.Rows(dnums_used_loc_lo).Cols(dnums_used_loc_lo) * Pker;
      cout << endl;
      cout << " elmat * P * ker_vecs: " << endl;
      cout << elmat_Pker << endl;
      FlatMatrix<double> PEP (nzero2, nzero2, lh);
      PEP = Trans(Pker) * elmat_Pker;
      cout << endl << "PEP: " << endl << PEP << endl;

      PEP = evecs_reg.Rows(0, nzero2) * PTP * Trans(evecs_reg.Rows(0, nzero2));
      cout << endl << "PEP2: " << endl << PEP << endl;
    }


    

    cout << "facet_elmat: " << endl << PTP << endl;
    
    auto & aux = (*aux_mat);
    for (auto I : Range(el_facet_nrs)) {
      auto facet_I = el_facet_nrs[I];
      const int osI = DPV() * I;
      for (int J = 0; J <= I; J++) {
	auto facet_J = el_facet_nrs[J];
	auto & etr = aux(facet_I, facet_J);
	auto & Tetr = aux(facet_J, facet_I);
	const int osJ = DPV() * J;
	if (true) {
	  etr += PTP.Rows(osI, osI+DPV()).Cols(osJ,osJ+DPV());
	  if (I != J) {
	    Tetr += Trans(PTP.Rows(osI, osI+DPV()).Cols(osJ,osJ+DPV()));
	  }
	}
	else {
	  Iterate<DPV()>([&](auto i) {
	      Iterate<DPV()>([&](auto j) {
		  auto x = PTP(osI+i, osJ+j);
		  etr(i.value, j.value) += x;
		  if (I != J) {
		    Tetr(j.value, i.value) += x;
		  }
		});
	    });
	}
      }
    }
  }
  

  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> ::
  AddElementMatrix3 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		     ElementId ei, LocalHeap & lh)
  {
    static Timer t("TDNNS_AMG_PC_AddElementMatrix");
    RegionTimer rt(t);
    HeapReset hr(lh);

    int ndof = dnums.Size();
    int ndof_used = 0;
    for (auto d : dnums)
      { if (d != -1) { ndof_used++; } }
    FlatArray<int> dnums_used (ndof_used, lh), dnums_used_loc(ndof_used, lh);
    ndof_used = 0;
    for (auto k : Range(dnums)) {
      auto d = dnums[k];
      if (d != -1) {
	dnums_used[ndof_used] = d;
	dnums_used_loc[ndof_used] = k;
	ndof_used++;
      }
    }

    cout << " dnums: "; prow2(dnums); cout << endl;
    cout << " dnums_used: "; prow2(dnums_used); cout << endl;
    // cout << " elmat: " << endl << elmat << endl;
    
    {
      FlatMatrix<double> used_elmat(ndof_used, ndof_used, lh), evecs(ndof_used, ndof_used, lh);
      FlatVector<double> evals(ndof_used, lh);
      used_elmat = elmat.Rows(dnums_used_loc).Cols(dnums_used_loc);
      LapackEigenValuesSymmetric(used_elmat, evals, evecs);
      cout << " orig USED elmat evals: "; prow(evals); cout << endl;
      // cout << " orig USED elmat evecs: " << endl << evecs << endl;
    }

    auto el_facet_nrs = ma->GetElFacets(ei);
    auto n_el_facets = el_facet_nrs.Size();

    Array<int> loc_lo_dofs(ndof_used); loc_lo_dofs.SetSize0();
    for (auto J : Range(el_facet_nrs)) {
      auto facet_J = el_facet_nrs[J];
      auto facet_J_lodofs = facet_lo_dofs[facet_J];
      for (auto jdof : facet_J_lodofs) {
	int pos = dnums_used.Pos(jdof);
	int pos2 = merge_pos_in_sorted_array(pos, loc_lo_dofs); // a[..pos2-1) <= pos < a[pos2..)
	if (pos2 == 0) 
	  { loc_lo_dofs.Insert(pos2, pos); }
	else if (loc_lo_dofs.Size() > pos2) {
	  if (loc_lo_dofs[pos2-1] != pos)
	    { loc_lo_dofs.Insert(pos2, pos); }
	}
	else
	  { loc_lo_dofs.Append(pos); }
      }
    }
    cout << " loc_lo_dofs: "; prow(loc_lo_dofs); cout << endl;
    FlatArray<int> lo_dofs(loc_lo_dofs.Size(), lh);
    for (auto j : Range(lo_dofs))
      { lo_dofs[j] = dnums_used[loc_lo_dofs[j]]; }
    cout << " lo_dofs: "; prow(lo_dofs); cout << endl;

    int ndof_used_lo = lo_dofs.Size();
    FlatArray<int> dnums_used_loc_lo(lo_dofs.Size(), lh);
    for (auto j : Range(ndof_used_lo))
      { dnums_used_loc_lo[j] = dnums_used_loc[loc_lo_dofs[j]]; }
    
    FlatMatrix<double> P_el (ndof_used_lo, DPV() * n_el_facets, lh); P_el = 0;
    for (auto I : Range(el_facet_nrs)) {
      auto facet_I = el_facet_nrs[I];
      auto facet_I_lodofs = facet_lo_dofs[facet_I];
      auto & facet_mat_I = facet_mat[facet_I];
      cout << "loc facet nr " << I << ", is " << facet_I << endl;
      cout << "facet dnrs :"; prow(facet_I_lodofs); cout << endl;
      cout << "loc   dnrs :";
      for (auto i : Range(facet_I_lodofs)) {
	auto idof = facet_I_lodofs[i];
	auto loc_idof = lo_dofs.Pos(idof);
	cout << loc_idof << " ";
	if (loc_idof == -1)
	  { throw Exception("i feared this might happen (maybe dirichlet/bddc or a combination thereof"); }
	/** strictly speaking, we have to differentiate between facet and edge hc dofs
	    but the hc facet dof rows are 0 anyways. **/
	bool is_hc = (ind_hc < ind_hd) ? (idof < os_hd) : (os_hc <= idof);
	double fac = (is_hc) ? 0.5 : 1;
	P_el.Rows(loc_idof, loc_idof+1).Cols(I*DPV(), (I+1)*DPV()) = fac * facet_mat_I.Row(i);
      }
      cout << endl;
    }
    
    cout << "P_el: " << endl << P_el << endl;

    FlatMatrix<double> PTP (P_el.Width(), P_el.Width(), lh);
    PTP = Trans(P_el) * P_el;

    FlatMatrix<double> evecs (P_el.Width(), P_el.Width(), lh);
    FlatVector<double> evals (P_el.Width(), lh);
    LapackEigenValuesSymmetric(PTP, evals, evecs);
    cout << " EVALS PTP "; prow(evals); cout << endl;

    PTP = Trans(P_el) * elmat.Rows(dnums_used_loc_lo).Cols(dnums_used_loc_lo) * P_el;

    auto nfacets = el_facet_nrs.Size();
    FlatMatrix<TM> facet_elmat(nfacets, nfacets, lh);
    for (auto I : Range(el_facet_nrs)) {
      for (auto J : Range(el_facet_nrs)) {
	facet_elmat(I,J) = PTP.Rows(I*DPV(), (I+1)*DPV()).Cols(J*DPV(),(J+1)*DPV());
      }
    }
    constexpr int N = mat_traits<TM>::HEIGHT;
    FlatArray<TM> sqrt_invs(nfacets, lh);
    for (auto I : Range(el_facet_nrs)) {
      sqrt_invs[I] = facet_elmat(I,I);
      CalcSqrtInv<N>(sqrt_invs[I]);
    }

    for (auto I : Range(el_facet_nrs)) {
      for (auto J : Range(el_facet_nrs)) {
	TM etr = sqrt_invs[I] * facet_elmat(I,J);
	facet_elmat(I,J) = etr * sqrt_invs[J];
      }
    }
    
    typedef typename ElasticityAMGFactory<DIM>::T_V_DATA TVD;
    TVD vi, vj;
    TM Qij, Qji;
    SetIdentity(Qij); SetIdentity(Qji);
    FlatMatrix<TM> facet_elmat2(nfacets, nfacets, lh); facet_elmat2 = 0;
    for (auto I : Range(el_facet_nrs)) {
      for (auto J : Range(I)) {
	double tr = fabsum(facet_elmat(I,J)) / N;
	vi.pos = facet_cos[el_facet_nrs[I]];
	vj.pos = facet_cos[el_facet_nrs[J]];
	ElasticityAMGFactory<DIM>::ModQs(vi, vj, Qij, Qji);
	facet_elmat2(I,I) += tr * Trans(Qij) * Qij;
	facet_elmat2(I,J) -= tr * Trans(Qij) * Qji;
	facet_elmat2(J,I) -= tr * Trans(Qji) * Qij;
	facet_elmat2(J,J) += tr * Trans(Qji) * Qji;
      }
    }

    auto & aux = *GetAuxMat();
    for (auto I : Range(el_facet_nrs)) {
      for (auto J : Range(el_facet_nrs)) {
	aux(el_facet_nrs[I], el_facet_nrs[J]) += facet_elmat2(I,J);
      }
    }
  } // TDNNS_AUX_AMG_Preconditioner::AddElementMatrix


  template<int DIM> void TDNNS_AUX_AMG_Preconditioner<DIM> ::
  AddElementMatrix1 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    /** seems to be broken anyways ... **/
    static Timer t("TDNNS_AMG_PC_AddElementMatrix");
    RegionTimer rt(t);
    HeapReset hr(lh);
    cout << " element " << ei << endl;
    cout << " dnums: "; prow2(dnums); cout << endl;
    cout << " elmat: " << endl << elmat << endl;
    auto el_facet_nrs = ma->GetElFacets(ei);
    cout << " el_facetsL : "; prow2(el_facet_nrs); cout << endl;
    Array<int> loc_lo_dofs(dnums.Size()); loc_lo_dofs.SetSize0();
    for (auto J : Range(el_facet_nrs)) {
      auto facet_J = el_facet_nrs[J];
      auto facet_J_lodofs = facet_lo_dofs[facet_J];
      for (auto jdof : facet_J_lodofs) {
	int pos = dnums.Pos(jdof);
	int pos2 = merge_pos_in_sorted_array(pos, loc_lo_dofs); // a[..pos2-1) <= pos < a[pos2..)
	if (pos2 == 0) //
	  { loc_lo_dofs.Insert(pos2, pos); }
	else if (loc_lo_dofs.Size() > pos2) {
	  if (loc_lo_dofs[pos2-1] != pos)
	    { loc_lo_dofs.Insert(pos2, pos); }
	}
	else
	  { loc_lo_dofs.Append(pos); }
      }
    }
    FlatArray<int> lo_dofs(loc_lo_dofs.Size(), lh);
    for (auto j : Range(lo_dofs))
      { lo_dofs[j] = dnums[loc_lo_dofs[j]]; }
    cout << "loc_lo_dofs: "; prow(loc_lo_dofs); cout << endl;
    cout << "lo_dofs: "; prow(lo_dofs); cout << endl;
    int nd_V = lo_dofs.Size();
    int nd_W = el_facet_nrs.Size();
    FlatMatrix<double> P(nd_V, DPV() * nd_W, lh); P = 0;
    Array<int> loc_jdofs;
    for (auto J : Range(el_facet_nrs)) {
      auto facet_J = el_facet_nrs[J];
      auto facet_J_lodofs = facet_lo_dofs[facet_J];
      loc_jdofs.SetSize(facet_J_lodofs.Size());
      auto & facet_mat_J = facet_mat[facet_J];
      cout << " facet " << J << " " << facet_J << endl;
      cout << " lo_dofs: "; prow(facet_J_lodofs); cout << endl;
      cout << " facet_mat: " << endl << facet_mat_J << endl;
      for (auto i : Range(facet_J_lodofs)) {
	auto idof = facet_J_lodofs[i];
	auto loc_idof = lo_dofs.Pos(idof);
	cout << i << " dof " << idof << " is loc " << loc_idof << endl;
	cout << "facet mat " << endl << facet_mat_J << endl;
	cout << " row " << endl; prow(facet_mat_J.Row(i)); cout << endl;
	bool is_hc = (ind_hc < ind_hd) ? (idof < os_hd) : (os_hc <= idof);
	auto fac = (is_hc) ? 0.5 : 1;
	cout << " p sumbat before: " << endl;
	cout << P.Rows(loc_idof, 1+loc_idof).Cols(J*DPV(), (J+1)*DPV()) << endl;
	cout << endl;
	P.Rows(loc_idof, 1+loc_idof).Cols(J*DPV(), (J+1)*DPV()) += fac * facet_mat_J.Row(i);
	cout << " p sumbat now: " << endl;
	cout << P.Rows(loc_idof, 1+loc_idof).Cols(J*DPV(), (J+1)*DPV()) << endl;
	cout << endl;
      }
    }
    cout << " element-P: " << endl << P << endl;
    FlatMatrix<double> facet_elmat(DPV()*nd_W, DPV()*nd_W, lh); facet_elmat = 0;

    FlatMatrix<double> elmat_P(nd_V, DPV()*nd_W, lh); facet_elmat = 0;
    elmat_P = elmat.Rows(loc_lo_dofs).Cols(loc_lo_dofs) * P;
    facet_elmat = Trans(P) * elmat_P;

    cout << " facet_elmat: " << endl << facet_elmat << endl;

    {
      FlatMatrix<double> evecs(facet_elmat.Height(), facet_elmat.Height(), lh);
      FlatVector<double> evals(facet_elmat.Height(), lh);
      LapackEigenValuesSymmetric(facet_elmat, evals, evecs);
      cout << " facet_elmat evals: "; prow(evals); cout << endl;
      cout << " facet_elmat evecs: " << endl << evecs << endl;
    }

    {
      FlatMatrix<double> loelmat(nd_V, nd_V, lh); loelmat = elmat.Rows(loc_lo_dofs).Cols(loc_lo_dofs);
      FlatMatrix<double> evecs(nd_V, nd_V, lh);
      FlatVector<double> evals(nd_V, lh);
      LapackEigenValuesSymmetric(loelmat, evals, evecs);
      cout << " lo elmat evals: "; prow(evals); cout << endl;
      cout << " lo elmat evecs: " << endl << evecs << endl;
    }

    auto & aux = (*aux_mat);
    for (auto I : Range(el_facet_nrs)) {
      auto facet_I = el_facet_nrs[I];
      const int osI = DPV() * I;
      for (int J = 0; J <= I; J++) {
	auto facet_J = el_facet_nrs[J];
	auto & etr = aux(facet_I, facet_J);
	auto & Tetr = aux(facet_J, facet_I);
	const int osJ = DPV() * J;
	Iterate<DPV()>([&](auto i) {
	    Iterate<DPV()>([&](auto j) {
		auto x = facet_elmat(osI+i, osJ+j);
		etr(i.value, j.value) += x;
		if constexpr(i.value != j.value) {
		    etr(j.value, i.value) += x;
		  }
		if (I != J) {
		  Tetr(j.value, i.value) += x;
		  if constexpr(i.value != j.value) {
		      Tetr(i.value, j.value) += x;
		    }
		}
	      });
	  });
      }
    }
    // aux_pc->AddElementMatrix (facet_elmat, facet_nrs, elid, ...)
  }

  template class TDNNS_AUX_AMG_Preconditioner<2>;
  template class TDNNS_AUX_AMG_Preconditioner<3>;
  

  RegisterPreconditioner<TDNNS_AUX_AMG_Preconditioner<2>> register_tdnnsamg_2d("ngs_amg.tdnns2d");
  RegisterPreconditioner<TDNNS_AUX_AMG_Preconditioner<3>> register_tdnnsamg_3d("ngs_amg.tdnns3d");

} // namespace amg

#include <python_ngstd.hpp>

namespace amg
{
  template<class PCC> void ExportTDNNSAUX (py::module & m, string name, string b)
  {
    auto pyclass = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, name.c_str(), b.c_str());
    pyclass.def(py::init([&](shared_ptr<BilinearForm> bfa, py::kwargs kwargs) {
	  // auto flags = CreateFlagsFromKwArgs(kwargs, h1s_class);
	  auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
	  return make_shared<PCC>(bfa, flags, "noname-pre");
	}), py::arg("bf"));
    pyclass.def_property_readonly("P", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetPMat();
      }, "");
    pyclass.def_property_readonly("aux_mat", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetAuxMat();
      }, "");
    pyclass.def_property_readonly("aux_freedofs", [](shared_ptr<PCC> pre) -> shared_ptr<BitArray> {
	return pre->GetAuxFreeDofs();
      }, "");
    pyclass.def("GetRBModes", [](shared_ptr<PCC> pre) -> py::tuple {
	auto rbms = pre->GetRBModes();
	auto tup = py::tuple(2);
	tup[0] = MakePyList(rbms[0]);
	tup[1] = MakePyList(rbms[1]);
	return tup;
      });
    pyclass.def("GetKerAux", [](shared_ptr<PCC> pre) -> py::tuple {
	const auto & cmat = *pre->GetAuxMat();
	Matrix<double> dense(PCC::DPV() * cmat.Width(), PCC::DPV() * cmat.Width());
	for (auto K : Range(cmat.Width()))
	  for (auto J : Range(cmat.Width()))
	    for (auto k : Range(PCC::DPV()))
	      for (auto j : Range(PCC::DPV()))
		{ dense(K*PCC::DPV() + k, J*PCC::DPV() + j) = cmat(K,J)(k,j); }
	// cout << "AUX dense mat: " << endl << dense << endl;
	Matrix<double> evecs(PCC::DPV() * cmat.Width(), PCC::DPV() * cmat.Width());
	Vector<double> evals(PCC::DPV() * cmat.Width());
	LapackEigenValuesSymmetric(dense, evals, evecs);
	// cout << "AUX evals: "; prow(evals); cout << endl;
	// cout << "AUX evecs: " << endl << evecs << endl;
	int nzero = 0;
	for (auto k : Range(evals))
	  if (fabs(evals(k)) < 1e-10)
	    { nzero++; }
	Array<shared_ptr<BaseVector>> ret(nzero);
	for (auto kv_nr : Range(nzero)) {
	  auto aux_kv = pre->CreateAuxVector();
	  auto f_aux = aux_kv->template FV<typename PCC::TV>();
	  for (auto k : Range(f_aux.Size())) {
	    for (auto j : Range(PCC::DPV())) {
	      f_aux[k](j) = evecs(kv_nr, k*PCC::DPV()+j);
	    }
	  }
	  ret[kv_nr] = aux_kv;
	}
	return MakePyList(ret);
      });
    pyclass.def("GetKerP", [](shared_ptr<PCC> pre) -> py::tuple {
	typename PCC::TPMAT_TM & P = *pre->GetPMat();
	auto PT = TransposeSPM(P);
	auto PT_P = MatMultAB(*PT, P);
	const auto & cmat = *PT_P;
	cout << "PTP: " << endl;
	print_tm_spmat(cout, *PT_P); cout << endl;
	Matrix<double> dense(PCC::DPV() * P.Width(), PCC::DPV() * P.Width());
	for (auto K : Range(P.Width()))
	  for (auto J : Range(P.Width()))
	    for (auto k : Range(PCC::DPV()))
	      for (auto j : Range(PCC::DPV()))
		{ dense(K*PCC::DPV() + k, J*PCC::DPV() + j) = cmat(K,J)(k,j); }
	cout << " dense mat: " << endl << dense << endl;
	Matrix<double> evecs(PCC::DPV() * P.Width(), PCC::DPV() * P.Width());
	Vector<double> evals(PCC::DPV() * P.Width());
	LapackEigenValuesSymmetric(dense, evals, evecs);
	cout << " evals: "; prow(evals); cout << endl;
	cout << " evecs: " << endl << evecs << endl;
	int nzero = 0;
	for (auto k : Range(evals))
	  if (fabs(evals(k)) < 1e-10)
	    { nzero++; }
	auto ret = py::tuple(nzero);
	for (auto kv_nr : Range(nzero)) {
	  auto aux_kv = pre->CreateAuxVector();
	  auto f_aux = aux_kv->template FV<typename PCC::TV>();
	  for (auto k : Range(f_aux.Size())) {
	    for (auto j : Range(PCC::DPV())) {
	      f_aux[k](j) = evecs(kv_nr, k*PCC::DPV()+j);
	    }
	  }
	  Array<shared_ptr<BaseVector>> Pkvs(f_aux.Size());
	  for (auto k : Range(f_aux.Size())) {
	    auto kv = pre->CreateColVector();
	    auto fk = kv.FVDouble(); fk = 0;
	    auto & aux_k = f_aux[k];
	    auto & fmat = pre->facet_mat[k];
	    auto & facet_dofs = pre->facet_lo_dofs[k];
	    Vector<double> v(PCC::DPV()); v = aux_k;
	    Vector<double> Pv(fmat.Height()); Pv = fmat * v;
	    for (auto j : Range(facet_dofs))
	      { fk[facet_dofs[j]] = Pv[j]; }
	    Pkvs[k] = kv;
	  }
	  ret[kv_nr] = MakePyList(Pkvs);
	}
	return ret;
      });
    
  }

  void ExportTDNNSStuff (py::module & m)
  {
    ExportTDNNSAUX<TDNNS_AUX_AMG_Preconditioner<2>>(m, "tdnns2d", "");
    ExportTDNNSAUX<TDNNS_AUX_AMG_Preconditioner<3>>(m, "tdnns3d", "");
  }

} // namespace amg
