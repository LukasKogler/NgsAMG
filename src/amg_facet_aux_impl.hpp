#ifndef FILE_AMG_FACET_AUX_IMPL_HPP
#define FILE_AMG_FACET_AUX_IMPL_HPP

namespace amg
{


  template<int DIM, class SPACEA, class SPACEB>
  const BaseMatrix & FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: GetAMatrix () const
  {
    return blf->GetMatrix();
  } // FacetWiseAuxiliarySpaceAMG::GetAMatrix


  template<int DIM, class SPACEA, class SPACEB>
  const BaseMatrix & FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: GetMatrix () const
  {
    return *this;
  } // FacetWiseAuxiliarySpaceAMG::GetMatrix


  template<int DIM, class SPACEA, class SPACEB>
  shared_ptr<BaseMatrix> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: GetMatrixPtr ()
  {
    return nullptr;
  } // FacetWiseAuxiliarySpaceAMG::GetMatrixPtr


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Mult (const BaseVector & b, BaseVector & x) const
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::Mult


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: MultTrans (const BaseVector & b, BaseVector & x) const
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::Mult


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
     ;
  } // FacetWiseAuxiliarySpaceAMG::MultAdd


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::MultTransAdd


  template<int DIM, class SPACEA, class SPACEB>
  AutoVector FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: CreateColVector () const
  {
    return shared_ptr<BaseVector>(nullptr);
  } // FacetWiseAuxiliarySpaceAMG::CreateColVector


  template<int DIM, class SPACEA, class SPACEB>
  AutoVector FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: CreateRowVector () const
  {
    return shared_ptr<BaseVector>(nullptr);
  } // FacetWiseAuxiliarySpaceAMG::CreateRowVector


  template<int DIM, class SPACEA, class SPACEB>
  shared_ptr<BaseVector> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: CreateAuxVector () const
  {
    return nullptr;
  } // FacetWiseAuxiliarySpaceAMG::CreateRowVector


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    AllocAuxMat ();
    SetUpFacetMats ();
    SetUpAuxParDofs();
    BuildPMat ();
  } // FacetWiseAuxiliarySpaceAMG::InitLevel


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: FinalizeLevel (const BaseMatrix * mat)
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::FinalizeLevel


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Update ()
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::Update


  template<int DIM, class SPACEA, class SPACEB>
  FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: FacetWiseAuxiliarySpaceAMG (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : Preconditioner (bfa, aflags, name)
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG(..)


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: AllocAuxMat ()
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
    aux_mat = make_shared<TAUX>(perow, nfacets);
    aux_mat->AsVector() = 0;
    perow = 0;
    itit([&](auto i, auto j) LAMBDA_INLINE {
	aux_mat->GetRowIndices(i)[perow[i]++] = j;
      });
    for (auto k : Range(nfacets))
      { QuickSort(aux_mat->GetRowIndices(k)); }
    auto auxrow = aux_mat->CreateRowVector();
  } // FacetWiseAuxiliarySpaceAMG::AllocAuxMat


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: SetUpFacetMats ()
  {
    LocalHeap clh (20*1024*1024, "SetUpFacetMats");
    size_t nfacets = ma->GetNFacets(), nedges = ma->GetNEdges();
    size_t cnt_buf;
    Array<int> dnums;
    /** Find low-order DOfs for each facet/edge **/
    TableCreator<int> cae(nedges), cbe(nedges);
    for ( ; !cae.Done(); cae++, cbe++) {
      if ( (DIM == 3) && (has_e_ctrbs) ) {
	for (auto enr : Range(nedges))
	  { ItLO_A(NodeId(NT_EDGE, enr), dnums, [&](auto i) LAMBDA_INLINE { cae.Add(enr, dnums[i]); }); }
	for (auto enr : Range(nedges))
	  { ItLO_B(NodeId(NT_EDGE, enr), dnums, [&](auto i) LAMBDA_INLINE { cbe.Add(enr, dnums[i]); }); }
      }
    }
    flo_a_e = cae.MoveTable();
    flo_b_e = cbe.MoveTable();
    TableCreator<int> caf(nfacets), cbf(nfacets);
    for ( ; !caf.Done(); caf++, cbf++) {
      cnt_buf = 0;
      for (auto facet_nr : Range(nfacets)) {
	if constexpr(DIM==3) {
	    for (auto fe : ma->GetFaceEdges(facet_nr))
	      { cnt_buf += flo_a_e[fe].Size() + flo_b_e[fe].Size(); }
	  }
	int c = 0;
	ItLO_A(NodeId(FACET_NT(DIM), facet_nr), dnums, [&](auto i) LAMBDA_INLINE { caf.Add(facet_nr, dnums[i]); c++; });
	cnt_buf += c; c = 0;
	ItLO_B(NodeId(FACET_NT(DIM), facet_nr), dnums, [&](auto i) LAMBDA_INLINE { cbf.Add(facet_nr, dnums[i]); c++; });
	cnt_buf += c;
      }
    }
    flo_a_f = caf.MoveTable();
    flo_b_f = cbf.MoveTable();
    /** Alloc facet matrices **/
    facet_mat_data.SetSize(cnt_buf * DPV());
    facet_mat.SetSize(nfacets); cnt_buf = 0;
    for (auto facet_nr : Range(nfacets)) {
      int c = flo_a_f[facet_nr].Size();
      if constexpr(DIM==3) {
	  for (auto fe : ma->GetFaceEdges(facet_nr))
	    { c += flo_a_e[fe].Size() + flo_b_e[fe].Size(); }
	}
      facet_mat[facet_nr].AssignMemory(c, DPV(), facet_mat_data.Addr(cnt_buf));
      cnt_buf += c * DPV();
    }
    SharedLoop2 sl_facets(nfacets);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
	 Array<int> elnums_of_facet;
    	 for (int facet_nr : sl_facets) {
	   HeapReset hr(lh);
	   ma->GetFacetElements(facet_nr, elnums_of_facet);
	   ElementId vol_elid(VOL, elnums_of_facet[0]);
	   ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
	   ELEMENT_TYPE et_vol = eltrans.GetElementType();
	   if constexpr( DIM == 2 ) {
	     switch(et_vol) {
	     case(ET_TRIG) : { CalcFacetMat<ET_TRIG> (vol_elid, facet_nr, facet_mat[facet_nr], lh); break; }
	     case(ET_QUAD) : { CalcFacetMat<ET_QUAD> (vol_elid, facet_nr, facet_mat[facet_nr], lh); break; }
	     }
	   }
	   else {
	     switch(et_vol) {
	     case(ET_TET) : { CalcFacetMat<ET_TET> (vol_elid, facet_nr, facet_mat[facet_nr], lh); break; }
	     case(ET_HEX) : { CalcFacetMat<ET_HEX> (vol_elid, facet_nr, facet_mat[facet_nr], lh); break; }
	     }
	   }
	 }
       });
  } // FacetWiseAuxiliarySpaceAMG::SetUpFacetMats


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: SetUpAuxParDofs ()
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::SetUpAuxParDofs


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: BuildPMat ()
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::BuildPMat


  template<int DIM, class SPACEA, class SPACEB> template<ELEMENT_TYPE ET> INLINE
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: CalcFacetMat (ElementId vol_elid, int facet_nr, FlatMatrix<double> fmat, LocalHeap & lh)
  {
    HeapReset hr(lh);

    /** element/facet info, eltrans, geometry stuff **/
    const int ir_order = 3;
    ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
    auto el_facet_nrs = ma->GetElFacets(vol_elid);
    int loc_facet_nr = el_facet_nrs.Pos(facet_nr);
    ELEMENT_TYPE et_vol = ET;
    ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
    Vec<DIM> mid, t; GetNodePos<DIM>(NodeId(FACET_NT(DIM), facet_nr), *ma, mid, t);

    /** finite elements **/
    auto & fea = static_cast<SPACE_EL<SPACEA, ET>&>(spacea->GetFE(vol_elid, lh)); const auto nda = fea.GetNDof();
    Array<int> adnrs (fea.GetNDof(), lh); spacea->GetDofNrs(vol_elid, adnrs);
    auto & feb = static_cast<SPACE_EL<SPACEB, ET>&>(spaceb->GetFE(vol_elid, lh)); const auto ndb = feb.GetNDof();
    Array<int> bdnrs (feb.GetNDof(), lh); spaceb->GetDofNrs(vol_elid, bdnrs);
    FacetRBModeFE<DIM> fec (mid); constexpr auto ndc = DPV();

    /** buffer for shapes **/
    FlatMatrix<double> sha(nda, DIM, lh), shb(ndb, DIM, lh), shc(ndc, DIM, lh), /** shapes **/
      dsha(nda, DIM, lh), dshb(ndb, DIM, lh); /** dual shapes **/

    /** Book-keeping **/
    int ff = (DIM==3) ? facet_nr : 0; // in 2D, pick 0 so we do not call some garbage here
    auto facet_edges = ma->GetFaceEdges(ff); const int nfe = facet_edges.Size();
    /** counts/offsets for facet-mat rows. ordering is [a_e1, a_e2.., a_f, b_e, b_f] **/
    FlatArray<int> ca(1 + nfe, lh), oa(2 + nfe, lh), cb(1 + nfe, lh), ob(2 + nfe, lh);
    oa[0] = ob[0] = 0;
    for (auto k : Range(facet_edges)) {
      auto fe = facet_edges[k];
      ca[k] = flo_a_e[fe].Size(); oa[1+k] = oa[k] + ca[k];
      cb[k] = flo_b_e[fe].Size(); ob[1+k] = ob[k] + cb[k];
    }
    auto laf = flo_a_f[facet_nr]; ca[nfe] = laf.Size();
    auto lbf = flo_b_f[facet_nr]; cb[nfe] = lbf.Size();
    /** facet-row ordering to vol-fel numbering **/
    FlatArray<int> a2vol(oa.Last(), lh), b2vol(ob.Last(), lh);
    for (auto k : Range(facet_edges)) {
      auto fe = facet_edges[k];
      auto lae = flo_a_e[fe];
      for (auto j : Range(ca[k]))
	{ a2vol[oa[k] + j] = adnrs.Pos(lae[j]); }
      auto lbe = flo_b_e[fe];
      for (auto j : Range(cb[k]))
	{ b2vol[ob[k] + j] = bdnrs.Pos(lbe[j]); }
    }
    for (auto j : Range(ca.Last()))
      { a2vol[oa[nfe] + j] = adnrs.Pos(laf[j]); }
    for (auto j : Range(cb.Last()))
      { b2vol[ob[nfe] + j] = bdnrs.Pos(lbf[j]); }

    const int na = oa.Last(), nb = ob.Last(), nc = DPV();

    FlatMatrix<double> ad_a(na, na, lh), ad_c(na, nc, lh), bd_b(nb, nb, lh), bd_c(nb, nc, lh);
    ad_a = 0; ad_c = 0; bd_b = 0; bd_c = 0;
    
    { /** Edge contributions (3d only, only potentially) **/
      if (DIM == 3) {
	if (has_e_ctrbs) {
	  const IntegrationRule & ir_seg = SelectIntegrationRule (ET_SEGM, ir_order); // reference segment
	  Facet2ElementTrafo seg_2_vol(et_vol, BBND);
	  auto vol_el_edges = ma->GetElEdges(vol_elid);
	  for (auto loc_enr : Range(nfe)) {
	    bool ahas = ca[loc_enr] > 0, bhas = cb[loc_enr] > 0;
	    if (ahas || bhas) {
	      auto a2v = a2vol.Part(oa[loc_enr], ca[loc_enr]);
	      auto b2v = b2vol.Part(ob[loc_enr], cb[loc_enr]);
	      int alo = oa[loc_enr], ahi = oa[1+loc_enr], blo = ob[loc_enr], bhi = ob[1+loc_enr];
	      auto volel_enr = vol_el_edges.Pos(facet_edges[loc_enr]);
	      IntegrationRule & ir_vol = seg_2_vol(volel_enr, ir_seg, lh); // reference VOL
	      auto & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(eltrans(ir_vol, lh))); // mapped VOL
	      for (auto ip_nr : Range(mir_vol)) {
		auto mip = mir_vol[ip_nr];
		fec.CalcMappedShape(mip, shc);
		if (ahas) {
		  // fea.CalcMappedShape(mip, sha);
		  // fea.CalcDualShape(mip, dsha);
		  CSDS_A(fea, mip, sha, dsha);
		  /** edge_dual x edge **/
		  ad_a.Rows(alo, ahi).Cols(alo,ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(sha.Rows(a2v));
		  /** edge_dual x aux **/
		  ad_c.Rows(alo,ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(shc);
		}
		if (bhas) {
		  // feb.CalcMappedShape(mip,shb);
		  // feb.CalcDualShape(mip,dshb);
		  CSDS_B(feb, mip, shb, dshb);
		  /** edge_dual x edge **/
		  bd_b.Rows(blo, bhi).Cols(blo, bhi) += mip.GetWeight() * dshb.Rows(b2v) * Trans(shb.Rows(b2v));
		  /** edge_dual x aux **/
		  bd_c.Rows(blo, bhi) += mip.GetWeight() * dshb.Rows(b2v) * Trans(shc);
		}
	      }
	    }
	  }
	}
      }
    } /** Edge contributions **/


    { /** Facet contributions **/
      const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
      Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
      IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
      BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
      MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
      mir_vol.ComputeNormalsAndMeasure(et_vol, loc_facet_nr); // ?? I THINK ??
      auto afds = flo_a_f[facet_nr];
      auto a2v = a2vol.Part(oa[nfe], ca[nfe]);
      auto b2v = b2vol.Part(ob[nfe], cb[nfe]);
      int alo = oa[nfe], ahi = oa[1+nfe], blo = ob[nfe], bhi = ob[1+nfe];
      for (auto ip_nr : Range(mir_vol)) {
	auto mip = mir_vol[ip_nr];
	// fea.CalcMappedShape(mip, sha);
	// fea.CalcDualShape(mip, dsha);
	CSDS_A(fea, mip, sha, dsha);
	// feb.CalcMappedShape(mip, shb);
	// feb.CalcDualShape(mip, dshb);
	CSDS_B(feb, mip, shb, dshb);
	fec.CalcMappedShape(mip, shc);
	/** facet_dual x facet **/
	ad_a.Rows(alo, ahi).Cols(alo, ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(sha.Rows(a2v));
	bd_b.Rows(blo, bhi).Cols(blo, bhi) += mip.GetWeight() * dshb.Rows(b2v) * Trans(shb.Rows(b2v));
	/** facet_dual x aux **/
	ad_c.Rows(alo, ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(shc);
	bd_c.Rows(blo, bhi) += mip.GetWeight() * dshb.Rows(b2v) * Trans(shc);
	/** facet_dual x edge **/
	for (auto k : Range(nfe)) {
	  if (ca[k] > 0) {
	    ad_a.Rows(alo, ahi).Cols(oa[k], oa[1+k]) += mip.GetWeight() * dsha.Rows(a2v) * Trans(sha.Rows(a2vol.Part(oa[k], ca[k])));
	  }
	  if (cb[k] > 0) {
	    bd_b.Rows(blo, bhi).Cols(ob[k], ob[1+k]) += mip.GetWeight() * dshb.Rows(b2v) * Trans(shb.Rows(b2vol.Part(ob[k], cb[k])));
	  }
	}
      }
    } /** Facet contributions **/
    CalcInverse(ad_a);
    fmat.Rows(0, oa.Last()) = ad_a * ad_c;
    CalcInverse(bd_b);
    fmat.Rows(oa.Last(), oa.Last()+ob.Last()) = bd_b * bd_c;
  } // FacetWiseAuxiliarySpaceAMG::CalcFacetMat


  /** Not sure what to do about BBND-elements (only relevant for some cases anyways) **/
  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									    ElementId ei, LocalHeap & lh)
  {
    ElementTransformation & eltrans = ma->GetTrafo (ei, lh);
    ELEMENT_TYPE et_vol = eltrans.GetElementType();
    if (DIM == 2) {
      switch(et_vol) {
      case(ET_SEGM)  : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_TRIG) : { Add_Vol (dnums, elmat, ei, lh); break; }
      case(ET_QUAD) : { Add_Vol (dnums, elmat, ei, lh); break; }
      default : { throw Exception("FacetWiseAuxiliarySpaceAMG for El-type not implemented"); break; }
      }
    }
    else {
      switch(et_vol) {
      case(ET_TRIG) : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_QUAD) : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_TET)  : { Add_Vol (dnums, elmat, ei, lh); break; }
      case(ET_HEX)  : { Add_Vol (dnums, elmat, ei, lh); break; }
      default : { throw Exception("FacetWiseAuxiliarySpaceAMG for El-type not implemented"); break; }
      }
    }
  } // FacetWiseAuxiliarySpaceAMG::AddElementMatrix


  template<int DIM, class SPACEA, class SPACEB>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Add_Facet (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
								     ElementId ei, LocalHeap & lh)
  {
    /** This should be straightforward **/
    auto facet_nr = ei.Nr();
    auto & P = facet_mat[facet_nr];
    FlatArray<int> rnums(P.Height(), lh);
    int cr = 0;
    int ff = (DIM==3) ? facet_nr : 0;
    auto facet_edges = ma->GetFaceEdges(ff);
    for (auto k : Range(facet_edges)) // a edge
      for (auto dof : flo_a_e[facet_edges[k]])
	{ rnums[cr++] = dnums.Pos(dof); }
    for (auto dof : flo_a_f[facet_nr]) // a facet
      { rnums[cr++] = dnums.Pos(dof); }
    for (auto k : Range(facet_edges)) // b edge
      for (auto dof : flo_b_e[facet_edges[k]])
	{ rnums[cr++] = dnums.Pos(dof); }
    for (auto dof : flo_b_f[facet_nr]) // b facet
      { rnums[cr++] = dnums.Pos(dof); }
    FlatMatrix<double> facet_elmat(DPV(), DPV(), lh);
    facet_elmat = Trans(P) * elmat.Rows(rnums).Cols(rnums) * P;
    auto ri = aux_mat->GetRowIndices(facet_nr);
    auto rv = aux_mat->GetRowValues(facet_nr);
    auto pos = ri.Pos(facet_nr);
    rv[pos] += facet_elmat;
  } // FacetWiseAuxiliarySpaceAMG::Add_Facet


  template<int DIM, class SPACEA, class SPACEB>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Add_Vol_simple (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										 ElementId ei, LocalHeap & lh)
  {
    HeapReset hr(lh);
    /** Define Element-P by facet-matrices and aux_elmat = el_P.T * elmat * el_P **/
    ElementTransformation & eltrans = ma->GetTrafo (ei, lh);
    ELEMENT_TYPE et_vol = eltrans.GetElementType();
    auto facet_nrs = ma->GetElFacets(ei); int loc_n_facets = facet_nrs.Size();
    auto vol_el_edges = ma->GetElEdges(ei);
    // Array<int> n_facet_edges(loc_n_facets); n_facet_edges = 0;
    // if constexpr (DIM == 3) {
    //   for (auto i : Range(facet_nrs)) {
    // 	auto facet_nr = facet_nrs[i];
    // 	auto facet_edges = ma->GetFaceEdges(facet_nr);
    // 	n_facet_edges[i] = facet_edges.Size();
    //   }
    // }

    /** Calc Element-P **/
    BitArray inrange(dnums.Size()); inrange.Clear();
    for (auto i : Range(facet_nrs)) {
      auto facet_nr = facet_nrs[i];
      for (auto dof : flo_a_f[facet_nr])
	{ inrange.SetBit(dnums.Pos(dof)); }
      for (auto dof : flo_b_f[facet_nr])
	{ inrange.SetBit(dnums.Pos(dof)); }
      if constexpr(DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  for (auto dof : flo_a_e[enr])
	    { inrange.SetBit(dnums.Pos(dof)); }
	  for (auto dof : flo_b_e[enr])
	    { inrange.SetBit(dnums.Pos(dof)); }
	}
      }
    }
    FlatArray<int> rownrs(inrange.NumSet(), lh), used_dnums(inrange.NumSet(), lh);
    int ninrange = 0;
    for (auto k : Range(dnums))
      if (inrange.Test(k))
	{ used_dnums[ninrange] = dnums[k]; rownrs[ninrange++] = k;}
    FlatMatrix<double> P(inrange.NumSet(), DPV() * loc_n_facets, lh);
    P = 0;
    for (auto i : Range(facet_nrs)) {
      int ilo = i*DPV(), ihi = (i+1)*DPV();
      auto facet_nr = facet_nrs[i];
      auto fmat = facet_mat[facet_nr];
      int cfrow = 0;
      auto add_fm = [&](const auto & dnrs) {
	FlatArray<int> rows(dnrs.Size(), lh);
	int c = 0;
	for (auto dof : dnrs)
	  { rows[c++] = used_dnums.Pos(dof); }
	P.Rows(rows).Cols(ilo,ihi) += fmat.Rows(cfrow, cfrow+c);
	cfrow += c;
      };
      if (DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  auto ednrs = flo_a_e[enr];
	  add_fm(ednrs);
	}
      }
      add_fm(flo_a_f[facet_nr]);
      if (DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  auto ednrs = flo_b_e[enr];
	  add_fm(ednrs);
	}
      }
      add_fm(flo_b_f[facet_nr]);
    }

    FlatMatrix<double> elm_P (ninrange, DPV() * loc_n_facets, lh);
    FlatMatrix<double> facet_elmat (DPV() * loc_n_facets, DPV() * loc_n_facets, lh);
    elm_P = elmat.Rows(rownrs).Cols(rownrs) * P;
    facet_elmat = Trans(P) * elm_P;

    for (auto I : Range(facet_nrs)) {
      auto FI = facet_nrs[I];
      auto riI = aux_mat->GetRowIndices(FI);
      auto rvI = aux_mat->GetRowValues(FI);
      int Ilo = DPV() * I, Ihi = (1+DPV()) * I;
      for (auto J : Range(I)) {
	auto FJ = facet_nrs[J];
	auto riJ = aux_mat->GetRowIndices(FJ);
	auto rvJ = aux_mat->GetRowValues(FJ);
	int Jlo = DPV() * J, Jhi = (1+DPV()) * J;
	rvI[riI.Pos(FJ)] += facet_elmat.Rows(Ilo, Ihi).Cols(Jlo, Jhi);
	rvJ[riJ.Pos(FI)] += facet_elmat.Rows(Jlo, Jhi).Cols(Ilo, Ihi);
      }
      rvI[riI.Pos(FI)] += facet_elmat.Rows(Ilo, Ihi).Cols(Ilo, Ihi);
    }

  } // FacetWiseAuxiliarySpaceAMG::Add_Vol_simple


  template<int DIM, class SPACEA, class SPACEB>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Add_Vol_rkP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									      ElementId ei, LocalHeap & lh)
  {
    /** Define Element-P by facet-matrices and aux_elmat = el_P.T * elmat * el_P.
	Then, regularize ker(el_P*el_P.T) **/
    throw Exception("Add_Vol_rkP todo");
  } // FacetWiseAuxiliarySpaceAMG::Add_Vol_rkP


  template<int DIM, class SPACEA, class SPACEB>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB> :: Add_Vol_elP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									      ElementId ei, LocalHeap & lh)
  {
    /** Asseble an extra element-P and define aux_elmat = el_P * elmat * el_P.T. **/
    throw Exception("Add_Vol_elP todo");
  } // FacetWiseAuxiliarySpaceAMG::Add_Vol_rkP
} // namespace amg

#endif
