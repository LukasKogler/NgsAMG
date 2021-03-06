#ifndef FILE_AMG_FACET_AUX_IMPL_HPP
#define FILE_AMG_FACET_AUX_IMPL_HPP

namespace amg
{

  /** FacetAuxSystem **/

  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: FacetAuxSystem (shared_ptr<BilinearForm> _bfa)
    : bfa(_bfa)
  {
    /** Find SPACEA and SPACEB  in the Compound Space **/
    auto fes = bfa->GetFESpace();
    comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
    ma = comp_fes->GetMeshAccess();
    if (comp_fes == nullptr)
      { throw Exception(string("Need a Compound space, got ") + typeid(*fes).name() + string("!")); }
    ind_sa = ind_sb = -1;
    for (auto space_nr : Range(comp_fes->GetNSpaces())) {
      auto s = (*comp_fes)[space_nr];
      if (auto sa = dynamic_pointer_cast<SPACEA>(s))
	{ spacea = sa; ind_sa = space_nr; }
      else if (auto sb = dynamic_pointer_cast<SPACEB>(s))
	{ spaceb = sb; ind_sb = space_nr; }
    }
    if (ind_sa == -1)
      { throw Exception(string("Could not find space of type ") + typeid(SPACEA).name() + string(" in compound space!")); }
    if (ind_sb == -1)
      { throw Exception(string("Could not find space of type ") + typeid(SPACEB).name() + string(" in compound space!")); }
    auto range_a = comp_fes->GetRange(ind_sa); os_sa = range_a.First();
    auto range_b = comp_fes->GetRange(ind_sb); os_sb = range_b.First();
    

    /** We need dummy ParallelDofs when running without MPI **/
    comp_pds = comp_fes->GetParallelDofs();
    if (comp_pds == nullptr) {
      Array<int> perow (comp_fes->GetNDof()); perow = 0;
      Table<int> dps (perow);
      NgMPI_Comm c(MPI_COMM_WORLD, false);
      Array<int> me({ c.Rank() });
      NgMPI_Comm mecomm = (c.Size() == 1) ? c : c.SubCommunicator(me);
      comp_pds = make_shared<ParallelDofs> ( mecomm , move(dps), 1, false);
    }

  } // FacetAuxSystem(..)


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  shared_ptr<trans_spm_tm<typename FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE>::TPMAT>>
  FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: GetPMatT () const
  {
    shared_ptr<TPMAT_TM> pmtm = pmat;
    auto pmatT_tm = TransposeSPM(*pmtm);
    auto & pmt = const_cast<shared_ptr<trans_spm<TPMAT>>&>(pmatT);
    pmt = make_shared<trans_spm<TPMAT>>(move(*pmatT_tm));
    return pmatT;
  } // FacetAuxSystem::GetPMatT


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Initialize (shared_ptr<BitArray> freedofs)
  {
    /** There are 3 relevant "freedofs" bitarrays:
	 i) comp_fds: free DOFs in compound space
	ii) finest_freedofs/aux_fds: free DOFs in (unsorted) aux-space
       iii) free_nodes : free nodes in (sorted) aux-space, used for coarsening (done by AMG_CLASS based on (ii) when constructing the finest mesh) **/

    if (freedofs != nullptr)
      { comp_fds = freedofs; }
    else
      {
	comp_fds = make_shared<BitArray>(comp_fes->GetNDof());
	comp_fds->Set();
      }

    /** Without BDDC these are the RAW FES freedofs - ignoring elimiante_int/elim_hidden etc!
	 i)  If eliminate_int is turned on, eliminate all CONDENSABLE_DOF
	 ii) if eliminate_hidden, remove HIDDEN_DOFs.
	iii) remove all DOFs not in SPACEA or SPACEB!
	With BDDC, the input are the actual freedofs, but this only clears bits that are not set in the first place. **/
    bool elint = bfa->UsesEliminateHidden(), elhid = bfa->UsesEliminateHidden();
    if ( elint || elhid ) {
      auto & free = *comp_fds;
      for (auto space_ind : Range(comp_fes->GetNSpaces())) {
	if ( (space_ind == ind_sa) || (space_ind == ind_sb) ) {
	  for (auto dof : comp_fes->GetRange(space_ind)) {
	    COUPLING_TYPE ct = comp_fes->GetDofCouplingType(dof);
	    if ( ( ct == UNUSED_DOF) || // definedon or refined mesh
		 ( elint && ((ct & CONDENSABLE_DOF) != 0) ) ||
		 ( elhid && ((ct & HIDDEN_DOF) != 0) ) )
	      { free.Clear(dof); }
	  }
	}
	else {
	  for (auto dof : comp_fes->GetRange(space_ind))
	    { free.Clear(dof); }
	}
      }
    }

    SetUpFacetMats (); // Need DOF-tables for auxiliary space freedofs already need dof-tables here
    
    /** A facet is Dirichlet in the auxiliary space iff:
          - all facet-A/facet-B DOfs are dirichlet
	  - there are no edge-A/edge-B DOFs
	All other cases have to be enforced weakly. **/
    auto n_facets = ma->GetNFacets();
    aux_fds = make_shared<BitArray>(n_facets);
    auto & afd = *aux_fds; afd.Clear();
    bool has_diri = false, diri_consistent = true;
    for (auto facet_nr : Range(n_facets)) {
      bool af_diri = ( (flo_a_f[facet_nr].Size() == 0) || !comp_fds->Test(os_sa + flo_a_f[facet_nr][0])) ? true : false;
      bool bf_diri = ( (flo_b_f[facet_nr].Size() == 0) || !comp_fds->Test(os_sb + flo_b_f[facet_nr][0])) ? true : false;
      if (af_diri != bf_diri)
	{ diri_consistent = false; }
      if (af_diri) {
	afd.SetBit(facet_nr);
	has_diri = true;
      }
    }
    if ( !diri_consistent )
      { throw Exception("Auxiliary Facet Space AMG needs same dirichlet-conditons for both space components!"); }
    if ( (DIM == 3) && has_diri && (has_a_e || has_b_e) )
      { throw Exception("Auxiliary Facet Space AMG with edge-contribs can not handle dirichlet BCs!"); }
    afd.Invert();

    /** Auxiliary space FEM matrix **/
    if (aux_elmats)
      { AllocAuxMat(); }
    /** ParallelDofs int he auxiliary space **/
    SetUpAuxParDofs();
    /** Auxiliary space -> Compound space embedding **/
    BuildPMat();
  } // FacetAuxSystem::Initialize
    

  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Finalize (shared_ptr<BaseMatrix> _comp_mat)
  {
    comp_mat = _comp_mat;

    cout << " aux_elmats " << &aux_elmats << " " << aux_elmats << endl;
    
    if (!aux_elmats) {
      shared_ptr<SparseMatrixTM<double>> cspm = dynamic_pointer_cast<SparseMatrixTM<double>>(comp_mat);
      if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(comp_mat)) {
	cspm = dynamic_pointer_cast<SparseMatrixTM<double>>(parmat->GetMatrix());
      }
      if (cspm == nullptr)
	{ throw Exception("Need SparseMatrix in comp space!!"); }
      shared_ptr<TPMAT_TM> pm = GetPMat();
      shared_ptr<trans_spm_tm<TPMAT_TM>> pmt = GetPMatT();
      auto aux_tm = RestrictMatrixTM<SparseMatrixTM<double>, TPMAT_TM> (*pmt, *cspm, *pm);
      aux_mat = make_shared<TAUX>(move(*aux_tm));
    }
  } // FacetAuxSystem::Finalize


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: AllocAuxMat ()
  {
    auto nfacets = ma->GetNFacets();
    Array<int> elnums;
    auto itit = [&](auto lam) LAMBDA_INLINE {
      for (auto fnr : Range(nfacets)) {
	ma->GetFacetElements(fnr, elnums);
	if (facet_mat[fnr].Height() > 0) { // not fine or unused facet
	  lam(fnr, fnr);
	  for (auto elnr : elnums) {
	    for (auto ofnum : ma->GetElFacets(ElementId(VOL, elnr)))
	      { if (ofnum != fnr) { lam(fnr, ofnum); } }
	  }
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
  } // FacetAuxSystem::AllocAuxMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: SetUpFacetMats ()
  {
    LocalHeap clh (ngcore::task_manager->GetNumThreads() * 20*1024*1024, "SetUpFacetMats");
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
    has_a_e = flo_a_e.AsArray().Size() > 0;
    has_b_e = flo_b_e.AsArray().Size() > 0;
    has_e_ctrbs = has_a_e || has_b_e;
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
    facet_mat_data.SetSize(cnt_buf * DPV);
    facet_mat.SetSize(nfacets); cnt_buf = 0;

    int nff = 0;
    fine_facet = make_shared<BitArray>(nfacets);
    fine_facet->Clear();

    for (auto facet_nr : Range(nfacets)) {
      int c = flo_a_f[facet_nr].Size() + flo_b_f[facet_nr].Size();
      if constexpr(DIM==3) {
	  for (auto fe : ma->GetFaceEdges(facet_nr))
	    { c += flo_a_e[fe].Size() + flo_b_e[fe].Size(); }
	}
      if (c > 0)
	{ fine_facet->SetBit(facet_nr); nff++; }
      facet_mat[facet_nr].AssignMemory(c, DPV, facet_mat_data.Addr(cnt_buf));
      cnt_buf += c * DPV;
    }

    f2a_facet.SetSize(nff); nff = 0;
    a2f_facet.SetSize(nfacets); a2f_facet = -1;
    for (auto k : Range(nfacets))
      if (fine_facet->Test(k))
	{ a2f_facet[k] = nff; f2a_facet[nff++] = k; }

    // cout << "have lo-dof tables " << endl;
    // cout << " A F " << endl << flo_a_f << endl;
    // cout << " A e " << endl << flo_a_e << endl;
    // cout << " B F " << endl << flo_b_f << endl;
    // cout << " B E " << endl << flo_b_e << endl;

    SharedLoop2 sl_facets(nfacets);
    ParallelJob
      ([&] (const TaskInfo & ti)
       {
	 LocalHeap lh = clh.Split(ti.thread_nr, ti.nthreads);
	 Array<int> elnums_of_facet;
    	 for (int facet_nr : sl_facets) {
	   HeapReset hr(lh);
	   ma->GetFacetElements(facet_nr, elnums_of_facet);
	   // if (facet_mat[facet_nr].Height() == 0) // not a fine facet, or not defined on this facet
	   if (!fine_facet->Test(facet_nr))
	     { continue; }
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
  } // FacetAuxSystem::SetUpFacetMats


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: SetUpAuxParDofs ()
  {
    /** Without MPI, we still need dummy ParallelDofs **/
    // if (!bfa->GetFESpace()->IsParallel())
      // { return; }

    auto nfacets = ma->GetNFacets();
    TableCreator<int> ct(nfacets);
    auto comm = comp_pds->GetCommunicator();
    for (; !ct.Done(); ct++) {
      if (comm.Size() > 1) { // cant ask mesh for dist-procs if not parallel
	for (auto facet_nr : Range(nfacets)) {
	  auto dps = ma->GetDistantProcs(NodeId(NT_FACET, facet_nr));
	  for (auto p : dps)
	    { ct.Add(facet_nr, p); }
	}
      }
    }
    aux_pds = make_shared<ParallelDofs> ( comm , ct.MoveTable(), DPV, false);
  } // FacetAuxSystem::SetUpAuxParDofs


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: BuildPMat ()
  {
    if (pmat != nullptr)
     { return; }

    auto H = comp_fes->GetNDof();
    auto nfacets = ma->GetNFacets(); // ...
    auto W = nfacets; // TODO: is this right ?? should I consider free facets etc. ??
    Array<int> perow_f(H), perow_d(H);
    perow_f = 0; perow_d = 0;
    const auto & free_facets = *comp_fds;
    /** get a lambda for evert facet and apply it to all lo dofs for that facet **/
    auto apply_lam = [&](auto lam) {
      for (auto facet_nr : Range(nfacets)) {
	lam(facet_nr, os_sa, flo_a_f[facet_nr]);
	if (DIM == 3)
	  for (auto enr : ma->GetFaceEdges(facet_nr))
	    { lam(facet_nr, os_sa, flo_a_e[enr]); }
	lam(facet_nr, os_sb, flo_b_f[facet_nr]);
	  if (DIM == 3)
	    for (auto enr : ma->GetFaceEdges(facet_nr))
	      { lam(facet_nr, os_sb, flo_b_e[enr]); }
      }
    };
    /** Create graph. Dirichlet-facets can be a problem, especially with edge-contribs.
	Dirichlet-edges can only be are only proled from dirichlet facets. Nasty.
	Even without edge-contribs, facets were only one of A/B are dirichlet and the other
	one is not are nasty. **/
    apply_lam([&](auto fnum, auto os, auto lodofs) {
	const bool free = comp_fds->Test(fnum);
	if (free)
	  for (auto l : Range(lodofs))
	    { perow_f[os + lodofs[l]]++; }
	else
	  for (auto l : Range(lodofs))
	    { perow_d[os + lodofs[l]]++; }
      });
    for (auto dof : Range(H))
      { perow_f[dof] = (perow_d[dof] == 0) ? perow_f[dof] : perow_d[dof]; }
    //  { perow_f[dof] += perow_d[dof]; }
    /** Alloc matrix. **/
    pmat = make_shared<TPMAT> (perow_f, W);
    /** Fill matrix. **/
    perow_f = 0; perow_d = 0; Array<int> locrow(nfacets); locrow = 0;
    apply_lam([&](auto fnum, auto os, auto lodofs) {
	// cout << "SET fnum " << fnum << endl;
	auto fmat = facet_mat[fnum];
	// cout << " ap to " << lodofs.Size() << " dofs: " << flush; prow(lodofs); cout << endl;
	// cout << "locrow is " << locrow[fnum] << endl;
	for (auto k : Range(lodofs)) {
	  auto dof = os + lodofs[k];
	  // cout << " add " << lodofs[k] << " (+os = " << dof << ") x " << fnum << endl;
	  auto ri = pmat->GetRowIndices(dof);
	  auto rv = pmat->GetRowValues(dof);
	  int ind = perow_f[dof]++;
	  ri[ind] = fnum;
	  rv[ind] = fmat.Row(locrow[fnum]++);
	}
      });

    cout << "pmat: " << endl;
    print_tm_spmat(cout, *pmat); cout << endl;

  } // FacetAuxSystem::BuildPMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE> template<ELEMENT_TYPE ET> INLINE
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcFacetMat (ElementId vol_elid, int facet_nr, FlatMatrix<double> fmat, LocalHeap & lh)
  {
    HeapReset hr(lh);

    // cout << " CalcFacetMat " << facet_nr << ", with vol " << vol_elid << endl;
    // cout << "pnums: " << endl << ma->GetFacePNums(facet_nr) << endl;
    

    /** element/facet info, eltrans, geometry stuff **/
    const int ir_order = 3;
    ElementTransformation & eltrans = ma->GetTrafo (vol_elid, lh);
    auto el_facet_nrs = ma->GetElFacets(vol_elid);
    int loc_facet_nr = el_facet_nrs.Pos(facet_nr);
    ELEMENT_TYPE et_vol = ET;
    ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
    Vec<DIM> mid, t; GetNodePos<DIM>(NodeId(FACET_NT(DIM), facet_nr), *ma, mid, t);

    // cout << "mis cos: " << mid << endl;
    
    /** finite elements **/
    auto & fea = static_cast<SPACE_EL<SPACEA, ET>&>(spacea->GetFE(vol_elid, lh)); const auto nda = fea.GetNDof();
    Array<int> adnrs (fea.GetNDof(), lh); spacea->GetDofNrs(vol_elid, adnrs);
    auto & feb = static_cast<SPACE_EL<SPACEB, ET>&>(spaceb->GetFE(vol_elid, lh)); const auto ndb = feb.GetNDof();
    Array<int> bdnrs (feb.GetNDof(), lh); spaceb->GetDofNrs(vol_elid, bdnrs);
    // FacetRBModeFE<DIM> fec (mid); constexpr auto ndc = DPV();
    // AUXFE fec (mid); constexpr auto ndc = DPV();
    AUXFE fec (NodeId(FACET_NT(DIM), facet_nr), *ma); constexpr auto ndc = DPV;

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
    auto laf = flo_a_f[facet_nr]; ca[nfe] = laf.Size(); oa.Last() = oa[nfe] + ca[nfe];
    auto lbf = flo_b_f[facet_nr]; cb[nfe] = lbf.Size(); ob.Last() = ob[nfe] + cb[nfe];
    // cout << "ca : "; prow2(ca); cout << endl;
    // cout << "oa : "; prow2(oa); cout << endl;
    // cout << "cb : "; prow2(cb); cout << endl;
    // cout << "ob : "; prow2(ob); cout << endl;
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

    const int na = oa.Last(), nb = ob.Last(), nc = DPV;

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
	      // mir_vol.ComputeNormalsAndMeasure(..)!!!
	      for (auto ip_nr : Range(mir_vol)) {
		auto mip = mir_vol[ip_nr];
		//TODO: ComputeNormalsAndMeasure!
		fec.CalcMappedShape(mip, shc);
		if (ahas) {
		  CSDS<SPACEA>(fea, mip, sha, dsha);
		  /** edge_dual x edge **/
		  ad_a.Rows(alo, ahi).Cols(alo,ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(sha.Rows(a2v));
		  /** edge_dual x aux **/
		  ad_c.Rows(alo,ahi) += mip.GetWeight() * dsha.Rows(a2v) * Trans(shc);
		}
		if (bhas) {
		  CSDS<SPACEB>(feb, mip, shb, dshb);
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
      // cout << "af -> av "; prow(a2v); cout << endl;
      // cout << "bf -> bv "; prow(b2v); cout << endl;
      for (auto ip_nr : Range(mir_vol)) {
	auto mip = mir_vol[ip_nr];
	CSDS<SPACEA>(fea, mip, sha, dsha);
	CSDS<SPACEB>(feb, mip, shb, dshb);
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
    // cout << " ad_a: " << endl << ad_a << endl;
    CalcInverse(ad_a);
    // cout << " inv ad_a: " << endl << ad_a << endl;
    // cout << " ad_c: " << endl << ad_c << endl;

    // cout << " bd_b: " << endl << bd_b << endl;
    CalcInverse(bd_b);
    // cout << " inv bd_b: " << endl << bd_b << endl;
    // cout << " bd_c: " << endl << bd_c << endl;

    fmat.Rows(0, oa.Last()) = ad_a * ad_c;

    // cout << " fmat with A " << endl << fmat << endl;

    fmat.Rows(oa.Last(), oa.Last()+ob.Last()) = bd_b * bd_c;

    // cout << " final fmat " << endl << fmat << endl;
  } // FacetAuxSystem::CalcFacetMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE> template<ELEMENT_TYPE ET> INLINE
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcElTrafoMat (ElementId vol_elid, FlatMatrix<double> elmat, LocalHeap & lh)
  {
  } // FacetAuxSystem::CalcElTrafoMat


    template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  Array<Vec<FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE>::DPV, double>> FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcFacetFlow ()
  {
    Array<Vec<DPV, double>> flow(f2a_facet.Size());
    LocalHeap lh(10 * 1024 * 1024, "Orrin_Oriscorin_Hiscorin_Sincorin_Alvin_Vladimir_Groolmoplong");
    Array<int> elnums;
    int curve_order = 1; // TODO: get this from mesh
    int ir_order = 1 + DIM * curve_order;
    auto nv_cf = NormalVectorCF(DIM);
    auto comm = aux_pds->GetCommunicator();
    for (auto kf : Range(flow)) {
      HeapReset hr(lh);
      auto facet_nr = f2a_facet[kf];
      AUXFE auxfe (NodeId(FACET_NT(DIM), facet_nr), *ma);
      ma->GetFacetElements(facet_nr, elnums);
      ElementId ei(VOL, elnums[0]);
      auto & trafo = ma->GetTrafo (ei, lh);
      auto facet_nrs = ma->GetElFacets(ei);
      int loc_facet_nr = facet_nrs.Pos(facet_nr);
      ELEMENT_TYPE et_vol = trafo.GetElementType();
      ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
      const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
      Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
      IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
      ElementTransformation & eltrans = ma->GetTrafo (ei, lh);
      BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
      MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL
      mir_vol.ComputeNormalsAndMeasure(et_vol, loc_facet_nr);
      FlatVector<double> facet_flow(DPV, lh); facet_flow = 0;
      FlatMatrix<double> auxval(DPV, DIM, lh);
      FlatVector<double> nvval(DIM, lh);
      double fac = 1.0;
      if (facet_nrs.Size() == 1) { // MPI or BND facet
	auto dps = ma->GetDistantProcs(NodeId(NT_FACET, facet_nr));
	if (dps.Size()) // MPI facet - orient flow from lower to higher rank
	  { fac = (dps[0] < comm.Rank()) ? -1.0 : 1.0; }
      }
      for (auto ip_nr : Range(mir_vol)) {
	auto mip = mir_vol[ip_nr];
	auxfe.CalcMappedShape(mip, auxval);
	nv_cf->Evaluate(mip, nvval);
	for (auto k : Range(DPV))
	  { facet_flow[k] += fac * mip.GetWeight() * InnerProduct(auxval.Row(k), nvval); }
      }
      flow[kf] = facet_flow;
    }
    return flow;
  } // FacetAuxSystem::CalcFlow


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  Table<int> FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcFacetLoops ()
  {
    if constexpr(DIM == 2) {
	return CalcFacetLoops2d();
      }
    else {
      return CalcFacetLoops3d();
    }
  } // FacetAuxSystem::CalcFacetLoops


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  Table<int> FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcFacetLoops3d ()
  {
    return Table<int>();
  } // FacetAuxSystem::CalcFacetLoops3d


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  Table<int> FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CalcFacetLoops2d ()
  {
    auto NV = ma->GetNV();
    Array<int> vels, vsels, edgels, selels;
    Array<int> loop_sizes(NV);
    for (auto vnr : Range(NV)) {
      ma->GetVertexElements(vnr, vels);
      ma->GetVertexSurfaceElements(vnr, vsels);
      loop_sizes[vnr] = vels.Size();
      if (vsels.Size()) {
	// TODO: not true for SELs between two domains, MPI BNDs...
	loop_sizes[vnr]++;
      }
    }
    Table<int> loops(loop_sizes);
    // Vec<2> vpos, epos, tvec;
    for (auto vnr : Range(NV)) {
      // cout << "loop for vertex " << vnr << endl;
      /** Find initial element **/
      int el0 = -1, e0 = -1, orient = -1;
      /** If we are at the mesh boundary, we should start with an element at the boundary,
	  so we can iterate through the loop in one go. **/
      ma->GetVertexSurfaceElements(vnr, vsels);
      if (vsels.Size()) {
	auto selfacets = ma->GetElFacets(ElementId(BND, vsels[0]));
	ma->GetFacetElements(selfacets[0], selels);
	el0 = selels[0];
	e0 = selfacets[0];
	orient = -1;
      }
      else { /** Otherwise, for now, pick an arbitrary element **/
	ma->GetVertexElements(vnr, vels);
	if (el0 == -1)
	  { el0 = vels[0]; }
	/** Now locate the initial edge - it is the one of el0 that contains vnr**/
	auto el0edges = ma->GetElEdges(ElementId(VOL, el0)); const int elos = el0edges.Size();
	for (int j = 0; (j < elos) && (e0 == -1); j++) {
	  auto enr = el0edges[j];
	  auto edverts = ma->GetEdgePNums(enr);
	  // for a "true" surf-facet orientation has to be -1
	  if ( (edverts[0] == vnr) || (edverts[1] == vnr) )
	    { e0 = enr; orient = -1; }
	}
      }
      // cout << " loop len " << loop_sizes[vnr] << endl;
      // cout << " start with el " << el0 << endl;
      // cout << " first edge "  << e0 << endl;
      /** Set loop start **/
      auto & loop = loops[vnr];
      const int loop_len = loop.Size();
      ma->GetFacetElements(e0, edgels);
      loop[0] = (edgels[0] == el0) ? -(1 + e0) : (1 + e0); // first edge should enter first el
      // loop[0] = (ma->GetEdgePNums(e0)[1] == el0) ? (1 + e0) : -(1 + e0); // this way no problem with orientation of facet 0...
      /** Iterate through the loop. **/
      int curr_edg = e0, next_el = el0;
      for (int k : Range(int(1), int(loop_len))) {
	// find next edge: the one that contains the vertex vnr and is not curr_edg!
	auto eledges = ma->GetElEdges(ElementId(VOL, next_el));
	const int neles = eledges.Size();
	int next_edg = -1;
	// cout << " k = " << k << ", el " << next_el << ", edges = "; prow(eledges); cout << endl;
	for (int j = 0; (j < neles) && (next_edg == -1); j++) {
	  auto enr = eledges[j];
	  if (enr != curr_edg) {
	    auto everts = ma->GetEdgePNums(enr);
	    if ( (everts[0] == vnr) || (everts[1] == vnr) ) {
	      // cout << "  new edge " << enr << endl;
	      next_edg = curr_edg = enr;
	      ma->GetFacetElements(enr, edgels);
	      // cout << "  els "; prow(edgels); cout << endl;
	      loop[k] = (edgels[0] == next_el) ? (1 + enr) : -(1 + enr);
	      if (edgels.Size() > 1)
		{ next_el = (edgels[0] == next_el) ? edgels[1] : edgels[0]; }
	      else {
		// cout << " no more els! " << endl;
		next_el = -1;
	      }
	    }
	  }
	}
	// cout << "  okay, next el/edge: " << next_el << ", " << next_edg << endl;
      }
      // cout << "loop: "; prow(loop); cout << endl;
    } // Range(NV)

    // cout << "loops: " << endl << loops << endl;

    return loops;
  } // FacetAuxSystem::CalcFacetLoops2d


  /** Not sure what to do about BBND-elements (only relevant for some cases anyways) **/
  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
								       ElementId ei, LocalHeap & lh)
  {
    if (!aux_elmats)
      { return; }

    ElementTransformation & eltrans = ma->GetTrafo (ei, lh);
    ELEMENT_TYPE et_vol = eltrans.GetElementType();
    if (DIM == 2) {
      switch(et_vol) {
      case(ET_SEGM)  : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_TRIG) : { Add_Vol (dnums, elmat, ei, lh); break; }
      case(ET_QUAD) : { Add_Vol (dnums, elmat, ei, lh); break; }
      default : { throw Exception("FacetAuxVertexAMGPC for El-type not implemented"); break; }
      }
    }
    else {
      switch(et_vol) {
      case(ET_TRIG) : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_QUAD) : { Add_Facet (dnums, elmat, ei, lh); break; }
      case(ET_TET)  : { Add_Vol (dnums, elmat, ei, lh); break; }
      case(ET_HEX)  : { Add_Vol (dnums, elmat, ei, lh); break; }
      default : { throw Exception("FacetAuxVertexAMGPC for El-type not implemented"); break; }
      }
    }
  } // FacetAuxSystem::AddElementMatrix


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Add_Facet (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
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
    FlatMatrix<double> facet_elmat(DPV, DPV, lh);
    facet_elmat = Trans(P) * elmat.Rows(rnums).Cols(rnums) * P;
    auto ri = aux_mat->GetRowIndices(facet_nr);
    auto rv = aux_mat->GetRowValues(facet_nr);
    auto pos = ri.Pos(facet_nr);
    rv[pos] += facet_elmat;
  } // FacetAuxSystem::Add_Facet


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  INLINE void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Add_Vol_simple (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										      ElementId ei, LocalHeap & lh)
  {
    HeapReset hr(lh);

    // auto & O (static_cast<Options&>(*options));

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

    // cout << "add vol simple, ei " << ei << endl;
    // cout << " dnums = "; prow2(dnums); cout << endl;
    // cout << " facet_nrs: "; prow(facet_nrs); cout << endl;

    /** Calc Element-P **/
    BitArray inrange(dnums.Size()); inrange.Clear();
    for (auto i : Range(facet_nrs)) {
      int pos;
      auto facet_nr = facet_nrs[i];
      // cout << " search for a f = "; prow(flo_a_f[facet_nr]); cout << endl;
      // cout << "w. os " << endl;
      // for (auto dof : flo_a_f[facet_nr])
	// cout << os_sa + dof << " ";
      // cout << endl;
      for (auto dof : flo_a_f[facet_nr])
	if ( (pos = dnums.Pos(os_sa + dof)) != -1) // BDDC eliminates dirichlet facets
	  { inrange.SetBit(pos); }
      // cout << " search for b f = "; prow(flo_b_f[facet_nr]); cout << endl;
      // cout << "w. os " << endl;
      // for (auto dof : flo_b_f[facet_nr])
	// cout << os_sb + dof << " ";
      // cout << endl;
      for (auto dof : flo_b_f[facet_nr])
	if ( (pos = dnums.Pos(os_sb + dof)) != -1) // BDDC eliminates dirichlet facets
	  { inrange.SetBit(pos); }

      if constexpr(DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  for (auto dof : flo_a_e[enr])
	    if ( (pos = dnums.Pos(os_sa + dof)) != -1) // BDDC eliminates dirichlet facets
	      { inrange.SetBit(pos); }
	  for (auto dof : flo_b_e[enr])
	    if ( (pos = dnums.Pos(os_sa + dof)) != -1) // BDDC eliminates dirichlet facets
	      { inrange.SetBit(pos); }
	}
      }
    }
    // cout << "inrange: "; prow2(inrange); cout << endl;
    FlatArray<int> rownrs(inrange.NumSet(), lh), used_dnums(inrange.NumSet(), lh);
    int ninrange = 0;
    for (auto k : Range(dnums))
      if (inrange.Test(k))
	{ used_dnums[ninrange] = dnums[k]; rownrs[ninrange++] = k;}
    // cout << " rownrs = "; prow2(rownrs); cout << endl;
    // cout << " used_dums = "; prow2(used_dnums); cout << endl;
    FlatMatrix<double> P(inrange.NumSet(), DPV * loc_n_facets, lh);
    P = 0;
    for (auto i : Range(facet_nrs)) {
      int ilo = i*DPV, ihi = (i+1)*DPV;
      auto facet_nr = facet_nrs[i];
      auto fmat = facet_mat[facet_nr];
      int cfrow = 0;
      auto add_fm = [&](const auto & dnrs, int os, double fac) {
	int c = 0, pos = -1;
	for (auto dof : dnrs)
	  if ( (pos = used_dnums.Pos(os + dof)) != -1)
	    { c++; }
	FlatArray<int> rows(c, lh), fmrows(c, lh);
	c = 0;
	for (auto k : Range(dnrs)) {
	  pos = used_dnums.Pos(os + dnrs[k]);
	  if (pos != -1) {
	    fmrows[c] = cfrow + k;
	    rows[c++] = pos;
	  }
	}
	P.Rows(rows).Cols(ilo,ihi) += fac * fmat.Rows(fmrows);
	cfrow += dnrs.Size();
      };
      if (DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  auto ednrs = flo_a_e[enr];
	  add_fm(ednrs, os_sa, 0.5);
	}
      }
      add_fm(flo_a_f[facet_nr], os_sa, 1.0);
      if (DIM == 3) {
	auto facet_edges = ma->GetFaceEdges(facet_nr);
	for (auto enr : facet_edges) {
	  auto ednrs = flo_b_e[enr];
	  add_fm(ednrs, os_sb, 0.5);
	}
      }
      add_fm(flo_b_f[facet_nr], os_sb, 1.0);
    }

    // cout << "element-P: " << endl << P << endl;

    FlatMatrix<double> elm_P (ninrange, DPV * loc_n_facets, lh);
    FlatMatrix<double> facet_elmat (DPV * loc_n_facets, DPV * loc_n_facets, lh);

    if (elmat_sc) { /** form schur-complement to low order part of elmat **/
      int nng = 0; // nr non garbage
      for (auto k : Range(dnums))
	if (dnums[k] >= 0)
	  { nng++; }
      int nelim = nng - ninrange;
      // cout << " nums " << nng << " " << ninrange << " " << nelim << endl;
      FlatArray<int> elim_rnrs(nelim, lh);
      nelim = 0;
      for (auto k : Range(dnums))
	if ( (dnums[k] > 0) && (!rownrs.Contains(k)) )
	  { elim_rnrs[nelim++] = k; }
      FlatMatrix<double> inner_inv(nelim, nelim, lh);
      inner_inv = elmat.Rows(elim_rnrs).Cols(elim_rnrs);
      CalcInverse(inner_inv);
      FlatMatrix<double> ii_ib(nelim, ninrange, lh);
      ii_ib = inner_inv * elmat.Rows(elim_rnrs).Cols(rownrs);
      FlatMatrix<double> elm_schur(ninrange, ninrange, lh);
      elm_schur = elmat.Rows(rownrs).Cols(rownrs);
      elm_schur -= elmat.Rows(rownrs).Cols(elim_rnrs) * ii_ib;
      elm_P = elm_schur * P;
    }
    else { /** take low order part of elmat **/
      elm_P = elmat.Rows(rownrs).Cols(rownrs) * P;
    }

    facet_elmat = Trans(P) * elm_P;


    auto check_ev = [&](auto &M, string name) {
      auto N = M.Height();
      FlatMatrix<double> evecs(N, N, lh);
      FlatVector<double> evals(N, lh);
      LapackEigenValuesSymmetric(M, evals, evecs);
      // cout << "evals " << name << ": "; prow2(evals); cout << endl;
    };
    
    // check_ev(elmat, "orig elmat"); cout << endl;
    // check_ev(facet_elmat, "facet_elmat"); cout << endl;
    
    for (auto I : Range(facet_nrs)) {
      auto FI = facet_nrs[I];
      auto riI = aux_mat->GetRowIndices(FI);
      auto rvI = aux_mat->GetRowValues(FI);
      int Ilo = DPV * I, Ihi = (1+DPV) * I;
      for (auto J : Range(I)) {
	auto FJ = facet_nrs[J];
	auto riJ = aux_mat->GetRowIndices(FJ);
	auto rvJ = aux_mat->GetRowValues(FJ);
	int Jlo = DPV * J, Jhi = (1+DPV) * J;
	rvI[riI.Pos(FJ)] += facet_elmat.Rows(Ilo, Ihi).Cols(Jlo, Jhi);
	rvJ[riJ.Pos(FI)] += facet_elmat.Rows(Jlo, Jhi).Cols(Ilo, Ihi);
      }
      rvI[riI.Pos(FI)] += facet_elmat.Rows(Ilo, Ihi).Cols(Ilo, Ihi);
    }

  } // FacetAuxSystem::Add_Vol_simple


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  INLINE void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Add_Vol_rkP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										   ElementId ei, LocalHeap & lh)
  {
    /** Define Element-P by facet-matrices and aux_elmat = el_P.T * elmat * el_P.
	Then, regularize ker(el_P*el_P.T) **/
    throw Exception("Add_Vol_rkP todo");
  } // FacetAuxSystem::Add_Vol_rkP


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  INLINE void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: Add_Vol_elP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										   ElementId ei, LocalHeap & lh)
  {
    /** Asseble an extra element-P and define aux_elmat = el_P * elmat * el_P.T. **/
    throw Exception("Add_Vol_elP todo");
  } // FacetAuxSystem::Add_Vol_rkP


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  Array<Array<AutoVector>> FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: GetRBModes () const
  {
    Array<Array<AutoVector>> rb_modes(2);
    auto bfa_mat = bfa->GetMatrixPtr();
    if (bfa_mat == nullptr)
      { throw Exception("mat not ready"); }
    if (pmat == nullptr)
      { throw Exception("pmat not ready"); }
    const auto & P = *pmat;
    /** displacements: (1,0,0), (0,1,0), (0,0,1) **/
    for (auto comp : Range(DIM)) {
      auto w = CreateAuxVector();
      w.SetParallelStatus(CUMULATED);
      auto fw = w.template FV<TV>();
      auto v = bfa_mat->CreateRowVector();
      v.SetParallelStatus(CUMULATED);
      auto fv = v.template FV<double>();
      for (auto k : Range(fw.Size()))
	{ fw(k) = 0; fw(k)(comp) = 1; }
      P.Mult(*w, *v);
      rb_modes[0].Append(move(v));
      rb_modes[1].Append(move(w));
    }
    /** rotations **/
    if constexpr(DIM == 3)
      {
	/** (1,0,0) \cross x = (0, -z, y)
	    (0,1,0) \cross x = (z, 0, -x)
	    (0,0,1) \cross x = (-y, x, 0) **/
	Vec<3> mid, r, cross, temp;
	for (auto rnr : Range(3)) {
	  r = 0; r(rnr) = 1;
	  auto w = CreateAuxVector();
	  w.SetParallelStatus(CUMULATED);
	  auto fw = w.template FV<TV>();
	  auto v = bfa_mat->CreateRowVector();
	  v.SetParallelStatus(CUMULATED);
	  auto fv = v.template FV<double>();
	  for (auto k : Range(fw.Size())) {
	    GetNodePos<DIM>(NodeId(FACET_NT(DIM), k), *ma, mid, temp);
	    fw(k) = 0;
	    cross = Cross(r, mid);
	    for (auto l : Range(DIM))
	      { fw(k)(l) = cross(l); }
	    fw(k)(DIM + rnr) = 1;
	  }
	  P.Mult(*w, *v);
	  rb_modes[0].Append(move(v));
	  rb_modes[1].Append(move(w));
	}
      }
    else
      {
	/** (0,0,1) \cross x = (y, -x, 0) **/
	throw Exception("2d GetRBModes not implemented");
      }
    return rb_modes;
  } // FacetAuxSystem::GetRBModes


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  AutoVector FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: CreateAuxVector () const
  {
    if (aux_pds != nullptr)
      { return make_unique<ParallelVVector<TV>> (pmat->Width(), aux_pds, DISTRIBUTED); }
    else
      { return make_unique<VVector<TV>> (pmat->Width()); }
  } // FacetAuxSystem::CreateAuxVector


  template<int DIM, class SPACEA, class SPACEB, class AUXFE>
  void FacetAuxSystem<DIM, SPACEA, SPACEB, AUXFE> :: __hacky__set__Pmat ( shared_ptr<BaseMatrix> embA, shared_ptr<BaseMatrix> embB )
  {
    /** merge mats (!! project out dirichlet dofs !!) **/
    auto mA = dynamic_pointer_cast<TPMAT_TM>(embA);
    auto mB = dynamic_pointer_cast<TPMAT_TM>(embB);
    if ( (mA == nullptr) || (mB == nullptr) )
      { throw Exception(" invalid mats!"); }
    Array<int> perow(comp_fes->GetNDof()); perow = 0;
    auto comp_free = comp_fes->GetFreeDofs(true); // ... oh well
    for (auto k : Range(mA->Height())) {
      if (comp_free->Test(os_sa + k))
	{ perow[os_sa + k] = mA->GetRowIndices(k).Size(); }
    }
    for (auto k : Range(mB->Height())) {
      if (comp_free->Test(os_sb + k))
	{ perow[os_sb + k] = mB->GetRowIndices(k).Size(); }
    }
    auto newP = make_shared<TPMAT>(perow, mA->Width());
    for (auto k : Range(mA->Height())) {
      int row = os_sa + k;
      if (comp_free->Test(row)) {
	auto ri = newP->GetRowIndices(row); auto ri2 = mA->GetRowIndices(k);
	auto rv = newP->GetRowValues(row); auto rv2 = mA->GetRowValues(k);
	ri = ri2;
	rv = rv2;
      }
    }
    for (auto k : Range(mB->Height())) {
      int row = os_sb + k;
      if (comp_free->Test(row)) {
	auto ri = newP->GetRowIndices(row); auto ri2 = mB->GetRowIndices(k);
	auto rv = newP->GetRowValues(row); auto rv2 = mB->GetRowValues(k);
	ri = ri2;
	rv = rv2;
      }
    }
    // cout << " new P mat: " << endl;
    // print_tm_spmat(cout, *newP); cout << endl;
    pmat = newP;
  } // FacetAuxSystem::__hacky__set__Pmat


  /** END FacetAuxSystem **/


  /** AuxiliarySpacePreconditioner **/

  template<class AUX_SYS, class BASE>
  AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: AuxiliarySpacePreconditioner (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : BASE(bfa, aflags, name)
  {
    /** Initialize Auxiliary System, set up transfer operators **/
    aux_sys = make_shared<AUX_SYS>(bfa);
  } // AuxiliarySpacePreconditioner(..)


  template<class AUX_SYS, class BASE>
  const BaseMatrix & AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: GetAMatrix () const
  {
    if ( (aux_sys == nullptr) || (aux_sys->GetCompMat() == nullptr) )
      { throw Exception("comp_mat not ready"); }
    return *aux_sys->GetCompMat();
  } // AuxiliarySpacePreconditioner::GetAMatrix


  template<class AUX_SYS, class BASE>
  const BaseMatrix & AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: GetMatrix () const
  {
    return *emb_amg_mat;
  } // AuxiliarySpacePreconditioner::GetMatrix


  template<class AUX_SYS, class BASE>
  shared_ptr<BaseMatrix> AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: GetMatrixPtr ()
  {
    return emb_amg_mat;
  } // AuxiliarySpacePreconditioner::GetMatrixPtr


  template<class AUX_SYS, class BASE>
  shared_ptr<EmbeddedAMGMatrix> AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: GetEmbAMGMat () const
  {
    return emb_amg_mat;
  } // AuxiliarySpacePreconditioner::GetEmbAMGMat


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: Mult (const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().Mult(b, x);
  } // AuxiliarySpacePreconditioner::Mult


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: MultTrans (const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultTrans(b, x);
  } // AuxiliarySpacePreconditioner::Mult


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultAdd(s, b, x);
  } // AuxiliarySpacePreconditioner::MultAdd


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultTransAdd(s, b, x);
  } // AuxiliarySpacePreconditioner::MultTransAdd


  template<class AUX_SYS, class BASE>
  AutoVector AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: CreateColVector () const
  {
    if (auto m = bfa->GetMatrixPtr())
      { return m->CreateColVector(); }
    else
      { throw Exception("BFA mat not ready!"); }
  } // AuxiliarySpacePreconditioner::CreateColVector


  template<class AUX_SYS, class BASE>
  AutoVector AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: CreateRowVector () const
  {
    if (auto m = bfa->GetMatrixPtr())
      { return m->CreateRowVector(); }
    else
      { throw Exception("BFA mat not ready!"); }
  } // AuxiliarySpacePreconditioner::CreateRowVector


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    const auto & O(static_cast<Options&>(*options));

    aux_sys->SetAuxElmats(O.aux_elmats);

    /** Initialize auxiliary system:
	  - filter freedofs
	  - set up auxiliary space freedofs
	  - set up auxiliary space ParallelDofs
	  - set up convert operator
	  - allocate auxiliary space matrix **/
    aux_sys->Initialize(freedofs);

    finest_freedofs = aux_sys->GetAuxFreeDofs();
  } // AuxiliarySpacePreconditioner::InitLevel


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: FinalizeLevel (const BaseMatrix * mat)
  {
    aux_sys->Finalize(shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter));

    // cout << "aux mat: " << endl;
    // print_tm_spmat(cout, *aux_sys->GetAuxMat()); cout << endl;

    // BASE::FinalizeLevel(mat);

    finest_mat = make_shared<ParallelMatrix> (aux_sys->GetAuxMat(), aux_sys->GetAuxParDofs(), aux_sys->GetAuxParDofs(), PARALLEL_OP::C2D);

    if (options->sync) {
      if (auto pds = finest_mat->GetParallelDofs()) {
	static Timer t(string("Sync1")); RegionTimer rt(t);
	pds->GetCommunicator().Barrier();
      }
    }

    /** Set dummy ParallelDofs **/
    if (mat->GetParallelDofs() == nullptr)
      { const_cast<BaseMatrix*>(mat)->SetParallelDofs(aux_sys->GetCompParDofs()); }

    factory = this->BuildFactory();

    this->BuildAMGMat();
  } // AuxiliarySpacePreconditioner::FinalizeLevel


  template<class AUX_SYS, class BASE>
  shared_ptr<BaseSmoother> AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: BuildFLS () const
  {
    const auto & O(static_cast<Options&>(*options));

    shared_ptr<BaseSmoother> sm = nullptr;

    cout << " comp_sm           = " << O.comp_sm << endl;
    cout << " comp_sm_blocks    = " << O.comp_sm_blocks  << endl;
    cout << " comp_sm_blocks_el = " << O.comp_sm_blocks_el  << endl;


    if (O.comp_sm) {
      if (O.comp_sm_blocks)
	{ sm = BuildFLS_EF(); }
      else {
	auto comp_mat = aux_sys->GetCompMat();
	shared_ptr<BaseSparseMatrix> comp_spmat;
	if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(comp_mat))
	  { comp_spmat = dynamic_pointer_cast<SparseMatrix<double>>(parmat->GetMatrix()); }
	else
	  { comp_spmat = dynamic_pointer_cast<SparseMatrix<double>>(comp_mat); }
	auto comp_pds = aux_sys->GetCompParDofs();
	auto comp_fds = aux_sys->GetCompFreeDofs();
	auto eqc_h = make_shared<EQCHierarchy>(comp_pds, false);
	sm = const_cast<AuxiliarySpacePreconditioner<AUX_SYS, BASE>*>(this)->BuildGSSmoother(comp_spmat, comp_pds, eqc_h, comp_fds);
      }
    }

    return sm;
  }


  template<class AUX_SYS, class BASE>
  shared_ptr<BaseSmoother> AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: BuildFLS_EF () const
  {
    const auto & O(static_cast<Options&>(*options));

    // cout << "build finest level block smoother " << endl;

    auto comp_space = aux_sys->GetCompSpace();
    auto spacea = aux_sys->GetSpaceA();
    auto spaceb = aux_sys->GetSpaceB();
    int os_sa = aux_sys->GetOsA(), os_sb = aux_sys->GetOsB();
    auto comp_pds = aux_sys->GetCompParDofs();
    auto comp_mat = aux_sys->GetCompMat();

    /** Element-Blocks **/
    auto free1 = aux_sys->GetCompFreeDofs(); // if BDDC this is the correct one
    auto free2 = comp_space->GetFreeDofs(bfa->UsesEliminateInternal()); // if el_int, this is the one
    auto is_free = [&](auto x) { return free1->Test(x) && free2->Test(x); };
    size_t n_blocks = O.comp_sm_blocks_el ? ma->GetNE() : ma->GetNFacets();
    TableCreator<int> cblocks(n_blocks);
    Array<int> dnums;
    auto add_dofs = [&](auto os, auto block_num, auto & dnums) LAMBDA_INLINE {
      for (auto loc_dof : dnums) {
	auto real_dof = os + loc_dof;
	// cout << "[" << loc_dof << " " << real_dof << " " << free->Test(real_dof) << "] ";
	if (is_free(real_dof))
	  { cblocks.Add(block_num, real_dof); }
      }
    };
    for (; !cblocks.Done(); cblocks++) {
      if (O.comp_sm_blocks_el) {
	for (auto el_nr : Range(n_blocks)) {
	  spacea->GetDofNrs(ElementId(VOL, el_nr), dnums);
	  add_dofs(os_sa, el_nr, dnums);
	  spaceb->GetDofNrs(ElementId(VOL, el_nr), dnums);
	  add_dofs(os_sb, el_nr, dnums);
	}
      }
      else {
	const FESpace &fesa(*spacea), &fesb(*spaceb);
	for (auto facet_nr : Range(n_blocks)) {
	  fesa.GetDofNrs(NodeId(NT_FACET, facet_nr), dnums);
	  add_dofs(os_sa, facet_nr, dnums);
	  fesb.GetDofNrs(NodeId(NT_FACET, facet_nr), dnums);
	  add_dofs(os_sb, facet_nr, dnums);
	}
      }
    }
    // auto comp_mat = bfa->GetMatrixPtr();
    auto eqc_h = make_shared<EQCHierarchy>(comp_pds, false); // TODO: get rid of these!

    auto sm = make_shared<HybridBS<double>> (comp_mat, eqc_h, cblocks.MoveTable(), O.sm_mpi_overlap, O.sm_mpi_thread, O.sm_shm, O.sm_sl2);

    return sm;
  } // AuxiliarySpacePreconditioner::BuildFLS


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: BuildAMGMat ()
  {
    /** If aux_only is false:
	   If smooth_aux_lo is true:
	     Block-Smooth for HO, then smooth all AMG-levels
	   else [ TODO!! ]
	     Block-Smooth for HO, then smooth all AMG-levels except first
	If aux_only:
	   Smooth all AMG-levels
     **/

    auto & O (static_cast<Options&>(*options));

    BASE::BuildAMGMat();

    if (__hacky_test) {
      /** Aux space AMG as a preconditioner for Auxiliary matrix **/
      auto i1 = printmessage_importance;
      auto i2 = netgen::printmessage_importance;
      printmessage_importance = 1;
      netgen::printmessage_importance = 1;
      cout << IM(1) << "Test AMG in auxiliary space " << endl;
      // EigenSystem eigen(*aux_mat, *amg_mat);
      EigenSystem eigen(*finest_mat, *amg_mat); // need parallel mat
      eigen.SetPrecision(1e-12);
      eigen.SetMaxSteps(1000); 
      eigen.Calc();
      cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
      cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      printmessage_importance = i1;
      netgen::printmessage_importance = i2;
    }

    /** Create the full preconditioner by adding a finest level smoother as a first step (or not, if aux_only) **/
    shared_ptr<BaseSmoother> fls = BuildFLS();
    /** okay, this is hacky - when we do not use a ProlMap for AssembleMatrix, we have to manually
	invert prol, and then convert it from SPM_TM to SPM **/

    auto pmat = aux_sys->GetPMat();
    auto comp_pds = aux_sys->GetCompParDofs();
    auto aux_pds = aux_sys->GetAuxParDofs();
    // cout << " fls : " << endl << fls << endl;
    auto pmatT = aux_sys->GetPMatT();
    auto ds = make_shared<ProlMap<typename AUX_SYS::TPMAT_TM>>(pmat, pmatT, comp_pds, aux_pds);
    emb_amg_mat = make_shared<EmbeddedAMGMatrix> (fls, amg_mat, ds);
    emb_amg_mat->SetSTK_outer(O.comp_sm_steps);

    if (__hacky_test) {
      /** Aux space AMG + finest level smoother as a preconditioner for the bilinear-form **/
      auto i1 = printmessage_importance;
      auto i2 = netgen::printmessage_importance;
      printmessage_importance = 1;
      netgen::printmessage_importance = 1;
      cout << IM(1) << "Test full Auxiliary space Preconditioner " << endl;
      this->Test();
      printmessage_importance = i1;
      netgen::printmessage_importance = i2;
    }
    
  } // AuxiliarySpacePreconditioner::BuildAMGMat


  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: Update ()
  {
    ;
  } // AuxiliarySpacePreconditioner::Update

  template<class AUX_SYS, class BASE>
  void AuxiliarySpacePreconditioner<AUX_SYS, BASE> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									ElementId ei, LocalHeap & lh)
  {
    aux_sys->AddElementMatrix(dnums, elmat, ei, lh);
  } // FacetAuxVertexAMGPC::AddElementMatrix


  /** END AuxiliarySpacePreconditioner **/


  /** FacetAuxVertexAMGPC **/

  template<int DIM, class AUX_SYS, class AMG_CLASS>
  FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: FacetAuxVertexAMGPC (shared_ptr<BilinearForm> bfa, const Flags & flags, const string name)
    :  BASE(bfa, flags, name)
  {
    options = this->MakeOptionsFromFlags(flags);
  } // FacetAuxVertexAMGPC(..)


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  void FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    BASE::InitLevel(freedofs);
  } // FacetAuxVertexAMGPC::InitLevel


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  void FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: SetUpMaps ()
  {
    static Timer t("SetUpMaps"); RegionTimer rt(t);
    auto & O (static_cast<Options&>(*options));
    auto n_facets = ma->GetNFacets();
    auto n_verts = n_facets;

    /** aux_mat is finest mat. [Gets re-sorted during TopMesh Setup]  **/
    use_v2d_tab = false;
    O.v_nodes.SetSize(n_facets);
    v2d_array.SetSize(n_facets);
    d2v_array.SetSize(n_facets);
    for (auto k : Range(n_facets)) {
      v2d_array[k] = d2v_array[k] = k;
      O.v_nodes[k] = NodeId(FACET_NT(DIM), k);
    }

  } // FacetAuxVertexAMGPC::SetUpMaps


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  shared_ptr<BaseAMGPC::Options> FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: NewOpts ()
  {
    return make_shared<Options>();
  } // FacetAuxVertexAMGPC::NewOpts


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  void FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: SetDefaultOptions (BaseAMGPC::Options & aO)
  {
    auto & O (static_cast<Options&>(aO));

    O.sm_shm = !bfa->GetFESpace()->IsParallel();

#ifdef ELASCTICITY
    if constexpr (is_same<AUXFE, FacetRBModeFE<DIM>>::value) {
	O.with_rots = true;
      }
#endif
    O.subset = AMG_CLASS::Options::DOF_SUBSET::RANGE_SUBSET;
    O.ss_ranges.SetSize(1);
    O.ss_ranges[0] = { 0, ma->GetNFacets() };
    O.dof_ordering = AMG_CLASS::Options::DOF_ORDERING::REGULAR_ORDERING;
    O.block_s.SetSize(1); O.block_s[0] = 1;
    O.topo = AMG_CLASS::Options::TOPO::ALG_TOPO;
    O.energy = AMG_CLASS::Options::ENERGY::ALG_ENERGY;

    O.store_v_nodes = false; // I do this manually

    /** Coarsening Algorithm **/
    O.crs_alg = AMG_CLASS::Options::CRS_ALG::AGG;
    O.ecw_geom = false;
    O.ecw_robust = false; // probably irrelevant as we are usually using this for MCS
    O.d2_agg = SpecOpt<bool>(false, { true });
    O.disc_max_bs = 1;

    /** Level-control **/
    // O.first_aaf = 1/pow(3, D);
    O.first_aaf = (DIM == 3) ? 0.025 : 0.05;
    O.aaf = 1/pow(2, DIM);

    /** Redistribute **/
    O.enable_redist = true;
    O.rdaf = 0.05;
    O.first_rdaf = O.aaf * O.first_aaf;
    O.rdaf_scale = 1;
    O.rd_crs_thresh = 0.7;
    O.rd_min_nv_gl = 5000;
    O.rd_seq_nv = 5000;
    
    /** Smoothed Prolongation **/
    O.enable_sp = true;
    O.sp_min_frac = (DIM == 3) ? 0.08 : 0.15;
    O.sp_omega = 1;
    O.sp_max_per_row = 1 + DIM;

    /** Discard **/
    O.enable_disc = false; // not sure, should probably work
    O.disc_max_bs = 1;

  } // FacetAuxVertexAMGPC::SetDefaultOptions


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  void FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: ModifyOptions (typename BaseAMGPC::Options & aO, const Flags & flags, string prefix)
  {
    auto & O (static_cast<Options&>(aO));

    BASE::__hacky_test = O.do_test;
    O.do_test = false;
  } // FacetAuxVertexAMGPC::ModifyOptions


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  shared_ptr<BaseDOFMapStep> FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: BuildEmbedding (BaseAMGFactory::AMGLevel & finest_level)
  {
    auto mesh = finest_level.cap->mesh;

    /** aux_mat is finest mat **/
    // shared_ptr<TPMAT_TM> emb_mat;
    // if ( aux_pds->GetCommunicator().Size() > 2) {
    //   auto & vsort = node_sort[NT_VERTEX];
    //   auto perm = BuildPermutationMatrix<typename TAUX_TM::TENTRY>(vsort);
    //   emb_mat = MatMultAB(*pmat, *perm);
    // }
    // else
    //   { emb_mat = pmat; }
    // auto step = make_shared<ProlMap<TPMAT_TM>>(emb_mat, comp_pds, aux_pds);
    // return step;

    /** comp_mat is finest mat **/
    auto comp_pds = aux_sys->GetCompParDofs();
    auto aux_pds = aux_sys->GetAuxParDofs();
    if (comp_pds->GetCommunicator().Size() > 2) {
      auto & vsort = node_sort[NT_VERTEX];
      auto perm = BuildPermutationMatrix<typename TAUX_TM::TENTRY>(vsort);
      auto step = make_shared<ProlMap<SparseMatrixTM<typename TAUX_TM::TENTRY>>>(perm, aux_pds, aux_pds);
      return step;
    }

    return nullptr;
  } // FacetAuxVertexAMGPC::BuildEmbedding


  template<int DIM, class AUX_SYS, class AMG_CLASS>
  void FacetAuxVertexAMGPC<DIM, AUX_SYS, AMG_CLASS> :: Update ()
  {
    ;
  } // FacetAuxVertexAMGPC::Update


  /** END FacetAuxVertexAMGPC **/


} // namespace amg


#include <python_ngstd.hpp>


namespace amg
{
  template <typename T>
  py::list MakePyList2 (FlatArray<T> ao)
  {
    size_t s = ao.Size();
    py::list l;
    for (size_t i = 0; i < s; i++)
      l.append (move(ao[i]));
    // for (size_t i = 0; i < s; i++)
      // l.append (move(ao.Data()[i]));
    return l;
  }

  template<class PCC, class TLAM> void ExportAuxiliaryAMG (py::module & m, string name, string descr, TLAM lam)
  {
    auto pyclass = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, name.c_str(), descr.c_str());

    pyclass.def(py::init([&](shared_ptr<BilinearForm> bfa, string name, py::kwargs kwa) {
	  auto flags = CreateFlagsFromKwArgs(kwa, py::none());
	  flags.SetFlag("__delay__opts", true);
	  return make_shared<PCC>(bfa, flags, name);
	}), py::arg("bf"), py::arg("name") = "SolvyMcAMGFace");
    pyclass.def_property_readonly("P", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetAuxSys()->GetPMat();
      }, "");
    pyclass.def_property_readonly("aux_mat", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetAuxSys()->GetAuxMat();
      }, "");
    pyclass.def_property_readonly("aux_freedofs", [](shared_ptr<PCC> pre) -> shared_ptr<BitArray> {
	return pre->GetAuxSys()->GetAuxFreeDofs();
      }, "");
    pyclass.def("CreateAuxVector", [](shared_ptr<PCC> pre) -> shared_ptr<BaseVector> {
	return pre->CreateAuxVector();
      });
    pyclass.def("GetRBModes", [](shared_ptr<PCC> pre) -> py::tuple {
	auto rbms = pre->GetAuxSys()->GetRBModes();
	auto tup = py::tuple(2);
	tup[0] = MakePyList2(rbms[0]);
	tup[1] = MakePyList2(rbms[1]);
	return tup;
      });
    pyclass.def("GetNLevels", [](PCC &pre, size_t rank) {
	return pre.GetEmbAMGMat()->GetNLevels(rank);
      }, py::arg("rank")=int(0));
    pyclass.def("GetNDof", [](PCC &pre, size_t level, size_t rank) {
	return pre.GetEmbAMGMat()->GetNDof(level, rank);
      }, py::arg("level"), py::arg("rank")=int(0));
    pyclass.def("GetBF", [](PCC &pre, shared_ptr<BaseVector> vec,
			    size_t level, size_t rank, size_t dof) {
		  pre.GetEmbAMGMat()->GetBF(level, rank, dof, *vec);
		}, py::arg("vec")=nullptr, py::arg("level")=size_t(0),
		py::arg("rank")=size_t(0), py::arg("dof")=size_t(0));
    pyclass.def("CINV", [](PCC &pre, shared_ptr<BaseVector> csol,
			   shared_ptr<BaseVector> rhs) {
		  pre.GetEmbAMGMat()->CINV(csol, rhs);
		}, py::arg("sol")=nullptr, py::arg("rhs")=nullptr);
    
    lam(pyclass);
  } // ExportAuxiliary

} // namespace amg

#endif
