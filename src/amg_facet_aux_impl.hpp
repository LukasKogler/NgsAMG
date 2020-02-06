#ifndef FILE_AMG_FACET_AUX_IMPL_HPP
#define FILE_AMG_FACET_AUX_IMPL_HPP

namespace amg
{



  /** Options **/

  void BaseFacetAMGOptions :: SetFromFlags (const Flags & flags, string prefix)
  {
    aux_only = flags.GetDefineFlagX("ngs_amg_aux_only").IsTrue();
    elmat_sc = !flags.GetDefineFlagX("ngs_amg_elmat_sc").IsFalse();
    el_blocks = !flags.GetDefineFlagX("ngs_amg_el_blocks").IsFalse();
    f_blocks_sc = flags.GetDefineFlagX("ngs_amg_fsc").IsTrue();
  } // BaseFacetAMGOptions :: SetFromFlags


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  class FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Options
    : public AMG_CLASS::Options,
      public BaseFacetAMGOptions
  {
  public:
    Options () { ; }

    virtual void SetFromFlags (shared_ptr<FESpace> fes, const Flags & flags, string prefix) override
    {
      AMG_CLASS::Options::SetFromFlags(fes, flags, prefix);
      BaseFacetAMGOptions::SetFromFlags(flags, prefix);
    }
  }; // FacetWiseAuxiliarySpaceAMG::Options

  /** END Options **/


  /** FacetWiseAuxiliarySpaceAMG **/

  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> ::
  FacetWiseAuxiliarySpaceAMG (shared_ptr<BilinearForm> bfa, const Flags & flags, const string name)
    :  AMG_CLASS(bfa, flags, name)
  {
    options = this->MakeOptionsFromFlags(flags);

    /** Find SPACEA and SPACEB  in the Compound Space **/
    auto fes = bfa->GetFESpace();
    comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes);
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
      NgsMPI_Comm c(MPI_COMM_WORLD);
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : AMG_ME_COMM;
      comp_pds = make_shared<ParallelDofs> ( mecomm , move(dps), 1, false);
    }
  } // FacetWiseAuxiliarySpaceAMG(..)


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  const BaseMatrix & FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: GetAMatrix () const
  {
    if (comp_mat == nullptr)
      { throw Exception("comp_mat not ready"); }
    return *comp_mat;
  } // FacetWiseAuxiliarySpaceAMG::GetAMatrix


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  const BaseMatrix & FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: GetMatrix () const
  {
    return *emb_amg_mat;
  } // FacetWiseAuxiliarySpaceAMG::GetMatrix


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  shared_ptr<BaseMatrix> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: GetMatrixPtr ()
  {
    return emb_amg_mat;
  } // FacetWiseAuxiliarySpaceAMG::GetMatrixPtr


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  shared_ptr<EmbeddedAMGMatrix> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: GetEmbAMGMat () const
  {
    return emb_amg_mat;
  } // FacetWiseAuxiliarySpaceAMG::GetEmbAMGMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Mult (const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().Mult(b, x);
  } // FacetWiseAuxiliarySpaceAMG::Mult


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: MultTrans (const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultTrans(b, x);
  } // FacetWiseAuxiliarySpaceAMG::Mult


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultAdd(s, b, x);
  } // FacetWiseAuxiliarySpaceAMG::MultAdd


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    GetMatrix().MultTransAdd(s, b, x);
  } // FacetWiseAuxiliarySpaceAMG::MultTransAdd


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  AutoVector FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: CreateColVector () const
  {
    if (auto m = bfa->GetMatrixPtr())
      { return m->CreateColVector(); }
    else
      { throw Exception("BFA mat not ready!"); }
  } // FacetWiseAuxiliarySpaceAMG::CreateColVector


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  AutoVector FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: CreateRowVector () const
  {
    if (auto m = bfa->GetMatrixPtr())
      { return m->CreateRowVector(); }
    else
      { throw Exception("BFA mat not ready!"); }
  } // FacetWiseAuxiliarySpaceAMG::CreateRowVector


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  shared_ptr<BaseVector> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: CreateAuxVector () const
  {
    if (aux_pardofs != nullptr)
      { return make_shared<ParallelVVector<TV>> (pmat->Width(), aux_pardofs, DISTRIBUTED); }
    else
      { return make_shared<VVector<TV>> (pmat->Width()); }
  } // FacetWiseAuxiliarySpaceAMG::CreateRowVector


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    /** There are 3 relevant "freedofs" bitarrays:
	 i) comp_fds: free DOFs in compound space
	ii) finest_freedofs: free DOFs in (unsorted) aux-space
       iii) free_verts : free verts in (sorted) aux-space, used for coarsening (done by AMG_CLASS) **/
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
	    if ( ( elint && ((ct & CONDENSABLE_DOF) != 0) ) ||
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
    finest_freedofs = make_shared<BitArray>(n_facets);
    auto & afd = *finest_freedofs; afd.Clear();
    bool has_diri = false, diri_consistent = true;
    for (auto facet_nr : Range(n_facets)) {
      bool af_diri = (flo_a_f[facet_nr].Size() && !comp_fds->Test(os_sa + flo_a_f[facet_nr][0])) ? true : false;
      bool bf_diri = (flo_b_f[facet_nr].Size() && !comp_fds->Test(os_sb + flo_b_f[facet_nr][0])) ? true : false;
      if (af_diri != bf_diri)
	{ diri_consistent = false; }
      if (af_diri) {
	afd.SetBit(facet_nr);
	has_diri = true;
      }
    }
    if ( !diri_consistent)
      { throw Exception("Auxiliary Facet Space AMG needs same dirichlet-conditons for both space components!"); }
    if ( (DIM == 3) && has_diri && (has_a_e || has_b_e) )
      { throw Exception("Auxiliary Facet Space AMG with edge-contribs can not handle dirichlet BCs!"); }
    afd.Invert();

    /** Auxiliary space FEM matrix **/
    AllocAuxMat();
    /** ParallelDofs int he auxiliary space **/
    SetUpAuxParDofs();
    /** Auxiliary space -> Compound space embedding **/
    BuildPMat();
  } // FacetWiseAuxiliarySpaceAMG::InitLevel


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: FinalizeLevel (const BaseMatrix * mat)
  {
    // auto aux_tm  = RestrictMatrixTM<SparseMatrixTM<double>, TPMAT_TM> (dynamic_cast<SparseMatrixTM<double>&>(const_cast<BaseMatrix&>(*mat)), *pmat);
    // aux_mat = make_shared<TAUX>(move(*aux_tm));

    comp_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter);

    finest_mat = make_shared<ParallelMatrix> (aux_mat, aux_pardofs, aux_pardofs, PARALLEL_OP::C2D);

    if (options->sync) {
      if (auto pds = finest_mat->GetParallelDofs()) {
	static Timer t(string("Sync1")); RegionTimer rt(t);
	pds->GetCommunicator().Barrier();
      }
    }

    /** Set dummy ParallelDofs **/
    if (mat->GetParallelDofs() == nullptr)
      { const_cast<BaseMatrix*>(mat)->SetParallelDofs(comp_pds); }

    factory = this->BuildFactory();

    this->BuildAMGMat();
  } // FacetWiseAuxiliarySpaceAMG::FinalizeLevel


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: BuildAMGMat ()
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

    AMG_CLASS::BuildAMGMat();

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
    shared_ptr<BaseSmoother> fls = O.aux_only ? nullptr : ( O.f_blocks_sc ? BuildFLS2() : BuildFLS() );
    /** okay, this is hacky - when we do not use a ProlMap for AssembleMatrix, we have to manually
	invert prol, and then convert it from SPM_TM to SPM **/

    // cout << " fls : " << endl << fls << endl;
    shared_ptr<trans_spm_tm<TPMAT_TM>> pmatT = TransposeSPM( ((TPMAT_TM&)(*pmat)) );
    pmatT = make_shared<trans_spm<TPMAT>>(move(*pmatT));
    auto ds = make_shared<ProlMap<TPMAT_TM>>(pmat, pmatT, comp_pds, aux_pardofs);
    emb_amg_mat = make_shared<EmbeddedAMGMatrix> (fls, amg_mat, ds);

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
    
  } // FacetWiseAuxiliarySpaceAMG::BuildAMGMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: SetUpMaps ()
  {
    static Timer t("SetUpMaps"); RegionTimer rt(t);
    auto & O (static_cast<Options&>(*options));
    auto n_facets = ma->GetNFacets();
    auto n_verts = n_facets;

    /** comp_mat is finest mat **/
    // use_v2d_tab = true;
    // TableCreator<int> cv2d(n_verts);
    // for (; !cv2d.Done(); cv2d++) {
    //   for (auto k : Range(n_facets)) {
    // 	if (DIM == 3) {
    // 	  for (auto enr : ma->GetFaceEdges(k))
    // 	    { cv2d.Add(k, flo_a_e[enr]); }
    // 	}
    // 	cv2d.Add(k, flo_b_f[k]);
    // 	if (DIM == 3) {
    // 	  for (auto enr : ma->GetFaceEdges(k))
    // 	    { cv2d.Add(k, flo_b_e[enr]); }
    // 	}
    // 	cv2d.Add(k, flo_b_f[k]);
    //   }
    // }
    // v2d_table = cv2d.MoveTable();
    // O.v_nodes.SetSize(n_facets);
    // d2v_array.SetSize(comp_fes->GetNDof()); d2v_array = -1;
    // for (auto k : Range(n_facets)) {
    //   for (auto dof : v2d_table[k])
    // 	{ d2v_array[dof] = k; }
    //   O.v_nodes[k] = NodeId(FACET_NT(DIM), k);
    // }

    /** aux_mat is finest mat. [Gets re-sorted during TopMesh Setup]  **/
    use_v2d_tab = false;
    O.v_nodes.SetSize(n_facets);
    v2d_array.SetSize(n_facets);
    d2v_array.SetSize(n_facets);
    for (auto k : Range(n_facets)) {
      v2d_array[k] = d2v_array[k] = k;
      O.v_nodes[k] = NodeId(FACET_NT(DIM), k);
    }

  } // FacetWiseAuxiliarySpaceAMG::SetUpMaps


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS> shared_ptr<BaseSmoother>
  FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: BuildFLS () const
  {
    const auto &O (static_cast<Options&>(*options));

    // cout << "build finest level block smoother " << endl;

    /** Element-Blocks **/
    auto& free1 = comp_fds; // if BDDC this is the correct one
    auto free2 = comp_fes->GetFreeDofs(bfa->UsesEliminateInternal()); // if el_int, this is the one
    auto is_free = [&](auto x) { return free1->Test(x) && free2->Test(x); };
    size_t n_blocks = O.el_blocks ? ma->GetNE() : ma->GetNFacets();
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
      if (O.el_blocks) {
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
	  fesa.GetDofNrs(NodeId(FACET_NT(DIM), facet_nr), dnums);
	  add_dofs(os_sa, facet_nr, dnums);
	  fesb.GetDofNrs(NodeId(FACET_NT(DIM), facet_nr), dnums);
	  add_dofs(os_sb, facet_nr, dnums);
	}
      }
    }
    // auto comp_mat = bfa->GetMatrixPtr();
    auto eqc_h = make_shared<EQCHierarchy>(comp_pds, false); // TODO: get rid of these!

    auto sm = make_shared<HybridBS<double>> (comp_mat, eqc_h, cblocks.MoveTable(), O.sm_mpi_overlap, O.sm_mpi_thread);

    return sm;
  } // FacetWiseAuxiliarySpaceAMG::BuildFLS


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS> shared_ptr<BaseSmoother>
  FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: BuildFLS2 () const
  {
    const auto &O(*this->options);

    auto& free1 = comp_fds; // if BDDC this is the correct one
    auto free2 = comp_fes->GetFreeDofs(bfa->UsesEliminateInternal()); // if el_int, this is the one
    auto is_free = [&](auto x) { return free1->Test(x) && free2->Test(x); };

    size_t n_blocks = ma->GetNFacets();
    TableCreator<int> cblocks(n_blocks), cbex(n_blocks);
    Array<int> elnums;
    Array<int> dnums_f, dnums_el, dnums_ex;
    auto add_facet = [&](auto os, const FESpace & fes, int facet_nr) LAMBDA_INLINE {
      auto add_dofs = [&](auto & tc, auto os, auto block_num, auto & dnums) LAMBDA_INLINE {
	bool set_any = false;
	for (auto loc_dof : dnums) {
	  auto real_dof = os + loc_dof;
	  if (is_free(real_dof))
	    { set_any = true; tc.Add(block_num, real_dof); }
	}
	return set_any;
      };
      fes.GetDofNrs(NodeId(FACET_NT(DIM), facet_nr), dnums_f);
      auto any_free = add_dofs(cblocks, os, facet_nr, dnums_f);
      int c = 0, pos = -1;
      if (any_free) {
	for (auto el_nr : elnums) {
	  fes.GetDofNrs(ElementId(VOL, el_nr), dnums_el);
	  c = 0; dnums_ex.SetSize(dnums_el.Size());
	  // iterate_anotb(dnums_el, dnums_f, [&](auto i) LAMBDA_INLINE { dnums_ex[c++] = dnums_el[i]; }); // not sorted!
	  for (auto el_dof : dnums_el)
	    if ( (pos = dnums_f.Pos(el_dof)) == -1)
	      { dnums_ex[c++] = el_dof; }
	  dnums_ex.SetSize(c);
	  add_dofs(cbex, os, facet_nr, dnums_ex);
	}
      }
    };
    for (; !cblocks.Done(); cblocks++, cbex++) {
      for (auto facet_nr : Range(n_blocks)) {
	ma->GetFacetElements(facet_nr, elnums);
	add_facet(os_sa, *spacea, facet_nr);
	add_facet(os_sb, *spaceb, facet_nr);
      }
    }

    auto eqc_h = make_shared<EQCHierarchy>(comp_pds, false); // TODO: get rid of these!
    auto sm = make_shared<HybridBS<double>> (comp_mat, eqc_h, cblocks.MoveTable(), cbex.MoveTable(), O.sm_mpi_overlap, O.sm_mpi_thread);

    return sm;
  } // FacetWiseAuxiliarySpaceAMG::BuildFLS


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  shared_ptr<BaseAMGPC::Options> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: NewOpts ()
  {
    return make_shared<Options>();
  } // FacetWiseAuxiliarySpaceAMG::NewOpts


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: SetDefaultOptions (BaseAMGPC::Options & aO)
  {
    auto & O (static_cast<Options&>(aO));

    O.with_rots = true;
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
    O.n_levels_d2_agg = 1;
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

  } // FacetWiseAuxiliarySpaceAMG::SetDefaultOptions


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: ModifyOptions (typename BaseAMGPC::Options & aO, const Flags & flags, string prefix)
  {
    auto & O (static_cast<Options&>(aO));

    __hacky_test = O.do_test;
    O.do_test = false;
  } // FacetWiseAuxiliarySpaceAMG::ModifyOptions


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  shared_ptr<BaseDOFMapStep> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: BuildEmbedding (shared_ptr<TopologicMesh> mesh)
  {

    /** aux_mat is finest mat **/
    // shared_ptr<TPMAT_TM> emb_mat;
    // if ( aux_pardofs->GetCommunicator().Size() > 2) {
    //   auto & vsort = node_sort[NT_VERTEX];
    //   auto perm = BuildPermutationMatrix<typename TAUX_TM::TENTRY>(vsort);
    //   emb_mat = MatMultAB(*pmat, *perm);
    // }
    // else
    //   { emb_mat = pmat; }
    // auto step = make_shared<ProlMap<TPMAT_TM>>(emb_mat, comp_pds, aux_pardofs);
    // return step;

    /** comp_mat is finest mat **/
    if (comp_pds->GetCommunicator().Size() > 2) {
      auto & vsort = node_sort[NT_VERTEX];
      auto perm = BuildPermutationMatrix<typename TAUX_TM::TENTRY>(vsort);
      auto step = make_shared<ProlMap<SparseMatrixTM<typename TAUX_TM::TENTRY>>>(perm, aux_pardofs, aux_pardofs);
      return step;
    }

    return nullptr;
  } // FacetWiseAuxiliarySpaceAMG::BuildEmbedding


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Update ()
  {
    ;
  } // FacetWiseAuxiliarySpaceAMG::Update




  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: AllocAuxMat ()
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


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: SetUpFacetMats ()
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
    facet_mat_data.SetSize(cnt_buf * DPV());
    facet_mat.SetSize(nfacets); cnt_buf = 0;
    for (auto facet_nr : Range(nfacets)) {
      int c = flo_a_f[facet_nr].Size() + flo_b_f[facet_nr].Size();
      if constexpr(DIM==3) {
	  for (auto fe : ma->GetFaceEdges(facet_nr))
	    { c += flo_a_e[fe].Size() + flo_b_e[fe].Size(); }
	}
      facet_mat[facet_nr].AssignMemory(c, DPV(), facet_mat_data.Addr(cnt_buf));
      cnt_buf += c * DPV();
    }
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


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: SetUpAuxParDofs ()
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
    aux_pardofs = make_shared<ParallelDofs> ( comm , ct.MoveTable(), DPV(), false);
  } // FacetWiseAuxiliarySpaceAMG::SetUpAuxParDofs


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: BuildPMat ()
  {
    auto H = comp_fes->GetNDof();
    auto nfacets = ma->GetNFacets(); // ...
    auto W = nfacets;
    Array<int> perow_f(H), perow_d(H);
    perow_f = 0; perow_d = 0;
    const auto & free_facets = *free_verts;
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
	const bool free = (free_verts == nullptr) ? true : free_verts->Test(fnum);
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

    // cout << "pmat: " << endl;
    // print_tm_spmat(cout, *pmat); cout << endl;
  } // FacetWiseAuxiliarySpaceAMG::BuildPMat


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS> template<ELEMENT_TYPE ET> INLINE
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: CalcFacetMat (ElementId vol_elid, int facet_nr, FlatMatrix<double> fmat, LocalHeap & lh)
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
    AUXFE fec (mid); constexpr auto ndc = DPV();

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
  } // FacetWiseAuxiliarySpaceAMG::CalcFacetMat


  /** Not sure what to do about BBND-elements (only relevant for some cases anyways) **/
  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
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


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Add_Facet (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
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


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Add_Vol_simple (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										 ElementId ei, LocalHeap & lh)
  {
    HeapReset hr(lh);

    auto & O (static_cast<Options&>(*options));

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
    FlatMatrix<double> P(inrange.NumSet(), DPV() * loc_n_facets, lh);
    P = 0;
    for (auto i : Range(facet_nrs)) {
      int ilo = i*DPV(), ihi = (i+1)*DPV();
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

    FlatMatrix<double> elm_P (ninrange, DPV() * loc_n_facets, lh);
    FlatMatrix<double> facet_elmat (DPV() * loc_n_facets, DPV() * loc_n_facets, lh);

    if (O.elmat_sc) { /** form schur-complement to low order part of elmat **/
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


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Add_Vol_rkP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									      ElementId ei, LocalHeap & lh)
  {
    /** Define Element-P by facet-matrices and aux_elmat = el_P.T * elmat * el_P.
	Then, regularize ker(el_P*el_P.T) **/
    throw Exception("Add_Vol_rkP todo");
  } // FacetWiseAuxiliarySpaceAMG::Add_Vol_rkP


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  INLINE void FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: Add_Vol_elP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
									      ElementId ei, LocalHeap & lh)
  {
    /** Asseble an extra element-P and define aux_elmat = el_P * elmat * el_P.T. **/
    throw Exception("Add_Vol_elP todo");
  } // FacetWiseAuxiliarySpaceAMG::Add_Vol_rkP


  template<int DIM, class SPACEA, class SPACEB, class AUXFE, class AMG_CLASS>
  Array<Array<shared_ptr<BaseVector>>> FacetWiseAuxiliarySpaceAMG<DIM, SPACEA, SPACEB, AUXFE, AMG_CLASS> :: GetRBModes () const
  {
    Array<Array<shared_ptr<BaseVector>>> rb_modes(2);
    auto bfa_mat = bfa->GetMatrixPtr();
    if (bfa_mat == nullptr)
      { throw Exception("mat not ready"); }
    if (pmat == nullptr)
      { throw Exception("pmat not ready"); }
    const auto & P = *pmat;
    /** displacements: (1,0,0), (0,1,0), (0,0,1) **/
    for (auto comp : Range(DIM)) {
      auto w = CreateAuxVector();
      w->SetParallelStatus(CUMULATED);
      auto fw = w->template FV<TV>();
      auto v = bfa_mat->CreateRowVector();
      v.SetParallelStatus(CUMULATED);
      auto fv = v.template FV<double>();
      for (auto k : Range(fw.Size()))
	{ fw(k) = 0; fw(k)(comp) = 1; }
      P.Mult(*w, *v);
      rb_modes[0].Append(v);
      rb_modes[1].Append(w);
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
	  w->SetParallelStatus(CUMULATED);
	  auto fw = w->template FV<TV>();
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
	  rb_modes[0].Append(v);
	  rb_modes[1].Append(w);
	}
      }
    else
      {
	/** (0,0,1) \cross x = (y, -x, 0) **/
	throw Exception("2d GetRBModes not implemented");
      }
    return rb_modes;
  } // FacetWiseAuxiliarySpaceAMG::GetRBModes

  /** END FacetWiseAuxiliarySpaceAMG **/


} // namespace amg


#include <python_ngstd.hpp>


namespace amg
{
  template<class PCC, class TLAM> void ExportFacetAux (py::module & m, string name, string descr, TLAM lam)
  {
    auto pyclass = py::class_<PCC, shared_ptr<PCC>, Preconditioner>(m, name.c_str(), descr.c_str());

    pyclass.def(py::init([&](shared_ptr<BilinearForm> bfa, string name, py::kwargs kwa) {
	  auto flags = CreateFlagsFromKwArgs(kwa, py::none());
	  flags.SetFlag("__delay__opts", true);
	  return make_shared<PCC>(bfa, flags, name);
	}), py::arg("bf"), py::arg("name") = "SolvyMcAMGFace");
    pyclass.def_property_readonly("P", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetPMat();
      }, "");
    pyclass.def_property_readonly("aux_mat", [](shared_ptr<PCC> pre) -> shared_ptr<BaseMatrix> {
	return pre->GetAuxMat();
      }, "");
    pyclass.def_property_readonly("aux_freedofs", [](shared_ptr<PCC> pre) -> shared_ptr<BitArray> {
	return pre->GetAuxFreeDofs();
      }, "");
    pyclass.def("CreateAuxVector", [](shared_ptr<PCC> pre) -> shared_ptr<BaseVector> {
	return pre->CreateAuxVector();
      });
    pyclass.def("GetRBModes", [](shared_ptr<PCC> pre) -> py::tuple {
	auto rbms = pre->GetRBModes();
	auto tup = py::tuple(2);
	tup[0] = MakePyList(rbms[0]);
	tup[1] = MakePyList(rbms[1]);
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
    
  }

} // namespace amg

#endif
