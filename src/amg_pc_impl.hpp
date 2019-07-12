
namespace amg
{

  /** Options **/

  // single base class so we only have the enums once
  struct BaseEmbedAMGOptions
  {
    /** Which subset of DOFs to perform the coarsening on **/
    enum DOF_SUBSET : char = { RANGE = 0,        // use Union { [ranges[i][0], ranges[i][1]) }
			       SELECTED = 1 };   // given by bitarray
    DOF_SUBSET subset = RANGE;
    Array<INT<2>> ss_ranges;
    shared_ptr<BitArray> ss_select;
    
    /** How the DOFs in the subset are mapped to vertices **/
    enum DOF_ORDERING : char = { REGULAR = 0,
				 VARIBALE = 1 };
    /**	REGULAR: sum(block_s) DOFs per "vertex", determined by ss_select and block_s
	   e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
	   each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks) 
	   
	VARIABLE: DOFs for vertex k: v_blocks[k] (not conistently implemented)
	subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13], but DOF 13 not in subet
    **/
    DOF_ORDERING dof_ordering = REGULAR;
    Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
    Table<int> v_blocks;

    /** How do we define the topology ? **/
    enum TOPOLOGY : char = { ALG = 0,        // by en entry in the finest level sparse matrix (restricted to subset)
			     MESH = 1,       // via the mesh
			     ELMAT = 2 };    // via element matrices
    TOPOLOGY topo = ALG;

    /** How do we compute vertex positions (if we need them) **/
    enum VERTEX_POSITION : char { VERTEX = 0,    // take from mesh vertex-positions
				  GIVEN = 1 };   // supplied from outside
    VERTEX_POSITION v_pos = VERTEX;
    FlatArray<Vec<3>> v_pos_array;

    /** How do we compute the replacement matrix **/
    enum ENERGY : char { TRIVIAL = 0,     // uniform weights
			 ALG = 1,         // from the sparse matrix
			 ELMAT = 2 };     // from element matrices
    ENERGY energy;
  };


  /** EmbedVAMG **/


  template<class Factory>
  struct EmbedVAMG<Factory>::Options : public Factory::Options,
				       public BaseEmbedAMGOptions
  {
    bool keep_vp = false;
    bool mat_ready = false;
    bool sync = false;
  };


  template<class Factory>
  void EmbedVAMG<Factory> :: MakeOptionsFromFlags (const Flags & flags, string prefix)
  {
    auto opts = make_shared<Factory::Options>();

    SetOptionsFromFlags(*opts, flags, prefix);

    return opts;
  }
  

  template<class Factory>
  void EmbedVAMG<Factory> :: SetOptionsFromFlags (Options & O, const Flags & flags, string prefix)
  {

    Factory::SetOptionsFromFlags(O, flags, frefix);

    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals, auto default_val) {
      string val = flags.GetStringFlag(prefix+key, "");
      bool found = false;
      for (auto k : Range(vals)) {
	if (v == vals[k]) {
	  found = true;
	  opt = decltype(opt)(k);
	  break;
	}
      }
      if (!found)
	{ opt = default_val; }
    };

    auto pfit = [] (string x) { return prefix + x; };

    set_enum_opt(O.subset, "on_dofs", {"range", "selected"}, RANGE);

    switch (O.subset) {
    case (RANGE) : {
      auto low = flags.GetNumListFlag(pfit("lower"));
      if (low.Size()) { // defined on multiple ranges
	auto up = flags.GetNumListFlag(pfit("upper"));
	ss_ranges.SetSize(low.Size());
	for (auto k : Range(low.Size()))
	  { ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }
      }
      else { // a single range
	ss_ranges.SetSize(1);
	size_t lowi = flags.GetNumFlag(pfit("lower"), 0);
	size_t upi = flags.GetNumFlag(pfit("upper"), bfa->GetFESpace()->GetNDof());
	// coarsen low order part, except if we are explicitely told not to
	if ( (lowi == 0) && (upi == bfa->GetFESpace()->GetNDof()) &&
	     (!flags.GetDefineFlagX(pfit("lo")).IsFalse()) )
	  if (auto lospace = bfa->GetFESpace()->LowOrderFESpacePtr()) // e.g compound has no LO space
	    { min_def_dof = 0; max_def_dof = lospace->GetNDof(); }
	ss_ranges[0] = { lowis, upi };
      }
      break;
    }
      // case(SELECTED) :
    default: { raise Exception("Not implemented"); break; }
    }
      
    set_enum_opt(O.dof_ordering, "dof_order", {"regular", "variable"}, REGULAR);

    set_enum_opt(O.topo, "edges", {"alg", "mesh", "elmat"}, ALG);

    set_enum_opt(O.v_pos, "vpos", {"vertex", "given"}, VERTEX);

    set_enum_opt(O.energy, "energy", {"triv", "alg", "elmat"}, ALG);

} // EmbedVAMG::MakeOptionsFromFlags


  template<class Factory>
  EmbedVAMG<Factory> :: EmbedVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string aname = "precond")
    : Preconditioner(blf, flags, name), bfa(blf)
  {
    
    options = MakeOptionsFromFlags (flags);

  } // EmbedVAMG::EmbedVAMG


  template<class Factory>
  void EmbedVAMG<Factory> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    if (freedofs == nullptr) // postpone to FinalizeLevel
      { return; }

    if (bfa->UsesEliminateInternal()) {
      auto fes = bfa->GetFESpace();
      finest_freedofs = make_shared<BitArray>(*freedofs);
      auto& ofd(*finest_freedofs);
      for (auto k : Range(freedofs->Size()))
	if (ofd.Test(k)) {
	  COUPLING_TYPE ct = fes->GetDofCouplingType(k);
	  if ((ct & CONDENSABLE_DOF) != 0)
	    ofd.Clear(k);
	}
    }
    else
      { finest_freedofs = freedofs; }

  } // EmbedVAMG::InitLevel


  template<class Factory>
  void EmbedVAMG<Factory> :: FinalizeLevel (const BaseMatrix * mat)
  {

    shared_ptr<BaseMatrix> fine_spm;
    if (mat != nullptr)
      { fine_spm = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { fine_spm = bfa->GetMatrixPtr(); }

    finest_mat = fine_spm; // embed-amg finest mat - need parallel matrix!

  } // EmbedVAMG::FinalizeLevel


  template<class Factory>
  void EmbedVAMG<Factory> :: Finalize ()
  {
    if (options->sync)
      {
	if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(mat)) {
	  static Timer t(string("NGsAMG - Initial Sync")); RegionTimer rt(t);
	  pmat->GetParallelDofs()->GetCommunicator().Barrier();
	}
      }

    if (finest_freedofs == nullptr)
      { finest_freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()); }
    
    shared_ptr<BaseMatrix> fine_spm = finest_mat;

    /** Set dummy-ParallelDofs **/
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
      { fine_spm = pmat->GetMatrix(); }
    else {
      if ( (GetEntryDim(fine_spm.get())==1) != (AMG_CLASS::DPN==1) )
	{ throw Exception("Not sure if this works..."); }
      NgsMPI_Comm c(MPI_COMM_WORLD);
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : AMG_ME_COMM;
      fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(pds), GetEntryDim(fine_spm.get()), false));
    }
    
    Finalize();
    auto mesh = BuildInitialMesh();

    factory = BuildFactory(mesh);

    // set mesh-rebuild here i guess

    /** Set up Smoothers **/
    BuildAMGMat();
    
  } // EmbedVAMG::Finalize


  template<class Factory>
  shared_ptr<Factory> EmbedVAMG<Factory> :: BuildFactory (shared_ptr<TMESH> mesh)
  {
    auto emb_step = BuildEmbedding();
    return make_shared<Factory>(mesh, emb_step, options);
  } // EmbedVAMG::BuildFactory


  template<class Factory>
  void EmbedVAMG<Factory> :: BuildAMGMat ()
  {
    /** Build coarse level matrices and grid-transfer **/
    Array<BaseSparseMatrix> mats;
    auto dof_map = make_shared<DOFMap>();
    factory->SetupLevels(mats, dof_map);

    // Set up smoothers
    Array<BaseSmoother> smoothers(mat.Size()-1);
    for (auto k : Range(size_t(0), mat.Size()-1)) {
      smoothers[k] = BuildSmoother (mats[k], dof_map->GetParDofs(k), (k==0) ? finest_free_dofs : nullptr);
      smoothers[k]->Finalize(); // do i even need this anymore ?
    }

    // Coarsest level setup

  } // EmbedVAMG::BuildAMGMat


  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BuildTopMesh ()
  {
    static Timer t("BuildTopMesh"); RegionTimer rt(t);

    auto & O(*options);

    /** Build inital EQC-Hierarchy **/
    shared_ptr<EQCHierarchy> eqc_h;
    auto fpd = finest_mat->GetParallelDofs();
    size_t maxset = 0;
    switch (O.subset) {
    case(RANGE): { maxset = ss_ranges.Last()[1]+1; break; }
    case(SELECTED): {
      auto sz = O.ss_select->Size();
      for (auto k : Range(sz--))
	if (O.ss_select->Test(sz--))
	  { maxset = k+1; break; }
      break;
    } }
    eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

    /** Build inital Mesh Topology **/
    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(MESH): { top_mesh = BTM_Mesh(eqc_h); break; }
    case(ELMAT): { top_mesh = BTM_Elmat(eqc_h); break; }
    case(ALG): { top_mesh = BTM_Alg(eqc_h); break; }
    }

    /** vertex positions, if we need them **/
    if (options->keep_vp) {
      auto & vsort = node_sort[0]; // also kinda hard coded for vertex-vertex pos, and no additional permutation
      auto & vpos(node_pos[NT_VERTEX]); vpos.SetSize(top_mesh->template GetNN<NT_VERTEX>());
      for (auto k : Range(vpos.Size()))
	ma->GetPoint(k,vpos[vsort[k]]);
    }
		
    /** Convert FreeDofs to FreeVerts (should probably do this above where I have v2d mapping1)**/

    // TODO:: this is hardcoded, but whatever ... 
    auto fvs = make_shared<BitArray>(top_mesh->GetNN<NT_VERTEX>()); fvs->Clear();
    auto & vsort = node_sort[NT_VERTEX];
    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>()))
      if (O.finest_free_dofs->Test(k))
	{ fvs->Set(vsort[k]); }
    free_verts = fvs;

    return top_mesh;
  }; //  EmbedVAMG::BuildTopMesh


  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    node_sort.SetSize(4);

    switch (O.v_pos) {
    case(VERTEX): {
      return MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
			      false, node_sort[2], false, node_sort[3]);
    }
    case(GIVEN): {
      throw Exception("Cannot combine custom vertices with topology from mesh");
      return nullptr;
    }
    default: { throw Exception("kinda unexpected case"); return nullptr; }
    }
  } // EmbedVAMG::BTM_Mesh
    

  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BTM_Alg (shared_ptr<EQCHierarchy> eqc_h)
  {
    node_sort.SetSize(4);

    auto top_mesh = make_shared<BlockTM>(eqc_h);

    size_t n_verts = 0, n_edges = 0;

    auto & vert_sort = node_sort[NT_VERTEX];

    // vertices
    auto set_vs = [&](auto nv, auto v2d) {
      vert_sort.SetSize(nv);
      top_mesh->SetVs (nv, [&](auto vnr)->FlatArray<int> { return fpd->GetDistantProcs(v2d(vnr)); },
		       [&vert_sort](auto i, auto j){ vert_sort[i] = j; });
    };

    // edges 
    auto create_edges = [&](auto v2d, auto d2v) {
      auto traverse_graph = [&](const auto& g, auto fun) { // vertex->dof,  // dof-> vertex
	for (auto k : Range(n_verts)) {
	  int row = v2d(k); // for find_in_sorted_array
	  auto ri = g.GetRowIndices(row);
	  auto pos = find_in_sorted_array(row, ri); // no duplicates
	  if (pos+1 < ri.Size()) {
	    for (auto col : ri.Part(pos+1)) {
	      auto j = d2v(col);
	      if (j != -1) {
		// cout << "dofs " << row << " " << col << " are vertices " << k << " " << j << endl;
		fun(vert_sort[k],vert_sort[j]);
	      }
	      // else {
	      //   cout << "dont use " << row << " " << j << " with dof " << col << endl;
	      // }
	    }
	  }
	}
      }; // traverse_graph
      auto bspm = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat);
	if (!bspm) { bspm = dynamic_pointer_cast<BaseSparseMatrix>( dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix()); }
	if (!bspm) { throw Exception("could not get BaseSparseMatrix out of finest_mat!!"); }
	n_edges = 0;
	traverse_graph(*bspm, [&](auto vk, auto vj) { n_edges++; });
	Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);
	n_edges = 0;
	traverse_graph(*bspm, [&](auto vk, auto vj) {
	    if (vk < vj) { epairs[n_edges++] = {vk, vj}; }
	    else { epairs[n_edges++] = {vj, vk}; }
	  });
	top_mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) { return epairs[num]; }, // (already v-sorted)
				     [](auto node_num, auto id) { /* dont care about edge-sort! */ });
      }; // create_edges

    if (O.dof_ordering == REGULAR) {
      if (O.subset == RANGE) {
	auto r0 = O.ss_ranges[0];
	const auto fes_bs = fpd->GetEntrySize();
	const auto bs0 = O.block_s[0]; // is this not kind of redundant ?
	int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	n_verts = (r0[1] - r0[0]) * fes_bs / dpv;
	auto d2v = [&](auto d) -> int { return ( (d%(fes_bs/bs0)) == 0) ? d*fes_bs/bs0 : -1; };
	auto v2d = [&](auto v) { return bs0/fes_bs * v; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
      }
      else { // SELECTED, subset by bitarray (is this tested??)
	size_t maxset = 0;
	auto sz = O.ss_select->Size();
	for (auto k : Range(sz--))
	  if (O.ss_select->Test(sz--))
	    { maxset = k+1; break; }
	n_verts = O.ss_select->NumSet() * fes_bs / dpv;
	Array<int> first_dof(n_verts);
	Array<int> compress(maxset); compress = -1;
	for (size_t k = 0, j = 0; k < n_verts; j += bs0 / fes_bs)
	  if (O.ss_select->Test(j))
	    { first_dof[k] = j; compress[j] = k++; }
	auto v2d = [&](auto v) { return first_dof[v]; };
	auto d2v = [&](auto d) -> int { return (d+1 > compress.Size()) ? -1 : compress[d]; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
	
      }
    }
    else { // VARIABLE, subset given via table anyways (is this even tested??)
      auto& vblocks = O.v_blocks;
      n_verts = vblocks.Size();
      auto v2d = [&](auto v) { return vblocks[v][0]; };
      Array<int> compress(vblocks[n_verts-1][0]); compress = -1;
      for (auto k : Range(compress.Size())) { compress[v2d(k)] = k; }
      auto d2v = [&](auto d) -> int { return (d+1 > compress.Size()) ? -1 : compress[d]; };
      set_vs (n_verts, v2d);
      create_edges ( v2d , d2v );
    }

    if (NgMPI_Comm(MPI_COMM_WORLD).Rank() == 1) {
      cout << "AMG performed on " << n_verts << " vertices, ndof local is: " << fpd->GetNDofLocal() << endl;
      cout << "free dofs " << options->finest_free_dofs->NumSet() << " ndof local is: " << fpd->GetNDofLocal() << endl;
    }

    return top_mesh;
  } // EmbedVAMG :: BTM_Alg


  /** EmbedWithElmats **/

  template<class Factory, class HTVD, class HTED>
  EmbedWithElmats<Factory, HTVD, HTED> :: EmbedWithElmats (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string aname = "precond")
    : EmbedVAMG<Factory>(bfa, aflags, aname)
  {
    if (options->energy == ELMATS) {
      shared_ptr<FESpace> lofes = fes;
      if (auto V = lofes->LowOrderFESpacePtr())
	{ lofes = V; }
      size_t NV = lofes->GetNDof(); // TODO: this overestimates for compound spaces
      ht_vertex = new HashTable<int, HTVD>(NV);
      ht_edge = new HashTable<INT<2,int>, HTED>(8*NV);
    }
  }


  template<class Factory, class HTVD, class HTED>
  EmbedWithElmats<Factory, HTVD, HTED> ::  ~EmbedWithElmats ()
  {
    if (ht_vertex != nullptr) delete ht_vertex;
    if (ht_edge   != nullptr) delete ht_edge;
  }


} // namespace amg
