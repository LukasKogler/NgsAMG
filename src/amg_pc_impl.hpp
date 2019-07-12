#ifndef FILE_AMGPC_IMPL_HPP
#define FILE_AMGPC_IMPL_HPP

namespace amg
{

  /** Options **/

  // single base class so we only have the enums once
  struct BaseEmbedAMGOptions
  {
    /** Which subset of DOFs to perform the coarsening on **/
    enum DOF_SUBSET : char { RANGE_SUBSET = 0,        // use Union { [ranges[i][0], ranges[i][1]) }
			     SELECTED_SUBSET = 1 };   // given by bitarray
    DOF_SUBSET subset = RANGE_SUBSET;
    Array<INT<2, size_t>> ss_ranges; // ranges must be non-overlapping and incresing
    shared_ptr<BitArray> ss_select;
    
    /** How the DOFs in the subset are mapped to vertices **/
    enum DOF_ORDERING : char { REGULAR_ORDERING = 0,
			       VARIBALE_ORDERING = 1 };
    /**	REGULAR: sum(block_s) DOFs per "vertex", determined by ss_select and block_s
	   e.g: block_s = [2,3], then we have NV blocks of 2 vertices, then NV blocks of 3 vertices
	   each block is increasing and continuous (neither DOFs [12,18] nor DOFs [5,4] are valid blocks) 
	   
	VARIABLE: DOFs for vertex k: v_blocks[k] (not conistently implemented)
	subset must be consistent for all dofs in each block ( so we cannot have a block of DOFs [12,13], but DOF 13 not in subet
    **/
    DOF_ORDERING dof_ordering = REGULAR_ORDERING;
    Array<int> block_s; // we are computing NV from this, so don't put freedofs in here, one BS per given range
    Table<int> v_blocks;

    /** How do we define the topology ? **/
    enum TOPO : char { ALG_TOPO = 0,        // by en entry in the finest level sparse matrix (restricted to subset)
		       MESH_TOPO = 1,       // via the mesh
		       ELMAT_TOPO = 2 };    // via element matrices
    TOPO topo = ALG_TOPO;

    /** How do we compute vertex positions (if we need them) **/
    enum POSITION : char { VERTEX_POS = 0,    // take from mesh vertex-positions
			   GIVEN_POS = 1 };   // supplied from outside
    POSITION v_pos = VERTEX_POS;
    FlatArray<Vec<3>> v_pos_array;

    /** How do we compute the replacement matrix **/
    enum ENERGY : char { TRIV_ENERGY = 0,     // uniform weights
			 ALG_ENERGY = 1,         // from the sparse matrix
			 ELMAT_ENERGY = 2 };     // from element matrices
    ENERGY energy = ALG_ENERGY;
  };


  template<class Factory>
  struct EmbedVAMG<Factory>::Options : public Factory::Options,
				       public BaseEmbedAMGOptions
  {
    bool keep_vp = false;
    bool mat_ready = false;
    bool sync = false;
    /** Smoothers **/
    bool old_smoothers = false;
    bool smooth_symmetric = false;
  };


  /** EmbedVAMG **/


  template<class Factory>
  shared_ptr<typename EmbedVAMG<Factory>::Options> EmbedVAMG<Factory> :: MakeOptionsFromFlags (const Flags & flags, string prefix)
  {
    auto opts = make_shared<Options>();

    SetOptionsFromFlags(*opts, flags, prefix);

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };
    
    set_bool(opts->old_smoothers, "oldsm");
    set_bool(opts->old_smoothers, "symsm");

    return opts;
  }
  

  template<class Factory>
  void EmbedVAMG<Factory> :: SetOptionsFromFlags (Options & O, const Flags & flags, string prefix)
  {

    Factory::SetOptionsFromFlags(O, flags, prefix);

    auto pfit = [prefix] (string x) { return prefix + x; };

    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals, auto default_val) {
      string val = flags.GetStringFlag(pfit(key), "");
      bool found = false;
      for (auto k : Range(vals)) {
	if (val == vals[k]) {
	  found = true;
	  opt = decltype(opt)(k);
	  break;
	}
      }
      if (!found)
	{ opt = default_val; }
    };

    typedef BaseEmbedAMGOptions BAO;

    set_enum_opt(O.subset, "on_dofs", {"range", "selected"}, BAO::RANGE_SUBSET);

    switch (O.subset) {
    case (BAO::RANGE_SUBSET) : {
      auto &low = flags.GetNumListFlag(pfit("lower"));
      if (low.Size()) { // defined on multiple ranges
	auto &up = flags.GetNumListFlag(pfit("upper"));
	O.ss_ranges.SetSize(low.Size());
	for (auto k : Range(low.Size()))
	  { O.ss_ranges[k] = { size_t(low[k]), size_t(up[k]) }; }
      }
      else { // a single range
	size_t lowi = flags.GetNumFlag(pfit("lower"), 0);
	size_t upi = flags.GetNumFlag(pfit("upper"), bfa->GetFESpace()->GetNDof());
	// coarsen low order part, except if we are explicitely told not to
	if ( (lowi == 0) && (upi == bfa->GetFESpace()->GetNDof()) &&
	     (!flags.GetDefineFlagX(pfit("lo")).IsFalse()) )
	  if (auto lospace = bfa->GetFESpace()->LowOrderFESpacePtr()) // e.g compound has no LO space
	    { lowi = 0; upi = lospace->GetNDof(); }
	O.ss_ranges.SetSize(1);
	O.ss_ranges[0] = { lowi, upi };
      }
      break;
    }
      // case(SELECTED) :
    default: { throw Exception("Not implemented"); break; }
    }
      
    set_enum_opt(O.dof_ordering, "dof_order", {"regular", "variable"}, BAO::REGULAR_ORDERING);

    set_enum_opt(O.topo, "edges", {"alg", "mesh", "elmat"}, BAO::ALG_TOPO);

    set_enum_opt(O.v_pos, "vpos", {"vertex", "given"}, BAO::VERTEX_POS);

    set_enum_opt(O.energy, "energy", {"triv", "alg", "elmat"}, BAO::ALG_ENERGY);

    ModifyOptions(O, flags, prefix);

  } // EmbedVAMG::MakeOptionsFromFlags


  template<class Factory>
  void EmbedVAMG<Factory> :: ModifyOptions (Options & O, const Flags & flags, string prefix)
  { ; }


  template<class Factory>
  EmbedVAMG<Factory> :: EmbedVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name)
    : Preconditioner(blf, flags, name), bfa(blf)
  {
    cout << "constr" << endl;
    options = MakeOptionsFromFlags (flags);
    cout << "have opts" << endl;
  } // EmbedVAMG::EmbedVAMG


  template<class Factory>
  void EmbedVAMG<Factory> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    cout << "init" << endl;

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
    cout << "finalizelevel" << endl;

    if (mat != nullptr)
      { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { finest_mat = bfa->GetMatrixPtr(); }

    Finalize();

  } // EmbedVAMG::FinalizeLevel


  template<class Factory>
  void EmbedVAMG<Factory> :: Finalize ()
  {
   cout << "finalize" << endl;

   if (options->sync)
      {
	if (auto pds = finest_mat->GetParallelDofs()) {
	  static Timer t(string("NGsAMG - Initial Sync")); RegionTimer rt(t);
	  pds->GetCommunicator().Barrier();
	}
      }

    if (finest_freedofs == nullptr)
      { finest_freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()); }
    
    cout << "finalize" << endl;

    /** Set dummy-ParallelDofs **/
    shared_ptr<BaseMatrix> fine_spm = finest_mat;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
      { fine_spm = pmat->GetMatrix(); }
    else {
      Array<int> perow (fine_spm->Height() ); perow = 0;
      Table<int> dps (perow);
      NgsMPI_Comm c(MPI_COMM_WORLD);
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : AMG_ME_COMM;
      fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(dps), GetEntryDim(fine_spm.get()), false));
    }

    cout << "finalize" << endl;

    auto mesh = BuildInitialMesh();

    cout << "finalize" << endl;
    factory = BuildFactory(mesh);

    cout << "finalize" << endl;
    // set mesh-rebuild here i guess

    /** Set up Smoothers **/
    BuildAMGMat();
    cout << "finalize" << endl;
    
  } // EmbedVAMG::Finalize


  template<class Factory>
  shared_ptr<Factory> EmbedVAMG<Factory> :: BuildFactory (shared_ptr<TMESH> mesh)
  {
    auto emb_step = BuildEmbedding();
    return make_shared<Factory>(mesh, options, emb_step);
  } // EmbedVAMG::BuildFactory


  template<class Factory>
  void EmbedVAMG<Factory> :: BuildAMGMat ()
  {
    /** Build coarse level matrices and grid-transfer **/

    if (finest_mat == nullptr)
      { throw Exception("Do not have a finest level matrix!"); }

    shared_ptr<BaseSparseMatrix> fspm = nullptr;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
      { fspm = dynamic_pointer_cast<BaseSparseMatrix>(pmat->GetMatrix()); }
    else
      { dynamic_pointer_cast<BaseSparseMatrix>(finest_mat); }

    if (fspm == nullptr)
      { throw Exception("Could not cast finest mat!"); }
    
    Array<shared_ptr<BaseSparseMatrix>> mats ({ fspm });
    auto dof_map = make_shared<DOFMap>();
    factory->SetupLevels(mats, dof_map);

    // Set up smoothers
    Array<shared_ptr<BaseSmoother>> smoothers(mats.Size()-1);
    for (auto k : Range(size_t(0), mats.Size()-1)) {
      smoothers[k] = BuildSmoother (mats[k], dof_map->GetParDofs(k), (k==0) ? finest_freedofs : nullptr);
      smoothers[k]->Finalize(); // do i even need this anymore ?
    }

    // Coarsest level setup

  } // EmbedVAMG::BuildAMGMat


  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BuildTopMesh ()
  {
    static Timer t("BuildTopMesh"); RegionTimer rt(t);

    cout << "top mesh" << endl;
    
    auto & O(*options);

    typedef BaseEmbedAMGOptions BAO;

    /** Build inital EQC-Hierarchy **/
    shared_ptr<EQCHierarchy> eqc_h;
    auto fpd = finest_mat->GetParallelDofs();
    size_t maxset = 0;
    switch (O.subset) {
    case(BAO::RANGE_SUBSET): { maxset = O.ss_ranges.Last()[1]; break; }
    case(BAO::SELECTED_SUBSET): {
      auto sz = O.ss_select->Size();
      for (auto k : Range(sz--))
	if (O.ss_select->Test(sz--))
	  { maxset = k+1; break; }
      break;
    } }
    cout << "maxset " << maxset << " " << fpd->GetNDofLocal() << endl;

    eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

    /** Build inital Mesh Topology **/
    shared_ptr<BlockTM> top_mesh;
    switch(O.topo) {
    case(BAO::MESH_TOPO): { top_mesh = BTM_Mesh(eqc_h); break; }
    case(BAO::ELMAT_TOPO): { top_mesh = BTM_Elmat(eqc_h); break; }
    case(BAO::ALG_TOPO): { top_mesh = BTM_Alg(eqc_h); break; }
    }

    /** vertex positions, if we need them **/
    if (O.keep_vp) {
      node_pos.SetSize(1);
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
      if (finest_freedofs->Test(k))
	{ fvs->Set(vsort[k]); }
    free_verts = fvs;

    cout << "top mesh done" << endl;

    return top_mesh;
  }; //  EmbedVAMG::BuildTopMesh


  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h)
  {
    cout << "Topmesh mesh" << endl;

    node_sort.SetSize(4);

    typedef BaseEmbedAMGOptions BAO;

    const auto &O(*options);

    switch (O.v_pos) {
    case(BAO::VERTEX_POS): {
      return MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
			      false, node_sort[2], false, node_sort[3]);
    }
    case(BAO::GIVEN_POS): {
      throw Exception("Cannot combine custom vertices with topology from mesh");
      return nullptr;
    }
    default: { throw Exception("kinda unexpected case"); return nullptr; }
    }
  } // EmbedVAMG::BTM_Mesh
    

  template<class Factory>
  shared_ptr<BlockTM> EmbedVAMG<Factory> :: BTM_Alg (shared_ptr<EQCHierarchy> eqc_h)
  {
    cout << "Topmesh alg" << endl;

    node_sort.SetSize(4);

    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    auto top_mesh = make_shared<BlockTM>(eqc_h);

    size_t n_verts = 0, n_edges = 0;

    auto & vert_sort = node_sort[NT_VERTEX];

    auto fpd = finest_mat->GetParallelDofs();

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

    if (O.dof_ordering == BAO::REGULAR_ORDERING) {
      cout << "case 1 " << endl;
      const auto fes_bs = fpd->GetEntrySize();
      int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
      const auto bs0 = O.block_s[0]; // is this not kind of redundant ?
      if (O.subset == BAO::RANGE_SUBSET) {
      cout << "case 1 " << endl;
	auto r0 = O.ss_ranges[0];
	int dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
	n_verts = (r0[1] - r0[0]) * fes_bs / dpv;
	auto d2v = [&](auto d) -> int { return ( (d%(fes_bs/bs0)) == 0) ? d*fes_bs/bs0 : -1; };
	auto v2d = [&](auto v) { return bs0/fes_bs * v; };
	set_vs (n_verts, v2d);
	create_edges ( v2d , d2v );
      }
      else { // SELECTED, subset by bitarray (is this tested??)
      cout << "case 2 " << endl;
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
      cout << "case 2 " << endl;
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
      cout << "free dofs " << finest_freedofs->NumSet() << " ndof local is: " << fpd->GetNDofLocal() << endl;
    }

    cout << "Topmesh alg done" << endl;

    return top_mesh;
  } // EmbedVAMG :: BTM_Alg


  /** EmbedWithElmats **/

  template<class Factory, class HTVD, class HTED>
  EmbedWithElmats<Factory, HTVD, HTED> :: EmbedWithElmats (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name)
    : EmbedVAMG<Factory>(bfa, aflags, name)
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    if (O.energy == BAO::ELMAT_ENERGY) {
      shared_ptr<FESpace> lofes = bfa->GetFESpace();
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

#endif //  FILE_AMGPC_IMPL_HPP
