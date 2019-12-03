#define FILE_AMGH1_CPP

#include "amg.hpp"
#include "amg_factory_impl.hpp"
#include "amg_pc_impl.hpp"

namespace amg
{

  /** H1AMGFactory **/


  H1AMGFactory :: H1AMGFactory (shared_ptr<H1AMGFactory::TMESH> mesh,  shared_ptr<H1AMGFactory::Options> opts,
				shared_ptr<BaseDOFMapStep> embed_step)
    : BASE(mesh, opts, embed_step)
  { ; }


  void H1AMGFactory :: SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix)
  {
    BASE::SetOptionsFromFlags(opts, flags, prefix);
  }


  void H1AMGFactory :: SetCoarseningOptions (VWCoarseningData::Options & opts, shared_ptr<H1Mesh> mesh) const
  {
    static Timer t("SetCoarseningOptions"); RegionTimer rt(t);
    const H1Mesh & rmesh(*mesh);
    const H1AMGFactory & self(*this);
    const auto & options = static_cast<const Options&>(*this->options);
    opts.free_verts = free_verts;
    auto NV = rmesh.template GetNN<NT_VERTEX>();
    auto NE = rmesh.template GetNN<NT_EDGE>();
    rmesh.CumulateData();
    auto vws = get<0>(rmesh.Data())->Data();
    auto ews = get<1>(rmesh.Data())->Data();
    Array<double> vcw(NV); vcw = 0;
    const auto & econ = *rmesh.GetEdgeCM();
    // cout << " vwts: " << endl;
    // prow2(vws); cout << endl << endl;
    // cout << " ewts: " << endl;
    // prow2(ews); cout << endl << endl;
    // cout << " SCO for mesh " << *mesh << endl;
    rmesh.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
	auto ew = ews[edge.id];
	vcw[edge.v[0]] += ew;
	vcw[edge.v[1]] += ew;
      }, true);
    rmesh.template AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) LAMBDA_INLINE { return sum_table(in); }, false);
    // cout << " ass vwts: " << endl;
    // prow2(vcw); cout << endl << endl;
    for (auto k : Range(vcw))
      { vcw[k] += vws[k]; }
    Array<double> ecw(NE);
    rmesh.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
	// double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
	// ecw[edge.id] = ews[edge.id] / vw;

	// ecw[edge.id] = ews[edge.id] / sqrt(vcw[edge.v[0]] * vcw[edge.v[1]]);

	// auto alpha = ews[edge.id];
	// auto aii = vcw[edge.v[0]] - alpha;
	// auto ajj = vcw[edge.v[0]] - alpha;
	// double rho = alpha * ( aii + ajj) / (aii * ajj);
	// ecw[edge.id] = rho / (1 + rho);

	// auto ecnt0 = econ.GetRowIndices(edge.v[0]).Size();
	// double x0 = 0.5 / ecnt0;
	// auto ecnt1 = econ.GetRowIndices(edge.v[1]).Size();
	// double x1 = 0.5 / ecnt0; 
	// if (ecw[edge.id] < min2(x0, x1))
	//   { ecw[edge.id] = 0; }

	ecw[edge.id] = ews[edge.id] * ( vcw[edge.v[0]] + vcw[edge.v[1]] ) / (vcw[edge.v[0]] * vcw[edge.v[1]]);
	
      }, false);
    // note: when using AMG as coarsetype of BDDC, dirichlet-dofs have no entries, so ecv/vcw[dir_dof] is 0 !
    for (auto v : Range(NV))
      { vcw[v] = (vcw[v] == 0) ? 0 : vws[v]/vcw[v]; }
    // cout << " vcws: " << endl; prow2(vcw); cout << endl << endl;
    // cout << " ecws: " << endl; prow2(ecw); cout << endl << endl;
    opts.vcw = move(vcw);
    opts.min_vcw = options.min_vcw;
    opts.ecw = move(ecw);
    opts.min_ecw = options.min_ecw;
  }


  /** EmbedVAMG<H1AMGFactory> **/


  template<> template<>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMGFactory> :: BuildEmbedding_impl<2> (shared_ptr<H1Mesh> mesh)
  { return nullptr; }


  template<> template<>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMGFactory> :: BuildEmbedding_impl<3> (shared_ptr<H1Mesh> mesh)
  { return nullptr; }


  template<> template<>
  shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMGFactory> :: BuildEmbedding_impl<6> (shared_ptr<H1Mesh> mesh)
  { return nullptr; }

  template<>
  void EmbedVAMG<H1AMGFactory> :: SetDefaultOptions (Options& O)
  {
    /** Coarsening Algorithm **/
    O.crs_alg = Options::CRS_ALG::AGG;
    O.agg_wt_geom = true;
    O.n_levels_d2_agg = 1;
    O.disc_max_bs = 5;

    /** Level-control **/
    // O.first_aaf = 1/pow(3, ma->GetDimension());
    O.first_aaf = (ma->GetDimension() == 3) ? 0.05 : 0.1;
    O.aaf = 1/pow(2, ma->GetDimension());

    /** Redistribute **/
    O.enable_ctr = true;
    O.ctraf = 0.05;
    O.first_ctraf = O.aaf * O.first_aaf;
    O.ctraf_scale = 1;
    O.ctr_crs_thresh = 0.9;
    O.ctr_min_nv_gl = 5000;
    O.ctr_seq_nv = 5000;
    
    /** Smoothed Prolongation **/
    O.enable_sm = true;
    O.sp_min_frac = (ma->GetDimension() == 3) ? 0.08 : 0.15;
    O.sp_omega = 1;
    O.sp_max_per_row = 1 + ma->GetDimension();

    /** Rebuild Mesh**/
    O.enable_rbm = true;
    O.rbmaf = O.aaf * O.aaf;
    O.first_rbmaf = O.aaf * O.first_aaf;

    /** Embed **/
    O.block_s = { 1 }; // scalar, so always one dof per vertex

  } // EmbedVAMG<H1AMGFactory>::SetDefaultOptions 


  template<>
  void EmbedVAMG<H1AMGFactory> :: ModifyOptions (Options & O, const Flags & flags, string prefix)
  {
    static Timer t_rbm("Rebuild Mesh");
    O.rebuild_mesh = [&](shared_ptr<H1Mesh> mesh, shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> pardofs) {
      cout << IM(4) << "REBUILD MESH, NV = " << mesh->GetNNGlobal<NT_VERTEX>() << ", NE = " << mesh->GetNNGlobal<NT_EDGE>() << endl;

      RegionTimer rt(t_rbm);
      /** New edges **/

      auto n_verts = mesh->GetNN<NT_VERTEX>();
      auto traverse_graph = [&](const auto& g, auto fun) LAMBDA_INLINE { // vertex->dof,  // dof-> vertex
	for (auto row : Range(n_verts)) {
	  auto ri = g.GetRowIndices(row);
	  auto pos = find_in_sorted_array(int(row), ri); // no duplicates
	  if (pos+1 < ri.Size())
	    for (auto j : ri.Part(pos+1))
	      { fun(row,j); }
	}
      }; // traverse_graph
      const auto& cspm = static_cast<SparseMatrixTM<double>&>(*mat);
      size_t n_edges = 0;
      traverse_graph(cspm, [&](auto vk, auto vj) LAMBDA_INLINE { n_edges++; });
      Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);
      n_edges = 0;
      traverse_graph(cspm, [&](auto vk, auto vj) LAMBDA_INLINE {
	  if (vk < vj) { epairs[n_edges++] = {int(vk), int(vj)}; }
	  else { epairs[n_edges++] = {int(vj), int(vk)}; }
	});
      mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) LAMBDA_INLINE { return epairs[num]; }, // (already v-sorted)
			       [](auto node_num, auto id) { /* dont care about edge-sort! */ });
      mesh->ResetEdgeCM();

      n_edges = mesh->GetNN<NT_EDGE>();
      
      cout << IM(4) << "REBUILT MESH, NV = " << mesh->GetNNGlobal<NT_VERTEX>() << ", NE = " << mesh->GetNNGlobal<NT_EDGE>() << endl;

      /** New weights **/
      auto avd = get<0>(mesh->Data()); avd->SetParallelStatus(DISTRIBUTED);
      auto vdata = avd->Data(); vdata = 0;
      avd->Cumulate();
      // for (auto k : Range(mat->Height())) {
      // 	double rs = 0;
      // 	for(auto v : cspm.GetRowValues(k))
      // 	  { rs += v; }
      // 	vdata[k] = rs;
      // }

      auto aed = get<1>(mesh->Data()); aed->SetParallelStatus(DISTRIBUTED);
      auto & edata = aed->GetModData(); edata.SetSize(n_edges);
      // off-diag entry -> is edge weight
      auto edges = mesh->GetNodes<NT_EDGE>();
      for (auto & e : edges)
	{ edata[e.id] = fabs(cspm(e.v[0], e.v[1])); }
      aed->Cumulate();

      return mesh;
    };
  }


  template<> shared_ptr<H1Mesh>
  EmbedVAMG<H1AMGFactory> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh)
  {
    static Timer t("BuildAlgMesh_TRIV"); RegionTimer rt(t);

    /**
       vertex-weights are 0
       edge-weihts are 1
     **/

    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); a->Data() = 0.0; 
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), CUMULATED); b->Data() = 1.0;
    auto mesh = make_shared<H1Mesh>(move(*top_mesh), a, b);
    return mesh;
  }


  template<> template<class TD2V, class TV2D> shared_ptr<H1Mesh>
  EmbedVAMG<H1AMGFactory> :: BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    static Timer t("BuildAlgMesh_ALG_scal"); RegionTimer rt(t);

    static_assert(is_same<int, decltype(D2V(0))>::value, "D2V mismatch");
    static_assert(is_same<int, decltype(V2D(0))>::value, "V2D mismatch");

    auto dspm = dynamic_pointer_cast<SparseMatrix<double>>(spmat);
    if (dspm == nullptr)
      { throw Exception("Could not cast sparse matrix!"); }

    // cout << "finest level mat: " << endl << *dspm << endl;

    const auto& cspm = *dspm;
    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED); auto ad = a->Data(); ad = 0;
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); auto bd = b->Data(); bd = 0;

    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>()))
      { auto d = V2D(k); ad[k] = cspm(d,d); }

    auto edges = top_mesh->GetNodes<NT_EDGE>();
    auto& fvs = *free_verts;
    for (auto & e : edges) {
      auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
      double v = cspm(di, dj);
      // bd[e.id] = fabs(v) / sqrt(cspm(di,di) * cspm(dj,dj)); ad[e.v[0]] += v; ad[e.v[1]] += v;
      bd[e.id] = fabs(v); ad[e.v[0]] += v; ad[e.v[1]] += v;
    }
    
    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>())) // -1e-16 can happen, is problematic
      { ad[k] = fabs(ad[k]); }
    auto mesh = make_shared<H1Mesh>(move(*top_mesh), a, b);
    return mesh;
  }

  template<> template<class TD2V, class TV2D> shared_ptr<H1Mesh>
  EmbedVAMG<H1AMGFactory> :: BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    throw Exception("not necessary, just implemented so it compiles!");
    return nullptr;
  }


  template<> template<int N>
  shared_ptr<stripped_spm_tm<typename strip_mat<Mat<N, 1, double>>::type>> EmbedVAMG<H1AMGFactory> :: BuildED (size_t subset_count, shared_ptr<H1Mesh> mesh)
  { return nullptr; }


  // template<> shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMGFactory> :: BuildEmbedding ()
  // {
  //   static Timer t("BuildEmbedding"); RegionTimer rt(t);
  //   typedef BaseEmbedAMGOptions BAO;
  //   const auto &O(*options);
  //   auto & vsort = node_sort[NT_VERTEX];
  //   shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();
  //   auto emb_mat = BuildPermutationMatrix<double>(vsort);
  //   if (O.subset == BAO::RANGE_SUBSET) {
  //     if ( (O.ss_ranges[0][0] != 0) || (O.ss_ranges[0][1] != fpds->GetNDofLocal()) ) { // otherwise, dont need additional cutoff
  // 	Array<int> perow(fpds->GetNDofLocal()); perow = 0;
  // 	for (auto r : O.ss_ranges) {
  // 	  for (auto l : Range(r[0], r[1]))
  // 	    perow[l] = 1;
  // 	}
  // 	auto mat = make_shared<SparseMatrixTM<double>>(perow, fpds->GetNDofLocal());
  // 	int cnt = 0;
  // 	for (auto k : Range(fpds->GetNDofLocal())) {
  // 	  auto ri = mat->GetRowIndices(k);
  // 	  if (ri.Size()) {
  // 	    ri[0] = cnt++;
  // 	    mat->GetRowValues(k) = 1;
  // 	  }
  // 	}
  // 	emb_mat = MatMultAB(*mat, *emb_mat);
  //     }
  //   }
  //   else if (O.subset == BAO::SELECTED_SUBSET) { // embed this
  //     Array<int> perow(fpds->GetNDofLocal());
  //     for (auto k : Range(fpds->GetNDofLocal()))
  // 	perow[k] = O.ss_select->Test(k) ? 1 : 0;
  //     auto mat = make_shared<SparseMatrixTM<double>>(perow, fpds->GetNDofLocal());
  //     int cnt = 0;
  //     for (auto k : Range(fpds->GetNDofLocal()))
  // 	if (O.ss_select->Test(k)) {
  // 	  mat->GetRowIndices(k) = cnt++;
  // 	  mat->GetRowValues(k) = 1;
  // 	}
  //     emb_mat = MatMultAB(*mat, *emb_mat);
  //   }
  //   else if (vsort.Size() != fpds->GetNDofLocal())
  //     { throw Exception("When seem to not be working on the full space, but we do not know where!"); }
  //   if (fpds->GetNDofLocal() != emb_mat->Height())
  //     throw Exception(string("EMBED MAT H does not fit: ") + to_string(fpds->GetNDofLocal()) + string("!=") + to_string(emb_mat->Height()));
  //   if (vsort.Size() != emb_mat->Width())
  //     throw Exception(string("EMBED MAT W does not fit: ") + to_string(vsort.Size()) + string("!=") + to_string(emb_mat->Height()));
  //   auto pmap = make_shared<ProlMap<SparseMatrixTM<double>>>(emb_mat, fpds, nullptr);
  //   return pmap;
  // } // EmbedVAMG<H1AMGFactory>::BuildEmbedding


  template<> shared_ptr<BaseSmoother> EmbedVAMG<H1AMGFactory> :: BuildSmoother (shared_ptr<BaseSparseMatrix> m, shared_ptr<ParallelDofs> pds,
										shared_ptr<BitArray> freedofs) const
  {
    // cout << "buildsmoother for " << pds->GetNDofGlobal() << "dofs " << endl;
    shared_ptr<SparseMatrix<double>> spmat = dynamic_pointer_cast<SparseMatrix<double>> (m);
    if (options->old_smoothers) {
      auto sm = make_shared<HybridGSS<1>> (spmat, pds, freedofs);
      sm->SetSymmetric(options->smooth_symmetric);
      // cout << "OK buildsmoother for " << pds->GetNDofGlobal() << "dofs " << endl;
      return sm;
    }
    else {

      auto parmat = make_shared<ParallelMatrix>(spmat, pds, pds, C2D);

      // cout << "OK buildsmoother for " << pds->GetNDofGlobal() << "dofs " << endl;

      // auto sm = make_shared<HybridGSS2<double>> (parmat, freedofs);
      // sm->SetSymmetric(options->smooth_symmetric);

      auto eqc_h = make_shared<EQCHierarchy>(pds, false); // todo: get rid of these!
      auto sm = make_shared<HybridGSS3<double>> (parmat, eqc_h, freedofs, options->mpi_overlap, options->mpi_thread);
      sm->SetSymmetric(options->smooth_symmetric);
      sm->Finalize();

      return sm;
    }
  }


  /** EmbedWithElmats<H1AMGFactory> **/


  template<> shared_ptr<H1Mesh> EmbedWithElmats<H1AMGFactory, double, double> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    if ( (ht_vertex == nullptr) || (ht_edge == nullptr) )
      { throw Exception("elmat-energy, but have to HTs! (HOW)"); }

    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED);
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);

    FlatArray<int> vsort = node_sort[NT_VERTEX];
    Array<int> rvsort(vsort.Size());
    for (auto k : Range(vsort.Size()))
      rvsort[vsort[k]] = k;
    auto ad = a->Data();
    for (auto key_val : *ht_vertex) {
      ad[rvsort[get<0>(key_val)]] = get<1>(key_val);
    }
    auto bd = b->Data();
    auto edges = top_mesh->GetNodes<NT_EDGE>();
    for (auto & e : edges) {
      bd[e.id] = (*ht_edge)[INT<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
    }

    auto mesh = make_shared<H1Mesh>(move(*top_mesh), a, b);
    return mesh;
  } // EmbedWithElmats<H1AMGFactory, double, double>::BuildAlgMesh


  template<> void EmbedWithElmats<H1AMGFactory, double, double> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
										     ElementId ei, LocalHeap & lh)
  {
    typedef BaseEmbedAMGOptions BAO;
    const auto &O(*options);

    if (O.energy != BAO::ELMAT_ENERGY)
      { return; }

    // vertex weights
    static Timer t("AddElementMatrix");
    static Timer t1("AddElementMatrix - inv");
    static Timer t3("AddElementMatrix - v-schur");
    static Timer t5("AddElementMatrix - e-schur");
    RegionTimer rt(t);
    size_t ndof = dnums.Size();
    BitArray used(ndof, lh);
    FlatMatrix<double> ext_elmat(ndof+1, ndof+1, lh);
    {
      ThreadRegionTimer reg (t5, TaskManager::GetThreadId());
      ext_elmat.Rows(0,ndof).Cols(0,ndof) = elmat;
      ext_elmat.Row(ndof) = 1;
      ext_elmat.Col(ndof) = 1;
      ext_elmat(ndof, ndof) = 0;
      CalcInverse (ext_elmat);
    }
    {
      RegionTimer reg (t1);
      for (size_t i = 0; i < dnums.Size(); i++)
        {
          Mat<2,2,double> ai;
          ai(0,0) = ext_elmat(i,i);
          ai(0,1) = ai(1,0) = ext_elmat(i, ndof);
          ai(1,1) = ext_elmat(ndof, ndof);
          ai = Inv(ai);
          double weight = fabs(ai(0,0));
          // vertex_weights_ht.Do(INT<1>(dnums[i]), [weight] (auto & v) { v += weight; });
          (*ht_vertex)[dnums[i]] += weight;
        }
    }
    {
      RegionTimer reg (t3);
      for (size_t i = 0; i < dnums.Size(); i++)
        for (size_t j = 0; j < i; j++)
          {
            Mat<3,3,double> ai;
            ai(0,0) = ext_elmat(i,i);
            ai(1,1) = ext_elmat(j,j);
            ai(0,1) = ai(1,0) = ext_elmat(i,j);
            ai(2,2) = ext_elmat(ndof,ndof);
            ai(0,2) = ai(2,0) = ext_elmat(i,ndof);
            ai(1,2) = ai(2,1) = ext_elmat(j,ndof);
            ai = Inv(ai);
            double weight = fabs(ai(0,0));
            // edge_weights_ht.Do(INT<2>(dnums[j], dnums[i]).Sort(), [weight] (auto & v) { v += weight; });
	    (*ht_edge)[INT<2, int>(dnums[j], dnums[i]).Sort()] += weight;
          }
    }
  } // EmbedWithElmats<H1AMGFactory, double, double>::AddElementMatrix

  RegisterPreconditioner<EmbedWithElmats<H1AMGFactory, double, double>> register_h1amg_scal("ngs_amg.h1_scal");

} // namespace amg

#include "amg_tcs.hpp"
