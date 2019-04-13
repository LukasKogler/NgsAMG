#ifdef ELASTICITY
#define FILE_AMGELAST_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{
  template<int D> shared_ptr<BaseSmoother>
  ElasticityAMG<D> :: BuildSmoother  (INT<3> level, shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> par_dofs,
				      shared_ptr<BitArray> free_dofs)
  {
    auto options = static_pointer_cast<Options>(this->options);
    if (shared_ptr<const SparseMatrix<Mat<dofpv(D), dofpv(D), double>>> spmat = dynamic_pointer_cast<SparseMatrix<Mat<dofpv(D), dofpv(D), double>>> (mat)) {
      if (options->regularize)
	{ return make_shared<StabHGSS<dofpv(D), disppv(D), dofpv(D)>> (spmat, par_dofs, free_dofs); }
      else
	{ return make_shared<HybridGSS<dofpv(D)>> (spmat, par_dofs, free_dofs); }
    }
    else if (shared_ptr<const SparseMatrix<Mat<disppv(D), disppv(D), double>>> spmat = dynamic_pointer_cast<SparseMatrix<Mat<disppv(D), disppv(D), double>>> (mat)) {
      return make_shared<HybridGSS<disppv(D)>> (spmat, par_dofs, free_dofs);
    }
    else if (shared_ptr<const SparseMatrix<double>> spmat = dynamic_pointer_cast<SparseMatrix<double>> (mat)) {
      return make_shared<HybridGSS<1>> (spmat, par_dofs, free_dofs);
    }
    throw Exception(string("Could not build a Smoother for mat-type ") + string(typeid(*mat).name()));
    return nullptr;
  }


  template<> shared_ptr<ElasticityAMG<3>::TSPMAT>
  ElasticityAMG<3> :: RegularizeMatrix (shared_ptr<ElasticityAMG<3>::TSPMAT> mat, shared_ptr<ParallelDofs> & pardofs)
  { return mat; }

  template<> void
  ElasticityAMG<3> :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts,
					    INT<3> level, shared_ptr<TMESH> _mesh)
  { BASE::SetCoarseningOptions(opts, level, mesh); };

  template<> shared_ptr<ElasticityAMG<2>::TSPMAT>
  ElasticityAMG<2> :: RegularizeMatrix (shared_ptr<ElasticityAMG<2>::TSPMAT> mat, shared_ptr<ParallelDofs> & pardofs)
  {
    cout << "REGULARIZE D 2" << endl;
    for(auto k : Range(mat->Height())) {
      auto& diag = (*mat)(k,k);
      if (diag(2,2) == 0.0) {
	diag(2,2) = 1.0;
	cout << "RD diag: " << endl; print_tm(cout, diag); cout << endl;
      }
      else {
	cout << "OK diag: " << endl; print_tm(cout, diag); cout << endl;
      }
    }
    return mat;
  }

  template<> void
  ElasticityAMG<2> :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts,
					    INT<3> level, shared_ptr<TMESH> _mesh)
  {
    cout << "SCE LEVEL " << level << endl;
    constexpr int D = 2;
    static Timer t(this->name+string("::SetCoarseningOptions")); RegionTimer rt(t);
    const ElasticityMesh<D> & mesh(*_mesh);
    auto NV = mesh.template GetNN<NT_VERTEX>();
    auto NE = mesh.template GetNN<NT_EDGE>();
    mesh.CumulateData();
    auto vdata = get<0>(mesh.Data())->Data();
    auto edata = get<1>(mesh.Data())->Data();
    double tol = 1e-8;
    Array<STABEW<D>> vstr(NV); vstr = 0;
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
  	vstr[edge.v[0]] += edata[edge.id];
  	vstr[edge.v[1]] += edata[edge.id];
      }, true);
    mesh.template AllreduceNodalData<NT_VERTEX>(vstr, [](auto & in) { return sum_table(in); }, false);
    Array<double> ecw(NE);
    Matrix<double> emat(dofpv(D), dofpv(D)), evecs(dofpv(D), dofpv(D));
    Vector<double> evals(dofpv(D));
    Mat<disppv(D), disppv(D)> ddm;
    Mat<rotpv(D), disppv(D)> rdm;
    Mat<rotpv(D), rotpv(D)> rrm;
    Matrix<double> ddemat(disppv(D), disppv(D)), ddevecs(disppv(D), disppv(D)),
      rremat(rotpv(D), rotpv(D)), rrevecs(rotpv(D), rotpv(D));
    Vector<double> ddevals(disppv(D)), rrevals(rotpv(D));
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	cout << "---------------------------" << endl << "ECW FOR EDGE " << edge << endl;

	double mev_w = 1e42, mev_b = 1e42;
	Iterate<2> ([&](auto i) {
	    cout << "VERTEX " << i.value << endl;
	    auto & vmat = vstr[edge.v[i.value]];
	    GetTMBlock<0,0>(ddm, vmat);
	    cout << "full V mat : " << endl << vmat << endl;
	    ddemat = ddm; LapackEigenValuesSymmetric(ddemat, ddevals, ddevecs);
	    cout << "dd-mat: " << endl << ddemat << endl;
	    cout << "dd evals: "; prow2(ddevals); cout << endl;
	    for (auto v : ddevals) if (v != 0.0) { mev_w = min2(mev_w, v); break; }
	    GetTMBlock<disppv(D),0>(rdm, vmat);
	    cout << "rd-mat: " << endl; print_tm(cout, rdm); cout << endl;
	    GetTMBlock<disppv(D),disppv(D)>(rrm, vmat);
	    cout << "rr-mat: " << endl; print_tm(cout, rrm); cout << endl;
	    CalcPseudoInv(ddm);
	    rrm -= rdm * ddm * Trans(rdm);
	    rremat = rrm; LapackEigenValuesSymmetric(rremat, rrevals, rrevecs);
	    cout << "rr-SC: " << endl << rremat << endl;
	    cout << "rr evals: "; prow2(rrevals); cout << endl;
	    for (auto v : rrevals) if (v != 0.0) { mev_b = min2(mev_b, v); break; }
	  });

	auto & emat = edata[edge.id];
	double edge_mev_w = 1e42, edge_mev_b = 1e42;
	GetTMBlock<0,0>(ddm, emat);
	ddemat = ddm; LapackEigenValuesSymmetric(ddemat, ddevals, ddevecs);
	cout << "edge dd-mat: " << endl << ddemat << endl;
	cout << "edge dd evals: "; prow2(ddevals); cout << endl;
	for (auto v : ddevals) if (v != 0.0) { edge_mev_w = min2(edge_mev_w, v); break; }
	GetTMBlock<disppv(D),0>(rdm, emat);
	cout << "edge rd-mat: " << endl; print_tm(cout, rdm); cout << endl;
	GetTMBlock<disppv(D),disppv(D)>(rrm, emat);
	cout << "edge rr-mat: " << endl; print_tm(cout, rrm); cout << endl;
	CalcPseudoInv(ddm);
	rrm -= rdm * ddm * Trans(rdm);
	rremat = rrm; LapackEigenValuesSymmetric(rremat, rrevals, rrevecs);
	cout << "edge rr-SC: " << endl << rremat << endl;
	cout << "edge rr evals: "; prow2(rrevals); cout << endl;
	for (auto v : rrevals) if (v != 0.0) { edge_mev_b = min2(edge_mev_b, v); break; }

	cout << "min_w, min_b: " << mev_w << " " << mev_b << endl;
	cout << "edge_mev_w/b: " << edge_mev_w << " " << edge_mev_b << endl;
	
	cout << "min of " << edge_mev_w/mev_w << " " << (mev_b*edge_mev_b==0) << " " << 1 << " " << mev_b/edge_mev_b << endl;
	ecw[edge.id] = min2(edge_mev_w/mev_w, ( (mev_b*edge_mev_b==0) ? 1 : edge_mev_b/mev_b));
	
	// emat = vstr[edge.v[0]];
	// cout << "mat v0 : " << endl; print_tm(cout, vstr[edge.v[0]]); cout << endl;
	// LapackEigenValuesSymmetric(emat, evals, evecs);
	// cout << "evals "; prow2(evals); cout << endl;
	// cout << "evecs " << endl << evecs << endl;
	// double tr0 = calc_trace(vstr[edge.v[0]])/dofpv(D);
	// double mev0 = 0; for (auto v : evals) if (v!=0.0) { mev0 = v; break; }
	// emat = vstr[edge.v[1]];
	// cout << "mat v1 : " << endl; print_tm(cout, vstr[edge.v[1]]); cout << endl;
	// LapackEigenValuesSymmetric(emat, evals, evecs);
	// cout << "evals "; prow2(evals); cout << endl;
	// cout << "evecs " << endl << evecs << endl;
	// double tr1 = calc_trace(vstr[edge.v[1]])/dofpv(D);
	// double mev1 = 0; for (auto v : evals) if (v!=0.0) { mev1 = v; break; }
	// emat = edata[edge.id];
	// cout << "mat edge : " << endl; print_tm(cout, edata[edge.id]); cout << endl;
	// LapackEigenValuesSymmetric(emat, evals, evecs);
	// cout << "evals "; prow2(evals); cout << endl;
	// cout << "evecs " << endl << evecs << endl;
	// double minev = 0; for(auto v : evals) if (v != 0.0) { minev = v; break; }
	// cout << "trces, minev: " << tr0 << " " << tr1 << " " << minev;
	// ecw[edge.id] = minev / min2(tr0, tr1);
	// ecw[edge.id] = minev / min2(mev0, mev1);

	cout << ", -> ecw " << ecw[edge.id] << endl;
      }, false);
    opts->ecw = move(ecw);
    opts->min_ecw = options->min_ecw;
    opts->vcw = Array<double>(NV); opts->vcw = 0;
    opts->min_vcw = options->min_vcw;
  }
    
  
  INLINE Timer & timer_hack_Hack_BuildAlgMesh () { static Timer t("ElasticityAMG::BuildAlgMesh"); return t; }
  template<class C, class D, class E> shared_ptr<typename C::TMESH>
  EmbedVAMG<C, D, E> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    Timer & t(timer_hack_Hack_BuildAlgMesh()); RegionTimer rt(t);
    auto a = new ElVData(Array<PosWV>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto b = new ElEData<C::DIM>(Array<STABEW<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);
    FlatArray<Vec<3,double>> vp = node_pos[NT_VERTEX];
    auto pwv = a->Data();
    for (auto k : Range(pwv.Size())) {
      pwv[k].wt = 0.0;
      pwv[k].pos = vp[k];
    }
    auto we = b->Data();
    if (options->energy == "TRIV") {
      for (auto & x : we) { SetIdentity(x); }
      b->SetParallelStatus(CUMULATED);
    }
    else if (options->energy == "ELMAT") {
      FlatArray<int> vsort = node_sort[NT_VERTEX];
      Array<int> rvsort(vsort.Size());
      for (auto k : Range(vsort.Size()))
	rvsort[vsort[k]] = k;
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      for (auto & e : edges) {
	we[e.id] = (*ht_edge)[INT<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
      }
    }
    else if (options->energy == "ALG")
      {
	if ( (options->block_s.Size() != 1) || (options->block_s[0] != disppv(C::DIM)) )
	  throw Exception("ALG WTS for elasticity only mit multidim+disp only (rest TODO).");
	shared_ptr<BaseMatrix> fseqmat = finest_mat;
	if (auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
	  fseqmat = fpm->GetMatrix();
	auto fspm = dynamic_pointer_cast<SparseMatrixTM<Mat<disppv(C::DIM), disppv(C::DIM), double>>>(fseqmat);
	const auto & cspm(*fspm);
	FlatArray<int> vsort = node_sort[NT_VERTEX];
	Array<int> rvsort(vsort.Size());
	for (auto k : Range(vsort.Size()))
	  rvsort[vsort[k]] = k;
	auto edges = top_mesh->GetNodes<NT_EDGE>();
	for (auto & e : edges) {
	  double fc = fabsum(cspm(rvsort[e.v[0]], rvsort[e.v[1]]));
	  SetScalIdentity(disppv(C::DIM) * fc / dofpv(C::DIM), we[e.id]);
	}
      }
    else
      { throw Exception(string("Invalid energy type ")+options->energy); }
    auto mesh = make_shared<ElasticityMesh<C::DIM>>(move(*top_mesh), a, b);
    return mesh;
  }

  template<class C, class D, class E> shared_ptr<BaseDOFMapStep>
  EmbedVAMG<C, D, E> :: BuildEmbedding ()
  {
    static Timer t(this->name+string("::BuildEmbedding")); RegionTimer rt(t);
    auto fpardofs = finest_mat->GetParallelDofs();
    auto & vsort = node_sort[NT_VERTEX];
    if (options->v_dofs == "NODAL") {
      if (options->block_s.Size() == 1 ) { // ndof/vertex != #kernel vecs
	if (options->block_s[0] != disppv(C::DIM)) {
	  // there is really only multidim=dofpv(D) and multidim=disppv(D) that make sense here...
	  throw Exception("This should not happen ... !");
	}
	using TESM = Mat<disppv(C::DIM), dofpv(C::DIM)>;
	options->regularize = true;
	auto pmap = make_shared<ProlMap<SparseMatrix<TESM>>> (fpardofs, nullptr);
	pmap->SetProl(BuildPermutationMatrix<TESM>(vsort));
	return pmap;
      }
      else if (options->block_s.Size() > 1) {
	using TESM = Mat<1, dofpv(C::DIM)>;
	// NOT CORRECT!! just for template insantiation so we get compile time checks
	auto pmap = make_shared<ProlMap<stripped_spm<TESM>>> (fpardofs, nullptr);
	pmap->SetProl(BuildPermutationMatrix<TESM>(vsort));
	throw Exception("Compound FES embedding not implemented, sorry!");
	return nullptr;
      }
      else { // ndof/vertex == #kernel vecs (so we have rotational DOFs)
	bool need_mat = false;
	for (int k : Range(vsort.Size()))
	  if (vsort[k]!=k) { need_mat = true; break; }
	if (need_mat == false) return nullptr;
	auto pmap = make_shared<ProlMap<SparseMatrix<typename C::TMAT>>>(fpardofs, nullptr);
	pmap->SetProl(BuildPermutationMatrix<typename C::TMAT>(vsort));
	return pmap;
      }
    }
    else {
      throw Exception("variable dofs embedding not implemented, sorry!");
      return nullptr;
    }
  }

  template<class C, class X, class Y> void EmbedVAMG<C, X, Y> ::
  AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
		    ElementId ei, LocalHeap & lh)
  {
    if (options->energy != "ELMAT") return;
    static Timer t(string("EmbedVAMG<ElasticityAMG<")+to_string(C::DIM)+">::AddElementMatrix");
    RegionTimer rt(t);
    constexpr int D = C::DIM;
    const bool vmajor = (options->block_s.Size()==1) ? 1 : 0;
    int ndof_vec = dnums.Size();
    int ndof = elmat.Height();
    int perv = 0;
    if(options->block_s.Size()==0) perv = dofpv(D);
    else for (auto v : options->block_s) perv += v; 
    // cout << perv << " " << disppv(D) << endl;
    if (perv==disppv(D)) {
      const int nv = ndof / perv;
      auto & vpos(node_pos[NT_VERTEX]);
      auto get_dof = [perv](auto v, auto comp) {
	return perv*v + comp;
      };
      cout << "dnums: "; prow(dnums); cout << endl;
      cout << "elmat: " << endl << elmat << endl;
      // cout << "disppv " << disppv(D) << endl;
      // cout << "rotpv " << rotpv(D) << endl;
      // cout << "perv " << perv << endl;
      // cout << "ndof " << ndof_vec << " " << ndof << endl;
      // cout << "nv " << nv << endl;
      int ext_ndof = ndof+disppv(D);
      FlatMatrix<double> extmat (ext_ndof, ext_ndof, lh);
      extmat.Rows(0, ndof).Cols(0,ndof) = elmat;
      extmat.Rows(ndof, ext_ndof) = 0; extmat.Cols(ndof, ext_ndof) = 0;
      // {u} = 0
      for (int i : Range(disppv(D)))
	for (int j : Range(nv))
	  extmat(ndof+i, get_dof(j, i)) = extmat(get_dof(j, i), ndof+i) = 1.0;
      Vec<3,double> mid = 0;
      for (int i : Range(nv)) mid += vpos[dnums[i]]; mid /= nv;
      FlatMatrixFixWidth<rotpv(D), double> rots(ext_ndof, lh); rots = 0;
      for (int i : Range(nv)) {
	Vec<3,double> tang = vpos[dnums[i]] - mid;
	if constexpr(D==3) {
	    // (0,-z,y)
	    rots(get_dof(i,1),0) = -tang(2);
	    rots(get_dof(i,2),0) = tang(1);
	    // (-z,0,x)
	    rots(get_dof(i,0),1) = -tang(2);
	    rots(get_dof(i,2),1) = tang(0);
	    // (y,-x,0)
	    rots(get_dof(i,0),2) = tang(1);
	    rots(get_dof(i,1),2) = -tang(0);
	  }
	else {
	  // (y, -x)
	  rots(get_dof(i,0),0) = tang(1);
	  rots(get_dof(i,1),0) = -tang(0);
	}
      }
      // cout << "extmat norots " << endl << extmat << endl;
      extmat += Trans(rots) * rots;
      // cout << "extmat with rots " << endl << extmat << endl;
      // FlatMatrix<double> evecs(ndof, ndof, lh);
      // FlatVector<double> evals(ndof, lh);
      // LapackEigenValuesSymmetric(elmat, evals, evecs);
      // FlatMatrix<double> evecs2(ext_ndof, ext_ndof, lh);
      // FlatVector<double> evals2(ext_ndof, lh);
      // LapackEigenValuesSymmetric(extmat, evals2, evecs2);
      // cout << "elmat evecs: " << endl << evecs << endl;
      // cout << "extmat evecs: " << endl << evecs2 << endl;
      // cout << "elmat evals: " << endl; prow2(evals); cout << endl;
      // cout << "extmat evals: " << endl; prow2(evals2); cout << endl;
      CalcInverse(extmat);
      // cout << "inv extmat: " << endl << extmat << endl;
      // LapackEigenValuesSymmetric(extmat, evals2, evecs2);
      // cout << "inv extmat evecs: " << endl << evecs2 << endl;
      // cout << "inv extmat evals: " << endl; prow2(evals2); cout << endl;
      for (int i = 0; i < nv; i++) {
	for (int j = i+1; j < nv; j++) {
	  constexpr int small_nd = 2*disppv(D)+disppv(D);
	  Mat<small_nd, small_nd> schur;
	  Array<int> inds (small_nd);
	  for (auto k : Range(disppv(D))) {
	    inds[k] = get_dof(i,k);
	    inds[disppv(D)+k] = get_dof(j,k);
	    inds[2*disppv(D)+k] = ndof+k;
	  }
	  schur = extmat.Rows(inds).Cols(inds);
	  cout << "i/j " << i << " " << j << endl;
	  cout << "di/dj " << dnums[i] << " " << dnums[j] << endl;
	  FlatMatrix<double> sm(small_nd, small_nd, lh), evecs(small_nd, small_nd, lh);
	  FlatVector<double> evals(small_nd, lh);
	  sm = extmat.Rows(inds).Cols(inds);
	  LapackEigenValuesSymmetric(sm, evals, evecs);
	  cout << "small mat evals: " << endl; prow2(evals); cout << endl;
	  cout << "small mat evercs: " << endl; cout << evecs; cout << endl;
	  cout << "schur block: " << endl; print_tm(cout, schur); cout << endl;

	  CalcInverse(schur);

	  cout << "schur block inved: " << endl; print_tm(cout, schur); cout << schur << endl;
	  sm = schur;
	  LapackEigenValuesSymmetric(sm, evals, evecs);
	  cout << "inv small mat evals: " << endl; prow2(evals); cout << endl;
	  cout << "inv small mat evercs: " << endl; cout << evecs; cout << endl;
	  auto & hte(*ht_edge);
	  STABEW<D> & elew = hte[INT<2,int>(dnums[i], dnums[j]).Sort()];
	  cout << "add to " << INT<2,int>(dnums[i], dnums[j]).Sort() << endl;
	  double lam = 0; for (auto k : Range(disppv(D))) lam += schur(k,k);
	  lam /= disppv(D);
	  Vec<3,double> tang = vpos[dnums[i]] - vpos[dnums[j]];
	  Iterate<disppv(D)>([&](auto i) {
	      Iterate<disppv(D)>([&](auto j) {
		  // elew(i.value, j.value) += lam * lam * tang(i.value) * tang(j.value); // ??why lam**2??
		  elew(i.value, j.value) += lam * tang(i.value) * tang(j.value);
		});
	    });
	  Iterate<rotpv(D)>([&](auto i) {
	      elew(disppv(D)+i.value, disppv(D)+i.value) = 1e-14;
	    });
	  cout << "ELEW: " << endl << elew << endl;
	}
      }
    }
    else {
      throw Exception("sorry, only disp X disp elmats implemented");
    }
  }


  template<> template<class TMESH>
  void ElEData<2> :: map_data (const CoarseMap<TMESH> & cmap, ElEData<2> & celed) const
  {
    static_assert(std::is_same<TMESH,ElasticityMesh<2>>::value==1, "ElEData with wrong map!");
    constexpr int D = 2;
    auto & mesh = static_cast<ElasticityMesh<D>&>(*this->mesh);
    // mesh.CumulateData(); // TODO: should not be necessary
    auto sp_eqc_h = mesh.GetEQCHierarchy();
    const auto & eqc_h = *sp_eqc_h;
    auto neqcs = mesh.GetNEqcs();
    const size_t NE = mesh.template GetNN<NT_EDGE>();
    ElasticityMesh<D> & cmesh = static_cast<ElasticityMesh<D>&>(*celed.mesh);
    const size_t NCE = cmap.template GetMappedNN<NT_EDGE>();
    celed.data.SetSize(NCE); celed.data = 0.0;
    auto e_map = cmap.template GetMap<NT_EDGE>();
    cout << "e_map: " << endl; prow2(e_map); cout << endl << endl;
    auto v_map = cmap.template GetMap<NT_VERTEX>();
    cout << "v_map: " << endl; prow2(v_map); cout << endl << endl;
    auto pecon = mesh.GetEdgeCM();
    cout << "fine mesh ECM: " << endl << *pecon << endl;
    const auto & econ(*pecon);
    auto fvd = get<0>(mesh.Data())->Data();
    auto cvd = get<0>(cmesh.Data())->Data();
    auto fed = Data();
    auto ced = celed.Data();
    auto edges = mesh.template GetNodes<NT_EDGE>();
    Table<int> c2fe;
    {
      TableCreator<int> cc2fe(NCE);
      for (; !cc2fe.Done(); cc2fe++)
	for (auto k : Range(NE))
	  if (is_valid(e_map[k]))
	    cc2fe.Add(e_map[k], k);
      c2fe = cc2fe.MoveTable();
    }

    Matrix<double> mat(dofpv(D), dofpv(D)), evecs(dofpv(D), dofpv(D));
    Vector<double> evals(dofpv(D));
    auto check_evals = [&](auto & M) {
      mat = M;
      LapackEigenValuesSymmetric(mat, evals, evecs);
      cout << "evals: " << endl; prow2(evals); cout << endl;
      cout << "evecs: " << endl << evecs << endl;
    };

    Matrix<double> mat_disp(disppv(D), disppv(D)), evecs_disp(disppv(D), disppv(D));
    Vector<double> evals_disp(disppv(D));
    auto check_evals_disp = [&](auto & M) {
      mat_disp = M;
      LapackEigenValuesSymmetric(mat_disp, evals_disp, evecs_disp);
      cout << "evals: " << endl; prow2(evals_disp); cout << endl;
      cout << "evecs: " << endl << evecs_disp << endl;
    };

    Matrix<double> mat_rot(rotpv(D), rotpv(D)), evecs_rot(rotpv(D), rotpv(D));
    Vector<double> evals_rot(rotpv(D));
    auto check_evals_rot = [&](auto & M) {
      mat_rot = M;
      LapackEigenValuesSymmetric(mat_rot, evals_rot, evecs_rot);
      cout << "evals: " << endl; prow2(evals_rot); cout << endl;
      cout << "evecs: " << endl << evecs_rot << endl;
    };

    Mat<dofpv(D), dofpv(D)> T, TTM;
    SetIdentity(T);
    
    Mat<disppv(D), disppv(D), double> W;
    Mat<disppv(D), rotpv(D), double> sktcf0, sktcf1, sktcc;
    Mat<rotpv(D), rotpv(D), double> B, adbm;
    auto calc_trafo = [](auto & T, const auto & tAi, const auto & tBj) {
      Vec<3, double> tang = 0.5 * (tAi + tBj);
      if constexpr(D==3) {
	  T(0,4) = -(T(1,3) =  tang[2]);
	  T(0,5) = -(T(2,3) = -tang[1]);
	  T(1,5) = -(T(2,4) =  tang[0]);
	}
      else {
	T(0,2) = -tang[1];
	T(1,2) =  tang[0];
      }
    };
    auto calc_cemat = [&](const auto & cedge, auto & cemat, auto use_fenr) {
      cout << "calc cemat!" << endl;
      cout << "cedge: " << cedge << endl;
      auto fenrs = c2fe[cedge.id];
      cemat = 0;
      cout << "fenrs: "; prow2(fenrs); cout << endl;
      // Vec<3> cc_tang = cvd[cedge.v[0]].pos - cvd[cedge.v[1]].pos;
      // calc_skt(cc_tang, sktcc);
      for (auto fenr : fenrs) {
	if (use_fenr(fenr)) {
	  const auto & fedge = edges[fenr];
	  int l = (v_map[fedge.v[0]] == cedge.v[0]) ? 0 : 1;
	  /**
	     I | sk(tcf0+tcf1)
	     0 | I
	   **/
	  Vec<3> tcf0 =  (fvd[fedge.v[0]].pos - cvd[cedge.v[l]].pos);
	  Vec<3> tcf1 =  (fvd[fedge.v[1]].pos - cvd[cedge.v[1-l]].pos);
	  calc_trafo(T, tcf0, tcf1);
	  cout << "fedge " << fedge << endl;
	  cout << " vs map: " << v_map[fedge.v[0]] << " " << v_map[fedge.v[1]] << endl;
	  cout << " l is " << l << endl;
	  cout << "cf_tang0: " << tcf0 << endl;
	  cout << "cf_tang1: " << tcf1 << endl;
	  auto & FM = fed[fedge.id];
	  cout << "FM: " << endl; print_tm(cout, FM); cout << endl;
	  cout << "trafo: " << endl; print_tm(cout, T); cout << endl;
	  TTM = Trans(T) * FM;
	  cemat += TTM * T;
	  cout << "intermed. cemat: " << endl; print_tm(cout, cemat); cout << endl;
	}
      }
      cout << "check evals of cemat: " << endl;
      check_evals(cemat);
    };
    typedef Mat<dofpv(D), dofpv(D), double> TM;
    auto calc_cedata = [&](const auto & edge, TM & cfullmat) {
      ced[edge.id] = cfullmat;
    };

    /** ex-edge matrices:  calc & reduce full cmats **/
    Table<TM> cemats;
    Array<int> perow(neqcs);
    if (neqcs>1) {
      perow[0] = 0;
      for (auto k : Range(neqcs))
	perow[k] = cmesh.template GetENN<NT_EDGE>(k) + cmesh.template GetCNN<NT_EDGE>(k);
      Table<TM> tcemats(perow); perow = 0;
      const PARALLEL_STATUS mesh_stat = GetParallelStatus();
      cmesh.ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  calc_cemat(cedge, tcemats[eqc][perow[eqc]++],
		     [&](auto fenr) { return mesh_stat==DISTRIBUTED || eqc_h.IsMasterOfEQC(eqc); } );
	}, false); // def, false!
      cemats = ReduceTable<TM,TM> (tcemats, sp_eqc_h, [&](auto & tab) { return sum_table(tab); });
      /** ex-edge matrices:  split to bend + wigg-mats **/
      perow = 0;
      cmesh.ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  calc_cedata(cedge, cemats[eqc][perow[eqc]++]);
	}, false); // def, false!
    }
    
    /** loc-edge matrices:  all in one **/
    TM cmat;
    cmesh.ApplyEQ<NT_EDGE>(Range(min(neqcs, size_t(1))), [&](auto eqc, const auto & cedge){
	calc_cemat(cedge, cmat, [&](auto fenr) { return true; });
	calc_cedata(cedge, cmat);
      }, false); // def, false!
    
    
    celed.SetParallelStatus(DISTRIBUTED);
  }

  
  template class ElEData<2>;
  // template class ElEData<3>;
  template void ElEData<2>::map_data (const CoarseMap<ElasticityMesh<2>> & cmap, ElEData<2> & cdata) const;


} // namespace amg

#include "amg_tcs.hpp"
#endif // ELASTICITY
