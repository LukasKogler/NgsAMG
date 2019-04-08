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
      if constexpr(D==3) {
	  if (options->regularize)
	    { cout << "DOF DOF REG " << endl; return make_shared<StabHGSS<dofpv(D), disppv(D), dofpv(D)>> (spmat, par_dofs, free_dofs); }
	}
      cout << "DOF DOF NO REG " << endl;
      return make_shared<HybridGSS<dofpv(D)>> (spmat, par_dofs, free_dofs);
    }
    else if (shared_ptr<const SparseMatrix<Mat<disppv(D), disppv(D), double>>> spmat = dynamic_pointer_cast<SparseMatrix<Mat<disppv(D), disppv(D), double>>> (mat)) {
      cout << "DISP x DISP SMOOTHER" << endl;
      return make_shared<HybridGSS<disppv(D)>> (spmat, par_dofs, free_dofs);
    }
    else if (shared_ptr<const SparseMatrix<double>> spmat = dynamic_pointer_cast<SparseMatrix<double>> (mat)) {
      cout << "1 x 1 SMOOTHER" << endl;
      return make_shared<HybridGSS<1>> (spmat, par_dofs, free_dofs);
    }
    throw Exception(string("Could not build a Smoother for mat-type ") + string(typeid(*mat).name()));
    return nullptr;
  }

  INLINE Timer & timer_hack_Hack_BuildAlgMesh () { static Timer t("ElasticityAMG::BuildAlgMesh"); return t; }
  template<class C, class D, class E> shared_ptr<typename C::TMESH>
  EmbedVAMG<C, D, E> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    Timer & t(timer_hack_Hack_BuildAlgMesh()); RegionTimer rt(t);
    auto a = new ElVData(Array<PosWV>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto b = new ElEData<C::DIM>(Array<ElEW<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);
    FlatArray<Vec<3,double>> vp = node_pos[NT_VERTEX];
    auto pwv = a->Data();
    for (auto k : Range(pwv.Size())) {
      pwv[k].wt = 0.0;
      pwv[k].pos = vp[k];
    }
    auto we = b->Data();
    if (options->energy == "TRIV") {
      for (auto & x : we) { SetIdentity(x.bend_mat()); SetIdentity(x.wigg_mat()); }
      b->SetParallelStatus(CUMULATED);
    }
    else if (options->energy == "ELMAT") {
      FlatArray<int> vsort = node_sort[NT_VERTEX];
      Array<int> rvsort(vsort.Size());
      for (auto k : Range(vsort.Size()))
	rvsort[vsort[k]] = k;
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      for (auto & e : edges) {
	auto ed = (*ht_edge)[INT<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
	we[e.id] = (*ht_edge)[INT<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
	// workaround for embedding
	SetScalIdentity(calc_trace(we[e.id].wigg_mat())/disppv(C::DIM), we[e.id].bend_mat());
	// cout << "edge " << e << endl;
	// cout << we[e.id].bend_mat() << endl;
	// cout << we[e.id].wigg_mat() << endl;
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
	  double fc = fabs(calc_trace(cspm(rvsort[e.v[0]], rvsort[e.v[1]])));
	  SetScalIdentity(fc, we[e.id].bend_mat());
	  SetScalIdentity(fc, we[e.id].wigg_mat());
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
      // cout << "dnums: "; prow(dnums); cout << endl;
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
      FlatMatrixFixWidth<dofpv(D), double> rots(ext_ndof, lh); rots = 0;
      for (int i : Range(nv)) {
	Vec<3,double> tang = vpos[dnums[i]] - mid;
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
      extmat += Trans(rots) * rots;
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
      // CalcInverse(extmat);
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
	  // FlatMatrix<double> sm(small_nd, small_nd, lh), evecs(small_nd, small_nd, lh);
	  // FlatVector<double> evals(small_nd, lh);
	  // sm = extmat.Rows(inds).Cols(inds);
	  // LapackEigenValuesSymmetric(sm, evals, evecs);
	  // cout << "small mat evals: " << endl; prow2(evals); cout << endl;
	  // cout << "small mat evercs: " << endl; cout << evecs; cout << endl;
	  // cout << "schur block: " << endl; print_tm(cout, schur); cout << endl;
	  // CalcInverse(schur);
	  // cout << "schur block inved: " << endl; print_tm(cout, schur); cout << schur << endl;
	  // sm = schur;
	  // LapackEigenValuesSymmetric(sm, evals, evecs);
	  // cout << "inv small mat evals: " << endl; prow2(evals); cout << endl;
	  // cout << "inv small mat evercs: " << endl; cout << evecs; cout << endl;
	  auto & hte(*ht_edge);
	  ElEW<D> & elew = hte[INT<2,int>(dnums[i], dnums[j]).Sort()];
	  // cout << "add to " << INT<2,int>(dnums[i], dnums[j]).Sort() << endl;
	  auto wigg_mat = elew.wigg_mat(); wigg_mat = 0;
	  for (auto k : Range(disppv(D))) {
	    for (auto j : Range(disppv(D))) {
	      wigg_mat(k,j) = schur(k,j);
	    }
	  }
	  // cout << "wigg_mat: " << endl << wigg_mat << endl;
	}
      }
    }
    else {
      throw Exception("sorry, only disp X disp elmats implemented");
    }
  }

  template <int D>
  PARALLEL_STATUS ElEData<D> :: map_data (const BaseCoarseMap & acmap, Array<ElEW<D>> & cdata) const
  {
    auto cmap = static_cast<const CoarseMap<ElasticityMesh<D>>&>(acmap);
    auto & mesh = static_cast<ElasticityMesh<D>&>(*cmap.GetMesh());

    cdata.SetSize(cmap.template GetMappedNN<NT_EDGE>()); cdata = 0.0;
    auto map = cmap.template GetMap<NT_EDGE>();
    auto econ = mesh.GetEdgeCM();
    auto edges = mesh.template GetNodes<NT_EDGE>();
    auto v_map = cmap.template GetMap<NT_VERTEX>();
    auto vd = get<0>(mesh.Data())->Data();
    Array<int> common_ovs;
    Mat<disppv(D), rotpv(D), double> skt, sktt;
    auto calc_skt = [&](const auto & t) {
      skt = 0;
      if constexpr(D==3) {
	  skt(0,1) = -(skt(1,0) =  t[2]);
	  skt(0,2) = -(skt(2,0) = -t[1]);
	  skt(1,2) = -(skt(2,1) =  t[0]);
	}
      else {
	skt(0,0) =  t[1];
	skt(1,0) = -t[0];
      }
      sktt = Trans(skt);
    };
    auto calc_admat = [&](const auto & edge) {
      Mat<rotpv(D), rotpv(D)> mat(0);
      auto fdm = data[edge.id].wigg_mat();
      mat = 0.25 * sktt * fdm * skt;
      // cout << "ad-mat: " << endl; print_tm(cout, mat); cout << endl;
      return mat;
    };
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	auto CE = map[edge.id];
	if (CE != -1) {
	  cdata[CE] += data[edge.id];
	}
	else if ( (v_map[edge.v[0]]==v_map[edge.v[1]]) && (v_map[edge.v[0]]!=-1) ) {
	  Vec<3> tang = vd[edge.v[0]].pos - vd[edge.v[1]].pos;
	  auto ri0 = econ->GetRowIndices(edge.v[0]);
	  auto ri1 = econ->GetRowIndices(edge.v[1]);
	  calc_skt(tang);
	  intersect_sorted_arrays(ri0, ri1, common_ovs);
	  for (auto ov : common_ovs) {
	    auto conn_edge_id = (*econ)(edge.v[0], ov);
	    if (map[conn_edge_id] == -1 ) continue;
	    auto ce_id = map[conn_edge_id];
	    auto& con_edge = edges[(*econ)(edge.v[0], ov)];
	    auto& con_edge2 = edges[(*econ)(edge.v[1], ov)];
	    auto add_here = cdata[ce_id].bend_mat();
	    add_here += calc_admat(con_edge);
	    add_here += calc_admat(con_edge2);
	  }
	}
      }, stat==CUMULATED);
    return DISTRIBUTED;
  }

  template class ElEData<2>;
  template class ElEData<3>;


} // namespace amg

#include "amg_tcs.hpp"
#endif // ELASTICITY
