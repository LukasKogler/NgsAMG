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
    throw Exception("Could not build a Smoother, WTF ???");
    return nullptr;
  }


  template<> shared_ptr<ElasticityAMG<3>::TSPMAT>
  ElasticityAMG<3> :: RegularizeMatrix (shared_ptr<ElasticityAMG<3>::TSPMAT> mat, shared_ptr<ParallelDofs> & pardofs)
  {
    // cerr << "reg. mat " << mat->Height() << " x " << mat->Width() << endl;
    // print_tm_spmat(cerr, *mat);
    // cerr << "reg diags: " << endl;
    // pardofs->GetCommunicator().Barrier();
    Matrix<double> M(6,6), evecs(6,6);
    Vector<double> evals(6);
    for(auto k : Range(mat->Height())) {
      // cerr << " k " << k << endl;
      auto& diag = (*mat)(k,k);

      // M = diag;
      // LapackEigenValuesSymmetric(M, evals, evecs);
      // cerr << "diag evals: " << endl; cerr << evals << endl;

      // print_tm(cerr, diag); cerr << endl;
      RegTM<3,3,6>(diag);
      // cerr << " reged " << endl;
      // print_tm(cerr, diag); cerr << endl;

      // M = diag;
      // LapackEigenValuesSymmetric(M, evals, evecs);
      // cerr << "reg diag evals: " << endl; cerr << evals << endl;
      
    }
    // cerr << "all ok " << endl;
    // pardofs->GetCommunicator().Barrier();
    return mat;
  }

  template<> shared_ptr<ElasticityAMG<2>::TSPMAT>
  ElasticityAMG<2> :: RegularizeMatrix (shared_ptr<ElasticityAMG<2>::TSPMAT> mat, shared_ptr<ParallelDofs> & pardofs)
  {
    for(auto k : Range(mat->Height())) {
      auto& diag = (*mat)(k,k);
      if (diag(2,2) == 0.0) {
	diag(2,2) = 1.0;
      }
    }
    return mat;
  }


  template<int N> INLINE double CalcMaxEV (const Matrix<double> & A, const Matrix<double> & B)
  {
    static Vector<double> v(N), v2(N), vn(N);
    Iterate<N>([&](auto i){ v[i] = double (rand()) / RAND_MAX; } );
    auto IP = [&](const auto & a, const auto & b) {
      vn = B * a;
      return InnerProduct(vn, b);
    };
    double ipv0 = IP(v,v);
    for (auto k : Range(3)) {
      v2 = A * v;
      v2 /= sqrt(IP(v2,v2));
      v = A * v2;
    }
    return sqrt(IP(v,v)/IP(v2,v2));
  }

  /**
     returns (approx) inf_x <Ax,x>/<Bx,x> = 1/sup_x <A_inv Bx,x>/<Ax,x>
   **/
  template<int N> INLINE double CalcMinGenEV (const Matrix<double> & A, const Matrix<double> & B)
  {
    static Timer t("CalcMinGenEV");
    RegionTimer rt(t);
    static Matrix<double> Ar(N,N), Ar_inv(N,N), Br(N,N), C(N,N);
    double trace = 0; Iterate<N>([&](auto i) { trace += A(N*i.value+i.value); });
    if (trace == 0.0) trace = 1e-2;
    double eps = max2(1e-3 * trace, 1e-10);
    Ar = A; Iterate<N>([&](auto i) { Ar(N*i.value+i.value) += eps; });
    Br = B; Iterate<N>([&](auto i) { Br(N*i.value+i.value) += eps; });
    // cerr << "B is " << endl << B << endl;
    // cerr << "A is " << endl << A << endl;
    // cerr << "calc inv for Ar " << endl << Ar << endl;
    {
      static Timer t("CalcMinGenEV - inv");
      RegionTimer rt(t);
      CalcInverse(Ar, Ar_inv);
    }
    {
      static Timer t("CalcMinGenEV - mult");
      RegionTimer rt(t);
      C = Ar_inv * Br;
    }
    double maxev = 0;
    {
      static Timer t("CalcMinGenEV - maxev");
      RegionTimer rt(t);
      maxev = CalcMaxEV<N>(C, Ar);
    }
    return 1/sqrt(maxev);
  }

  template<int N> INLINE double CalcMinGenEV2 (const Matrix<double> & A, const Matrix<double> & B)
  {
    static Timer t("CalcMinGenEV");
    RegionTimer rt(t);
    static Timer tlap("CalcMinGenEV::Lapack");
    static Matrix<double> evecs(N,N), bia(N,N), sqrt_b(N,N);
    static Vector<double> evals(N);
    // cerr << "CalcMinGenEV, A: " << endl << A << endl;
    // cerr << "CalcMinGenEV, B: " << endl << B << endl;

    // LapackEigenValuesSymmetric(A, evals, evecs);
    // cout << "A evals: " << endl; prow2(evals, cout); cout << endl;
    // cout << "A evecs: " << endl << evecs;

    // cerr << "LEV B " << endl;
    tlap.Start();
    LapackEigenValuesSymmetric(B, evals, evecs);
    tlap.Stop();
    // cerr << "B evals: " << endl; prow2(evals, cerr); cerr << endl;
    // cout << "B evecs: " << endl << evecs;
    const double tol = 1e-13;
    for (auto & v : evals) // pseudo inverse ??
      v = (v>tol) ? 1.0/sqrt(sqrt(v)) : 0;
    // cerr << "B rescaled evals: " << endl; prow2(evals, cerr); cerr << endl;
    Iterate<N>([&](auto i) {
	Iterate<N>([&](auto j) {
	    evecs(i.value,j.value) *= evals(i.value);
	  });
      });
    sqrt_b = Trans(evecs) * evecs;
    // cout << "sqrt_inv_B: " << endl << sqrt_b << endl;
    evecs = A * sqrt_b;
    bia = sqrt_b * evecs;
    // cerr << "BinvA: " << endl << bia << endl;
    // cerr << "LEV BIA " << endl;
    // cerr << bia << endl;
    tlap.Start();
    LapackEigenValuesSymmetric(bia, evals, evecs);
    tlap.Stop();
    // cout << "BinvA evals: " << endl; prow2(evals, cout); cout << endl;
    // cout << "BinvA evecs: " << endl << evecs << endl;
    double minev = 0;
    for (auto v : evals) // 0/0 := 1
      if (fabs(v) > tol) {
	minev = v;
	break;
      }
    return minev;
    // return evals(0);
  };


  template<int D> Array<double> 
  ElasticityAMG<D> :: CalcECWSimple (shared_ptr<TMESH> _mesh)
  {
    const ElasticityMesh<D> & mesh(*_mesh);
    auto NV = mesh.template GetNN<NT_VERTEX>();
    auto NE = mesh.template GetNN<NT_EDGE>();
    auto vdata = get<0>(mesh.Data())->Data();
    auto edata = get<1>(mesh.Data())->Data();
    const auto& econ(*mesh.GetEdgeCM());
    Array<double> ecw(NE);
    Array<double> vcw(NV); vcw = 0;
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);
	vcw[edge.v[0]] += tr;
	vcw[edge.v[1]] += tr;
      }, true);
    mesh.template AllreduceNodalData<NT_VERTEX>(vcw, [](auto & in) { return sum_table(in); }, false);
    mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	auto tr = calc_trace(edata[edge.id]);
	double vw = min(vcw[edge.v[0]], vcw[edge.v[1]]);
	ecw[edge.id] = tr / vw;
      }, false);
    return move(ecw);
  }
  
  template<int D> Array<double> 
  ElasticityAMG<D> :: CalcECWRobust (shared_ptr<TMESH> _mesh)
  {
    const ElasticityMesh<D> & mesh(*_mesh);
    auto NV = mesh.template GetNN<NT_VERTEX>();
    auto NE = mesh.template GetNN<NT_EDGE>();
    auto vdata = get<0>(mesh.Data())->Data();
    auto edata = get<1>(mesh.Data())->Data();
    const auto& econ(*mesh.GetEdgeCM());
    Array<double> ecw(NE);
    // TODO: only works "sequential" for now ...
    Matrix<TMAT> emat(2,2);
    Matrix<double> schur(dofpv(D), dofpv(D)), emoo(dofpv(D), dofpv(D));
    Array<TMAT> vblocks(NV); vblocks = 0;
    auto edges = mesh.template GetNodes<NT_EDGE>();
    {
      static Timer t(this->name+string("::SetCoarseningOptions - Collect")); RegionTimer rt(t);
      mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	  CalcRMBlock(mesh, edge, emat);
	  vblocks[edge.v[0]] += emat(0,0);
	  vblocks[edge.v[1]] += emat(1,1);
	}, true);
    }
    mesh.template AllreduceNodalData<NT_VERTEX, TMAT>(vblocks, [](auto & tab){ return move(sum_table(tab)); });
    {
      static Timer t(this->name+string("::SetCoarseningOptions - Calc")); RegionTimer rt(t);
      mesh.template Apply<NT_EDGE>([&](const auto & edge) {
	double cws[2] = {0,0};
  	CalcRMBlock(mesh, edge, emat);
	double tr = 0; Iterate<dofpv(D)>([&](auto i) { tr += emat(0,0)(i.value,i.value); } );
	tr /= dofpv(D);
	emat /= tr;
	// cout << "edge mat: " << endl; print_tm_mat(cout, emat); cout << endl;
	Iterate<2>([&](auto i) {
	    // cout << "edge " << edge << " i " << i.value << endl;
	    constexpr int j = 1-i;
	    emoo = vblocks[edge.v[i.value]];
	    emoo /= tr;
	    // cout << "invert emoo: " << endl << emoo << endl;
	    CalcInverse(emoo);
	    // CalcPseudoInverse<dofpv(D)>(emoo);
	    // cout << "inverted emoo: " << endl << emoo << endl;
	    schur = emat(j, j) - emat(j,i.value) * emoo * emat(i.value, j);
	    emoo = emat(j, j);
	    // cout << "j-block: " << endl << emoo << endl;
	    // cout << "schur: " << endl << schur << endl;
	    cws[i.value] = sqrt(1 - CalcMinGenEV<dofpv(D)>(schur, emoo));
	  });
	// ecw[edge.id] = sqrt(cws[0]*cws[1]);
	ecw[edge.id] = cws[0] + cws[1];
	// cout << "okj, next edge " << endl;
      }, false);
    }
    return move(ecw);
  }
    
  template<int D> void
  ElasticityAMG<D> :: SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts,
  					    INT<3> level, shared_ptr<TMESH> _mesh)
  {
    static Timer t(this->name+string("::SetCoarseningOptions")); RegionTimer rt(t);
    auto &options = static_cast<ElasticityAMG<D>::Options&>(*this->options);
    const ElasticityMesh<D> & mesh(*_mesh);
    mesh.CumulateData();
    auto NV = mesh.template GetNN<NT_VERTEX>();
    opts->min_vcw = options.min_vcw;
    opts->vcw = Array<double>(NV); opts->vcw = 0;
    opts->min_ecw = options.min_ecw;
    if (options.soc == "SIMPLE")
      { opts->ecw = CalcECWSimple(_mesh); }
    else if (options.soc == "ROBUST")
      { opts->ecw = CalcECWRobust(_mesh); }
    else
      { throw Exception("INVALID SOC!"); }
  }
  
  INLINE Timer & timer_hack_Hack_BuildAlgMesh () { static Timer t("ElasticityAMG::BuildAlgMesh"); return t; }
  template<class C, class D, class E> shared_ptr<typename C::TMESH>
  EmbedVAMG<C, D, E> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    Timer & t(timer_hack_Hack_BuildAlgMesh()); RegionTimer rt(t);
    auto a = new ElVData(Array<PosWV>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto b = new ElEData<C::DIM>(Array<STABEW<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);
    auto pwv = a->Data();
    if (options->energy == "ELMAT") {
      FlatArray<int> vsort = node_sort[NT_VERTEX];
      auto & vp = node_pos[NT_VERTEX];
      for (auto k : Range(pwv.Size())) {
	pwv[vsort[k]].wt = 0.0;
	pwv[vsort[k]].pos = vp[k];
      }
      vp.SetSize(0);
    }
    else {
      auto & vp = node_pos[NT_VERTEX];
      for (auto k : Range(pwv.Size())) {
	pwv[k].wt = 0.0;
	pwv[k].pos = vp[k];
      }
      vp.SetSize(0);
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
      // cout << "dnums: "; prow(dnums); cout << endl;
      // cout << "elmat: " << endl << elmat << endl;
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
      double trace = 0;
      for(auto k : Range(dnums.Size())) trace += elmat(k,k);
      trace = sqrt(trace/dnums.Size());
      Vec<3,double> mid = 0;
      for (int i : Range(nv)) mid += vpos[dnums[i]]; mid /= nv;
      FlatMatrixFixWidth<rotpv(D), double> rots(ext_ndof, lh); rots = 0;
      for (int i : Range(nv)) {
	Vec<3,double> tang = vpos[dnums[i]] - mid;
	tang *= trace/L2Norm(tang);
	if constexpr(D==3) {
	    // (0,-z,y)
	    rots(get_dof(i,1),0) = -tang(2);
	    rots(get_dof(i,2),0) =  tang(1);
	    // (-z,0,x)
	    rots(get_dof(i,0),1) = -tang(2);
	    rots(get_dof(i,2),1) =  tang(0);
	    // (y,-x,0)
	    rots(get_dof(i,0),2) =  tang(1);
	    rots(get_dof(i,1),2) = -tang(0);
	  }
	else {
	  // (y, -x)
	  rots(get_dof(i,0),0) =  tang(1);
	  rots(get_dof(i,1),0) = -tang(0);
	}
      }
      // cout << "extmat norots " << endl << extmat << endl;

      // FlatMatrix<double> evecs(ndof, ndof, lh);
      // FlatVector<double> evals(ndof, lh);
      // LapackEigenValuesSymmetric(elmat, evals, evecs);
      // cout << "elmat evals: " << endl; prow2(evals); cout << endl;

      // FlatMatrix<double> evecs2(ext_ndof, ext_ndof, lh);
      // FlatVector<double> evals2(ext_ndof, lh);
      // LapackEigenValuesSymmetric(extmat, evals2, evecs2);
      // cout << "extmat 1 evals: " << endl; prow2(evals2); cout << endl;

      extmat += Trans(rots) * rots;

      // LapackEigenValuesSymmetric(extmat, evals2, evecs2);
      // cout << "extmat 2 evals: " << endl; prow2(evals2); cout << endl;

      CalcInverse(extmat);
      // cerr << "inv extmat: " << endl << extmat << endl;
      // LapackEigenValuesSymmetric(extmat, evals2, evecs2);
      // cout << "inv extmat evecs: " << endl << evecs2 << endl;
      // cout << "inv extmat evals: " << endl; prow2(evals2); cout << endl;
      for (int i = 0; i < nv; i++) {
	for (int j = i+1; j < nv; j++) {
	  constexpr int small_nd = 2*disppv(D)+disppv(D);
	  Mat<small_nd, small_nd> schur;
	  Array<int> inds (small_nd);
	  Iterate<disppv(D)>([&](auto k) {
	      inds[k.value] = get_dof(i,k.value);
	      inds[disppv(D)+k.value] = get_dof(j,k.value);
	      inds[2*disppv(D)+k.value] = ndof+k.value;
	    });
	  schur = extmat.Rows(inds).Cols(inds);
	  // cout << "i/j " << i << " " << j << endl;
	  // cout << "di/dj " << dnums[i] << " " << dnums[j] << endl;

	  // FlatMatrix<double> sm(small_nd, small_nd, lh), evecs(small_nd, small_nd, lh);
	  // FlatVector<double> evals(small_nd, lh);
	  // sm = extmat.Rows(inds).Cols(inds);
	  // LapackEigenValuesSymmetric(sm, evals, evecs);
	  // cout << "small mat evals: " << endl; prow2(evals); cout << endl;
	  // cout << "small mat evercs: " << endl; cout << evecs; cout << endl;

	  // cout << "schur block: " << endl; print_tm(cout, schur); cout << endl;
	  CalcInverse(schur);
	  // cout << endl << "schur block inved: " << endl; print_tm(cout, schur); cout << endl;

	  // sm = schur;
	  // LapackEigenValuesSymmetric(sm, evals, evecs);
	  // cout << "inv small mat evals: " << endl; prow2(evals); cout << endl;
	  // cout << "inv small mat evercs: " << endl; cout << evecs; cout << endl;

	  auto & hte(*ht_edge);
	  STABEW<D> & elew = hte[INT<2,int>(dnums[i], dnums[j]).Sort()];
	  // cout << "add to " << INT<2,int>(dnums[i], dnums[j]).Sort() << endl;
	  double lam = 0; for (auto k : Range(disppv(D))) lam += schur(k,k);
	  lam /= disppv(D);
	  Vec<3,double> tang = vpos[dnums[i]] - vpos[dnums[j]];
	  // cout << "lam: " << lam << endl;
	  // cout << "ELEW bef: " << endl; print_tm(cout, elew); cout << endl;
	  Iterate<disppv(D)>([&](auto i) {
	      Iterate<disppv(D)>([&](auto j) {
		  // elew(i.value, j.value) += lam * lam * tang(i.value) * tang(j.value); // ??why lam**2??
		  elew(i.value, j.value) += lam * tang(i.value) * tang(j.value);
		});
	    });
	  // cout << "ELEW after: " << endl; print_tm(cout, elew); cout << endl;
	  Iterate<rotpv(D)>([&](auto i) {
	      elew(disppv(D)+i.value, disppv(D)+i.value) += 1e-10 * lam;
	    });
	  // if (lam<0) {
	  //   throw Exception("LAM<0!!");
	  // }
	}
      }
    }
    else {
      throw Exception("sorry, only disp X disp elmats implemented");
    }
  }


  template<int D> template<class TMESH>
  void ElEData<D> :: map_data (const CoarseMap<TMESH> & cmap, ElEData<D> & celed) const
  {
    // cerr << " map edges " << endl;
    static_assert(std::is_same<TMESH,ElasticityMesh<D>>::value==1, "ElEData with wrong map?!");
    auto & mesh = static_cast<ElasticityMesh<D>&>(*this->mesh);
    mesh.CumulateData();
    auto sp_eqc_h = mesh.GetEQCHierarchy();
    const auto & eqc_h = *sp_eqc_h;
    // eqc_h.GetCommunicator().Barrier();
    auto neqcs = mesh.GetNEqcs();
    const size_t NE = mesh.template GetNN<NT_EDGE>();
    ElasticityMesh<D> & cmesh = static_cast<ElasticityMesh<D>&>(*celed.mesh);
    const size_t NCE = cmap.template GetMappedNN<NT_EDGE>();

    // cout << " NE NCE " << NE << " " << NCE << endl;
    
    celed.data.SetSize(NCE); celed.data = 0.0;
    auto e_map = cmap.template GetMap<NT_EDGE>();
    // cout << "e_map: " << endl; prow2(e_map); cout << endl << endl;
    auto v_map = cmap.template GetMap<NT_VERTEX>();
    // cout << "v_map: " << endl; prow2(v_map); cout << endl << endl;
    auto pecon = mesh.GetEdgeCM();
    // cout << "fine mesh ECM: " << endl << *pecon << endl;
    const auto & econ(*pecon);
    // cout << "cumulate fm vd!" << endl;
    auto fvd = get<0>(mesh.Data())->Data();
    // cout << "cumulate cm vd!" << endl;
    get<0>(cmesh.Data())->Cumulate();
    // cout << "cumulates done!" << endl;
    auto cvd = get<0>(cmesh.Data())->Data();
    auto fed = this->Data();
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
    Mat<dofpv(D), dofpv(D)> T, TTM;
    T = 0; Iterate<dofpv(D)>([&](auto i) { T(i.value, i.value) = 1.0; });
    Mat<disppv(D), disppv(D), double> W;
    Mat<disppv(D), rotpv(D), double> sktcf0, sktcf1, sktcc;
    Mat<rotpv(D), rotpv(D), double> B, adbm;
    auto calc_trafo = [](auto & T, const auto & tAi, const auto & tBj) {
      Vec<3, double> tang = 0.5 * (tAi + tBj); // t is flipped
      if constexpr(D==3) {
	  T(2,4) = - (T(1,5) = tang(0));
	  T(0,5) = - (T(2,3) = tang(1));
	  T(1,3) = - (T(0,4) = tang(2));
	}
      else {
	T(1,2) =  tang(0);
	T(0,2) = -tang(1);
      }
    };
    auto calc_cemat = [&](const auto & cedge, auto & cemat, auto use_fenr) {
      auto fenrs = c2fe[cedge.id];
      cemat = 0;
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
	  // cout << "t0 " << tcf0 << endl;
	  // cout << "t1 " << tcf1 << endl;
	  calc_trafo(T, tcf0, tcf1);
	  // cout << "fedge " << fedge << endl << " to cedge " << cedge << endl;
	  auto & FM = fed[fedge.id];
	  // cout << "FM: " << endl; print_tm(cout, FM); cout << endl;
	  // cout << "trans: " << endl; print_tm(cout, T); cout << endl;
	  TTM = Trans(T) * FM;
	  cemat += TTM * T;
	  // cout << "cemat now: " << endl; print_tm(cout, cemat); cout << endl;
	}
      }
    };
    typedef Mat<dofpv(D), dofpv(D), double> TM;
    auto calc_cedata = [&](const auto & edge, TM & cfullmat) {
      ced[edge.id] = cfullmat;
    };
    /** ex-edge matrices:  calc & reduce full cmats **/
    if (neqcs>1) {
      Array<int> perow(neqcs);
      perow[0] = 0;
      for (auto k : Range(neqcs))
	perow[k] = cmesh.template GetENN<NT_EDGE>(k) + cmesh.template GetCNN<NT_EDGE>(k);
      Table<TM> tcemats(perow); perow = 0;
      cmesh.template ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  calc_cemat(cedge, tcemats[eqc][perow[eqc]++],
		     [&](auto fenr) { return eqc_h.IsMasterOfEQC(mesh.template GetEqcOfNode<NT_EDGE>(fenr)); } );
	}, false); // def, false!
      Table<TM> cemats = ReduceTable<TM,TM> (tcemats, sp_eqc_h, [&](auto & tab) { return sum_table(tab); });
      /** ex-edge matrices:  split to bend + wigg-mats **/
      perow = 0;
      cmesh.template ApplyEQ<NT_EDGE>(Range(size_t(1), neqcs), [&](auto eqc, const auto & cedge){
	  ced[cedge.id] = cemats[eqc][perow[eqc]++];
	}, false); // def, false!
    }
    /** loc-edge matrices:  all in one **/
    cmesh.template ApplyEQ<NT_EDGE>(Range(min(neqcs, size_t(1))), [&](auto eqc, const auto & cedge){
	calc_cemat(cedge, ced[cedge.id], [&](auto fenr) { return true; });
      }, false); // def, false!
    celed.SetParallelStatus(CUMULATED);
    // cout << "wait map edge data " << endl;
    // eqc_h.GetCommunicator().Barrier();
    // cout << "wait map edge data done " << endl;
  }

  
  template class ElEData<2>;
  template class ElEData<3>;
  template void ElEData<2>::map_data (const CoarseMap<ElasticityMesh<2>> & cmap, ElEData<2> & cdata) const;
  template void ElEData<3>::map_data (const CoarseMap<ElasticityMesh<3>> & cmap, ElEData<3> & cdata) const;


} // namespace amg

#include "amg_tcs.hpp"
#endif // ELASTICITY
