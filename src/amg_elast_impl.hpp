#ifndef FILE_AMGELAST_IMPL
#define FILE_AMGELAST_IMPL

#ifdef ELASTICITY

namespace amg
{

  constexpr int disppv (int dim)
  { return dim; }
  constexpr int rotpv (int dim)
  { return dim*(dim-1)/2; }
  constexpr int dofpv (int dim)
  { return disppv(dim) + rotpv(dim); }

  
  /** Elasticity Vertex Data **/
  template<class TMESH> INLINE void AttachedEVD :: map_data (const AgglomerateCoarseMap<TMESH> & agg_map, AttachedEVD & cevd) const
  {
    static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
    const auto & M = *mesh;
    auto & cdata = cevd.data;
    const auto & CM = static_cast<BlockTM&>(*agg_map.GetMappedMesh());
    Cumulate();
    cdata.SetSize(agg_map.template GetMappedNN<NT_VERTEX>()); cdata = 0.0;
    auto vmap = agg_map.template GetMap<NT_VERTEX>();
    /** Set coarse vertex pos to the pos of agg centers. **/
    const auto & ctrs = *agg_map.GetAggCenter();
    M.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto v) LAMBDA_INLINE {
	auto cv = vmap[v];
	if (cv != -1) {
	  if (ctrs.Test(v))
	    { cdata[vmap[v]].pos = data[v].pos; }
	  cdata[vmap[v]].wt += data[v].wt;
	}
      }, true);

    // cout << "vmap: " << endl; prow2(vmap); cout << endl;

    // Array<double> ccnt(agg_map.template GetMappedNN<NT_VERTEX>()); ccnt = 0;
    // M.template Apply<NT_VERTEX> ([&](auto v) LAMBDA_INLINE {
    // 	auto cv = vmap[v];
    // 	if (cv != -1)
    // 	  { ccnt[cv]++; }
    //   }, true);
    // CM.template AllreduceNodalData<NT_VERTEX> (ccnt, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });
    // for (auto & val : ccnt)
    //   { val = 1.0 / val; }
    // M.template Apply<NT_VERTEX>([&](auto v) LAMBDA_INLINE {
    // 	auto cv = vmap[v];
    // 	if (cv != -1) {
    // 	  cdata[cv].pos += ccnt[cv] * data[v].pos;
    // 	  cdata[cv].wt += data[v].wt;
    // 	}
    //   }, true);

    cevd.SetParallelStatus(DISTRIBUTED);
  } // AttachedEVD::map_data


  /** Elasticity Edge Data **/


  /** ElasticityAMGFactory **/

  template<int D>
  struct ElasticityAMGFactory<D> :: Options : public ElasticityAMGFactory<D>::BASE::Options // in impl because sing_diag
  {
    bool with_rots = false;            // do we have rotations on the finest level ?

    bool reg_mats = false;             // regularize coarse level matrices
    bool reg_rmats = false;            // regularize replacement-matrix contributions for smoothed prolongation

    /** different algorithms for computing strength of connection **/
    enum SOC_ALG : char { SIMPLE = 0,   // 
			  ROBUST = 1 }; // experimental, probably works for 2d, much more expensive
    SOC_ALG soc_alg = SOC_ALG::SIMPLE;

  };


  template<int D> template<NODE_TYPE NT>
  INLINE double ElasticityAMGFactory<D> :: GetWeight (const ElasticityMesh<D> & mesh, const AMG_Node<NT> & node) const
  {
    if constexpr(NT==NT_VERTEX) { return get<0>(mesh.Data())->Data()[node].wt; }
    else if constexpr(NT==NT_EDGE) { return calc_trace(get<1>(mesh.Data())->Data()[node.id]); }
    else return 0;
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: ModQ  (const Vec<3> & t, TM & Q)
  {
    if constexpr(D == 2) {
	Q(0,2) = -t(1);
	Q(1,2) =  t(0);
      }
    else {
      // Q(1,5) = - (Q(2,4) = t(0));
      // Q(2,3) = - (Q(0,5) = t(1));
      // Q(0,4) = - (Q(1,3) = t(2));
      Q(1,5) = - (Q(2,4) = -t(0));
      Q(2,3) = - (Q(0,5) = -t(1));
      Q(0,4) = - (Q(1,3) = -t(2));
    }
  }

  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcQ  (const Vec<3> & t, TM & Q)
  {
    Q = 0;
    Iterate<dofpv(D)>([&] (auto i) LAMBDA_INLINE { Q(i.value, i.value) = 1.0; } );
    if constexpr(D == 2) {
	Q(0,2) = -t(1);
	Q(1,2) =  t(0);
      }
    else {
      // Q(1,5) = - (Q(2,4) = t(0));
      // Q(2,3) = - (Q(0,5) = t(1));
      // Q(0,4) = - (Q(1,3) = t(2));
      Q(1,5) = - (Q(2,4) = -t(0));
      Q(2,3) = - (Q(0,5) = -t(1));
      Q(0,4) = - (Q(1,3) = -t(2));
    }
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcQij (const T_V_DATA & di, const T_V_DATA & dj, TM & Qij)
  {
    Vec<3> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
  }

  template<int D>
  INLINE void ElasticityAMGFactory<D> :: ModQij (const T_V_DATA & di, const T_V_DATA & dj, TM & Qij)
  {
    Vec<3> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcQHh (const T_V_DATA & dH, const T_V_DATA & dh, TM & QHh)
  {
    Vec<3> t = dh.pos - dH.pos; // H -> h
    CalcQ(t, QHh);
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: ModQHh (const T_V_DATA & dH, const T_V_DATA & dh, TM & QHh)
  {
    Vec<3> t = dh.pos - dH.pos; // H -> h
    ModQ(t, QHh);
  }

  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcQs  (const T_V_DATA & di, const T_V_DATA & dj, TM & Qij, TM & Qji)
  {
    Vec<3> t = 0.5 * (dj.pos - di.pos); // i -> j
    CalcQ(t, Qij);
    t *= -1;
    CalcQ(t, Qji);
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: ModQs  (const T_V_DATA & di, const T_V_DATA & dj, TM & Qij, TM & Qji)
  {
    Vec<3> t = 0.5 * (dj.pos - di.pos); // i -> j
    ModQ(t, Qij);
    t *= -1;
    ModQ(t, Qji);
  }

  template<int D>
  INLINE typename ElasticityAMGFactory<D>::T_V_DATA ElasticityAMGFactory<D> :: CalcMPData (const ElasticityAMGFactory<D>::T_V_DATA & da, const ElasticityAMGFactory<D>::T_V_DATA & db) {
    T_V_DATA o; o.pos = 0.5 * (da.pos + db.pos); o.wt = da.wt + db.wt;
    return move(o);
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcPWPBlock (const ElasticityMesh<D> & fmesh, const ElasticityMesh<D> & cmesh,
  						       AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, ElasticityAMGFactory<D>::TM & mat) const
  {
    CalcQHh(get<0>(cmesh.Data())->Data()[cv], get<0>(fmesh.Data())->Data()[v], mat);
  }


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcRMBlock (const ElasticityMesh<D> & fmesh, const AMG_Node<NT_EDGE> & edge,
  						      FlatMatrix<typename ElasticityAMGFactory<D>::TM> mat) const
  {
    static TM Qij, Qji, QiM, QjM;
    auto vd = get<0>(fmesh.Data())->Data();
    auto ed = get<1>(fmesh.Data())->Data();
    CalcQs( vd[edge.v[0]], vd[edge.v[1]], Qij, Qji);
    auto & M = ed[edge.id];
    // Vec<3> t = vd[edge.v[1]].pos - vd[edge.v[0]].pos; cout << "tang 0 -> 1: " << t << endl;
    // cout << " emat: " << endl; print_tm(cout, M); cout << endl;
    // cout << " Qij: " << endl; print_tm(cout, Qij); cout << endl;
    // cout << " Qji: " << endl; print_tm(cout, Qji); cout << endl;
    
    QiM = Trans(Qij) * M;
    QjM = Trans(Qji) * M;
    mat(0,0) =  QiM * Qij;
    mat(0,1) = -QiM * Qji;
    mat(1,0) = -QjM * Qij;
    mat(1,1) =  QjM * Qji;
  }


  /** approx. max ev of A in IP given by B via vector iteration **/
  template<int N> INLINE double CalcMaxEV (const Matrix<double> & A, const Matrix<double> & B)
  {
    static Vector<double> v(N), v2(N), vn(N);
    Iterate<N>([&](auto i){ v[i] = double (rand()) / RAND_MAX; } ); // TODO: how much does this cost?
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
  } // CalcMinGenEV


  /** EmbedVAMG **/


  template<class C> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh)
  {
    auto a = new AttachedEVD(Array<ElasticityVertexData>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto vdata = a->Data();
    FlatArray<int> vsort = node_sort[NT_VERTEX];
    auto & vp = node_pos[NT_VERTEX];
    for (auto k : Range(vdata)) {
      auto& x = vdata[k];
      x.wt = 0.0;
      x.pos = vp[k];
    }

    auto b = new AttachedEED<C::DIM>(Array<ElasticityEdgeData<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), CUMULATED);
    for (auto & x : b->Data()) { SetIdentity(x); }

    auto mesh = make_shared<typename C::TMESH>(move(*top_mesh), a, b);

    return mesh;
  } // EmbedVAMG::BuildAlgMesh_TRIV


  template<class C> template<class TD2V, class TV2D> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    if (spmat == nullptr)
      { throw Exception("BuildAlgMesh_ALG_scal called with no mat!"); }

    const auto & O(*options);
    auto a = new AttachedEVD(Array<ElasticityVertexData>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
    const auto & vsort = node_sort[NT_VERTEX];

    // BDDC does not work with multidim anyways, so this should usually be correct!

    /** vertex-points **/
    Vec<3> t; const auto & MA(*ma);
    for (auto k : Range(O.v_nodes)) {
      auto vnum = vsort[k];
      vdata[vnum].wt = 0;
      GetNodePos(O.v_nodes[k], MA, vdata[vnum].pos, t);
    }

    auto b = new AttachedEED<C::DIM>(Array<ElasticityEdgeData<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); // !! has to be distr
    auto edata = b->Data();

    const auto& dof_blocks(options->block_s);
    // auto& fvs = *free_verts;
    // cout << endl << " CALC INIT EMATS " << endl << endl;
    const auto& ffds = *finest_freedofs;
    if ( (dof_blocks.Size() == 1) && (dof_blocks[0] == 1) ) { // multidim
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<Mat<disppv(C::DIM),disppv(C::DIM),double>>>(spmat)) { // disp only
	const auto& A(*spm_tm);
	for (auto & e : edges) {
	  auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
	  // cout << "edge " << e << endl << " dofs " << di << " " << dj << endl;
	  // cout << " mat etr " << endl; print_tm(cout, A(di, dj)); cout << endl;
	  // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
	  double etrs = fabsum(A(di,dj));
	  // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / disppv(C::DIM) : 1e-4; // after BBDC, diri entries are compressed and mat has no entry 
	  double fc = (etrs != 0.0) ? etrs / disppv(C::DIM) : 1e-4;
	  // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / sqrt(fabsum(A(di,di)) * fabsum(A(dj,dj))) / disppv(C::DIM) : 1e-4;
	  auto & emat = edata[e.id]; emat = 0;
	  // Iterate<disppv(C::DIM)>([&](auto i) LAMBDA_INLINE { emat(i.value, i.value) = fc; });

	  Vec<3> tang = vdata[e.v[1]].pos - vdata[e.v[0]].pos;
	  double len = L2Norm(tang);
	  fc /= (len * len);
	  // if (ffds.Test(di) && ffds.Test(dj)) {
	  //   auto tri = calc_trace(A(di,di))/disppv(C::DIM);
	  //   auto trj = calc_trace(A(dj,dj))/disppv(C::DIM);
	  //   fc /= sqrt(fabs(tri*trj));
	  // }
	  Iterate<disppv(C::DIM)>([&](auto i) LAMBDA_INLINE {
	      Iterate<disppv(C::DIM)>([&](auto j) LAMBDA_INLINE {
		  emat(i.value, j.value) = fc * tang(i.value) * tang(j.value);
		});
	    });
	  auto fsem = fabsum(emat);
	  emat *= etrs / fsem;

	  // cout << " emat: " << endl; print_tm(cout, emat); cout << endl;
	  // TM X, Qij, Qji;
	  // C::CalcQs(vdata[e.v[0]], vdata[e.v[0]], Qij, Qji);
	  // X = - Trans(Qij) * emat * Qji;
	  // cout << " repl entry: " << endl; print_tm(cout, X); cout << endl;
	  // Iterate<disppv(C::DIM)>([&](auto i) LAMBDA_INLINE {
	      // Iterate<disppv(C::DIM)>([&](auto j) LAMBDA_INLINE {
		  // X(i.value, j.value) -= A(di,dj)(i.value, j.value);
		// });
	    // });
	  // cout << " diff " << endl; print_tm(cout, X);
    	}
      }
      else if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<Mat<dofpv(C::DIM),dofpv(C::DIM),double>>>(spmat)) { // disp+rot
	const auto& A(*spm_tm);
	for (auto & e : edges) {
	  auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
	  // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
	  double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / dofpv(C::DIM) : 1e-4; // after BBDC, diri entries are compressed and mat has no entry 
	  // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / sqrt(fabsum(A(di,di)) * fabsum(A(dj,dj))) / dofpv(C::DIM) : 1e-4;
	  auto & emat = edata[e.id]; emat = 0;
	  Iterate<dofpv(C::DIM)>([&](auto i) LAMBDA_INLINE { emat(i.value, i.value) = fc; });
	}
      }
      else
	{ throw Exception(string("not sure how to compute edge weights from mat of type ") + typeid(*spmat).name() + string("!")); }
    }
    else
      { throw Exception("block_s for compound, but called algmesh_alg_scal!"); }

    // cout << endl << " DONE W. INIT EMATS " << endl << endl;
    // cout << " v data : " << endl; prow2(vdata); cout << endl;
    // cout << " e data : " << endl; prow2(edata); cout << endl;

    auto mesh = make_shared<typename C::TMESH>(move(*top_mesh), a, b);

    return mesh;
  } // EmbedVAMG::BuildAlgMesh_ALG_scal


  template<class C> template<class TD2V, class TV2D> shared_ptr<typename C::TMESH>
  EmbedVAMG<C> :: BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    // cout << " mesh mat " << top_mesh << " " << spmat << endl;
    // cout << " BLK NV " << top_mesh->GetNN<NT_VERTEX>() << endl;
    auto a = new AttachedEVD(Array<ElasticityVertexData>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
    auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
    const auto & vsort = node_sort[NT_VERTEX];
    const auto & O(*options);

    // cout << " vsort: " << endl; prow2(vsort); cout << endl;

    // With [on_dofs = select, subset = free], not all vertices in the mesh have a "vertex" in the alg-mesh !

    /** vertex-points **/
    Vec<3> t; const auto & MA(*ma);
    for (auto k : Range(O.v_nodes)) {
      auto vnum = vsort[k];
      vdata[vnum].wt = 0;
      GetNodePos(O.v_nodes[k], MA, vdata[vnum].pos, t);
    }

    // cout << "have vdata: " << endl; prow(vdata); cout << endl;

    auto b = new AttachedEED<C::DIM>(Array<ElasticityEdgeData<C::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); // !! has to be distr
    auto edata = b->Data();

    auto edges = top_mesh->GetNodes<NT_EDGE>();

    if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<double>>(spmat)) {
      const auto& ffds = *finest_freedofs;
      const auto & MAT = *spm_tm;
      for (const auto & e : edges) {
	auto dis = V2D(e.v[0]); auto djs = V2D(e.v[1]); auto diss = dis.Size();
	auto & ed = edata[e.id]; ed = 0;
	// after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
	if (ffds.Test(dis[0]) && ffds.Test(djs[0])) {
	  double x = 0;
	  // TODO: should I scale with diagonal inverse here ??
	  // actually, i think i should scale with diag inv, then sum up, then scale back
	  TM aij(0);
	  for (auto i : Range(dis)) { // this could be more efficient
	    x += fabs(MAT(dis[i], djs[i]));
	    for (auto j = i+1; j < diss; j++)
	      { x += 2*fabs(MAT(dis[i], djs[j])); }
	  }
	  x /= (diss * diss);
	  if (diss == disppv(C::DIM)) {
	    Vec<3> tang = vdata[e.v[1]].pos - vdata[e.v[0]].pos;
	    Iterate<disppv(C::DIM)>([&](auto i) LAMBDA_INLINE {
		Iterate<disppv(C::DIM)>([&](auto j) LAMBDA_INLINE {
		    ed(i.value, j.value) = x * tang(i.value) * tang(j.value);
		  });
	      });
	  }
	  else {
	    for (auto j : Range(diss))
	      { ed(j,j) = x; }
	  }
	}
	else { // does not matter what we give in here, just dont want nasty NaNs below, however this is admittedly a bit hacky
	  ed(0,0) = 0.00042;
	}
      }
    }
    else
      { throw Exception(string("not sure how to compute edge weights from mat of type (_blk version called)") + typeid(*spmat).name() + string("!")); }

    auto mesh = make_shared<typename C::TMESH>(move(*top_mesh), a, b);

    return mesh;
  } // EmbedVAMG::BuildAlgMesh_ALG_blk

  template<> template<>
  shared_ptr<BaseDOFMapStep> INLINE EmbedVAMG<ElasticityAMGFactory<2>> :: BuildEmbedding_impl<6> (shared_ptr<ElasticityMesh<2>> mesh)
  { return nullptr; }


  template<> template<>
  shared_ptr<BaseDOFMapStep> INLINE EmbedVAMG<ElasticityAMGFactory<3>> :: BuildEmbedding_impl<2> (shared_ptr<ElasticityMesh<3>> mesh)
  { return nullptr; }

  /** EmbedWithElmats **/

  template<class C, class D, class E> shared_ptr<typename EmbedWithElmats<C,D,E>::TMESH>
  EmbedWithElmats<C,D,E> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
  {
    return nullptr;
  } // EmbedWithElmats::BuildAlgMesh_ELMAT


  // template<class C, class D, class E> void EmbedWithElmats<C,D,E> ::
  // AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
  // 		    ElementId ei, LocalHeap & lh)
  // {
  //   // if (options->energy != Options::ENERGY::ELMAT_ENERGY)
  //     // { return; }
  //   // static Timer t(string("EmbedVAMG<ElasticityAMG<") + to_string(C::DIM) + ">::AddElementMatrix");
  //   // RegionTimer rt(t);
  //   // const bool vmajor = (options->block_s.Size()==1) ? 1 : 0;
  // } // EmbedWithElmats::AddElementMatrix

} // namespace amg

#endif

#endif
