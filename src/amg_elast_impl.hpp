#ifndef FILE_AMGELAST_IMPL
#define FILE_AMGELAST_IMPL

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
      Q(1,5) = - (Q(2,4) = t(0));
      Q(2,3) = - (Q(0,5) = t(1));
      Q(0,4) = - (Q(1,3) = t(2));
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

} // namespace amg

#endif
