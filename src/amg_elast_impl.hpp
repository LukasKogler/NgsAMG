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
  INLINE void ElasticityAMGFactory<D> :: CalcPWPBlock (const ElasticityMesh<D> & fmesh, const ElasticityMesh<D> & cmesh,
						       const CoarseMap<ElasticityMesh<D>> & map, AMG_Node<NT_VERTEX> v,
						       AMG_Node<NT_VERTEX> cv, ElasticityAMGFactory<D>::TM & mat) const
  {
    Vec<3,double> tang = get<0>(cmesh.Data())->Data()[cv].pos;
    tang -= get<0>(fmesh.Data())->Data()[v].pos;
    mat = 0;
    for (auto k : Range(dofpv(D)))
      { mat(k,k) = 1.0; }
    if constexpr(D==2) {
	mat(1,2) = -tang(0);
	mat(0,2) =  tang(1);
      }
    else {
      mat(1,5) = - (mat(2,4) = tang(0));
      mat(2,3) = - (mat(0,5) = tang(1));
      mat(0,4) = - (mat(1,3) = tang(2));
    }
  } // ElasticityAMGFactory::CalcPWPBlock


  template<int D>
  INLINE void ElasticityAMGFactory<D> :: CalcRMBlock (const ElasticityMesh<D> & fmesh, const AMG_Node<NT_EDGE> & edge,
						      FlatMatrix<typename ElasticityAMGFactory<D>::TM> mat) const
  {
    /**
       I |  0.5 * sk(t)| -I |  0.5 * sk(t) 
       0 |      I      |  0 |     -I
    **/
    const auto & O = static_cast<const ElasticityAMGFactory<D>::Options &>(*this->options);

    Vec<3,double> tang = get<0>(fmesh.Data())->Data()[edge.v[1]].pos
      - get<0>(fmesh.Data())->Data()[edge.v[0]].pos;
    // cout << "v0: " << get<0>(fmesh.Data())->Data()[edge.v[0]].pos << endl;
    // cout << "v1: " << get<0>(fmesh.Data())->Data()[edge.v[1]].pos << endl;
    // cout << "tang: " << tang << endl;
    static Matrix<TM> M(1,1);
    static Matrix<TM> T(1,2);
    static Matrix<TM> TTM(2,1);
    SetIdentity(T(0,0));
    if constexpr(D==2) {
  	T(0,0)(1,2) =  0.5 * tang(0);
  	T(0,0)(0,2) = -0.5 * tang(1);
      }
    else {
      T(0,0)(2,4) = - (T(0,0)(1,5) = 0.5 * tang(0));
      T(0,0)(0,5) = - (T(0,0)(2,3) = 0.5 * tang(1));
      T(0,0)(1,3) = - (T(0,0)(0,4) = 0.5 * tang(2));
    }
    T(0,1) = T(0,0);
    Iterate<dofpv(D)>([&](auto i) {
  	T(0,1)(i.value, i.value) = -1.0;
      });
    M(0,0) = get<1>(fmesh.Data())->Data()[edge.id]; // cant do Matrix<TM> * TM
    // cout << "(unreged) M: " << endl; print_tm_mat(cout, M); cout << endl;
    if (O.reg_rmats) // not so sure if this is the right way to do it...
      { RegTM<disppv(D), rotpv(D), dofpv(D)>(M(0,0)); }
    Iterate<2>([&](auto i) {
  	TTM(i.value,0) = Trans(T(0,i.value)) * M(0,0);
      });
    // TTM = TT * M;
    mat = TTM * T;
    // cout << "M: " << endl; print_tm_mat(cout, M); cout << endl;
    // cout << "T: " << endl; print_tm_mat(cout, T); cout << endl;
    // cout << "TTM: " << endl; print_tm_mat(cout, TTM); cout << endl;
    // cout << "emat: " << endl; print_tm_mat(cout, mat); cout << endl;
  } // ElasticityAMGFactory::CalcRMBlock


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
