#ifndef FILE_MIS_AGG_IMPL_HPP
#define FILE_MIS_AGG_IMPL_HPP

#ifdef MIS_AGG

#include "mis_agg.hpp"

#include <utils_denseLA.hpp>

// namespace amg
// {

// INLINE double MIN_EV_HARM (double A, double B, double R) { return 0.5 * R * (A+B)/(A*B); }
// template<int N>
// INLINE double MIN_EV_HARM (const Mat<N,N,double> & A, const Mat<N,N,double> & B, const Mat<N,N,double> & aR)
// {
//   /**
//      Compute inf <Rx,x> / <A(A+B)^(-1)Bx,x> (<= 1!):
//         - decompose L
//   - project R to ortho(ker(L))
//       [ we can ignore any x in ker(L), if its in ker(R), we have 1, else "inf", both not interesting ]
//   - return minimum EV of L^(-1/2) R L^(-1/2) [projected to ortho(ker(L))]
//     **/
//   static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
//   HeapReset hr(lh);

//   static Mat<N,N,double> SUM;

//   SUM = A + B;
//   CalcPseudoInverse<N>(SUM); // Is CPI<3,3,6> valid in 3d ??

//   FlatMatrix<double> L(N, N, lh), R(N, N, lh);
//   L = A * SUM * B;
//   R = aR;

//   return 0.5 * MEV<N>(L,R);
// } // MIN_EV_HARM

// } // namespace amg

namespace amg
{
/** Agglomerator **/

template<class ENERGY, class TMESH, bool ROBUST>
MISAgglomerator<ENERGY, TMESH, ROBUST> :: MISAgglomerator (shared_ptr<TMESH> _mesh)
  : Agglomerator<TMESH>(_mesh)
{
  assert(_mesh != nullptr); // obviously this would be bad
} // MISAgglomerator(..)


template<class ENERGY, class TMESH, bool ROBUST>
void MISAgglomerator<ENERGY, TMESH, ROBUST> :: Initialize (const MISAggOptions & opts, int level)
{
  lazy_neib_boost = opts.lazy_neib_boost.GetOpt(level);
  use_minmax_soc  = opts.ecw_minmax.GetOpt(level);
  dist2           = opts.mis_dist2.GetOpt(level);
  minmax_avg      = opts.agg_minmax_avg.GetOpt(level);
  cw_geom         = opts.ecw_geom.GetOpt(level);
} // MISAgglomerator::InitializeMIS


template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
INLINE void MISAgglomerator<ENERGY, TMESH, ROBUST> :: GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data)
{
  if constexpr(std::is_same<TED, TMU>::value)
  {
    out_data.FlatArray<TMU>::Assign(in_data);
  }
  else if constexpr(std::is_same<TMU, double>::value)
  { /** use trace **/
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = ENERGY::GetApproxWeight(in_data[k]); }
	}
  else
  {
    /**
     *  H11 needs "double -> TM" extension for higher dim only to make it
     *  compile, we never call this.
     *  We actually go here for Stokes where energy has two mats per edge.
     *  h1, double -> TM extension is necessary pro forma
     */
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = ENERGY::GetEMatrix(in_data[k]); }
  }
}

template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
INLINE void MISAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  static_assert ( (std::is_same<TMU, TM>::value || std::is_same<TMU, double>::value), "Only 2 options make sense!");

  static Timer t("FormAgglomerates"); RegionTimer rt(t);

  static Timer tdiag("FormAgglomerates - diags");
  static Timer tecw("FormAgglomerates - init. ecw");
  static Timer t1("FormAgglomerates - 1"); // prep
  static Timer t2("FormAgglomerates - 2"); // first loop
  static Timer t3("FormAgglomerates - 3"); // second loop

  constexpr int N = ngbla::Height<TMU>();
  const auto & M = this->GetMesh();
  M.CumulateData();
  const auto & eqc_h = *M.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();
  const auto NV = M.template GetNN<NT_VERTEX>();
  const auto & econ = *M.GetEdgeCM();

  const double MIN_ECW = this->edge_thresh;
  const bool dist2 = this->dist2;
  const bool geom = this->cw_geom;
  const int MIN_NEW_AGG_SIZE = 2;
  const bool enable_neib_boost = this->neib_boost;
  const bool lazy_neib_boost = this->lazy_neib_boost;
  /** use if explicitely turned on, or if harmonic mean and not turned off **/
  const bool use_stab_ecw_hack = this->use_stab_ecw_hack.IsTrue() || ( (!this->use_stab_ecw_hack.IsFalse()) && (!geom) );
  /** use if explicitely turned on, or if not turned off and geometric mean is used **/
  const bool use_minmax_soc = this->use_minmax_soc.IsTrue() || ( (!this->use_minmax_soc.IsFalse()) && geom );
  const AVG_TYPE minmax_avg = this->minmax_avg;

  FlatArray<TVD> vdata = get<0>(M.AttachedData())->Data();
  Array<TMU> edata; GetEdgeData<TMU>(get<1>(M.AttachedData())->Data(), edata);
  // Array<TMU> edata; GetEdgeData<TMU>(edata);

  if (this->print_aggs) {
    cout << " agg coarsen params: " << endl;
    cout << " dist2 = " << dist2 << endl;
    cout << " geom = " << geom << endl;
    cout << " enable_neib_boost = " << enable_neib_boost << endl;
    cout << " lazy_neib_boost = " << lazy_neib_boost << endl;
    cout << " stab_ecw_hack = " << use_stab_ecw_hack << endl;
    cout << " use_minmax_soc = " << use_minmax_soc << endl;
    cout << " minmax_avg = " << minmax_avg << endl;
    cout << " min new agg size = " << MIN_NEW_AGG_SIZE << endl;
    cout << " neibs per v " << double(2 * M.template GetNN<NT_EDGE>())/NV << endl;
    size_t mpr = 0;
    for (auto k : Range(econ.Height()))
      { mpr = max2(mpr, econ.GetRowIndices(k).Size()); }
    Array<int> pr(mpr+1); pr = 0;
    for (auto k : Range(econ.Height()))
      { pr[econ.GetRowIndices(k).Size()]++; }
    cout << " econ perow: " << endl; prow2(pr); cout << endl;
    // cout << " ECON: " << endl << econ << endl;
  }

  /** replacement-matrix diagonals **/
  Array<TMU> repl_diag(M.template GetNN<NT_VERTEX>()); repl_diag = 0;

  /** collapse weights for edges - we use these as upper bounds for weights between agglomerates (TODO: should not need anymore) **/
  Array<double> ecw(M.template GetNN<NT_EDGE>()); ecw = 0;

  /** vertex -> agglomerate map **/
  BitArray marked(M.template GetNN<NT_VERTEX>()); marked.Clear();
  Array<int> dist2agg(M.template GetNN<NT_VERTEX>()); dist2agg = -1;
  v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;


  /** Am I the one that determines the agg for this vertex? **/
  auto solid_verts = this->GetSolidVerts();
  auto allowed_edges = this->GetAllowedEdges();
  const bool use_sv = solid_verts != nullptr;

  auto is_mine_eq = [&](auto v, auto eq) LAMBDA_INLINE { return use_sv ? solid_verts->Test(v) :
                eqc_h.IsMasterOfEQC(eq); };

  auto is_mine = [&](auto v) LAMBDA_INLINE { return use_sv ? solid_verts->Test(v) :
                eqc_h.IsMasterOfEQC(M.template GetEQCOfNode<NT_VERTEX>(v)); };

  /** Can we add something from eqa to eqb?? **/
  auto eqa_to_eqb = [&](auto eqa, auto eqb) { // PER DEFINITION, we are master of eqb!
    return eqc_h.IsLEQ(eqa, eqb);
  };

  LocalHeap lh(10 * 1024 * 1024, "agglh", false);

  auto fixed_aggs = this->GetFixedAggs();

  if (fixed_aggs.Size()) {
    for (auto k : Range(fixed_aggs.Size())) {
      for (auto v : fixed_aggs[k])
        { marked.SetBit(v); }
    }
  }

  /** non-master/non-solid vertices **/
  M.template ApplyEQ2<NT_VERTEX>([&](auto eq, auto vs) {
    if (use_sv) {
      if (eq > 0)
        for (auto v : vs)
          if (!solid_verts->Test(v))
            { marked.SetBit(v); }
    }
    else {
      if (!eqc_h.IsMasterOfEQC(eq))
        for (auto v : vs)
          { marked.SetBit(v); }
    }
  }, false); // obviously, not master only


  Array<TMU> agg_diag;

  auto get_vwt = [&](auto v) {
    return ENERGY::GetApproxVWeight(vdata[v]);
    // if constexpr(is_same<TVD, double>::value) { return vdata[v]; }
    // else { return vdata[v].wt; }
  };


  Array<int> neibs_in_agg;
  size_t cnt_prtm = 0;
  Array<int> common_neibs(20), aneibs(20), bneibs(20);
  Array<FlatArray<int>> neib_tab(20);
  TMU Qij(0), Qji(0), emat(0), Q(0), Ein(0), Ejn(0), Esum(0), addE(0), Q2(0), Aaa(0), Abb(0);
  SetIdentity(Qij); SetIdentity(Qji); SetIdentity(emat); SetIdentity(Q); SetIdentity(Ein); SetIdentity(Ejn);
  SetIdentity(Esum); SetIdentity(addE); SetIdentity(Q2); SetIdentity(Aaa); SetIdentity(Abb);

  /** Add a contribution from a neighbour common to both vertices of an edge to the edge's matrix.
      Lazy version where we only compute traces and multiply by a factor. **/
  auto add_neib_edge_lazy = [&](const TVD & h_data, const auto & amems, const auto & bmems, auto N, auto & mat) LAMBDA_INLINE {
    double Ein = 0, Ejn = 0;
    constexpr int N2 = ngbla::Height<TMU>();
    auto rowis = econ.GetRowIndices(N);
    intersect_sorted_arrays(rowis, amems, neibs_in_agg);
    for (auto amem : neibs_in_agg)
      { Ein += calc_trace(edata[int(econ(amem,N))]); }
    Ejn = 0;
    intersect_sorted_arrays(rowis, bmems, neibs_in_agg);
    for (auto bmem : neibs_in_agg)
      { Ejn += calc_trace(edata[int(econ(bmem,N))]); }
    /** tr(fac*mat) = tr(mat + hm*Id) **/
    double hm = 2.0 * (Ein * Ejn) / (Ein + Ejn);
    mat *= (1 + hm/calc_trace(mat));
  };

  /** Add a contribution from a neighbour common to both vertices of an edge to the edge's matrix **/
  auto add_neib_edge = [&](const TVD & h_data, const auto & amems, const auto & bmems, auto N, auto & mat) LAMBDA_INLINE {
    if (lazy_neib_boost) {
      add_neib_edge_lazy(h_data, amems, bmems, N, mat);
      return;
    }
    constexpr int N2 = ngbla::Height<TMU>();
    auto rowis = econ.GetRowIndices(N);
    Ein = 0;
    intersect_sorted_arrays(rowis, amems, neibs_in_agg);
    for (auto amem : neibs_in_agg) {
      ModQij(vdata[N], vdata[amem], Q2);
      AddQtMQ(1.0, Ein, Q2, edata[int(econ(amem,N))]);
    }
    Ejn = 0;
    intersect_sorted_arrays(rowis, bmems, neibs_in_agg);
    for (auto bmem : neibs_in_agg) {
      ModQij(vdata[N], vdata[bmem], Q2);
      AddQtMQ(1.0, Ejn, Q2, edata[int(econ(bmem,N))]);
    }
    if constexpr(is_same<TMU, double>::value)
    {
      addE = (Ein + Ejn) / max(1e-20, Esum);
    }
    else
    {
      // pseudo inv - 3d-elasticity!
      Esum = Ein + Ejn;
      CalcPseudoInverse(Esum, lh);
      addE = TripleProd(Ein, Esum, Ejn);
    }
    ModQHh(h_data, vdata[N], Q2); // QHN
    AddQtMQ (2.0, mat, Q2, addE);
  }; // add_neib_edge

  auto calc_soc_mats = [&]() LAMBDA_INLINE {
    double mmev;
    if (geom)
      { mmev = MIN_EV_FG2(Aaa, Abb, emat); }
    else
      { mmev = MIN_EV_HARM2(Aaa, Abb, emat); }
    return mmev;
  };

  auto calc_emat = [&](TVD & H_data,
      auto ca, FlatArray<int> memsa, auto cb, FlatArray<int> memsb,
      bool common_neib_boost) LAMBDA_INLINE {
    const auto memsas = memsa.Size();
    const auto memsbs = memsb.Size();
    bool vv_case = (memsas == memsbs) && (memsas == 1);
    if ( vv_case ) {// simple vertex-vertex case
      int eid = int(econ(ca, cb));
      emat = edata[eid];
      if (enable_neib_boost && common_neib_boost) { // on the finest level, this is porbably 0 in most cases, but still expensive
        intersect_sorted_arrays(econ.GetRowIndices(ca), econ.GetRowIndices(cb), common_neibs);
        for (auto v : common_neibs)
          { add_neib_edge(H_data, memsa, memsb, v, emat); }
      }
    }
    else { // find all edges connecting the agglomerates and most shared neibs
      emat = 0;
      for (auto amem : memsa) { // add up emat contributions
        intersect_sorted_arrays(econ.GetRowIndices(amem), memsb, common_neibs);
        for (auto bmem : common_neibs) {
          int eid = int(econ(amem, bmem));
          TVD h_data = ENERGY::CalcMPData(vdata[amem], vdata[bmem]);
          ModQHh (H_data, h_data, Q);
          AddQtMQ(1.0, emat, Q, edata[eid]);
        }
      }
      if (enable_neib_boost && common_neib_boost) {
        auto get_all_neibs = [&](auto mems, auto & ntab, auto & out) LAMBDA_INLINE {
          ntab.SetSize0(); ntab.SetSize(mems.Size());
          for (auto k : Range(mems))
            { ntab[k].Assign(econ.GetRowIndices(mems[k])); }
          merge_arrays(ntab, out, [&](const auto & i, const auto & j) LAMBDA_INLINE { return i < j; });
        };
        get_all_neibs(memsa, neib_tab, aneibs);
        get_all_neibs(memsb, neib_tab, bneibs);
        intersect_sorted_arrays(aneibs, bneibs, common_neibs); // can contain members of a/b
        for (auto N : common_neibs) {
          auto pos = find_in_sorted_array(N, memsa);
          if (pos == -1) {
            pos = find_in_sorted_array(N, memsb);
            if (pos == -1)
              { add_neib_edge(H_data, memsa, memsb, N, emat); }
            }
        }
        // cout << endl;
      }
    }
  };

  auto CalcSOC = [&](auto ca, FlatArray<int> memsa, const auto & diaga,
          auto cb, FlatArray<int> memsb, const auto & diagb,
          bool common_neib_boost) LAMBDA_INLINE {
    if (allowed_edges) {
      bool allowit = false;
      for (auto amem : memsa) {
        auto rvs = econ.GetRowValues(amem);
        iterate_intersection(econ.GetRowIndices(amem),
                             memsb,
                             [&](auto i, auto j) {
                               allowit |= allowed_edges->Test(int(rvs[i]));
        });
        if (allowit)
          { break; }
      }
      if (!allowit) // OK, no idea if we divide by this somewhere, so just return a small value
        { cout << "FORBID " << ca << ", mems "; prow(memsa); cout << " - " << cb << ", mems "; prow(memsb); cout << endl; }
      if (!allowit) // OK, no idea if we divide by this somewhere, so just return a small value
        { return 1e-10; }
    }
    /** Transformed diagonal matrices **/
    TVD H_data = ENERGY::CalcMPData(vdata[ca], vdata[cb]);
    ModQHh(H_data, vdata[ca], Q);
    SetQtMQ(Aaa, Q, diaga);
    ModQHh(H_data, vdata[cb], Q);
    SetQtMQ(Abb, Q, diagb);
    /** Calc edge matrix connecting agglomerates **/
    calc_emat(H_data, ca, memsa, cb, memsb, common_neib_boost);
    /** This is a bit ugly, but i want to avoid CalcSOC1/2, that compiles too long. Here, calc_soc_mats is only called ONCE!**/
    double mmev = 1;
    if (use_minmax_soc) {
      double mtra = 0, mtrb = 0;
      for (auto amem : memsa) { // get max trace of edge mat leading outwards
        auto rvs = econ.GetRowValues(amem);
        iterate_anotb(econ.GetRowIndices(amem), memsa, [&](auto inda) LAMBDA_INLINE {
            mtra = max2(mtra, calc_trace(edata[int(rvs[inda])]));
        });
      }
      for (auto bmem : memsb) { // get max trace of edge mat leading outwards
        auto rvs = econ.GetRowValues(bmem);
        iterate_anotb(econ.GetRowIndices(bmem), memsb, [&](auto indb) LAMBDA_INLINE {
            mtrb = max2(mtrb, calc_trace(edata[int(rvs[indb])]));
        });
      }
      double mtr = 0;
      switch(minmax_avg) {
      case(MIN): { mtr = min(mtra, mtrb); break; }
      case(GEOM): { mtr = sqrt(mtra * mtrb); break; }
      case(HARM): { mtr = 2 * (mtra * mtrb) / (mtra + mtrb); break; }
      case(ALG): { mtr = (mtra + mtrb) / 2; break; }
      case(MAX): { mtr = max(mtra, mtrb); break; }
      }
      mmev = calc_trace(emat) / mtr;
    }
    constexpr int N2 = ngbla::Height<TMU>();
    if constexpr(N2 == 1) {
      if (!use_minmax_soc)
        { mmev = calc_soc_mats(); }
    }
    else {
      if (use_minmax_soc) {
        Aaa /= calc_trace(Aaa);
        Abb /= calc_trace(Abb);
        emat /= calc_trace(emat);
      }
      double soc_mat = calc_soc_mats();
      mmev = min2(mmev, soc_mat);
    }
    /** Admittedly crude hack for l2 weights **/
    double vw0 = get_vwt(ca);
    double vw1 = get_vwt(cb);
    double maxw = max(vw0, vw1);
    double minw = min(vw0, vw1);
    double fac = (fabs(maxw) < 1e-12) ? 1.0 : (0.1+minw)/(0.1+maxw);
    return fac * mmev;
  }; // CalcSOC


  Array<int> dummya(1), dummyb(1);
  auto CalcSOC_av = [&](const auto & agg, auto v, auto cnb) LAMBDA_INLINE {
    dummyb = v;
    return CalcSOC(agg.center(), agg.members(), agg_diag[agg.id], v, dummyb, repl_diag[v], cnb);
  };

  auto CalcSOC_aa = [&](const auto & agga, const auto & aggb, auto cnb) LAMBDA_INLINE {
    return CalcSOC(agga.center(), agga.members(), agg_diag[agga.id],
        aggb.center(), aggb.members(), agg_diag[aggb.id],
        cnb);
  };

  auto add_v_to_agg = [&](auto & agg, auto v) LAMBDA_INLINE {
    /**
  Variant 1:
      diag = sum_{k in agg, j not in agg} Q(C->mid(k,j)).T Ekj Q(C->mid(k,j))
      So, when we add a new member, we have to remove contributions from edges that are now
      internal to the agglomerate, and add new edges instead!
      !!! Instead we add Q(C->j).T Ajj Q(C->j) and substract Q(C->mid(k,j)).T Ekj Q(C->mid(k,j)) TWICE !!
      We know that any edge from a new agg. member to the agg is IN-EQC, so we KNOW we LOCALLY have the edge
      and it's full matrix.
      However, we might be missing some new out-of-agg connections from the other vertex. So we just add the entire
      diag and substract agg-v twice.
  Probably leads to problems with "two-sided" conditions like:
      alpha_ij * (aii+ajj) / (aii*ajj)
  So we would need something like:
      alpha_ij / sqrt(aii*ajj)
  Alternatively, we take a two-sided one, and use
    CW(agg, v) = min( max_CW_{n in Agg}(n,v), CW_twoside(agg, v))
    [ So agglomerating vertices can only decrease the weight of edges but never increase it ]
    **/
    auto vneibs = econ.GetRowIndices(v);
    auto eids = econ.GetRowValues(v);
    double fac = 2;
    if (use_stab_ecw_hack  && (agg.members().Size() == 1) )
      { fac = 1.5; }
    for (auto j : Range(vneibs)) {
      auto neib = vneibs[j];
      auto pos = find_in_sorted_array(neib, agg.members());
      if (pos != -1) {
        int eid = int(eids[j]);
        TVD mid_vd = ENERGY::CalcMPData(vdata[neib], vdata[v]);
        ModQHh(vdata[agg.center()], mid_vd, Q); // Qij or QHh??
        agg_diag[agg.id] -= fac * Trans(Q) * edata[eid] * Q;
      }
    }
    ModQHh(vdata[agg.center()], vdata[v], Q);
    agg_diag[agg.id] += 1.0 * Trans(Q) * repl_diag[v] * Q;
    agg.AddSort(v);
  }; // add_v_to_agg

  Array<int> neib_ecnt(30), qsis(30);
  Array<double> ntraces(30);
  auto init_agglomerate = [&](auto v, auto v_eqc, bool force) LAMBDA_INLINE {
    auto agg_nr = agglomerates.Size();
    agglomerates.Append(Agglomerate(v, agg_nr)); // TODO: does this do an allocation??
    agg_diag.Append(repl_diag[v]);
    marked.SetBit(v);
    v_to_agg[v] = agg_nr;
    dist2agg[v] = 0;
    auto& agg = agglomerates.Last();
    auto & aggd = agg_diag[agg_nr];
    auto neibs_v = econ.GetRowIndices(v);
    int cnt_mems = 1;
    auto may_check_neib = [&](auto N) -> bool LAMBDA_INLINE
      { return (!marked.Test(N)) && eqa_to_eqb(M.template GetEQCOfNode<NT_VERTEX>(N), v_eqc) ;  };
    /** First, try to find ONE neib which we can add. Heuristic: Try descending order of trace(emat).
        I think we should give common neighbour boost here. **/
    auto neibs_e = econ.GetRowValues(v);
    ntraces.SetSize0(); ntraces.SetSize(neibs_v.Size());
    qsis.SetSize0(); qsis.SetSize(neibs_v.Size());
    int first_N_ind = -1;
    for (auto k : Range(ntraces)) {
      ntraces[k] = calc_trace(edata[int(neibs_e[k])]);
      qsis[k] = k;
    }
    QuickSortI(ntraces, qsis, [&] (const auto & i, const auto & j) LAMBDA_INLINE { return i > j; });
    for (auto j : Range(ntraces)) {
      auto indN = qsis[j];
      auto N = neibs_v[indN];
      if ( may_check_neib(N) ) {
        dummyb = N;
        auto soc = CalcSOC (v, agg.members(), aggd,
                N, dummyb, repl_diag[N],
                true); // neib boost probably worth it ...
        if (soc > MIN_ECW) {
          first_N_ind = indN;
          v_to_agg[N] = agg_nr;
          dist2agg[N] = 1;
          marked.SetBit(N);
          add_v_to_agg (agg, N);
          cnt_mems++;
          break;
        }
      }
    }
    if (first_N_ind != -1) { // if we could not add ANY neib, nohing has changed
      /** We perform a greedy strategy: Keep adding neighbours have the highest number of connections
          leading into the aggregate. If not dist2, only check neighbours common to the two first verts in the aggregate,
          otherwise check all neibs of v.
          Its not perfect - if i check a neib, and do not add it, but later on add a common neib, i could potentially
          have added it in the first place. On the other hand - di I WANT to re-check all the time?
          Can I distinguish between "weak" and "singular" connections? Then I could only re-check the singular ones? **/
      int qss = neibs_v.Size();
      neib_ecnt.SetSize0(); neib_ecnt.SetSize(qss); neib_ecnt = 1; // per definition every neighbour has one edge
      qsis.SetSize0(); qsis.SetSize(qss); qsis = -1; qss = 0;
      for (auto j : Range(neibs_v))
        if (!may_check_neib(neibs_v[j]))
          { neib_ecnt[j] = -1; }
        else
          { qsis[qss++] = j; }
      int first_valid_ind = 0;
      /** inc edge count for common neibs of v and first meber**/
      iterate_intersection(econ.GetRowIndices(neibs_v[first_N_ind]), neibs_v,
                [&](auto i, auto j) {
                  if (neib_ecnt[j] != -1)
              { neib_ecnt[j]++; }
                });
      /**  lock out all neibs of v that are not neibs of N **/
      int inc_fv = 0;
      if ( !dist2 ) {
        iterate_anotb(neibs_v, econ.GetRowIndices(neibs_v[first_N_ind]),
          [&](auto inda) LAMBDA_INLINE {
            if (neib_ecnt[inda] != -1) {
              neib_ecnt[inda] = -1;
              inc_fv++;
            }
          });
      }
      QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });
      first_valid_ind += inc_fv;
      qss -= inc_fv;
      while(qss>0) {
        auto n_ind = qsis[first_valid_ind + qss - 1];
        auto N = neibs_v[n_ind]; dummyb = N;
        auto soc = CalcSOC (v, agg.members(), aggd,
                N, dummyb, repl_diag[N],
                true); // neib boost should not be necessary
        qss--; // done with this neib
        neib_ecnt[n_ind] = -1; // just to check if i get all - i should not access this anymore anyways
        if (soc > MIN_ECW) { // add N to agg, update edge counts and re-sort neibs
          v_to_agg[N] = agg_nr;
          dist2agg[N] = 1;
          marked.SetBit(N);
          add_v_to_agg (agg, N); cnt_mems++;
          iterate_intersection(econ.GetRowIndices(N), neibs_v, // all neibs of v and N now have an additional edge into the agglomerate
              [&](auto inda, auto indb) LAMBDA_INLINE {
                if (neib_ecnt[indb] != -1)
                  { neib_ecnt[indb]++; }
              });
          QuickSort(qsis.Part(first_valid_ind, qss), [&](auto i, auto j) { return neib_ecnt[i] < neib_ecnt[j]; });
        }
      } // while(qss)
    } // first_N_ind != -1
    if ( force || (cnt_mems >= MIN_NEW_AGG_SIZE) )
      { return true; }
    else { // remove the aggregate again -
      for (auto M : agg.members()) {
        v_to_agg[M] = -1;
        dist2agg[M] = -1;
        marked.Clear(M);
      }
      agglomerates.SetSize(agg_nr);
      agg_diag.SetSize(agg_nr);
      return false;
    }
  }; // init_agglomerate (..)


  /** Should I make a new agglomerate around v? **/
  auto check_v = [&](auto v) LAMBDA_INLINE {
    auto myeq = M.template GetEQCOfNode<NT_VERTEX>(v);
    if ( marked.Test(v) ) // any "inactive" (=ghost/minion) vert would be marked
      { return false; }
    auto neibs = econ.GetRowIndices(v);
    for (auto n : neibs) { // any vert in an agg must be solid, no need to check is_mine
      if ( (marked.Test(n)) && eqa_to_eqb(myeq, M.template GetEQCOfNode<NT_VERTEX>(n)) )  {
        auto n_agg_nr = v_to_agg[n];
        if (n_agg_nr != -1) // can still do this later ...
          { return false; }
      }
    }
    return init_agglomerate(v, myeq, false);
  };

  t1.Start();

  /** Deal with dirichlet vertices **/
  auto free_verts = this->GetFreeVerts();
  if (free_verts != nullptr)
  {
    const auto & fvs = *free_verts;
    for (auto k : Range(M.template GetNN<NT_VERTEX>())) {
      if (!fvs.Test(k))
        { marked.SetBit(k); }
    }
  }

  /** Calc replacement matrix diagonals **/
  tdiag.Start();

  M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
    ModQs(vdata[edge.v[0]], vdata[edge.v[1]], Qij, Qji);
    const auto & em = edata[edge.id];
    AddQtMQ(1.0, repl_diag[edge.v[0]], Qij, em);
    AddQtMQ(1.0, repl_diag[edge.v[1]], Qji, em);
  }, true); // only master, we cumulate this afterwards
  M.template AllreduceNodalData<NT_VERTEX>(repl_diag, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

  tdiag.Stop();


  /** Calc initial edge weights. TODO: I dont think i need this anymore at all! **/
  tecw.Start();
  M.template Apply<NT_EDGE>([&](const auto & edge) {
    dummya = edge.v[0]; dummyb = edge.v[1];
    // ecw[edge.id] = CalcSOC(edge.v[0], dummya, repl_diag[edge.v[0]],
    // 		       edge.v[1], dummyb, repl_diag[edge.v[1]],
    // 		       true);
    ecw[edge.id] = 1;
  });
  tecw.Stop();

  size_t n_strong_e = 0;
  M.template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_strong_e++; } }, true);
  double s_e_per_v = (M.template GetNN<NT_VERTEX>() == 0) ? 0 : 2 * double(n_strong_e) / double(M.template GetNN<NT_VERTEX>());
  size_t approx_nagg = max2(size_t(1), size_t(NV / ( 1 + s_e_per_v )));
  agglomerates.SetSize(1.2 * approx_nagg); agglomerates.SetSize0();
  agg_diag.SetSize(1.2 * approx_nagg); agg_diag.SetSize0();

  t1.Stop(); t2.Start();

  if constexpr(false)
  {
    /** Iterate through vertices and start new agglomerates if the vertex is at least at distance 1 from
        any agglomerate (so distance 2 from any agg. center) **/
    Array<int> vqueue(M.template GetNN<NT_VERTEX>()); vqueue.SetSize0();
    BitArray checked(M.template GetNN<NT_VERTEX>()); checked.Clear();
    BitArray queued(M.template GetNN<NT_VERTEX>()); queued.Clear();
    size_t nchecked = 0; const auto NV = M.template GetNN<NT_VERTEX>();
    int cntq = 0, cntvr = NV-1;
    int vnum;
    while (nchecked < NV) {
      /** if there are any queued vertices, handle them first, otherwise take next vertex by reverse counting
          ( ex-vertices have higher numbers) **/
      bool from_q = cntq < vqueue.Size();
      if (from_q)
        { vnum = vqueue[cntq++]; }
      else
        { vnum = cntvr--; }
      if (!checked.Test(vnum)) {
        // cout << " from queue ? " << from_q << endl;
        bool newagg = check_v(vnum);
        checked.SetBit(vnum);
        nchecked++;
        if (newagg) { // enqueue neibs of neibs of the agg (if they are not checked, queued or at least marked yet)
          const auto & newagg = agglomerates.Last();
          int oldqs = vqueue.Size();
          for (auto mem : newagg.members()) {
            auto mem_neibs = econ.GetRowIndices(mem);
            for (auto x : mem_neibs) {
        auto y = econ.GetRowIndices(x);
        for (int jz = int(y.Size()) - 1; jz>= 0; jz--) { // enqueue neibs of neibs - less local ones first
          auto z = y[jz];
          if (  (!marked.Test(z)) && (!checked.Test(z)) && (!queued.Test(z)) )
            { vqueue.Append(z); queued.SetBit(z); }
        }
            }
          }
          int newqs = vqueue.Size();
          if (newqs > oldqs + 1) // not sure ?
            { QuickSort(vqueue.Part(oldqs, (newqs-oldqs)), [&](auto i, auto j) { return i>j; }); }
        }
      }
    }
  }
  else {
    /** Iterate through vertices by ascending #of edges. **/
    size_t maxedges = 0;
    for (int k = econ.Height()-1; k>=0; k--)
      { maxedges = max2(maxedges, econ.GetRowIndices(k).Size()); }
    TableCreator<int> cvb(1+maxedges);
    for ( ; !cvb.Done(); cvb++ )
      for (int k = econ.Height()-1; k>=0; k--)
        { cvb.Add(econ.GetRowIndices(k).Size(), k); }
          auto vbuckets = cvb.MoveTable();
          for (auto vnum : vbuckets.AsArray())
            { check_v(vnum); }
  }



  if (this->print_aggs) {
    //   cout << endl << " FIRST loop done " << endl;
    cout << "frac marked: " << double(marked.NumSet()) / marked.Size() << endl;
    cout << " INTERMED agglomerates : " << agglomerates.Size() << endl;
    cout << agglomerates << endl;
    //  Array<int> naggs;
    //  auto resize_to = [&](auto i) {
    // 	auto olds = naggs.Size();
    // 	if (olds < i) {
    // 	  naggs.SetSize(i);
    // 	  for (auto j : Range(olds, i))
    // 	    { naggs[j] = 0; }
    // 	}
    //  };
    //  for (const auto & agg : agglomerates) {
    // // 	auto ags = agg.members().Size();
    // 	resize_to(1+ags);
    // 	naggs[ags]++ ;
    //  }
    //  cout << " INTERMED agg size distrib: "; prow2(naggs); cout << endl;
  }

  t2.Stop(); t3.Start();

  /** Assign left over vertices to some neighbouring agglomerate, or, if not possible, start a new agglomerate with them.
      Also try to weed out any dangling vertices. **/
  Array<int> neib_aggs(20), notake(20), index(20);
  Array<double> na_soc(20);
  M.template ApplyEQ<NT_VERTEX>([&] (auto eqc, auto v) LAMBDA_INLINE {
    // TODO: should do this twice, once with max_dist 1, once 2 (second round, steal verts from neibs ?)
    if ( marked.Test(v) )
      { return; }
    if ( use_sv && (!is_mine_eq(v, eqc)) )
      { return; }
    auto neibs = econ.GetRowIndices(v);
    if (!marked.Test(v)) {
      if (neibs.Size() == 1) {  // Check for hanging vertices
        // Collect neighbouring agglomerates we could append ourselfs to
        auto N = neibs[0];
        auto neib_agg_id = v_to_agg[neibs[0]];
        if (neib_agg_id != -1) { // neib is in an agglomerate - must be active
          auto & neib_agg = agglomerates[neib_agg_id];
          auto N_eqc = M.template GetEQCOfNode<NT_VERTEX>(N);
          bool can_add = eqa_to_eqb(eqc, N_eqc);
          if (can_add) {
            auto soc = CalcSOC_av (neib_agg, v, true);
            can_add &= (soc > MIN_ECW);
          }
          if (can_add) { // lucky!
            marked.SetBit(v);
            v_to_agg[v] = neib_agg_id;
            dist2agg[v] = 1 + dist2agg[N]; // must be correct - N is the only neib
            add_v_to_agg (neib_agg, v);
          }
          else // unfortunate - new single agg!
          { init_agglomerate(v, eqc, true); }
        } else { // neib is not in an agg - force start a new one at neib and add this vertex (which must be OK!)
          auto N_eqc = M.template GetEQCOfNode<NT_VERTEX>(N);
          // if ( (eqc_h.IsMasterOfEQC(N_eqc)) && (eqa_to_eqb(eqc, N_eqc) ) ) { // only if it is OK eqc-wise
          if ( is_mine(N) && eqa_to_eqb(eqc, N_eqc) ) { // have to check if neib is active!
            init_agglomerate(N, N_eqc, true); // have to force new agg even if we dont really want to ...
            auto new_agg_id = v_to_agg[N];
            if (!marked.Test(v)) { // might already have been added by init_agg, but probably not
              marked.SetBit(v);
              v_to_agg[v] = new_agg_id;
              dist2agg[v] = 1;
                add_v_to_agg(agglomerates[new_agg_id], v);
            }
          }
          else // unfortunate - new single agg!
            { init_agglomerate(v, eqc, true); }
        }
      }
      else {
        neib_aggs.SetSize0();
        notake.SetSize0();
        na_soc.SetSize0();
        for (auto n : neibs) {
          if ( (marked.Test(n)) && (dist2agg[n] <= (dist2 ? 2 : 1))) { // max_dist 2 maybe, idk ??
            auto agg_nr = v_to_agg[n];
            if (agg_nr != -1) { // check if neib is active
              auto & n_agg = agglomerates[agg_nr];
              if ( eqa_to_eqb(eqc, M.template GetEQCOfNode<NT_VERTEX>(n_agg.center())) &&
                    (!notake.Contains(agg_nr)) && (!neib_aggs.Contains(agg_nr)) ) {
                auto soc = CalcSOC_av(n_agg, v, true); // use neib_boost here - we want as few new aggs as possible
                if (soc > MIN_ECW) {
                  /** Desirability of adding v to A:
                      1 / (1 + |A|) * Sum_{k in A} alpha(e_vk)/dist[k]
                      So desirability grows when we have many strong connections to an agglomerate and decreases
                      the farther away from the center of A we are and the larger A already is. **/
                  intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
                  double mindist = 20;
                  double des = 0;
                  for (auto k : neibs_in_agg) {
                    des += ecw[int(econ(v,k))] / ( 1 + dist2agg[k] );
                    mindist = min2(mindist, double(dist2agg[k]));
                  }
                  des = (soc/MIN_ECW) / (mindist * n_agg.members().Size());
                  neib_aggs.Append(agg_nr);
                  na_soc.Append(des);
                }
                else // just so we do not compute SOC twice for no reason
                  { notake.Append(agg_nr); }
              }
            }
          }
        }
        if (neib_aggs.Size()) { // take the most desirable neib
          auto mi = ind_of_max(na_soc);
          auto agg_nr = neib_aggs[mi]; auto& n_agg = agglomerates[agg_nr];
          add_v_to_agg(n_agg, v);
          marked.SetBit(v);
          v_to_agg[v] = agg_nr;
          intersect_sorted_arrays(n_agg.members(), neibs, neibs_in_agg);
          int mindist = 1000;
          for (auto k : neibs_in_agg)
            { mindist = min2(mindist, dist2agg[k]); }
          dist2agg[v] = 1 + mindist;
        }
        else {
          init_agglomerate(v, eqc, true); // have to force new agg even if we dont really want to ...
          // Maybe we should check if we should steal some vertices from surrounding aggs, at least those with
          // dist >= 2. Maybe init_agg(v, eqc, steal=true) ??
        }
      } // neibs.Size > 1
    } // if (!marked.Test(v))
  }, !use_sv); // also only master verts!

  t3.Stop();

  if (fixed_aggs.Size()) {
    auto ags = agglomerates.Size();
    agglomerates.SetSize(ags + fixed_aggs.Size());
    agglomerates.SetSize(ags);
    for (auto k : Range(fixed_aggs.Size())) {
      auto fagg = fixed_aggs[k];
      auto agg_nr = agglomerates.Size();
      agglomerates.Append(Agglomerate(fagg[0], agg_nr)); // TODO: does this do an allocation??
      v_to_agg[fagg[0]] = agg_nr;
      auto& agg = agglomerates.Last();
      for (auto k : Range(size_t(1), fagg.Size())) {
        v_to_agg[fagg[k]] = agg_nr;
        agg.AddSort(fagg[k]);
      }
    }
  }

  if (this->print_aggs) {
    cout << " FINAL agglomerates : " << agglomerates.Size() << endl;
    cout << agglomerates << endl;
    cout << endl;
  }
} // MISAgglomerator::FormAgglomerates_impl


template<class ENERGY, class TMESH, bool ROBUST>
void MISAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  if constexpr (ROBUST) {
    if (this->robust_crs) /** cheap, but not robust for some corner cases **/
      { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
    else /** (much) more expensive, but also more robust **/
      { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
  }
  else // do not even compile the robust version - saves a lot of ti
    { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
} // MISAgglomerator::FormAgglomerates

} // namespace amg

#endif // MIS_AGG

#endif // FILE_MIS_AGG_IMPL_HPP
