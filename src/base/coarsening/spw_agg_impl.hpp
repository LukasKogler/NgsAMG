#ifndef FILE_SPW_AGG_IMPL_HPP
#define FILE_SPW_AGG_IMPL_HPP

#ifdef SPW_AGG

#include <utils.hpp>
#include <utils_denseLA.hpp>
#include <utils_arrays_tables.hpp>
#include <utils_io.hpp>

#include "spw_agg.hpp"

namespace amg
{

template<int N>
INLINE void print_rank(string name, const Mat<N,N,double> & A)
{
  static LocalHeap lh ( 5 * 9 * sizeof(double) * N * N, "mmev", false); // about 5 x used mem
  HeapReset hr(lh);
  FlatMatrix<double> a(N, N, lh); a = A;
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(a, evals);
  int neg = 0, pos = 0, rk = 0;
  double eps = 1e-10 * fabs(evals(N-1));
  for (auto k :Range(evals))
    if (evals(k) > eps)
  { pos++; rk++; }
    else if (evals(k) < -eps)
  { neg++; rk++; }
  // cout << " trace of " << name << " = " << calc_trace(A) << endl;
  cout << " rank of " << name << " = " << rk << " of " << N << ", pos = " << pos << ", neg = " << neg << ", min = " << evals(0) << ", max = " << evals(N-1) << ", all evals = "; prow(evals); cout << endl;
              // cout << " all evals are = "; prow(evals); cout << endl;
}

INLINE void print_rank(string name, FlatMatrix<double> A, LocalHeap & lh, ostream & os)
{
  HeapReset hr(lh);
  int N = A.Height();
  FlatMatrix<double> a(N, N, lh); a = A;
  FlatVector<double> evals(N, lh);
  LapackEigenValuesSymmetric(a, evals);
  int neg = 0, pos = 0, rk = 0;
  double eps = 1e-10 * fabs(evals(N-1));
  double minpos = fabs(evals(N-1));
  for (auto k :Range(evals))
    if (evals(k) > eps)
  { pos++; rk++; minpos = min(evals(k), minpos); }
    else if (evals(k) < -eps)
  { neg++; rk++; }
  // os << " trace of " << name << " = " << calc_trace(A) << endl;
  // os << " rank of " << name << " = " << rk << " of " << N << ", pos = " << pos << ", neg = " << neg << ", min = " << evals(0) << ", max = " << evals(N-1) << ", all evals = "; prow(evals); cout << endl;
  os << " rank of " << name << " = " << rk << " of " << N << ", pos = " << pos << ", neg = " << neg << ", min = " << evals(0) << ", minpos = " << minpos << ", max = " << evals(N-1) << ", all evals = " << endl;
              // cout << " all evals are = "; prow(evals); cout << endl;
}

INLINE void print_rank(string name, FlatMatrix<double> A, LocalHeap & lh)
{
  print_rank(name, A, lh, cout);
}

/** for some reaason, ENERGY::Calc/Add stuff segfaults. Actually, these segfault too?? **/

template<class ENERGY>
static INLINE void CalcMQ (double scal, const typename ENERGY::TM & M, const typename ENERGY::TM & Qij, typename ENERGY::TM & out)
{
  /** I   0    A  B   =     A        B
  QT s*I   BT C   =  QTA+s*BT QTB+s*C **/
  static Mat<ENERGY::DISPPV, ENERGY::ROTPV, double> QTA;
  static Mat<ENERGY::ROTPV, ENERGY::ROTPV, double> QTB;
  const double s = Qij(ENERGY::DISPPV, ENERGY::DISPPV);
  QTA = Trans(MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(Qij)) * MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(M);
  QTB = Trans(MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(Qij)) * MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(M);
  MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(out) = scal * MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(M);
  MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(out) = scal * MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(M);
  MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, 0, ENERGY::DISPPV>(out) = scal * ( (s*MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, 0, ENERGY::DISPPV>(M)) + QTA );
  MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, ENERGY::DISPPV, ENERGY::ROTPV>(out) = scal * ( (s*MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, ENERGY::DISPPV, ENERGY::ROTPV>(M)) + QTB);
}

template<class ENERGY>
static INLINE void AddMQ (double scal, const typename ENERGY::TM & M, const typename ENERGY::TM & Qij, typename ENERGY::TM & out)
{
  /** I  0   A  B   =    A      B
  QT I   BT C   =  QTA+BT QTB+C **/
  static Mat<ENERGY::DISPPV, ENERGY::ROTPV, double> QTA;
  static Mat<ENERGY::ROTPV, ENERGY::ROTPV, double> QTB;
  const double s = Qij(ENERGY::DISPPV, ENERGY::DISPPV);
  QTA = Trans(MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(Qij)) * MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(M);
  QTB = Trans(MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(Qij)) * MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(M);
  MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(out) += scal * MakeFlatMat<0, ENERGY::DISPPV, 0, ENERGY::DISPPV>(M);
  MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(out) += scal * MakeFlatMat<0, ENERGY::DISPPV, ENERGY::DISPPV, ENERGY::ROTPV>(M);
  MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, 0, ENERGY::DISPPV>(out) += scal * ( (s*MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, 0, ENERGY::DISPPV>(M)) + QTA );
  MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, ENERGY::DISPPV, ENERGY::ROTPV>(out) += scal * ( (s*MakeFlatMat<ENERGY::DISPPV, ENERGY::ROTPV, ENERGY::DISPPV, ENERGY::ROTPV>(M)) + QTB);
}

template<class T>
INLINE void print_tm (ostream &os, int prec, const T & mat)
{
  constexpr int H = mat_traits<T>::HEIGHT;
  constexpr int W = mat_traits<T>::WIDTH;
  for (int kH : Range(H)) {
    for (int jW : Range(W)) { os << setprecision(prec) << mat(kH,jW) << " "; }
    os << endl;
  }
}
template<> INLINE void print_tm (ostream &os, int prec, const double & mat) { os << setprecision(prec) << mat << endl; }


/** LocCoarseMap **/

class LocCoarseMap : public BaseCoarseMap
{

  template<class ATENERGY, class ATMESH, bool AROBUST> friend class SPWAgglomerator;

public:

  LocCoarseMap (shared_ptr<TopologicMesh> mesh, shared_ptr<TopologicMesh> mapped_mesh = nullptr)
    : BaseCoarseMap(mesh, mapped_mesh)
  {
    const auto & M = *mesh;
    Iterate<4>([&](auto k) {
    NN[k] = M.template GetNN<NODE_TYPE(k.value)>();
    node_maps[k].SetSize(NN[k]);
  });
  }

  FlatArray<int> GetV2EQ () const { return cv_to_eqc; }
  void SetV2EQ (Array<int> && _cv_to_eqc) { cv_to_eqc = _cv_to_eqc; }

  size_t & GetNCV () { return mapped_NN[NT_VERTEX]; }

  void FinishUp ()
  {
    // cout << " FinishUp, NCV = " << GetMappedNN<NT_VERTEX>();
    // cout << " NCV 2 = " << GetNCV() << endl;

    static Timer t("LocCoarseMap::FinishUp"); RegionTimer rt(t);
    /** mapped_mesh **/
    auto eqc_h = mesh->GetEQCHierarchy();
    auto cmesh = make_shared<TopologicMesh>(eqc_h, GetMappedNN<NT_VERTEX>(), 0, 0, 0); // NOTE: crs edges are only built below
    // NOTE: verts are never set, they are unnecessary because we dont build coarse BlockTM!
    /** crs vertex eqcs **/
    auto pbtm = dynamic_pointer_cast<BlockTM>(mesh);
    auto c2fv = GetMapC2F<NT_VERTEX>();
    if (pbtm != nullptr) { // we are the first map - construct cv_to_eqc. for other maps, concatenate it!
      const auto & btm = *pbtm;
      cv_to_eqc.SetSize(GetMappedNN<NT_VERTEX>());
      // cout << " FUP c2fv " << endl << c2fv << endl;
      for(auto cvnr : Range(cv_to_eqc)) {
        auto fvs = c2fv[cvnr];
        if (fvs.Size() == 1)
        { cv_to_eqc[cvnr] = btm.GetEQCOfNode<NT_VERTEX>(fvs[0]); }
        else {
        // NOTE: this only works as long as i only go up/down eqc-hierararchy
        int eqa = btm.GetEQCOfNode<NT_VERTEX>(fvs[0]), eqb = btm.GetEQCOfNode<NT_VERTEX>(fvs[1]);
        // cout << fvs[0] << " " << fvs[1] << " -> " << cenr << ", eqs " << eqa << " " << eqb << ", isleq " << eqc_h->IsLEQ(eqa, eqb) << endl;
        // cv_to_eqc = eqc_h->IsLEQ(eqa, eqb) ? eqb : eqa;
        cv_to_eqc[cvnr] = (eqa == 0) ? eqb : ( (eqb == 0) ? eqa : (eqc_h->IsLEQ(eqa, eqb) ? eqb : eqa) );
        }
      }
    // cout << " FIRST cv2eq = "; prow2(cv_to_eqc); cout << endl;
    }
    /** crs edge connectivity and edges **/
    const auto & fecon = *mesh->GetEdgeCM();
    auto vmap = GetMap<NT_VERTEX>();
    TableCreator<int> ccg(GetMappedNN<NT_VERTEX>());
    Array<int> vneibs(50);
    for (; !ccg.Done(); ccg++)
      for (auto cvnr : Range(c2fv)) {
        vneibs.SetSize0();
        for (auto fvnr : c2fv[cvnr])
        for (auto vneib : fecon.GetRowIndices(fvnr)) {
          int cneib = vmap[vneib];
          if ( (cneib != -1) &&  ( cneib != cvnr) )
        { insert_into_sorted_array_nodups(cneib, vneibs); }
        }
        ccg.Add(cvnr, vneibs);
      }
    auto graph = ccg.MoveTable();
    // cout << " cecon graph " << endl << graph << endl;
    Array<int> perow(graph.Size());
    for (auto k : Range(perow))
      { perow[k] = graph[k].Size(); }
    auto pcecon = make_shared<SparseMatrix<double>>(perow, GetMappedNN<NT_VERTEX>());
    const auto & cecon = *pcecon;
    size_t & cnt = mapped_NN[NT_EDGE]; cnt = 0;
    auto & cedges = cmesh->edges;
    cedges.SetSize(graph.AsArray().Size()/2);  // every edge is counted twice
    for (int k : Range(cecon)) {
      auto ris = cecon.GetRowIndices(k);
      ris = graph[k];
      auto rvs = cecon.GetRowValues(k);
      for (auto l : Range(ris)) {
        if (k < ris[l])
        { int cid = cnt++; rvs[l] = cid; cedges[cid].v = { k, ris[l] }; cedges[cid].id = cid; }
        else
        { rvs[l] = cecon(ris[l], k); }
      }
    }
    // cout << " cnt " << cnt << ", mappednn " << mapped_NN[NT_EDGE] << " " << GetMappedNN<NT_EDGE>() << endl;
    cmesh->econ = pcecon;
    cmesh->nnodes[NT_EDGE] = GetMappedNN<NT_EDGE>();
    // cout << " cecon " << endl << cecon << endl;
    /** edge map **/
    auto fedges = mesh->template GetNodes<NT_EDGE>();
    auto & emap = node_maps[NT_EDGE]; emap.SetSize(GetNN<NT_EDGE>());
    // cout << " FU fedges " << fedges.Size() << " " << GetNN<NT_EDGE>() << " " << mesh->template GetNN<NT_EDGE>() << endl;
    // cout << " MAPPED " << GetMappedNN<NT_EDGE>() << endl;
    for (auto fenr : Range(emap)) {
      auto & edge = fedges[fenr];
      int cv0 = vmap[edge.v[0]], cv1 = vmap[edge.v[1]];
      if ( (cv0 != -1) && (cv1 != -1) && (cv0 != cv1) )
        { emap[fenr] = cecon(cv0, cv1); }
      else
        { emap[fenr] = -1; }
    }
    // cout << " FU emap " << emap.Size() << " "; prow2(emap); cout << endl;
    this->mapped_mesh = cmesh;
    // actually, I never need c_solid_verts!
    // if (f_solid_verts) {
      // c_solid_verts = make_shared<BitArray>(cmesh->template GetNN<NT_VERTEX>());
      // c_solid_verts->Clear();
      // for (auto k : Range(vmap))
    // if (vmap[k] != -1)
    // { c_solid_verts->SetBit(vmap[k]); }
    // }
    if (f_allowed_edges) {
      c_allowed_edges = make_shared<BitArray>(cmesh->template GetNN<NT_EDGE>());
      c_allowed_edges->Clear();
      for (auto k : Range(fedges))
        if ( (emap[k] != -1) && (f_allowed_edges->Test(k)) )
        { c_allowed_edges->SetBit(emap[k]); }
    }
  } // FinishUp

  virtual shared_ptr<LocCoarseMap> ConcatenateLCM (shared_ptr<LocCoarseMap> right_map)
  {
    static Timer t("LocCoarseMap::ConcatenateLCM(cmap)"); RegionTimer rt(t);
    /** with concatenated vertex/edge map **/
    auto concmap = make_shared<LocCoarseMap>(this->mesh, right_map->mapped_mesh);
    SetConcedMap(right_map, concmap);
    // auto concmap = BaseCoarseMap::Concatenate(right_map);
    /** concatenated aggs! **/
    const size_t NCV = right_map->GetMappedNN<NT_VERTEX>();
    FlatTable<int> aggs1 = GetMapC2F<NT_VERTEX>(), aggs2 = right_map->GetMapC2F<NT_VERTEX>();
    // cout << " ConcatenateLCM, aggs1 = " << endl << aggs1 << endl;
    // cout << " ConcatenateLCM, aggs2 = " << endl << aggs2 << endl;
    TableCreator<int> ct(NCV);
    for (; !ct.Done(); ct++)
      for (auto k : Range(NCV))
        for (auto v : aggs2[k])
          { ct.Add(k, aggs1[v]); }
    concmap->rev_node_maps[NT_VERTEX] = ct.MoveTable();
    // cout << " ConcatenateLCM, aggs3 = " << endl << concmap->rev_node_maps[NT_VERTEX] << endl;
    /** coarse vertex->eqc mapping - right map does not have it yet (no BTM to construct it from) **/
    auto eqc_h = mesh->GetEQCHierarchy();
    auto & cv2eq = concmap->cv_to_eqc; cv2eq.SetSize(NCV);
    for (auto k : Range(NCV)) {
      auto agg = aggs2[k];
      if (agg.Size() == 1)
        { cv2eq[k] = cv_to_eqc[agg[0]]; }
      else {
        int eqa = cv_to_eqc[agg[0]], eqb = cv_to_eqc[agg[1]];
        cv2eq[k] = (eqa == 0) ? eqb : ( (eqb == 0) ? eqa : (eqc_h->IsLEQ(eqa, eqb) ? eqb : eqa) );
      }
    }
    // concmap->f_solid_verts = f_solid_verts;
    // concmap->c_solid_verts = right_map->c_solid_verts;
    concmap->f_allowed_edges = f_allowed_edges;
    concmap->c_allowed_edges = right_map->c_allowed_edges;
    // cout << " CONC cv2eq = "; prow2(cv2eq); cout << endl;
    return concmap;
  } // ConcatenateLCM

  void Concatenate (size_t NCV, FlatArray<int> rvmap)
  {
    static Timer t("LocCoarseMap::Concatenate(vmap)"); RegionTimer rt(t);
    /** no mesh on coarse level **/
    this->mapped_mesh = nullptr;
    /** no edges on coarse level **/
    mapped_NN[NT_VERTEX] = NCV;
    mapped_NN[NT_EDGE] = 0;
    /** concatenate vertex map **/
    auto & vmap = node_maps[NT_VERTEX];
    for (auto & val : vmap)
      { val = (val == -1) ? val : rvmap[val]; }
    /** set up reverse vertex map **/
    // cout << " Concatenate vmap, r vmap = "; prow2(rvmap); cout << endl;
    // cout << " orig aggs = " << endl << rev_node_maps[NT_VERTEX] << endl << endl;
    auto & aggs = rev_node_maps[NT_VERTEX];
    TableCreator<int> ct(NCV);
    for (; !ct.Done(); ct++)
      for (auto k : Range(aggs))
        if (rvmap[k] != -1)
        { ct.Add(rvmap[k], aggs[k]); }
    rev_node_maps[NT_VERTEX] = ct.MoveTable();
    // cout << endl << " new aggs = " << endl << rev_node_maps[NT_VERTEX] << endl;
    /** no vertex->eqc mapping on coarse level **/
    cv_to_eqc.SetSize(NCV); cv_to_eqc = -1;
  } // Concatenate(array)

  INLINE int CV2EQ (int v) const { return cv_to_eqc[v]; }

  protected:

  Array<int> cv_to_eqc; /** (coarse) vert->eqc mapping **/

  /** generates coarse level allowed edes IF FINE LEVEL is set before FinishUp **/
  shared_ptr<BitArray> f_solid_verts, c_solid_verts;
  /** non-solid verts are set to "handled" below anyways! **/
  shared_ptr<BitArray> f_allowed_edges = nullptr, c_allowed_edges = nullptr;


  public:
  void SetAllowedFEdges (shared_ptr<BitArray> _f_allowed_edges) { f_allowed_edges = _f_allowed_edges; }
  shared_ptr<BitArray> GetAllowedFEdges () const { return f_allowed_edges; }
  shared_ptr<BitArray> GetAllowedCEdges () const { return c_allowed_edges; }

  void SetSolidFVerts (shared_ptr<BitArray> _f_solid_verts) { f_solid_verts = _f_solid_verts; }
  shared_ptr<BitArray> GetSolidFVerts () const { return f_solid_verts; }
  shared_ptr<BitArray> GetSolidCVerts () const { return c_solid_verts; }

}; // class LocCoarseMap

/** END LocCoarseMap **/


/** SPWAgglomerator **/


template<class ENERGY, class TMESH, bool ROBUST>
SPWAgglomerator<ENERGY, TMESH, ROBUST> :: SPWAgglomerator (shared_ptr<TMESH> mesh)
  :	Agglomerator<TMESH>(mesh)
{
  assert(mesh != nullptr); // obviously this would be bad
} // SPWAgglomerator(..)


template<class ENERGY, class TMESH, bool ROBUST>
void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: Initialize (const SPWAggOptions & opts, int level)
{
	num_rounds = opts.spw_rounds.GetOpt(level);
	neib_boost = opts.spw_neib_boost.GetOpt(level);
	allrobust = opts.spw_allrobust.GetOpt(level);
	bdiag = opts.spw_cbs_bdiag.GetOpt(level);
	pick_cw_type = opts.spw_pick_cwt.GetOpt(level);
	pick_mma_scal = opts.spw_pick_mma_scal.GetOpt(level);
	pick_mma_mat = opts.spw_pick_mma_mat.GetOpt(level);
	check_cw_type = opts.spw_check_cwt.GetOpt(level);
	check_mma_scal = opts.spw_check_mma_scal.GetOpt(level);
	check_mma_mat = opts.spw_check_mma_mat.GetOpt(level);
	checkbigsoc = opts.spw_cbs.GetOpt(level);
	cbs_spd_hack = opts.spw_cbs_spd_hack.GetOpt(level);
	simple_checkbigsoc = !opts.spw_cbs_robust.GetOpt(level);
	print_params = opts.spw_print_params.GetOpt(level);
	print_summs = opts.spw_print_summs.GetOpt(level);
	weed_out = opts.spw_wo.GetOpt(level);
	all_cmk = opts.spw_cmk.GetOpt(level);
} // SPWAgglomerator<ENERGY, TMESH, ROBUST>::SetLevelSPWOptions


template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
INLINE void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data)
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
} // SPWAgglomerator::GetEdgeData


template<class ENERGY, class TMESH, bool ROBUST> template<class TFULL, class TAPPROX>
INLINE void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: GetApproxEdgeData (FlatArray<TFULL> in_data, Array<TAPPROX> & out_data)
{
  if constexpr(std::is_same<TFULL, TAPPROX>::value)
  {
    out_data.FlatArray<TAPPROX>::Assign(in_data);
  }
  else
  {
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = calc_trace(in_data[k]); }
  }

} // SPWAgglomerator::GetApproxEdgeData


template<class ENERGY, class TMESH, bool ROBUST>
void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  if constexpr (ROBUST)
  {
    if (this->robust_crs) /** cheap, but not robust for some corner cases **/
      { FormAgglomerates_impl<TM> (agglomerates, v_to_agg); }
    else /** (much) more expensive, but also more robust **/
      { FormAgglomerates_impl<double> (agglomerates, v_to_agg); }
  }
  else // do not even compile the robust version - saves a lot of ti
  {
    FormAgglomerates_impl<double> (agglomerates, v_to_agg);
  } 
} // SPWAgglomerator::FormAgglomerates


template<class ENERGY, class TMESH, bool ROBUST> Timer<TTracing, TTiming>& GetRoundTimer (int round) {
static Array<Timer<TTracing, TTiming>> timers ( { Timer("FormAggloemrates - round 0"),
	Timer("FormAggloemrates - round 1"),
	Timer("FormAggloemrates - round 2"),
	Timer("FormAggloemrates - round 3"),
	Timer("FormAggloemrates - round 4"),
	Timer("FormAggloemrates - round 5"),
	Timer("FormAggloemrates - rounds > 5") } );
return timers[min(5, round)];
} // GetRoundTimer

template<class ENERGY, class TMESH, bool ROBUST> template<class TMU>
INLINE void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg)
{
  static_assert ( (std::is_same<TMU, TM>::value || std::is_same<TMU, double>::value), "Only 2 options make sense!");

  static Timer t("SPWAgglomerator::FormAgglomerates_impl"); RegionTimer rt(t);
  static Timer tfaggs("FormAgglomerates - finalize aggs");
  static Timer tmap("FormAgglomerates - map loc mesh");
  static Timer tprep("FormAgglomerates - prep");
  static Timer tsv("FormAgglomerates - spec verts");
  static Timer tvp("FormAgglomerates - pair verts");
  static Timer tassb("FormAgglomerates - ass.block");
  static Timer tassbs("FormAgglomerates - ass.block.simple");
  static Timer tbste("FormAgglomerates - boost edge");
  static Timer tdangle("FormAgglomerates - dangling verts");
  tprep.Start();

  // auto tm_mesh = GetMesh();
  // const TMESH & M = *tm_mesh; M.CumulateData();
  TMESH &ncMesh = this->GetMesh();
  const TMESH &M = this->GetMesh(); M.CumulateData();
  const auto & eqc_h = *M.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();
  const auto NV = M.template GetNN<NT_VERTEX>();
  const auto NE = M.template GetNN<NT_EDGE>();
  const auto & base_econ = *M.GetEdgeCM();

  const bool print_aggs = this->print_aggs;    // actual aggs
  const bool print_summs =  ( this->print_aggs || this->print_summs );   // summary info
  const bool print_params =  ( this->print_summs || this->print_params );  // parameters for every round
  const bool havelocinfo = (eqc_h.GetCommunicator().Size() == 1) || (eqc_h.GetCommunicator().Rank() != 0);

  // what? why, what posible use can this have? probably left over from a debugging/experimenting session??
  int abs = (eqc_h.GetCommunicator().Rank() == 0) ? 1 : 0;
  eqc_h.GetCommunicator().NgMPI_Comm::Bcast(abs, 0);
  const bool m_print_summs = abs;

  const int num_rounds = this->num_rounds;
  const bool bdiag = this->bdiag;
  const bool cbs_spd_hack = this->cbs_spd_hack;

  constexpr int BS = ngbla::Height<TM>();
  constexpr int BSU = ngbla::Height<TMU>();
  constexpr bool robust = (BS == BSU) && (BSU > 1);

  if (print_params) {
    ;
  }

  /** Calc trace of sum of off-proc contrib to diagonal. use this as additional trace to take the max over for "mmx" CWT.
  Only need to do this if the mesh is parallel, and if we are actually using "mmx" anywhere.
  This !!IS MORE RESTRICTIVE!! than normal "mmx", because we take the sum off off-proc connections.
  We cannot calc max off-proc connection for every vertex and use max of those on coarse level.
  To be exact, we would also need to properly map the off-proc edges, which sums up fine connections and then take
  the max trace of coarse off-proc connections. But we only want to map everything locally. Taking sum of off-proc entries
  and summing them up for the coarse level is easier, and becuase it is MORE restrictive, cannot deteriorate the condition!  **/
  bool any_mmx = (M.template GetNNGlobal<NT_VERTEX>() - M.template GetNN<NT_VERTEX>() > 0);
  if (any_mmx) {
    any_mmx = false;
    for (auto k : Range(num_rounds)) {
      if ( ( this->pick_cw_type.GetOpt(k) == MINMAX ) ||
          ( this->robust_crs && (!this->allrobust.GetOpt(k)) && (this->check_cw_type.GetOpt(k) == MINMAX) ) )
        { any_mmx = true; break; }
    }
  }
  const bool need_mtrod = any_mmx;

  const double MIN_ECW = this->edge_thresh;
  const double MIN_VCW = this->vert_thresh;

  if (havelocinfo && print_aggs) {
    if (M.template GetNN<NT_VERTEX>() < 200)
     { cout << " FMESH : " << endl << M << endl; }
  }

  // cout << " pcwt " << settings.pick_cw_type << endl;
  // cout << " pmmas " << settings.pick_mma_scal << endl;
  // cout << " pmmam " << settings.pick_mma_mat << endl;
  // cout << " ccwt " << settings.check_cw_type << endl;
  // cout << " cmmas " << settings.check_mma_scal << endl;
  // cout << " cmmam " << settings.check_mma_mat << endl;

  if (print_params) {
    cout << "SPWAgglomerator::FormAgglomerates_impl, params are: " << endl;
    cout << " ROBUST = " << robust << endl;
    cout << " num_rounds = " << num_rounds << endl;
    cout << " MIN_ECW = " << MIN_ECW << endl;
    cout << " MIN_VCW = " << MIN_VCW << endl;
  }

  shared_ptr<LocCoarseMap> conclocmap;

  FlatArray<TVD> base_vdata = get<0>(ncMesh.AttachedData())->Data();
  Array<TMU> base_edata_full; GetEdgeData<TMU>(get<1>(ncMesh.AttachedData())->Data(), base_edata_full);
  Array<double> base_edata; GetApproxEdgeData<TMU, double>(base_edata_full, base_edata);
  Array<TMU> base_diags(M.template GetNN<NT_VERTEX>()); base_diags = 0.0;
  Array<double> base_maxtrace(M.template GetNN<NT_VERTEX>()); base_diags = 0.0;

  TM Qij(0), Qji(0), contrib(0); SetIdentity(Qij); SetIdentity(Qji);

  M.template Apply<NT_EDGE>([&](const auto & edge) LAMBDA_INLINE {
      constexpr int rrobust = robust;
      const auto & em = base_edata_full[edge.id];
      if constexpr(rrobust) {
        ENERGY::ModQs(base_vdata[edge.v[0]], base_vdata[edge.v[1]], Qij, Qji);
        ENERGY::AddQtMQ(1.0, base_diags[edge.v[0]], Qij, em);
        ENERGY::AddQtMQ(1.0, base_diags[edge.v[1]], Qji, em);
      }
      else {
        base_diags[edge.v[0]] += em;
        base_diags[edge.v[1]] += em;
      }
    }, true); // only master, we cumulate this afterwards

  size_t mtrs = need_mtrod ? M.template GetNN<NT_VERTEX>() : 0;
  Array<double> base_mtrod(mtrs);

  if (need_mtrod) {
    for (auto k : Range(base_mtrod))
      { base_mtrod[k] = calc_trace(base_diags[k]); }
  }

  M.template AllreduceNodalData<NT_VERTEX>(base_diags, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });

  if (need_mtrod) {
    for (auto k : Range(base_mtrod))
      { base_mtrod[k] = (calc_trace(base_diags[k]) - base_mtrod[k]) / BS; }
  }

  // cout << " intermed base_diags: " << endl;
  // for (auto k : Range(base_diags)) {
  //   cout << "(" << k << "::" << calc_trace(base_diags[k]) << " " << calc_trace(base_diags[k])/BS << ") ";
  // }
  // cout << endl << endl;

  /** NOTE: We use l2 weights on both sides only for CBS. for pair-wise, l2 is only on diag. **/
  M.template Apply<NT_VERTEX>([&](auto v) {
    constexpr int rrobust = robust;
    if constexpr(rrobust)
      { base_diags[v] += ENERGY::GetVMatrix(base_vdata[v]); }
    else
      { base_diags[v] += ENERGY::GetApproxVWeight(base_vdata[v]); }
  }, false ); // everyone

  Array<TVD> cvdata;
  Array<TMU> cedata_full, cdiags;
  Array<double> cedata, cmtrod;
  FlatArray<TVD> fvdata; fvdata.Assign(base_vdata);
  FlatArray<TMU> fedata_full; fedata_full.Assign(base_edata_full);
  FlatArray<TMU> fdiags; fdiags.Assign(base_diags);
  FlatArray<double> fedata; fedata.Assign(base_edata);
  FlatArray<double> fmtrod; fmtrod.Assign(base_mtrod);

  shared_ptr<BitArray> curr_master_verts = this->GetSolidVerts();
  shared_ptr<BitArray> curr_allowed_edges = this->GetAllowedEdges();

  v_to_agg.SetSize(M.template GetNN<NT_VERTEX>()); v_to_agg = -1;

  // cout << " final base_diags: " << endl;
  // for (auto k : Range(base_diags)) {
  //   cout << "(" << k << "::" << calc_trace(fdiags[k]) << " " << calc_trace(fdiags[k])/BS << ") ";
  // }
  // cout << endl << endl;

  // cout << " fedata: "; prow2(fedata); cout << endl;
  // cout << endl << endl;

  // if (print_aggs) {
  //   cout << " FINE l2 wts; " << endl;
  //   for (auto k : Range(fvdata)) {
  // 	cout << k << " = " << fvdata[k] << endl;
  // 	print_tm(cout, ENERGY::GetVMatrix(fvdata[k])); cout << endl;
  //   }
  //   cout << endl << endl;
  // }

  /** 1.5 times min(150MB, max(20MB, max i need for cbs)) **/
  size_t lhs = 2.0 * min2(size_t(157286400),
        max2( size_t(20971520),
          size_t( max(pow(2*BS, num_rounds), 20.0) * 10 ) * sizeof(double) ) );

  // cout << " alloc localheap with size = " << lhs << ", in MBS = " << lhs/1024/1024 << endl;

  LocalHeap lh(lhs, "cthulu");

  /** check_pair/check_agg/mmax_scal/mmx_mat/l2/pair/noallowed **/
  IVec<7, int> rej; rej = 0;
  IVec<7, int> allrej = 0;
  IVec<7, int> acc; acc = 0;
  IVec<7, int> allacc = 0;

  auto calc_avg_scal = [&](AVG_TYPE avg, double mtra, double mtrb) LAMBDA_INLINE {
    switch(avg) {
    case(MIN): { return min(mtra, mtrb); break; }
    case(GEOM): { return sqrt(mtra * mtrb); break; }
    case(HARM): { return 2 * (mtra * mtrb) / (mtra + mtrb); break; }
    case(ALG): { return (mtra + mtrb) / 2; break; }
    case(MAX): { return max(mtra, mtrb); break; }
    default: { return -1.0; }
    }
  };

  /** vertex coll wt. only computed in round 0 **/
  auto calc_vcw = [&](auto v) {
    constexpr bool rrobust = robust;
    if constexpr (rrobust) {
      if (fabs(calc_trace(ENERGY::GetVMatrix(base_vdata[v]))) < 1e-15)
        { return 0.0; }
      HeapReset hr(lh);
      FlatMatrix<double> L(BS, BS, lh), R(BS, BS, lh);
      L = base_diags[v]; R = ENERGY::GetVMatrix(base_vdata[v]);
      auto soc = MEV<BS>(L, R); // lam L \leq R
      // if (soc > 1e-4) {
        // cout << " calc vcw for " << v << endl; // << ", L = " << endl << L << endl << " R = " << endl << R << endl;
        // cout << " soc = " << soc << endl;
      // }
      if (soc > MIN_VCW)
        { acc[4]++; }
      else
        { rej[4]++; }
      return soc;
    }
    else {
      return ENERGY::GetApproxVWeight(base_vdata[v]) / base_diags[v];
    }
  };


  /** usedverts == master_verts **/
  auto allow_merge_eq = [&](auto vi, auto eqi, auto vj, auto eqj) LAMBDA_INLINE {
    if (eqi == 0)
      { return eqc_h.IsMasterOfEQC(eqj); }
    else if (eqj == 0)
      { return eqc_h.IsMasterOfEQC(eqi); }
    else
      { return eqc_h.IsMasterOfEQC(eqi) && eqc_h.IsMasterOfEQC(eqj) && ( eqc_h.IsLEQ(eqi, eqj) || eqc_h.IsLEQ(eqj, eqi) ); }
  };


  /** used verts == solid_verts **/
  auto allow_merge_sv = [&](auto vi, auto eqi, auto vj, auto eqj) LAMBDA_INLINE {
    if (eqi == 0)
      { return curr_master_verts->Test(vj); }
    else if (eqj == 0)
      { return curr_master_verts->Test(vi); }
    else
      { return curr_master_verts->Test(vi) && curr_master_verts->Test(vj) && ( eqc_h.IsLEQ(eqi, eqj) || eqc_h.IsLEQ(eqj, eqi) ); }
  };

  auto allow_merge = [&](auto vi, auto eqi, auto vj, auto eqj) LAMBDA_INLINE {
    return (curr_master_verts == nullptr) ? allow_merge_eq(vi, eqi, vj, eqj) : allow_merge_sv(vi, eqi, vj, eqj);
  };

  /** Initial SOC to pick merge candidate. Can be EVP or scalar based. **/
  double da, db, dedge;
  auto calc_soc_scal = [&](CW_TYPE cw_type, AVG_TYPE mm_avg, auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
    /** calc_trace does nothing for scalar case **/
    dedge = calc_trace(fedata[int(fecon(vi, vj))]);
    // constexpr bool rrobust = robust;
    // if constexpr(rrobust) {
    // cout << " i vi j vj edge " << calc_trace(fdiags[vi]) << " " << calc_trace(fvdata[vi].wt);
    // cout << " " << calc_trace(fdiags[vj]) << " " << calc_trace(fvdata[vj].wt) << " " << dedge << endl;
  // }
    switch(cw_type) {
    case(SPW_CW_TYPE::HARMONIC) : { return BSU * dedge / calc_avg_scal(HARM, calc_trace(fdiags[vi]), calc_trace(fdiags[vj])); break; }
    case(SPW_CW_TYPE::GEOMETRIC) : { return BSU * dedge / calc_avg_scal(GEOM, calc_trace(fdiags[vi]), calc_trace(fdiags[vj])); break; }
    case(SPW_CW_TYPE::MINMAX) : {
      da = db = 0;
      for (auto eid : fecon.GetRowValues(vi))
        { da = max2(da, calc_trace(fedata[int(eid)])); }
      da = max2(da, ENERGY::GetApproxVWeight(fvdata[vi])*BS);
      if (need_mtrod)
        { da = max2(da, fmtrod[vi]); }
      for (auto eid : fecon.GetRowValues(vj))
        { db = max2(db, calc_trace(fedata[int(eid)])); }
      db = max2(db, ENERGY::GetApproxVWeight(fvdata[vj])*BS); // this is divided by BS, but should not be
      if (need_mtrod)
        { db = max2(db, fmtrod[vj]); }
      double soc = dedge / calc_avg_scal(mm_avg, da, db);
      if (soc < MIN_ECW)
        { rej[2]++; }
      else
        { acc[2]++; }
      return soc;
      break;
    }
    default : { return 0.0; break; }
    }
  };

  /** EVP based pairwise SOC. **/
  TM Q2(0), Ein(0), Ejn(0), Esum(0), addE(0); SetIdentity(Q2);
  auto calc_emat = [&](TVD & H_data, TM & emat, auto vi, auto vj, bool boost, const auto & fecon) LAMBDA_INLINE {
    constexpr bool rrobust = robust;
    if constexpr(rrobust) { // dummy - should never be called anyways!
      emat = fedata_full[int(fecon(vi, vj))];
      if (boost) {
        RegionTimer rt(tbste);
        // print_rank("initial emat", emat);
        auto neibsi = fecon.GetRowIndices(vi);
        auto neibsj = fecon.GetRowIndices(vj);
        iterate_intersection(neibsi, neibsj, [&](auto ki, auto kj) {
          int N = neibsi[ki];
          ENERGY::ModQij(fvdata[N], fvdata[vi], Q2);
          ENERGY::SetQtMQ(1.0, Ein, Q2, fedata_full[int(fecon(vi,N))]);
          ENERGY::ModQij(fvdata[N], fvdata[vj], Q2);
          ENERGY::SetQtMQ(1.0, Ejn, Q2, fedata_full[int(fecon(vj,N))]);
          Esum = Ein + Ejn;
          // CalcPseudoInverse(Esum, lh);
          CalcPseudoInverseNew(Esum, lh);
          addE = TripleProd(Ein, Esum, Ejn);
          ENERGY::ModQHh(H_data, fvdata[N], Q2);
          ENERGY::AddQtMQ (2.0, emat, Q2, addE);
        });
        // print_rank("final emat", emat);
      }
    }
  };

  // /** does not work as intended (is not enough) **/
  // auto calc_emat_filtered = [&](TM & emat, auto vi, auto vj, auto filter) LAMBDA_INLINE {
  //   constexpr bool rrobust = robust;
  //   if constexpr(rrobust) { // dummy - should never be called anyways!
  // 	emat = base_edata_full[int(base_econ(vi, vj))];
  // 	// print_rank("init emat", emat);
  // 	TVD H_data = ENERGY::CalcMPData(base_vdata[vi], base_vdata[vj]);
  // 	// print_rank("initial emat", emat);
  // 	auto neibsi = base_econ.GetRowIndices(vi);
  // 	auto neibsj = base_econ.GetRowIndices(vj);
  // 	int cnt = 0;
  // 	iterate_intersection(neibsi, neibsj, [&](auto ki, auto kj) {
  // 	    int N = neibsi[ki];
  // 	    if (filter(N)) {
  // 	      cnt++;
  // 	      ENERGY::ModQij(base_vdata[N], base_vdata[vi], Q2);
  // 	      ENERGY::SetQtMQ(1.0, Ein, Q2, base_edata_full[int(base_econ(vi,N))]);
  // 	      ENERGY::ModQij(base_vdata[N], base_vdata[vj], Q2);
  // 	      ENERGY::SetQtMQ(1.0, Ejn, Q2, base_edata_full[int(base_econ(vj,N))]);
  // 	      Esum = Ein + Ejn;
  // 	      CalcPseudoInverse(Esum, lh);
  // 	      addE = TripleProd(Ein, Esum, Ejn);
  // 	      ENERGY::ModQHh(H_data, base_vdata[N], Q2);
  // 	      ENERGY::AddQtMQ (2.0, emat, Q2, addE);
  // 	    }
  // 	  });
  // 	// print_rank("final emat (+" + to_string(cnt) + ")", emat);
  //   }
  // };

  TM QiM(0), QjM(0), ed(0); //, dgb(0);
  auto assemble_block_simple = [&](FlatArray<int> mems, FlatMatrix<double> A) {
    RegionTimer rt(tassbs);
    constexpr bool rrobust = robust; // constexpr capture ...
    if constexpr(rrobust) {
      const int n = mems.Size(), N = BS * mems.Size();
      for (auto ki : Range(mems)) {
        auto vi = mems[ki];
        const int Ki = BS*ki, Kip = Ki + BS;
        auto neibsa = base_econ.GetRowIndices(vi);
        auto eids = base_econ.GetRowValues(vi);
        A.Rows(Ki, Kip).Cols(Ki, Kip) += ENERGY::GetVMatrix(base_vdata[vi]); // include l2 weight
        iterate_intersection(neibsa, mems, [&](auto kneib, auto kj) {
          if (kj > ki) {
            const int vj = neibsa[kneib], Kj = BS*kj, Kjp = Kj + BS;
            const TMU & ed = base_edata_full[int(eids[kneib])];
            // calc_emat_filtered(ed, vi, vj, [&](auto n){ return find_in_sorted_array(n, mems) == -1; });
            ENERGY::ModQs(base_vdata[vi], base_vdata[vj], Qij, Qji);
            QiM = Trans(Qij) * ed;
            QjM = Trans(Qji) * ed;
            A.Rows(Ki, Kip).Cols(Ki, Kip) +=  QiM * Qij;
            A.Rows(Ki, Kip).Cols(Kj, Kjp) -= QiM * Qji;
            A.Rows(Kj, Kjp).Cols(Ki, Kip) -= QjM * Qij;
            A.Rows(Kj, Kjp).Cols(Kj, Kjp) +=  QjM * Qji;
          }
        });
      }
    }
  };

  // Array<int> extmems;
  // auto assemble_block = [&](FlatArray<int> mems, FlatMatrix<double> A, bool boost) {
  //   RegionTimer rt(tassb);
  //   constexpr bool rrobust = robust; // constexpr capture ...
  //   if constexpr(rrobust) {
  // 	if (!boost) /** straight A-block **/
  // 	  { assemble_block_simple(mems, A); }
  // 	else { /** Assemble diagonal block for mems + neibs of mems, then calc (generalized) SC to mems **/
  // 	  HeapReset hr(lh);
  // 	  extmems.SetSize(3 * mems.Size()); extmems.SetSize0();
  // 	  for (auto mem : mems) // no need to add "mem" itself - we assume mems to form connected graph
  // 	    for (auto neib : base_econ.GetRowIndices(mem))
  // 	      { insert_into_sorted_array_nodups(neib, extmems); }
  // 	  int n = mems.Size(), N = BS * n, m = extmems.Size() - n, M = BS * m;
  // 	  FlatMatrix<double> Aext(N+M, N+M, lh); Aext = 0;
  // 	  assemble_block_simple(extmems, Aext);
  // 	  FlatArray<int> locin(N, lh);
  // 	  iterate_intersection(mems, extmems, [&](auto ki, auto ke) {
  // 	      int indi = BS * ki, inde = BS * ke;
  // 	      for (auto l : Range(BS))
  // 		{ locin[indi++] = inde++; }
  // 	    });
  // 	  FlatArray<int> locout(M, lh); int co = 0;
  // 	  iterate_anotb(extmems, mems, [&](auto ke) {
  // 	      int inde = BS * ke;
  // 	      for (auto l : Range(BS))
  // 		{ locout[co++] = inde++; }
  // 	    });
  // 	  FlatMatrix<double> C(M, M, lh);
  // 	  C = Aext.Rows(locout).Cols(locout);
  // 	  // print_rank("as sblock C", C, lh, cerr);
  // 	  // cerr << " calc Pinv for " << endl << C << endl;
  // 	  CalcPseudoInverseFM(C, lh);
  // 	  // print_rank("inved C", C, lh, cerr);
  // 	  A = Aext.Rows(locin).Cols(locin);
  // 	  FlatMatrix<double> AC(N, M, lh);
  // 	  AC = Aext.Rows(locin).Cols(locout) * C;
  // 	  // print_rank("assblock A I", Aext, lh);
  // 	  A -= AC * Aext.Rows(locout).Cols(locin);
  // 	  // print_rank("assblock A II  ", Aext, lh);
  // 	}
  //   }
  // };

  Array<int> extmems, inmems;
  TM tm_tmp(0);
  auto assemble_block = [&](FlatArray<int> mems, FlatMatrix<double> A, bool boost) {
    RegionTimer rt(tassb);
    constexpr bool rrobust = robust; // constexpr capture ...
    if constexpr(rrobust) {
      /** straight A-block **/
      assemble_block_simple(mems, A);
      /** For every member, calc SC w.r.t mems  **/
      if (boost) {
        extmems.SetSize(3 * mems.Size()); extmems.SetSize0();
        extmems.Part(0, mems.Size()) = mems;
      // print_rank("assblock init A", A, lh);
        for (auto mem : mems) { // no need to add "mem" itself - we assume mems to form connected graph
          for (auto neib : base_econ.GetRowIndices(mem)) {
            if ( insert_into_sorted_array_nodups(neib, extmems) ) {
              HeapReset hr(lh);
              auto neibneibs = base_econ.GetRowIndices(neib);
              intersect_sorted_arrays(neibneibs, mems, inmems);
              if (inmems.Size() < 2)
                { continue; }
              // cout << " add neib " << neib << endl;
              // cout << " inmems: "; prow2(inmems); cout << endl;
              Esum = 0;
              FlatArray<int> Aris(BS*inmems.Size(), lh); int cnt_Aris = 0;
              FlatMatrix<double> A_n_mems(BS, BS*inmems.Size(), lh);
              for (auto li : Range(inmems)) {
                int vi = inmems[li], Ki = find_in_sorted_array(vi, mems) * BS, Kip = Ki + BS,
                Li = BS * li, Lip = Li + BS;
                // cout << " vi ki kip " << vi << " " << Ki << " " << Kip << endl;
                for (auto l : Range(Ki, Kip))
                { Aris[cnt_Aris++] = l; }
                const TM & ed = base_edata_full[int(base_econ(vi, neib))];
                ENERGY::ModQs(base_vdata[vi], base_vdata[neib], Qij, Qji);

                ENERGY::CalcQTM(1.0, Qji, ed, QjM);
                // CalcQTM<ENERGY>(1.0, Qji, ed, QjM);
                // QjM = Trans(Qji) * ed;

                ENERGY::CalcQTM(1.0, Qij, ed, QiM);
                // CalcQTM<ENERGY>(1.0, Qij, ed, QiM);
                // QiM = Trans(Qij) * ed;

                // A.Rows(Ki, Kip).Cols(Ki, Kip) += QiM * Qij;
                // ENERGY::CalcMQ(1.0, QiM, Qij, tm_tmp);
                CalcMQ<ENERGY>(1.0, QiM, Qij, tm_tmp);
                // tm_tmp = QiM * Qij;
                A.Rows(Ki, Kip).Cols(Ki, Kip) += tm_tmp;

                // A_n_mems.Cols(Li, Lip) = -1.0 * QjM * Qij;
                // ENERGY::CalcMQ(-1.0, QjM, Qij, tm_tmp);
                CalcMQ<ENERGY>(-1.0, QjM, Qij, tm_tmp);
                A_n_mems.Cols(Li, Lip) = tm_tmp;

                // Esum += QjM * Qji;
                // ENERGY::AddMQ(1.0, QjM, Qji, Esum);
                AddMQ<ENERGY>(1.0, QjM, Qji, Esum);
              }
              // cout << "Aris "; prow(Aris); cout << endl;
              // print_rank("assblock A " + to_string(mem) + ".I", A, lh);
              // cout << " Esum " << endl; print_tm(cout, Esum); cout << endl;
              // TM esum2 = Esum;
              // print_rank("esum", Esum);
              // cout << " Esum = " << endl; print_tm(cout, Esum);

              // CalcPseudoInverse(Esum, lh);
              CalcPseudoInverseNew(Esum, lh);

              // CalcPseudoInverseNew(esum2, lh);
              // cout << " inv Esum    = " << endl; print_tm(cout, Esum);
              // cout << " newinv Esum = " << endl; print_tm(cout, esum2);
              // esum2 -= Esum;
              // double nrm = 0;
              // for (auto k : Range(BS))
              //   for (auto j : Range(BS))
              //     nrm += fabs(esum2(k,j));
              // if (nrm > 1e-12)
              //   { cout << " DIFF = " << nrm << endl << endl; }

              // cout << " pinv Esum " << endl; print_tm(cout, Esum); cout << endl;
              FlatMatrix<double> A_mems_n(BS*inmems.Size(), BS, lh);
              // cout << " A_n_mems " << endl << A_n_mems << endl;
              A_mems_n = Trans(A_n_mems) * Esum;
              /** The SC **/
              A.Rows(Aris).Cols(Aris) -= A_mems_n * A_n_mems;
              // print_rank("assblock A " + to_string(mem) + ".II", A, lh);
            }
          }
        }
      }
    }
  };

  TM dma(0), dmb(0), dmedge(0), Q(0); SetIdentity(Q);
  auto calc_soc_robust = [&](CW_TYPE cw_type, bool neib_boost, AVG_TYPE mma_scal, bool mma_mat_harm, auto vi, auto vj, const auto & fecon) LAMBDA_INLINE {
    constexpr bool rrobust = robust;
    double soc = 1.0;
    if constexpr(rrobust) { // dummy - should never be called anyways!
      soc = 0;
      // cout << " CSR " << vi << " " << vj << endl;
      /** Transform diagonal matrices **/
      TVD H_data = ENERGY::CalcMPData(fvdata[vi], fvdata[vj]);
      ENERGY::ModQHh(H_data, fvdata[vi], Q);
      ENERGY::SetQtMQ(1.0, dma, Q, fdiags[vi]);
      ENERGY::ModQHh(H_data, fvdata[vj], Q);
      ENERGY::SetQtMQ(1.0, dmb, Q, fdiags[vj]);
      /** TODO:: neib bonus **/
      // dmedge = fedata_full[int(fecon(vi,vj))];
      calc_emat(H_data, dmedge, vi, vj, neib_boost, fecon);
      // cout << " dga " << endl; print_tm(cout, fdiags[vi]); cout << endl;
      // print_rank("dma", dma);
      // print_rank("dmb", dma);
      // print_rank("dmedge", dmedge);
      // cout << " dma " << endl; print_tm(cout, 16, dma); cout << endl;
      // cout << " dgb " << endl; print_tm(cout, fdiags[vj]); cout << endl;
      // cout << " dmb " << endl; print_tm(cout, 16, dmb); cout << endl;
      // cout << " dmedge " << endl; print_tm(cout, 16, dmedge); cout << endl;
      // cout << " fecon inds row " << vi << " = "; prow(fecon.GetRowIndices(vi)); cout << endl;
      // cout << " fecon vals row " << vi << " = "; prow(fecon.GetRowValues(vi)); cout << endl;
      // cout << " fecon inds row " << vj << " = "; prow(fecon.GetRowIndices(vj)); cout << endl;
      // cout << " fecon vals row " << vj << " = "; prow(fecon.GetRowValues(vj)); cout << endl;
      // cout << " fed size " << fedata_full.Size() << ", get from " << int(fecon(vi, vj)) << endl;
      switch(cw_type) {
        case(CW_TYPE::HARMONIC) : { soc = MIN_EV_HARM2(dma, dmb, dmedge); /** cout << " soc harm = " << soc << endl; **/ break; }
        case(CW_TYPE::GEOMETRIC) : { soc = MIN_EV_FG2(dma, dmb, dmedge); /** cout << " soc geom = " << soc << endl; **/ break; }
        case(CW_TYPE::MINMAX) : {
          double mtra = 0, mtrb = 0;
          for (auto eid : fecon.GetRowValues(vi))
            { mtra = max2(mtra, calc_trace(fedata[int(eid)])); }
          mtra = max2(mtra, ENERGY::GetApproxVWeight(fvdata[vi])*BS); // this is divided by BS, but should not be
          if (need_mtrod)
            { mtra = max2(mtra, fmtrod[vi]); } //
          for (auto eid : fecon.GetRowValues(vj))
            { mtrb = max2(mtrb, calc_trace(fedata[int(eid)])); }
          mtrb = max2(mtrb, ENERGY::GetApproxVWeight(fvdata[vj])*BS); // this is divided by BS, but should not be
          if (need_mtrod)
            { mtrb = max2(mtrb, fmtrod[vj]); } //
          double etrace = calc_trace(dmedge);
          soc = etrace / calc_avg_scal(mma_scal, mtra, mtrb);
          if (soc < MIN_ECW)
            { rej[2]++; }
          else
            { acc[2]++; }
          // cout << " mtra mtrb etrace socscal " << mtra << " " << mtrb << " " << etrace << " " << soc << endl;
          if (soc > MIN_ECW) {
            dma /= calc_trace(dma);
            dmb /= calc_trace(dmb);
            dmedge /= etrace;
            double socmat;
            if (mma_mat_harm)
              { socmat = MIN_EV_HARM2(dma, dmb, dmedge); /** cout << " soc mmx harm = " << soc << endl; **/ }
            else
              { socmat = MIN_EV_FG2(dma, dmb, dmedge); /** cout << " soc mmx geom = " << soc << endl; **/ }
            if (socmat < MIN_ECW)
              { rej[3]++; }
            else
              { acc[3]++; }
            soc = min2(soc, socmat);
            // cout << " soc mat " << soc << endl;
          }
          break;
        }
      } // switch
    } // robust
    return soc;
  };

  auto calc_soc_pair = [&](bool dorobust, bool neib_boost, CW_TYPE cwt, AVG_TYPE mma_scal,
          AVG_TYPE mma_mat, auto vi, auto vj, const auto & fecon) { // maybe not force inlining this?
    // if (double(lh.Available())/lhs < 0.9)
  // cout << " calc_soc_pair, free      = " << double(lh.Available())/1024/1024 << ", frac = " << double(lh.Available())/lhs << endl;
    constexpr bool rrobust = robust;
    {
      HeapReset hr(lh);
      if constexpr(rrobust) {
        if (dorobust)
        { return calc_soc_robust(cwt, neib_boost, mma_scal, (mma_mat==HARM), vi, vj, fecon); }
        else
        { return calc_soc_scal(cwt, mma_scal, vi, vj, fecon); }
      }
      else
        { return calc_soc_scal(cwt, mma_scal, vi, vj, fecon); }
    }
  };

  auto check_soc_aggs_scal = [&](auto memsi, auto memsj) LAMBDA_INLINE {
    HeapReset hr(lh);
    /** TODO: this does not use fdiags, so l2 weights are not considered! **/
    /** simplified scalar version, based on traces of mats **/
    int n = memsi.Size() + memsj.Size();
    // FlatArray<int> allmems = merge_arrays_lh(memsi, memsj, lh); // actually, these are not ordered in the first place..
    FlatArray<int> allmems(n, lh);
    allmems.Part(0, memsi.Size()) = memsi;
    allmems.Part(memsi.Size(), memsj.Size()) = memsj;
    QuickSort(allmems);
    /** A - the sub-assembled diag block including l2, but excluding external connections **/
    FlatMatrix<double> A(n, n, lh); A = 0.0;
    for (auto ki : Range(allmems)) {
      auto vi = allmems[ki];
      auto neibsa = base_econ.GetRowIndices(vi);
      auto eids = base_econ.GetRowValues(vi);
      A(ki, ki) += ENERGY::GetApproxVWeight(base_vdata[vi]); // divide by BS is probably important here
      iterate_intersection(neibsa, allmems, [&](auto kneib, auto kj) {
        if (kj > ki) {
          const double x = base_edata[int(eids[kneib])];
          A(ki, ki) += x;
          A(ki, kj) -= x;
          A(kj, ki) -= x;
          A(kj, kj) += x;
        }
      });
    }
    /** P, PT**/
    FlatMatrix<double> P(n, 1, lh), PT(1, n, lh);
    P = 1.0; PT = 1.0;
    /**  M is the diag block of the smoother - either block-diag or diag  **/
    FlatMatrix<double> M(n, n, lh);
    if (bdiag)
      { M = A; }
    else
      { M = 0.0; }
    for (auto ki : Range(allmems)) {
      auto vi = allmems[ki];
      M(ki, ki) = calc_trace(base_diags[vi])/BS;
    }
    /** Project out ran(P) **/
    FlatMatrix<double> PTM(1, n, lh);
    PTM = PT * M;
    FlatMatrix<double> PTMP(1, 1, lh);
    PTMP = PTM * P;
    double invPTMP = 1.0/PTMP(0,0);
    // cout << " (scal) CBS for " << memsi.Size() << " + " << memsj.Size() << " = " << n << endl;
    // print_rank("CBS A", A, lh);
    // print_rank("CBS M I ", M, lh);
    M -= invPTMP * Trans(PTM) * PTM;
    // M -= Trans(PTM) * invPTMP * PTM;
    // print_rank("CBS, M II", M, lh);
    /** we have to check: MIN_ECW * M < A, ot A-MIN_ECW*M >= 0 **/
    // print_rank("CBS M II ", M, lh);
    A -= MIN_ECW * M;
    /** we KNOW the kernel exactly - regularize, so we can use dpotrf/dpstrf [[-eps eval is a problem]] **/
    A += calc_trace(A)/n * P * PT;
    // print_rank("CBS checked mat", A, lh);
    // print_rank("CBS, checked mat", A, lh);
    // print_rank("(scal) CBS checked mat ", A, lh);
    bool isspd = CheckForSPD(A, lh);
    // cout << " is spd = " << isspd << endl << endl;
    return isspd;
  };


  auto check_soc_aggs_robust = [&](auto memsi, auto memsj, bool boost) LAMBDA_INLINE {
    /** TODO: this does not use fdiags, so l2 weights are not considered! **/
    constexpr bool rrobust = robust;
    if constexpr(rrobust) {
      /** "full" version, assembles the entire system **/
      int n = memsi.Size() + memsj.Size(), N = BS * n;
      HeapReset hr(lh);
      if (n == 0)
        { return true; }
      // FlatArray<int> allmems = merge_arrays_lh(memsi, memsj, lh); // actually, these are not ordered in the first place..
      FlatArray<int> allmems(n, lh);
      allmems.Part(0, memsi.Size()) = memsi;
      allmems.Part(memsi.Size(), memsj.Size()) = memsj;
      QuickSort(allmems);
      /** A - the sub-assembled diag block **/
      FlatMatrix<double> A(N, N, lh); A = 0.0;
      assemble_block(allmems, A, boost);

      // /** P, PT **/
      FlatMatrix<double> P(N, BS, lh), PT(BS, N, lh);
      for (auto k : Range(allmems)) {
        ENERGY::ModQHh(base_vdata[allmems[0]], base_vdata[allmems[k]], Qij);
        P.Rows(k*BS, (k+1)*BS) = Qij;
      }
      PT = Trans(P);
      /**  M is the diag block of the smoother - ATM I use the full diag block of Ahat (BJAC),
         not only the diagonal entries (JAC). Here we also need external (including off-proc) connections and l2 weight **/
      FlatMatrix<double> M(N, N, lh);
      if (bdiag)
        { M = A; }
      else
        { M = 0.0; }
      for (auto ki : Range(allmems)) {
        auto vi = allmems[ki];
        const int Ki = BS*ki, Kip = Ki + BS;
        M.Rows(Ki, Kip).Cols(Ki, Kip) = base_diags[vi];
      }
      FlatMatrix<double> PTM(BS, N, lh);
      PTM = PT * M;
      FlatMatrix<double> PTMP(BS, BS, lh);
      PTMP = PTM * P;
      /** this CAN be singular (in rare cases...) **/
      // CalcInverse(PTMP);
      CalcPseudoInverseNew(PTMP, lh);
      FlatMatrix<double> inv_PTMP_PTM(BS, N, lh);
      inv_PTMP_PTM = PTMP * PTM;
      // cout << " (vec) CBS for " << memsi.Size() << " + " << memsj.Size() << " = " << n << endl;
      // print_rank("CBS A", A, lh);
      // print_rank("CBS M I ", M, lh);
      /** M - M P (PTMP)^{-1} PT M **/
      M -= Trans(PTM) * inv_PTMP_PTM;
      // print_rank("CBS A ", A, lh);
      // print_rank("CBS M ", M, lh);
      if (cbs_spd_hack) {
        /** Here we do not necessarily know the entire kernel - vertices can be on a line (in 3d).
           Regularizing with RBMs only might not be enough. So we substract a bit more than MIN_ECW,
          and then add a bit to the diagonal. Then we can use dpotrf. **/
        double tra = calc_trace(A)/N;
        double reg = 1e-12 * tra;
        A -= (MIN_ECW + reg) * M;
        // print_rank("CBS checked mat I ", A, lh);
        A += tra * P * PT;
        for (auto k : Range(N))
        { A(k,k) += reg; }
        // print_rank("(hacked) CBS checked mat ", A, lh);
        // FlatMatrix<double> A2(N, N, lh); A2 = A;
        bool isspd = CheckForSPD(A, lh);
        // bool issspd = CheckForSSPD(A2, lh);
        // cout << " SSPD = " << isspd << endl;
        return isspd;
      }
      else { /** Do not regularize anything and use dpstrf. **/
        A -= MIN_ECW * M;
        // print_rank("CBS checked mat ", A, lh);
        bool issspd = CheckForSSPD(A, lh);
        // cout << " SSPD = " << issspd << endl;
        return issspd;
      }
      // return CheckForSSPD(A, lh);
    }
    else {
      return true;
    }
  };

  /** SOC for pair of agglomerates w.r.t original matrix **/
  auto check_soc_aggs = [&](bool simplify, bool boost, auto memsi, auto memsj) LAMBDA_INLINE {
    /** TODO: this does not use fdiags, so l2 weights are not considered! **/
    if ( (BSU == 1) || simplify)
      { return check_soc_aggs_scal(memsi, memsj); }
    else
      { return check_soc_aggs_robust(memsi, memsj, boost); }
    return true;
  };

  /** Finds a neighbor to merge vertex v with. Returns -1 if no suitable ones found **/
  auto find_neib = [&](auto v, const auto & fecon, auto allowed, auto get_mems, bool robust_pick, bool neib_boost,
        auto cwt_pick, auto pmmas, auto pmmam, auto cwt_check, auto cmmas, auto cmmam,
        bool checkbigsoc, bool simple_cbs) LAMBDA_INLINE {
    constexpr bool rrobust = robust;
    HeapReset hr(lh);
    /** SOC for all neibs **/
    double max_soc = 0, msn = -1, c = 0;
    auto neibs = fecon.GetRowIndices(v);
    auto rvs = fecon.GetRowValues(v);
    FlatArray<IVec<2,double>> bsocs(neibs.Size(), lh);
    for (auto jn : Range(neibs)) {
      auto neib = neibs[jn];
      if ( curr_allowed_edges && !curr_allowed_edges->Test(int(rvs[jn])) )
        { cout << "FORBID " << v << " " << neib << endl; }
      if ( !curr_allowed_edges || curr_allowed_edges->Test(int(rvs[jn])) ) {
        if (allowed(v, neib))
        { double thesoc = calc_soc_pair(robust_pick, neib_boost, cwt_pick, pmmas, pmmam, v, neib, fecon); bsocs[c++] = IVec<2, double>(neib, thesoc); }
      }
    }
    // cout << endl << " ALL possible neibs for " << v << " = ";
    // for (auto v : bsocs)
  // { cout << "[" << v[0] << " " << v[1] << "] "; }
    // cout << endl;
    auto socs = bsocs.Part(0, c);
    QuickSort(socs, [&](const auto & a, const auto & b) LAMBDA_INLINE { return a[1] > b[1]; });
    // cout << " possible neibs for " << v << " = ";
    // for (auto v : socs)
    // 	{ cout << "[" << v[0] << " " << v[1] << "] "; }
    // cout << endl;

    int candidate = ( (c > 0) && (socs[0][1] > MIN_ECW) ) ? int(socs[0][0]) : -1;

    if ( (c > 0) && allowed(v, int(socs[0][0])) ) {
      if (candidate == -1)
        { rej[5]++; }
      else
        { acc[5]++; }
    }

    if (c == 0)
      { rej[6]++; }

    // cout << " candidate is " << candidate << endl;
    /** check candidate - either small EVP, or large EVP, or both! **/
    bool need_check = (rrobust && (!robust_pick)) || checkbigsoc;
    // cout << " rr = " << rrobust << " rp = " << robust_pick << ", cbs = " << checkbigsoc << endl;
    // cout << " need_check = " << (rrobust && (!robust_pick)) << " || " << checkbigsoc << endl;
    if (need_check) {
      candidate = -1; // !! important, otherwise the loop fails if last entry has soc>MIN, but stabsoc < MIN
      for (int j = 0; j < socs.Size(); j++) {
        double stabsoc = socs[j][1];
        if (stabsoc < MIN_ECW)
          { candidate = -1; break; }
        if constexpr(rrobust) {
          if (!robust_pick) /** small EVP soc **/
            { stabsoc = calc_soc_pair(true, neib_boost, cwt_check, cmmas, cmmam, v, int(socs[j][0]), fecon); }
        }
      if (stabsoc < MIN_ECW) /** this neib has strong stable connection **/
        { rej[0]++; }
      else
        { acc[0]++; }
      if (checkbigsoc && (stabsoc > MIN_ECW)) { /** big EVP soc **/
        stabsoc = check_soc_aggs(simple_cbs, neib_boost, get_mems(v), get_mems(int(socs[j][0]))) ? stabsoc : 0.0;
        if (stabsoc < MIN_ECW) /** this neib has strong stable connection **/
          { rej[1]++; }
        else
          { acc[1]++; }
      }
      if (stabsoc > MIN_ECW) /** this neib has strong stable connection **/
        { candidate = int(socs[j][0]); break; }
      }
    } // need_check

    // cout << " final candidate is " << candidate << endl;
    return candidate;
  };

  /** Iterate through unhandled vertices and pair them up **/
  auto pair_vertices = [&](FlatArray<int> vmap, size_t & NCV,
          int num_verts, auto get_vert, const auto & fecon,
          BitArray & handled,
          auto get_mems, auto allowed,// auto set_pair,
          bool r_ar,  bool r_nb, auto r_cwtp, auto r_pmmas, auto r_pmmam, auto r_cwtc, auto r_cmmas, auto r_cmmam, bool r_cbs, bool r_scbs
          ) LAMBDA_INLINE {
    // if (print_summs) {
    // 	cout << "  pair_vertices done, check " << num_verts << " of " << fecon.Height() << endl;
    // 	cout << "  " << handled.NumSet() << " alreay handled " << endl;
    // }

    RegionTimer rt(tvp);
    for (auto k : Range(num_verts)) {
      auto vnr = get_vert(k);
      if (!handled.Test(vnr)) { // try to find a neib
        // cout << " find neib for " << vnr << endl;
        int neib = find_neib(vnr, fecon, [&](auto vi, auto vj) { return (!handled.Test(vj)) && allowed(vi, vj); },
              get_mems, r_ar, r_nb, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, r_cbs, r_scbs);
        if (neib != -1) {
          // cout << " new pair: " << vnr << " " << neib << endl;
          vmap[neib] = NCV;
          handled.SetBit(neib);
        }
        vmap[vnr] = NCV++;
        handled.SetBit(vnr);
        // set_pair(vnr, neib);
      }
    }
    if ( havelocinfo && print_summs ) {
      cout << "  pair_vertices done, NFV = " << fecon.Height() << ", NCV = " << NCV << endl;
    }
    if (havelocinfo && print_aggs) {
      cout << "  round vmap: "; prow2(vmap); cout << endl;
    }
  };

  tprep.Stop();

  // pair vertices
  Array<Agglomerate> spec_aggs;
  for (int round : Range(num_rounds)) {
    HeapReset hr(lh); // probably unnecessary
    const bool r_ar = robust && this->allrobust.GetOpt(round);
    const bool r_nb = robust && this->neib_boost.GetOpt(round);
    const CW_TYPE r_cwtp = this->pick_cw_type.GetOpt(round);
    const AVG_TYPE r_pmmas = this->pick_mma_scal.GetOpt(round);
    const AVG_TYPE r_pmmam = this->pick_mma_mat.GetOpt(round);
    const CW_TYPE r_cwtc = this->check_cw_type.GetOpt(round);
    const AVG_TYPE r_cmmas = this->check_mma_scal.GetOpt(round);
    const AVG_TYPE r_cmmam = this->check_mma_mat.GetOpt(round);
    const bool r_cbs = (round > 0) && this->checkbigsoc; // round 0 no bigsoc hardcoded anyways
    const bool r_scbs = this->simple_checkbigsoc;
    const bool r_wo = this->weed_out;
    const bool do_cmk = (round == 0) || this->all_cmk;

    const bool use_hack_stab = this->use_stab_ecw_hack.IsTrue() ||
      ( (!this->use_stab_ecw_hack.IsFalse())
        && ( (   this->allrobust.GetOpt(round)  && (this->pick_cw_type.GetOpt(round)  == CW_TYPE::HARMONIC) ) ||
          ( (!this->allrobust.GetOpt(round)) && (this->check_cw_type.GetOpt(round) == CW_TYPE::HARMONIC)) ) ) ;

    if (print_params) {
      cout << endl << " SPW agglomerates, round " << round << " of " << num_rounds << endl;
      cout << "   allrobust      = " << r_ar << endl;
      cout << "   neib_boost     = " << r_nb << endl;
      cout << "   cwt pick       = " << r_cwtp << endl;
      cout << "   mma pick scal  = " << r_pmmas << endl;
      cout << "   mma pick mat   = " << r_pmmam << endl;
      cout << "   cwt check      = " << r_cwtc << endl;
      cout << "   mma check scal = " << r_cmmas << endl;
      cout << "   mma check mat  = " << r_cmmam << endl;
      cout << "   checkbigsoc    = " << r_cbs << endl;
      cout << "   use_hack_stab  = " << use_hack_stab << endl;
      cout << "   weed_out    = " << r_wo << endl;
    }

    /** Set up a local map to represent this round of pairing vertices **/
    if ( (round > 0) && m_print_summs ) {
      if ( print_summs && havelocinfo ) {
        cout << " next loc mesh NV NE " << conclocmap->GetMappedMesh()->template GetNN<NT_VERTEX>() << " "
             << conclocmap->GetMappedMesh()->template GetNN<NT_EDGE>() << endl;
      }
      if ( eqc_h.GetCommunicator().Size() > 1 ) {
        auto rednv = eqc_h.GetCommunicator().Reduce(conclocmap->GetMappedMesh()->template GetNN<NT_VERTEX>(), NG_MPI_SUM, 0);
        if ( print_summs && (eqc_h.GetCommunicator().Rank() == 0) )
          { cout << " next glob NV = " << rednv << endl; }
      }
    }
    auto locmap = make_shared<LocCoarseMap>( (round == 0) ? this->_mesh : conclocmap->GetMappedMesh());
    locmap->SetAllowedFEdges( (round == 0) ? this->GetAllowedEdges() : conclocmap->GetAllowedCEdges());
    auto vmap = locmap->template GetMap<NT_VERTEX>();
    const TopologicMesh & fmesh = *locmap->GetMesh();
    const SparseMatrix<double> & fecon = *fmesh.GetEdgeCM();

    // if (fecon.Height())
    // 	{
    // 	  IVec<5, int> cnts = 0;
    // 	  for (auto k : Range(fecon.Height())) {
    // 	    if (fecon.GetRowIndices(k).Size() < 5) {
    // 	      cnts[fecon.GetRowIndices(k).Size()]++;
    // 	    }
    // 	  }
    // 	  cout << " cnts: " << endl;
    // 	  for (auto k : Range(5)) {
    // 	    cout << k << " " << cnts[k] << "   " << double(cnts[k])/fecon.Height() << endl;
    // 	  }
    // 	}

    BitArray handled(NV); handled.Clear();
    size_t & NCV = locmap->GetNCV(); NCV = 0;

    if (round == 0) {
      tsv.Start();
      
      /** dirichlet vertices **/
      auto free_verts = this->GetFreeVerts();
      if (free_verts != nullptr) {
        const auto & fvs = *free_verts;
        // cout << "FV LOOP, " << fvs.Size() << " " << vmap.Size() << endl;
        for (auto v : Range(vmap))
        if (!fvs.Test(v))
          { vmap[v] = -1; handled.SetBit(v); }
        // for (auto v : Range(vmap))
        // if (!fvs.Test(v))
        //   { cout << "STDIRI " << v << endl; }
        // cout << endl << endl;
      }
      
      /** non-master vertices **/
      M.template ApplyEQ2<NT_VERTEX>([&](auto eq, auto vs) {
        if (curr_master_verts != nullptr) {
          if (eq > 0)
        for (auto v : vs)
          if (!curr_master_verts->Test(v))
          { vmap[v] = -1; handled.SetBit(v); }
        }
        else {
          if (!eqc_h.IsMasterOfEQC(eq))
        for (auto v : vs)
          { vmap[v] = -1; handled.SetBit(v); }
        }
      }, false); // obviously, not master only
  
      /** Fixed Aggs - set verts to diri, handle afterwards **/
      auto fixed_aggs = this->GetFixedAggs();
      for (auto row : fixed_aggs) {
        for (auto v : row) {
          if (!handled.Test(v))
            { vmap[v] = -1; handled.SetBit(v); }
        }
      }
      /** collapsed vertices **/
  
      if (MIN_VCW > 0) { // otherwise, no point
        HeapReset hr(lh);
        for(auto v : Range(vmap)) {
          if (!handled.Test(v)) {
            double cw = calc_vcw(v);
            if (cw > MIN_VCW)
              { vmap[v] = -1; handled.SetBit(v); }
          }
        }
      }

      auto allowit = [&](auto vi, auto vj) { return allow_merge(vi, M.template GetEQCOfNode<NT_VERTEX>(vi),
                                                                vj, M.template GetEQCOfNode<NT_VERTEX>(vj)); };

      /** Try to weed out dangling vertices. Do not include these aggs in local crs levels if they have > 2 members ! **/
      if (r_wo)
      {
        RegionTimer rt(tdangle);
        Array<int> possmems, dummy;
        bool simplecbs = r_scbs || (!r_cbs);
        const int maxaggsize = pow(2, num_rounds);
        int acnt = 0;
        if (M.template GetNN<NT_VERTEX>() > 0)
        for (auto k : M.template GetENodes<NT_VERTEX>(0))
          if (fecon.GetRowIndices(k).Size() == 1)
            { acnt++; }
        spec_aggs.SetSize(acnt); spec_aggs.SetSize0();
        /** try only local vertices (but ctr can be in another eqc) **/
        if (M.template GetNN<NT_VERTEX>() > 0) {
          for (auto k : M.template GetENodes<NT_VERTEX>(0)) {
            // for (auto k : Range(fecon.Height())) {
            auto ris = fecon.GetRowIndices(k);
            if ( (ris.Size() == 1) && (! (handled.Test(k) || handled.Test(ris[0])) ) && (allowit(k, ris[0])) ) {
              int ctrv = ris[0];
              auto ctrnbs = fecon.GetRowIndices(ctrv);
              /** all allowed, dangling neibs of "ctrv" **/
              int c = 1; possmems.SetSize(1+ctrnbs.Size()); possmems[0] = ctrv;
              for (auto cn : ctrnbs)
              if ( (fecon.GetRowIndices(cn).Size() == 1) && (!handled.Test(cn)) && allowit(cn, c) )
                { possmems[c++] = cn; }
              possmems.SetSize(min(c, maxaggsize));
              if ( c == 2 ) { // force this connection - it HAS to be at least "OK" per definition!
                handled.SetBit(k); handled.SetBit(ctrv);
                vmap[k] = NCV; vmap[ctrv] = NCV;
                NCV++;
              }
              else {
                HeapReset hr(lh);
                bool aggok = check_soc_aggs( r_scbs, r_nb, FlatArray<int>(possmems), FlatArray<int>(dummy) );
                if ( aggok ) { /** build the agg **/
                  int aggid = spec_aggs.Size();
                  handled.SetBit(ctrv); vmap[ctrv] = -1;
                  v_to_agg[ctrv] = aggid;
                  spec_aggs.Append(Agglomerate(ctrv, aggid));
                  auto & agg = spec_aggs.Last();
                  for (auto mem : possmems.Part(1)) {
                    handled.SetBit(mem);
                    vmap[mem] = -1;
                    agg.Add(mem);
                    v_to_agg[mem] = aggid;
                  }
                  // cout << " added a dangling agg for " << k <<  ", agg = " << agg << endl;
                } // TOOD: I should probably at least take "k + the neib or so..."
              }
            }
          }
        }
      }
      tsv.Stop();
      /** CMK ordering **/
      Array<int> cmk; CalcCMK(handled, fecon, cmk);
      /** Find pairs for vertices **/
      Array<int> dummy(1);
      pair_vertices(vmap, NCV, cmk.Size(), [&](auto k) { return cmk[k]; }, fecon, handled,
            [&](auto v) ->FlatArray<int> { dummy[0] = v; return dummy; }, // get_mems
            allowit,
            r_ar, r_nb, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, false, r_scbs); // no big soc necessary
    }
    else
    {
      /** Find pairs for vertices **/
      auto c2fv = conclocmap->template GetMapC2F<NT_VERTEX>();
      auto veqs = conclocmap->GetV2EQ();
      Array<int> cmk;
      if (do_cmk)
        { CalcCMK(handled, fecon, cmk); }
      // cout << " CMK: "; prow2(cmk); cout << endl;
      size_t NGV = do_cmk ? cmk.Size() : vmap.Size();
      size_t MAX = vmap.Size() - 1;
      auto get_v = [&](auto k) { return do_cmk ? cmk[k] : MAX - k; };
      pair_vertices(vmap, NCV, NGV, get_v,
            fecon, handled,
            [&](auto v) LAMBDA_INLINE { return c2fv[v]; }, // get_mems
            [&](auto vi, auto vj) LAMBDA_INLINE { return allow_merge(vi, veqs[vi], vj, veqs[vj]); }, // allowed
            r_ar, r_nb, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, r_cbs, r_scbs);

        // pair_vertices(vmap, NCV,
      // 	      // cmk.Size(), [&](auto k) { return cmk[k]; },
      // 	      // vmap.Size(), [&](auto i) LAMBDA_INLINE { return i; }, // no CMK on later rounds!
      // 	      vmap.Size(), [&](auto i) LAMBDA_INLINE { return MAX-i; }, // no CMK on later rounds!
      // 	      fecon, handled,
      // 	      [&](auto v) LAMBDA_INLINE { return c2fv[v]; }, // get_mems
      // 	      [&](auto vi, auto vj) LAMBDA_INLINE { return allow_merge(vi, veqs[vi], vj, veqs[vj]); }, // allowed
      // 	      r_ar, r_nb, r_cwtp, r_pmmas, r_pmmam, r_cwtc, r_cmmas, r_cmmam, r_cbs, r_scbs);
    }

    if (round < num_rounds - 1) /** proper concatenated map, with coarse mesh **/
    { 
      RegionTimer art(tmap);
      /** Build edge map, C2F vertex map, edge connectivity and coarse mesh **/
      locmap->FinishUp();
      /** Coarse vertex data **/
      Array<TVD> ccvdata(NCV);
      auto c2fv = locmap->template GetMapC2F<NT_VERTEX>();
      for (auto cvnr : Range(ccvdata)) {
        auto fvs = c2fv[cvnr];
        if (fvs.Size() == 1)
          { ccvdata[cvnr] = fvdata[fvs[0]]; }
        else
         { ccvdata[cvnr] = ENERGY::CalcMPDataWW(fvdata[fvs[0]], fvdata[fvs[1]]); }
      }
      /** Coarse edge data **/
      size_t NCE = locmap->template GetMappedNN<NT_EDGE>();
      Array<TMU> ccedata_full(NCE); ccedata_full = 0;
      auto fedges = locmap->GetMesh()->template GetNodes<NT_EDGE>();
      auto cedges = locmap->GetMappedMesh()->template GetNodes<NT_EDGE>();
      auto emap = locmap->template GetMap<NT_EDGE>();
      // cout << " mapped mesh " << locmap->GetMappedMesh() << endl;
      // cout << " fedges " << fedges.Size() << " "; prow(fedges); cout << endl;
      // cout << " cedges " << cedges.Size() << " "; prow(cedges); cout << endl;
      // cout << " emap " << emap.Size() << " "; prow2(emap); cout << endl;
      TM Q(0); SetIdentity(Q);
      TVD femp, cemp;
      for (auto fenr : Range(fedges)) {
        auto & fedge = fedges[fenr];
        auto cenr = emap[fenr];
        if (cenr != -1) {
          auto & cedge = cedges[cenr];
          if constexpr(robust) {
            femp = ENERGY::CalcMPData(fvdata[fedge.v[0]], fvdata[fedge.v[1]]);
            cemp = ENERGY::CalcMPData(ccvdata[cedge.v[0]], ccvdata[cedge.v[1]]);
            ENERGY::ModQHh(cemp, femp, Q);
            ENERGY::AddQtMQ(1.0, ccedata_full[cenr], Q, fedata_full[fenr]);
          }
          else
            { ccedata_full[cenr] += fedata_full[fenr]; }
        }
        else { ; } /** if only one vertex drops, no need to do anything - the connection is already in fdiags ! **/
      }
      // cout << " ccedata_full " << ccedata_full.Size() << endl; prow2(ccedata_full); cout << endl;
      /** Coarse diags, I have to do this here because off-proc entries.
        Maps are only local on master, so cannot cumulate on coarse level **/
      Array<TMU> ccdiags(NCV);
      TM Qij(0), Qji(0); SetIdentity(Qij); SetIdentity(Qji);
      for (auto cvnr : Range(ccvdata)) {
        auto fvs = c2fv[cvnr];
        if (fvs.Size() == 1)
          { ccdiags[cvnr] = fdiags[fvs[0]]; }
        else
        { /** sum up diags, remove contribution of connecting edge **/
          if constexpr(robust) {
            ENERGY::ModQs(fvdata[fvs[0]], fvdata[fvs[1]], Qij, Qji);
            ENERGY::SetQtMQ(1.0, ccdiags[cvnr], Qji, fdiags[fvs[0]]);
            ENERGY::AddQtMQ(1.0, ccdiags[cvnr], Qij, fdiags[fvs[1]]);
          }
          else
            { ccdiags[cvnr] = fdiags[fvs[0]] + fdiags[fvs[1]]; }
          /** note: this should be fine, on coarse level, at least one vert is already in an agg,
          or it would already have been paired in first round **/
          double fac = ( (round == 0) && use_hack_stab ) ? -1.5 : -2.0;
          /** note: coarse vert is exactly ad edge-midpoint **/
          ccdiags[cvnr] -= fac * fedata_full[int(fecon(fvs[0], fvs[1]))];
        }
      }
      // cout << " ccdiags " << ccdiags.Size() << endl; prow2(ccdiags); cout << endl;

      Array<double> ccmtrod(0);
      if (need_mtrod) {
        ccmtrod.SetSize(NCV); ccmtrod = 0;
        for (auto k : Range(vmap)) {
        auto cvnr = vmap[k];
        if (cvnr != -1)
          { ccmtrod[cvnr] += fmtrod[k]; }
        }
      }

      if (havelocinfo && print_aggs) {
        cout << "   round aggs: " << endl;
        cout << locmap->template GetMapC2F<NT_VERTEX>();
      }

      /** Concatenate local maps **/
      conclocmap = (round == 0) ? locmap : conclocmap->ConcatenateLCM(locmap);

      // both are nullptr, but let us make it explicit
      // curr_master_verts = conclocmap->GetSolidCVerts();
      curr_master_verts = nullptr;

      curr_allowed_edges = conclocmap->GetAllowedCEdges();

      fvdata.Assign(0, lh); cvdata = std::move(ccvdata);
      fedata_full.Assign(0, lh); cedata_full = std::move(ccedata_full);
      fedata.Assign(0, lh); GetApproxEdgeData<TMU, double>(cedata_full, cedata);
      fdiags.Assign(0, lh); cdiags = std::move(ccdiags);
      fmtrod.Assign(0, lh); cmtrod = std::move(ccmtrod);

      fvdata.Assign(cvdata);
      fedata.Assign(cedata);
      fedata_full.Assign(cedata_full);
      fdiags.Assign(cdiags);
      fmtrod.Assign(cmtrod);
    }
    else if (round == 0 ) // only 1 round of pairing
     { conclocmap = locmap; }
    else {
      /** Only concatenate vertex map, no coarse mesh **/
      conclocmap->Concatenate(NCV, vmap);
    }

    // rej[0] = eqc_h.GetCommunicator().Reduce(rej[0], NG_MPI_SUM);
    // rej[1] = eqc_h.GetCommunicator().Reduce(rej[1], NG_MPI_SUM);
    if (m_print_summs) {
      auto redc = [&](auto & tup) {
        for (auto k : Range(7)) {
          auto redval = eqc_h.GetCommunicator().Reduce(tup[k], NG_MPI_SUM, 0);
          if (eqc_h.GetCommunicator().Rank() == 0)
            { tup[k] = redval; }
        }
      };
      redc(rej); redc(acc);
      allrej += rej; allacc += acc;
      int NFVG;
      if (round == 0)
        { NFVG = M.template GetNNGlobal<NT_VERTEX>(); }
      else {
        NFVG = fecon.Height();
        int rval = eqc_h.GetCommunicator().Reduce(NFVG, NG_MPI_SUM, 0);
        if ( eqc_h.GetCommunicator().Rank() == 0)
        { NFVG = rval; }
      }
      if (print_summs) {
        /** TODO: use print_summs here instead, but for now keep it as on rank 1 until stable! **/
        auto prtt = [&](auto tup) { for (auto l : Range(6)) { cout << tup[l] << " / "; } cout << endl; };
        cout << "  rej for reasons check_pair/check_agg/mmax_scal/mmx_mat/l2/allpair/noallow " << endl;
        cout << "  round " << round << " rej = "; prtt(rej);
        cout << "  cumulative  rej   = "; prtt(allrej);
        cout << "  no valid partner for " << rej[6] << ", frac = " << double(rej[6])/NFVG << endl;
        cout << "  acc for reasons check_pair/check_agg/mmax_scal/mmx_mat/l2/allpair/noallow " << endl;
        cout << "  round " << round << " acc = "; prtt(acc);
        cout << "  cumulative acc   = "; prtt(allacc);
      }
      rej = 0; acc = 0;
    }
  } // round-loop


  if ( this->weed_out && ( eqc_h.GetCommunicator().Size() > 2 ) )
  {
    std::cout << " WARNING, spw_wo is turned on but final orphan pass only works in serial, skipping it! " << std::endl;
  }

  if ( this->weed_out && ( eqc_h.GetCommunicator().Size() <= 2 ) )
  {
    auto aggs = conclocmap->template GetMapC2F<NT_VERTEX>();

    auto fecon = *conclocmap->GetMesh()->GetEdgeCM();

    auto vmap = conclocmap->template GetMap<NT_VERTEX>();

    int cntSingles = 0;
    Array<IVec<2, int>> joins;
    for (auto K : Range(aggs))
    {
      auto agg = aggs[K];
      if (agg.Size() == 1)
      {
        cntSingles++;
        // for first test, just add it anywhere!
        auto v = agg[0];
        auto neibs = fecon.GetRowIndices(v);
        int partner = -1;
        for (auto j : Range(neibs))
        {
          auto jNeib = neibs[j];
          auto J = vmap[jNeib];
          if (J != -1)
          {
            auto agg_j = aggs[J];
            if ( (agg_j.Size() > 1) &&
                 allow_merge(v,     M.template GetEQCOfNode<NT_VERTEX>(v),
                             jNeib, M.template GetEQCOfNode<NT_VERTEX>(jNeib)) )
            {
              partner = J;
              break;
            }
          }
        }
        if (partner != -1)
        {
          joins.Append(IVec<2, int>(K, partner));
        }
      }
    }
// 
    // std::cout << " FOUND NEW AGGS FOR " << joins.Size() << " of " << cntSingles << " singles! " << std::endl;

    if (joins.Size())
    {
      Array<int> cVmap(aggs.Size());

      cVmap = 1;
      for (auto k : Range(joins))
      {
        cVmap[joins[k][0]] = 0;
      }    

      int cnt = 0;
      for (auto k : Range(cVmap))
      {
        auto newCnt = cnt + cVmap[k];      
        cVmap[k] = cnt;
        cnt = newCnt;
      }

      for (auto k : Range(joins))
      {
        cVmap[joins[k][0]] = cVmap[joins[k][1]];
      }

      int NCV = cnt;
      conclocmap->Concatenate(NCV, cVmap);

      // std::cout << " REDUCED " << aggs.Size() << " -> " << NCV << std::endl;
    }
  }


  /** Build final aggregates **/
  tfaggs.Start();
  auto fixed_aggs = this->GetFixedAggs();
  size_t n_aggs_spec = spec_aggs.Size(), n_aggs_p = conclocmap->template GetMappedNN<NT_VERTEX>(), n_aggs_f = fixed_aggs.Size();
  size_t n_aggs_tot = n_aggs_spec + n_aggs_p + n_aggs_f, agg_id = n_aggs_spec;
  if (m_print_summs) {
    int nspacs = eqc_h.GetCommunicator().Reduce(int(n_aggs_spec), NG_MPI_SUM, 0);
    if (eqc_h.GetCommunicator().Rank() > 0)
      { nspacs = n_aggs_spec; }
    if (print_summs)
      { cout << " # spec aggs = " << nspacs << endl; }
  }

  agglomerates.SetSize(n_aggs_tot);
  for (auto k : Range(n_aggs_spec))
    { agglomerates[k] = std::move(spec_aggs[k]); }

  auto set_agg = [&](int agg_nr, auto vs) {
    auto & agg = agglomerates[agg_nr];
    agg.id = agg_nr;
    int ctr_eqc = M.template GetEQCOfNode<NT_VERTEX>(vs[0]), v_eqc = -1;
    agg.ctr = vs[0]; // TODO: somehow mark agg ctrs [[ must be in the largest eqc ]] - which one is the best choice?
    agg.mems.SetSize(vs.Size());
    for (auto l : Range(vs)) {
      v_eqc = M.template GetEQCOfNode<NT_VERTEX>(vs[l]);
      if ( (v_eqc != 0) && (ctr_eqc != v_eqc) && (eqc_h.IsLEQ( ctr_eqc, v_eqc) ) )
        { agg.ctr = vs[l]; ctr_eqc = v_eqc; }
      agg.mems[l] = vs[l];
      v_to_agg[vs[l]] = agg_nr;
    }
  };

  /** aggs from pairing **/
  auto c2fv = conclocmap->template GetMapC2F<NT_VERTEX>();
  // cout << " final vmap " << endl; prow2(conclocmap->template GetMap<NT_VERTEX>()); cout << endl;
  // cout << " final C->F vertex " << endl << c2fv << endl;
  for (auto agg_nr : Range(n_aggs_p)) {
    auto aggvs = c2fv[agg_nr];
    QuickSort(aggvs);
    set_agg(agg_id++, aggvs);
  }
  /** pre-determined fixed aggs **/
  // TODO: should I filter fixed_aggs by master here, or rely on this being OK from outside?
  for (auto k : Range(fixed_aggs))
    { set_agg(agg_id++, fixed_aggs[k]); }
  tfaggs.Stop();

  if (havelocinfo && print_aggs) {
    cout << " SPW FINAL agglomerates : " << agglomerates.Size() << endl;
    cout << agglomerates << endl;
  }

  MapVertsTest (agglomerates, v_to_agg);

} // SPWAgglomerator::FormAgglomerates_impl


template<class ENERGY, class TMESH, bool ROBUST>
void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: CalcCMK (const BitArray & skip, const SparseMatrix<double> & econ, Array<int> & cmk)
{
/** Calc a CMK ordering of vertices. Connectivity given by "econ". Only orders those where "skip" is not set **/
static Timer t("CalcCMK"); RegionTimer rt(t);
size_t numtake = econ.Height() - skip.NumSet();
if (numtake == 0)
	{ cmk.SetSize0(); return; }
cmk.SetSize(numtake); cmk = -12;
size_t cnt = 0, cnt2 = 0;
BitArray handled(skip);
Array<int> neibs;
while (cnt < numtake) {
	/** pick some minimum degree vertex to start with **/
	int mnc = econ.Height(), nextvert = -1; // max # of possible cons should be H - 1
	int ncons;
	for (auto k : Range(econ.Height()))
if (!handled.Test(k))
	if ( (ncons = econ.GetRowIndices(k).Size()) < mnc )
	{ mnc = ncons; nextvert = k; }
	if (nextvert == -1) // all verts are fully connected (including to itself, which should be illegal)
for (auto k : Range(econ.Height()))
	if (!handled.Test(k))
	{ nextvert = k; break; }
	cmk[cnt++] = nextvert; handled.SetBit(nextvert);
	while((cnt < numtake) && (cnt2 < cnt)) {
/** add (unadded) neibs of cmk[cnt2] to cmk, ordered by degree, until we run out of vertices
	to take neibs from - if that happens, we have to pick a new minimum degree vertex to start from! **/
int c = 0; auto ris = econ.GetRowIndices(cmk[cnt2]);
neibs.SetSize(ris.Size());
for(auto k : Range(ris)) {
	auto neib = ris[k];
	if (!handled.Test(neib))
	{ neibs[c++] = neib; handled.SetBit(neib); }
}
neibs.SetSize(c);
QuickSort(neibs, [&](auto vi, auto vj) { return econ.GetRowIndices(vi).Size() < econ.GetRowIndices(vj).Size(); });
for (auto l : Range(c))
	{ cmk[cnt++] = neibs[l]; handled.SetBit(neibs[l]); }
cnt2++;
	}
}
} // SPWAgglomerator::CalcCMK

/** END SPWAgglomerator **/


template<class ENERGY, class TMESH, bool ROBUST>
void SPWAgglomerator<ENERGY, TMESH, ROBUST> :: MapVertsTest (FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg)
{
/** TODO: remove this again, but for now keep it so I can overload it in one of the cpp files and do some checks... **/ ;
}

} // namespace amg

#endif // SPW_AGG

#endif // FILE_SPW_AGG_IMPL_HPP
