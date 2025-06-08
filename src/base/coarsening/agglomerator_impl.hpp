#ifndef FILE_AGGLOMERATOR_IMPL
#define FILE_AGGLOMERATOR_IMPL

#include "agglomerator.hpp"

#include <utils_denseLA.hpp>

namespace amg
{

template<class ENERGY, class TMESH>
template<class TMU>
INLINE void
VertexAgglomerator<ENERGY, TMESH>::
GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data)
{
  if constexpr(std::is_same<TED, TMU>::value)
  {
    out_data.FlatArray<TMU>::Assign(in_data);
  }
  else if constexpr(std::is_same<TMU, double>::value ||
                    std::is_same<TMU, float>::value)
  { /** use trace **/
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = ENERGY::GetApproxWeight(in_data[k]); }
  }
  else
  {
    /**
    *    i) H1: never called, but make it compile
    *   ii) Stoks: We actually go here for Stokes where energy has two mats per edge.
    *  iii) Elasticity double -> float conversion
    */
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = ENERGY::template GetEMatrix<TScal<TMU>>(in_data[k]); }
  }
} // VertexAgglomerator::GetEdgeData


template<class ENERGY, class TMESH>
template<class TFULL, class TAPPROX>
INLINE void
VertexAgglomerator<ENERGY, TMESH>::
GetApproxEdgeData (FlatArray<TFULL>  in_data,
                  Array<TAPPROX>   &out_data)
{
  if constexpr(std::is_same<TFULL, TAPPROX>::value)
  {
    out_data.FlatArray<TAPPROX>::Assign(in_data);
  }
  else
  {
    out_data.SetSize(in_data.Size());
    for (auto k : Range(in_data))
      { out_data[k] = ENERGY::template GetApproxWeight<TAPPROX>(in_data[k]); }
  }
} // VertexAgglomerator::GetApproxEdgeData


template<class ENERGY, class TMESH>
template<class TMU, class TWEIGHT>
INLINE void
VertexAgglomerator<ENERGY, TMESH>::
InitializeAggData(AgglomerationData<TMESH, ENERGY, TMU, TWEIGHT> &aggData)
{
  static Timer t("VertexAgglomerator::InitializeAggData");
  RegionTimer rt(t);

  constexpr bool ROBUST = Height<TMU>() > 1;

  TMESH       &ncFM = this->GetMesh();
  TMESH const &FM   = ncFM;
  FM.CumulateData();

  auto const &eqc_h = *FM.GetEQCHierarchy();

  const auto FNV = FM.template GetNN<NT_VERTEX>();
  const auto FNE = FM.template GetNN<NT_EDGE>();

  constexpr int BS  = ENERGY::DPV;
  constexpr int BSU = ngbla::Height<TMU>();

  auto baseVData = get<0>(ncFM.AttachedData())->Data();
  auto baseEData = get<1>(ncFM.AttachedData())->Data();

  auto &vData = aggData.vData;
  vData.FlatArray<typename ENERGY::TVD>::Assign(baseVData);

  // always traces, just a flat-array for scalar
  auto &edgeTrace = aggData.edgeTrace;
  GetApproxEdgeData(baseEData, edgeTrace);

  // robust->full edge-matrices, else->traces
  auto &edgeMats = aggData.edgeMats;
  GetEdgeData(baseEData, edgeMats);

  /**
   * In parallel, we cannot take the maximum of approx-edge-weights for vertices in boundary-rows locally.
   * Therefore, pre-compute the difference between the total diagonal trace and the locally computed diagonal trace,
   * that is, the sum of all off-proc diagonal traces and use that.
   * On intermediate levels, we sum up these max-od contribs instead of taking the max. The reason is that strictly
   * speaking we would need the max of summed-up off-proc-leading edge-traces, but we do not map off-proc-leading edges
   * at all.
   *
   * This is MORE RESTRICTIVE than actual max-off-diag traces so it is not a problem stability-wise
   */

  auto &baseDiags    = aggData.auxDiags;
  auto &maxTrOD      = aggData.maxTrOD;
  auto &offProcTrace = aggData.offProcTrace;

  aggData.hasOffProcTrace = eqc_h.IsTrulyParallel();

  maxTrOD.SetSize(FNV);
  baseDiags.SetSize(FNV);
  offProcTrace.SetSize(aggData.hasOffProcTrace ? FNV : 0);

  maxTrOD   = 0;
  baseDiags = 0;

  // assemble local diag - contribs where I am not master of one of the two
  // vertices can (and should) be included in offProcTrace
  FM.template Apply<NT_EDGE>([&](auto const &edge) LAMBDA_INLINE
  {
    auto const vi = edge.v[0];
    auto const vj = edge.v[1];

    if constexpr( ROBUST )
    {
      TMU const E = ENERGY::template GetEMatrix<TScal<TMU>>(baseEData[edge.id]);

      auto [Qij, Qji] = ENERGY::GetQijQji(baseVData[vi],
                                          baseVData[vj]);

      TMU dvi = Qij.GetQTMQ(1.0, E);

      // cout << " ASS FL, edge " << edge << endl;
      // cout << "  E: " << endl; print_tm(cout, E); cout << endl;
      // TMU E_Q = Qij.GetMQ(1.0, E);
      // TMU QT_E_Q = Qij.GetQTM(1.0, E_Q);
      // cout << " E_Q = " << endl; print_tm(cout, E_Q); cout << endl;
      // cout << " QT_E_Q = " << endl; print_tm(cout, QT_E_Q); cout << endl;

      baseDiags[vi] += dvi;
      maxTrOD[vi]    = max(maxTrOD[vi], CalcAvgTrace<TWEIGHT>(dvi));
      // cout << "  dvi contrib " << endl; print_tm(cout, dvi); cout << endl;
      // cout << " baseDiags[" << vi << "] now " << endl; print_tm(cout, baseDiags[vi]); cout << endl;
      // cout << " maxTrOD[" << vi << "] now " << maxTrOD[vi] << endl;

      TMU dvj = Qji.GetQTMQ(1.0, E);
      baseDiags[vj] += dvj;
      maxTrOD[vj]    = max(maxTrOD[vj], CalcAvgTrace<TWEIGHT>(dvj));
      // cout << "  dvj contrib " << endl; print_tm(cout, dvj); cout << endl;
      // cout << " baseDiags[" << vj << "] now " << endl; print_tm(cout, baseDiags[vj]); cout << endl;
      // cout << " maxTrOD[" << vj << "] now " << maxTrOD[vj] << endl;
    }
    else
    {
      auto const &E = ENERGY::template GetApproxWeight<TWEIGHT>(baseEData[edge.id]);

      baseDiags[vi] += E;
      maxTrOD[vi]    = max(maxTrOD[vi], E / BS);

      baseDiags[vj] += E;
      maxTrOD[vj]    = max(maxTrOD[vj], E / BS);
    }
  }, true);

  // stash trace
  if ( eqc_h.IsTrulyParallel() )
  {
    FM.template Apply<NT_VERTEX>([&](auto vi)
    {
      offProcTrace[vi] = calc_trace(baseDiags[vi]);
    }, true);

    // reduce diagonals
    FM.template AllreduceNodalData<NT_VERTEX>(baseDiags, [&](auto tab_in) LAMBDA_INLINE { return sum_table(tab_in); });

    // off-proc contrib = reduced diag - local trace
    FM.template Apply<NT_VERTEX>([&](auto vi)
    {
      offProcTrace[vi] = ( calc_trace(baseDiags[vi]) - offProcTrace[vi] ) / BS;
    }, true);
  }

  // L2 contribution (add AFTER reduction above)
  FM.template Apply<NT_VERTEX>([&](auto v)
  {
    if constexpr( ROBUST )
    {
      auto const &vM = ENERGY::GetVMatrix(baseVData[v]);

      baseDiags[v] += vM;
      maxTrOD[v]    = max(maxTrOD[v], CalcAvgTrace<TWEIGHT>(vM));
    }
    else
    {
      TWEIGHT const vM = ENERGY::template GetApproxVWeight<TWEIGHT>(baseVData[v]);

      baseDiags[v] += vM;
      maxTrOD[v]    = max(maxTrOD[v], vM);
    }
  }, false ); // everyone
} // VertexAgglomerator::PrepareAgglomerationData


} // namespace amg

#endif // FILE_AGGLOMERATOR_IMPL