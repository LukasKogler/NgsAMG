#ifndef FILE_AUX_MAT_HPP
#define FILE_AUX_MAT_HPP

#include <base.hpp>
#include <utils.hpp>

#include <alg_mesh.hpp>
#include <utils_denseLA.hpp>

namespace amg
{


template<class ENERGY, class TMESH, class TSPM, class TLAM>
shared_ptr<TSPM>
AssembleAhatSparse (TMESH const &FM,
                    bool  const &includeVertexContribs,
                    TLAM lamAdd,
                    BitArray *exclude = NULL)
{
  int const NV = FM.template GetNN<NT_VERTEX>();

  auto const &econ = *FM.GetEdgeCM();

  auto fvdata = get<0>(FM.Data())->Data();
  auto fedata = get<1>(FM.Data())->Data();

  Array<int> perow(NV);

  for (auto k : Range(NV))
    { perow[k] = 1 + econ.GetRowIndices(k).Size(); }

  auto spAhat = make_shared<TSPM>(perow, NV);

  for (auto k : Range(NV))
  {
    auto ris = spAhat->GetRowIndices(k);
    auto rvs = spAhat->GetRowValues(k);

    auto neibs = econ.GetRowIndices(k);

    int const pos = merge_pos_in_sorted_array(k, neibs);

    if ( pos > 0)
      { ris.Range(0, pos) = neibs.Range(0, pos); }

    ris[pos] = k;

    if ( pos + 1 < ris.Size())
      { ris.Part(pos + 1) = neibs.Part(pos); }

    rvs = 0.0;
  }

  typedef typename strip_mat<typename ENERGY::TM>::type TM;
  Matrix<TM> eblock(2, 2);

  /**
   * Only assemble master-edges, thus with CUMULATED edge-data we get
   * a C2D compatible matrix!
   */
  FM.template ApplyEQ<NT_EDGE>([&](auto eqc, const auto &fEdge)
  {
    if ( exclude && ( exclude->Test(fEdge.v[0]) || exclude->Test(fEdge.v[1]) ) )
      { return; }

    auto ris0 = spAhat->GetRowIndices(fEdge.v[0]);
    auto rvs0 = spAhat->GetRowValues(fEdge.v[0]);

    // auto col00 = find_in_sorted_array(fEdge.v[0], ris0);
    // auto col01 = find_in_sorted_array(fEdge.v[1], ris0);

    auto ris1 = spAhat->GetRowIndices(fEdge.v[1]);
    auto rvs1 = spAhat->GetRowValues(fEdge.v[1]);

    // auto col10 = find_in_sorted_array(fEdge.v[0], ris1);
    // auto col11 = find_in_sorted_array(fEdge.v[1], ris1);

    ENERGY::CalcRMBlock(
      fedata[fEdge.id],
      fvdata[fEdge.v[0]],
      fvdata[fEdge.v[1]],
      [&](auto li, auto lj, auto const &v)
      {
        auto &rvs = li == 0 ? rvs0 : rvs1;
        auto &ris = li == 0 ? ris0 : ris1;

        auto const col = find_in_sorted_array(fEdge.v[lj], ris);

        lamAdd(rvs[col], v);
      });

    // ENERGY::CalcRMBlock(eblock, fedata[fEdge.id], fvdata[fEdge.v[0]], fvdata[fEdge.v[1]]);

    // lamAdd(rvs0[col00], eblock(0, 0));
    // lamAdd(rvs0[col01], eblock(0, 1));
    // lamAdd(rvs1[col10], eblock(1, 0));
    // lamAdd(rvs1[col11], eblock(1, 1));
  }, true);

  if (includeVertexContribs)
  {
    auto &Ahat = *spAhat;

    FM.template ApplyEQ<NT_VERTEX>([&](auto eqc, const auto &vNr) {
      lamAdd(Ahat(vNr, vNr), ENERGY::GetVMatrix(fvdata[vNr]));
    }, true);
  }

  return spAhat;
} // AssembleAhatSparse

template<class ENERGY, class TMESH>
shared_ptr<stripped_spm<typename ENERGY::TM>>
AssembleAhatSparse (TMESH const &FM,
                    bool  const &includeVertexContribs,
                    BitArray *exclude = nullptr)
{
  return AssembleAhatSparse<ENERGY, TMESH, stripped_spm<typename ENERGY::TM>>(
    FM,
    includeVertexContribs,
    [&](auto &a, const auto &b) { a += b; },
    exclude
  );
} // AssembleAhatSparse


template<class ENERGY, class TMESH, class TSPM, class TLAM>
void
addInAggregateEdgeBoost (TMESH const &FM,
                         bool  const &includeVertexContribs,
                         double const &theta,
                         FlatArray<int> vmap,
                         shared_ptr<TSPM> spA,
                         TLAM lamAdd,
                         BitArray *exclude = NULL)
{
  typedef typename ENERGY::TM TM;

  auto const &A = *spA;

  int const FNV = FM.template GetNN<NT_VERTEX>();

  auto const &econ = *FM.GetEdgeCM();

  FM.CumulateData();

  auto fvdata = get<0>(FM.Data())->Data();
  auto fedata = get<1>(FM.Data())->Data();

  Array<TM> diags(FNV);

  for (auto k : Range(FNV))
  {
    if ( exclude && exclude->Test(k) )
    {
      diags[k] = 0.0;
    }
    else
    {
      diags[k] = A(k,k);
    }
  }

  FM.template AllreduceNodalData<NT_VERTEX>(diags, [&](auto tab_in) LAMBDA_INLINE { return sum_table(tab_in); });

  Array<double> lambda(50);

  FM.template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes)
  {
    for (auto k : Range(FNV))
    {
      auto const CV = vmap[k];

      if ( CV == -1 )
      {
        continue;
      }

      auto ris = A.GetRowIndices(k);
      auto rvs = A.GetRowValues(k);

      lambda.SetSize(ris.Size());

      double lsum = .0;

      TM diag = diags[k];

      for (auto j : Range(ris))
      {
        auto neib = ris[j];

        if ( ( neib != k ) && ( vmap[neib] == CV ) )
        {
          double const tr = calc_trace(rvs[j]);

          lambda[j] = tr;
          lsum += tr;
        }
      }

      if ( lsum == 0.0 )
      {
        continue;
      }

      for (auto j : Range(ris))
      {
        auto neib = ris[j];

        if ( neib == k )
        {
          // diag is cumulated, rvs could be distr. val, we are adding
          // the cumulated bonus on the master so it is OK either way!
          rvs[j] += theta * diag;
        }
        else if ( vmap[neib] == CV )
        {
          // !! Q^{neib -> k} !!
          auto const Q = ENERGY::GetQiToj(fvdata[neib], fvdata[k]);

          double const fac = -1 * lambda[j] / lsum * theta;

          // a_ij -> a_ij - lambda_j * theta * diag * Q^{j->i}
          rvs[j] += Q.GetMQ(fac, diag);
        }
      }
    }
  },
  true); // master only
}

/**
 * A version of AssembleAhatSparse where we inflate the importance of connections leading to
 * neighbours in the same aggregate:
 *    ahat_ij -> ahat_ij + theta * lambda_j Q^{i->j} ahat_ii
 *    ahat_ii -> (1 + theta) ahat_ii
 * where
 *    sum_{j \in { neighbors in same agg}} lambda_j = 1
 * A single prolongation smoothing step with this modified Ahat, P_pw -> (I-Dhat^{-1}Ahat)P_pw,
 * gives the same result as a dampened prolongation smoothing step with normal Ahat, 
 * P_pw -> (I-\omega \cdot Dhat^{-1}Ahat)P_pw with \omega = 1 / (1 + \theta).
 *
 * This is true if using full columns in smoothed prol and only for a single smoothing step!
 *
 * TODO: look into adding vertex-contribs in a similar manner, keep a_ii + vertex_contrib, but add
 * compensating -QT * contrib * Q to off-diagonal to keep RBs in kernel or something
 */
template<class ENERGY, class TMESH, class TSPM, class TLAM>
shared_ptr<TSPM>
AssembleAhatSparseStabSP (TMESH const &FM,
                          bool  const &includeVertexContribs,
                          double const &theta,
                          FlatArray<int> vmap,
                          TLAM lamAdd,
                          BitArray *exclude = NULL)
{
  if ( includeVertexContribs )
  {
    throw Exception("AssembleAhatSparseStabSP with vertex-contribs is TODO!");
  }

  auto spAhat = AssembleAhatSparse<ENERGY, TMESH, TSPM>(FM, includeVertexContribs, lamAdd, exclude);

  addInAggregateEdgeBoost<ENERGY>(FM, includeVertexContribs, theta, vmap, spAhat, lamAdd, exclude);

  return spAhat;
}

template<class ENERGY, class TMESH>
shared_ptr<stripped_spm<typename ENERGY::TM>>
AssembleAhatSparseStabSP (TMESH const &FM,
                          bool  const &includeVertexContribs,
                          double const &theta,
                          FlatArray<int> vmap,
                          BitArray *exclude = nullptr)
{
  // typedef typename EMERGY::TM TM;

  // static constexpr int BS = Height<TM>();

  return AssembleAhatSparseStabSP<ENERGY, TMESH, stripped_spm<typename ENERGY::TM>>(
    FM,
    includeVertexContribs,
    theta,
    vmap,
    [&](auto &a, const auto &b)
    {
      a += b;
      // for assembling FMAT w. different block-size
      // if constexpr( BS != 1 ) // so it compiles
      // {
      //   Iterate<BSF>([&](auto j) {
      //     Iterate<BSF>([&](auto i) {
      //       a(i.value, j.value) += b(i.value, j.value);
      //     });
      //   });
      // }
      // else
      // {
      //   a += b;
      // }
    },
    exclude
  );
} // AssembleAhatSparse

} // namespace amg

#endif // FILE_AUX_MAT_HPP

