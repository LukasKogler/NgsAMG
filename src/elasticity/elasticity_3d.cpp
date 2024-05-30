#include "utils_denseLA.hpp"
#ifdef ELASTICITY

#define FILE_AMG_ELAST_CPP
#define FILE_AMG_ELAST_3D_CPP

#include "elasticity.hpp"
#include <amg_pc_vertex.hpp>

#include "elasticity_impl.hpp"
#include <amg_pc_vertex_impl.hpp>
#include "elasticity_pc_impl.hpp"

#include <utils_sparseLA.hpp>
#include <utils_sparseMM.hpp>

#include <plate_test_agg_impl.hpp>

namespace amg
{

// template<> template<>
// shared_ptr<stripped_spm_tm<Mat<3, 6, double>>>
// VertexAMGPC<ElasticityAMGFactory<3>> :: BuildEDC<3> (size_t height, shared_ptr<TopologicMesh> mesh)
// {
//   const auto & O(static_cast<Options&>(*options));

//   const auto & M(*mesh);

//   constexpr int BS  = ElasticityAMGFactory<3>::BS;
//   constexpr int BSA = 3;

//   typedef stripped_spm_tm<Mat<3, 6, double>> T_E_D_C;

//   Array<int> perow(M.template GetNN<NT_VERTEX>()); perow = 1;
//   auto E_D_C = make_shared<T_E_D_C>(perow, M.template GetNN<NT_VERTEX>());
//   for (auto k : Range(perow))
//   {
//     E_D_C->GetRowIndices(k)[0] = k;
//     auto & v = E_D_C->GetRowValues(k)[0];
//     v = 0;
//     Iterate<3>([&](auto i) { v(i.value, 3 + i.value) = 1; });
//   }
//   return E_D_C;
// }


template<> shared_ptr<ElasticityMesh<3>>
ElmatVAMG<ElasticityAMGFactory<3>, double, double> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
{
  return nullptr;
}

INLINE std::tuple<double, double, double> DenseEquivTestAAAA(ngbla::FlatMatrix<double> A, ngbla::FlatMatrix<double> B, ngcore::LocalHeap & lh, bool print = false)
{
  HeapReset hr(lh);

  int const N = A.Height();

  FlatMatrix<double> evecsA(N, N, lh);
  FlatVector<double> evalsA(N, lh);
  TimedLapackEigenValuesSymmetric(A, evalsA, evecsA);

  // cout << " DenseEquivTest, evalsA = " << evalsA << endl;

  // {
  //   FlatMatrix<double> evecsB(N, N, lh);
  //   FlatVector<double> evalsB(N, lh);
  //   LapackEigenValuesSymmetric(B, evalsB, evecsB);
  //   cout << " DenseEquivTest, evalsB = " << evalsB << endl;
  // }

  double eps = max(1e-14, 1e-6 * evalsA(N-1));

  // scale eigenvectors with 1/sqrt(evals(k))
  FlatMatrix<double> sqAi(N, N, lh); sqAi = 0.0;
  int count_zero = 0;
  for (auto k : Range(N)) {
    if (evalsA(k) > eps)
    {
      double fac = 1.0 / sqrt(evalsA(k));
      // sqAi += fac * evecsA.Rows(k, k+1) * Trans(evecsA.Rows(k, k+1));
      for (auto l : Range(N))
        for (auto m : Range(N))
          { sqAi(l,m) += fac * evecsA(k, l) * evecsA(k, m); }
    }
    else
    {
      count_zero++;
    }
  }

  // cout << " sqAi: " << endl << sqAi << endl;

  // sqrt(Ainv) * B * sqrt(Ainv)
  FlatMatrix<double> BAi(N, N, lh);
  BAi = B * sqAi;
  FlatMatrix<double> AiBAi(N, N, lh);
  AiBAi = sqAi * BAi;

  // cout << " AiBAi: " << endl << AiBAi << endl;

  FlatMatrix<double> evecsABA(N, N, lh);
  FlatVector<double> evalsABA(N, lh);
  TimedLapackEigenValuesSymmetric(AiBAi, evalsABA, evecsABA);

  double min_nz = evalsABA(count_zero);

  double kappa = (min_nz == 0.0) ? std::numeric_limits<double>::infinity() : evalsABA(N-1) / min_nz;

  // cout << " min EV row: " << endl;
  // for (auto k : Range(3)) {
  //   for(auto j : Range(N/3))
  //   {
  //     // move such that we have 0 displacement at vertex 0
  //     cout << setw(6) << evecsABA(min_nz, 3*j+k) - evecsABA(min_nz, k)<< " ";
  //   }
  //   cout << endl;
  // }


  cout << " evalsABA: " << endl << evalsABA << endl;

  if (print)
  {
    std::cout << " DenseEqivTest, A kernel dim = " << count_zero
              << ", bounds: " << min_nz << "*A <= B <= " << evalsABA(N-1) << "*A, kappa = "
              << kappa << std::endl;

    FlatMatrix<double> genEvecs(N, N, lh);
    // realEvecs = A^(-1/2) * evecs, but evecs are row-wise, so:
    genEvecs = evecsABA * sqAi;

    FlatVector<double> vec(N, lh);
    FlatVector<double> Avec(N, lh);
    FlatVector<double> Bvec(N, lh);
    
    double ip;

    cout << " smallest eval " << min_nz << endl;//", evec = " << endl << genEvecs.Row(count_zero) << endl;

    // vec = genEvecs.Row(count_zero);

    // Avec = A * vec;
    // Bvec = B * vec;

    // cout << " A * vec = " << endl << Avec << endl;
    // Avec *= min_nz;
    // cout << " lam * A * vec = " << endl << Avec << endl;
    // cout << " B * vec = " << endl << Bvec << endl;


    cout << " largest  eval " << evalsABA(N-1) << endl;//<< ", evec = " << endl << genEvecs.Row(N-1) << endl;

    // vec = genEvecs.Row(N-1);
    // Avec = A * vec;
    // Bvec = B * vec;
    // for (auto k : Range(6))
    // {
    //   ip = InnerProduct(Avec, genEvecs.Row(k));
    //   Avec -= ip * genEvecs.Row(k);

    //   ip = InnerProduct(Avec, genEvecs.Row(k));
    //   Avec -= ip * genEvecs.Row(k);
    // }

    // cout << " A * vec = " << endl << Avec << endl;
    // Avec *= evalsABA(N-1);
    // cout << " lam * A * vec = " << endl << Avec << endl;
    // cout << " B * vec = " << endl << Bvec << endl;

  }

  return std::make_tuple(min_nz, evalsABA(N-1), kappa);
}


template<>
void
ElmatVAMG<ElasticityAMGFactory<3>, double, double> :: 
CalcAuxWeightsALG (FlatArray<int>            dnums,
                   FlatMatrix<double> const &elmat,
                   ElementId                 ei,
                   LocalHeap                &lh)
{
  const auto & O(static_cast<Options&>(*options));

  bool elmat_evs = O.calc_elmat_evs;

  std::cout << " LS-fit aux-elmat for ei " << ei << ", dnums = "; prow2(dnums); cout << std::endl;

  constexpr int B = 3;

  int const n = dnums.Size();
  int const N = n * B;

  int const nE = n * (n - 1) / 2;

  /** position & tangents **/
  FlatArray<Vec<3, double>> pos(n, lh);
  FlatArray<Vec<3, double>> tang(nE, lh);

  Vec<3, double> tmpVec; // GetNodePos needs this for edge-pos
  for (auto k : Range(n))
  {
    GetNodePos<3>(NodeId(NT_VERTEX, dnums[k]), *ma, pos[k], tmpVec);
  }

  int cit = 0;
  auto itEdges = [&](auto lam) {
    int c = 0;
    // cout << " itEdges " << cit << endl;
    for (auto vi : Range(n))
    {
      for (auto vj : Range(vi+1, n))
      {
      // cout << "  " << c << " " << vi << "-" << vj << endl;
        lam(c++, vi, vj);
      }
    }
    cit++;
  };

  itEdges([&](auto k, auto vi, auto vj) {
    tang[k] = pos[ vj ] - pos[ vi ];
    auto const len = L2Norm(tang[k]);
    tang[k] /= len;
  });

  auto tOf = [&](auto const &t, int const &ri, int const &rj) {
    double sum = 0.0;
    Iterate<B>([&](auto ii) { 
      Iterate<B>([&](auto jj) { 
        sum += t(ii.value) * t(jj.value) * elmat(B*ri + ii.value, B*rj + jj.value);
      });
    });
    return sum;
  };

  // system to solve for least-squares fit!
  FlatMatrix<double> theta(nE, nE, lh);
  FlatVector<double> rhs(nE, lh);
  FlatVector<double> sol(nE, lh);

  theta = 0.0;

  itEdges([&](auto eNr, auto vi, auto vj) {

    Vec<3, double> &ti = tang[eNr];

    // rhs(eNr) = tOf(ti, vi, vi) - tOf(ti, vi, vj) - tOf(ti, vj, vi) + tOf(ti, vj, vj);
    rhs(eNr) = tOf(ti, vi, vi) - 2 * tOf(ti, vi, vj) + tOf(ti, vj, vj);

    theta(eNr, eNr) = 4.0;

    for (auto eJ : Range(eNr + 1, nE))
    {
      Vec<3, double> &tj = tang[eJ];

      double const ip = InnerProduct(ti, tj);

      theta(eNr, eJ) = ip * ip;
      theta(eJ, eNr) = ip * ip;

      // double const fac = (eJ == eNr) ? 1.0 : -1.0;
      // theta(eNr, eJ) = fac * ip * ip;
    }
  });

  // cout << " THETA " << endl << theta << endl;

  CalcInverse(theta);
  sol = theta * rhs;

  // cout << " THETA INV " << endl << theta << endl;
  // cout << " rhs " << endl << rhs << endl;
  // cout << " sol " << endl << sol << endl;


  /** Assemble aux-elmat and test approximation quality! */

  FlatMatrix<double> auxElmat(N, N, lh);

  auxElmat = 0.0;

  for (auto eNr : Range(nE))
  {
    int const vi = eNr;
    int const vj = (eNr + 1) % n;

    Vec<3, double> &t = tang[eNr];

    Iterate<B>([&](auto i) {
      Iterate<B>([&](auto j) {
        double const val = sol(eNr) * t(i.value) * t(j.value);
        auxElmat(vi * B + i.value, vi * B + j.value) += val;
        auxElmat(vi * B + i.value, vj * B + j.value) -= val;
        auxElmat(vj * B + i.value, vi * B + j.value) -= val;
        auxElmat(vj * B + i.value, vj * B + j.value) += val;
      });
    });
  }

  // cout << " ELMAT " << ei << endl << elmat << endl;
  // cout << " LS-AUX ELMAT " << ei << endl << auxElmat << endl;

  if (elmat_evs)
  {
    // bool print = O.log_level_pc == Options::LOG_LEVEL_PC::DBG;
    bool const print = true;
    if ( print )
    {
      std::cout << " Test LS elmat " << ei << std::endl;
    }
    auto [minev, maxev, kappa] = DenseEquivTestAAAA(elmat, auxElmat, lh, print);
    this->elmat_evs[0] = min(minev, this->elmat_evs[0]);
    this->elmat_evs[1] = max(maxev, this->elmat_evs[1]);
    this->elmat_evs[2] = max(kappa, this->elmat_evs[2]);
  }

  auxElmat -= elmat;
  // cout << " AUX-ELMAT - ELMAT: " << endl << auxElmat << endl;
  cout << " L2-DIFF " << L2Norm(auxElmat)<< endl;
  cout << " REL L2-diff " << L2Norm(auxElmat)/L2Norm(elmat) << endl;
}

template<>
void
ElmatVAMG<ElasticityAMGFactory<3>, double, double> :: 
CalcAuxWeightsSC (FlatArray<int>            dnums,
                  FlatMatrix<double> const &elmat,
                  ElementId                 ei,
                  LocalHeap                &lh)
{
  const auto & O(static_cast<Options&>(*options));

  bool elmat_evs = O.calc_elmat_evs;

  cout << " CalcAuxWeightsSC, dnums = "; prow2(dnums); cout << endl;
  int const n  = dnums.Size();
  int const nb = 2;
  int const ni = n - nb;

  constexpr int B = 3;
  int const N  = dnums.Size() * B;
  int const NB = nb * B;
  int const NI = ni * B;

  HeapReset hr(lh);

  FlatMatrix<double> S(NB, NB, lh);
  FlatMatrix<double> Aii(NI, NI, lh);
  FlatMatrix<double> A_bi_A_ii(NB, NI, lh);
  FlatMatrix<double> auxElmat(N, N, lh);

  FlatArray<int> bRows(NB, lh);
  FlatArray<int> iRows(NI, lh);

  auxElmat = 0.0;

  for (auto i : Range(n))
  {
    for (auto l : Range(i))
    {
      Iterate<B>([&](auto ll){
        iRows[l * B + ll.value] = l * B + ll;
      });
    }
    Iterate<B>([&](auto ii){
      bRows[ii.value] = i * B + ii;
    });
    
    for (auto j : Range(i + 1, n))
    {
      for (auto l : Range(i + 1, j))
      {
        Iterate<B>([&](auto ll){
          iRows[(l - 1) * B + ll.value] = l * B + ll;
        });
      }
      Iterate<B>([&](auto jj){
        bRows[B + jj.value] = j * B + jj;
      });
      for (auto l : Range(j + 1, n))
      {
        Iterate<B>([&](auto ll){
          iRows[(l - 2)* B + ll.value] = l * B + ll;
        });
      }

      // cout << " i, j = " << i << " " << j << endl;
      // cout << " iRows = "; prow2(iRows); cout << endl;
      // cout << " bRows = "; prow2(bRows); cout << endl;


      Aii = elmat.Rows(iRows).Cols(iRows);
      CalcPseudoInverseNew(Aii, lh);
      A_bi_A_ii = elmat.Rows(bRows).Cols(iRows) * Aii;
      S = elmat.Rows(bRows).Cols(bRows);
      S -= A_bi_A_ii * elmat.Rows(iRows).Cols(bRows);

      // cout << " S = " << endl << S << endl;

      auxElmat.Rows(bRows).Cols(bRows) += S;

      // addToTM((*ht_edge)[IVec<2, int>(dnums[j], dnums[i]).Sort()], 1.0, 0, 0, S);
    }
  }

  cout << " EMAT " << ei << endl << elmat << endl;
  cout << " AUX-EMAT " << ei << endl << auxElmat << endl;

  if (elmat_evs)
  {
    // bool print = O.log_level_pc == Options::LOG_LEVEL_PC::DBG;
    bool const print = true;
    if ( print )
    {
      std::cout << " Test elmat " << ei << std::endl;
    }
    auto [minev, maxev, kappa] = DenseEquivTestAAAA(elmat, auxElmat, lh, print);
    this->elmat_evs[0] = min(minev, this->elmat_evs[0]);
    this->elmat_evs[1] = max(maxev, this->elmat_evs[1]);
    this->elmat_evs[2] = max(kappa, this->elmat_evs[2]);
  }

  auxElmat -= elmat;
  cout << " L2-DIFF " << L2Norm(auxElmat)<< endl;
  // cout << " AUX-ELMAT - ELMAT: " << endl << auxElmat << endl;
  cout << " REL L2-diff " << L2Norm(auxElmat)/L2Norm(elmat) << endl;

  // CalcAuxWeightsALG(dnums, elmat, ei, lh);

  // cout << " auxElmat: " << endl << auxElmat << endl;

  // CalcPseudoInverseNew(auxElmat, lh);

  // cout << " SC to first 2: " << endl << S << endl;

  // FlatMatrix<double> Abb(6, 6, lh);
  // (*ht_egge)[IVec<2, int>(dnums[j], dnums[i]).Sort()] += weight;

} // ElmatVAMG::AddElementMatrix


template<>
void
ElmatVAMG<ElasticityAMGFactory<3>, double, double> :: 
CalcAuxWeightsLSQ (FlatArray<int>            dnums,
                   FlatMatrix<double> const &elmat,
                   ElementId                 ei,
                   LocalHeap                &lh)
{
  if (ei.Nr() > 100)
  {
    return;
  }

  /**
   * Ensure ||A - Ahat||^2 -> min by obtaining alpha as the solution of
   *    C alpha = b
   */

  cout << endl << endl << " CalcAuxWeightsLSQ for EI " << ei << endl;


  int const nV = dnums.Size();
  int const nE = ( nV * ( nV - 1 ) ) / 2;

  FlatArray<Vec<3, double>> pos(nV, lh);

  Vec<3, double> tmpVec; // GetNodePos needs this for edge-pos
  for (auto k : Range(nV))
  {
    GetNodePos<3>(NodeId(NT_VERTEX, dnums[k]), *ma, pos[k], tmpVec);
  }


  FlatMatrix<double> C(nE, nE, lh);
  FlatVector<double> b(nE, lh);
  FlatVector<double> alpha(nE, lh);

  C = 0;
  b = 0;

  auto eNr = [&](auto i, auto j)
  {
    // indexing for upper triangular (without diag) matrix:
    // position (k, l), with L > k in the values is off_k + (l - k - 1) where
    //   off_k = k*N - (k (k+1))/2;
    // that is k*(n-1) - ( k(k+1)/2 ) + l - 1
    auto const k   = std::min(i, j);
    auto const l   = std::max(i, j);

    // auto const off = k * nV - ( k * ( k + 1 ) ) / 2;
    // return off + l - k - 1;

    return k * ( nV- 1 ) - ( k * ( k + 1 ) ) / 2 + l - 1;
    // return off + l - k - 1;
  };

  auto itEdges = [&](auto lam)
  {
    int cntE = 0;

    for (auto k : Range(nV))
    {
      for (auto j : Range(k + 1, nV))
      {
        int locEId = cntE++;

        lam(k, j, locEId);
      }
    }

  };

  auto itEdgePairs = [&](auto lam)
  {
    for (auto i : Range(nV)) // loop over all vertices
    {
      // inner loop over pairs of neibs
      for (auto j : Range(nV))
      {
        if ( i != j )
        {
          int locEij = eNr(i, j);

          for (auto k : Range(j+1, nV)) // no duplicates!
          {
            if ( k != i )
            {
              int locEik = eNr(i, k);

              lam(i, j, locEij, k, locEik);
            }
          }
        }
      }
    }

  };
  
  itEdges([&](auto vi, auto vj, auto locEid)
  {
    // b_{ij} = < (A_ii - A_ij - A_ji + A_jj) tij, tij >
    Mat<DIM, DIM, double> A = 0;

    cout << " b, " << vi << "-" << vj << " as locEid " << locEid << endl;

    auto const offi = DIM * vi;
    auto const offj = DIM * vj;

    addToTM(A,  1.0, elmat, offi, offi);
    addToTM(A, -1.0, elmat, offi, offj);
    addToTM(A, -1.0, elmat, offj, offi);
    addToTM(A,  1.0, elmat, offj, offj);

    Vec<DIM, double> tang = pos[vj] - pos[vi];

    Vec<DIM, double> At = A * tang;

    double const Att = InnerProduct(At, tang);
    // double const Att = L2Norm(A);

    b(locEid) = Att;

    // C_{ij, ij} = \sum_kl 4 alpha_kl <tij, tij>^2
    double const tij_4   = sqr(InnerProduct(tang, tang));
    // double const tij_4   = sqr(sqr(L2Norm(tang)));

    C(locEid, locEid) += 4 * tij_4;
  });


  int minI = 0;
  int minJ = 0;
  int minK = 0;
  double maxCOS = 0;

  itEdgePairs([&](auto vi, auto vj, auto locEij, auto vk, auto locEik)
  {
    cout << " C, " << vi << "-" << vj << " as " << locEij << " and " << vi << "-" << vk << " as " << locEik << endl;

    Vec<DIM, double> tij = pos[vj] - pos[vi];
    Vec<DIM, double> tik = pos[vk] - pos[vi];

    // cout << "   pi: "; prow(pos[vi]); cout << endl;
    // cout << "   pj: "; prow(pos[vj]); cout << endl;
    // cout << "   pk: "; prow(pos[vk]); cout << endl;

    cout << "   tij "; prow(tij); cout << endl;
    cout << "   tik "; prow(tik); cout << endl;

    double const tij_tik = sqr(InnerProduct(tij, tik));

    double const cos = InnerProduct(tij, tik) / L2Norm(tij) / L2Norm(tik);

    if (cos > maxCOS)
    {
      maxCOS = cos;
      minI = vi;
      minJ = vj;
      minK = vk;
    }
    cout << "   cos " << vi << "-" << vj << "x" << vi << "-" << vk << " = " << cos << endl;


    // double const tij_tik = sqr(L2Norm(tij) * L2Norm(tik));

    // C_{ij, kl} = 2 * alpha_kl <tij, tkl>^2
    C(locEij, locEik) += tij_tik;
    C(locEik, locEij) += tij_tik;
  });

  cout << " MAX cos " << minI << "-" << minJ << "x" << minI << "-" << minK << " = " << maxCOS << endl;

  // cout << " C: " << endl << C << endl;

  CalcInverse(C);

  // cout << " C inv: " << endl << C << endl;

  alpha = C * b;

  // cout << " b: " << endl << b << endl;
  // cout << " alpha: " << endl << alpha << endl;

  FlatMatrix<double> auxElmat(DIM * nV, DIM * nV, lh);

  auxElmat = 0;

  itEdges([&](auto vi, auto vj, auto locEid)
  {
    Vec<DIM, double> tij = pos[vj] - pos[vi];
    Mat<DIM, DIM, double> T = 0;

    Iterate<DIM>([&](auto i) {
      Iterate<DIM>([&](auto j) {
        T(i, j) = tij(i) * tij(j);
      });
    });

    double const alpha_ij = abs(alpha(locEid));

    auto const offi = vi * DIM;
    auto const offj = vj * DIM;

    addTM(auxElmat, offi, offi,  alpha_ij, T);
    addTM(auxElmat, offi, offj, -alpha_ij, T);
    addTM(auxElmat, offj, offi, -alpha_ij, T);
    addTM(auxElmat, offj, offj,  alpha_ij, T);
  });
  
  // cout << " elmat:    " << endl << elmat << endl;
  // cout << " auxElmat: " << endl << auxElmat << endl;

  // if (elmat_evs)
  {
    // bool print = O.log_level_pc == Options::LOG_LEVEL_PC::DBG;
    bool const print = true;
    if ( print )
    {
      std::cout << " Test elmat " << ei << std::endl;
    }
    auto [minev, maxev, kappa] = DenseEquivTestAAAA(elmat, auxElmat, lh, print);
    // this->elmat_evs[0] = min(minev, this->elmat_evs[0]);
    // this->elmat_evs[1] = max(maxev, this->elmat_evs[1]);
    // this->elmat_evs[2] = max(kappa, this->elmat_evs[2]);
  }

  auxElmat -= elmat;
  // cout << " AUX-ELMAT - ELMAT: " << endl << auxElmat << endl;
  cout << " L2-DIFF " << L2Norm(auxElmat)<< endl;
  cout << " REL L2-diff " << L2Norm(auxElmat)/L2Norm(elmat) << endl;

} // ElmatVAMG::AddElementMatrix



template<>
void ElasticityAMGFactory<3> :: CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  auto gcomm = amg_levels[0]->cap->eqc_h->GetCommunicator();

  auto intr = [&](auto tmmm) {
    cout << endl;
    auto mvd = get<0>(tmmm->Data())->Data();
    cout << " eqc vs: " << endl;
    tmmm->template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
      if (nodes.Size() > 0)
        cout << " eqc " << eqc << " = [" << nodes[0] << " ... " << nodes.Last() << "]" << endl;
      else
        cout << " eqc " << eqc << " = []" << endl;
    }, false);
    cout << " v POS (max. first 100): " << endl;
    size_t S = min(size_t(100), mvd.Size());
    for (auto k : Range(S))
      { cout << k << " " << mvd[k].pos << endl; }
    cout << endl;
  };

  cout << " Check KVECS, meshes" << endl;

  for (auto l : Range(amg_levels)) {
    cout << " level " << l << endl;
    if (auto cap = amg_levels[l]->cap) {
      if (auto mesh = cap->mesh) {
        intr(static_pointer_cast<TMESH>(mesh));
      }
      else
        { cout << " no mesh " << endl; }
    }
    else
      { cout << " no cap " << endl; }
  }


  auto chkab = [&](auto fva, auto fvb, int n, shared_ptr<BitArray> fnodes, string title) {
    if ( n == 0)
      { return; }
    cout << " check " << title << endl;
    int bs = fva.Size() / n;
    int numdf = 0;
    for (auto k : Range(n)) {
      if ( (fnodes==nullptr) || (fnodes->Test(k)) ) {
        double df = 0;
        double sqa = 0.0;
        double sqb = 0.0;
        for (auto l : Range(k*bs, (k+1)*bs)) {
          df += sqr(fva(l)-fvb(l));
          sqa += sqr(fva(l));
          sqb += sqr(fvb(l));
        }
        df = sqrt(df);
        double avg = sqrt(sqrt(sqa) * sqrt(sqb));
        double rel = avg == 0.0 ? 0.0 : df / avg;
        if (df > 1e-14 && rel > 1e-12) {
          numdf++;
          cout << " DIFF " << k << " norm = " << df << ", diff = " << ", rel = " << rel << ": ";
          for (auto l : Range(k*bs, (k+1)*bs))
            { cout << "(" << fva(l) << "/" << fvb(l) << "/" << fva(l)-fvb(l) << ") "; }
          cout << endl;
        }
      }
    }
    if (numdf > 0)
      { cout << " done " << title << ", mismatches = " << numdf << endl; }
    if (fnodes != nullptr)
      { cout << " fnodes non-set = " << fnodes->Size() - fnodes->NumSet() << endl; }
  };


  auto prtv = [&](auto & vec, int nv, string title, double thresh = 0.0) {
    if (nv == 0)
      { return; }
    auto fv = vec.FVDouble();
    int vbs = fv.Size()/nv;
    cout << title << " = " << endl;
    cout << "  stat = " << vec.GetParallelStatus() << endl;
    cout << "  vals = " << endl;
    for (auto vnr : Range(nv)) {
      bool first = true;
      for (int l = 0; l < vbs; l++)
      {
        auto val = fv(vbs * vnr + l);
        if (fabs(val) > thresh) {
          if(first) {
            first = false;
            cout << "  " << vnr << " = ";
          }
          cout << "(" << l << ":" << val << ") ";
        }
      }
        // cout << fv(vbs * vnr + l) << " ";
      if (!first)
      cout << endl;
    }
    cout << endl;
  };


  auto set_kvec = [&](auto & vec, int kvnr, BaseAMGFactory::AMGLevel & alev, shared_ptr<BitArray> free_nodes) {
    typename ENERGY::TVD opt(0); /** just an arbitrary point ... **/
    // for (auto l : Range(3))
      // { opt.pos(0) = l * gcomm.Size() + gcomm.Rank(); }
    typename ENERGY::TM Q; SetIdentity(Q);
    Vec<BS, double> vcos, ovec;
    ovec = 0; ovec(kvnr) = 1;
    auto mesh = static_pointer_cast<TMESH>(alev.cap->mesh);
    auto vdata = get<0>(mesh->Data())->Data();
    vec.SetParallelStatus(CUMULATED);
    vec.FVDouble() = 0;
    cout << " set kvec : " << endl;
    if (mesh->template GetNN<NT_VERTEX>() > 0) {
      // opt = vdata[0]; /// DONT DO THAT !!! compares different kvecs then !!!
      int vbs = vec.FVDouble().Size()/mesh->template GetNN<NT_VERTEX>();
      auto fv = vec.FVDouble();
      for (auto vnr : Range(mesh->template GetNN<NT_VERTEX>())) {
        if ( (free_nodes == nullptr) || (free_nodes->Test(vnr)) ) {
          ENERGY::CalcQHh(opt, vdata[vnr], Q);
          vcos = Q * ovec;
          for (int l = 0; l < vbs; l++)
            { fv(vbs * vnr + l) = vcos(l); }
          // cout << vnr << " = ";
          // for (int l = 0; l < vbs; l++)
          //   cout << fv(vbs * vnr + l) << " ";
          // cout << endl;
        }
        else {
          for (int l = 0; l < vbs; l++)
            { fv(vbs * vnr + l) = 0.0; }
        }
      }
      // cout << endl;
    }
  };

  auto clcenrg = [&](auto & lev, auto & v, string title)
  {
    cout << " MAT = " << lev.cap->mat << endl;
    cout << " mat type " << typeid(*lev.cap->mat).name() << endl;
    // auto pds = lev.embed_map != nullptr ? lev.embed_map->GetParDofs() : lev.cap->pardofs;

    // auto pds = lev.embed_map != nullptr ? lev.embed_map->GetUDofs().GetParallelDofs() : lev.cap->uDofs.GetParallelDofs();
    UniversalDofs uDofs = lev.embed_map != nullptr ? lev.embed_map->GetUDofs() : lev.cap->uDofs;

    auto A = WrapParallelMatrix(lev.cap->mat, uDofs, uDofs, C2D);
    // auto A = make_shared<ParallelMatrix>(lev.cap->mat, pds, pds, C2D);
    // prtv(v, lev.cap->mesh->template GetNN<NT_VERTEX>(), "vec v");
    auto Av = A->CreateColVector();
    A->Mult(v, *Av);
    double enrg = sqrt(fabs(InnerProduct(*Av, v)));
    double rel = enrg/sqrt(InnerProduct(v, v));
    cout << title << ", energy = " << enrg << ", vv = " << InnerProduct(v,v) << ", relative = " << rel << endl;

    if (rel > 1e-4)
    {
      prtv(*Av, lev.cap->mesh->template GetNN<NT_VERTEX>(), "vec Av", 1e-4 * enrg);
    }
  };

  for (int kvnr = 0; kvnr < BS; kvnr++)
  {
    int nlevsglob = gcomm.AllReduce(amg_levels.Size(), NG_MPI_MAX);
    int nlevsloc = map->GetNLevels();
    unique_ptr<BaseVector> cvec = std::move(map->CreateVector(nlevsloc-1));

    if ( (nlevsloc == nlevsglob) && (cvec != nullptr) )
    {
      cout << " set_kvec " << nlevsloc-1 << endl;
      set_kvec(*cvec, kvnr, *amg_levels[nlevsloc-1], nullptr);

      cout << " clcenrg " << nlevsloc-1 << endl;
      clcenrg(*amg_levels[nlevsloc-1], *cvec,
        string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(nlevsloc-1));
    }

    for (int lev = nlevsloc-2; lev >= 0; lev--)
    {
      bool havemb = (lev == 0) && (amg_levels[0]->embed_map != nullptr);
      unique_ptr<BaseVector> fvec1 = map->CreateVector(lev), fvec2 = havemb ? amg_levels[0]->embed_map->CreateMappedVector() : map->CreateVector(lev);
      map->TransferC2F(lev, fvec1.get(), cvec.get());
      set_kvec(*fvec2, kvnr, *amg_levels[lev], amg_levels[lev]->cap->free_nodes);
      if ( havemb ) {
        unique_ptr<BaseVector> fvec3 = amg_levels[0]->embed_map->CreateVector();
        amg_levels[0]->embed_map->Finalize(); // this can be  concatenated before crs grid projection!
        amg_levels[0]->embed_map->TransferC2F(fvec3.get(), fvec2.get());
        fvec2 = std::move(fvec3);
      }
      chkab(fvec1->FVDouble(), fvec2->FVDouble(), amg_levels[lev]->cap->mesh->template GetNN<NT_VERTEX>(), nullptr,
            string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(lev));
      clcenrg(*amg_levels[lev], *fvec1,
        string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(lev));
      cvec = std::move(fvec2);
    }

  }

}

using T_MESH           = ElasticityAMGFactory<3>::TMESH;
using T_ENERGY         = ElasticityAMGFactory<3>::ENERGY;
using T_MESH_WITH_DATA = typename T_MESH::T_MESH_W_DATA;

extern template class SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;
extern template class MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;

template class DiscreteAgglomerateCoarseMap<T_MESH, SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;
template class DiscreteAgglomerateCoarseMap<T_MESH, MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;

template class ElasticityAMGFactory<3>;
template class VertexAMGPC<ElasticityAMGFactory<3>>;
template class ElmatVAMG<ElasticityAMGFactory<3>, double, double>;

template class PlateTestAgglomerator<T_MESH_WITH_DATA>;
template class DiscreteAgglomerateCoarseMap<T_MESH, PlateTestAgglomerator<T_MESH_WITH_DATA>>;

using PCCBASE = VertexAMGPC<ElasticityAMGFactory<3>>;
using PCC = ElmatVAMG<ElasticityAMGFactory<3>, double, double>;
// using PCC = PCCBASE;

// RegisterPreconditioner<PCC> register_elast_3d ("ATAmg.elast_3d");
RegisterElasticityAMGSolver<PCC> register_elast_3d ("NgsAMG.elast_3d");

} // namespace amg


#endif // ELASTICITY
