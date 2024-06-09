#include <ostream>
#ifdef ELASTICITY

#define FILE_AMG_ELAST_CPP
#define FILE_AMG_ELAST_2D_CPP

#include "elasticity.hpp"
#include <amg_pc_vertex.hpp>

#include "elasticity_impl.hpp"
#include <amg_pc_vertex_impl.hpp>
#include "elasticity_pc_impl.hpp"

#include "plate_test_agg_impl.hpp"

namespace amg
{

// template<> template<>
// shared_ptr<stripped_spm_tm<Mat<1, 3, double>>>
// VertexAMGPC<ElasticityAMGFactory<2>> :: BuildEDC<2> (size_t height, shared_ptr<TopologicMesh> mesh)
// {
//   const auto & O(static_cast<Options&>(*options));

//   const auto & M(*mesh);

//   constexpr int BS  = ElasticityAMGFactory<3>::BS;
//   constexpr int BSA = 1;

//   typedef stripped_spm_tm<Mat<1, 3, double>> T_E_D_C;

//   Array<int> perow(M.template GetNN<NT_VERTEX>()); perow = 1;
//   auto E_D_C = make_shared<T_E_D_C>(perow, M.template GetNN<NT_VERTEX>());
//   for (auto k : Range(perow))
//   {
//     E_D_C->GetRowIndices(k)[0] = k;
//     auto & v = E_D_C->GetRowValues(k)[0];
//     v = 0;
//     v(0, 2) = 1.0;
//   }
//   return E_D_C;
// }


template<> shared_ptr<ElasticityMesh<2>>
ElmatVAMG<ElasticityAMGFactory<2>, double, double> :: BuildAlgMesh_ELMAT (shared_ptr<BlockTM> top_mesh)
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

  double eps = 1e-12 * evalsA(N-1);

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

  // sqrt(Ainv) * B * sqrt(Binv)
  FlatMatrix<double> AiBAi(N, N, lh); AiBAi = sqAi * B * sqAi;

  FlatMatrix<double> evecsABA(N, N, lh);
  FlatVector<double> evalsABA(N, lh);
  TimedLapackEigenValuesSymmetric(AiBAi, evalsABA, evecsABA);

  double min_nz = evalsABA(count_zero);

  double kappa = (min_nz == 0.0) ? std::numeric_limits<double>::infinity() : evalsABA(N-1) / min_nz;

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

    cout << " smallest eval " << min_nz << ", evec = " << endl << genEvecs.Row(count_zero) << endl;

    vec = genEvecs.Row(count_zero);

    Avec = A * vec;
    Bvec = B * vec;

    cout << " A * vec = " << endl << Avec << endl;
    Avec *= min_nz;
    cout << " lam * A * vec = " << endl << Avec << endl;
    cout << " B * vec = " << endl << Bvec << endl;


    cout << " largest  eval " << evalsABA(N-1) << ", evec = " << endl << genEvecs.Row(N-1) << endl;

    vec = genEvecs.Row(N-1);
    Avec = A * vec;
    Bvec = B * vec;
    for (auto k : Range(6))
    {
      ip = InnerProduct(Avec, genEvecs.Row(k));
      Avec -= ip * genEvecs.Row(k);

      ip = InnerProduct(Avec, genEvecs.Row(k));
      Avec -= ip * genEvecs.Row(k);
    }

    cout << " A * vec = " << endl << Avec << endl;
    Avec *= evalsABA(N-1);
    cout << " lam * A * vec = " << endl << Avec << endl;
    cout << " B * vec = " << endl << Bvec << endl;

  }

  return std::make_tuple(min_nz, evalsABA(N-1), kappa);
}


template<>
void
ElmatVAMG<ElasticityAMGFactory<2>, double, double> :: 
CalcAuxWeightsLSQ (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
        ElementId ei, LocalHeap & lh)
{
  throw Exception("CalcAuxWeightsLSQ in 2D!");
}


template<>
void
ElmatVAMG<ElasticityAMGFactory<2>, double, double> :: 
CalcAuxWeightsALG (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
        ElementId ei, LocalHeap & lh)
{
  std::cout << " LS-fit aux-elmat for ei " << ei << ", dnums = " << dnums << std::endl;

  constexpr int B = 2;

  int const n = dnums.Size();
  int const N = n * B;

  int const nE = 3;

  /** position & tangents **/
  FlatArray<Vec<2, double>> pos(n, lh);
  FlatArray<Vec<2, double>> tang(nE, lh);

  Vec<2, double> tmpVec; // GetNodePos needs this for edge-pos
  for (auto k : Range(n))
  {
    GetNodePos<2>(NodeId(NT_VERTEX, dnums[k]), *ma, pos[k], tmpVec);
  }

  for (auto k : Range(n))
  {
    tang[k] = pos[ (k + 1) % n ] - pos[ k ];
    auto const len = L2Norm(tang[k]);
    tang[k] /= len;
  }

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

  for (auto eNr : Range(nE))
  {
    int const vi = eNr;
    int const vj = (eNr + 1) % n;

    Vec<2, double> &ti = tang[eNr];

    // rhs(eNr) = tOf(ti, vi, vi) - tOf(ti, vi, vj) - tOf(ti, vj, vi) + tOf(ti, vj, vj);
    rhs(eNr) = tOf(ti, vi, vi) - 2 * tOf(ti, vi, vj) + tOf(ti, vj, vj);

    theta(eNr, eNr) = 4.0;

    for (auto eJ : Range(eNr + 1, nE))
    {
      Vec<2, double> &tj = tang[eJ];

      double const ip = InnerProduct(ti, tj);

      theta(eNr, eJ) = ip * ip;
      theta(eJ, eNr) = ip * ip;

      // double const fac = (eJ == eNr) ? 1.0 : -1.0;
      // theta(eNr, eJ) = fac * ip * ip;
    }
  }

  cout << " THETA " << endl << theta << endl;

  CalcInverse(theta);
  sol = theta * rhs;

  cout << " THETA INV " << endl << theta << endl;
  cout << " rhs " << endl << rhs << endl;
  cout << " sol " << endl << sol << endl;


  /** Assemble aux-elmat and test approximation quality! */

  FlatMatrix<double> auxElmat(N, N, lh);

  auxElmat = 0.0;

  for (auto eNr : Range(nE))
  {
    int const vi = eNr;
    int const vj = (eNr + 1) % n;

    Vec<2, double> &t = tang[eNr];

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

  cout << " ELMAT " << ei << endl << elmat << endl;
  cout << " LS-AUX ELMAT " << ei << endl << auxElmat << endl;

  {
    // bool print = O.log_level_pc == Options::LOG_LEVEL_PC::DBG;
    bool const print = true;
    if ( print )
    {
      std::cout << " Test LS elmat " << ei << std::endl;
    }
    auto [minev, maxev, kappa] = DenseEquivTestAAAA(elmat, auxElmat, lh, print);
    // this->elmat_evs[0] = min(minev, this->elmat_evs[0]);
    // this->elmat_evs[1] = max(maxev, this->elmat_evs[1]);
    // this->elmat_evs[2] = max(kappa, this->elmat_evs[2]);
  }

  auxElmat -= elmat;
  cout << " AUX-ELMAT - ELMAT: " << endl << auxElmat << endl;
  cout << " REL L2-diff " << L2Norm(auxElmat)/L2Norm(elmat) << endl;


}

template<>
void
ElmatVAMG<ElasticityAMGFactory<2>, double, double> :: 
CalcAuxWeightsSC (FlatArray<int>            dnums,
                  FlatMatrix<double> const &elmat,
                  ElementId                 ei,
                  LocalHeap                &lh)
{
  cout << " CalcAuxWeightsSC, dnums = "; prow2(dnums); cout << endl;
  int const n  = dnums.Size();
  int const nb = 2;
  int const ni = n - nb;

  constexpr int B = 2;
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

      cout << " i, j = " << i << " " << j << endl;
      cout << " iRows = "; prow2(iRows); cout << endl;
      cout << " bRows = "; prow2(bRows); cout << endl;


      Aii = elmat.Rows(iRows).Cols(iRows);
      CalcPseudoInverseNew(Aii, lh);
      A_bi_A_ii = elmat.Rows(bRows).Cols(iRows) * Aii;
      S = elmat.Rows(bRows).Cols(bRows);
      S -= A_bi_A_ii * elmat.Rows(iRows).Cols(bRows);

      cout << " S = " << endl << S << endl;

      auxElmat.Rows(bRows).Cols(bRows) += S;
    }
  }

  cout << " EMAT " << ei << endl << elmat << endl;
  cout << " AUX-EMAT " << ei << endl << auxElmat << endl;

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
  cout << " AUX-ELMAT - ELMAT: " << endl << auxElmat << endl;
  cout << " REL L2-diff " << L2Norm(auxElmat)/L2Norm(elmat) << endl;

  CalcAuxWeightsALG(dnums, elmat, ei, lh);
  // cout << " auxElmat: " << endl << auxElmat << endl;

  // CalcPseudoInverseNew(auxElmat, lh);

  // cout << " SC to first 2: " << endl << S << endl;

  // FlatMatrix<double> Abb(6, 6, lh);
  // (*ht_egge)[IVec<2, int>(dnums[j], dnums[i]).Sort()] += weight;

} // ElmatVAMG::AddElementMatrix



  template<>
  void ElasticityAMGFactory<2> :: CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
  {
  }

  using T_MESH           = ElasticityAMGFactory<2>::TMESH;
  using T_ENERGY         = ElasticityAMGFactory<2>::ENERGY;
  using T_MESH_WITH_DATA = typename T_MESH::T_MESH_W_DATA;

  extern template class SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;
  extern template class MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>;

  template class DiscreteAgglomerateCoarseMap<T_MESH, SPWAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;
  template class DiscreteAgglomerateCoarseMap<T_MESH, MISAgglomerator<T_ENERGY, T_MESH_WITH_DATA, T_ENERGY::NEED_ROBUST>>;

  template class ElasticityAMGFactory<2>;
  template class VertexAMGPC<ElasticityAMGFactory<2>>;
  template class ElmatVAMG<ElasticityAMGFactory<2>, double, double>;

  template class PlateTestAgglomerator<T_MESH_WITH_DATA>;
  template class DiscreteAgglomerateCoarseMap<T_MESH, PlateTestAgglomerator<T_MESH_WITH_DATA>>;

  using PCCBASE = VertexAMGPC<ElasticityAMGFactory<2>>;
  using PCC = ElmatVAMG<ElasticityAMGFactory<2>, double, double>;
  // using PCC = PCCBASE;

  // RegisterPreconditioner<PCC> register_elast_2d ("ATAmg.elast_2d");
  RegisterElasticityAMGSolver<PCC> register_elast_2d ("NgsAMG.elast_2d");
  // RegisterAMGSolver<PCC> register_elast_2d
} // namespace amg


#endif // ELASTICITY
