#include "dyn_block.hpp"
#include "utils_sparseLA.hpp"
#include "utils.hpp"
#include "utils_io.hpp"
#include "utils_arrays_tables.hpp"
#include "utils_sparseMM.hpp"

#include <elementbyelement.hpp>
#include <memory>
#include <ngs_stdcpp_include.hpp>
#include <sparsematrix.hpp>
#include <universal_dofs.hpp>
namespace amg
{

// TODO: to header!

PARALLEL_OP GetOpType(BaseMatrix const &A)
{
  if (auto pmat = dynamic_cast<ParallelMatrix const *>(&A))
  {
    return pmat->GetOpType();
  }
  else
  {
    return C2C;
  }
}

template<int BS>
void
AMGBFCheck(SparseMat<BS, BS> const &A,
           SparseMat<BS, BS> const &M,
           double            const &thresh = 1e-10)
{
  auto const N = A.Height();

  double badRE = 0.0;
  int    badK  = -1;
  int    badC  = -1;

  double avgRE = 0.0;
  int    cntRE = 0;

  Vec<BS, double> badREPC = 0;
  IVec<BS, int>    badKPC  = -1;

  Vec<BS, double> avgREPC = 0;
  IVec<BS, int>    cntREPC = 0;

  for (auto k : Range(N))
  {
    auto const &Akk = A(k,k);
    auto const &Mkk = M(k,k);

    for (auto l : Range(BS))
    {
      double Akkl;
      double Mkkl;

      if constexpr(BS > 1)
      {
        Akkl = sqrt(Akk(l,l));
        Mkkl = sqrt(Mkk(l,l));
      }
      else
      {
        Akkl = sqrt(Akk);
        Mkkl = sqrt(Mkk);
      }

      if (Mkkl > thresh)
      {
        double rel = Akkl / Mkkl;

        avgRE += rel;
        cntRE += 1;

        avgREPC(l) += rel;
        cntREPC[l] += 1;

        if ( rel > badREPC(l) )
        {
          badREPC(l) = rel;
          badKPC[l] = k;

          // cout << " component " << l << " - new worst RE = " << rel << " for DOF (" << k << "." << l << ")" << endl;
          // cout << "               L2-norm = " << Mkkl << ", A-norm = " << Akkl << endl;
        }

        if ( rel > badRE )
        {
          badRE = rel;
          badK = k;
          badC = l;
          // cout << " new worst RE = " << rel << " for DOF (" << k << "." << l << ")" << endl;
          // cout << "               L2-norm = " << Mkkl << ", A-norm = " << Akkl << endl;
        }
      }
    }
  }

  avgRE = cntRE == 0 ? -1 : avgRE / cntRE;

  for (auto l : Range(BS))
  {
    avgREPC(l) = cntREPC[l] == 0 ? -1 : avgREPC(l) / cntREPC[l];
  }

  cout << "  OVERALL avg RE = " << avgRE << ", taken from " << cntRE << "/" << A.Height() * BS << " DOFs." << endl;
  cout << "  OVERALL worst RE = " << badRE << " for DOF (" << badK << "." << badC << ")" << endl;

  cout << "  PER COMPONENT: " << endl;

  for (auto l : Range(BS))
  {
    cout << "   COMPONENT " << l << ": " << endl;
    cout << "     avg RE = " << avgREPC(l) << ", taken from " << cntREPC[l] << "/" << A.Height() << " DOFs." << endl;
    cout << "     worst RE = " << badREPC(l) << " for DOF (" << badKPC[l] << "." << l << ")" << endl;
  }
}

void
AMGBFCheck(shared_ptr<BaseMatrix> const &A,
           shared_ptr<BaseMatrix> const &M,
           double     const &thresh = 1e-10)
{
  auto baseSPA = dynamic_pointer_cast<BaseSparseMatrix>(A);
  auto baseSPB = dynamic_pointer_cast<BaseSparseMatrix>(M);

  if (baseSPA == nullptr || baseSPB == nullptr)
  {
    cout << " AMGBFCheck needs SP-mats!" << endl;
    return;
  }

  DispatchSquareMatrix(*baseSPA, [&](auto const &spA, auto BS)
  {
    auto spBP = dynamic_pointer_cast<SparseMat<BS, BS>>(baseSPB);

    if (spBP == nullptr)
    {
      cout << " AMGBFCheck, block-sizes do not match!" << endl;
      return;
    }

    AMGBFCheck<BS>(spA, *spBP, thresh);
  });
}

template <class TSPM>
INLINE
shared_ptr<TSPM>
AddMergeSPM (double const &alpha,
             TSPM   const &A,
             double const &beta,
             TSPM   const &B)
{
  Array<int> cols(100);

  auto const H = A.Height();
  // LocalHeap lh(12582912, "GezwicktesMurmeltier"); // 12 MB

  auto itRCV = [&](auto lam)
  {
    for (auto k : Range(H))
    {
      auto risA = A.GetRowIndices(k);
      auto rvsA = A.GetRowValues(k);
      auto risB = B.GetRowIndices(k);
      auto rvsB = B.GetRowValues(k);

      if (risB.Size() == 0)
      {
        for (auto j : Range(risA))
        {
          lam(k, risA[j], rvsA[j]);
        }
      }
      else if (risA.Size() == 0)
      {
        for (auto j : Range(risB))
        {
          lam(k, risB[j], rvsB[j]);
        }
      }
      else
      {
        iterate_ABC(risA, risB,
          [&](auto const &idxA) { lam(k, risA[idxA], rvsA[idxA]); },
          [&](auto const &idxB) { lam(k, risB[idxB], rvsA[idxB]); },
          [&](auto const &idxA, auto const &idxB) { lam(k, risA[idxA], rvsA[idxA] + rvsA[idxB]); });
      }
    }
  };

  Array<int> perow(A.Height());

  perow = 0;
  itRCV([&](auto row, auto col, auto const &val) { perow[row]++; });

  auto sum = make_shared<TSPM>(perow, A.Width());

  perow = 0;
  itRCV([&](auto row, auto col, auto const &val) {
    auto const idxInRow = perow[row]++;
    sum->GetRowIndices(row)[idxInRow] = col;
    sum->GetRowValues(row)[idxInRow] = val;
  });
  return sum;
} // AddMergeSPM

template <class TSPM>
INLINE
shared_ptr<TSPM>
AddSPM (double const &alpha,
        TSPM   const &A,
        double const &beta,
        TSPM   const &B)
{
  Array<int> perow(A.Height());
  for (auto k : Range(perow))
  {
    perow[k] = A.GetRowIndices(k).Size();
  }

  auto sum = make_shared<TSPM>(perow, A.Width());

  for (auto k : Range(perow))
  {
    sum->GetRowIndices(k) = A.GetRowIndices(k);

    auto valsSum = sum->GetRowValues(k);
    auto valsA   = A.GetRowValues(k);
    auto valsB   = B.GetRowValues(k);

    for (auto j : Range(valsSum))
    {
      valsSum[j] = alpha * valsA[j] + beta * valsB[j];
    }
  }

  return sum;
} // AddSPM

INLINE
std::tuple<double, shared_ptr<BaseMatrix>>
UnwrapScaleExpression(shared_ptr<BaseMatrix> A)
{
  if (auto scaledA = dynamic_pointer_cast<VScaleMatrix<double>>(A))
  {
    // "GetScalingFactor" not available in upstream NGSolve 
    std::string opInfoName = scaledA->GetOperatorInfo().name;
    auto pos = opInfoName.find('=');
    double scale = std::stod(opInfoName.substr(pos, opInfoName.size() - pos));
    return std::make_tuple(scale, scaledA->SPtrMat());

    // throw Exception("UnwrapScaleExpression, GetScalingFactor not available in NGSolve!");

    // return std::make_tuple(scaledA->GetScalingFactor(), scaledA->SPtrMat());
  }
  else
  {
    return std::make_tuple(1.0, A);
  }
}

INLINE
shared_ptr<BaseMatrix>
AddABGeneric (double                 const &inAlpha,
              shared_ptr<BaseMatrix> const &inA,
              double                 const &inBeta,
              shared_ptr<BaseMatrix> const &inB)
{
  auto [extraAlpha, A] = UnwrapScaleExpression(inA);
  auto [extraBeta,  B] = UnwrapScaleExpression(inB);


  double const alpha = inAlpha * extraAlpha;
  double const beta  = inBeta  * extraBeta;

  auto [rowUDA, colUDA, locASB, opA] = UnwrapParallelMatrix(A);

  auto locA = locASB; // no capture of structured binding

  if ( (A->Height() != B->Height()) || (A->Width() != B->Width()) )
  {
    std::stringstream stream;
    stream << "AddABGeneric, cannot add ("
           << A->Height() << "x" << A->Width() << ") and ("
           << B->Height() << "x" << B->Width() << ")!";
    throw Exception(stream.str());
  }

  if (opA != GetOpType(*B))
  {
    std::stringstream stream;
    stream << "AddABGeneric, cannot add " << opA << " and " << GetOpType(*B) << "!";
    throw Exception(stream.str());
  }

  int const WA = rowUDA.GetBS();
  int const HA = colUDA.GetBS();

  shared_ptr<BaseMatrix> sum;

  DispatchRectangularMatrix(*B, [&](auto const &locB, auto HB, auto WB) {
    if ( (HB != HA) || (WB != WA) )
    {
      std::stringstream stream;
      stream << "AddABGeneric, cannot add dims (" << HA << "x" << WA << ") and (" << HB << "x" << WB << ")!";
      throw Exception(stream.str());
    }

    auto const& locASPM = static_cast<stripped_spm<Mat<HB, WB, double>>const&>(*locA);

    sum = AddMergeSPM(alpha, locASPM, beta, locB);
    // sum = AddSPM(alpha, locASPM, beta, locB);
  });

  return WrapParallelMatrix(sum, rowUDA, colUDA, opA);
} // AddABGeneric

size_t GetScalNZE (BaseMatrix const *bMat)
{
  auto [rowUD, colUD, locMat, opType] = UnwrapParallelMatrix(bMat);

  size_t nze_scal = 0;

  if (auto spm = dynamic_cast<BaseSparseMatrix const *>(locMat.get()))
  {
    nze_scal = spm->NZE() * GetEntrySize(spm);
  }
  else if (auto ptr = dynamic_cast<DynBlockSparseMatrix<double> const *>(locMat.get()))
  {
    nze_scal = ptr->GetNZE();
  }

  size_t nzeGlob  = rowUD.GetCommunicator().AllReduce(nze_scal, NG_MPI_SUM);

  return rowUD.GetCommunicator().Rank() == 0 ? nzeGlob : nze_scal;
} // GetScalNZE


shared_ptr<BaseMatrix> TransposeAGeneric (shared_ptr<BaseMatrix> A)
{
  auto [rowUDA, colUDA, locA, opA] = UnwrapParallelMatrix(A);

  shared_ptr<BaseSparseMatrix> spA = my_dynamic_pointer_cast<BaseSparseMatrix>(locA, "TransposeAGeneric - Sparse-A");

  PARALLEL_OP opAT = ParallelOp(ColType(opA), RowType(opA));

  shared_ptr<BaseMatrix> AT = nullptr;

  Switch<AMG_MAX_SYS_DIM> (GetEntryHeight(spA.get())-1,
    [&] (auto H_A_M) {
      Switch<AMG_MAX_SYS_DIM> (GetEntryWidth(spA.get())-1,
      [&] (auto W_A_M) {
        constexpr int H_A = H_A_M + 1;
        constexpr int W_A = W_A_M + 1;
        if constexpr (isSparseMatrixCompiled<H_A, W_A>()) {
          typedef typename strip_mat<Mat<H_A, W_A, double>>::type TMA;
          typedef typename strip_mat<Mat<W_A, H_A, double>>::type TMAT;
          auto spA_tm = dynamic_pointer_cast<SparseMatrixTM<TMA>>(spA);
          auto AT_tm = TransposeSPM(*spA_tm);
          AT = make_shared<SparseMatrix<TMAT>>(std::move(*AT_tm));
        }
      });
    });

  return WrapParallelMatrix(AT, colUDA, rowUDA, opAT);
} // TransposeAGeneric


shared_ptr<BaseMatrix> TransposeSPMGeneric (shared_ptr<BaseMatrix> A)
{
  auto parA = dynamic_pointer_cast<ParallelMatrix>(A);
  shared_ptr<BaseSparseMatrix> spA = (parA == nullptr) ? dynamic_pointer_cast<BaseSparseMatrix>(A) :
    dynamic_pointer_cast<BaseSparseMatrix>(parA->GetMatrix());
  shared_ptr<BaseMatrix> AT = nullptr;
  Switch<AMG_MAX_SYS_DIM> (GetEntryHeight(spA.get())-1,
    [&] (auto H_A_M) {
      Switch<AMG_MAX_SYS_DIM> (GetEntryWidth(spA.get())-1,
      [&] (auto W_A_M) {
        constexpr int H_A = H_A_M + 1;
        constexpr int W_A = W_A_M + 1;
        if constexpr (isSparseMatrixCompiled<H_A, W_A>()) {
          typedef typename strip_mat<Mat<H_A, W_A, double>>::type TMA;
          typedef typename strip_mat<Mat<W_A, H_A, double>>::type TMAT;
          auto spA_tm = dynamic_pointer_cast<SparseMatrixTM<TMA>>(spA);
          auto AT_tm = TransposeSPM(*spA_tm);
          AT = make_shared<SparseMatrix<TMAT>>(std::move(*AT_tm));
        }
      });
    });
  return AT;
}

// mult mats. A~NxN/C->D, B~NxM/C->C
shared_ptr<BaseMatrix> MatMultABGeneric (shared_ptr<BaseMatrix> A, shared_ptr<BaseMatrix> B)
{
  return MatMultABGeneric(*A, *B);
}

shared_ptr<BaseMatrix> MatMultABGeneric (BaseMatrix const &A, BaseMatrix const &B)
{
  auto [rowUDA, colUDA, locA, opA] = UnwrapParallelMatrix(A);
  auto [rowUDB, colUDB, locB, opB] = UnwrapParallelMatrix(B);

  if (ColType(opB) != RowType(opA))
  {
    std::stringstream stream;
    stream << "MatMultABGeneric - cannot mult " << opA << " x " << opB;
    throw Exception(stream.str());
  }

  PARALLEL_OP opAB = ParallelOp(RowType(opB), ColType(opA));

  shared_ptr<BaseMatrix> AB = nullptr;


  if (auto dynSPA = dynamic_pointer_cast<DynBlockSparseMatrix<double>>(locA))
  {
    auto dynSPB = my_dynamic_pointer_cast<DynBlockSparseMatrix<double>>(locB, "MatMultABGeneric - DynSPM&Non-DynSPM");

    AB = MatMultAB(*dynSPA, *dynSPB);
  }
  else
  {
    shared_ptr<BaseSparseMatrix> spA = my_dynamic_pointer_cast<BaseSparseMatrix>(locA, "MatMultABGeneric - Sparse-A");
    shared_ptr<BaseSparseMatrix> spB = my_dynamic_pointer_cast<BaseSparseMatrix>(locB, "MatMultABGeneric - Sparse-B");

    int const EHA = GetEntryHeight(spA.get());
    int const EWA = GetEntryWidth(spA.get());
    int const EHB = GetEntryHeight(spB.get());
    int const EWB = GetEntryWidth(spB.get());

    if ( EWA != EHB )
    {
      std::stringstream stream;
      stream << "MatMultABGeneric - cannot multiply (" << EHA << "x" << EWA << ") x (" << EHB << "x" << EWB << ")!";
      throw Exception(stream.str());
    }

    Switch<AMG_MAX_SYS_DIM> (EHA-1, [&] (auto H_A_M) {
      Switch<AMG_MAX_SYS_DIM> (EWA-1, [&] (auto W_A_M) {
        Switch<AMG_MAX_SYS_DIM> (EWB-1, [&] (auto W_B_M)
        {
          constexpr int H_A = H_A_M + 1;
          constexpr int W_A = W_A_M + 1;
          constexpr int W_B = W_B_M + 1;
          if constexpr ( IsSparseMMCompiled<H_A, W_A, W_B>() ) {
            typedef typename strip_mat<Mat<H_A, W_A, double>>::type TMA;
            typedef typename strip_mat<Mat<W_A, W_B, double>>::type TMB;
            typedef stripped_spm_tm<Mat<H_A, W_B, double>> TSPM_TM_C;
            typedef stripped_spm<Mat<H_A, W_B, double>> TSPM_C;
            auto spA_tm = dynamic_pointer_cast<SparseMatrixTM<TMA>>(spA);
            auto spB_tm = dynamic_pointer_cast<SparseMatrixTM<TMB>>(spB);
            // cout << " -> CALL MatMultAB, " << spA_tm << " " << spB_tm << " !" << endl;
            shared_ptr<TSPM_TM_C> C_tm = MatMultAB(*spA_tm, *spB_tm);
            AB = make_shared<TSPM_C>(std::move(*C_tm));
          }
          else {
            std::stringstream stream;
            stream << "MatMultABGeneric - sparse matrix multiplication "
                    << "(" << EHA << "x" << EWA << ") x (" << EHB << "x" << EWB << ") !"
                    << " NOT COMPILED! ";
            throw Exception(stream.str());
          }
        });
      });
    });
  }

  return WrapParallelMatrix(AB, rowUDB, colUDA, opAB);
} // MatMultABGeneric


std::tuple<shared_ptr<BaseMatrix>, // AP
           shared_ptr<BaseMatrix>> // PT A P
RestrictMatrixKeepFactor(BaseMatrix const &A,
                         BaseMatrix const &P,
                         BaseMatrix const &PT)
{
  auto sparseA  = dynamic_cast<BaseSparseMatrix const *>(&A);
  auto sparseP  = dynamic_cast<BaseSparseMatrix const *>(&P);
  auto sparsePT = dynamic_cast<BaseSparseMatrix const *>(&PT);

  shared_ptr<BaseMatrix> AP   = nullptr;
  shared_ptr<BaseMatrix> PTAP = nullptr;

  if ( ( sparseA  != nullptr ) && 
       ( sparseP  != nullptr ) && 
       ( sparsePT != nullptr ) )
  {
    AP = MatMultABGeneric(*sparseA, *sparseP);
    PTAP = MatMultABGeneric(*sparsePT, *AP);
  }

  return std::make_tuple(AP, PTAP);
} // RestrictMatrixKeepFactor


shared_ptr<BaseMatrix> BaseMatrixToSparse (shared_ptr<BaseMatrix> A);

template<int BH, int BW>
INLINE
shared_ptr<stripped_spm_tm<Mat<BH, BW, double>>>
Block2SparseMatrix_impl (BlockMatrix &A)
{
  typedef stripped_spm_tm<Mat<BH, BW, double>> TSPM;

  // cout << "Block2SparseMatrix_impl " << BH << " x " << BW << endl;

  int const H = A.BlockRows();
  int const W = A.BlockCols();

  int totalH = 0;
  int totalW = 0;

  Array<int> rowOff(H + 1);
  Array<int> colOff(W + 1);

  auto getAkj = [&](auto k, auto j) -> TSPM const* {
    if (auto p = A(k, j))
    {
      return my_dynamic_cast<TSPM const>(p.get(), "Block2SparseMatrix_impl - block-cast");
    }
    return nullptr;
  };

  auto getBlockInRow = [&](auto k) -> TSPM const* {
    for (auto j : Range(W))
    {
      if (auto p = A(k, j))
      {
        return getAkj(k, j);
      }
    }
    return nullptr;
  };

  auto getBlockInCol = [&](auto j) -> TSPM const* {
    for (auto k : Range(H))
    {
      if (auto p = A(k, j))
      {
        return getAkj(k, j);
      }
    }
    return nullptr;
  };

  for (auto k : Range(H))
  {
    rowOff[k] = totalH;
    totalH += getBlockInRow(k)->Height();
  }

  for (auto j : Range(W))
  {
    rowOff[j] = totalH;
    totalH += getBlockInCol(j)->Width();
  }

  auto iterateBlockWise = [&](auto lam)
  {
    for (auto k : Range(H))
    {
      int const offRow = rowOff[k];
      for (auto j : Range(W))
      {
        int const offCol = colOff[j];
        if (auto pAkj = getAkj(k, j))
        {
          auto const &Akj = *pAkj;
          lam(k, offRow, j, offCol, Akj);
        }
      }
    }
  };

  Array<int> perow(totalH);
  perow = 0;

  iterateBlockWise([&](auto k, auto offRow, auto j, auto offCol, auto const &Akj)
  {
    for (auto kk : Range(Akj.Height()))
    {
      perow[offRow + kk] += Akj.GetRowIndices(kk).Size();
    }
  });

  auto spA = make_shared<stripped_spm_tm<Mat<BH, BW, double>>>(perow, totalW);

  perow = 0;

  iterateBlockWise([&](auto k, auto offRow, auto j, auto offCol, auto const &Akj)
  {
    for (auto kk : Range(Akj.Height()))
    {
      auto Akj_ris = Akj.GetRowIndices(kk);
      auto Akj_rvs = Akj.GetRowValues(kk);

      auto ris = spA->GetRowIndices(kk);
      auto rvs = spA->GetRowValues(kk);

      for (auto l : Range(ris))
      {
        int const row = offRow + kk;
        int const inRow = perow[row]++;
        ris[inRow] = offCol + Akj_ris[j];
        rvs[inRow] = Akj_rvs[j];
      }
    }
  });

  return spA;
}

shared_ptr<ParallelDofs>
fuseParallelDOFs(FlatArray<shared_ptr<ParallelDofs>> parDofs)
{
  int N = parDofs.Size();

  if (N == 0)
  {
    return nullptr;
  }

  Array<int> offsets(N);
  int nD = 0;
  for (auto k : Range(N))
  {
    offsets[k] = nD;
    nD += parDofs[k]->GetNDofLocal();
  }

  TableCreator<int> createDPs(nD);
  for (;!createDPs.Done(); createDPs++)
  {
    for (auto k : Range(N))
    {
      auto const  off = offsets[k];
      auto const &pdK = *parDofs[k];

      for (auto j : Range(pdK.GetNDofLocal()))
      {
        auto distProcs = pdK.GetDistantProcs(j);
        if (distProcs.Size())
        {
          createDPs.Add(off + j, distProcs);
        }
      }
    }
  }

  return make_shared<ParallelDofs>(parDofs[0]->GetCommunicator(), createDPs.MoveTable(), parDofs[0]->GetEntrySize());
}


shared_ptr<BaseMatrix>
Block2SparseMatrix (BlockMatrix &A)
{
  int const H = A.BlockRows();
  int const W = A.BlockCols();

  if (H * W == 0)
  {
    return nullptr;
  }

  int BH = -1;
  int BW = -1;

  PARALLEL_OP blockOp = C2C;
  int blockPar = -1;

  // convert all blocks to sparse matrices, make sure the block-dims are okay
  Array<Array<shared_ptr<BaseMatrix>>> parConvBlocks(H);

  Array<shared_ptr<ParallelDofs>> rowPds(W);
  Array<shared_ptr<ParallelDofs>> colPds(H);

  Array<Array<shared_ptr<BaseMatrix>>> convBlocks(H);

  for (auto k : Range(H))
  {
    parConvBlocks[k].SetSize(W);
    convBlocks[k].SetSize(W);
    for (auto j : Range(W))
    {
      parConvBlocks[k][j] = nullptr;
      convBlocks[k][j] = nullptr;
      if (auto p = A(k, j))
      {
        auto sparseKJ = BaseMatrixToSparse(p);

        auto parAkj = dynamic_pointer_cast<ParallelMatrix>(sparseKJ);

        bool const kjPar = ( parAkj != nullptr );

        if (kjPar)
        {
          rowPds[k] = parAkj->GetRowParallelDofs();
          colPds[j] = parAkj->GetColParallelDofs();
        }

        if (blockPar == -1)
        {
          blockPar = kjPar ? 1 : 0;
          blockOp = kjPar ? parAkj->GetOpType() : C2C;
        }
        else if (blockPar != kjPar)
        {
          throw Exception("Block2SparseMatrix - mixed local and parallel matrices!");
          return nullptr;
        }
        else if (kjPar && ( parAkj->GetOpType() != blockOp ) )
        {
          throw Exception("Block2SparseMatrix - mixed op-types!");
          return nullptr;
        }

        parConvBlocks[k][j] = sparseKJ;
        convBlocks[k][j]    = GetLocalMat(sparseKJ);
        auto [X, Y] = GetEntryDims(*convBlocks[k][j]);
        if (BH == -1)
        {
          BH = X;
          BW = Y;
        }
        else
        {
          if ( (X != BH) || (Y != BW) )
          {
            throw Exception("Block2SparseMatrix - block-dims do not match!");
            return nullptr;
          }
        }
      }
    }
  }

  BlockMatrix convA(convBlocks);

  shared_ptr<BaseMatrix> singleSparseMat = nullptr;

  Switch<AMG_MAX_SYS_DIM> (BH - 1, [&](auto HM) {
    Switch<AMG_MAX_SYS_DIM> (BW - 1, [&](auto WM) {
      constexpr int H = HM + 1;
      constexpr int W = WM + 1;
      if constexpr(isSparseMatrixCompiled<H, W>())
      {
        singleSparseMat = Block2SparseMatrix_impl<H, W>( convA );
      }
    });
  });

  shared_ptr<ParallelDofs> fusedRowParDOFs = nullptr;
  shared_ptr<ParallelDofs> fusedColParDOFs = nullptr;

  if (blockPar)
  {
    fusedRowParDOFs = fuseParallelDOFs(rowPds);
    fusedColParDOFs = fuseParallelDOFs(colPds);
  }

  return WrapParallelMatrix(singleSparseMat,
                            UniversalDofs(fusedRowParDOFs, singleSparseMat->Height(), BH),
                            UniversalDofs(fusedColParDOFs, singleSparseMat->Width(), BW),
                            blockOp);
}

shared_ptr<BaseMatrix>
Block2SparseMatrix (BlockMatrix const &A)
{
  // (i,j) operator is not const in NGSolve
  return Block2SparseMatrix(const_cast<BlockMatrix&>(A));
}

template<class TSPM>
INLINE
shared_ptr<TSPM>
EmbedSparseMatrix_impl(TSPM const &smallA, IntRange r, int const bigH)
{
  Array<int> perow(bigH);
  perow = 0;

  int const off = r.First();

  int const smallH = smallA.Height();

  for (auto k : Range(smallH))
  {
    perow[off + k] = smallA.GetRowIndices(k).Size();
  }

  // cout << " EmbedSparseMatrix_impl " << r << " -> " << bigH << endl;
  // cout << "  Entry-DIMs " << EntryHeight<TSPM>() << " x " << EntryWidth<TSPM>() << endl;
  // cout << " W = " << smallA.Width() << endl;
  // cout << " perow = " << perow << endl;

  // the width is the one of the small matrix
  //         |    0   |
  // bigA =  | smallA |
  //         |    0   |
  auto bigA = make_shared<TSPM>(perow, smallA.Width());

  for (auto k : Range(smallH))
  {
    auto ris = bigA->GetRowIndices(off + k);
    auto rvs = bigA->GetRowValues(off + k);

    auto smallRis = smallA.GetRowIndices(k);
    auto smallRvs = smallA.GetRowValues(k);

    ris = smallRis;
    rvs = smallRvs;
  }

  // cout <<  "OK ! " << endl;

  return bigA;
}

shared_ptr<BaseMatrix>
EmbedSparseMatrix(shared_ptr<BaseMatrix> A, IntRange r, int const bigH)
{
  shared_ptr<BaseMatrix> embeddedA = nullptr;
  DispatchOverMatrixDimensions(*A, [&](auto const &spA, auto H, auto W) {
    embeddedA = EmbedSparseMatrix_impl(spA, r, bigH);
  });
  return embeddedA;
}

template<int BS>
INLINE
shared_ptr<stripped_spm_tm<Mat<BS, BS, double>>>
createSparseIdentityImpl(int H, BitArray const *onRows)
{
  Array<int> perow(H); perow = 1;

  if (onRows)
  {
    for (auto k : Range(perow))
    {
      if (!onRows->Test(k))
      {
        perow[k] = 0;
      }
    }
  }

  auto spI = make_shared<stripped_spm_tm<Mat<BS, BS, double>>>(perow, H);

  for (auto k : Range(perow))
  {
    if ( (onRows == nullptr) || (onRows->Test(k) ) )
    {
      spI->GetRowIndices(k)[0] = 1;
      spI->GetRowValues(k)[0] = 1.0;
    }
  }

  return spI;
}

shared_ptr<BaseMatrix>
createSparseIdentity(int H, int BS, BitArray const *onRows = nullptr)
{
  shared_ptr<BaseMatrix> spI = nullptr;
  Switch<AMG_MAX_SYS_DIM>(BS-1, [&](auto BSM) {
    constexpr int BS = BSM + 1;
    spI = createSparseIdentityImpl<BS>(H, onRows);
  });
  return spI;
}

template<int BH, int BW>
shared_ptr<stripped_spm<Mat<BH, BW, double>>>
assembleElByEl (ElementByElementMatrix<double> const &A)
{
  throw Exception("assembleElByEl does not work with upstream NGSolve!");

  int const hScal = A.Height();
  int const H  = hScal / BH;

  int const wScal = A.Width();
  int const W  = wScal / BW;

  auto toBlkRow = [&](auto k) { return std::make_tuple(k%BH == 0, k / BH); };
  auto toBlkCol = [&](auto k) { return std::make_tuple(k%BW == 0, k / BW); };

  auto mapThem = [&](FlatArray<int> cols, Array<int> &bCols, auto map) -> FlatArray<int> {
    bCols.SetSize(cols.Size());
    int c = 0;
    for (auto j : Range(cols))
    {
      auto col = cols[j];
      auto [ isFirst, blockCol ] = map(col);
      if (isFirst)
      {
        bCols[c++] = blockCol;
      }
    }
    bCols.SetSize(c);
    return bCols;
  };

  Array<int> bRows(100);
  Array<int> bCols(100);
  auto mapCols = [&](auto cols) { return mapThem(cols, bCols, toBlkCol); };
  auto mapRows = [&](auto rows) { return mapThem(rows, bRows, toBlkRow); };

  // auto const nMats = A.GetNumElMats();
  auto const nMats = -1;

  Array<int> perowR(nMats);
  Array<int> perowC(nMats);

  for (auto k : Range(nMats))
  {
    auto rowNums = A.GetElementRowDNums(k);
    auto colNums = A.GetElementColumnDNums(k);

    perowR[k] = rowNums.Size() / BH;
    perowC[k] = colNums.Size() / BW;
  }

  Table<int> rowTable(perowR);
  Table<int> colTable(perowC);

  for (auto k : Range(nMats))
  {
    auto rowNums = A.GetElementRowDNums(k);
    auto colNums = A.GetElementColumnDNums(k);

    rowTable[k] = mapRows(rowNums);
    colTable[k] = mapCols(rowNums);
  }

  auto spA = make_shared<stripped_spm<Mat<BH, BW, double>>>(H, W, rowTable, colTable, false);

  for (auto k : Range(nMats))
  {
    // shmem: atomic!
    spA->AddElementMatrix(rowTable[k], colTable[k], A.GetElementMatrix(k), false);
  }

  return spA;
}


template<int N, int M> INLINE double abssum_tm(Mat<N, M, double> const &x)
{
  double s = 0.0;
  Iterate<N>([&](auto i) {
    Iterate<M>([&](auto j) {
      s += abs(x(i.value, j.value));
    });
  });
  return s;
}

INLINE double abssum_tm(double const &x) { return x; }

template<class TSPM>
shared_ptr<TSPM>
CompressA_impl(TSPM const &A, double const &thresh = 1e-20)
{
  int const H = A.Height();

  auto itFilteredGraph = [&](auto lam) {
    for (auto k : Range(H))
    {
      auto ris = A.GetRowIndices(k);
      auto rvs = A.GetRowValues(k);

      for (auto j : Range(ris))
      {
        auto const &v = rvs[j];
        if (abssum_tm(v) > thresh)
        {
          lam(k, ris[j], v);
        }
      }
    }
  };

  Array<int> perow(H);

  perow = 0;
  itFilteredGraph([&](auto row, auto col, auto const &val) { perow[row]++; });

  auto compressedA = make_shared<TSPM>(perow, A.Width());

  perow = 0;
  itFilteredGraph([&](auto row, auto col, auto const &val) {
    int const idxInRow = perow[row]++;
    compressedA->GetRowIndices(row)[idxInRow] = col;
    compressedA->GetRowValues(row)[idxInRow]  = val;
  });

  return compressedA;
}

shared_ptr<BaseMatrix>
CompressAGeneric (shared_ptr<BaseMatrix> A, double const &thresh)
{
  auto [rowUD, colUD, locMat, op] = UnwrapParallelMatrix(A);
  shared_ptr<BaseMatrix> compressedA = nullptr;
  DispatchOverMatrixDimensions(*locMat, [&](auto const &spA, auto H, auto W) {
    // filter based on average entry by using thresh * H * W
    compressedA = CompressA_impl(spA, thresh * H * W);
  });
  return WrapParallelMatrix(compressedA,
                            rowUD,
                            colUD,
                            op);
}

shared_ptr<BaseMatrix>
createSparseIdentity(shared_ptr<BaseMatrix> A)
{
  auto [rowUD, colUD, locMat, op] = UnwrapParallelMatrix(A);

  if (!(rowUD == colUD))
  {
    throw Exception("Cannot create non-square sparse Identity Matrix!");
  }

  BitArray const *masterRows = rowUD.IsParallel() ? &rowUD.GetParallelDofs()->MasterDofs() : nullptr;

  return WrapParallelMatrix(createSparseIdentity(rowUD.GetNDofLocal(), rowUD.GetBS(), masterRows),
                            rowUD,
                            colUD,
                            op);
}

template<class TM>
INLINE
shared_ptr<TM>
restrictMatrixToBlocksImpl (TM &A,
                            FlatTable<int> rowBlocks,
                            FlatTable<int> colBlocks,
                            double const &compressTol=1e-20)
{
  for (auto k : Range(rowBlocks))
  {
    auto rowBlock = rowBlocks[k];
    auto colBlock = colBlocks[k];
    for (auto l : Range(rowBlock))
    {
      int const row = rowBlock[l];

      auto ris = A.GetRowIndices(row);
      auto rvs = A.GetRowValues(row);

      iterate_anotb(ris, colBlock, [&](auto idxRi) { rvs[idxRi] = 0.0; });
    }
  }

  return CompressA_impl(A, compressTol);
}


shared_ptr<BaseMatrix>
restrictMatrixToBlocks(shared_ptr<BaseMatrix> A,
                       FlatTable<int> rowBlocks,
                       FlatTable<int> colBlocks,
                       double const &compressTol=1e-20)
{
  auto [rowUD, colUD, locMat, op] = UnwrapParallelMatrix(A);

  shared_ptr<BaseMatrix> rMat = nullptr;

  DispatchOverMatrixDimensions(*locMat, [&](auto &spA, auto H, auto W) {
    rMat = restrictMatrixToBlocksImpl(spA, rowBlocks, colBlocks, compressTol);
  });

  return WrapParallelMatrix(rMat,
                            rowUD,
                            colUD,
                            op);
}

extern
shared_ptr<SparseMatrix<double>>
makeLOToHOFacetEmbedding(shared_ptr<FESpace> orig,
                         shared_ptr<FESpace> dest,
                         shared_ptr<BitArray> destFree = nullptr)
{
  int NO = orig->GetNDof();
  int ND = dest->GetNDof();

  Array<int> perow(ND);
  perow = 0;

  Array<int> dFacetDOFs;
  Array<int> oFacetDOFs;

  auto const &MA = *dest->GetMeshAccess();

  for (auto fnum : Range(MA.GetNFacets()))
  {
    dest->GetDofNrs(NodeId(NT_FACET, fnum), dFacetDOFs);
    orig->GetDofNrs(NodeId(NT_FACET, fnum), oFacetDOFs);

    if (dFacetDOFs.Size())
    {
      auto loDOF = dFacetDOFs[0];
      if (loDOF != -1)
      {
        if ( ( destFree == nullptr ) || ( destFree->Test(loDOF) ) )
        {
          perow[loDOF] = oFacetDOFs.Size();
        }
      }
    }
  }

  auto emb = make_shared<SparseMatrix<double>>(perow, NO);

  for (auto fnum : Range(MA.GetNFacets()))
  {
    dest->GetDofNrs(NodeId(NT_FACET, fnum), dFacetDOFs);
    orig->GetDofNrs(NodeId(NT_FACET, fnum), oFacetDOFs);

    if (dFacetDOFs.Size())
    {
      auto loDOF = dFacetDOFs[0];
      if (loDOF != -1)
      {
        if ( ( destFree == nullptr ) || ( destFree->Test(loDOF) ) )
        {
          auto ris = emb->GetRowIndices(loDOF);
          auto rvs = emb->GetRowValues(loDOF);

          for (auto j : Range(ris))
          {
            ris[j] = oFacetDOFs[j];
            rvs[j] = 1;
          }
        }
      }
    }
  }

  return emb;
}


shared_ptr<BaseMatrix> BaseMatrixToSparse (shared_ptr<BaseMatrix> A)
{
  // std::cout << " BaseMatrixToSparse " << A << endl;
  // if (A)
  // {
  //   cout << "   BaseMatrixToSparse " << typeid(*A).name() << endl;
  // }
  if (auto spA = dynamic_pointer_cast<BaseSparseMatrix>(A))
  {
    // cout << " BaseMatrixToSparse BSPM" << endl;
    return A;
  }
  else if (auto parA = dynamic_pointer_cast<ParallelMatrix>(A))
  {
    // cout << " BaseMatrixToSparse parmat" << endl;
    auto [rowUD, colUD, locMat, op] = UnwrapParallelMatrix(A);

    // cout << " BaseMatrixToSparse parmat check loc " << typeid(*locMat).name() << endl;
    if (auto spLocA = dynamic_pointer_cast<BaseSparseMatrix>(locMat))
    {
      // cout << " BaseMatrixToSparse parmat loc sparse" << endl;
      return A;
    }
    else if (auto locI = dynamic_pointer_cast<IdentityMatrix>(locMat))
    {
      auto locSparse = createSparseIdentity(rowUD.GetNDofLocal(), rowUD.GetBS(), &rowUD.GetParallelDofs()->MasterDofs());
      return WrapParallelMatrix(locSparse, rowUD, colUD, op);
    }
    else
    {
      // cout << " BaseMatrixToSparse parmat loc recurse" << endl;
      auto locSparse = BaseMatrixToSparse(locMat);
      return WrapParallelMatrix(locSparse, rowUD, colUD, op);
    }
  }
  else if (auto T = dynamic_pointer_cast<Transpose>(A) )
  {
    // cout << " BaseMatrixToSparse transpose" << endl;
    return TransposeAGeneric(BaseMatrixToSparse(T->SPtrMat()));
  }
  else if (auto prod = dynamic_pointer_cast<ProductMatrix>(A))
  {
    // could check for block-matrix here and do the product block-wise, that would probably be better?
    shared_ptr<BaseMatrix> A = prod->SPtrA();
    shared_ptr<BaseMatrix> B = prod->SPtrB();

    if (auto idB = dynamic_pointer_cast<IdentityMatrix>(B))
    {
      return BaseMatrixToSparse(A);
    }
    else if (auto idA = dynamic_pointer_cast<IdentityMatrix>(A))
    {
      return BaseMatrixToSparse(B);
    }
    else
    {
      if (auto ebeA = dynamic_pointer_cast<ElementByElementMatrix<double>>(GetLocalMat(A)))
      {
        // cout << " BaseMatrixToSparse prod - convert B" << endl;
        B = BaseMatrixToSparse(B);
        if (auto sparseB = dynamic_pointer_cast<BaseSparseMatrix>(GetLocalMat(B)))
        {
          DispatchMatrixHeight(*sparseB, [&](auto HBP)
          {
            constexpr int HB = HBP;
            A = assembleElByEl<1, HB>(*ebeA);
          });
        }
        else
        {
          throw Exception("BaseMatrixToSparse prod - could not sparsify B!");
        }
      }
      else if (auto ebeB = dynamic_pointer_cast<ElementByElementMatrix<double>>(GetLocalMat(B)))
      {
        // cout << " BaseMatrixToSparse prod - convert A" << endl;
        A = BaseMatrixToSparse(A);
        // cout << " BaseMatrixToSparse prod - EBE assemble B" << endl;
        if (auto sparseA = dynamic_pointer_cast<BaseSparseMatrix>(GetLocalMat(A)))
        {
          DispatchMatrixWidth(*sparseA, [&](auto WA)
          {
            B = assembleElByEl<WA, 1>(*ebeB);
          });
        }
        else
        {
          throw Exception("BaseMatrixToSparse prod - could not sparsify A!");
        }
      }
      else
      {
        cout << " BaseMatrixToSparse prod - convert A" << endl;
        A = BaseMatrixToSparse(A);
        cout << " BaseMatrixToSparse prod - convert B" << endl;
        B = BaseMatrixToSparse(B);
      }

      // cout << " BaseMatrixToSparse prod MULT" << endl;
      auto AB = MatMultABGeneric(A, B);

      return AB;
    }
  }
  else if (auto sumA = dynamic_pointer_cast<SumMatrix>(A))
  {
    shared_ptr<BaseMatrix> A = sumA->SPtrA();
    shared_ptr<BaseMatrix> B = sumA->SPtrB();

    if (auto idB = dynamic_pointer_cast<IdentityMatrix>(B))
    {
      A = BaseMatrixToSparse(A);
      B = createSparseIdentity(A);
    }
    else if (auto idA = dynamic_pointer_cast<IdentityMatrix>(A))
    {
      B = BaseMatrixToSparse(B);
      A = createSparseIdentity(B);
    }
    else
    {
      A = BaseMatrixToSparse(A);
      B = BaseMatrixToSparse(B);
    }

    return AddABGeneric(1.0, A, 1.0, B);
  }
  else if (auto scaledA = dynamic_pointer_cast<VScaleMatrix<double>>(A))
  {
    throw Exception("BaseMatrixToSparse - scale not implemented!");
    return nullptr;
  }
  else if (auto blockA = dynamic_pointer_cast<BlockMatrix>(A))
  {
    cout << " BaseMatrixToSparse sum - BLOCK" << endl;
    return Block2SparseMatrix(*blockA);
  }
  else if (auto embA = dynamic_pointer_cast<EmbeddedMatrix>(A))
  {
    // cout << " BaseMatrixToSparse sum - EMB, convert small" << endl;
    auto smallA = BaseMatrixToSparse(embA->GetMatrix());

    // cout << " BaseMatrixToSparse sum - EMB" << endl;
    return EmbedSparseMatrix(smallA, embA->GetRange(), embA->Height());
  }
  else if (auto elByEl = dynamic_pointer_cast<ElementByElementMatrix<double>>(A))
  {
    // return assembleElByEl(elByEl);
    throw Exception("BaseMatrixToSparse - cannot sparsify ElementByElementMatrix, no access to NELs in NGSolve!");
  }
  else if (auto idB = dynamic_pointer_cast<IdentityMatrix>(A))
  {
    throw Exception("BaseMatrixToSparse - cannot do anything with IdentityMatrix, need size & block-size!");
  }
  else
  {
    throw Exception("BaseMatrixToSparse - not implemented!");
    return nullptr;
  }
} // BaseMatrixToSparse

} // namespace amg
