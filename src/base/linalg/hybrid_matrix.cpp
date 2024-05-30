#include "hybrid_matrix.hpp"
#include "utils.hpp"
#include "utils_sparseLA.hpp"

#include <utils_arrays_tables.hpp>
#include <utils_io.hpp>

namespace amg
{

namespace
{

/**
 * Decompose a regular sparse matrix into M and G, suitable for a hybrid sparse matrix
 */
template<class TM>
std::tuple<shared_ptr<SparseMatrix<TM>>,
           shared_ptr<SparseMatrix<TM>>>
DecomposeSparseMatrixHybrid (shared_ptr<SparseMatrix<TM>> anA,
                             shared_ptr<ParallelDofs>       pardofs,
                             DCCMap<typename mat_traits<TM>::TSCAL> const &dCCMap)
{
  string cn = string("DecomposeSparseMatrixHybrid<" + to_string(ngbla::Height<TM>()) + string(">"));
  static Timer t(cn + "::SetUpMats");
  RegionTimer rt(t);

  auto& A(*anA);
  auto& pds(*pardofs);

  auto H = A.Height();
  NgsAMG_Comm comm(pds.GetCommunicator());
  auto ex_procs = pds.GetDistantProcs();
  auto nexp = ex_procs.Size();

  // cout << " COMM R " << comm.Rank() << " S " << comm.Size() << endl;
  // cout << " PDS ND L " << pds.GetNDofLocal() << " G " << pds.GetNDofGlobal() << endl;
  // cout << " HYBM EX_PROCS = "; prow2(ex_procs); cout << endl;
  // cout << " PDS ARE " << pardofs << endl;
  // cout << " PDS ARE " << endl << pds << endl;

  shared_ptr<SparseMatrix<TM>> spM;
  shared_ptr<SparseMatrix<TM>> spG;

  if (nexp == 0)
  {
    // we are still not "dummy", because we are still working with parallel vectors
    spM = anA;
    spG = nullptr;
    return std::make_tuple(spM, spG);
  }

  typedef SparseMatrixTM<TM> TSPMAT_TM;
  Array<TSPMAT_TM*> send_add_diag_mats(nexp);
  Array<NG_MPI_Request> rsdmat (nexp);
  { // create & send diag-blocks for M to masters
    auto SPM_DIAG = [&] (auto dofs) LAMBDA_INLINE
    {
      auto iterate_coo = [&](auto fun) LAMBDA_INLINE
      {
        for (auto i : Range(dofs.Size()))
        {
          auto d = dofs[i];
          auto ris = A.GetRowIndices(d);
          auto rvs = A.GetRowValues(d);

          for (auto j : Range(ris.Size()))
          {
            auto pos = find_in_sorted_array(ris[j], dofs);

            if (pos != -1)
            {
              fun(i, pos, rvs[j]);
            }
          }
        }
      };

      Array<int> perow(dofs.Size()); perow = 0;

      iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE { perow[i]++; });

      TSPMAT_TM* dspm = new TSPMAT_TM(perow, perow.Size());

      perow = 0;

      iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE
      {
        dspm->GetRowIndices(i)[perow[i]] = j;
        dspm->GetRowValues(i)[perow[i]++] = val;
      });

      return dspm;
    };

    for (auto kp : Range(nexp))
    {
      send_add_diag_mats[kp] = SPM_DIAG(dCCMap.GetGDOFs(kp));
      rsdmat[kp] = comm.ISend(*send_add_diag_mats[kp], ex_procs[kp], NG_MPI_TAG_AMG);
    }
  } // create & send diag-blocks for M to masters

  Array<shared_ptr<TSPMAT_TM>> recv_mats(nexp);
  { // recv diag-mats
    static Timer t(cn + "::SetUpMats - recv diag");
    RegionTimer rt(t);

    for (auto kp : Range(nexp))
    {
      comm.Recv(recv_mats[kp], ex_procs[kp], NG_MPI_TAG_AMG);
      // cout << " MAT FROM kp " << kp << " proc " << ex_procs[kp] << endl;
      // print_tm_spmat(cout, *recv_mats[kp]); cout << endl;
    }
  }

  { // merge diag-mats
    static Timer t(cn + "::SetUpMats - merge");
    RegionTimer rt(t);

    auto& mf_dofs = *dCCMap.GetMasterDOFs();

    Array<int> perow(H); perow = 0;
    Array<size_t> at_row(nexp);
    Array<int> row_matis(nexp);
    Array<FlatArray<int>> all_cols;
    Array<int> mrowis(50); mrowis.SetSize0(); // col-inds for a row of orig mat (but have to remove a couple first)

    auto iterate_rowinds = [&](auto fun, bool map_exd) LAMBDA_INLINE
    {
      at_row = 0; // restart counting through rows of recv-mats

      for (auto rownr : Range(H))
      {
        if (mf_dofs.Test(rownr))
        { // I am master of this dof - merge recved rows with part of original row
          row_matis.SetSize0(); // which mats I received have this row?

          if (pds.GetDistantProcs(rownr).Size()) // local master
          {

            for (auto kp : Range(nexp))
            {
              auto exds = dCCMap.GetMDOFs(kp);
              if (at_row[kp] == exds.Size()) continue; // no more rows to take from there
              auto ar = at_row[kp]; // the next row for that ex-mat
              size_t ar_dof = exds[ar]; // the dof that row belongs to
              if (ar_dof > rownr) continue; // not yet that row's turn
              row_matis.Append(kp);
              // cout << "row " << at_row[kp] << " from " << kp << endl;
              at_row[kp]++;
            }
            
            all_cols.SetSize0(); // otherwise tries to copy FA I think
            all_cols.SetSize(1 + row_matis.Size());
            
            for (auto k : Range(all_cols.Size()-1)) {
              auto kp = row_matis[k];
              auto cols = recv_mats[kp]->GetRowIndices(at_row[kp]-1);
              if (map_exd) // !!! <- remap col-nrs of received rows, only do this ONCE!!
              {
                auto mxd = dCCMap.GetMDOFs(kp);
                // cout << "k " << k << ", kp " << kp << ", proc = " << ex_procs[kp] << endl;
                // cout << "cols = "; prow2(cols); cout << endl;
                // cout << "mxd  = "; prow2(mxd); cout << endl;
                for (auto j : Range(cols.Size()))
                  { cols[j] = mxd[cols[j]]; }
                // cout << " was ok " << endl;
              }
              all_cols[k].Assign(cols);
            }

            // only master-master (not master-all) goes into M
            auto aris = A.GetRowIndices(rownr);
            mrowis.SetSize(aris.Size()); int c = 0;
  
            for (auto col : aris)
            {
              if (mf_dofs.Test(col))
                { mrowis[c++] = col; }
            }
  
            mrowis.SetSize(c);
            all_cols.Last().Assign(mrowis);

            // cout << "merge cols: " << endl;
            // for (auto k : Range(all_cols.Size()))
            // 	{ cout << k << " || "; prow2(all_cols[k]); cout << endl; }
            // cout << endl;

            auto merged_cols = merge_arrays(all_cols, [](const auto&a, const auto &b){return a<b; });

            // cout << "merged cols: "; prow2(merged_cols); cout << endl;

            fun(rownr, row_matis, merged_cols);
          } // mf-exd.Test(rownr);
          else
          { // local row - pick out only master-cols
            auto aris = A.GetRowIndices(rownr);
            mrowis.SetSize(aris.Size()); int c = 0;
            for (auto col : aris)
            {
              if (mf_dofs.Test(col))
                { mrowis[c++] = col; }
            }

            mrowis.SetSize(c);
            fun(rownr, row_matis, mrowis);
          }
        } // mf_dofs.Test(rownr)
      }
    };

    iterate_rowinds([&](auto rownr, const auto &matis, const auto &rowis) LAMBDA_INLINE
    {
      perow[rownr] = rowis.Size();
    }, true);

    spM = make_shared<SparseMatrix<TM>>(perow);

    // cout << "M NZE : " << M->NZE() << endl;

    iterate_rowinds([&](auto rownr, const auto & matis, const auto & rowis)
    {
      auto ris = spM->GetRowIndices(rownr); ris = rowis;
      auto rvs = spM->GetRowValues(rownr); rvs = 0;
      // cout << "rownr, rowis: " << rownr << ", "; prow2(rowis); cout << endl;
      // cout << "rownr, matis: " << rownr << ", "; prow2(matis); cout << endl;
      // cout << ris.Size() << " " << rvs.Size() << " " << rowis.Size() << endl;
      auto add_vals = [&](auto cols, auto vals) LAMBDA_INLINE
      {
        for (auto l : Range(cols))
        {
          auto pos = find_in_sorted_array<int>(cols[l], ris);
          // cout << "look for " << cols[l] << " in "; prow(ris); cout << " -> pos " << pos << endl;
          if (pos != -1)
            { rvs[pos] += vals[l]; }
        }
      };

      add_vals(A.GetRowIndices(rownr), A.GetRowValues(rownr));

      for (auto kp : matis)
      {
        // cout << "row " << at_row[kp] -1 << " from kp " << kp << endl;
        add_vals(recv_mats[kp]->GetRowIndices(at_row[kp]-1),
                 recv_mats[kp]->GetRowValues(at_row[kp]-1));
      }
    }, false);

  } // merge diag-mats

  // cout << endl << "ORIG A MAT: " << endl << A << endl << endl ;
  // cout << endl  << "M done: " << endl << *spM << endl << endl ;

  { // build G-matrix
    static Timer t(cn + "::SetUpMats - S-mat"); RegionTimer rt(t);
    // ATTENTION: explicitely only for symmetric matrices!
    Array<int> master_of(H); // DOF to master mapping (0 .. nexp-1 ordering // -1 -> I am master)
    master_of = -1;
    for (auto kp : Range(ex_procs))
    {
      for (auto d : dCCMap.GetGDOFs(kp))
        { master_of[d] = kp; }
    }

    Array<int> perow(A.Height());

    auto iterate_coo = [&](auto fun) LAMBDA_INLINE // could be better if we had GetMaxExchangeDof(), or GetExchangeDofs()
    {
      perow = 0; // <- !! is set to zero here ...
      for (auto rownr : Range(A.Height())) {
        auto ris = A.GetRowIndices(rownr);
        auto rvs = A.GetRowValues(rownr);
        auto mrow = master_of[rownr];
        for (auto j : Range(ris)) {
          if (master_of[ris[j]] != mrow) {
            fun(rownr, ris[j], rvs[j]);
          }
        }
      }
    };

    iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE
    {
      perow[i]++;
    });
    
    spG = make_shared<SparseMatrix<TM>>(perow);

    iterate_coo([&](auto i, auto j, auto val) LAMBDA_INLINE
    {
      spG->GetRowIndices(i)[perow[i]] = j;
      spG->GetRowValues(i)[perow[i]++] = val;
    });
  } // build G-matrix

  // cout << endl  << "G done: " << endl << *G << endl << endl ;

  {
    static Timer t(cn + "::SetUpMats - finish send"); RegionTimer rt(t);
    MyMPI_WaitAll(rsdmat);
    for (auto kp : Range(nexp))
      { delete send_add_diag_mats[kp]; }
  }

  return std::make_tuple(spM, spG);
} // DecomposeSparseMatrixHybrid

}


/** HybridBaseMatrix **/

shared_ptr<BaseMatrix>
MakeSumMatrix(shared_ptr<BaseMatrix> A,
              shared_ptr<BaseMatrix> B)
{
  if (A != nullptr)
  {
    if (B != nullptr)
    {
      return make_shared<SumMatrix>(A, B);
    }
    else
    {
      return A;
    }
  }
  else
  {
    return B;
  }
}


template<class TSCAL>
HybridBaseMatrix<TSCAL>::
HybridBaseMatrix(shared_ptr<ParallelDofs>  parDOFs,
                 shared_ptr<DCCMap<TSCAL>> dCCMap,
                 shared_ptr<BaseMatrix>    aM,
                 shared_ptr<BaseMatrix>    aG)
  : BaseMatrix(parDOFs)
  , _dCCMap(dCCMap)
  , _M(aM)
  , _G(aG)
  , _MPG(MakeSumMatrix(_M, _G))
{
  if (dCCMap == nullptr)
  {
    dummy = true;
    g_zero = true;
  }
  else
  {
    dummy = false;

    int nzg = (_G != nullptr) ? 1 : 0;

    nzg = this->GetParallelDofs()->GetCommunicator().AllReduce(nzg, NG_MPI_SUM);

    g_zero = (nzg == 0);
  }
}


template<class TSCAL>
shared_ptr<DCCMap<TSCAL>>
HybridBaseMatrix<TSCAL>::
GetDCCMapPtr ()
{
  return _dCCMap;
} // HybridBaseMatrix::getDCCMapPtr


template<class TSCAL>
DCCMap<TSCAL> &
HybridBaseMatrix<TSCAL>::
GetDCCMap ()
{
  return *_dCCMap;
} // HybridBaseMatrix::getDCCMap


template<class TSCAL>
DCCMap<TSCAL> const&
HybridBaseMatrix<TSCAL>::
GetDCCMap () const
{
  return *_dCCMap;
} // HybridBaseMatrix::getDCCMap


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
MultAdd (double s, const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::MultAdd");
  RegionTimer rt(t);

  x.Cumulate();
  y.Distribute();

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();

  GetM()->MultAdd(s, locX, locY);

  if ( HasGLocal() )
    { GetG()->MultAdd(s, locX, locY); }
} // HybridBaseMatrix::MultTrans


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::MultAdd");
  RegionTimer rt(t);

  x.Cumulate();
  y.Distribute();

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();

  GetM()->MultAdd(s, locX, locY);

  if ( HasGLocal() )
    { GetG()->MultAdd(s, locX, locY); }
} // HybridBaseMatrix::MultTrans


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
Mult (const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::Mult");
  RegionTimer rt(t);

  x.Cumulate();
  y.SetParallelStatus(DISTRIBUTED);

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();

  GetM()->Mult(locX, locY);

  if ( HasGLocal() )
    { GetG()->MultAdd(1.0, locX, locY); }
} // HybridBaseMatrix::MultTrans


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::MultTransAdd");
  RegionTimer rt(t);

  x.Cumulate();
  y.Distribute();

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();
  
  GetM()->MultTransAdd(s, locX, locY);

  if ( HasGLocal() )
    { GetG()->MultTransAdd(s, locX, locY); }
} // HybridBaseMatrix::MultTrans


template<class TSCAL>
void
HybridBaseMatrix<TSCAL> :: MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::MultTransAdd");
  RegionTimer rt(t);

  x.Cumulate();
  y.Distribute();

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();

  GetM()->MultTransAdd(s, locX, locY);

  if ( HasGLocal() )
    { GetG()->MultTransAdd(s, locX, locY); }
} // HybridBaseMatrix::MultTrans


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
MultTrans (const BaseVector & x, BaseVector & y) const
{
  static Timer t("HybridBaseMatrix::MultTrans");
  RegionTimer rt(t);

  x.Cumulate();
  y.SetParallelStatus(DISTRIBUTED);

  auto const &locX = *x.GetLocalVector();
  auto       &locY = *y.GetLocalVector();

  GetM()->MultTrans(locX, locY);

  if ( HasGLocal() )
    { GetG()->MultTransAdd(1.0, locX, locY); }

} // HybridBaseMatrix::MultTrans


template<class TSCAL>
AutoVector
HybridBaseMatrix<TSCAL>::
CreateVector () const
{
  auto parDOFs = this->GetParallelDofs();

  return CreateSuitableVector(parDOFs->GetNDofLocal(),
                              parDOFs->GetEntrySize(),
                              parDOFs);
} // HybridBaseMatrix::CreateVector


template<class TSCAL>
AutoVector
HybridBaseMatrix<TSCAL>::
CreateRowVector () const
{
  return CreateVector();
} // HybridBaseMatrix::CreateRowVector


template<class TSCAL>
AutoVector
HybridBaseMatrix<TSCAL>::
CreateColVector () const
{
  return CreateVector();
} // HybridBaseMatrix::CreateColVector



template<class TSCAL>
bool
HybridBaseMatrix<TSCAL>::
IsComplex() const
{
  return is_same<double, TSCAL>::value ? false : true;
} // HybridBaseMatrix::IsComplex


template<class TSCAL>
size_t
HybridBaseMatrix<TSCAL>::
NZE () const
{
  return this->GetM()->NZE() + ( HasGLocal() ? this->GetG()->NZE() : 0ul ); 
} // HybridBaseMatrix::NZE


template<class TSCAL>
void
HybridBaseMatrix<TSCAL>::
SetMG(shared_ptr<BaseMatrix> aM, shared_ptr<BaseMatrix> aG)
{
  _M = aM;
  _G = aG;
  _MPG = MakeSumMatrix(_M, _G);

  int nzg = (_G != nullptr) ? 1 : 0;

  nzg = this->GetParallelDofs()->GetCommunicator().AllReduce(nzg, NG_MPI_SUM);

  g_zero = (nzg == 0);
}

/** END HybridBaseMatrix **/


/** HybridMatrix **/

template<class TM>
HybridMatrix<TM>::
HybridMatrix (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map)
  : HybridBaseMatrix<TSCAL>(mat->GetParallelDofs(),
                            _dcc_map)
  , spM(nullptr)
  , spG(nullptr)
{
  if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat))
  {
    auto loc_spmat = my_dynamic_pointer_cast<SparseMatrix<TM>>(parmat->GetMatrix(),
                                                               "HybriMatrix constructor");
    // can still have dummy-pardofs (ngs-amg w. NP==1) or no ex-procs (NP==2)
    SetUpMats(loc_spmat);
  }
  else if (auto spmat = dynamic_pointer_cast<SparseMatrix<TM>>(mat))
  {
    if (spmat == nullptr)
      { throw Exception("Dummy-HybridMatrix not actually a dummy!"); }

    spM = spmat;
    spG = nullptr;
  }

  this->SetMG(spM, spG);
} // HybridMatrix(..)


template<class TM>
HybridMatrix<TM>::
HybridMatrix (shared_ptr<BaseMatrix> mat)
  : HybridMatrix<TM>(mat,
                     mat->GetParallelDofs() == nullptr ? nullptr
                                                       : make_shared<BasicDCCMap<TSCAL>>(mat->GetParallelDofs()))
{
} // HybridMatrix(..)


template<class TM>
void
HybridMatrix<TM>::
SetUpMats (shared_ptr<SparseMatrix<TM>> anA)
{
  auto [ sM, sG ] = DecomposeSparseMatrixHybrid<TM>(anA, this->GetParallelDofs(), this->GetDCCMap());

  spM = sM;
  spG = sG;
} // HybridMatrix::SetUpMats


template<class TM>
size_t
HybridMatrix<TM>::
EntrySize () const
{
  return ( spM == nullptr ) ? 1 : GetEntrySize(spM.get());
}


template<class TM>
void
HybridMatrix<TM>::
ReplaceM (shared_ptr<BaseMatrix> M)
{
  spM = nullptr;

  this->SetMG(M, this->GetG());
}
/** END HybridMatrix **/


/** DynamicBlockHybridMatrix **/

template<class TSCAL>
DynamicBlockHybridMatrix<TSCAL>::
DynamicBlockHybridMatrix (shared_ptr<BaseMatrix> mat, shared_ptr<DCCMap<TSCAL>> _dcc_map)
  : HybridBaseMatrix<TSCAL>(mat->GetParallelDofs(), _dcc_map)
{
  if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(mat))
  {
    auto loc_spmat = my_dynamic_pointer_cast<SparseMatrix<TSCAL>>(parmat->GetMatrix(),
                                                                  "DynamicBlockHybridMatrix constructor");

    auto [ sM, sG ] = DecomposeSparseMatrixHybrid<TSCAL>(loc_spmat, this->GetParallelDofs(), this->GetDCCMap());

    dynSpM = make_shared<DynBlockSparseMatrix<TSCAL>>(*sM);
    dynSpG = ( sG == nullptr ) ? nullptr : make_shared<DynBlockSparseMatrix<TSCAL>>(*sG);
  }
  else if (auto spmat = dynamic_pointer_cast<SparseMatrix<TSCAL>>(mat))
  {
    if (spmat == nullptr)
      { throw Exception("Dummy-DynamicBlockHybridMatrix not actually a dummy!"); }

    dynSpM = make_shared<DynBlockSparseMatrix<TSCAL>>(*spmat);
    dynSpG = nullptr;
  }

  this->SetMG(dynSpM, dynSpG);
} // DynamicBlockHybridMatrix(..)


template<class TSCAL>
DynamicBlockHybridMatrix<TSCAL>::
DynamicBlockHybridMatrix (shared_ptr<BaseMatrix> mat)
  : DynamicBlockHybridMatrix<TSCAL>(mat,
                                    mat->GetParallelDofs() == nullptr ? nullptr
                                                       : make_shared<BasicDCCMap<TSCAL>>(mat->GetParallelDofs()))
{
} // DynamicBlockHybridMatrix(..)


/** END DynamicBlockHybridMatrix **/


template class HybridBaseMatrix<double>;

template class HybridMatrix<double>;
template class HybridMatrix<Mat<2,2,double>>;
template class HybridMatrix<Mat<3,3,double>>;

#ifdef ELASTICITY
template class HybridMatrix<Mat<4,4,double>>;
template class HybridMatrix<Mat<5,5,double>>;
template class HybridMatrix<Mat<6,6,double>>;
#endif // ELASTICITY

template class DynamicBlockHybridMatrix<double>;

} // namespace amg