#include "amg.hpp"
#include "python_ngstd.hpp"
#include "amg_pc.hpp"
#include "amg_matrix.hpp"

namespace amg
{

  template<class TPMAT, class TPMAT_TM>
  shared_ptr<TPMAT> compose_embs ( shared_ptr<BaseMatrix> embA, shared_ptr<BaseMatrix> embB )
  {
    /** merge mats (!! project out dirichlet dofs !!) **/
    auto mA = dynamic_pointer_cast<TPMAT_TM>(embA);
    auto mB = dynamic_pointer_cast<TPMAT_TM>(embB);
    if ( (mA == nullptr) || (mB == nullptr) )
      { throw Exception(" invalid mats!"); }
    int os_sa = 0, os_sb = mA->Height();
    Array<int> perow(mA->Height() + mB->Height()); perow = 0;
    for (auto k : Range(mA->Height()))
      { perow[os_sa + k] = mA->GetRowIndices(k).Size(); }
    for (auto k : Range(mB->Height()))
      { perow[os_sb + k] = mB->GetRowIndices(k).Size(); }
    auto newP = make_shared<TPMAT>(perow, mA->Width());
    for (auto k : Range(mA->Height())) {
      int row = os_sa + k;
      auto ri = newP->GetRowIndices(row); auto ri2 = mA->GetRowIndices(k);
      auto rv = newP->GetRowValues(row); auto rv2 = mA->GetRowValues(k);
      ri = ri2;
      rv = rv2;
    }
    for (auto k : Range(mB->Height())) {
      int row = os_sb + k;
      auto ri = newP->GetRowIndices(row); auto ri2 = mB->GetRowIndices(k);
      auto rv = newP->GetRowValues(row); auto rv2 = mB->GetRowValues(k);
      ri = ri2;
      rv = rv2;
    }
    // cout << " new P mat: " << endl;
    // print_tm_spmat(cout, *newP); cout << endl;
    return newP;
  } // compose_embs


  void ExportHackyStuff (py::module &m)
  {
    m.def("compose_embs", [](shared_ptr<BaseMatrix> embA, shared_ptr<BaseMatrix> embB) -> shared_ptr<BaseMatrix> {
	return compose_embs<SparseMatrix<Mat<1,3,double>>, SparseMatrixTM<Mat<1,3,double>>>(embA, embB);
      });

    m.def("E_A_ET", [](shared_ptr<BaseMatrix> E, shared_ptr<BaseMatrix> A) -> shared_ptr<BaseMatrix> {
	auto spA = dynamic_pointer_cast<SparseMatrixTM<double>>(A);
	auto spE = dynamic_pointer_cast<SparseMatrixTM<Mat<1,3,double>>>(E);

	auto ET = TransposeSPM(*spE);

	auto EAET = RestrictMatrixTM<SparseMatrixTM<double>, SparseMatrixTM<Mat<1,3,double>>>(*ET, *spA, *spE);

	auto spm = make_shared<SparseMatrix<Mat<3,3,double>>>(move(*EAET));
	
	return spm;
	// if ( (spA == nullptr) || (spB = nullptr) )
	//   { throw Exception("spA/B!"); }
	// shared_ptr<BaseSparseMatrix> EAET;
	// Switch<MAX_SYS_DIM> // 0-based
	//   (GetEntryDim(spA.get())-1, [&] (auto BSM) {
	//     constexpr int BS = BSM + 1;
	//   });
      });

    m.def("CheckEI3", [](shared_ptr<Preconditioner> pc, int level, shared_ptr<BitArray> freedofs) {
	auto amg_pc = dynamic_pointer_cast<BaseAMGPC>(pc);
	auto prow3 = [](const auto & ar, std::ostream &os = cout) {
	  for (auto k : Range(ar.Size())) os << "(" << k << "::" << ar(k) << ") ";
	};
	auto chkab = [&](auto fva, auto fvb, string c, shared_ptr<BitArray> freedofs) {
	  cout << c << endl;
	  Vec<3,double> diff;
	  for (auto k : Range(fva.Size())) {
	    if ( (!freedofs) || freedofs->Test(k)) {
	      diff = fva(k) - fvb(k);
	      double nd = L2Norm(diff);
	      if ( nd > 1e-14 )
		{ cout << " DIFF " << k << " norm = " << nd << ", diff = "; prow3(diff); cout << endl; }
	    }
	  }
	};
	auto prtv = [&](auto & avec) {
	  cout << " STAT: " << avec.GetParallelStatus() << endl;
	  auto fv = avec.template FV<Vec<6, double>>();
	  for (auto k : Range(fv.Size()))
	    { cout << k << ": "; prow3(fv(k)); cout << endl; }
	  cout << endl;
	};
	auto prtv0 = [&](auto & avec) {
	  cout << " STAT: " << avec.GetParallelStatus() << endl;
	  auto fv = avec.template FV<Vec<3, double>>();
	  for (auto k : Range(fv.Size()))
	    { cout << k << ": "; prow3(fv(k)); cout << endl; }
	  cout << endl;
	};
	auto amg_mat = amg_pc->GetAMGMatrix();
	auto map = amg_mat->GetMap();
	int nlevs = map->GetNLevels();
	int maxlev = min2(nlevs-1, level);
	cout << " lev, nlevs, maxlev " << level << " " << nlevs << " " << maxlev << endl;
	for (int comp = 0; comp < 3; comp++) {
	  auto cvec = map->CreateVector(maxlev);
	  auto fvec = map->CreateVector(0);
	  auto tvec = map->CreateVector(0);
	  {
	    auto fv = tvec->FV<Vec<3, double>>();
	    fv = 0; tvec->SetParallelStatus(CUMULATED);
	    for (auto k : Range(fv.Size())) {
	      fv(k) = 0;
	      fv(k)(comp) = 1;
	    }
	  }
	  /** On ranks that have the coarsest level, set the vector **/
	  if ( (cvec != nullptr) && (maxlev == nlevs-1) ) {
	    auto fv = cvec->FV<Vec<6, double>>();
	    fv = 0; cvec->SetParallelStatus(CUMULATED);
	    for (auto k : Range(fv.Size())) {
	      fv(k) = 0;
	      fv(k)(comp) = 1;
	    }
	    // cout << " vec @ level " << maxlev << endl;
	    // prtv(*cvec);
	  }
	  for (int l = maxlev-1; l >= 0; l--) {
	    unique_ptr<BaseVector> mvec = map->CreateVector(l);
	    BaseVector & uvec = (l == 0) ? *fvec : *mvec;
	    map->TransferC2F(l, &uvec, cvec.get());
	    cout << " vec @ level " << l << endl;
	    if (l>0) {
	      auto uvec2 = uvec.CreateVector();
	      uvec2.SetParallelStatus(uvec.GetParallelStatus());
	      uvec2.FVDouble() = uvec.FVDouble();
	      uvec.Distribute();
	      uvec.Cumulate();
	      chkab(uvec.FV<Vec<6, double>>(), uvec2.FV<Vec<6, double>>(), string("diff vec @ level ") + to_string(l), nullptr);
	      // prtv(uvec);
	    } else {
	      prtv0(uvec);
	    }
	    if (l > 0)
	      { cvec = move(mvec); }
	  }
	  {
	    auto fva = fvec->FV<Vec<3, double>>();
	    auto fvb = tvec->FV<Vec<3, double>>();
	    chkab(fva, fvb, "FLEV diff", freedofs);
	  }
	}
      });
    
  }

} // namespace amg
