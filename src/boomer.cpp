#ifdef USE_BOOMER

#include "amg.hpp"

#include "HYPRE_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// internals!
#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

#include <python_ngstd.hpp>

namespace amg
{


  INLINE void check_error (int herr, string tag = string("unnamed"))
  {
    if (herr == 0) return;
    static char* buffer = (char*) malloc(500*sizeof(char));
    HYPRE_DescribeError(herr, buffer);
    cout << "HYPRE-error for call " << tag << ": " << string(buffer) << endl;
    HYPRE_ClearAllErrors();
  };

  class HypreMatrix
  {
  protected:
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    Array<int> global_nums;
    int ilower, iupper, BS;
    shared_ptr<BitArray> freedofs;
    shared_ptr<ParallelDofs> pardofs;
  public:
    HypreMatrix (int _BS, shared_ptr<ParallelDofs> _pardofs, shared_ptr<BitArray> _freedofs);
    HYPRE_ParCSRMatrix GetParCSRMat() const { return parcsr_A; }
    int GetILow () const { return ilower; }
    int GetIUpp () const { return iupper; }
    int GetBS () const { return BS; }
    FlatArray<int> GetGlobNums () const { return global_nums; }
  };

  template<class TM>
  class HypreMatrixTM : public HypreMatrix
  {
  public:
    HypreMatrixTM (shared_ptr<SparseMatrixTM<TM>> spmat, shared_ptr<ParallelDofs> pardofs,
		   shared_ptr<BitArray> freedofs);
  };

  class BoomerEAMG : public Preconditioner
  {
  protected:
    shared_ptr<BilinearForm> bfa;
    HypreMatrix* hypre_mat;
    HYPRE_Solver precond;
    shared_ptr<BitArray> freedofs;
    shared_ptr<ParallelDofs> pardofs;
    mutable Array<int> iluis, allis;
    mutable Array<double> zerovs, allvs;
    HYPRE_IJVector ij_rhs, ij_sol;
  public:
    BoomerEAMG (shared_ptr<BilinearForm> _bfa, Array<shared_ptr<BaseVector>> & kvecs);
    ~BoomerEAMG () { if(hypre_mat) delete hypre_mat; }
    virtual void FinalizeLevel (const ngla::BaseMatrix * mat = NULL) override {};
    virtual void Update () override {}
    virtual void Mult (const BaseVector & f, BaseVector & u) const override;
    virtual int VHeight() const override { return pardofs->GetNDofLocal();}
    virtual int VWidth() const override { return pardofs->GetNDofLocal();}
    virtual const BaseMatrix & GetAMatrix() const override { return bfa->GetMatrix(); }
  };


  HypreMatrix :: HypreMatrix (int _BS, shared_ptr<ParallelDofs> _pardofs, shared_ptr<BitArray> _freedofs)
    : BS(_BS), freedofs(_freedofs), pardofs(_pardofs)
  {
    auto ndof = pardofs->GetNDofLocal();
    auto comm = pardofs->GetCommunicator();
    
    // find global dof enumeration 
    global_nums.SetSize(ndof); global_nums = -1;

    int num_master_dofs = 0;
    for (auto i : Range(ndof))
      if (pardofs -> IsMasterDof (i) && (!freedofs || freedofs -> Test(i)))
	global_nums[i] = num_master_dofs++;
    
    Array<int> first_master_dof(comm.Size());
    MPI_Allgather (&num_master_dofs, 1, MPI_INT, 
		   &first_master_dof[0], 1, MPI_INT, 
		   comm);
    
    int num_glob_dofs = 0;
    for (auto i : Range(comm.Size())) {
      int cur = first_master_dof[i];
      first_master_dof[i] = num_glob_dofs;
      num_glob_dofs += cur;
    }
    first_master_dof.Append(num_glob_dofs);

    auto rank = comm.Rank();
    for (int i = 0; i < ndof; i++)
      if (global_nums[i] != -1)
	global_nums[i] += first_master_dof[rank];
    
    ScatterDofData (global_nums, pardofs);
    // range of my master dofs ...
    ilower = BS * first_master_dof[rank];
    iupper = BS * first_master_dof[rank+1] - 1;

    cout << "num glob pds: " << pardofs->GetNDofGlobal() << endl;
    cout << "num glob dofs: " << num_glob_dofs << endl;
    cout << "ilow/up: " << ilower << " " << iupper << endl;
    // cout << "glob nums: " << endl; prow2(global_nums); cout << endl;
  }

  template<class TM>
  HypreMatrixTM<TM> :: HypreMatrixTM (shared_ptr<SparseMatrixTM<TM>> spmat, shared_ptr<ParallelDofs> pardofs,
				      shared_ptr<BitArray> freedofs)
    : HypreMatrix(mat_traits<TM>::HEIGHT, pardofs, freedofs)
  {
    static Timer t("HypreMatrixTM::Constructor"); RegionTimer rt(t);
    constexpr int BS = mat_traits<TM>::HEIGHT;
    auto comm = pardofs->GetCommunicator();
    
    
    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetPrintLevel (A, 1);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    Array<int> cols_global;
    Array<double> vals_global;
    const auto mat(*spmat);
    for ( auto row : Range(mat.Height())) {
      if (global_nums[row] == -1) continue;
      // cout << "row " << row << " glob num: " << global_nums[row] << endl;
      auto ri = mat.GetRowIndices(row);
      auto rv = mat.GetRowValues(row);
      cols_global.SetSize(0);
      for (auto j : Range(ri.Size())) {
	auto globnum = global_nums[ri[j]];
	if (globnum != -1) {
	  auto base = BS * globnum;
	  for (int bj : Range(BS)) {
	    cols_global.Append(base+bj);
	  }
	}
      }
      for (int bi : Range(BS)) {
	vals_global.SetSize(0);
	for (auto j : Range(ri.Size())) {
	  auto globnum = global_nums[ri[j]];
	  if (globnum != -1) {
	    if constexpr(BS == 1) {
		vals_global.Append(rv[j]);
	      }
	    else {
	      auto& v = rv[j];
	      for (int bj : Range(BS)) {
		vals_global.Append(v(bi,bj));
	      }
	    }
	  }
	}
	int size = cols_global.Size();
	int at_row = BS * global_nums[row] + bi;
	// cout << "add to row " << at_row << ":    "; prow(cols_global); cout << endl;
	HYPRE_IJMatrixAddToValues(A, 1, &size, &at_row, &cols_global[0], &vals_global[0]);
      }
    }

    {
      static Timer t("HypreMatrixTM::Assemble"); RegionTimer rt(t);
      check_error(HYPRE_IJMatrixAssemble(A), "ass mat");
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    }

    HYPRE_IJMatrixPrint(A, "A.out");
    
  }

  
  BoomerEAMG :: BoomerEAMG (shared_ptr<BilinearForm> _bfa, Array<shared_ptr<BaseVector>>& kvecs)
    : Preconditioner(_bfa, Flags()), bfa(_bfa)
  {
    int herr;

    auto fes = bfa->GetFESpace();
    freedofs = fes->GetFreeDofs();
    auto parmat = dynamic_pointer_cast<ParallelMatrix>(bfa->GetMatrixPtr());
    pardofs = parmat->GetParallelDofs();
    auto comm = pardofs->GetCommunicator();
    auto ndof = pardofs->GetNDofLocal();
    auto seqmat = dynamic_pointer_cast<BaseSparseMatrix>(parmat->GetMatrix());

    {
      static Timer t("Boomer::Convert Matrix");
      if (auto spm = dynamic_pointer_cast<SparseMatrixTM<double>>(seqmat)) { hypre_mat = new HypreMatrixTM<double>(spm, pardofs, freedofs); }
      else if (auto spm = dynamic_pointer_cast<SparseMatrixTM<Mat<2,2,double>>>(seqmat)) { hypre_mat = new HypreMatrixTM<Mat<2,2,double>>(spm, pardofs, freedofs); }
      else if (auto spm = dynamic_pointer_cast<SparseMatrixTM<Mat<3,3,double>>>(seqmat)) { hypre_mat = new HypreMatrixTM<Mat<3,3,double>>(spm, pardofs, freedofs); }
      else if (auto spm = dynamic_pointer_cast<SparseMatrixTM<Mat<4,4,double>>>(seqmat)) { hypre_mat = new HypreMatrixTM<Mat<4,4,double>>(spm, pardofs, freedofs); }
      else if (auto spm = dynamic_pointer_cast<SparseMatrixTM<Mat<5,5,double>>>(seqmat)) { hypre_mat = new HypreMatrixTM<Mat<5,5,double>>(spm, pardofs, freedofs); }
      else if (auto spm = dynamic_pointer_cast<SparseMatrixTM<Mat<6,6,double>>>(seqmat)) { hypre_mat = new HypreMatrixTM<Mat<6,6,double>>(spm, pardofs, freedofs); }
      else { throw Exception("Could not convert matrix!"); }
    }

    int ilower = hypre_mat->GetILow();
    int iupper = hypre_mat->GetIUpp();
    HYPRE_IJVectorCreate(comm, ilower, iupper, &ij_rhs);
    HYPRE_IJVectorSetObjectType(ij_rhs, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(comm, ilower, iupper, &ij_sol);
    HYPRE_IJVectorSetObjectType(ij_sol, HYPRE_PARCSR);

    zerovs.SetSize(0); iluis.SetSize(0);
    if (iupper >= ilower) {
      zerovs.SetSize(iupper-ilower+1); zerovs = 0;
      iluis.SetSize(iupper-ilower+1);
      for (auto k : Range(iluis.Size())) {
	iluis[k] = ilower + k;
      }
    }

    auto glob_nums = hypre_mat->GetGlobNums();
    int BS = hypre_mat->GetBS();
    allvs.SetSize(BS*glob_nums.Size()); allvs = -1;
    allis.SetSize(BS*glob_nums.Size()); allis.SetSize0();
    for (auto k : Range(glob_nums.Size())) {
      auto num = glob_nums[k];
      if (num != -1) {
	auto BSnum = BS*num;
	for (auto l : Range(BS)) {
	  allis.Append(BSnum+l);
	}
      }
    }

    
    int num_kvecs = kvecs.Size();
    Array<HYPRE_ParVector> hkv(num_kvecs);
    {
      static Timer t("Boomer::Kernel-Vecs");
      for (auto k : Range(num_kvecs)) {
	int cnt = 0;
	HYPRE_IJVector ijvec;
	HYPRE_IJVectorCreate(comm, ilower, iupper, &ijvec);
	HYPRE_IJVectorSetPrintLevel (ijvec, 1);
	HYPRE_IJVectorSetObjectType(ijvec, HYPRE_PARCSR);
	HYPRE_IJVectorInitialize(ijvec);
	if (iluis.Size()) HYPRE_IJVectorSetValues(ijvec, iluis.Size(), &iluis[0], &zerovs[0]);
	kvecs[k]->Distribute();
	auto fv = kvecs[k]->FVDouble();
	for (auto k : Range(glob_nums.Size())) {
	  auto num = glob_nums[k];
	  if (num != -1) {
	    auto nbase = BS*k;
	    for (auto l : Range(BS)) {
	      allvs[cnt++] = fv[nbase+l];
	    }
	  }
	}
	// cout << "add vals: " << endl;
	// cout << endl; prow2(allvs); cout << endl;
	// cout << endl << endl;
	if (allis.Size()) HYPRE_IJVectorAddToValues(ijvec, allis.Size(), &allis[0], &allvs[0]);
	check_error(HYPRE_IJVectorAssemble(ijvec), "ass vec");
	HYPRE_ParVector par_vec; HYPRE_IJVectorGetObject(ijvec, (void **) &par_vec);
	hkv[k] = par_vec;
	// string name = string("kvec")+to_string(k)+string(".out");
	// HYPRE_IJVectorPrint(ijvec, name.c_str());
      }
    }    

    check_error(HYPRE_BoomerAMGCreate(&precond));
    check_error(HYPRE_BoomerAMGSetPrintLevel(precond, 1), "set print level");
    
    cout << "BS: " << BS << endl;
    check_error(HYPRE_BoomerAMGSetNumFunctions(precond, BS), "set num funcs");
    // HYPRE_BoomerAMGSetDofFunc(precond, &dof_func[0]);

    // Coarsening
    // 0 CLJP // 1 RS(no bnd) // 3 RS + bnd // 6 Falgout(default) // 7 CLJP (debug) // 8 PMIS // 9 PMIS (debug)
    // 10 HMIS // 11 one-pass RS (no bnd) // 21 CGC // 22 CGC-E
    // block i think: 6 7 8 9 10 21
    HYPRE_BoomerAMGSetCoarsenType(precond, 10);
    check_error(HYPRE_BoomerAMGSetNodal (precond, 2), "set nodal"); // 1 frobenius // 2 sum abs val // 3 largest el // 4 row-sum // 6 sum vals
    // HYPRE_BoomerAMGSetAggNumLevels(precond,1);
    
    // Interpolation
    // 0 classic // 1 LS // 2 class for hyp // 3 direct // 4 multipass // 5 multipass + sep
    // 6 ext+i // 7 ext+i (if no common) // 8 standard // 9 standard+sep // 10 classic block // 11 classic block+diag
    // 12 FF // 13 FF1 // 14 ext
    // block i think: 11 22 23 20 21 24 (10)
    // works: 0, 6
    // broken: 10, 21, 22
    // HYPRE_BoomerAMGSetInterpType(precond, 10);
    HYPRE_BoomerAMGSetPMaxElmts(precond, 36);
    // cout << "num k-vecs: " << num_kvecs << endl;
    check_error(HYPRE_BoomerAMGSetInterpVectors(precond, num_kvecs, &hkv[0]), "set interp-vecs");
    check_error(HYPRE_BoomerAMGSetInterpVecVariant(precond, 2), "set vec-interp type"); // 1 GM-v1 // 2 GM-v2 (better than 1) // 3 LN
    HYPRE_BoomerAMGSetInterpVecQMax (precond, 36);
    HYPRE_BoomerAMGSetInterpVecAbsQTrunc (precond, 0.05); // ???
    // HYPRE_BoomerAMGSetTruncFactor(precond, 0.05);
    
    // Misc.
    HYPRE_BoomerAMGSetStrongThreshold(precond, 0.15);
    HYPRE_BoomerAMGSetMaxRowSum (precond, 1.0);

    // Levels
    HYPRE_BoomerAMGSetMaxLevels(precond, 20);  /* maximum number of levels */

    // Cycle & Smoothing
    // 3 hGS fw, 4 hGS bw, 6 sym hGS , 8 sym hGS l1, 9 GAUSS ELIM, 16 cheby

    // check_error(HYPRE_BoomerAMGSetRelaxType(precond, 16), "set relax type");
    // HYPRE_BoomerAMGSetChebyOrder(precond, 1);
      
    check_error(HYPRE_BoomerAMGSetRelaxType(precond, 6), "set relax type");
    // check_error(HYPRE_BoomerAMGSetOuterWt(precond, -10), "swt outer wt");
    // check_error(HYPRE_BoomerAMGSetOuterWt(precond, 0.5), "swt outer wt");
    

    // check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 6, 1), "set relax type");
    // check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 6, 2), "set relax type");
    // check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 9, 3), "set relax type");

    check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 6, 1), "set relax type");
    check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 6, 2), "set relax type");
    check_error(HYPRE_BoomerAMGSetCycleRelaxType(precond, 9, 3), "set relax type");

    
    // HYPRE_BoomerAMGSetNumSweeps(precond, 1);   /* Sweeeps on each level */
    HYPRE_BoomerAMGSetTol(precond, 0.0);      /* conv. tolerance */
    HYPRE_BoomerAMGSetMinIter(precond,1);
    HYPRE_BoomerAMGSetMaxIter(precond,1);
    // HYPRE_BoomerAMGSetMinCoarseSize(precond,100);
    // HYPRE_BoomerAMGSetMaxCoarseSize(precond,100);
    
    {
      comm.Barrier();
      static Timer t("Boomer::Setup"); RegionTimer rt(t);
      HYPRE_ParVector dummy1, dummy2;
      check_error(HYPRE_BoomerAMGSetup (precond, hypre_mat->GetParCSRMat(), dummy1, dummy2), "Boomer setup");
      comm.Barrier();
    }

    hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) precond;
    cout << "coarse relax typ: " << hypre_ParAMGDataUserCoarseRelaxType(amg_data) << endl;
    
  }


  void BoomerEAMG :: Mult (const BaseVector & f, BaseVector & u) const
  {
    f.Distribute();
    u.SetParallelStatus(DISTRIBUTED);

    auto fvf = f.FVDouble();
    auto fvsol = u.FVDouble();

    int BS = hypre_mat->GetBS();
    int ilower = hypre_mat->GetILow();
    int iupper = hypre_mat->GetIUpp();
    auto glob_nums = hypre_mat->GetGlobNums();

    HYPRE_IJVectorInitialize(ij_sol);
    HYPRE_IJVectorInitialize(ij_rhs);
    if (iupper >= ilower) {
      HYPRE_IJVectorSetValues(ij_sol, iluis.Size(), &iluis[0], &zerovs[0]);
      HYPRE_IJVectorSetValues(ij_rhs, iluis.Size(), &iluis[0], &zerovs[0]);
    }
    check_error(HYPRE_IJVectorAssemble(ij_sol), "ass sol");
    HYPRE_ParVector par_sol; HYPRE_IJVectorGetObject(ij_sol, (void **) &par_sol);

    int cnt = 0;
    for (auto k : Range(glob_nums.Size())) {
      auto num = glob_nums[k];
      if (num != -1) {
	auto BSnum = BS*num;
	auto base = BS*k;
	for (auto l : Range(BS)) {
	  allvs[cnt++] = fvf[base+l];
	}
      }
    }
    if (allis.Size()) HYPRE_IJVectorAddToValues(ij_rhs, allis.Size(), &allis[0], &allvs[0]);
    check_error(HYPRE_IJVectorAssemble(ij_rhs), "ass_rhs");
    HYPRE_ParVector par_rhs; HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);


    // HYPRE_IJVectorPrint(ij_sol, "inisol.out");
    // {
    //   static int cnt_rhs = 0;
    //   string name = string("rhs")+to_string(cnt_rhs)+string(".out");
    //   HYPRE_IJVectorPrint(ij_rhs, name.c_str());
    //   cnt_rhs++;
    // }

    {
      static Timer t("Boomer::Solve"); RegionTimer rt(t);
      check_error(HYPRE_BoomerAMGSolve(precond, hypre_mat->GetParCSRMat(), par_rhs, par_sol), "solve");
    }

    // {
    //   static int cnt_sol = 0;
    //   string name = string("sol")+to_string(cnt_sol)+string(".out");
    //   HYPRE_IJVectorPrint(ij_sol, name.c_str());
    //   cnt_sol++;
    // }
    
    if (iluis.Size()) HYPRE_IJVectorGetValues (ij_sol, iluis.Size(), &iluis[0], &allvs[0]);
    
    cnt = 0;
    for (auto k : Range(pardofs->GetNDofLocal())) {
      auto base = k * BS;
      if (pardofs->IsMasterDof(k) && glob_nums[k] != -1) 
	for (auto l : Range(BS)) fvsol(base + l) = allvs[cnt++];
      else
	for (auto l : Range(BS)) fvsol(base + l) = 0;
    }
  }

  
  void ExportBoomer (py::module & m)
  {
    py::class_<BoomerEAMG, shared_ptr<BoomerEAMG>, BaseMatrix>
      (m, "BoomerEAMG", "")
      .def(py::init<>
	   ( [&] (shared_ptr<BilinearForm> blf, py::list py_kvecs) {
	     Array<shared_ptr<BaseVector>> kvecs = makeCArraySharedPtr<shared_ptr<BaseVector>>(py_kvecs);
	     return new BoomerEAMG(blf, kvecs);
	   }), py::arg("blf") = nullptr,
	   py::arg("kvecs") = py::list())
      .def ("Test", [](BoomerEAMG &pre) { pre.Test();} )
      ;
  }
  
}

#endif
