
#include "amg.hpp"
#include "amg_matrix.hpp"

namespace amg
{

  AMGMatrix :: AMGMatrix (shared_ptr<DOFMap> _map, FlatArray<shared_ptr<BaseSmoother>> _smoothers)
    : map(_map), smoothers(_smoothers)
  {
    /** local & global nr of levels **/
    n_levels = map->GetNLevels();
    n_levels_glob = map->GetParDofs(0)->GetCommunicator().AllReduce(n_levels, MPI_MAX);

    /** work vectors **/
    res_level.SetSize(map->GetNLevels());
    x_level.SetSize(map->GetNLevels());
    rhs_level.SetSize(map->GetNLevels());
    for (auto l : Range(map->GetNLevels())) {
      x_level[l] = map->CreateVector(l);
      rhs_level[l] = map->CreateVector(l);
      res_level[l] = map->CreateVector(l);
    }

    /** is this proc dropped by redistribution at some point? **/
    drops_out = (x_level.Last() == nullptr);
  } // AMGMatrix::AMGMatrix


  void AMGMatrix :: SetCoarseInv (shared_ptr<BaseMatrix> _crs_inv)
  { crs_inv = _crs_inv; has_crs_inv = true; }


  void AMGMatrix :: SmoothW (BaseVector & x, const BaseVector & b) const
  {
    b.Distribute();

    static Timer tt ("AMGMatrix::SmoothW"); RegionTimer rt(tt);

    // static Timer t4("coarse inv");
    // static Timer t5("drop, wait ");
    // static Timer t_l0 ("level 0");    
    // static Timer t_l1 ("level 1");    
    // static Timer t_l2 ("level 2");    
    // static Timer t_l3 ("level 3");    
    // static Timer t_lr ("rest");    
    // static Timer t_lc ("coarsest");    
    // static Array<Timer*> lev_timers({ &t_l0, &t_l1, &t_l2, &t_l3, &t_lr });
    // auto ltimer = [](int level) -> Timer& { return *lev_timers[min2(level,4)]; };

    auto comm = map->GetParDofs(0)->GetCommunicator();

    function<void(int)> smit = [&](int level) {
      if ( level == 0 ) {
	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());

	smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);
	rl.Distribute();

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
      }
      if ( level + 1 < n_levels ) {
	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());


	smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);
	rl.Distribute();

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(level + 1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothK(stk, xl, bl, rl, false, false, false);
	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(level + 1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
      }
      else {
	if (!drops_out) {
	  if (has_crs_inv) { /** exact solve on coarsest level **/
	    if (crs_inv == nullptr)
	      { x_level[n_levels-1]->FVDouble() = 0; }
	    else
	      { crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]); }
	    x_level[n_levels-1]->Cumulate();
	  }
	  else
	    { *x_level[n_levels-1] = 0; x_level[n_levels-1]->Cumulate(); }
	}
      }
    };

    smit(0);
  } // AMGMatrix::SmoothW


  void AMGMatrix :: SmoothBS (BaseVector & x, const BaseVector & b) const
  {
    b.Distribute();

    static Timer tt ("AMGMatrix::SmoothBS"); RegionTimer rt(tt);

    static Timer t4("coarse inv");
    static Timer t5("drop, wait ");
    static Timer t_l0 ("level 0");    
    static Timer t_l1 ("level 1");    
    static Timer t_l2 ("level 2");    
    static Timer t_l3 ("level 3");    
    static Timer t_lr ("rest");    
    static Timer t_lc ("coarsest");    
    static Array<Timer*> lev_timers({ &t_l0, &t_l1, &t_l2, &t_l3, &t_lr });
    auto ltimer = [](int level) -> Timer& { return *lev_timers[min2(level,4)]; };

    auto comm = map->GetParDofs(0)->GetCommunicator();

    function<void(int, bool)> smit = [&](int level, bool bs) {
      auto & t = ltimer(level);
      if ( level == 0 ) {
	RegionTimer rt(t);
	// t.Start();
      	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
      	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
      	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
      	xl.SetParallelStatus(CUMULATED);
      	rl.FVDouble() = bl.FVDouble();
      	rl.SetParallelStatus(bl.GetParallelStatus());

      	smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);
      	rl.Distribute();

      	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	// t.Stop();
      	smit(1, true);
	// t.Start();
      	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

      	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
	// t.Stop();
      }
      if ( level + 1 < n_levels ) {
	RegionTimer rt(t);
	// t.Start();
	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());

	if (bs) { // BS: MG as smoother
	  smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);
	  rl.Distribute();
	  map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	  // t.Stop();
	  smit(level + 1, false);
	  // t.Start();
	  map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
	  smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, true, false);
	}
	else
	  { smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true); }

	// recursive MG
	rl.Distribute();
	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	// t.Stop();
	smit(level + 1, true);
	// t.Start();
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	if (bs) {
	  smoothers[level]->SmoothK(stk, xl, bl, rl, false, true, false);
	  map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	  // t.Stop();
	  smit(level + 1, false);
	  // t.Start();
	  map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
	  smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
	}
	else
	  { smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false); }
	// t.Stop();
      }
      else {
	RegionTimer rt(t4);
	if (!drops_out) {
	  if (has_crs_inv) { /** exact solve on coarsest level **/
	    if (crs_inv == nullptr)
	      { x_level[n_levels-1]->FVDouble() = 0; }
	    else
	      { crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]); }
	    x_level[n_levels-1]->Cumulate();
	  }
	  else
	    { *x_level[n_levels-1] = 0; x_level[n_levels-1]->Cumulate(); }
	}
      }
    };

    smit(0, true);
  } // AMGMatrix::SmoothBS


  void AMGMatrix :: SmoothV (BaseVector & x, const BaseVector & b) const
  {
    b.Distribute();

    static Timer tt ("AMGMatrix::Mult"); RegionTimer rt(tt);

    static Timer t4("coarse inv");
    static Timer t5("drop, wait ");
    static Timer t_l0 ("level 0");    
    static Timer t_l1 ("level 1");    
    static Timer t_l2 ("level 2");    
    static Timer t_l3 ("level 3");    
    static Timer t_lr ("rest");    
    static Timer t_lc ("coarsest");    
    static Array<Timer*> lev_timers({ &t_l0, &t_l1, &t_l2, &t_l3, &t_lr });

    auto ltimer = [](int level) -> Timer& { return *lev_timers[min2(level,4)]; };

    auto comm = map->GetParDofs(0)->GetCommunicator();

    /** pre-smooth **/
    for (auto level : Range(n_levels-1))
      {
	RegionTimer rtl(ltimer(level));

	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

	/** Start with initial guess 0 and keep residum up to date **/

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);

	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());

	// smoothers[level]->Smooth(xl, bl, rl, true, true, true);
	smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);

	// cout << " x 1  level " << level << ": " << endl << xl.FVDouble() << endl;

	rl.Distribute();

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());

      }

    /** coarsest level **/    
    if (!drops_out) {
      RegionTimer rtl(t_lc);
      if (has_crs_inv) { /** exact solve on coarsest level **/
	t4.Start();

	// if (comm.Rank() == 1) {
	// cout << " coarse rhs: " << endl;
	// cout << (*rhs_level[n_levels-1]).FVDouble() << endl;
	// }
	
	//{ x_level[n_levels-1]->FVDouble() = 0; }
	if (crs_inv == nullptr)
	  { x_level[n_levels-1]->FVDouble() = 0; }
	else
	  { crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]); }

	x_level[n_levels-1]->Cumulate();

	// if (comm.Rank() == 1) {
	// cout << " coarse sol: " << endl;
	// cout << (*x_level[n_levels-1]).FVDouble() << endl;
	// }

	t4.Stop();
      }
      else {
	// throw Exception("Coarsest level smooth kind of not implemented ...");
	*x_level[n_levels-1] = 0;
	x_level[n_levels-1]->Cumulate();
      }
    }

    /** post-smooth **/
    for (int level = n_levels-2; level>=0; level--)
      {
	RegionTimer rtl(ltimer(level));

	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

	/** apply coarse grid correction - residuum is no longer updated **/
	if (x_level[level+1] == nullptr)
	  { t5.Start(); }

	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	// cout << " x 2  level " << level << ": " << endl << xl.FVDouble() << endl;

	if (x_level[level+1] == nullptr)
	  { t5.Stop(); }

	// smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);
	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
	
	// cout << " x 3  level " << level << ": " << endl << xl.FVDouble() << endl;
      }

  } // AMGMatrix::SmoothV


  void AMGMatrix :: SmoothVFromLevel (int startlevel, BaseVector  &x, const BaseVector &b, BaseVector  &res,
				     bool res_updated, bool update_res, bool x_zero) const
  {
    static Timer t ("SmoothFromLevel"); RegionTimer rt(t);
    // cout << "SVL " << startlevel << " of " << n_levels << endl;
    // cout << " x b r s " << x.Size() << " " << b.Size() << " " << res.Size() << endl;

    // cout << " SVL (F) " << startlevel << " " << startlevel << endl;
    smoothers[startlevel]->SmoothK(stk, x, b, res, res_updated, true, x_zero);
    res.Distribute();
    // cout << " SVL (F) " << startlevel << " " << startlevel << endl;
    map->TransferF2C(startlevel, &res, rhs_level[startlevel+1].get());
    // cout << " SVL (F) " << startlevel << " " << startlevel << endl;

    if (startlevel + 2 < n_levels)
      for (int level = startlevel + 1; level+1 < n_levels; level++ ) {
	BaseVector &xl ( *x_level[level] ), &rl (*res_level[level]);
	const BaseVector &bl ( *rhs_level[level] );
	// cout << " xl bl l" << level << " lens " << xl.Size() << " " << bl.Size() << endl;
	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());
	// cout << " SVL (F) " << startlevel << " " << level << endl;
	smoothers[level]->SmoothK(stk, xl, bl, rl, true, true, true);
	rl.Distribute();
	// cout << " SVL (F) " << startlevel << " " << level << endl;
	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
      }

    if (!drops_out) {
      if (has_crs_inv) { /** exact solve on coarsest level **/
	if (crs_inv == nullptr)
	  { x_level[n_levels-1]->FVDouble() = 0; }
	else
	  { crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]); }
	x_level[n_levels-1]->Cumulate();
      }
      else {
	*x_level[n_levels-1] = 0;
	x_level[n_levels-1]->Cumulate();
      }
    }

    if (startlevel + 2 < n_levels)
      for (int level = n_levels - 2; level > startlevel; level--) {
	BaseVector &xl ( *x_level[level] ), &rl (*res_level[level]);
	const BaseVector &bl ( *rhs_level[level] );
	// cout << " SVL (B) " << startlevel << " " << level << endl;
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
	// cout << " SVL (B) " << startlevel << " " << level << endl;
	smoothers[level]->SmoothBackK(stk, xl, bl, rl, false, false, false);
      }
    
    // cout << " SVL (B) " << startlevel << " " << startlevel << endl;
    map->AddC2F(startlevel, 1.0, &x, x_level[startlevel+1].get());
    // cout << " SVL (B) " << startlevel << " " << startlevel << endl;
    smoothers[startlevel]->SmoothBackK(stk, x, b, res, false, update_res, false);

    // cout << "SVL " << startlevel << " done" << endl;

  } // AMGMatrix::SmoothVFromLevel


  void AMGMatrix :: Mult (const BaseVector & b, BaseVector & x) const
  { Smooth(x, b); }


  void AMGMatrix :: MultTrans (const BaseVector & b, BaseVector & x) const
  { Mult(b, x); }


  void AMGMatrix :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    this->Smooth(*x_level[0], b);
    x += s * *x_level[0];
  }


  void AMGMatrix :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
  { MultAdd(s, b, x); }


  /** used for visualizing base functions **/


  size_t AMGMatrix :: GetNLevels (int rank) const
  {
    if (rank == 0) return n_levels_glob;
    auto comm = map->GetParDofs(0)->GetCommunicator();
    return comm.AllReduce((comm.Rank() == rank) ? (1 + smoothers.Size()) : 0, MPI_SUM);
  }


  void AMGMatrix :: CINV (shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const
  {
    auto tmp = x_level[0];
    x_level[0] = x;
    auto tmp2 = rhs_level[0];
    rhs_level[0] = b;
    rhs_level[0]->Distribute();
    for (auto level : Range(n_levels-1))
      map->TransferF2C(level, rhs_level[level].get(), rhs_level[level+1].get());
    if (!drops_out) {
      crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]);
      cout << endl << endl << "CINV --  crs rhs/sol (len = " << rhs_level[n_levels-1]->FVDouble().Size()<< "): " << endl;
      for (auto k : Range(rhs_level[n_levels-1]->FVDouble().Size())) {
	cout << k << ": " << rhs_level[n_levels-1]->FVDouble()[k] << " " << x_level[n_levels-1]->FVDouble()[k] << endl;
      }
      cout << endl;
    }
    for (int level = n_levels-2; level>=0; level--)
      map->TransferC2F(level, x_level[level].get(), x_level[level+1].get());
    x_level[0] = tmp;
    rhs_level[0] = tmp2;

    cout << endl << endl << "CINV --  finest level rhs/sol (len = " << rhs_level[0]->FVDouble().Size()<< "): " << endl;
    for (auto k : Range(rhs_level[0]->FVDouble().Size())) {
      cout << k << ": " << rhs_level[0]->FVDouble()[k] << " " << x_level[0]->FVDouble()[k] << endl;
    }
    cout << endl;
  } // AMGMatrix::CINV


  // void AMGMatrix :: GetEV(size_t level, int arank, size_t k_num, BaseVector & vec) const
  // {
  //   auto comm = map->GetParDofs(0)->GetCommunicator();
  //   auto rank = comm.Rank();
  //   auto np = comm.Size();
  //   if (np>1) throw Exception("GetEV doesnt work with NP>1.");
  //   cout << "GET EVEC " << level << " " << rank << " " << k_num << endl;
  //   int input_notok = 1;
  //   if ( (arank<0) || (arank>np) )
  //     cout << "Invalid rank " << arank << ", only have " << np << endl;
  //   else if ( (rank==arank) && (level>=x_level.Size() || x_level[level]==nullptr) )
  //     cout << "rank " << rank << "does not have that many levels!!" << endl;
  //   else
  //     input_notok = 0;
  //   input_notok = comm.AllReduce(input_notok, MPI_MAX);
  //   for (auto xl : x_level)
  //     if (xl!=nullptr) xl->FVDouble() = 0.0;
  //   cout << "GARBAGE IN? " << input_notok << endl;
  //   vec.FVDouble() = 0.0;
  //   vec.SetParallelStatus(CUMULATED);
  //   if (input_notok) return;
  //   cout << "STILL HERE!!" << endl;
  //   size_t max_x_access = x_level.Size() - (x_level.Last()==nullptr ? 2 : 1);
  //   size_t my_max_level = min2(max_x_access, (size_t)level);
  //   /** set coarsest vector **/
  //   x_level[my_max_level]->FVDouble() = 0.0;
  //   if (level==my_max_level && k_num<x_level[my_max_level]->FVDouble().Size()) {
  //     auto mat = dynamic_pointer_cast<const SparseMatrix<double>>(mats[level]);
  //     size_t h = mat->Height();
  //     if (h > 3000) {
  // 	cout << "mat size " << h << ", too large, return 0 vec!" << endl;
  // 	return;
  //     }
  //     Matrix<double> full(h,h);
  //     if (h) full = 0.0;
  //     for (auto k : Range(h))
  // 	for (auto j : Range(h)) {
  // 	  full(k,j) = (*mat)(k,j);
  // 	}
  //     Matrix<double> evecs(h,h);
  //     Vector<double> evals(h);
  //     LapackEigenValuesSymmetric(full, evals, evecs);
  //     // cout << "full mat: " << endl << full << endl;
  //     cout << "evals: " << endl;prow(evals); cout << endl;
  //     *testout << "evals: " << endl;
  //     for (auto k : Range(h)) *testout << evals[k] << endl;
      
  //     for (auto k : Range(h))
  // 	{ x_level[my_max_level]->FVDouble()[k] = evecs(k_num, k); }

  //     // const auto & fv = x_level[my_max_level]->FVDouble();
  //     // auto nv = h/3;
  //     // if (k_num==1) {
  //     // 	fv = 0.0;
  //     // 	fv(0) = 0;
  //     // 	fv(1) = 1;
  //     // 	fv(2) = fv(3) = 1;
  //     // }
  //     // else if (k_num==2) {
  //     // 	fv = 0.0;
  //     // 	fv(0) = fv(1) = 0;
  //     // 	fv(4) = fv(5) = 1;
  //     // }

  //     cout << "get EVEC nr " << k_num << " for eval: " << evals(k_num) << endl;
  //   }
  //   else {
  //     cout << "DONT HAVE THAT MANY EVS!!!" << endl;
  //   }
  //   /** prolongate down **/
  //   if (my_max_level == level) { //does not drop out -> #trans=#levels-1
  //     for (int k = level-1; k>=0; k--) {
  // 	map->TransferC2F(k, x_level[k], x_level[k+1]);
  //     }
  //   }
  //   else { //drops out -> #trans=#levels
  //     map->TransferC2F(my_max_level, x_level[my_max_level], nullptr);
  //     for (int k = my_max_level-1; k>=0; k--) {
  // 	map->TransferC2F(k, x_level[k], x_level[k+1]);
  //     }
  //   }
  //   x_level[0]->Cumulate();
  //   for (auto k : Range(vec.FVDouble().Size()))
  //     { vec.FVDouble()[k] = x_level[0]->FVDouble()[k]; }
  //   vec.SetParallelStatus(CUMULATED);
  // }

  
  void AMGMatrix :: GetBF(size_t level, int arank, size_t dof, BaseVector & vec) const
  {
    auto comm = map->GetParDofs(0)->GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    cout << "GET BF " << level << " " << rank << " " << dof << endl;
    int input_notok = 1;
    if ( (arank<0) || (arank>np) )
      cout << "Invalid rank " << arank << ", only have " << np << endl;
    else if ( (rank==arank) && (level>=x_level.Size() || x_level[level]==nullptr) )
      cout << "rank " << rank << "does not have that many levels!!" << endl;
    else if ( (rank==arank) && (rank!=0) && (dof>x_level[level]->FVDouble().Size()))
      cout << "rank " << rank << "only has " << x_level[level]->FVDouble().Size() << " dofs on level " << level << " (wanted dof " << dof << ")" << endl;
    else if ( (rank==arank) && (rank==0) && (dof>map->GetParDofs(level)->GetNDofGlobal()*map->GetParDofs(level)->GetEntrySize()))
      cout << "global ndof on level " << level << "is " << map->GetParDofs(level)->GetNDofGlobal() << " (wanted dof " << dof << ")" << endl;
    else
      input_notok = 0;
    input_notok = comm.AllReduce(input_notok, MPI_MAX);
    for (auto xl : x_level)
      if (xl!=nullptr) xl->FVDouble() = 0.0;
    cout << "GARBAGE IN? " << input_notok << endl;
    vec.FVDouble() = 0.0;
    vec.SetParallelStatus(CUMULATED);
    if (input_notok) return;
    cout << "STILL HERE!!" << endl;
    size_t max_x_access = x_level.Size() - (x_level.Last()==nullptr ? 2 : 1);
    size_t my_max_level = min2(max_x_access, (size_t)level);
    /** set coarsest vector **/
    if ( (arank!=0) && (rank==arank) ) {
      x_level[level]->FVDouble() = 0.0;
      x_level[level]->FVDouble()[dof] = 1.0;
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
      cout << "SET x_level[" << level << "] [" << dof << "] to 1.0!!" << endl;
    }
    else if ( (arank==0) && (my_max_level==level) ){
      auto pd = map->GetParDofs(level);
      auto BS = pd->GetEntrySize();
      auto n = pd->GetNDofLocal();
      auto all = make_shared<BitArray>(n);
      all->Set();
      Array<int> gdn;
      int gn;
      pd->EnumerateGlobally(all, gdn, gn);
      size_t bdof = dof/BS;
      for (auto k : Range(n)) {
	if (size_t(gdn[k])==bdof && pd->IsMasterDof(k)) {
	  cout << "SET x_level[" << level << "] [" << BS*k+(dof%BS) << "] to 1.0!! (is global nr " << dof << ") " << endl;
	  x_level[level]->FVDouble() = 0.0;
	  x_level[level]->FVDouble()[BS*k+(dof%BS)] = 1.0;
	}
      }
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
    }
    else if (my_max_level==level) {
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
    }
    /** prolongate down **/
    if (my_max_level == level) { //does not drop out -> #trans=#levels-1
      for (int k = level-1; k >= 0; k--) {
	map->TransferC2F(k, x_level[k].get(), x_level[k+1].get());
      }
    }
    else { //drops out -> #trans=#levels
      map->TransferC2F(my_max_level, x_level[my_max_level].get(), nullptr);
      for (int k = my_max_level-1; k >= 0; k--) {
	map->TransferC2F(k, x_level[k].get(), x_level[k+1].get());
      }
    }
    x_level[0]->Cumulate();
    for (auto k : Range(vec.FVDouble().Size()))
      { vec.FVDouble()[k] = x_level[0]->FVDouble()[k]; }
    vec.SetParallelStatus(CUMULATED);
  }


  size_t AMGMatrix :: GetNDof(size_t level, int arank) const
  {
    auto comm = map->GetParDofs(0)->GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    int input_notok = 0;
    if (arank<0 || arank>np) {
      cout << "Invalid rank" << arank << ", only have " << np << endl;
      return -1;
    }
    else if ( (rank==arank) && (level>=x_level.Size() || x_level[level]==nullptr) ) {
      cout << "rank " << rank << "does not have that many levels!!" << endl;
      input_notok = 1;
    }
    input_notok = comm.AllReduce(input_notok, MPI_MAX);
    if (input_notok) return -1;
// #ifdef ELASTICITY
    if (arank==0) return comm.AllReduce((rank==arank) ? map->GetParDofs(level)->GetNDofGlobal() * map->GetParDofs(level)->GetEntrySize() : 0, MPI_SUM);
// #else
//     if (arank==0) return map->GetParDofs(level)->GetNDofGlobal();
// #endif
    else return comm.AllReduce((rank==arank) ? x_level[level]->FVDouble().Size() : 0, MPI_SUM);
  }


  /** EmbeddedAMGMatrix **/

  
  void EmbeddedAMGMatrix :: MultTrans (const BaseVector & b, BaseVector & x) const
  {
    Mult(b, x);
  } // EmbeddedAMGMatrix::MultTrans


  void EmbeddedAMGMatrix :: Mult (const BaseVector & b, BaseVector & x) const
  {
    static Timer tt ("EmbeddedAMGMatrix::Mult"); RegionTimer rt(tt);
    static Timer tpre ("EmbeddedAMGMatrix::Pre");
    static Timer tpost ("EmbeddedAMGMatrix::Post");

    auto & cx = clm->x_level[0];
    auto & cr = clm->rhs_level[0];

    b.Distribute();

    if ( (fls != nullptr) ) {
      tpre.Start();
      x.FVDouble() = 0;
      x.SetParallelStatus(CUMULATED);
      res->FVDouble() = b.FVDouble();
      res->SetParallelStatus(b.GetParallelStatus());
      // fls->Smooth(x, b, *res, true, true, true);
      fls->SmoothK(stk, x, b, *res, true, true, true);
      // fls->Smooth(x, b, *res, false, false, false);
      ds->TransferF2C(res.get(), cr.get());
      tpre.Stop();
      clm->Smooth(*cx, *cr);
      tpost.Start();
      ds->AddC2F(1.0, &x, cx.get());
      // fls->SmoothBack(x, b, *res, false, false, false);
      fls->SmoothBackK(stk, x, b, *res, false, false, false);
      tpost.Stop();
    }
    else { /** **/
      ds->TransferF2C(&b, cr.get());
      clm->SmoothV(*cx, *cr);
      ds->TransferC2F(&x, cx.get());
    }

    // x *= 4.4;

  } // EmbeddedAMGMatrix::MultTrans


  void EmbeddedAMGMatrix :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    MultAdd(s, b, x);
  } // EmbeddedAMGMatrix::MultTrans


  void EmbeddedAMGMatrix :: MultAdd (double s, const BaseVector & b, BaseVector & x) const
  {
    Mult(b, *xbuf);
    x += s * *xbuf;
  } // EmbeddedAMGMatrix::MultTrans


  void EmbeddedAMGMatrix :: CINV (shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const
  {
    auto & cx = clm->x_level[0];
    auto & cr = clm->rhs_level[0];
    ds->TransferF2C(b.get(), cr.get());
    clm->CINV(cx, cr);
    ds->TransferC2F(x.get(), cx.get());
  } // EmbeddedAMGMatrix::CINV


  size_t EmbeddedAMGMatrix :: GetNLevels (int rank) const
  {
    return clm->GetNLevels(rank);
  } // EmbeddedAMGMatrix::GetNLevels


  size_t EmbeddedAMGMatrix :: GetNDof (size_t level, int rank) const
  {
    return clm->GetNDof(level, rank);
  } // EmbeddedAMGMatrix::GetNDof


  void EmbeddedAMGMatrix :: GetBF (size_t level, int rank, size_t dof, BaseVector & vec) const
  {
    auto & cx = clm->x_level[0];
    clm->GetBF(level, rank, dof, *cx);
    cout << "aux vec: " << endl;
    for (auto k : Range(cx->FVDouble().Size()))
      { if (cx->FVDouble()[k]!=0.0) { cout << "(" << k << ":" << cx->FVDouble()[k] << ") "; } }
    cout << endl;
    ds->TransferC2F(&vec, cx.get());
  } // EmbeddedAMGMatrix::GetBF




} // namespace amg
