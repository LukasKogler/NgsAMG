
#include <amg_matrix.hpp>

namespace amg
{

  /** AMGMatrix **/

  AMGMatrix :: AMGMatrix (shared_ptr<DOFMap> _map, FlatArray<shared_ptr<BaseSmoother>> _smoothers)
    : BaseMatrix(_map->GetParDofs(0))
    , map(_map)
    , smoothers(_smoothers)
  {
    /** local & global nr of levels **/
    n_levels = map->GetNLevels();
    n_levels_glob = map->GetUDofs().GetCommunicator().AllReduce(n_levels, NG_MPI_MAX);

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


  void AMGMatrix :: SetCoarseInv (shared_ptr<BaseMatrix> _crs_inv, shared_ptr<BaseMatrix> _crs_mat)
  { crs_inv = _crs_inv; crs_mat = _crs_mat; has_crs_inv = true; }


  void AMGMatrix :: SmoothW (BaseVector & x, const BaseVector & b) const
  {
    b.Distribute();

    static Timer tt ("AMGMatrix::SmoothW"); RegionTimer rt(tt);

    auto comm = map->GetUDofs().GetCommunicator();

    function<void(int)> smit = [&](int level) {
      if ( level == 0 ) {
	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());

	smoothers[level]->Smooth(xl, bl, rl, true, true, true);
	rl.Distribute();

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);
      }
      if ( level + 1 < n_levels ) {
	BaseVector &xl ( (level == 0) ? x : *x_level[level]);
	const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
	BaseVector &rl (*res_level[level]);

       	xl.FVDouble() = 0.0;
	xl.SetParallelStatus(CUMULATED);
	rl.FVDouble() = bl.FVDouble();
	rl.SetParallelStatus(bl.GetParallelStatus());

	smoothers[level]->Smooth(xl, bl, rl, true, true, true);
	rl.Distribute();

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(level + 1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothBack(xl, bl, rl, false, true, false);
	smoothers[level]->Smooth(xl, bl, rl, true, true, false);

	map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
	smit(level + 1);
	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());

	smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);
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

    b.Distribute();

    /** pre-smooth **/
    for (auto level : Range(n_levels-1)) {
      BaseVector &xl ( (level == 0) ? x : *x_level[level]);
      const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
      BaseVector &rl (*res_level[level]);
      xl.FVDouble() = 0.0;
      xl.SetParallelStatus(CUMULATED);
      rl.FVDouble() = bl.FVDouble();
      rl.SetParallelStatus(bl.GetParallelStatus());
      SmoothVFromLevel(level, xl, bl, rl, true, true, true);
      rl.Distribute();
      map->TransferF2C(level, res_level[level].get(), rhs_level[level+1].get());
    }

    /** coarsest level **/
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

    /** post-smooth **/
    for (int level = n_levels-2; level>=0; level--)
    {
      BaseVector &xl ( (level == 0) ? x : *x_level[level]);
      const BaseVector &bl ( (level == 0) ? b : *rhs_level[level]);
      BaseVector &rl (*res_level[level]);
      map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
      SmoothVFromLevel(level, xl, bl, rl, false, false, false);
    }

  } // AMGMatrix::SmoothBS


  void AMGMatrix :: SmoothV (BaseVector & x, const BaseVector & b) const
  {
    // std::cout << "AMGMAT::V, size = " << x.Size() << " types " << typeid(x).name() << " " << typeid(b).name() << " stats " << x.GetParallelStatus() << " " << b.GetParallelStatus() << std::endl;

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
    static Array<Timer<TTracing, TTiming>*> lev_timers({ &t_l0, &t_l1, &t_l2, &t_l3, &t_lr });

    auto ltimer = [](int level) -> Timer<TTracing, TTiming>& { return *lev_timers[min2(level,4)]; };

    auto comm = map->GetUDofs().GetCommunicator();

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

      // std::cout << "AMGMAT::V, level " << level
      //           << ", size = " << xl.Size() << " " << bl.Size() << " " << rl.Size()
      //           << ", size = " << xl.FVDouble().Size() << " " << bl.FVDouble().Size() << " " << rl.FVDouble().Size()
      //           << " types " << typeid(xl).name() << " " << typeid(rl).name() << " stats " << xl.GetParallelStatus() << " " << rl.GetParallelStatus() << std::endl;

      rl.FVDouble() = bl.FVDouble();
      rl.SetParallelStatus(bl.GetParallelStatus());


      // smoothers[level]->Smooth(xl, bl, rl, true, true, true);
      smoothers[level]->Smooth(xl, bl, rl, true, true, true);

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

  // if (level+1 == n_levels-1)
  // {
  // 	map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
  // }
  // else
  // {
  //   double alpha = 1.0;

  //   auto corr = x_level[level+1]->CreateVector();
  //   *corr = *x_level[level+1];

  //   auto Acorr = (*corr).CreateVector();
  //   smoothers[level+1]->GetAMatrix()->Mult(*corr, *Acorr);

  //   double const a = InnerProduct(*corr, *res_level[level + 1]);
  //   double const b = InnerProduct(*Acorr, *corr);

  //   if (a*b > 0)
  //   {
  //     if (a > 2 * b)
  //       alpha = 2;
  //     else
  //       alpha = a/b;
  //   }

    // double const alpha = a / b;

    // map->AddC2F(level, alpha, &xl, &*corr);
  // }


	// cout << " x 2  level " << level << ": " << endl << xl.FVDouble() << endl;

	if (x_level[level+1] == nullptr)
	  { t5.Stop(); }

	// smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);
	smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);

	// cout << " x 3  level " << level << ": " << endl << xl.FVDouble() << endl;
      }

  } // AMGMatrix::SmoothV


  void AMGMatrix :: SmoothVFromLevel (int startlevel, BaseVector  &x, const BaseVector &b, BaseVector  &res,
				     bool res_updated, bool update_res, bool x_zero) const
  {
    static Timer t ("SmoothFromLevel");
    RegionTimer rt(t);

    // cout << "SVL " << startlevel << " of " << n_levels << endl;
    // cout << " x b r s " << x.Size() << " " << b.Size() << " " << res.Size() << endl;

    // cout << " SVL (F) " << startlevel << " " << startlevel << endl;
    smoothers[startlevel]->Smooth(x, b, res, res_updated, true, x_zero);
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
        smoothers[level]->Smooth(xl, bl, rl, true, true, true);
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

    if (startlevel + 2 < n_levels) {
      for (int level = n_levels - 2; level > startlevel; level--) {
        BaseVector &xl ( *x_level[level] ), &rl (*res_level[level]);
        const BaseVector &bl ( *rhs_level[level] );
        // cout << " SVL (B) " << startlevel << " " << level << endl;
        map->AddC2F(level, 1.0, &xl, x_level[level+1].get());
        // cout << " SVL (B) " << startlevel << " " << level << endl;
        smoothers[level]->SmoothBack(xl, bl, rl, false, false, false);
      }
    }

    // cout << " SVL (B) " << startlevel << " " << startlevel << endl;
    map->AddC2F(startlevel, 1.0, &x, x_level[startlevel+1].get());
    // cout << " SVL (B) " << startlevel << " " << startlevel << endl;
    smoothers[startlevel]->SmoothBack(x, b, res, false, update_res, false);

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
    auto comm = map->GetUDofs().GetCommunicator();
    return comm.AllReduce((comm.Rank() == rank) ? (1 + smoothers.Size()) : 0, NG_MPI_SUM);
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
      // cout << endl << endl << "CINV --  crs rhs/sol (len = " << rhs_level[n_levels-1]->FVDouble().Size()<< "): " << endl;
      // for (auto k : Range(rhs_level[n_levels-1]->FVDouble().Size())) {
      //   cout << k << ": " << rhs_level[n_levels-1]->FVDouble()[k] << " " << x_level[n_levels-1]->FVDouble()[k] << endl;
      // }
      // cout << endl;
    }
    for (int level = n_levels-2; level>=0; level--)
      { map->TransferC2F(level, x_level[level].get(), x_level[level+1].get()); }
    // cout << endl << endl << "CINV --  finest level rhs/sol (len = " << rhs_level[0]->FVDouble().Size()<< "): " << endl;
    // for (auto k : Range(rhs_level[0]->FVDouble().Size())) {
    //   cout << k << ": " << rhs_level[0]->FVDouble()[k] << " " << x_level[0]->FVDouble()[k] << endl;
    // }
    // cout << endl;

    x_level[0] = tmp;
    rhs_level[0] = tmp2;
  } // AMGMatrix::CINV



  void AMGMatrix :: GetBF (BaseVector & vec, int level, size_t dof, int comp, int arank, int onLevel) const
  {
    const auto & map = *GetMap();
    auto gcomm = map.GetUDofs().GetCommunicator();
    int nlev_loc = map.GetNLevels();
    int nlev_loc_real = map.GetUDofs(map.GetNLevels() - 1).IsValid() ? nlev_loc - 1 : nlev_loc;
    // auto lev_pds = (level < nlev_loc) ? map.GetParDofs(level) : nullptr;
    UniversalDofs lev_uDofs = map.GetUDofs(level);
    bool const have_lev = lev_uDofs.IsValid();
    int  const lev_bs   = lev_uDofs.GetBS();

    // cout << " getBF " << level << " have_lev " << have_lev << ", uDofs " << lev_uDofs << endl;

    vec.FVDouble() = 0.0;
    vec.SetParallelStatus(CUMULATED);

    { // Check if input is OK
      int input_notok = 0;
      if ( (arank < 0) || (arank > gcomm.Size()) )
      	{ input_notok = 1; }
      if ( have_lev && (comp > lev_bs) )
	      { input_notok = 2; }
      if ( (gcomm.Rank() == arank) && (!have_lev) ) // e.g contracted out
	      { input_notok = 3; }
      if (have_lev) {
        if ( (gcomm.Rank() == arank) && (gcomm.Rank() > 0) && (dof > lev_uDofs.GetND()) )
          { input_notok = 4; }
        if ( (arank == 0) && (dof > lev_uDofs.GetNDGlob()) )
          { input_notok = 5; }
      }
      input_notok = gcomm.AllReduce(input_notok, NG_MPI_MAX);
      if (input_notok != 0) {
        if (gcomm.Rank() == 0) {
          cerr << "GetBF invalid input, reason = ";
          switch(input_notok) {
          case(1) : { cerr << "invalid rank " << arank << endl; break; }
          case(2) : { cerr << "component " << comp << " invalid on level " << level << endl; break; }
          case(3) : { cerr << "rank " << arank << " does not have DOFs on level " << level << endl; break; }
          case(4) : { cerr << "rank " << arank << " DOFs on level " << level << " < " << dof << endl; break; }
          case(5) : { cerr << "global DOFs on level " << level << " < " << dof << endl; break; }
          default : { break; }
          }
          return;
        }
      }
    }

    /** set vector on level **/
    if (have_lev) { // have && not empty!
      auto BS = lev_bs;
      // cout << " x_level[level] = " << x_level[level] << endl;
      x_level[level]->FVDouble() = 0.0;
      if ( ( gcomm.Size() == 1 ) || ( arank > 0 ) ) {
        // if ( gcomm.Rank() == arank)
        //   { cout << " SET VEC " << level << " @ " << BS*dof + comp << endl; }
        if ( gcomm.Rank() == arank)
          { x_level[level]->FVDouble()[BS*dof + comp] = 1.0; }
        x_level[level]->SetParallelStatus(DISTRIBUTED);
        x_level[level]->Cumulate();
      }
      else {
        auto ndloc = lev_uDofs.GetND();
        auto gdn = const_cast<DOFMap&>(map).GetGlobDNums(level);
        for (auto k : Range(ndloc))
          if (size_t(gdn[k]) == dof)
            { x_level[level]->FVDouble()[BS*k + comp] = 1.0; }
        x_level[level]->SetParallelStatus(CUMULATED);
      }
    }

    /** prolongate down to level 0 **/
    map.TransferAtoB (min(level, nlev_loc), onLevel, x_level[level].get(), &vec);
  } // AMGMatrix :: GetBF


  tuple<size_t, int>  AMGMatrix :: GetNDof (int level, int arank) const
  {
    const auto & map = *GetMap();
    NgMPI_Comm gcomm(map.GetUDofs().GetCommunicator());
    int nlev_loc = map.GetNLevels();
    int nlev_loc_real = map.GetUDofs(map.GetNLevels() - 1).IsValid() ? nlev_loc - 1 : nlev_loc;
    // int nlev_loc_real = (map.GetParDofs(map.GetNLevels() - 1) == nullptr) ? nlev_loc - 1 : nlev_loc;
    UniversalDofs lev_uDofs = map.GetUDofs(level);
    bool const have_lev = lev_uDofs.IsValid();
    int  const lev_bs   = lev_uDofs.GetBS();

    int input_notok = 0;
    { // Check input
      if ( have_lev && (arank >= lev_uDofs.GetCommunicator().Size()) ) {
      	input_notok = 1;
        if ( gcomm.Rank() == 0 )
          { cerr << " no rank " << arank << " on level " << level << endl; }
      }
      input_notok = gcomm.AllReduce(input_notok, NG_MPI_MAX);
      if ( input_notok )
      	{ return make_tuple(0, 1); }
    }

    IVec<2, size_t> nd_bs(0);
    if ( have_lev ) {
      nd_bs[1] = lev_bs;
      if ( (arank > 0) && (gcomm.Rank() == arank) )
      	{ nd_bs[0] = lev_uDofs.GetND(); }
      else if ( arank == 0 )
	      { nd_bs[0] = lev_uDofs.GetNDGlob(); }
    }
    bool any_nolevel = (!have_lev) || (gcomm.Size() > lev_uDofs.GetCommunicator().Size());
    if ( (arank > 0) || any_nolevel) // ND from a rank != 0, or at least one proc does not have level
      { gcomm.Bcast(nd_bs, arank); }

    return make_tuple(nd_bs[0], nd_bs[1]);
  }

  Array<double> AMGMatrix :: GetOC () const
  {
    auto comm = map->GetUDofs().GetCommunicator();
    size_t n_levs_loc = (crs_inv == nullptr) ? smoothers.Size() : smoothers.Size() + 1;
    size_t n_levs_glob = comm.AllReduce(n_levs_loc, NG_MPI_MAX);
    Array<size_t> nzes(n_levs_glob), nops(n_levs_glob); nzes = 0.0; nops = 0.0;

    // for (auto k : Range(smoothers)) {
    //   if (smoothers[k]) {
    //   cout << " smoother " << k << " / " << smoothers.Size() << endl;
    //   smoothers[k]->PrintTo(cout); cout << endl << endl;
    // }
    // } cout << endl;

    for (auto k : Range(smoothers)) {
      nzes[k] = smoothers[k]->GetANZE();
      nops[k] = smoothers[k]->GetNOps();
    }
    comm.AllReduceFA(nzes);
    comm.AllReduceFA(nops);
    Array<double> occs(1 + n_levs_loc);
    for (auto k : Range(n_levs_loc)) {
      switch(vwb) {
      case(0) : { occs[1+k] = nops[k] / double(nzes[0]); break; }
      case(1) : { occs[1+k] = pow(2,k) * nops[k] / double(nzes[0]); break; }
      case(2) : { occs[1+k] = 2*(1+k) * nops[k] / double(nzes[0]); break; }
      }
    }
    occs[0] = 0.0;
    occs[0] = std::accumulate(occs.begin(), occs.end(), 0.0);
    return occs;
  } // AMGMatrix::GetOC

  /** END AMGMatrix **/


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
      fls->Smooth(x, b, *res, true, true, true);
      // fls->Smooth(x, b, *res, false, false, false);
      ds->TransferF2C(res.get(), cr.get());
      tpre.Stop();
      clm->Smooth(*cx, *cr);
      tpost.Start();
      ds->AddC2F(1.0, &x, cx.get());
      // fls->SmoothBack(x, b, *res, false, false, false);
      fls->SmoothBack(x, b, *res, false, false, false);
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


  tuple<size_t, int> EmbeddedAMGMatrix :: GetNDof (int level, int rank) const
  {
    return clm->GetNDof(level, rank);
  } // EmbeddedAMGMatrix::GetNDof


  void EmbeddedAMGMatrix :: GetBF (BaseVector & vec, int level, size_t dof, int comp, int rank) const
  {
    auto & cx = clm->x_level[0];
    clm->GetBF(*cx, level, dof, comp, rank);
    // cout << "aux vec: " << endl;
    // for (auto k : Range(cx->FVDouble().Size()))
      // { if (cx->FVDouble()[k]!=0.0) { cout << "(" << k << ":" << cx->FVDouble()[k] << ") "; } }
    // cout << endl;
    ds->TransferC2F(&vec, cx.get());
  } // EmbeddedAMGMatrix::GetBF


/** AMGSmoother **/

AMGSmoother::
AMGSmoother (shared_ptr<AMGMatrix> _amg_mat, int _start_level, bool _singleFLS)
  : BaseSmoother(_amg_mat->GetSmoother(_start_level)->GetAMatrix())
  , start_level(_start_level)
  , singleFLS(_singleFLS)
  , amg_mat(_amg_mat)
  { ; }


void
AMGSmoother::
Smooth (BaseVector  &x, const BaseVector &b,
        BaseVector  &res, bool res_updated,
        bool update_res, bool x_zero) const
{
  if (singleFLS)
  {
    if (!res_updated)
    {
      this->CalcResiduum(x, b, res, x_zero);
    }

    auto const &map = *amg_mat->GetMap();

    bool const haveNextLevel = map.GetNLevels() > start_level;

    if (haveNextLevel)
    {
      auto xC   = amg_mat->x_level[start_level + 1];
      auto bC   = amg_mat->rhs_level[start_level + 1];
      auto resC = amg_mat->res_level[start_level + 1];

      map.TransferAtoB(start_level, start_level + 1, &res, bC.get());
      *xC = 0.0;

      if (map.GetNLevels() > start_level)
      {
        amg_mat->SmoothVFromLevel(start_level + 1, *xC, *bC, *resC, false, false, true);
      }

      map.AddC2F(start_level, 1.0, &x, xC.get());
    }
    else
    {
      map.TransferAtoB(start_level, start_level + 1, &res, nullptr);

      map.AddC2F(start_level, 1.0, &x, nullptr);
    }

    amg_mat->GetSmoother(start_level)->Smooth(x, b, res, false, update_res, false);
  }
  else
  {
    amg_mat->SmoothVFromLevel(start_level, x, b, res, res_updated, update_res, x_zero);
  }
} // AMGSmoother::Smooth


void
AMGSmoother::
SmoothBack (BaseVector  &x, const BaseVector &b,
            BaseVector &res, bool res_updated,
            bool update_res, bool x_zero) const
{
  if (singleFLS)
  {
    auto const &map = *amg_mat->GetMap();

    bool const haveNextLevel = map.GetNLevels() > start_level;

    amg_mat->GetSmoother(start_level)->SmoothBack(x, b, res, res_updated, true, x_zero);

    if (haveNextLevel)
    {
      auto xC   = amg_mat->x_level[start_level + 1];
      auto bC   = amg_mat->res_level[start_level + 1];
      auto resC = amg_mat->rhs_level[start_level + 1];

      map.TransferAtoB(start_level, start_level + 1, &res, bC.get());
      *xC = 0.0;

      if (map.GetNLevels() > start_level)
      {
        amg_mat->SmoothVFromLevel(start_level + 1, *xC, *bC, *resC, false, false, true);
      }

      map.AddC2F(start_level, 1.0, &x, xC.get());
    }
    else
    {
      map.TransferAtoB(start_level, start_level + 1, &res, nullptr);

      map.AddC2F(start_level, 1.0, &x, nullptr);
    }

    if (update_res)
    {
      this->CalcResiduum(x, b, res, false);
    }
  }
  else
  {
    amg_mat->SmoothVFromLevel(start_level, x, b, res, res_updated, update_res, x_zero);
  }
} // AMGSmoother::SmoothBack

size_t
AMGSmoother::
GetNOps () const
{
  size_t nops = 0;
  auto smoothers = amg_mat->GetSmoothers();
  for (auto k : Range(smoothers.Part(start_level)))
    { nops += smoothers[k]->GetNOps(); }
  if (auto cmat = amg_mat->GetCMat())
    { nops += GetScalNZE(cmat.get()); }
  return nops;
} // AMGSmoother::GetNOps

/** END AMGSmoother **/


} // namespace amg
