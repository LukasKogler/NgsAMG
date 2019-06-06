
#include "amg.hpp"

namespace amg
{
  void AMGMatrix :: CINV(shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const
  {
    auto tmp = x_level[0];
    x_level[0] = x;
    auto tmp2 = rhs_level[0];
    rhs_level[0] = b;
    rhs_level[0]->Distribute();
    for(auto level:Range(n_levels-1))
      map->TransferF2C(level, rhs_level[level], rhs_level[level+1]);
    if(!drops_out) {
      crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]);
    }
    for(int level = n_levels-2; level>=0; level--)
      map->TransferC2F(level, x_level[level], x_level[level+1]);
    x_level[0] = tmp;
    rhs_level[0] = tmp2;
  }

  
  void AMGMatrix :: SmoothV(const shared_ptr<BaseVector> & x,
			    const shared_ptr<const BaseVector> & b,
			    FlatArray<int> on_levels) const
  {
    // cout << "n_levels: " << n_levels << endl;
    // cout << "vec mat dim check: " << endl;
    // cout << "x len: " << x->FVDouble().Size() << endl;
    // for(auto [k,vec]:Enumerate(x_level))
    //   if(vec!=nullptr)
    // 	cout << k << ": " << vec->FVDouble().Size() << endl;
    //   else
    // 	cout << k << ": nullptr" << endl;
    // cout << endl << "mat dims: " << endl;
    // for(auto [k,mat]:Enumerate(mats))
    //   if(mat!=nullptr)
    // 	cout << k << ": " << mat->Width() << endl;
    //   else
    // 	cout << k << ": nullptr" << endl;
    // cout << endl;
    static Timer tt ("AMG - Matrix - Mult - total");    
    RegionTimer rt(tt);
    static Timer t1("AMG - Matrix - Mult - smooth");
    static Timer t3("AMG - Matrix - Mult - transfer");
    static Timer t4("AMG - Matrix - Mult - coarse inv");
    static Timer t5("AMG - Matrix - Mult - drop, wait ");
    static Timer t_l0 ("AMG - Matrix - Mult - level 0");    
    static Timer t_l1 ("AMG - Matrix - Mult - level 1");    
    static Timer t_l2 ("AMG - Matrix - Mult - level 2");    
    static Timer t_l3 ("AMG - Matrix - Mult - level 3");    
    static Timer t_l4 ("AMG - Matrix - Mult - level 4");    
    static Timer t_l5 ("AMG - Matrix - Mult - level 5");    
    static Timer t_lr ("AMG - Matrix - Mult - rest");    
    static Timer t_lc ("AMG - Matrix - Mult - coarsest");    
    static Timer t_ld ("AMG - Matrix - Mult - drop");    
    Array<Timer*> lev_timers;
    lev_timers.Append(&t_l0);
    lev_timers.Append(&t_l1);
    lev_timers.Append(&t_l2);
    lev_timers.Append(&t_l3);
    lev_timers.Append(&t_l4);
    lev_timers.Append(&t_l5);
    lev_timers.Append(&t_lr);
    auto tmp = x_level[0];
    x_level[0] = x;
    shared_ptr<BaseVector> tmp2 = rhs_level[0];
    rhs_level[0] = shared_ptr<BaseVector>(const_cast<BaseVector*>(b.get()), NOOP_Deleter);
    rhs_level[0]->Distribute();
    // cout << "type x b " << typeid(*x).name() << " " << typeid(*b).name() << endl;
    // static double tf = 0;
    // static double tb = 0;
    /** pre-smooth **/

    // auto check_vec = [](auto & vecp) {
    //   if(vecp==nullptr) { cout << " VEC IS NULLPTR" << endl; return; }
    //   if(vecp->Size()==0) { cout << " VEC IS LEN 0 " << endl; return; }
    //   cout << "check vec, len " << vecp->Size() << " FVD s " << vecp->FVDouble().Size() << endl;
    //   cout << "factor: " << vecp->FVDouble().Size()/(1.0*vecp->Size()) << endl;
    //   // cout << "vec: " << endl << *vecp << endl;
    // };
    
    for(auto level:Range(n_levels-1))
      {
	RegionTimer rtl(*lev_timers[min2(level,size_t(6))]);
	t1.Start();
	// cout << " x  level " << level << ": "; check_vec(x_level[level]); cout << endl;
	// cout << "rhs level " << level << ": "; check_vec(rhs_level[level]); cout << endl;
	// cout << "res level " << level << ": "; check_vec(res_level[level]); cout << endl;
       	x_level[level]->FVDouble() = 0.0;
	x_level[level]->SetParallelStatus(CUMULATED);	
	res_level[level]->FVDouble() = rhs_level[level]->FVDouble();
	res_level[level]->SetParallelStatus(rhs_level[level]->GetParallelStatus());
	// cout << "smooth level " << level << endl;
	// double ts0 = -MPI_Wtime();
	// cout << "rhs " << level << *rhs_level[level] << endl;
	// cout << "res " << level << *res_level[level] << endl;
	smoothers[level]->Smooth(*x_level[level], *rhs_level[level], *res_level[level],
				 true, true, true);
	// cout << "x after sm " << level << endl << *x_level[level] << endl;
	// ts0 += MPI_Wtime();
	// if(level==0) {
	//   tf = 0.5 * (tf + ts0);
	//   cout << "avg fw took " << tf << endl;
	// }
	// cout << "sm done " << endl; 
	res_level[level]->Distribute();
	// cout << "res after sm " << level << endl << *res_level[level] << endl;
	t1.Stop();
	t3.Start();
	if(!drops_out || level+1<n_levels-2) rhs_level[level+1]->FVDouble() = 0.0;
	// cout << "trans up " << endl;
	map->TransferF2C(level, res_level[level], rhs_level[level+1]);
	// cout << "done trans up " << endl;
	t3.Stop();
      }
    /** coarsest level **/    
    if(!drops_out) {
      RegionTimer rtl(t_lc);
      t3.Start();
      // transfers.Last()->TransferF2C(res_level[n_levels-1], rhs_level[n_levels]);
      // map->TransferF2C(n_levels-2,res_level[n_levels-2], rhs_level[n_levels-1]);
      t3.Stop();
      if (has_crs_inv) { /** exact solve on coarsest level **/
	t4.Start();
	// cout << "crs inv " << endl;
	// cout << "crs rhs " << endl;
	// for(auto k:Range(rhs_level[n_levels-1]->FVDouble().Size()))
	//   cout << k << ": " << rhs_level[n_levels-1]->FVDouble()[k] << endl;
	// cout << endl;
	if (crs_inv==nullptr)
	  x_level[n_levels-1]->FVDouble() = 0;
	else
	  crs_inv->Mult(*rhs_level[n_levels-1], *x_level[n_levels-1]);
	// cout << "crs sol " << endl;
	// for(auto k:Range(x_level[n_levels-1]->FVDouble().Size()))
	//   cout << k << ": " << x_level[n_levels-1]->FVDouble()[k] << endl;
	// cout << endl;
	// cout << "done crs inv " << endl;
	x_level[n_levels-1]->Cumulate();
	t4.Stop();
      }
      else {
	throw Exception("Coarsest level smooth kind of not implemented ...");
      }
      t3.Start();
      //transfers.Last()->TransferC2F(res_level[n_levels-1], x_level[n_levels]);
      // map->TransferC2F(n_levels-2,res_level[n_levels-2], x_level[n_levels-1]);
      t3.Stop();
    }
    // else { /** proc is contracted out **/
    //   RegionTimer rtl(t_ld);
    //   t3.Start();
    //   // cout << "drop!" << endl;
    //   //transfers.Last()->TransferF2C(res_level.Last());
    //   map->TransferF2C(n_levels, x_level[n_levels], x_level[n_levels+1]);
    //   // cout << "join!" << endl;
    //   //transfers.Last()->TransferC2F(res_level.Last());
    //   map->TransferC2F(n_levels, x_level[n_levels], x_level[n_levels+1]);
    //   // cout << "joined!" << endl;
    //   t3.Stop();
    //   *x_level[n_levels-1] += *res_level.Last();
    // }
    /** post-smooth **/
    for(int level = n_levels-2; level>=0; level--)
      {
	auto & tlev = *lev_timers[min2(level,6)];
	if(x_level[level+1]!=nullptr) {
	  tlev.Start();
	  t3.Start();
	}
	else t5.Start();
	x_level[level]->Cumulate();
	map->TransferC2F(level, res_level[level], x_level[level+1]);
	// cout << "level " << level << " CGC: " << endl << *res_level[level] << endl;
	*x_level[level] += *res_level[level];
	if(x_level[level+1]!=nullptr) t3.Stop();
	else t5.Stop();
	t1.Start();
	// cout << "smooth level back" << level << endl;
	// cout << x_level[level]->FVDouble().Size() << " " << rhs_level[level]->FVDouble().Size() << " " << res_level[level]->FVDouble().Size() << " " << endl;
	// double ts0 = -MPI_Wtime();
	// if(level!=0)
	smoothers[level]->SmoothBack(*x_level[level], *rhs_level[level], *res_level[level],
	  			     false, false, false);

	// cout << "x after smb " << level << endl << *x_level[level] << endl;
	// ts0 += MPI_Wtime();
	// if(level==0) {
	//   tb = 0.5 * (tb + ts0);
	//   cout << "avg bw took " << tb << endl;
	// }// cout << "done smooth back" << level << endl;
	t1.Stop();
	if(x_level[level+1]!=nullptr) {
	  tlev.Stop();
	}
      }
    // TODO: remove this

    // cout << "stats 1 " << x_level[0]->GetParallelStatus() << " " << tmp2->GetParallelStatus() << endl;
    // mats[0]->Mult(*x_level[0], *tmp2);
    // cout << "stats 2 " << x_level[0]->GetParallelStatus() << " " << tmp2->GetParallelStatus() << endl;
    // double ip = InnerProduct(*x_level[0], *tmp2);
    // cout << "stats 3 " << x_level[0]->GetParallelStatus() << " " << tmp2->GetParallelStatus() << endl;
    // if(ip < 0)
    //   cout << " ---------- IP: " << ip << endl;

    x_level[0] = tmp;
    rhs_level[0] = tmp2;
    return;
  }

  void AMGMatrix :: GetEV(size_t level, int arank, size_t k_num, BaseVector & vec) const
  {
    auto comm = map->GetParDofs(0)->GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    if(np>1) throw Exception("GetEV doesnt work with NP>1.");
    cout << "GET EVEC " << level << " " << rank << " " << k_num << endl;
    int input_notok = 1;
    if( (arank<0) || (arank>np) )
      cout << "Invalid rank " << arank << ", only have " << np << endl;
    else if( (rank==arank) && (level>=x_level.Size() || x_level[level]==nullptr) )
      cout << "rank " << rank << "does not have that many levels!!" << endl;
    else
      input_notok = 0;
    input_notok = comm.AllReduce(input_notok, MPI_MAX);
    for(auto xl : x_level)
      if(xl!=nullptr) xl->FVDouble() = 0.0;
    cout << "GARBAGE IN? " << input_notok << endl;
    vec.FVDouble() = 0.0;
    vec.SetParallelStatus(CUMULATED);
    if(input_notok) return;
    cout << "STILL HERE!!" << endl;
    size_t max_x_access = x_level.Size() - (x_level.Last()==nullptr ? 2 : 1);
    size_t my_max_level = min2(max_x_access, (size_t)level);
    /** set coarsest vector **/
    x_level[my_max_level]->FVDouble() = 0.0;
    if(level==my_max_level && k_num<x_level[my_max_level]->FVDouble().Size()) {
      auto mat = dynamic_pointer_cast<const SparseMatrix<double>>(mats[level]);
      size_t h = mat->Height();
      if(h>3000) {
	cout << "mat size " << h << ", too large, return 0 vec!" << endl;
	return;
      }
      Matrix<double> full(h,h);
      if(h) full = 0.0;
      for(auto k:Range(h))
	for(auto j:Range(h)) {
	  full(k,j) = (*mat)(k,j);
	}
      Matrix<double> evecs(h,h);
      Vector<double> evals(h);
      LapackEigenValuesSymmetric(full, evals, evecs);
      // cout << "full mat: " << endl << full << endl;
      cout << "evals: " << endl;prow(evals); cout << endl;
      *testout << "evals: " << endl;
      for(auto k:Range(h)) *testout << evals[k] << endl;
      
      for(auto k:Range(h)) {
	x_level[my_max_level]->FVDouble()[k] = evecs(k_num, k);
      }

      // const auto & fv = x_level[my_max_level]->FVDouble();
      // auto nv = h/3;
      // if(k_num==1) {
      // 	fv = 0.0;
      // 	fv(0) = 0;
      // 	fv(1) = 1;
      // 	fv(2) = fv(3) = 1;
      // }
      // else if (k_num==2) {
      // 	fv = 0.0;
      // 	fv(0) = fv(1) = 0;
      // 	fv(4) = fv(5) = 1;
      // }

      cout << "get EVEC nr " << k_num << " for eval: " << evals(k_num) << endl;
    }
    else {
      cout << "DONT HAVE THAT MANY EVS!!!" << endl;
    }
    /** prolongate down **/
    if(my_max_level == level) { //does not drop out -> #trans=#levels-1
      for(int k = level-1; k>=0; k--) {
	map->TransferC2F(k, x_level[k], x_level[k+1]);
      }
    }
    else { //drops out -> #trans=#levels
      map->TransferC2F(my_max_level, x_level[my_max_level], nullptr);
      for(int k = my_max_level-1; k>=0; k--) {
	map->TransferC2F(k, x_level[k], x_level[k+1]);
      }
    }
    x_level[0]->Cumulate();
    for(auto k:Range(vec.FVDouble().Size()))
      vec.FVDouble()[k] = x_level[0]->FVDouble()[k];
    vec.SetParallelStatus(CUMULATED);
  }

  
  void AMGMatrix :: GetBF(size_t level, int arank, size_t dof, BaseVector & vec) const
  {
    auto comm = map->GetParDofs(0)->GetCommunicator();
    auto rank = comm.Rank();
    auto np = comm.Size();
    cout << "GET BF " << level << " " << rank << " " << dof << endl;
    int input_notok = 1;
    if( (arank<0) || (arank>np) )
      cout << "Invalid rank " << arank << ", only have " << np << endl;
    else if( (rank==arank) && (level>=x_level.Size() || x_level[level]==nullptr) )
      cout << "rank " << rank << "does not have that many levels!!" << endl;
    else if( (rank==arank) && (rank!=0) && (dof>x_level[level]->FVDouble().Size()))
      cout << "rank " << rank << "only has " << x_level[level]->FVDouble().Size() << " dofs on level " << level << " (wanted dof " << dof << ")" << endl;
#ifdef ELASTICITY
    else if( (rank==arank) && (rank==0) && (dof>map->GetParDofs(level)->GetNDofGlobal()*map->GetParDofs(level)->GetEntrySize()))
#else
    else if( (rank==arank) && (rank==0) && (dof>map->GetParDofs(level)->GetNDofGlobal()))
#endif
      cout << "global ndof on level " << level << "is " << map->GetParDofs(level)->GetNDofGlobal() << " (wanted dof " << dof << ")" << endl;
    else
      input_notok = 0;
    input_notok = comm.AllReduce(input_notok, MPI_MAX);
    for(auto xl : x_level)
      if(xl!=nullptr) xl->FVDouble() = 0.0;
    cout << "GARBAGE IN? " << input_notok << endl;
    vec.FVDouble() = 0.0;
    vec.SetParallelStatus(CUMULATED);
    if(input_notok) return;
    cout << "STILL HERE!!" << endl;
    size_t max_x_access = x_level.Size() - (x_level.Last()==nullptr ? 2 : 1);
    size_t my_max_level = min2(max_x_access, (size_t)level);
    /** set coarsest vector **/
    if( (arank!=0) && (rank==arank) ) {
      x_level[level]->FVDouble() = 0.0;
      x_level[level]->FVDouble()[dof] = 1.0;
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
      cout << "SET x_level[" << level << "] [" << dof << "] to 1.0!!" << endl;
    }
    else if ( (arank==0) && (my_max_level==level) ){
      // auto pd = mats[level]->GetParallelDofs();
      auto pd = map->GetParDofs(level);
#ifdef ELASTICITY
      auto BS = pd->GetEntrySize();
#else
      auto BS = 1;
#endif
      auto n = pd->GetNDofLocal();
      auto all = make_shared<BitArray>(n);
      all->Set();
      Array<int> gdn;
      int gn;
      pd->EnumerateGlobally(all, gdn, gn);
      size_t bdof = dof/BS;
      for(auto k:Range(n))
	if(size_t(gdn[k])==bdof && pd->IsMasterDof(k)) {
	  cout << "SET x_level[" << level << "] [" << BS*k+(dof%BS) << "] to 1.0!! (is global nr " << dof << ") " << endl;
	  x_level[level]->FVDouble() = 0.0;
	  x_level[level]->FVDouble()[BS*k+(dof%BS)] = 1.0;
	}
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
    }
    else if(my_max_level==level) {
      x_level[level]->SetParallelStatus(DISTRIBUTED);
      x_level[level]->Cumulate();
    }
    /** prolongate down **/
    if(my_max_level == level) { //does not drop out -> #trans=#levels-1
      for(int k = level-1; k>=0; k--) {
	map->TransferC2F(k, x_level[k], x_level[k+1]);
      }
    }
    else { //drops out -> #trans=#levels
      map->TransferC2F(my_max_level, x_level[my_max_level], nullptr);
      for(int k = my_max_level-1; k>=0; k--) {
	map->TransferC2F(k, x_level[k], x_level[k+1]);
      }
    }
    x_level[0]->Cumulate();
    for(auto k:Range(vec.FVDouble().Size()))
      vec.FVDouble()[k] = x_level[0]->FVDouble()[k];
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
#ifdef ELASTICITY
    if (arank==0) return comm.AllReduce((rank==arank) ? map->GetParDofs(level)->GetNDofGlobal() * map->GetParDofs(level)->GetEntrySize() : 0, MPI_SUM);
#else
    if (arank==0) return map->GetParDofs(level)->GetNDofGlobal();
#endif
    else return comm.AllReduce((rank==arank) ? x_level[level]->FVDouble().Size() : 0, MPI_SUM);
  }

} // namespace amg
