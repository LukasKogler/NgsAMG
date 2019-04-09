#include "amg.hpp"

namespace amg {


  shared_ptr<SparseMatrix<double>> AssBlockRepl_wip
  (int BS, const shared_ptr<const BaseAlgebraicMesh> & mesh,
   const shared_ptr<const AMGOptions> & opts,
   std::function<void(size_t vi, size_t vj, FlatMatrix<double>&)> get_block,
   std::function<bool(size_t vi, size_t vj)> filter_block,
   bool smooth, double rho)
  {
    static Timer t("AssBlockRepl");
    RegionTimer rt(t);
    auto econ = mesh->GetEdgeConnectivityMatrix();
    // const auto & eqc_h = mesh->GetEQCHierarchy();
    auto nv = mesh->NN<NT_VERTEX>();
    auto ne = mesh->NN<NT_EDGE>();
    auto ND = nv * BS;
    auto get_dof = [&](auto v, auto comp) { return nv*comp + v; };
    Table<int> v_graph;
    // cout << "nv ne ND " << nv << " " << ne << " " << ND << endl;
    {
      TableCreator<int> cvg(nv);
      while(!cvg.Done()) {
	for(const auto & vi : mesh->GetNodes<NT_VERTEX>()) {
	  auto vjs = econ->GetRowIndices(vi);
	  bool dd = false;
	  for(auto vj:vjs) {
	    if(filter_block(vi,vj)) continue;
	    if(vj>vi && !dd) {
	      cvg.Add(vi,vi);
	      dd = true;
	    }
	    cvg.Add(vi,vj);
	  }
	  if(!dd) cvg.Add(vi,vi);
	}
	cvg++;
      }
      v_graph = cvg.MoveTable();
      // cout << "v_graph: " << endl << v_graph << endl;
    }
    Array<int> nperow (ND);
    if(smooth)
      for(auto l:Range(BS))
	for(auto k:Range(nv))
	  nperow[get_dof(k,l)] = v_graph[k].Size() ? BS * (v_graph[k].Size()-1) + 1 : 0;
    else
      for(auto l:Range(BS))
	for(auto k:Range(nv))
	  nperow[get_dof(k,l)] = BS * v_graph[k].Size();
    Table<int> graph(nperow);
    if(smooth) {
      for(auto lk:Range(BS)) {
	for(auto k:Range(nv)) {
	  auto dofk = get_dof(k,lk);
	  auto row = graph[dofk];
	  int rc = 0;
	  for(auto lj:Range(BS))
	    for(auto vj:v_graph[k])
	      if(vj==k) { if(lk==lj) row[rc++] = get_dof(vj,lk); }
	      else row[rc++] = get_dof(vj,lj);
	}
      }
    }
    else {
      for(auto lk:Range(BS))
	for(auto k:Range(nv)) {
	  auto dofk = get_dof(k,lk);
	  auto row = graph[dofk];
	  int rc = 0;
	  for(auto lj:Range(BS))
	    for(auto vj:v_graph[k])
	      row[rc++] = get_dof(vj,lj);
	}  
    }
    cout << "dof_graph: " << endl << graph << endl;
    auto ahat = make_shared<SparseMatrix<double>> (nperow, ND);
    {
      constexpr int hs = 2*1024*1024;
      LocalHeap lh(hs, "Lukas", false); // ~2 MB LocalHeap
      // LocalHeap lh(2000000, "Lukas", false); // ~2 MB LocalHeap
      Array<int> wt_cols(2*BS);
      Array<int> get_rows(BS);
      Array<int> dcols(BS);
      FlatMatrix<double> block (2*BS, 2*BS, lh);
      for (auto vi:Range(nv)) {
	HeapReset hr(lh);
	auto vcols = v_graph[vi];
	size_t nv_used = vcols.Size();
	auto get_col = [&](auto posk, auto comp) { return nv_used * comp + posk;};
	FlatMatrix<double> rows(BS, BS*nv_used, lh);
	rows = 0.0;
	auto posi = vcols.Pos(vi);
	for(auto [posj, vj] : Enumerate(vcols)) {
	  if(vj==vi) continue;
	  // bool lr = vi<vj;
	  for(auto l : Range(BS)) {
	    // wt_cols[2*l + (lr ? 0 : 1)] = get_col(posi, l);
	    // wt_cols[2*l + (lr ? 1 : 0)] = get_col(posj, l);
	    // get_rows[l] = 2*l + (lr ? 0 : 1);
	    wt_cols[2*l] = get_col(posi, l);
	    wt_cols[2*l+1] = get_col(posj, l);
	    get_rows[l] = 2*l;
	  }
	  get_block(vi,vj, block);
	  rows.Cols(wt_cols) += block.Rows(get_rows);
	}
	if (!smooth) {
	  // throw Exception("Not debugged/tested at all... ");
	  for(auto li : Range(BS)) {
	    auto dofi = get_dof(vi,li);
	    auto ris = ahat->GetRowIndices(dofi);
	    ris = graph[dofi];
	    auto rvs = ahat->GetRowValues(dofi);
	    for(auto [k,val] : Enumerate(rvs))
	      val = rows(li,k);
	  }
	}
	else {
	  for(auto l : Range(BS))
	    dcols[l] = get_col(posi, l);
	  FlatMatrix<double> diag_inv(BS, BS, lh);
	  diag_inv = rows.Cols(dcols);
	  CalcInverse(diag_inv);
	  FlatMatrix<double> inv_t_row(BS, BS*nv_used, lh);
	  inv_t_row = -rho * diag_inv * rows;
	  diag_inv = -1000000.0;
	  inv_t_row.Cols(dcols) = diag_inv;
	  for(auto li : Range(BS)) {
	    int crv = 0;
	    int crb = 0;
	    auto dofi = get_dof(vi,li);
	    auto ris = ahat->GetRowIndices(dofi);
	    auto rvs = ahat->GetRowValues(dofi);
	    ris = graph[dofi];
	    for(auto lj:Range(BS)) {
	      for(auto posk:Range(posi))
		rvs[crv++] = inv_t_row(li, crb++);
	      crb ++;
	      if(li==lj) rvs[crv++] = 1.0-rho;
	      for(auto posk:Range(posi+1, nv_used))
		rvs[crv++] = inv_t_row(li, crb++);
	    }
	  }
	  for(auto li : Range(BS)) {
	    auto dofi = get_dof(vi,li);
	    auto rvs = ahat->GetRowValues(dofi);
	  }
	  // cout << endl;
	}
      }
    }
    return move(ahat);
  } // end AssBlockRepl
  
  
} // end namespace amg
