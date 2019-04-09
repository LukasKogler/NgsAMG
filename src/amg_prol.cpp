#include "amg.hpp"

namespace amg {

  ProlongationClasses & GetProlongationClasses()
  {
    static ProlongationClasses smcl;
    return smcl;
  }


  

  shared_ptr<BaseSparseMatrix>
  BuildH1PWProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
			 const shared_ptr<const BaseAlgebraicMesh> & cmesh,
			 const shared_ptr<const CoarseMapping> & cmap)
  {
    size_t nv = fmesh->NV();
    size_t ncv = cmesh->NV();
    auto& vertex_coarse = cmap->GetVMAP();
    size_t nverts = vertex_coarse.Size();
    Array<int> non_zero_per_row(nverts);
    non_zero_per_row = 0;
    for (int i = 0; i < nverts; ++i)
      if (vertex_coarse[i] != -1) {
	non_zero_per_row[i] = 1;
      }
    auto prol = make_shared<SparseMatrix<double> >(non_zero_per_row, ncv);
    for(int i=0;i<nverts;i++)
      if(vertex_coarse[i]!=-1) {
	prol->GetRowIndices(i)[0] = vertex_coarse[i];
	prol->GetRowValues(i)[0] = 1.0;
      }    
    return std::move(prol);
  }



  
  shared_ptr<BaseSparseMatrix>
  BuildH1HierarchicProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				 const shared_ptr<const BaseAlgebraicMesh> & acmesh,
				 const shared_ptr<const CoarseMapping> & cmap)
  {
    static Timer t1("ASCAMG - BuildProlongation - Hierarchic");
    RegionTimer rt1(t1);

    const auto & mesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (fmesh);
    const auto & eqc_h = mesh->GetEQCHierarchy();

    const auto & cmesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (acmesh);
    
    int nv = mesh->NV();
    int ncv = cmesh->NV();

    MPI_Comm comm = eqc_h->GetCommunicator();
    int np,rank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    const auto & vmap = cmap->GetVMAP();

    size_t MAX_PER_ROW = 4;
    double MIN_COLLAPSE_WEIGHT = 0.05;
    
    /** 
	- Build edge connectivity matrices (on fine and coare level)
	TODO: this should not be done multiple times, ideally this should come from outside!!
    **/
    // auto build_econ = [](auto & mesh) -> unique_ptr<SparseMatrix<double>>
    //   {
    // 	int nv = mesh->NV();
    // 	Array<int> econ_s(nv);
    // 	econ_s = 0;
    // 	for(auto & edge: mesh->Edges())
    // 	  for(auto l:Range(2))
    // 	    econ_s[edge.v[l]]++;
    // 	Table<INT<2> > econ_i(econ_s);
    // 	econ_s = 0;
    // 	for(auto & edge: mesh->Edges())
    // 	  for(auto l:Range(2))
    // 	    econ_i[edge.v[l]][econ_s[edge.v[l]]++] = INT<2>(edge.v[1-l], edge.id);
    // 	for(auto row:econ_i)
    // 	  QuickSort(row, [](auto & a, auto & b) { return a[0]<b[0]; });
    // 	auto mat = make_unique<SparseMatrix<double> >(econ_s, nv);
    // 	for(auto k:Range(nv)) {
    // 	  auto rinds = mat->GetRowIndices(k);
    // 	  auto rvals = mat->GetRowValues(k);
    // 	  rinds = -1;
    // 	  rvals = -1;
    // 	  for(auto j:Range(econ_i[k].Size())) {
    // 	    rinds[j] = econ_i[k][j][0];
    // 	    rvals[j] = econ_i[k][j][1]; //e-ids
    // 	  }
    // 	}
    // 	return std::move(mat);
    //   };
    // auto cecon = build_econ(cmesh);
    // auto fecon = build_econ(mesh);

    auto cecon = cmesh->GetEdgeConnectivityMatrix();
    auto fecon = mesh->GetEdgeConnectivityMatrix();

    
    /** Create c2f-mapping**/
    Table<int> c2f;
    {
      TableCreator<int> create_c2f(ncv);
      while(!create_c2f.Done()) {
	for(auto k:Range(nv))
	  if(vmap[k]!=-1)
	    create_c2f.Add(vmap[k], k);
	create_c2f++;
      }
      c2f = create_c2f.MoveTable();
    }
    
    ParallelVVector<double> vec_is_single(cmap->GetCParallelDofs(), DISTRIBUTED);
    for(auto k:Range(ncv))
      vec_is_single.FVDouble()[k] = c2f[k].Size();
    ( (BaseVector*) (&vec_is_single) )->Cumulate();
    for(auto k:Range(ncv))
      vec_is_single.FVDouble()[k] = (vec_is_single.FVDouble()[k] == (cmap->GetCParallelDofs()->GetDistantProcs(k).Size()+1))?1.0:0.0;


    /**
       tentative graph+vals:
       - graph: ALL connected cdofs
       - vals: sum of all FINE edge-weights I am master of
    **/
    Table<int> tent_graph;
    Table<double> tent_vals;
    if(nv && ncv)
      {
	// auto & ew = mesh->GetWE();
	TableCreator<int> create_tent_graph(nv);
	TableCreator<double> create_tent_vals(nv);
	size_t sz = 0;
	if(rank)
	  sz = 5*max2(cecon->GetRowIndices(0).Size(), MAX_PER_ROW);
	Array<double> buf(sz);
	auto round = 0;
	Array<int> cri2(buf.Size());
	while(!create_tent_graph.Done()) {
	  for(auto k:Range(nv)) {
	    if(vmap[k]!=-1) {
	      if(vec_is_single.FVDouble()[vmap[k]]==1.0) {
		create_tent_graph.Add(k, vmap[k]);
		continue;
	      }
	      auto cri = cecon->GetRowIndices(vmap[k]);
	      cri2.SetSize(0);
	      for(auto j:Range(cri.Size()))
		if(eqc_h->IsLEQ(mesh->GetEQCOfV(k), cmesh->GetEQCOfV(cri[j])))
		  {
		    cri2.Append(cri[j]);
		  }
	      cri2.Append(vmap[k]);
	      QuickSort(cri2);
	      for(auto j:Range(cri2.Size()))
		create_tent_graph.Add(k, cri2[j]);
	      buf.SetSize(cri2.Size()); //+1 for vmap[me]
	      buf = 0.0;
	      auto fri = fecon->GetRowIndices(k);
	      auto frv = fecon->GetRowValues(k);
	      int pos = -1;
	      for(auto j:Range(fri.Size()))
		if( (vmap[fri[j]]!=-1) &&
		    (-1 != (pos = cri2.Pos(vmap[fri[j]]))) ) {
		  auto eqc_cut = eqc_h->GetCommonEQC(mesh->GetEQCOfV(k), mesh->GetEQCOfV(fri[j]));
		  if( (!eqc_h->GetDistantProcs(eqc_cut).Size()) ||
		      (rank<eqc_h->GetDistantProcs(eqc_cut)[0]) )
		    {
		      /**
			 we should only consider strong connections here
			 graph could come out differently on different procs if
			 COLW - MIN_COLW < eps !!
			 so, give all connections but set weight to 0 if not strong
		      **/
		      if(mesh->GetECW(int(frv[j])) > MIN_COLLAPSE_WEIGHT)
			buf[pos] += mesh->GetEW(frv[j]);
		      // buf[pos] += ew[(int)frv[j]];
		    }
		}
	      for(auto j:Range(buf.Size()))
		create_tent_vals.Add(k, buf[j]);
	    }
	  }
	  create_tent_graph++; create_tent_vals++;
	}
	tent_graph = create_tent_graph.MoveTable();
	tent_vals = create_tent_vals.MoveTable();
      }
    else if(nv && !ncv) {
      Array<int> dummy(nv);
      dummy = 0;
      tent_graph = Table<int>(dummy);
      tent_vals = Table<double>(dummy);
    }
    else {
      Array<int> dummy(0);
      tent_graph = Table<int>(dummy);
      tent_vals = Table<double>(dummy);
    }

    /**
       Accumulate weights
    **/
    Array<size_t> eqcs(nv);
    for(auto k:Range(nv))
      eqcs[k] = mesh->GetEQCOfV(k);
    Table<double> reduced_vals = ReduceTable<double, double>(tent_vals, eqcs, eqc_h, [](auto tab) {
	Array<double> out;
	if(!tab.Size() || !tab[0].Size())
	  return out;
	out.SetSize(tab[0].Size());
	out = 0.0;
	for(auto k:Range(tab.Size()))
	  for(auto j:Range(tab[k].Size()))
	    out[j] += tab[k][j];
	return out;
      } );
    
    
    /** Create Prolongation-matrix**/
    /** only allow vertices such that weight/sum>0.1 or other dof == coarse of**/
    Array<int> s1(nv);
    for(auto k:Range(nv)) {
      if( ((!reduced_vals[k].Size()) && ((vmap[k])!=-1)) ) {
	s1[k] = 1; continue; }
      else if(vmap[k]==-1) {
	s1[k] = 0; continue; }
      size_t nnz = 1;

      double sumv = 0.0;
      for(auto v:reduced_vals[k])
	if(v>sumv)
	  sumv += v;

      auto ind_other = tent_graph[k].Pos(vmap[k]);

      Array<int> indices(reduced_vals[k].Size());
      for(auto j:Range(indices.Size()))
	indices[j] = j;
      QuickSortI(reduced_vals[k], indices, [](auto a, auto b){ return a>b; });

      size_t found_index = 0;
      sumv = reduced_vals[k][indices[0]];
      int j = 1;
      bool found = false;
      // bool other_in = false;
      if(sumv!=0.0) {
	while( (j<min2(MAX_PER_ROW, reduced_vals[k].Size())) && (!found) ) {
	  auto J = indices[j];
	  // if(indices[j] == ind_other) other_in = true;
	  sumv += reduced_vals[k][J];
	  if(reduced_vals[k][J]/sumv < 0.1)
	    found = true;
	  else
	    j++;
	}      
      }
      // TODO: zero row happens (very) rarely ... why??
      s1[k] = j;
    }
    
    Array<int> epr_dummy (0);
    
    shared_ptr<SparseMatrix<double>> prol = (nv>0) ? make_shared<SparseMatrix<double>>(s1, ncv) :
      make_shared<SparseMatrix<double>>(epr_dummy,ncv);

    /**
       Fill Prolongation-matrix. pick the sizes[k] strongest connections
    **/
    Array<int> indices(50);
    for(auto row:Range(nv)) {
      auto nc = s1[row];
      if( ((!reduced_vals[row].Size()) && ((vmap[row])!=-1)) ||
	  (nc==1) ) { //single V
	prol->GetRowIndices(row)[0] = vmap[row];
	prol->GetRowValues(row)[0] = 1.0;
	continue;
      } //collapsed v
      else if(!nc) //grounded V
	continue;
      
      auto ri = prol->GetRowIndices(row);
      auto rv = prol->GetRowValues(row);
      indices.SetSize(tent_graph[row].Size());
      for(auto k:Range(indices.Size()))
	indices[k] = k;
      QuickSortI(reduced_vals[row], indices, [](auto a, auto b){ return a>b; });

      auto pos_other = indices.Pos(tent_graph[row].Pos(vmap[row]));
      if(pos_other>=s1[row]) {
	swap(indices[s1[row]-1], indices[pos_other]);
      }
      /** ...what??? **/
      for(auto k:Range((size_t)s1[row], tent_graph[row].Size())) {
	tent_graph[row][indices[k]] = 2000*ncv+1;
      }
      QuickSortI(tent_graph[row], indices);

      pos_other = indices.Pos(tent_graph[row].Pos(vmap[row]));
      rv = 0.0;
      double sum = 0.0;
      
      for(auto k:Range(s1[row])) {
	ri[k] = tent_graph[row][indices[k]];
	auto v = reduced_vals[row][indices[k]];
	rv[k] += 0.5*v;
	rv[pos_other] += 0.5*v;
	sum += v;
      }
      for(auto &v:rv)
	v /= sum;
    }
    
    return prol;
  } // end BuildH1HierarchicProlongation

  
  shared_ptr<BaseSparseMatrix>
  BuildlastiPWProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
			    const shared_ptr<const BaseAlgebraicMesh> & cmesh,
			    const shared_ptr<const CoarseMapping> & cmap,
			    Array<Vec<3,double> > & p_coords,
			    Array<Vec<3,double> > & cp_coords)
  {
    /**
       prol-mat for 2 verts:
       1 0  0.5*l*n_x
       1 0 -0.5*l*n_x
       0 1  0.5*l*n_y
       0 1 -0.5*l*n_y
       0 0     1
       0 0     1

       nze/row:
       - 0 if grounded
       - else 1 if not collapsed
       - 2 if collapsed and displ-component
       - 1 if collapsed and rot-component
    **/

    size_t nv = fmesh->NV();
    size_t ncv = cmesh->NV();
    auto& vertex_coarse = cmap->GetVMAP();

    Array<int> non_zero_per_row(3*nv);
    if(nv)non_zero_per_row = 0;

    Array<size_t> fineof(ncv);
    if(ncv)fineof=-1;
    for(auto k:Range(nv)) {
      if (vertex_coarse[k] != -1) {
	if(fineof[vertex_coarse[k]]!=-1) {
	  non_zero_per_row[k] = non_zero_per_row[fineof[vertex_coarse[k]]] = 2; //x-disp
	  non_zero_per_row[nv+k] = non_zero_per_row[nv+fineof[vertex_coarse[k]]] = 2; //y-disp
	  non_zero_per_row[2*nv+k] = non_zero_per_row[2*nv+fineof[vertex_coarse[k]]] = 1; //rota
	  /** !! fineof now has the larger of the two vertices! !!**/
	  fineof[vertex_coarse[k]] = k;
	}
	else {
	  fineof[vertex_coarse[k]] = k;
	}
      }
    }
    for(auto k:Range(nv))
      if( (vertex_coarse[k]!=-1) && (non_zero_per_row[k]==0) ) {
	non_zero_per_row[k] = 1;
	non_zero_per_row[nv+k] = 1;
	non_zero_per_row[2*nv+k] = 1;
      }

    auto prol = make_shared<SparseMatrix<double> >(non_zero_per_row, 3*ncv);

    for(auto k:Range(3*nv)) {
      auto ri = prol->GetRowIndices(k);
      if(!ri.Size()) continue;
      auto rv = prol->GetRowValues(k);
      ri = -1;
      rv = 0.0;
    }
    
    
    for(auto k:Range(nv)) {
      if (vertex_coarse[k] != -1) {
	prol->GetRowIndices(k)[0] = vertex_coarse[k];
	prol->GetRowValues(k)[0] = 1.0;
	prol->GetRowIndices(nv+k)[0] = ncv+vertex_coarse[k];
	prol->GetRowValues(nv+k)[0] = 1.0;
	prol->GetRowIndices(2*nv+k)[0] = 2*ncv+vertex_coarse[k];
	prol->GetRowValues(2*nv+k)[0] = 1.0;
	if(fineof[vertex_coarse[k]]!=k) {
	  // cout << "fineof[" << vertex_coarse[k] << "] = " << fineof[vertex_coarse[k]]
	  //      << "!= " << k << " !!" << endl;
	  auto ko = fineof[vertex_coarse[k]];
	  // auto pvals = pvs[k];
	  Vec<3,double> dif = 0.5*(p_coords[fineof[vertex_coarse[k]]]-p_coords[k]);
	  auto pvals = INT<2,double>(-dif[1], dif[0]);
	  // cout << "get PV for " << INT<2>(k, fineof[vertex_coarse[k]]) << ", vals: " << pvals[0] 
	  //      << " " << pvals[1] << endl;
	  // cout << "orig vals were: " << pvals2[0] << " " << pvals2[1] << endl;
	  // cout << "alternative would be: " << pvals3[0] << " " << pvals3[1] << endl;
	  // TODO: should this be times -1??
	  prol->GetRowIndices(k)[1] = 2*ncv+vertex_coarse[k];
	  prol->GetRowValues(k)[1] = pvals[0];
	  prol->GetRowIndices(ko)[1] = 2*ncv+vertex_coarse[k];
	  prol->GetRowValues(ko)[1] = -pvals[0];
	  prol->GetRowIndices(nv+k)[1] = 2*ncv+vertex_coarse[k];
	  prol->GetRowValues(nv+k)[1] = pvals[1];
	  prol->GetRowIndices(nv+ko)[1] = 2*ncv+vertex_coarse[k];
	  prol->GetRowValues(nv+ko)[1] = -pvals[1];
	  // cout << "prol is now: " << endl << *prol << endl;
	}
      }
    }

    /** for next level **/
    cp_coords = Array<Vec<3,double> > (ncv);
    for(auto k:Range(nv)) {
      if(vertex_coarse[k]==-1) continue;
      if( ( non_zero_per_row[k]==2) && (fineof[vertex_coarse[k]] == k) ) continue;
      cp_coords[vertex_coarse[k]] = 0.5*p_coords[k] + 0.5*p_coords[fineof[vertex_coarse[k]]];
    }

    return prol;
  }




  
  shared_ptr<BaseSparseMatrix>
  BuildElastiHierarchicProlongation
  ( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
    const shared_ptr<const BaseAlgebraicMesh> & acmesh,
    const shared_ptr<const CoarseMapping> & cmap,
    Array<Vec<3,double> > & p_coords,
    Array<Vec<3,double> > & cp_coords,
    HashTable<INT<2>, INT<2,double>>* lamijs,
    const shared_ptr<const BaseMatrix> & amat)
  {
    const auto & mesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (fmesh);
    const auto & eqc_h = mesh->GetEQCHierarchy();
    const auto & cmesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (acmesh);
    
    auto pmat = dynamic_pointer_cast<const ParallelMatrix>(amat);
    auto spmat = dynamic_pointer_cast<const SparseMatrix<double>>(pmat->GetMatrix());
    shared_ptr<ParallelDofs> pardofs  = pmat->GetParallelDofs();

    auto pwp1 = BuildlastiPWProlongation(fmesh, cmesh, cmap, p_coords, cp_coords);
    shared_ptr<const SparseMatrix<double> > pwprol =
      dynamic_pointer_cast<const SparseMatrix<double>>(pwp1);

    //cout << "PWPROL: " << endl << *pwprol << endl;
    
    int nv = mesh->NV();
    int ncv = cmesh->NV();

    MPI_Comm comm = eqc_h->GetCommunicator();
    int np,rank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    const auto & vmap = cmap->GetVMAP();

    size_t MAX_PER_ROW = 3;
    double MIN_COLLAPSE_WEIGHT = 0.05;
    
    /** 
  	- Build edge connectivity matrices (on fine and coare level)
  	TODO: this should not be done multiple times, ideally this should come from outside!!
    **/
    auto build_econ = [](auto & mesh) -> unique_ptr<SparseMatrix<double>>
      {
  	int nv = mesh->NV();
  	Array<int> econ_s(nv);
  	econ_s = 0;
  	for(auto & edge: mesh->Edges())
  	  for(auto l:Range(2))
  	    econ_s[edge.v[l]]++;
  	Table<INT<2> > econ_i(econ_s);
  	econ_s = 0;
  	for(auto & edge: mesh->Edges())
  	  for(auto l:Range(2))
  	    econ_i[edge.v[l]][econ_s[edge.v[l]]++] = INT<2>(edge.v[1-l], edge.id);
  	for(auto row:econ_i)
  	  QuickSort(row, [](auto & a, auto & b) { return a[0]<b[0]; });
  	auto mat = make_unique<SparseMatrix<double> >(econ_s, nv);
  	for(auto k:Range(nv)) {
  	  auto rinds = mat->GetRowIndices(k);
  	  auto rvals = mat->GetRowValues(k);
  	  rinds = -1;
  	  rvals = -1;
  	  for(auto j:Range(econ_i[k].Size())) {
  	    rinds[j] = econ_i[k][j][0];
  	    rvals[j] = econ_i[k][j][1]; //e-ids
  	  }
  	}
  	return std::move(mat);
      };
    auto cecon = build_econ(cmesh);
    auto fecon = build_econ(mesh);

    // cout << "cecon: " << endl << *cecon << endl;
    // cout << "fecon: " << endl << *fecon << endl;
    
    /** Create c2f-mapping**/
    Table<int> c2f;
    {
      TableCreator<int> create_c2f(ncv);
      while(!create_c2f.Done()) {
  	for(auto k:Range(nv))
  	  if(vmap[k]!=-1)
  	    create_c2f.Add(vmap[k], k);
  	create_c2f++;
      }
      c2f = create_c2f.MoveTable();
    }
    
    ParallelVVector<double> vec_is_single(cmap->GetCParallelDofs(), DISTRIBUTED);
    for(auto k:Range(ncv))
      vec_is_single.FVDouble()[k] = c2f[k].Size();
    ( (BaseVector*) (&vec_is_single) )->Cumulate();
    for(auto k:Range(ncv))
      vec_is_single.FVDouble()[k] = (vec_is_single.FVDouble()[k] == (cmap->GetCParallelDofs()->GetDistantProcs(k).Size()+1))?1.0:0.0;


    /**
       tentative graph+vals:
       - graph: ALL connected cdofs
       - vals: sum of all FINE edge-weights I am master of
    **/
    Table<int> tent_graph;
    Table<double> tent_vals;
    if(nv && ncv)
      {
  	// auto & ew = mesh->GetWE();
  	TableCreator<int> create_tent_graph(nv);
  	TableCreator<double> create_tent_vals(nv);
  	size_t sz = 0;
  	if(rank)
  	  sz = 5*max2(cecon->GetRowIndices(0).Size(), MAX_PER_ROW);
  	Array<double> buf(sz);
  	auto round = 0;
  	Array<int> cri2(buf.Size());
  	while(!create_tent_graph.Done()) {
  	  for(auto k:Range(nv)) {
  	    if(vmap[k]!=-1) {
  	      if(vec_is_single.FVDouble()[vmap[k]]==1.0) {
  		create_tent_graph.Add(k, vmap[k]);
  		continue;
  	      }
  	      auto cri = cecon->GetRowIndices(vmap[k]);
  	      cri2.SetSize(0);
  	      for(auto j:Range(cri.Size()))
  		if(eqc_h->IsLEQ(mesh->GetEQCOfV(k), cmesh->GetEQCOfV(cri[j])))
  		  {
  		    cri2.Append(cri[j]);
  		  }
  	      cri2.Append(vmap[k]);
  	      QuickSort(cri2);
  	      for(auto j:Range(cri2.Size()))
  		create_tent_graph.Add(k, cri2[j]);
  	      buf.SetSize(cri2.Size()); //+1 for vmap[me]
  	      buf = 0.0;
  	      auto fri = fecon->GetRowIndices(k);
  	      auto frv = fecon->GetRowValues(k);
  	      int pos = -1;
  	      for(auto j:Range(fri.Size()))
  		if( (vmap[fri[j]]!=-1) &&
  		    (-1 != (pos = cri2.Pos(vmap[fri[j]]))) ) {
  		  auto eqc_cut = eqc_h->GetCommonEQC(mesh->GetEQCOfV(k), mesh->GetEQCOfV(fri[j]));
  		  if( (!eqc_h->GetDistantProcs(eqc_cut).Size()) ||
  		      (rank<eqc_h->GetDistantProcs(eqc_cut)[0]) )
  		    {
  		      /**
  			 we should only consider strong connections here
  			 graph could come out differently on different procs if
  			 COLW - MIN_COLW < eps !!
  			 so, give all connections but set weight to 0 if not strong
  		      **/
  		      if(mesh->GetECW(size_t(frv[j])) > MIN_COLLAPSE_WEIGHT)
  			buf[pos] += mesh->GetEW(frv[j]);
  		      // buf[pos] += ew[(int)frv[j]];
  		    }
  		}
  	      for(auto j:Range(buf.Size()))
  		create_tent_vals.Add(k, buf[j]);
  	    }
  	  }
  	  create_tent_graph++; create_tent_vals++;
  	}
  	tent_graph = create_tent_graph.MoveTable();
  	tent_vals = create_tent_vals.MoveTable();
      }
    else if(nv && !ncv) {
      Array<int> dummy(nv);
      dummy = 0;
      tent_graph = Table<int>(dummy);
      tent_vals = Table<double>(dummy);
    }
    else {
      Array<int> dummy(0);
      tent_graph = Table<int>(dummy);
      tent_vals = Table<double>(dummy);
    }

    //cout << "tent_graph: " << endl << tent_graph << endl;
    
    /**
       Accumulate weights
    **/
    Array<size_t> eqcs(nv);
    for(auto k:Range(nv))
      eqcs[k] = mesh->GetEQCOfV(k);
    Table<double> reduced_vals = ReduceTable<double, double>(tent_vals, eqcs, eqc_h, [](auto tab) {
  	Array<double> out;
  	if(!tab.Size() || !tab[0].Size())
  	  return out;
  	out.SetSize(tab[0].Size());
  	out = 0.0;
  	for(auto k:Range(tab.Size()))
  	  for(auto j:Range(tab[k].Size()))
  	    out[j] += tab[k][j];
  	return out;
      } );

    cout << "reduced_vals: " << endl << reduced_vals << endl;

    /** Create Prolongation-matrix**/
    /** only allow vertices such that weight/sum>0.1 or other dof == coarse of**/
    Array<int> s1(3*nv);
    for(auto k:Range(nv)) {
      if( ((!reduced_vals[k].Size()) && ((vmap[k])!=-1)) ) {
  	s1[k] = 1; continue; }
      else if(vmap[k]==-1) {
  	s1[k] = 0; continue; }
      size_t nnz = 1;
      double sumv = 0.0;
      for(auto v:reduced_vals[k])
  	if(v>sumv)
  	  sumv += v;
      auto ind_other = tent_graph[k].Pos(vmap[k]);
      Array<int> indices(reduced_vals[k].Size());
      for(auto j:Range(indices.Size()))
  	indices[j] = j;
      QuickSortI(reduced_vals[k], indices, [](auto a, auto b){ return a>b; });
      size_t found_index = 0;
      sumv = reduced_vals[k][indices[0]];
      int j = 1;
      bool found = false;
      // bool other_in = false;
      if(sumv!=0.0) {
  	while( (j<min2(MAX_PER_ROW, reduced_vals[k].Size())) && (!found) ) {
  	  auto J = indices[j];
  	  // if(indices[j] == ind_other) other_in = true;
  	  sumv += reduced_vals[k][J];
  	  if(reduced_vals[k][J]/sumv < 0.1)
  	    found = true;
  	  else
  	    j++;
  	}      
      }
      // TODO: zero row happens (very) rarely ... why??
      s1[k] = j;
    }

    /** allocate matrix **/
    for(auto k:Range(nv)) {
      auto dps = pardofs->GetDistantProcs(k);
      /** 3x3 mat entries!! **/
      if( (s1[k]>1) && (dps.Size()==0) && false) { 
	s1[nv+k] = (s1[2*nv+k] = (s1[k] = 3 * s1[k]));
      }
      else {
	for(auto j: {k, nv+k, 2*nv+k})
	  s1[j] = pwprol->GetRowIndices(j).Size();
      }
    }    

    //cout << "s1: " << endl << s1 << endl;

    Array<int> epr_dummy (0);
    shared_ptr<SparseMatrix<double>> prol = (nv>0) ? make_shared<SparseMatrix<double>>(s1, 3*ncv) :
      make_shared<SparseMatrix<double>>(epr_dummy,3*ncv);
    /** repurpose it again ... **/
    for(auto k:Range(nv)) {
      auto dps = pardofs->GetDistantProcs(k);
      /** 3x3 mat entries!! **/
      if( (s1[k]>2) && (dps.Size()==0)) {
	s1[nv+k] = (s1[2*nv+k] = (s1[k] = s1[k]/3));
      }
      else if(s1[k]!=0) { /** 1 to one **/
	s1[2*nv+k] = (s1[nv+k] = (s1[k] = 1));
      }
    }    

    /** compute edge-contribution of replacement matrix **/
    auto calc_edge_repl = [](Vec<3, double> & p0, Vec<3, double> & p1,
			     INT<2,double> & lamis, FlatMatrix<double> & mat,
			     LocalHeap & lh)
      {
	FlatVector<double> t(2, lh);
	FlatVector<double> n(2, lh);
	t[0] = p1[0]-p0[0];
	t[1] = p1[1]-p0[1];
	double l = L2Norm(t);
	t /= l;
	n[0] = -t[1];
	n[1] = t[0];
	// cout << "n: " << n[0] << " " << n[1] << endl;
	// cout << "t: " << t[0] << " " << t[1] << endl;
	// cout << "l: " << l << endl;
	double lamA = lamis[0];
	double lamC = lamis[1];
	double llamC = l*lamC;
	double lllamC = l*llamC;
	/** write repl mat in tt/nn/rr base **/
	mat = 0.0;
	mat(0,0) = mat(1,1) = lamA;
	mat(1,0) = mat(0,1) = -lamA;
	mat(2,2) = mat(3,3) = 2*lamC;
	mat(2,3) = mat(3,2) = -2*lamC;
	mat(2,4) = mat(2,5) = mat(4,2) = mat(5,2) = -llamC;
	mat(3,4) = mat(3,5) = mat(4,3) = mat(5,3) = +llamC;
	mat(4,4) = mat(5,5) = lllamC;
	// cout << "t/n/r base edge block: " << endl << mat << endl;
	/** transform to xx/yy/rr base **/
	FlatMatrix<double> trans(6,6,lh);
	trans = 0.0;
	trans(0,0) = trans(1,1) = t(0);
	trans(2,0) = trans(3,1) = t(1);
	trans(0,2) = trans(1,3) = n(0);
	trans(2,2) = trans(3,3) = n(1);
	trans(4,4) = trans(5,5) = 1.0;
	// cout << "trans: " << endl << trans << endl;
	// cout << "Trans(trans): " << endl << Trans(trans) << endl;
	FlatMatrix<double> mat2(6,6,lh);
	mat2 = trans * mat;
	mat = mat2 * Trans(trans);
	return;
      };

    auto prow = [](const auto & row) {
      // for(auto v:row) cout << v << " ";
      // cout << endl;
    };


    // cout << "p_coords: " << endl << p_coords << endl;
    // cout << endl;
    // cout << "NV: " << nv << endl;
    
    
    /** fill matrix **/
    LocalHeap lh(2000000, "Lukas", false); // ~ 2 MB Localheap
    for(auto V:Range(nv)) {
      if(s1[V]==0) continue; //collapsed
      if(s1[V]==1) { //either taken to coarse level or direct pwprol!
	// for(auto j:Array<size_t>({V, nv+V, 2*nv+V})) {
	int inds[3] = {V, nv+V, 2*nv+V};
	for(auto j:inds) {
	  prol->GetRowIndices(j) = pwprol->GetRowIndices(j);
	  prol->GetRowValues(j) = pwprol->GetRowValues(j);
	  // cout << "RI: " << endl << prol->GetRowIndices(j);
	  // cout << "RV: " << endl << prol->GetRowValues(j);
	}	  
	// prol->GetRowIndices(V) = pwprol->GetRowIndices(V);
	// prol->GetRowIndices(nv+V) = pwprol->GetRowIndices(nv+V);
	// prol->GetRowIndices(2*nv+V) = pwprol->GetRowIndices(2*nv+V);
	// prol->GetRowValues(V) = pwprol->GetRowValues(V);
	// prol->GetRowValues(nv+V)[0] = 1.0;
	// prol->GetRowValues(2*nv+V)[0] = 1.0;
	continue;
      }

      HeapReset hr(lh);
      
      FlatArray<int> all_dofs = spmat->GetRowIndices(V);
      FlatMatrix<double> evecs(all_dofs.Size(), all_dofs.Size(), lh);
      FlatVector<double> evals(all_dofs.Size(), lh);
      // cout << "all_dofs for vertex " << V << ": ";
      prow(all_dofs);
      size_t nv_all = all_dofs.Size()/3;
      FlatMatrix<double> ad_block(all_dofs.Size(), all_dofs.Size(), lh);
      ad_block = 0.0;
      FlatMatrix<double> edge_block(6,6,lh);
      Vec<3,double> p0, p1;
      for(auto k:Range(nv_all)) {
	if(all_dofs[k]==V) continue;
	int v1 = all_dofs[k];
	int v2 = V;
	if(v1>v2) swap(v1,v2);
	
	p0 = p_coords[v1];
	p1 = p_coords[v2];

	auto lijs = (*lamijs)[INT<2>(v1,v2)];
	calc_edge_repl(p0, p1, (*lamijs)[INT<2>(v1,v2)], edge_block, lh);

	// cout << "v1: " << v1 << endl;
	// cout << "v2: " << v2 << endl;
	// cout << "p0: " << p0 << endl;
	// cout << "p1: " << p1 << endl;
	// cout << "lamijs: " << lijs[0] << " " << lijs[1] << endl;
	// cout << "edge block: " << endl << edge_block << endl;
	
	FlatArray<int> inds(6,lh);
	auto i1 = all_dofs.Pos(v1);
	auto i2 = all_dofs.Pos(v2);
	for(auto l:Range(3)) {
	  inds[2*l]   = nv_all*l+i1;
	  inds[2*l+1] = nv_all*l+i2;
	}
	for(auto i:Range(6))
	  for(auto j:Range(6))
	    ad_block(inds[i], inds[j]) += edge_block(i,j);

	LapackEigenValuesSymmetric(ad_block, evals, evecs);
	cout << "ad_block evals after " << k << " of " << nv_all << ": "
	     << endl << evals << endl;
	cout << "ad_block evecs after " << k << " of " << nv_all << ": "
	     << endl << evecs << endl;
	
	FlatMatrix<double> evecs(6,6,lh);
	FlatVector<double> evals(6, lh);
	LapackEigenValuesSymmetric(edge_block, evals, evecs);
	cout << "eblock evals: " << endl << evals << endl;
	cout << "eblock evecs: " << endl << evecs << endl;
	
      }

      cout << "ad_block: "<< endl << ad_block << endl;
      // cout << "evecs: " << endl << evecs << endl;


            /** which ones do i take?? **/
      auto trow = tent_graph[V];
      FlatArray<int> indices(trow.Size(), lh);
      for(auto k:Range(trow.Size())) indices[k] = k;
      QuickSortI(reduced_vals[V], indices, [](auto a, auto b){return (a>b); });
      cout << "indices: " << endl << indices << endl;
      Array<int> tcverts(s1[V], lh);
      tcverts.SetSize(0);
      for(auto k:Range(s1[V])) tcverts.Append(trow[indices[k]]);
      if(!tcverts.Contains(vmap[V])) tcverts.Last() = vmap[V];
      QuickSort(tcverts);
      Array<int> tverts(2*tcverts.Size(), lh);
      tverts.SetSize(0);
      for(auto k:Range(tcverts.Size())) {
	auto finevs = c2f[tcverts[k]];
	for(auto j:Range(finevs.Size()))
	  if(all_dofs.Contains(finevs[j]))
	    tverts.Append(finevs[j]);
      }
      QuickSort(tverts);

      cout << "all_dofs for vertex " << V << ": ";
      prow(all_dofs);
      cout << "tcverts: ";
      prow(tcverts);
      cout << "tverts: ";
      prow(tverts);
      cout << "crs[rv]: ";
      for(auto v:tverts) cout << vmap[v] << " ";
      cout << endl;
      cout << "nv_all: " << nv_all << endl;
      cout << "nv_take: " << tverts.Size() << endl;
      
      /** calc schur for tdofs **/
      BitArray barray(all_dofs.Size());
      barray.Clear();
      for(auto k:Range(tverts.Size())) {
	auto pos = all_dofs.Pos(tverts[k]);
	auto offset = all_dofs.Size()/3;
	barray.Set(pos);
	barray.Set(offset + pos);
	barray.Set(2*offset + pos);
      }
      size_t ss = 3 * tverts.Size();
      cout << "take_dofs: " << endl << barray << endl;
      FlatMatrix<double> S(ss,ss,lh);
      if(ss<ad_block.Height())
	CalcSchurComplement(ad_block, S, barray, lh);
      else
	S = ad_block;
      
      cout << "S: " << endl << S << endl; 
      FlatMatrix<double> sevecs(ss,ss,lh);
      FlatVector<double> sevals(ss,lh);
      // for(auto l:Range(2))
      // 	for(auto k:Range(tverts.Size()))
      // 	  for(auto j:Range(tverts.Size()))
      // 	    S(tverts.Size()*l+k, tverts.Size()*l+j) += 1.0;
      LapackEigenValuesSymmetric(S, sevals, sevecs);
      for(auto j:Range(3))
      	for(auto l:Range(2))
      	  for(auto k:Range(tverts.Size()))
      	    sevecs(j, l*tverts.Size()+k) -= sevecs(j, (l+1)*tverts.Size()-1);
		 
      for(auto j:Range(3))
	for(auto k:Range(ss))
	  if(sevecs(j,ss-1)*sevecs(j,ss-1)>1e-16)
	    sevecs(j,k) /= sevecs(j,ss-1);

      for(auto k:Range(ss))
	for(auto j:Range(ss))
	  if(sevecs(k,j)*sevecs(k,j)<1e-16)
	    sevecs(k,j) = 0.0;

      cout << "S evals: " << endl << sevals << endl;
      cout << "S evecs: " << endl << sevecs << endl;
      // for(auto l:Range(2))
      // 	for(auto k:Range(tverts.Size()))
      // 	  for(auto j:Range(tverts.Size()))
      // 	    S(tverts.Size()*l+k, tverts.Size()*l+j) -= 1.0;
      
      FlatVector<double> v1(ss, lh);
      FlatVector<double> v2(ss, lh);

      for(auto k:Range(2)) {
	v1 = 0.0;
	for(auto j:Range(k*ss/3, (k+1)*ss/3))
	  v1[j] = 1.0;
	v2 = S*v1;
	cout << "S*v: " << L2Norm(v2) << endl;
      }
      
      /** Inv diag block **/
      FlatMatrix<double> Dblock(3,3,lh);
      auto mypos = tverts.Pos(V);
      auto offset = tverts.Size();
      FlatArray<int> myinds(3,lh);
      myinds[0] = mypos;
      myinds[1] = offset + mypos;
      myinds[2] = 2*offset + mypos;
      for(auto k:Range(3))
	for(auto j:Range(3))
	  Dblock(k,j) = S(myinds[k], myinds[j]);
      cout << "D:" << endl << Dblock << endl;
      CalcInverse(Dblock);
      cout << "Dinv:" << endl << Dblock << endl;
      
      
      /** **/
      FlatMatrix<double> OD(3,ss-3,lh);
      FlatMatrix<double> DtOD(3,ss-3,lh);
      FlatArray<int> oinds(ss-3,lh);
      size_t s = 0;
      for(auto k:Range(ss))
	if(!myinds.Contains(k)) oinds[s++] = k;
      for(auto k:Range(3))
	for(auto j:Range(oinds.Size()))
	  OD(k,j) = -S(myinds[k], oinds[j]);

      cout << "OD: " << endl << OD << endl;

      DtOD = Dblock * OD;
      cout << "Dinv * OD: " << endl << DtOD << endl;

      cout << "myinds: " << endl << myinds << endl;
      cout << "oinds: " << endl << oinds << endl;
      
      /**  block-rows of (I - 0.5 * D^{-1}A)**/
      FlatMatrix<double> row(3, ss, lh);
      row = 0.0;
      for(auto k:Range(3)) {
      	row(k,myinds[k]) = 0.5;
      	for(auto j:Range(oinds.Size())) {
      	  row(k,oinds[j]) = 0.5*DtOD(k,j);
      	}      
      }

      cout << "row: " << endl << row << endl;
      
      /** pw-prol block **/
      size_t vh = tverts.Size();
      size_t vw = tcverts.Size();
      FlatMatrix<double> P(ss, 3*vw, lh);
      for(auto OS1:Range(3)) {
	for(auto OS2:Range(3)) {
	  for(auto k:Range(vh)) {
	    for(auto j:Range(vw))
	      P(OS1*vh+k,OS2*vw+j) = (*pwprol)(nv*OS1+tverts[k], ncv*OS2+tcverts[j]);
	  }
	}
      }

      cout << "pwp block: " << endl << P << endl;
      
      FlatMatrix<double> rowtP(3, 3*vw, lh);
      rowtP = row * P;

      cout << "rowtP: " << endl << rowtP << endl;

      FlatVector<double> v3(3*vw, lh);
      FlatVector<double> v4(3, lh);
      for(auto k:Range(2)) {
	v3 = 0.0;
	for(auto j:Range(k*vw, (k+1)*vw))
	  v3[j] = 1.0;
	cout << "v3: " << endl << v3 << endl;
	v4 = rowtP*v3;
	cout << "rowtp*v3: " << endl << v4 << endl;
      }
      
      /** fill graph & vals **/
      auto ri1 = prol->GetRowIndices(V);
      auto rv1 = prol->GetRowValues(V);
      auto ri2 = prol->GetRowIndices(nv+V);
      auto rv2 = prol->GetRowValues(nv+V);
      auto ri3 = prol->GetRowIndices(2*nv+V);
      auto rv3 = prol->GetRowValues(2*nv+V);
      for(auto k:Range(s1[V])) {
	auto off = ri1.Size()/3;
	for(auto l:Range(3)) {
	  ri1[l*off+k] = (ri2[l*off+k] = (ri3[l*off+k] = l*ncv+tcverts[k]));
	  rv1[l*off+k] = rowtP(0, l*off+k);
	  rv2[l*off+k] = rowtP(1, l*off+k);
	  rv3[l*off+k] = rowtP(2, l*off+k);
	}
      }


      cout << "ris: " << endl;
      prow(ri1);
      prow(ri2);
      prow(ri3);
      cout << "rvs: " << endl;
      prow(rv1);
      prow(rv2);
      prow(rv3);

      
      
    } // end V:Range(nv)

    cout << "prol: " << endl << *prol << endl;
    VVector<double> v1(3*ncv);
    VVector<double> v2(3*nv);
    for(auto k:Range(2)) {
      v1 = 0.0;
      for(auto j:Range(k*ncv, (k+1)*ncv))
	v1.FVDouble()[j] = 1.0;
      prol->Mult(v1,v2);
      cout << "NV NCV: "<< nv << " " << ncv << endl;
      cout << "trans " << k << ": " << endl;
      for(auto j:Range(nv)) {
	cout << j << ": " << v2.FVDouble()[j] << " " << v2.FVDouble()[nv+j]
	     << " " << v2.FVDouble()[2*nv+j] << endl;
      }
      cout << endl;
    }
    
    
    return prol;
  }


  /** 3d vertsion smoothed **/
  shared_ptr<BaseSparseMatrix>
  BuildElastiHierarchicProlongation
  ( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
    const shared_ptr<const BaseAlgebraicMesh> & acmesh,
    const shared_ptr<const CoarseMapping> & cmap,
    Array<Vec<3,double> > & p_coords,
    Array<Vec<3,double> > & cp_coords,
    HashTable<INT<2>, INT<4,double>>* lamijs,
    const shared_ptr<const BaseMatrix> & amat)
  {
    return nullptr;
  }


  
  /** 
      pwp: piecewise prolongation
      bsize: # dofs per block
      get_repl(k,j, mat):
         writes replacement-edge-mat (dim 2*bsize) for dof-blocks k,j
	 into mat

      (currently broken for hierarchic collapse!)
      (uses collapse-weight of edges to decide on graph)
  **/
  shared_ptr<BaseSparseMatrix>
  BuildVertexBasedHierarchicProlongation
  (const shared_ptr<const BaseAlgebraicMesh> & afmesh,
   const shared_ptr<const BaseAlgebraicMesh> & acmesh,
   const shared_ptr<const CoarseMapping> & cmap,
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwprol,
   size_t bsize, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl)
  {
    static Timer t("BuildVertexBasedHierarchicProlongation");
    RegionTimer rt(t);
    
    const auto & mesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (afmesh);
    const auto & eqc_h = mesh->GetEQCHierarchy();
    const auto & cmesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (acmesh);
    const auto & vmap = cmap->GetVMAP();

    const size_t MAX_PER_ROW = 4;
    double MIN_COLLAPSE_WEIGHT = 0.05;
    double MIN_PROL_FRAC = 0.1;

    size_t nv = mesh->NV();
    size_t ncv = cmesh->NV();
    
    Table<size_t> tent_graph;
    Array<int> sz;

    // col-wt for coarse VERT is only onesided if CROSS-COLL!! -> piggyback off of merged!!
    ParallelVVector<double> merged(cmap->GetCParallelDofs(), DISTRIBUTED);
    if(ncv) merged.FVDouble() = 0.0;
    BitArray has_loc_partner(ncv);
    has_loc_partner.Clear();

    const auto & edges = mesh->Edges();
    const auto & cedges = mesh->Edges();

    if(nv && ncv) {
      TableCreator<INT<2, double>> ctt(nv);
      while(!ctt.Done()) {
	for(auto k:Range(nv)) {
	  // in case of cross-collapse, have to force coarse V to be in graph!
	  auto cv = vmap[k];
	  if(cv!=-1) ctt.Add(k, INT<2,double>(cv, 0.0));
	}
	for(const auto & edge : edges) { //broken with collapsed cross-edges!
	  auto ew = mesh->GetECW(edge);
	  INT<2, size_t> cv(vmap[edge.v[0]], vmap[edge.v[1]]);
	  if( (cv[0]==-1) || (cv[1]==-1) ) continue;
	  INT<2, size_t> eqcs (mesh->GetEQCOfV(edge.v[0]),
			       mesh->GetEQCOfV(edge.v[1]));
	  if( (cv[0]==cv[1]) ) {// should be irrelevant
	    auto edge_eq = eqc_h->GetCommonEQC(eqcs[0], eqcs[1]);
	    if(eqc_h->IsMasterOfEQC(edge_eq)) {
	      merged.FVDouble()[cv[0]] = ew;
	    }
	    continue;
	  }
	  /** do not use weak connections!! **/
	  if(ew < MIN_COLLAPSE_WEIGHT) continue;
	  // INT<2, size_t> ceqcs = (cmesh->GetEQCOfV(cv[0]),
	  // 			  cmesh->GetEQCOfV(cv[1]));
	  for(auto l:Range(2)) {
	    // if(eqc_h->IsLEQ(eqcs[l], ceqcs[1-l])) {
	    if(eqc_h->IsLEQ(eqcs[l], eqcs[1-l])) { // see used_verts!
	      ctt.Add(edge.v[l], INT<2, double>(cv[1-l], ew));
	    }
	  }	  
	}
	ctt++;
      }
      auto ttab = ctt.MoveTable(); // can have duplicate entries!!

      /** 
	  - get rid of duplicate entries (cumulate ewts) 
	  - sort by weight (except own coarse is always strongest!)
	  - take strongest
      **/
      merged.Cumulate(); // <-- !! actually, does nothing right now because only blockwise crsening
      TableCreator<size_t> ctt2(nv);
      sz = Array<int>(ttab.Size());
      sz = 0;
      size_t rnd = 0;
      while(!ctt2.Done()) {
	Array<int > crowis(20);
	Array<INT<2,double> > crow(20);
	for(auto rnr:Range(nv)) {
	  if(vmap[rnr]==-1) continue;
	  if(merged.FVDouble()[vmap[rnr]] == 0.0) {
	    sz[rnr] = 1; // coarse BF==fine BF
	    ctt2.Add(rnr, vmap[rnr]);
	    continue;
	  }
	  crow.SetSize(0);
	  crowis.SetSize(0);
	  auto row = ttab[rnr];
	  for(auto j:Range(row.Size()))
	    if(!crowis.Contains((int)row[j][0])) {
	      crowis.Append(row[j][0]);
	      crow.Append(INT<2,double>(row[j][0], 0.0));
	    }
	  for(auto j:Range(row.Size())) {
	    // if(crowis[j] == vmap[rnr]) continue; //piggybacked weight!
	    auto pos = crowis.Pos(row[j][0]);
	    crow[pos][1] += row[j][1];
	  }
	  size_t vc = vmap[rnr];
	  QuickSort(crow, [vc](const auto &a, const auto & b) {
	      if(a[0]==b[0]) return false; // if compare to self
	      if(a[0]==vc) return true;
	      if(b[0]==vc) return false;
	      return (a[1]>b[1]);
	    });
	  if(!crow.Size()) continue;
	  if(crow.Size()==1) {
	    ctt2.Add(rnr, crow[0][0]);
	    sz[rnr] = 1;
	    continue;
	  }
	  size_t rsz = 1;
	  double sum = merged.FVDouble()[vc]; //piggybacked!! 
	  for(int j:Range((size_t)1, crow.Size())) {
	    sum += crow[j][1];
	    if(crow[j][1]/sum<MIN_PROL_FRAC) break;
	    rsz++;
	  }
	  rsz = min2(rsz, MAX_PER_ROW);
	  sz[rnr] = rsz;
	  for(auto j:Range(rsz)) {
	    ctt2.Add(rnr, crow[j][0]);
	  }
	}
	ctt2++;
      }
      tent_graph = ctt2.MoveTable();
      for(auto k:Range(nv)) QuickSort(tent_graph[k]);
    }
    else {
      sz = Array<int>(nv);
      if(nv) sz = 0;
      tent_graph = Table<size_t>(sz);
    }

    // {
    //   Array<size_t> teqcs(nv);
    //   Table<INT<2,int>> mtg;
    //   {
    // 	Array<int> tsz(nv);
    // 	for(auto k:Range(nv))
    // 	  tsz[k] = tent_graph[k].Size();
    // 	mtg = Table<INT<2,int>>(tsz);
    // 	for(auto k:Range(nv)) {
    // 	  teqcs[k] = mesh->GetEQCOfV(k);
    // 	  for(auto j:Range(tsz[k])) mtg[k][j] = INT<2,int>(cmesh->MapVToEQC(tent_graph[k][j]),
    // 							   eqc_h->GetEQCID(cmesh->GetEQCOfV(tent_graph[k][j])));
    // 	}
    //   }
    //   auto tg2 = ReduceTable<INT<2,int>, INT<2,int>>(mtg, teqcs, eqc_h, [&cntr](auto & tab) {
    // 	  Array<INT<2,int>> out;
    // 	  if(!tab.Size()) return out;
    // 	  bool ok = true;
    // 	  out = Array<int>(tab[0].Size());
    // 	  out = tab[0];
    // 	  for(auto k:Range(tab.Size())) {
    // 	    if(tab[k].Size()!=tab[0].Size()) {
    // 	      return out;
    // 	    }
    // 	  }
    // 	  for(auto k:Range(tab.Size())) {
    // 	    for(auto j:Range(tab[0].Size())) {
    // 	      if(tab[k][j]!=tab[0][j]) {
    // 		return out;
    // 	      }
    // 	    }
    // 	  }
    // 	  return out;
    // 	});
    // }
    
    /** 
	Reducing graph is not necessary because the only non-upward connection
	we can prolongate from is the one to partner, which is the strongest 
	one by definition!!
    **/

    
    /** allocate matrix **/
    size_t ndof = nv * bsize;
    Array<int> psz(ndof);
    BitArray calcit(nv);
    calcit.Clear();
    for(auto k:Range(nv)) {
      if(sz[k] == 0) { // drops out
	for(auto l:Range(bsize))
	  psz[k+l*nv] = 0;
      }
      else if(merged.FVDouble()[vmap[k]]==0) { // coarse BF == fine BF
	for(auto l:Range(bsize))
	  psz[k+l*nv] = 1;
      }
      else if( (sz[k] == 1) ) { // ||(cmap->GetParallelDofs()->GetDistantProcs(k).Size()) ){ // no strong connectiosn
	for(auto l:Range(bsize))
	  psz[k+l*nv] = pwprol->GetRowIndices(k+l*nv).Size();
      }
      else if(sz[k] > 1) {
	/** 
	    actually prols from others (assume dense block in prol because
	    of diagonal inverse!! 
	**/
	calcit.Set(k);
	size_t s = sz[k]*bsize;
	for(auto l:Range(bsize))
	  psz[k+l*nv] = s;
      }
    }
    auto prol = make_shared<SparseMatrix<double>>(psz, bsize*ncv);
    /** 
	get vertex-patches used for prol
	
	cumulate repl-blocks for [vertex, partner], because we might not
	actually have this available for a collapsed cross-edge
    **/
    // (fine) verices used for prol (including vertex itself!!);
    Table<int> used_verts;
    // (fine) edges   used for prol (including partner only if edge is IN!!);
    Table<int> used_edges; 
    {
      TableCreator<INT<2,int>> cuv(nv);
      while(!cuv.Done()) {
	const auto & edges = mesh->Edges();
	for(const auto & edge : edges) {
	  auto ew = mesh->GetECW(edge);
	  for(auto l:Range(2)) {
	    if( (calcit.Test(edge.v[l])) && (vmap[edge.v[1-l]]!=-1) &&
		(tent_graph[edge.v[l]].Contains(vmap[edge.v[1-l]])) &&
		(eqc_h->IsLEQ(mesh->GetEQCOfV(edge.v[l]), mesh->GetEQCOfV(edge.v[1-l]))) ) {
	      /**
		 we can only take FINE connections that go upwards in the hierarchy 
		 (the COARSE connection can go upward even if the fine one does not!!)
	      **/
	      cuv.Add(edge.v[l], INT<2,int>(edge.v[1-l], edge.id));
	    }
	  }
	}
	for(auto k:Range(nv)) if(calcit.Test(k)) cuv.Add(k, INT<2, int>(k, -1));
	cuv++;
      }
      auto uvtab = cuv.MoveTable();
      Array<size_t> rs(nv);
      for(auto k:Range(nv)) {
	QuickSort(uvtab[k], [](const auto & a, const auto & b)
		  { return a[0]<b[0]; });
	rs[k] = uvtab[k].Size();
      }
      used_verts = Table<int>(rs);
      used_edges = Table<int>(rs);
      for(auto k:Range(nv)) {
	for(auto j:Range(rs[k])){
	  used_verts[k][j] = uvtab[k][j][0];
	  used_edges[k][j] = uvtab[k][j][1];
	}
      }
    }

    /** finally, fill matrix **/
    LocalHeap lh(2000000, "Lukas", false); // ~2 MB LocalHeap
    Array<size_t> offset(bsize);
    Array<size_t> coffset(bsize);
    Array<int> r1({0, 2, 4, 6, 8, 10});
    Array<int> r2({1, 3, 5, 7, 9, 11});
    // off-diagonal block for dist partner
    // FlatMatrix<double> admat(bsize, bsize, lh);
    for(auto l:Range(bsize)) {
      offset[l] = l*nv;
      coffset[l] = l*ncv;
    }

    // {
    //   INT<4> counts;
    //   for(auto l:Range(4)) counts[l] = 0;
    //   for(auto k:Range(nv)) {
    // 	if(psz[k]==0) {
    // 	  cout << "vertex " << k << " drops!!" << endl;
    // 	  counts[0]++;
    // 	  continue; // drops out
    // 	}
    // 	if(merged.FVDouble()[vmap[k]]==0) { // coarse BF == fine BF
    // 	  counts[1]++;
    // 	  cout << "vertex " << k << " SINGLE!!" << endl;
    // 	  continue;
    // 	}
    // 	if(!calcit.Test(k)) { // no strong connections
    // 	  counts[2]++;
    // 	  cout << "vertex " << k << " NO STRONG!!" << endl;
    // 	  continue;
    // 	}
    // 	counts[3]++;
    // 	cout << "vertex " << k << " CALC!!" << endl;
    //   }
    //   cout << "S-prol categories: drop single nostrong calc: " << counts << endl;
    // }

    
    for(auto k:Range(nv)) {
      if(psz[k]==0) continue; // drops out
      if(merged.FVDouble()[vmap[k]]==0) { // coarse BF == fine BF
	for(auto l:Range(bsize)) {
	  prol->GetRowIndices(k+offset[l])[0] = vmap[k]+coffset[l];
	  prol->GetRowValues(k+offset[l])[0] = 1.0;
	}
	continue;
      }
      if(!calcit.Test(k)) { // no strong connections
	for(auto l:Range(bsize)) {
	  auto ri0 = prol->GetRowIndices(k+offset[l]);
	  auto rv0 = prol->GetRowValues(k+offset[l]);
	  auto ri1 = pwprol->GetRowIndices(k+offset[l]);
	  auto rv1 = pwprol->GetRowValues(k+offset[l]);
	  ri0 = ri1;
	  rv0 = rv1;
	}
	continue;
      }

      /** actually calc!! **/
      HeapReset hr(lh);

      auto uverts = used_verts[k]; 
      size_t unv = uverts.Size(); // # of vertices used
      size_t und = bsize * unv; // # of dofs used
      FlatArray<int> oss(bsize, lh);
      for(auto l:Range(bsize)) oss[l] = l*unv;

      // FlatMatrix<double> fullblock(unv*bsize, unv*bsize, lh);
      // fullblock = 0.0;
      FlatMatrix<double> mat(bsize, unv*bsize, lh);
      mat = 0.0;
      FlatMatrix<double> block(2*bsize, 2*bsize, lh);
      FlatArray<int> odc(bsize, lh);
      // FlatArray<int> au(2*bsize, lh);
      auto posk = uverts.Pos(k);
      bool hadit = false;
      for(auto l:Range(unv)) {
	if(k==uverts[l]) continue;
	// if(vmap[uverts[l]] == vmap[k]) hadit = true;
	auto v0 = min2((int)k, uverts[l]);
	auto v1 = max2((int)k, uverts[l]);
	bool flip = get_repl(edges[used_edges[k][l]], block);
	// flip = flip != (v1==k);
	flip = v1==k;
	const Array<int> & rk = flip ? r2 : r1;
	for(auto j:Range(bsize))
	  odc[j] = posk+oss[j];
	mat.Cols(odc) += block.Rows(rk).Cols(rk);
	const Array<int> & rl = flip ? r1 : r2;
	for(auto j:Range(bsize))
	  odc[j] = l+oss[j];
	mat.Cols(odc) = block.Rows(rk).Cols(rl);
	// int cau = 0;
	// int pos0 = (k<uverts[l]) ? posk : l;
	// int pos1 = (k<uverts[l]) ? l : posk;
	// for(auto j:Range(bsize)) {
	//   au[cau++] = pos0+oss[j];
	//   au[cau++] = pos1+oss[j];
	// }
	// fullblock.Rows(au).Cols(au) += block;
      }

      // {
      // 	FlatMatrix<double> evecs(unv*bsize, unv*bsize, lh);
      // 	FlatVector<double> evals(unv*bsize, lh);
      // 	LapackEigenValuesSymmetric(fullblock, evals, evecs);
      // 	int nkernel = 0;
      // 	for(auto v:evals) if(sqr(v)<sqr(5e-15)) nkernel++;
      // 	for(auto k:Range(nkernel)) {
      // 	  double fac = 1.0;
      // 	  for(int j=0;j<unv*bsize&&fac==1.0;j++)
      // 	    if(sqr(evecs(k,j))>1e-8)
      // 	      fac = evecs(k,j);
      // 	  for(auto l:Range(unv*bsize))
      // 	    if(sqr(evecs(k,l))<sqr(1e-14)) evecs(k,l) = 0.0;
      // 	    else evecs(k,l) /= fac;
      // 	}
      // }
	
      
      FlatArray<int> rdiag(bsize, lh);
      for(auto l:Range(bsize))
	rdiag[l] = posk+oss[l];
      FlatArray<int> rodiag((unv-1)*bsize, lh);
      size_t cc = 0;
      size_t cc2 = 0;
      for(auto j:Range(und)) {
	if( (cc>=bsize) || (j!=rdiag[cc]) )rodiag[cc2++] = j;
	else cc++;
      }
      /** diagonal block! **/
      FlatMatrix<double> inv(bsize, bsize, lh); //saves the diagonal block
      FlatMatrix<double> row(bsize, unv*bsize, lh);
      row = 0;
      inv = mat.Cols(rdiag);

      /** 
	  I CANNOT DO IT THIS WAY !!
	  (for elast: rot-flip for partner)
	  We just ignore the (vert, partner)-term, but this should not
	  be too bad because we have a dampening parameter
	  If this is the ONLY good connection, we do not "calcit"
	  but simply pwprol it anyways
       **/
      // if(!hadit) { // non-local conntection
      // 	auto flip = get_repl(idedge({bare_edge(k,-1),size_t(-1)}), block);
      // 	cout << "take block " << (flip?2:1) << " for vertex " << k << endl;
      // 	const Array<int> & rd = flip ? r2 : r1;
      // 	const Array<int> & ro = flip ? r1 : r2;
      // 	// inv += block.Rows(rd).Cols(rd);
      // 	// admat = block.Rows(rd).Cols(ro);
      // }
      CalcInverse(inv);
      // if(!hadit) row.Cols(rdiag) = -0.5 * inv * admat;
      for(auto j:Range(bsize)) row(j, rdiag[j]) += 0.5;
      row.Cols(rodiag) = -0.5 * inv * mat.Cols(rodiag);

      /** pwprol-block **/
      Array<int> ucverts(sz[k]);
      size_t uncv = ucverts.Size();
      for(auto l:Range(uncv)) ucverts[l] = tent_graph[k][l];
      FlatMatrix<double> pwpblock(unv*bsize, sz[k]*bsize, lh);
      const SparseMatrix<double> & pwp(*pwprol);
      size_t pos = 0;
      for(auto lk:Range(bsize)) {
	for(auto k:Range(unv)) {
	  auto rip = pwp.GetRowIndices(uverts[k]+offset[lk]);
	  auto rvp = pwp.GetRowValues(uverts[k]+offset[lk]);
	  for(auto j:Range(uncv)) {
	    for(auto lj:Range(bsize)) {
	      if((pos = rip.Pos(ucverts[j]+coffset[lj])) == -1)
		pwpblock(k+unv*lk, j+uncv*lj) = 0.0;
	      else
		pwpblock(k+unv*lk, j+uncv*lj) = rvp[pos];
	      // pwpblock(k+unv*lk, j+uncv*lj) = pwp(uverts[k]+offset[lk],
	      // 					  ucverts[j]+coffset[lj]);
	    }
	  }
	}
      }
      
      FlatMatrix<double> vals(bsize, sz[k]*bsize, lh);
      vals = row * pwpblock | Lapack;

      for(auto l:Range(bsize)) {
	auto ri = prol->GetRowIndices(k+offset[l]);
	auto rv = prol->GetRowValues(k+offset[l]);
	size_t s = 0;
	for(auto lj:Range(bsize)) {
	  for(auto j:Range(uncv)) {
	    ri[s] = ucverts[j]+coffset[lj]; 
	    rv[s] = vals(l, s);
	    s++;
	  }
	}
      }

      // {
      // 	FlatMatrix<double> ctmp(unv*bsize, sz[k]*bsize, lh);
      // 	FlatMatrix<double> cblock(sz[k]*bsize, sz[k]*bsize, lh);
      // 	FlatMatrix<double> evecs2(sz[k]*bsize, sz[k]*bsize, lh);
      // 	FlatVector<double> evals2(sz[k]*bsize, lh);

      // 	ctmp = fullblock * pwpblock;
      // 	cblock = Trans(pwpblock) * ctmp;
      // 	LapackEigenValuesSymmetric(cblock, evals2, evecs2);
      // 	nkernel = 0;
      // 	for(auto v:evals2) if(sqr(v)<sqr(5e-15)) nkernel++;
      // 	for(auto k:Range(nkernel)) {
      // 	  double fac = 1.0;
      // 	  for(int j=0;j<sz[k]*bsize&&fac==1.0;j++)
      // 	    if(sqr(evecs2(k,j))>1e-8)
      // 	      fac = evecs2(k,j);
      // 	  for(auto l:Range(sz[k]*bsize))
      // 	    if(sqr(evecs2(k,l))<sqr(1e-14)) evecs2(k,l) = 0.0;
      // 	    else evecs2(k,l) /= fac;
      // 	}
      // 	cout << "CBLOCK nr kernel: " << nkernel << endl;
      // 	cout << "all evals: ";for(auto v:evals2) cout << v << " "; cout << endl;
      // 	cout << "fullblock kernel vecs:" << endl << Trans(evecs2.Rows(0, nkernel)) << endl;
      // 	cout << "cblock is: " << endl << cblock << endl;
      // }
    }
    
    return prol;
  }


  shared_ptr<BaseSparseMatrix>
  BuildVertexBasedHierarchicProlongation2
  (const shared_ptr<const BaseAlgebraicMesh> & afmesh,
   const shared_ptr<const BaseAlgebraicMesh> & acmesh,
   const shared_ptr<const CoarseMapping> & cmap,
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwprol,
   const shared_ptr<BitArray> & freedofs,
   int BS, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl,
   std::function<double(size_t, size_t)> get_wt)
  {
    static Timer t("Vertex-based H-prol (v2)");
    RegionTimer rt(t);

    const auto & mesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (afmesh);
    auto fecon = mesh->GetEdgeConnectivityMatrix();
    const auto & eqc_h = mesh->GetEQCHierarchy();
    const auto & cmesh = dynamic_pointer_cast<const BlockedAlgebraicMesh> (acmesh);
    auto cecon = cmesh->GetEdgeConnectivityMatrix();
    const auto & vmap = cmap->GetVMAP();
    const auto & emap = cmap->GetEMAP();

    const size_t MAX_PER_ROW = 4;
    double MIN_PROL_FRAC = 0.1;
    double MIN_PROL_WT = 0.025;
    size_t NV = mesh->NV();
    size_t NCV = cmesh->NV();

    cout << "NV " << NV << " CNV " << NCV << endl;
    
    if(freedofs) {
      cout << "got freedofs: " << endl << *freedofs << endl;
      cout << "not free: " << endl;
      for(auto k:Range(freedofs->Size())) if(!freedofs->Test(k)) cout << k << " ";
      cout << endl;
    }
    
    /**
       For each vertex (that is free), we solve a local Problems.
       We take into account a set of (non-grounded) neighbours of each vertex, such that
       they map to at most MAX_PER_ROW coarse vertices. We use a dampening parameter.
              - A grounded neighbour is a zero-row in the PW prolongation, so we ignore it.
              - Every collapsed vertex always prolongates from it's coarse vertex. 
	  - QUESTION: For a grounded , FREE vertex, should we ignore the dampening parameter??
                        (I think so, because otherwise we dont preserve RBMs!)
	  - QUESTION: What do we do for non-grounded, non-collapsed vertices?
	                (For now, just take the pw-prol 1:1 ... )
     **/
    
    /** 
	If cross-collapse, we do not necessarily know which vertices are actually collapsed!
	If a cross-edge is collapsed, all ranks need to know it's collapse-weight.
	We have to share this information.
    **/
    ParallelVVector<double> colv_wts(cmap->GetParallelDofs(), DISTRIBUTED);
    colv_wts.FVDouble() = 0.0;
    const auto & fedges = mesh->Edges();
    {
      INT<2, size_t> cv;
      auto doit = [&](const auto & the_edges) {
	for(const auto & edge : the_edges) {
	  if( (emap[edge.id]!=-1) ||
	      ((cv[0]=vmap[edge.v[0]]) == -1 ) ||
	      ((cv[1]=vmap[edge.v[1]]) == -1 ) ||
	      (cv[0]!=cv[1]) ) continue;
	  auto com_wt = max2(get_wt(edge.id, edge.v[0]),get_wt(edge.id, edge.v[1]));
	  colv_wts.FVDouble()[edge.v[0]] = com_wt;
	  colv_wts.FVDouble()[edge.v[1]] = com_wt;
	  // cv_wts.FVDouble()[cv[0]] = get_wt(edge.id);
	}
      };
      for(auto eqc:Range(eqc_h->GetNEQCS())) {
	if(!eqc_h->IsMasterOfEQC(eqc)) continue;
	doit((*mesh)[eqc]->Edges());
	doit(mesh->GetPaddingEdges(eqc));
      }
    }
    colv_wts.Cumulate();
    
    /**
       Build the vertex-graph.
	    - For V, if V is not free, take nothing! 
	    - For V, if V is not collapsed or grounded, take only coarse(V)
       For grounded+free or collapsed vertices we do:
           We build tentative rows with (cv, weight) entries:
	        - For V, if V is collapsed, coarse(V) is never in the tentative row (colw is in cv_colwts)
	        - For V, add (coarse(v2),colwt of [v,v2]) if eqc(V) <= eqc(v2)
           We get rid of duplicate entries in the tentative row (sum up weights), then sort by descending weight.
	   We take the strongest from the tentative row, but only if  
                                    ### weight > MIN_FRAC * sum_weights ###
	   (We need cv_colwts because they belong to sum!)
	   For collapsed, we take at most MAX_PER_ROW-1, then add coarse(V).
	   For grounded+free, we take at most MAX_PER_ROW
     **/
    Table<size_t> v_graph(NV, MAX_PER_ROW);
    v_graph.AsArray() = -1;
    Array<int> vgs(NV);
    vgs = 0;
    {
      Array<INT<2,double>> trow;
      Array<INT<2,double>> tcv;
      Array<size_t> fin_row;
      for(auto V:Range(NV)) {
	if(freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if( (CV!=-1) && (colv_wts.FVDouble()[V]==0.0) ){
	  vgs[V] = 1;
	  v_graph[V][0] = CV;
	  continue;
	}
	trow.SetSize(0);
	tcv.SetSize(0);
	auto EQ = mesh->GetEQCOfV(V);
	auto ovs = fecon->GetRowIndices(V);
	auto eis = fecon->GetRowValues(V);
	size_t pos;
	for(auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if(cov==-1 || cov==CV) continue;
	  auto oeq = mesh->GetEQCOfV(ov);
	  if(eqc_h->IsLEQ(EQ, oeq)) {
	    auto wt = get_wt(eis[j], V);
	    if( (pos=tcv.Pos(cov)) == -1) {
	      trow.Append(INT<2,double>(cov, wt));
	      tcv.Append(cov);
	    }
	    else {
	      trow[pos][1] += wt;
	    }
	  }
	}
	cout << endl;
	cout << "tent row for V " << V << endl;
	for(auto v:trow)
	  cout << v[0] << " " << v[1]  << endl;
	QuickSort(trow, [](const auto & a, const auto & b) {
	    if(a[0]==b[0]) return false;
	    return a[1]>b[1];
	  });
	cout << "sorted tent row for V " << V << endl;
	for(auto v:trow)
	  cout << v[0] << " " << v[1]  << endl;
	if(CV==-1)
	  cout << "spec vert!!" << endl;
	else
	  cout << "colv_wt: " << colv_wts.FVDouble()[V] << endl;
	double cw_sum = (CV!=-1) ? colv_wts.FVDouble()[V] : 0.0;
	fin_row.SetSize(0);
	if(CV!=-1) fin_row.Append(CV); //collapsed vertex
	size_t max_adds = (CV!=-1) ? min2(MAX_PER_ROW-1, trow.Size()) : trow.Size();
	for(auto j:Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if(CV!=-1) {
	    if(fin_row.Size() && (trow[j][1] < MIN_PROL_WT)) break;
	    if(trow[j][1]/cw_sum < MIN_PROL_FRAC) break;
	  }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	cout << "fin row for V " << V << endl << fin_row << endl;
	vgs[V] = fin_row.Size();
	for(auto j:Range(fin_row.Size()))
	  v_graph[V][j] = fin_row[j];
	if(fin_row.Size()==1 && CV==-1) {
	  cout << "whoops for dof " << V << endl;
	}
      }
    }

    cout << "v_graph: " << endl << v_graph << endl;
    
    /**
       Build the graph for the matrix.
       For V, if not free, -> empty row.
       For V, if not coll or ground -> ri-size from pw-prol.
       For V, otherwise, ri are all DOFs of blocks belonging to coarse vertices in vertex-graph.
    **/
    Array<int> nperow(BS*NV);
    auto get_dof = [&] (auto block, auto comp) {
      return NV*comp + block;
    };
    auto get_Cdof = [&] (auto block, auto comp) {
      return NCV*comp + block;
    };
    for(auto V:Range(NV)) {
      if(freedofs && !freedofs->Test(V)) {
	for(auto j:Range(BS))
	  nperow[get_dof(V, j)] = 0;
	continue;
      }
      else if(vgs[V]==1 && vmap[V]!=-1) {
	for(auto j:Range(BS)) {
	  auto dof = get_dof(V, j);
	  nperow[dof] = pwprol->GetRowIndices(dof).Size();
	}
	continue;
      }
      for(auto j:Range(BS)) {
	nperow[get_dof(V,j)] = vgs[V]*BS;
      }
    }
    auto prol = make_shared<SparseMatrix<double>>(nperow, BS*NCV);
      
    /**
       Fill the matrix.
       For V, if not free, skip.
       For V, if not collapsed or grounded, copy ri/rv from pw-prol.
       Otherwise:
       - Find out OTHER fine verts+fine edges we need,
       only take edges [v,V2] if coarse(V2) is in ri
       ## and eqc(v)<=eqc(V2) ##
       (otherwise, the graph might be right, but one side might take into account
       fine connections the other side does not know)
       - If V is collapsed, find the collapsed edge for special treatment
    **/
    {
      LocalHeap lh(2000000, "Lukas", false); // ~2 MB LocalHeap
      Array<int> r1({0, 2, 4, 6, 8, 10});
      Array<int> r2({1, 3, 5, 7, 9, 11});
      Array<size_t> uverts, uedges;
      Array<INT<2,size_t>> uve;
      const auto & edges = mesh->Edges();
      Array<size_t> count_spec;
      for(auto V:Range(NV)) {
	if(freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if(vgs[V]==1 && CV!=-1) {
	  // cgs can still be 1 if CV is -1
	  for(auto l:Range(BS)) {
	    auto dof = get_dof(V,l);
	    prol->GetRowIndices(dof) = pwprol->GetRowIndices(dof);
	    prol->GetRowValues(dof) = pwprol->GetRowValues(dof);
	  }
	  continue;
	}
	cout << "CALC for V " << V << endl;
	uve.SetSize(0);
	auto vg_row = v_graph[V];
	if(CV==-1) count_spec.Append(V);
	// bool is_col = cv_colwts.FVDouble()[CV]!=0.0;
	auto EQ = mesh->GetEQCOfV(V);
	auto all_ov = fecon->GetRowIndices(V);
	auto all_oe = fecon->GetRowValues(V);
	for(auto j:Range(all_ov.Size())) {
	  auto ov = all_ov[j];
	  auto cov = vmap[ov];
	  if(cov==-1) continue;
	  if(!vg_row.Contains(cov)) continue;
	  auto eq = mesh->GetEQCOfV(ov);
	  if(!eqc_h->IsLEQ(EQ, eq)) continue;
	  uve.Append(INT<2>(ov,all_oe[j]));
	}
	uve.Append(INT<2>(V,-1));
	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];});
	uverts.SetSize(uve.Size());
	uedges.SetSize(uve.Size());
	for(auto k:Range(uve.Size())) {
	  uverts[k] = uve[k][0];
	  uedges[k] = uve[k][1];
	}
	cout << "use_v: " << endl << uverts << endl;
	cout << "use_e: " << endl << uedges << endl;
	// calc (I - omega D^-1 A)
	auto posV = uverts.Pos(V);
      	size_t unv = uverts.Size(); // # of vertices used
	size_t und = BS * unv; // # of dofs used
	FlatArray<int> oss(BS, lh);
	for(auto l:Range(BS)) oss[l] = l*unv;
	FlatMatrix<double> mat(BS, unv*BS, lh);
	mat = 0.0;
	FlatMatrix<double> block(2*BS, 2*BS, lh);
	FlatArray<int> odc(BS, lh);
	for(auto l:Range(unv)) {
	  if(l==posV) continue;
	  auto v0 = min2(V, uverts[l]);
	  auto v1 = max2(V, uverts[l]);
	  bool flip = get_repl(edges[uedges[l]], block);
	  cout << "block for V ov " << V << " " << uverts[l] << " (edge " << edges[uedges[l]] << ")"
	       << endl << block << endl;
	  flip = v1==V;
	  const Array<int> & rk = flip ? r2 : r1;
	  for(auto j:Range(BS))
	    odc[j] = posV+oss[j];
	  mat.Cols(odc) += block.Rows(rk).Cols(rk);
	  const Array<int> & rl = flip ? r1 : r2;
	  for(auto j:Range(BS))
	    odc[j] = l+oss[j];
	  mat.Cols(odc) = block.Rows(rk).Cols(rl);
	  cout << "mat now: " << endl << mat << endl;
	}
	FlatArray<int> rdiag(BS, lh);
	FlatArray<int> rodiag((unv-1)*BS, lh);
	for(auto l:Range(BS))
	  rdiag[l] = posV+oss[l];
	size_t cc = 0;
	size_t cc2 = 0;
	for(auto j:Range(und)) {
	  if( (cc>=BS) || (j!=rdiag[cc]) ) rodiag[cc2++] = j;
	  else cc++;
	}
	/** diagonal block! **/
	FlatMatrix<double> inv(BS, BS, lh); //saves the diagonal block
	FlatMatrix<double> row(BS, unv*BS, lh);
	row = 0;
	inv = mat.Cols(rdiag);
	cout << "diag: " << endl << inv << endl;
	CalcInverse(inv);
	cout << "inv diag: " << endl << inv << endl;
	// if(!hadit) row.Cols(rdiag) = -0.5 * inv * admat;
	for(auto j:Range(BS)) row(j, rdiag[j]) += 0.5;
	double alpha = (CV==-1) ? 1.0 : 0.5;
	row.Cols(rodiag) = -alpha * inv * mat.Cols(rodiag);
	/** pwprol-block **/
	Array<int> ucverts(vgs[V]);
	size_t uncv = ucverts.Size();
	for(auto l:Range(uncv)) ucverts[l] = v_graph[V][l];
	FlatMatrix<double> pwpblock(unv*BS, vgs[V]*BS, lh);
	const SparseMatrix<double> & pwp(*pwprol);
	size_t pos = 0;
	for(auto lk:Range(BS)) {
	  for(auto k:Range(unv)) {
	    auto rip = pwp.GetRowIndices(get_dof(uverts[k],lk));
	    auto rvp = pwp.GetRowValues(get_dof(uverts[k],lk));
	    for(auto j:Range(uncv)) {
	      for(auto lj:Range(BS)) {
		if((pos = rip.Pos(get_Cdof(ucverts[j], lj))) == -1)
		  pwpblock(k+unv*lk, j+uncv*lj) = 0.0;
		else
		  pwpblock(k+unv*lk, j+uncv*lj) = rvp[pos];
	      }
	    }
	  }
	}
	cout << "row: " << endl << row << endl;
	cout << "pwb: " << endl << pwpblock << endl;
	FlatMatrix<double> vals(BS, vgs[V]*BS, lh);
	vals = row * pwpblock | Lapack;
	cout << "vals: " << endl << vals << endl;
	for(auto l:Range(BS)) {
	  auto dof = get_dof(V,l);
	  auto ri = prol->GetRowIndices(dof);
	  auto rv = prol->GetRowValues(dof);
	  size_t s = 0;
	  for(auto lj:Range(BS)) {
	    for(auto j:Range(uncv)) {
	      ri[s] = get_Cdof(ucverts[j],lj); 
	      rv[s] = vals(l, s);
	      s++;
	    }
	  }
	}
      }
    cout << "special verts (" << count_spec.Size() << ")" << endl << count_spec << endl;
    }

    
    return prol;
  }


  shared_ptr<SparseMatrix<double>>
  HProl_Vertex
  (const shared_ptr<const BlockedAlgebraicMesh> & fmesh,
   Array<int> & vmap, //CoarseMap & cmap,
   const shared_ptr<ParallelDofs> & fpd, 
   const shared_ptr<ParallelDofs> & cpd, 
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwprol,
   const shared_ptr<BitArray> & freedofs,
   int BS, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl,
   std::function<double(size_t, size_t)> get_wt)
  {
    static Timer t("Vertex-based H-prol (v2)");
    RegionTimer rt(t);

    const auto & mesh = fmesh;
    auto fecon = mesh->GetEdgeConnectivityMatrix();
    const auto & eqc_h = mesh->GetEQCHierarchy();
    // const auto & vmap = cmap.GetMap<NT_VERTEX>();
    // const auto & emap = cmap.GetMap<NT_EDGE>();

    const size_t MAX_PER_ROW = (opts!=nullptr) ? opts->GetProlMaxPerRow() : 4;
    double MIN_PROL_FRAC = (opts!=nullptr) ? opts->GetProlMinFrac() : 0.1;
    double MIN_PROL_WT = (opts!=nullptr) ? opts->GetProlMinWt() : 0.05;
    size_t NV = mesh->NV();
    size_t NCV = 0;
    for(auto v:vmap) if(v+1>NCV) NCV = v+1;

    // cout << "NV " << NV << " CNV " << NCV << endl;

    // if(freedofs) {
    //   cout << "got freedofs: " << endl << *freedofs << endl;
    //   cout << "not free: " << endl;
    //   for(auto k:Range(freedofs->Size())) if(!freedofs->Test(k)) cout << k << " ";
    //   cout << endl;
    // }
    
    /**
       For each vertex (that is free), we solve a local Problems.
       We take into account a set of (non-grounded) neighbours of each vertex, such that
       they map to at most MAX_PER_ROW coarse vertices. We use a dampening parameter.
       - A grounded neighbour is a zero-row in the PW prolongation, so we ignore it.
       - Every collapsed vertex always prolongates from it's coarse vertex. 
       - QUESTION: For a grounded , FREE vertex, should we ignore the dampening parameter??
       (I think so, because otherwise we dont preserve RBMs!)
       - QUESTION: What do we do for non-grounded, non-collapsed vertices?
       (For now, just take the pw-prol 1:1 ... )
    **/
    
    /** 
	If cross-collapse, we do not necessarily know which vertices are actually collapsed!
	If a cross-edge is collapsed, all ranks need to know it's collapse-weight.
	We have to share this information.
    **/
    ParallelVVector<double> colv_wts(fpd, DISTRIBUTED);
    colv_wts.FVDouble() = 0.0;
    const auto & fedges = mesh->Edges();
    {
      INT<2, size_t> cv;
      auto doit = [&](const auto & the_edges) {
	for(const auto & edge : the_edges) {
	  // if( (emap[edge.id]!=-1) ||
	  if( ((cv[0]=vmap[edge.v[0]]) == -1 ) ||
	      ((cv[1]=vmap[edge.v[1]]) == -1 ) ||
	      (cv[0]!=cv[1]) ) continue;
	  auto com_wt = max2(get_wt(edge.id, edge.v[0]),get_wt(edge.id, edge.v[1]));
	  colv_wts.FVDouble()[edge.v[0]] = com_wt;
	  colv_wts.FVDouble()[edge.v[1]] = com_wt;
	  // cv_wts.FVDouble()[cv[0]] = get_wt(edge.id);
	}
      };
      for(auto eqc:Range(eqc_h->GetNEQCS())) {
	if(!eqc_h->IsMasterOfEQC(eqc)) continue;
	doit((*mesh)[eqc]->Edges());
	doit(mesh->GetPaddingEdges(eqc));
      }
    }
    colv_wts.Cumulate();
    
    /**
       Build the vertex-graph.
       - For V, if V is not free, take nothing! 
       - For V, if V is not collapsed or grounded, take only coarse(V)
       For grounded+free or collapsed vertices we do:
       We build tentative rows with (cv, weight) entries:
       - For V, if V is collapsed, coarse(V) is never in the tentative row (colw is in cv_colwts)
       - For V, add (coarse(v2),colwt of [v,v2]) if eqc(V) <= eqc(v2)
       We get rid of duplicate entries in the tentative row (sum up weights), then sort by descending weight.
       We take the strongest from the tentative row, but only if  
       ### weight > MIN_FRAC * sum_weights ###
       (We need cv_colwts because they belong to sum!)
       For collapsed, we take at most MAX_PER_ROW-1, then add coarse(V).
       For grounded+free, we take at most MAX_PER_ROW
    **/
    Table<size_t> v_graph(NV, MAX_PER_ROW);
    v_graph.AsArray() = -1;
    Array<int> vgs(NV);
    vgs = 0;
    {
      Array<INT<2,double>> trow;
      Array<INT<2,double>> tcv;
      Array<size_t> fin_row;
      for(auto V:Range(NV)) {
	if(freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if( (CV!=-1) && (colv_wts.FVDouble()[V]==0.0) ){
	  vgs[V] = 1;
	  v_graph[V][0] = CV;
	  continue;
	}
	trow.SetSize(0);
	tcv.SetSize(0);
	auto EQ = mesh->GetEQCOfV(V);
	auto ovs = fecon->GetRowIndices(V);
	auto eis = fecon->GetRowValues(V);
	size_t pos;
	for(auto j:Range(ovs.Size())) {
	  auto ov = ovs[j];
	  auto cov = vmap[ov];
	  if(cov==-1 || cov==CV) continue;
	  auto oeq = mesh->GetEQCOfV(ov);
	  if(eqc_h->IsLEQ(EQ, oeq)) {
	    auto wt = get_wt(eis[j], V);
	    if( (pos=tcv.Pos(cov)) == -1) {
	      trow.Append(INT<2,double>(cov, wt));
	      tcv.Append(cov);
	    }
	    else {
	      trow[pos][1] += wt;
	    }
	  }
	}
	// cout << endl;
	// cout << "tent row for V " << V << endl;
	// for(auto v:trow)
	//   cout << v[0] << " " << v[1]  << endl;
	QuickSort(trow, [](const auto & a, const auto & b) {
	    if(a[0]==b[0]) return false;
	    return a[1]>b[1];
	  });
	// cout << "sorted tent row for V " << V << endl;
	// for(auto v:trow)
	//   cout << v[0] << " " << v[1]  << endl;
	// if(CV==-1)
	//   cout << "spec vert!!" << endl;
	// else
	//   cout << "colv_wt: " << colv_wts.FVDouble()[V] << endl;
	double cw_sum = (CV!=-1) ? colv_wts.FVDouble()[V] : 0.0;
	fin_row.SetSize(0);
	if(CV!=-1) fin_row.Append(CV); //collapsed vertex
	size_t max_adds = (CV!=-1) ? min2(MAX_PER_ROW-1, trow.Size()) : trow.Size();
	for(auto j:Range(max_adds)) {
	  cw_sum += trow[j][1];
	  if(CV!=-1) {
	    if(fin_row.Size() && (trow[j][1] < MIN_PROL_WT)) break;
	    if(trow[j][1] < MIN_PROL_FRAC*cw_sum) break;
	  }
	  fin_row.Append(trow[j][0]);
	}
	QuickSort(fin_row);
	// cout << "fin row for V " << V << endl << fin_row << endl;
	vgs[V] = fin_row.Size();
	for(auto j:Range(fin_row.Size()))
	  v_graph[V][j] = fin_row[j];
	if(fin_row.Size()==1 && CV==-1) {
	  cout << "whoops for dof " << V << endl;
	}
      }
    }

    // cout << "v_graph: " << endl << v_graph << endl;
    
    /**
       Build the graph for the matrix.
       For V, if not free, -> empty row.
       For V, if not coll or ground -> ri-size from pw-prol.
       For V, otherwise, ri are all DOFs of blocks belonging to coarse vertices in vertex-graph.
    **/
    Array<int> nperow(BS*NV);
    auto get_dof = [&] (auto block, auto comp) {
      return NV*comp + block;
    };
    auto get_Cdof = [&] (auto block, auto comp) {
      return NCV*comp + block;
    };
    for(auto V:Range(NV)) {
      if(freedofs && !freedofs->Test(V)) {
	for(auto j:Range(BS))
	  nperow[get_dof(V, j)] = 0;
	continue;
      }
      else if(vgs[V]==1 && vmap[V]!=-1) {
	for(auto j:Range(BS)) {
	  auto dof = get_dof(V, j);
	  nperow[dof] = pwprol->GetRowIndices(dof).Size();
	}
	continue;
      }
      for(auto j:Range(BS)) {
	nperow[get_dof(V,j)] = vgs[V]*BS;
      }
    }
    auto prol = make_shared<SparseMatrix<double>>(nperow, BS*NCV);
      
    /**
       Fill the matrix.
       For V, if not free, skip.
       For V, if not collapsed or grounded, copy ri/rv from pw-prol.
       Otherwise:
       - Find out OTHER fine verts+fine edges we need,
       only take edges [v,V2] if coarse(V2) is in ri
       ## and eqc(v)<=eqc(V2) ##
       (otherwise, the graph might be right, but one side might take into account
       fine connections the other side does not know)
       - If V is collapsed, find the collapsed edge for special treatment
    **/
    {
      LocalHeap lh(2000000, "Lukas", false); // ~2 MB LocalHeap
      Array<int> r1({0, 2, 4, 6, 8, 10});
      Array<int> r2({1, 3, 5, 7, 9, 11});
      Array<size_t> uverts, uedges;
      Array<INT<2,size_t>> uve;
      const auto & edges = mesh->Edges();
      Array<size_t> count_spec;
      for(auto V:Range(NV)) {
	HeapReset hr(lh);
	if(freedofs && !freedofs->Test(V)) continue;
	auto CV = vmap[V];
	if(vgs[V]==1 && CV!=-1) {
	  // cgs can still be 1 if CV is -1
	  for(auto l:Range(BS)) {
	    auto dof = get_dof(V,l);
	    prol->GetRowIndices(dof) = pwprol->GetRowIndices(dof);
	    prol->GetRowValues(dof) = pwprol->GetRowValues(dof);
	  }
	  continue;
	}
	// cout << "CALC for V " << V << endl;
	uve.SetSize(0);
	auto vg_row = v_graph[V];
	if(CV==-1) count_spec.Append(V);
	// bool is_col = cv_colwts.FVDouble()[CV]!=0.0;
	auto EQ = mesh->GetEQCOfV(V);
	auto all_ov = fecon->GetRowIndices(V);
	auto all_oe = fecon->GetRowValues(V);
	for(auto j:Range(all_ov.Size())) {
	  auto ov = all_ov[j];
	  auto cov = vmap[ov];
	  if(cov==-1) continue;
	  if(!vg_row.Contains(cov)) continue;
	  auto eq = mesh->GetEQCOfV(ov);
	  if(!eqc_h->IsLEQ(EQ, eq)) continue;
	  uve.Append(INT<2>(ov,all_oe[j]));
	}
	uve.Append(INT<2>(V,-1));
	QuickSort(uve, [](const auto & a, const auto & b){return a[0]<b[0];});
	uverts.SetSize(uve.Size());
	uedges.SetSize(uve.Size());
	for(auto k:Range(uve.Size())) {
	  uverts[k] = uve[k][0];
	  uedges[k] = uve[k][1];
	}
	// cout << "use_v: " << endl << uverts << endl;
	// cout << "use_e: " << endl << uedges << endl;
	// calc (I - omega D^-1 A)
	auto posV = uverts.Pos(V);
      	size_t unv = uverts.Size(); // # of vertices used
	size_t und = BS * unv; // # of dofs used
	FlatArray<int> oss(BS, lh);
	for(auto l:Range(BS)) oss[l] = l*unv;
	FlatMatrix<double> mat(BS, unv*BS, lh);
	mat = 0.0;
	FlatMatrix<double> block(2*BS, 2*BS, lh);
	FlatArray<int> odc(BS, lh);
	for(auto l:Range(unv)) {
	  if(l==posV) continue;
	  auto v0 = min2(V, uverts[l]);
	  auto v1 = max2(V, uverts[l]);
	  bool flip = get_repl(edges[uedges[l]], block);
	  // cout << "block for V ov " << V << " " << uverts[l] << " (edge " << edges[uedges[l]] << ")"
	  //      << endl << block << endl;
	  flip = v1==V;
	  const Array<int> & rk = flip ? r2 : r1;
	  for(auto j:Range(BS))
	    odc[j] = posV+oss[j];
	  mat.Cols(odc) += block.Rows(rk).Cols(rk);
	  const Array<int> & rl = flip ? r1 : r2;
	  for(auto j:Range(BS))
	    odc[j] = l+oss[j];
	  mat.Cols(odc) = block.Rows(rk).Cols(rl);
	  // cout << "mat now: " << endl << mat << endl;
	}
	FlatArray<int> rdiag(BS, lh);
	FlatArray<int> rodiag((unv-1)*BS, lh);
	for(auto l:Range(BS))
	  rdiag[l] = posV+oss[l];
	size_t cc = 0;
	size_t cc2 = 0;
	for(auto j:Range(und)) {
	  if( (cc>=BS) || (j!=rdiag[cc]) ) rodiag[cc2++] = j;
	  else cc++;
	}
	/** diagonal block! **/
	FlatMatrix<double> inv(BS, BS, lh); //saves the diagonal block
	FlatMatrix<double> row(BS, unv*BS, lh);
	row = 0;
	inv = mat.Cols(rdiag);
	// cout << "diag: " << endl << inv << endl;
	CalcInverse(inv);
	// cout << "inv diag: " << endl << inv << endl;
	// if(!hadit) row.Cols(rdiag) = -0.5 * inv * admat;
	for(auto j:Range(BS)) row(j, rdiag[j]) += 0.5;
	double alpha = (CV==-1) ? 1.0 : 0.5;
	row.Cols(rodiag) = -alpha * inv * mat.Cols(rodiag);
	/** pwprol-block **/
	Array<int> ucverts(vgs[V]);
	size_t uncv = ucverts.Size();
	for(auto l:Range(uncv)) ucverts[l] = v_graph[V][l];
	FlatMatrix<double> pwpblock(unv*BS, vgs[V]*BS, lh);
	const SparseMatrix<double> & pwp(*pwprol);
	size_t pos = 0;
	for(auto lk:Range(BS)) {
	  for(auto k:Range(unv)) {
	    auto rip = pwp.GetRowIndices(get_dof(uverts[k],lk));
	    auto rvp = pwp.GetRowValues(get_dof(uverts[k],lk));
	    for(auto j:Range(uncv)) {
	      for(auto lj:Range(BS)) {
		if((pos = rip.Pos(get_Cdof(ucverts[j], lj))) == -1)
		  pwpblock(k+unv*lk, j+uncv*lj) = 0.0;
		else
		  pwpblock(k+unv*lk, j+uncv*lj) = rvp[pos];
	      }
	    }
	  }
	}
	// cout << "row: " << endl << row << endl;
	// cout << "pwb: " << endl << pwpblock << endl;
	FlatMatrix<double> vals(BS, vgs[V]*BS, lh);
	vals = row * pwpblock | Lapack;
	// cout << "vals: " << endl << vals << endl;
	for(auto l:Range(BS)) {
	  auto dof = get_dof(V,l);
	  auto ri = prol->GetRowIndices(dof);
	  auto rv = prol->GetRowValues(dof);
	  size_t s = 0;
	  for(auto lj:Range(BS)) {
	    for(auto j:Range(uncv)) {
	      ri[s] = get_Cdof(ucverts[j],lj); 
	      rv[s] = vals(l, s);
	      s++;
	    }
	  }
	}
      }
      // cout << "special verts (" << count_spec.Size() << ")" << endl << count_spec << endl;
    }

    
    return prol;
  }

  

  RegisterProlongation reg_h1p("h1_pw", true, BuildH1PWProlongation);
  RegisterProlongation reg_h1h("h1_hierarchic", true, BuildH1HierarchicProlongation);

} // end namespace amg
