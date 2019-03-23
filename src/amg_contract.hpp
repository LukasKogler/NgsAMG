#ifndef FILE_AMGCTR
#define FILE_AMGCTR

namespace amg
{

  template<class TV>
  class CtrMap : public BaseDOFMapStep
  {
  public:
    using TM = typename strip_mat<Mat<mat_traits<TV>::HEIGHT, mat_traits<TV>::HEIGHT, typename mat_traits<TV>::TSCAL>>::type;
    using TSPM = SparseMatrix<TM,TV,TV>;
    CtrMap (shared_ptr<ParallelDofs> pardofs, shared_ptr<ParallelDofs> mapped_pardofs,
	    Array<int> && group, Table<int> && dof_maps);
    ~CtrMap ();
    
    virtual void TransferF2C(const shared_ptr<const BaseVector> & x_fine,
			     const shared_ptr<BaseVector> & x_coarse) const override { ; }
    virtual void TransferC2F(const shared_ptr<BaseVector> & x_fine,
			     const shared_ptr<const BaseVector> & x_coarse) const override { ; }

    virtual shared_ptr<BaseSparseMatrix> AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const override
    {
      // TODO: static cast this??
      shared_ptr<TSPM> spm = dynamic_pointer_cast<TSPM>(mat);
      if (spm == nullptr) {
	throw Exception("CtrMap cast did not work!!");
      }
      return DoAssembleMatrix(spm);
    }
    
    
    virtual shared_ptr<BaseDOFMapStep> Concatenate (shared_ptr<BaseDOFMapStep> other) override
    { return nullptr; }
    
    INLINE bool IsMaster () const { return is_gm; }

    // TODO: bad hack because NgsAMG_Comm -> MPI_Comm -> NgsMPI_Comm in pardofs constructor (ownership lost!)
    NgsAMG_Comm _comm_keepalive_hack;

  protected:
    shared_ptr<TSPM> DoAssembleMatrix (shared_ptr<TSPM> mat) const;
    using BaseDOFMapStep::pardofs, BaseDOFMapStep::mapped_pardofs;

    Array<int> group;
    int master;
    bool is_gm;
    Table<int> dof_maps;
    Array<MPI_Request> reqs;
    Array<MPI_Datatype> mpi_types;
    Table<double> buffers;

  };

  INLINE Timer & timer_hack_gccm1 () { static Timer t("GridContractMap::MapNodeData"); return t; }
  template<NODE_TYPE NT> INLINE Timer & timer_hack_gccm2 () {
    if constexpr(NT==NT_VERTEX) { static Timer t("GridContractMap::MapNodeData, V"); return t; }
    if constexpr(NT==NT_EDGE)   { static Timer t("GridContractMap::MapNodeData, E"); return t; }
    if constexpr(NT==NT_FACE)   { static Timer t("GridContractMap::MapNodeData, F"); return t; }
    if constexpr(NT==NT_CELL)   { static Timer t("GridContractMap::MapNodeData, C"); return t; }
  }

  Table<int> PartitionProcsMETIS (BlockTM & mesh, int nparts);
  

  template<class TMESH>
  class GridContractMap : public GridMapStep<TMESH>
  {
    static_assert(std::is_base_of<BlockTM, TMESH>::value, "GridContractMap can only be constructed for Meshes that inherit from BlockTM!");
  public:
    GridContractMap (Table<int> && groups, shared_ptr<TMESH> mesh);

    // template<NODE_TYPE NT, typename T>
    // void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const;

    INLINE bool IsMaster () const { return is_gm; }
    INLINE FlatArray<int> GetGroup () const { return my_group; }
    template<NODE_TYPE NT> INLINE FlatArray<amg_nts::id_type> GetNodeMap (int member) const { return node_maps[NT][member]; }

    INLINE shared_ptr<EQCHierarchy> GetEQCHierarchy () const { return eqc_h; }
    INLINE shared_ptr<EQCHierarchy> GetMappedEQCHierarchy () const { return c_eqc_h; }
    
  protected:
    using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;
    
    shared_ptr<EQCHierarchy> eqc_h = nullptr;
    shared_ptr<EQCHierarchy> c_eqc_h = nullptr;

    /** proc-maps **/
    bool is_gm = true; // is group master
    Array<int> proc_map;
    Table<int> groups;
    FlatArray<int> my_group;

    /** EQC-maps **/
    Table<int> map_om; // (maps eqcs) orig --> merged
    Array<int> map_mc; // (maps eqcs) merged --> contr
    Table<int> map_oc; // (maps eqcs) orig --> contr
    // (I think) un-mapped members of merged eqcs
    Table<int> mmems; // merged (pre-contracted)\members
    
    /** node-maps **/
    size_t mapped_NN[4];
    Array<Table<amg_nts::id_type>> node_maps;
    Array<Table<amg_nts::id_type>> annoy_nodes;

    void BuildCEQCH ();
    void BuildNodeMaps ();

  public:
    template<NODE_TYPE NT, typename T>
    void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const
    {
      cout << "MAPNODEDATA " << endl;
      RegionTimer rt1(timer_hack_gccm1());
      RegionTimer rt2(timer_hack_gccm2<NT>());
      auto comm = eqc_h->GetCommunicator();
      int master = my_group[0];
      if(!is_gm) {
	cout << "send " << data.Size() << " times " << typeid(T).name() << " to " << master << endl;
	comm.Send(data, master, MPI_TAG_AMG);
	return;
      }

      cout << "send " << data.Size() << " times " << typeid(T).name() << " to myself, LOL" << endl;
      Array<T> &out(*cdata);
      MPI_Request req = MyMPI_ISend(data, master, MPI_TAG_AMG, comm); // todo: avoid local send please...
      MPI_Request_free(&req);
      out.SetSize(mapped_NN[NT]);
      // if (stat==DISTRIBUTED) out = T(0);
      out = T(0); // TODO: above is better...
      auto & maps = node_maps[NT];
      Array<T> buf;
      for(auto pn:Range(my_group.Size())) {
	auto p = my_group[pn];
	auto map = maps[pn];
	cout << "get " << buf.Size() << " times " << typeid(T).name() << " from " << p << endl;
	comm.Recv(buf, p, MPI_TAG_AMG);
	cout << "got " << buf.Size() << " times " << typeid(T).name() << " from " << p << endl;
	cout << "map size is: " << map.Size() << endl;
	// cout << "map : "; prow2(map); cout << endl;
	if(stat==CUMULATED)
	  for(auto k:Range(map.Size()))
	    out[map[k]] = buf[k];
	else
	  for(auto k:Range(map.Size()))
	    out[map[k]] += buf[k];
      }
      // annoy data is not scattered correctly now for edge-based data!!
      auto & anodes = annoy_nodes[NT];
      if(!anodes.Size()) { return; } // no annoying nodes for this type!!
      auto cneqcs = c_eqc_h->GetNEQCS();
      Array<size_t> sz(cneqcs);
      for(auto k:Range(cneqcs))
	sz[k] = anodes[k].Size();
      Table<T> adata(sz);
      for(auto k:Range(cneqcs)) {
	for(auto j:Range(anodes[k].Size()))
	  adata[k][j] = out[anodes[k][j]];
      }
      auto radata = ReduceTable<T,T>(adata, c_eqc_h, [&](auto & t) {
	  // cout << "lambda for t: " << endl;
	  // for(auto k:Range(t.Size())) { cout<<k<<"(" << t[k].Size() << "):  ";prow(t[k]);cout<<endl; }
	  Array<T> out;
	  if(!t.Size()) return out;
	  if(!t[0].Size()) return out;
	  out.SetSize(t[0].Size());
	  out = t[0];
	  if(t.Size()==1) return out;
	  T zero(0);
	  for(auto k:Range((size_t)1, t.Size())) {
	    for(auto j:Range(t[k].Size())) {
	      if( (stat==DISTRIBUTED) || (t[k][j]==zero) ) {
		// hacked_add(out[j],t[k][j]);
		out[j] += t[k][j];
	      }
	      else {
		out[j] = t[k][j];
	      }
	    }
	  }
	  return out;
	});
      for(auto k:Range(cneqcs)) {
	for(auto j:Range(anodes[k].Size()))
	  out[anodes[k][j]] = radata[k][j];
      }
      return;
    }
    
  };



} // namespace amg

#endif
