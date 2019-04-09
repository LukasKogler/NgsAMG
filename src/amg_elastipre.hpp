#ifndef AMG_ELASTIPRE
#define AMG_ELASTIPRE

namespace amg
{

  /** **/
  template<int D, int R>
  struct struct_edge_data {
    INT<D*D+R*R, double> data;
    // Mat<disppv(D), disppv(D), double>   edisp;
    // Mat<rotpv(D), rotpv(D), double> erot;
    // so we can write this into an array/hash-table etc.
    INLINE FlatMatrixFixWidth<D, double> edisp()
    { return FlatMatrixFixWidth<D, double>(D, &data[0]);}
    INLINE FlatMatrixFixWidth<R, double> erot()
    { return FlatMatrixFixWidth<R, double>(R, &data[D*D]);}
    struct_edge_data (struct_edge_data<D,R> && b) {
      data = std::move(b.data);
    }
    struct_edge_data (const struct_edge_data<D,R> & val) {
      data = val.data;
    }
    struct_edge_data (double val) : data(val) {}
    struct_edge_data (){}
    INLINE const struct_edge_data & operator += (const struct_edge_data<D,R> & b) {
      this->data += b.data;
      return *this;
    }
    INLINE const struct_edge_data &  operator -= (const struct_edge_data<D,R> & b) {
      this->data -= b.data;
      return *this;
    }
    INLINE const struct_edge_data &  operator = (double val) {
      this->data = val;
      return *this;
    }
    INLINE const struct_edge_data &  operator = (const struct_edge_data<D,R> & b) {
      this->data = b.data;
      return *this;
    }
    INLINE const bool operator == (const struct_edge_data<D,R> & b) const
    { return this->data == b.data; }
  };

  template<int D, int R>
  std::ostream & operator<<(std::ostream &os, struct_edge_data<D,R>& ed)  {
    os << "disp: ";
    for(auto k:Range(D*D)) os << ed.data[k] << " ";
    os << "   ||  rot: ";
    for(auto k:Range(D*D, D*D+R*R)) os << ed.data[k] << " ";
    // os << "disp: " << endl << ed.edisp();
    // os << "rot: " << endl << ed.erot();
    return os;
  }
  

  
  template<>
  INLINE void hacked_add<struct_edge_data<2,1>>(struct_edge_data<2,1> & a, const struct_edge_data<2,1> & b) { a+=b; }
  template<>
  INLINE void hacked_add<struct_edge_data<3,3>>(struct_edge_data<3,3> & a, const struct_edge_data<3,3> & b) { a+=b; }

  /** Nodal data that is attached to a mesh **/
  template<NODE_TYPE NT, class T, class SELF>
  class AttachedNodeData
  {
  protected:
    BlockedAlgebraicMesh * mesh;
    PARALLEL_STATUS stat;
    Array<T> data;
  public:
    AttachedNodeData(Array<T> && _data, PARALLEL_STATUS _stat) : data(move(_data)), stat(_stat) {}
    void AttachTo (BlockedAlgebraicMesh * _mesh) { mesh = _mesh; }
    Array<T> & Data() const { return data; }
    unique_ptr<SELF> Map (CoarseMap & cmap) {
      // cout << "COARSEMAP DATA, NT =  " << NT << ", T = " << typeid(T).name() << endl;
      // cout << " map " << cmap.GetNN<NT>() << " to " << cmap.GetMappedNN<NT>() << endl;
      // if(NT==NT_EDGE) cout << " data sz: " << data.Size() << endl;
      // if(NT==NT_EDGE) cout << " data: " << endl << data << endl;
      Array<T> cdata(cmap.GetMappedNN<NT>());
      auto cstat = static_cast<SELF&>(*this).reduce_func(cmap, data, cdata);
      // auto cstat = reduce_func(cmap, data, cdata); // what does this do with status??
      // return AttachedNodeData<NT, T>(cdata, cmap.GetMappedMesh(), cstat);
      // if(NT==NT_EDGE) cout << "COARSE data (cumulated? " << (cstat==CUMULATED) << "): " << endl << cdata << endl;
      auto cm_wd = make_unique<SELF>(move(cdata), cstat);
      // cm_wd->AttachTo((BlockedAlgebraicMesh*)cmap.GetMappedMesh().get());
      return cm_wd;
    }
    unique_ptr<SELF> Map (GridContractMap & cmap) {
      Array<T> cdata(cmap.GetMappedNN<NT>());
      auto cstat = cmap.MapNodeData<NT,T>(data, cdata);
      auto cm_wd = make_unique<SELF>(move(cdata), cstat);
      return cm_wd;
    }
    unique_ptr<SELF> Map (NodeDiscardMap & cmap) {
      Array<T> cdata(cmap.GetMappedNN<NT>());
      auto cstat = cmap.MapNodeData<NT,T>(data, cdata);
      auto cm_wd = make_unique<SELF>(move(cdata), cstat);
      return cm_wd;
    }
    // virtual unique_ptr<AttachedNodeData<NT, T>> make_self(Array<T> && _data, PARALLEL_STATUS _stat, BlockedAlgebraicMesh* bm) = 0;
    // virtual PARALLEL_STATUS reduce_func(CoarseMap & cmap, FlatArray<T> _data, FlatArray<T> _cdata) = 0;
    // PARALLEL_STATUS reduce_func(CoarseMap & cmap, FlatArray<T> _data, FlatArray<T> _cdata);
    virtual void Cumulate() const {
      if(stat == DISTRIBUTED) {
	// cout << "att data CUMULATE, NT = " << NT << endl;
	if constexpr(NT == NT_VERTEX) {
	    mesh->CumulateVertexData(data);
	  }
	if constexpr(NT == NT_EDGE) {
	    mesh->CumulateEdgeData(data);
	  }
	// cout << "CUMULATED DATA: " << endl << data << endl;
	stat = CUMULATED;
      }
    }
    virtual void Distribute() const {
      if(stat==CUMULATED) {
	cout << "att data DISTRIBUTE, NT = " << NT << endl;
	for(auto [eqc,block] : *mesh) {
	  if(mesh->GetEQCHierarchy()->IsMasterOfEQC(eqc)) continue;
	  if constexpr(NT==NT_VERTEX) {
	      for(auto v:block->Vertices())
		data[v] = 0;
	    }
	  if constexpr(NT==NT_EDGE) {
	      for(auto e:block->Edges())
		data[e.id] = 0;
	      for(auto e:mesh->GetPaddingEdges(eqc))
		data[e.id] = 0;
	    }
	}
	stat = DISTRIBUTED;
      }
    }
    PARALLEL_STATUS GetParallelStatus () { return stat; }
    void SetParallelStatus (PARALLEL_STATUS _stat) { stat = _stat; }
  };

  template<NODE_TYPE NT, class T, class SELF>
  std::ostream & operator<<(std::ostream &os, AttachedNodeData<NT,T, SELF>& nd)
    {
      os << endl << "stat: " << nd.GetParallelStatus() << endl;
      os << endl << "data: " << nd.Data() << endl;
      return os;
    }

  
  /** This is an algebraic mesh with various attached data**/
  template<class... T>
  class MeshWithData : public BlockedAlgebraicMesh
  {
  protected:
      std::tuple<unique_ptr<T>...> node_data;
  public:
    MeshWithData ( size_t nv, size_t ne, size_t nf,
		   Array<edge> && edges, Array<face> && faces,
		   Array<double> && awv, Array<double> && awe, Array<double> && awf,
		   Array<size_t> && eqc_v, const shared_ptr<EQCHierarchy> & aeqc_h,
		   unique_ptr<T>...t)
      : BlockedAlgebraicMesh(nv, ne, nf, move(edges), move(faces), move(awv), move(awe), move(awf), move(eqc_v), aeqc_h),
	node_data(move(t)...)
    { std::apply([&](auto& ...x){(..., x->AttachTo(this));}, node_data); }

    MeshWithData ( BlockedAlgebraicMesh && am, unique_ptr<T> ...t)
      : BlockedAlgebraicMesh(move(am)), node_data(move(t)...)
    { std::apply([&](auto& ...x){(..., x->AttachTo(this));}, node_data); }

    
    virtual shared_ptr<BaseAlgebraicMesh> Map (CoarseMap & map)
    {
      auto cbm = dynamic_pointer_cast<BlockedAlgebraicMesh>(BlockedAlgebraicMesh::Map(map));
      //TODO: this is ugly, but i need it for cumulating in EVD
      auto cm = std::apply([&](auto&&... x){ return make_shared<MeshWithData<T...>>(move(*cbm), x->Map(map)...); }, node_data);

      // std::apply([&](auto&&... x){ (x->AttachTo(cm.get()), ...); }, cm->Data());
      std::apply([&](auto&&... x){ (x->Cumulate(), ...); }, cm->Data());
      return move(cm);
    }

    virtual shared_ptr<BaseAlgebraicMesh> Map (GridContractMap & map)
    {
      auto cbm = dynamic_pointer_cast<BlockedAlgebraicMesh>(BlockedAlgebraicMesh::Map(map));
      if (cbm!=nullptr) {
	// TODO: BMB should generally not be necessary...
	cbm->BuildMeshBlocks();
	// TODO: is the order well defined by standard? alternatively, first make tuple of coarse data, then unpack into constructor
	auto cm = std::apply([&](auto&&... x){ return make_shared<MeshWithData<T...>>(move(*cbm), x->Map(map)...); }, node_data);
	// done in constructor
	std::apply([&](auto&&... x){ (x->AttachTo(cm.get()), ...); }, cm->Data());
	// no idea why we need this here, it should be built in move constructor...
	cbm->BuildMeshBlocks();
      	return move(cm);
      }
      // does this reverse order??
      // std::apply([&](auto&&... x){ (x->Map(map),...); }, node_data);
      // std::apply([&](auto&&... x){ (...,x->Map(map)); }, node_data);
      std::apply([&](auto&&... x){ make_tuple(x->Map(map)...); }, node_data);
      return nullptr;
    }

    virtual shared_ptr<BaseAlgebraicMesh> Map (NodeDiscardMap & map)
    {
      auto cbm = dynamic_pointer_cast<BlockedAlgebraicMesh>(BlockedAlgebraicMesh::Map(map));
      auto cm = std::apply([&](auto&&... x){ return make_shared<MeshWithData<T...>>(move(*cbm), x->Map(map)...); }, node_data);
      return move(cm);
    }

    
    // template<class C>
    // shared_ptr<BaseAlgebraicMesh> Map (C & map)
    // {
    //   auto cbm = dynamic_pointer_cast<BlockedAlgebraicMesh>(BlockedAlgebraicMesh::Map(map));
    //   return std::apply([&](auto&&... x){ return make_shared<MeshWithData<T...>>(move(*cbm), x->Map(map)...); }, node_data);
    // }
    
    // std::tuple<T...> & Data () { return node_data; }
    std::tuple<unique_ptr<T>...> & Data () { return node_data; }

    template<typename... T2>
    friend std::ostream & operator<<(std::ostream &os, MeshWithData<T2...> & m);
    
  };

  template<class... T>
  std::ostream & operator<<(std::ostream &os, MeshWithData<T...> & m)
  {
    os << "MeshWithData:" << endl;
    os << "NV NE: " << m.NV() << " " << m.NE() << endl;
    os << "eqc_h: " << *m.GetEQCHierarchy() << endl;
    os << "vert: " << endl; print_ft(os, m.eqc_verts); cout << endl;
    os << "edges: " << endl; print_ft(os, m.eqc_edges); cout << endl;
    os << "cross_edges: " << endl; print_ft(os, m.eqc_pad_edges); cout << endl;
    apply([&](auto&&... x) {(os << ... << *x) << endl; }, m.Data()); return os;
  }


  // this is just a workaround because I dont want BAMG-PC here ... 
  class BAMG_PC_NEW
  {
  public:
    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, CoarseMap & map) { return nullptr; };
    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, GridContractMap & map) { return nullptr; };
    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, NodeDiscardMap & map) { return nullptr; };
  };


  /**
     TODO(!): 
        we assume 6 one-dimensional spaces
	we could also have 3d-disp, 3d-rot spaces (or even one 6-dimensional space!)
	   (in 2d: 2d-disp, 1d-rot)

     TODO(...): 
        we also want to handle cases with only disp-components (-> extend to disp/rot forulation??)
	       
   **/
  template<int D>
  class ElasticityAMGPreconditioner : public BaseAMGPreconditioner
  {
    static constexpr int disppv (int dim)
    { return dim; }
    static constexpr int rotpv (int dim)
    { return dim*(dim-1)/2; }
    static constexpr int dofpv (int n)
    { return disppv(n)+rotpv(n); }
    static constexpr int dofpv(int n, int type)
    { return (type==0)?disppv(n):rotpv(n); }

    // nested classes for data attached to the mesh
    class EVD : public AttachedNodeData<NT_VERTEX, Vec<3,double>, EVD>
    {
    public:
      // using AttachedNodeData<NT_VERTEX, Vec<3,double>, EVD> :: Cumulate;
      // using AttachedNodeData<NT_VERTEX, Vec<3,double>, EVD> :: Distribute;
      EVD (Array<Vec<3,double>> && _data, PARALLEL_STATUS _stat)
	: AttachedNodeData<NT_VERTEX, Vec<3,double>, EVD>(move(_data), _stat)
      {}
      PARALLEL_STATUS reduce_func(CoarseMap & cmap, FlatArray<Vec<3,double>> _data, FlatArray<Vec<3,double>> _cdata);
    };
    using edge_data = struct_edge_data<disppv(D), rotpv(D)>;
    class EED : public AttachedNodeData<NT_EDGE, edge_data, EED>
    {
    public:
      EED (Array<edge_data> && _data, PARALLEL_STATUS _stat)
	: AttachedNodeData<NT_EDGE, edge_data, EED>(move(_data), _stat)
      {}
      PARALLEL_STATUS reduce_func(CoarseMap & cmap, FlatArray<edge_data> _data, FlatArray<edge_data> _cdata);
    };

    /** 
	For the case of elasticity, we attach:
	- coordinates for each vertex
	- one NDxND and one NRxNR matrix for each edge
    **/
    using ElasticityMesh = MeshWithData<EVD, EED>;
    /**
       This is ugly. We have to dynamically cast to elasticitymesh, because
       we need the node data, and we cannot use ElasticityMesh as a template arg
       for EED because we would have a circular template dependency which i dont know how
       to resolve
    **/

    
  public:
 
    ElasticityAMGPreconditioner ( const PDE & apde,
				  const Flags & aflags,
				  const string aname = "ElastiticyAMGPrecond")
      : BaseAMGPreconditioner(apde, aflags, aname),
	my_lh(10000, "something", false)
    { throw Exception("elasticity amg from pde not implemented!"); }
    
    
    ElasticityAMGPreconditioner ( shared_ptr<BilinearForm> abfa,
				  const Flags & aflags,
				  const string aname = "ElasticityAMGPreconditioner")
      : BaseAMGPreconditioner(abfa, aflags, aname),
	my_lh(10000, "something", false)
    {
      fes = abfa->GetFESpace();
      amg_options = make_shared<AMGOptions>(aflags);
      amg_options->SetMaxLevels(10);
      amg_info = make_shared<AMGInfo>();
    }

    virtual const BaseMatrix & GetMatrix() const override
    { return *(dynamic_cast<BaseMatrix*>(amg_matrix.get())); }
    virtual void AddElementMatrix (FlatArray<int> dnums,
  				   const FlatMatrix<double> & elmat,
  				   ElementId ei,
  				   LocalHeap & lh);
    virtual void InitLevel (shared_ptr<BitArray> afreedofs) override;
    virtual void FinalizeLevel (const BaseMatrix * a_mat) override;

    virtual size_t GetNLevels(size_t rank) const override
    {return this->amg_matrix->GetNLevels(rank); }
    virtual void GetBF(size_t level, size_t rank, size_t dof, BaseVector & vec) const override
    {this->amg_matrix->GetBF(level, rank, dof, vec); }
    virtual size_t GetNDof(size_t level, size_t rank) const override
    { return this->amg_matrix->GetNDof(level, rank); }

    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, CoarseMap & map) override;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, GridContractMap & map) override;
    virtual shared_ptr<BaseDOFMapStep> BuildDOFMapStep (shared_ptr<ParallelDofs> pd, NodeDiscardMap & map) override;


    void CINV(shared_ptr<BaseVector> x, shared_ptr<BaseVector> b)
    { amg_matrix->CINV(x, b); }

    void SetCutBLF (shared_ptr<BilinearForm> acut_blf) {
      cut_blf = acut_blf;
    }


    void MyTest () const
    {
      cout << IM(1) << "Compute eigenvalues" << endl;
      const BaseMatrix & amat = GetAMatrix();
      const BaseMatrix & pre = GetMatrix();

      int eigenretval;

      EigenSystem eigen (amat, pre);
      eigen.SetPrecision(1e-30);
      eigen.SetMaxSteps(1000); 
        
      eigen.SetPrecision(1e-15);
      eigenretval = eigen.Calc();
      eigen.PrintEigenValues (*testout);
      cout << IM(1) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      cout << IM(1) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
      cout << IM(1) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      (*testout) << " Min Eigenvalue : "  << eigen.EigenValue(1) << endl; 
      (*testout) << " Max Eigenvalue : " << eigen.MaxEigenValue() << endl; 
        
      if(testresult_ok) *testresult_ok = eigenretval;
      if(testresult_min) *testresult_min = eigen.EigenValue(1);
      if(testresult_max) *testresult_max = eigen.MaxEigenValue();
        
        
      //    (*testout) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl; 
      //    for (int i = 1; i < min2 (10, eigen.NumEigenValues()); i++)
      //      cout << "cond(i) = " << eigen.MaxEigenValue() / eigen.EigenValue(i) << endl;
      (*testout) << " Condition   " << eigen.MaxEigenValue()/eigen.EigenValue(1) << endl;
        
    }

    
    
  protected:

    shared_ptr<ElasticityMesh> BuildInitialAlgMesh ();
    void SetupAMGMatrix();
    shared_ptr<BaseGridMapStep> TryElim (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh);
    shared_ptr<BaseGridMapStep> TryCoarsen (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh);
    shared_ptr<BaseGridMapStep> TryContract (INT<3> level, const shared_ptr<BaseAlgebraicMesh> & _mesh);
    // shared_ptr<BaseDOFMapStep> SmoothDOFStep (shared_ptr<ProlMap> pw_map, shared_ptr<ElasticityMesh> mesh);
    shared_ptr<SparseMatrix<double>> SmoothProl (shared_ptr<ProlMap> pw_map, shared_ptr<ElasticityMesh> mesh);

    
    // Array<shared_ptr<ParallelDofs>> dummy_fds;
    shared_ptr<ParallelDofs> finest_pds;
    bool cut_prol = false;
    bool has_dummies = false;
    shared_ptr<BilinearForm> cut_blf = nullptr; //not used ATM (see branch "testing")
    shared_ptr<FESpace> fes;
    HashTable<INT<2, size_t>, edge_data>* hash_edge;
    Array<edge_data> edge_mats;
    Array<edge_data> cedge_mats;
    Array<Vec<3,double> > p_coords;
    Array<INT<2,double>> vertex_wt;
    Array<size_t> marked_vs;
    bool smooth_next_prol = false;
    bool use_hierarch = false;
    bool last_hierarch = false;
    LocalHeap my_lh;
    
    // not const bc. of local heap...
    void CalcReplMatrix (edge_data & edata, Vec<3,double> cv0, Vec<3,double> cv1, FlatMatrix<double> & mat);
    INLINE INT<2, double> CalcEdgeWeights(edge_data & e_data)
    {
      // cout << "mat rot: " << endl << e_data.erot() << endl;
      // cout << "mat disp: " << endl << e_data.edisp() << endl;
      auto calc_det = [](const auto mat) -> double {
	switch(mat.Height()) {
	case(1):
	return mat(0,0);
	case(2):
	return mat(0,0)*mat(1,1)-mat(1,0)*mat(0,1);
	case(3):
	return mat(0,0)*mat(1,1)*mat(2,2) + mat(0,1)*mat(1,2)*mat(2,0)+mat(0,2)*mat(1,0)*mat(1,2)
	- mat(2,0)*mat(1,1)*mat(0,2) - mat(2,1)*mat(1,2)*mat(0,0) - mat(2,2)*mat(1,0)*mat(0,1);
	}
      };
      // double det_rot = calc_det(e_data.erot());
      // double det_disp = calc_det(e_data.edisp());
      auto calc_trace = [](const auto mat) -> double {
	double t = 0.0;
	for(auto l:Range(mat.Height()))
	  t += mat(l,l);
	t /= mat.Height();
	return t;
      };
      double det_rot = calc_trace(e_data.erot());
      double det_disp = calc_trace(e_data.edisp());
      // cout << "det rot/disp -> " << det_rot << "  " << det_disp << endl;
      return INT<2, double>(det_rot, det_disp);
    }

    void CalcRBM (Vec<D> & t, FlatMatrix<double> & rbm);
    
  }; // end class ElasticityAMGPreconditioner



    
} // namespace amg


namespace ngcore
{
  template<int D, int R> struct MPI_typetrait<amg::struct_edge_data<D,R>> {
    static MPI_Datatype MPIType () 
    { return MyGetMPIType<INT<D*D+R*R, double>>(); }
  };
} // namespace ngcore

#endif
