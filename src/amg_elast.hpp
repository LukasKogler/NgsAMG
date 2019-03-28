#ifndef FILE_AMGELAST
#define FILE_AMGELAST


namespace amg
{

  constexpr int disppv (int dim)
  { return dim; }
  constexpr int rotpv (int dim)
  { return dim*(dim-1)/2; }
  constexpr int dofpv (int dim)
  { return disppv(dim) + rotpv(dim); }

  struct PosWV { // position + weight
    Vec<3,double> pos;
    double wt;
    PosWV (double val) { wt = val; pos = val; }
    PosWV () : PosWV (0) { ; }
    PosWV (Vec<3,double> &&_pos, double &&_wt) : pos(_pos), wt(_wt) { ; }
    PosWV (Vec<3,double> _pos, double _wt) : pos(_pos), wt(_wt) { ; }
    // PosWV (PosWV && other) : pos(move(other.pos)), wt(move(other.wt)) { ; }
    PosWV (PosWV && other) = default;
    PosWV (PosWV & other) = default;
    INLINE void operator = (double x) { wt = x; pos = x; }; // for Cumulate
    INLINE void operator = (const PosWV & other) { wt = other.wt; pos = other.pos; }; // for Cumulate
    INLINE void operator += (const PosWV & other) { pos += other.pos; wt += other.wt; }; // for Cumulate
    INLINE bool operator == (const PosWV & other) {
      return wt==other.wt && pos(0)==other.pos(0) && pos(1)==other.pos(1) && pos(2)==other.pos(2); } 
  };
  // INLINE void operator + (PosWV & a, const PosWV & b) { a.wt += b.wt; }; // for Cumulate
  class ElVData : public AttachedNodeData<NT_VERTEX, PosWV, ElVData>
  {
  public:
    using AttachedNodeData<NT_VERTEX, PosWV, ElVData>::map_data;
    ElVData (Array<PosWV> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, PosWV, ElVData>(move(_data), _stat) {}
    INLINE PARALLEL_STATUS map_data (const BaseCoarseMap & cmap, Array<PosWV> & cdata) const
    {
      Cumulate();
      cdata.SetSize(cmap.GetMappedNN<NT_VERTEX>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_VERTEX>();
      Array<int> touched(map.Size()); touched = 0;
      mesh->Apply<NT_EDGE>([&](const auto & e) { // set coarse data for all coll. vertices
	  auto CV = map[e.v[0]];
	  if ( (CV == -1) || (map[e.v[1]] != CV) ) return;
	  touched[e.v[0]] = touched[e.v[1]] = 1;
	  cdata[CV].pos = 0.5 * (data[e.v[0]].pos + data[e.v[1]].pos);
	  cdata[CV].wt = data[e.v[0]].wt + data[e.v[1]].wt;
	}, true); // if stat is CUMULATED, only master of collapsed edge needs to set wt 
      mesh->AllreduceNodalData<NT_VERTEX>(touched, [](auto & in) { return move(sum_table(in)); } , false);
      mesh->Apply<NT_VERTEX>([&](auto v) { // set coarse data for all "single" vertices
	  auto CV = map[v];
	  if ( (CV != -1) && (touched[v] == 0) )
	    { cdata[CV] = data[v]; }
	}, true);
      return DISTRIBUTED;
    }
  };

  template<int D>
  struct ElEW { // edge + weight
    INT<disppv(D)*disppv(D)+rotpv(D)*rotpv(D),double> wt_data;
    ElEW (double val) { wt_data = val; }
    ElEW () : ElEW<D>(0.0) { ; }
    ElEW (ElEW<D> && other) = default;
    ElEW (ElEW<D> & other) = default;
    INLINE void operator = (double x) { wt_data = x; }; // for Cumulate
    INLINE void operator = (const ElEW<D> & other) { wt_data = other.wt_data; }
    INLINE void operator += (const ElEW<D> & other)
    { for (auto l : Range(disppv(D)*disppv(D)+rotpv(D)*rotpv(D))) wt_data[l] += other.wt_data[l]; }; // for Cumulate
    INLINE bool operator == (const ElEW<D> & other) { return wt_data == other.wt_data; }
    INLINE double get_wt () const { return wt_data[0]; }
  };
  template struct ElEW<2>;
  template struct ElEW<3>;
  template<int D>
  class ElEData : public AttachedNodeData<NT_EDGE, ElEW<D>, ElEData<D>>
  {
  public:
    using AttachedNodeData<NT_EDGE, ElEW<D>, ElEData<D>>::map_data;
    using AttachedNodeData<NT_EDGE, ElEW<D>, ElEData<D>>::mesh;
    using AttachedNodeData<NT_EDGE, ElEW<D>, ElEData<D>>::stat;
    using AttachedNodeData<NT_EDGE, ElEW<D>, ElEData<D>>::data; // ?? why do I need all of this> ??
    ElEData (Array<ElEW<D>> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, ElEW<D>, ElEData>(move(_data), _stat) {}
    INLINE PARALLEL_STATUS map_data (const BaseCoarseMap & cmap, Array<ElEW<D>> & cdata) const
    {
      cdata.SetSize(cmap.GetMappedNN<NT_EDGE>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_EDGE>();
      mesh->template Apply<NT_EDGE>([&](const auto & e) {
      	  auto CE = map[e.id];
      	  if (CE != -1) cdata[CE] += data[e.id];
      	}, stat==CUMULATED);
      return DISTRIBUTED;
    }
  };

  template<int D>
  using ElasticityMesh = BlockAlgMesh<ElVData, ElEData<D>>;

  template<int D>
  class ElasticityAMG : public VWiseAMG<ElasticityAMG<D>, ElasticityMesh<D>, Mat<dofpv(D), dofpv(D), double>>
  {
  public:
    using TMESH = ElasticityMesh<D>;
    using TMAT = Mat<dofpv(D), dofpv(D), double>;
    using BASE = VWiseAMG<ElasticityAMG<D>, ElasticityMesh<D>, TMAT>;
    using Options = typename BASE::Options;
    ElasticityAMG (shared_ptr<ElasticityMesh<D>> mesh, shared_ptr<Options> opts)
      : VWiseAMG<ElasticityAMG<D>, ElasticityMesh<D>, Mat<dofpv(D), dofpv(D), double>>(mesh, opts) { ; }
    template<NODE_TYPE NT> INLINE double GetWeight (const TMESH & mesh, const AMG_Node<NT> & node) const
    { // TODO: should this be in BlockAlgMesh instead???
      if constexpr(NT==NT_VERTEX) { return get<0>(mesh.Data())->Data()[node].wt; }
      else if constexpr(NT==NT_EDGE) { return get<1>(mesh.Data())->Data()[node.id].get_wt(); }
      else return 0;
    }
    // INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap<TMESH> & map,
    // 			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, TMAT & mat) const
    // { BASE::SetIdentity(mat); }
    INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap<TMESH> & map,
			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, TMAT & mat) const
    {
      Vec<3,double> tang = get<0>(cmesh.Data())->Data()[cv].pos;
      tang -= get<0>(fmesh.Data())->Data()[v].pos;
      mat = 0;
      for (auto k : Range(disppv(D))) {
	mat(k,k) = 1.0;
	if constexpr(D==3) for (auto j : Range(rotpv(D))) mat(k,disppv(D)+j) = tang(k)*tang(j);
      }
      if constexpr(D==2) {
	  mat(0,2) = tang(1);
	  mat(1,2) = -tang(0);
	}
      for (auto k : Range(disppv(D), disppv(D)+rotpv(D)))
	mat(k,k) = 1.0;
    }
    INLINE void CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<TMAT> mat) const
    {
      // [r]
      for (int i = 0; i < rotpv(D); i++) {
	
      }
      
      SetIdentity(mat(0,0));
      mat(0,1) = -mat(0,0);
      mat(1,0) = -mat(0,0);
      SetIdentity(mat(1,1));
    }
  };

} // namespace amg

namespace ngcore
{
  template<> struct MPI_typetrait<amg::PosWV> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if(!MPI_T)
  	{
  	  int block_len[2] = {1,1};
  	  MPI_Aint displs[2] = {0, sizeof(ngbla::Vec<3,double>)};
	  MPI_Datatype types[2] = {GetMPIType<ngbla::Vec<3,double>>(), GetMPIType<double>()};
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  };
  template<> struct MPI_typetrait<amg::ElEW<2>> {
    static MPI_Datatype MPIType () {
      return GetMPIType<decltype(amg::ElEW<2>::wt_data)>();
    }
  };
  template<> struct MPI_typetrait<amg::ElEW<3>> {
    static MPI_Datatype MPIType () {
      return GetMPIType<decltype(amg::ElEW<3>::wt_data)>();
    }
  };
} // namespace ngcore

#endif
