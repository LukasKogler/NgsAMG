#ifdef ELASTICITY
#ifndef FILE_AMGELAST
#define FILE_AMGELAST

namespace amg
{
  /**
     Elasticity Preconditioner

     The data we need to attach to the mesh is:
        - per vertex: one double as l2-weight, Vec<3,double> for vertex postition
	- per edge: dofpv x dofpv matrix
   **/

  struct ElasticityVertexData
  {
    Vec<3,double> pos;
    double wt;
    ElasticityVertexData (double val) { wt = val; pos = val; }
    ElasticityVertexData () : ElasticityVertexData (0) { ; }
    ElasticityVertexData (Vec<3,double> &&_pos, double &&_wt) : pos(_pos), wt(_wt) { ; }
    ElasticityVertexData (Vec<3,double> _pos, double _wt) : pos(_pos), wt(_wt) { ; }
    // ElasticityVertexData (ElasticityVertexData && other) : pos(move(other.pos)), wt(move(other.wt)) { ; }
    ElasticityVertexData (ElasticityVertexData && other) = default;
    ElasticityVertexData (const ElasticityVertexData & other) = default;
    INLINE void operator = (double x) { wt = x; pos = x; }; // for Cumulate
    INLINE void operator = (const ElasticityVertexData & other) { wt = other.wt; pos = other.pos; }; // for Cumulate
    INLINE void operator += (const ElasticityVertexData & other) { pos += other.pos; wt += other.wt; }; // for Cumulate
    INLINE bool operator == (const ElasticityVertexData & other)
    { return wt==other.wt && pos(0)==other.pos(0) && pos(1)==other.pos(1) && pos(2)==other.pos(2); } 
  }; // struct ElasticityVertexData
  INLINE std::ostream & operator<<(std::ostream &os, ElasticityVertexData& V)
  { os << "[" << V.wt << " | " << V.pos << "]"; return os; }
  INLINE bool is_zero (const ElasticityVertexData & m) { return is_zero(m.wt) && is_zero(m.pos); }


  class AttachedEVD : public AttachedNodeData<NT_VERTEX, ElasticityVertexData, AttachedEVD>
  {
  public:
    using BASE = AttachedNodeData<NT_VERTEX, ElasticityVertexData, AttachedEVD>;
    using BASE::map_data;

    AttachedEVD (Array<ElasticityVertexData> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    // templated because i cant static_cast basecoarsemap to elasticitymesh (dont have dim)
    template<class TMESH> INLINE void map_data (const CoarseMap<TMESH> & cmap, AttachedEVD & cevd) const;
  }; // class AttachedEVD


  template<int D> struct EED_TRAIT { typedef void type; };
  template<> struct EED_TRAIT<2> { typedef Mat<3, 3, double> type; };
  template<> struct EED_TRAIT<3> { typedef Mat<6, 6, double> type; };
  template<int D> using ElasticityEdgeData = typename EED_TRAIT<D>::type;
  
  template<int D>
  class AttachedEED : public AttachedNodeData<NT_EDGE, ElasticityEdgeData<D>, AttachedEED<D>>
  {
  public:
    using BASE = AttachedNodeData<NT_EDGE, ElasticityEdgeData<D>, AttachedEED<D>>;
    using BASE::map_data;

    AttachedEED (Array<ElasticityEdgeData<D>> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    void map_data (const BaseCoarseMap & cmap, AttachedEED<D> & ceed) const;
  }; // class AttachedEED


  template<int D>
  using ElasticityMesh = BlockAlgMesh<AttachedEVD, AttachedEED<D>>;

  template<int D> struct ELTM_TRAIT { typedef void type; };
  template<> struct ELTM_TRAIT<2> { typedef Mat<3,3,double> type; };
  template<> struct ELTM_TRAIT<3> { typedef Mat<6,6,double> type; };
  template<int D> using ELTM = typename ELTM_TRAIT<D>::type;

  template<int D>
  class ElasticityAMGFactory : public VertexBasedAMGFactory<ElasticityAMGFactory<D>, ElasticityMesh<D>, ELTM<D>>
  {
  public:
    constexpr static int DIM = D;
    using TMESH = ElasticityMesh<D>;
    using TM = ELTM<D>;

    using BASE = VertexBasedAMGFactory<ElasticityAMGFactory<D>, ElasticityMesh<D>, ELTM<D>>;
    struct Options;

    ElasticityAMGFactory (shared_ptr<TMESH> mesh, shared_ptr<Options> options, shared_ptr<BaseDOFMapStep> _embed_step = nullptr);

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");

    virtual void SetCoarseningOptions (VWCoarseningData::Options & opts, shared_ptr<TMESH> mesh) const override;

    template<NODE_TYPE NT> INLINE double GetWeight (const TMESH & mesh, const AMG_Node<NT> & node) const;

    INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap<TMESH> & map,
			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, TM & mat) const;

    INLINE void CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<TM> mat) const;

  protected:
    Array<double> CalcECWSimple (shared_ptr<TMESH> mesh) const;
    Array<double> CalcECWRobust (shared_ptr<TMESH> mesh) const;
  }; // class ElasticityAMGFactory


  template<class TMESH> void AttachedEVD :: map_data (const CoarseMap<TMESH> & cmap, AttachedEVD & cevd) const
  {
    static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
    auto & cdata = cevd.data;
    Cumulate();
    // cout << "(cumul) f-pos: " << endl;
    // for (auto V : Range(data.Size())) cout << V << ": " << data[V].pos << endl;
    // cout << endl;
    cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0.0;
    auto map = cmap.template GetMap<NT_VERTEX>();
    // cout << "v_map: " << endl; prow2(map); cout << endl << endl;
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
    // cout << "(distr) c-pos: " << endl;
    // for (auto CV : Range(cmap.GetMappedNN<NT_VERTEX>())) cout << CV << ": " << cdata[CV].pos << endl;
    // cout << endl;
    cevd.SetParallelStatus(DISTRIBUTED);
  } // AttachedEVD::map_data

} // namespace amg


namespace ngcore
{
  template<> struct MPI_typetrait<amg::ElasticityVertexData> {
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
} // namespace ngcore


#endif
#endif
