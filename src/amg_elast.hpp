#ifdef ELASTICITY

#ifndef FILE_AMG_ELAST_HPP
#define FILE_AMG_ELAST_HPP

#include "amg_coarsen.hpp"
#include "amg_discard.hpp"
#include "amg_contract.hpp"
#include "amg_factory.hpp"
#include "amg_factory_nodal.hpp"
#include "amg_factory_vertex.hpp"
#include "amg_energy.hpp"

namespace amg
{

  /** Vertex Data **/

  template<int DIM>
  class ElastVData
  {
  public:
    Vec<DIM, double> pos;
    double wt;
    ElastVData (double val) : pos(val), wt(val) { ; }
    ElastVData () : ElastVData(0) { ; }
    ElastVData (Vec<DIM,double> _pos, double _wt) : pos(_pos), wt(_wt) { ; }
    ElastVData (Vec<DIM,double> && _pos, double && _wt) : pos(_pos), wt(_wt) { ; }
    ElastVData (ElastVData<DIM> && other) = default;
    ElastVData (const ElastVData<DIM> & other) = default;
    INLINE void operator = (double x) { pos = x; wt = x; }
    INLINE void operator = (const ElastVData<DIM> & other) { pos = other.pos; wt = other.wt; }
    INLINE void operator += (const ElastVData<DIM> & other) { pos += other.pos; wt += other.wt; }
    INLINE bool operator == (const ElastVData<DIM> & other)
    {
      if constexpr (DIM == 2)
	{ return ( wt == other.wt ) && ( pos(0) == other.pos(0) ) && ( pos(1) == other.pos(1) ); }
      else
	{ return ( wt == other.wt ) && ( pos(0) == other.pos(0) ) && ( pos(1) == other.pos(1) ) && ( pos(2) == other.pos(2) ); }
    }
  }; // class ElastVData

  template<int DIM> INLINE std::ostream & operator << (std::ostream & os, ElastVData<DIM> & V)
  { os << "[" << V.wt << " | " << V.pos << "]"; return os; }

  template<int DIM> INLINE bool is_zero (const ElastVData<DIM> & m) { return is_zero(m.wt) && is_zero(m.pos); }


  template<int DIM>
  class AttachedEVD : public AttachedNodeData<NT_VERTEX, ElastVData<DIM>, AttachedEVD<DIM>>
  {
  public:
    using BASE = AttachedNodeData<NT_VERTEX, ElastVData<DIM>, AttachedEVD<DIM>>;
    using BASE::map_data, BASE::Cumulate, BASE::mesh, BASE::data;

    AttachedEVD (Array<ElastVData<DIM>> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    template<class TMESH> inline void map_data (const CoarseMap<TMESH> & cmap, AttachedEVD<DIM> & cevd) const
    {
      /** ECOL coarsening -> set midpoints in edges **/
      static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
      Cumulate();
      auto & cdata = cevd.data; cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0;
      auto vmap = cmap.template GetMap<NT_VERTEX>();
      Array<int> touched(vmap.Size()); touched = 0;
      mesh->template Apply<NT_EDGE>([&](const auto & e) { // set coarse data for all coll. vertices
	  auto CV = vmap[e.v[0]];
	  if ( (CV != -1) || (vmap[e.v[1]] == CV) ) {
	    touched[e.v[0]] = touched[e.v[1]] = 1;
	    cdata[CV].pos = 0.5 * (data[e.v[0]].pos + data[e.v[1]].pos);
	    cdata[CV].wt = data[e.v[0]].wt + data[e.v[1]].wt;
	  }
	}, true); // if stat is CUMULATED, only master of collapsed edge needs to set wt 
      mesh->template AllreduceNodalData<NT_VERTEX>(touched, [](auto & in) { return move(sum_table(in)); } , false);
      mesh->template Apply<NT_VERTEX>([&](auto v) { // set coarse data for all "single" vertices
	  auto CV = vmap[v];
	  if ( (CV != -1) && (touched[v] == 0) )
	    { cdata[CV] = data[v]; }
	}, true);
      cevd.SetParallelStatus(DISTRIBUTED);
    } // AttachedEVD::map_data

    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedEVD<DIM> & cevd) const
    {
      /** AGG coarsening -> set midpoints in agg centers **/
      static Timer t("AttachedEVD::map_data"); RegionTimer rt(t);
      Cumulate();
      auto & cdata = cevd.data; cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0;
      auto vmap = cmap.template GetMap<NT_VERTEX>();
      const auto & M = *mesh;
      const auto & CM = static_cast<BlockTM&>(*cmap.GetMappedMesh()); // okay, kinda hacky, the coarse mesh already exists, but only as BlockTM i think
      const auto & ctrs = *cmap.GetAggCenter();
      M.template ApplyEQ<NT_VERTEX> ([&](auto eqc, auto v) LAMBDA_INLINE {
	  auto cv = vmap[v];
	  if (cv != -1) {
	    if (ctrs.Test(v))
	      { cdata[cv].pos = data[v].pos; }
	    cdata[cv].wt += data[v].wt;
	  }
	}, true);
      cevd.SetParallelStatus(DISTRIBUTED);
    } // AttachedEVD::map_data

  }; // class AttachedEVD

  /** End Vertex Data **/


  /** Edge Data **/

  template<int DIM> struct EED_TRAIT { typedef void type; };
  template<> struct EED_TRAIT<2> { typedef Mat<3, 3, double> type; };
  template<> struct EED_TRAIT<3> { typedef Mat<6, 6, double> type; };
  template<int DIM> using ElasticityEdgeData = typename EED_TRAIT<DIM>::type;

  template<int DIM>
  class AttachedEED : public AttachedNodeData<NT_EDGE, ElasticityEdgeData<DIM>, AttachedEED<DIM>>
  {
  public:
    using BASE = AttachedNodeData<NT_EDGE, ElasticityEdgeData<DIM>, AttachedEED<DIM>>;
    using BASE::map_data, BASE::Cumulate, BASE::mesh;
    static constexpr int BS = mat_traits<ElasticityEdgeData<DIM>>::HEIGHT;

    AttachedEED (Array<ElasticityEdgeData<DIM>> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    void map_data (const BaseCoarseMap & cmap, AttachedEED<DIM> & ceed) const; // in impl header beacust I static_cast to elasticity-mesh

  }; // class AttachedEED

  /** End Edge Data **/


  /** Factory **/

  template<int DIM> using ElasticityMesh = BlockAlgMesh<AttachedEVD<DIM>, AttachedEED<DIM>>;


  template<int ADIM>
  class ElasticityAMGFactory : public VertexAMGFactory<EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>,
						       ElasticityMesh<ADIM>,
						       mat_traits<ElasticityEdgeData<ADIM>>::HEIGHT>
  {
  public:
    static constexpr int DIM = ADIM;
    using ENERGY = EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>;
    using TMESH = ElasticityMesh<DIM>;
    static constexpr int BS = ENERGY::DPV;
    using BASE = VertexAMGFactory<EpsEpsEnergy<ADIM, ElastVData<ADIM>, ElasticityEdgeData<ADIM>>,
				  ElasticityMesh<ADIM>, mat_traits<ElasticityEdgeData<ADIM>>::HEIGHT>;

    class Options : public BASE::Options
    {
    public:
      bool with_rots = false;
    }; // class ElasticityAMGFactory::Options

  protected:
    using BASE::options;

  public:
    ElasticityAMGFactory (shared_ptr<Options> _opts)
      : BASE(_opts)
    { ; }

    /** Misc **/
    virtual void CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
    {
      auto intr = [&](auto tmmm) {
	cout << endl;
	auto mvd = get<0>(tmmm->Data())->Data();
	cout << " eqc vs: " << endl;
	tmmm->template ApplyEQ2<NT_VERTEX>([&](auto eqc, auto nodes) {
	    if (nodes.Size() > 0)
	      cout << " eqc " << eqc << " = [" << nodes[0] << " ... " << nodes.Last() << "]" << endl;
	    else
	      cout << " eqc " << eqc << " = []" << endl;
	  }, false);
	cout << " v data: " << endl;
	for (auto k : Range(mvd))
	  { cout << k << " " << mvd[k] << endl; }
	cout << endl;
      };
      cout << " Check KVECS, meshes" << endl;
      for (auto l : Range(amg_levels)) {
	cout << " level " << l << endl;
	if (auto cap = amg_levels[l]->cap) {
	  if (auto mesh = cap->mesh) {
	    intr(static_pointer_cast<TMESH>(mesh));
	  }
	  else
	    { cout << " no mesh " << endl; }
	}
	else
	  { cout << " no cap " << endl; }
      }
      auto chkab = [&](auto fva, auto fvb, int n, auto fnodes, string title) {
	if ( n == 0)
	  { return; }
	cout << " check " << title << endl;
	int bs = fva.Size() / n;
	int numdf = 0;
	for (auto k : Range(n)) {
	  if ( (fnodes==nullptr) || (fnodes->Test(k)) ) {
	    double df = 0;
	    for (auto l : Range(k*bs, (k+1)*bs))
	      { df += sqr(fva(l)-fvb(l)); }
	    df = sqrt(df);
	    if (df > 1e-14) {
	      numdf++;
	    cout << " DIFF " << k << " norm = " << df << ", diff = ";
	    for (auto l : Range(k*bs, (k+1)*bs))
	      { cout << "(" << fva(l) << "/" << fvb(l) << "/" << fva(l)-fvb(l) << ") "; }
	    cout << endl;
	    }
	  }
	}
	cout << " done " << title << ", mismates = " << numdf << endl;
	if (fnodes != nullptr)
	  { cout << " fnodes non-set = " << fnodes->Size() - fnodes->NumSet() << endl; }
      };
      auto set_kvec = [&](auto & vec, int kvnr, BaseAMGFactory::AMGLevel & alev) {
	typename ENERGY::TVD opt(0); /** just an arbitrary point ... **/
	typename ENERGY::TM Q; SetIdentity(Q);
	Vec<BS, double> vcos, ovec;
	ovec = 0; ovec(kvnr) = 1;
	auto mesh = static_pointer_cast<TMESH>(alev.cap->mesh);
	auto vdata = get<0>(mesh->Data())->Data();
	vec.SetParallelStatus(CUMULATED);
	vec.FVDouble() = 0;
	cout << " set kvec : " << endl;
	if (mesh->template GetNN<NT_VERTEX>() > 0) {
	  // opt = vdata[0]; /// DONT DO THAT !!! compares different kvecs then !!!
	  int vbs = vec.FVDouble().Size()/mesh->template GetNN<NT_VERTEX>();
	  auto fv = vec.FVDouble();
	  for (auto vnr : Range(mesh->template GetNN<NT_VERTEX>())) {
	    ENERGY::CalcQHh(opt, vdata[vnr], Q);
	    vcos = Q * ovec;
	    for (int l = 0; l < vbs; l++)
	      { fv(vbs * vnr + l) = vcos(l); }
	    cout << vnr << " = ";
	    for (int l = 0; l < vbs; l++)
	      cout << fv(l) << " ";
	    cout << endl;
	  }
	  cout << endl;
	}
      };

      for (int kvnr = 0; kvnr < BS; kvnr++) {
	auto gcomm = amg_levels[0]->cap->eqc_h->GetCommunicator();
	int nlevsglob = gcomm.AllReduce(amg_levels.Size(), MPI_MAX);
	int nlevsloc = map->GetNLevels();
	unique_ptr<BaseVector> cvec = move(map->CreateVector(nlevsloc-1));
	if ( (nlevsloc == nlevsglob) && (cvec != nullptr) )
	  { set_kvec(*cvec, kvnr, *amg_levels[nlevsloc-1]); }
	for (int lev = nlevsglob-2; lev >= 0; lev--) {
	  unique_ptr<BaseVector> fvec1 = map->CreateVector(lev), fvec2 = map->CreateVector(lev);
	  map->TransferC2F(lev, fvec1.get(), cvec.get());
	  set_kvec(*fvec2, kvnr, *amg_levels[lev]);
	  chkab(fvec1->FVDouble(), fvec2->FVDouble(), amg_levels[lev]->cap->mesh->template GetNN<NT_VERTEX>(), amg_levels[lev]->cap->free_nodes,
		string("kvec ") + to_string(kvnr) + string(" on lev ") + to_string(lev));
	  cvec = move(fvec2);
	}
      }
    }

  }; // class ElasticityAMGFactory

  /** END Factory **/

} // namespace amg


namespace ngcore
{

  /** MPI extensions **/

  template<> struct MPI_typetrait<amg::ElastVData<2>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
  	{
  	  int block_len[2] = { 1, 1 };
  	  MPI_Aint displs[2] = { 0, sizeof(ngbla::Vec<2,double>) };
	  MPI_Datatype types[2] = { GetMPIType<ngbla::Vec<2,double>>(), GetMPIType<double>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  };


  template<> struct MPI_typetrait<amg::ElastVData<3>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
  	{
  	  int block_len[2] = { 1, 1 };
  	  MPI_Aint displs[2] = { 0, sizeof(ngbla::Vec<3,double>) };
	  MPI_Datatype types[2] = { GetMPIType<ngbla::Vec<3,double>>(), GetMPIType<double>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  };

  /** END MPI extensions **/

}; // namespace ngcore

#endif
#endif
