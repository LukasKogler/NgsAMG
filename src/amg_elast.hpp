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

  /** blocks size for differnt dimensions **/
  template<int DIM> struct ETM_TRAIT { typedef void type; };
  template<> struct ETM_TRAIT<2> { typedef Mat<3, 3, double> type; };
  template<> struct ETM_TRAIT<3> { typedef Mat<6, 6, double> type; };


  /** Vertex Data **/

  template<int DIM>
  class ElastVData
  {
  public:
    using TM = typename ETM_TRAIT<DIM>::type;
    Vec<DIM, double> pos;
    TM wt;
    ElastVData (double val) : pos(val), wt(0) { SetScalIdentity(val, wt); }
    ElastVData () : ElastVData(0) { ; }
    ElastVData (Vec<DIM,double> _pos, double _wt) : pos(_pos), wt(0) { SetScalIdentity(_wt, wt); }
    ElastVData (Vec<DIM,double> && _pos, double && _wt) : pos(_pos), wt(0) { SetScalIdentity(_wt, wt); }
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

    template<class TMESH> INLINE void map_data (const CoarseMap<TMESH> & cmap, AttachedEVD<DIM> & cevd) const;

    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedEVD<DIM> & cevd) const;

  }; // class AttachedEVD

  /** End Vertex Data **/


  /** Edge Data **/

  // template<int DIM> struct EED_TRAIT { typedef void type; };
  // template<> struct EED_TRAIT<2> { typedef Mat<3, 3, double> type; };
  // template<> struct EED_TRAIT<3> { typedef Mat<6, 6, double> type; };
  // template<int DIM> using ElasticityEdgeData = typename EED_TRAIT<DIM>::type;

  template<int DIM> using ElasticityEdgeData = typename ETM_TRAIT<DIM>::type;

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
    void CheckKVecs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map) override;

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
	  MPI_Datatype types[2] = { GetMPIType<ngbla::Vec<2,double>>(), GetMPIType<typename ElastVData<2>::TM>() };
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
	  MPI_Datatype types[2] = { GetMPIType<ngbla::Vec<3,double>>(), GetMPIType<typename ElastVData<3>::TM>() };
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
