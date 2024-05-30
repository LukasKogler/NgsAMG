#ifndef FILE_ELASTICITY_MESH_HPP
#define FILE_ELASTICITY_MESH_HPP

#include <alg_mesh.hpp>
#include <utils_buffering.hpp>

#include "elasticity_energy.hpp"

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

  static constexpr int DOUBLE_BUFFER_SIZE = 1 + SIZE_IN_BUFFER<Vec<DIM,double>>() + SIZE_IN_BUFFER<Mat<DIM,DIM,double>>();

  Vec<DIM, double> pos;
  TM wt;
  double rot_scaling; // scaling of rotations for better numerical stability
  ElastVData (double val) : pos(val), wt(0), rot_scaling(1.0) { SetScalIdentity(val, wt); }
  ElastVData () : ElastVData(0) { ; }
  ElastVData (Vec<DIM,double> _pos, double _wt, double _rot_scaling = 1.0)
    : pos(_pos), wt(0), rot_scaling(_rot_scaling)
  { SetScalIdentity(_wt, wt); }
  ElastVData (Vec<DIM,double> && _pos, double && _wt, double _rot_scaling = 1.0)
    : pos(_pos), wt(0), rot_scaling(_rot_scaling)
  { SetScalIdentity(_wt, wt); }
  ElastVData (ElastVData<DIM> && other) = default;
  ElastVData (const ElastVData<DIM> & other) = default;
  INLINE void operator = (double x) { pos = x; wt = x; rot_scaling = x; }
  INLINE void operator = (const ElastVData<DIM> & other) { pos = other.pos; wt = other.wt; rot_scaling = other.rot_scaling; }
  INLINE void operator += (const ElastVData<DIM> & other) { pos += other.pos; wt += other.wt; rot_scaling += other.rot_scaling; }
  INLINE bool operator == (const ElastVData<DIM> & other)
  {
    if constexpr (DIM == 2) {
      return ( wt == other.wt )
        && ( pos(0) == other.pos(0) ) && ( pos(1) == other.pos(1) ) &&
        ( rot_scaling == other.rot_scaling );
    }
    else {
      return ( wt == other.wt ) &&
        ( pos(0) == other.pos(0) ) && ( pos(1) == other.pos(1) ) && ( pos(2) == other.pos(2) )
        ( rot_scaling == other.rot_scaling );
    }
  }
}; // class ElastVData


template<int DIM> struct SIZE_IN_BUFFER_TRAIT<ElastVData<DIM>> { static constexpr int value = ElastVData<DIM>::DOUBLE_BUFFER_SIZE; };

template<int DIM>
INLINE int
PackIntoBuffer(ElastVData<DIM> const &d, double *buf)
{
  int c = 0;
  c += PackIntoBuffer(d.rot_scaling, buf + c);
  c += PackIntoBuffer(d.pos, buf + c);
  c += PackIntoBuffer(d.wt, buf + c);
  return SIZE_IN_BUFFER<ElastVData<DIM>>();
}

template<int DIM>
INLINE int
UnpackFromBuffer(ElastVData<DIM> &d, double const *buf)
{
  int c = 0;
  c += UnpackFromBuffer(d.rot_scaling, buf + c);
  c += UnpackFromBuffer(d.pos, buf + c);
  c += UnpackFromBuffer(d.wt, buf + c);
  return SIZE_IN_BUFFER<ElastVData<DIM>>();
}


template<int DIM>
INLINE std::ostream&
operator << (std::ostream & os, ElastVData<DIM> const &V)
{
  os << "[ pos: " << V.pos << " | rot-scale: " << V.rot_scaling << " | wt: " << V.wt << "]";
  return os;
}

template<int DIM>
INLINE bool
is_zero (ElastVData<DIM> const &m)
{
  return is_zero(m.wt) && is_zero(m.pos);
}

/** End Vertex Data **/


/** Edge Data **/

// template<int DIM> struct EED_TRAIT { typedef void type; };
// template<> struct EED_TRAIT<2> { typedef Mat<3, 3, double> type; };
// template<> struct EED_TRAIT<3> { typedef Mat<6, 6, double> type; };
// template<int DIM> using ElasticityEdgeData = typename EED_TRAIT<DIM>::type;

template<int DIM> using ElasticityEdgeData = typename ETM_TRAIT<DIM>::type;


/** End Edge Data **/

} // namespace amg



namespace ngcore
{

  /** MPI extensions **/

  template<> struct MPI_typetrait<amg::ElastVData<2>> {
    static NG_MPI_Datatype MPIType () {
      static NG_MPI_Datatype NG_MPI_T = 0;
      if (!NG_MPI_T)
      {
        int block_len[3] = { 1, 1, 1 };
        NG_MPI_Aint displs[3] = { 0, sizeof(ngbla::Vec<2,double>), sizeof(ngbla::Vec<2,double>) + sizeof(amg::ElastVData<2>::TM) };
        NG_MPI_Datatype types[3] = { GetMPIType<ngbla::Vec<2,double>>(), GetMPIType<typename amg::ElastVData<2>::TM>(), NG_MPI_DOUBLE };
        NG_MPI_Type_create_struct(3, block_len, displs, types, &NG_MPI_T);
        NG_MPI_Type_commit ( &NG_MPI_T );
      }
      return NG_MPI_T;
    }
  };


  template<> struct MPI_typetrait<amg::ElastVData<3>> {
    static NG_MPI_Datatype MPIType () {
      static NG_MPI_Datatype NG_MPI_T = 0;
      if (!NG_MPI_T)
      {
        int block_len[3] = { 1, 1, 1 };
        NG_MPI_Aint displs[3] = { 0, sizeof(ngbla::Vec<3,double>), sizeof(ngbla::Vec<3,double>) +  sizeof(amg::ElastVData<3>::TM)};
        NG_MPI_Datatype types[3] = { GetMPIType<ngbla::Vec<3,double>>(), GetMPIType<typename amg::ElastVData<3>::TM>(), NG_MPI_DOUBLE };
        NG_MPI_Type_create_struct(3, block_len, displs, types, &NG_MPI_T);
        NG_MPI_Type_commit ( &NG_MPI_T );
      }
      return NG_MPI_T;
    }
  };

  /** END MPI extensions **/

}; // namespace ngcore

#endif // FILE_ELASTICITY_MESH_HPP