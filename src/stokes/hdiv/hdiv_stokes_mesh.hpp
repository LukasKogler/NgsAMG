#ifndef FILE_NC_STOKES_MESH_HPP
#define FILE_NC_STOKES_MESH_HPP

#include <stokes_mesh.hpp>
#include <stokes_energy.hpp>
#include <h1_energy.hpp>

// include the H1 energy impl header (plus required utility headers)
// here since there is no Stokes energy impl header
#include <utils_denseLA.hpp>
#include <h1_energy_impl.hpp>

namespace amg
{

/** Stokes raw Data  **/

template<int ADIM, int AMAX_DOFS, class ATED = double>
struct HDivStokesEData
{
  static constexpr int DIM      = ADIM;
  static constexpr int MAX_DOFS = AMAX_DOFS;
  using TED   = ATED;
  using TFLOW = Vec<MAX_DOFS, double>;
  
  TED edi;      // energy contribution v_i-f_ij 
  TED edj;      // energy contribution v_j-f_ij
  TFLOW flow;   // flow of base functions
  
  HDivStokesEData (double val) : edi(val), edj(val), flow(val) { ; }
  HDivStokesEData () : HDivStokesEData(0) { ; }
  HDivStokesEData (TED _edi, TED _edj, Vec<MAX_DOFS, double> _flow) : edi(_edi), edj(_edj), flow(_flow) { ; }
  HDivStokesEData (TED && _edi, TED && _edj, Vec<MAX_DOFS, double> && _flow) : edi(std::move(_edi)), edj(std::move(_edj)), flow(std::move(_flow)) { ; }
  HDivStokesEData (HDivStokesEData<DIM, MAX_DOFS, TED> && other) : edi(std::move(other.edi)), edj(std::move(other.edj)), flow(std::move(other.flow)) { ; }
  HDivStokesEData (const HDivStokesEData<DIM, MAX_DOFS, TED> & other) : edi(other.edi), edj(other.edj), flow(other.flow) { ; }
  INLINE void operator = (double x) { edi = x; edj = x; flow = x; }
  INLINE void operator = (const HDivStokesEData<DIM, MAX_DOFS, TED> & other) { edi = other.edi; edj = other.edj; flow = other.flow; }
  INLINE void operator += (const HDivStokesEData<DIM, MAX_DOFS, TED> & other) { edi += other.edi; edj += other.edj; flow += other.flow; }
  INLINE void operator == (const HDivStokesEData<DIM, MAX_DOFS, TED> & other) { return (edi == other.edi) && (edj == other.edj) && (flow = other.flow); }
}; // struct HDivStokesEData

template<int DIM, int MAX_DOFS, class TED>
INLINE std::ostream & operator<< (std::ostream & os, HDivStokesEData<DIM, MAX_DOFS, TED> & e)
{
  os << "[f:" << e.flow << " | ij:" << e.edi << " | ji:" << e.edj << "]";
  return os;
}

template<int DIM, int MAX_DOFS, class TED>
INLINE bool is_zero (const HDivStokesEData<DIM, MAX_DOFS, TED> & ed)
{
  return is_zero(ed.edi) && is_zero(ed.edj) && is_zero(ed.flow);
}

/** END Stokes raw Data **/


/** StokesEnergy **/

// preserves constants + "n"
template<int DIM> using HDivStokesGGVD     = StokesVData<DIM, double>;
template<int DIM> using HDivStokesGGED     = HDivStokesEData<DIM, 1 + DIM, double>;
template<int DIM> using HDivGGStokesEnergy = StokesEnergy<H1Energy<1, double, double>, // <- SPM-BS is taken from energy
                                                          HDivStokesGGVD<DIM>,
                                                          HDivStokesGGED<DIM>>;

/** END StokesEnergy **/


/** Stokes Attached Data **/


template<class ATED>
class AttachedSED : public AttachedNodeData<NT_EDGE, ATED>
{
public:
  using TED = ATED;
  static constexpr int DIM = TED::DIM;
  using BASE = AttachedNodeData<NT_EDGE, ATED>;
  using BASE::mesh;
  using BASE::data;

  AttachedSED (Array<TED> && _data, PARALLEL_STATUS stat)
    : BASE(std::move(_data), stat)
  { ; }

  void map_data (const BaseCoarseMap & cmap, AttachedSED<TED> *ceed) const;
}; // class AttachedSED

/** END Stokes Attached Data **/


/** GGStokesMesh */

template<int DIM> using HDivGGStokesMesh = StokesMesh<AttachedSVD<HDivStokesGGVD<DIM>>,
                                                      AttachedSED<HDivStokesGGED<DIM>>>;


/** END GGStokesMesh */

} // namespace amg


namespace ngcore
{
  template<> struct MPI_typetrait<amg::HDivStokesGGED<2>> {
    static NG_MPI_Datatype MPIType () {
      static NG_MPI_Datatype NG_MPI_T = 0;
      if (!NG_MPI_T)
      {
        int block_len[2] = { 2, 1 };
        NG_MPI_Aint displs[2] = { 0, 2 * sizeof(amg::HDivStokesGGED<2>::TED) };
        NG_MPI_Datatype types[2] = { GetMPIType<amg::HDivStokesGGED<2>::TED>(), GetMPIType<amg::HDivStokesGGED<2>::TFLOW>() };
        NG_MPI_Type_create_struct(2, block_len, displs, types, &NG_MPI_T);
        NG_MPI_Type_commit ( &NG_MPI_T );
      }
      return NG_MPI_T;
    }
  }; // struct MPI_typetrait


  template<> struct MPI_typetrait<amg::HDivStokesGGED<3>> {
    static NG_MPI_Datatype MPIType () {
      static NG_MPI_Datatype NG_MPI_T = 0;
      if (!NG_MPI_T)
      {
        int block_len[2] = { 2, 1 };
        NG_MPI_Aint displs[2] = { 0, 2 * sizeof(amg::HDivStokesGGED<3>::TED) };
        NG_MPI_Datatype types[2] = { GetMPIType<amg::HDivStokesGGED<3>::TED>(), GetMPIType<amg::HDivStokesGGED<3>::TFLOW>() };
        NG_MPI_Type_create_struct(2, block_len, displs, types, &NG_MPI_T);
        NG_MPI_Type_commit ( &NG_MPI_T );
      }
      return NG_MPI_T;
    }
  }; // struct MPI_typetrait

} // namespace ngcore

#endif // FILE_NC_STOKES_MESH_HPP
