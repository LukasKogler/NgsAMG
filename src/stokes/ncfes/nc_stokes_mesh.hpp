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

template<int ADIM, int ABS, class ATED>
struct StokesEData
{
  static constexpr int DIM = ADIM;
  static constexpr int BS = ABS;
  using TED = ATED;
  using TFLOW = Vec<BS, double>;
  
  TED edi;      // energy contribution v_i-f_ij 
  TED edj;      // energy contribution v_j-f_ij
  TFLOW flow;   // flow of base functions
  
  StokesEData (double val) : edi(val), edj(val), flow(val) { ; }
  StokesEData () : StokesEData(0) { ; }
  StokesEData (TED _edi, TED _edj, Vec<BS, double> _flow) : edi(_edi), edj(_edj), flow(_flow) { ; }
  StokesEData (TED && _edi, TED && _edj, Vec<BS, double> && _flow) : edi(std::move(_edi)), edj(std::move(_edj)), flow(std::move(_flow)) { ; }
  StokesEData (StokesEData<DIM, BS, TED> && other) : edi(std::move(other.edi)), edj(std::move(other.edj)), flow(std::move(other.flow)) { ; }
  StokesEData (const StokesEData<DIM, BS, TED> & other) : edi(other.edi), edj(other.edj), flow(other.flow) { ; }
  INLINE void operator = (double x) { edi = x; edj = x; flow = x; }
  INLINE void operator = (const StokesEData<DIM, BS, TED> & other) { edi = other.edi; edj = other.edj; flow = other.flow; }
  INLINE void operator += (const StokesEData<DIM, BS, TED> & other) { edi += other.edi; edj += other.edj; flow += other.flow; }
  INLINE void operator == (const StokesEData<DIM, BS, TED> & other) { return (edi == other.edi) && (edj == other.edj) && (flow = other.flow); }
}; // struct StokesEData

template<int DIM, int BS, class TED>
INLINE std::ostream & operator<< (std::ostream & os, StokesEData<DIM, BS, TED> & e)
{
  os << "[" << e.flow << " | " << e.edi << " | " << e.edj << "]";
  return os;
}

template<int DIM, int BS, class TED>
INLINE bool is_zero (const StokesEData<DIM, BS, TED> & ed)
{
  return is_zero(ed.edi) && is_zero(ed.edj) && is_zero(ed.flow);
}

/** END Stokes raw Data **/


/** StokesEnergy **/

template<int DIM> using NCStokesGGVD = StokesVData<DIM, double>;
template<int DIM> using NCStokesGGED = StokesEData<DIM, DIM, double>;
template<int DIM> using NCGGStokesEnergy = StokesEnergy<H1Energy<DIM, double, double>,
                                                        NCStokesGGVD<DIM>,
                                                        NCStokesGGED<DIM>>;

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

template<int DIM> using NCGGStokesMesh = StokesMesh<AttachedSVD<NCStokesGGVD<DIM>>,
                                                    AttachedSED<NCStokesGGED<DIM>>>;


/** END GGStokesMesh */

} // namespace amg


namespace ngcore
{
  template<> struct MPI_typetrait<amg::NCStokesGGED<2>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
      {
        int block_len[2] = { 2, 1 };
        MPI_Aint displs[2] = { 0, 2 * sizeof(amg::NCStokesGGED<2>::TED) };
        MPI_Datatype types[2] = { GetMPIType<amg::NCStokesGGED<2>::TED>(), GetMPIType<amg::NCStokesGGED<2>::TFLOW>() };
        MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
        MPI_Type_commit ( &MPI_T );
      }
      return MPI_T;
    }
  }; // struct MPI_typetrait


  template<> struct MPI_typetrait<amg::NCStokesGGED<3>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
      {
        int block_len[2] = { 2, 1 };
        MPI_Aint displs[2] = { 0, 2 * sizeof(amg::NCStokesGGED<3>::TED) };
        MPI_Datatype types[2] = { GetMPIType<amg::NCStokesGGED<3>::TED>(), GetMPIType<amg::NCStokesGGED<3>::TFLOW>() };
        MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
        MPI_Type_commit ( &MPI_T );
      }
      return MPI_T;
    }
  }; // struct MPI_typetrait

} // namespace ngcore

#endif // FILE_NC_STOKES_MESH_HPP