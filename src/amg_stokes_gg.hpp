#ifdef STOKES

#ifndef FILE_STOKES_GG_HPP
#define FILE_STOKES_GG_HPP

namespace amg
{
  /** H1 Stokes Data */

  template<int DIM>
  using GGSVD = StokesVData<DIM, double>;

  template<int DIM>
  using GGSED = StokesEData<DIM, DIM, double>;

  /** END H1 Stokes Data */


  /** H1 Stokes Attached Data **/

  template<class C> template<class TMESH>
  INLINE void AttachedSVD<C> :: map_data (const CoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const
  {

  } // AttachedSVD<C> :: map_data


  template<class C> template<class TMESH>
  INLINE void AttachedSVD<C> :: map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const
  {

  } // AttachedSVD<C> :: map_data


  template<class C>
  void AttachedSED<C> :: map_data (const BaseCoarseMap & cmap, AttachedSED<TED> & ceed) const
  {

  } // AttachedSED :: map_data

  /** END H1 Stokes Attached Data **/


  /** StokesMesh **/

  template<int DIM> using GGStokesMesh = BlockAlgMesh<AttachedSVD<GGSVD<DIM>>,
						      AttachedSED<GGSED<DIM>> >;

  /** END StokesMesh **/


  /** StokesEnergy **/

  template<int DIM> using GGStokesEnergy = StokesEnergy<H1Energy<DIM, typename GGSVD<DIM>::TVD, typename GGSED<DIM>::TED>, GGSVD<DIM>, GGSED<DIM>>;

  /** END StokesEnergy **/


#ifdef FILE_AMG_PC_STOKES_HPP
/** StokesAMGPC **/

  template<class FACTORY, class AUX_SYS> shared_ptr<typename StokesAMGPC<FACTORY, AUX_SYS>::TMESH>
  StokesAMGPC<FACTORY, AUX_SYS> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const
  {
    constexpr int DIM = FACTORY::ENERGY::DIM;

    auto a = new AttachedSVD<GGSVD<DIM>>(Array<GGSVD<DIM>>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); a->Data() = 0.0;
    auto b = new AttachedSED<GGSED<DIM>>(Array<GGSED<DIM>>(top_mesh->GetNN<NT_EDGE>()), CUMULATED); b->Data() = 0.0;

    auto mesh = make_shared<GGStokesMesh<DIM>>(move(*top_mesh), a, b);

    return mesh;
  } // StokesAMGPC::BuildAlgMesh_TRIV

  /** END StokesAMGPC **/

#endif // FILE_AMG_PC_STOKES_HPP


} // namespace amg


namespace ngcore
{

  template<> struct MPI_typetrait<amg::GGSVD<2>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
	{
	  int block_len[2] = { 1, 1};
	  MPI_Aint displs[2] = { 0, sizeof(amg::GGSVD<2>::TVD) };
	  MPI_Datatype types[2] = { GetMPIType<amg::GGSVD<2>::TVD>(), GetMPIType<double>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  }; // struct MPI_typetrait


  template<> struct MPI_typetrait<amg::GGSVD<3>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
	{
	  int block_len[2] = { 1, 1};
	  MPI_Aint displs[2] = { 0, sizeof(amg::GGSVD<3>::TVD) };
	  MPI_Datatype types[2] = { GetMPIType<amg::GGSVD<3>::TVD>(), GetMPIType<double>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  }; // struct MPI_typetrait


  template<> struct MPI_typetrait<amg::GGSED<2>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
	{
	  int block_len[2] = { 2, 1 };
	  MPI_Aint displs[2] = { 0, 2 * sizeof(amg::GGSED<2>::TED) };
	  MPI_Datatype types[2] = { GetMPIType<amg::GGSED<2>::TED>(), GetMPIType<ngbla::Vec<amg::GGSED<2>::BS, double>>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  }; // struct MPI_typetrait


  template<> struct MPI_typetrait<amg::GGSED<3>> {
    static MPI_Datatype MPIType () {
      static MPI_Datatype MPI_T = 0;
      if (!MPI_T)
	{
	  int block_len[2] = { 2, 1 };
	  MPI_Aint displs[2] = { 0, 2 * sizeof(amg::GGSED<3>::TED) };
	  MPI_Datatype types[2] = { GetMPIType<amg::GGSED<3>::TED>(), GetMPIType<ngbla::Vec<amg::GGSED<3>::BS, double>>() };
  	  MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
  	  MPI_Type_commit ( &MPI_T );
	}
      return MPI_T;
    }
  }; // struct MPI_typetrait

} // namespace ngcore

#endif
#endif // STOKES
