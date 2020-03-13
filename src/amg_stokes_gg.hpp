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

  /** StokesEnergy **/

  template<int DIM> using GGStokesEnergy = StokesEnergy<H1Energy<DIM, typename GGSVD<DIM>::TVD, typename GGSED<DIM>::TED>, GGSVD<DIM>, GGSED<DIM>>;

  /** END StokesEnergy **/



  /** StokesMesh **/

  template<int DIM> using GGStokesMesh = BlockAlgMesh<AttachedSVD<GGSVD<DIM>>,
						      AttachedSED<GGSED<DIM>> >;

  /** END StokesMesh **/

  /** H1 Stokes Attached Data **/

  template<class C> template<class TMESH>
  INLINE void AttachedSVD<C> :: map_data (const CoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const
  {
    throw Exception("this should never happen!");
  } // AttachedSVD<C> :: map_data


  template<class C> template<class TMESH>
  INLINE void AttachedSVD<C> :: map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const
  {
    auto & cdata = cevd.data;
    cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0.0;
    static_cast<TMESH*>(mesh)->CumulateData(); // lmao ...
    auto vmap = cmap.template GetMap<NT_VERTEX>();
    mesh->template Apply<NT_VERTEX>([&](auto v) {
	auto cv = vmap[v];
	if (cv != -1) {
	  cdata[cv].vol += data[v].vol;
	  cdata[cv].vd += data[v].vd;
	}
      }, true);
    cevd.SetParallelStatus(DISTRIBUTED);
  } // AttachedSVD<C> :: map_data


  template<class C>
  void AttachedSED<C> :: map_data (const BaseCoarseMap & cmap, AttachedSED<TED> & ceed) const
  {
    auto & cdata = ceed.data;
    auto ftm = static_pointer_cast<GGStokesMesh<C::DIM>>(cmap.GetMesh());
    auto fvdata = get<0>(ftm->Data())->Data();
    auto ctm = static_pointer_cast<GGStokesMesh<C::DIM>>(cmap.GetMappedMesh());
    auto cvdata = get<0>(ctm->Data())->Data();
    cdata.SetSize(cmap.template GetMappedNN<NT_EDGE>()); cdata = 0.0;
    auto vmap = cmap.template GetMap<NT_VERTEX>();
    auto emap = cmap.template GetMap<NT_EDGE>();
    auto cedges = ctm->template GetNodes<NT_EDGE>();
    // typename GGStokesEnergy<C::DIM>::TVD vdh, vdH;
    typename GGStokesEnergy<C::DIM>::TM QHh; SetIdentity(QHh);
    mesh->template Apply<NT_EDGE>([&](const auto & e) {
	auto cenr = emap[e.id];
	if (cenr != -1) {
	  const auto & ce = cedges[cenr];
	  // vdh = GGStokesEnergy<C::DIM>::CalcMPData(fvdata[e.v[0]], fvdata[e.v[1]]);
	  // vdH = GGStokesEnergy<C::DIM>::CalcMPData(cvdata[ce.v[0]], cvdata[ce.v[1]]);
	  // GGStokesEnergy<C::DIM>::ModQHh(vdH, vdh, QHh);
	  if (vmap[e.v[0]] == ce.v[0]) {
	    cdata[cenr].edi += data[e.id].edi;
	    cdata[cenr].edj += data[e.id].edj;
	  }
	  else {
	    cdata[cenr].edj += data[e.id].edi;
	    cdata[cenr].edi += data[e.id].edj;
	  }
	  Iterate<C::DIM>([&](auto i) { cdata[cenr].flow[i.value] += InnerProduct(data[e.id].flow, QHh.Col(i.value)); });
	}
      }, true);
    ceed.SetParallelStatus(CUMULATED);
  } // AttachedSED :: map_data

  /** END H1 Stokes Attached Data **/



#ifdef FILE_AMG_PC_STOKES_HPP
/** StokesAMGPC **/

  template<class FACTORY, class AUX_SYS> shared_ptr<typename StokesAMGPC<FACTORY, AUX_SYS>::TMESH>
  StokesAMGPC<FACTORY, AUX_SYS> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const
  {
    constexpr int DIM = FACTORY::ENERGY::DIM;

    /** we already get an ALG-Mesh, but it only has some algebraic data (v-pos and v-vol) **/
    auto alg_mesh = dynamic_pointer_cast<TMESH>(top_mesh);

    // actually, not sure what to do here ... in triv case, nothing I guess
    auto fdata = get<1>(alg_mesh->Data())->Data();

    // TODO: PARSTAT is NOT yet correct - we shoudl check wether we are master of facet everywhere
    auto edata = get<1>(alg_mesh->Data())->Data();
    auto flows = aux_sys->CalcFacetFlow();
    auto edges = alg_mesh->template GetNodes<NT_EDGE>();
    FlatArray<int> vsort = node_sort[NT_VERTEX];
    FlatArray<int> esort = node_sort[NT_EDGE];
    Array<int> elnums;
    auto comm = aux_sys->GetAuxParDofs()->GetCommunicator();
    auto f2a_facet = aux_sys->GetFMapF2A();
    for (auto k : Range(f2a_facet)) {
      auto facet_nr = f2a_facet[k];
      auto enr = esort[f2a_facet[k]]; // TODO: garbage with MPI
      const auto & edge = edges[enr];
      ma->GetFacetElements(facet_nr, elnums);
      double fac = 1.0;
      if (elnums.Size() == 2) {
	if (vsort[elnums[0]] == edge.v[0])
	  { fac = 1.0; }
	else if (vsort[elnums[0]] == edge.v[1])
	  { fac = -1; }
	else
	  { throw Exception("ummm .. WTF??"); }
      }
      else {
	// I think GetDistantProcs for non-parallel mesh just segfaults ...
	if ( ( comm.Size() > 1 ) && ma->GetDistantProcs(NodeId(NT_FACET, facet_nr)).Size() ) {
	  throw Exception("I dont even know ...");
	}
	else {
	  fac = (vsort[elnums[0]] == edge.v[0]) ? 1.0 : -1.0; // should only flip if surface vertex is sorted before vol vertex - never without MPI??
	}
      }
      edata[enr].flow = fac * flows[k];
      edata[enr].edi = 1;
      edata[enr].edj = 1;
    }

    return alg_mesh;
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
