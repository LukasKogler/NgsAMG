#ifdef STOKES

#ifndef FILE_AMG_STOKES_HPP
#define FILE_AMG_STOKES_HPP

namespace amg
{

  /** Stokes AMG (grad-grad + div-div penalty):
      We assume that we have DIM DOFs per facet of the mesh. Divergence-The divergence 
       - DOFs are assigned to edges of the dual mesh.
   **/

  /** Laplace + div-div: displacements **/
  template<int D> struct StokesTM_TRAIT { typedef void type; };
  template<> struct StokesTM_TRAIT<2> { typedef Mat<2,2,double> type; };
  template<> struct StokesTM_TRAIT<3> { typedef Mat<3,3,double> type; };
  template<int D> using StokesTM = typename StokesTM_TRAIT<D>::type;


  template<int D>
  struct H1StokesVData
  {
    double area;
  }; // StokesH1VData


  // template<int D>
  // struct EpsEpsStokesVData
  // {
  //   Vec<D> pos;
  //   double area;
  // }; // StokesEpsVData


  template<int D>
  class AttachedSVD : public AttachedNodeData<NT_VERTEX, StokesVData<D>, AttachedSVD<D>>
  {
  public:
    using BASE = AttachedNodeData<NT_VERTEX, StokesVData<D>, AttachedSVD<D>>;
    using BASEL::map_data;

    AttachedSVD (Array<StokesVData<D>> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    INLINE void map_data (const BaseCoarseMap & cmap, AttachedSVD & csvd) const
    {
      Cumulate();
      auto cdata = csvd.data; cdata.SetSize(map.GetMappedNN<NT>());
      cdata = 0;
      Array<int> cnt (cdata.Size()); cnt = 0;
      auto map = cmap.GetMap<NT_VERTEX>();
      mesh->Apply<NT_VERTEX>([&](auto v) {
	  auto cv = map[v];
	  if (cv != -1) {
	    cnt[cv]++;
	    cdata[cv].pos += data[v].pos;
	    cdata[cv].area += data[v].area;
	  }
	}, true);
      mesh->AllreduceNodalData<NT_VERTEX>(cnt, [](auto & in) { return move(sum_table(in)); } , false);
      for (auto k : Range(cdata))
	{ cdata[k].pos *= 1.0/cnt[k]; }
      csvd.SetParallelStatus(DISTRIBUTED);
    } // AttachedSVD::map_data

  }; // class AttachedSVD


  template<int D>
  struct H1StokesEData
  {
    H1Energy<D>::TM mat;
    Vec<D,double> flow; // \int e_1i*n
  };


  template<int D>
  class AttachedSED : public AttachedEdgeData<NT_EDGE, StokesEData<D>, AttachedSED<D>>
  {
  public:
    using BASE = public AttachedEdgeData<NT_EDGE, StokesEData<D>, AttachedSED<D>>;
    using BASE::map_data;

    template<class TMESH> INLINE void map_data (const BaseCoarseMap & map, AttachedSED<D> & csed) const;

  }; // class AttachedSED


  template<int D>
  using H1StokesMesh = BlockAlgMesh<AttachedH1SVD<D>, AttachedH1SED<D>>;

  // template<int D>
  // using EpsEpsStokesMesh = BlockAlgMesh<AttachedEpsEpsSVD<D>, AttachedEpsEpsSED<D>>;


  template<int DIM, class ATMESH, class TENERGY>
  class StokesAMGFactory : public NodalAMGFactory<NT_EDGE, ATMESH, TENERGY::TM>
  {
    // static_assert( std::is_same<ENERGY::TVD, StokesEData<DIM>>::value, "stokes factory with wrong vertex data!");
  public:
    using TMESH = ATMESH;
    using ENERGY = TENERGY;
    using TM = AMG_CLASS::TM;
  protected:
  public:
  protected:
    template<class TMAP> shared_ptr<TSPM_TM> BuildPWProl_impl (shared_ptr<TMAP> cmap, shared_ptr<ParallelDofs> fpd) const;
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<CoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const override
    { return BuildPWProl_impl(cmap, fpd); }
    virtual shared_ptr<TSPM_TM> BuildPWProl (shared_ptr<AgglomerateCoarseMap<TMESH>> cmap, shared_ptr<ParallelDofs> fpd) const override
    { return BuildPWProl_impl(cmap, fpd); }
  }; // class StokesAMGFactory


  // /** Stokes AMG Preconditioner for facet-nodal discretizations. E.g. from an auxiliary space.
  //     Does not directly work with HDiv spaces. **/
  // template<class TFACTORY>
  // class StokesAMGPC : public Preconditioner
  // {
  // }; // StokesAMGPC


} // namespace amg

#endif

#endif
