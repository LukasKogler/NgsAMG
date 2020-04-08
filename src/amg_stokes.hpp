#ifdef STOKES

#ifndef FILE_AMG_STOKES_HPP
#define FILE_AMG_STOKES_HPP

namespace amg
{

  /** Stokes AMG (grad-grad + div-div penalty):
      We assume that we have DIM DOFs per facet of the mesh. Divergence-The divergence 
       - DOFs are assigned to edges of the dual mesh.
   **/

  /** Stokes Data **/

  template<int ADIM, class ATVD>
  struct StokesVData
  {
    static constexpr int DIM = ADIM;
    using TVD = ATVD;
    TVD vd;
    double vol;                  // if positive, the volume. if negative, the vertex is fictitious [[added for non-diri boundary facets]]
    INLINE bool IsReal () { return vol > 0; }     // a regular vertex that stands for a volume
    INLINE bool IsImag () { return vol < 0; }     // an imaginary vertex, appended for boundary facets
    INLINE bool IsWeird () { return vol == 0; }   // a temporary vertex, usually from CalcMPData
    StokesVData (double val) : vd(val), vol(val) { ; }
    StokesVData () : StokesVData (0) { ; }
    StokesVData (TVD _vd, double _vol) : vd(_vd), vol(_vol) { ; }
    StokesVData (TVD && _vd, double && _vol) : vd(move(_vd)), vol(move(_vol)) { ; }
    StokesVData (StokesVData<DIM, TVD> && other) : vd(move(other.vd)), vol(move(other.vol)) { ; }
    StokesVData (const StokesVData<DIM, TVD> & other) : vd(other.vd), vol(other.vol) { ; }
    INLINE void operator = (double x) { vd = x; vol = x; }
    INLINE void operator = (const StokesVData<DIM, TVD> & other) { vd = other.vd; vol = other.vol; }
    INLINE void operator += (const StokesVData<DIM, TVD> & other) { vd += other.vd; vol += other.vol; }
    INLINE bool operator == (const StokesVData<DIM, TVD> & other) { return (vd == other.vd) && (vol == other.vol); }
  }; // struct StokesVData

  template<int DIM, class TVD> INLINE std::ostream & operator << (std::ostream & os, StokesVData<DIM, TVD> & v)
  { os << "[" << v.vol << " | " << v.vd << "]"; return os; }

  template<int DIM, class TVD> INLINE bool is_zero (const StokesVData<DIM, TVD> & vd) { return is_zero(vd.vd) && is_zero(vd.vol); }


  template<int ADIM, int ABS, class ATED>
  struct StokesEData
  {
    static constexpr int DIM = ADIM;
    static constexpr int BS = ABS;
    using TED = ATED;
    TED edi, edj;                // energy contribs v_i-f_ij and v_j-f_ij
    Vec<BS, double> flow;        // flow of base functions
    StokesEData (double val) : edi(val), edj(val), flow(val) { ; }
    StokesEData () : StokesEData(0) { ; }
    StokesEData (TED _edi, TED _edj, Vec<BS, double> _flow) : edi(_edi), edj(_edj), flow(_flow) { ; }
    StokesEData (TED && _edi, TED && _edj, Vec<BS, double> && _flow) : edi(move(_edi)), edj(move(_edj)), flow(move(_flow)) { ; }
    StokesEData (StokesEData<DIM, BS, TED> && other) : edi(move(other.edi)), edj(move(other.edj)), flow(move(other.flow)) { ; }
    StokesEData (const StokesEData<DIM, BS, TED> & other) : edi(other.edi), edj(other.edj), flow(other.flow) { ; }
    INLINE void operator = (double x) { edi = x; edj = x; flow = x; }
    INLINE void operator = (const StokesEData<DIM, BS, TED> & other) { edi = other.edi; edj = other.edj; flow = other.flow; }
    INLINE void operator += (const StokesEData<DIM, BS, TED> & other) { edi += other.edi; edj += other.edj; flow += other.flow; }
    INLINE void operator == (const StokesEData<DIM, BS, TED> & other) { return (edi == other.edi) && (edj == other.edj) && (flow = other.flow); }
  }; // struct StokesEData

  template<int DIM, int BS, class TED> INLINE std::ostream & operator << (std::ostream & os, StokesEData<DIM, BS, TED> & e)
  { os << "[" << e.flow << " | " << e.edi << " | " << e.edj << "]"; return os; }

  template<int DIM, int BS, class TED> INLINE bool is_zero (const StokesEData<DIM, BS, TED> & ed) { return is_zero(ed.edi) && is_zero(ed.edj) && is_zero(ed.flow); }

  /** END Stokes Data **/


  /** StokesEnergy **/

  template<class AENERGY, class ATVD, class ATED>
  class StokesEnergy
  {
  public:

    /** A wrapper around a normal energy **/

    using ENERGY = AENERGY;
    using TVD = ATVD;
    using TED = ATED;

    static constexpr int DIM = ENERGY::DIM;
    static constexpr int DPV = ENERGY::DPV;
    static constexpr bool NEED_ROBUST = ENERGY::NEED_ROBUST;

    using TM = typename ENERGY::TM;

    static INLINE double GetApproxWeight (const TED & ed) {
      double wi = ENERGY::GetApproxWeight(ed.edi), wj = ENERGY::GetApproxWeight(ed.edj);
      return ( (wi>0) && (wj>0) ) ? (wi + wj) : 0;
    }

    static INLINE double GetApproxVWeight (const TVD & vd) {
      return ENERGY::GetApproxVWeight(vd.vd);
    }

    static INLINE const TM & GetEMatrix (const TED & ed) {
      static TM emi, emj;
      emi = ENERGY::GetEMatrix(ed.edi);
      double tri = calc_trace(emi) / DPV;
      emi /= tri;
      emj = ENERGY::GetEMatrix(ed.edj);
      double trj = calc_trace(emj) / DPV;
      emj /= trj;
      double f = ( (emi > 0) && (emj > 0) ) ? ( 2 * emi * emj / (emi+emj) ) : 0;
      return f * (emi + emj);
    }

    static INLINE const TM & GetVMatrix (const TVD & vd)
    { return vd.IsReal() ? ENERGY::GetVMatrix(vd) : TM(0); }

    static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij)
    { ENERGY::CalcQij(di.vd, dj.vd, Qij); }
    static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij)
    { ENERGY::ModQij(di.vd, dj.vd, Qij); }
    static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh)
    { ENERGY::CalcQHh(dH.vd, dh.vd, QHh); }
    static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh)
    { ENERGY::CalcQHh(dH.vd, dh.vd, QHh); }
    static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
    { ENERGY::CalcQs(di.vd, dj.vd, Qij, Qji); }
    static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji)
    { ENERGY::ModQs(di.vd, dj.vd, Qij, Qji); }
    static INLINE TVD CalcMPData (const TVD & da, const TVD & db)
    {
      TVD dc;
      dc.vd = ENERGY::CalcMPData(da.vd, db.vd);
      dc.vol = 0.0; // ... idk what us appropriate ...
      return dc;
    }

    static INLINE void QtMQ(const TM & Qij, const TM & M)
    { ENERGY::QtMQ(Qij, M); }

    static INLINE void AddQtMQ(double val, TM & A, const TM & Qij, const TM & M)
    { ENERGY::AddQtMQ(val, A, Qij, M); }


    // I dont think I need this in Agglomerator !
    // static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
    // { ENERGY::CalcRMBlock(mat, GetEMatrix(ed), vdi, vdj); }

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat,
				    const TVD & vi, const TVD & vj, const TVD & vk,
				    const TED & eij, bool revij,
				    const TED & eik, bool revik)
    {
      static typename TVD::TVD vij, vik;
      static typename ENERGY::TM EM, Qij_ik, Qik_ij;

      /** facet mid points **/
      vij = ENERGY::CalcMPData(vi.vd, vj.vd);
      vik = ENERGY::CalcMPData(vi.vd, vk.vd);

      /** half-edge mats **/
      const typename ENERGY::TM EM_ij = revij ? ENERGY::GetEMatrix(eij.edj) : ENERGY::GetEMatrix(eij.edi);
      const typename ENERGY::TM EM_ik = revik ? ENERGY::GetEMatrix(eik.edj) : ENERGY::GetEMatrix(eik.edi);

      /** trafo half-edge mats to edge-MP **/
      ENERGY::CalcQs(vij, vik, Qij_ik, Qik_ij);
      ENERGY::QtMQ(Qij_ik, EM_ij);
      ENERGY::QtMQ(Qik_ij, EM_ik);

      /** (fake-) harmonic mean (should 0.5 times harmonic mean) **/
      EM = ENERGY::HMean(EM_ij, EM_ik);

      /** Calc contrib **/
      ENERGY::CalcRMBlock2(mat, EM, vij, vik);
    }

  }; // class StokesEnergy

  /** END StokesEnergy **/


  /** Stokes Attached Data **/

  template<class ATVD>
  class AttachedSVD : public AttachedNodeData<NT_VERTEX, ATVD, AttachedSVD<ATVD>>
  {
  public:
    using TVD = ATVD;
    using BASE = AttachedNodeData<NT_VERTEX, ATVD, AttachedSVD<ATVD>>;
    using BASE::mesh;
    using BASE::data;
    using BASE::map_data;

    AttachedSVD (Array<ATVD> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    template<class TMESH> INLINE void map_data (const CoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const;
    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, AttachedSVD<TVD> & cevd) const;
  }; // class AttachedSVD


  template<class ATED>
  class AttachedSED : public AttachedNodeData<NT_EDGE, ATED, AttachedSED<ATED>>
  {
  public:
    using TED = ATED;
    using BASE = AttachedNodeData<NT_EDGE, ATED, AttachedSED<ATED>>;
    using BASE::mesh;
    using BASE::data;
    using BASE::map_data;

    AttachedSED (Array<TED> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    void map_data (const BaseCoarseMap & cmap, AttachedSED<TED> & ceed) const; // in impl header beacust I static_cast to elasticity-mesh
  }; // class AttachedSED

  /** END Stokes Attached Data **/



  /** StokesMesh **/


  /**
     Extends a Mesh with facet-loops needed for Hiptmair smoother.
   **/
  template<class... T>
  class StokesMesh : public BlockAlgMesh<T...>
  {
    using BASE = BlockAlgMesh<T...>;
    using THIS_CLASS = StokesMesh<T...>;

  protected:

    Table<int> loops;

  public:

    StokesMesh (BlockTM && _mesh, T*... _data)
      : BASE(move(_mesh), _data...)
    { ; }

    StokesMesh (BlockTM && _mesh, tuple<T*...> _data)
      : BASE(move(_mesh), _data)
    { ; }

    ~StokesMesh () { ; }

    FlatTable<int> GetLoops () const { return loops; }
    void SetLoops (Table<int> && _loops) { loops = move(_loops); }

    virtual void MapAdditionalData (const BaseGridMapStep & amap) override
    {
      // TERRIBLE (!!), but I really don't feel like thinking of something better ...
      if (auto ctr_map = dynamic_cast<const GridContractMap<THIS_CLASS>*>(&amap))
	{ MapAdditionalData_impl(*ctr_map); }
      else if (auto crs_map = dynamic_cast<const BaseCoarseMap*>(&amap))
	{ MapAdditionalData_impl(*crs_map); }
      else
	{ throw Exception(string("Not Map for ") + typeid(amap).name()); }
    }

    void MapAdditionalData_impl (const GridContractMap<THIS_CLASS> & cmap)
    { throw Exception("MapAdditionalData contract TODO!"); }

    void MapAdditionalData_impl (const BaseCoarseMap & cmap)
    {
      auto & fmesh = *this;
      fmesh.CumulateData();

      // NOTE: this prooobably does not work ... 
      auto & cmesh = *static_pointer_cast<THIS_CLASS>(cmap.GetMappedMesh());

      auto fedges = fmesh.template GetNodes<NT_EDGE>();
      auto cedges = cmesh.template GetNodes<NT_EDGE>();

      auto vmap = cmap.GetMap<NT_VERTEX>();
      auto emap = cmap.GetMap<NT_EDGE>();

      Array<int> loop_map(loops.Size());
      Array<int> ucfacets(30); // unique coarse facets
      Array<int> cloop(30);
      TableCreator<int> ccl;
      for (; !ccl.Done(); ccl++) {
	int cnt_c_loops = 0;
	for (auto flnr : Range(loops.Size())) {
	  cout << "map loop " << flnr << " = "; prow(loops[flnr]); cout << endl;
	  auto floop = loops[flnr];
	  bool cloop_ok = true;
	  int c = 0;
	  ucfacets.SetSize0();
	  cloop.SetSize(floop.Size());
	  const int fls = floop.Size();
	  for (int j = 0; (j < fls) && cloop_ok; j++) {
	    auto sfenr = floop[j];
	    bool fflip = (sfenr < 0);
	    int fenr = abs(sfenr) - 1;
	    auto cenr = emap[fenr];
	    cout << " enr " << fenr << " -> " << cenr << endl;
	    if (cenr != -1) {
	      auto pos = merge_pos_in_sorted_array(cenr, ucfacets);
	      cout << " looked for " << cenr << " in ucfacets = "; prow(ucfacets); cout << endl;
	      cout << " pos is " << pos << endl;
	      if ( (pos != -1) && (pos > 0) && (ucfacets[pos-1] == cenr) )
		{ cloop_ok = false; break; }
	      else if (pos >= 0)
		{ ucfacets.Insert(pos, cenr); }
	      cout << " now ucfacets is "; prow(ucfacets); cout << endl;
	      bool cflip = cedges[cenr].v[0] == vmap[fedges[fenr].v[0]];
	      cloop[c++] = (fflip ^ cflip) ? -(1 + cenr) : (1 + cenr);
	    }
	  }
	  cloop.SetSize(c);
	  cout << " maps to cloop "; prow(cloop); cout << endl;
	  cloop_ok &= (c > 1);
	  if (cloop_ok) {
	    loop_map[flnr] = cnt_c_loops;
	    ccl.Add(cnt_c_loops++, cloop);
	  }
	  else
	    { loop_map[flnr] = -1; }
	}
      }
      cmesh.loops = ccl.MoveTable();

      cout << " c loops: " << endl << cmesh.loops << endl;

    } // StokesMesh::map_loops


    // Table<int> map_loops (const GridContractMap<BASE> & cmap);

  protected:

  }; // class StokesMesh


  /** END StokesMesh **/

} // namespace amg

#endif

#endif
