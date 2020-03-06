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

    static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj)
    { ENERGY::CalcRMBlock(mat, GetEMatrix(ed), vdi, vdj); }

  }; // class StokesEnergy

  /** END StokesEnergy **/


  /** Stokes Attached Data **/

  template<class ATVD>
  class AttachedSVD : public AttachedNodeData<NT_VERTEX, ATVD, AttachedSVD<ATVD>>
  {
  public:
    using TVD = ATVD;
    using BASE = AttachedNodeData<NT_VERTEX, ATVD, AttachedSVD<ATVD>>;
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
    using BASE::map_data;

    AttachedSED (Array<TED> && _data, PARALLEL_STATUS stat)
      : BASE(move(_data), stat)
    { ; }

    void map_data (const BaseCoarseMap & cmap, AttachedSED<TED> & ceed) const; // in impl header beacust I static_cast to elasticity-mesh
  }; // class AttachedSED

  /** END Stokes Attached Data **/
} // namespace amg

#endif

#endif
