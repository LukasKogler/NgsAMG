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
    double vol;                  // volume
  }; // struct StokesVData


  template<int ADIM, int ABS, class ATED>
  struct StokesEData
  {
    static constexpr int DIM = ADIM;
    static constexpr int BS = ABS;
    using TED = ATED;
    TED edi, edj;                // energy contribs v_i-f_ij and v_j-f_ij
    Vec<BS, double> flow;        // flow of base functions
  }; // struct StokesEData

  /** END Stokes Data **/


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
