#ifndef FILE_AMG_PAGG_HPP
#define FILE_AMG_PAGG_HPP

#include "amg.hpp"
#include "amg_map.hpp"
#include "amg_coarsen.hpp"

namespace amg
{

  /** Successive pairwise agglomeration **/
  template<class ATENERGY, class ATMESH, bool AROBUST = true>
  class SPAgglomerator : public AgglomerateCoarseMap<ATMESH> 
  {
  public:

    static constexpr bool ROBUST = AROBUST;
    using TMESH = ATMESH;
    using ENERGY = ATENERGY;
    using TM = typename ENERGY::TM;
    using TVD = typename ENERGY::TVD;

    struct Options
    {
      bool robust = true;                       // use robust coarsening via EVPs
      double edge_thresh = 0.025;
      double vert_thresh = 0.0;
      int num_rounds = 3;                       // # of rounds of coarsening
      enum CW_TYPE : char { HARMONIC,           // most stable
			    GEOMETRIC,          // if it works, best condition, most restrictive
			    MINMAX };           // best with unbalanced aggs (??) 
      /** when picking neib **/
      bool allrobust = false;                   // true: EVP to choose neib candidate, otherwise EVP only to confirm
      CW_TYPE
      /** when checking neib with small EVP **/
      /** when checking neib with big EVP **/
      bool checkbigsoc = true;                  // check big EVP is pos. def for agg-agg merge


      SpecOpt<CW_TYPE> cw_type_scal = HARMONIC;
      SpecOpt<CW_TYPE> cw_type_mat = HARMONIC;
      SpecOpt<bool> neib_boost = false;         // use connections to common neibs to boost edge matrices
      SpecOpt<bool> lazy_nb = false;            // to a "lazy" boost, which requires no EVPs (probably a bad idea)
      SpecOpt<xbool> use_stab_ecw_hack = maybe; // useful to make HARMONIC behave a bit more like geometric 
      SpecOpt<AVG_TYPE> minmax_avg = MIN;       // used with MINMAX
      SpecOpt<AVG_TYPE> minmax_avg_mat = GEOM;  // must be GEOM or HARM
      bool print_aggs = false;                  // output
    };
  protected:
    
    using AgglomerateCoarseMap<TMESH>::mesh;

    shared_ptr<BitArray> free_verts;

    Options settings;

    Table<int> fixed_aggs; // hacky for a test

  public:

    SPAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts, Options && _settings);

    SPAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts = nullptr);

    virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;

    template<class TMU> INLINE FlatArray<TMU> GetEdgeData (shared_ptr<TMESH> mesh);

    template<class TMU> void FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);

  }; // class SPAgglomerator

} //namespace amg
