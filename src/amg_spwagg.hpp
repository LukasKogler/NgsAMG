#ifndef FILE_AMG_SPWAGG_HPP
#define FILE_AMG_SPWAGG_HPP

#ifdef SPWAGG

#include "amg.hpp"
#include "amg_map.hpp"
#include "amg_coarsen.hpp"
#include "amg_agg.hpp"

namespace amg
{

  enum SPW_CW_TYPE : int { HARMONIC = 0,                // most stable
			   GEOMETRIC = 1,               // if it works, best condition, most restrictive
			   MINMAX = 2 };                // best with unbalanced aggs (??) 

  /** Successive pairwise agglomeration **/
  template<class ATENERGY, class ATMESH, bool AROBUST = true>
  class SPWAgglomerator : public AgglomerateCoarseMap<ATMESH> 
  {
  public:

    static constexpr bool ROBUST = AROBUST;
    using TMESH = ATMESH;
    using ENERGY = ATENERGY;
    using TED = typename ENERGY::TED;
    using TVD = typename ENERGY::TVD;
    using TM = typename ENERGY::TM;

    struct Options
    {
      using CW_TYPE = SPW_CW_TYPE;
      bool robust = true;                            // use robust coarsening via EVPs
      double edge_thresh = 0.025;
      double vert_thresh = 0.0;
      int num_rounds = 3;                            // # of rounds of coarsening
      /** when picking neib **/
      SpecOpt<bool> allrobust = false;               // true: EVP to choose neib candidate, otherwise EVP only to confirm
      SpecOpt<CW_TYPE> pick_cw_type = HARMONIC;      // which kind of SOC to use here
      SpecOpt<AVG_TYPE> pick_mma_scal = HARM;        // which averaging to use for MINMAX SOC 
      SpecOpt<AVG_TYPE> pick_mma_mat = HARM;         // which averaging to use for MINMAX SOC (only HARM/GEOM are valid)
      /** when checking neib with small EVP (only relevant for robust && (!allrobust) **/
      SpecOpt<CW_TYPE> check_cw_type = HARMONIC;     // which kind of SOC to use here
      SpecOpt<AVG_TYPE> check_mma_scal = HARM;       // which averaging to use for traces in MINMAX SOC
      SpecOpt<AVG_TYPE> check_mma_mat = HARM;        // which averaging to use for mats in MINMAX SOC
      /** when checking neib with big EVP **/
      bool checkbigsoc = true;                       // check big EVP is pos. def for agg-agg merge
      bool simple_checkbigsoc = false;               // use simplified big EVP based on traces
      bool bdiag = false;                            // check big EVP correspoding to BGS, or GS smoother?
      /** used for EVPs **/
      SpecOpt<bool> neib_boost = false;              // use connections to common neibs to boost edge matrices
      // SpecOpt<bool> lazy_nb = false;                 // to a "lazy" boost, which requires no EVPs (probably a bad idea)
      xbool use_stab_ecw_hack = maybe;               // useful to make HARMONIC behave a bit more like geometric 
      /** misc. **/
      bool print_aggs = false;                       // output
      // Options (Options && other)
      // 	: robust(other.robust), edge_thresh(other.edge_thresh), vert_thresh(other.vert_thresh), num_rounds(other.num_rounds),
      // 	  allrobust(move(other.allrobust)), pick_cw_type(move(other.pick_cw_type)), pick_mma_scal(move(other.pick_mma_scal)),
      // 	  pick_mma_mat(move(other.pick_mma_mat)), check_cw_type(move(other.check_cw_type)), check_mma_scal(move(other.check_mma_scal)),
      // 	  check_mma_mat(move(other.check_mma_mat)), checkbigsoc(other.checkbigsoc), use_stab_ecw_hack(other.use_stab_ecw_hack), print_aggs(other.print_aggs)
      // { ; }
    };
  protected:
    
    using AgglomerateCoarseMap<TMESH>::mesh;

    shared_ptr<BitArray> free_verts;

    Options settings;

    Table<int> fixed_aggs; // hacky for a test

  public:

    SPWAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts, Options && _settings);

    SPWAgglomerator (shared_ptr<TMESH> _mesh, shared_ptr<BitArray> _free_verts = nullptr);

  protected:
    virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;

    template<class ATD, class TMU> INLINE void GetEdgeData (FlatArray<ATD> in_data, Array<TMU> & out_data);

    template<class TMU> void FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);

    void CalcCMK (const BitArray & skip, const SparseMatrix<double> & econ, Array<int> & cmk);

  }; // class SPWAgglomerator

} //namespace amg

#endif // SPWAGG

#endif
