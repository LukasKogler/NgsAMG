#ifndef FILE_AMG_SPW_AGG_HPP
#define FILE_AMG_SPW_AGG_HPP

#ifdef SPW_AGG

#include <utils_numeric_types.hpp>

#include "agglomerator.hpp"


namespace amg
{

enum SPW_CW_TYPE : int {
    HARMONIC = 0,   // most stable
    GEOMETRIC = 1,  // if it works, best condition, most restrictive
    MINMAX = 2      // best with unbalanced aggs (??)
};

class SPWAggOptions
{
public:
  /** SPW-AGG **/
  SpecOpt<int> spw_rounds = 3;
  SpecOpt<bool> spw_allrobust = true;
  SpecOpt<bool> spw_neib_boost = false;
  SpecOpt<bool> spw_cbs = true;
  SpecOpt<bool> spw_cbs_bdiag = false;
  SpecOpt<bool> spw_cbs_robust = true;
  SpecOpt<bool> spw_cbs_spd_hack = false;
  SpecOpt<SPW_CW_TYPE> spw_pick_cwt = SPW_CW_TYPE::MINMAX;
  SpecOpt<AVG_TYPE> spw_pick_mma_scal = AVG_TYPE::GEOM;
  SpecOpt<AVG_TYPE> spw_pick_mma_mat = AVG_TYPE::GEOM;
  SpecOpt<SPW_CW_TYPE> spw_check_cwt = SPW_CW_TYPE::HARMONIC;
  SpecOpt<AVG_TYPE> spw_check_mma_scal = AVG_TYPE::GEOM;
  SpecOpt<AVG_TYPE> spw_check_mma_mat = AVG_TYPE::GEOM;
  SpecOpt<bool> spw_print_params = false;
  SpecOpt<bool> spw_print_summs = false;
  SpecOpt<bool> spw_wo = false;
  SpecOpt<bool> spw_cmk = false;

  SPWAggOptions () { ; }

  ~SPWAggOptions () { ; }

  void SetSPWFromFlags (const Flags & flags, string prefix)
  {
    spw_rounds.SetFromFlags(flags,    prefix + "spw_rounds");
    spw_allrobust.SetFromFlags(flags, prefix +  "spw_pick_robust");
    spw_cbs.SetFromFlags(flags,       prefix + "spw_cbs");
    if ( (flags.GetDefineFlagX(prefix + "spw_bdiag").IsTrue() || flags.GetDefineFlagX(prefix + "spw_bdiag").IsFalse()) ||
          (flags.GetNumListFlag(prefix + "spw_bdiag_spec").Size() > 0) ) {
        /** it is set explicitely!**/
        spw_cbs_bdiag.SetFromFlags(flags, prefix + "spw_bdiag");
    }
    else {
      /** if nothing is given explicitely, try to take a guess on which sm_types are given. default is "gs", so initialize with "false" **/
      spw_cbs_bdiag = false;
      spw_cbs_bdiag.SetFromFlagsEnum(flags, prefix+"sm_type", { "gs", "bgs" }, Array<bool>{ false, true });
    }
    spw_cbs_robust.SetFromFlags(flags,         prefix + "spw_cbs_robust");
    spw_cbs_spd_hack.SetFromFlags(flags,       prefix + "spw_cbs_spd_hack");
    spw_pick_cwt.SetFromFlagsEnum(flags,       prefix + "spw_pcwt", { "harm", "geom", "mmx" }, Array<SPW_CW_TYPE>{ HARMONIC, GEOMETRIC, MINMAX });
    spw_pick_mma_scal.SetFromFlagsEnum(flags,  prefix + "spw_pmmas", { "min", "geom", "harm", "alg", "max" }, Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG, MAX });
    spw_pick_mma_mat.SetFromFlagsEnum(flags,   prefix + "spw_pmmam", { "min", "geom", "harm", "alg", "max" }, Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG, MAX });
    spw_check_cwt.SetFromFlagsEnum(flags,      prefix + "spw_ccwt", { "harm", "geom", "mmx" }, Array<SPW_CW_TYPE>{ HARMONIC, GEOMETRIC, MINMAX });
    spw_check_mma_scal.SetFromFlagsEnum(flags, prefix + "spw_cmmas", { "min", "geom", "harm", "alg", "max" }, Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG, MAX });
    spw_check_mma_mat.SetFromFlagsEnum(flags,  prefix + "spw_cmmam", { "min", "geom", "harm", "alg", "max" },  Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG, MAX });
    spw_print_params.SetFromFlags(flags,       prefix + "spw_print_params");
    spw_print_summs.SetFromFlags(flags,        prefix + "spw_print_summs");
    spw_wo.SetFromFlags(flags,                 prefix + "spw_wo");
    spw_neib_boost.SetFromFlags(flags,         prefix + "spw_neib_boost");
    spw_cmk.SetFromFlags(flags,                prefix + "spw_cmk");
  } // SPWAggOptions::SetFromFlags

}; // class SPWAggOptions


template<class ATENERGY, class ATMESH, bool AROBUST = true>
class SPWAgglomerator : public Agglomerator<ATMESH>                        
{
  using TMESH = ATMESH;
  using ENERGY = ATENERGY;
  using TED = typename ENERGY::TED;
  using TVD = typename ENERGY::TVD;
  using TM = typename ENERGY::TM;
public:

  SPWAgglomerator (shared_ptr<TMESH> mesh);

  void Initialize (const SPWAggOptions & opts, int level);

  virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;

protected:
  /** settings **/
  using CW_TYPE = SPW_CW_TYPE;
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
  bool cbs_spd_hack = false;                     // regularize robust CBS EVP such that we can use dpotrf instead of dpstrf
  /** used for EVPs **/
  SpecOpt<bool> neib_boost = false;              // use connections to common neibs to boost edge matrices
  /** misc. **/
  bool print_params = false;                       // output
  bool print_summs = false;                       // output
  bool weed_out = false;
  bool all_cmk = false;

protected:
  template<class TMU>
  INLINE void GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data);
  
  template<class TFULL, class TAPPROX>
  INLINE void GetApproxEdgeData (FlatArray<TFULL> in_data, Array<TAPPROX> & out_data);

  template<class TMU>
  void FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);
  
  void CalcCMK (const BitArray & skip, const SparseMatrix<double> & econ, Array<int> & cmk);
  
  void MapVertsTest  (FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
}; // class SPWAgglomerator


} //namespace amg

#endif // SPW_AGG

#endif
