#ifndef FILE_AMG_SPW_AGG_HPP
#define FILE_AMG_SPW_AGG_HPP

#ifdef SPW_AGG

#include <utils_sparseMM.hpp>
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
  SpecOpt<int>      numRounds   = 3;
  SpecOpt<bool>     robustPick  = true;  // whether picking or just checking with robust SOC
  SpecOpt<bool>     neibBoost   = true;  // use boost for robust SOC
  SpecOpt<AVG_TYPE> scalAvg     = GEOM;  // type of avg for scalar neib-filtering
  SpecOpt<bool>     orphanRound = true;  // probably a good idea generally

  SpecOpt<bool>     checkBigSOC           = false; // disabled by default
  SpecOpt<bool>     bigSOCBlockDiagSM     = false; // use blk-diag for big-soc check (more permissive)
  SpecOpt<bool>     bigSOCRobust          = false; // do scalar check only
  SpecOpt<double>   bigSOCCheckHackThresh = 0.0;   // 0.0 -> hack disabled

  /**
   *  fraction of base-level in-agg edge-contribs that are removed from coarse diags
   *  improves stability, but decreases valid matches
   *   0   .. full remove -> max. matches
   *   1   .. no remove   -> max. stab
   */
  SpecOpt<double> diagStabBoost = 0.5;


  SpecOpt<bool> printParams  = false;
  SpecOpt<bool> printSummary = false;

  SPWAggOptions () { ; }

  ~SPWAggOptions () { ; }

  void SetSPWFromFlags (const Flags & flags, string prefix)
  {
    numRounds.SetFromFlags(flags,             prefix + "spw_rounds");
    robustPick.SetFromFlags(flags,            prefix + "spw_pick_robust");
    neibBoost.SetFromFlags(flags,             prefix + "spw_neib_boost");
    checkBigSOC.SetFromFlags(flags,           prefix + "spw_cbs");
    diagStabBoost.SetFromFlags(flags,         prefix + "spw_diag_stab_boost");
    bigSOCRobust.SetFromFlags(flags,          prefix + "spw_big_soc_robust");
    bigSOCCheckHackThresh.SetFromFlags(flags, prefix + "spw_big_soc_hack_thresh");
    orphanRound.SetFromFlags(flags,           prefix + "spw_orphan_treatment");

    scalAvg.SetFromFlagsEnum(flags,
                             prefix + "spw_pick_avg",
                             { "min", "geom", "harm", "alg", "max" },
                             Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG, MAX });

    bigSOCBlockDiagSM.SetFromFlags(flags, prefix + "spw_big_soc_bdiag");

    printParams.SetFromFlags(flags,  prefix + "spw_print_params");
    printSummary.SetFromFlags(flags, prefix + "spw_print_summ");

    // if ( (flags.GetDefineFlagX(prefix + "spw_bdiag").IsTrue() || flags.GetDefineFlagX(prefix + "spw_bdiag").IsFalse()) ||
    //       (flags.GetNumListFlag(prefix + "spw_bdiag_spec").Size() > 0) ) {
    //     /** it is set explicitely!**/
    //     checkBigSOCBlockDiagSM.SetFromFlags(flags, prefix + "spw_bdiag");
    // }
    // else {
    //   /** if nothing is given explicitely, try to take a guess on which sm_types are given. default is "gs", so initialize with "false" **/
    //   spw_cbs_bdiag = false;
    //   checkBigSOCBlockDiagSM.SetFromFlagsEnum(flags, prefix+"sm_type", { "gs", "bgs" }, Array<bool>{ false, true });
    // }
  } // SPWAggOptions::SetFromFlags

}; // class SPWAggOptions


// using float here would be nice, but does not work (yet?) for elasicity?
using TWEIGHT = double;

struct SPWConfig
{
  int  numRounds;
  bool dropDDVerts; // drop diagonally dominant vertices (that is, large L2-weight)
  TWEIGHT vertThresh;
  bool orphanRound;
  TWEIGHT diagStabBoost;

  /**
  * Partner identification during neighbor matching has 2 phases:
  *   a) initial neighbor filtering
  *   b) picking from viable neigbhors
  *
  * (a) is done using an approximate, scalar, weight
  *     based on max-off-diags.
  * (b) uses stable harmonic-mean based EVPs to PICK OR CONFIRM
  */

  // (a)
  AVG_TYPE avgTypeScal;   // the mean-type used for (a) (I like GEOM here)
  TWEIGHT  scalRelThresh; // relative threshold for (a)
  // bool     l2BoostScal;   // boost for connections between vertices with l2-weights

  // (b)
  bool     robustPick;   // whether (b) uses stable SOC for picking or only checking
  bool     neibBoost;    // whether we compute neighbor-boost for (b)
  TWEIGHT  absRobThresh; // absolute threshold for (b)
  // bool     l2BoostRob;   // boost for connections between vertices with l2-weights

  // (b) bigSOC
  bool     checkBigSOC;      // whether (b) additionally checks the agg-wise SOCs
  bool     robBigSOC;        // whether big-SOC is done for full energy or only for scalar
  bool     bigSOCUseBDG;     // whether agg-wise block-smoother is assumed for big-SOC
  TWEIGHT  bigSOCHackThresh; // threshold for BIG-SOC hack (potenitally ignores evals smaller than this)
  TWEIGHT  absBigThresh; // threshold for agg-wise SOC in (b)

  /** misc. **/
  bool printParams = false;
  bool printSumms  = false;
  bool all_cmk = false;
};

template<class ATENERGY, class ATMESH, bool COMPILE_EV_BASED = true>
class SPWAgglomerator : public VertexAgglomerator<ATENERGY, ATMESH>
{
public:

  using TMESH = typename Agglomerator<ATMESH>::TMESH;

  using ENERGY = typename VertexAgglomerator<ATENERGY, ATMESH>::ENERGY;
  using TED    = typename VertexAgglomerator<ATENERGY, ATMESH>::TED;
  using TVD    = typename VertexAgglomerator<ATENERGY, ATMESH>::TVD;

  using TM = StripTM<ENERGY::DPV, ENERGY::DPV, TWEIGHT>;

  SPWAgglomerator (shared_ptr<TMESH> mesh);

  void Initialize (const SPWAggOptions & opts, int level);

  virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;

protected:
  /** settings **/
  using CW_TYPE = SPW_CW_TYPE;

  SPWConfig cfg;

protected:

  template<class TMU>
  void FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);

  void MapVertsTest  (FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg);
}; // class SPWAgglomerator


} //namespace amg

#endif // SPW_AGG

#endif
