#ifndef FILE_MIS_AGG_HPP
#define FILE_MIS_AGG_HPP

#ifdef MIS_AGG

#include <utils_numeric_types.hpp>

#include "agglomerator.hpp"

namespace amg
{

  /** Maximal Independent Set (MIS) style agglomeration **/

  class MISAggOptions
  {
  public:
    SpecOpt<bool> mis_neib_boost = false;
    SpecOpt<bool> lazy_neib_boost = false;
    SpecOpt<AVG_TYPE> agg_minmax_avg = AVG_TYPE::GEOM;
    SpecOpt<xbool> ecw_minmax = xbool(maybe);
    SpecOpt<bool> mis_dist2 = SpecOpt<bool>(false, Array<bool>({ true }));
    SpecOpt<bool> ecw_geom = true;           // use geometric instead of harmonic mean when determining strength of connection

    MISAggOptions () { ; }

    ~MISAggOptions () { ; }

    void SetMISFromFlags (const Flags & flags, string prefix)
    {
      mis_neib_boost.SetFromFlags(flags, prefix + "mis_neib_boost");
      lazy_neib_boost.SetFromFlags(flags, prefix + "mis_lazy_neib_boost");
      agg_minmax_avg.SetFromFlagsEnum(flags, prefix + "mis_minmax_avg", { "min", "geom", "harm", "alg", "max" }, Array<AVG_TYPE>{ MIN, GEOM, HARM, ALG,MAX });
      ecw_minmax.SetFromFlags(flags, prefix + "mis_ecw_minmax");
      mis_dist2.SetFromFlags(flags, prefix + "mis_dist2");
      ecw_geom.SetFromFlags(flags, prefix + "mis_ecw_geom");
    }
  }; // class MISAggOptions


  template<class ATENERGY, class ATMESH, bool AROBUST = true>
  class MISAgglomerator : public Agglomerator<ATMESH>
  {
    static constexpr bool ROBUST = AROBUST;
    using TMESH = ATMESH;
    using ENERGY = ATENERGY;
    using TM = typename ENERGY::TM;
    using TED = typename ENERGY::TED;
    using TVD = typename ENERGY::TVD;

  public:
    MISAgglomerator (shared_ptr<TMESH> _mesh);

    void Initialize (const MISAggOptions & opts, int level);

  protected:
    void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;
    void FormAgglomeratesOld (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);
    template<class TMU> void FormAgglomerates_impl (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg);

    // template<class TMU> INLINE FlatArray<TMU> GetEdgeData ();

    template<class TMU>
    INLINE void GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data);
  

  public:
    /** settings **/
    bool cw_geom = false;
    double dist2 = false;
    int min_new_aggs = 3;
    bool neib_boost = true;
    bool lazy_neib_boost = false;
    xbool use_minmax_soc = maybe;
    AVG_TYPE minmax_avg = MIN;

  protected:
    template<class TMU> INLINE void CalcQs (const TVD & di, const TVD & dj, TMU & Qij, TMU & Qji)
    {
      if constexpr(std::is_same<TMU, TM>::value)
      	{ ENERGY::CalcQs(di, dj, Qij, Qji); }
      else
      	{ Qij = Qji = 1; }
    }

    template<class TMU> INLINE void SetQtMQ (TMU & A, const TMU & Q, const TMU & M)
    {
      if constexpr(std::is_same<TMU, TM>::value)
	      { ENERGY::SetQtMQ(1.0, A, Q, M); }
      else // scalar
      	{ A = M; }
    }

    // A += fac * QT * M * Q
    template<class TMU> INLINE void AddQtMQ (double fac, TMU & A, const TMU & Q, const TMU & M)
    {
      if constexpr(std::is_same<TMU, TM>::value)
      	{ ENERGY::AddQtMQ(fac, A, Q, M); }
      else // scalar
	      { A += fac * M; }
    }

    template<class TMU> INLINE void ModQs (const TVD & di, const TVD & dj, TMU & Qij, TMU & Qji)
    {
      if constexpr(std::is_same<TMU, TM>::value)
      	{ ENERGY::ModQs(di, dj, Qij, Qji); }
      else
      	{ Qij = Qji = 1; }
    }

    template<class TMU> INLINE void ModQij (const TVD & di, const TVD & dj, TMU & Qij)
    {
      if constexpr(std::is_same<TMU, TM>::value)
      	{ ENERGY::ModQij(di, dj, Qij); }
      else
      	{ Qij = 1; }
    }

    template<class TMU> INLINE void ModQHh (const TVD & di, const TVD & dj, TMU & QHh)
    {
      if constexpr(std::is_same<TMU, TM>::value)
      	{ ENERGY::ModQHh(di, dj, QHh); }
      else
      	{ QHh = 1; }
    }

  }; // class MISAgglomerator


} // namespace amg

#endif // MIS_AGG

#endif // FILE_MIS_AGG_HPP
