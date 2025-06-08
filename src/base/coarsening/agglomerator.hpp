#ifndef FILE_BASE_AGG
#define FILE_BASE_AGG

#include "base_mesh.hpp"

#include <SpecOpt.hpp>
#include <utils_sparseLA.hpp>
#include <utils_denseLA.hpp>

namespace amg
{

class AggOptions
{
public:
  SpecOpt<double> edge_thresh = 0.025;
  SpecOpt<double> vert_thresh = 0.0;
  SpecOpt<bool> crs_robust = true;         // use robust coarsening via EVPs (if available)
  SpecOpt<xbool> ecw_stab_hack = xbool(maybe);
  SpecOpt<bool> print_aggs = false;           // print agglomerates (for debugging purposes) (TODO: still only used by MIS)
  SpecOpt<bool> print_vmap = false;           // debugging output
  SpecOpt<bool> check_iso  = true;            // check for isolated vertices

  AggOptions () { ; }

  ~AggOptions () { ; }

  void SetAggFromFlags (const Flags & flags, string prefix)
  {
    edge_thresh.SetFromFlags(flags, prefix + "edge_thresh");
    vert_thresh.SetFromFlags(flags, prefix + "vert_thresh");
    crs_robust.SetFromFlags(flags, prefix + "crs_robust");
    print_aggs.SetFromFlags(flags, prefix + "agg_print_aggs");
    print_vmap.SetFromFlags(flags, prefix + "agg_print_vmap");
    print_vmap |= print_aggs;
    ecw_stab_hack.SetFromFlags(flags, prefix + "ecw_stab_hack");
    check_iso.SetFromFlags(flags, prefix + "check_isolated_vertices");
  } // AgglomerateCoarseMap::SetFromFlags

}; // class AggOptions


struct Agglomerate
{
  int id, ctr;
  ArrayMem<int, 30> mems;
  Agglomerate () { id = -2; ctr = -1; }
  Agglomerate (int c, int d) { mems.SetSize(1); mems[0] = c; id = d; ctr = c; }
  INLINE Agglomerate (Agglomerate && other) {
    id = std::move(other.id);
    ctr = std::move(other.ctr);
    mems = std::move(other.mems);
  }
  INLINE Agglomerate& operator = (Agglomerate && other) {
    id = std::move(other.id);
    ctr = std::move(other.ctr);
    mems = std::move(other.mems);
    return *this;
  }
  // we never want to copy/copy construct
  INLINE Agglomerate (const Agglomerate & other) = delete;
  INLINE Agglomerate& operator = (const Agglomerate & other) = delete;
  ~Agglomerate () { ; }
  INLINE int center () const { return ctr; }
  INLINE FlatArray<int> members () const { return mems; }
  INLINE void AddSort (int v) { insert_into_sorted_array(v, mems); }
  INLINE void Add (int v) { AddSort(v); }
}; // Agglomerate


INLINE std::ostream & operator<<(std::ostream &os, const Agglomerate& agg) {
  os << "[id " << agg.id << ", #mems " << agg.mems.Size() << ", center " << agg.center() << ", mems: ";
  for (auto j : Range(agg.mems))
    { os << "( " << j << "::" << agg.mems[j] << ")]"; }
  return os;
}

/**
 * Forms agglomerates for a mesh. This is split into it's own class
 * so the agglomeration algorithms, some of which can be very expensive to
 * compilte, only need to include very minimal headers!
*/
class BaseAgglomerator
{
public:
  BaseAgglomerator() = default;

  virtual ~BaseAgglomerator() = default;

  virtual void InitializeBaseAgg (const AggOptions & opts, int level);

  void SetFreeVerts    (shared_ptr<BitArray> &free_verts);
  void SetSolidVerts   (shared_ptr<BitArray> &solid_verts);
  void SetAllowedEdges (shared_ptr<BitArray> &allowed_edges);

  void SetFixedAggs (Table<int> && fixed_aggs);

  virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) = 0;
protected:

  INLINE shared_ptr<BitArray> GetFreeVerts()    { return _free_verts; }
  INLINE shared_ptr<BitArray> GetSolidVerts()   { return _solid_verts; }
  INLINE shared_ptr<BitArray> GetAllowedEdges() { return _allowed_edges; }

  INLINE FlatTable<int> GetFixedAggs() { return _fixed_aggs; }

  virtual BlockTM const &GetBTM() const = 0;

  /** settings **/
  double edge_thresh = 0.025;
  double vert_thresh = 0.0;
  bool robust_crs = true;
  bool print_aggs = false;     // print agglomerates (for debugging purposes) (TODO: still only used by MIS)
  xbool use_stab_ecw_hack = maybe;
  shared_ptr<BitArray> _free_verts = nullptr;    // coarsening on these vertices
  shared_ptr<BitArray> _solid_verts = nullptr;   // when set, locally coarsen on these vertices instead of master vertices.
  shared_ptr<BitArray> _allowed_edges = nullptr; // lock certain edges from coarsening
  bool check_isolated_vertices = true;

  Table<int> _fixed_aggs; // hacky for a test
};


template<class ATMESH>
class Agglomerator : public BaseAgglomerator
{
public:
  typedef ATMESH TMESH;

  Agglomerator(shared_ptr<TMESH> mesh)
    : BaseAgglomerator()
    , _mesh(mesh)
  { ; }

  virtual ~Agglomerator() = default;

  INLINE TMESH const &GetMesh() const { return *_mesh; }
  INLINE TMESH &GetMesh() { return *_mesh; }

protected:

  BlockTM const &GetBTM() const override { return GetMesh(); };

  shared_ptr<TMESH> _mesh;
}; // class Agglomerator


/**
* utility data structur to use during coarsening
*/
template<class TMESH, class TENERGY, class ATMU, class ATWEIGHT = double>
class AgglomerationData
{
public: // all public, for my sanity
  /**
  * Implementation of aggregation-data for full weights
  */
  using ENERGY  = TENERGY;
  using TVD     = typename ENERGY::TVD;
  using TM      = typename ENERGY::TM;
  using TMU     = ATMU;
  using TWEIGHT = ATWEIGHT;

  // static constexpr bool ROBUST = std::is_same<TMU, typename ENERGY::TM>::value;

  static constexpr bool ROBUST = Height<TMU>() > 1;

  static_assert(ROBUST != std::is_same_v<TMU, TWEIGHT>, "TMU/TWEIGHT must coincide for scal-crs");

  AgglomerationData (TMESH const &aMesh)
    : mesh(aMesh)
  {}

  virtual ~AgglomerationData () {};

  INLINE TMESH                const& GetMesh ()   const { return mesh; }
  INLINE SparseMatrix<double> const& GetEdgeCM () const { return *GetMesh().GetEdgeCM(); }

  INLINE TVD const &GetVData      (int const &vertNum) const { return vData[vertNum]; }
  INLINE TMU const &GetAuxDiag    (int const &vertNum) const { return auxDiags[vertNum]; }
  INLINE TMU const &GetEdgeMatrix (int const &edgeNum) const { return edgeMats[edgeNum]; }

  INLINE TWEIGHT
  GetApproxVWeight (int const &vertNum) const
  {
    return ENERGY::GetApproxEdgeWeight(GetVData(vertNum));
  }

  INLINE TWEIGHT
  GetApproxEdgeWeight (int const &edgeNum) const
  {
    if constexpr(Height<TMU>() > 1)
    {
      return CalcAvgTrace<TWEIGHT>(GetEdgeMatrix(edgeNum));
    }
    else
    {
      return GetEdgeMatrix(edgeNum);
    }
  }

  INLINE TWEIGHT
  GetMaxTrOD (int const &vertNum) const
  {
    return hasOffProcTrace ? std::max(maxTrOD[vertNum], offProcTrace[vertNum])
                          : maxTrOD[vertNum];
  }

  TMESH const &mesh;

  // Note: may or may not own this
  Array<TVD>     vData;
  Array<TMU>     auxDiags;
  Array<TWEIGHT> maxTrOD;    // max trace of off-diag contribs

  Array<TMU>     edgeMats;
  Array<TWEIGHT> edgeTrace;

  bool hasOffProcTrace;
  Array<TWEIGHT> offProcTrace;
}; // class AgglomerationData

/**
 *  Agglomerators that fit the usual schema (so, currently all) derive from this.
 *  The class onluy provides some utility for coarsening with approximate or full
 *  weights, TENERGY must be a usual energy with edge- and vertex-data and eneregy
 *  contribs from edges.
 */
template<class TENERGY, class ATMESH>
class VertexAgglomerator : public Agglomerator<ATMESH>
{
public:
  using TMESH = typename Agglomerator<ATMESH>::TMESH;

  typedef TENERGY ENERGY;

  using TED = typename ENERGY::TED;
  using TVD = typename ENERGY::TVD;

  VertexAgglomerator(shared_ptr<TMESH> mesh)
    : Agglomerator<TMESH>(mesh)
  { ; }

  virtual ~VertexAgglomerator() = default;

protected:

  template<class TMU, class TWEIGHT>
  INLINE void
  InitializeAggData(AgglomerationData<TMESH, TENERGY, TMU, TWEIGHT> &aggData);

  template<class TMU>
  INLINE void
  GetEdgeData (FlatArray<TED> in_data, Array<TMU> & out_data);

  template<class TFULL, class TAPPROX>
  INLINE void
  GetApproxEdgeData (FlatArray<TFULL>  in_data,
                    Array<TAPPROX>   &out_data);
}; //




} // namesapce amg


#endif // FILE_BASE_AGG
