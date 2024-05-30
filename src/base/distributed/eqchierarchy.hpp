#ifndef FILE_EQCHIERARCHY_HPP
#define FILE_EQCHIERARCHY_HPP

#include <base.hpp>

#include "mpiwrap_extension.hpp"

namespace amg
{

enum EQC_TYPE {VOL_EQC, FACE_EQC, WIRE_EQC};

static const size_t NO_EQC = -1;

class EQCHierarchy
{
public:

  /**
   * Dummy Constructor !
   * Okay, this is kinda nasty, it would be better if the other constructors
   * did something useful when given nullptr pardofs, but WHATEVER, we just do it
   * this way.
   **/
  EQCHierarchy ();

  /**
    setup from paralleldofs;
    if do_cutunion==true, also constructs all possible cuts/unions of EQCs
    if _max_nd != -1, only looks at dofs 0.._max_nd
  **/
  EQCHierarchy (const shared_ptr<ParallelDofs> & apd, bool do_cutunion = false,
                size_t _max_nd = -1, shared_ptr<BitArray> select = nullptr);

  /**
    setup from finished eqc-dps;
    if do_cutunion==true, also constructs all possible cuts/unions of EQCs
  **/
  EQCHierarchy (Table<int> && eqcs, NgsAMG_Comm acomm, bool do_cutunion = false, bool isRankZeroIdle = true);

  /**
    take vertex-eqcs
  **/
  EQCHierarchy (const shared_ptr<MeshAccess> & ma, Array<NODE_TYPE> nts, bool do_cutunion);

  /**
    setup from:
      - a list of all possible neibs
      - a table of eqc-members
    only one proc needs to have each eqc-mem array
    **/
  EQCHierarchy (FlatArray<int> alldps, FlatTable<int> aloc_mems, NgsAMG_Comm acomm, bool do_cutunion = false);


  virtual ~EQCHierarchy () { ; }

  INLINE bool IsDummy () const { return is_dummy; }
  INLINE bool IsRankZeroIdle () const { return _isRankZeroIdle; }
  INLINE bool IsTrulyParallel        () const { return _isTrulyParallel; }

  INLINE size_t GetNEQCS () const { return neqcs; }
  const Table<int> & GetDPTable () const { return dist_procs; }
  const Array<size_t> & GetEqcIds () const { return eqc_ids; }

  INLINE NgsAMG_Comm const& GetCommunicator () const { return comm; }

  INLINE int GetMasterOfEQC (size_t eqc) const
    { return dist_procs[eqc].Size() ? min2(rank, dist_procs[eqc][0]) : rank; }

  INLINE bool IsMasterOfEQC (size_t eqc) const
    { return ( (!dist_procs[eqc].Size()) || (rank<dist_procs[eqc][0]) ); }

  /** get (global) ID of eqc **/
  INLINE int GetEQCID (size_t eqc) const
    { return eqc_ids[eqc]; }

  /** get (local) eqc that can be used as arguments  methods below**/
  INLINE size_t GetEQCOfID (int eqc_id) const
    { return idf_2_ind[eqc_id]; }

  FlatArray<int> GetDistantProcs (size_t eqc) const
    { return dist_procs[eqc]; }

  FlatArray<int> GetDistantProcs () const
    { return all_dist_procs; }

  /** Find eqc that corresponds to a_dps**/
  INLINE size_t FindEQCWithDPs (FlatArray<int> a_dps) const
  {
    for(auto k:Range(neqcs))
      if( dist_procs[k] == a_dps )
        { return k; }
    return -1;
  }

  /** true if eqc1-procs \union eqc2-procs corresponds to a valid eqc **/
  bool IsMergeValid (size_t eqc1, size_t eqc2) const
    { return (mat_merge(eqc1, eqc2)==NO_EQC)? false : true; }

  INLINE int GetMergedEQCIfExists (size_t eqc1, size_t eqc2) const
    { return mat_merge(eqc1, eqc2); }

  /** get eqc correpsonding to eqc1-procs \union eqc2-procs **/
  INLINE int GetMergedEQC (size_t eqc1, size_t eqc2) const
    { return mat_merge(eqc1, eqc2); }

  /** get eqc correpsonding to eqc1-procs \cap eqc2-procs **/
  INLINE int GetCommonEQC (size_t eqc1, size_t eqc2) const
    { return mat_intersect(eqc1, eqc2); }

  INLINE bool IsLEQ (size_t eqc1, size_t eqc2) const
    { return hierarchic_order.Test(eqc1*neqcs + eqc2);}


  /**
   * @brief Divide data of type T into eqc-classes defined by f
   * @param n_data size of data
   * @param (int k)->size_t; gesize_t_of_data has to get eqc of datum nr k; has to return NO_EQC if not to write
   * @param (int k)->T; get_data_to_write get datum nr k
   * @details This is written with get_data-function pointer because
   I did not want to limit it to work with arrays etc.
    EG, for vertex-partition we would need [1..nv] - array
    which does not need to be constructed!
  **/
  template<typename T, typename TF1, typename TF2>
  Table<T> PartitionData ( size_t n_data,  TF1 gesize_t_of_data, TF2 get_data_to_write) const
  {
    Array<size_t> d2e(n_data);
    d2e = -1;
    size_t pfd;
    for(auto k:Range(n_data))
      if( (pfd = gesize_t_of_data(k)) != NO_EQC )
        { d2e[k] = pfd; }
    Array<int> sizes(neqcs);
    sizes = 0;
    for(auto k:Range(n_data))
      if(d2e[k]!=NO_EQC)
        { sizes[d2e[k]]++; }
    Table<T> tc(sizes);
    sizes = 0;
    for(auto k:Range(n_data))
      if(d2e[k]!=NO_EQC)
        { tc[d2e[k]][sizes[d2e[k]]++] = get_data_to_write(k); }
    return std::move(tc);
  }

  template<class T> Array<MPI_Request> ScatterEQCData (T & tab_in) const
  {
    int nreqs = 0;
    for (auto k : Range(neqcs)) {
      auto dpsk = dist_procs[k];
      if (auto dps = dpsk.Size()) {
        if ( rank < dpsk[0] )
          { nreqs += dps; }
        else
          { nreqs++; }
      }
    }
    Array<MPI_Request> reqs(nreqs); nreqs = 0;
    for (auto k : Range(neqcs)) {
      auto dpsk = dist_procs[k];
      if (dpsk.Size()) {
        if ( rank < dpsk[0] ) {
          for (auto p : dpsk)
            { reqs[nreqs++] = comm.ISend(tab_in[k], p, MPI_TAG_AMG); }
        }
        else
          { reqs[nreqs++] = comm.IRecv(tab_in[k], dpsk[0], MPI_TAG_AMG); }
      }
    }
    return reqs;
  }

  template<class T>
  void
  ScatterEQCArray (FlatArray<T> arr) const
  {
    int nreqs = 0;
    for (auto k : Range(neqcs)) {
      auto dpsk = dist_procs[k];
      if (auto dps = dpsk.Size()) {
        if ( rank < dpsk[0] )
          { nreqs += dps; }
        else
          { nreqs++; }
      }
    }
    Array<MPI_Request> reqs(nreqs); nreqs = 0;
    for (auto k : Range(neqcs)) {
      auto dpsk = dist_procs[k];
      if (dpsk.Size()) {
        if ( rank < dpsk[0] ) {
          for (auto p : dpsk)
            { reqs[nreqs++] = comm.ISend(arr[k], p, MPI_TAG_AMG); }
        }
        else
          { reqs[nreqs++] = comm.IRecv(arr[k], dpsk[0], MPI_TAG_AMG); }
      }
    }
    MyMPI_WaitAll(reqs);
  }
  
  protected:

  void SetupFromInitialDPs (Table<int> && vanilla_dps);
  void SetupFromInitialDPs2 (Table<int> && vanilla_dps);
  void SetupFromDPs (Table<int> && new_dps);

  bool is_dummy = false;

  NgsAMG_Comm comm;
  int rank, np;

  size_t neqcs, neqcs_glob;

  Array<size_t> eqc_ids;
  Table<int> dist_procs;
  Array<int> all_dist_procs;

  Array<size_t> merge;
  FlatMatrix<size_t> mat_merge;
  Array<size_t> intersect;
  FlatMatrix<size_t> mat_intersect;
  BitArray hierarchic_order;

  /** maps identifier to index **/
  Array<size_t> idf_2_ind;

  bool _isRankZeroIdle;
  bool _isTrulyParallel;

  friend std::ostream & operator<<(std::ostream &os, const EQCHierarchy& p);
}; // class EQCHierarchy


class DummyEQCHierarchy : public EQCHierarchy
{
public:
  DummyEQCHierarchy();
}; // class DummyEQCHierarchy


std::ostream & operator<<(std::ostream &os, const EQCHierarchy& p);


} // namespace amg

#endif // FILE_EQCHIERARCHY_HPP
