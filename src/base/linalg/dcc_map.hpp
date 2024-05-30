#ifndef FILE_AMG_CC_MAP_HPP
#define FILE_AMG_CC_MAP_HPP

#include <base.hpp>

namespace amg
{
/**
 * DCCMap adds a third "valid" parallel status for parallel vectors:
 *     "CONCENTRATED":  DISTRIBUTED, but for each DOF a designated proc has the full value
 *                     and the others have zeros.
 *     !!! in this context, the master of a DOF is not necessarily the lowest proc that shares it
 *         (but it is always ONE of them) !!!
 *   relevant transformations are:
 *     - DISTRIBUTED -> CONCENTRATED
 *     - CONCENTRATED -> CUMULATED
 *   master of each DOF not necessarily lowest rank
 */
template<class TSCAL>
class DCCMap
{
public:

  DCCMap (shared_ptr<ParallelDofs> _pardofs);

  ~DCCMap ();

  shared_ptr<BitArray> GetMasterDOFs () const { return m_dofs; }

  shared_ptr<ParallelDofs> GetParallelDofs () const { return pardofs; }

  /** buffer G vals, start M recv / G send, and zeros out G vec vals (-> reserve M/G buffers)**/
  void StartDIS2CO (BaseVector & vec) const;

  /** wait for M recv to finish, add M buf vals to vec, free M buffer  **/
  void ApplyDIS2CO (BaseVector & vec) const;

  /** wait for G send to finish, free G buffer */
  void FinishDIS2CO () const;

  /** buffer M vals, start M send / G recv (-> reserve M/G buffers) **/
  void StartCO2CU (BaseVector & vec) const;

  /** wait for G recv to finish, replace G values, free G buffer **/
  void ApplyCO2CU (BaseVector & vec) const;

  /*** wait for M send to finish, free M buffer */
  void FinishCO2CU () const;

  FlatArray<int> GetMDOFs (int kp) const { return m_ex_dofs[kp]; }
  FlatArray<int> GetGDOFs (int kp) const { return g_ex_dofs[kp]; }

  /** used for DIS2CO and CO2CU **/
  void WaitM () const;
  void WaitG () const;
  void WaitD2C () const;

  /** used for DIS2CO **/
  void BufferG (BaseVector & vec) const;
  void ApplyM (BaseVector & vec) const;

  /** used for CO2CU **/
  void BufferM (BaseVector & vec) const;
  void ApplyG (BaseVector & vec)const;

protected:

  /** Call in constructor to allocate MPI requests and buffers **/
  void AllocMPIStuff ();

  /** Implement sth like that to decide who is master of which DOFs (constructs m_dofs, m_ex_dofs, g_ex_dofs) **/
  // virtual void CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h = nullptr) = 0;

  shared_ptr<ParallelDofs> pardofs;

  int block_size;

  shared_ptr<BitArray> m_dofs;

  Array<NG_MPI_Request> m_reqs;
  Array<NG_MPI_Request> m_send, m_recv;
  Table<int> m_ex_dofs;      // master ex-DOFs for each dist-proc (we are master of these)
  Table<TSCAL> m_buffer;    // buffer for master-DOF vals for each dist-proc

  Array<NG_MPI_Request> g_reqs;
  Array<NG_MPI_Request> g_send, g_recv;
  Table<int> g_ex_dofs;      // ghost ex-DOFs  for each dist-proc (they are master of these)
  Table<TSCAL> g_buffer;    // buffer for ghost-DOF vals  for each dist-proc

}; // class DCCMap



/**
    Splits eqch EQC into evenly sized chunks, one chunk goes to each proc.
    There is a minimum chunk size. If there are more procs than chunks for an EQC,
    randomly select ranks to assign them to.
**/
template<class TSCAL>
class ChunkedDCCMap : public DCCMap<TSCAL>
{
public:
  ChunkedDCCMap (shared_ptr<EQCHierarchy> eqc_h, shared_ptr<ParallelDofs> _pardofs,
      int _MIN_CHUNK_SIZE = 50);

protected:

  using DCCMap<TSCAL>::pardofs;
  using DCCMap<TSCAL>::m_dofs;
  using DCCMap<TSCAL>::m_ex_dofs;
  using DCCMap<TSCAL>::g_ex_dofs;

  const int MIN_CHUNK_SIZE;

  virtual void CalcDOFMasters (shared_ptr<EQCHierarchy> eqc_h);
};


/** the trivial choice, master of each DOF is the rank **/
template<class TSCAL>
class BasicDCCMap : public DCCMap<TSCAL>
{
public:
  BasicDCCMap (shared_ptr<ParallelDofs> _pardofs);

protected:

  using DCCMap<TSCAL>::pardofs;
  using DCCMap<TSCAL>::m_dofs;
  using DCCMap<TSCAL>::m_ex_dofs;
  using DCCMap<TSCAL>::g_ex_dofs;

  virtual void CalcDOFMasters ();
};

/** END DCCMap **/
} // namespace amg

#endif // FILE_AMG_CC_MAP_HPP