#ifndef FILE_AMGH1
#define FILE_AMGH1

namespace amg
{

  /**
     Scalar, H1 preconditioner.
     
     The data we need to attach to the mesh is:
        - one double per vertex (for coarsening weights)
        - one double per edge (for coarsening weights)
  **/

  /** data which we attach to each vertex in the mesh **/
  class H1VData : public AttachedNodeData<NT_VERTEX, double, H1VData>
  {
  public:
    H1VData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, double, H1VData>(move(_data), _stat) {}
    /** 
	What we need to implement is how we want to map the data from the fine to the coarse level.
	In this case we just sum up the weights of all nodes that map to the same coarse node.
	We first distribute and then add values locally. The resulting coarse data is DISTRIBUTED (!).
	
	  Note: We cannot sum up cumulated fine values and get cumulated coarse values because 
	        vertices can (depending on the coarsening algorithm) change equivalence class between levels.

	(public BC it is called from AND<...>, can I find a way to make this protected??)
     **/
    INLINE PARALLEL_STATUS map_data (const CoarseMap & cmap, Array<double> & cdata) const
    {
      Distribute();
      cdata.SetSize(cmap.GetMappedNN<NT_VERTEX>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_VERTEX>();
      // cout << "V map: " << endl; prow(map); cout << endl;
      // cout << "orig V data: "; prow(data); cout << endl;
      for (auto k : Range(map.Size()))
	{ auto cid = map[k]; if ( cid != decltype(cid)(-1)) cdata[cid] += data[k]; }
      // cout << "mapped V data: "; prow(cdata); cout << endl;
      return DISTRIBUTED;
    }
  };
  /** data which we attach to each edge in the mesh **/
  class H1EData : public AttachedNodeData<NT_EDGE, double, H1EData>
  {
  public:
    H1EData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, double, H1EData>(move(_data), _stat) {}
    INLINE PARALLEL_STATUS map_data (const CoarseMap & cmap, Array<double> & cdata) const
    {
      Distribute();
      cdata.SetSize(cmap.GetMappedNN<NT_EDGE>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_EDGE>();
      // cout << "E map: " << endl; prow(map); cout << endl;
      // cout << "orig E data: "; prow(data); cout << endl;
      for (auto k : Range(map.Size()))
	{ auto cid = map[k]; if ( cid != decltype(cid)(-1)) cdata[cid] += data[k]; }
      // cout << "mapped E data: "; prow(cdata); cout << endl;
      return DISTRIBUTED;
    }
  };
  using H1Mesh = BlockAlgMesh<H1VData, H1EData>;
  class H1AMG : public VWiseAMG<H1AMG, H1Mesh, double>
  {
  public:
    using TMESH = H1Mesh;
    using VWiseAMG<H1AMG, H1Mesh, double>::Options;
    H1AMG (shared_ptr<TMESH> mesh,  shared_ptr<Options> opts);
    INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap & map,
			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, double & mat) const
    { SetIdentity(mat); }
  protected:
    virtual void SetCoarseningOptions (shared_ptr<VWCoarseningData::Options> & opts, INT<3> level, shared_ptr<TMESH> mesh) override;
  };


  
#ifndef FILE_AMGH1_CPP
  extern template class EmbedVAMG<H1AMG>;
#endif

#ifndef FILE_AMGCTR_CPP
  extern template class GridContractMap<H1Mesh>;
#endif

} // namespace amg

#endif
