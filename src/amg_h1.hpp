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
    using AttachedNodeData<NT_VERTEX, double, H1VData>::map_data;
    H1VData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_VERTEX, double, H1VData>(move(_data), _stat) {}
    /** 
	What we need to implement is how we want to map the data from the fine to the coarse level.
	In this case we just sum up the weights of all nodes that map to the same coarse node.
	We first distribute and then add values locally. The resulting coarse data is DISTRIBUTED (!).
	
	  Note: We cannot sum up cumulated fine values and get cumulated coarse values because 
	        vertices can (depending on the coarsening algorithm) change equivalence class between levels.

	(public BC it is called from AND<...>, can I find a way to make this protected??)
     **/
    INLINE PARALLEL_STATUS map_data (const BaseCoarseMap & cmap, Array<double> & cdata) const
    {
      cdata.SetSize(cmap.GetMappedNN<NT_VERTEX>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_VERTEX>();
      auto lam_v = [&](const AMG_Node<NT_VERTEX> & v)
	{ auto cv = map[v]; if (cv != -1) cdata[cv] += data[v]; };
      bool master_only = (GetParallelStatus()==CUMULATED);
      mesh->Apply<NT_VERTEX>(lam_v, master_only);
      return DISTRIBUTED;
    }
  };
  /** data which we attach to each edge in the mesh **/
  class H1EData : public AttachedNodeData<NT_EDGE, double, H1EData>
  {
  public:
    using AttachedNodeData<NT_EDGE, double, H1EData>::map_data;
    H1EData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, double, H1EData>(move(_data), _stat) {}
    INLINE PARALLEL_STATUS map_data (const BaseCoarseMap & cmap, Array<double> & cdata) const
    {
      cdata.SetSize(cmap.GetMappedNN<NT_EDGE>()); cdata = 0.0;
      auto map = cmap.GetMap<NT_EDGE>();
      auto lam_e = [&](const AMG_Node<NT_EDGE>& e)
	{ auto cid = map[e.id]; if ( cid != decltype(cid)(-1)) { cdata[cid] += data[e.id]; } };
      bool master_only = (GetParallelStatus()==CUMULATED);
      mesh->Apply<NT_EDGE>(lam_e, master_only);
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
    template<NODE_TYPE NT> INLINE double GetWeight (const TMESH & mesh, const AMG_Node<NT> & node) const
    { // TODO: should this be in BlockAlgMesh instead???
      if constexpr(NT==NT_VERTEX) { return get<0>(mesh.Data())->Data()[node]; }
      else if constexpr(NT==NT_EDGE) { return get<1>(mesh.Data())->Data()[node.id]; }
      else return 0;
    }
    INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap<TMESH> & map,
			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, double & mat) const
    { SetIdentity(mat); }
    INLINE void CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<double> mat) const
    {
      auto w = GetWeight<NT_EDGE>(fmesh, edge); cout << "wt is " << w << endl;
      mat(0,1) = mat(1,0) = - (mat(0,0) = mat(1,1) = w);
    }
  };


} // namespace amg

#endif
