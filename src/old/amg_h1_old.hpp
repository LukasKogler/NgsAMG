#ifndef FILE_AMGH1_HPP
#define FILE_AMGH1_HPP

namespace amg
{

  /**
     H1 preconditioner.

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
    **/
    template<class TMAP> INLINE void map_data_impl (const TMAP & cmap, H1VData & ch1v) const;
    INLINE void map_data (const BaseCoarseMap & cmap, H1VData & ch1v) const
    { map_data_impl(cmap, ch1v); }
    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1VData & ch1v) const
    { map_data_impl(cmap, ch1v); }
  }; // class H1VData


  /** data which we attach to each edge in the mesh **/
  class H1EData : public AttachedNodeData<NT_EDGE, double, H1EData>
  {
  public:
    using AttachedNodeData<NT_EDGE, double, H1EData>::map_data;
    H1EData (Array<double> && _data, PARALLEL_STATUS _stat) : AttachedNodeData<NT_EDGE, double, H1EData>(move(_data), _stat) {}
    template<class TMESH> INLINE void map_data_impl (const TMESH & cmap, H1EData & ch1e) const;
    INLINE void map_data (const BaseCoarseMap & cmap, H1EData & ch1e) const
    { map_data_impl(cmap, ch1e); }
    template<class TMESH> INLINE void map_data (const AgglomerateCoarseMap<TMESH> & cmap, H1EData & ch1e) const
    { map_data_impl(cmap, ch1e); }
  }; // class H1EData


  using H1Mesh = BlockAlgMesh<H1VData, H1EData>;

  class H1AMGFactory : public VertexBasedAMGFactory<H1AMGFactory, H1Mesh, double>
  {
  public:
    using TMESH = H1Mesh;
    using T_V_DATA = double;
    using TM = double;

    using BASE = VertexBasedAMGFactory<H1AMGFactory, H1Mesh, double>;
    using BASE::Options;

    H1AMGFactory (shared_ptr<TMESH> mesh,  shared_ptr<Options> opts, shared_ptr<BaseDOFMapStep> _embed_step = nullptr);

    static void SetOptionsFromFlags (Options& opts, const Flags & flags, string prefix = "ngs_amg_");
    
    virtual void SetCoarseningOptions (VWCoarseningData::Options & opts, shared_ptr<H1Mesh> mesh) const override;

    // these are not the weights for the coarsening - these are just used for determining the S-prol graph!
    template<NODE_TYPE NT> INLINE double GetWeight (const TMESH & mesh, const AMG_Node<NT> & node) const;

    INLINE void CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh,
			      AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, double & mat) const;

    INLINE void CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<TM> mat) const;

    static INLINE void CalcQ (double vdi, double vdj, double & c) { c = 1; }
    static INLINE void ModQ (double vdi, double vdj, double & c) { c = 1; }
    static INLINE void CalcQs  (double a, double b, double &c, double &d) { c = 1; d = 1; }
    static INLINE void ModQs  (double a, double b, double &c, double &d) { c = 1; d = 1; }
    static INLINE void CalcQij (double a, double b, double &c) { c = 1; }
    static INLINE void ModQij (double a, double b, double &c) { c = 1; }
    static INLINE void CalcQHh (double a, double b, double &c) { c = 1; }
    static INLINE void ModQHh (double a, double b, double &c) { c = 1; }
    static INLINE double CalcMPData (double a, double b) { return 0; }


  }; // class H1AMGFactory


} // namespace amg

#include "amg_h1_impl.hpp"

#endif // FILE_AMGH1_HPP
