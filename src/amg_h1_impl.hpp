#ifndef FILE_AMGH1_IMPL_HPP
#define FILE_AMGH1_IMPL_HPP

namespace amg
{

  INLINE void H1VData :: map_data (const BaseCoarseMap & cmap, H1VData & ch1v) const
  {
    auto & cdata = ch1v.data;
    cdata.SetSize(cmap.GetMappedNN<NT_VERTEX>()); cdata = 0.0;
    auto map = cmap.GetMap<NT_VERTEX>();
    auto lam_v = [&](const AMG_Node<NT_VERTEX> & v)
      { auto cv = map[v]; if (cv != -1) cdata[cv] += data[v]; };
    bool master_only = (GetParallelStatus()==CUMULATED);
    mesh->Apply<NT_VERTEX>(lam_v, master_only);
    ch1v.SetParallelStatus(DISTRIBUTED);
  }


  INLINE void H1EData :: map_data (const BaseCoarseMap & cmap, H1EData & ch1e) const
  {
    auto & cdata = ch1e.data;
    cdata.SetSize(cmap.GetMappedNN<NT_EDGE>()); cdata = 0.0;
    auto map = cmap.GetMap<NT_EDGE>();
    auto lam_e = [&](const AMG_Node<NT_EDGE>& e)
      { auto cid = map[e.id]; if ( cid != decltype(cid)(-1)) { cdata[cid] += data[e.id]; } };
    bool master_only = (GetParallelStatus()==CUMULATED);
    mesh->Apply<NT_EDGE>(lam_e, master_only);
    ch1e.SetParallelStatus(DISTRIBUTED);
  }


  template<NODE_TYPE NT> INLINE double H1AMGFactory :: GetWeight (const TMESH & mesh, const AMG_Node<NT> & node) const
  {
    if constexpr(NT==NT_VERTEX) { return get<0>(mesh.Data())->Data()[node]; }
    else if constexpr(NT==NT_EDGE) { return get<1>(mesh.Data())->Data()[node.id]; }
    else return 0;
  }


  INLINE void H1AMGFactory :: CalcPWPBlock (const TMESH & fmesh, const TMESH & cmesh, const CoarseMap<TMESH> & map,
					    AMG_Node<NT_VERTEX> v, AMG_Node<NT_VERTEX> cv, double & mat) const
  { SetIdentity(mat); }


  INLINE void H1AMGFactory :: CalcRMBlock (const TMESH & fmesh, const AMG_Node<NT_EDGE> & edge, FlatMatrix<double> mat) const
  {
    auto w = GetWeight<NT_EDGE>(fmesh, edge);
    mat(0,1) = mat(1,0) = - (mat(0,0) = mat(1,1) = w);
  }

} // namespace amg

#endif // FILE_AMGH1_IMPL_HPP
