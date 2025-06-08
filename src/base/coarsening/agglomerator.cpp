
#include "agglomerator.hpp"

namespace amg
{

void
BaseAgglomerator::
InitializeBaseAgg (const AggOptions & opts, int level)
{
  edge_thresh = opts.edge_thresh.GetOpt(level);
  vert_thresh = opts.vert_thresh.GetOpt(level);
  robust_crs = opts.crs_robust.GetOpt(level);
  print_aggs = opts.print_aggs.GetOpt(level);
  use_stab_ecw_hack = opts.ecw_stab_hack.GetOpt(level);
  check_isolated_vertices = opts.check_iso.GetOpt(level);
} // BaseAgglomerator::Initialize


void
BaseAgglomerator::
SetFreeVerts (shared_ptr<BitArray> &free_verts)
{
  _free_verts = free_verts;

  if (check_isolated_vertices)
  {
    shared_ptr<BitArray> freeAndConn;

    if ( _free_verts )
    {
      // copy free-verts
      freeAndConn = make_shared<BitArray>(*_free_verts);
    }
    else
    {
      freeAndConn = make_shared<BitArray>(GetBTM().GetNN<NT_VERTEX>());
      freeAndConn->Set();
    }

    size_t cntIso = 0;
    if ( GetBTM().GetNN<NT_VERTEX>() > 0 )
    {
      auto const &econ = *GetBTM().GetEdgeCM();
      // can only make this decision for local vertices,
      // shared vertices would need an exchange
      for (auto vnr : GetBTM().GetENodes<NT_VERTEX>(0))
      {
        if ( ( _free_verts == nullptr ) || _free_verts->Test(vnr) )
        {
          if ( econ.GetRowIndices(vnr).Size() == 0 )
          {
            freeAndConn->Clear(vnr);
            cntIso++;
          }
        }
      }
    }

    cout << " Found " << cntIso << " isolated vertices, keeping them out of next level!" << endl;

  // disable for debugging!
    // if ( cntIso > 0 )
    // {
    //   _free_verts = freeAndConn;
    // }
  }
} // BaseAgglomerator::SetFreeVerts


void
BaseAgglomerator::
SetSolidVerts (shared_ptr<BitArray> &solid_verts)
{
  _solid_verts = solid_verts;
} // BaseAgglomerator::SetSolidVerts


void
BaseAgglomerator::
SetFixedAggs (Table<int> && fixed_aggs)
{
  _fixed_aggs = std::move(fixed_aggs);
} // BaseAgglomerator::SetFixedAggs


void
BaseAgglomerator::
SetAllowedEdges (shared_ptr<BitArray> &allowed_edges)
{
  _allowed_edges = allowed_edges;
} // BaseAgglomerator::SetAllowedEdges


} // namespace amg