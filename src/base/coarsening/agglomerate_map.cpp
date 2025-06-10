
#include "agglomerate_map.hpp"

namespace amg
{

/** BaseAgglomerateCoarseMap **/

BaseAgglomerateCoarseMap :: BaseAgglomerateCoarseMap(shared_ptr<TopologicMesh> mesh)
  : BaseCoarseMap(mesh, nullptr)
  , _agglomerator(nullptr)
{
  assert(mesh != nullptr); // obviously this would be bad
  for (NODE_TYPE NT : {NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL} )
    { NN[NT] = mapped_NN[NT] = 0; }
  NN[NT_VERTEX] = mesh->template GetNN<NT_VERTEX>();
  NN[NT_EDGE] = mesh->template GetNN<NT_EDGE>();
} // BaseAgglomerateCoarseMap(..)


shared_ptr<TopologicMesh> BaseAgglomerateCoarseMap :: GetMappedMesh () const
{
  if (mapped_mesh == nullptr)
    { const_cast<BaseAgglomerateCoarseMap&>(*this).BuildMappedMesh(); }
  return mapped_mesh;
} // BaseBaseAgglomerateCoarseMap::GetMappedMesh


void BaseAgglomerateCoarseMap :: InitializeAgg (const AggOptions & opts, int level)
{
  _agglomerator = createAgglomerator();

  print_vmap = opts.print_vmap.GetOpt(level);

  getAgglomerator().InitializeBaseAgg(opts, level);
} // BaseBaseAgglomerateCoarseMap::SetLevelAggOptions


void BaseAgglomerateCoarseMap :: SetFreeVerts (shared_ptr<BitArray> &free_verts)
{
  getAgglomerator().SetFreeVerts(free_verts);
} // BaseBaseAgglomerateCoarseMap::SetFreeVerts


void BaseAgglomerateCoarseMap :: SetSolidVerts (shared_ptr<BitArray> &solid_verts)
{
  _solid_verts = solid_verts;
  getAgglomerator().SetSolidVerts(solid_verts);
} // BaseBaseAgglomerateCoarseMap::SetSolidVerts


void BaseAgglomerateCoarseMap :: SetFixedAggs (Table<int> && fixed_aggs)
{
  getAgglomerator().SetFixedAggs(std::move(fixed_aggs));
} // BaseBaseAgglomerateCoarseMap::setFixedAggs


void BaseAgglomerateCoarseMap :: SetAllowedEdges (shared_ptr<BitArray> &allowed_edges)
{
  getAgglomerator().SetAllowedEdges(allowed_edges);
} // BaseBaseAgglomerateCoarseMap::SetAllowedEdges


void BaseAgglomerateCoarseMap :: BuildMappedMesh ()
{
  static Timer t("BaseBaseAgglomerateCoarseMap::BuildMappedMesh");
  RegionTimer rt(t);

  auto eqc_h = mesh->GetEQCHierarchy();
  auto &agglomerator = getAgglomerator();

  Array<Agglomerate> aggs;
  Array<int> v_to_agg;
  agglomerator.FormAgglomerates(aggs, v_to_agg);

  _is_center = make_shared<BitArray>(mesh->template GetNN<NT_VERTEX>());
  _is_center->Clear();
  for (const auto & agg : aggs)
    { _is_center->SetBit(agg.center()); }

  allocateMappedMesh(eqc_h);

  auto mapped_btm = dynamic_pointer_cast<BlockTM>(GetMappedMesh());

  assert(mapped_btm != nullptr);

  mapped_btm->has_nodes[NT_VERTEX] = mapped_btm->has_nodes[NT_EDGE] = true;
  mapped_btm->has_nodes[NT_FACE] = mapped_btm->has_nodes[NT_CELL] = false;

  if (_solid_verts == nullptr)
    { MapVerts(*mapped_btm, aggs, v_to_agg); }
  else
    { MapVerts_sv(*mapped_btm, aggs, v_to_agg); }

  if (this->print_vmap)
    { cout << " vmap: " << endl; prow2(this->GetMap<NT_VERTEX>()); cout << endl; }

  MapEdges(*mapped_btm, aggs, v_to_agg);

  fillCoarseMesh();

} // BaseBaseAgglomerateCoarseMap::BuildMappedMesh


void BaseAgglomerateCoarseMap :: MapVerts (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg)
{
  static Timer t("MapVerts"); RegionTimer rt(t);

  /**
     We need:
        I) No non-master vertex is in any agglomerate I have.
  II) For each member v of an agg with center c:
            i) eqc(v) <= eqc(c)
      ii) I am master of c (and therefore also of v)
      [So no agglomerates that cross process boundaries]
      -> each coarse vertex, defined by an agglomerate is in the eqc of the (fine) center vertex v of that agglomerate
    **/

  auto tm_mesh = dynamic_pointer_cast<BlockTM>(GetMesh());
  const auto & M = *tm_mesh;
  const auto & eqc_h = *M.GetEQCHierarchy();
  auto comm = eqc_h.GetCommunicator();
  auto neqcs = eqc_h.GetNEQCS();

  // cout << " EQCH: " << endl << eqc_h << endl;
  // cout << " fmesh NV " << M.nnodes[NT_VERTEX] << " " << M.nnodes_glob[NT_VERTEX] << endl;
  // cout << " fmesh eqc_verts " << endl << M.eqc_verts << endl;
  // cout << " agglomerates: " << endl << agglomerates << endl;
  // cout << " v_to_agg: " << endl; prow2(v_to_agg); cout << endl;

  auto& vmap = node_maps[0];
  vmap.SetSize(M.template GetNN<NT_VERTEX>()); vmap = -1;

  /** EX-data: [crs_cnt, Nsame, Nchange, (locvmap, eqid)-array
crs_cnt .. #of verts in eqc on coarse level
Nsame, Nchange .. #of verts that stay in this eqc / that change eqc **/
  Array<int> cnt_vs(neqcs); cnt_vs = 0;

  Array<int> agg_map(agglomerates.Size()); agg_map = -1;
  Array<int> agg_eqc(agglomerates.Size()); agg_eqc = -1;

  /** **/
  Array<int> exds(neqcs); exds = 0;
  Table<int> exdata;
  if (neqcs > 1)
    for (auto k : Range(size_t(1), neqcs))
{ exds[k] = 3 + 2 * M.template GetENN<NT_VERTEX>(k); }
  exdata = Table<int> (exds);
  // if (neqcs > 1)
  //   for (auto eqc : Range(size_t(1), neqcs)) {
  // 	if (eqc_h.IsMasterOfEQC(eqc)) { exdata[eqc] = -2; }
  // 	else { exdata[eqc] = -3; }
  //   }
  if (neqcs > 1) {
    for (auto eqc : Range(size_t(1), neqcs)) {
if (eqc_h.IsMasterOfEQC(eqc)) {
  auto exrow = exdata[eqc];
  exrow[0] = 0;
  auto & cnt_same = exrow[1]; cnt_same = 0;
  auto & cnt_change = exrow[2]; cnt_change = 0;
  auto & loc_map = exrow.Part(3); // cout << " loc map s " << loc_map.Size() << endl;
  auto eqc_verts = M.template GetENodes<NT_VERTEX>(eqc);
  for (auto j : Range(eqc_verts)) {
    auto v = eqc_verts[j];
    auto agg_nr = v_to_agg[v];
    // cout << "eqc j v agg_nr " << eqc << " " << j << " " << v << " " << agg_nr << endl;
    if (agg_nr == -1) { // dirichlet
      loc_map[2*j]   = -1;
      loc_map[2*j+1] = -1;
    }
    else { // not dirichlet
      auto & cid = agg_map[agg_nr];
      auto & ceq = agg_eqc[agg_nr];
      if (agg_map[agg_nr] == -1) {
  auto & agg = agglomerates[agg_nr];
  // cout << "agg: " << agg << endl;
  ceq = M.template GetEQCOfNode<NT_VERTEX>(agg.center());
  agg_map[agg_nr] = cnt_vs[ceq]++;
      }
      // cout << " ceq " << ceq << " cid " << cid << endl;
      loc_map[2*j]   = cid;
      loc_map[2*j+1] = eqc_h.GetEQCID(ceq);
      if (eqc == ceq) { cnt_same++; }
      else { cnt_change++; }
    }
  }
}
    }
    for (auto eqc : Range(size_t(1), neqcs))
{ exdata[eqc][0] = cnt_vs[eqc]; }
  }

  // cout << "(unred) exdata: " << endl << exdata << endl;

  auto reqs = eqc_h.ScatterEQCData(exdata);

  // cout << "(red) exdata: " << endl << exdata << endl;

  /** map agglomerates that do not touch a subdomain boundary **/
  for (auto & agg : agglomerates) { // per definition I am master of all verts in this agg
    auto eqc = M.template GetEQCOfNode<NT_VERTEX>(agg.center());
    if (eqc == 0) { // ALL mems must be local!
auto cid = cnt_vs[0]++;
for (auto m : agg.members())
  { vmap[m] = cid; }
    }
  }

  /** finish EQC scatter, make displ-array **/
  MyMPI_WaitAll(reqs);

  // cout << "(red)   exdata: " << endl << exdata << endl;

  if (neqcs > 1)
    for (auto eqc : Range(size_t(1), neqcs))
{ cnt_vs[eqc] = exdata[eqc][0]; }
  Array<int> disps(1+neqcs); disps = 0;
  for (auto k : Range(neqcs))
    { disps[k+1] = disps[k] + cnt_vs[k]; }

  if (neqcs > 1)
    for (auto eqc : Range(size_t(1), neqcs)) {
auto exrow = exdata[eqc];
auto & cnt_same = exrow[1];
auto & cnt_change = exrow[2];
auto & loc_map = exrow.Part(3);
auto eqc_verts = M.template GetENodes<NT_VERTEX>(eqc);
for (auto j : Range(eqc_verts)) {
  auto v = eqc_verts[j];
  auto cid_loc = loc_map[2*j];
  if (cid_loc != -1) { // dirichlet, once again!
    // cout << v << ", " << j << " in " << eqc << ", locmap " << loc_map[2*j] << flush;
    // cout << " " << loc_map[2*j+1] << flush;
    int ceq = eqc_h.GetEQCOfID(loc_map[2*j+1]);
    // cout << ", ceq " << ceq;
    auto cid = disps[ceq] + cid_loc;
    // cout << ", disp " << disps[ceq] << " -> " << cid << endl;
    vmap[v] = cid;
    auto agg_nr = v_to_agg[v];
    if (agg_nr != -1) { // a local agg, but might also contain verts from eqc 0!
      auto & agg = agglomerates[agg_nr];
      if (agg.center() == v) {
  for (auto mem : agg.members())
    { vmap[mem] = cid; }
      }
    }
  }
}
    }

  mapped_NN[NT_VERTEX] = disps.Last();
  cmesh.nnodes[NT_VERTEX] = disps.Last();
  cmesh.verts.SetSize(cmesh.nnodes[NT_VERTEX]);
  auto & cverts = cmesh.verts;
  for (auto k : Range(cmesh.nnodes[NT_VERTEX]) )
    { cverts[k] = k; }
  auto & disp_veq = cmesh.disp_eqc[NT_VERTEX];
  disp_veq = std::move(disps);
  cmesh.nnodes_eqc[NT_VERTEX] = std::move(cnt_vs);
  cmesh.nnodes_cross[NT_VERTEX].SetSize(neqcs); cmesh.nnodes_cross[NT_VERTEX] = 0;
  cmesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, cmesh.disp_eqc[NT_VERTEX].Data(), cmesh.verts.Data());
  auto ncv_master = 0;
  for (auto eqc : Range(neqcs))
    if (eqc_h.IsMasterOfEQC(eqc))
{ ncv_master += cmesh.GetENN<NT_VERTEX>(eqc); }
  cmesh.nnodes_glob[NT_VERTEX] = comm.AllReduce(ncv_master, NG_MPI_SUM);

  // cout << " fmesh NV " << M.nnodes[NT_VERTEX] << " " << M.nnodes_glob[NT_VERTEX] << endl;
  // cout << " fmesh eqc_verts " << endl << M.eqc_verts << endl;

  // cout << " vmap: " << endl; prow2(vmap); cout << endl;
  // cout << " EQCH: " << endl << eqc_h << endl;
  // cout << " cmesh NV " << cmesh.nnodes[NT_VERTEX] << " " << cmesh.nnodes_glob[NT_VERTEX] << endl;
  // cout << " cmesh eqc_verts " << endl << cmesh.eqc_verts << endl;

  // {
  //   Array<int> cnt(cmesh.template GetNN<NT_VERTEX>()); cnt = 1;
  //   cmesh.template AllreduceNodalData<NT_VERTEX>(cnt, [&](auto in) { return sum_table(in); });
  //   cout << " vertex test cnt: " << endl;
  //   prow2(cnt); cout << endl;
  // }

} // BaseAgglomerateCoarseMap::MapVerts


void BaseAgglomerateCoarseMap :: MapVerts_sv (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v_to_agg)
{
  static Timer t("MapVerts_sv"); RegionTimer rt(t);

  /** Vertex-map for agglomerates constructed on "solid" vertices!
I)    every vertex is in exactly one agg
II)   every agg is local to exactly the proc that has it in the "agglomerates" array
III)  EQC of agg == eqc of agg. center (so all vertices are in a smaller or equal eqc of center vertex)
IV)   difference from generic: no guarantee that the proc that has an agg is the master of the aggs EQC. **/

  auto &solid_vderts = this->_solid_verts;
  auto tm_mesh = dynamic_pointer_cast<BlockTM>(mesh);
  const auto & M = *tm_mesh;
  auto p_eqc_h = M.GetEQCHierarchy();
  const auto & eqc_h = *p_eqc_h;
  auto comm = eqc_h.GetCommunicator();
  auto neqcs = eqc_h.GetNEQCS();
  auto& vmap = node_maps[0];
  vmap.SetSize(M.template GetNN<NT_VERTEX>()); vmap = -1;

  /** We do it like this:
  (I)     per EQC list of cnts per member rank
  (II)    prefix-sums of these for in-eqc offset:
  (III)   cross-eqc prefix sums
  (IV.0)  local vertices are mapped locally
  (IV.1)  ex-vertex map ided by [ eqid, kp, cnt ], new local nr is then
          offset[eqc][kp]++; **/

  // int spec_vert = -1;
  // if (comm.Rank() == 33)
  //   { spec_vert = 432294; }
  // if (comm.Rank() == 35)
  //   { spec_vert = 425201; }
  // bool doco = spec_vert != -1;


  /** (I), and agg-eqcs  **/
  Array<int> perow(neqcs); perow = 1;
  Table<int> tent_cnts(perow); tent_cnts.AsArray() = 0;
  Array<int> agg_eqcs(agglomerates.Size());
  for (auto agg_nr : Range(agglomerates)) {
    auto ctr = agglomerates[agg_nr].center();
    auto eqc = M.template GetEQCOfNode<NT_VERTEX>(ctr);
    agg_eqcs[agg_nr] = eqc;
    tent_cnts[eqc][0]++;
  }
  Table<int> red_cnts = ReduceTable<int, int>(tent_cnts, p_eqc_h, [&](auto & in) {
Array<int> out(in.Size());
for (auto k : Range(out))
  { out[k] = in[k][0]; }
return out;
    });

  // cout << " agg_eqcs " << endl; prow2(agg_eqcs); cout << endl;
  // cout << " tent_cnts " << endl << tent_cnts << endl;
  // cout << " red_cnts " << endl << red_cnts << endl;

  /** (II)+(III) in-and cross-eqc prefix sum **/
  auto & disp_veq = cmesh.disp_eqc[NT_VERTEX];
  disp_veq.SetSize(neqcs+1); disp_veq = 0;
  auto & cnt_vs = cmesh.nnodes_eqc[NT_VERTEX];
  cnt_vs.SetSize(neqcs);
  int sum = 0;
  for (auto k : Range(neqcs)) {
    disp_veq[k] = sum;
    for (auto j : Range(red_cnts[k])) {
int x = red_cnts[k][j];
red_cnts[k][j] = sum;
sum += x;
    }
    cnt_vs[k] = sum - disp_veq[k];
  }
  disp_veq.Last() = sum;

  // cout << " prefixes " << endl << red_cnts << endl;

  /** (IV.0/1) vertex-map! **/
  perow.SetSize(neqcs);
  for (auto k : Range(perow))
    { perow[k] = (k>0) ? M.template GetENN<NT_VERTEX>(k) : 0; }
  Table<IVec<3, int>> exv_map(perow); exv_map.AsArray() = IVec<3, int>(-1); // -1 for verts that are not set (ex diri vs)
  { // set up exv_map
    const auto nvloc = M.template GetENN<NT_VERTEX>(0);
    perow.SetSize(neqcs);
    for (auto k : Range(perow))
{ perow[k] = red_cnts[k].Size(); }
    Table<int> rc2(perow); // we iterate through offsets twice b.c tablecreator, so we have to reset!
    rc2.AsArray() = 0;
    for (auto agg_nr : Range(agglomerates)) {
auto & agg = agglomerates[agg_nr];
auto cv_eq = agg_eqcs[agg_nr];
auto cv_eqid = eqc_h.GetEQCID(cv_eq);
int kp = merge_pos_in_sorted_array(int(comm.Rank()), eqc_h.GetDistantProcs(cv_eq));
// cout << " agg " << agg << " -> CVEQ " << cv_eq << ", kp = " << kp << endl;
// cout << " DPS "; prow(eqc_h.GetDistantProcs(cv_eq)); cout << endl;
auto cv_locnr = rc2[cv_eq][kp]++; // ! "loc" in eqc/proc block, not in eqc !
auto cv_nr = red_cnts[cv_eq][kp] + cv_locnr;
// cout << " cv locnr/nr " << cv_locnr << " " << cv_nr << endl;
if (cv_eq == 0) { // must all be local
  // cout << " -> MAP TO " << cv_nr << endl;
  for (auto v : agg.members())
    { vmap[v] = cv_nr; }
} else { // at least some ex-verts
  for (auto v : agg.members()) {
    vmap[v] = cv_nr;
    if ( !(v < nvloc) ) { // ex vertex
      auto [veq, v_locnr] = M.template MapENodeToEQLNR<NT_VERTEX>(v);
      // cout << "eq, lnr " << veq << " " << v_locnr << endl;
      // cout << " -> write (" << cv_eq << " " << kp << " " << cv_locnr << ") " << endl;
      exv_map[veq][v_locnr] = IVec<3>( { cv_eqid, kp, cv_locnr } );
    }
  }
}
    }
  }  // set up exv_map

  // if (doco)
  // cout << " SPEC V VMAP I " << spec_vert << " -> " << vmap[spec_vert] << endl;

  // cout << " exc_map " << endl << exv_map << endl;

  // cannot use sum_table because of "-1" entries!!
  // auto red_exv_map = ReduceTable<IVec<3>, IVec<3>>(exv_map, p_eqc_h, [&](auto & in) { return std::move(sum_table(in)); });

  auto red_exv_map = ReduceTable<IVec<3>, IVec<3>>(exv_map, p_eqc_h, [&](const auto & tab) {
Array<IVec<3>> out;
auto nrows = tab.Size();
if (nrows == 0) return out;
auto row_s = tab[0].Size();
if (row_s == 0) return out;
out.SetSize(row_s); out = tab[0];
if (nrows == 1) { return out; }
for (size_t k = 1; k < tab.Size(); k++) {
  auto row = tab[k];
  for (auto l : Range(row_s)) {
    // if ( (row[l][0] != -1) && (out[l][0] != -1) )
      // { cout << "ERR MAP " << endl; }
    if (row[l][0] != -1)
      { out[l] = row[l]; }
  }
}
return out;
    });

  // cout << " red_exv_map " << endl << red_exv_map << endl;

  for (auto eqc : Range(red_exv_map)) {
    auto row = red_exv_map[eqc];
    auto eq_vs = M.template GetENodes<NT_VERTEX>(eqc);
    for (auto j : Range(row)) {
auto [ cv_eqid, kp, cv_locnr ] = row[j];
if (cv_eqid == -1)
  { continue; }
auto cv_eq = eqc_h.GetEQCOfID(cv_eqid);
auto cv_nr = red_cnts[cv_eq][kp] + cv_locnr;
vmap[eq_vs[j]] = cv_nr;
// if (doco && (eq_vs[j]==spec_vert)) {
  // cout << "eq " << eqc << " j " << j << ", v " << eq_vs[j] << " rec " << cv_eqid << " " << kp << " " << cv_locnr << " -> " << vmap[eq_vs[j]] << endl;
// }
    }
  }

  // if (doco)
  // cout << " SPEC V VMAP II " << spec_vert << " -> " << vmap[spec_vert] << endl;

  // cout << " okay, final vmap" << endl; prow2(vmap); cout << endl;

  mapped_NN[NT_VERTEX] = disp_veq.Last();
  cmesh.nnodes[NT_VERTEX] = disp_veq.Last();
  cmesh.verts.SetSize(cmesh.nnodes[NT_VERTEX]);
  auto & cverts = cmesh.verts;
  for (auto k : Range(cmesh.nnodes[NT_VERTEX]) )
    { cverts[k] = k; }
  cmesh.nnodes_cross[NT_VERTEX].SetSize(neqcs); cmesh.nnodes_cross[NT_VERTEX] = 0;
  cmesh.eqc_verts = FlatTable<AMG_Node<NT_VERTEX>> (neqcs, cmesh.disp_eqc[NT_VERTEX].Data(), cmesh.verts.Data());
  auto ncv_master = 0;
  for (auto eqc : Range(neqcs))
    if (eqc_h.IsMasterOfEQC(eqc))
{ ncv_master += cmesh.GetENN<NT_VERTEX>(eqc); }
  cmesh.nnodes_glob[NT_VERTEX] = comm.AllReduce(ncv_master, NG_MPI_SUM);

} // BaseAgglomerateCoarseMap::MapVerts_sv


void BaseAgglomerateCoarseMap :: MapEdges (BlockTM & cmesh, FlatArray<Agglomerate> agglomerates, FlatArray<int> v2agg)
{
  // TODO: this would probably be MUCH faster with a sparse matrix-matrix like algorithm to compute local coarse edges
  //       and then some reordering/shuffling around for parallel

  /** Say there is a coarse edge between coarse vertices A and B. Everyone looks at all their edges that map to this coarse edge.
     -) If I have some fine edge in a different EQC that A-B, add the coarse eqc to a table of "additional edges".
        If I do not, increment counter for eq/cross edges in the coarse eqc. If they are local edges, I can instantly map them!
      -) Mark all these fine edges as done.
MPI-gather additional edges.
Go through the gathered coarse edges. Look at all fine edges that map to it:
      -) If all of them are in the same eqc, decreace counter for eq/cross edges by one  and again mark fine edges as done.
Now we have the correct coarse counts.
Go through gathered coarse edges again and map all fine edges concerned.
Go through all UNMAPPED fine edges again, their coarse ids are defined by the offsets + the (consistent) ordering of the
fine edge with the lowest number for each coarse edge.
    **/
  static Timer t("MapEdges"); RegionTimer rt(t);

  auto tm_mesh = dynamic_pointer_cast<BlockTM>(mesh);
  const auto & M = *tm_mesh;
  const auto FNV = M.template GetNN<NT_VERTEX>();
  const auto FNE = M.template GetNN<NT_EDGE>();
  const auto & fecon = *M.GetEdgeCM();

  const auto & CM = cmesh;
  const auto CNV = CM.template GetNN<NT_VERTEX>();

  auto sp_eqc_h = M.GetEQCHierarchy();
  const auto & eqc_h = *sp_eqc_h;
  auto neqcs = eqc_h.GetNEQCS();
  auto comm = eqc_h.GetCommunicator();

  const auto & vmap = node_maps[NT_VERTEX];
  auto & emap = node_maps[NT_EDGE]; emap.SetSize(GetNN<NT_EDGE>()); emap = -1;

  auto fedges = M.template GetNodes<NT_EDGE>();

  // vertex-maps are alreadt done
  auto c2fv = this->template GetMapC2F<NT_VERTEX>();

  // cout << " c2fv TABLE: " << endl << c2fv << endl << endl;

  /**
   *  Iterates through fine edges. If fine and coarse are loc, call lambda on edge.
   *  If not, call lambda only if edge is the designated one. Designated is the one
   *  with the lowest number in the highest eqc.
   *  On first call, sets the "desig" flag for the designated edges.
   *
   *  TODO: this could probably be sped up a bunch by only calling the expensive it_feids once
   *        and caching the edge group
   */
  BitArray mark(FNE);
  BitArray desig(FNE);

  auto wrap_elam = [&](auto eqc,
                       auto edges,
                       auto check_lam,
                       auto order_matters,
                       auto edge_lam) LAMBDA_INLINE
  {
    for (const auto & e : edges)
    {
      // cout << " wrap_elam around " << e << " " << mark.Test(e.id) << " " << desig.Test(e.id) << " " << check_lam(e) << endl;
      if ( (!mark.Test(e.id)) && check_lam(e) )
      {
        auto cv0 = vmap[e.v[0]];
        auto cv1 = vmap[e.v[1]];

        if ( ( cv0 != -1 ) && ( cv1 != -1 ) && ( cv0 != cv1 ) )
        {
          auto ceq0 = CM.template GetEQCOfNode<NT_VERTEX>(cv0);
          auto ceq1 = CM.template GetEQCOfNode<NT_VERTEX>(cv1);

          bool const c_cross = ( ceq0 != ceq1 );

          auto const ceq = c_cross ? eqc_h.GetCommonEQC(ceq0, ceq1) : ceq0;

          /**
          * look through fine edges that map to the same coarse edge,
          * MARK ALL and call lambda FOR ALL.
          */

          auto it_feids = [&](auto ind_lam) LAMBDA_INLINE
          {
            FlatArray<int> memsa = c2fv[cv0];
            FlatArray<int> memsb = c2fv[cv1];

            for (auto x : memsa)
            {
              auto eids = fecon.GetRowValues(x);

              iterate_intersection(fecon.GetRowIndices(x), memsb,
                        [&](auto i, auto j) LAMBDA_INLINE
                        {
                          auto feid = int(eids[i]);
                          mark.SetBit(feid);
                          ind_lam(feid);
                        });
            }
          };

          /**
          * look through all fine edges that map to the same coarse edge,
          * pick the one with the smallest number in the largest eqc
          *   ( marks all others as DONE)
          */

          auto max_eq = eqc;
          auto spec_feid = e.id;

          /** Not a local edge, designated edge not found. **/
          if ( ( order_matters ) && ( ceq > 0 ) && ( !desig.Test(e.id) ) )
          {
            FlatArray<int> memsa = c2fv[cv0];
            FlatArray<int> memsb = c2fv[cv1];

            it_feids ([&](auto feid) LAMBDA_INLINE
            {
              /** Find and mark one fine edge as designated, mark rest as done! **/
              auto eq_fe = M.template GetEQCOfNode<NT_EDGE>(feid);
              // edges do not need to come in sorted
              if ( eq_fe == max_eq )
                { spec_feid = min2(spec_feid, feid); }
              else if ( eqc_h.IsLEQ(max_eq, eq_fe) ) // (eq_fe != max_eq)
              {
                max_eq =  eq_fe;
                spec_feid = feid;
              }
              // edges need come in sorted
              // if ( (eq_fe != max_eq) && (eqc_h.IsLEQ(eq_fe, max_eq)) ) {
              //   /** must be the lowest edge id in the "biggest" eqc (if there is one. if not, edge changes anyways) **/
              //   max_eq =  eq_fe;
              //   spec_feid = feid;
              // }
            });
          }
          // if (doco)
            // { cout << " FE" << e.id << " -> SPEC FE " << spec_feid << endl; }
          if (spec_feid != e.id) // call this later
            { desig.SetBit(spec_feid); continue; }

          // we only get here for the designated edges

          /** Either order does not matter (count), or we must be either local, or the designated fine edge! **/
          if (cv0 > cv1) { swap(cv0, cv1); swap(ceq0, ceq1); }

          // finally, call lambda
          edge_lam (spec_feid, max_eq, ceq, c_cross,
                    cv0, ceq0, cv1, ceq1, it_feids);
        }
      }
    }
  };

  /** ~ iterate edges unique **/
  auto it_es_u = [&](bool doloc,
                     auto check_lam,
                     auto edge_lam,
                     bool order_matters) LAMBDA_INLINE
  {
    for (int eqc = ( doloc ? 0 : 1 ); eqc < neqcs; eqc++ )
    {
      wrap_elam(eqc, M.template GetENodes<NT_EDGE>(eqc), check_lam, order_matters, edge_lam);
      wrap_elam(eqc, M.template GetCNodes<NT_EDGE>(eqc), check_lam, order_matters, edge_lam);
    }
  };

  auto sort_tup = [&](auto & t) LAMBDA_INLINE {
    if (t[0] == t[1]) { if (t[2] > t[3]) { swap(t[2], t[3]); } }
    else if (t[0] > t[1]) { swap(t[1], t[0]); swap(t[2], t[3]); }
  };
  auto less_tup = [&](const auto & a, const auto & b) LAMBDA_INLINE {
    const bool ain = a[0] == a[1], bin = b[0] == b[1];
    if (ain && !bin) { return true; } else if (bin && !ain) { return false; }
    else if (a[0] < b[0]) { return true; } else if (a[0] > b[0]) { return false; }
    else if (a[1] < b[1]) { return true; } else if (a[1] > b[1]) { return false; }
    else if (a[2] < b[2]) { return true; } else if (a[2] > b[2]) { return false; }
    else if (a[3] < b[3]) { return true; } else if (a[3] > b[3]) { return false; }
    else { return false; }
  };

  Array<int> cnt_eq(neqcs);
  Array<int> cnt_cr(neqcs);
  Array<int> cnt_add_ce(neqcs);

  cnt_eq = 0;
  cnt_cr = 0;
  cnt_add_ce = 0;

  // first round: check all edges, also local ones, increment counters

  mark.Clear();
  desig.Clear();

  it_es_u(true, // use local ones
          [&](auto feid) LAMBDA_INLINE { return true; }, // check all edges
          [&](auto spec_feid, auto feqc, auto ceqc, auto ccross,
              auto cv0, auto ceq0, auto cv1, auto ceq1, auto it_feids ) LAMBDA_INLINE
            {
              if ( ceqc != feqc )
              {
                cnt_add_ce[ceqc]++;
              }

              if (ccross)
                { cnt_cr[ceqc]++; }
              else
                { cnt_eq[ceqc]++; }

              // mark other fine edges mapping to the same coarse edge
              it_feids( [&](auto feid) LAMBDA_INLINE { /*cout << fedges[feid] << " "; */; } );
            },
          false // order does not matter
          );

  // cout << " cnt_add_ce : " << endl << cnt_add_ce << endl;

  Table<IVec<4,int>> tent_add_ce(cnt_add_ce);

  cnt_add_ce = 0;

  // fill add_ce table
  if ( neqcs > 1 )
  {
    mark.Clear(); // leave desig

    it_es_u(true, // use local ones
            [&](auto feid) LAMBDA_INLINE { return true; }, // check all edges
            [&](auto spec_feid, auto feqc, auto ceqc, auto ccross,
                auto cv0, auto ceq0, auto cv1, auto ceq1, auto it_feids ) LAMBDA_INLINE
              {
                if ( ceqc != feqc )
                {
                  auto ceq0_id = eqc_h.GetEQCID(ceq0);
                  auto ceq1_id = eqc_h.GetEQCID(ceq1);
                  auto cv0_loc = CM.template MapENodeToEQC<NT_VERTEX>(ceq0, cv0);
                  auto cv1_loc = CM.template MapENodeToEQC<NT_VERTEX>(ceq1, cv1);

                  // cout << "FE " << fedges[spec_feid] << endl
                  //      << "   CV0 " << cv0 << " eq " << ceq0 << " id " << ceq0_id << " loc " << cv0_loc << endl
                  //      << "   CV1 " << cv1 << " eq " << ceq1 << " id " << ceq1_id << " loc " << cv1_loc << endl;
                  // cout << "      -> INTO " << ceqc << " " << cnt_add_ce[ceqc] << endl;

                  IVec<4,int> tup( { ceq0_id, ceq1_id, cv0_loc, cv1_loc } );
                  sort_tup(tup);

                  tent_add_ce[ceqc][cnt_add_ce[ceqc]++] = tup;
                }

                // mark others mapping to the same edge as DONE
                it_feids( [&](auto feid) LAMBDA_INLINE { /*cout << fedges[feid] << " "; */; } );
              },
            false // order does not matter
            );

    // should be cheap (few add_edges!). rows are duplicate-less, but diffrent rows can contain same edge
    for (auto row : tent_add_ce)
    {
      QuickSort(row, less_tup);
    }
  }

  // cout << endl << "LOC ADD_CE: " << endl;
  // cout << tent_add_ce << endl << endl;

  Table<IVec<4,int>> add_ce = ReduceTable<IVec<4,int>,IVec<4,int>>(
    tent_add_ce,
    sp_eqc_h,
    [&less_tup](const auto & tab) LAMBDA_INLINE
    {
      return merge_arrays(tab, less_tup);
    }
  );

  // cout << endl << "RED ADD_CE: " << endl;
  // cout << add_ce << endl << endl;

  // cout << endl << "LOC CNTS cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
  // cout << endl << "LOC CNTS cnt_cr     " << endl; prow2(cnt_cr); cout << endl;

  /**
   * Now we have to iterate through add_cedges, and decrease counters if we have
   * any fine edge mapping to it. Counters for add_edges are correct already.
   */
  Array<int> cnt_add_eq(neqcs);
  Array<int> cnt_add_cr(neqcs);

  cnt_add_eq = 0;
  cnt_add_cr = 0;

  for (auto eqc : Range(neqcs))
  {
    auto exrow = add_ce[eqc];
    int n_add_in_es = exrow.Size();

    for (auto j : Range(exrow))
    {
      auto & tup = exrow[j];
      auto eq0 = eqc_h.GetEQCOfID(tup[0]);
      auto v0 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[2], eq0);
      auto eq1 = (tup[1] != tup[0]) ? eqc_h.GetEQCOfID(tup[1]) : eq0;
      auto v1 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[3], eq1);
      auto memsa = c2fv[v0]; auto memsb = c2fv[v1];
      // cout << "count " << eqc << " " << j << ", tup " << tup << endl;
      auto ccross = tup[1] != tup[0];
      if (ccross) { cnt_add_cr[eqc]++; }
      else {cnt_add_eq[eqc]++; }
      // cout << " memsa: "; prow(memsa); cout << endl;
      // cout << " memsb: "; prow(memsb); cout << endl;
      for (auto x : memsa) /** find out if we have already counted this coarse edge (there might not be) **/
      {
        // cout << x << " con to "; prow(fecon.GetRowIndices(x)); cout << endl;
        if (!is_intersect_empty(fecon.GetRowIndices(x), memsb)) // there is SOME fine edge, which we have counted already
        {
          if (ccross)
            { cnt_cr[eqc]--; /* cout << "dec cnt_cr[" << eqc << "], now " << cnt_cr[eqc] << endl; */ }
          else
            { cnt_eq[eqc]--; /*cout << "dec cnt_eq[" << eqc << "], now " << cnt_eq[eqc] << endl; */ }
          break;
        }
      }
    }
  }

  /** Counters are now correct, we can set up the offset arrays **/
  Array<int> os_eq(1+neqcs);
  Array<int> os_cr(1+neqcs);
  
  os_eq[0] = 0;
  os_cr[0] = 0;

  for (auto k : Range(neqcs))
  {
    os_eq[1+k] = os_eq[k] + cnt_eq[k] + cnt_add_eq[k];
    os_cr[1+k] = os_cr[k] + cnt_cr[k] + cnt_add_cr[k];
  }

  size_t neq = os_eq.Last();
  size_t ncr = os_cr.Last();
  size_t nce = neq + ncr;

  // cout << endl << " nce neq ncr " << nce << " " << neq << " " << ncr << endl;
  // cout << "cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
  // cout << "cnt_add_eq " << endl; prow2(cnt_add_eq); cout << endl;
  // cout << "os_eq " << endl; prow2(os_eq); cout << endl;
  // cout << "cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
  // cout << "cnt_add_cr " << endl; prow2(cnt_add_cr); cout << endl;
  // cout << "os_cr " << endl; prow2(os_cr); cout << endl << endl;


  auto & cedges = cmesh.edges;

  cedges.SetSize(nce);

  for (auto & cedge : cedges)
    { cedge.id = -4242; cedge.v = { -42, -42 }; }

  mark.Clear();
  desig.Clear();

  /** we have to set add_edges manually - we might have some which are not mapped to locally **/
  if (neqcs > 1) {
    auto set_cedge = [&](auto id, auto& tup)
    {
      auto eq0 = eqc_h.GetEQCOfID(tup[0]);
      auto v0 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[2], eq0);
      auto eq1 = (tup[1] != tup[0]) ? eqc_h.GetEQCOfID(tup[1]) : eq0;
      auto v1 = CM.template MapENodeFromEQC<NT_VERTEX>(tup[3], eq1);
      cedges[id].id = id;
      cedges[id].v = (v0 < v1) ? IVec<2, int>({v0, v1}) : IVec<2, int>({v1, v0});
      // cout << " add edge id " << id << ", tup " << tup << " -> " << cedges[id] << endl;
      // cout << " map fedges ";
      FlatArray<int> memsa = c2fv[v0], memsb = c2fv[v1];
      for (auto x : memsa) {
        auto eids = fecon.GetRowValues(x);
        iterate_intersection(fecon.GetRowIndices(x), memsb,
                  [&](auto i, auto j) LAMBDA_INLINE {
              auto feid = int(eids[i]);
              emap[feid] = id; mark.SetBit(feid); // i think i dont ned mark because i check emap below ??
              // cout << feid << " ";
                  });
      }
      // cout << endl;
    };

    for (auto eqc : Range(size_t(1), neqcs))
    {
      auto exrow = add_ce[eqc];
      auto cid_eq = os_eq[eqc] + cnt_eq[eqc];

      // cout << " FILL ADD EX-EQ, " << cnt_add_eq[eqc] << " into " << os_eq[eqc] << " + " << cnt_eq[eqc] << " = " << os_eq[eqc] + cnt_eq[eqc] << endl;

      for (auto j : Range(cnt_add_eq[eqc]))
        { set_cedge(cid_eq++, exrow[j]); }
      
      auto cid_cr = neq + os_cr[eqc] + cnt_cr[eqc];
      
      // cout << " FILL ADD EX-CROSS, " << cnt_add_eq[eqc] << " into " << os_cr[eqc] << " + " << cnt_cr[eqc] << " = " << os_cr[eqc] + cnt_cr[eqc] << endl;

      for (auto j : Range(cnt_add_eq[eqc], int(exrow.Size())))
        { set_cedge(cid_cr++, exrow[j]); }
    }
  }

  // mark already cleared
  cnt_eq = 0;
  cnt_cr = 0;

  it_es_u(true, // god dammit, i HAVE to loop through ALL edges because i only alloc coarse edges after finished counts
    [&](const auto & e) LAMBDA_INLINE { return emap[e.id] == -1; }, // edges mapping to add_edges are already mapped
    [&](auto spec_id, auto feqc, auto ceqc, auto ccross,
        auto cv0, auto ceq0, auto cv1, auto ceq1, auto it_feids ) LAMBDA_INLINE
  {
      int loc_id = ccross ? cnt_cr[ceqc]++ : cnt_eq[ceqc]++;
      auto cid = (ccross ? (neq + os_cr[ceqc]) : os_eq[ceqc]) + loc_id;


      cedges[cid].id = cid;
      cedges[cid].v = IVec<2, int>({cv0, cv1});
      it_feids([&](auto feid) LAMBDA_INLINE
      {
        emap[feid] = cid;
      });


  }, true); // order matters here!

  // for (auto k : Range(FNE))
  // {
  //   auto const &fEdge = fedges[k];

  //   int cv0 = vmap[fEdge.v[0]];
  //   int cv1 = vmap[fEdge.v[1]];

  //   if ( (cv0 != cv1) && ( emap[k] == -1 ) && ( cv0 != -1 ) && ( cv1 != -1 ) )
  //   {
  //     cout << " FEDGE " << fEdge << " NO MAPPED, CVS = " << cv0 << " " << cv1 << endl;
  //   }
  // }


  // cout << endl << "FIN 1 cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
  // cout << "FIN 1 os_eq " << endl; prow2(os_eq); cout << endl;
  // cout << "FIN 1 cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
  // cout << "FIN 1 os_cr " << endl; prow2(os_cr); cout << endl << endl;

  for (auto k : Range(neqcs))
  {
    cnt_eq[k] += cnt_add_eq[k];
    cnt_cr[k] += cnt_add_cr[k];
  }

  // cout << endl << "FIN 2 cnt_eq     " << endl; prow2(cnt_eq); cout << endl;
  // cout << "FIN 2 os_eq " << endl; prow2(os_eq); cout << endl;
  // cout << "FIN 2 cnt_cr     " << endl; prow2(cnt_cr); cout << endl;
  // cout << "FIN 2 os_cr " << endl; prow2(os_cr); cout << endl << endl;

  mapped_NN[NT_EDGE] = nce;
  cmesh.nnodes[NT_EDGE] = nce;
  cmesh.disp_eqc[NT_EDGE] = std::move(os_eq);
  cmesh.disp_cross[NT_EDGE] = std::move(os_cr);
  cmesh.nnodes_glob[NT_EDGE] = 0;
  cmesh.nnodes_eqc[NT_EDGE] = cnt_eq;
  cmesh.nnodes_cross[NT_EDGE] = cnt_cr;
  cmesh.eqc_edges  = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_eqc[NT_EDGE].Data(), cmesh.edges.Data());
  cmesh.cross_edges = FlatTable<AMG_Node<NT_EDGE>> (neqcs, cmesh.disp_cross[NT_EDGE].Data(), cmesh.edges.Part(neq).Data());
  cmesh.nnodes_glob[NT_EDGE] = 0;
  for (auto eqc : Range(neqcs))
    if (eqc_h.IsMasterOfEQC(eqc))
{ cmesh.nnodes_glob[NT_EDGE] += cnt_eq[eqc] + cnt_cr[eqc]; }
  cmesh.nnodes_glob[NT_EDGE] = comm.AllReduce(cmesh.nnodes_glob[NT_EDGE], NG_MPI_SUM);

  cmesh.nnodes[NT_FACE] = 0;
  cmesh.nnodes[NT_CELL] = 0;

  /**
     cout << " fmesh NE " << M.nnodes[NT_EDGE] << " " << M.nnodes_glob[NT_EDGE] << endl << endl;
      cout << " fmesh disp e eqc " << endl; prow2(M.disp_eqc[NT_EDGE]); cout << endl << endl;
      cout << " fmesh disp e cr " << endl; prow2(M.disp_cross[NT_EDGE]); cout << endl << endl;
      cout << " fmesh eqc e " << endl; cout << M.eqc_edges << endl << endl;
      cout << " fmesh cr e " << endl; cout << M.cross_edges << endl << endl;
      cout << " cmesh NE " << cmesh.nnodes[NT_EDGE] << " " << cmesh.nnodes_glob[NT_EDGE] << endl << endl;
      cout << " cmesh eqc e " << endl; cout << cmesh.eqc_edges << endl << endl;
      cout << " cmesh cr e " << endl; cout << cmesh.cross_edges << endl << endl;
  **/

  /**
     Array<int> ce_cnt(cmesh.nnodes[NT_EDGE]); ce_cnt = 1;
      cout << "eqc_h: " << endl << eqc_h << endl;
      cout << " CM eqc_h: " << endl << *CM.GetEQCHierarchy() << endl;
      cout << " test allred edge data " << endl;
      CM.template AllreduceNodalData<NT_EDGE>(ce_cnt, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });
      cout << " have tested allred edge data! " << endl;
      cout << " ce_cnt: " << endl; prow2(ce_cnt); cout << endl;
      Array<IVec<2>> ce_cnt2(cmesh.nnodes[NT_EDGE]); ce_cnt2 = 0;
      int I = eqc_h.GetCommunicator().Rank() - 1;
      if ( (I == 0) || (I == 1) )
      for (auto k : Range(neqcs)) {
      for(const auto & edge : cmesh.template GetENodes<NT_EDGE>(k))
      { ce_cnt2[edge.id][I] = 1; }
      for(const auto & edge : cmesh.template GetCNodes<NT_EDGE>(k))
      { ce_cnt2[edge.id][I] = 2; }
      }
      CM.template AllreduceNodalData<NT_EDGE>(ce_cnt2, [&](auto tab) LAMBDA_INLINE { return sum_table(tab); });
      cout << " ce_cnt2: " << endl; prow2(ce_cnt2); cout << endl;
  **/

} // BaseAgglomerateCoarseMap::MapEdges

/** END BaseAgglomerateCoarseMap **/

} // namespace amg
