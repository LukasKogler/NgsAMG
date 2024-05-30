#ifndef FILE_BASE_MESH_IMPL_HPP
#define FILE_BASE_MESH_IMPL_HPP

#include <algorithm>

#include <base.hpp>

namespace amg
{

INLINE Timer<TTracing, TTiming>& GetSetNodesTimer (NODE_TYPE NT)
{
  static Timer tv("BlockTM - Set Vertices");
  static Timer te("BlockTM - Set Edges");
  static Timer tf("BlockTM - Set Faces");
  static Timer tc("BlockTM - Set Cells");
  switch(NT) {
    case(NT_VERTEX): { return tv; }
    case(NT_EDGE  ): { return te; }
    case(NT_FACE  ): { return tf; }
    case(NT_CELL  ): { return tc; }
    default: { throw Exception("Invalid NODE_TYPE in GetSetNodesTimer"); }
  }
}


template<class TGET, class TSET>
INLINE void BlockTM :: SetVs  (size_t annodes, TGET get_dps, TSET set_sort)
{
  RegionTimer rt(GetSetNodesTimer(NT_VERTEX));

  const auto & eqc_h = *this->eqc_h;
  auto neqcs = eqc_h.GetNEQCS();
  has_nodes[NT_VERTEX] = true;

  if (neqcs == 0) { // rank 0
    nnodes_glob[NT_VERTEX] = eqc_h.GetCommunicator().AllReduce(size_t(0), MPI_SUM);
    return;
  }

  size_t nv = annodes;
  this->nnodes[NT_VERTEX] = nv;

  this->verts.SetSize(nv);
  std::iota(verts.begin(), verts.end(), 0); // verts[k] = k

  Array<size_t> & disp(this->disp_eqc[NT_VERTEX]);
  disp.SetSize(neqcs + 1); disp = 0;

  auto lam_veq = [&](auto fun) LAMBDA_INLINE {
    for (auto vnr : Range(nv)) {
      auto dps = get_dps(vnr);
      auto eqc = eqc_h.FindEQCWithDPs(dps);
      fun(vnr,eqc);
    }
  };

  // eqc_verts table
  if (eqc_h.IsDummy()) {
    disp[0] = 0;
    disp[1] = nv;
  } else {
    lam_veq([&](auto vnr, auto eqc) LAMBDA_INLINE { disp[eqc + 1]++; });
    std::partial_sum(disp.begin(), disp.end(), disp.begin());
  }
  this->eqc_verts = FlatTable<AMG_Node<NT_VERTEX>>(neqcs, this->disp_eqc[NT_VERTEX].Data(), this->verts.Data());
  
  // vertex sorting
  if (eqc_h.IsDummy()) {
    for (auto vnr : Range(nv))
      { set_sort(vnr, vnr); }
  } else {
    Array<size_t> vcnt(neqcs + 1);
    vcnt = disp;  
    lam_veq([&](auto vnr, auto eqc) LAMBDA_INLINE { set_sort(vnr, vcnt[eqc]++); });
  }

  // // vertex -> eqc mapping; (I think we don't need this anymore!)
  // Array<int> v2eq(nv);
  // size_t cnt = 0;
  // for (auto k:Range(neqcs)) {
  //   auto d = disp[k];
  //   for (auto j:Range(disp[k+1]-disp[k]))
  //     { v2eq[cnt++] = d + j; }
  // }

  // # if in-eqc vertices
  this->nnodes_eqc[NT_VERTEX].SetSize(neqcs);
  for (auto eqc : Range(neqcs))
    { nnodes_eqc[NT_VERTEX][eqc] = disp_eqc[NT_VERTEX][eqc+1]-disp_eqc[NT_VERTEX][eqc]; }

  // there are no cross-EQC vertices!
  this->nnodes_cross[NT_VERTEX].SetSize(neqcs);
  this->nnodes_cross[NT_VERTEX] = 0;

  // # of global vertices
  size_t nv_master = 0;
  for (auto eqc : Range(neqcs)) {
    if (eqc_h.IsMasterOfEQC(eqc))
      { nv_master += eqc_verts[eqc].Size(); }
  }
  this->nnodes_glob[NT_VERTEX] = eqc_h.GetCommunicator().AllReduce(nv_master, MPI_SUM);
} // BlockTM::SetVs



template<ngfem::NODE_TYPE NT, class TGET, class TSET, typename T2> //  = typename std::enable_if<NT!=NT_VERTEX>::type>
INLINE void BlockTM :: SetNodes (size_t annodes, TGET get_node, TSET set_sort)
{
  RegionTimer rt(GetSetNodesTimer(NT));

  constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);

  const auto & eqc_h = *this->eqc_h;
  auto neqcs = eqc_h.GetNEQCS();

  this->has_nodes[NT] = true;

  if (neqcs == 0) {
    // empty mesh on rank 0 of a parallel mesh
    nnodes_glob[NT] = eqc_h.GetCommunicator().AllReduce(size_t(0), MPI_SUM);
    return;
  }

  auto get_node_array = [&]()->Array<AMG_Node<NT>>&{
    if constexpr(NT == NT_EDGE) { return this->edges; }
    if constexpr(NT == NT_FACE) { return this->faces; }
    if constexpr(NT == NT_CELL) { return this->cells; }
  };

  size_t tot_nnodes, tot_nnodes_eqc, tot_nnodes_cross;
  Array<AMG_Node<NT>> &nodes(get_node_array());
  Array<size_t> &node_disp_eqc(this->disp_eqc[NT]), &node_disp_cross(this->disp_cross[NT]);

  if (eqc_h.IsDummy()) { // local, so this is simple!
    nnodes_glob[NT] = ( nnodes[NT] = annodes );
    tot_nnodes = ( tot_nnodes_eqc = annodes );
    tot_nnodes_cross = 0;    
    node_disp_eqc.SetSize(2); node_disp_eqc = { 0, annodes };
    node_disp_cross.SetSize(2); node_disp_cross = { 0, 0 };
    nodes.SetSize(annodes);
    for (auto k : Range(annodes)) {
      nodes[k].id = k;
      set_sort(k, k);
      nodes[k].v = get_node(k);
    }
  }
  else { // parallel case - need to exchange and merge shared nodes
    // call fun_eqc on in-eqc and fun_cross on cross-eqc nodes
    auto lam_neq = [&](auto fun_eqc, auto fun_cross) LAMBDA_INLINE {
      constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v) / sizeof(AMG_Node<NT_VERTEX>) ;
      INT<NODE_SIZE,int> eqcs;
      for (auto node_num : Range(annodes)) {
        auto vs = get_node(node_num);
        auto eq_in = GetEQCOfNode<NT_VERTEX>(vs[0]);
        auto eq_cut = eq_in;
        for (auto i : Range(NODE_SIZE)) {
          auto eq_v = GetEQCOfNode<NT_VERTEX>(vs[i]);
          eqcs[i] = eqc_h.GetEQCID(eq_v);
          eq_in = (eq_in == eq_v) ? eq_in : -1;
          eq_cut = (eq_cut == eq_v) ? eq_cut : eqc_h.GetCommonEQC(eq_cut, eq_v);
        }
        AMG_CNode<NT> node = {{vs}, eqcs};
        if (eq_in != size_t(-1))
          { fun_eqc(node_num, node, eq_in); }
        else
          { fun_cross(node_num, node, eq_cut); }
      }
    };

    // create tentative ex-nodes
    TableCreator<AMG_CNode<NT>> cten(neqcs);
    Table<AMG_CNode<NT>> tent_ex_nodes;
    node_disp_eqc.SetSize(neqcs + 1); node_disp_cross.SetSize(neqcs + 1);

    constexpr int NODE_SIZE = sizeof(AMG_CNode<NT>::v)/sizeof(AMG_Node<NT_VERTEX>);
    auto add_node_eqc = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
      if (eqc == 0)
        { node_disp_eqc[1]++; }
      else {
        for(auto i : Range(NODE_SIZE))
          { node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]); }
        cten.Add(eqc, node);
      }
    };
    auto add_node_cross = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
      if (eqc == 0)
        { node_disp_cross[1]++; }
      else {
        for(auto i : Range(NODE_SIZE))
          { node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]); }
        cten.Add(eqc, node);
      }
    };

    for(; !cten.Done(); cten++) {
      node_disp_eqc = 0;
      node_disp_cross = 0;
      lam_neq(add_node_eqc, add_node_cross);
    }
    tent_ex_nodes = cten.MoveTable();

    // merge ex-nodes
    // Table<AMG_CNode<NT>> tent_ex_nodes = cten.MoveTable();
    auto smaller = [&](const auto & a, const auto & b) -> bool {
      // Note: only works for NT_EDGE
      bool isina = (a.eqc[0] == a.eqc[1]); // is a in-eqc?
      bool isinb = (b.eqc[0] == b.eqc[1]); // is b in-eqc?
      if      ( isina && !isinb)     { return true;}
      else if (!isina &&  isinb)     { return false; }
      else if ( isina &&  isinb)      { return a.v < b.v; }
      else if (a.eqc[0] < b.eqc[0]) { return true;  }
      else if (a.eqc[0] > b.eqc[0]) { return false; }
      else if (a.eqc[1] < b.eqc[1]) { return true;  }
      else if (a.eqc[1] > b.eqc[1]) { return false; }
      else                          { return a.v < b.v; }
    };
    for (auto k : Range(size_t(1), neqcs))
      { QuickSort(tent_ex_nodes[k], smaller); }

    // cout << "tent_ex_nodes : " << endl << tent_ex_nodes << endl;
    Table<AMG_CNode<NT>> ex_nodes;
    if (!eqc_h.IsDummy()) {
      auto merge_with_smaller = [&](auto & input) LAMBDA_INLINE { return merge_arrays(input, smaller); };
      ex_nodes = ReduceTable<AMG_CNode<NT>,AMG_CNode<NT>> (tent_ex_nodes, this->eqc_h, merge_with_smaller);
    } else {
      ex_nodes = std::move(tent_ex_nodes);
    }

    // cout << "ex_nodes : " << endl << ex_nodes << endl;
    tot_nnodes_eqc = node_disp_eqc[1];
    tot_nnodes_cross = node_disp_cross[1];
    tot_nnodes = tot_nnodes_eqc + tot_nnodes_cross;

    // count in/cross nodes in exchange-nodes, add to offsets/nnodes
    for (auto k : Range(size_t(1), neqcs)) {
      auto isin = [](auto X) {
        auto eq = X.eqc[0];
        for (int k = 1; k < NODE_SIZE; k++) {
          if (X.eqc[k] != eq)
            { return false; }
        }
        return true;
      };
      auto row = ex_nodes[k];
      size_t n_in = 0;
      auto row_s = row.Size();
      while(n_in < row_s) {
        if(!isin(row[n_in]))
          { break; }
        n_in++;
      }
      tot_nnodes += row_s;
      node_disp_eqc[k + 1] = node_disp_eqc[k] + n_in;
      tot_nnodes_eqc += n_in;
      auto n_cross = row_s - n_in;
      node_disp_cross[k + 1] = node_disp_cross[k] + n_cross;
      tot_nnodes_cross += n_cross;
    }

    // final # of nodes
    this->nnodes[NT] = tot_nnodes;

    // final node array
    // Array<AMG_Node<NT>> &nodes(get_node_array()); nodes.SetSize(tot_nnodes);
    nodes.SetSize(tot_nnodes);
    
    // cross-nodes come after in-nodes offset, whe need the offsets from the START now
    for (auto k : Range(neqcs + 1))
      { node_disp_cross[k] += tot_nnodes_eqc; }

    // write exchange nodes into node array
    Array<size_t> nin(neqcs);
    nin[0] = -1; // just to ensure this blows up if we access here
    for (auto k : Range(size_t(1), neqcs)) {
      auto row = ex_nodes[k];
      auto n_in = node_disp_eqc[k + 1] - node_disp_eqc[k];
      nin[k] = n_in;
      auto cr = 0;
      auto os_in = node_disp_eqc[k];
      for (auto j : Range(n_in)) {
        auto id = os_in + j;
        auto & node = nodes[id];
        node.id = id;
        const auto & exn_v = row[cr].v;
        for (auto l : Range(NODE_SIZE))
          { node.v[l] = MapENodeFromEQC<NT_VERTEX>(exn_v[l], k); }
        cr++;
      }
      auto n_cross = row.Size() - n_in;
      auto os_cross = node_disp_cross[k];
      if (n_cross > 0) {
        for (auto j : Range(n_cross)) {
          auto id = os_cross + j;
          auto & node = nodes[id];
          node.id = id;
          auto & exn_v  = row[cr].v;
          auto & exn_eq = row[cr].eqc;
          for (auto l : Range(NODE_SIZE))
            { node.v[l] = MapENodeFromEQC<NT_VERTEX>(exn_v[l], eqc_h.GetEQCOfID(exn_eq[l])); }
          cr++;
        }
      }
    }

    // write+map local nodes and map ex-nodes!
    size_t cnt0 = 0;
    auto add_node_eqc2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
      if (eqc == 0) {
        // set_sort(node_num,id);
        amg_nts::id_type id = cnt0;
        nodes[id] = {{node.v}, id};
        set_sort(node_num, id);
        cnt0++;
      }
      else {
        for(auto i : Range(NODE_SIZE))
          { node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]); }
        auto pos = ex_nodes[eqc].Pos(node);
        // auto loc_id = pos-nin[eqc]; // WHAT???? no...
        auto loc_id = pos;
        set_sort(node_num, MapENodeFromEQC<NT>(loc_id, eqc));
      }
    };

    size_t cnt0c = tot_nnodes_eqc;
    auto add_node_cross2 = [&](auto node_num, AMG_CNode<NT>& node, auto eqc) LAMBDA_INLINE {
      if (eqc == 0) {
        // set_sort(node_num,id);
        amg_nts::id_type id = cnt0c;
        nodes[id] = {{node.v}, id};
        set_sort(node_num, id);
        cnt0c++;
      }
      else {
        for(auto i : Range(NODE_SIZE))
          { node.v[i] = MapNodeToEQC<NT_VERTEX>(node.v[i]); }
        auto pos = ex_nodes[eqc].Pos(node);
        auto loc_id = pos - nin[eqc];
        // set_sort(node_num,MapCNodeFromEQC<NT>(loc_id, eqc));
        set_sort(node_num, loc_id + node_disp_cross[eqc]); // ndc has tot_nnodes_eqc added ATM!
      }
    };

    lam_neq(add_node_eqc2, add_node_cross2);

    // reset the cross-offsets again to relative to the start OF THE CROSS-NDOES part!
    for (auto k : Range(neqcs + 1))
      { node_disp_cross[k] -= tot_nnodes_eqc; }    

  } // parallel case

  // 
  auto writeit = [neqcs, tot_nnodes, tot_nnodes_eqc](auto & _arr, auto & _disp_eqc, auto & _tab_eqc,
                                                     auto & _disp_cross, auto & _tab_cross) LAMBDA_INLINE
  {
    if (_disp_eqc.Last() > _disp_eqc[0])
      { _tab_eqc = FlatTable<AMG_Node<NT>>(neqcs, _disp_eqc.Data(), &(_arr[0])); }
    else
      { _tab_eqc = FlatTable<AMG_Node<NT>>(neqcs, _disp_eqc.Data(), nullptr); }
    if (tot_nnodes > tot_nnodes_eqc)
      { _tab_cross = FlatTable<AMG_Node<NT>>(neqcs, _disp_cross.Data(), &(_arr[tot_nnodes_eqc])); }
    else
      { _tab_cross = FlatTable<AMG_Node<NT>>(neqcs, _disp_cross.Data(), nullptr); }
  };

  if constexpr(NT==NT_EDGE)
    { writeit(edges, disp_eqc[NT], eqc_edges, disp_cross[NT], cross_edges); }
  else if constexpr(NT==NT_FACE)
    { writeit(faces, disp_eqc[NT], eqc_faces, disp_cross[NT], cross_faces); }

  // eqc-wiser NNODES
  nnodes_eqc[NT].SetSize(neqcs);
  nnodes_cross[NT].SetSize(neqcs);
  for (auto eqc : Range(neqcs)) {
    nnodes_eqc[NT][eqc]   = disp_eqc[NT][eqc + 1]   - disp_eqc[NT][eqc];
    nnodes_cross[NT][eqc] = disp_cross[NT][eqc + 1] - disp_cross[NT][eqc];
  }

  // global NNODES
  size_t nn_master = 0;
  for (auto eqc : Range(neqcs))
    if (eqc_h.IsMasterOfEQC(eqc))
      { nn_master += GetENN<NT>(eqc) + GetCNN<NT>(eqc); }
  nnodes_glob[NT] = eqc_h.GetCommunicator().AllReduce(nn_master, MPI_SUM);
}// SetNodes


template<ngfem::NODE_TYPE NT>
INLINE void BlockTM :: SetEQOS (Array<size_t> && _adisp)
{
  disp_eqc[NT] = std::move(_adisp);
  auto neqcs = eqc_h->GetNEQCS();
  switch(NT) {
    case(NT_VERTEX) : { eqc_verts = MakeFT<AMG_Node<NT_VERTEX>> (neqcs, disp_eqc[NT_VERTEX], verts, 0); break; }
    case(NT_EDGE)   : { eqc_edges = MakeFT<AMG_Node<NT_EDGE>>   (neqcs, disp_eqc[NT_EDGE], edges, 0); break; }
    case(NT_FACE)   : { eqc_faces = MakeFT<AMG_Node<NT_FACE>>   (neqcs, disp_eqc[NT_FACE], faces, 0); break; }
  }
  nnodes_eqc[NT].SetSize(neqcs);
  for (auto k : Range(nnodes_eqc[NT]))
    { nnodes_eqc[NT][k] = disp_eqc[NT][k + 1] - disp_eqc[NT][k]; }
} // BlockTM::SetEQOS


} // namespace amg

#endif //  FILE_BASE_MESH_IMPL_HPP