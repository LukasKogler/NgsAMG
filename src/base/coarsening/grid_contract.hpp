#ifndef FILE_CONTRACT_MAP_HPP
#define FILE_CONTRACT_MAP_HPP

#include <base.hpp>
#include <utils_tuples.hpp>

#include "grid_map.hpp"

#include <utils_numeric_types.hpp>
#include <reducetable.hpp>

namespace amg
{

// dont need to expose that I think??
Table<int> PartitionProcsMETIS (BlockTM & mesh, int nparts, bool sep_p0 = true);


INLINE Timer<TTracing, TTiming>& timer_hack_gccm1 () { static Timer t("GridContractMap::MapNodeData"); return t; }
template<NODE_TYPE NT> INLINE Timer<TTracing, TTiming>& timer_hack_gccm2 () {
  if constexpr(NT==NT_VERTEX) { static Timer t("GridContractMap::MapNodeData, V"); return t; }
  if constexpr(NT==NT_EDGE)   { static Timer t("GridContractMap::MapNodeData, E"); return t; }
  if constexpr(NT==NT_FACE)   { static Timer t("GridContractMap::MapNodeData, F"); return t; }
  if constexpr(NT==NT_CELL)   { static Timer t("GridContractMap::MapNodeData, C"); return t; }
}


class GridContractMap : public BaseGridMapStep
{
  // static_assert(std::is_base_of<BlockTM, TMESH>::value, "GridContractMap can only be constructed for Meshes that inherit from BlockTM!");
public:
  GridContractMap (Table<int> && groups, shared_ptr<BlockTM> mesh, bool oriented = false);

  virtual ~GridContractMap() = default;

  // template<NODE_TYPE NT, typename T>
  // void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const;

  INLINE bool IsMaster () const { return is_gm; }
  INLINE FlatArray<int> GetGroup () const { return my_group; }
  template<NODE_TYPE NT> INLINE FlatArray<amg_nts::id_type> GetNodeMap (int member) const { return node_maps[NT][member]; }
  template<NODE_TYPE NT> INLINE FlatTable<amg_nts::id_type> GetNodeMaps () const { return node_maps[NT]; }

  INLINE shared_ptr<EQCHierarchy> GetEQCHierarchy () const { return eqc_h; }
  INLINE shared_ptr<EQCHierarchy> GetMappedEQCHierarchy () const { return c_eqc_h; }

  INLINE FlatArray<int> GetProcMap () const { return proc_map; }

  template<NODE_TYPE NT> INLINE FlatArray<BitArray> GetFlipNodes () const { return flip_nodes[NT]; }
  template<NODE_TYPE NT> INLINE const BitArray& GetFlipNodes (int member) const { return flip_nodes[NT][member]; }
  
  template<NODE_TYPE NT> INLINE size_t GetMappedNN () const { return (is_gm) ? mapped_NN[NT] : 0; }

  /**
   * A couple of notes here:
   *    i) BuildNodeMaps actually:
   *         - computes the node-maps
   *         - sets up the contracted mesh
   *         - fills the contracted data by calling FillContractedMesh
   *       So BuildNodeMaps should actually be BuildMappedMesh and we should call BuildCEQH from there 
   *   ii) We are doing the GetMappedMesh overload instead of calling BuildMappedMesh from any constructor
   *       because when we are overloading FillContractedMesh in a derived class that would not be called
   *       correctly.
   * TOOD: It would probably be cleaner to add a "Finalize" method to the grid-maps.
  */

  shared_ptr<TopologicMesh> GetMappedMesh () const override
  {
    const_cast<GridContractMap&>(*this).BuildMappedMesh();
    return mapped_mesh;
  }

  virtual void BuildMappedMesh()
  {
    if (cmesh_constructed)
      { return; }
    static Timer t("GridContractMap MapMesh");
    RegionTimer rt(t);

    /**
     * Note: we have to set this BEFORE calling into BuildCEQCH/BuildNodeMaps,
     *       as we call GetMappedMesh there too.
     *       That is why a finalize method would be cleaner!
     */
    cmesh_constructed = true;

    BuildCEQCH();
    BuildNodeMaps();
  }

  virtual void PrintTo (std::ostream & os, string prefix = "") const override;

protected:

  virtual shared_ptr<BlockTM> AllocateContractedMesh (shared_ptr<EQCHierarchy> _cEQCH) = 0;
  virtual void FillContractedMesh() = 0;

  using BaseGridMapStep::mesh, BaseGridMapStep::mapped_mesh;
  
  shared_ptr<EQCHierarchy> eqc_h = nullptr;
  shared_ptr<EQCHierarchy> c_eqc_h = nullptr;

  bool cmesh_constructed = false;

  /** proc-maps **/
  bool is_gm = true; // is group master
  Array<int> proc_map;
  Table<int> groups;
  FlatArray<int> my_group;

  /** EQC-maps **/
  Table<int> map_om; // (maps eqcs) orig --> merged
  Array<int> map_mc; // (maps eqcs) merged --> contr
  Table<int> map_oc; // (maps eqcs) orig --> contr
  // (I think) un-mapped members of merged eqcs
  Table<int> mmems; // merged (pre-contracted)\members
  
  /** node-maps **/
  bool oriented_maps = false; // keeps track of chenge in orientation of nodes
  size_t mapped_NN[4];
  Array<Table<amg_nts::id_type>> node_maps;
  Array<Table<amg_nts::id_type>> annoy_nodes;
  Array<Array<BitArray>> flip_nodes;

  void BuildCEQCH ();
  void BuildNodeMaps ();

public:

  template<NODE_TYPE NT, typename T>
  INLINE void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const
  {

    // cout << "MAPNODEDATA, NT " << NT << ", stat " << stat << endl;
    // cout << "orig data: " << endl << "___" << endl;
    // for (auto k : Range(data.Size())) {
    // 	cout << "orig data " << k;
    // 	if (is_gm) cout << ", maps to " << node_maps[NT][0][k];
    // 	else cout << ", maps to NOT MASTER";
    // 	cout << ": " << endl << data[k] << endl << "___" << endl;
    // }
    // cout << endl;

    RegionTimer rt1(timer_hack_gccm1());
    RegionTimer rt2(timer_hack_gccm2<NT>());
    auto comm = eqc_h->GetCommunicator();
    int master = my_group[0];
    if(!is_gm) {
      // cout << "send " << data.Size() << " times " << typeid(T).name() << " to " << master << endl;
      comm.Send(data, master, NG_MPI_TAG_AMG);
      return;
    }

    // cout << "send " << data.Size() << " times " << typeid(T).name() << " to myself, LOL" << endl;
    Array<T> &out(*cdata);
    out.SetSize(mapped_NN[NT]);
    // if (stat==DISTRIBUTED) out = T(0);
    out = T(0); // TODO: above is better...
    auto & maps = node_maps[NT];
    Array<T> buf;
    auto it_vals = [&] (auto & map, auto & buf) LAMBDA_INLINE {
      if(stat==CUMULATED)
        for(auto k:Range(map.Size()))
          { out[map[k]] = buf[k]; }
      else
        for(auto k:Range(map.Size()))
          { out[map[k]] += buf[k]; }
    };

    if (my_group.Size() > 1) { // DIST DATA
      for(auto pn : Range(size_t(1), my_group.Size())) {
        auto p = my_group[pn];
        auto map = maps[pn];
        // cout << "get " << buf.Size() << " times " << typeid(T).name() << " from " << p << endl;
        comm.Recv(buf, p, NG_MPI_TAG_AMG);
        // cout << "got " << buf.Size() << " times " << typeid(T).name() << " from " << p << endl;
        // cout << "map size is: " << map.Size() << endl;
        // cout << "map : "; prow2(map); cout << endl;
        it_vals(map, buf);
      }
    }
    it_vals(maps[0], data); // LOC DATA
    
    // annoy data is not scattered correctly now for edge-based data!!
    auto & anodes = annoy_nodes[NT];

    // cout << "1, contr data: " << endl << "___" << endl;
    // for (auto k : Range(out.Size())) {
    // 	cout << "1, contr data " << k << ": " << endl << out[k] << endl << "___" << endl;
    // }
    // cout << endl;
    // cout << "MND2" << endl;

    if(!anodes.Size())
      { return; } // no annoying nodes for this type!!
    
    auto cneqcs = c_eqc_h->GetNEQCS();
    Array<size_t> sz(cneqcs);
    for(auto k:Range(cneqcs))
      { sz[k] = anodes[k].Size(); }

    Table<T> adata(sz);

    for(auto k:Range(cneqcs)) {
      for(auto j:Range(anodes[k].Size()))
        { adata[k][j] = out[anodes[k][j]]; }
    }

    // for (auto k : Range(adata.Size())) {
    // 	for (auto j : Range(adata[k].Size())) {
    // 	  cout << "anode " << anodes[k][j] << "(is " << k << " " << j << "):" << endl;
    // 	  cout << adata[k][j];
    // 	  cout << endl << "__" << endl;
    // 	}
    // }

    auto radata = ReduceTable<T,T>(adata, c_eqc_h, [&](auto & t) {
      // cout << "lambda for t: " << endl;
      // for(auto k:Range(t.Size())) { cout<<k<<"(" << t[k].Size() << "):  ";prow(t[k]);cout<<endl; }
      Array<T> out;
      if(!t.Size()) return out;
      if(!t[0].Size()) return out;
      out.SetSize(t[0].Size());
      out = t[0];
      if(t.Size()==1) return out;
      for(auto k:Range((size_t)1, t.Size())) {
        for(auto j:Range(t[k].Size())) {
          // if( (stat==DISTRIBUTED) || (t[k][j]==zero) ) { // TODO: why??
          if( stat == DISTRIBUTED ) { 
            // hacked_add(out[j],t[k][j]);
      out[j] += t[k][j];
          }
          else {
      if (is_zero(out[j]) && !is_zero(t[k][j]))
        out[j] = t[k][j];
          }
        }
      }
      return out;
    });

    for(auto k:Range(cneqcs)) {
      for(auto j:Range(anodes[k].Size()))
        { out[anodes[k][j]] = radata[k][j]; }
    }

    // cout << "2, contr data: " << endl << "___" << endl;
    // for (auto k : Range(out.Size())) {
    // 	cout << "2, contr data " << k << ": " << endl << out[k] << endl << "___" << endl;
    // }
    // cout << endl;

    return;
  } // GridContractMap::MapNodeData

  template<NODE_TYPE NT, typename T>
  INLINE void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat) const
  {
    MapNodeData<NT, T>(data, stat, nullptr);
  } // GridContractMap::MapNodeData

}; // class GridContractMap


template<class TMESH>
class AlgContractMap : public GridContractMap
{
public:
  AlgContractMap (Table<int> && groups, shared_ptr<BlockTM> mesh, bool oriented = false)
    : GridContractMap(std::move(groups), mesh, oriented)
  { ; }

  virtual ~AlgContractMap() = default;

protected:
  virtual shared_ptr<BlockTM> AllocateContractedMesh (shared_ptr<EQCHierarchy> cEQCH) override
  {
    return make_shared<TMESH>(cEQCH);
  }

  virtual void FillContractedMesh() override
  {
    auto f_mesh = dynamic_pointer_cast<TMESH>(GetMesh());

    assert(f_mesh != nullptr);

    if (IsMaster())
    {
      auto c_mesh = my_dynamic_pointer_cast<TMESH>(GetMappedMesh(),
                      "AlgContractMap::AllocateContractedMesh - CMESH!");

      c_mesh->AllocateAttachedData();

      // auto mapLam = [&](auto &fAD, auto &cAD) {
      //   auto fstat = fAD.GetParallelStatus();
      //   auto &cdata = cAD.GetModData();
      //   this->template MapNodeData<decltype(fAD)::TNODE>(fAD.Data(), fstat, &cdata);
      //   cAD.SetParallelStatus(fstat);
      // };

      ApplyComponentWise([&](auto pFAD, auto pCAD) {
        auto &fAD = *pFAD;
        auto fstat = fAD.GetParallelStatus();
        auto &cdata = pCAD->GetModData();
        static constexpr NODE_TYPE NT = std::remove_reference<decltype(fAD)>::type::TNODE;
        this->template MapNodeData<NT>(fAD.Data(), fstat, &cdata);
        pCAD->SetParallelStatus(fstat);        
      },
      f_mesh->Data(),
      c_mesh->ModData());

      // std::apply( [&](auto& ...x, auto& ...y) {
      //   (..., mapLam(x, y));
      // },
      // f_mesh->AttachedData(),
      // c_mesh->AttachedData());
    }
    else
    {
      ApplyComponentWise([&](auto pFAD) {
        auto &fAD = *pFAD;
        auto fstat = fAD.GetParallelStatus();
        static constexpr NODE_TYPE NT = std::remove_reference<decltype(fAD)>::type::TNODE;
        this->template MapNodeData<NT>(fAD.Data(), fstat);
      },
      f_mesh->Data());
    }
  }

}; // class AlgContractMap


// template<NODE_TYPE NT, typename T>
// INLINE void GridContractMap :: MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, Array<T> * cdata) const
 

} // namespace amg

#endif // FILE_CONTRACT_MAP_HPP
