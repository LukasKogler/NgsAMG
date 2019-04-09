#ifndef FILE_AMGCTR
#define FILE_AMGCTR

namespace amg
{

  // template<class TV>
  // class CtrMap : public BaseDOFMapStep
  // {

  // };

  Table<int> PartitionProcsMETIS (BlockTM & mesh, int nparts);
  
  template<class TMESH>
  class GridContractMap : public GridMapStep<TMESH>
  {
  public:
    GridContractMap (Table<int> && groups, shared_ptr<TMESH> mesh);

    // template<class TV>
    // shared_ptr<CtrMap<TV>> DMS (shared_ptr<ParallelDofs> fpd)

    // template<NODE_TYPE NT, typename T>
    // void MapNodeData (FlatArray<T> data, PARALLEL_STATUS stat, FlatArray<T> * cdata);
    
  protected:
    // using GridMapStep<TMESH>::mesh, GridMapStep<TMESH>::mapped_mesh;
    
    shared_ptr<EQCHierarchy> eqc_h = nullptr;
    shared_ptr<EQCHierarchy> c_eqc_h = nullptr;

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
    size_t mapped_NN[4];
    Array<Table<size_t>> node_maps;
    Array<Table<size_t>> annoy_nodes;

    void BuildCEQCH ();
    void BuildNodeMaps ();
  };

} // namespace amg

#endif
