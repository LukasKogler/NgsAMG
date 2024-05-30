#ifndef FILE_AMG_FACTORY_NODAL_IMPL_HPP
#define FILE_AMG_FACTORY_NODAL_IMPL_HPP

#include "nodal_factory.hpp"
#include "universal_dofs.hpp"
namespace amg
{

  /** NodalAMGFactory **/

  template<NODE_TYPE NT, class TMESH, int BS>
  NodalAMGFactory<NT, TMESH, BS> :: NodalAMGFactory (shared_ptr<Options> _opts)
    : BaseAMGFactory(_opts)
  {
    ;
  } // NodalAMGFactory(..)


  template<NODE_TYPE NT, class TMESH, int BS>
  size_t NodalAMGFactory<NT, TMESH, BS> :: ComputeMeshMeasure (const TopologicMesh & m) const
  {
    return m.template GetNNGlobal<NT>();
  } // NodalAMGFactory::ComputeMeshMeasure


  template<NODE_TYPE NT, class TMESH, int BS>
  double NodalAMGFactory<NT, TMESH, BS> :: ComputeLocFrac (const TopologicMesh & am) const
  {
    auto btm_ptr = dynamic_cast<const BlockTM*>(&am);
    if (btm_ptr == nullptr)
      { return 1.0; }
    else {
      const auto & m (*btm_ptr);
      auto nng = m.template GetNNGlobal<NT_VERTEX>();
      size_t nnloc = (m.GetEQCHierarchy()->GetNEQCS() > 1) ? m.template GetENN<NT_VERTEX>(0) : 0;
      auto nnlocg = m.GetEQCHierarchy()->GetCommunicator().AllReduce(nnloc, NG_MPI_SUM);
      return double(nnlocg) / nng;
    }
  } // NodalAMGFactory::ComputeLocFrac


  template<NODE_TYPE NT, class TMESH, int BS>
  UniversalDofs
  NodalAMGFactory<NT, TMESH, BS> ::
  BuildUDofs (LevelCapsule const &cap) const
  {
    const BlockTM & mesh = *my_dynamic_pointer_cast<BlockTM>(cap.mesh, "NodalAMGFactory::BuildUDofs");

    const auto & eqc_h = *mesh.GetEQCHierarchy();

    if (eqc_h.IsDummy())
    {
      return UniversalDofs(mesh.GetNN<NT>(), BS);
    }
    else
    {
      TableCreator<int> cdps(mesh.GetNN<NT>());

      for (; !cdps.Done(); cdps++) { // also goes through cross nodes
        mesh.ApplyEQ2<NT>([&](auto eqc, auto nodes) {
          auto dps = eqc_h.GetDistantProcs(eqc);
          for (const auto & node : nodes)
            if constexpr(NT == NT_VERTEX)
              { cdps.Add(node, dps); }
            else
              { cdps.Add(node.id, dps); }
        }, false); // not master only
      }

      auto tab = cdps.MoveTable();
      // auto pds = make_shared<ParallelDofs> (eqc_h.GetCommunicator(), std::move(tab) /* cdps.MoveTable() */, BS, false, eqc_h.IsRankZeroIdle());
      auto pds = CreateParallelDOFs(eqc_h.GetCommunicator(), std::move(tab), BS, false, eqc_h.IsRankZeroIdle());

      return UniversalDofs(pds);
    }
  } // NodalAMGFactory::BuildDofs


  template<NODE_TYPE NT, class TMESH, int BS>
  size_t NodalAMGFactory<NT, TMESH, BS> :: ComputeGoal (const shared_ptr<AMGLevel> & f_lev, State & state)
  {
    auto &O(*options);

    // TODO: we have to respect enable_multistep/interleave here

    auto fmesh = f_lev->cap->mesh;

    auto curr_meas = ComputeMeshMeasure(*fmesh);

    size_t goal_meas = (curr_meas == 0) ? 0 : 1;

    /** static coarsening ratio **/
    if (O.use_static_crs) {
      double af = ( (f_lev->level == 0) && (O.first_aaf != -1) ) ?
	O.first_aaf : ( pow(O.aaf_scale, f_lev->level - ( (O.first_aaf == -1) ? 0 : 1) ) * O.aaf );
      goal_meas = max( size_t(min(af, 0.9) * curr_meas), max(O.max_meas, size_t(1)));
    }

    /** dynamic coarsening ratio **/
    // if (O.use_dyn_crs) {
    // }
    // dynamic coarsening ratio has to be handled by derived classes
    // TODO: There should also be a ComputeGoal in VertexAMGFactory which can take advantage of this!
    /** We want to find the right agglomerate size, as a heutristic take 1/(1+avg number of strong neighbours) **/
    // size_t curr_ne = fmesh->template GetNNGlobal<NT_EDGE>();
    // size_t curr_nv = fmesh->template GetNNGlobal<NT_VERTEX>();
    // double edge_per_v = 2 * double(curr_ne) / double(curr_nv);
    // if (O.enable_dyn_aaf) {
    //   const double MIN_ECW = coarsen_opts->min_ecw;
    //   const auto& ecw = coarsen_opts->ecw;
    //   size_t n_s_e = 0;
    //   cmesh->template Apply<NT_EDGE>([&](const auto & e) { if (ecw[e.id] > MIN_ECW) { n_s_e++; } }, true);
    //   n_s_e = cmesh->GetEQCHierarchy()->GetCommunicator().AllReduce(n_s_e, NG_MPI_SUM);
    //   double s_e_per_v = 2 * double(n_s_e) / double(cmesh->template GetNNGlobal<NT_VERTEX>());
    //   double dynamic_goal_fac = 1.0 / ( 1 + s_e_per_v );
    //   goal_meas = max( size_t(min2(0.5, dynamic_goal_fac) * curr_meas), max(O.max_meas, size_t(1)));
    // } // dyn_aaf

    return goal_meas;
  } // NodalAMGFactory::ComputeGoal


  template<NODE_TYPE NT, class TMESH, int BS>
  shared_ptr<BaseGridMapStep> NodalAMGFactory<NT, TMESH, BS> :: BuildContractMap (double factor, shared_ptr<TopologicMesh> mesh, shared_ptr<BaseAMGFactory::LevelCapsule> & mapped_cap) const
  {
    static Timer t("BuildContractMap");
    RegionTimer rt(t);

    auto &O(*options);

    if (mesh == nullptr)
      { throw Exception("BuildContractMap needs a mesh!"); }

    auto m = my_dynamic_pointer_cast<TMESH>(mesh, "NodalAMGFactory::BuildContractMap - mesh");

    if (O.log_level >= Options::LOG_LEVEL::DBG)
    {
      ofstream out ("alg_mesh_preCtr_rk_" + to_string(mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
        + to_string(mapped_cap->baselevel) + ".out");
      out << *mesh << endl;
    }

    bool const sepPZero = mesh->GetEQCHierarchy()->IsRankZeroIdle();

    int const effSeqSize = sepPZero ? 2 : 1;

    // at least 2 groups - dont send everything from 1 to 0 for no reason
    int n_groups = (factor == -1) ? effSeqSize : max2(int(2), int(1 + std::round( (mesh->GetEQCHierarchy()->GetCommunicator().Size()-1) * factor)));

    Table<int> groups = PartitionProcsMETIS (*m, n_groups, sepPZero);

    // auto cm = make_shared<GridContractMap>(std::move(groups), m);
    auto cm = AllocateContractMap(std::move(groups), m);

    mapped_cap->mesh = cm->GetMappedMesh();
    mapped_cap->free_nodes = nullptr;
    mapped_cap->eqc_h = cm->IsMaster() ? mapped_cap->mesh->GetEQCHierarchy() : nullptr;
    mapped_cap->uDofs = UniversalDofs(); // is set in BuildDOFContractMap // probably unnecessary to set it explicitly


    if (O.log_level >= Options::LOG_LEVEL::DBG)
    {
      if (mapped_cap->mesh != nullptr)
      {
        ofstream outMap ("alg_mesh_ctrMap_rk_" + to_string(mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
          + to_string(mapped_cap->baselevel) + ".out");
        outMap << *cm << endl;

        ofstream out ("alg_mesh_postCtr_rk_" + to_string(mesh->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_"
          + to_string(mapped_cap->baselevel) + ".out");
        out << *mapped_cap->mesh << endl;
      }
    }

    return cm;
  } // NodalAMGFactory::BuildContractMap


  template<NODE_TYPE NT, class TMESH, int BS>
  shared_ptr<BaseDOFMapStep>
  NodalAMGFactory<NT, TMESH, BS> ::
  BuildContractDOFMap (shared_ptr<BaseGridMapStep> cmap,
                       shared_ptr<LevelCapsule> &fCap,
                       shared_ptr<LevelCapsule> &mapped_cap) const
  {
    static Timer t("BuildContractDOFMap"); RegionTimer rt(t);

    if (cmap == nullptr)
      { throw Exception("BuildContractDOFMap needs a mesh!"); }
    auto cm = my_dynamic_pointer_cast<GridContractMap>(cmap, "BuildContractDOFMap - grid-map");

    auto fg = cm->GetGroup();
    Array<int> group(fg.Size()); group = fg;
    Table<int> dof_maps;
    if (cm->IsMaster()) {
      // const TMESH& cmesh(*static_cast<const TMESH&>(*grid_step->GetMappedMesh()));
      shared_ptr<TMESH> cmesh = static_pointer_cast<TMESH>(cm->GetMappedMesh());
      mapped_cap->uDofs = BuildUDofs(*mapped_cap);
      Array<int> perow (group.Size()); perow = 0;
      for (auto k : Range(group.Size()))\
        { perow[k] = cm->template GetNodeMap<NT>(k).Size(); } 
      dof_maps = Table<int>(perow);
      for (auto k : Range(group.Size()))
        { dof_maps[k] = cm->template GetNodeMap<NT>(k); }
    }
    else
      { mapped_cap->uDofs = UniversalDofs(); }

    auto ctr_map = make_shared<CtrMap<typename strip_vec<Vec<BS, double>>::type>> (fCap->uDofs,
                                                                                   mapped_cap->uDofs,
                                                                                   std::move(group),
                                                                                   std::move(dof_maps));

    if (cm->IsMaster())
      { ctr_map->_comm_keepalive_hack = cm->GetMappedEQCHierarchy()->GetCommunicator(); }

    return std::move(ctr_map);
  } // NodalAMGFactory::BuildContractDOFMap


  /** END NodalAMGFactory **/


} // namespace amg

#endif
