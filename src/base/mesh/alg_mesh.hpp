#ifndef FILE_ALG_MESH_HPP
#define FILE_ALG_MESH_HPP

#include <base.hpp>
#include <utils.hpp>               // param-pack utils
#include <utils_arrays_tables.hpp> // for algos
#include <utils_io.hpp> // for algos
#include <reducetable.hpp>

#include <base_mesh.hpp>

#include "alg_mesh_nodes.hpp"

namespace amg
{

/** Nodal data that can be attached to a Mesh to form an AlgebraicMesh **/
template<NODE_TYPE NT, class T>
class AttachedNodeData // : public NodeData<NT, T>
{
public:
  using TATTACHED = AttachedNodeData<NT, T>;
  using TDATA = T;
  static constexpr NODE_TYPE TNODE = NT;
protected:
  BlockTM * mesh;
  Array<T> data;
  PARALLEL_STATUS stat;
public:
  AttachedNodeData (Array<T> && _data, PARALLEL_STATUS _stat) : data(std::move(_data)), stat(_stat) {}
  virtual ~AttachedNodeData () { ; }
  void SetMesh (BlockTM * _mesh) { mesh = _mesh; }
  PARALLEL_STATUS GetParallelStatus () const { return stat; }
  void SetParallelStatus (PARALLEL_STATUS astat) { stat = astat; }
  Array<T> & GetModData () { return data; }
  FlatArray<T> Data () const { return data; }

  virtual void Cumulate () const
  {
    if (stat == DISTRIBUTED) {
      AttachedNodeData<NT,T>& nc_ref(const_cast<AttachedNodeData<NT,T>&>(*this));

      mesh->AllreduceNodalData<NT, T> (nc_ref.data, [](auto & tab){return std::move(sum_table(tab)); }, false);

      nc_ref.stat = CUMULATED;
    }
  }

  virtual void Distribute () const
  {
    AttachedNodeData<NT,T>& nc_ref(const_cast<AttachedNodeData<NT,T>&>(*this));
    const auto & eqc_h = *mesh->GetEQCHierarchy();

    if (stat == CUMULATED) {
      mesh->template ApplyEQ2<NT>([&](auto eqc, auto nodes) LAMBDA_INLINE {
        if (!eqc_h.IsMasterOfEQC(eqc))
        {
          for (const auto &node : nodes)
            { data[GetNodeId<NT>(node)] = 0; }
        }
      });
    }
    nc_ref.stat = DISTRIBUTED;
  }
}; // class AttachedNodeData


template<class ...T_A>
class BlockTMWithData : public BlockTM
{
protected:
  std::tuple<T_A*...> node_attached_data;

public:
  BlockTMWithData (BlockTM && _mesh, std::tuple<T_A*...> _data)
    : BlockTM(std::move(_mesh))
    , node_attached_data(_data)
  {
    // std::apply([&](auto& ...x){(..., x->SetMesh(this));}, AttachedData());
    ApplyToAttachedData([&](auto &ad) { ad->SetMesh(this); });
  }

  BlockTMWithData (BlockTM && _mesh, T_A*... _data)
    : BlockTMWithData (std::move(_mesh), std::tuple<T_A*...>(_data...))
  { ; }

  BlockTMWithData (shared_ptr<EQCHierarchy> eqc_h)
    : BlockTM(eqc_h)
  { ; } // dummy node_data
  
  virtual ~BlockTMWithData()
  {
    ApplyToAttachedData([&](auto ad) { delete ad; });
  }

  INLINE const std::tuple<T_A*...>& AttachedData () const { return node_attached_data; }

  INLINE void CumulateData () const
  {
    ApplyToAttachedData([&](auto ad) { ad->Cumulate(); });
  }

  INLINE void DistributeData () const
  {
    ApplyToAttachedData([&](auto ad) { ad->Distribute(); });
  }
  
  template<class TLAM>
  INLINE void ApplyToAttachedData(const TLAM & lam) const
  {
    std::apply([&](auto& ...x){ (lam(x),...); }, AttachedData());
  }

  template<class TLAM>
  INLINE void ApplyToAttachedData(const TLAM & lam)
  {
    std::apply([&](auto& ...x){ (lam(x),...); }, AttachedData());
  }
  
}; // class BlockTMWithData

/** Mesh topology + various attached nodal data **/
template<class... T>
class BlockAlgMesh : public BlockTMWithData<typename T::TATTACHED...>
{
public:
  using T_MESH_W_DATA = BlockTMWithData<typename T::TATTACHED...>;

protected:
  std::tuple<T*...> node_data;

public:
  using TTUPLE = std::tuple<T*...>;

  BlockAlgMesh (BlockTM && _mesh, std::tuple<T*...> _data)
    : T_MESH_W_DATA(std::move(_mesh), toBaseTuple(_data))
    , node_data(_data)
  {
    // std::apply([&](auto& ...x){(..., x->SetMesh(this));}, AttachedData());
    // ApplyToData([&](auto &ad) { ad->SetMesh(this); });
  }

  BlockAlgMesh (BlockTM && _mesh, T*... _data)
    : BlockAlgMesh (std::move(_mesh), std::tuple<T*...>(_data...))
  { ; }

  BlockAlgMesh (shared_ptr<EQCHierarchy> eqc_h)
    : T_MESH_W_DATA(eqc_h)
  { ; } // dummy node_data

  virtual void printTo(std::ostream &os) const override;

  INLINE tuple<typename T::TATTACHED*...> toBaseTuple(tuple<T*...> const &derivedTup)
  {
    return std::apply([&](auto ...x) { return make_tuple(
                ((typename remove_pointer<typename remove_reference<decltype(x)>::type>::type::TATTACHED*)(x))...
              );
            }, derivedTup);
  }

  INLINE void AllocateAttachedData()
  {
    this->node_data = make_tuple<T*...>( new T(Array<typename T::TDATA>(this->template GetNN<T::TNODE>()), DISTRIBUTED)... );
    this->node_attached_data = toBaseTuple(node_data);
    ApplyToData([&](auto &ad) { ad->SetMesh(this); });
  }

  virtual ~BlockAlgMesh () = default;

  INLINE const std::tuple<T*...>& Data () const { return node_data; }
  INLINE       std::tuple<T*...>& ModData () { return node_data; }

  template<class TLAM>
  INLINE void ApplyToData(const TLAM & lam) const
  {
    std::apply([&](auto& ...x){ (lam(x),...); }, Data());
  }

  template<class TLAM>
  INLINE void ApplyToData(const TLAM & lam)
  {
    std::apply([&](auto& ...x){ (lam(x),...); }, Data());
  }

}; // class BlockAlgMesh


template<NODE_TYPE NT, class T>
INLINE std::ostream & operator<<(std::ostream &os, AttachedNodeData<NT,T>& nd)
{
  os << "Data for NT=" << NT;;
  os << ", status: " << nd.GetParallelStatus();
  os << ", data:" << endl; prow3(nd.Data(), os, "   ", 1); os << endl;
  return os;
}

template<class... T>
void BlockAlgMesh<T...> :: printTo(std::ostream &os) const
{
  BlockTM::printTo(os);
  os << endl << "BlockAlgMesh Node-Data:" << endl;

  ApplyToData([&](auto &ad) {
    os << "Data @" << ad << ": " << endl; 
    os << *ad << endl << endl;
  });

  // apply([&](auto&&... x) {
  //   (os << ... << *x) << endl << endl;
  //   },
  //   Data()
  // );
}

} // namespace amg


#endif // FILE_ALG_MESH_HPP
