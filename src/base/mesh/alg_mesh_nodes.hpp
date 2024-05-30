#ifndef FILE_ALG_MESH_NODES_HPP
#define FILE_ALG_MESH_NODES_HPP

#include <base.hpp>

namespace amg
{

// confine this garbage to it's own namespace  
namespace amg_nts
{
  typedef int vert;
  typedef int id_type;
  struct edge { INT<2,vert> v; };
  struct face { INT<3,vert> v; };
  struct cell { INT<4,vert> v; };
  struct idedge : edge { id_type id; };
  struct idface : face { id_type id; };
  struct idcell : cell { id_type id; };
  struct cedge : edge { INT<2,int> eqc; };
  struct cface : face { INT<3,int> eqc; };
  struct ccell : cell { INT<4,int> eqc; };

  INLINE bool operator == (const idedge & a, const idedge & b) { return (a.v==b.v) && (a.id==b.id); }
  INLINE bool operator == (const idface & a, const idface & b) { return (a.v==b.v) && (a.id==b.id); }
  INLINE bool operator == (const idcell & a, const idcell & b) { return (a.v==b.v) && (a.id==b.id); }
  INLINE bool operator == (const cedge & a, const cedge & b) { return (a.v==b.v) && (a.eqc==b.eqc); }
  INLINE bool operator == (const cface & a, const cface & b) { return (a.v==b.v) && (a.eqc==b.eqc); }
  INLINE bool operator == (const ccell & a, const ccell & b) { return (a.v==b.v) && (a.eqc==b.eqc); }

  INLINE ostream & operator << (ostream & ost, const idedge & idnode)
    { return ost << "[" << idnode.id << ": " << "(" << idnode.v << ")]"; }
  INLINE ostream & operator << (ostream & ost, const idface & idnode)
    { return ost << "[" << idnode.id << ": " << "(" << idnode.v << ")]"; }
  INLINE ostream & operator << (ostream & ost, const idcell & idnode)
    { return ost << "[" << idnode.id << ": " << "(" << idnode.v << ")]"; }
  INLINE ostream & operator << (ostream & ost, const cedge & cnode)
    { return ost << "[v(" << cnode.v << ")eq(" << cnode.eqc << ")]"; }
  INLINE ostream & operator << (ostream & ost, const cface & cnode)
    { return ost << "[v(" << cnode.v << ")eq(" << cnode.eqc << ")]"; }
  INLINE ostream & operator << (ostream & ost, const ccell & cnode)
    { return ost << "[v(" << cnode.v << ")eq(" << cnode.eqc << ")]"; }

} // namespace amg_nts

template<ngfem::NODE_TYPE NT> struct amg_type_trait { typedef void type; };
template<> struct amg_type_trait<ngfem::NT_VERTEX> { typedef amg_nts::vert type; };
template<> struct amg_type_trait<ngfem::NT_EDGE>   { typedef amg_nts::idedge type; };
template<> struct amg_type_trait<ngfem::NT_FACE>   { typedef amg_nts::idface type; };
template<> struct amg_type_trait<ngfem::NT_CELL>   { typedef amg_nts::idcell type; };
template<ngfem::NODE_TYPE NT> using AMG_Node = typename amg_type_trait<NT>::type;

template<ngfem::NODE_TYPE NT> struct amg_type_trait_cross { typedef void type; };
template<> struct amg_type_trait_cross<ngfem::NT_EDGE>   { typedef amg_nts::cedge type; };
template<> struct amg_type_trait_cross<ngfem::NT_FACE>   { typedef amg_nts::cface type; };
template<> struct amg_type_trait_cross<ngfem::NT_CELL>   { typedef amg_nts::ccell type; };
template<ngfem::NODE_TYPE NT> using AMG_CNode = typename amg_type_trait_cross<NT>::type;


template<ngfem::NODE_TYPE NT> INLINE amg_nts::id_type GetNodeId (const AMG_Node<NT> & node)
  { return node.id; }

template<> INLINE amg_nts::id_type GetNodeId<NT_VERTEX> (const AMG_Node<NT_VERTEX> & node)
  { return node; }

} // namespace amg



namespace ngcore
{
template<> struct MPI_typetrait<amg::amg_nts::edge> {
  static MPI_Datatype MPIType () {
    return GetMPIType<typename ngstd::INT<2> >();
  }
};

template<> struct MPI_typetrait<amg::AMG_Node<ngfem::NT_EDGE>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_Node<ngfem::NT_EDGE>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_Node<ngfem::NT_EDGE>::v)>(), GetMPIType<decltype(amg::AMG_Node<ngfem::NT_EDGE>::id)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

template<> struct MPI_typetrait<amg::AMG_Node<ngfem::NT_FACE>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_Node<ngfem::NT_FACE>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_Node<ngfem::NT_FACE>::v)>(), GetMPIType<decltype(amg::AMG_Node<ngfem::NT_FACE>::id)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

template<> struct MPI_typetrait<amg::AMG_Node<ngfem::NT_CELL>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_Node<ngfem::NT_CELL>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_Node<ngfem::NT_CELL>::v)>(), GetMPIType<decltype(amg::AMG_Node<ngfem::NT_CELL>::id)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

template<> struct MPI_typetrait<amg::AMG_CNode<ngfem::NT_EDGE>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_CNode<ngfem::NT_EDGE>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_EDGE>::v)>(), GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_EDGE>::eqc)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

template<> struct MPI_typetrait<amg::AMG_CNode<ngfem::NT_FACE>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_CNode<ngfem::NT_FACE>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_FACE>::v)>(), GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_FACE>::eqc)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

template<> struct MPI_typetrait<amg::AMG_CNode<ngfem::NT_CELL>> {
  static MPI_Datatype MPIType() {
    static MPI_Datatype MPI_T = 0;
    if(!MPI_T)
  {
    int block_len[2] = {1,1};
    MPI_Aint displs[2] = {0, sizeof(decltype(amg::AMG_CNode<ngfem::NT_CELL>::v))};
  MPI_Datatype types[2] = {GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_CELL>::v)>(), GetMPIType<decltype(amg::AMG_CNode<ngfem::NT_CELL>::eqc)>()};
    MPI_Type_create_struct(2, block_len, displs, types, &MPI_T);
    MPI_Type_commit ( &MPI_T );
  }
    return MPI_T;
  }
};

} // namespace ngcore

#endif
