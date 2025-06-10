#pragma once

#include <base.hpp>

namespace ngcore
{
#define MPI_IN_PLACE 0
#define NG_MPI_REQUEST_NULL 0
#define NG_MPI_DOUBLE 0
#define NG_MPI_INT 0
#define NG_MPI_SUM 0
#define NG_MPI_MAX 0
#define NG_MPI_MIN 0
#define NG_MPI_STATUS_IGNORE 0

    using NG_MPI_Aint = int;
    using NG_MPI_Group = int;

// struct NgMPI_Comm
// {
//     static constexpr bool valid_comm = false;
//     int comm = 0;
// };

#define DUMMY_IMPL(NAME) template<class... Args> inline void NAME(Args... args){};


    DUMMY_IMPL(NG_MPI_Type_create_struct);
    DUMMY_IMPL(NG_MPI_Comm_create);
    DUMMY_IMPL(NG_MPI_Comm_group);
    DUMMY_IMPL(NG_MPI_Group_incl);
    DUMMY_IMPL(NG_MPI_Request_free);
    DUMMY_IMPL(NG_MPI_Bcast);
    DUMMY_IMPL(NG_MPI_Wait);
    DUMMY_IMPL(NG_MPI_Isend);
    DUMMY_IMPL(NG_MPI_Irecv);
    DUMMY_IMPL(NG_MPI_Type_free);
    DUMMY_IMPL(NG_MPI_Startall);
    DUMMY_IMPL(NG_MPI_Recv_init);
    DUMMY_IMPL(NG_MPI_Send_init);
    DUMMY_IMPL(NG_MPI_Type_indexed);
    DUMMY_IMPL(NG_MPI_Send);
    DUMMY_IMPL(NG_MPI_Recv);
    DUMMY_IMPL(NG_MPI_Waitany);
    DUMMY_IMPL(NG_MPI_Gatherv);
    DUMMY_IMPL(NG_MPI_Allgather);
    

#undef DUMMY_IMPL
}