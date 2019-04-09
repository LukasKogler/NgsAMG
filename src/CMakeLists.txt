project(ngs_amg)

cmake_minimum_required(VERSION 3.8)

#add_library(ngs_amg SHARED petsc_interface.cpp python_petsc_if.cpp)

# add_ngsolve_python_module(ngs_amg SHARED
#   algebraicmesh.cpp amg_contract.cpp amg_h1precond.cpp amg_precond.cpp
#   amg_repl.cpp eqchierarchy.cpp amg_coarsen.cpp amg_elastipre.cpp
#   amg_map.cpp amg_prol.cpp amg_smoother.cpp python_amg.cpp)

add_ngsolve_python_module(ngs_amg SHARED
  eqchierarchy.cpp amg_mesh.cpp amg_precond.cpp
  amg_coarsen.cpp python_amg.cpp amg_h1.cpp
  amg_spmstuff.cpp amg_map.cpp amg_smoother.cpp
  amg_matrix.cpp amg_contract.cpp)


include_directories(${MPI_CXX_INCLUDE_DIRS})

#target_link_libraries(ngs_amg ${PETSC_LIBRARIES})
#target_include_directories(ngs_amg PRIVATE ${PETSC_INCLUDES})

install(TARGETS ngs_amg DESTINATION ${CMAKE_INSTALL_PREFIX})
