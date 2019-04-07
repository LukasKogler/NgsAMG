project(ngs_amg)

cmake_minimum_required(VERSION 3.8)

add_ngsolve_python_module(ngs_amg SHARED
  eqchierarchy.cpp amg_mesh.cpp
  amg_coarsen.cpp python_amg.cpp amg_h1.cpp
  amg_spmstuff.cpp amg_map.cpp amg_smoother.cpp
  amg_matrix.cpp amg_contract.cpp amg_elast.cpp)

include_directories(${MPI_CXX_INCLUDE_DIRS})

install(TARGETS ngs_amg DESTINATION ${CMAKE_INSTALL_PREFIX})
