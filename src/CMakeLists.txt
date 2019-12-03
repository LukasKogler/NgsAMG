project(ngs_amg)

cmake_minimum_required(VERSION 3.8)
add_ngsolve_python_module(ngs_amg SHARED
  eqchierarchy.cpp amg_mesh.cpp
  amg_coarsen.cpp amg_h1.cpp python_amg.cpp
  amg_spmstuff.cpp amg_map.cpp amg_smoother.cpp
  amg_matrix.cpp amg_contract.cpp amg_elast.cpp
  boomer.cpp amg_smoother2.cpp amg_smoother3.cpp
  amg_discard.cpp amg_coarsen2.cpp)

include_directories(${MPI_CXX_INCLUDE_DIRS})

if (USE_BOOMER)
   target_link_libraries(ngs_amg ${BOOMER_LIBRARIES})
endif (USE_BOOMER)

if(USE_TAU)
  #target_link_libraries(ngs_amg TAU)
  # if (TAU_PRELOAD_LIB)
  #   target_link_libraries(ngs_amg ${TAU_PRELOAD_LIB})
  # endif (TAU_PRELOAD_LIB)
  target_link_libraries(ngs_amg TAU-pthread)
endif(USE_TAU)

install(TARGETS ngs_amg DESTINATION ${CMAKE_INSTALL_PREFIX})
