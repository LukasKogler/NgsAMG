project(ngs_amg)

cmake_minimum_required(VERSION 3.8)

# amg_agg_*.*pp compile a long time, so start with those
add_ngsolve_python_module(ngs_amg SHARED
  amg_agg_h1scal.cpp amg_agg_el2d.cpp amg_agg_el3d.cpp
  eqchierarchy.cpp amg_mesh.cpp amg_coarsen.cpp
  amg_factory.cpp python_amg.cpp amg_pc.cpp
  amg_h1_dim1.cpp amg_h1_dim2.cpp amg_h1_dim3.cpp
  amg_elast_2d.cpp amg_elast_3d.cpp
  amg_spmstuff.cpp amg_map.cpp amg_smoother.cpp
  amg_matrix.cpp amg_contract.cpp python_smoothers.cpp
  boomer.cpp amg_smoother2.cpp amg_smoother3.cpp
  amg_discard.cpp amg_blocksmoother.cpp
  amg_mcs_gg_2d.cpp amg_mcs_gg_3d.cpp
  amg_mcs_epseps_2d.cpp amg_mcs_epseps_3d.cpp
  amg_stokes_gg_2d.cpp)

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
