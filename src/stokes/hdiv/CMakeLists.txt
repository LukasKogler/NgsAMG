target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_mesh.hpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_mesh_impl.hpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_pc.hpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_pc_impl.hpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_2d.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_2d_spw.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_2d_mis.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_2d_factory.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_3d.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_3d_spw.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_3d_mis.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_stokes_gg_3d_factory.cpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_hdg_embedding.hpp
${CMAKE_CURRENT_SOURCE_DIR}/hdiv_hdg_embedding.cpp
)

