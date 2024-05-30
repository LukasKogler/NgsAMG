target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_mesh_nodes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_mesh.hpp
)
