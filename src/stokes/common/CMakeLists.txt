target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/loop_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loop_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_mesh_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_map.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_map_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_energy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_factory_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_pc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_pc_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stokes_pc.cpp
)

