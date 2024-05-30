target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_energy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_energy_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_2d_spw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_3d_spw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_2d_mis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_3d_mis.cpp
)

if (NETGEN_USE_PYTHON)
    target_sources(NgsAMG
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/python_elasticity.cpp
    )
endif (NETGEN_USE_PYTHON)
