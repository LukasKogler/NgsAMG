target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR};${MUMPS_INCLUDE_DIR})

if (NETGEN_USE_PYTHON)
  # dyn_block header is used for python-export
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_sparseMM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_sparseMM.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_sparseLA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_sparseLA.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dyn_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dyn_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dyn_block_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dcc_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dcc_map.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bla_extension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bla_extension.hpp
)

