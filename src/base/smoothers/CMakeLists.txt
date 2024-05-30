target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base_smoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_smoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_base_smoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_base_smoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_smoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_smoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gssmoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gssmoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_gssmoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_gssmoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loc_block_gssmoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loc_block_gssmoother.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loc_block_gssmoother_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dyn_block_smoother.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dyn_block_smoother.hpp
)

if (NETGEN_USE_PYTHON)
    target_sources(NgsAMG
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/python_smoothers.cpp
    )
endif (NETGEN_USE_PYTHON)