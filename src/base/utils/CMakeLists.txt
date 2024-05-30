target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR};${MUMPS_INCLUDE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_denseLA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_denseLA.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_buffering.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_arrays_tables.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_tuples.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SpecOpt.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_numeric_types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amg_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amg_register.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amg_solver_settings.hpp
)

if (NETGEN_USE_PYTHON)
  target_sources(NgsAMG
  PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/python_utils.cpp
  )
endif (NETGEN_USE_PYTHON)
