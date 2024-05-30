target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base.hpp
)

add_subdirectory(utils)
add_subdirectory(distributed)
add_subdirectory(mesh)
add_subdirectory(coarsening)
add_subdirectory(factory)
add_subdirectory(smoothers)
add_subdirectory(solve)
add_subdirectory(precond)
add_subdirectory(ngs_extension)
add_subdirectory(linalg)


if (NETGEN_USE_PYTHON)
    add_subdirectory(python)
endif (NETGEN_USE_PYTHON)
