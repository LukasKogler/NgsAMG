# dont remember why
# cmake_minimum_required(VERSION 3.8)

# project(at_amg)
# cmake_policy(SET CMP0076 OLD)

# add_library(NgsAMG SHARED "")


# We are building two libraries:
#  1. NgsAMGlib.so, contains everything except the python-module
#  2. NgsAMG.so, contains ONLY the python module
# add_ngsolve_python_module sets the install-path to the python-lib-dir


link_directories(${CMAKE_CURRENT_BINARY_DIR})

# We need to include the python headers also for the C library because python headers leak through
# when we include NGSolve headers (e.g. comp.hpp)
if (NETGEN_USE_PYTHON)
  target_link_libraries(NgsAMGlib PUBLIC ngsolve Python3::Module)
  target_link_libraries(NgsAMG PUBLIC NgsAMGlib)
endif(NETGEN_USE_PYTHON)

include_directories(${NG_MPI_CXX_INCLUDE_DIRS})

add_subdirectory(base)
add_subdirectory(h1)

add_subdirectory(elasticity)

if (ENABLE_STOKES)
  add_subdirectory(stokes)
endif(ENABLE_STOKES)


if(NETGEN_USE_PYTHON)
  # message(STATUS "INSTALLING NgsAMG.so to ${AMG_PYTHON_INSTALL_DIR}")
  install(TARGETS NgsAMG DESTINATION ${AMG_PYTHON_INSTALL_DIR})
else()
  install(TARGETS NgsAMGlib DESTINATION ${AMG_CLIB_INSTALL_DIR})
endif(NETGEN_USE_PYTHON)

