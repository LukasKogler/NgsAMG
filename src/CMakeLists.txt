# dont remember why
# cmake_minimum_required(VERSION 3.8)

# project(at_amg)
# cmake_policy(SET CMP0076 OLD)

# add_library(NgsAMG SHARED "")


# We are building two libraries:
#  1. NgsAMGlib.so, contains everything except the python-module
#  2. NgsAMG.so, contains ONLY the python module
# add_ngsolve_python_module sets the install-path to the python-lib-dir

add_library(NgsAMGlib SHARED)
target_link_libraries(NgsAMGlib PUBLIC ngsolve)
set_target_properties(NgsAMGlib PROPERTIES INSTALL_RPATH "${AMG_CLIB_INSTALL_DIR}")

link_directories(${NGSOLVE_LIBRARY_DIR})
link_directories(${CMAKE_CURRENT_BINARY_DIR})

# We need to include the python headers also for the C library because python headers leak through
# when we include NGSolve headers (e.g. comp.hpp)
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMGlib PUBLIC ${NETGEN_PYTHON_INCLUDE_DIRS})
endif(NETGEN_USE_PYTHON)

if (NETGEN_USE_PYTHON)
  add_ngsolve_python_module(NgsAMG SHARED "")
  add_dependencies(NgsAMG NgsAMGlib)
  target_link_libraries(NgsAMG PUBLIC NgsAMGlib)
endif(NETGEN_USE_PYTHON)

include_directories(${MPI_CXX_INCLUDE_DIRS})

set_property(GLOBAL PROPERTY AT_AMG_SOURCE "")
set_property(GLOBAL PROPERTY BASE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/")

function(add_source_file SRCFILE)
  # get the relative path starting from HERE
  # get_property(bdir GLOBAL PROPERTY BASE_SRC_DIR)
  # string(REPLACE "${bdir}" "" FINAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${SRCFILE}")
 
  # get_property(tmp  GLOBAL PROPERTY AT_AMG_SOURCE)
  # # set(tmp "${tmp} ${FINAL_SRC}")  
  # set(tmp "${tmp} ${CMAKE_CURRENT_SOURCE_DIR}/${SRCFILE}")  
  # set_property(GLOBAL PROPERTY AT_AMG_SOURCE "${tmp}")
endfunction()

add_subdirectory(base)
add_subdirectory(h1)

add_subdirectory(elasticity)

if (ENABLE_STOKES)
  add_subdirectory(stokes)
endif(ENABLE_STOKES)

get_property(source_list GLOBAL PROPERTY AT_AMG_SOURCE)


install(TARGETS NgsAMGlib DESTINATION ${AMG_CLIB_INSTALL_DIR})

if(NETGEN_USE_PYTHON)
  install(TARGETS NgsAMG DESTINATION ${AMG_PYTHON_INSTALL_DIR})
endif(NETGEN_USE_PYTHON)

