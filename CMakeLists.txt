# we need to set SOME minimum version
# but I don't know what we really need
cmake_minimum_required(VERSION 3.16)

option( ENABLE_COMPILE_TIMING                 "time compiling"                                  ON  )
option( USE_CCACHE                            "Use CCache"                                      OFF )
option( ENABLE_ELASTICITY                     "enable H1-ELASTICITY AMG preconditioners"        ON  )
option( ENABLE_ROBUST_ELASTICITY_COARSENING   "enable more robust elasticity coarsening"        ON  )
option( ENABLE_MIS_AGG                        "Enable (M)aximal (I)ndependent (S)et coarsening" ON  )
option( ENABLE_SPW_AGG                        "Enable (S)uccessive(P)air(W)ise coarsening"      ON  )
option( ENABLE_STOKES                         "Enable Stokes AMG"                               ON  )
option( NGS_COMPATIBILITY                     "Compatibility with public NGSolve "              ON  )
option( ADD_TO_NGS_INSTALL                    "Install wherever Netgen/NGSolve is installed "   OFF )

include(ExternalProject)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(addon_name NgsAMG)
project(NgsAMG LANGUAGES CXX )
include(ngsolve_addon.cmake)

if(ADD_TO_NGS_INSTALL)
  message(STATUS "Installing to Netgen/NGSolve python-dir ${NETGEN_PYTHON_DIR}")
  set(CMAKE_INSTALL_PREFIX ${NETGEN_PYTHON_DIR})
endif(ADD_TO_NGS_INSTALL)

set(AMG_PYTHON_INSTALL_DIR NgsAMG)
set(AMG_CLIB_INSTALL_DIR NgsAMG)

if(NETGEN_USE_PYTHON)
  add_ngsolve_addon(NgsAMG)
  add_library(NgsAMGlib OBJECT)
  set_target_properties(NgsAMGlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
else(NETGEN_USE_PYTHON)
  add_library(NgsAMGlib SHARED)
endif(NETGEN_USE_PYTHON)

# # ninja -t graph > graph
# # dot -Tsvg graph > ngsGraph.svg
# message(STATUS "   ")
# message(STATUS "   ")
# message(STATUS "Giving out ALL variables:")
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
# message(STATUS "   ")
# message(STATUS "   ")
# message(STATUS "CALLED WITH CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
# message(STATUS "CALLED WITH CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")

if (NGSOLVE_USE_MUMPS)
  if (MUMPS_INCLUDE_DIR)
    message(STATUS "NGSolve was built with MUMPS enabled, using MUMPS_INCLUDE_DIR=${MUMPS_INCLUDE_DIR}")
  else(MUMPS_INCLUDE_DIR)
    message(WARNING "NGSolve was built with MUMPS enabled but MUMPS_INCLUDE_DIR was not provided!")
  endif (MUMPS_INCLUDE_DIR)
endif(NGSOLVE_USE_MUMPS)

if(APPLE)
  # hard code
  set(LAPACK_LIBRARIES /System/Library/Frameworks/Accelerate.framework )
else(APPLE)
  if (NGSOLVE_USE_MKL)
    message(STATUS "NGSolve was built with MKL, attempting trying to find that")
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    message(STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
    set(MKL_INTERFACE "lp64") # 32 bit integer interface
    find_package(MKL)
    if (MKL_FOUND)
      message(STATUS "Found MKL!")
      set(LAPACK_LIBRARIES ${NGSOLVE_MKL_LIBRARIES})
      message(STATUS "Linking against MKL libraries ${LAPACK_LIBRARIES}")
    endif (MKL_FOUND)
  endif(NGSOLVE_USE_MKL)
endif(APPLE)

if (NOT LAPACK_LIBRARIES)
  message(STATUS "Trying to find Lapack")
  find_package(LAPACK REQUIRED)
endif(NOT LAPACK_LIBRARIES)

message(STATUS "Linking against Lapack libraries ${LAPACK_LIBRARIES}")
target_link_libraries(NgsAMGlib PRIVATE ${LAPACK_LIBRARIES})

if (NOT NGSOLVE_USE_MPI)
  message(FATAL_ERROR "NGsAMG requires the MPI-Version of NGSolve!")
endif (NOT NGSOLVE_USE_MPI)

if(USE_CCACHE)
  message(STATUS "CCache is turned ON")
  find_program(CCACHE_FOUND NAMES ccache ccache.bat)
  if(CCACHE_FOUND)
    message(STATUS "Found CCache executable ${CCACHE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_FOUND})
  else(CCACHE_FOUND)
    message(FATAL_ERROR "Could not find CCache executable")
  endif(CCACHE_FOUND)
endif(USE_CCACHE)

if (NGSOLVE_USE_NUMA)
  message(WARNING "NGSolve was configured with USE_NUMA=ON. This might not yield optimal performance for MPI-computations.")
endif(NGSOLVE_USE_NUMA)

if (NOT NG_MPI_FOUND)
  find_package(MPI REQUIRED)
endif(NOT NG_MPI_FOUND)

# Try to find METIS header/library that was built into Netgen, otherwise, if applicable, use the provided METIS
if (NETGEN_METIS_INCLUDE_DIR AND NETGEN_METIS_LIBRARY)
  message(STATUS "Using METIS from Netgen-build")
  find_file(METIS_HEADER metis.h ${NETGEN_METIS_INCLUDE_DIR} NO_CACHE)

  if (NOT METIS_HEADER)
    message(STATUS "Could not find METIS-header used to build Netgen!!")
    set(BUILD_METIS 1)
  else(NOT METIS_HEADER)
    message(STATUS "Using METIS from Netgen-build")
    set(METIS_INCLUDE_DIR ${NETGEN_METIS_INCLUDE_DIR})
    set(METIS_LIBRARY     nglib)
    set(BUILD_METIS 0)
    endif(NOT METIS_HEADER)
else()
  if (NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(STATUS "Could not find METIS from Netgen-build and no Metis installation was provided.")
    set(BUILD_METIS 1)
  endif()
endif()

if (BUILD_METIS)
  enable_language(C)
  include(cmake/metis.cmake)
  message(STATUS "Building METIS as a dependency to ${METIS_DIR}")
  add_dependencies(NgsAMGlib project_metis)
  set(METIS_INCLUDE_DIR ${METIS_DIR}/include)
  # find_library(METIS_LIB metis HINTS ${METIS_DIR})
  set(METIS_LIBRARY ${METIS_DIR}/lib/libmetis.a)
endif(BUILD_METIS)


message(STATUS "Using METIS-header from ${METIS_INCLUDE_DIR}")
message(STATUS "USing METIS-library ${METIS_LIBRARY}")

# message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
# message(STATUS "CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT=${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT}")
# message(STATUS "SKBUILD_PLATLIB_DIR=${SKBUILD_PLATLIB_DIR}")

message(STATUS " CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
# message(STATUS " NgsAMG  C-library      will be installed to: ${AMG_CLIB_INSTALL_DIR}")
# message(STATUS " NgsAMG  python-package will be installed to: ${AMG_PYTHON_INSTALL_DIR}")

if (NOT ENABLE_MIS_AGG)
  if (NOT ENABLE_SPW_AGG)
    message(FATAL_ERROR "Need to enable at least one of either MIS or SPW agglomeration algorithms (ENABLE_MIS_AGG, ENABLE_SPW_AGG)")
  endif (NOT ENABLE_SPW_AGG)
endif (NOT ENABLE_MIS_AGG)

if (ENABLE_MIS_AGG)
  message(STATUS "Turning ON SPW agglomeration algorithms")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIS_AGG")
else (ENABLE_MIS_AGG)
  message(STATUS "Turning OFF SPW agglomeration algorithms")
endif (ENABLE_MIS_AGG)

if (ENABLE_SPW_AGG)
  message(STATUS "Turning ON MIS agglomeration algorithms")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPW_AGG")
else (ENABLE_SPW_AGG)
  message(STATUS "Turning OFF MIS agglomeration algorithms")
endif (ENABLE_SPW_AGG)

if (ENABLE_ELASTICITY)
  message(STATUS "Turning ON elasticity AMG")
  if ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")
    message(WARNING "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. Elasticity preconditioners performance in 3d may be impaired for MAX_SYS_DIM<6!")
  endif ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELASTICITY")

  if (ENABLE_ROBUST_ELASTICITY_COARSENING)
    message(STATUS "Turning ON robust elasticity agglomeration algorithms")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELASTICITY_ROBUST_ECW")
  else(ENABLE_ROBUST_ELASTICITY_COARSENING)
    message(STATUS "Turning OFF robust elasticity agglomeration algorithms")
  endif(ENABLE_ROBUST_ELASTICITY_COARSENING)
else(ENABLE_ELASTICITY)
  message(STATUS "Turning OFF elasticity AMG")
endif(ENABLE_ELASTICITY)

if (ENABLE_STOKES)
  message(STATUS "Turning ON Stokes AMG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTOKES_AMG")
else (ENABLE_STOKES)
  message(STATUS "Turning OFF Stokes AMG")
endif (ENABLE_STOKES)

if (ENABLE_COMPILE_TIMING)
  message(STATUS "measuring compile times")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
else (ENABLE_COMPILE_TIMING)
  message(STATUS "NOT measuring compile times")
endif (ENABLE_COMPILE_TIMING)

if (NGS_COMPATIBILITY)
  message(STATUS "Compiling in NGSolve compatibility mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGS_COMPATIBILITY")
else (NGS_COMPATIBILITY)
endif (NGS_COMPATIBILITY)


add_subdirectory(src)


