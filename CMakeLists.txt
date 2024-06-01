# we need to set SOME minimum version
# but I don't know what we really need
cmake_minimum_required(VERSION 3.16)

include(ExternalProject)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(NgsAMG LANGUAGES CXX )

add_library(NgsAMGlib SHARED)

option( ENABLE_COMPILE_TIMING                 "time compiling"                                                ON  )
option( USE_CCACHE                            "Use CCache"                                                    OFF )
option( ENABLE_ELASTICITY                     "enable H1-ELASTICITY AMG preconditioners"                      OFF )
option( ENABLE_ROBUST_ELASTICITY_COARSENING   "enable more robust elasticity coarsening (very long compile)"  OFF )
option( ENABLE_MIS_AGG                        "Enable (M)aximal (I)ndependent (S)et agglomeration algorithms" ON  )
option( ENABLE_SPW_AGG                        "Enable (S)uccessive(P)air(W)ise agglomeration algorithms"      ON  )
option( ENABLE_STOKES                         "Enable Stokes AMG"                                             OFF )
option( NGS_COMPATIBILITY                     "Compatibility with public NGSolve "                            ON  )
option( NGS_PYTHON_CONFIG                     "Use python to find NGSolve (for pip build)"                    OFF )

# for pip build, use python to find NGSolve, otherwise, e.g. for non-python builds, have to rely on NETGENDIR being set 
if(NGS_PYTHON_CONFIG)
  message(STATUS "Using python to find NGSolve...")
  if(CMAKE_VERSION VERSION_LESS "3.18")
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  endif()
  execute_process(COMMAND ${Python3_EXECUTABLE} -m netgen.config OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE Netgen_DIR)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m ngsolve.config OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NGSolve_DIR)
  message(STATUS  "Found Netgen  at ${Netgen_DIR}")
  message(STATUS  "Found NGSolve at ${NGSolve_DIR}")
else(NGS_PYTHON_CONFIG)
  message(STATUS "Looking for NGSolve in the usual places...")
endif(NGS_PYTHON_CONFIG)

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

if (NGSOLVE_USE_MUMPS)
  if (MUMPS_INCLUDE_DIR)
    message(STATUS "NGSolve was built with MUMPS enabled, using MUMPS_INCLUDE_DIR=${MUMPS_INCLUDE_DIR}")
  else(MUMPS_INCLUDE_DIR)
    message(FATAL_ERROR "NGSolve was built with MUMPS enabled but MUMPS_INCLUDE_DIR was not provided!")
  endif (MUMPS_INCLUDE_DIR)
endif(NGSOLVE_USE_MUMPS)

# should be fine without MKL I think??
# if (NOT NGSOLVE_USE_MKL)
#   message(FATAL_ERROR "NGsAMG requires NGSolve to be built with LAPACK!")
# endif (NOT NGSOLVE_USE_MKL)

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
    message(STATUS "Could not fine METIS-header used to build Netgen, building METIS as dependency!")
    include(cmake/metis.cmake)
    add_dependencies(NgsAMGlib project_metis)
    message(STATUS "Building METIS to ${METIS_DIR}")
    set(METIS_INCLUDE_DIR ${METIS_DIR}/include)
    find_library(METIS_LIB metis HINTS ${METIS_DIR})
    set(METIS_LIBRARY ${METIS_DIR}/lib/libmetis.a)
  else(NOT METIS_HEADER)
    message(STATUS "Using METIS from Netgen-build")
    set(METIS_INCLUDE_DIR ${NETGEN_METIS_INCLUDE_DIR})
    set(METIS_LIBRARY     nglib)
    endif(NOT METIS_HEADER)
else()
  if (NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "Could not find METIS from Netgen-build and no Metis installation was provided.")
  endif()
endif()

message(STATUS "Using METIS-header from ${METIS_INCLUDE_DIR}")
message(STATUS "USing METIS-library ${METIS_LIBRARY}")
message(STATUS "METIS-HEADER = ${METIS_HEADER}")


# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve lib/python-lib directories
if(NGS_PYTHON_CONFIG)
  message(STATUS "Setting Install dir based on NGSolve python-config")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys,sysconfig,os.path; print(os.path.relpath(sysconfig.get_path('platlib'), sys.prefix))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE python3_library_dir
  )

  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.prefix)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE install_prefix
  )
  execute_process(COMMAND ${Python3_EXECUTABLE} -m site --user-base
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE user_base RESULT_VARIABLE ret
  )

  if (ret EQUAL 0)
    set(install_prefix ${user_base})
  endif()
  set(CMAKE_INSTALL_PREFIX ${install_prefix} CACHE PATH "Install dir" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OFF)

  set(AMG_PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${python3_library_dir}/NgsAMG CACHE PATH "Install dir" FORCE)
  set(AMG_CLIB_INSTALL_DIR   ${AMG_PYTHON_INSTALL_DIR} CACHE PATH "Install dir" FORCE)
else(NGS_PYTHON_CONFIG)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      message(STATUS "Installing NgsAMG into NGSolve-installation!")
      message(STATUS "   NGSOLVE_INSTALL_DIR        = ${NGSOLVE_INSTALL_DIR}")
      message(STATUS "   NGSOLVE_LIBRARY_DIR        = ${NGSOLVE_LIBRARY_DIR}")
      message(STATUS "   NGSOLVE_INSTALL_DIR_PYTHON = ${NGSOLVE_INSTALL_DIR_PYTHON}")
      set(AMG_CLIB_INSTALL_DIR   ${NGSOLVE_LIBRARY_DIR} CACHE PATH "Install dir" FORCE)
      set(AMG_PYTHON_INSTALL_DIR ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON}/NgsAMG CACHE PATH "Install dir" FORCE)
    else(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      message(STATUS "Installing NgsAMG to explicitly specified ${CMAKE_INSTALL_PREFIX}!")
      set(AMG_CLIB_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX} CACHE PATH "Install dir" FORCE)
      set(AMG_PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} CACHE PATH "Install dir" FORCE)
    endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(NGS_PYTHON_CONFIG)


message(STATUS " CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS " NgsAMG  C-library      will be installed to: ${AMG_CLIB_INSTALL_DIR}")
message(STATUS " NgsAMG  python-package will be installed to: ${AMG_PYTHON_INSTALL_DIR}")

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
    message(FATAL_ERROR "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. Need at least 6 for elasticity preconditioners!")
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
  if ( "${NGSOLVE_MAX_SYS_DIM}" LESS "3")
    message(FATAL_ERROR "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. Need at least 3!")
  endif ( "${NGSOLVE_MAX_SYS_DIM}" LESS "3")
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


