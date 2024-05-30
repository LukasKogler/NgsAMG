# we need to set SOME minimum version
# but I don't know what we really need
cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(NgsAMG LANGUAGES CXX )

option( ENABLE_COMPILE_TIMING                 "time compiling"                                                ON  )
option( USE_CCACHE                            "Use CCache"                                                    OFF )
option( ENABLE_ELASTICITY                     "enable H1-ELASTICITY AMG preconditioners"                      OFF )
option( ENABLE_ROBUST_ELASTICITY_COARSENING   "enable more robust elasticity coarsening (very long compile)"  OFF )
option( ENABLE_MIS_AGG                        "Enable (M)aximal (I)ndependent (S)et agglomeration algorithms" ON  )
option( ENABLE_SPW_AGG                        "Enable (S)uccessive(P)air(W)ise agglomeration algorithms"      ON  )
option( ENABLE_STOKES                         "Enable Stokes AMG"                                             OFF )
option( NGS_COMPATIBILITY                     "Compatibility with public NGSolve "                            ON  )

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

if (NGSOLVE_USE_MUMPS)
  if (MUMPS_INCLUDE_DIR)
    message(STATUS "NGSolve was built with MUMPS enabled, using MUMPS_INCLUDE_DIR=${MUMPS_INCLUDE_DIR}")
  else(MUMPS_INCLUDE_DIR)
    message(FATAL_ERROR "NGSolve was built with MUMPS enabled but MUMPS_INCLUDE_DIR provided!")
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

if (NOT MPI_FOUND)
  find_package(MPI REQUIRED)
endif(NOT MPI_FOUND)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve lib/python-lib directories
if(NOT FORCED_INSTALL_DIR)
  message(STATUS "No FORCED_INSTALL_DIR was provided, installing under NGSolve!")
  set(AMG_CLIB_INSTALL_DIR ${NGSOLVE_LIBRARY_DIR} CACHE PATH "Install dir" FORCE)
  set(AMG_PYTHON_INSTALL_DIR ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON}/NgsAMG CACHE PATH "Install dir" FORCE)
else(NOT FORCED_INSTALL_DIR)
  message(STATUS "A FORCED_INSTALL_DIR was provided, installing there!")
  set(AMG_CLIB_INSTALL_DIR ${FORCED_INSTALL_DIR} CACHE PATH "Install dir" FORCE)
  set(AMG_PYTHON_INSTALL_DIR ${FORCED_INSTALL_DIR} CACHE PATH "Install dir" FORCE)
endif(NOT FORCED_INSTALL_DIR)

message(STATUS "NGSOLVE_LIBRARY_DIR = ${NGSOLVE_LIBRARY_DIR}")
message(STATUS "        C-library will be installed to: ${AMG_CLIB_INSTALL_DIR}")
message(STATUS "   python-package will be installed to: ${AMG_PYTHON_INSTALL_DIR}")

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


