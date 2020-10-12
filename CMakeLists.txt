include (ExternalProject)
project(ngs_amg)

cmake_minimum_required(VERSION 3.8)

option( ELASTICITY   "Enable Elasticity AMG" OFF)
option( ENABLE_ELASTICITY_ROBUST_ECW   "Enable robust elasticity coarsening (compiles a long time!)" ON)
option( USE_TAU   "Enable TAU instrumentation" OFF)
option( COMPILE_TIMING   "time compiling" ON)
option( AUX_AMG   "Enable auxiliary space AMG preconditioners." ON)
option( SPWAGG "Enable successive pairwise agglomeration" ON)
option( STOKES "Enable AMG preconditioners for divdiv-penalty (Stokes)" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_modules")

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

if (NOT NGSOLVE_USE_MKL)
  message(FATAL_ERROR "NGsAMG requires NGSolve to be built with LAPACK!")
endif (NOT NGSOLVE_USE_MKL)

if (NOT NGSOLVE_USE_MPI)
  message(FATAL_ERROR "NGsAMG requires the MPI-Version of NGSolve!")
endif (NOT NGSOLVE_USE_MPI)

if (NGSOLVE_USE_NUMA)
  message(WARNING "NGSolve was configured with USE_NUMA=ON. This might not yield optimal performance for MPI-computations.")
endif(NGSOLVE_USE_NUMA)

find_package(MPI REQUIRED)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if ( "${NGSOLVE_MAX_SYS_DIM}" LESS "3")
  message(FATAL_ERROR "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. Need at least 3!")
endif ( "${NGSOLVE_MAX_SYS_DIM}" LESS "3")

if (ELASTICITY)
  message(STATUS "Checking prerequisites for Elasticity AMG")

  if ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")
    message(FATAL_ERROR "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. The Elasticity NGsAMG solver needs at least 6!")
  else ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")
    message(STATUS "NGSolve was built with MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}. OK, large enough.")
  endif ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")

  message(STATUS "Turning ON elasticity AMG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELASTICITY")

  if (ENABLE_ELASTICITY_ROBUST_ECW)
    message(STATUS "Turn ON robust elasticity coarsening (warning: takes a long time to compile!)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELASTICITY_ROBUST_ECW")
  else (ENABLE_ELASTICITY_ROBUST_ECW)
    message(STATUS "Turn OFF robust elasticity coarsening (warning: takes a long time to compile!)")
  endif (ENABLE_ELASTICITY_ROBUST_ECW)

else (ELASTICITY)
  message(STATUS "Turning OFF elasticity AMG")
endif (ELASTICITY)

if (AUX_AMG)
  message(STATUS "Turning ON auxiliary space AMG.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUX_AMG")
else (AUX_AMG)
  message(STATUS "Turning OFF auxiliary space AMG.")
endif (AUX_AMG)

if (SPWAGG)
  message(STATUS "Turning ON successive pairwise agglomeration (SPWAGG)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPWAGG")
endif (SPWAGG)

if (STOKES)
  # if (NOT AUX_AMG)
    # message(FATAL_ERROR "Need auxiliary space AMG for Stokes AMG!!")
  # endif(NOT AUX_AMG)
  message(STATUS "Turning ON H1 Stokes AMG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTOKES")
  if (ELASTICITY)
    message(STATUS "Turning ON Elasticity Stokes AMG")
  else (ELASTICITY)
    message(STATUS "Turning OFF Elasticity Stokes AMG")
  endif (ELASTICITY)
else(STOKES)
  message(STATUS "Turning OFF H1/Elasticity Stokes AMG")
endif(STOKES)


if (USE_BOOMER)
  message(STATUS "Turning on BOOMER test preconditioner")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BOOMER")
endif (USE_BOOMER)


if (USE_TAU)
  message(STATUS "Turning on TAU")
  find_package(TAU REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TAU_INCLUDE_FLAGS} ${TAU_COMPILE_FLAGS} -DUSE_TAU")
  set(CMAKE_SHARED_LINKER_FLAGS "${TAU_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
endif (USE_TAU)


if (COMPILE_TIMING)
  message(STATUS "measuring compile times")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
else (COMPILE_TIMING)
  message(STATUS "NOT measuring compile times")
endif (COMPILE_TIMING)

include(CTest)

add_subdirectory(src)
add_subdirectory(tests)

