include (ExternalProject)
project(ngs_amg)

cmake_minimum_required(VERSION 3.8)

option( ELASTICITY   "Enable Elasticity AMG" OFF)
option( USE_TAU   "Enable TAU instrumentation" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_modules")

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

find_package(MPI REQUIRED)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if (ELASTICITY)
  message(STATUS "Turning on elasticity AMG")

  if ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")
    message(WARNING "WARNING, MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}, need at least 6 for Inverse Matrix!!")
  else ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")
    message(STATUS "MAX_SYS_DIM = ${NGSOLVE_MAX_SYS_DIM}, ok, large enough.")
  endif ( "${NGSOLVE_MAX_SYS_DIM}" LESS "6")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELASTICITY")
else (ELASTICITY)
  message(STATUS "Turning off elasticity AMG")
endif (ELASTICITY)

if (USE_BOOMER)
  message(STATUS "Turning on BOOMER test preconditioner")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BOOMER")
endif (USE_BOOMER)

if (USE_TAU)
  message(STATUS "Turning on TAU")
  find_package(TAU REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TAU ${TAU_INCLUDE_FLAGS} ${TAU_COMPILE_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${TAU_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS} -lTAU-preload")
endif (USE_TAU)


add_subdirectory(src)
