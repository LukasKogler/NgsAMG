target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(NGS_PYTHON_CONFIG)
    target_include_directories(NgsAMG PRIVATE ${Python3_INCLUDE_DIRS})
  endif(NGS_PYTHON_CONFIG)
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMG
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/python_amg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python_amg.cpp
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
    IMMEDIATE @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION ${AMG_PYTHON_INSTALL_DIR})