target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMG
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/python_amg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python_amg.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${AMG_PYTHON_INSTALL_DIR})