target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ncfespace.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ncfespace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/facet_aux_info.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/facet_aux_info.cpp
)
