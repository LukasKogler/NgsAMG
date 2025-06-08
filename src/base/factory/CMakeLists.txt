target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/aux_mat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodal_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodal_factory_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vertex_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vertex_factory_impl.hpp
)
