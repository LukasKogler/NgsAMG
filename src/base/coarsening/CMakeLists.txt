
target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}; ${METIS_INCLUDE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_link_libraries(NgsAMGlib PRIVATE ${METIS_LIBRARY})

target_sources(NgsAMGlib
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/base_coarse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/base_coarse.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerate_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerate_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerator_impl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerator_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/agglomerator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spw_agg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spw_agg_impl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spw_agg_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg_impl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg_map.hpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg_elast.hpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg_elast_impl.hpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/mis_agg_elast_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/plate_test_agg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/plate_test_agg_impl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/plate_test_agg_map.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_contract.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_contract.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_contract.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_contract.hpp
)

if (NETGEN_USE_PYTHON)
  target_sources(NgsAMG
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/python_coarse.cpp
  )
endif(NETGEN_USE_PYTHON)