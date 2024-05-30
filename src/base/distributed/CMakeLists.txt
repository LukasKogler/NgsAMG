
target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/universal_dofs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/universal_dofs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eqchierarchy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eqchierarchy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpiwrap_extension.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reducetable.hpp
)
