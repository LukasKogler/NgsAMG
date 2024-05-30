target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(NETGEN_USE_PYTHON)
  target_include_directories(NgsAMG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif(NETGEN_USE_PYTHON)

target_sources(NgsAMGlib
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/amg_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amg_matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSolver.hpp
)

if(NETGEN_USE_PYTHON)
    target_sources(NgsAMG
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/python_solve.cpp
    )
endif(NETGEN_USE_PYTHON)