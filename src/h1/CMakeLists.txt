target_include_directories(NgsAMGlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(NgsAMGlib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim1.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim1_agg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim2_agg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_dim3_agg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_impl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_energy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h1_energy_impl.hpp
)


if (NETGEN_USE_PYTHON)
    target_sources(NgsAMG
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/python_h1.cpp
    )
endif (NETGEN_USE_PYTHON)