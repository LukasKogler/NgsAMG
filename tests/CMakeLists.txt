
include(CTest)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/clean_cache.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(h1)
add_subdirectory(elasticity)
