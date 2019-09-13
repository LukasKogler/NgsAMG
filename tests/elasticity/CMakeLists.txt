function(myadd_test NAME NP EXEC ARGS)
  if (NP EQUAL 1)
    add_test(NAME "${NAME}" COMMAND ${EXEC} ${ARGS})
  else (NP EQUAL 1)
    set (PIPED_CMD "-np" "1" ${EXEC} ${ARGS})
    math(EXPR NPMO "${NP} - 1")
    foreach(rk RANGE 1 ${NPMO} 1)
      set (PIPED_CMD ${PIPED_CMD} ":" "-np" "1" bash "-c" "${EXEC} ${ARGS} 1>out_p${rk} 2>err_p${rk}")
    endforeach(rk)
    add_test(NAME "${NAME}" COMMAND mpirun ${PIPED_CMD})
  endif (NP EQUAL 1)
endfunction(myadd_test)

# Copy test-files to build-directory
message(STATUS "Copy elasticity test-files from ${CMAKE_CURRENT_SOURCE_DIR} to ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/amg_utils.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mdim DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Adding elasticity tests")

# mdim - simple
set (MDIM_SIMPLE_CMD "-m pytest -k mdim/simple")
myadd_test("el_mdim_simple_seq" 1 ngspy ${MDIM_SIMPLE_CMD})
myadd_test("el_mdim_simple_p2"  2 ngspy ${MDIM_SIMPLE_CMD})
myadd_test("el_mdim_simple_p5"  5 ngspy ${MDIM_SIMPLE_CMD})

# mdim - jump
# mdim - corner
# compound - simple
# compound - jump
# compound - corner
