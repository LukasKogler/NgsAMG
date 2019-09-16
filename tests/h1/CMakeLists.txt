# Copy test-files to build-directory
message(STATUS "Copy H1 test-files from ${CMAKE_CURRENT_SOURCE_DIR} to ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/amg_utils.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/simple DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/jump DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# add tests
message(STATUS "Adding H1 tests")

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

# simple H1 tests - no parameter jumps
set (SIMPLEPYT "-m pytest -k simple")
myadd_test("h1_seq" 1 ngspy ${SIMPLEPYT})
myadd_test("h1_p2"  2 ngspy ${SIMPLEPYT})
myadd_test("h1_p5"  5 ngspy ${SIMPLEPYT})

# H1 tests with coefficient jumps
set (JUMPPYT "-m pytest -k jump")
myadd_test("h1_jump_seq" 1 ngspy ${JUMPPYT})
myadd_test("h1_jump_p2"  2 ngspy ${JUMPPYT})
myadd_test("h1_jump_p5"  5 ngspy ${JUMPPYT})
