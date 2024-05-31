
message(STATUS "Trying to figure out how to compile with TAU..")

if (NOT TAU_MAKEFILE)
  message(STATUS "TAU_MAKEFILE not given explicitely, checking environment")
  if (DEFINED ENV{TAU_MAKEFILE})
    set(TAU_MAKEFILE $ENV{TAU_MAKEFILE})
    message(STATUS "Using environment TAU_MAKEFILE = ${TAU_MAKEFILE}")
  else (DEFINED ENV{TAU_MAKEFILE})
    message(FATAL_ERROR "No TAU_MAKEFILE in environment either, could not find TAU!")
  endif (DEFINED ENV{TAU_MAKEFILE})
else (NOT TAU_MAKEFILE)
  message(STATUS "Using TAU_MAKEFILE = ${TAU_MAKEFILE}")
endif (NOT TAU_MAKEFILE)
  
message(STATUS "Trying to figure out include flags")
execute_process(COMMAND tau_cxx.sh -tau_makefile=${TAU_MAKEFILE} -tau:showincludes OUTPUT_VARIABLE TAU_INCLUDE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
# message(STATUS "TAU include flags: ${TAU_INCLUDE_FLAGS}")

message(STATUS "Trying to figure out link flags")
execute_process(COMMAND tau_cxx.sh -tau_makefile=${TAU_MAKEFILE} -tau:showlibs OUTPUT_VARIABLE TAU_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "TAU link flags: ${TAU_LINK_FLAGS}")

message(STATUS "Trying to figure out compile flags")
execute_process(COMMAND tau_cxx.sh -tau_makefile=${TAU_MAKEFILE} -tau:show OUTPUT_VARIABLE TAU_ALL_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

# -tau:show should give: COMPILE_CMD TAU_INCLUDE [we want this! > [COMPILE_DEFINITIONS]] TAU_LINK
string(FIND ${TAU_ALL_FLAGS} ${TAU_INCLUDE_FLAGS} INC_START)
string(LENGTH ${TAU_INCLUDE_FLAGS} INC_LEN)
math(EXPR LOWER "${INC_START} + ${INC_LEN}")
string(FIND ${TAU_ALL_FLAGS} ${TAU_LINK_FLAGS} UPPER)
math(EXPR TCF_LEN "${UPPER}-${LOWER}" )
string(SUBSTRING ${TAU_ALL_FLAGS} ${LOWER} ${TCF_LEN} TAU_COMPILE_FLAGS1) 
string(STRIP ${TAU_COMPILE_FLAGS1} TAU_COMPILE_FLAGS)
# message(STATUS "TAU compile flags: " ${TAU_COMPILE_FLAGS})
  
