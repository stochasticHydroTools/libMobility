set(NAME PSE)
add_library(libMobility_${NAME} SHARED extra/uammd_wrapper.cu)
target_include_directories(libMobility_${NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(libMobility_${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
install(TARGETS libMobility_${NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mobility.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/libMobility/solvers/${NAME})
nanobind_add_module(
  ${NAME}
  STABLE_ABI
  python_wrapper.cu
)
target_link_libraries(${NAME} PRIVATE libMobility_${NAME})
install(TARGETS ${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/libMobility)
target_link_libraries(libMobility_${NAME} cufft cublas)
file(GLOB EXTRA_HEADERS extra/*.h)
install(FILES ${EXTRA_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/libMobility/solvers/${NAME}/extra)
