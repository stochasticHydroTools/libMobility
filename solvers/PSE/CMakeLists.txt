set(NAME PSE)
add_library(libMobility_${NAME} SHARED extra/uammd_wrapper.cu)
uammd_setup_target(libMobility_${NAME})
nanobind_add_module(
  ${NAME}
  STABLE_ABI
  python_wrapper.cu
)
target_link_libraries(${NAME} PRIVATE libMobility_${NAME})
install(TARGETS libMobility_${NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS ${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/libMobility)
