set(NAME SelfMobility)
add_library(libMobility_${NAME} SHARED selfmobility.cu)
uammd_setup_target(libMobility_${NAME})
target_include_directories(libMobility_${NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
nanobind_add_module(
  ${NAME}
  STABLE_ABI
  python_wrapper.cu
)
target_link_libraries(${NAME} PRIVATE libMobility_${NAME})
install(TARGETS ${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/libMobility)
