set(NAME PSE)
add_library(libMobility_${NAME} SHARED extra/uammd_wrapper.cu)
uammd_setup_target(libMobility_${NAME})

setup_nanobind_module(${NAME})

uammd_setup_target(${NAME})
target_link_libraries(${NAME} PRIVATE libMobility_${NAME})
install(TARGETS libMobility_${NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS ${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/libMobility)
