function(compile_uammd_module NAME)
uammd_setup_target(libMobility_${NAME})

  nanobind_add_module(
      ${NAME}
      STABLE_ABI
      python_wrapper.cu
  )
  uammd_setup_target(${NAME})
  target_link_libraries(libMobility_${NAME} PRIVATE cblas)
  target_link_libraries(${NAME} PRIVATE libMobility_${NAME} cblas)
  install(TARGETS libMobility_${NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(TARGETS ${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/libMobility)

  nanobind_add_stub(
      ${NAME}_stub
      INSTALL_TIME
      MODULE ${NAME}
      OUTPUT "${Python_SITEARCH}/libMobility/${NAME}.pyi"
      PYTHON_PATH "${Python_SITEARCH}/libMobility"
      MARKER_FILE "${Python_SITEARCH}/libMobility/py.typed"
  )

endfunction()

add_subdirectory(DPStokes)
add_subdirectory(PSE)
add_subdirectory(NBody)
add_subdirectory(SelfMobility)

set(INIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
install(FILES "${INIT_SOURCE_DIR}/__init__.py" DESTINATION ${Python_SITEARCH}/libMobility)
