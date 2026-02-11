function(setup_nanobind_module NAME)
  nanobind_add_module(
    ${NAME}
    STABLE_ABI
    python_wrapper.cu
  )

  nanobind_add_stub(
    ${NAME}_stub
    MODULE ${NAME}
    OUTPUT ${NAME}.pyi
    DEPENDS ${NAME}
    MARKER_FILE py.typed
  )

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${Python_SITEARCH}/libMobility
  )

endfunction()

add_subdirectory(DPStokes)
add_subdirectory(PSE)
add_subdirectory(NBody)
add_subdirectory(SelfMobility)

set(INIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
install(FILES "${INIT_SOURCE_DIR}/__init__.py" DESTINATION ${Python_SITEARCH}/libMobility)
