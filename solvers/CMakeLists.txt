add_subdirectory(DPStokes)
add_subdirectory(PSE)
add_subdirectory(NBody)
add_subdirectory(SelfMobility)

set(INIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
install(FILES "${INIT_SOURCE_DIR}/__init__.py" DESTINATION ${Python_SITEARCH}/libMobility)
