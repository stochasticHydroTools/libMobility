add_subdirectory(DPStokes)
add_subdirectory(PSE)
add_subdirectory(NBody)
add_subdirectory(SelfMobility)

set(INIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INIT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/path/to/installation/directory")
file(COPY "${INIT_SOURCE_DIR}/__init__.py" DESTINATION ${Python3_SITEARCH}/libMobility)