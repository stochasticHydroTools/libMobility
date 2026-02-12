set(NAME NBody)
add_library(libMobility_${NAME} SHARED extra/NbodyRPY.cu)
compile_uammd_module(${NAME})