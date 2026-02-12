set(NAME PSE)
add_library(libMobility_${NAME} SHARED extra/uammd_wrapper.cu)
compile_uammd_module(${NAME})