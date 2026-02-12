set(NAME SelfMobility)
add_library(libMobility_${NAME} SHARED selfmobility.cu)
compile_uammd_module(${NAME})