cmake_minimum_required(VERSION 3.22)
project(libMobility_example)
enable_language(CUDA)
# add_compile_definitions(PUBLIC CUDA_ENABLED) # uncommenting this makes the code not compile- I don't know why
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(ROOT_DIR ../../)
# enable the line below if libmobility is compiled in double
# add_compile_definitions(PUBLIC DOUBLE_PRECISION)
include(FetchContent)
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd/
  GIT_TAG        v2.8.5
)
FetchContent_MakeAvailable(uammd)
FetchContent_Declare(
  lanczos
  GIT_REPOSITORY https://github.com/RaulPPelaez/LanczosAlgorithm
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(lanczos)
include_directories(${lanczos_SOURCE_DIR}/include)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${ROOT_DIR}/include)
include_directories(${ROOT_DIR})
link_libraries(libMobility_PSE libMobility_NBody)

add_executable(example example.cu)
add_executable(example_cuda example_gpu.cu)
uammd_setup_target(example)
uammd_setup_target(example_cuda)
