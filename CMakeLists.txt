cmake_minimum_required(VERSION 3.24)
project(libMobility)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
# Set CUDA archs so all supported GPUs are covered
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all")
endif()
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G -src-in-ptx -fPIC")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -fPIC")
ENDIF()
if(DOUBLEPRECISION) # checks if equal to 1
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/include/third_party)
find_package(Python 3
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind/
  GIT_TAG        v2.4.0
)
FetchContent_MakeAvailable(nanobind)
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd/
  GIT_TAG        v2.7.3
)
FetchContent_MakeAvailable(uammd)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
include_directories(${THIRD_PARTY_DIR}/LanczosAlgorithm/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()
add_subdirectory(solvers)
enable_testing()
add_subdirectory(tests/cpp)


# Install headers under include/ to the CMAKE_INSTALL_PREFIX include directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/MobilityInterface DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
install(DIRECTORY ${THIRD_PARTY_DIR}/LanczosAlgorithm/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/MobilityInterface/)
install(DIRECTORY ${THIRD_PARTY_DIR}/uammd/src/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/uammd/src)

install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/libMobility/tests/ FILES_MATCHING PATTERN "*.py" PATTERN "*.npz")
# Add a test for each Python file in the tests directory
file(GLOB TESTS tests/*.py)
foreach(TEST ${TESTS})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  add_test(NAME ${TEST_NAME} COMMAND pytest -vs ${TEST})
endforeach()
