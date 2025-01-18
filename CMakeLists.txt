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
if(DEFINED DOUBLEPRECISION)
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/include/third_party)
set(UAMMD_INCLUDE ${THIRD_PARTY_DIR}/uammd/src
  ${THIRD_PARTY_DIR}/uammd/src/third_party)

set(BLA_STATIC OFF)
find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
  set(MKL_DETECTED FALSE)
  foreach(lib ${BLAS_LIBRARIES})
    get_filename_component(filename "${lib}" NAME)
    string(REGEX REPLACE "^lib|\\.(so|dylib|dll)$" "" filename_stripped "${filename}")
    string(TOLOWER "${filename_stripped}" filename_lower)
    if(filename_lower MATCHES "mkl")
      set(MKL_DETECTED TRUE)
      break()
    endif()
  endforeach()
  if(MKL_DETECTED)
    message(STATUS "Intel MKL environment detected")
    add_compile_definitions(PUBLIC USE_MKL)
    find_path(MKL_INCLUDE_DIR NAMES mkl.h PATHS
      $ENV{CONDA_PREFIX}/include
      $ENV{PREFIX}/include
      $ENV{MKLROOT}/include
      $ENV{BLAS_HOME}/include
      /usr/include
      /usr/local/include
    )
    include_directories(${MKL_INCLUDE_DIR})
  else()
    find_path(BLAS_INCLUDE_DIR NAMES lapacke.h PATHS
      $ENV{CONDA_PREFIX}/include
      $ENV{PREFIX}/include
      $ENV{MKLROOT}/include
      $ENV{BLAS_HOME}/include
      /usr/include
      /usr/local/include
    )
    find_path(LAPACKE_INCLUDE_DIR lapacke.h)
    find_library(LAPACKE_LIBRARY NAMES lapacke)
    if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
      set(LAPACKE_FOUND TRUE)
      set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
      set(LAPACKE_INCLUDE_DIRS ${LAPACKE_INCLUDE_DIR})
    else()
      message(FATAL_ERROR "LAPACKE not found.")
    endif()
    include_directories(${LAPACKE_INCLUDE_DIRS})
    include_directories(${BLAS_INCLUDE_DIR})
    link_libraries(${LAPACKE_LIBRARIES})
  endif()
else()
  message(FATAL_ERROR "BLAS and LAPACK not found. Please install them.")
endif()
link_libraries(BLAS::BLAS LAPACK::LAPACK cudart)
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

add_compile_options(-DSPDLOG_COMPILED_LIB)
FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(spdlog)
link_libraries(spdlog::spdlog)
# Set log level to all
add_compile_definitions(PUBLIC SPDLOG_ACTIVE_LEVEL=0)
include_directories(${UAMMD_INCLUDE} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
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
