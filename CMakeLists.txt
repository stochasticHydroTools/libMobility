cmake_minimum_required(VERSION 3.24)
project(libMobility)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
set(CMAKE_BUILD_TYPE Release)
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
# Set CUDA archs so all supported GPUs are covered
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all")
endif()
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G  -lineinfo -src-in-ptx")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")
ENDIF()

# add_compile_definitions(PUBLIC DOUBLE_PRECISION)
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
link_libraries(BLAS::BLAS LAPACK::LAPACK)
find_package(Python 3
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
include_directories(${UAMMD_INCLUDE} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
include_directories(${THIRD_PARTY_DIR}/LanczosAlgorithm/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()
add_subdirectory(solvers)

# Install headers under include/ to the CMAKE_INSTALL_PREFIX include directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/MobilityInterface DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
install(DIRECTORY ${THIRD_PARTY_DIR}/LanczosAlgorithm/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/MobilityInterface/)
install(DIRECTORY ${THIRD_PARTY_DIR}/uammd/src/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/uammd/src)


enable_testing()
install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/libMobility/tests/ FILES_MATCHING PATTERN "*.py" PATTERN "*.npz")
# Add a test for each Python file in the tests directory
file(GLOB TESTS tests/*.py)
foreach(TEST ${TESTS})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  add_test(NAME ${TEST_NAME} COMMAND pytest -vs ${TEST})
endforeach()
