cmake_minimum_required(VERSION 3.24)
project(libMobility)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
set(CMAKE_BUILD_TYPE Release)
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
# Set CUDA archs so all supported GPUs are covered
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all")
endif()

#add_compile_definitions(PUBLIC DOUBLE_PRECISION)
set(UAMMD_INCLUDE include/third_party/uammd/src include/third_party/uammd/src/third_party)
# When UAMMD is available in conda-forge, use the following lines to find it instead of the above
# find_package(UAMMD REQUIRED)
# set(UAMMD_INCLUDE ${UAMMD_INCLUDE_DIR})
set(BLA_STATIC OFF)
find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
  message(STATUS "BLAS and LAPACK found")
  # Additional check for MKL
  find_path(MKL_INCLUDE_DIR NAMES mkl.h PATHS $ENV{MKLROOT}/include)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Intel MKL environment detected")
    add_compile_definitions(PUBLIC USE_MKL)
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
  link_libraries(BLAS::BLAS LAPACK::LAPACK)
else()
  message(FATAL_ERROR "BLAS and LAPACK not found. Please install them.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${UAMMD_INCLUDE} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
include_directories(include/third_party/LanczosAlgorithm/include)
include_directories(include)
link_libraries(${CUDA_LIBRARY})
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()

# Ensure ${Python3_SOABI} is not empty
if(NOT Python3_SOABI)
  message(FATAL_ERROR "Python3_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
endif()
add_subdirectory(solvers)


IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G  -lineinfo -src-in-ptx")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")
ENDIF()

# Install headers under include/ to the CMAKE_INSTALL_PREFIX include directory
install(DIRECTORY include/MobilityInterface DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
install(DIRECTORY include/third_party/LanczosAlgorithm/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/MobilityInterface/)
install(DIRECTORY include/third_party/uammd/src/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/uammd/src)
