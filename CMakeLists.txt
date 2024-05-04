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
set(CMAKE_CUDA_ARCHITECTURES "all")

#add_compile_definitions(PUBLIC DOUBLE_PRECISION)
set(UAMMD_INCLUDE third_party/uammd/src third_party/uammd/src/third_party)
# When UAMMD is available in conda-forge, use the following lines to find it instead of the above
# find_package(UAMMD REQUIRED)
# set(UAMMD_INCLUDE ${UAMMD_INCLUDE_DIR})
set(BLA_STATIC OFF)
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if(BLAS_FOUND)
  message("mkl environment detected")
  add_compile_definitions(PUBLIC USE_MKL)
  link_libraries(
    BLAS::BLAS
  )
  find_path(BLAS_INCLUDE_DIRS mkl.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
else()
  unset(BLA_VENDOR)
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  find_package(BLAS REQUIRED)
  link_libraries(${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  find_path(BLAS_INCLUDE_DIRS cblas.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
  find_path(LAPACKE_INCLUDE_DIRS lapacke.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{LAPACKE_HOME}/include
  )

endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${UAMMD_INCLUDE} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
include_directories(third_party/LanczosAlgorithm/include)
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
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY third_party/LanczosAlgorithm/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/MobilityInterface)
install(DIRECTORY third_party/uammd/src/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/uammd/src)
