cmake_minimum_required(VERSION 3.22)
project(libMobility_example)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(UAMMD_INCLUDE include/third_party/uammd/src include/third_party/uammd/src/third_party)
# enable the line below if libmobility is compiled in double
# add_compile_definitions(PUBLIC DOUBLE_PRECISION)
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
include_directories(${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARY} libMobility_PSE libMobility_NBody cufft cublas curand cusolver cuda cudart)

add_executable(example example.cpp)
