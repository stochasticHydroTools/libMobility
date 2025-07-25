#pragma once
#include "cublasDebug.h"
#include "cuda_lib_defines.h"
#include "defines.h"
namespace lanczos {
struct Blas {
  cublasHandle_t cublas_handle;
  Blas() { CublasSafeCall(cublasCreate(&cublas_handle)); }
  ~Blas() { CublasSafeCall(cublasDestroy(cublas_handle)); }
  template <typename... Args> void gemv(Args &&...args) {
    CublasSafeCall(
        cublasSgemv(cublas_handle, CUBLAS_OP_N, std::forward<Args>(args)...));
  }

  template <typename... Args> void nrm2(Args &&...args) {
    CublasSafeCall(cublasSnrm2(cublas_handle, std::forward<Args>(args)...));
  }

  template <typename... Args> void axpy(Args &&...args) {
    CublasSafeCall(cublasSaxpy(cublas_handle, std::forward<Args>(args)...));
  }

  template <typename... Args> void dot(Args &&...args) {
    CublasSafeCall(cublasSdot(cublas_handle, std::forward<Args>(args)...));
  }

  template <typename... Args> void scal(Args &&...args) {
    CublasSafeCall(cublasSscal(cublas_handle, std::forward<Args>(args)...));
  }
};
} // namespace lanczos
