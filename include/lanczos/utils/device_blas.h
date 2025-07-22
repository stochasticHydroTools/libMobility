#pragma once
#ifdef CUDA_ENABLED
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
#else
#include "lapack_and_blas_defines.h"
namespace lanczos {
struct Blas {
  void gemv(int n, int m, real *alpha, real *A, int inca, real *B, int incb,
            real *beta, real *C, int incc) {
    cblas_gemv(CblasColMajor, CblasNoTrans, n, m, *alpha, A, inca, B, incb,
               *beta, C, incc);
  }
  void nrm2(int n, const real *A, int inca, real *res) {
    *res = cblas_nrm2(n, A, inca);
  }

  void axpy(int n, real *alpha, real *A, int inca, real *B, int incb) {
    cblas_axpy(n, *alpha, A, inca, B, incb);
  }

  void dot(int n, real *A, int inca, real *B, int incb, real *alpha) {
    *alpha = cblas_dot(n, A, inca, B, incb);
  }

  void scal(int n, real *alpha, real *A, int inca) {
    cblas_scal(n, *alpha, A, inca);
  }
};
} // namespace lanczos
#endif
