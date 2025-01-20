#pragma once
#include "memory/allocator.h"
#include "memory/container.h"
#include <cublas_v2.h>
#include <mkl.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
using namespace thrust::placeholders;
namespace lanczos {
using libmobility::device;
using libmobility::device_adapter;
using libmobility::device_span;
using libmobility::numeric;

namespace detail {
template <numeric T>
void cublas_gemv(cublasHandle_t handle, int n, int m, T alpha,
                 device_span<const T> A, device_span<const T> x,
                 device_span<T> y) {
  cublasStatus_t status;
  T beta = 1.0;
  if constexpr (std::is_same_v<T, double>)
    status = cublasDgemv(handle, CUBLAS_OP_N, n, m, &alpha, A.data(), n,
                         x.data(), 1, &beta, y.data(), 1);
  else {
    status = cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, A.data(), n,
                         x.data(), 1, &beta, y.data(), 1);
  }
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUBLAS gemv failed with status " +
                             std::to_string(status));
}

template <numeric T>
void cblas_gemv(int n, int m, T alpha, device_span<const T> A,
                device_span<const T> x, device_span<T> y) {
  if constexpr (std::is_same_v<T, double>)
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, m, alpha, A.data(), n, x.data(),
                1, 1.0, y.data(), 1);
  else
    cblas_sgemv(CblasColMajor, CblasNoTrans, n, m, alpha, A.data(), n, x.data(),
                1, 1.0, y.data(), 1);
}

} // namespace detail
struct Algebra {

  template <numeric T> T nrm2(device_span<const T> v) {
    return sqrt(thrust::inner_product(v.begin(), v.end()));
  }

  template <numeric T>
  void axpy(T a, device_span<const T> x, device_span<T> y) {
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), a * _1 + _2);
  }

  template <numeric T> void scal(T a, device_span<T> x) {
    thrust::transform(x.begin(), x.end(), x.begin(), a * _1);
  }

  template <numeric T> T dot(device_span<const T> x, device_span<const T> y) {
    return thrust::inner_product(x.begin(), x.end(), y.begin());
  }

  template <numeric T>
  void gemv(int n, int m, T alpha, device_span<const T> A,
            device_span<const T> x, device_span<T> y) {
    if (A.dev == device::cuda) {
      detail::cublas_gemv(handle, n, m, alpha, A, x, y);
    } else {
      detail::cblas_gemv(n, m, alpha, A, x, y);
    }
  }
  cublasHandle_t handle;
  Algebra() { cublasCreate(&handle); }
  ~Algebra() { cublasDestroy(handle); }
};
} // namespace lanczos
