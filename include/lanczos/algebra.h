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
                 const device_span<T> &A, const device_span<T>& x,
                 device_span<T> &y) {
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
void cblas_gemv(int n, int m, T alpha, const device_span<T> &A,
                const device_span<T>& x, device_span<T> &y) {
  if constexpr (std::is_same_v<T, double>)
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, m, alpha, A.data(), n, x.data(),
                1, 1.0, y.data(), 1);
  else
    cblas_sgemv(CblasColMajor, CblasNoTrans, n, m, alpha, A.data(), n, x.data(),
                1, 1.0, y.data(), 1);
}

} // namespace detail
struct Algebra {

  template <typename Iter> auto norm2(Iter &v) {
    return sqrt(*thrust::inner_product(v.begin(), v.end(), v.begin(), v.begin()));
  }

  template <numeric T, typename Iter1, typename Iter2>
  void axpy(T a, const Iter1 &x, Iter2 &y){
    //thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), a * _1 + _2);
  }

  template <typename Iter, numeric T> void scal(Iter& x, T a) {
    //thrust::transform(x.begin(), x.end(), x.begin(), a * _1);
  }

  template <typename Iter1, typename Iter2> auto dot(const Iter1 &x, const Iter2 &y) {
    return 1.0;//*thrust::inner_product(x.begin(), x.end(), y.begin(), y.begin());
  }

  template <numeric T>
  void gemv(int n, int m, T alpha, const device_span<T> &A,
            const device_span<T>& x, device_span<T> &y) {
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
