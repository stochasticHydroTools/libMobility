/* Raul P. Pelaez 2025. Tensor creation utilities for the libMobility Python
   interface

   This file provides utilities to create tensors in different frameworks

 */
#pragma once
#include "allocator.h"
#include <array>
#include <map>
#include <nanobind/ndarray.h>
#include <numeric>
#include <vector>
namespace libmobility {
namespace python {
namespace nb = nanobind;

/* Frameworks supported by the library */
enum class framework { numpy, torch, cupy, jax, tensorflow };

namespace detail {
static const std::map<framework, std::string> framework_names = {
    {framework::numpy, "numpy"},
    {framework::torch, "torch"},
    {framework::cupy, "cupy"},
    {framework::jax, "jax"},
    {framework::tensorflow, "tensorflow"}};

template <typename Framework, typename T>
auto create_array(std::vector<size_t> shape = {}, bool is_cuda = false) {
  nb::ndarray<T, nb::c_contig> gen_array;
  const size_t total_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (!is_cuda) {
    using alloc = allocator::host_cached_allocator<T>;
    auto *v = new std::vector<T, alloc>(total_size, T{});
    nb::capsule del(v, [](void *v) noexcept {
      delete static_cast<std::vector<T, alloc> *>(v);
    });
    auto arr = nb::ndarray<Framework, T, nb::device::cpu, nb::c_contig>(
        v->data(), shape.size(), shape.data(), std::move(del));
    gen_array = nb::cast<nb::ndarray<T, nb::c_contig>>(nb::cast(arr));
  } else {

    using alloc = allocator::device_cached_allocator<T>;
    T *v = alloc().allocate(total_size);
    thrust::fill_n(thrust::cuda::par, v, total_size, T{});
    nb::capsule deleter(v, [](void *v) noexcept {
      alloc().deallocate(static_cast<T *>(v), 0);
    });
    auto arr = nb::ndarray<Framework, T, nb::device::cuda, nb::c_contig>(
        v, shape.size(), shape.data(), std::move(deleter));
    gen_array = nb::cast<nb::ndarray<T, nb::c_contig>>(nb::cast(arr));
  }
  return gen_array;
}

} // namespace detail

template <typename pyarray> inline framework get_framework(pyarray &a) {
  nb::object obj = nb::find(a);
  std::string tn = nb::str(nb::getattr(obj, "__class__")).c_str();
  for (auto [k, v] : detail::framework_names) {
    if (tn.find(v) != std::string::npos) {
      return k;
    }
  }
  return framework::numpy;
}

template <typename T>
inline auto create_with_framework(std::vector<size_t> shape, int device_type,
                                  framework f) {
  bool is_cuda = device_type == nb::device::cuda::value;
  bool is_cpu = device_type == nb::device::cpu::value;
  if (!is_cpu && !is_cuda) {
    throw std::runtime_error("Unsupported device type " +
                             std::to_string(device_type));
  }
  switch (f) {
  case framework::numpy:
    return detail::create_array<nb::numpy, T>(shape, is_cuda);
  case framework::torch:
    return detail::create_array<nb::pytorch, T>(shape, is_cuda);
  case framework::cupy:
    return detail::create_array<nb::cupy, T>(shape, is_cuda);
  case framework::jax:
    return detail::create_array<nb::jax, T>(shape, is_cuda);
  case framework::tensorflow:
    return detail::create_array<nb::tensorflow, T>(shape, is_cuda);
  }
  return detail::create_array<nb::numpy, T>(shape);
}
} // namespace python
} // namespace libmobility
