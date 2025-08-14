#pragma once
#include <thrust/device_vector.h>
namespace lanczos {
template <class T> using device_container = thrust::device_vector<T>;
namespace detail {
template <class T> auto getRawPointer(thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <class Iter, class Iter2>
void device_copy(Iter begin, Iter end, Iter2 out) {
  thrust::copy(thrust::cuda::par, begin, end, out);
}
template <class Iter, class T> void device_fill(Iter begin, Iter end, T value) {
  thrust::fill(thrust::cuda::par, begin, end, value);
}

} // namespace detail
} // namespace lanczos
