#pragma once
#include "allocator.h"
#include <span>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <vector>

namespace libmobility {
enum class device { cpu, cuda, unknown };
template <typename T>
concept numeric = std::is_arithmetic_v<T>;
template <numeric T> struct device_span : public std::span<T> {
  device dev;
  device_span(std::span<T> data, device dev) : std::span<T>(data), dev(dev) {}
};

template <numeric T> class device_adapter : public device_span<T> {
  // This class takes a device_span and a target device, if the target is
  // different from the source, it will copy the data to the target device
  using bT = std::remove_const_t<T>;
  std::vector<bT, allocator::host_cached_allocator<bT>> host_data;
  thrust::device_vector<bT, allocator::thrust_cached_allocator<bT>> device_data;
  device_span<T> original_span;

public:
  device_adapter(device_span<T> span, device target_device)
      : device_span<T>(span), original_span(span) {
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    if (span.dev != target_device) {
      if (target_device == device::cuda) {
        device_data.assign(span.begin(), span.end());
        device_span<T> tmp_span({device_data.data().get(),
                                 device_data.data().get() + device_data.size()},
                                target_device);
        this->device_span<T>::operator=(tmp_span);
      } else if (target_device == device::cpu) {
        host_data.resize(span.size());
        thrust::cuda::pointer<T> d_ptr(span.data());
        thrust::copy_n(d_ptr, span.size(), host_data.begin());
        device_span<T> tmp_span(host_data, target_device);
        this->device_span<T>::operator=(tmp_span);
      } else {
        throw std::runtime_error("Unsupported device");
      }
    }
  }

  ~device_adapter() {
    if constexpr (!std::is_const_v<T>) {
      if (this->dev != original_span.dev) {
        if (this->dev == device::cuda && original_span.dev == device::cpu) {
          thrust::cuda::pointer<T> d_ptr(this->data());
          thrust::copy_n(d_ptr, this->size(), original_span.data());
        } else if (this->dev == device::cpu &&
                   original_span.dev == device::cuda) {
          thrust::cuda::pointer<T> d_ptr(original_span.data());
          thrust::copy_n(this->data(), this->size(), d_ptr);
        }
      }
    }
  }
};

} // namespace libmobility
