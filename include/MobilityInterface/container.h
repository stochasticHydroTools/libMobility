#pragma once
#include "allocator.h"
#include <span>
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
  thrust::device_vector<bT, allocator::device_cached_allocator<bT>> device_data;
  device target_device;

public:
  device_adapter(device_span<T> span, device target_device)
      : device_span<T>(span), target_device(target_device) {
    if (span.dev != target_device) {
      if (target_device == device::cuda) {
        device_data.assign(span.begin(), span.end());
        typename device_span<T>::device_span(
            {device_data.data(), device_data.size()}, target_device);
      } else if (target_device == device::cpu) {
        host_data.assign(span.begin(), span.end());
        typename device_span<T>::device_span(host_data, target_device);
      } else {
        throw std::runtime_error("Unsupported device");
      }
    }
  }

  ~device_adapter() {
    // If the target device is different from the source device, we need to copy
    // the data back to the source device
    if (this->dev != target_device) {
      if (target_device == device::cuda) {
        thrust::copy(this->begin(), this->end(), device_data.begin());
      } else if (target_device == device::cpu) {
        thrust::copy(this->begin(), this->end(), host_data.begin());
      }
    }
  }
};

} // namespace libmobility
