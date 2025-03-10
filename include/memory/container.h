/* Raul  P. Pelaez  2025.  Multi device  memory  management
 * This file  contains a  device_span class that  is a  wrapper around
 * std::span  that  includes  a  device field.   It  also  contains  a
 * device_adapter class that takes a  device_span and a target device.
 * The adapter  will provide  a valid address  for the  target device,
 * regardless of the origin.
 */
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
/**
 * @brief A wrapper around std::span that includes a device field.
 *
 * @tparam T The type of the data.
 */
template <numeric T> struct device_span : public std::span<T> {
  device dev;
  device_span(std::span<T> data, device dev) : std::span<T>(data), dev(dev) {}
  device_span() : std::span<T>(), dev(device::unknown) {}
  template <class Allocator>
  device_span(std::vector<T, Allocator> &data)
      : std::span<T>(data.data(), data.size()), dev(device::cpu) {}
  template <class Allocator>
  device_span(const std::vector<std::remove_const_t<T>, Allocator> &data)
      : device_span<const T>(std::span<const T>{data.data(), data.size()}, device::cpu) {}
  template <class Allocator>
  device_span(thrust::device_vector<T, Allocator> &data)
      : device_span<T>(
            std::span<T>{thrust::raw_pointer_cast(data.data()), data.size()},
            device::cuda) {}
  template <class Allocator>
  device_span(
      const thrust::device_vector<std::remove_const_t<T>, Allocator> &data)
      : device_span<T>(
            std::span<T>{thrust::raw_pointer_cast(data.data()), data.size()},
            device::cuda) {}
  /**
   * @brief Implicit conversion to a `device_span<const T>` so that it can be
   *        passed to functions that accept `device_span<const T>`.
   */
  operator device_span<const T>() const
    requires(!std::is_const_v<T>)
  {
    // Construct from *this (which is std::span<T>) to std::span<const T>
    // preserving the same device.
    return device_span<const T>(std::span<const T>(*this), dev);
  }
};
/**
 * @brief Adapts a device_span to a target device. RAII-enabled to keep original
 * up to date.
 *
 * This class takes a device_span and a target device. If the target device is
 * different from the source device, it copies the data to the target device.
 * The adpater becomes a device_span that provides valid addresses on the target
 * device, regardless of the original device.
 * The adapter takes care of updating the origin at destruction.
 * @tparam T The type of the data.
 */
template <numeric T> class device_adapter : public device_span<T> {

  using bT = std::remove_const_t<T>;
  std::vector<bT, allocator::host_cached_allocator<bT>> host_data;
  thrust::device_vector<bT, allocator::thrust_cached_allocator<bT>> device_data;
  device_span<T> original_span;

public:
  device_adapter(device_span<T> span, device target_device)
      : device_span<T>(span), original_span(span) {
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
