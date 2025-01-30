/* Raul P. Pelaez 2025. Allocators for libMobility.

   This file provides a series of memory resources and allocators that
   can be used with the C++17 allocator interface.

   The main use for this file is to provide cached allocators that can
   be used with std::vector  for host memory and thrust::device_vector
   for GPU memory.

   Example:

   using namespace libmobility::allocator;
   using float_vector = std::vector<float, host_cached_allocator<float>>;
   using float_device_vector = thrust::device_vector<float, thrust_cached_allocator<float>>;

   float_vector v(1000); // Allocates 1000 floats in host memory
   float_device_vector dv(1000); // Allocates 1000 floats in device memory

   The cached allocators  will store previously allocated  blocks in a
   cache, and  retrieve them when  a similar block is  requested. This
   makes them very efficient for repeated allocations.
 */
#pragma once
#include <cstddef>
#include <map>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/memory.h>
namespace libmobility {
namespace allocator {
namespace detail {

template <class T> using cuda_ptr = thrust::cuda::pointer<T>;

template <class T> class memory_resource {
public:
  using pointer = T;
  using max_align_t =
      long double; // This C++11 alias is not available in std with g++-4.8.5
  pointer allocate(std::size_t bytes,
                   std::size_t alignment = alignof(max_align_t)) {
    return do_allocate(bytes, alignment);
  }

  void deallocate(pointer p, std::size_t bytes,
                  std::size_t alignment = alignof(max_align_t)) {
    return do_deallocate(p, bytes, alignment);
  }

  bool is_equal(const memory_resource &other) const noexcept {
    return do_is_equal(other);
  }

  virtual pointer do_allocate(std::size_t bytes, std::size_t alignment) = 0;

  virtual void do_deallocate(pointer p, std::size_t bytes,
                             std::size_t alignment) = 0;

  virtual bool do_is_equal(const memory_resource &other) const noexcept {
    return this == &other;
  }
};
} // namespace detail

template <class MR> MR *get_default_resource() {
  static MR default_resource;
  return &default_resource;
}

class device_memory_resource : public detail::memory_resource<void *> {
  using super = detail::memory_resource<void *>;

public:
  using pointer = typename super::pointer;

  pointer do_allocate(std::size_t bytes, std::size_t alignment) override {
    auto ptr = thrust::raw_pointer_cast(thrust::cuda::malloc<char>(bytes));
    return reinterpret_cast<pointer>(ptr);
  }

  void do_deallocate(pointer p, std::size_t bytes,
                     std::size_t alignment) override {
    thrust::cuda::pointer<void> void_ptr(p);
    thrust::cuda::free(void_ptr);
  }
};

class managed_memory_resource : public detail::memory_resource<void *> {
  using super = detail::memory_resource<void *>;

public:
  using pointer = typename super::pointer;

  virtual pointer do_allocate(std::size_t bytes,
                              std::size_t alignment) override {
    void *result;
    cudaMallocManaged(&result, bytes, cudaMemAttachGlobal);
    return static_cast<pointer>(result);
  }

  virtual void do_deallocate(pointer p, std::size_t bytes,
                             std::size_t alignment) override {
    cudaFree(thrust::raw_pointer_cast(p));
  }
};

class host_memory_resource : public detail::memory_resource<void *> {
  using super = detail::memory_resource<void *>;

public:
  using pointer = typename super::pointer;

  virtual pointer do_allocate(std::size_t bytes,
                              std::size_t alignment) override {
    return static_cast<pointer>(new char[bytes]);
  }

  virtual void do_deallocate(pointer p, std::size_t bytes,
                             std::size_t alignment) override {
    delete[] static_cast<char *>(p);
  }
};

// A pool device memory_resource, stores previously allocated blocks in a cache
//  and retrieves them fast when similar ones are allocated again (without
//  calling malloc everytime).
template <class MR> struct pool_memory_resource_adaptor {
private:
  MR *res;

public:
  using pointer = typename MR::pointer;

  ~pool_memory_resource_adaptor() noexcept {
    try {
      free_all();
    } catch (...) {
    }
  }

  pool_memory_resource_adaptor(MR *resource) : res(resource) {}
  pool_memory_resource_adaptor()
      : pool_memory_resource_adaptor(get_default_resource<MR>()) {}

  using FreeBlocks = std::multimap<std::ptrdiff_t, void *>;
  using AllocatedBlocks = std::map<void *, std::ptrdiff_t>;
  FreeBlocks free_blocks;
  AllocatedBlocks allocated_blocks;

  pointer do_allocate(std::size_t bytes, std::size_t alignment) {
    pointer result;
    std::ptrdiff_t blockSize = 0;
    auto available_blocks = free_blocks.equal_range(bytes);
    auto available_block = available_blocks.first;
    // Look for a block of the same size
    if (available_block == free_blocks.end()) {
      available_block = available_blocks.second;
    }
    // Try to find a block greater than requested size
    if (available_block != free_blocks.end()) {
      result = pointer(available_block->second);
      blockSize = available_block->first;
      free_blocks.erase(available_block);
    } else {
      result = res->do_allocate(bytes, alignment);
      blockSize = bytes;
    }
    allocated_blocks.insert(
        std::make_pair(thrust::raw_pointer_cast(result), blockSize));
    return result;
  }

  void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) {
    auto block = allocated_blocks.find(thrust::raw_pointer_cast(p));
    if (block == allocated_blocks.end()) {
      throw std::system_error(EFAULT, std::generic_category(),
                              "Address is not handled by this instance.");
    }
    std::ptrdiff_t num_bytes = block->second;
    allocated_blocks.erase(block);
    free_blocks.insert(std::make_pair(num_bytes, thrust::raw_pointer_cast(p)));
  }

  bool
  do_is_equal(const pool_memory_resource_adaptor<MR> &other) const noexcept {
    return res->do_is_equal(other);
  }

  bool has_allocated_blocks() const noexcept {
    return allocated_blocks.size() > 0;
  }

  void free_all() {
    for (auto &i : free_blocks)
      res->do_deallocate(static_cast<pointer>(i.second), i.first, 0);
    for (auto &i : allocated_blocks)
      res->do_deallocate(static_cast<pointer>(i.first), i.second, 0);
    free_blocks.clear();
    allocated_blocks.clear();
  }
};

namespace detail {
// Takes a pointer type (including smart pointers) and returns a reference to
// the underlying type
template <class T> struct pointer_to_lvalue_reference {
private:
  using element_type = typename std::pointer_traits<T>::element_type;

public:
  using type = typename std::add_lvalue_reference<element_type>::type;
};

// Specialization for special thrust pointer/reference types...
template <class T> struct pointer_to_lvalue_reference<detail::cuda_ptr<T>> {
  using type = thrust::system::cuda::reference<T>;
};

template <class T> struct non_void_value_type {
  using type = T;
};
template <> struct non_void_value_type<void> {
  using type = char;
};

} // namespace detail

// An allocator that can be used for any type using the same underlying
// memory_resource. pointer type can be specified to work with thrust cuda
// pointers
template <class T, class MR, class void_pointer = T *>
class polymorphic_allocator {
  MR *res;

public:
  // C++17 definitions for allocator interface
  using size_type = std::size_t;

  using value_type = T;
  using value_size_type =
      typename detail::non_void_value_type<value_type>::type;

  // All of the traits below are deprecated in C++17, but thrust counts on them
  // using void_pointer = T*;
  using pointer =
      typename std::pointer_traits<void_pointer>::template rebind<value_type>;

  using reference = typename detail::pointer_to_lvalue_reference<pointer>::type;
  using const_reference = typename detail::pointer_to_lvalue_reference<
      std::add_const<pointer>>::type;

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  template <class Other>
  polymorphic_allocator(Other other) : res(other.resource()) {}

  polymorphic_allocator(MR *resource) : res(resource) {}

  polymorphic_allocator() : polymorphic_allocator(get_default_resource<MR>()) {}

  MR *resource() const { return this->res; }

  pointer allocate(size_type n) const {
    return static_cast<pointer>(
        static_cast<value_type *>(this->res->do_allocate(
            n * sizeof(value_size_type), alignof(value_size_type))));
  }

  void deallocate(pointer p, size_type n = 0) const {
    return this->res->do_deallocate(thrust::raw_pointer_cast(p),
                                    n * sizeof(value_size_type),
                                    alignof(value_size_type));
  }
};

template <class T, class MR>
using polymorphic_cached_allocator =
    polymorphic_allocator<T, pool_memory_resource_adaptor<MR>>;
template <class T>
using device_cached_allocator =
    polymorphic_cached_allocator<T, device_memory_resource>;
template <class T>
using managed_cached_allocator =
    polymorphic_cached_allocator<T, managed_memory_resource>;
template <class T>
using host_cached_allocator =
    polymorphic_cached_allocator<T, host_memory_resource>;
template <class T>
using thrust_cached_allocator =
    polymorphic_allocator<T, device_memory_resource, detail::cuda_ptr<T>>;
} // namespace allocator
} // namespace libmobility
