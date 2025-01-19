#include "gtest/gtest.h"
#include <vector>
#include "memory/allocator.h"
#include "memory/container.h"
#include "thrust/device_vector.h"
using namespace libmobility;
using namespace libmobility::allocator;
TEST(Alloc, CPU) {
  std::vector<float, host_cached_allocator<float>> a(100, 1.0f);
  std::vector<float, host_cached_allocator<float>> b(100, 2.0f);
  ASSERT_EQ(a[0], 1.0f);
}

TEST(Alloc, cudamalloc){
  char* iptr = nullptr;
  size_t size = 400;
  int st = cudaMalloc(&iptr, size);
  ASSERT_EQ(st, cudaSuccess);
  ASSERT_NE(iptr, nullptr);
  cudaFree(iptr);
}
TEST(Alloc, device_resource) {
  device_memory_resource alloc;
  auto ptr = alloc.do_allocate(1, 0);
  alloc.do_deallocate(ptr, 0, 0 );
}

TEST(Alloc, device_pool){
  pool_memory_resource_adaptor<device_memory_resource> pool;
  auto ptr = pool.do_allocate(1,0);
  pool.do_deallocate(ptr, 1,0);
}

TEST(Alloc, thrust_allocator){
  thrust::device_vector<float, thrust_cached_allocator<float>> a(100, 1.0f);
}

TEST(Alloc, host_cached_allocator){
  std::vector<float, host_cached_allocator<float>> a(100, 1.0f);
  a.push_back(2.0f);
  a.resize(100);
  std::fill(a.begin(), a.end(), 3.0f);
}

TEST(Container, span){
  std::vector<float> a(100, 1.0f);
  device_span<float> s(a, device::cpu);

  ASSERT_EQ(s[0], 1.0f);
  ASSERT_EQ(s.size(), a.size());
}

TEST(Container, adapterHostToHost){
  std::vector<float> a(100, 1.0f);
  device_span<float> s(a, device::cpu);
  {
    device_adapter<float> d(s,device::cpu);

    ASSERT_EQ(d[0], 1.0f);
    ASSERT_EQ(d.size(), a.size());
    ASSERT_EQ(d.dev, device::cpu);
    std::fill(d.begin(), d.end(), 2.0f);
  }
  ASSERT_EQ(a[0], 2.0f);
  ASSERT_EQ(a.size(), 100);
}

TEST(Container, adapterHostToDevice){
  std::vector<float> a(100, 1.0f);
  device_span<float> s(a, device::cpu);
  {
    device_adapter<float> adap(s, device::cuda);
    ASSERT_EQ(adap.dev, device::cuda);
    ASSERT_EQ(adap.size(), a.size());
    thrust::fill(thrust::cuda::par, adap.begin(), adap.end(), 2.0f);
  }
  ASSERT_EQ(a[0], 2.0f);
  ASSERT_EQ(a.size(), 100);
}

TEST(Container, adapterDeviceToHost){
  thrust::device_vector<float> a(100, 1.0f);
  device_span<float> s({a.data().get(), a.data().get()+a.size()}, device::cuda);
  {
    device_adapter<float> adap(s, device::cpu);
    ASSERT_EQ(adap.dev, device::cpu);
    ASSERT_EQ(adap.size(), a.size());
    std::fill(adap.begin(), adap.end(), 2.0f);
  }
  ASSERT_EQ(a[0], 2.0f);
  ASSERT_EQ(a.size(), 100);
}

TEST(Container, adapterDeviceToDevice){
  thrust::device_vector<float> a(100, 1.0f);
  device_span<float> s({a.data().get(), a.data().get()+a.size()}, device::cuda);
  {
    device_adapter<float> adap(s, device::cuda);
    ASSERT_EQ(adap.dev, device::cuda);
    ASSERT_EQ(adap.size(), a.size());
    thrust::fill(thrust::cuda::par, adap.begin(), adap.end(), 2.0f);
  }
  ASSERT_EQ(a[0], 2.0f);
  ASSERT_EQ(a.size(), 100);
}

TEST(Container, adapterHostToDeviceConst){
  // Use const float span
  std::vector<float> a(100, 1.0f);
  device_span<const float> s(a, device::cpu);
  {
    device_adapter<const float> adap(s, device::cuda);
    ASSERT_EQ(adap.dev, device::cuda);
    ASSERT_EQ(adap.size(), a.size());
  }
  ASSERT_EQ(a[0], 1.0f);
}

TEST(Alloc, empty){
  // Try to deallocate an empty vector
  {
    std::vector<float, host_cached_allocator<float>> a;
  }
  {
    thrust::device_vector<float, thrust_cached_allocator<float>> a;
  }

  thrust_cached_allocator<float>alloc;
  alloc.deallocate(nullptr, 0);

}
