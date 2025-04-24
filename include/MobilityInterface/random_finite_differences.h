/*Raul P. Pelaez 2025. Random Finite Differences for computing thermal drift in
 * libMobility
 */
#pragma once

#include "defines.h"
#include "memory/container.h"
#include "third_party/saruprng.cuh"
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace libmobility {

void fill_with_random(device_span<real> input_vector, uint seed) {
  device_adapter<real> input_vector_cuda(input_vector, device::cuda);
  auto cit = thrust::make_counting_iterator<uint>(0);
  thrust::transform(thrust::cuda::par, cit, cit + input_vector_cuda.size(),
                    input_vector_cuda.begin(), [seed] __device__(uint i) {
                      Saru rng(seed, i);
                      return rng.gf(0, 1).x;
                    });
}

// Thermal drift is approximated by using Random Finite
// Differences: RFD works by approxmating:
// kT\partial_q\dot M = 1/\delta\langle M(q+\delta/2 W)W-M(q-\delta/2 W) W
// \rangle Where delta is a small number, and W is a normal random vector of
// unit length
template <typename Mdot>
void random_finite_differences(Mdot mdot, device_span<const real> positions,
                               device_span<real> ilinear, uint seed,
                               real prefactor = 1) {
  constexpr real delta = 1e-3;
  const uint N = ilinear.size() / 3;
  using device_vector =
      thrust::device_vector<real, allocator::thrust_cached_allocator<real>>;
  using namespace thrust::placeholders;
  device_vector noise(N * 3);
  device_vector pos_delta(N * 3);
  device_vector Mpd(N * 3, 0);
  device_vector Mmd(N * 3, 0);
  device_span<real> noise_span(noise);
  fill_with_random(noise_span, seed);
  thrust::transform(thrust::cuda::par, positions.begin(), positions.end(),
                    noise.begin(), pos_delta.begin(), _1 + (delta * 0.5) * _2);
  mdot(pos_delta, noise, Mpd);
  thrust::transform(thrust::cuda::par, positions.begin(), positions.end(),
                    noise.begin(), pos_delta.begin(), _1 - (delta * 0.5) * _2);
  mdot(pos_delta, noise, Mmd);
  device_adapter<real> linear(ilinear, device::cuda);
  thrust::transform(thrust::cuda::par, Mpd.begin(), Mpd.end(), Mmd.begin(),
                    linear.begin(), (prefactor / delta) * (_1 - _2));
}

} // namespace libmobility
