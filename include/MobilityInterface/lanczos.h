/*Raul P. Pelaez 2022. Interface between libMobility and LanczosAlgorithm

 */
#ifndef LIBMOBILITY_LANCZOS_ADAPTOR_H
#define LIBMOBILITY_LANCZOS_ADAPTOR_H
#define CUDA_ENABLED
#include "lanczos/LanczosAlgorithm.h"
#include "third_party/saruprng.cuh"
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace detail {
using real = lanczos::real;
struct SaruFill {
  uint seed1, seed2;
  __device__ real operator()(uint id) {
    Saru prng(seed1, seed2, id);
    return prng.gf(real(0), real(1.0)).x;
  }
};

} // namespace detail
// This class uses the LanczosAlgorithm library to compute fluctuations.
class LanczosStochasticVelocities {
  using real = lanczos::real;
  lanczos::Solver lanczos;
  // std::vector<real> lanczosNoise;
  thrust::device_vector<real> lanczosNoise;
  real lanczosTolerance;
  std::mt19937 engine;

public:
  LanczosStochasticVelocities(real tol, std::uint64_t seed) {
    this->lanczosTolerance = tol;
    engine = std::mt19937{seed};
  }

  // Given a functor that applies the mobility operator, returns prefactor*(B
  // dW). Where B is an operator that applies the square root of the provided
  // mobility.
  template <class MobilityDot>
  void sqrtMdotW(MobilityDot idot, real *result, int numberParticles,
                 std::function<void(int, float)> callback) {
    lanczosNoise.resize(3 * numberParticles);
    // std::generate(lanczosNoise.begin(), lanczosNoise.end(), gen);
    uint seed1 = std::uniform_int_distribution<uint>(0, UINT32_MAX)(engine);
    uint seed2 = std::uniform_int_distribution<uint>(0, UINT32_MAX)(engine);
    auto cit = thrust::make_counting_iterator<uint>(0);
    thrust::transform(cit, cit + 3 * numberParticles, lanczosNoise.begin(),
                      detail::SaruFill{seed1, seed2});
    std::function<void(real *, real *)> dot = [&](real *f, real *mv) {
      idot(f, mv);
    };
    lanczos.run(dot, result, lanczosNoise.data().get(), lanczosTolerance,
                3 * numberParticles, callback);
  }
};

#endif
