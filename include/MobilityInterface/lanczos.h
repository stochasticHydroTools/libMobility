/*Raul P. Pelaez 2022. Interface between libMobility and LanczosAlgorithm

 */
#ifndef LIBMOBILITY_LANCZOS_ADAPTOR_H
#define LIBMOBILITY_LANCZOS_ADAPTOR_H
#include "LanczosAlgorithm.h"
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// This class uses the LanczosAlgorithm library to compute fluctuations.
class LanczosStochasticVelocities {
  using real = lanczos::real;
  lanczos::Solver lanczos;
  std::vector<real> lanczosNoise;
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
  void sqrtMdotW(MobilityDot dot, real *result, int numberParticles,
                 real prefactor = 1) {
    std::normal_distribution<real> dist{0, 1};
    auto gen = [&]() { return dist(engine); };
    lanczosNoise.resize(3 * numberParticles);
    std::generate(lanczosNoise.begin(), lanczosNoise.end(), gen);
    lanczos.run(dot, result, lanczosNoise.data(), lanczosTolerance,
                3 * numberParticles);
  }
};

#endif
