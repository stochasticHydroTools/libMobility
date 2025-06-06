/*Raul P. Pelaez 2022. SelfMobility example sovler.

  This solver ignores hydrodynamic interactions. The mobility is the identity
  matrix scaled with 1/(6pi*eta*a). This is a simple example on how to implement
  a new solver. Note that is a purely CPU implementation.

 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include <MobilityInterface/MobilityInterface.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

class SelfMobility : public libmobility::Mobility {
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  template <class T> using device_span = libmobility::device_span<T>;
  Parameters par;
  // std::vector<real> positions;
  real linearMobility;
  real angularMobility;
  real temperature;
  int numberParticles = 0;
  std::mt19937 rng;

public:
  SelfMobility(Configuration conf) {
    if (conf.periodicityX != periodicity_mode::open or
        conf.periodicityY != periodicity_mode::open or
        conf.periodicityZ != periodicity_mode::open)
      throw std::runtime_error("[Mobility] This is an open boundary solver");
  }

  void initialize(Parameters ipar) override {
    auto seed = ipar.seed;
    if (not seed)
      seed = std::random_device()();
    this->rng = std::mt19937{seed};
    this->temperature = ipar.temperature;
    real hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->linearMobility =
        1.0 / (6 * M_PI * ipar.viscosity * hydrodynamicRadius);
    this->angularMobility =
        1.0 / (8 * M_PI * ipar.viscosity * hydrodynamicRadius *
               hydrodynamicRadius * hydrodynamicRadius);
    Mobility::initialize(ipar);
  }

  // An example of how to take in extra parameters. This function is supposed to
  // be called BEFORE initialize
  void setParametersSelfMobility(real some_unnecesary_parameter) {}

  void setPositions(device_span<const real> ipositions) override {
    this->numberParticles = ipositions.size() / 3;
    // positions.assign(ipositions.begin(), ipositions.end());
  }

  uint getNumberParticles() override { return this->numberParticles; }

  void Mdot(device_span<const real> iforces, device_span<const real> itorques,
            device_span<real> ilinear, device_span<real> iangular) override {
    if (!iforces.empty()) {
      int numberParticles = getNumberParticles();
      auto forces =
          libmobility::device_adapter(iforces, libmobility::device::cpu);
      auto linear =
          libmobility::device_adapter(ilinear, libmobility::device::cpu);
      for (int i = 0; i < 3 * numberParticles; i++) {
        linear[i] = forces[i] * linearMobility;
      }
    }
    if (!itorques.empty()) {
      int numberParticles = getNumberParticles();
      auto torques =
          libmobility::device_adapter(itorques, libmobility::device::cpu);
      auto angular =
          libmobility::device_adapter(iangular, libmobility::device::cpu);
      for (int i = 0; i < 3 * numberParticles; i++) {
        angular[i] = torques[i] * angularMobility;
      }
    }
  }

  // If this function is not present the default behavior is invoked, which uses
  // the Lanczos algorithm
  void sqrtMdotW(device_span<real> ilinear, device_span<real> iangular,
                 real prefactor = 1) override {
    std::normal_distribution<real> d{0, 1};
    auto linear =
        libmobility::device_adapter(ilinear, libmobility::device::cpu);
    if (!linear.empty()) {
      int numberParticles = getNumberParticles();
      for (int i = 0; i < 3 * numberParticles; i++) {
        real dW = d(rng);
        linear[i] += prefactor * sqrt(2 * temperature * linearMobility) * dW;
      }
    }
    if (!iangular.empty()) {
      int numberParticles = getNumberParticles();
      auto angular =
          libmobility::device_adapter(iangular, libmobility::device::cpu);
      for (int i = 0; i < 3 * numberParticles; i++) {
        real dW = d(rng);
        angular[i] += prefactor * sqrt(2 * temperature * angularMobility) * dW;
      }
    }
  }
};
#endif
