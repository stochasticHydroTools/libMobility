/* Raul P. Pelaez 2021-2025. The libMobility interface.
   Every mobility implement must inherit from the Mobility virtual base class.
   See solvers/SelfMobility for a simple example
 */
#ifndef MOBILITYINTERFACE_H
#define MOBILITYINTERFACE_H
#include "defines.h"
#include "lanczos.h"
#include "memory/container.h"
#include <random>
#include <stdexcept>
#include <vector>

namespace libmobility {

enum class periodicity_mode {
  single_wall,
  two_walls,
  open,
  periodic,
  unspecified
};

// Parameters that are proper to every solver.
struct Parameters {
  std::vector<real> hydrodynamicRadius;
  real viscosity = 1;
  real tolerance = 1e-4; // Tolerance for Lanczos fluctuations
  std::uint64_t seed = 0;
  bool includeAngular = false;
};

// A list of parameters that cannot be changed by reinitializing a solver and/or
// are properties of the solver. For instance, an open boundary solver will only
// accept open periodicity. Another solver might be set up for either cpu or gpu
// at creation
struct Configuration {
  periodicity_mode periodicityX = periodicity_mode::unspecified;
  periodicity_mode periodicityY = periodicity_mode::unspecified;
  periodicity_mode periodicityZ = periodicity_mode::unspecified;
};

// This is the virtual base class that every solver must inherit from.
class Mobility {
private:
  bool initialized = false;
  std::uint64_t lanczosSeed;
  real lanczosTolerance;
  std::shared_ptr<LanczosStochasticVelocities> lanczos;
  std::vector<real> lanczosOutput;
  bool includeAngular = false;
  std::mt19937 rng;

protected:
  Mobility() {};

public:
  // These constants are available to all solvers
  static constexpr auto version = LIBMOBILITYVERSION; // The interface version
#if defined SINGLE_PRECISION
  static constexpr auto precision = "float";
#else
  static constexpr auto precision = "double";
#endif
  // The constructor should accept a Configuration object and ensure the
  // requested parameters are acceptable (an open boundary solver should
  // complain if periodicity is selected). A runtime_exception should be thrown
  // if the configuration is invalid. The constructor here is just an example,
  // since this is a pure virtual class
  /*
  Mobility(Configuration conf){
    if(conf.periodicityX != periodicity::open or
    conf.periodicityY != periodicity::open or
    conf.periodicityZ != periodicity::open)
      throw std::runtime_error("[Mobility] This is an open boundary solver");
  }
  */
  // Outside of the common interface, solvers can define a function called
  // setParameters[ModuleName] , with arbitrary input, that simply acknowledges
  // a set of values proper to the specific solver. These new parameters should
  // NOT take effect until initialize is called afterwards.
  //  void setParametersModuleName(MyParameters par){
  //    //Store required parameters
  //  }

  // Initialize should leave the solver in a state ready for setPositions to be
  //  called. Furthermore, initialize can be called again if some parameter
  //  changes
  virtual void initialize(Parameters par) {
    // Clean if the solver was already initialized
    if (initialized)
      this->clean();
    if (par.seed == 0)
      par.seed = std::random_device()();
    this->rng = std::mt19937(par.seed);
    this->initialized = true;
    this->lanczosSeed = this->rng();
    this->lanczosTolerance = par.tolerance;
    this->includeAngular = par.includeAngular;
  }

  // Set the positions to construct the mobility operator from
  virtual void setPositions(device_span<const real> positions) = 0;

  // Return the current number of particles in the system
  virtual uint getNumberParticles() = 0;

  // Apply the grand mobility operator (\Omega) to a series of forces (F) and
  // torques (T) to get the resulting linear (V) and angular (W) velocities
  virtual void Mdot(device_span<const real> forces,
                    device_span<const real> torques, device_span<real> linear,
                    device_span<real> angular) = 0;

  // Compute the stochastic displacements as result=prefactor*sqrt(\Omega)*dW.
  // Where dW is a vector of Gaussian random numbers If the solver does not
  // provide a stochastic displacement implementation, the Lanczos algorithm
  // will be used automatically
  virtual void sqrtMdotW(device_span<real> ilinear, device_span<real> iangular,
                         real prefactor = 1) {
    if (prefactor == 0)
      return;
    if (not this->initialized)
      throw std::runtime_error(
          "[libMobility] You must initialize the base class in order to use "
          "the default stochastic displacement computation");
    const auto numberParticles = this->getNumberParticles();
    if (numberParticles <= 0)
      throw std::runtime_error(
          "[libMobility] The number of particles is not set. Did you "
          "forget to call setPositions?");
    device_adapter<real> linear(ilinear, device::cpu);
    device_adapter<real> angular(iangular, device::cpu);
    if (linear.empty())
      throw std::runtime_error(
          "[libMobility] This solver requires linear velocities");
    if (linear.size() / 3 != numberParticles) {
      throw std::runtime_error(
          "[libMobility] The number of linear velocities does not match the "
          "number of particles");
    }
    const auto numberElements =
        numberParticles + (this->includeAngular ? numberParticles : 0);
    if (not lanczos) {
      lanczos = std::make_shared<LanczosStochasticVelocities>(
          this->lanczosTolerance, this->lanczosSeed);
    }
    lanczosOutput.resize(3 * numberElements);
    std::fill(lanczosOutput.begin(), lanczosOutput.end(), 0);
    auto dev = linear.dev;
    lanczos->sqrtMdotW(
        [this, dev, numberParticles](const real *f, real *mv) {
          // Torques are stored at the end of the force array
          // After, results are separated into linear and angular velocities
          const auto N = numberParticles;
          const real *t = this->includeAngular ? (f + 3 * N) : nullptr;
          real *mt = this->includeAngular ? (mv + 3 * N) : nullptr;
          device_span<const real> s_t({t, t + (t ? (3 * N) : 0)}, dev);
          device_span<real> s_mt({mt, mt + (mt ? (3 * N) : 0)}, dev);
          device_span<const real> s_f({f, f + 3 * N}, dev);
          device_span<real> s_mv({mv, mv + 3 * N}, dev);
          Mdot(s_f, s_t, s_mv, s_mt);
        },
        lanczosOutput.data(), numberElements, prefactor);
    std::transform(lanczosOutput.begin(),
                   lanczosOutput.begin() + 3 * numberParticles, linear.begin(),
                   linear.begin(), thrust::plus<real>());
    if (this->includeAngular)
      std::transform(lanczosOutput.begin() + 3 * numberParticles,
                     lanczosOutput.end(), angular.begin(), angular.begin(),
                     thrust::plus<real>());
  }


  // computes velocities according to the Langevin equation.
  virtual void LangevinVelocities(device_span<const real> forces,
                                      device_span<const real> torques,
                                      device_span<real> linear,
                                      device_span<real> angular,
                                      real dt, real kbt){
    if (!forces.empty() or !torques.empty()) {
      Mdot(forces, torques, linear, angular);
    }

    // prefactors (last argument) are chosen for dimensional consistency
    if (kbt != 0){
      const real sqrt_prefactor = std::sqrt(2 * kbt / dt);
      sqrtMdotW(linear, angular, sqrt_prefactor);
      divM(linear, angular, kbt);
    }
  }

  // Compute the divergence of the mobility matrix,
  // :math:`\boldsymbol{\partial}_\boldsymbol{x}\cdot
  // \boldsymbol{\mathcal{M}}`.
  virtual void divM(device_span<real> ilinear, device_span<real> angular,
                    real prefactor = 1) {
    // For open and periodic solvers the thermal drift is zero, so this function
    // does nothing by default.
  }

  bool getIncludeAngular() const { return this->includeAngular; }

  // Clean any memory allocated by the solver
  virtual void clean() {
    lanczos.reset();
    lanczosOutput.clear();
  }
};
} // namespace libmobility
#endif
