/* Raul P. Pelaez 2021-2025. The libMobility interface.
   Every mobility implement must inherit from the Mobility virtual base class.
   See solvers/SelfMobility for a simple example
 */
#ifndef MOBILITYINTERFACE_H
#define MOBILITYINTERFACE_H
#include "defines.h"
#include "lanczos/lanczos.h"
#include "memory/container.h"
#include "third_party/uammd/src/third_party/saruprng.cuh"
#include <random>
#include <stdexcept>
#include <thrust/transform.h>
#include <vector>
namespace libmobility {

struct SaruNoise {
  uint seed1;
  uint seed2;
  SaruNoise(uint s1, uint s2) : seed1(s1), seed2(s2) {}

  __device__ real operator()(uint i) {
    Saru s(seed1, seed2, i);
    return s.gf(0.0f, 1.0f).x;
  }
};

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
  real temperature = 0;
  real tolerance = 1e-4; // Tolerance for Lanczos fluctuations
  int numberParticles = -1;
  std::uint64_t seed = 0;
  bool needsTorque = false;
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
  int numberParticles;
  bool initialized = false;
  std::uint64_t lanczosSeed;
  real lanczosTolerance;
  // std::shared_ptr<LanczosStochasticVelocities> lanczos;
  lanczos::Lanczos<real> lanczos_solver;
  thrust::device_vector<real> lanczosOutput, lanczosNoise;
  real temperature;
  bool needsTorque = false;
  void fillGaussianNoise(size_t elements) {
    static int ncalls = 0;
    lanczosNoise.resize(elements);
    auto cit = thrust::make_counting_iterator<uint>(0);
    // thrust::transform(thrust::cuda::par, cit,
    //                   cit + elements,
    //                   lanczosNoise.begin(),
    // 		      SaruNoise(lanczosSeed, ncalls));
    ncalls++;
  }

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
    this->initialized = true;
    this->numberParticles = par.numberParticles;
    this->lanczosSeed = par.seed;
    this->lanczosTolerance = par.tolerance;
    this->temperature = par.temperature;
    this->needsTorque = par.needsTorque;
  }

  // Set the positions to construct the mobility operator from
  virtual void setPositions(device_span<const real> positions) = 0;

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
    if (this->temperature == 0)
      return;
    if (prefactor == 0)
      return;
    if (not this->initialized)
      throw std::runtime_error(
          "[libMobility] You must initialize the base class in order to use "
          "the default stochastic displacement computation");
    lanczosOutput.resize(3 * this->numberParticles +
                         (this->needsTorque ? 3 * this->numberParticles : 0));
    if (this->needsTorque && iangular.empty())
      throw std::runtime_error("[libMobility] This solver requires angular "
                               "velocities when configured with torques");
    thrust::fill(lanczosOutput.begin(), lanczosOutput.end(), 0);
    device_adapter<real> linear(ilinear, device::cuda);
    device_adapter<real> angular(iangular, device::cuda);
    std::cerr << "Using Lanczos" << std::endl;
    auto dot = [this](device_span<const real> v, device_span<real> mv) {
      // Torques are stored at the end of the force array
      // After, results are separated into linear and angular velocities
      const int N = this->numberParticles;
      const real *t = this->needsTorque ? (v.data() + 3 * N) : nullptr;
      real *mt = this->needsTorque ? (mv.data() + 3 * N) : nullptr;
      auto dev = device::cuda;
      device_span<const real> s_t({t, t + (t ? (3 * N) : 0)}, dev);
      device_span<real> s_mt({mt, mt + (mt ? (3 * N) : 0)}, dev);
      device_span<const real> s_f({v.data(), v.data() + 3 * N}, dev);
      device_span<real> s_mv({mv.data(), mv.data() + 3 * N}, dev);
      Mdot(s_f, s_t, s_mv, s_mt);
    };
    device_span<real> output(lanczosOutput);
    this->fillGaussianNoise(lanczosOutput.size());
    device_span<const real> noise_span(lanczosNoise);
    lanczos_solver(dot, noise_span, output, lanczosTolerance);
    auto l_ptr = thrust::cuda::pointer<real>(linear.data());
    thrust::copy(thrust::cuda::par, lanczosOutput.begin(),
                 lanczosOutput.begin() + 3 * this->numberParticles, l_ptr);
    if (this->needsTorque) {
      auto a_ptr = thrust::cuda::pointer<real>(angular.data());
      thrust::copy(thrust::cuda::par,
                   lanczosOutput.begin() + 3 * this->numberParticles,
                   lanczosOutput.end(), a_ptr);
    }
  }

  // Equivalent to calling Mdot and then stochasticDisplacements, can be faster
  // in some solvers
  virtual void hydrodynamicVelocities(device_span<const real> forces,
                                      device_span<const real> torques,
                                      device_span<real> linear,
                                      device_span<real> angular,
                                      real prefactor = 1) {
    if (!forces.empty() or !torques.empty()) {
      Mdot(forces, torques, linear, angular);
    }
    sqrtMdotW(linear, angular, prefactor);
  }

  int getNumberParticles() const { return this->numberParticles; }

  bool getNeedsTorque() const { return this->needsTorque; }

  // Clean any memory allocated by the solver
  virtual void clean() { lanczosOutput.clear(); }
};
} // namespace libmobility
#endif
