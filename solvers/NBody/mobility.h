/* Raul P. Pelaez 2022-2025. NBody libMobility solver.

 */
#ifndef MOBILITY_NBODY_H
#define MOBILITY_NBODY_H
#include "extra/interface.h"
#include <MobilityInterface/MobilityInterface.h>
#include <MobilityInterface/random_finite_differences.h>
#include <cmath>
#include <optional>
#include <type_traits>
#include <vector>

class NBody : public libmobility::Mobility {
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  using device = libmobility::device;
  template <class T> using device_span = libmobility::device_span<T>;
  template <class T> using device_adapter = libmobility::device_adapter<T>;
  nbody_rpy::kernel_type kernel;
  thrust::device_vector<real> positions;
  real transMobility;
  real rotMobility;
  real transRotMobility;
  real hydrodynamicRadius;
  nbody_rpy::algorithm algorithm = nbody_rpy::algorithm::advise;

  real wallHeight; // location of the wall in z

  // Batched functionality configuration
  int Nbatch;
  int NperBatch;

  real temperature;
  std::mt19937 rng;

public:
  NBody(Configuration conf) {
    if (conf.periodicityX != periodicity_mode::open or
        conf.periodicityY != periodicity_mode::open)
      throw std::runtime_error("[Mobility] NBody must be open in the plane");
    if (conf.periodicityZ == periodicity_mode::open)
      this->kernel = nbody_rpy::kernel_type::open_rpy;
    else if (conf.periodicityZ == periodicity_mode::single_wall)
      this->kernel = nbody_rpy::kernel_type::bottom_wall;
    else
      throw std::runtime_error("[Mobility] Invalid periodicity");
  }

  struct NBodyParameters {
    nbody_rpy::algorithm algo = nbody_rpy::algorithm::advise;
    int Nbatch = -1;
    int NperBatch = -1;
    std::optional<real> wallHeight = std::nullopt;
  };
  /**
   * @brief Sets the parameters for the N-body computation
   *
   * NBody allows for different strategies: Naive, Block and Fast. The algorithm
   * is selected through the "algo" parameter. The possible values are:
   * - fast: Fast multipole method
   * - naive: Direct computation
   * - block: Block-based computation
   * - advise: Automatically selects the best strategy based on input
   *
   * NBody can work on "batches" of particles, where all batches must have the
   * same size. A single batch represents full particle interactions within that
   * batch - only the mobility matrix elements corresponding to pairs within the
   * same batch are non-zero. This is equivalent to computing NPerBatch^2
   * matrix-vector products for each batch separately.
   *
   * The data layout uses 3 interleaved coordinates with consecutive batches:
   * [x_1_1, y_1_1, z_1_1, ..., x_1_NperBatch, ..., x_Nbatches_NperBatch]
   *
   * @param par The NBody parameters struct containing algorithm choice and
   * batch settings
   */
  void setParametersNBody(NBodyParameters par) {
    this->algorithm = par.algo;
    this->Nbatch = par.Nbatch;
    this->NperBatch = par.NperBatch;
    if (kernel == nbody_rpy::kernel_type::bottom_wall) {
      if (par.wallHeight) {
        this->wallHeight = par.wallHeight.value();
      } else {
        throw std::runtime_error(
            "[Mobility] Wall height parameter is required for a bottom wall. "
            "If you want to use a wall, set the wallHeight parameter.");
      }
    } else if (par.wallHeight) {
      throw std::runtime_error(
          "[Mobility] Wall height parameter is only valid for bottom wall. If "
          "you want to use a wall, set periodicityZ to single_wall in the "
          "configuration.");
    }
  }

  void initialize(Parameters ipar) override {
    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->transMobility =
        1.0 / (6 * M_PI * ipar.viscosity * hydrodynamicRadius);
    this->transRotMobility = 1.0 / (8 * M_PI * ipar.viscosity *
                                    hydrodynamicRadius * hydrodynamicRadius);
    this->rotMobility = 1.0 / (8 * M_PI * ipar.viscosity * hydrodynamicRadius *
                               hydrodynamicRadius * hydrodynamicRadius);
    this->temperature = ipar.temperature;
    if (ipar.seed == 0) {
      ipar.seed = std::random_device()();
    }
    this->rng = std::mt19937(ipar.seed);
    ipar.seed = rng();

    Mobility::initialize(ipar);
  }

  void setPositions(device_span<const real> ipositions) override {
    positions.assign(ipositions.begin(), ipositions.end());
    const auto numberParticles = this->getNumberParticles();
    int i_Nbatch = (this->Nbatch < 0) ? 1 : this->Nbatch;
    int i_NperBatch = (this->NperBatch < 0) ? numberParticles : this->NperBatch;
    if (i_NperBatch * i_Nbatch != numberParticles)
      throw std::runtime_error("[Mobility] Invalid batch parameters for NBody. "
                               "If in doubt, use the defaults.");
  }

  uint getNumberParticles() override { return this->positions.size() / 3; }

  void Mdot(device_span<const real> forces, device_span<const real> torques,
            device_span<real> linear, device_span<real> angular) override {
    const auto numberParticles = this->getNumberParticles();
    int i_Nbatch = (this->Nbatch < 0) ? 1 : this->Nbatch;
    int i_NperBatch = (this->NperBatch < 0) ? numberParticles : this->NperBatch;
    if (numberParticles <= 0)
      throw std::runtime_error(
          "[Mobility] Positions have 0 particles. Did you call "
          "setPositions?");
    using device_vector = thrust::device_vector<
        real, libmobility::allocator::thrust_cached_allocator<real>>;
    device_vector posZ(positions);
    if (wallHeight != 0) { // shifts positions so the wall is at z=0 since the
                           // kernels are programmed as such.
      using namespace thrust::placeholders;
      auto index_3 = thrust::make_transform_iterator(
          thrust::make_counting_iterator(0), _1 * 3);
      auto iposition =
          thrust::make_permutation_iterator(positions.begin() + 2, index_3);
      auto opositionZ =
          thrust::make_permutation_iterator(posZ.begin() + 2, index_3);
      thrust::transform(thrust::cuda::par, iposition,
                        iposition + numberParticles, opositionZ,
                        _1 - wallHeight);
    }
    device_span<const real> pos(posZ);
    nbody_rpy::callBatchedNBody(pos, forces, torques, linear, angular, i_Nbatch,
                                i_NperBatch, transMobility, rotMobility,
                                transRotMobility, hydrodynamicRadius,
                                this->getIncludeAngular(), algorithm, kernel);
  }

  void thermalDrift(device_span<real> ilinear, device_span<real> iangular,
                    real prefactor = 1) override {
    if (temperature == 0 || prefactor == 0) {
      return;
    }
    if (this->kernel == nbody_rpy::kernel_type::open_rpy)
      return; // No wall, no thermal drift
    const auto numberParticles = this->getNumberParticles();
    if (ilinear.size() != 3 * numberParticles) {
      throw std::runtime_error(
          "[libMobility] The number of forces does not match the "
          "number of particles");
    }
    if (this->getIncludeAngular() && iangular.size() != 3 * numberParticles) {
      throw std::runtime_error(
          "[libMobility] The number of torques does not match the "
          "number of particles");
    }
    if (!this->getIncludeAngular() && iangular.size() != 0) {
      throw std::runtime_error("[libMobility] Received torques but the solver "
                               "was initialized with includeAngular=False");
    }
    using device_vector = thrust::device_vector<
        real, libmobility::allocator::thrust_cached_allocator<real>>;
    const auto stored_positions =
        thrust::device_ptr<const real>(positions.data());
    device_vector original_pos(stored_positions,
                               stored_positions + 3 * numberParticles);
    auto mdot = [this](device_span<const real> positions,
                       device_span<const real> v, device_span<real> result_m,
                       device_span<real> result_d) {
      this->setPositions(positions);
      this->Mdot(v, device_span<const real>(), result_m, result_d);
    };
    uint seed = rng();
    device_vector thermal_drift_m(ilinear.size(), 0);
    device_vector thermal_drift_d(iangular.size(), 0);

    libmobility::random_finite_differences(mdot, original_pos, thermal_drift_m,
                                           thermal_drift_d, seed,
                                           prefactor * temperature);
    device_adapter<real> linear(ilinear, device::cuda);
    this->setPositions(original_pos);
    thrust::transform(thrust::cuda::par, thermal_drift_m.begin(),
                      thermal_drift_m.end(), linear.begin(), linear.begin(),
                      thrust::plus<real>());
    if (this->getIncludeAngular()) {
      device_adapter<real> angular(iangular, device::cuda);
      thrust::transform(thrust::cuda::par, thermal_drift_d.begin(),
                        thermal_drift_d.end(), angular.begin(), angular.begin(),
                        thrust::plus<real>());
    }
  }
};
#endif
