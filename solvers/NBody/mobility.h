/* Raul P. Pelaez 2022-2025. NBody libMobility solver.

 */
#ifndef MOBILITY_NBODY_H
#define MOBILITY_NBODY_H
#include "extra/interface.h"
#include <MobilityInterface/MobilityInterface.h>
#include <cmath>
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
  int numberParticles;
  nbody_rpy::algorithm algorithm = nbody_rpy::algorithm::advise;

  // Batched functionality configuration
  int Nbatch;
  int NperBatch;

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
  }

  void initialize(Parameters ipar) override {
    this->numberParticles = ipar.numberParticles;
    if (Nbatch < 0)
      Nbatch = 1;
    if (NperBatch < 0)
      NperBatch = ipar.numberParticles;
    if (NperBatch * Nbatch != numberParticles)
      throw std::runtime_error("[Mobility] Invalid batch parameters for NBody. "
                               "If in doubt, use the defaults.");

    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->transMobility =
        1.0 / (6 * M_PI * ipar.viscosity * hydrodynamicRadius);
    this->transRotMobility = 1.0 / (8 * M_PI * ipar.viscosity *
                                    hydrodynamicRadius * hydrodynamicRadius);
    this->rotMobility = 1.0 / (8 * M_PI * ipar.viscosity * hydrodynamicRadius *
                               hydrodynamicRadius * hydrodynamicRadius);
    Mobility::initialize(ipar);
  }

  void setPositions(device_span<const real> ipositions) override {
    positions.assign(ipositions.begin(), ipositions.end());
  }

  void Mdot(device_span<const real> iforces, device_span<const real> itorques,
            device_span<real> linear, device_span<real> angular) override {
    device_adapter<const real> forces(iforces, device::cuda);
    device_adapter<const real> torques(itorques, device::cuda);
    int numberParticles = positions.size() / 3;
    if (numberParticles != this->numberParticles)
      throw std::runtime_error(
          "[libMobility] Wrong number of particles in positions. Did you "
          "forget to call setPositions?");
    device_span<const real> pos{{positions.data().get(), positions.size()},
                                libmobility::device::cuda};
    nbody_rpy::callBatchedNBody(
        pos, forces, torques, linear, angular, Nbatch, NperBatch, transMobility,
        rotMobility, transRotMobility, hydrodynamicRadius, algorithm, kernel);
  }
};
#endif
