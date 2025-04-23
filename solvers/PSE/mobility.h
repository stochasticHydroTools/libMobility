// Raul P. Pelaez 2021-2022. libmobility interface for UAMMD PSE module
#ifndef MOBILITY_PSE_H
#define MOBILITY_PSE_H
#include "extra/uammd_interface.h"
#include <MobilityInterface/MobilityInterface.h>
#include <cmath>
#include <type_traits>
#include <vector>

static_assert(
    std::is_same<libmobility::real, uammd_pse::real>::value,
    "Trying to compile PSE with a different precision to MobilityInterface.h");

class PSE : public libmobility::Mobility {
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  using device = libmobility::device;
  template <class T> using device_span = libmobility::device_span<T>;
  template <class T> using device_adapter = libmobility::device_adapter<T>;

  thrust::device_vector<real> positions;
  std::shared_ptr<uammd_pse::UAMMD_PSE_Glue> pse;
  uammd_pse::PyParameters psepar, currentpsepar;
  uint currentNumberParticles = 0;
  real temperature;

public:
  PSE(Configuration conf) {
    if (conf.periodicityX != periodicity_mode::periodic or
        conf.periodicityY != periodicity_mode::periodic or
        conf.periodicityZ != periodicity_mode::periodic)
      throw std::runtime_error("[Mobility] This is a triply periodic solver");
  }

  // If the initialize function is called two times only changing the shear
  // strain the module is not reinitialized entirely
  void initialize(Parameters ipar) override {
    if (pse and onlyShearStrainChanged(ipar)) {
      pse->setShearStrain(psepar.shearStrain);
    } else {
      this->temperature = ipar.temperature;
      psepar.viscosity = ipar.viscosity;
      psepar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];
      psepar.tolerance = ipar.tolerance;
      Mobility::initialize(ipar);
    }
    currentpsepar = psepar;
    if (ipar.needsTorque)
      throw std::runtime_error("[PSE] Torque is not implemented");
  }

  struct PSEParameters {
    real psi, Lx, Ly, Lz, shearStrain;
  };

  void setParametersPSE(PSEParameters i_par) {
    psepar.psi = i_par.psi;
    psepar.Lx = i_par.Lx;
    psepar.Ly = i_par.Ly;
    psepar.Lz = i_par.Lz;
    psepar.shearStrain = i_par.shearStrain;
  }

  void setPositions(device_span<const real> ipositions) override {
    if (ipositions.size() / 3 != this->currentNumberParticles &&
        ipositions.size() > 0) {
      this->currentNumberParticles = ipositions.size() / 3;
      pse = std::make_shared<uammd_pse::UAMMD_PSE_Glue>(
          currentpsepar, this->currentNumberParticles);
    }
    positions.assign(ipositions.begin(), ipositions.end());
  }

  uint getNumberParticles() override { return this->currentNumberParticles; }

  void Mdot(device_span<const real> iforces, device_span<const real> itorques,
            device_span<real> linear, device_span<real> angular) override {
    if (this->getNumberParticles() <= 0)
      throw std::runtime_error(
          "[PSE] The number of particles is not set. Did you "
          "forget to call setPositions?");
    if (itorques.size())
      throw std::runtime_error("[PSE] Torque is not implemented");
    if (!pse)
      throw std::runtime_error("[PSE] PSE is not initialized. Did you "
                               "forget to call initialize?");
    if (linear.size() != 3 * currentNumberParticles)
      throw std::runtime_error(
          "[libMobility] The number of linear velocities does not match the "
          "number of particles");
    if (iforces.size() != 3 * currentNumberParticles)
      throw std::runtime_error(
          "[libMobility] The number of forces does not match the "
          "number of particles");
    pse->computeHydrodynamicDisplacements(positions.data().get(),
                                          iforces.data(), linear.data(), 0, 0);
  }

  void sqrtMdotW(device_span<real> linear, device_span<real> angular,
                 real prefactor = 1) override {
    if (this->getNumberParticles() <= 0)
      throw std::runtime_error(
          "[PSE] The number of particles is not set. Did you "
          "forget to call setPositions?");
    if (angular.size())
      throw std::runtime_error("[PSE] Torque is not implemented");
    if (positions.size() != 3 * currentNumberParticles)
      throw std::runtime_error(
          "[libMobility] Wrong number of particles in positions. Did you "
          "forget to call setPositions?");
    if (!pse)
      throw std::runtime_error("[libMobility] PSE is not initialized. Did you "
                               "forget to call initialize?");
    pse->computeHydrodynamicDisplacements(
        positions.data().get(), nullptr, linear.data(), temperature, prefactor);
  }

  virtual void hydrodynamicVelocities(device_span<const real> forces,
                                      device_span<const real> torques,
                                      device_span<real> linear,
                                      device_span<real> angular,
                                      real prefactor = 1) override {
    if (this->getNumberParticles() <= 0)
      throw std::runtime_error(
          "[PSE] The number of particles is not set. Did you "
          "forget to call setPositions?");
    if (angular.size())
      throw std::runtime_error("[PSE] Torque is not implemented");
    pse->computeHydrodynamicDisplacements(positions.data().get(), forces.data(),
                                          linear.data(), temperature,
                                          prefactor);
  }

private:
  bool onlyShearStrainChanged(Parameters i_par) {
    if (currentpsepar.psi != psepar.psi or currentpsepar.Lx != psepar.Lx or
        currentpsepar.Ly != psepar.Ly or currentpsepar.Lz != psepar.Lz)
      return false;
    if (this->temperature != i_par.temperature or
        psepar.viscosity != i_par.viscosity or
        psepar.hydrodynamicRadius != i_par.hydrodynamicRadius[0] or
        psepar.tolerance != i_par.tolerance)
      return false;
    return true;
  }
};
#endif
