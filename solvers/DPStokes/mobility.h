/*Raul P. Pelaez 2022-2025. libMobility interface for UAMMD's DPStokes module

References:
[1] Computing hydrodynamic interactions in confined doubly periodic geometries
in linear time. A. Hashemi et al. J. Chem. Phys. 158, 154101 (2023)
https://doi.org/10.1063/5.0141371
 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include "extra/poly_fits.h"
#include "extra/uammd_interface.h"
#include <MobilityInterface/MobilityInterface.h>
#include <MobilityInterface/random_finite_differences.h>
#include <cmath>
#include <vector>

class DPStokes : public libmobility::Mobility {
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using DPStokesParameters = uammd_dpstokes::PyParameters;
  using real = libmobility::real;
  using DPStokesUAMMD = uammd_dpstokes::DPStokesGlue;
  using device = libmobility::device;
  template <class T> using device_span = libmobility::device_span<T>;
  template <class T> using device_adapter = libmobility::device_adapter<T>;
  uint numberParticles = 0;
  Parameters par;
  std::shared_ptr<DPStokesUAMMD> dpstokes;
  DPStokesParameters dppar;
  real lanczosTolerance;
  std::string wallmode;
  std::mt19937 rng;

public:
  DPStokes(Configuration conf) {
    if (conf.periodicityX != periodicity_mode::periodic or
        conf.periodicityY != periodicity_mode::periodic or
        not(conf.periodicityZ == periodicity_mode::open or
            conf.periodicityZ == libmobility::periodicity_mode::single_wall or
            conf.periodicityZ == libmobility::periodicity_mode::two_walls))
      throw std::runtime_error("[DPStokes] This is a doubly periodic solver");
    if (conf.periodicityZ == periodicity_mode::open)
      wallmode = "nowall";
    else if (conf.periodicityZ == periodicity_mode::single_wall)
      wallmode = "bottom";
    else if (conf.periodicityZ == periodicity_mode::two_walls)
      wallmode = "slit";
  }

  void setParametersDPStokes(DPStokesParameters i_dppar) {
    this->dppar = i_dppar;
    dpstokes = std::make_shared<uammd_dpstokes::DPStokesGlue>();
  }

  void initialize(Parameters ipar) override {
    this->dppar.viscosity = ipar.viscosity;
    this->lanczosTolerance = ipar.tolerance;
    this->dppar.mode = this->wallmode;
    this->dppar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    if (ipar.seed == 0) {
      ipar.seed = std::random_device()();
    }
    this->rng = std::mt19937(ipar.seed);
    ipar.seed = rng();
    real h;
    if (ipar.includeAngular) {
      this->dppar.w = 6;
      this->dppar.w_d = 6;
      this->dppar.beta = {real(1.327) * this->dppar.w,
                          real(1.327) * this->dppar.w, -1.0};
      this->dppar.beta_d = {real(2.217) * this->dppar.w_d,
                            real(2.217) * this->dppar.w_d, -1.0};
      h = this->dppar.hydrodynamicRadius / 1.731;
      this->dppar.alpha_d = this->dppar.w_d * 0.5;
    } else {
      // w=4
      this->dppar.w = 4;
      this->dppar.beta = {real(1.785) * this->dppar.w,
                          real(1.785) * this->dppar.w, -1.0};
      h = this->dppar.hydrodynamicRadius / 1.205;

      // w=6
      // this->dppar.w = 6;
      // this->dppar.beta = {real(1.714) * this->dppar.w, real(1.714) *
      // this->dppar.w, -1.0};
      // h = this->dppar.hydrodynamicRadius / 1.554;
    }
    this->dppar.alpha = this->dppar.w * 0.5;
    this->dppar.tolerance = 1e-6;

    int Nx = floor(this->dppar.Lx / h);
    Nx += Nx % 2;
    int Ny = floor(this->dppar.Ly / h);
    Ny += Ny % 2;

    this->dppar.nx = Nx;
    this->dppar.ny = Ny;

    if (this->dppar.allowChangingBoxSize) { // adjust box size to suit h
      this->dppar.Lx = Nx * h;
      this->dppar.Ly = Ny * h;
    } else { // adjust h so that L/h is an integer
      real h_x = this->dppar.Lx / Nx;
      real h_y = this->dppar.Ly / Ny;
      h = min(h_x, h_y);
      double arg = this->dppar.hydrodynamicRadius / (this->dppar.w * h_x);
      real beta_x =
          dpstokes_polys::polyEval(dpstokes_polys::cbeta_monopole_inv, arg);
      arg = this->dppar.hydrodynamicRadius / (this->dppar.w * h_y);
      real beta_y =
          dpstokes_polys::polyEval(dpstokes_polys::cbeta_monopole_inv, arg);

      if (beta_x < 4.0 || beta_x > 18.0 || beta_y < 4.0 || beta_y > 18.0) {
        throw std::runtime_error(
            "[DPStokes] Could not find (h,beta) within interp range. This "
            "means the particle radius and grid spacing are incompatible- try "
            "a square domain.");
      }

      this->dppar.beta = {beta_x, beta_y, min(beta_x, beta_y)};

      if (ipar.includeAngular) {
        arg = this->dppar.hydrodynamicRadius / (this->dppar.w_d * h_x);
        real beta_xd =
            dpstokes_polys::polyEval(dpstokes_polys::cbeta_dipole_inv, arg);
        arg = this->dppar.hydrodynamicRadius / (this->dppar.w_d * h_y);
        real beta_yd =
            dpstokes_polys::polyEval(dpstokes_polys::cbeta_dipole_inv, arg);
        if (beta_xd < 4.0 || beta_xd > 18.0 || beta_yd < 4.0 ||
            beta_yd > 18.0) {
          throw std::runtime_error(
              "[DPStokes] Could not find (h,beta) within interp range. This "
              "means the particle radius and grid spacing are incompatible- "
              "try "
              "a square domain.");
        }
        this->dppar.beta_d = {beta_xd, beta_yd, min(beta_xd, beta_yd)};
      }
    }

    // Add a buffer of 1.5*w*h/2 when there is an open boundary
    if (this->wallmode == "nowall") {
      this->dppar.zmax += 1.5 * this->dppar.w * h / 2;
      this->dppar.zmin -= 1.5 * this->dppar.w * h / 2;
    }
    if (this->wallmode == "bottom") {
      this->dppar.zmax += 1.5 * this->dppar.w * h / 2;
    }
    real Lz = this->dppar.zmax - this->dppar.zmin;
    if (Lz <= 2 * this->dppar.hydrodynamicRadius) {
      throw std::runtime_error("[DPStokes] The box size in z is too small to "
                               "fit the particles. Try increasing zmax.");
    }
    real H = Lz / 2;
    // sets chebyshev node spacing at its coarsest (in the middle) to be h
    real nz_actual = M_PI / (asin(h / H)) + 1;

    // pick nearby N such that 2(Nz-1) has two factors of 2 and is FFT friendly
    this->dppar.nz = (int)floor(nz_actual);
    this->dppar.nz += (this->dppar.nz - 1) % 2;

    dpstokes->initialize(dppar);
    Mobility::initialize(ipar);
  }

  void setPositions(device_span<const real> ipositions) override {
    this->numberParticles = ipositions.size() / 3;
    device_adapter<const real> positions(ipositions, device::cuda);
    dpstokes->setPositions(positions.data(), this->numberParticles);
  }

  uint getNumberParticles() override { return this->numberParticles; }

  void Mdot(device_span<const real> iforces, device_span<const real> itorques,
            device_span<real> ilinear, device_span<real> iangular) override {
    if (this->numberParticles <= 0) {
      throw std::runtime_error("[libMobility] Positions are not set. Did you "
                               "forget to call setPositions?");
    }
    device_adapter<const real> forces(iforces, device::cuda);
    device_adapter<const real> torques(itorques, device::cuda);
    device_adapter<real> linear(ilinear, device::cuda);
    device_adapter<real> angular(iangular, device::cuda);

    dpstokes->Mdot(forces.data(), torques.data(), linear.data(), angular.data(),
                   this->getNumberParticles(), this->getIncludeAngular());
  }

  void divM(device_span<real> ilinear, device_span<real> iangular,
            real prefactor = 1) override {
    if (prefactor == 0) {
      return;
    }
    if (ilinear.size() != 3 * this->numberParticles) {
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
    device_adapter<real> linear(ilinear, device::cuda);
    using device_vector = thrust::device_vector<
        real, libmobility::allocator::thrust_cached_allocator<real>>;
    const auto stored_positions =
        thrust::device_ptr<const real>(dpstokes->getStoredPositions());
    device_vector original_pos(stored_positions,
                               stored_positions + 3 * this->numberParticles);
    auto mdot = [this](device_span<const real> positions,
                       device_span<const real> v, device_span<real> result_m,
                       device_span<real> result_d) {
      this->setPositions(positions);
      this->Mdot(v, device_span<const real>(), result_m, result_d);
    };
    uint seed = rng();
    device_vector thermal_drift_m(ilinear.size(), 0);
    device_vector thermal_drift_d(iangular.size(), 0);
    libmobility::random_finite_differences(
        mdot, original_pos, thermal_drift_m, thermal_drift_d, seed,
        this->dppar.delta * this->dppar.hydrodynamicRadius, prefactor);
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

  void clean() override {
    Mobility::clean();
    dpstokes->clear();
  }
};
#endif
