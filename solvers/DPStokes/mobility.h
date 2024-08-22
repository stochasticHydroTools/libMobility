/*Raul P. Pelaez 2022. libMobility interface for UAMMD's DPStokes module

References:
[1] Computing hydrodynamic interactions in confined doubly periodic geometries in linear time.
A. Hashemi et al. J. Chem. Phys. 158, 154101 (2023) https://doi.org/10.1063/5.0141371
 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include <MobilityInterface/MobilityInterface.h>
#include"extra/uammd_interface.h"
#include"extra/poly_fits.h"
#include<vector>
#include<cmath>
#include<type_traits>

class DPStokes: public libmobility::Mobility{
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using DPStokesParameters = uammd_dpstokes::PyParameters;
  using real = libmobility::real;
  using DPStokesUAMMD = uammd_dpstokes::DPStokesGlue;
  Parameters par;
  int numberParticles;
  std::shared_ptr<DPStokesUAMMD> dpstokes;
  DPStokesParameters dppar;
  real temperature;
  real lanczosTolerance;
  std::uint64_t lanczosSeed;
  std::shared_ptr<LanczosStochasticVelocities> lanczos;
  std::string wallmode;
public:

  DPStokes(Configuration conf){
    if(conf.periodicityX != periodicity_mode::periodic or
       conf.periodicityY != periodicity_mode::periodic or
       not (conf.periodicityZ == periodicity_mode::open or
	    conf.periodicityZ == libmobility::periodicity_mode::single_wall or
	    conf.periodicityZ == libmobility::periodicity_mode::two_walls)
	    )
      throw std::runtime_error("[DPStokes] This is a doubly periodic solver");
    if(conf.periodicityZ == periodicity_mode::open) wallmode = "nowall";
    else if(conf.periodicityZ == periodicity_mode::single_wall) wallmode = "bottom";
    else if(conf.periodicityZ == periodicity_mode::two_walls) wallmode = "slit";
  }

  void setParametersDPStokes(DPStokesParameters i_dppar){
    this->dppar = i_dppar;

    if(this->dppar.Lx != this->dppar.Ly) throw std::runtime_error("[DPStokes] Only square periodic boxes (Lx = Ly) are currently supported.\n");
    
    dpstokes = std::make_shared<uammd_dpstokes::DPStokesGlue>();
  }

  void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    this->dppar.viscosity = ipar.viscosity;
    this->temperature = ipar.temperature;
    this->lanczosTolerance = ipar.tolerance;
    this->dppar.mode = this->wallmode;
    this->dppar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];

    // sets up kernel parameters & grid based on optimal parameters from [1]
    this->dppar.w = 4;
    this->dppar.beta = 1.785*this->dppar.w;
    real h = this->dppar.hydrodynamicRadius/1.205;
    this->dppar.alpha = this->dppar.w/2.0;
    this->dppar.tolerance = 1e-6;

    // although this h is optimal for grid invariance, we need to adjust either L or h to get an integer number of points
    int N = floor(this->dppar.Lx/h);
    N += N % 2;

    this->dppar.nx = N;
    this->dppar.ny = N;

    // note: this part is only configured for square boxes
    if(this->dppar.allowChangingBoxSize){ // adjust box size to suit h
      this->dppar.Lx = N*h;
      this->dppar.Ly = N*h;
    } else{ // adjust h so that L/h is an integer
      h = this->dppar.Lx/N;
      double arg = this->dppar.hydrodynamicRadius/(this->dppar.w*h);
      this->dppar.beta = dpstokes_polys::polyEval(dpstokes_polys::cbetam_inv, arg);
    }

    // Add a buffer of 1.5*w*h/2 when there is an open boundary
    if(this->wallmode == "nowall"){
      this->dppar.zmax += 1.5*this->dppar.w*h/2;
      this->dppar.zmin -= 1.5*this->dppar.w*h/2;
    }
    if(this->wallmode == "bottom"){
      this->dppar.zmax += 1.5*this->dppar.w*h/2;
    }
    real Lz = this->dppar.zmax - this->dppar.zmin;
    real H = Lz/2;
    // sets chebyshev node spacing at its coarsest (in the middle) to be h
    real nz_actual = M_PI/(asin(h/H)) + 1;

    // pick nearby N such that 2(Nz-1) has two factors of 2 and is FFT friendly
    this->dppar.nz = floor(nz_actual);
    this->dppar.nz += (int)ceil(nz_actual) % 2;

    dpstokes->initialize(dppar, this->numberParticles);
    Mobility::initialize(ipar);
  }

  void setPositions(const real* ipositions) override{
    dpstokes->setPositions(ipositions);
  }

  void Mdot(const real* forces, real* result) override{
    dpstokes->Mdot(forces, nullptr, result, nullptr);
  }

  void clean() override{
    Mobility::clean();
    dpstokes->clear();
  }
};
#endif
