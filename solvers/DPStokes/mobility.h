/*Raul P. Pelaez 2022. libMobility interface for UAMMD's DPStokes module
 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include <MobilityInterface/MobilityInterface.h>
#include"extra/uammd_interface.h"
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
    dpstokes = std::make_shared<uammd_dpstokes::DPStokesGlue>();
  }

  void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    this->dppar.viscosity = ipar.viscosity;
    this->temperature = ipar.temperature;
    this->lanczosTolerance = ipar.tolerance;
    this->dppar.mode = this->wallmode;
    this->dppar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->dppar.w = 4;
    this->dppar.beta = 1.785*this->dppar.w;
    real h = this->dppar.hydrodynamicRadius/1.205;
    this->dppar.alpha = this->dppar.w/2.0;
    this->dppar.tolerance = 1e-6;

    // adjust box size to be a multiple of h
    real N_in = this->dppar.Lx/h;
    int N_up = ceil(N_in);
    int N_down = floor(N_in);
    int N;
    // either N_up or N_down will be a multiple of 2. pick the even one for a more FFT friendly grid.
    if(N_up % 2 == 0){
      N = N_up;
    }else{
      N = N_down;
    }

    // note: only works for square boxes
    this->dppar.Lx = N*h;
    this->dppar.Ly = N*h;
    this->dppar.nx = N;
    this->dppar.ny = N;

    // Add a buffer of w*h/2 when there is an open boundary
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

    // pick nearby N such that 2(Nz-1) is FFT friendly
    N_up = ceil(nz_actual);
    N_down = floor(nz_actual);
    if(N_up % 2 == 1){
      this->dppar.nz = N_up;
    } else {
      this->dppar.nz = N_down;
    }

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
