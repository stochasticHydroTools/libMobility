/*Raul P. Pelaez 2022. libMobility interface for UAMMD's DPStokes module
 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include <MobilityInterface/MobilityInterface.h>
#include"DPStokes/uammd_interface.h"
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
public:

  DPStokes(Configuration conf){
    if(conf.periodicityX != periodicity_mode::periodic or
       conf.periodicityY != periodicity_mode::periodic or
       not (conf.periodicityZ == periodicity_mode::open or
	    conf.periodicityZ == libmobility::periodicity_mode::single_wall or
	    conf.periodicityZ == libmobility::periodicity_mode::two_walls)
	    )
      throw std::runtime_error("[DPStokes] This is a doubly periodic solver");
  }

  // I am confused by this function. Isn't it supposed to be called setParameters?
  void setParametersDPStokes(DPStokesParameters i_dppar){
    this->dppar = i_dppar;
    // DPStokesParameters dppar;
    // int nx = -1;
    // int ny = -1;
    // int nz = -1;
    // dppar.dt
    // dppar.viscosity
    // dppar.Lx
    // dppar.Ly
    // dppar.zmin
    //   dppar. zmax
    // //Tolerance will be ignored in DP mode, TP will use only tolerance and nxy/nz
    // dppar.tolerance = 1e-5;
    // dppar.w
    // dppar.w_d;
    // dppar.hydrodynamicRadius = -1;
    // dppar.beta = -1;
    // dppar.beta_d = -1;
    // dppar.alpha = -1;
    // dppar.alpha_d = -1;
    // //Can be either none, bottom, slit or periodic
    // dppar.mode;
    dpstokes = std::make_shared<uammd_dpstokes::DPStokesGlue>();
  }

  void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    this->dppar.viscosity = ipar.viscosity;
    this->temperature = ipar.temperature;
    this->lanczosTolerance = ipar.tolerance;
    dpstokes->initialize(dppar, this->numberParticles);
    Mobility::initialize(ipar);
  }

  void setPositions(const real* ipositions) override{
    dpstokes->setPositions(ipositions);
  }

  void Mdot(const real* forces, real* result) override{
    dpstokes->Mdot(forces, nullptr, result, nullptr);
  }

  void sqrtMdotW(real* result, real prefactor = 1) override{
    if(this->temperature == 0) return;
    if(not dpstokes)
      throw std::runtime_error("[libMobility] You must initialize the base class in order to use the default stochastic displacement computation");
    if(not lanczos){
      if(this->lanczosSeed==0){//If a seed is not provided, get one from random device
	this->lanczosSeed = std::random_device()();
      }
      lanczos = std::make_shared<LanczosStochasticVelocities>(this->numberParticles,
							      this->lanczosTolerance, this->lanczosSeed);
    }
    lanczos->sqrtMdotW([this](const real*f, real* mv){Mdot(f, mv);}, result, prefactor);
  }

  void clean() override{
    Mobility::clean();
    dpstokes->clear();
  }
};
#endif
