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
    this->dppar.w = 6;
    this->dppar.beta = 1.714*this->dppar.w;
    real h = this->dppar.hydrodynamicRadius/1.554;
    this->dppar.alpha = this->dppar.w/2.0;
    this->dppar.tolerance = 1e-6;
    this->dppar.nx = int(this->dppar.Lx/h + 0.5);
    this->dppar.ny = int(this->dppar.Ly/h + 0.5);
    // Add a buffer of w*h/2 when there is an open boundary
    if(this->wallmode == "nowall"){
      this->dppar.zmax += this->dppar.w*h/2;
      this->dppar.zmin -= this->dppar.w*h/2;
    }
    if(this->wallmode == "bottom"){
      this->dppar.zmin -= this->dppar.w*h/2;
    }
    real Lz = this->dppar.zmax - this->dppar.zmin;
    this->dppar.nz = M_PI*Lz/(h);
    dpstokes->initialize(dppar, this->numberParticles);
    Mobility::initialize(ipar);
    if(ipar.needsTorque) throw std::runtime_error("[DPStokes] Torque is not implemented");
  }

  void setPositions(const real* ipositions) override{
    dpstokes->setPositions(ipositions);
  }

  void Mdot(const real* forces, const real* torques, real* linear, real* angular) override{
    dpstokes->Mdot(forces, torques, linear, angular);
  }

  void clean() override{
    Mobility::clean();
    dpstokes->clear();
  }
};
#endif
