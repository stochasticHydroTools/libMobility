/*Raul P. Pelaez 2022. libMobility interface for UAMMD's DPStokes module
Donev: Consider making this the interface to both the CPU and GPU versions of DPStokes
 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include <MobilityInterface/MobilityInterface.h>
#include"DPStokes/uammd_interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

class DPStokes: public libmobility::Mobility{
  using device = libmobility::device;
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using DPStokesParameters = uammd_dpstokes::PyParameters;
  using real = libmobility::real;
  using DPStokesUAMMD = uammd_dpstokes::DPStokesGlue;
  Parameters par;
  int numberParticles;
  std::shared_ptr<DPStokesUAMMD> dpstokes;
public:

  DPStokes(Configuration conf){
    if(conf.dev != device::gpu)
      throw std::runtime_error("[DPStokes] This is a GPU-only solver");
    if(conf.periodicity != periodicity_mode::doubly_periodic)
      throw std::runtime_error("[DPStokes] This is a doubly periodic solver");
  }

  // I am confused by this function. Isn't it supposed to be called setParameters?
  void setParametersDPStokes(DPStokesParameters dppar){
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
    dpstokes->initialize(dppar, this->numberParticles);
  }

  void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    Mobility::initialize(ipar);
  }

  void setPositions(const real* ipositions) override{
    dpstokes->setPositions(ipositions);
  }

  void Mdot(const real* forces, const real *torques, real* result) override{
    // Donev: I don't get why result is passed twice here
    dpstokes->Mdot(forces, torques, result, result);
  }

  void clean() override{
    Mobility::clean();
    dpstokes->clear();
  }
};
#endif
