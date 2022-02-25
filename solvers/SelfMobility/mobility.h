/*Raul P. Pelaez 2022. SelfMobility example sovler. 
  
  This solver ignores hydrodynamic interactions and uses the default mechanism to compute stochastic displacements (the Lanczos algorithm).
  Donev: You should change this to actually over-ride the default to show how it can be done, since here it is trivial to construct sqrt
  One issue I want to understand better is that of seeding and RNGs. How many streams should there be, and who seeds them and when, etc.

 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include<MobilityInterface/MobilityInterface.h>
#include<vector>
#include<cmath>
#include<type_traits>

class SelfMobility: public libmobility::Mobility{
  using device = libmobility::device;
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  Parameters par;
  std::vector<real> positions;
  real selfMobility;
  int numberParticles;
public:

  SelfMobility(Configuration conf){
    if(conf.numberSpecies!=1)
      throw std::runtime_error("[Mobility] I can only deal with one species");
    if(conf.dev == device::gpu)
      throw std::runtime_error("[Mobility] This is a CPU-only solver");
    if(conf.periodicity != periodicity_mode::open)
      throw std::runtime_error("[Mobility] This is an open boundary solver");
  }

  void initialize(Parameters ipar) override{
    Mobility::initialize(ipar);
    this->numberParticles = ipar.numberParticles;
    real hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->selfMobility = 1.0/(6*M_PI*ipar.viscosity*hydrodynamicRadius); 
  }

  void setPositions(const real* ipositions) override{ }

  void Mdot(const real* forces, const real *torques, real* result) override{
    if(torques) throw std::runtime_error("SelfMobility can only compute monopole displacements");
    for(int i = 0; i<3*numberParticles; i++){
      result[i] = forces[i]*selfMobility;
    }
  }
};
#endif

