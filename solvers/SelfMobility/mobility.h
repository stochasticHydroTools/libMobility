/*Raul P. Pelaez 2022. SelfMobility example sovler.

  This solver ignores hydrodynamic interactions. The mobility is the identity matrix scaled with 1/(6pi*eta*a).
  This is a simple example on how to implement a new solver. Note that is a purely CPU implementation.

 */
#ifndef MOBILITY_SELFMOBILITY_H
#define MOBILITY_SELFMOBILITY_H
#include<MobilityInterface/MobilityInterface.h>
#include <cstdint>
#include <random>
#include<vector>
#include<cmath>
#include<type_traits>

class SelfMobility: public libmobility::Mobility{
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  Parameters par;
  std::vector<real> positions;
  real selfMobility;
  real temperature;
  int numberParticles;
  std::mt19937 rng;
public:

  SelfMobility(Configuration conf){
    if(conf.periodicityX != periodicity_mode::open or
       conf.periodicityY != periodicity_mode::open or
       conf.periodicityZ != periodicity_mode::open)
      throw std::runtime_error("[Mobility] This is an open boundary solver");
  }

  void initialize(Parameters ipar) override{
    auto seed = ipar.seed;
    if(not seed) seed = std::random_device()();
    this->rng = std::mt19937{seed};
    this->temperature = ipar.temperature;
    this->numberParticles = ipar.numberParticles;
    real hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->selfMobility = 1.0/(6*M_PI*ipar.viscosity*hydrodynamicRadius);
    Mobility::initialize(ipar);
    if(ipar.needsTorque)
      throw std::runtime_error("[SelfMobility] Torque is not implemented");
  }

  //An example of how to take in extra parameters. This function is supposed to be called BEFORE initialize
  void setParametersSelfMobility(real some_unnecesary_parameter){

  }

  void setPositions(const real* ipositions) override{ }

  void Mdot(const real* forces, const real* torques, real* linear, real* angular) override{
    if(torques or angular)
      throw std::runtime_error("[SelfMobility] Torque is not implemented");
    if(not forces){
      std::fill(linear, linear+3*numberParticles, 0);
    }
    else{
      for(int i = 0; i<3*numberParticles; i++){
	linear[i] = forces[i]*selfMobility;
      }
    }
  }

  //If this function is not present the default behavior is invoked, which uses the Lanczos algorithm
  void sqrtMdotW(real* linear, real* angular, real prefactor = 1) override{
    if(angular)
      throw std::runtime_error("[SelfMobility] Torque is not implemented");
    std::normal_distribution<real> d{0,1};
    for(int i = 0; i<3*numberParticles; i++){
      real dW = d(rng);
      linear[i] = prefactor*sqrt(2*temperature*selfMobility)*dW;
    }
  }

};
#endif
