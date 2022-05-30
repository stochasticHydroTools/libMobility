// Donev: Is this now redundant with mobility_cuda since both provide the same thing? It seems strange to have both
// In particular they are called NBody and NBody_wall. Surely they should be together, and just switch based on the periodicity? Why do we need two repos, two different solver classes, etc., when they are basically the same and only the kernel changes
// In particular, if you make the kernel a template parameter you can make your fancy NBody work for any kernel including the one with a single wall

#ifndef MOBILITY_NBODY_H
#define MOBILITY_NBODY_H
#include<MobilityInterface/MobilityInterface.h>
#include"BatchedNBodyRPY/source/interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

static_assert(std::is_same<libmobility::real, nbody_rpy::real>::value,
	      "Trying to compile NBody with a different precision to MobilityInterface.h");

class NBody: public libmobility::Mobility{
  using device = libmobility::device;
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  std::vector<real> positions;
  real selfMobility;
  real hydrodynamicRadius;
  int numberParticles;
  nbody_rpy::algorithm algorithm = nbody_rpy::algorithm::advise;
public:

  NBody(Configuration conf){
    if(conf.dev == device::cpu)
      throw std::runtime_error("[Mobility] This is a GPU-only solver");
    if(conf.periodicity != periodicity_mode::open)
      throw std::runtime_error("[Mobility] This is an open boundary solver");
  }

  void setParametersNBody(nbody_rpy::algorithm algo){
    this->algorithm = algo;
  }

  virtual void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->selfMobility = 1.0/(6*M_PI*ipar.viscosity*this->hydrodynamicRadius);
    Mobility::initialize(ipar);
  }

  virtual void setPositions(const real* ipositions) override{
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());
  }

  virtual void Mdot(const real* forces, const real *torques, real* result) override{
    if(torques) throw std::runtime_error("NBody can only compute monopole displacements");
    int numberParticles = positions.size()/3;
    nbody_rpy::callBatchedNBodyRPY(positions.data(),
				   forces,
				   result,
				   1, numberParticles,
				   selfMobility, hydrodynamicRadius,
				   algorithm);
  }
};
#endif

