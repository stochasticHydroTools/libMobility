
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
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  Parameters par;
  std::vector<real> positions;
  real selfMobility;
  real hydrodynamicRadius;
public:

  virtual void initialize(Parameters ipar) override{
    this->hydrodynamicRadius = ipar.hydrodynamicRadius;
    this->selfMobility = 1.0/(6*M_PI*ipar.viscosity*this->hydrodynamicRadius); 
  }

  virtual void setPositions(const real* ipositions, int numberParticles) override{
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
				   nbody_rpy::algorithm::advise);
  }

};
#endif
