/* Raul P. Pelaez 2021. libMobility C++ interface for NBody_wall.
   Many solvers are available in the mobility_cuda repo. Only some of them are exposed via this interface.
 */
#ifndef MOBILITY_NBODY_WALL_H
#define MOBILITY_NBODY_WALL_H
#include<MobilityInterface/MobilityInterface.h>
#include"Mobility_NBody_wall/interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

static_assert(std::is_same<libmobility::real, mobility_cuda::real>::value,
	      "Trying to compile NBody_wall with a different precision to MobilityInterface.h");

class NBody_wall: public libmobility::Mobility{
  using real = libmobility::real;
  std::vector<real> positions;
  real viscosity;
  real hydrodynamicRadius;
  libmobility::BoxSize box;
public:
  using Parameters = libmobility::Parameters;

  virtual void initialize(Parameters ipar) override{
    this->hydrodynamicRadius = ipar.hydrodynamicRadius;
    this->viscosity = ipar.viscosity;
    this->box = ipar.boxSize;
  }

  virtual void setPositions(const real* ipositions, int numberParticles) override{
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());
  }

  virtual void Mdot(const real* forces, const real *torques, real* result) override{
    if(torques) throw std::runtime_error("NBody_wall can only compute monopole displacements");
    int numberParticles = positions.size()/3;
    mobility_cuda::single_wall_mobility_trans_times_force_cuda(positions.data(),
							       forces,
							       result,
							       viscosity, hydrodynamicRadius,
							       box.x, box.y, box.z,
							       numberParticles);
  }  

};
#endif
