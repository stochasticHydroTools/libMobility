/* Raul P. Pelaez 2021. libMobility C++ interface for NBody_wall.
   Many solvers are available in the mobility_cuda repo. Only some of them are exposed via this interface.
 */
#ifndef MOBILITY_NBODY_WALL_H
#define MOBILITY_NBODY_WALL_H
#include<MobilityInterface/MobilityInterface.h>
#include"Mobility_NBody_wall/interface.h"
#include <stdexcept>
#include<vector>
#include<cmath>
#include<type_traits>

static_assert(std::is_same<libmobility::real, mobility_cuda::real>::value,
	      "Trying to compile NBody_wall with a different precision to MobilityInterface.h");

class NBody_wall: public libmobility::Mobility{
  using real = libmobility::real;
  using device = libmobility::device;
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;

  std::vector<real> positions;
  real viscosity;
  real hydrodynamicRadius;
  int numberParticles;
  bool initialized = false;
  real lx, ly, lz;
public:

  NBody_wall(Configuration conf){
    if(conf.numberSpecies!=1)
      throw std::runtime_error("[Mobility] I can only deal with one species");
    if(conf.dev == device::cpu)
      throw std::runtime_error("[Mobility] This is a GPU-only solver");
    if(conf.periodicity != periodicity_mode::single_wall)
      throw std::runtime_error("[Mobility] Only single mode is allowed");
  }
  
  void initialize(Parameters ipar) override{
    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->viscosity = ipar.viscosity;
    this->numberParticles = ipar.numberParticles;
  }

  void setParametersNBody_wall(real lx, real ly, real lz){
    this->initialized = true;
    this->lx = lx;
    this->ly = ly;
    this->lz = lz;
  }

  void setPositions(const real* ipositions) override{
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());
  }

  void Mdot(const real* forces, const real *torques, real* result) override{
    if(torques) throw std::runtime_error("NBody_wall can only compute monopole displacements");
    if(not initialized) throw std::runtime_error("[NBody_wall] You must call setParametersNBody_wall first.");
    
    int numberParticles = positions.size()/3;
    mobility_cuda::single_wall_mobility_trans_times_force_cuda(positions.data(),
							       forces,
							       result,
							       viscosity, hydrodynamicRadius,
							       lx, ly, lz,
							       numberParticles);
  }  

};
#endif
