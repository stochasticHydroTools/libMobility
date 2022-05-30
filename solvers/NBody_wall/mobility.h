/* Raul P. Pelaez 2021. libMobility C++ interface for NBody_wall.
   Many solvers are available in the mobility_cuda repo. Only some of them are exposed via this interface.
 */
// Donev: Well, only ONE is exposed here ;-)
// Donev: One thing that I cannot figure out on my own is this. If one wants to use GPU mode, is everything going to be kept on the GPU. For example, is the Lanczos going to be done on the GPU, and the Mdot product, etc., and nothing will be copied to and from the CPU? If this is not accomplished, the utility of the GPU implementation is greatly diminished. We already have a python Lanczos implementation. The point here is to accelerate all that on the GPU without going through interpeter but also without copying memory back and forth with every call to Mdot etc. Let's discuss 
 
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
  // Donev: I think you inherited these from single_wall_mobility_trans_times_force_cuda
  // That implements fake periodicity. If you do that, then the parameters should be the number of images you do in x/y/z directions. We need to discuss that and I need to look at the code. Remember I really wanted to have scalar kernel functions without loops over particles/images...
  
public:

  NBody_wall(Configuration conf){
    if(conf.dev == device::cpu)
      throw std::runtime_error("[Mobility] This is a GPU-only solver");
    if(conf.periodicity != periodicity_mode::single_wall)
      throw std::runtime_error("[Mobility] Only single mode is allowed");
  }
  
  void initialize(Parameters ipar) override{
    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->viscosity = ipar.viscosity;
    this->numberParticles = ipar.numberParticles;
    Mobility::initialize(ipar);
  }

  // Donev: Why are lx, ly, lz parameters when they don't make sense. This solver needs no other parameters. 
  void setParametersNBody_wall(real lx, real ly, real lz){
    this->initialized = true;
    // Donev: The rest of this should not be here
    this->lx = lx;
    this->ly = ly;
    this->lz = lz;
  }

  // Donev: What are the requiremets on the positions -- user needs to know. For example, is it assumed all particles have z>=0 (I think it is) 
  // Is the wall at z=0 always?
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
