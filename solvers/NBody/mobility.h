/* Raul P. Pelaez 2022. NBody libMobility solver.

 */
#ifndef MOBILITY_NBODY_H
#define MOBILITY_NBODY_H
#include<MobilityInterface/MobilityInterface.h>
#include"extra/interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

class NBody: public libmobility::Mobility{
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  enum class kernel_type{open_rpy, bottom_wall};
  kernel_type kernel;
  std::vector<real> positions;
  real transMobility;
  real rotMobility;
  real transRotMobility;
  real hydrodynamicRadius;
  int numberParticles;
  nbody_rpy::algorithm algorithm = nbody_rpy::algorithm::advise;

  //Batched functionality configuration
  int Nbatch;
  int NperBatch;
public:

  NBody(Configuration conf){
    if(conf.periodicityX != periodicity_mode::open or conf.periodicityY != periodicity_mode::open)
      throw std::runtime_error("[Mobility] NBody must be open in the plane");
    if(conf.periodicityZ == periodicity_mode::open)
      this->kernel = kernel_type::open_rpy;
    else if(conf.periodicityZ == periodicity_mode::single_wall)
      this->kernel = kernel_type::bottom_wall;
    else
      throw std::runtime_error("[Mobility] Invalid periodicity");
  }

  struct NBodyParameters{
    nbody_rpy::algorithm algo = nbody_rpy::algorithm::advise;
    int Nbatch = -1;
    int NperBatch = -1;
  };

  //NBody allows for different strategies: Naive, Block and Fast. The "algo" argument allows to choose one. The possible values are:
  // fast, naive, block, advise. The last one tries to choose the best strategy according to the input.
  // NBody can work on "batches" of particles, all batches must have the same size. Note that a single batch is equivalent to every particle interacting to every other.
  // Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.
  // The data layout is 3 interleaved coordinates with each batch placed after the previous one: [x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]
  void setParametersNBody(NBodyParameters par){
    this->algorithm = par.algo;
    this->Nbatch = par.Nbatch;
    this->NperBatch = par.NperBatch;
  }

  virtual void initialize(Parameters ipar) override{
    this->numberParticles = ipar.numberParticles;
    if(Nbatch<0) Nbatch = 1;
    if(NperBatch<0) NperBatch = ipar.numberParticles;
    if(NperBatch*Nbatch != numberParticles)
      throw std::runtime_error("[Mobility] Invalid batch parameters for NBody. If in doubt, use the defaults.");

    this->hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    this->transMobility = 1.0/(6*M_PI*ipar.viscosity*hydrodynamicRadius);
    this->transRotMobility = 1.0/(8*M_PI*ipar.viscosity*hydrodynamicRadius*hydrodynamicRadius);
    this->rotMobility = 1.0/(8*M_PI*ipar.viscosity*hydrodynamicRadius*hydrodynamicRadius*hydrodynamicRadius);
    Mobility::initialize(ipar);
  }

  virtual void setPositions(const real* ipositions) override{
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());
  }

  virtual void Mdot(const real* forces, const real* torques, real* linear, real* angular) override{
    int numberParticles = positions.size()/3;
    // Donev: Why can't there be a single callBatchedNBody routine that does if(kernel == kernel_type::bottom_wall) internally?
    // Example, what if in the future we add manually periodized RPY where one repeats a unit cell a certain number of times in each direction. This is actually easy to do with 3 loops and useful, and only requires adding a parameter nUnitCellsRepeat[3] and removing the error if some direction is periodic and only spitting an error if two walls are asked for.

    real* tempTorques;
    real* tempAngular;
    bool torquesPassed = torques;
    // this is a temporary hacky fix. Mdot passes in nullptr if no torques are given
    // but NBody currently always computes torques, even if they aren't needed.
    if(!torques){
      tempTorques = new real[3*numberParticles];
      std::fill(tempTorques, tempTorques + 3*numberParticles, 0);

      tempAngular = new real[3*numberParticles];
      std::fill(tempAngular, tempAngular + 3*numberParticles, 0);

      torques = tempTorques;
      angular = tempAngular;
    } else if(torques && kernel == kernel_type::bottom_wall){
      throw std::runtime_error("[Mobility] Bottom wall kernel not implemented for torques.\n");
    }

    auto solver = nbody_rpy::callBatchedNBodyOpenBoundaryRPY;
    if(kernel == kernel_type::bottom_wall)
      solver = nbody_rpy::callBatchedNBodyBottomWallRPY;
    solver(positions.data(), forces, torques, linear, angular,
	   Nbatch, NperBatch,
	   transMobility, rotMobility, transRotMobility, hydrodynamicRadius,
	   algorithm);

    // part 2 of hacky fix
    if(!torquesPassed){
      delete[] tempTorques;
      delete[] tempAngular;
      torques = nullptr;
      angular = nullptr;
    }

  }
};
#endif
