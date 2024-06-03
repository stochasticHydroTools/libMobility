/* Raul P. Pelaez 2021-2022. The libMobility interface.
   Every mobility implement must inherit from the Mobility virtual base class.
   See solvers/SelfMobility for a simple example
 */
#ifndef MOBILITYINTERFACE_H
#define MOBILITYINTERFACE_H
#include<stdexcept>
#include <vector>
#include <random>
#include <array>
#include"lanczos.h"
#include"defines.h"
namespace libmobility{

  enum class periodicity_mode{single_wall, two_walls, open, periodic, unspecified};

  //Parameters that are proper to every solver.
  struct Parameters{
    std::vector<real> hydrodynamicRadius;
    real viscosity = 1;
    real temperature = 0;
    real tolerance = 1e-4; //Tolerance for Lanczos fluctuations
    int numberParticles = -1;
    std::uint64_t seed = 0;
    bool needsTorque = false;
  };

  //A list of parameters that cannot be changed by reinitializing a solver and/or are properties of the solver.
  //For instance, an open boundary solver will only accept open periodicity.
  //Another solver might be set up for either cpu or gpu at creation
  struct Configuration{
    periodicity_mode periodicityX = periodicity_mode::unspecified;
    periodicity_mode periodicityY = periodicity_mode::unspecified;
    periodicity_mode periodicityZ = periodicity_mode::unspecified;
  };

  //This is the virtual base class that every solver must inherit from.
  class Mobility{
  private:
    int numberParticles;
    bool initialized = false;
    std::uint64_t lanczosSeed;
    real lanczosTolerance;
    std::shared_ptr<LanczosStochasticVelocities> lanczos;
    std::vector<real> lanczosOutput;
    real temperature;
    bool needsTorque = false;
  protected:
    Mobility(){};
  public:
    //These constants are available to all solvers
    static constexpr auto version = LIBMOBILITYVERSION; //The interface version
#if defined SINGLE_PRECISION
    static constexpr auto precision = "float";
#else
    static constexpr auto precision = "double";
#endif
    //The constructor should accept a Configuration object and ensure the requested parameters are acceptable (an open boundary solver should complain if periodicity is selected).
    //A runtime_exception should be thrown if the configuration is invalid.
    //The constructor here is just an example, since this is a pure virtual class
    /*
    Mobility(Configuration conf){
      if(conf.periodicityX != periodicity::open or
      conf.periodicityY != periodicity::open or
      conf.periodicityZ != periodicity::open)
    	throw std::runtime_error("[Mobility] This is an open boundary solver");
    }
    */
    //Outside of the common interface, solvers can define a function called setParameters[ModuleName]
    //, with arbitrary input, that simply acknowledges a set of values proper to the specific solver.
    //These new parameters should NOT take effect until initialize is called afterwards.
    // void setParametersModuleName(MyParameters par){
    //   //Store required parameters
    // }

    //Initialize should leave the solver in a state ready for setPositions to be
    // called. Furthermore, initialize can be called again if some parameter changes
    virtual void initialize(Parameters par){
      //Clean if the solver was already initialized
      if(initialized)
	this->clean();
      this->initialized = true;
      this->numberParticles = par.numberParticles;
      this->lanczosSeed = par.seed;
      this->lanczosTolerance = par.tolerance;
      this->temperature = par.temperature;
      this->needsTorque = par.needsTorque;
    }

    //Set the positions to construct the mobility operator from
    virtual void setPositions(const real* positions) = 0;

    //Apply the grand mobility operator (\Omega) to a series of forces (F) and torques (T) to get the resulting linear (V) and angular (W) velocities
    virtual void Mdot(const real* forces, const real* torques, real* linear, real* angular) = 0;

    //Compute the stochastic displacements as result=prefactor*sqrt(\Omega)*dW. Where dW is a vector of Gaussian random numbers
    //If the solver does not provide a stochastic displacement implementation, the Lanczos algorithm will be used automatically
    virtual void sqrtMdotW(real* linear, real* angular, real prefactor = 1){
      if(this->temperature == 0) return;
      if(not this->initialized)
	throw std::runtime_error("[libMobility] You must initialize the base class in order to use the default stochastic displacement computation");
      if(not lanczos){
	if(this->lanczosSeed==0){//If a seed is not provided, get one from random device
	  this->lanczosSeed = std::random_device()();
	}
	int numberElements = this->needsTorque ? (2*this->numberParticles) : this->numberParticles;
	lanczos = std::make_shared<LanczosStochasticVelocities>(numberElements,
								this->lanczosTolerance, this->lanczosSeed);
	lanczosOutput.resize(3*numberElements);
      }
      if(this->needsTorque and not angular)
	throw std::runtime_error("[libMobility] This solver requires angular velocities when configured with torques");
      std::fill(lanczosOutput.begin(), lanczosOutput.end(), 0);
      lanczos->sqrtMdotW([this](const real*f, real* mv){
	const real* t = this->needsTorque ? (f+3*this->numberParticles) : nullptr;
        real* mt = this->needsTorque ? (mv+3*this->numberParticles) : nullptr;
	Mdot(f, t, mv, mt);
      }, lanczosOutput.data(), prefactor);
      std::copy(lanczosOutput.begin(), lanczosOutput.begin()+3*this->numberParticles, linear);
      if(this->needsTorque)
	std::copy(lanczosOutput.begin()+3*this->numberParticles, lanczosOutput.end(), angular);
    }

    //Equivalent to calling Mdot and then stochasticDisplacements, can be faster in some solvers
    virtual void hydrodynamicVelocities(const real* forces, const real* torques, real* linear, real* angular, real prefactor = 1){
      if(forces or torques){
	Mdot(forces, torques, linear, angular);
      }
      sqrtMdotW(linear, angular, prefactor);
    }

    int getNumberParticles() const{
      return this->numberParticles;
    }

    //Clean any memory allocated by the solver
    virtual void clean(){
      lanczos.reset();
      lanczosOutput.clear();
    }
  };
}
#endif
