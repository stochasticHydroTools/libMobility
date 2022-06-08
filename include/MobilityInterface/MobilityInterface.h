/* Raul P. Pelaez 2021-2022. The libMobility interface v1.0.
   Every mobility implement must inherit from the Mobility virtual base class.
   See solvers/SelfMobility for a simple example
 */
#ifndef MOBILITYINTERFACE_H
#define MOBILITYINTERFACE_H
#define LIBMOBILITYVERSION "1.0"
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
#include<stdexcept>
#include <vector>
#include<random>
#include"lanczos.h"
namespace libmobility{
#if defined SINGLE_PRECISION
  using  real  = float;
#else
  using  real  = double;
#endif

  // Donev: Fix this
  enum class periodicity_mode{triply_periodic, doubly_periodic, single_wall, open, unspecified};

  enum class device{cpu, gpu, automatic};

  //Parameters that are proper to every solver.  
  struct Parameters{
    std::vector<real> hydrodynamicRadius;
    real viscosity = 1;
    real temperature = 0; //Donev: If set to zero, the calculation of stochastic displacements is skipped
    real tolerance = 1e-4; //Tolerance for Lanczos fluctuations
    int numberParticles = -1;
    std::uint64_t seed = 0;
  };

  //A list of parameters that cannot be changed by reinitializing a solver and/or are properties of the solver.
  //For instance, an open boundary solver will only accept open periodicity.
  //Another solver might be set up for either cpu or gpu at creation
  struct Configuration{
    int dimensions = 3; 
    periodicity_mode periodicity = periodicity_mode::unspecified;
    device dev = device::automatic;
  };

  //This is the virtual base class that every solver must inherit from.
  class Mobility{
  private:
    int numberParticles;
    bool initialized = false;
    std::shared_ptr<LanczosStochasticDisplacements> lanczos;
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
      if(conf.device == device::gpu)
    	throw std::runtime_error("[Mobility] This is a CPU-only solver");
      // Donev: Fix this once periodicity is changed	
      if(conf.periodicity != periodicity::open)
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
      // Donev: Why always build a Lanczos structure if it is not used
      // You say in top README  "The initialize function of a new solver must call the ```libmobility::Mobility::initialize``` function at some point"
      // So even for PSE, which does not need Lanczos, the object lanczos will be created, which seems wrong
      // How about doing this below in stochasticDisplacements instead, so it is only done when
      // that virtual function is not overridden?
      if(not lanczos){
	if(par.seed==0){//If a seed is not provided, get one from random device
	  par.seed = std::random_device()();
	}
	lanczos = std::make_shared<LanczosStochasticDisplacements>(par.numberParticles, par.tolerance, par.seed);
      }
    }

    //Set the positions to construct the mobility operator from
    virtual void setPositions(const real* positions) = 0;

    // Donev SERIOUS comment: See comments in top README
    // result should be split into velocities and angVelocities
    // Same applies to all other functions below
    //Apply the mobility operator (M) to a series of forces (F) and/or torques (T), returns M*[F; T]
    virtual void Mdot(const real* forces, const real *torques, real* result) = 0;

    //Compute the stochastic displacements as result=prefactor*sqrt(M)*dW. Where dW is a vector of Gaussian random numbers   
    //If the solver does not provide a stochastic displacement implementation, the Lanczos algorithm will be used automatically
    virtual void stochasticDisplacements(real* result, real prefactor = 1){
      if(not this->initialized)
	throw std::runtime_error("[libMobility] You must initialize the base class in order to use the default stochastic displacement computation");
	  // Donev: This assumes only stochastic linear velocities are wanted, which is not quite right
	  // It make completes sense to also have stochastic rotational velocities
      lanczos->stochasticDisplacements([this](const real*f, real*mv){Mdot(f, nullptr, mv);}, result, prefactor);
    }

    //Equivalent to calling Mdot and then stochasticDisplacements, can be faster in some solvers
    // Donev: If this is a GPU solver like NbodyRPY.cu, how many times will there be some exchange of arrays between GPU and CPU memory? Can the code somehow be written to minimize this, for example, stochasticDisplacement could create result arrays on the GPU and only copy the final result to the CPU array when mode is GPU. It seems to me the current result arrays are CPU arrays    
    virtual void hydrodynamicDisplacements(const real* forces, const real *torques, real* result, real prefactor = 1){
      Mdot(forces, torques, result);
      stochasticDisplacements(result, prefactor);
    }

    //Clean any memory allocated by the solver
    virtual void clean(){
      lanczos.reset();
    }
  };
}
#endif
