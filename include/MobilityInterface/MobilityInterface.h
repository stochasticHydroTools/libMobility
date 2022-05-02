/* Raul P. Pelaez 2021. The libMobility interface v1.0.
   Every mobility implement must inherit from the Mobility virtual base class.

 */
#ifndef MOBILITYINTERFACE_H
#define MOBILITYINTERFACE_H
#define LIBMOBILITYVERSION "1.0"
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
#include<stdexcept>
#include<vector>
#include"lanczos.h"
namespace libmobility{
#if defined SINGLE_PRECISION
  using  real  = float;
#else
  using  real  = double;
#endif

  enum class periodicity_mode{triply_periodic, doubly_periodic, single_wall, open, unspecified};

  enum class device{cpu, gpu, automatic};

  //Parameters that are proper to every solver.  
  struct Parameters{
    std::vector<real> hydrodynamicRadius;
    real viscosity = 1;
    real temperature = 0;
    real tolerance = 1e-4; // Donev: Add comment to explain what this is. Could be tolerance for M*F calculation, or for Lanczos. Two separate and different things
    int numberParticles = -1;
  };

  //A list of parameters that cannot be changed by reinitializing a solver and/or are properties of the solver.
  //For instance, an open boundary solver will only accept open periodicity.
  //Another solver might be set up for either cpu or gpu at creation
  struct Configuration{
    int dimensions = 3; // Donev: Some stuff like periodicity_mode seem very specific to 3D. I don't think we can really nail down how to do this and handle different dimensions until we have an example (say the quasi2D stuff including Saffman mobility. Perhaps we should just limit to 3D. There is always a tradeoff between generality and simplicity of use.
    int numberSpecies = 1; // Donev: Not clear to me what this means and why we need it, see comments in README. Maybe best to omit it until we have some example with more than one species and we can figure out how to use this. At this point it is too abstract.
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
      if(conf.numberSpecies!=1)
    	throw std::runtime_error("[Mobility] I can only deal with one species");
      if(conf.device == device::gpu)
    	throw std::runtime_error("[Mobility] This is a CPU-only solver");
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
    // called. Furthermore, initialize can be called again if some parameter
    // changes
    virtual void initialize(Parameters par){
      // Donev: If already initialized, shouldn't this call clean first (I know these are all virtual methods but just as an example)?
      this->initialized = true;
      this->numberParticles = par.numberParticles;
      if(not lanczos){
	lanczos = std::make_shared<LanczosStochasticDisplacements>(par.numberParticles, par.temperature, par.tolerance);
      }

    }

    virtual void setPositions(const real* positions) = 0;

    virtual void Mdot(const real* forces, const real *torques, real* result) = 0;

    //If the solver does not provide a stochastic displacement implementation, the Lanczos algorithm will be used automatically
    virtual void stochasticDisplacements(real* result, real prefactor = 1){
      if(not this->initialized)
	throw std::runtime_error("[libMobility] You must initialize the base class in order to use the default stochastic displacement computation");
      lanczos->stochasticDisplacements([this](const real*f, real*mv){Mdot(f, nullptr, mv);}, result, prefactor);
    }
    
    virtual void hydrodynamicDisplacements(const real* forces, const real *torques, real* result, real prefactor = 1){
      Mdot(forces, torques, result);
      stochasticDisplacements(result, prefactor);
    }

    virtual void clean(){}
  };
}
#endif
