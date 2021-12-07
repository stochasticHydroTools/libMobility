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
namespace libmobility{
#if defined SINGLE_PRECISION
  using  real  = float;
#else
  using  real  = double;
#endif

  struct Periodicity{
    Periodicity(){}
    Periodicity(bool x, bool y, bool z): x(x), y(y), z(z){}
    bool x,y,z;
  };

  struct BoxSize{
    BoxSize(){}
    BoxSize(real x, real y, real z): x(x), y(y), z(z){}
    real x = -1;
    real y = -1;
    real z = -1;
  };

  struct Parameters{
    real hydrodynamicRadius;
    real viscosity;
    real temperature;
    real tolerance = 1e-4;
    BoxSize boxSize;
    Periodicity periodicity;
  };
  
  class Mobility{
  public:
    static constexpr auto version = LIBMOBILITYVERSION; //The interface version
#if defined SINGLE_PRECISION
    static constexpr auto precision = "float";
#else
    static constexpr auto precision = "double";
#endif

    virtual void initialize(Parameters par) = 0;
    
    virtual void setPositions(const real* positions, int numberParticles) = 0;

    virtual void Mdot(const real* forces, const real *torques, real* result) = 0;

    virtual void stochasticDisplacements(real* result, real prefactor = 1){ //Use Lanczos here
      throw std::runtime_error("Stochastic displacements are not implemented for this solver");      
    }

    virtual void hydrodynamicDisplacements(const real* forces, const real *torques, real* result, real prefactor = 1){
      Mdot(forces, torques, result);
      stochasticDisplacements(result, prefactor);
    }

    virtual void clean(){}
  };

  
}
#endif
