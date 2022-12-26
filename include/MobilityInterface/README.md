## Mobility interface for use in libMobility
<!--- Donev: This seems to simply duplicate stuff from the top-level README, nothing new
It only invites troubles to maintain consistency. I suggest deleting or drastically shortening to not repeat stuff from toplevel README like the requried methods.--->

The files in this repository are not very useful on their on, rather being a part of the libMobility repo.  

A C++ pure virtual base class called ```libmobility::Mobility``` is provided in MobilityInterface.h.  
A macro to export (via pybind11) any class inherited from ```libmobility::Mobility``` is available at pythonify.h.  

## The libMobility interface

A class inheriting from ```libmobility::Mobility``` provides the following set of functions (called the same in C++ and python and described here in a kind of language agnostic way):  
  * **initialize(parameters)**: Initializes the module according to the parameters (see below).  
  * **setPositions(positions, numberParticles)**: Sets the positions to compute the mobility of.  
  * **Mdot(forces = null, torques = null, result)**: Computes the deterministic hydrodynamic displacements, i.e applies the mobility operator. If either the onopolar (force) or dipolar (torque) contributions are not desired, the relevant argument can be ommited.  
  * **stochasticDisplacements(result, prefactor = 1)**: Computes the stochastic displacements and multiplies them by the provided prefactor.  
  * **hydrodynamicDisplacements(forces = null, torques = null, result, prefactor = 1)**: Equivalent to calling Mdot followed by stochastichDisplacements (some algorithms might benefit from doing so).  
  * **clean()**: Cleans any memory allocated by the module. The initialization function must be called again in order to use the module again.  
The many examples in this repository offer more insight about the interface and how to use them. See cpp/example.cpp or python/example.py. See solvers/NBody for an example of a module implementation. Even though the algorithm behind it is quite convoluted, the files in this directory are short and simple, since they are only a thin wrapper to the actual algorithm, located under BatchedNBodyRPY there.  
An equal sign denotes defaults.  

### Data format
Positions, forces, torques and the results provided by the functions are packed in a 3*numberParticles contiguos array containing [x_1, y_1, z_1, x_2,...z_N] .  



### Parameters
The valid parameters accepted by the interface are:  
  * **temperature**.  
  * **hydrodynamicRadius**: The hydrodynamic radius of the particles.  
  * **viscosity**: The fluid viscosity.  
  * **tolerance = 1e-4**: If the solver is not exact this provides an error tolerance.  
  * **boxSize**: The domain size in each direction (if the system is open in some or all of them).  
  * **periodicity**: Whether the domain is periodic or not in each direction (if the module can make use of it).  

An equal sign denotes default values.  


