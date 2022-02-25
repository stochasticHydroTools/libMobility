# libMobility
This repository contains several solvers that can compute the action of the hydrodynamic mobility (at the RPY level) of a group of particles (in different geometries) with forces and/or torques acting on them.  

In particular, given a group of forces and torques, T=[forces; torques], the libMobility solvers can compute:

	[dX, dA] = M·T + prefactor*sqrt(2*temperature*M)·dW  

Where dX and dA are the linear and angular displacements respectively, prefactor is an user provided prefactor and dW is a collection of i.i.d Weinner processes.  
Each solver in libMobility allows to compute either the deterministic term, the stochastic term or both at the same time.  

For each solver, a python and a C++ interface are provided. All solvers have the same interface, although some input parameters might change (an open boundaries solver does not accept a box size as a parameter).  

Some of the solvers have different functionalities than the others. For instance, some modules might be able to accept torques while others dont. The documentation for a particular solver must be visited to know more.  

## How this repository is organized

The directory **solvers** contains a subfolder for each module. Each module exists on a different git repository (and included here as a submodule). Besides the submodule repo, the folder for each solver contains a wrapper to the afforementioned repository that allows to use it under a common interface.  
The directory **python** contains the file libMobility.py, which can be imported giving access to all the compiled modules.  
The directory **cpp** contains an example usage of the C++ interface to the modules.  
The directory **include** contains the C++ base class and utilities used to construct the modules.  

## The libMobility interface
Each solver is encased in a single class which is default constructible (no arguments required for its constructor).  
Each solver provides the following set of functions (called the same in C++ and python and described here in a kind of language agnostic way):  
  * **[Constructor] (configuration)**: The solver constructors must be provided with a series of system-related parameters (see below).  
  * **initialize(parameters)**: Initializes the module according to the parameters (see below).  
  * **setParameters[SolverName]([extra parameters])**: Some modules might need special parameters, in these instances this function must also be called.  
  * **setPositions(positions)**: Sets the positions to compute the mobility of.  
  * **Mdot(forces = null, torques = null, result)**: Computes the deterministic hydrodynamic displacements, i.e applies the mobility operator. If either the monopolar (force) or dipolar (torque) contributions are not desired, the relevant argument can be ommited.  
  * **stochasticDisplacements(result, prefactor = 1)**: Computes the stochastic displacements and multiplies them by the provided prefactor.  
  * **hydrodynamicDisplacements(forces = null, torques = null, result, prefactor = 1)**: Equivalent to calling Mdot followed by stochastichDisplacements (some algorithms might benefit from doing these operations together, e.g., solvers based on fluctuating hydrodynamics).  
  * **clean()**: Cleans any memory allocated by the module. The initialization function must be called again in order to use the module again.  
The many examples in this repository offer more insight about the interface and how to use them. See cpp/example.cpp or python/example.py. See solvers/NBody for an example of a module implementation. Even though the algorithm behind it is quite convoluted, the files in this directory are short and simple, since they are only a thin wrapper to the actual algorithm, located under BatchedNBodyRPY there.  
An equal sign denotes defaults.  

### Data format
Positions, forces, torques and the results provided by the functions are packed in a 3*numberParticles contiguos array containing [x_1, y_1, z_1, x_2,...z_N] .  



### Parameters
The valid parameters accepted by the interface are:  
  * **temperature**. In units of energy (AKA k_BT).  
  * **hydrodynamicRadius**: The hydrodynamic radii of the particles. Donev: [CHECK and CLARIFY] This can be one radius per species, or one radius per particle ???
  * **viscosity**: The fluid viscosity.  
  * **tolerance = 1e-4**: If the solver is not exact this provides an error tolerance. Donev: What about Lanczos tolerance?  

Donev: What about numberParticles?
An equal sign denotes default values.  

### Configuration parameters
At contruction, solvers must be provided with the following information:
  * **device**. Can be either "cpu", "gpu" or "automatic".  
  * **dimension**: The dimensionality of the Stokes flow problem (typically 3 but solvers may work in 2D also).  
  * **numberSpecies**: The number of different species/types of particles. Donev: This is unclear and not so useful. There is no common interface for setting species of particles so why is it useful to have numberSpecies? Without any example of use of this it is hard to imagine how touse this. Maybe delete / discuss?
  * **periodicity**: Donev: [Added some clarifications] The periodicity, can any of "triply_periodic", "doubly_periodic" (periodic in xy but not z), "single_wall" (unbounded in xy but wall at z=0), "open", "unspecified".  Donev: This clearly assumes 3D and also assumes that only things we know how to implement now make sense. For example, how about "singly_periodic" or "slit_channel"? The most flexible and best way to do this is not via an enumerator but rather a vector periodic of size [2,dimension], where 1 means periodic along that direction (both must be 1], -1 means wall, and 0 means open. Needs discussion...
  
The solvers constructor will check the provided configuration and throw an error if something invalid is requested of it (for instance, the PSE solver will complain if open boundaries are chosen).



## How to use this repo
Each solver is included as a git submodule (So that each solver has its own, separated, repository). Be sure to clone this repository recursively (using ```git clone --recurse```).  
After compilation (see below) a python

## Compilation
Running ```make``` on the root of the repository will compile all modules under the solvers directory as long as they adhere to the conventions described in "Adding a new solver".  
Compilation for each module happens separatedly. Since each module might have its own particular dependencies, it is quite possible that compilation fails for some of them. The user must manually address these issues by modifying the relevant Makefiles for the modules they intend to make use of.  
Note that only the modules that are going to be used need to be compiled. It is possible to compile only a particular module (or list of them) by calling:  
```
make solvers/SolverName/
```
Where SolverName is the name of the solver directory, for instance, NBody.  
Any uncompiled modules will simply be ignored by python/libMobility.py (although a warning is issued when importing).  
## Python Usage

Importing libMobility.py will make available any module under "solvers" that has been compiled correctly (or, in the case of python-only modules, any module that provides a valid SolverName.py script).  

An usage example is available in python/example.py.  
Calling
```python
	help(SolverName)
```
will provide more in depth information about the solver.  

## C++ Usage

In order to use a module called SolverName, the header solvers/SolverName/mobility.h must be included.  
If the module has been compiled correctly the definitions required for the functions in mobility.h will be available at solvers/SolverName/mobility.so.  
An example is available in cpp/example.cpp and the accompanying Makefile.  
## Adding a new solver

Solvers must be added following the ```libmobility::Mobility``` directives (see include/MobilityInterface). This C++ base class joins every solver under a common interface. When the C++ interface is prepared, the python bindings can be added automagically by using ```pythonify.h``` (a tool under MobilityInterface).  

Solvers are exposed to Python and C++ via a single class that inherits from ```libmobility::Mobility```.  

A new solver must add a directory under the "solvers", which must be the name that both the python and C++ classes.  

Under the directory proper to this new module, the repository containing the solver must be added as a submodule. See for instance solvers/NBody, whose actual code is under solvers/NBody/BatchedNBodyRPY (a git submodule).

Besides the folder containing the repository, inside the solver directory a series of files (and only these files) must exist (additional files might be placed under an "extra" directory inside the solver directory). These files are:  
  * **mobility.h**: Must include ```MobilityInterface.h``` and define a single class, named as the directory for the solver, that inherits the ```libmobility::Mobility``` base class. For instance, in solvers/NBody/mobility.h we have:  
  ```c++
  ...
  class NBody: public libmobility::Mobility{
  ...
  ```
Every pure virtual function in ```libmobility::Mobility``` must be overriden and defined. See more about the mobility interface in include/MobilityInterface.  
mobility.h can include and make use of any existent utilities/files from the solvers repository.  
  * **python_wrapper.cpp**: This file must provide the python bindings for the class in mobility.h. If this class follows the ```libmobility::Mobility``` correctly, this file can in general be quite simple, having only a single line using the ```MOBILITY_PYTHONIFY``` utility in include/MobilityInterface/pythonify.h. See solvers/NBody/python_wrapper.cpp for an example.  
  In the case of a module being python only (or in general not providing a correct child of ```libmobility::Mobility```), python_wrapper.cpp might be ommited, but a file called [solver].py must exist instead, providing a python class that is compatible with ```libmobility::Mobility``` (so that the user can write ```from solver import *``` and get a class, called "solver" that adheres to the libmobility requirements).  
  * **Makefile**: This Makefile must contain rules to create two files:  
	* **mobility.so**: A shared library that provides access to the solver's C++ class.  
	* **[solver].[python-config --extension-suffix]**: A shared library that provides access to the solver's Python class.  
Furthermore, the Makefile must provide an "all" rule (which creates both libraries) and a "clean" rule.  
In order to create the two libraries, the Makefile has the freedom to call, for instance, any build tools in the solvers repository (such as another Makefile).  
 * **example.py**: An example script using the python bindings of the module.  
 * **test.py**: A correctness test (as simple as possible) that ensures the module is working as intended.  
 * **README.md**: The documentation for the specific module (stating the geometry, required arguments,...).  
If these conventions are followed, the Makefile in the root directory will compile the module along the rest and libMobility.py will import it without any modifications to each of them.  

Regarding the different functions of the interface, some of them provide default behavior if not defined. In particular, the stochastich displacements will be computed using a Lanczos solver if the module does not override the corresponding function. Additionally, the hydrodynamicDisplacements functions defaults to calling Mdot followed by stochasticDisplacements. Finally, the clean function defaults to doing nothing.  
An example of this is NBody, which only provides an initialization and Mdot functions.  

**The initialize function of a new solver must call the ```libmobility::Mobility::initialize``` function at some point.**  

**See solvers/SelfMobility for a basic example.**

When a module needs additional parameters to those provided to initialize an additional function, called ```setParameters[SolverName]``` must be defined and exposed to python. See solvers/PSE/mobility.h and solver/PSE/python_wrapper.cpp for an example. Donev: It is up to users of the library to call setParameters before calling initialize with the required arguments.

