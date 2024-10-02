# libMobility

Documentation is available at [libmobility.readthedocs.io](https://libmobility.readthedocs.io)  

This repository contains several GPU solvers that can compute the action of the hydrodynamic mobility (at the RPY/FCM level) of a group of particles (in different geometries) with forces acting on them.  

In particular, given a group of forces, $\boldsymbol{F}$, and torques, $\boldsymbol{\tau}$, acting on a group of positions, $\boldsymbol{X}$ and directions $\boldsymbol{\tau}, the libMobility solvers can compute:

$$\begin{bmatrix}d\boldsymbol{X}\\
d\boldsymbol{\tau}\end{bmatrix} = \boldsymbol{\mathcal{\Omega}}\begin{bmatrix}\boldsymbol{F}\\
\boldsymbol{T}\end{bmatrix}dt + \text{prefactor}\sqrt{2 k_B T \boldsymbol{\mathcal{\Omega}}}d\boldsymbol{W}$$  


Where $d\boldsymbol{X}$ are the linear displacements, $\boldsymbol{d\tau}$ are the angular displacements, $\boldsymbol{\mathcal{\Omega}}$ is the grand mobility tensor, $\boldsymbol{F}$ are the forces, $\boldsymbol{T}$ are the torques, $\text{prefactor}$ is an user provided prefactor and $d\boldsymbol{W}$ is a collection of i.i.d Weinner processes and $T$ is the temperature.

Each solver in libMobility allows to compute either the deterministic term, the stochastic term, or both at the same time.  

For each solver, a python and a C++ interface are provided. All solvers have the same interface, although some input parameters might change (an open boundaries solver does not accept a box size as a parameter).  

## Repository Structure  

This repository is organized into the following directories:  

- **solvers/**: This directory hosts a subfolder for each solver module. Each subfolder contains the implementation of the `libMobility` interface specific to that solver.  

- **examples/**: Contains examples on how to use the library from Python and C++.  

- **include/**: Includes the essential C++ base classes and utility files needed to construct the modules.  

- **devtools/**: Contains dev specific scripts and files, like a meta.yaml file to build a conda package for libMobility using conda-build.  

- **docs/**: Contains the source files for the documentation.  

- **tests/**: Contains the tests for the library.  




<!-- ## The libMobility interface -->

<!-- Each solver is encased in a single class which is default constructible (no arguments required for its constructor).   -->
<!-- Each solver provides the following set of functions (called the same in C++ and python and described here in a kind of language agnostic way):   -->
<!--   * **[Constructor] (configuration)**: The solver constructors must be provided with a series of system-related parameters (see below).   -->
<!--   * **initialize(parameters)**: Initializes the module according to the parameters (see below).   -->
<!--   * **setParameters[SolverName]([extra parameters])**: Some modules might need special parameters, in these instances this function must also be called. Check the README for each module and its mobility.h file.   -->
<!--   * **setPositions(positions)**: Sets the positions to compute the mobility of.   -->
<!--   * **Mdot(forces, result)**: Computes the deterministic hydrodynamic displacements, i.e applies the mobility operator.  -->
<!--   * **sqrtMdotW(result, prefactor = 1)**: Computes the stochastic displacements and multiplies them by the provided prefactor. The computation will be skipped if prefactor is 0. -->

<!--   * **hydrodynamicVelocities(forces = null, result, prefactor = 1)**: Equivalent to calling Mdot followed by sqrtMdotW (some algorithms might benefit from doing these operations together, e.g., solvers based on fluctuating hydrodynamics).   -->

<!--   * **clean()**: Cleans any memory allocated by the module. The initialization function must be called again in order to use the module again.   -->
<!-- The many examples in this repository offer more insight about the interface and how to use them. See cpp/example.cpp or python/example.py.   -->
<!-- An equal sign denotes defaults.   -->

<!-- ### Data format -->
<!-- Positions, forces, and the results provided by the functions are packed in a 3*numberParticles contiguous array containing ```[x_1, y_1, z_1, x_2,...z_N]```. -->


<!-- ### Parameters -->
<!-- The valid parameters accepted by the interface are:   -->
<!--   * **temperature**. In units of energy (AKA k_BT).   -->
<!--   * **hydrodynamicRadius**: The hydrodynamic radii of the particles. Note that many solvers only allow for all particles having the same radius, in those cases this vector should be of size one.   -->
<!--   * **viscosity**: The fluid viscosity.   -->
<!--   * **tolerance = 1e-4**: Tolerance for the Lanczos algorithm.   -->
<!--   * **numberParticles**: The number of particles   -->

<!-- An equal sign denotes default values.   -->

<!-- ### Configuration parameters -->
<!-- At contruction, solvers must be provided with the following information: -->
<!--   * **periodicityX**, **periodicityY**, **periodicityZ**: The periodicity, can be any of "periodic", "open", "single_wall", "two_walls", "unspecified".   -->
  
<!-- The solvers constructor will check the provided configuration and throw an error if something invalid is requested of it (for instance, the PSE solver will complain if open boundaries are chosen). -->


## Installation

You can install libMobility latest release through our conda channel:

```shell
$ conda install -c conda-forge -c stochasticHydroTools libmobility
```

Check the documentation for additional information on installation, such as how to compile from source.


## Python Usage

Importing libMobility will make available any module under "solvers".  

An usage example is available in python/example.py.  
Calling
```python
	help(SolverName)
```
will provide more in depth information about the solver.  

## C++ Usage

In order to use a module called SolverName, the header solvers/SolverName/mobility.h must be included.  
If the module has been compiled correctly the definitions required for the functions in mobility.h will be available at solvers/SolverName/mobility.so.  
An example is available in cpp/example.cpp.  
