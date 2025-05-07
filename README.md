# libMobility

Documentation is available at [libmobility.readthedocs.io](https://libmobility.readthedocs.io)  

This repository contains several GPU solvers that can compute the action of the hydrodynamic mobility (at the RPY/FCM level) of a group of particles (in different geometries) with forces acting on them.  

In particular, given a group of forces, $\boldsymbol{F}$, and torques, $\boldsymbol{\tau}$, acting on a group of positions, $\boldsymbol{X}$ and directions $\boldsymbol{\tau}$, the libMobility solvers can compute:

$$\begin{bmatrix}d\boldsymbol{X}\\
d\boldsymbol{\tau}\end{bmatrix} = \boldsymbol{\mathcal{\Omega}}\begin{bmatrix}\boldsymbol{F}\\
\boldsymbol{T}\end{bmatrix}dt + \text{prefactor}\sqrt{2 k_B T \boldsymbol{\mathcal{\Omega}}}d\boldsymbol{W} + \begin{\bmatrix}k_BT\boldsymbol{\partial}_\boldsymbol{X}\cdot \boldsymbol{\mathcal{M}}\\ \boldsymbol{0}\end{bmatrix}dt$$  


Where $d\boldsymbol{X}$ are the linear displacements, $\boldsymbol{d\tau}$ are the angular displacements, $\boldsymbol{\mathcal{\Omega}}$ is the grand mobility tensor, $\boldsymbol{F}$ are the forces, $\boldsymbol{T}$ are the torques, $\text{prefactor}$ is an user provided prefactor and $d\boldsymbol{W}$ is a collection of i.i.d Weinner processes and $T$ is the temperature.

Each solver in libMobility allows to compute either the deterministic term, the stochastic term, or both at the same time.  

For each solver, a python interface is provided. All solvers have the same interface, although some input parameters might change (an open boundaries solver does not accept a box size as a parameter).  

## Repository Structure  

This repository is organized into the following directories:  

- **solvers/**: This directory hosts a subfolder for each solver module. Each subfolder contains the implementation of the `libMobility` interface specific to that solver.  

- **examples/**: Contains examples on how to use the library from Python and C++.  

- **include/**: Includes the essential C++ base classes and utility files needed to construct the modules.  

- **devtools/**: Contains dev specific scripts and files, like a meta.yaml file to build a conda package for libMobility using conda-build.  

- **docs/**: Contains the source files for the documentation.  

- **tests/**: Contains the tests for the library.  


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
