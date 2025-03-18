libMobility: GPU Solvers for Hydrodynamic Mobility
==================================================

libMobility is a C++ library with Python bindings offering several GPU solvers that can compute the action of the hydrodynamic mobility (at the RPY/FCM level) of a group of particles (in different geometries) with forces and torques acting on them.

Functionality
-------------

Given a group of forces, :math:`\boldsymbol{F}`, and torques, :math:`\boldsymbol{\tau}`, acting on a group of positions, :math:`\boldsymbol{X}` and directions :math:`\boldsymbol{\tau}`, the libMobility solvers can compute:

.. math::

   \begin{bmatrix}d\boldsymbol{X}\\d\boldsymbol{\tau}\end{bmatrix} = \boldsymbol{\mathcal{M}}\begin{bmatrix}\boldsymbol{F}\\\boldsymbol{T}\end{bmatrix}dt + \text{prefactor}\sqrt{2 k_B T \boldsymbol{\mathcal{M}}}d\boldsymbol{W}

Where:

- :math:`d\boldsymbol{X}` are the linear displacements
- :math:`d\boldsymbol{\tau}` are the angular displacements
- :math:`\boldsymbol{\mathcal{M}}` is the grand mobility tensor
- :math:`\boldsymbol{F}` are the forces
- :math:`\boldsymbol{T}` are the torques
- :math:`\text{prefactor}` is a user-provided prefactor
- :math:`d\boldsymbol{W}` is a collection of i.i.d Weiner processes
- :math:`T` is the temperature

Solver Capabilities
-------------------

Each solver in libMobility allows computation of:

- The deterministic term
- The stochastic term
- Both terms simultaneously

Interfaces
----------

For each solver, a Python interface is provided.

All solvers have the same interface, although some input parameters might change (e.g., an open boundaries solver does not accept a box size as a parameter).
    

.. toctree::
   :hidden:
      
   installation
   usage
   solvers
   api
   new-solver

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
