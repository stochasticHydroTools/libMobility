Usage
-----

libMobility offers a set of solvers to compute the hydrodynamic displacements of particles in a fluid under different constraints and boundary conditions. The solvers are written in C++ and wrapped in Python.

In particular, libMobility solvers can compute the different elements on the right hand side of the following stochastic differential equation:

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

.. hint:: See the :ref:`solvers` section for a list of available solvers.

.. warning:: libMobility does *not* include the thermal drift :math:`k_B T \nabla_{\boldsymbol{X}} \cdot M` and the user must supply their own implementation in order to maintain detailed balance. The thermal drift can be approximated in libMobility using Random Finite Differences (RFD)  

	.. math::

	   \nabla_{\boldsymbol{X}} \cdot \mathcal{M} = \lim_{\delta \to 0} \frac{1}{\delta} \left\langle \mathcal{M}\left(\boldsymbol{X} + \frac{\delta}{2} \boldsymbol{W}  \right) - \mathcal{M}\left(\boldsymbol{X} - \frac{\delta}{2} \boldsymbol{W}  \right) \right\rangle_{\boldsymbol{W}}, \hspace{1cm} \boldsymbol{W} \sim \mathcal{N}\left(0,1 \right)

Each solver in libMobility allows to compute either the deterministic term, the stochastic term, or both at the same time.  


The libMobility interface
~~~~~~~~~~~~~~~~~~~~~~~~~

All solvers present the same set of methods, which are used to set the parameters of the solver, initialize it, and compute the different terms in the equation above.

As an example, the :py:mod:`libMobility.SelfMobility` solver has the following API:

.. autoclass:: libMobility.SelfMobility
   :members:
   :inherited-members:
   :no-index:
  

   
Example
~~~~~~~

The following example shows how to use the :py:class:`libMobility.SelfMobility` solver to compute the different terms in the equation above.

A libMobility solver is initialized in three steps:

1. Creation of the solver object, specifying the periodicity of the system.
2. Setting the parameters proper to the solver.
3. Initialization of the solver with global the parameters.

.. code:: python


		import numpy as np
		from libMobility import SelfMobility
		numberParticles = 100;
		precision = np.float32 if SelfMobility.precision == "float" else np.float64
		solver = SelfMobility("open", "open", "open")
		solver.setParameters(parameter=5) # SelfMobility only exposes an example parameter
		solver.initialize(
		  temperature=0.0,
		  viscosity=1.0,
		  hydrodynamicRadius=1.0,
		  numberParticles=numberParticles,
		  needsTorque=True,
		)
		positions = np.random.rand(numberParticles, 3).astype(precision)
		forces = np.random.rand(numberParticles, 3).astype(precision)
		torques = np.random.rand(numberParticles, 3).astype(precision)
		solver.setPositions(positions)
		linear, angular = solver.Mdot(forces, torques)
		
