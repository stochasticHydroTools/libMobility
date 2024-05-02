Usage
-----

libMobility offers a set of solvers to compute the hydrodynamic displacements of particles in a fluid under different constraints and boundary conditions. The solvers are written in C++ and wrapped in Python.

In particular, libMobility solvers can compute the different elements in the right hand side of the following stochastic differential equation:

.. math::
   
   d\boldsymbol{X} = \boldsymbol{\mathcal{M}}\boldsymbol{F}dt + \text{prefactor}\sqrt{2 k_B T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}

.. hint:: See the :ref:`solvers` section for a list of available solvers.

Where dX are the linear displacements, prefactor is a user provided prefactor and dW is a collection of i.i.d Weinner processes and T is the temperature. Finally $\boldsymbol{\mathcal{M}}$ represents the mobility tensor. 

.. warning:: libMobility does *not* include the thermal drift :math:`k_B T \nabla_{\boldsymbol{X}} \cdot \mathcal{M}` and the user must supply their own implementation in order to maintain detailed ballance. The thermal drift can be approximated in libMobility using Random Finite Differences (RFD)  

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
          import libMobility

	  numberParticles = 3
	  Solver = libMobility.SelfMobility
	  precision = np.float32 if Solver.precision=="float" else np.float64
	  pos = np.random.rand(3*numberParticles).astype(precision)
	  force = np.ones(3*numberParticles).astype(precision)
	  result = np.zeros(3*numberParticles).astype(precision)

          # The solver will fail if it is not compatible with the provided periodicity
	  nb = Solver(periodicityX='open',periodicityY='open',periodicityZ='open')
	  # Other solvers might need an intermediate step here to provide some extra parameters:
	  # nb.setParameters(parameter1 = 1, ...)
	  nb.initialize(temperature=1.0, viscosity = 1/(6*np.pi),
                        hydrodynamicRadius = 1.0,
                        numberParticles = numberParticles)
	  nb.setPositions(pos)
	  nb.Mdot(forces = force, result = result)
	  print(f"{numberParticles} particles located at ( X Y Z ): {pos}")
	  print("Forces:", force)
	  print("M*F:", result)
	  # result = prefactor*sqrt(2*temperature*M)*dW
	  nb.sqrtMdotW(prefactor = 1.0, result = result)
	  print("sqrt(2*T*M)*N(0,1):", result)
	  nb.clean()
