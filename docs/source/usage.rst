Usage
-----

libMobility offers a set of solvers to compute the hydrodynamic displacements of particles in a fluid under different constraints and boundary conditions. The solvers are written in C++ and wrapped in Python.

In particular, libMobility solvers can compute the different elements in the right hand side of the following stochastic differential equation:

.. math::
   
   d\boldsymbol{X} = \boldsymbol{\mathcal{M}}\boldsymbol{F}dt + \text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}

.. hint:: See the :ref:`solvers` section for a list of available solvers.


The libMobility interface
~~~~~~~~~~~~~~~~~~~~~~~~~

All solvers present the following common methods:


.. py:class:: Solver

   .. py:function:: Solver(periodicityX, periodicityY, periodicityZ)
		  
     The constructor of the solver. Only requires to know the periodicity of the system in the x, y, and z directions.
  
     :param str periodicityX: The periodicity of the system in the x direction.
     :param str periodicityY: The periodicity of the system in the y direction.
     :param str periodicityZ: The periodicity of the system in the z direction.

   .. py:function:: setParameters(**kwargs)

      Sets the parameters of the solver. The parameters that can be set are specific to each solver. Check the Solver page for specifics.
      **This function must be called before the initialize function.**

      :param kwargs: The parameters to set.
		     
   .. py:function:: initialize(temperature, viscosity, hydrodynamicRadius, numberParticles)

     Initializes the solver with the following parameters:
 
     :param float temperature: The temperature of the system.
     :param float viscosity: The viscosity of the fluid.
     :param float hydrodynamicRadius: The hydrodynamic radius of the particles.
     :param int numberParticles: The number of particles in the system.


   .. py:function:: setPositions(positions)

      Sets the positions of the particles in the system.

      :param numpy.ndarray positions: The positions of the particles in the system. The array must have a length of 3 times the number of particles in the system.

   .. py:function:: Mdot(forces, result)
		 
      Computes the deterministic hydrodynamic displacements of the particles in the system.

     :param numpy.ndarray forces: The forces acting on the particles in the system. The array must have a length of 3 times the number of particles in the system.
     :param numpy.ndarray result: The array where the displacements will be stored. The array must have a length of 3 times the number of particles in the system.
  
  .. py:function:: sqrtMdotW(prefactor, result)
  		 
     Computes the stochastic displacements of the particles in the system.
  
     :param float prefactor: The prefactor to multiply the stochastic displacements by.
     :param numpy.ndarray result: The array where the displacements will be stored. The array must have a length of 3 times the number of particles in the system.
  
  .. py:function:: clean()
  		 
     Cleans the memory allocated by the solver. The initialization function must be called again in order to use the solver again.
  

   
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
