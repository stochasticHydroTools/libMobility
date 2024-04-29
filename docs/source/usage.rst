Usage
-----


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
