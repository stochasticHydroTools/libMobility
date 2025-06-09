"""
Raul P. Pelaez 2021-2025. libMobility python interface usage example.
One of the available solvers is chosen and then, using the same interface the solver is initialized and the mobility is applied.
"""

import numpy as np
import libMobility

# can also import modules individually by name, e.g.
# from libMobility import SelfMobility
# from libMobility import PSE
# ... etc

# Each solver has its own help available, and libMobility itself offers information about the common interface.
# help(libMobility)
# help(libMobility.NBody)
# help(libMobility.PSE)
# help(...


numberParticles = 3
# Each module can be compiled using a different precision, you can know which one with this
precision = np.float32 if libMobility.SelfMobility.precision == "float" else np.float64
pos = np.random.rand(3 * numberParticles).astype(precision)
force = np.ones(3 * numberParticles).astype(precision)

# All solvers present a common interface, but some of them
# need additional initialization via a "setParameters" function
solver = libMobility.SelfMobility(
    periodicityX="open", periodicityY="open", periodicityZ="open"
)

# For the NBody solver, periodicityZ can also be single_wall, e.g.
# solver = libMobility.NBody(periodicityX='open',periodicityY='open',periodicityZ='open')
# solver.setParameters(algorithm="advise")

# solver = libMobility.PSE(periodicityX='periodic',periodicityY='periodic',periodicityZ='periodic')
# solver.setParameters(psi=1, Lx=128, Ly=128, Lz=128,shearStrain=1)

solver.initialize(
    temperature=1.0,
    viscosity=1.0,
    hydrodynamicRadius=1.0,
    includeAngular=False,
)
solver.setPositions(pos)

# Some solvers can include angular velocities by changing includeAngular=True in initialize().
# The return is always a tuple of (linear_velocities, angular_velocities).
linear_velocities, _ = solver.Mdot(forces=force)
print(f"{numberParticles} particles located at ( X Y Z ): {pos}")
print("Forces:", force)
print("M*F:", linear_velocities)

linear_fluctuations, _ = solver.sqrtMdotW()
print("M^{1/2} * dW:", linear_fluctuations)

# Some solvers (e.g. SelfMobility) have no thermal drift and return all zeros.
# In general, the thermal drift is non-zero and other solvers (e.g. DPStokes) return a non-zero value.
linear_drift, _ = solver.thermalDrift()
print("Thermal drift:", linear_drift)
solver.clean()
