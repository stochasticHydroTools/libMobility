"""
Raul P. Pelaez 2021. libMobility python interface usage example.
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

# Any solver can be used interchangeably, some of them need additional initialization via a "setParameters" function

nb = libMobility.SelfMobility(
    periodicityX="open", periodicityY="open", periodicityZ="open"
)
# to call with the alternate import, use the below
# nb = SelfMobility(periodicityX='open',periodicityY='open',periodicityZ='open')

# For NBody periodicityZ can also be single_wall
# nb = libMobility.NBody(periodicityX='open',periodicityY='open',periodicityZ='open')
# nb.setParameters(algorithm="advise", Nbatch=1, NperBatch=numberParticles)

# nb = libMobility.PSE(periodicityX='periodic',periodicityY='periodic',periodicityZ='periodic')
# nb.setParameters(psi=1,   Lx=128, Ly=128, Lz=128,shearStrain=1)

nb.initialize(
    temperature=1.0,
    viscosity=1 / (6 * np.pi),
    hydrodynamicRadius=1.0,
    numberParticles=numberParticles,
    needsTorque=False,
)
nb.setPositions(pos)

result, _ = nb.Mdot(forces=force)
print(f"{numberParticles} particles located at ( X Y Z ): {pos}")
print("Forces:", force)
print("M*F:", result)
# result = prefactor*sqrt(2*temperature*M)*dW
result = nb.sqrtMdotW(prefactor=1.0)
print("sqrt(2*T*M)*N(0,1):", result)
nb.clean()
