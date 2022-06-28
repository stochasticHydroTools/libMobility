from NBody import *
import numpy as np

#Constructor requires periodicity in each dimension
nb = NBody("open", "open", "open")

numberParticles = 10
precision = np.float32 if nb.precision=="float" else np.float64
pos = np.linspace(-10, 10, 3*numberParticles).astype(precision)
force = np.linspace(-1, 1, 3*numberParticles).astype(precision)
result = np.zeros(3*numberParticles).astype(precision)

par = Parameters(temperature = 0.0, viscosity = 1.0, hydrodynamicRadius = 1.0)
nb.initialize(par);
#Batched parameters to -1 turns the functionality off (default)
nb.setParametersNBody(algorithm="advise", Nbatch=-1, NperBatch=-1);
nb.setPositions(pos, numberParticles)
nb.Mdot(forces = force, result = result)
nb.clean()
print(result)
