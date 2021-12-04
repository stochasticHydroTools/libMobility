import numpy as np
import libMobility

#help(libMobility)
#help(libMobility.NBody)
#help(libMobility.PSE)

numberParticles = 10
precision = np.float32 if libMobility.NBody.precision=="float" else np.float64 
pos = np.linspace(-10, 10, 3*numberParticles).astype(precision)
force = np.linspace(-1, 1, 3*numberParticles).astype(precision)
result = np.zeros(3*numberParticles).astype(precision)
par = libMobility.Parameters(temperature = 0.0, viscosity = 1.0, hydrodynamicRadius = 1.0)

nb = libMobility.NBody()
nb.initialize(par);
nb.setPositions(pos, numberParticles)
nb.Mdot(forces = force, result = result)
nb.clean()
print(result)

par.boxSize = libMobility.BoxSize(128, 128, 128)
pse = libMobility.PSE()
pse.initialize(par);
pse.setPositions(pos, numberParticles)
pse.Mdot(forces = force, result = result)
pse.clean()
print(result)



