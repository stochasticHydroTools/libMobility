#Raul P. Pelaez 2021. PSE Mobility python example.
from PSE import *
#help(PSE)
import numpy as np

pse = PSE()

numberParticles = 10
precision = np.float32 if pse.precision=="float" else np.float64 
pos = np.linspace(-10, 10, 3*numberParticles).astype(precision)
force = np.linspace(-1, 1, 3*numberParticles).astype(precision)
result = np.zeros(3*numberParticles).astype(precision)

par = Parameters(temperature = 1.0, viscosity = 1.0, hydrodynamicRadius = 1.0, boxSize=BoxSize(128,128,128))
pse.initialize(par);
pse.setPositions(pos, numberParticles)
pse.Mdot(forces = force, result = result)
pse.clean()
print(result)


