#Raul P. Pelaez 2021-2022. PSE Mobility test.
from PSE import *
import numpy as np

pse = PSE("periodic", "periodic", "periodic")

numberParticles = 10
precision = np.float32 if pse.precision=="float" else np.float64
pos = np.linspace(-10, 10, 3*numberParticles).astype(precision)
force = np.linspace(-1, 1, 3*numberParticles).astype(precision)
result = np.zeros(3*numberParticles).astype(precision)


pse.setParametersPSE(psi=1.0, Lx=32, Ly=32, Lz=32, shearStrain=0)
pse.initialize(temperature=1.0, viscosity=1.0, hydrodynamicRadius=1, numberParticles=numberParticles);
#If the second call to setParametersPSE+initialize only differs in the shear strain the module is not reinitialized. Only the shear strain is updated
#pse.setParametersPSE(psi=1.0, Lx=32, Ly=32, Lz=32, shearStrain=1)
#pse.initialize(temperature=1.0, viscosity=1.0, hydrodynamicRadius=1, numberParticles=numberParticles);
pse.setPositions(pos)
pse.Mdot(forces = force, result = result)
pse.clean()
print(result)
