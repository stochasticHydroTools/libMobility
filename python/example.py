'''
Raul P. Pelaez 2021. libMobility python interface usage example.
One of the available solvers is chosen and then, using the same interface the solver is initialized and the mobility is applied.

'''
import numpy as np
import libMobility
#Each solver has its own help available, and libMobility itself offers information about the common interface.
#help(libMobility)
#help(libMobility.NBody)
#help(libMobility.PSE)
#help(...

numberParticles = 3
#Each module can be compiled using a different precision, you can know which one with this
precision = np.float32 if libMobility.SelfMobility.precision=="float" else np.float64 
pos = np.random.rand(3*numberParticles).astype(precision)
force = np.ones(3*numberParticles).astype(precision)
result = np.zeros(3*numberParticles).astype(precision)

#Any solver can be used interchangeably, some of them need additional initialization via a "setParameters" function


nb = libMobility.SelfMobility(dimension=3, periodicity='open', numberSpecies=1, device='cpu')


#nb = libMobility.NBody(dimension=3, periodicity='open', numberSpecies=1, device='gpu')

#nb = libMobility.NBody_wall(dimension=3, periodicity='single_wall', numberSpecies=1, device='gpu')
#nb.setParametersNBody_wall(Lx=128, Ly=128, Lz=128);

#nb = libMobility.PSE(dimension=3, periodicity='triply_periodic', numberSpecies=1, device='gpu')
#nb.setParametersPSE(psi=1, Lx=128, Ly=128, Lz=128 );

nb.initialize(temperature=1.0, viscosity = 1/(6*np.pi),
              hydrodynamicRadius = 1.0,
              numberParticles = numberParticles)
nb.setPositions(pos)
#result = M*F
nb.Mdot(forces = force, result = result)
print(f"{numberParticles} particles located at ( X Y Z ): {pos}")
print("Forces:", force)
print("M*F:", result)
#result = prefactor*sqrt(2*temperature*M)*dW
nb.stochasticDisplacements(prefactor = 1.0, result = result)
print("sqrt(2*T*M)*N(0,1):", result)
nb.clean()



