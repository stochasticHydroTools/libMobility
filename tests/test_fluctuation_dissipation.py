import pytest
import numpy as np

from libMobility import *

PSE.setParameters = PSE.setParametersPSE
NBody.setParameters = NBody.setParametersNBody
# DPStokes.setParameters = DPStokes.setParametersDPStokes

sane_parameters = {
    "PSE": {"psi": 1.0, "Lx": 32, "Ly": 32, "Lz": 32, "shearStrain": 0.0},
    "NBody": {"algorithm": "advise"},
}


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (PSE, ("periodic", "periodic", "periodic")),
        (NBody, ("open", "open", "open")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
def test_fluctuation_dissipation(Solver, periodicity, hydrodynamicRadius):
    solver = Solver(*periodicity)

    solver.setParameters(**sane_parameters[Solver.__name__])
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )
    positions = np.random.rand(numberParticles, 3)
    forces = np.zeros((numberParticles, 3))
    mf = np.zeros((numberParticles, 3))
    solver.setPositions(positions)
    forces[1] = 1.0
    solver.Mdot(forces, mf)
    sqrtmnoise = np.zeros((numberParticles, 3))
    solver.sqrtMdotW(sqrtmnoise, prefactor=1.0)
