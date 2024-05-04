import pytest
from libMobility import *
import numpy as np


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [(SelfMobility, ("open", "open", "open"))],
)
def test_contiguous(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    solver.setParameters(parameter=1)
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)

    solver.setPositions(positions)

    # Set positions so that they are not contiguous
    matrix = np.random.rand(numberParticles * 3, numberParticles * 3).astype(precision)
    positions = matrix[:, 0]
    with pytest.raises(RuntimeError):
        solver.setPositions(positions)


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (SelfMobility, ("open", "open", "open")),
        (PSE, ("periodic", "periodic", "periodic")),
        (NBody, ("open", "open", "open")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
def test_returns(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    solver.setParameters(parameter=1)
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    mf = solver.Mdot(forces)
    assert mf.shape == forces.shape
    forces = forces.reshape(3 * numberParticles)
    mf = solver.Mdot(forces)
    assert mf.shape == forces.shape
