import pytest
from libMobility import *
import numpy as np
from utils import sane_parameters


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
def test_contiguous(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
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
def test_returns_mf(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
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
    mf, _ = solver.Mdot(forces)
    assert mf.shape == (numberParticles, 3)
    forces = forces.reshape(3 * numberParticles)
    mf, _ = solver.Mdot(forces)
    assert mf.shape == (numberParticles, 3)


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
def test_returns_sqrtM(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
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
    sqrtmw, _ = solver.sqrtMdotW()
    assert sqrtmw.shape == (numberParticles, 3)


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
def test_returns_hydrodisp(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
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
    sqrtmw, _ = solver.hydrodynamicVelocities()
    assert sqrtmw.shape == (numberParticles, 3)
    sqrtmw, _ = solver.hydrodynamicVelocities(forces)
    assert sqrtmw.shape == (numberParticles, 3)
