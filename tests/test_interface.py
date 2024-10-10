import pytest
from libMobility import *
import numpy as np
from utils import sane_parameters, initialize_solver


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

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles)

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

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles)

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
        (NBody, ("open", "open", "open")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
def test_returns_mf_mt(Solver, periodicity):

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles, needsTorque=True)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    torques = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    u, w = solver.Mdot(forces, torques)
    assert u.shape == (numberParticles, 3)
    assert w.shape == (numberParticles, 3)
    forces = forces.reshape(3 * numberParticles)
    torques = torques.reshape(3 * numberParticles)
    u, w = solver.Mdot(forces, torques)
    assert u.shape == (numberParticles, 3)
    assert w.shape == (numberParticles, 3)


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

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles)

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

@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (SelfMobility, ("open", "open", "open")),
        (NBody, ("open", "open", "open")),
        (NBody, ("open", "open", "single_wall")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
def test_no_torques_error(Solver, periodicity):
    # Test that the solver raises an error if torques are provided but solver was not initialized with needsTorque=True
    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles, needsTorque=False)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64

    forces = np.random.rand(numberParticles, 3).astype(precision)
    torques = np.random.rand(numberParticles, 3).astype(precision)
    posistions = np.random.rand(numberParticles, 3).astype(precision)

    solver.setPositions(posistions)

    with pytest.raises(RuntimeError):
        u, w = solver.Mdot(forces, torques)

@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (SelfMobility, ("open", "open", "open")),
        (NBody, ("open", "open", "open")),
        (NBody, ("open", "open", "single_wall")),
        (PSE, ("periodic", "periodic", "periodic")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
def test_no_positions_error(Solver, periodicity):
    # Test that the solver raises an error if Mdot, sqrtMdot, and hydroDisp are called before setting positions
    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, numberParticles)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64

    forces = np.random.rand(numberParticles, 3).astype(precision)

    with pytest.raises(RuntimeError):
        u, _ = solver.Mdot(forces)

    with pytest.raises(RuntimeError):
        sqrtmw, _ = solver.sqrtMdotW()

    with pytest.raises(RuntimeError):
        u, _ = solver.hydrodynamicVelocities()