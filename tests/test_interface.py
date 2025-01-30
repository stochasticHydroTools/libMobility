import pytest
from libMobility import *
import numpy as np
from utils import (
    sane_parameters,
    initialize_solver,
    solver_configs_all,
    solver_configs_torques,
    generate_positions_in_box,
)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_returns_mf_mt(Solver, periodicity):
    numberParticles = 1
    parameters = sane_parameters[Solver.__name__]
    solver = initialize_solver(
        Solver, periodicity, numberParticles, needsTorque=True, parameters=parameters
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = generate_positions_in_box(parameters, numberParticles)
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_sqrtM(Solver, periodicity):

    numberParticles = 1
    parameters = sane_parameters[Solver.__name__]
    solver = initialize_solver(
        Solver, periodicity, numberParticles, parameters=parameters, temperature=1.0
    )
    # Set precision to be the same as compiled precision
    positions = generate_positions_in_box(parameters, numberParticles)
    solver.setPositions(positions)
    sqrtmw, _ = solver.sqrtMdotW()
    assert sqrtmw.shape == (numberParticles, 3)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_hydrodisp(Solver, periodicity):
    parameters = sane_parameters[Solver.__name__]
    numberParticles = 1
    solver = initialize_solver(
        Solver, periodicity, numberParticles, parameters=parameters, temperature=1.0
    )
    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = generate_positions_in_box(parameters, numberParticles)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    sqrtmw, _ = solver.hydrodynamicVelocities()
    assert sqrtmw.shape == (numberParticles, 3)
    sqrtmw, _ = solver.hydrodynamicVelocities(forces)
    assert sqrtmw.shape == (numberParticles, 3)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_no_positions_error(Solver, periodicity):
    # Test that the solver raises an error if Mdot, sqrtMdot, and hydroDisp are called before setting positions
    if Solver.__name__ == "SelfMobility":
        pytest.skip("SelfMobility does not require positions")
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("needsTorque", [True, False])
def test_hydrodisp_equivalent(Solver, periodicity, needsTorque):
    #  Check that calling Mdot is equivalent to calling hydrodynamicVelocities with temperature = 0
    if needsTorque and Solver.__name__ == "PSE":
        pytest.skip("PSE does not support torques")
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    numberParticles = 1
    solver.initialize(
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
        needsTorque=needsTorque,
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    torques = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    args = (forces, torques) if needsTorque else (forces,)
    mf, mt = solver.Mdot(*args)
    bothmf, bothmt = solver.hydrodynamicVelocities(*args)
    assert np.allclose(mf, bothmf, atol=1e-6)
    if needsTorque:
        assert np.allclose(mt, bothmt, atol=1e-6)
