import pytest
from libMobility import *
import numpy as np
from utils import (
    get_sane_params,
    initialize_solver,
    solver_configs_all,
    solver_configs_torques,
    generate_positions_in_box,
)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_contiguous(Solver, periodicity):

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity)

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

    numberParticles = 2
    solver = initialize_solver(Solver, periodicity, numberParticles)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    mf, _ = solver.Mdot(forces)
    assert mf.shape == forces.shape

    forces = forces.reshape(numberParticles * 3)
    mf, _ = solver.Mdot(forces)
    assert mf.shape == forces.shape

    position = positions.reshape(numberParticles * 3)
    solver.setPositions(position)
    mf, _ = solver.Mdot(forces)
    assert mf.shape == forces.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_returns_mf_mt(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, needsTorque=True, parameters=parameters
    )

    shapes = [[(3,), (1, 3)], [(1, 3), (3,)], [(1, 3), (1, 3)], [(3,), (3,)]]

    for f_shapes, t_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        torques = np.random.rand(*t_shapes).astype(precision)

        solver.setPositions(positions)
        u, w = solver.Mdot(forces, torques)
        assert u.shape == forces.shape
        assert w.shape == torques.shape

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        u, w = solver.Mdot(forces, torques)
        assert u.shape == forces.shape
        assert w.shape == torques.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_sqrtM(Solver, periodicity):

    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, parameters=parameters, temperature=1.0
    )
    # Set precision to be the same as compiled precision
    positions = generate_positions_in_box(parameters, numberParticles)
    solver.setPositions(positions)
    sqrtmw, _ = solver.sqrtMdotW()
    assert sqrtmw.shape == positions.shape

    positions = positions.reshape(numberParticles * 3)
    solver.setPositions(positions)
    sqrtmw, _ = solver.sqrtMdotW()
    assert sqrtmw.shape == positions.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_hydrodisp(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, parameters=parameters, temperature=1.0, needsTorque=False
    )

    shapes = [(3,), (1, 3), (1, 3), (3,)]

    for f_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        solver.setPositions(positions)
        v, _ = solver.hydrodynamicVelocities()
        assert v.shape == positions.shape
        v, _ = solver.hydrodynamicVelocities(forces)
        assert v.shape == forces.shape

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        v, _ = solver.hydrodynamicVelocities()
        assert v.shape == positions.shape
        v, _ = solver.hydrodynamicVelocities(forces)
        assert v.shape == forces.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_returns_hydrodisp_torques(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, parameters=parameters, temperature=1.0, needsTorque=True
    )

    shapes = [[(3,), (1, 3)], [(1, 3), (3,)], [(1, 3), (1, 3)], [(3,), (3,)]]

    for f_shapes, t_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        torques = np.random.rand(*t_shapes).astype(precision)
        solver.setPositions(positions)
        v, _ = solver.hydrodynamicVelocities()
        assert v.shape == positions.shape
        v, w = solver.hydrodynamicVelocities(forces, torques)
        assert v.shape == forces.shape
        assert w.shape == torques.shape

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        v, _ = solver.hydrodynamicVelocities()
        assert v.shape == positions.shape
        v, w = solver.hydrodynamicVelocities(forces, torques)
        assert v.shape == forces.shape
        assert w.shape == torques.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_no_torques_error(Solver, periodicity):
    # Test that the solver raises an error if torques are provided but solver was not initialized with needsTorque=True
    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, needsTorque=False)

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
    solver = initialize_solver(Solver, periodicity)

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
@pytest.mark.parametrize("shape", ((4, 6), (3, 1), (1, 4)))
def test_bad_positions_shape(Solver, periodicity, shape):
    solver = initialize_solver(Solver, periodicity)

    precision = np.float32 if Solver.precision == "float" else np.float64

    positions = np.random.rand(*shape).astype(precision)

    with pytest.raises(RuntimeError):
        solver.setPositions(positions)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("needsTorque", [True, False])
def test_hydrodisp_equivalent(Solver, periodicity, needsTorque):
    #  Check that calling Mdot is equivalent to calling hydrodynamicVelocities with temperature = 0
    if needsTorque and Solver.__name__ == "PSE":
        pytest.skip("PSE does not support torques")

    numberParticles = 1
    solver = initialize_solver(
        Solver,
        periodicity,
        needsTorque=needsTorque,
        temperature=0.0,
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


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("needsTorque", [True, False])
def test_changing_number_particles(Solver, periodicity, needsTorque):
    if needsTorque and Solver.__name__ == "PSE":
        pytest.skip("PSE does not support torques")

    solver = initialize_solver(
        Solver,
        periodicity,
        needsTorque=needsTorque,
        temperature=1.0,
    )
    for numberParticles in [1, 2, 3]:
        # Set precision to be the same as compiled precision
        positions = np.random.rand(numberParticles, 3)
        forces = np.random.rand(numberParticles, 3)
        torques = np.random.rand(numberParticles, 3)
        solver.setPositions(positions)
        args = (forces, torques) if needsTorque else (forces,)
        mf, mt = solver.Mdot(*args)
        assert mf.shape == (numberParticles, 3)
        dwf, dmt = solver.sqrtMdotW()
        assert dwf.shape == (numberParticles, 3)
        bothmf, bothmt = solver.hydrodynamicVelocities(*args)
        assert bothmf.shape == (numberParticles, 3)
        if needsTorque:
            assert bothmt.shape == (numberParticles, 3)
            assert dmt.shape == (numberParticles, 3)
            assert mt.shape == (numberParticles, 3)
