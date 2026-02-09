import pytest

# from libMobility import *
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


# without includeAngular=True, Mdot should return only linear forces
@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_includeAngular_false_returns_linear(Solver, periodicity):

    numberParticles = 2
    solver = initialize_solver(Solver, periodicity)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    mf, mt = solver.Mdot(forces)
    assert mf.shape == forces.shape
    assert mt is None

    forces = forces.reshape(numberParticles * 3)
    mf, mt = solver.Mdot(forces)
    assert mf.shape == forces.shape
    assert mt is None

    position = positions.reshape(numberParticles * 3)
    solver.setPositions(position)
    mf, mt = solver.Mdot(forces)
    assert mf.shape == forces.shape
    assert mt is None


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_torques_returns_both(Solver, periodicity):

    def check_results(mf, mt, ivec, Solver):
        assert mt.shape == ivec.shape
        assert mf.shape == ivec.shape
        assert np.linalg.norm(mt) > 0
        # selfmobility has no hydrodynamic interactions so a torque won't create a linear velocity
        if Solver.__name__ != "SelfMobility":
            assert np.linalg.norm(mf) > 0

    numberParticles = 5
    solver = initialize_solver(Solver, periodicity, includeAngular=True)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    torques = np.random.rand(numberParticles, 3).astype(precision)

    solver.setPositions(positions)
    mf, mt = solver.Mdot(torques=torques)
    check_results(mf, mt, torques, Solver)

    torques = torques.reshape(numberParticles * 3)
    mf, mt = solver.Mdot(torques=torques)
    check_results(mf, mt, torques, Solver)

    position = positions.reshape(numberParticles * 3)
    solver.setPositions(position)
    mf, mt = solver.Mdot(torques=torques)
    check_results(mf, mt, torques, Solver)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_forces_with_includeAngular_returns_both(Solver, periodicity):

    numberParticles = 5
    solver = initialize_solver(Solver, periodicity, includeAngular=True)

    def check_results(mf, mt, ivec, Solver):
        assert mt.shape == ivec.shape
        assert mf.shape == ivec.shape
        assert np.linalg.norm(mf) > 0
        # selfmobility has no hydrodynamic interactions so a force won't create an angular velocity
        if Solver.__name__ != "SelfMobility":
            assert np.linalg.norm(mt) > 0

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)

    solver.setPositions(positions)
    mf, mt = solver.Mdot(forces=forces)
    check_results(mf, mt, forces, Solver)

    forces = forces.reshape(numberParticles * 3)
    mf, mt = solver.Mdot(forces=forces)
    check_results(mf, mt, forces, Solver)

    position = positions.reshape(numberParticles * 3)
    solver.setPositions(position)
    mf, mt = solver.Mdot(forces=forces)
    check_results(mf, mt, forces, Solver)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_returns_mf_mt(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, includeAngular=True, parameters=parameters
    )

    def check_results(mf, mt, fvec, tvec):
        assert mf.shape == fvec.shape
        assert mt.shape == tvec.shape
        assert np.linalg.norm(mf) > 0
        assert np.linalg.norm(mt) > 0

    shapes = [[(3,), (1, 3)], [(1, 3), (3,)], [(1, 3), (1, 3)], [(3,), (3,)]]

    for f_shapes, t_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        torques = np.random.rand(*t_shapes).astype(precision)

        solver.setPositions(positions)
        u, w = solver.Mdot(forces, torques)
        check_results(u, w, forces, torques)

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        u, w = solver.Mdot(forces, torques)
        check_results(u, w, forces, torques)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_sqrtM(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(Solver, periodicity, parameters=parameters)
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
def test_returns_divM(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(Solver, periodicity, parameters=parameters)

    positions = generate_positions_in_box(parameters, numberParticles)
    solver.setPositions(positions)
    sqrtmw, _ = solver.divM()
    assert sqrtmw.shape == positions.shape

    positions = positions.reshape(numberParticles * 3)
    solver.setPositions(positions)
    sqrtmw, _ = solver.divM()
    assert sqrtmw.shape == positions.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_returns_hydrodisp(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, parameters=parameters, includeAngular=False
    )

    shapes = [(3,), (1, 3), (1, 3), (3,)]

    for f_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        solver.setPositions(positions)
        v, _ = solver.LangevinVelocities(1.0, 1.0)
        assert v.shape == positions.shape
        v, _ = solver.LangevinVelocities(1.0, 1.0, forces)
        assert v.shape == forces.shape

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        v, _ = solver.LangevinVelocities(1.0, 1.0)
        assert v.shape == positions.shape
        v, _ = solver.LangevinVelocities(1.0, 1.0, forces)
        assert v.shape == forces.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_returns_hydrodisp_torques(Solver, periodicity):
    numberParticles = 1
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver, periodicity, parameters=parameters, includeAngular=True
    )

    shapes = [[(3,), (1, 3)], [(1, 3), (3,)], [(1, 3), (1, 3)], [(3,), (3,)]]

    for f_shapes, t_shapes in shapes:
        # Set precision to be the same as compiled precision
        precision = np.float32 if Solver.precision == "float" else np.float64
        positions = generate_positions_in_box(parameters, numberParticles)
        forces = np.random.rand(*f_shapes).astype(precision)
        torques = np.random.rand(*t_shapes).astype(precision)
        solver.setPositions(positions)
        v, _ = solver.LangevinVelocities(dt=1.0, kbt=1.0)
        assert v.shape == positions.shape
        v, w = solver.LangevinVelocities(1.0, 1.0, forces, torques)
        assert v.shape == forces.shape
        assert w.shape == torques.shape

        positions = positions.reshape(numberParticles * 3)
        solver.setPositions(positions)
        v, _ = solver.LangevinVelocities(dt=1.0, kbt=1.0)
        assert v.shape == positions.shape
        v, w = solver.LangevinVelocities(1.0, 1.0, forces, torques)
        assert v.shape == forces.shape
        assert w.shape == torques.shape


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_no_torques_error(Solver, periodicity):
    # Test that the solver raises an error if torques are provided but solver was not initialized with includeAngular=True
    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, includeAngular=False)

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
        u, _ = solver.LangevinVelocities(dt=1.0, kbt=1.0)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("shape", ((4, 6), (3, 1), (1, 4)))
def test_bad_positions_shape(Solver, periodicity, shape):
    solver = initialize_solver(Solver, periodicity)

    precision = np.float32 if Solver.precision == "float" else np.float64

    positions = np.random.rand(*shape).astype(precision)

    with pytest.raises(RuntimeError):
        solver.setPositions(positions)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_bad_force_shape(Solver, periodicity):
    numberParticles = 5
    solver = initialize_solver(Solver, periodicity)

    precision = np.float32 if solver.precision == "float" else np.float64
    forces = np.random.rand(3 * (numberParticles - 1)).astype(precision)

    for n_wrong in [numberParticles - 1, numberParticles + 1]:
        forces = np.random.rand(3 * n_wrong).astype(precision)
        with pytest.raises(RuntimeError):
            solver.Mdot(forces=forces)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
def test_bad_torque_shape(Solver, periodicity):
    numberParticles = 5
    solver = initialize_solver(Solver, periodicity, includeAngular=True)
    precision = np.float32 if solver.precision == "float" else np.float64

    for n_wrong in [numberParticles - 1, numberParticles + 1]:
        torques = np.random.rand(3 * n_wrong).astype(precision)
        with pytest.raises(RuntimeError):
            solver.Mdot(torques=torques)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("includeAngular", [True, False])
def test_hydrodisp_equivalent(Solver, periodicity, includeAngular):
    #  Check that calling Mdot is equivalent to calling LangevinVelocities with temperature set to 0
    if includeAngular and Solver.__name__ == "PSE":
        pytest.skip("PSE does not support torques")

    numberParticles = 1
    solver = initialize_solver(Solver, periodicity, includeAngular=includeAngular)

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    torques = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    temp_args = (forces, torques) if includeAngular else (forces,)
    mf, mt = solver.Mdot(*temp_args)
    bothmf, bothmt = solver.LangevinVelocities(1.0, 0.0, *temp_args)
    assert np.allclose(mf, bothmf, atol=1e-6)
    if includeAngular:
        assert np.allclose(mt, bothmt, atol=1e-6)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("includeAngular", [True, False])
def test_changing_number_particles(Solver, periodicity, includeAngular):
    if includeAngular and Solver.__name__ == "PSE":
        pytest.skip("PSE does not support torques")

    solver = initialize_solver(Solver, periodicity, includeAngular=includeAngular)
    for numberParticles in [1, 2, 3]:
        # Set precision to be the same as compiled precision
        positions = np.random.rand(numberParticles, 3)
        forces = np.random.rand(numberParticles, 3)
        torques = np.random.rand(numberParticles, 3)
        solver.setPositions(positions)
        args = (forces, torques) if includeAngular else (forces,)
        mf, mt = solver.Mdot(*args)
        assert mf.shape == (numberParticles, 3)
        dwf, dmt = solver.sqrtMdotW()
        assert dwf.shape == (numberParticles, 3)
        bothmf, bothmt = solver.LangevinVelocities(1.0, 1.0, *args)
        assert bothmf.shape == (numberParticles, 3)
        if includeAngular:
            assert bothmt.shape == (numberParticles, 3)
            assert dmt.shape == (numberParticles, 3)
            assert mt.shape == (numberParticles, 3)


# def test_prefactor(Solver, periodicity):
#     solver = initialize_solver(Solver, periodicity)
#     numberParticles = 1

#     precision = np.float32 if Solver.precision == "float" else np.float64
#     positions = np.random.rand(numberParticles, 3).astype(precision)
#     forces = np.random.rand(numberParticles, 3).astype(precision)
#     solver.setPositions(positions)

#     mf, mt = solver.Mdot(forces=forces, prefactor=2.0)
#     mf_no_prefac, mt_no_prefac = solver.Mdot(forces=forces, prefactor=1.0)
#     assert np.allclose(mf, 2.0 * mf_no_prefac, atol=1e-6)
#     if mt is not None:
#         assert np.allclose(mt, 2.0 * mt_no_prefac, atol=1e-6)

#     sqrtmw, _ = solver.sqrtMdotW(prefactor=2.0)
#     sqrtmw_no_prefac, _ = solver.sqrtMdotW(prefactor=1.0)
#     assert np.allclose(sqrtmw, np.sqrt(2.0) * sqrtmw_no_prefac, atol=1e-6)
