import pytest
import numpy as np
from libMobility import SelfMobility, PSE, NBody, DPStokes
from utils import (
    solver_configs_all,
    solver_configs_torques,
    get_sane_params,
    generate_positions_in_box,
)


def average(function, num_averages):
    result_m, result_d = function()
    for i in range(num_averages - 1):
        new_result_m, new_result_d = function()
        result_m += new_result_m
        if result_d is not None:
            result_d += new_result_d
    if result_d is not None:
        result_d /= num_averages
    return result_m / num_averages, result_d


def divM_rfd(solver, positions):
    # RFD works by approximating \partial_q \dot M = 1/\delta \langle M(q+\delta/2 W)W - M(q-\delta/2 W)W \rangle
    delta = 1e-3

    def rfd_func():
        W = np.random.normal(size=positions.shape).astype(positions.dtype)
        solver.setPositions(positions + delta / 2 * W)
        _tdriftp_m, _tdriftp_d = solver.Mdot(W)
        solver.setPositions(positions - delta / 2 * W)
        _tdriftm_m, _tdriftm_d = solver.Mdot(W)
        _tdrift_m = (_tdriftp_m - _tdriftm_m) / delta
        _tdrift_d = (
            (_tdriftp_d - _tdriftm_d) / delta if _tdriftm_d is not None else None
        )
        return _tdrift_m, _tdrift_d

    solver.setPositions(positions)
    tdrift_m, tdrift_d = average(lambda: average(rfd_func, 400), 100)
    return tdrift_m, tdrift_d


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_divM_does_not_change_positions(Solver, periodicity):
    # Compute the Mdot with some random forces, then compute divM, then compute Mdot again
    # Check that Mdot has not changed
    includeAngular = False
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        includeAngular=includeAngular,
    )
    positions = generate_positions_in_box(parameters, 10).astype(precision)
    solver.setPositions(positions)
    forces = np.random.normal(size=positions.shape).astype(precision)
    mf, _ = solver.Mdot(forces)
    solver.divM()
    mf2, _ = solver.Mdot(forces)
    assert np.allclose(
        mf,
        mf2,
        atol=1e-7,
        rtol=1e-7,
    ), f"Mdot has changed: {np.max(np.abs(mf - mf2))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
@pytest.mark.parametrize("includeAngular", [False, True])
def test_divM_is_zero(
    Solver, periodicity, hydrodynamicRadius, numberParticles, includeAngular
):
    if not np.all(np.array(periodicity) == "open") and not np.all(
        np.array(periodicity) == "periodic"
    ):
        pytest.skip("Only periodic and open boundary conditions have zero divergence")
    if Solver.__name__ == "PSE" and includeAngular:
        pytest.skip("PSE does not support torques")
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        includeAngular=includeAngular,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    thermal_drift_m, thermal_drift_d = average(solver.divM, 3000)
    assert np.allclose(
        np.abs(thermal_drift_m),
        0.0,
        atol=1e-5,
        rtol=0,
    ), f"Linear RFD drift is not zero: {np.max(np.abs(thermal_drift_m))}"
    if thermal_drift_d is not None:
        assert np.allclose(
            np.abs(thermal_drift_d),
            0.0,
            atol=1e-5,
            rtol=0,
        ), f"Dipolar RFD drift is not zero: {np.max(np.abs(thermal_drift_d))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
@pytest.mark.parametrize("includeAngular", [False, True])
def test_divM_returns_different_numbers(
    Solver, periodicity, hydrodynamicRadius, numberParticles, includeAngular
):
    if Solver.__name__ == "PSE" and includeAngular:
        pytest.skip("PSE does not support torques")

    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        includeAngular=includeAngular,
    )
    positions = np.asarray(
        generate_positions_in_box(parameters, numberParticles).astype(precision) * 0.8
    )
    solver.setPositions(positions)
    rfd1, _ = solver.divM()
    if np.all(rfd1 == 0):
        pytest.skip("RFD is zero, skipping test")
    rfd2, _ = solver.divM()
    assert np.any(
        np.abs(rfd1 - rfd2) > 1e-5
    ), f"RFD is not different: {np.max(np.abs(rfd1 - rfd2))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [0.95])
@pytest.mark.parametrize("numberParticles", [1, 10])
@pytest.mark.parametrize("includeAngular", [False, True])
def test_divM_matches_rfd(
    Solver, periodicity, hydrodynamicRadius, numberParticles, includeAngular
):
    if Solver.__name__ == "PSE" and includeAngular:
        pytest.skip("PSE does not support torques")
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        includeAngular=includeAngular,
    )
    positions = np.asarray(
        generate_positions_in_box(parameters, numberParticles).astype(precision)
    )
    solver.setPositions(positions)
    reference_m, reference_d = divM_rfd(solver, positions)

    solver.setPositions(positions)
    rfd_m, rfd_d = average(lambda: average(solver.divM, 400), 100)
    assert np.allclose(
        reference_m,
        rfd_m,
        atol=1e-3,
        rtol=1e-3,
    ), f"Linear RFD does not match: {np.max(np.abs(reference_m - rfd_m))}"
    if reference_d is not None and rfd_d is not None:
        assert np.allclose(
            reference_d,
            rfd_d,
            atol=1e-3,
            rtol=1e-3,
        ), f"Dipole RFD does not match: {np.max(np.abs(reference_d - rfd_d))}"
