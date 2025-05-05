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


def thermal_drift_rfd(solver, positions):
    # RFD works by approximating kT\partial_q \dot M = 1/\delta \langle M(q+\delta/2 W)W - M(q-\delta/2 W)W \rangle
    delta = 1e-3

    def thermal_drift_func():
        W = np.random.normal(size=positions.shape).astype(positions.dtype)
        solver.setPositions(positions + delta / 2 * W)
        _tdriftp_m, _tdriftp_d = solver.Mdot(W)
        solver.setPositions(positions - delta / 2 * W)
        _tdriftm_m, _tdriftm_d = solver.Mdot(W)
        _tdrift_m = (_tdriftp_m - _tdriftm_m)/delta
        _tdrift_d = (_tdriftp_d - _tdriftm_d)/delta if _tdriftm_d is not None else None
        return _tdrift_m, _tdrift_d

    solver.setPositions(positions)
    tdrift_m, tdrift_d = average(lambda: average(thermal_drift_func, 400), 100)
    return tdrift_m, tdrift_d


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_thermal_drift_does_not_change_positions(Solver, periodicity):
    # Compute the Mdot with some random forces, then compute the thermal drift, then compute Mdot again
    # Check that Mdot has not changed
    needsTorque = False
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, 10).astype(precision)
    solver.setPositions(positions)
    forces = np.random.normal(size=positions.shape).astype(precision)
    mf, _ = solver.Mdot(forces)
    solver.thermalDrift()
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
@pytest.mark.parametrize("needsTorque", [False, True])
def test_thermal_drift_is_zero(
        Solver, periodicity, hydrodynamicRadius, numberParticles, needsTorque
):
    if not np.all(np.array(periodicity) == "open") and not np.all(
        np.array(periodicity) == "periodic"
    ):
        pytest.skip(
            "Only periodic and open boundary conditions have zero thermal drift"
        )
    if(Solver.__name__ == "PSE" and needsTorque):
        pytest.skip("PSE does not support torques")
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    thermal_drift_m, thermal_drift_d = average(solver.thermalDrift, 3000)
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
@pytest.mark.parametrize("needsTorque", [False, True])
def test_thermal_drift_returns_different_numbers(
        Solver, periodicity, hydrodynamicRadius, numberParticles, needsTorque
):
    if(Solver.__name__ == "PSE" and needsTorque):
        pytest.skip("PSE does not support torques")

    temperature = 1.2
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=temperature,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        needsTorque=needsTorque,
    )
    positions = np.asarray(
        generate_positions_in_box(parameters, numberParticles).astype(precision) * 0.8
    )
    solver.setPositions(positions)
    rfd1,_ = solver.thermalDrift()
    if np.all(rfd1 == 0):
        pytest.skip("RFD is zero, skipping test")
    rfd2,_ = solver.thermalDrift()
    assert np.any(
        np.abs(rfd1 - rfd2) > 1e-5
    ), f"RFD is not different: {np.max(np.abs(rfd1 - rfd2))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [0.95])
@pytest.mark.parametrize("numberParticles", [1, 10])
@pytest.mark.parametrize("needsTorque", [False, True])
def test_thermal_drift_matches_rfd(
        Solver, periodicity, hydrodynamicRadius, numberParticles, needsTorque
):
    if(Solver.__name__ == "PSE" and needsTorque):
        pytest.skip("PSE does not support torques")
    temperature = 1.2
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=temperature,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        needsTorque=needsTorque,
    )
    positions = np.asarray(
        generate_positions_in_box(parameters, numberParticles).astype(precision)
    )
    solver.setPositions(positions)
    reference_m, reference_d = thermal_drift_rfd(solver, positions)
    reference_m = reference_m * temperature
    if reference_d is not None:
        reference_d = reference_d * temperature

    solver.setPositions(positions)
    rfd_m, rfd_d = average(lambda: average(solver.thermalDrift, 400), 100)
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
