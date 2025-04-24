import pytest
import numpy as np
from libMobility import SelfMobility, PSE, NBody, DPStokes
from utils import (
    solver_configs_all,
    solver_configs_torques,
    get_sane_params,
    generate_positions_in_box,
)


def online_average(current_mean, current_count, new_value):
    # Update the mean and count
    new_mean = current_mean + (new_value - current_mean) / (current_count + 1)
    return new_mean


def average(function, num_averages):
    result = function()
    for i in range(num_averages - 1):
        result += function()
    return result / num_averages


def thermal_drift_rfd(solver, positions):
    # RFD works by approxmating kT\partial_q \dot M = 1/\delta \langle M(q+\delta/2 W)W - M(q-\delta/2 W)W \rangle
    # Where delta is a small number, and W is a normal random vector of unit length
    delta = 1e-4

    def thermal_drift_func():
        W = np.random.normal(size=positions.shape)
        solver.setPositions(positions + delta / 2 * W)
        _tdrift = solver.Mdot(W)[0]
        solver.setPositions(positions - delta / 2 * W)
        _tdrift -= solver.Mdot(W)[0]
        _tdrift /= delta
        return _tdrift

    tdrift = average(lambda: average(thermal_drift_func, 100), 100)
    solver.setPositions(positions)
    return tdrift


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
def test_thermal_drift_is_zero(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    if not np.all(np.array(periodicity) == "open") and not np.all(
        np.array(periodicity) == "periodic"
    ):
        pytest.skip(
            "Only periodic and open boundary conditions have zero thermal drift"
        )

    needsTorque = False
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
    thermal_drift = average(solver.thermalDrift, 3000)
    assert np.allclose(
        np.abs(thermal_drift),
        0.0,
        atol=1e-5,
        rtol=0,
    ), f"RFD drift is not zero: {np.max(np.abs(thermal_drift))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
def test_thermal_drift_matches_rfd(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    temperature = 1.2
    needsTorque = False
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
    reference = temperature * thermal_drift_rfd(solver, positions)
    solver.setPositions(positions)
    rfd = average(lambda: average(solver.thermalDrift, 100), 1000)
    assert np.allclose(
        reference,
        rfd,
        atol=1e-5,
        rtol=1e-5,
    ), f"RFD does not match: {np.max(np.abs(reference - rfd))}"
