import pytest
import numpy as np
from libMobility import SelfMobility, PSE, NBody, DPStokes
from utils import (
    solver_configs_all,
    solver_configs_torques,
    get_sane_params,
    generate_positions_in_box,
)


def accurate_average(current_mean, current_count, new_value):
    # Update the mean and count
    new_mean = current_mean + (new_value - current_mean) / (current_count + 1)
    return new_mean


def thermal_drift_rfd(solver, positions):
    # RFD works by approxmating kT\partial_q \dot M = 1/\delta \langle M(q+\delta/2 W)W - M(q-\delta/2 W)W \rangle
    # Where delta is a small number, and W is a normal random vector of unit length
    tdrift = np.zeros_like(positions)
    delta = 1e-4
    for i in range(30000):
        W = np.random.normal(size=positions.shape)
        qpd = positions + delta / 2 * W
        solver.setPositions(qpd)
        dMf, _ = solver.Mdot(W)
        _tdrift = dMf
        qmd = positions - delta / 2 * W
        solver.setPositions(qmd)
        dMf, _ = solver.Mdot(W)
        _tdrift -= dMf
        _tdrift /= delta
        tdrift = accurate_average(tdrift, i, _tdrift)
    return tdrift


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
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    reference = thermal_drift_rfd(solver, positions)
    assert np.allclose(
        np.abs(reference),
        0.0,
        atol=1e-5,
        rtol=0,
    ), f"RFD drift is not zero: {np.max(np.abs(reference))}"


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
def test_thermal_drift_matches_rfd(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    needsTorque = False
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    reference = thermal_drift_rfd(solver, positions)
    solver.setPositions(positions)
    rfd = solver.thermalDrift()
    assert np.allclose(
        reference,
        rfd,
        atol=1e-5,
        rtol=1e-5,
    ), f"RFD does not match: {np.max(np.abs(reference - rfd))}"
