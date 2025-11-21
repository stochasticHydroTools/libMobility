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


def average_with_error(function, num_averages, soln, N):
    # Track cumulative (running) error: error of the running average after each new sample
    avg_err_m = np.zeros(num_averages)
    rfd_m = np.zeros(3 * N)
    avg_err_d = np.zeros(num_averages)
    rfd_d = np.zeros(3 * N)

    for i in range(num_averages):
        new_result_m, new_result_d = function()
        rfd_m += new_result_m
        running_m = rfd_m / float(i + 1)
        avg_err_m[i] = np.linalg.norm(running_m - soln)
        if new_result_d is not None:
            rfd_d += new_result_d
            running_d = rfd_d / float(i + 1)
            avg_err_d[i] = np.linalg.norm(running_d - soln) / np.linalg.norm(soln)

    rfd_m /= float(num_averages)
    rfd_d /= float(num_averages)

    import matplotlib.pyplot as plt

    plt.figure()
    n = np.arange(num_averages)
    plt.loglog(n, avg_err_m, marker="o", linestyle="-")
    plt.loglog(n, 0.1 * 1 / np.sqrt(n), linestyle="--")
    plt.legend(["Measured error", "O(1/sqrt(N))"])
    # plt.plot(
    #     np.arange(num_averages),
    #     avg_err_d,
    #     linestyle="--",
    #     color="red",
    # )
    plt.xlabel("Sample index")
    plt.ylabel("Error (||running_avg - sol||) or relative for dipole")
    plt.title("Cumulative running error of the averaged estimate (linear)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_err_m.png", dpi=150)
    plt.close()
    return rfd_m, rfd_d


def deterministic_div_m(solver, positions, delta):
    N = np.size(positions) // 3
    print("Number of particles:", N)

    positions = positions.flatten()
    div_M = np.zeros(3 * N, dtype=positions.dtype)
    for i in range(3 * N):
        for j in range(3 * N):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[j] += 0.5 * delta
            pos_minus[j] -= 0.5 * delta
            solver.setPositions(pos_plus)
            e_j = np.zeros(3 * N, dtype=positions.dtype)
            e_j[j] = 1.0
            M_plus, _ = solver.Mdot(e_j)
            solver.setPositions(pos_minus)
            M_minus, _ = solver.Mdot(e_j)
            div_M[i] += M_plus[i] - M_minus[i]

    div_M /= delta
    return div_M


def divM_rfd(solver, positions, delta):
    # RFD works by approximating \partial_q \dot M = 1/\delta \langle M(q+\delta/2 W)W - M(q-\delta/2 W)W \rangle

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
def test_deterministic_divM_matches_rfd(Solver, periodicity):
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver.setParameters(**parameters)
    precision = np.float32 if solver.precision == "float" else np.float64
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        includeAngular=False,
    )
    numberParticles = 10
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)

    delta = 1e-3
    det_div_m = deterministic_div_m(solver, positions, delta)
    # rfd_dM, _ = divM_rfd(solver, positions, delta)
    # rfd_m, rfd_d = average(solver.divM, 10000)
    m, d = average_with_error(solver.divM, 10000, det_div_m, numberParticles)

    breakpoint()
    print("-----------------------")
    print("Deterministic divM:", det_div_m)
    print("RFD divM:", rfd_m)

    assert np.allclose(
        det_div_m,
        rfd_m,
        atol=1e-3,
        rtol=1e-3,
    ), f"Deterministic divM does not match RFD divM: {np.max(np.abs(det_div_m - rfd_m))}"


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
