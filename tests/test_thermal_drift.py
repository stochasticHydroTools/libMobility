import pytest
import numpy as np
from utils import (
    solver_configs_all,
    get_sane_params,
    generate_positions_in_box,
)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("includeAngular", [True, False])
@pytest.mark.parametrize("a", [0.4, 1.0, 2.5])
def test_deterministic_divM_matches_rfd(Solver, periodicity, a, includeAngular):
    if Solver.__name__ == "PSE" and includeAngular:
        pytest.skip("PSE does not support torques")
    N_iter = 250001 if Solver.__name__ == "NBody" else 25001
    tol = 1e-2 if Solver.__name__ == "DPStokes" else 1e-3
    solver = Solver(*periodicity)
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    delta = 1e-3
    if "delta" in parameters:
        parameters["delta"] = delta
    solver.setParameters(**parameters)
    precision = np.float32 if solver.precision == "float" else np.float64
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=a,
        includeAngular=includeAngular,
    )
    numberParticles = 10
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)

    # use delta*a in deterministic since libMobility solvers expect delta to be in units of a
    det_div_m = deterministic_div_m(solver, positions, includeAngular, delta * a)
    rfd_m, rfd_d = average_until_error_tolerance(
        solver.divM, N_iter, det_div_m, numberParticles, includeAngular, tol
    )

    assert np.allclose(det_div_m[0 : 3 * numberParticles], rfd_m, atol=tol, rtol=tol)
    if includeAngular:
        assert np.allclose(det_div_m[3 * numberParticles :], rfd_d, atol=tol, rtol=tol)


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


def average_until_error_tolerance(function, num_averages, soln, N, includeAngular, tol):
    # track error of the running average of RFDs after each new sample. exit early if the error is below the tolerance
    avg_err_m = np.zeros(num_averages)
    rfd_m = np.zeros(3 * N)
    avg_err_d = np.zeros(num_averages)
    rfd_d = np.zeros(3 * N)

    soln_m = soln[0 : 3 * N]
    soln_d = soln[3 * N :] if includeAngular else None

    err_check_iter = 0
    check_every = 5000
    exit_early = False
    last_iter = -1
    for i in range(num_averages):
        new_result_m, new_result_d = function()
        rfd_m += new_result_m
        running_m = rfd_m / float(i + 1)
        avg_err_m[i] = np.linalg.norm(running_m - soln_m)
        if new_result_d is not None:
            rfd_d += new_result_d
            running_d = rfd_d / float(i + 1)
            avg_err_d[i] = np.linalg.norm((running_d - soln_d))

        if err_check_iter == check_every:
            err_m = np.allclose(running_m, soln_m, atol=tol, rtol=tol)
            errs = [err_m]
            if includeAngular:
                err_d = np.allclose(running_d, soln_d, atol=tol, rtol=tol)
                errs.append(err_d)
            if all(errs):
                last_iter = i
                exit_early = True
                break
            err_check_iter = 0
        err_check_iter += 1

    if exit_early:
        num_averages = last_iter + 1

    rfd_m /= float(num_averages)
    rfd_d /= float(num_averages)

    avg_err_m = avg_err_m[:num_averages]
    avg_err_d = avg_err_d[:num_averages]

    # plot_error(avg_err_m, avg_err_d, num_averages, includeAngular, delta, a)

    return rfd_m, rfd_d


def plot_error(avg_err_m, avg_err_d, num_averages, includeAngular):
    import matplotlib.pyplot as plt

    plt.figure()
    n = np.arange(1, num_averages + 1)
    plt.loglog(n, avg_err_m, marker="o", linestyle="-")
    plt.loglog(n, avg_err_d, marker="o", linestyle="-")
    plt.loglog(n, 0.1 * 1 / np.sqrt(n), linestyle="--", color="black")
    if includeAngular:
        plt.legend(
            ["Measured error (linear)", "Measured error (dipole)", "O(1/sqrt(N))"]
        )
    else:
        plt.legend(["Measured error", "O(1/sqrt(N))"])
    plt.xlabel("RFD count")
    plt.ylabel("Error (||running_avg - sol||)")
    plt.title("Error of running average of RFDs compared to deterministic divM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rfd_error.png", dpi=150)
    plt.close()


def deterministic_div_m(solver, positions, includeAngular, delta):
    N = np.size(positions) // 3
    size = 6 * N if includeAngular else 3 * N

    positions = positions.flatten()
    div_M = np.zeros(size, dtype=positions.dtype)
    for j in range(3 * N):
        pos_plus = positions.copy()
        pos_minus = positions.copy()
        pos_plus[j] += 0.5 * delta
        pos_minus[j] -= 0.5 * delta

        e_j = np.zeros(size, dtype=positions.dtype)
        e_j[j] = 1.0
        f = e_j[0 : 3 * N] if includeAngular else e_j
        t = e_j[3 * N :] if includeAngular else None

        solver.setPositions(pos_plus)
        Mf_plus, Mt_plus = solver.Mdot(forces=f, torques=t)

        solver.setPositions(pos_minus)
        Mf_minus, Mt_minus = solver.Mdot(forces=f, torques=t)

        div_M[0 : 3 * N] += Mf_plus - Mf_minus
        if includeAngular:
            div_M[3 * N :] += Mt_plus - Mt_minus

    div_M /= delta
    return div_M
