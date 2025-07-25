import pytest
import numpy as np
from scipy.linalg import pinv, sqrtm
from scipy.stats import kstest, norm
from numpy.linalg import eig
import logging

from libMobility import PSE
from utils import (
    compute_M,
    generate_positions_in_box,
    get_sane_params,
    solver_configs_all,
    solver_configs_torques,
    initialize_solver,
)


def fluctuation_dissipation_KS(M, fluctuation_method):
    R"""
    Test the fluctuation-dissipation theorem using the Kolmogorov-Smirnov test.

    Parameters
    ----------
    M : np.ndarray
        Mobility matrix, :math:`\boldsymbol{\mathcal{M}}`
    fluctuation_method : function
        Function that returns :math:`\sqrt{\boldsymbol{\mathcal{M}}} \cdot \xi`, where :math:`\xi` is a random vector
    """
    if M.shape[0] != M.shape[1] or not np.allclose(M, M.T, rtol=0, atol=5e-5):
        raise ValueError("Matrix M must be square and symmetric.")
    Sigma, Q = eig(M)
    ind = np.argsort(Sigma)
    Sigma = np.sort(Sigma)
    Q = Q[:, ind]
    MInvhalf = sqrtm(pinv(np.diag(Sigma))) @ Q.T
    # pick n so that the mean is
    # within +/- mu_a of the real mean
    # with probability mu_alpha for all N components
    N = Sigma.size
    mu_alpha = 0.999 ** (1 / N)
    mu_a = 0.05
    Ns = int(round(2 * (norm.ppf(mu_alpha) / mu_a) ** 2))
    ScaledNoise = np.full((N, Ns), np.nan)
    sing_check = Sigma < 1e-6

    ScaledNoise = (
        MInvhalf @ np.array([fluctuation_method() for _ in range(Ns)]).T
    ).squeeze()

    for m in range(N):
        if sing_check[m]:
            logging.info(f"Component {m}: Skipped due to zero singular value")
            continue
        noise_scaled = ScaledNoise[m, :]
        _, p = kstest(noise_scaled, "norm")
        assert p > (
            1 - mu_alpha
        ), f"KS test failed for component {m}, p = {p}, 1-mu_alpha = {1-mu_alpha}. This stochastic test may fail occasionally, try running it again."


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("hydrodynamicRadius", [0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 10])
def test_fluctuation_dissipation_linear_displacements(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    includeAngular = False
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
    M = compute_M(solver, numberParticles, includeAngular=includeAngular)

    def fluctuation_method():
        return solver.sqrtMdotW(prefactor=1.0)[0].flatten()

    fluctuation_dissipation_KS(M, fluctuation_method)


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_torques)
@pytest.mark.parametrize("hydrodynamicRadius", [0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 10])
def test_fluctuation_dissipation_angular_displacements(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    includeAngular = True
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
    M = compute_M(solver, numberParticles, includeAngular=includeAngular)

    def fluctuation_method():
        u, omega = solver.sqrtMdotW(prefactor=1.0)
        return np.concatenate((u.flatten(), omega.flatten()))

    fluctuation_dissipation_KS(M, fluctuation_method)


@pytest.mark.parametrize("includeAngular", [True, False])
@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_matrix_pos_def(Solver, periodicity, includeAngular):

    if includeAngular and Solver == PSE:
        pytest.skip("PSE does not support torques")

    numberParticles = 10
    parameters = get_sane_params(Solver.__name__, periodicity[2])
    solver = initialize_solver(
        Solver,
        periodicity,
        includeAngular=includeAngular,
        parameters=parameters,
    )

    n_iter = 20
    for _ in range(n_iter):
        positions = generate_positions_in_box(parameters, numberParticles)
        solver.setPositions(positions)
        M = compute_M(solver, numberParticles, includeAngular=includeAngular)
        assert np.all(
            np.linalg.eigvals(M) > 0
        ), "Mobility matrix is not positive definite."
