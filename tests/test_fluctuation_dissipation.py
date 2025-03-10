import pytest
import numpy as np
from scipy.linalg import pinv, sqrtm
from scipy.stats import kstest, norm
from numpy.linalg import eig
import logging

from libMobility import SelfMobility, PSE, NBody, DPStokes
from utils import (
    compute_M,
    sane_parameters,
    generate_positions_in_box,
    solver_configs_all,
    initialize_solver,
)


def fluctuation_dissipation_KS(M, fluctuation_method, needsTorques):
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
    if needsTorques:
        # TODO this won't work yet for torques
        # will also probably need to modify fluctuation_method's return
        ScaledNoise = (
            MInvhalf @ np.array([fluctuation_method() for _ in range(Ns)]).T
        ).squeeze()
    else:
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
    needsTorques = False
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    numberParticles = 10
    solver.initialize(
        temperature=0.5,  # needs to be 1/2 to cancel out the sqrt(2*T) when computing Mdot
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles, needsTorque=needsTorques)

    def fluctuation_method():
        return solver.sqrtMdotW(prefactor=1.0)[0].flatten()

    fluctuation_dissipation_KS(M, fluctuation_method, needsTorques)


@pytest.mark.parametrize("needsTorques", [True, False])
@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_matrix_pos_def(Solver, periodicity, needsTorques):

    if needsTorques and Solver == PSE:
        pytest.skip("PSE does not support torques")

    numberParticles = 10
    parameters = sane_parameters[Solver.__name__]
    solver = initialize_solver(
        Solver,
        periodicity,
        numberParticles,
        needsTorque=needsTorques,
        parameters=parameters,
    )

    n_iter = 20
    for _ in range(n_iter):
        positions = generate_positions_in_box(parameters, numberParticles)
        solver.setPositions(positions)
        M = compute_M(solver, numberParticles, needsTorque=needsTorques)
        assert np.all(
            np.linalg.eigvals(M) > 0
        ), "Mobility matrix is not positive definite."
