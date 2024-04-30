import pytest
import numpy as np
from scipy.linalg import pinv, sqrtm
from scipy.stats import kstest, norm
from numpy.linalg import eig
import logging

from libMobility import *

sane_parameters = {
    "PSE": {"psi": 1.0, "Lx": 32, "Ly": 32, "Lz": 32, "shearStrain": 0.0},
    "NBody": {"algorithm": "advise"},
    "SelfMobility": {"parameter": 5.0},
}


def compute_M(solver, numberParticles):

    precision = np.float32 if solver.precision == "float" else np.float64

    # forces = np.zeros((numberParticles, 3), dtype=precision)
    # mf = np.zeros((numberParticles, 3), dtype=precision)

    size = 3*numberParticles
    M = np.zeros((size,size), dtype=precision)

    I = np.identity(size, dtype=precision)
    for i in range(0,size):
        forces = I[:,i].copy()
        mf = np.zeros(size, dtype=precision)
        solver.Mdot(forces, mf)

        M[:,i] = mf

    return M


def fluctuation_dissipation_KS(M, fluctuation_method):
    if M.shape[0] != M.shape[1] or not np.allclose(M, M.T):
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
    mu_alpha = 0.99 ** (1 / N)
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
        ), f"KS test failed for component {m}, p = {p}, 1-mu_alpha = {1-mu_alpha}"


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (SelfMobility, ("open", "open", "open")),
        # (PSE, ("periodic", "periodic", "periodic")),
        # (NBody, ("open", "open", "open")),
        # (DPStokes, ("periodic", "periodic", "open")),
        # (DPStokes, ("periodic", "periodic", "single_wall")),
        # (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
def test_fluctuation_dissipation(Solver, periodicity, hydrodynamicRadius):

    precision = np.float32 if Solver.precision == "float" else np.float64

    solver = Solver(*periodicity)
    solver.setParameters(**sane_parameters[Solver.__name__])
    numberParticles = 1
    solver.initialize(
        temperature=0.5, # needs to be 1/2 to cancel out the sqrt(2*T) when computing Mdot
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )
    positions = np.random.rand(numberParticles, 3).astype(precision)
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles)

    def fluctuation_method():
        sqrtmnoise = np.zeros(numberParticles*3).astype(precision)
        solver.sqrtMdotW(sqrtmnoise, prefactor=1.0)
        return sqrtmnoise

    fluctuation_dissipation_KS(M, fluctuation_method)
