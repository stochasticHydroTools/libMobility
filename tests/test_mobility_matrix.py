from utils import sane_parameters, compute_M
import pytest
import numpy as np
from libMobility import SelfMobility, PSE, NBody, DPStokes


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (SelfMobility, ("open", "open", "open")),
        (PSE, ("periodic", "periodic", "periodic")),
        (NBody, ("open", "open", "open")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
@pytest.mark.parametrize("hydrodynamicRadius", [1.0, 0.95, 1.12])
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
def test_mobility_matrix(Solver, periodicity, hydrodynamicRadius, numberParticles):
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )
    positions = np.random.rand(numberParticles, 3).astype(precision) - 0.5
    if "Lx" in parameters:
        positions[:, 0] *= parameters["Lx"]
    if "Ly" in parameters:
        positions[:, 1] *= parameters["Ly"]
    if "Lz" in parameters:
        positions[:, 2] *= parameters["Lz"]
    if "zmin" in parameters:
        positions[:, 2] *= (parameters["zmax"] - parameters["zmin"]) * 0.5
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles)
    assert M.shape == (3 * numberParticles, 3 * numberParticles)
    assert M.dtype == precision
    sym = M - M.T
    assert np.allclose(
        sym, 0.0, rtol=0, atol=1e-6
    ), f"Mobility matrix is not symmetric within 1e-6, max diff: {np.max(np.abs(sym))}"
