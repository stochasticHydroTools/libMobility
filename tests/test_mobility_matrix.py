from utils import sane_parameters, compute_M, generate_positions_in_box
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
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles)
    assert M.shape == (3 * numberParticles, 3 * numberParticles)
    assert M.dtype == precision
    sym = M - M.T
    assert np.allclose(
        sym, 0.0, rtol=0, atol=5e-5
    ), f"Mobility matrix is not symmetric within 1e-5, max diff: {np.max(np.abs(sym))}"


def test_self_mobility_selfmobility():
    # Mobility should be just 1/(6\pi\eta R) for a single particle.
    precision = np.float32 if SelfMobility.precision == "float" else np.float64
    solver = SelfMobility("open", "open", "open")
    parameters = sane_parameters[SelfMobility.__name__]
    solver.setParameters(**parameters)
    hydrodynamicRadius = 0.9123
    viscosity = 1.123
    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=1,
    )
    positions = np.zeros((1, 3), dtype=precision)
    solver.setPositions(positions)
    forces = np.ones(3, dtype=precision)
    result = solver.Mdot(forces)
    m0 = 1.0 / (6 * np.pi * viscosity * hydrodynamicRadius)
    assert np.allclose(result, m0 * forces, rtol=0, atol=1e-7)


def test_self_mobility_nbody():
    # Mobility should be just 1/(6\pi\eta R)
    Solver = NBody
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver("open", "open", "open")
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    hydrodynamicRadius = 0.9123
    viscosity = 1.123
    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=1,
    )
    positions = np.zeros((1, 3), dtype=precision)
    solver.setPositions(positions)
    forces = np.ones(3, dtype=precision)
    result = solver.Mdot(forces)
    m0 = 1.0 / (6 * np.pi * viscosity * hydrodynamicRadius)
    assert np.allclose(result, m0 * forces, rtol=0, atol=1e-7)


@pytest.mark.parametrize("psi", [0.0, 0.5, 1.0])
def test_self_mobility_pse_cubic_box(psi):
    # Mobility should be just 1/(6\pi\eta a)*(1 - 2.83729748 a/L) for a single particle.
    Solver = PSE
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver("periodic", "periodic", "periodic")
    parameters = {}
    parameters["psi"] = psi
    # Set Lx=Ly=Lz
    parameters["Lx"] = 64.0
    parameters["Ly"] = parameters["Lx"]
    parameters["Lz"] = parameters["Lx"]
    parameters["shearStrain"] = 0.0
    solver.setParameters(**parameters)
    hydrodynamicRadius = 2
    viscosity = 1.123
    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=1,
        tolerance=1e-6,
    )
    positions = np.zeros((1, 3), dtype=precision)
    solver.setPositions(positions)
    forces = np.ones(3, dtype=precision)
    result = solver.Mdot(forces)
    leff = hydrodynamicRadius / parameters["Lx"]
    m0 = (
        1.0
        / (6 * np.pi * viscosity * hydrodynamicRadius)
        * (1 - 2.83729748 * leff + 4.0 * np.pi / 3.0 * leff**3)
    )
    assert np.allclose(result, m0 * forces, rtol=0, atol=1e-6)
