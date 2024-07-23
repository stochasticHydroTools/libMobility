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
def test_mobility_matrix_linear(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    needsTorque = False
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles, needsTorque)
    assert M.shape == (3 * numberParticles, 3 * numberParticles)
    assert M.dtype == precision
    sym = M - M.T
    assert np.allclose(
        sym, 0.0, rtol=0, atol=5e-5
    ), f"Mobility matrix is not symmetric within 5e-5, max diff: {np.max(np.abs(sym))}"


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
@pytest.mark.parametrize("numberParticles", [1, 2, 3, 10])
def test_mobility_matrix_angular(
    Solver, periodicity, hydrodynamicRadius, numberParticles
):
    needsTorque = True
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver(*periodicity)
    parameters = sane_parameters[Solver.__name__]
    solver.setParameters(**parameters)
    solver.initialize(
        temperature=0.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
        needsTorque=needsTorque,
    )
    positions = generate_positions_in_box(parameters, numberParticles).astype(precision)
    solver.setPositions(positions)
    M = compute_M(solver, numberParticles, needsTorque)
    size = 6 * numberParticles
    assert M.shape == (size, size)
    assert M.dtype == precision
    sym = M - M.T
    assert np.allclose(
        sym, 0.0, rtol=0, atol=5e-5
    ), f"Mobility matrix is not symmetric within 5e-5, max diff: {np.max(np.abs(sym))}"


def test_self_mobility_linear_selfmobility():
    # linear mobility should be just 1/(6\pi\eta R) for a single particle.
    # angular mobility should be just 1/(8\pi\eta R^3) for a single particle.
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
    torques = np.ones(3, dtype=precision)
    linear, angular = solver.Mdot(forces, torques)
    m0 = 1.0 / (6 * np.pi * viscosity * hydrodynamicRadius)
    t0 = 1.0 / (8 * np.pi * viscosity * hydrodynamicRadius**3)
    assert np.allclose(linear, m0 * forces, rtol=0, atol=1e-7)
    assert np.allclose(angular, t0 * torques, rtol=0, atol=1e-7)

# @pytest.mark.parametrize("algorithm", ["naive", "block", "fast", "advise"])
@pytest.mark.parametrize("algorithm", ["naive", "fast"])
def test_self_mobility_linear_nbody(algorithm):
    # Mobility should be just 1/(6\pi\eta R)
    Solver = NBody
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver("open", "open", "open")
    parameters = {"algorithm": algorithm}
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
    result, _ = solver.Mdot(forces)
    m0 = 1.0 / (6 * np.pi * viscosity * hydrodynamicRadius)
    assert np.allclose(result, m0 * forces, rtol=0, atol=1e-7)

# @pytest.mark.parametrize("algorithm", ["naive", "block", "fast", "advise"])
@pytest.mark.parametrize("algorithm", ["naive", "fast"])
def test_self_mobility_angular_nbody(algorithm):
    # Mobility should be just 1/(6\pi\eta R)
    Solver = NBody
    precision = np.float32 if Solver.precision == "float" else np.float64
    solver = Solver("open", "open", "open")
    parameters = {"algorithm": algorithm}
    solver.setParameters(**parameters)
    hydrodynamicRadius = 0.9123
    viscosity = 1.123
    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=1,
        needsTorque=True
    )
    positions = np.zeros((1, 3), dtype=precision)
    solver.setPositions(positions)
    forces = np.ones(3, dtype=precision)
    torques = np.ones(3, dtype=precision)
    linear, angular = solver.Mdot(forces, torques)
    m0 = 1.0 / (6 * np.pi * viscosity * hydrodynamicRadius)
    t0 = 1.0 / (8 * np.pi * viscosity * hydrodynamicRadius**3)
    assert np.allclose(linear, m0 * forces, rtol=0, atol=1e-7)
    assert np.allclose(angular, t0 * torques, rtol=0, atol=1e-7)

@pytest.mark.parametrize("psi", [0.0, 0.5, 1.0])
def test_self_mobility_linear_pse_cubic_box(psi):
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
    result, _ = solver.Mdot(forces)
    leff = hydrodynamicRadius / parameters["Lx"]
    m0 = (
        1.0
        / (6 * np.pi * viscosity * hydrodynamicRadius)
        * (1 - 2.83729748 * leff + 4.0 * np.pi / 3.0 * leff**3)
    )
    assert np.allclose(result, m0 * forces, rtol=0, atol=1e-6)
