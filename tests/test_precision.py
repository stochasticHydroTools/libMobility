import pytest
import numpy as np

from libMobility import *
from utils import solver_configs_all, get_sane_params


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_precision(Solver, periodicity):
    # tests that Mdot gives a non-zero result when using the precision that libMobility was compiled in

    hydrodynamicRadius = 1.0

    solver = Solver(*periodicity)
    solver.setParameters(**get_sane_params(Solver.__name__, periodicity[2]))
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)

    size = 3 * numberParticles

    solver.setPositions(positions)
    forces = np.ones(size, dtype=precision)
    mf, _ = solver.Mdot(forces)

    zeros = np.zeros(size, dtype=precision)
    assert (
        mf != zeros
    ).all(), "Mdot should not come out to be zero when using correct precision."


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_incorrect_precision(Solver, periodicity):
    # libMobility should work even if the inputs have an unexpected precision

    hydrodynamicRadius = 1.0

    solver = Solver(*periodicity)
    solver.setParameters(**get_sane_params(Solver.__name__, periodicity[2]))
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
    )

    # Set precision to be opposite from compiled precision
    precision_bad = np.float32 if Solver.precision != "float" else np.float64
    precision_good = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision_bad)

    size = 3 * numberParticles
    solver.setPositions(positions)
    positions = positions.astype(precision_good)
    solver.setPositions(positions)
    forces = np.ones(size, dtype=precision_bad)
    mf, _ = solver.Mdot(forces)
