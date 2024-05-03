import pytest
from libMobility import *
import numpy as np


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [(SelfMobility, ("open", "open", "open"))],
)
def test_contiguous(Solver, periodicity):
    hydrodynamicRadius = 1.0
    solver = Solver(*periodicity)
    solver.setParameters(parameter=1)
    numberParticles = 1
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )

    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions = np.random.rand(numberParticles, 3).astype(precision)

    solver.setPositions(positions)

    # Set positions so that they are not contiguous
    matrix = np.random.rand(numberParticles * 3, numberParticles * 3).astype(precision)
    positions = matrix[:, 0]
    with pytest.raises(RuntimeError):
        solver.setPositions(positions)
