import pytest
import numpy as np
import scipy.io

from libMobility import DPStokes
from utils import compute_M

@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall")),
    ],
)
def test_self_mobility(Solver, periodicity):
    zmax = 19.2
    xymax = 76.8
    params = {"dt": 1, "Lx": xymax, "Ly": xymax, "zmin": 0, "zmax": zmax}

    hydrodynamicRadius = 1.0
    eta = 1/4/np.sqrt(np.pi)

    precision = np.float32 if Solver.precision == "float" else np.float64

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 1
    solver.initialize(
        temperature=0,
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=numberParticles,
    )

    nHeights = 60
    heights = np.linspace(0, zmax/4, 50)
    heights = np.concatenate((heights, np.linspace(zmax/4 + heights[2]-heights[1], zmax/2, 10)))

    allM = np.zeros((nHeights, 3*numberParticles, 3*numberParticles), dtype=precision)
    for i in range(0,nHeights):
        positions = np.array([[xymax/2, xymax/2, heights[i]]], dtype=precision)
        solver.setPositions(positions)

        M = compute_M(solver, numberParticles)
        allM[i] = M
        # breakpoint()

    scipy.io.savemat('self_mobility_bw.mat', {'M': allM, 'heights': heights/hydrodynamicRadius})

if __name__ == "__main__":
    test_self_mobility(DPStokes, ("periodic", "periodic", "single_wall"))