import pytest
import numpy as np
import scipy.interpolate
import scipy.io

from libMobility import DPStokes, NBody
from utils import compute_M

self_mobility_params = {
    "DPStokes": {"dt": 1, "Lx": 76.8, "Ly": 76.8, "zmin": 0, "zmax": 19.2},
    "NBody": {"algorithm": "advise"},
}

@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc.mat"),
        (NBody, ("open", "open", "single_wall"), "self_mobility_bw_ref_noimg.mat")
    ],
)
def test_self_mobility(Solver, periodicity, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    ref_dir = "./ref/"
    ref = scipy.io.loadmat(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'][0]

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

    nHeights = len(refHeights)

    normMat = np.ones((3*numberParticles, 3*numberParticles), dtype=precision)
    diag_ind = np.diag_indices_from(normMat)
    normMat[diag_ind] = 1/(6*np.pi*eta*hydrodynamicRadius) # only for diag elements as this is for self mobility

    allM = np.zeros((nHeights, 3*numberParticles, 3*numberParticles), dtype=precision)
    for i in range(0,nHeights):
        positions = np.array([[xymax/2, xymax/2, refHeights[i]]], dtype=precision)
        solver.setPositions(positions)
        
        M = compute_M(solver, numberParticles)
        M /= normMat
        allM[i] = M

    # scipy.io.savemat('./temp/test_' + ref_file, {'M': allM, 'heights': heights})

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [np.diag(matrix)[0:3] for matrix in refM] # only take diagonal elements from forces

    if Solver.__name__ == "DPStokes": # NBody ref only goes down to a height of z=1
        assert np.all(np.diag(allM[0]) == [0,0,0]), "Self mobility is not zero on the wall at z=0"

    assert np.allclose(diags, ref_diags, atol=1e-2), "Self mobility does not match reference"