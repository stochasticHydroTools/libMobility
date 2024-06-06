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
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_w4_gpu.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_w4_gpu.mat"),
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

    # uncomment to save datafile for test plots
    # scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights})

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [np.diag(matrix)[0:3] for matrix in refM] # only take diagonal elements from forces

    if Solver.__name__ == "DPStokes": # NBody ref only goes down to a height of z=1
        assert np.all(np.diag(allM[0]) == [0,0,0]), "Self mobility is not zero on the wall at z=0"

    assert np.allclose(diags, ref_diags, atol=1e-6), "Self mobility does not match reference"

@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "pair_mobility_bw_w4_gpu.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "pair_mobility_sc_w4_gpu.mat"),
        (NBody, ("open", "open", "single_wall"), "pair_mobility_bw_ref_noimg.mat")
    ],
)
def test_pair_mobility(Solver, periodicity, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    ref_dir = "./ref/"
    ref = scipy.io.loadmat(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()
    nHeights = len(refHeights)

    radH = 1.0 # hydrodynamic radius
    eta = 1/4/np.sqrt(np.pi)

    precision = np.float32 if Solver.precision == "float" else np.float64

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 2
    solver.initialize(
        temperature=0,
        viscosity=eta,
        hydrodynamicRadius=radH,
        numberParticles=numberParticles,
    )
    
    normMat = (1/(6*np.pi*eta))*np.ones((3*numberParticles, 3*numberParticles), dtype=precision)

    seps = np.array([3 * radH, 4 * radH, 8 * radH])
    nSeps = len(seps)

    allM = np.zeros((nSeps, nHeights, 3*numberParticles, 3*numberParticles), dtype=precision)
    for i in range(0,nSeps):
        for k in range(0, nHeights):
            xpos = xymax/2
            positions = np.array([[xpos+seps[i]/2, xpos, refHeights[k]],
                                  [xpos-seps[i]/2, xpos, refHeights[k]]], dtype=precision)
            solver.setPositions(positions)
            
            M = compute_M(solver, numberParticles)
            M /= normMat
            allM[i][k] = M

    # uncomment to save datafile for test plots
    # scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights})

    ## xx component
    indx = 4
    indy = 1
    checkComponent(indx, indy, allM, refM, nSeps, nHeights)

    ## yy component
    indx = 5
    indy = 2
    checkComponent(indx, indy, allM, refM, nSeps, nHeights)

    ## zz component
    indx = 6
    indy = 3
    checkComponent(indx, indy, allM, refM, nSeps, nHeights)

    ## xz component
    indx = 3
    indy = 4
    checkComponent(indx, indy, allM, refM, nSeps, nHeights)


def checkComponent(indx, indy, allM, refM, nSeps, nHeights):

    indx -= 1 # shift from matlab to python indexing
    indy -= 1
    for i in range(0,nSeps):
        for k in range(0, nHeights):

            xx = allM[i, k, indx, indy]
            xx_ref = refM[i, k, indx, indy]

            if xx_ref == 0.0:
                diff = np.abs(xx - xx_ref)
            else:
                diff = np.abs(xx - xx_ref)/xx_ref

            assert diff < 1e-3, f"Pair mobility does not match reference for component {indx}, {indy}, {xx}, {xx_ref}"