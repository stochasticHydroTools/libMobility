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
    ("Solver", "periodicity", "tol", "start_height", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), 5e-3, 4, "self_mobility_bw_ref.mat"), # correctness check
        (DPStokes, ("periodic", "periodic", "single_wall"), 1e-6, 0, "self_mobility_bw_w4.mat"), # consistency check
        (DPStokes, ("periodic", "periodic", "two_walls"), 1e-6, 0, "self_mobility_sc_w4.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-6, 1, "self_mobility_bw_ref_noimg.mat")
    ],
)
def test_self_mobility(Solver, periodicity, tol, start_height, ref_file):
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

    start_ind = np.where(refHeights >= start_height)[0][0]
    refHeights = refHeights[start_ind:]
    nHeights = len(refHeights)

    refM = refM[start_ind:]

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

    diff = np.abs([(diag - ref_diag) for diag, ref_diag in zip(diags, ref_diags)])

    avgErr = np.mean(diff)

    assert avgErr < tol, "Self mobility does not match reference"

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), 1e-6, "pair_mobility_bw_w4.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), 1e-6, "pair_mobility_sc_w4.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-4, "pair_mobility_bw_ref_noimg.mat"),
    ],
)
def test_pair_mobility(Solver, periodicity, ref_file, tol):
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


    indx, indy = 4, 1 ## xx
    checkComponent(indx, indy, allM, refM, nSeps, tol)

    indx, indy = 5, 2 # yy
    checkComponent(indx, indy, allM, refM, nSeps, tol)

    indx, indy = 6, 3 # zz
    checkComponent(indx, indy, allM, refM, nSeps, tol)

    indx, indy = 5, 1 # yx
    checkComponent(indx, indy, allM, refM, nSeps, tol)

    indx, indy = 3, 4 # zx
    checkComponent(indx, indy, allM, refM, nSeps, tol)

    indx, indy = 3, 5 # zy
    checkComponent(indx, indy, allM, refM, nSeps, tol)

def checkComponent(indx, indy, allM, refM, nSeps, tol):

    indx -= 1 # shift from matlab to python indexing
    indy -= 1
    for i in range(0, nSeps):

        xx = allM[i, :, indx, indy]
        xx_ref = refM[i, :, indx, indy]

        relDiff = np.abs([np.linalg.norm(xx - xx_ref)/np.linalg.norm(xx_ref + 1e-6) for xx, xx_ref in zip(xx, xx_ref)])
        avgErr = np.mean(relDiff)

        assert avgErr < tol, f"Pair mobility does not match reference for component {indx+1}, {indy+1}"