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
        (DPStokes, ("periodic", "periodic", "single_wall"), 5e-3, 4, "self_mobility_bw_ref.mat"),
        (DPStokes, ("periodic", "periodic", "single_wall"), 1e-5, 0, "self_mobility_bw_w4.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), 1e-5, 0, "self_mobility_sc_w4.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-6, 1, "self_mobility_bw_ref_noimg.mat")
    ],
)
def test_self_mobility_linear(Solver, periodicity, tol, start_height, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    needsTorque = False

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
        
        M = compute_M(solver, numberParticles, needsTorque)
        M /= normMat
        allM[i] = M

    # uncomment to save datafile for test plots
    # scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights})

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [np.diag(matrix)[0:3] for matrix in refM] # only take diagonal elements from forces

    diff = np.abs([(diag - ref_diag) for diag, ref_diag in zip(diags, ref_diags)])

    avgErr = np.mean(diff)

    assert avgErr < tol, f"Self mobility does not match reference. Average error: {avgErr}"

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), 5e-5, "pair_mobility_bw_w4.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), 5e-5, "pair_mobility_sc_w4.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-4, "pair_mobility_bw_ref_noimg.mat"),
    ],
)
def test_pair_mobility_linear(Solver, periodicity, ref_file, tol):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    needsTorque = False

    ref_dir = "./ref/"
    ref = scipy.io.loadmat(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()
    nHeights = len(refHeights)

    radH = 1.0 # hydrodynamic radius
    eta = 1/4/np.sqrt(np.pi)

    precision = np.float32 if Solver.precision == "float" else np.float64
    tol = 100*np.finfo(precision).eps

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
            
            M = compute_M(solver, numberParticles, needsTorque)
            M /= normMat
            allM[i][k] = M

    # uncomment to save datafile for test plots
    # scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights})

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i,k] - refM[i,k][0:6,0:6])
            assert np.all(diff < tol)

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "start_height", "ref_file"),
    [
        # (DPStokes, ("periodic", "periodic", "single_wall"), 1e-5, 0, "self_mobility_bw_torque.mat"),
        # (DPStokes, ("periodic", "periodic", "two_walls"), 1e-5, 0, "self_mobility_sc_torque.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-6, 0, "self_mobility_bw_ref_noimg.mat")
    ],
)
def test_self_mobility_angular(Solver, periodicity, tol, start_height, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    needsTorque = True

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
        needsTorque=needsTorque
    )

    start_ind = np.where(refHeights >= start_height)[0][0]
    refHeights = refHeights[start_ind:]
    nHeights = len(refHeights)

    refM = refM[start_ind:]

    normMat = np.zeros((6*numberParticles, 6*numberParticles), dtype=precision)
    normMat[0:3,0:3] = 1/(6*np.pi*eta*hydrodynamicRadius) # tt
    normMat[3:, 3: ] = 1/(8*np.pi*eta*hydrodynamicRadius**3) # rr
    normMat[3: ,0:3] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # tr
    normMat[0:3,3: ] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # rt

    allM = np.zeros((nHeights, 6*numberParticles, 6*numberParticles), dtype=precision)
    for i in range(0,nHeights):
        positions = np.array([[xymax/2, xymax/2, refHeights[i]]], dtype=precision)
        solver.setPositions(positions)
        
        M = compute_M(solver, numberParticles, needsTorque)
        M /= normMat
        allM[i] = M

    # uncomment to save datafile for test plots
    # scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights})

    for i in range(0, nHeights):
        diff = abs(allM[i] - refM[i])
        assert np.all(diff < tol)

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "start_height", "ref_file"),
    [
        # (DPStokes, ("periodic", "periodic", "single_wall"), 1e-6, 0, "pair_mobility_bw_torque.mat"),
        # (DPStokes, ("periodic", "periodic", "two_walls"), 1e-6, 0, "pair_mobility_sc_torque.mat"),
        (NBody, ("open", "open", "single_wall"), 1e-6, 0, "pair_mobility_bw_ref_noimg.mat")
    ],
)
def test_pair_mobility_angular(Solver, periodicity, tol, start_height, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = self_mobility_params[Solver.__name__]

    needsTorque = True

    ref_dir = "./ref/"
    ref = scipy.io.loadmat(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()

    hydrodynamicRadius = 1.0
    eta = 1/4/np.sqrt(np.pi)

    precision = np.float32 if Solver.precision == "float" else np.float64

    nP = 2
    solver = Solver(*periodicity)
    solver.setParameters(**params)
    solver.initialize(
        temperature=0,
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
        numberParticles=nP,
        needsTorque=needsTorque
    )

    start_ind = np.where(refHeights >= start_height)[0][0]
    refHeights = refHeights[start_ind:]
    nHeights = len(refHeights)

    seps = np.array([3 * hydrodynamicRadius, 4 * hydrodynamicRadius, 8 * hydrodynamicRadius])
    nSeps = len(seps)

    refM = refM[start_ind:]

    normMat = np.zeros((6*nP, 6*nP), dtype=precision)
    normMat[0:3*nP,0:3*nP] = 1/(6*np.pi*eta*hydrodynamicRadius) # tt
    normMat[3*nP:, 3*nP: ] = 1/(8*np.pi*eta*hydrodynamicRadius**3) # rr
    normMat[3*nP: ,0:3*nP] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # tr
    normMat[0:3*nP,3*nP: ] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # rt

    xpos = xymax/2
    allM = np.zeros((nSeps, nHeights, 6*nP, 6*nP), dtype=precision)
    for i in range(0, nSeps):
        for k in range(0,nHeights):
            positions = np.array([[xpos+seps[i]/2, xpos, refHeights[k]],
                                  [xpos-seps[i]/2, xpos, refHeights[k]]], dtype=precision)
            solver.setPositions(positions)
            
            M = compute_M(solver, nP, needsTorque)
            M /= normMat
            allM[i,k] = M

    # uncomment to save datafile for test plots
    scipy.io.savemat('./temp/test_data/test_' + ref_file, {'M': allM, 'heights': refHeights, 'seps': seps})

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i,k] - refM[i,k])
            temp = diff < tol
            x = temp[0:6,6:12]
            # breakpoint()
            assert np.all(diff < tol)

def checkPairComponent(indx, indy, allM, refM, nSeps, tol):

    indx -= 1 # shift from matlab to python indexing
    indy -= 1
    for i in range(0, nSeps):

        x = allM[i, :, indx, indy]
        x_ref = refM[i, :, indx, indy]

        for j in range(0, len(x)):

            diff = x[j]-x_ref[j]

            if x[j] > tol and x_ref[j] > tol and x_ref[j] != 0:
                err = diff/x_ref[j]
            else:
                err = diff

        assert err < tol, f"Pair mobility does not match reference for component {indx+1}, {indy+1}"