import pytest
import numpy as np

from libMobility import DPStokes, NBody
from utils import compute_M

# NOTE: Some of the following tests will only pass if compiled with double precision.
# This is because the reference data was generated in double precision.

wall_params = {
    "DPStokes": {"dt": 1, "Lx": 76.8, "Ly": 76.8, "zmin": 0, "zmax": 19.2},
    "NBody": {"algorithm": "advise"},
}

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "start_height", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), 5e-3, 4, "self_mobility_bw_ref.npz"),
        (DPStokes, ("periodic", "periodic", "single_wall"), 1e-6, 0, "self_mobility_bw_w4.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), 1e-6, 0, "self_mobility_sc_w4.npz"),
        (NBody, ("open", "open", "single_wall"), 1e-6, 1, "self_mobility_bw_ref_noimg.npz")
    ],
)
def test_self_mobility_linear(Solver, periodicity, tol, start_height, ref_file):
    xymax = 76.8
    params = wall_params[Solver.__name__]
    needsTorque = False

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()

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

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [np.diag(matrix)[0:3] for matrix in refM] # only take diagonal elements from forces

    for diag, ref_diag in zip(diags, ref_diags):
        assert np.allclose(diag, ref_diag, rtol=tol, atol=tol), f"Self mobility does not match reference"

@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), 1e-6, "pair_mobility_bw_w4.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), 1e-6, "pair_mobility_sc_w4.npz"),
        (NBody, ("open", "open", "single_wall"), 1e-4, "pair_mobility_bw_ref_noimg.npz"),
    ],
)
def test_pair_mobility_linear(Solver, periodicity, ref_file, tol):
    xymax = 76.8
    params = wall_params[Solver.__name__]
    needsTorque = False

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights']
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

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i,k] - refM[i,k][0:6,0:6])
            assert np.all(diff < tol)

@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_torque.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_torque.npz"),
        (NBody, ("open", "open", "single_wall"), "self_mobility_bw_ref_noimg.npz")
    ],
)
def test_self_mobility_angular(Solver, periodicity, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = wall_params[Solver.__name__]

    hydrodynamicRadius = 1.0
    eta = 1/4/np.sqrt(np.pi)

    needsTorque = True
    tol = 1e-6

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()

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

    nHeights = len(refHeights)

    normMat = np.zeros((6*numberParticles, 6*numberParticles), dtype=precision)
    normMat[0:3,0:3] = 1/(6*np.pi*eta*hydrodynamicRadius) # tt
    normMat[3:, 3: ] = 1/(8*np.pi*eta*hydrodynamicRadius**3) # rr
    normMat[3: ,0:3] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # tr
    normMat[0:3,3: ] = 1/(6*np.pi*eta*hydrodynamicRadius**2) # rt

    allM = np.zeros((nHeights, 6*numberParticles, 6*numberParticles), dtype=precision)
    for i in range(0,nHeights):
        # breakpoint()
        positions = np.array([[xymax/2, xymax/2, refHeights[i]]], dtype=precision)
        solver.setPositions(positions)
        
        M = compute_M(solver, numberParticles, needsTorque)
        M /= normMat
        allM[i] = M

    for i in range(0, nHeights):
        diff = abs(allM[i] - refM[i])
        assert np.all(diff < tol)

@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "pair_mobility_bw_torque.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "pair_mobility_sc_torque.npz"),
        (NBody, ("open", "open", "single_wall"), "pair_mobility_bw_ref_noimg.npz")
    ],
)
def test_pair_mobility_angular(Solver, periodicity, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = wall_params[Solver.__name__]
    hydrodynamicRadius = 1.0
    eta = 1/4/np.sqrt(np.pi)
    needsTorque = True

    tol = 1e-6

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'].flatten()

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

    nHeights = len(refHeights)

    seps = np.array([3 * hydrodynamicRadius, 4 * hydrodynamicRadius, 8 * hydrodynamicRadius])
    nSeps = len(seps)

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

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i,k] - refM[i,k])
            temp = diff < tol
            x = temp[0:6,6:12]
            assert np.all(diff < tol)