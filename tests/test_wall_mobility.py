import pytest
import numpy as np
import scipy.interpolate
import scipy.io

from libMobility import DPStokes
from utils import compute_M

@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_ref.mat"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_ref.mat"),
    ],
)
def test_self_mobility(Solver, periodicity, ref_file):
    zmax = 19.2
    xymax = 76.8
    params = {"dt": 1, "Lx": xymax, "Ly": xymax, "zmin": 0, "zmax": zmax}
    ref_dir = "./ref/"

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

    normMat = np.ones((3*numberParticles, 3*numberParticles), dtype=precision)
    diag_ind = np.diag_indices_from(normMat)
    normMat[diag_ind] *= 1/(6*np.pi*eta*hydrodynamicRadius) # only for diag elements as this is for self mobility

    allM = np.zeros((nHeights, 3*numberParticles, 3*numberParticles), dtype=precision)
    for i in range(0,nHeights):
        positions = np.array([[xymax/2, xymax/2, heights[i]]], dtype=precision)
        solver.setPositions(positions)
        
        M = compute_M(solver, numberParticles)
        M /= normMat
        allM[i] = M

    # scipy.io.savemat('./temp/self_mobility_bw.mat', {'M': allM, 'heights': heights})
    ref = scipy.io.loadmat(ref_dir + ref_file)
    refM = ref['M']
    refHeights = ref['heights'][0]

    f = scipy.interpolate.interp1d(heights, allM, axis=0)
    Minterp = f(refHeights)

    interp_diags = [np.diag(matrix) for matrix in Minterp]
    ref_diags = [np.diag(matrix)[0:3] for matrix in refM] # only take diagonal elements from forces

    assert np.all(np.diag(allM[0]) == [0,0,0]), "Self mobility is not zero on the wall at z=0"

    assert np.allclose(interp_diags, ref_diags, atol=3e-2), "Self mobility does not match reference"

if __name__ == "__main__":
    # test_self_mobility(DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_ref.mat")
    test_self_mobility(DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_ref.mat")