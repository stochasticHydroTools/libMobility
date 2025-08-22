import pytest
import numpy as np

from libMobility import DPStokes, NBody
from utils import compute_M, get_wall_params
import os

# NOTE: Some of the following tests will only pass if compiled with double precision.
# This is because the reference data was generated in double precision.

precision_str = "single" if NBody.precision == "float" else "double"
ref_dir = os.path.dirname(os.path.abspath(__file__)) + "/ref/"


@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_w4.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_w4.npz"),
        (NBody, ("open", "open", "single_wall"), "self_mobility_bw_ref_noimg.npz"),
    ],
)
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_self_mobility_linear(Solver, periodicity, ref_file, wallHeight):
    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    includeAngular = False
    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"].flatten()

    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)

    tol = 100 * np.finfo(precision_str).eps

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 1
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
    )

    nHeights = len(refHeights)

    normMat = np.ones((3 * numberParticles, 3 * numberParticles), dtype=precision_str)
    diag_ind = np.diag_indices_from(normMat)
    normMat[diag_ind] = 1 / (
        6 * np.pi * eta * hydrodynamicRadius
    )  # only for diag elements as this is for self mobility

    allM = np.zeros(
        (nHeights, 3 * numberParticles, 3 * numberParticles), dtype=precision_str
    )
    for i in range(0, nHeights):
        positions = np.array(
            [[xymax / 2, xymax / 2, refHeights[i] + wallHeight]], dtype=precision_str
        )
        solver.setPositions(positions)

        M = compute_M(solver, numberParticles, includeAngular)
        M /= normMat
        allM[i] = M

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [
        np.diag(matrix)[0:3] for matrix in refM
    ]  # only take diagonal elements from forces

    assert np.allclose(diags, ref_diags, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "pair_mobility_bw_w4.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "pair_mobility_sc_w4.npz"),
        (
            NBody,
            ("open", "open", "single_wall"),
            "pair_mobility_bw_ref_noimg_offset_x.npz",
        ),
    ],
)
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_pair_mobility_linear(Solver, periodicity, ref_file, wallHeight):
    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    includeAngular = False
    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"]
    nHeights = len(refHeights)

    radH = 1.0  # hydrodynamic radius
    eta = 1 / 4 / np.sqrt(np.pi)

    tol = 100 * np.finfo(precision_str).eps

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 2
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=radH,
    )

    normMat = (1 / (6 * np.pi * eta)) * np.ones(
        (3 * numberParticles, 3 * numberParticles), dtype=precision_str
    )

    seps = np.array([3 * radH, 4 * radH, 8 * radH])
    nSeps = len(seps)

    allM = np.zeros(
        (nSeps, nHeights, 3 * numberParticles, 3 * numberParticles), dtype=precision_str
    )
    for i in range(0, nSeps):
        for k in range(0, nHeights):
            xpos = xymax / 2
            positions = np.array(
                [
                    [xpos + seps[i] / 2, xpos, refHeights[k] + wallHeight],
                    [xpos - seps[i] / 2, xpos, refHeights[k] + wallHeight],
                ],
                dtype=precision_str,
            )
            solver.setPositions(positions)

            M = compute_M(solver, numberParticles, includeAngular)
            M /= normMat
            allM[i][k] = M

    assert np.allclose(allM, refM[:, :, 0:6, 0:6], atol=tol, rtol=tol)


@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "self_mobility_bw_w6.npz"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "self_mobility_sc_w6.npz"),
        (NBody, ("open", "open", "single_wall"), "self_mobility_bw_ref_noimg.npz"),
    ],
)
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_self_mobility_angular(Solver, periodicity, ref_file, wallHeight):
    if precision_str == "single" and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)

    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)

    tol = 100 * np.finfo(precision_str).eps

    includeAngular = True

    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"].flatten()

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 1
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
        includeAngular=includeAngular,
    )

    nHeights = len(refHeights)

    normMat = np.zeros((6 * numberParticles, 6 * numberParticles), dtype=precision_str)
    normMat[0:3, 0:3] = 1 / (6 * np.pi * eta * hydrodynamicRadius)  # tt
    normMat[3:, 3:] = 1 / (8 * np.pi * eta * hydrodynamicRadius**3)  # rr
    normMat[3:, 0:3] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # tr
    normMat[0:3, 3:] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # rt

    allM = np.zeros(
        (nHeights, 6 * numberParticles, 6 * numberParticles), dtype=precision_str
    )
    for i in range(0, nHeights):
        positions = np.array(
            [[xymax / 2, xymax / 2, refHeights[i] + wallHeight]], dtype=precision_str
        )
        solver.setPositions(positions)

        M = compute_M(solver, numberParticles, includeAngular)
        M /= normMat
        allM[i] = M

    assert np.allclose(allM, refM, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    ("Solver", "periodicity", "ref_file"),
    [
        (DPStokes, ("periodic", "periodic", "single_wall"), "pair_mobility_bw_w6"),
        (DPStokes, ("periodic", "periodic", "two_walls"), "pair_mobility_sc_w6"),
        (NBody, ("open", "open", "single_wall"), "pair_mobility_bw_ref_noimg"),
    ],
)
@pytest.mark.parametrize("offset", ["x", "y"])
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_pair_mobility_angular(Solver, periodicity, ref_file, offset, wallHeight):
    if precision_str == "single" and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)
    includeAngular = True

    tol = 100 * np.finfo(precision_str).eps

    ref_file += "_offset_" + offset + ".npz"
    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"].flatten()

    nP = 2
    solver = Solver(*periodicity)
    solver.setParameters(**params)
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
        includeAngular=includeAngular,
    )

    nHeights = len(refHeights)

    seps = np.array(
        [3 * hydrodynamicRadius, 4 * hydrodynamicRadius, 8 * hydrodynamicRadius]
    )
    nSeps = len(seps)

    normMat = np.zeros((6 * nP, 6 * nP), dtype=precision_str)
    normMat[0 : 3 * nP, 0 : 3 * nP] = 1 / (6 * np.pi * eta * hydrodynamicRadius)  # tt
    normMat[3 * nP :, 3 * nP :] = 1 / (8 * np.pi * eta * hydrodynamicRadius**3)  # rr
    normMat[3 * nP :, 0 : 3 * nP] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # tr
    normMat[0 : 3 * nP, 3 * nP :] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # rt

    if offset == "x":
        offset_vec = np.array([1, 0, 0])
    elif offset == "y":
        offset_vec = np.array([0, 1, 0])
    else:
        raise ValueError("Test for offset in {} not implemented".format(offset))

    xpos = xymax / 2
    allM = np.zeros((nSeps, nHeights, 6 * nP, 6 * nP), dtype=precision_str)
    for i in range(0, nSeps):
        seps_vec = (seps[i] * offset_vec) / 2
        for k in range(0, nHeights):
            positions = np.array(
                [
                    [xpos, xpos, refHeights[k] + wallHeight] + seps_vec,
                    [xpos, xpos, refHeights[k] + wallHeight] - seps_vec,
                ],
                dtype=precision_str,
            )
            solver.setPositions(positions)

            M = compute_M(solver, nP, includeAngular)
            M /= normMat
            allM[i, k] = M

    assert np.allclose(allM, refM, atol=tol, rtol=tol)


def test_dpstokes_matching_rpy():

    a = np.random.uniform(0.5, 2.0)
    eta = 1 / (6 * np.pi * a)
    L_min = 100 * a  # need a fairly large domain to neglect periodic effects
    L_fact = np.random.uniform(1.0, 3.0)
    L = L_min * L_fact
    max_height = 10.0 * a
    min_height = 2.0 * a  # need a buffer for regularized kernels to match

    solver = DPStokes("periodic", "periodic", "single_wall")
    solver.setParameters(Lx=L, Ly=L, zmin=0.0, zmax=max_height)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta, includeAngular=True)

    rpy = NBody("open", "open", "single_wall")
    rpy.setParameters(wallHeight=0.0)
    rpy.initialize(hydrodynamicRadius=a, viscosity=eta, includeAngular=True)

    nP = 10
    # generate particles in interior of domain to avoid periodic artifacts
    pos = np.random.uniform(L / 4, 3 * L / 4, (nP, 3)).astype(np.float32)
    pos_z = np.random.uniform(min_height, max_height, nP).astype(np.float32)
    pos[:, 2] = pos_z
    F = np.eye(6 * nP)

    results_dpstokes = np.zeros((2 * 6 * nP, 3 * nP))
    results_rpy = np.zeros((2 * 6 * nP, 3 * nP))

    rpy.setPositions(pos)
    solver.setPositions(pos)
    for j in range(6 * nP):
        f_j = F[: 3 * nP, j].copy()
        t_j = F[3 * nP :, j].copy()
        mf_j, mt_j = solver.Mdot(forces=f_j, torques=t_j)
        results_dpstokes[j] += mf_j
        results_dpstokes[j + 6 * nP] += mt_j

        mf_j, mt_j = rpy.Mdot(forces=f_j, torques=t_j)
        results_rpy[j] += mf_j
        results_rpy[j + 6 * nP] += mt_j

    assert np.allclose(results_dpstokes, results_rpy, rtol=1e-3, atol=1e-2)
