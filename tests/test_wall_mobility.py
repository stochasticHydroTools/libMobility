import pytest
import numpy as np

from libMobility import DPStokes, NBody
import libMobility
from utils import compute_M, get_wall_params

# NOTE: Some of the following tests will only pass if compiled with double precision.
# This is because the reference data was generated in double precision.

precision = np.float32 if NBody.precision == "float" else np.float64


@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "ref_file"),
    [
        (
            DPStokes,
            ("periodic", "periodic", "single_wall"),
            1e-6,
            "self_mobility_bw_w4.npz",
        ),
        (
            DPStokes,
            ("periodic", "periodic", "two_walls"),
            1e-6,
            "self_mobility_sc_w4.npz",
        ),
        (
            NBody,
            ("open", "open", "single_wall"),
            1e-6,
            "self_mobility_bw_ref_noimg.npz",
        ),
    ],
)
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_self_mobility_linear(Solver, periodicity, tol, ref_file, wallHeight):
    if precision == np.float32 and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    includeAngular = False

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"].flatten()

    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 1
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=hydrodynamicRadius,
    )

    nHeights = len(refHeights)

    normMat = np.ones((3 * numberParticles, 3 * numberParticles), dtype=precision)
    diag_ind = np.diag_indices_from(normMat)
    normMat[diag_ind] = 1 / (
        6 * np.pi * eta * hydrodynamicRadius
    )  # only for diag elements as this is for self mobility

    allM = np.zeros(
        (nHeights, 3 * numberParticles, 3 * numberParticles), dtype=precision
    )
    for i in range(0, nHeights):
        positions = np.array(
            [[xymax / 2, xymax / 2, refHeights[i] + wallHeight]], dtype=precision
        )
        solver.setPositions(positions)

        M = compute_M(solver, numberParticles, includeAngular)
        M /= normMat
        allM[i] = M

    diags = [np.diag(matrix) for matrix in allM]
    ref_diags = [
        np.diag(matrix)[0:3] for matrix in refM
    ]  # only take diagonal elements from forces

    for diag, ref_diag in zip(diags, ref_diags):
        assert np.allclose(
            diag, ref_diag, rtol=tol, atol=tol
        ), f"Self mobility does not match reference"


@pytest.mark.parametrize(
    ("Solver", "periodicity", "tol", "ref_file"),
    [
        (
            DPStokes,
            ("periodic", "periodic", "single_wall"),
            1e-6,
            "pair_mobility_bw_w4.npz",
        ),
        (
            DPStokes,
            ("periodic", "periodic", "two_walls"),
            1e-6,
            "pair_mobility_sc_w4.npz",
        ),
        (
            NBody,
            ("open", "open", "single_wall"),
            1e-4,
            "pair_mobility_bw_ref_noimg_offset_x.npz",
        ),
    ],
)
@pytest.mark.parametrize("wallHeight", [0, 5.4, -10])
def test_pair_mobility_linear(Solver, periodicity, ref_file, tol, wallHeight):
    if precision == np.float32 and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    includeAngular = False

    ref_dir = "./ref/"
    ref = np.load(ref_dir + ref_file)
    refM = ref["M"]
    refHeights = ref["heights"]
    nHeights = len(refHeights)

    radH = 1.0  # hydrodynamic radius
    eta = 1 / 4 / np.sqrt(np.pi)

    tol = 100 * np.finfo(precision).eps

    solver = Solver(*periodicity)
    solver.setParameters(**params)
    numberParticles = 2
    solver.initialize(
        viscosity=eta,
        hydrodynamicRadius=radH,
    )

    normMat = (1 / (6 * np.pi * eta)) * np.ones(
        (3 * numberParticles, 3 * numberParticles), dtype=precision
    )

    seps = np.array([3 * radH, 4 * radH, 8 * radH])
    nSeps = len(seps)

    allM = np.zeros(
        (nSeps, nHeights, 3 * numberParticles, 3 * numberParticles), dtype=precision
    )
    for i in range(0, nSeps):
        for k in range(0, nHeights):
            xpos = xymax / 2
            positions = np.array(
                [
                    [xpos + seps[i] / 2, xpos, refHeights[k] + wallHeight],
                    [xpos - seps[i] / 2, xpos, refHeights[k] + wallHeight],
                ],
                dtype=precision,
            )
            solver.setPositions(positions)

            M = compute_M(solver, numberParticles, includeAngular)
            M /= normMat
            allM[i][k] = M

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i, k] - refM[i, k][0:6, 0:6])
            assert np.all(diff < tol)


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
    if precision == np.float32 and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)

    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)

    includeAngular = True
    tol = 1e-6

    ref_dir = "./ref/"
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

    normMat = np.zeros((6 * numberParticles, 6 * numberParticles), dtype=precision)
    normMat[0:3, 0:3] = 1 / (6 * np.pi * eta * hydrodynamicRadius)  # tt
    normMat[3:, 3:] = 1 / (8 * np.pi * eta * hydrodynamicRadius**3)  # rr
    normMat[3:, 0:3] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # tr
    normMat[0:3, 3:] = 1 / (6 * np.pi * eta * hydrodynamicRadius**2)  # rt

    allM = np.zeros(
        (nHeights, 6 * numberParticles, 6 * numberParticles), dtype=precision
    )
    for i in range(0, nHeights):
        positions = np.array(
            [[xymax / 2, xymax / 2, refHeights[i] + wallHeight]], dtype=precision
        )
        solver.setPositions(positions)

        M = compute_M(solver, numberParticles, includeAngular)
        M /= normMat
        allM[i] = M

    for i in range(0, nHeights):
        diff = abs(allM[i] - refM[i])
        assert np.all(diff < tol)


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
    if precision == np.float32 and Solver.__name__ == "DPStokes":
        pytest.skip(
            "The test is only valid for double precision due to how reference data was generated."
        )

    xymax = 76.8
    params = get_wall_params(Solver.__name__, wallHeight)
    hydrodynamicRadius = 1.0
    eta = 1 / 4 / np.sqrt(np.pi)
    includeAngular = True

    tol = 1e-6

    ref_dir = "./ref/"
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

    normMat = np.zeros((6 * nP, 6 * nP), dtype=precision)
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
    allM = np.zeros((nSeps, nHeights, 6 * nP, 6 * nP), dtype=precision)
    for i in range(0, nSeps):
        seps_vec = (seps[i] * offset_vec) / 2
        for k in range(0, nHeights):
            positions = np.array(
                [
                    [xpos, xpos, refHeights[k] + wallHeight] + seps_vec,
                    [xpos, xpos, refHeights[k] + wallHeight] - seps_vec,
                ],
                dtype=precision,
            )
            solver.setPositions(positions)

            M = compute_M(solver, nP, includeAngular)
            M /= normMat
            allM[i, k] = M

    for i in range(0, nSeps):
        for k in range(0, nHeights):
            diff = abs(allM[i, k] - refM[i, k])
            assert np.all(diff < tol)
