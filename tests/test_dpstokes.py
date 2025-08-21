import numpy as np
import libMobility as lm


def compute_with_dpstokes(pos, forces, lx, ly):
    params = {"viscosity": 1 / (6 * np.pi), "hydrodynamicRadius": 1.73}
    dpstokes = lm.DPStokes(
        periodicityX="periodic", periodicityY="periodic", periodicityZ="single_wall"
    )
    dpstokes.setParameters(Lx=lx, Ly=ly, zmin=-8.0, zmax=8.0)
    dpstokes.initialize(**params)
    dpstokes.setPositions(pos)
    mf_dp, _ = dpstokes.Mdot(forces=forces)
    return mf_dp


def test_non_square_box():
    nP = 10
    pos_or = np.random.uniform(-8, 8, (nP, 3)).astype(np.float32)
    forces_or = np.random.uniform(-1, 1, (nP, 3)).astype(np.float32)
    forces_or -= np.mean(forces_or, axis=0)  # Center the forces around zero

    L_fact = 2.0
    L_short = 16.0
    L_long = L_short * L_fact

    mf_dp_cube = compute_with_dpstokes(pos_or, forces_or, lx=L_short, ly=L_short)

    # This must be identical than doubling the box size in x and repeating the particles
    pos = np.vstack((pos_or, pos_or + np.array([0.5 * L_long, 0.0, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_lx = compute_with_dpstokes(pos, forces, lx=L_long, ly=L_short)[
        : len(pos_or), :
    ]

    # And the same for doubling the box size in y
    pos = np.vstack((pos_or, pos_or + np.array([0.0, 0.5 * L_long, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_ly = compute_with_dpstokes(pos, forces, lx=L_short, ly=L_long)[
        : len(pos_or), :
    ]
    lx_match = np.allclose(mf_dp_lx, mf_dp_cube, rtol=1e-3, atol=1e-2)
    ly_match = np.allclose(mf_dp_ly, mf_dp_cube, rtol=1e-3, atol=1e-2)

    if not lx_match:
        print("DPStokes lx=32.0 does not match cube")
    if not ly_match:
        print("DPStokes ly=32.0 does not match cube")

    assert lx_match and ly_match, "DPStokes non-square box test failed"


def test_isotropy():
    L0 = 16.0
    L_fact = 2.0

    f_x = np.array([1.0, 0.0, 0.0])
    f_y = np.array([0.0, 1.0, 0.0])

    n_runs = 5
    for _ in range(n_runs):
        pos = np.random.uniform(-8, 8, (3)).astype(np.float32)

        mf_x = compute_with_dpstokes(pos, f_x, lx=L0 * L_fact, ly=L0)
        mf_y = compute_with_dpstokes(pos, f_y, lx=L0, ly=L0 * L_fact)

        assert np.allclose(mf_x[0], mf_y[1], rtol=1e-3, atol=1e-2)
        assert np.allclose(mf_x[1], mf_y[0], rtol=1e-3, atol=1e-2)
        assert np.allclose(mf_x[2], mf_y[2], rtol=1e-3, atol=1e-2)


def test_non_square_matching_rpy():

    a = np.random.uniform(0.5, 2.0)
    eta = 1 / (6 * np.pi * a)
    L_min = 100 * a  # need a fairly large domain to neglect periodic effects
    Lx = np.random.uniform(L_min, 4.0 * L_min)
    Ly = np.random.uniform(L_min, 4.0 * L_min)
    max_height = 10.0 * a
    min_height = 1.5 * a  # need a buffer for regularized kernels to match

    solver = lm.DPStokes("periodic", "periodic", "single_wall")
    solver.setParameters(Lx=Lx, Ly=Ly, zmin=0.0, zmax=max_height)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    rpy = lm.NBody("open", "open", "single_wall")
    rpy.setParameters(wallHeight=0.0)
    rpy.initialize(hydrodynamicRadius=a, viscosity=eta)

    F = np.eye(3)

    heights = np.linspace(min_height, max_height, 20)

    results_dpstokes = np.zeros((len(heights), 3))
    results_rpy = np.zeros((len(heights), 3))

    for i, h in enumerate(heights):
        pos = np.array([0.0, 0.0, h])

        rpy.setPositions(pos)
        solver.setPositions(pos)
        for j in range(3):
            f_j = F[:, j].copy()
            mf_j = solver.Mdot(forces=f_j)[0]
            results_dpstokes[i] += mf_j

            mf_j = rpy.Mdot(forces=f_j)[0]
            results_rpy[i] += mf_j

    assert np.allclose(results_dpstokes[:, 0], results_rpy[:, 0], rtol=1e-3, atol=1e-2)
    assert np.allclose(results_dpstokes[:, 1], results_rpy[:, 1], rtol=1e-3, atol=1e-2)
    assert np.allclose(results_dpstokes[:, 2], results_rpy[:, 2], rtol=1e-3, atol=1e-2)
