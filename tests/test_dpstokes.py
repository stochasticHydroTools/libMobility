import numpy as np
import libMobility as lm


def compute_with_dpstokes(pos, lx, ly, forces=None, torques=None):
    params = {"viscosity": 1 / (6 * np.pi), "hydrodynamicRadius": 1.73}
    dpstokes = lm.DPStokes("periodic", "periodic", "single_wall")
    torques_on = True if torques is not None else False
    dpstokes.setParameters(Lx=lx, Ly=ly, zmin=-8.0, zmax=8.0)
    dpstokes.initialize(**params, includeAngular=torques_on)
    dpstokes.setPositions(pos)
    mf_dp, mt_dp = dpstokes.Mdot(forces=forces, torques=torques)
    if torques_on:
        return mt_dp
    else:
        return mf_dp


def test_non_square_box():
    nP = 10
    pos_or = np.random.uniform(-8, 8, (nP, 3)).astype(np.float32)
    forces_or = np.random.uniform(-1, 1, (nP, 3)).astype(np.float32)
    forces_or -= np.mean(forces_or, axis=0)  # Center the forces around zero

    L_fact = 2.0
    L_short = 16.0
    L_long = L_short * L_fact

    mf_dp_cube = compute_with_dpstokes(pos_or, forces=forces_or, lx=L_short, ly=L_short)
    mt_dp_cube = compute_with_dpstokes(
        pos_or, torques=forces_or, lx=L_short, ly=L_short
    )

    # This must be identical than doubling the box size in x and repeating the particles
    pos = np.vstack((pos_or, pos_or + np.array([0.5 * L_long, 0.0, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_lx = compute_with_dpstokes(pos, forces=forces, lx=L_long, ly=L_short)[
        : len(pos_or), :
    ]
    mt_dp_lx = compute_with_dpstokes(pos, torques=forces, lx=L_long, ly=L_short)[
        : len(pos_or), :
    ]

    # And the same for doubling the box size in y
    pos = np.vstack((pos_or, pos_or + np.array([0.0, 0.5 * L_long, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_ly = compute_with_dpstokes(pos, forces=forces, lx=L_short, ly=L_long)[
        : len(pos_or), :
    ]
    mt_dp_ly = compute_with_dpstokes(pos, torques=forces, lx=L_short, ly=L_long)[
        : len(pos_or), :
    ]

    assert np.allclose(mf_dp_lx, mf_dp_cube, rtol=1e-3, atol=1e-2)
    assert np.allclose(mf_dp_ly, mf_dp_cube, rtol=1e-3, atol=1e-2)

    assert np.allclose(mt_dp_lx, mt_dp_cube, rtol=1e-3, atol=1e-2)
    assert np.allclose(mt_dp_ly, mt_dp_cube, rtol=1e-3, atol=1e-2)


def test_isotropy():
    L0 = 16.0
    L_fact = 2.0

    f_x = np.array([1.0, 0.0, 0.0])
    f_y = np.array([0.0, 1.0, 0.0])

    n_runs = 5
    for _ in range(n_runs):
        pos = np.random.uniform(-8, 8, (3)).astype(np.float32)

        mf_x = compute_with_dpstokes(pos, forces=f_x, lx=L0 * L_fact, ly=L0)
        mf_y = compute_with_dpstokes(pos, forces=f_y, lx=L0, ly=L0 * L_fact)

        mt_x = compute_with_dpstokes(pos, torques=f_x, lx=L0 * L_fact, ly=L0)
        mt_y = compute_with_dpstokes(pos, torques=f_y, lx=L0, ly=L0 * L_fact)

        assert np.allclose(mf_x[0], mf_y[1], rtol=1e-3, atol=1e-2)
        assert np.allclose(mf_x[1], mf_y[0], rtol=1e-3, atol=1e-2)
        assert np.allclose(mf_x[2], mf_y[2], rtol=1e-3, atol=1e-2)

        assert np.allclose(mt_x[0], mt_y[1], rtol=1e-3, atol=1e-2)
        assert np.allclose(mt_x[1], mt_y[0], rtol=1e-3, atol=1e-2)
        assert np.allclose(mt_x[2], mt_y[2], rtol=1e-3, atol=1e-2)


def test_non_square_matching_rpy():

    a = np.random.uniform(0.5, 2.0)
    eta = 1 / (6 * np.pi * a)
    L_min = 100 * a  # need a fairly large domain to neglect periodic effects
    Lx = np.random.uniform(L_min, 3.0 * L_min)
    Ly = np.random.uniform(L_min, 3.0 * L_min)
    max_height = 10.0 * a
    min_height = 1.5 * a  # need a buffer for regularized kernels to match

    solver = lm.DPStokes("periodic", "periodic", "single_wall")
    solver.setParameters(Lx=Lx, Ly=Ly, zmin=0.0, zmax=max_height)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta, includeAngular=True)

    rpy = lm.NBody("open", "open", "single_wall")
    rpy.setParameters(wallHeight=0.0)
    rpy.initialize(hydrodynamicRadius=a, viscosity=eta, includeAngular=True)

    F = np.eye(6)

    heights = np.linspace(min_height, max_height, 20)

    results_dpstokes = np.zeros((2 * len(heights), 3))
    results_rpy = np.zeros((2 * len(heights), 3))

    for i, h in enumerate(heights):
        pos = np.array([0.0, 0.0, h])

        rpy.setPositions(pos)
        solver.setPositions(pos)
        for j in range(6):
            f_j = F[0:3, j].copy()
            t_j = F[3:6, j].copy()
            mf_j, mt_j = solver.Mdot(forces=f_j, torques=t_j)
            results_dpstokes[i] += mf_j
            results_dpstokes[i + 3] += mt_j

            mf_j, mt_j = rpy.Mdot(forces=f_j, torques=t_j)
            results_rpy[i] += mf_j
            results_rpy[i + 3] += mt_j

    assert np.allclose(results_dpstokes, results_rpy, rtol=1e-3, atol=1e-2)
