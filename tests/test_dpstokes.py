import numpy as np
import libMobility as lm


def compute_with_dpstokes(pos, forces, lx, ly):
    params = {"viscosity": 1 / (6 * np.pi), "hydrodynamicRadius": 1.0}
    dpstokes = lm.DPStokes(
        periodicityX="periodic", periodicityY="periodic", periodicityZ="single_wall"
    )
    dpstokes.setParameters(Lx=lx, Ly=ly, zmin=-8.0, zmax=8.0)
    dpstokes.initialize(**params)
    dpstokes.setPositions(pos)
    mf_dp, _ = dpstokes.Mdot(forces=forces)
    return mf_dp


def test_non_square_box():
    nP = 2
    pos_or = np.random.uniform(-8, 8, (nP, 3)).astype(np.float32)
    forces_or = np.random.uniform(-1, 1, (nP, 3)).astype(np.float32)
    forces_or -= np.mean(forces_or, axis=0)  # Center the forces around zero

    L_mult = 1.0
    L_short = 16.0 * L_mult
    L_fact = 2.0 * L_mult
    L_long = L_short * L_fact

    mf_dp_cube = compute_with_dpstokes(pos_or, forces_or, lx=L_short, ly=L_short)

    # This must be identical than doubling the box size in x and repeating the particles
    pos = np.vstack((pos_or, pos_or + np.array([L_short, 0.0, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_lx = compute_with_dpstokes(pos, forces, lx=L_long, ly=L_short)[
        : len(pos_or), :
    ]

    # And the same for doubling the box size in y
    pos = np.vstack((pos_or, pos_or + np.array([0.0, L_short, 0.0])))
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


def test_hydro_radius():
    eta = 2.5
    a = 1.2
    L0 = 50.0
    L_fact = 1.0
    L = [L0 * L_fact, L0]
    # L = [L0, L0 * L_fact]
    solver = lm.DPStokes("periodic", "periodic", "open")
    solver.setParameters(Lx=L[0], Ly=L[1], zmin=-8.0, zmax=8.0)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    pos = np.random.uniform(-8, 8, (3)).astype(np.float32)
    forces = np.random.uniform(-10, 10, (3)).astype(np.float32)
    # forces = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    n_reps = 1
    pos_tiled = np.tile(pos, (n_reps, 1))
    z_facts = np.arange(-(n_reps // 2), n_reps // 2 + 1)
    pos_tiled[:, 2] = pos_tiled[:, 2] + 16 * z_facts
    forces_tiled = np.tile(forces, (n_reps, 1))
    solver.setPositions(pos_tiled)
    mf, _ = solver.Mdot(forces=forces_tiled)

    print(pos_tiled)

    # solver.setPositions(pos)
    # mf, _ = solver.Mdot(forces=forces)

    mf_center = mf[n_reps // 2]
    # mf_center = mf

    Km = 2.83
    a_x = Km / L[0] + 6 * np.pi * eta * mf_center[0] / forces[0]
    a_y = Km / L[1] + 6 * np.pi * eta * mf_center[1] / forces[1]
    a_z = Km / L[1] + 6 * np.pi * eta * mf_center[2] / forces[2]
    #
    print("1/a:", 1 / a)
    print(a_x, a_y, a_z)
    # print("1/6 pi eta a:", 1 / (6 * np.pi * eta * a))
    # print("mf: ", mf_center)
