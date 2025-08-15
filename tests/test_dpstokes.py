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

    mf_dp_cube = compute_with_dpstokes(pos_or, forces_or, lx=16.0, ly=16.0)

    # This must be identical than doubling the box size in x and repeating the particles
    pos = np.vstack((pos_or, pos_or + np.array([16.0, 0.0, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_lx = compute_with_dpstokes(pos, forces, lx=32.0, ly=16.0)[: len(pos_or), :]

    # And the same for doubling the box size in y
    pos = np.vstack((pos_or, pos_or + np.array([0.0, 16.0, 0.0])))
    forces = np.vstack((forces_or, forces_or))
    mf_dp_ly = compute_with_dpstokes(pos, forces, lx=16.0, ly=32.0)[: len(pos_or), :]
    assert np.allclose(
        mf_dp_lx, mf_dp_cube, rtol=1e-3, atol=1e-2
    ), "DPStokes lx=32.0 does not match cube"
    assert np.allclose(
        mf_dp_ly, mf_dp_cube, rtol=1e-3, atol=1e-2
    ), "DPStokes ly=32.0 does not match cube"


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
