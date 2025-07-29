import numpy as np
import libMobility as lm


def compute_with_dpstokes(pos, forces, lx, ly):
    lz = 15
    params = {"viscosity": 1 / (6 * np.pi), "hydrodynamicRadius": 1.0}
    dpstokes = lm.DPStokes(
        periodicityX="periodic", periodicityY="periodic", periodicityZ="single_wall"
    )
    dpstokes.setParameters(Lx=lx, Ly=ly, zmin=-1.0, zmax=lz)
    dpstokes.initialize(**params)
    dpstokes.setPositions(pos)
    mf_dp, _ = dpstokes.Mdot(forces=forces)
    return mf_dp


def test_non_square_box():
    pos_or = np.random.uniform(-8, 8, (10, 3)).astype(np.float32)
    forces_or = np.random.uniform(-1, 1, (10, 3)).astype(np.float32)
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
