import numpy as np


sane_parameters = {
    "PSE": {"psi": 1.0, "Lx": 32, "Ly": 32, "Lz": 32, "shearStrain": 0.0},
    "NBody": {"algorithm": "advise"},
    "DPStokes": {"dt": 1, "Lx": 16, "Ly": 16, "zmin": -6, "zmax": 6},
    "SelfMobility": {"parameter": 5.0},
}


def compute_M(solver, numberParticles):
    precision = np.float32 if solver.precision == "float" else np.float64
    size = 3 * numberParticles
    M = np.zeros((size, size), dtype=precision)
    I = np.identity(size, dtype=precision)
    for i in range(0, size):
        forces = I[:, i].copy()
        mf = np.zeros(size, dtype=precision)
        solver.Mdot(forces, mf)
        M[:, i] = mf
    return M
