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
        M[:, i] = solver.Mdot(forces)[0].reshape(3 * numberParticles)
    return M


def generate_positions_in_box(parameters, numberParticles):
    positions = np.random.rand(numberParticles, 3) - 0.5
    if "Lx" in parameters:
        positions[:, 0] *= parameters["Lx"]
    if "Ly" in parameters:
        positions[:, 1] *= parameters["Ly"]
    if "Lz" in parameters:
        positions[:, 2] *= parameters["Lz"]
    if "zmin" in parameters:
        positions[:, 2] *= parameters["zmax"] - parameters["zmin"]
    return positions
