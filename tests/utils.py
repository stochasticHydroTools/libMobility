import numpy as np


sane_parameters = {
    "PSE": {"psi": 1.0, "Lx": 32, "Ly": 32, "Lz": 32, "shearStrain": 0.0},
    "NBody": {"algorithm": "advise"},
    "DPStokes": {"dt": 1, "Lx": 16, "Ly": 16, "zmin": -6, "zmax": 6, "allowChangingBoxSize": False},
    "SelfMobility": {"parameter": 5.0},
}

def initialize_solver(Solver, periodicity, numberParticles, needsTorque=False):
    solver = Solver(*periodicity)
    solver.setParameters(**sane_parameters[Solver.__name__])
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        numberParticles=numberParticles,
        needsTorque=needsTorque,
    )
    return solver


def compute_M(solver, numberParticles, needsTorque):
    precision = np.float32 if solver.precision == "float" else np.float64
    if needsTorque:
        size = 6 * numberParticles
    else:
        size = 3 * numberParticles
    M = np.zeros((size, size), dtype=precision)
    I = np.identity(size, dtype=precision)
    for i in range(0, size):
        forces = I[0 : 3 * numberParticles, i].copy()
        if needsTorque:
            torques = I[3 * numberParticles :, i].copy()
            linear, angular = solver.Mdot(forces, torques)
            M[:, i] = np.concatenate(
                (
                    linear.reshape(3 * numberParticles),
                    angular.reshape(3 * numberParticles),
                )
            )
        else:
            linear, _ = solver.Mdot(forces)
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

    # generates positions for NBody
    if "algorithm" in parameters:
        positions[:, 2] += 0.5
        positions *= 10

    return positions