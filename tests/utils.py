import numpy as np
from libMobility import SelfMobility, NBody, PSE, DPStokes


sane_parameters = {
    "PSE": {"psi": 1.0, "Lx": 32, "Ly": 32, "Lz": 32, "shearStrain": 0.0},
    "NBody": {"algorithm": "advise"},
    "NBody_wall": {"algorithm": "advise", "wallHeight": 0.0},
    "DPStokes": {
        "Lx": 16,
        "Ly": 16,
        "zmin": -6,
        "zmax": 6,
        "allowChangingBoxSize": False,
    },
    "SelfMobility": {"parameter": 5.0},
}

wall_parameters = {
    "NBody": {"algorithm": "advise", "wallHeight": 0.0},
    "DPStokes": {
        "Lx": 76.8,
        "Ly": 76.8,
        "zmin": 0,
        "zmax": 19.2,
        "allowChangingBoxSize": False,
    },
}


solver_configs_all = [
    (SelfMobility, ("open", "open", "open")),
    # (NBody, ("open", "open", "open")),
    # (NBody, ("open", "open", "single_wall")),
    # (PSE, ("periodic", "periodic", "periodic")),
    # (DPStokes, ("periodic", "periodic", "open")),
    # (DPStokes, ("periodic", "periodic", "single_wall")),
    # (DPStokes, ("periodic", "periodic", "two_walls")),
]

solver_configs_torques = [
    (Solver, periodicity)
    for Solver, periodicity in solver_configs_all
    if not (Solver == PSE)
]


def initialize_solver(
    Solver, periodicity, includeAngular=False, parameters=None, **kwargs
):
    solver = Solver(*periodicity)
    if parameters is not None:
        solver.setParameters(**parameters)
    else:
        solver.setParameters(**get_sane_params(Solver.__name__, periodicity[2]))
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        includeAngular=includeAngular,
    )
    return solver


def compute_M(solver, numberParticles, includeAngular):
    precision = np.float32 if solver.precision == "float" else np.float64
    if includeAngular:
        size = 6 * numberParticles
    else:
        size = 3 * numberParticles
    M = np.zeros((size, size), dtype=precision)
    I = np.identity(size, dtype=precision)
    for i in range(0, size):
        forces = I[0 : 3 * numberParticles, i].copy()
        if includeAngular:
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
        positions[:, 2] += 0.5  # [-0.5, 0.5] -> [0, 1]
        positions *= 10  # [0, 1] -> [0, 10]

    return positions


def get_sane_params(solverName, geom=None):
    if solverName == "NBody" and geom == "single_wall":
        params = sane_parameters["NBody_wall"].copy()
    else:
        params = sane_parameters[solverName].copy()
    return params


def get_wall_params(solverName, wallHeight):
    params = wall_parameters[
        solverName
    ].copy()  # copy is necessary, otherwise modifications accumulate
    if solverName == "DPStokes":
        params["zmax"] += wallHeight
        params["zmin"] += wallHeight
    elif solverName == "NBody":
        params["wallHeight"] = wallHeight

    return params
