import pytest
import numpy as np
from libMobility import *
from utils import sane_parameters, solver_configs_all

# List of classes inside libMobility

solver_list = [PSE, NBody, DPStokes, SelfMobility]


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
def test_periodic_initialization(Solver, periodicity):
    solver = Solver(*periodicity)


def test_pse_torques_unsupported():
    hydrodynamicRadius = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    parameters = sane_parameters[PSE.__name__]
    solver.setParameters(**parameters)
    with pytest.raises(RuntimeError):
        solver.initialize(
            viscosity=1.0,
            hydrodynamicRadius=hydrodynamicRadius,
            includeAngular=True,
        )


@pytest.mark.parametrize("Solver", solver_list)
def test_invalid_throws(Solver):
    with pytest.raises(RuntimeError):
        solver = Solver("periodicasdas", "periodic", "open")


@pytest.mark.parametrize(
    ("NBatch", "NperBatch"),
    [
        (1, 12),
        (12, 1),
    ],
)
def test_nbody_good_batch_parameters(NBatch, NperBatch):
    solver = NBody("open", "open", "open")
    solver.setParameters("advise", NBatch, NperBatch)
    numberParticles = NBatch * NperBatch
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.5,
    )
    positions = np.random.rand(numberParticles, 3)
    solver.setPositions(positions)


def test_nbody_default_parameters():
    solver = NBody("open", "open", "open")
    solver.setParameters("advise")
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.5,
    )
    positions = np.random.rand(10, 3)
    solver.setPositions(positions)


def test_pse_no_shear():
    solver = PSE("periodic", "periodic", "periodic")
    solver.setParameters(psi=1.0, Lx=10.0, Ly=10.0, Lz=10.0)
    solver.initialize(hydrodynamicRadius=1.0, viscosity=1.0)


# NBody should use default alg and all particles in one batch if no params set
def test_nbody_no_params():
    solver = NBody("open", "open", "open")
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.5,
    )
    positions = np.random.rand(10, 3)
    solver.setPositions(positions)
    mf, _ = solver.Mdot(positions)


@pytest.mark.parametrize(
    ("NBatch", "NperBatch", "numberParticles"),
    [
        (5, 1, 10),
        (2, -1, 8),
    ],
)
def test_nbody_bad_parameters(NBatch, NperBatch, numberParticles):
    solver = NBody("open", "open", "open")
    solver.setParameters("advise", NBatch, NperBatch)
    with pytest.raises(RuntimeError):
        solver.initialize(
            viscosity=1.0,
            hydrodynamicRadius=1.5,
        )
        positions = np.random.rand(numberParticles, 3)
        solver.setPositions(positions)


def test_nbody_wall_height_parameter():

    solver = NBody("open", "open", "open")
    with pytest.raises(RuntimeError):
        solver.setParameters(
            wallHeight=0.5
        )  # wallHeight is not valid for open periodicity
    solver.setParameters("advise")

    solver = NBody("open", "open", "single_wall")
    with pytest.raises(RuntimeError):
        solver.setParameters(
            "advise"
        )  # single_wall periodicity requires wall height param

    solver.setParameters(wallHeight=0.5)


def test_dpstokes_narrow_channel():
    solver = DPStokes("periodic", "periodic", "two_walls")
    solver.setParameters(Lx=3.0, Ly=3.0, zmin=0.0, zmax=1.0)
    with pytest.raises(RuntimeError, match="DPStokes"):
        solver.initialize(hydrodynamicRadius=1.0, viscosity=1.0)
