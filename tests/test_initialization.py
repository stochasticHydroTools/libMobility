import pytest

from libMobility import *
from utils import sane_parameters, solver_configs_all

# List of classes inside libMobility

solver_list = [PSE, NBody, DPStokes, SelfMobility]


@pytest.mark.parametrize(
    ("Solver", "periodicity"), solver_configs_all
)
def test_periodic_initialization(Solver, periodicity):
    solver = Solver(*periodicity)

def test_pse_torques_unsupported():
    hydrodynamicRadius = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    parameters = sane_parameters[PSE.__name__]
    solver.setParameters(**parameters)
    numberParticles = 1
    with pytest.raises(RuntimeError):
        solver.initialize(
            temperature=1.0,
            viscosity=1.0,
            hydrodynamicRadius=hydrodynamicRadius,
            numberParticles=numberParticles,
            needsTorque=True
        )

@pytest.mark.parametrize("Solver", solver_list)
def test_invalid_throws(Solver):
    with pytest.raises(RuntimeError):
        solver = Solver("periodicasdas", "periodic", "open")

def test_dpstokes_invalid_box():
    with pytest.raises(RuntimeError):
        Lx, Ly = 10, 20
        solver = DPStokes("periodic", "periodic", "single_wall")
        params = {"dt": 1, "Lx": Lx, "Ly": Ly, "zmin": 0, "zmax": 19.2}
        solver.setParameters(**params)

@pytest.mark.parametrize(("NBatch", "NperBatch"),
                         [
                            (1, 12),
                            (12, 1),
                         ])
def test_nbody_good_batch_parameters(NBatch, NperBatch):
    solver = NBody("open", "open", "open")
    solver.setParameters("advise", NBatch, NperBatch)
    numberParticles = NBatch*NperBatch
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=1.5,
        numberParticles=numberParticles)
    
def test_nbody_default_parameters():
    solver = NBody("open", "open", "open")
    solver.setParameters("advise")
    numberParticles = 10
    solver.initialize(
        temperature=1.0,
        viscosity=1.0,
        hydrodynamicRadius=1.5,
        numberParticles=numberParticles)
    
@pytest.mark.parametrize(("NBatch", "NperBatch", "numberParticles"),
                         [
                            (5, 1, 10),
                            (2, -1, 8),
                         ])
def test_nbody_bad_parameters(NBatch, NperBatch, numberParticles):
    solver = NBody("open", "open", "open")
    solver.setParameters("advise", NBatch, NperBatch)
    with pytest.raises(RuntimeError):
        solver.initialize(
            temperature=1.0,
            viscosity=1.0,
            hydrodynamicRadius=1.5,
            numberParticles=numberParticles)