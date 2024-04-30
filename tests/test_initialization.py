import pytest

from libMobility import *

# List of classes inside libMobility

solver_list = [PSE, NBody, DPStokes, SelfMobility]


@pytest.mark.parametrize(
    ("Solver", "periodicity"),
    [
        (PSE, ("periodic", "periodic", "periodic")),
        (NBody, ("open", "open", "open")),
        (DPStokes, ("periodic", "periodic", "open")),
        (DPStokes, ("periodic", "periodic", "single_wall")),
        (DPStokes, ("periodic", "periodic", "two_walls")),
    ],
)
def test_periodic_initialization(Solver, periodicity):
    solver = Solver(*periodicity)


@pytest.mark.parametrize("Solver", solver_list)
def test_invalid_throws(Solver):
    with pytest.raises(RuntimeError):
        solver = Solver("periodicasdas", "periodic", "open")

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