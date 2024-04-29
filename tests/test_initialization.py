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
