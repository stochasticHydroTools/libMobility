import pytest
from libMobility import *
import numpy as np
import importlib

from utils import (
    initialize_solver,
    solver_configs_all,
)


def setup_inputs(framework, use_torques, numberParticles, precision):
    try:
        # import the framework using the string name
        importlib.import_module(framework)
    except ImportError:
        pytest.skip(f"{framework} not available")

    positions = np.random.rand(numberParticles, 3).astype(precision)
    forces = np.random.rand(numberParticles, 3).astype(precision)
    torques = (
        np.random.rand(numberParticles, 3).astype(precision) if use_torques else None
    )
    if framework == "torch":
        import torch

        positions = torch.from_numpy(positions)
        forces = torch.from_numpy(forces)
        torques = torch.from_numpy(torques) if use_torques else None
        if torch.cuda.is_available():
            positions = positions.cuda()
            forces = forces.cuda()
            torques = torques.cuda() if use_torques else None
    elif framework == "cupy":
        import cupy as cp

        # skip if CUDA is not available
        if not cp.cuda.is_available():
            pytest.skip("CUDA not available")
        positions = cp.random.rand(numberParticles, 3).astype(precision)
        forces = cp.random.rand(numberParticles, 3).astype(precision)
        torques = (
            cp.random.rand(numberParticles, 3).astype(precision)
            if use_torques
            else None
        )
    elif framework == "jax":
        import jax

        device = jax.devices("cpu")[0]
        positions = jax.device_put(positions, device)
        forces = jax.device_put(forces, device)
        torques = jax.device_put(torques, device) if use_torques else None
    elif framework == "tensorflow":
        import tensorflow as tf

        positions = tf.convert_to_tensor(positions)
        forces = tf.convert_to_tensor(forces)
        torques = tf.convert_to_tensor(torques) if use_torques else None

    return positions, forces, torques


@pytest.mark.parametrize(("Solver", "periodicity"), solver_configs_all)
@pytest.mark.parametrize("framework", ["numpy", "torch", "cupy", "jax", "tensorflow"])
@pytest.mark.parametrize("use_torques", [False, True])
@pytest.mark.parametrize("method", ["Mdot", "thermalDrift", "sqrtMdotW", "hydrodynamicVelocities"])
def test_framework(Solver, periodicity, framework, use_torques, method):
    if use_torques and Solver.__name__ in "PSE":
        pytest.skip("PSE does not support torques")
    numberParticles = 10
    solver = initialize_solver(Solver, periodicity, use_torques)
    # Set precision to be the same as compiled precision
    precision = np.float32 if Solver.precision == "float" else np.float64
    positions, forces, torques = setup_inputs(
        framework, use_torques, numberParticles, precision
    )
    solver.setPositions(positions)
    if method == "thermalDrift":
        mf, mt = solver.thermalDrift()
    elif method == "Mdot":
        mf, mt = solver.Mdot(forces, torques)
    elif method == "sqrtMdotW":
        mf, mt = solver.sqrtMdotW()
    elif method == "hydrodynamicVelocities":
        mf, mt = solver.hydrodynamicVelocities(forces, torques)
    else:
        raise ValueError(f"Unknown method: {method}")
    assert type(mf) == type(positions)
    if use_torques:
        assert type(mt) == type(positions)
    # Assert device origin is equal to the input
    if framework != "numpy":
        assert mf.device == positions.device
        if use_torques:
            assert mt.device == positions.device
    del mf
