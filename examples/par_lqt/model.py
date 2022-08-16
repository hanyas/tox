from functools import partial
from typing import Callable

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tox.objects import Trajectory, Box

simulation_step = 0.01
downsampling = 1
horizon = 50
state_dim = 2
action_dim = 1


def _final_cost(state: jnp.ndarray) -> float:
    goal: jnp.ndarray = jnp.array([10.0, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))

    c = (state - goal).T @ final_state_cost @ (state - goal)
    return c * (simulation_step * downsampling)


def _transient_cost(
        state: jnp.ndarray, action: jnp.ndarray
) -> float:
    goal: jnp.ndarray = jnp.array([10.0, 0.0])
    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e0]))

    c = (state - goal).T @ state_cost @ (state - goal)
    c += action.T @ action_cost @ action
    return c * (simulation_step * downsampling)


def par_double_integrator(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    A: jnp.ndarray = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B: jnp.ndarray = jnp.array([[0.0], [1.0]])

    return A @ state + B @ action


def par_runge_kutta(
        state: jnp.ndarray,
        action: jnp.ndarray,
        ode: Callable,
        step: float,
):
    k1 = ode(state, action)
    k2 = ode(state + 0.5 * step * k1, action)
    k3 = ode(state + 0.5 * step * k2, action)
    k4 = ode(state + step * k3, action)
    return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _dynamics(state: jnp.ndarray, action: jnp.ndarray):
    return par_runge_kutta(state, action, par_double_integrator, simulation_step)


_state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)
_action_space: Box = Box(
    low=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(action_dim,),
)
_reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)


def model_parameters():
    final_cost = partial(_final_cost)
    transient_cost = partial(_transient_cost)
    dynamics = partial(_dynamics)
    state_space = partial(_state_space)
    action_space = partial(_action_space)
    reference = _reference

    return final_cost, transient_cost, dynamics, state_space, action_space, reference
