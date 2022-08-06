from jax.config import config

config.update("jax_enable_x64", True)

from jax import vmap
import jax.random as jr
import jax.numpy as jnp
from jax import block_until_ready

from tox.objects import Box
from tox.utils import discretize_dynamics
from tox.solvers import maxent_lqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))

    c = 0.5 * (state - goal_state).T @ final_state_cost @ (state - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e0]))

    c = 0.5 * (state - goal_state).T @ state_cost @ (state - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def double_integrator(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    A: jnp.ndarray = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B: jnp.ndarray = jnp.array([[0.0], [1.0]])
    return A @ state + B @ action


simulation_step = 0.01
downsampling = 10
dynamics = discretize_dynamics(
    ode=double_integrator,
    simulation_step=simulation_step,
    downsampling=downsampling,
)

state_dim = 2
action_dim = 1

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(action_dim,),
)

goal_state: jnp.ndarray = jnp.array([10.0, 0.0])
horizon = 50

start = clock.time()
policy = maxent_lqr.solver(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    state_space,
    action_space,
    horizon,
)

key = jr.PRNGKey(737)
episode_keys = jr.split(key, 100)

init_state: jnp.ndarray = jnp.array([0.0, 0.0])
episode = vmap(
    maxent_lqr.rollout,
    in_axes=(None, None, None, None, None, None, None, None, None, 0),
)(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    horizon,
    episode_keys,
)
block_until_ready(episode)
end = clock.time()
print("Compilation + Execution Time:", end - start)

state, action, total_cost = episode

plt.subplot(3, 1, 1)
plt.plot(state[..., 0].T)
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[..., 1].T)
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[..., 0].T)
plt.ylabel("u")
plt.xlabel("t")
plt.show()
