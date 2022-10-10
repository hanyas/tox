from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from tox.objects import Box
from tox.utils import discretize_dynamics
from tox.solvers import ddp as ddp

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e0, 1e0]))
    c = 0.5 * (state - goal_state).T @ final_state_cost @ (state - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e0, 1e0]))
    action_cost = jnp.diag(jnp.array([0.1, 0.1]))

    c = 0.5 * (state - goal_state).T @ state_cost @ (state - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def car(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    length = 0.1
    return jnp.hstack(
        [
            state[3] * jnp.cos(state[2]),
            state[3] * jnp.sin(state[2]),
            state[3] * jnp.tan(action[1]) / length,
            action[0],
        ]
    )


simulation_step = 0.01
downsampling = 5
dynamics = discretize_dynamics(
    ode=car, simulation_step=simulation_step, downsampling=downsampling
)

state_dim = 4  # x, y, r, v
action_dim = 2

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=jnp.array([-10.0, -2.0 * jnp.pi]),
    high=jnp.array([10.0, 2.0 * jnp.pi]),
    shape=(action_dim,),
)

init_state = jnp.array([5.0, 5.0, jnp.pi/3.0, 0.0])
goal_state = jnp.array([0.0, 0.0, 0.0, 0.0])

nb_steps = 75
horizon = 15

key = jr.PRNGKey(1337)
control = 1e-4 * jr.normal(key, shape=(horizon, action_dim))

options = ddp.Hyperparameters(max_iter=100)

start = clock.time()
state, action, _ = ddp.exact_mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    control,
    action_space,
    horizon,
    nb_steps,
    options,
)
end = clock.time()
print("Compilation + Execution Time:", end - start)

plt.subplot(6, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("x")
plt.subplot(6, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("y")
plt.subplot(6, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("r")
plt.subplot(6, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("v")
plt.subplot(6, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u1")
plt.subplot(6, 1, 6)
plt.plot(action[:, 1])
plt.ylabel("u2")
plt.xlabel("t")
plt.show()
