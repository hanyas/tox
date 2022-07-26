from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.lax import fori_loop
from jax import block_until_ready

from tox.objects import Trajectory
from tox.utils import runge_kutta
from tox.solvers import lqr

import time as clock
import matplotlib.pyplot as plt


simulation_step = 0.01
downsampling = 10

horizon = 50
state_dim = 2
action_dim = 1


def final_cost(state: jnp.ndarray) -> float:
    goal: jnp.ndarray = jnp.array([10.0, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))

    c = (state - goal).T @ final_state_cost @ (state - goal)
    return c * (simulation_step * downsampling)


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> float:

    goal: jnp.ndarray = jnp.array([10.0, 0.0])
    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e0]))

    c = (state - goal).T @ state_cost @ (state - goal)\
        + action.T @ action_cost @ action
    return c * (simulation_step * downsampling)


def double_integrator(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    A: jnp.ndarray = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B: jnp.ndarray = jnp.array([[0.0], [1.0]])
    c: jnp.ndarray = jnp.array([0.0, 0.0])
    return A @ state + B @ action + c


def dynamics(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    def _step(t, state):
        next_state = runge_kutta(
            state,
            action,
            time + t * simulation_step,
            double_integrator,
            simulation_step,
        )
        return next_state

    return fori_loop(
        lower=0,
        upper=downsampling,
        body_fun=_step,
        init_val=state,
    )


reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

start = clock.time()
policy = lqr.solver(
    final_cost, transient_cost, dynamics, reference
)

init_state = jnp.array([0.0, 0.0])
episode = lqr.rollout(
    final_cost, transient_cost, dynamics, init_state, policy, reference
)
block_until_ready(episode)
end = clock.time()
print("Compilation + Execution Time:", end - start)

state, action, total_cost = episode

plt.subplot(3, 1, 1)
plt.plot(state[:, 0].T)
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1].T)
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[:, 1].T)
plt.ylabel("u")
plt.xlabel("t")
plt.show()
