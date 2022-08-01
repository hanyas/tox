from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.lax import fori_loop

from tox.objects import Trajectory, Box
from tox.utils import runge_kutta
from tox.solvers import ilqr

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
    return A @ state + B @ action


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


init_reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

init_policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=jnp.zeros((horizon, action_dim)),
)

init_state = jnp.array([0., 0.])

options = ilqr.Hyperparameters()

start = clock.time()
policy, reference, trace = ilqr.py_solver(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    init_policy,
    action_space,
    init_reference,
    init_state,
    options,
)

end = clock.time()
print("Compilation + Execution Time:", end - start)

episode = ilqr.rollout(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    policy,
    action_space,
    reference,
    init_state,
)

state, action, total_cost = episode

plt.subplot(3, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("x1")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("x2")
plt.subplot(3, 1, 3)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
