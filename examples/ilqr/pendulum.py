from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)

import jax.numpy as jnp
import jax.random as jr

from tox.objects import Trajectory, Box
from tox.utils import discretize_dynamics, wrap_angle
from tox.solvers import ilqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e-1]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += action.T @ action_cost @ action
    return c


def pendulum(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    gravity: float = 9.81
    length: float = 1.0
    mass: float = 1.0
    damping: float = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )


simulation_step = 0.01
downsampling = 5
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=simulation_step, downsampling=downsampling
)

state_dim = 2
action_dim = 1

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=-5.0 * jnp.ones((action_dim,)),
    high=5.0 * jnp.ones((action_dim,)),
    shape=(action_dim,),
)

init_state = jnp.array([wrap_angle(0.01), 0.0])
goal_state = jnp.array([jnp.pi, 0.0])

horizon = 100

key = jr.PRNGKey(1337)
init_policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(key, shape=(horizon, action_dim)),
)

init_reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

options = ilqr.Hyperparameters()

start = clock.time()
policy, reference, _ = ilqr.py_solver(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    init_policy,
    action_space,
    init_reference,
    options,
)
end = clock.time()
print("Compilation + Execution Time:", end - start)

episode = ilqr.rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    reference,
)

state, action, total_cost = episode

plt.subplot(3, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("q")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("dq")
plt.subplot(3, 1, 3)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
