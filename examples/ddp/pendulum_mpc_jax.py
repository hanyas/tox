from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jax import block_until_ready

from tox.objects import Box
from tox.utils import discretize_dynamics, wrap_angle
from tox.solvers import ddp as ddp

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e0, 1e-1]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def pendulum(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            - gravity / length * jnp.sin(position)
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
    low=-5. * jnp.ones((action_dim,)),
    high=5. * jnp.ones((action_dim,)),
    shape=(action_dim,),
)

init_state = jnp.array([wrap_angle(0.01), -0.01])
goal_state = jnp.array([jnp.pi, 0.0])

nb_steps = 100
horizon = 25

key = jr.PRNGKey(1337)

key, policy_key = jr.split(key, 2)
control = 1e-2 * jr.normal(policy_key, shape=(horizon, action_dim))

options = ddp.Hyperparameters(max_iter=50)

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
block_until_ready(state)
end = clock.time()
print("Compilation + Execution Time:", end - start)

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


init_state = jnp.array([wrap_angle(-0.01), 0.01])

key, policy_key = jr.split(key, 2)
control = 1e-2 * jr.normal(policy_key, shape=(horizon, action_dim))

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
block_until_ready(state)
end = clock.time()
print("Execution Time:", end - start)

plt.subplot(3, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("q")
plt.subplot(3, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("dq")
plt.subplot(3, 1, 3)
plt.plot(action[:, 1])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
