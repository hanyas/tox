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
    final_state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (
        0.5
        * (_wrapped - goal_state).T
        @ final_state_cost
        @ (_wrapped - goal_state)
    )
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def cartpole(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action
        + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=cartpole, simulation_step=simulation_step, downsampling=downsampling
)

state_dim = 4
action_dim = 1

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=-50.0 * jnp.ones((action_dim,)),
    high=50.0 * jnp.ones((action_dim,)),
    shape=(action_dim,),
)

init_state = jnp.array([0.01, wrap_angle(-0.01), 0.01, -0.01])
goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])

nb_steps = 80
horizon = 15

key = jr.PRNGKey(1337)

key, policy_key = jr.split(key, 2)
control = 1e-2 * jr.normal(policy_key, shape=(horizon, action_dim))

options = ddp.Hyperparameters(max_iter=150)

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

plt.subplot(5, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("x")
plt.subplot(5, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("q")
plt.subplot(5, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("dx")
plt.subplot(5, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("dq")
plt.subplot(5, 1, 5)
plt.plot(action[:, 1])
plt.ylabel("u")
plt.xlabel("t")
plt.show()

init_state = jnp.array([-0.01, wrap_angle(0.01), -0.01, 0.01])

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

plt.subplot(5, 1, 1)
plt.plot(state[:, 0])
plt.ylabel("x")
plt.subplot(5, 1, 2)
plt.plot(state[:, 1])
plt.ylabel("q")
plt.subplot(5, 1, 3)
plt.plot(state[:, 2])
plt.ylabel("dx")
plt.subplot(5, 1, 4)
plt.plot(state[:, 3])
plt.ylabel("dq")
plt.subplot(5, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
