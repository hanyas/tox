from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)

import jax.numpy as jnp
import jax.random as jr
from jax import block_until_ready

from tox.objects import Trajectory, Box
from tox.utils import discretize_dynamics, wrap_angle
from tox.solvers import ilqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += action.T @ action_cost @ action
    return c


def cartpole(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity: float = 9.81
    pole_length: float = 0.5
    cart_mass: float = 10.0
    pole_mass: float = 1.0
    total_mass: float = cart_mass + pole_mass

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


simulation_step = 0.04
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

init_state: jnp.ndarray = jnp.array([0.01, wrap_angle(-0.01), 0.01, -0.01])
goal_state: jnp.ndarray = jnp.array([0.0, jnp.pi, 0.0, 0.0])

nb_steps = 100
horizon = 15

key = jr.PRNGKey(747)

key, policy_key = jr.split(key, 2)
policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-1 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

options = ilqr.Hyperparameters(max_iter=100)

start = clock.time()
state, action = ilqr.mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    reference,
    options,
    nb_steps,
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
policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-1 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

start = clock.time()
state, action = ilqr.mpc_rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    reference,
    options,
    nb_steps,
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
