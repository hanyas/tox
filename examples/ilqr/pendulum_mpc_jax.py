from typing import Callable
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)

import jax.numpy as jnp
import jax.random as jr

from jax import jit
from jax import block_until_ready
from jax.lax import fori_loop, scan

from tox.objects import Trajectory, Box
from tox.utils import runge_kutta
from tox.solvers import ilqr
from tox.solvers.ilqr import Hyperparameters, LinearPolicy

import time as clock
import matplotlib.pyplot as plt


simulation_step = 0.01
downsampling = 5

state_dim = 2
action_dim = 1


def wrap_angle(x):
    # wrap angle between [0, 2*pi]
    return x % (2.0 * jnp.pi)


def final_cost(state: jnp.ndarray) -> float:
    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e-1]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = (_wrapped - goal).T @ final_state_cost @ (_wrapped - goal)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> float:

    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0])
    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((wrap_angle(state[0]), state[1]))
    c = (_wrapped - goal).T @ state_cost @ (_wrapped - goal)
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
            -3.0 * gravity / (2.0 * length) * jnp.sin(position)
            + 3.0 * (action - damping * velocity) / (mass * length**2),
        )
    )


state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

action_space: Box = Box(
    low=-2.5 * jnp.ones((action_dim,)),
    high=2.5 * jnp.ones((action_dim,)),
    shape=(action_dim,),
)


def dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    def _step(t, state):
        next_state = runge_kutta(
            state,
            action,
            time + t * simulation_step,
            pendulum,
            simulation_step,
        )
        return next_state

    return fori_loop(
        lower=0,
        upper=downsampling,
        body_fun=_step,
        init_val=state,
    )


@partial(jit, static_argnums=(0, 1, 2, -1))
def mpc_rollout(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    init_state: jnp.ndarray,
    options: Hyperparameters,
    nb_steps: int,
) -> (jnp.ndarray, jnp.ndarray):
    def mpc_step(carry, args):
        policy, reference, state = carry

        next_policy, next_reference = ilqr.jax_solver(
            final_cost,
            transient_cost,
            dynamics,
            state_space,
            policy,
            action_space,
            reference,
            state,
            options,
        )

        action = next_reference.action[0]
        next_state = state_space.clip(dynamics(state, action, 0))

        return (next_policy, next_reference, next_state), (next_state, action)

    _, (state, action) = scan(
        mpc_step, init=(policy, reference, init_state), xs=jnp.arange(nb_steps)
    )

    state = jnp.vstack((init_state, state))
    return state, action

nb_steps = 100
horizon = 25

key = jr.PRNGKey(747)
key, policy_key = jr.split(key, 2)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

init_state = jnp.array([0.01, -0.01])
options = ilqr.Hyperparameters(max_iter=25)

start = clock.time()
state, action = mpc_rollout(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    policy,
    action_space,
    reference,
    init_state,
    options,
    nb_steps,
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
plt.plot(action[:, 1])
plt.ylabel("u")
plt.xlabel("t")
plt.show()


key, policy_key = jr.split(key, 2)

reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(policy_key, shape=(horizon, action_dim)),
)

init_state = jnp.array([-0.01, 0.01])

start = clock.time()
state, action = mpc_rollout(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    policy,
    action_space,
    reference,
    init_state,
    options,
    nb_steps,
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
