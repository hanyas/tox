from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from jax import jacobian as jac
from jax import block_until_ready

from jax.lax import fori_loop
from jax.lax import stop_gradient

from tox.spaces import Box

from tox.objects import Trajectory
from tox.utils import runge_kutta
from tox.solvers import ilqr

import time as clock
import matplotlib.pyplot as plt


simulation_step = 0.01
downsampling = 5

horizon = 100
state_dim = 2
action_dim = 1


def cost_features(state):
    return jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])


# I don't know why it works better with this :/
def linear_cost_approx(state: jnp.ndarray) -> float:
    def _features_jacobian(state):
        J = jac(cost_features, 0)
        j = cost_features(state) - J(state) @ state
        return J, j

    J, j = _features_jacobian(stop_gradient(state))
    return J(stop_gradient(state)) @ state + j


def final_cost(state: jnp.ndarray) -> float:
    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0])
    final_state_cost: jnp.ndarray = jnp.diag(
        jnp.array([1e1, 1e1, 1e-1])
    )  # in feature space

    state_feat_approx = linear_cost_approx(state)
    goal_feat = cost_features(goal)
    c = (
        (state_feat_approx - goal_feat).T
        @ final_state_cost
        @ (state_feat_approx - goal_feat)
    )
    return c * (simulation_step * downsampling)


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> float:

    goal: jnp.ndarray = jnp.array([jnp.pi, 0.0])
    state_cost: jnp.ndarray = jnp.diag(
        jnp.array([1e1, 1e1, 1e-1])
    )  # in feature space
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e-3]))

    state_feat_approx = linear_cost_approx(state)
    goal_feat = cost_features(goal)
    c = (state_feat_approx - goal_feat).T @ state_cost @ (
        state_feat_approx - goal_feat
    ) + action.T @ action_cost @ action
    return c * (simulation_step * downsampling)


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


# limits
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


init_reference = Trajectory(
    state=jnp.zeros((horizon + 1, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

key = jr.PRNGKey(137)

init_policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(key, shape=(horizon, action_dim)),
)

init_state = jnp.array([0.0, 0.0])

options = ilqr.Hyperparameters()

start = clock.time()
policy, reference, trace = ilqr.solver(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    init_policy,
    action_space,
    init_reference,
    options,
    init_state,
)

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
