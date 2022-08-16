from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from tox.objects import BeliefTrajectory, Box
from tox.solvers import bsp_ilqr

import time as clock
import matplotlib.pyplot as plt


def final_cost(
    bel_mu: jnp.ndarray, bel_cov: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    final_mean_cost: jnp.ndarray = jnp.diag(jnp.array([10.0]))
    final_covariance_cost: jnp.ndarray = jnp.diag(jnp.array([100.0]))

    c = 0.5 * (bel_mu - goal_state).T @ final_mean_cost @ (bel_mu - goal_state)
    c += jnp.trace(final_covariance_cost @ bel_cov)
    return c


def transient_cost(
    bel_mu: jnp.ndarray,
    bel_cov: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
    goal_state: jnp.ndarray,
) -> float:

    mean_cost: jnp.ndarray = jnp.diag(jnp.array([0.0]))
    covariance_cost: jnp.ndarray = jnp.diag(jnp.array([10.0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([0.5]))

    c = 0.5 * (bel_mu - goal_state).T @ mean_cost @ (bel_mu - goal_state)
    c += jnp.trace(covariance_cost @ bel_cov)
    c += 0.5 * action.T @ action_cost @ action
    return c


def dynamics_mean(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    simulation_step = 0.1
    return state + simulation_step * action


def dynamics_noise(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    return jnp.eye(state_dim) * 0.0


def observation_mean(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    return state


def observation_noise(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    beacon: jnp.ndarray = jnp.array([5.0])
    return 0.5 * (beacon - state) ** 2 * jnp.eye(observation_dim)


state_dim = 1
observation_dim = 1
action_dim = 1

state_space: Box = Box(
    low=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((state_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(state_dim,),
)

observation_space: Box = Box(
    low=jnp.ones((observation_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((observation_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(observation_dim,),
)

action_space: Box = Box(
    low=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(action_dim,),
)

init_mu = jnp.array([-5.0])
init_cov = jnp.eye(state_dim) * 5.0

goal_state: jnp.ndarray = jnp.array([0.0])

horizon = 100

key = jr.PRNGKey(1137)
init_policy = bsp_ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, state_dim)),
    kff=1e-2 * jr.normal(key, shape=(horizon, action_dim)),
)

init_reference = BeliefTrajectory(
    bel_mu=jnp.zeros((horizon + 1, state_dim)),
    bel_cov=jnp.zeros((horizon + 1, state_dim, state_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

options = bsp_ilqr.Hyperparameters()

start = clock.time()
policy, reference, trace = bsp_ilqr.py_solver(
    final_cost,
    transient_cost,
    goal_state,
    dynamics_mean,
    dynamics_noise,
    init_mu,
    init_cov,
    state_space,
    observation_mean,
    observation_noise,
    observation_space,
    init_policy,
    action_space,
    init_reference,
    options,
)
end = clock.time()
print("Compilation + Execution Time:", end - start)

start = clock.time()
policy, reference = bsp_ilqr.jax_solver(
    final_cost,
    transient_cost,
    goal_state,
    dynamics_mean,
    dynamics_noise,
    init_mu,
    init_cov,
    state_space,
    observation_mean,
    observation_noise,
    observation_space,
    init_policy,
    action_space,
    init_reference,
    options,
)
end = clock.time()
print("Compilation + Execution Time:", end - start)

plt.subplot(3, 1, 1)
plt.plot(reference.bel_mu[:, 0])
plt.ylabel("x")
plt.subplot(3, 1, 2)
plt.plot(reference.bel_cov[:, 0, 0])
plt.ylabel("s")
plt.subplot(3, 1, 3)
plt.plot(reference.action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
