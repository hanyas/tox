from jax.config import config

config.update("jax_enable_x64", True)

from jax import vmap
import jax.random as jr
import jax.numpy as jnp
from jax import block_until_ready

from tox.objects import Box
from tox.utils import discretize_dynamics
from tox.solvers import maxent_leqg

import time as clock
import matplotlib.pyplot as plt


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e1, 1e0]))
    c = 0.5 * (state - goal_state).T @ final_state_cost @ (state - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost = jnp.diag(jnp.array([1e0]))

    c = 0.5 * (state - goal_state).T @ state_cost @ (state - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def double_integrator(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    c = jnp.array([0.0, 0.0])
    return A @ state + B @ action + c


simulation_step = 0.01
downsampling = 10
dynamics = discretize_dynamics(
    ode=double_integrator,
    simulation_step=simulation_step,
    downsampling=downsampling,
)

state_dim = 2
action_dim = 1

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

goal_state = jnp.array([10.0, 0.0])
horizon = 50

risk_param = 0
noise_var = 1e-1

start = clock.time()
policy = maxent_leqg.solver(
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    action_space,
    risk_param,  # risk_param
    noise_var,   # noise_var
    goal_state,
    horizon,
)

key = jr.PRNGKey(737)
episode_keys = jr.split(key, 100)

init_state = jnp.array([0.0, 0.0])

episode = vmap(
    maxent_leqg.rollout,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None),
)(
    episode_keys,
    final_cost,
    transient_cost,
    dynamics,
    state_space,
    action_space,
    policy,
    init_state,
    goal_state,
    noise_var,
    horizon,    
)

block_until_ready(episode)
end = clock.time()
print("Compilation + Execution Time:", end - start)

state, action, total_cost = episode

print("state.shape: ", state.shape)
print("action.shape: ", action.shape)
print("total_cost.shape: ", total_cost.shape)
print("total_cost: ", total_cost)

total_cost_mu = jnp.mean(total_cost)
total_cost_std = jnp.std(total_cost)

print("total_cost_mu:",total_cost_mu)
print("total_cost_std:",total_cost_std)

title_text = "Total cost mu: " + str(jnp.round(total_cost_mu))
title_text += " std: " + str(jnp.round(total_cost_std))


fig = plt.figure(figsize=(8, 8))
plt.subplot(4, 1, 1)
plt.plot(state[..., 0].T)
plt.ylabel("x1")
plt.grid()
plt.subplot(4, 1, 2)
plt.plot(state[..., 1].T)
plt.ylabel("x2")
plt.grid()
plt.subplot(4, 1, 3)
plt.plot(action[..., 0].T)
plt.ylabel("u")
plt.xlabel("t")
plt.grid()
plt.subplot(4, 1, 4)
plt.plot(total_cost)
plt.plot(total_cost_mu*jnp.ones(shape=total_cost.shape))
plt.ylabel("Total Cost")
plt.grid()
plt.title(title_text)
plt.tight_layout()
plt.show()
