from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from jax import jacobian as jac

from tox.objects import Trajectory, Box
from tox.utils import symmetrize
from tox.solvers import ilqr

import time as clock
import matplotlib.pyplot as plt


state_dim = 1
observation_dim = 1
belief_dim = 2  # mu + cov
action_dim = 1


def final_cost(belief: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_mean_cost: jnp.ndarray = jnp.diag(jnp.array([10.0]))
    final_covariance_cost: jnp.ndarray = jnp.diag(jnp.array([100.0]))

    mu = belief[:state_dim]
    cov = jnp.reshape(belief[state_dim:], (state_dim, state_dim))

    c = 0.5 * (mu - goal_state).T @ final_mean_cost @ (mu - goal_state)
    c += jnp.trace(final_covariance_cost @ cov)
    return c


def transient_cost(
    belief: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    mean_cost: jnp.ndarray = jnp.diag(jnp.array([0.0]))
    covariance_cost: jnp.ndarray = jnp.diag(jnp.array([10.0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([0.5]))

    mu = belief[:state_dim]
    cov = jnp.reshape(belief[state_dim:], (state_dim, state_dim))

    c = 0.5 * (mu - goal_state).T @ mean_cost @ (mu - goal_state)
    c += jnp.trace(covariance_cost @ cov)
    c += 0.5 * action.T @ action_cost @ action
    return c


def dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    simulation_step = 0.1
    return state + simulation_step * action


def process_noise(
    state: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    return jnp.eye(state_dim) * 0.0


def observation(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    return state


def observation_noise(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    beacon: jnp.ndarray = jnp.array([5.0])
    return 0.5 * (beacon - state)**2 * jnp.eye(observation_dim)


def belief_dynamics(belief: jnp.ndarray,
                    action: jnp.ndarray,
                    time: int) -> jnp.ndarray:

    mu = belief[:state_dim]
    cov = jnp.reshape(belief[state_dim:], (state_dim, state_dim))

    # linearize process and observation
    A = jac(dynamics, 0)(mu, action, time)
    H = jac(observation, 0)(mu, action, time)

    # evaluate state-action-dependent noise
    M = process_noise(mu, action, time)
    N = observation_noise(dynamics(mu, action, time), action, time)

    # extended Kalman filter updates
    G = symmetrize(A @ cov @ A.T) + M
    K = jnp.linalg.solve(H @ G @ H.T + N, H @ G.T).T

    next_mu = dynamics(mu, action, time)
    next_cov = symmetrize(G - K @ H @ G)

    next_belief = jnp.hstack((next_mu, jnp.ravel(next_cov)))
    return next_belief


belief_space: Box = Box(
    low=jnp.ones((belief_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((belief_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(belief_dim,),
)

action_space: Box = Box(
    low=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).min,
    high=jnp.ones((action_dim,)) * jnp.finfo(jnp.float64).max,
    shape=(action_dim,),
)

init_mu = jnp.array([-5.0])
init_cov = jnp.eye(state_dim) * 5.0
init_belief = jnp.hstack((init_mu, jnp.ravel(init_cov)))

goal_state: jnp.ndarray = jnp.array([0.0])

horizon = 100

key = jr.PRNGKey(1337)
init_policy = ilqr.LinearPolicy(
    K=jnp.zeros((horizon, action_dim, belief_dim)),
    kff=1e-2 * jr.normal(key, shape=(horizon, action_dim)),
)

init_reference = Trajectory(
    state=jnp.zeros((horizon + 1, belief_dim)),
    action=jnp.zeros((horizon, action_dim)),
)

options = ilqr.Hyperparameters()

start = clock.time()
policy, reference, trace = ilqr.py_solver(
    final_cost,
    transient_cost,
    goal_state,
    belief_dynamics,
    init_belief,
    belief_space,
    init_policy,
    action_space,
    init_reference,
    options,
)
end = clock.time()
print("Compilation + Execution Time:", end - start)


plt.subplot(3, 1, 1)
plt.plot(reference.state[:, 0])
plt.ylabel("x")
plt.subplot(3, 1, 2)
plt.plot(reference.state[:, 1])
plt.ylabel("s")
plt.subplot(3, 1, 3)
plt.plot(reference.action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
