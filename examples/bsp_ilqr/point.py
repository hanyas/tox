from jax.config import config
config.update("jax_enable_x64", True)

from jax import vmap
import jax.numpy as jnp
import jax.random as jr

from tox.objects import Box
from tox.utils import unflatten_gaussian, flatten_gaussian

from tox.filtering import sqrt_kalman_filter_dynamics
from tox.filtering import sqrt_kalman_filter

from tox.solvers import bsp_ilqr as bsp_ilqr

import matplotlib.pyplot as plt


state_dim = 1
belief_dim = state_dim + int(state_dim + state_dim * (state_dim - 1) / 2)
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


def dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    delta: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    simulation_step = 0.1
    return state + simulation_step * action + 1e-2 * jnp.eye(1) @ delta


def observation(
    state: jnp.ndarray,
    eta: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    beacon = jnp.array([5.0])
    return (
        state
        + (
            1e-2 * jnp.eye(1)
            + jnp.linalg.cholesky(0.5 * (beacon - state) ** 2 * jnp.eye(1))
        )
        @ eta
    )


def unflatten_belief(state):
    return unflatten_gaussian(state, state_dim)


def flatten_belief(mu, chol):
    return flatten_gaussian(mu, chol, state_dim)


def final_belief_cost(
    belief: jnp.ndarray,
    goal_state: jnp.ndarray,
) -> float:

    bel_mu, bel_chol = unflatten_belief(belief)

    final_mean_cost = jnp.diag(jnp.array([10.0]))
    final_covariance_cost = jnp.diag(jnp.array([100.0]))

    c = 0.5 * (bel_mu - goal_state).T @ final_mean_cost @ (bel_mu - goal_state)
    c += 0.5 * jnp.trace(final_covariance_cost @ (bel_chol @ bel_chol.T))
    return c


def transient_belief_cost(
    belief: jnp.ndarray,
    action: jnp.ndarray,
    time: int,
    goal_state: jnp.ndarray,
) -> float:

    bel_mu, bel_chol = unflatten_belief(belief)

    mean_cost = jnp.diag(jnp.array([0.0]))
    covariance_cost = jnp.diag(jnp.array([10.0]))
    action_cost = jnp.diag(jnp.array([0.5]))

    c = 0.5 * (bel_mu - goal_state).T @ mean_cost @ (bel_mu - goal_state)
    c += 0.5 * jnp.trace(covariance_cost @ (bel_chol @ bel_chol.T))
    c += 0.5 * action.T @ action_cost @ action
    return c


belief_dynamics = sqrt_kalman_filter_dynamics(
    dynamics,
    state_space,
    observation,
    observation_space,
    flatten_belief,
    unflatten_belief,
)

horizon = 100

init_mu = jnp.array([-5.0])
init_chol = jnp.eye(state_dim) * 1.0

init_belief = flatten_belief(init_mu, init_chol)
goal_state = jnp.array([0.0])

key = jr.PRNGKey(1337)
key, control_key = jr.split(key, 2)
control = 1e-4 * jr.normal(control_key, shape=(horizon, action_dim))

options = bsp_ilqr.Hyperparameters()

trajectory, control, trace = bsp_ilqr.jax_solver(
    final_belief_cost,
    transient_belief_cost,
    goal_state,
    belief_dynamics,
    init_belief,
    control,
    action_space,
    horizon,
    options,
)

bel_mu, bel_chol = vmap(unflatten_belief, in_axes=(0,))(trajectory)
bel_cov = jnp.einsum("nkh,ndl->nkd", bel_chol, bel_chol)

plt.subplot(3, 1, 1)
plt.plot(bel_mu[:, 0])
plt.ylabel("m")
plt.subplot(3, 1, 2)
plt.plot(bel_cov[:, 0, 0])
plt.ylabel("s")
plt.subplot(3, 1, 3)
plt.plot(control)
plt.ylabel("u")
plt.xlabel("t")
plt.show()

bayes_filter = sqrt_kalman_filter(
    dynamics,
    state_space,
    observation,
    observation_space,
    flatten_belief,
    unflatten_belief,
)

state_key, key = jr.split(key, 2)
init_state = jr.multivariate_normal(state_key, mean=init_mu, cov=init_chol @ init_chol.T)

state, belief, action, cost = bsp_ilqr.approximate_closed_loop_rollout(
    final_belief_cost,
    transient_belief_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    observation,
    observation_space,
    belief_dynamics,
    init_belief,
    bayes_filter,
    control,
    action_space,
    horizon,
    options,
    key,
)

bel_mu, bel_chol = vmap(unflatten_belief, in_axes=(0,))(belief)
bel_cov = jnp.einsum("nkh,ndl->nkd", bel_chol, bel_chol)

plt.subplot(3, 1, 1)
plt.plot(state[:, 0])
plt.plot(bel_mu[:, 0], color='r')
plt.ylabel("x/m")
plt.subplot(3, 1, 2)
plt.plot(bel_cov[:, 0, 0])
plt.ylabel("s")
plt.subplot(3, 1, 3)
plt.plot(action[:, 0])
plt.ylabel("u")
plt.xlabel("t")
plt.show()
