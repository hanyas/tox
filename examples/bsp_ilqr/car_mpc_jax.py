from jax.config import config
config.update("jax_enable_x64", True)

from jax import vmap
import jax.numpy as jnp
import jax.random as jr

from tox.objects import Box
from tox.utils import discretize_dynamics
from tox.utils import unflatten_gaussian, flatten_gaussian

from tox.filtering import sqrt_kalman_filter_dynamics
from tox.filtering import sqrt_kalman_filter

from tox.solvers import bsp_ilqr as bsp_ilqr

from time import time as clock
import matplotlib.pyplot as plt


state_dim = 4
belief_dim = state_dim + int(state_dim + state_dim * (state_dim - 1) / 2)
observation_dim = 2
action_dim = 2

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
    low=jnp.array([-10.0, -2.0 * jnp.pi]),
    high=jnp.array([10.0, 2.0 * jnp.pi]),
    shape=(action_dim,),
)


def car(state: jnp.ndarray, action: jnp.ndarray, time: int) -> jnp.ndarray:
    length = 0.1
    return jnp.hstack(
        [
            state[3] * jnp.cos(state[2]),
            state[3] * jnp.sin(state[2]),
            state[3] * jnp.tan(action[1]) / length,
            action[0],
        ]
    )


simulation_step = 0.1
downsampling = 1
dynamics_mean = discretize_dynamics(
        ode=car, simulation_step=simulation_step, downsampling=downsampling
    )


def dynamics(
    state: jnp.ndarray,
    action: jnp.ndarray,
    delta: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    return dynamics_mean(state, action, time) + 1e-2 * jnp.eye(4) @ delta


def observation(
    state: jnp.ndarray,
    eta: jnp.ndarray,
    time: int,
) -> jnp.ndarray:
    beacon = jnp.array([-5.0, 5.0])
    return (
        state[:2]
        + (
            jnp.linalg.cholesky(
                (1e-4 + 0.1 * (beacon - state[:2]).T @ (beacon - state[:2]))
                * jnp.eye(2)
            )
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

    final_mean_cost = jnp.diag(jnp.array([1e2, 1e2, 1e2, 1e2]))
    final_covariance_cost = jnp.diag(jnp.array([1e2, 1e2, 1e2, 1e2]))

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

    mean_cost = jnp.diag(jnp.array([1e0, 1e0, 1e0, 1e0]))
    covariance_cost = jnp.diag(jnp.array([1e2, 1e2, 1e2, 1e2]))
    action_cost = jnp.diag(jnp.array([0.1, 0.1]))

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

bayes_filter = sqrt_kalman_filter(
    dynamics,
    state_space,
    observation,
    observation_space,
    flatten_belief,
    unflatten_belief,
)

nb_steps = 100
horizon = 50

init_mu = jnp.array([5.0, 5.0, jnp.pi, 0.0])
init_chol = jnp.eye(state_dim) * 1.0

init_belief = flatten_belief(init_mu, init_chol)
goal_state = jnp.array([0.0, 0.0, 0.0, 0.0])

key = jr.PRNGKey(1337)
key, control_key = jr.split(key, 2)
control = 1e-4 * jr.normal(control_key, shape=(horizon, action_dim))

key, state_key = jr.split(key, 2)
init_state = jr.multivariate_normal(state_key, mean=init_mu, cov=init_chol @ init_chol.T)

options = bsp_ilqr.Hyperparameters(max_iter=250)

start = clock()
state, belief, action, cost = bsp_ilqr.approximate_mpc_rollout(
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
    nb_steps,
    options,
    key,
)
end = clock()
print(end - start)

bel_mu, bel_chol = vmap(unflatten_belief, in_axes=(0,))(belief)
bel_cov = jnp.einsum("nkh,ndl->nkd", bel_chol, bel_chol)

plt.subplot(6, 1, 1)
plt.plot(state[:, 0])
plt.plot(bel_mu[:, 0], color="r")
plt.ylabel("x")
plt.subplot(6, 1, 2)
plt.plot(state[:, 1])
plt.plot(bel_mu[:, 1], color="r")
plt.ylabel("y")
plt.subplot(6, 1, 3)
plt.plot(state[:, 2])
plt.plot(bel_mu[:, 2], color="r")
plt.ylabel("r")
plt.subplot(6, 1, 4)
plt.plot(state[:, 3])
plt.plot(bel_mu[:, 3], color="r")
plt.ylabel("v")
plt.subplot(6, 1, 5)
plt.plot(action[:, 0])
plt.ylabel("u1")
plt.subplot(6, 1, 6)
plt.plot(action[:, 1])
plt.ylabel("u2")
plt.xlabel("t")
plt.show()
