from jax.config import config
config.update("jax_enable_x64", True)

from jax import jacobian as jac
import jax.numpy as jnp
from jax import block_until_ready

from tox.objects import Box
from tox.utils import discretize_dynamics, symmetrize
from tox.solvers import lqr


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))

    c = 0.5 * (state - goal_state).T @ final_state_cost @ (state - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, time: int, goal_state: jnp.ndarray
) -> float:

    state_cost: jnp.ndarray = jnp.diag(jnp.array([1e1, 1e0]))
    action_cost: jnp.ndarray = jnp.diag(jnp.array([1e0]))

    c = 0.5 * (state - goal_state).T @ state_cost @ (state - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def double_integrator(
    state: jnp.ndarray, action: jnp.ndarray, time: int
) -> jnp.ndarray:
    A: jnp.ndarray = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B: jnp.ndarray = jnp.array([[0.0], [1.0]])
    c: jnp.ndarray = jnp.array([0.0, 0.0])
    return A @ state + B @ action + c


simulation_step = 0.01
downsampling = 10
dynamics = discretize_dynamics(
    ode=double_integrator, simulation_step=simulation_step, downsampling=downsampling
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

goal_state: jnp.ndarray = jnp.array([10.0, 0.0])
horizon = 50

policy = lqr.solver(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    state_space,
    action_space,
    horizon
)

init_state: jnp.ndarray = jnp.array([0.0, 0.0])
state, action, total_cost = lqr.rollout(
    final_cost,
    transient_cost,
    goal_state,
    dynamics,
    init_state,
    state_space,
    policy,
    action_space,
    horizon
)


def observation_mu(goal_state, state, time):
    policy = lqr.solver(final_cost,
                        transient_cost,
                        goal_state,
                        dynamics,
                        state_space,
                        action_space,
                        horizon)

    action = action_space.clip(policy(state, time))
    next_state = state_space.clip(dynamics(state, action, time))
    return jnp.hstack((next_state, action))


observation_cov = jnp.eye(state_dim + action_dim) * 1e-2

bel_mu = jnp.array([0., 0.])
bel_cov = jnp.eye(2)

for t in range(1, horizon):
    obs = jnp.hstack((state[t], action[t-1]))

    obs_jac = jac(observation_mu, 0)(bel_mu, state[t - 1], t - 1)
    obs_cov = observation_cov

    K = bel_cov @ obs_jac.T @ jnp.linalg.inv(obs_jac @ bel_cov @ obs_jac.T + obs_cov)

    bel_mu = bel_mu + K @ (obs - observation_mu(bel_mu, state[t - 1], t - 1))
    bel_cov = symmetrize(bel_cov - K @ (obs_jac @ bel_cov @ obs_jac.T + obs_cov) @ K.T)

block_until_ready(bel_mu)
