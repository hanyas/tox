from typing import NamedTuple, Callable
from functools import partial

import jax.random
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsc

from jax import jit
from jax.lax import scan

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearGaussianDynamics,
    Trajectory,
    Box,
)

from tox.approximations import (
    quadratize_final_cost,
    quadratize_transient_cost,
    linearize_dynamics,
)

from tox.utils import symmetrize


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(
        self,
        state: jnp.ndarray,
        time: int,
    ) -> jnp.ndarray:
        return self.K[time] @ state + self.kff[time]


def _backward_pass(
        quadratic_final_cost: QuadraticFinalCost,
        quadratic_transient_cost: QuadraticTransientCost,
        linear_dynamics: LinearGaussianDynamics,
        risk_param: float
) -> LinearPolicy:

    fCxx, fcx, _ = quadratic_final_cost
    Cxx, Cuu, Cxu, cx, cu, _ = quadratic_transient_cost
    A, B, c, sigma = linear_dynamics

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B, c, sigma = params

        Vxx, vx = carry

        rVxx = jnp.linalg.inv(jnp.linalg.inv(Vxx) - risk_param * sigma)

        Qxx = Cxx + A.T @ rVxx @ A
        Quu = Cuu + B.T @ rVxx @ B
        Qux = (Cxu + A.T @ rVxx @ B).T

        qx = cx + A.T @ rVxx @ c + A.T @ vx
        qu = cu + B.T @ rVxx @ c + B.T @ vx

        K = -jsc.linalg.solve(Quu, Qux, assume_a='pos')
        kff = -jsc.linalg.solve(Quu, qu, assume_a='pos')

        Vxx = symmetrize(Qxx + Qux.T @ K)
        vx = qx + Qux.T @ kff

        return (Vxx, vx), (K, kff)

    K, kff = scan(
        f=_backwards,
        init=(fCxx, fcx),
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B, c, sigma),
        reverse=True,
    )[1]

    return LinearPolicy(K, kff)


@partial(jit, static_argnums=(0, 1, 2, 3, 4, -1))
def solver(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    action_space: Box,
    risk_param: float,
    goal_state: jnp.ndarray,
    horizon: int,
) -> LinearPolicy:

    time = jnp.linspace(0, horizon, horizon + 1)

    # reference needed only for automatic Taylor expansions
    reference = Trajectory(
        state=jnp.tile(jnp.zeros(state_space.shape), (horizon + 1, 1)),
        action=jnp.tile(jnp.zeros(action_space.shape), (horizon, 1)),
    )

    quadratic_final_cost = quadratize_final_cost(
        final_cost,
        goal_state,
        reference.final,
    )
    quadratic_transient_cost = quadratize_transient_cost(
        transient_cost, goal_state, reference.transient, time[:-1]
    )
    linear_dynamics = linearize_dynamics(
        dynamics, state_space, reference.transient, time[:-1]
    )

    A, b, c = linear_dynamics

    Q = jnp.tile(jnp.eye(state_space.shape[0]) * 1e-2, (A.shape[0], 1, 1))

    linear_stoch_dynamics = LinearGaussianDynamics(A, b, c, Q)

    return _backward_pass(
        quadratic_final_cost, quadratic_transient_cost, linear_stoch_dynamics, risk_param
    )


@partial(jit, static_argnums=(1, 2, 3, 4, 5, -1))
def rollout(
    key: jr.PRNGKey,
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    action_space: Box,
    policy: LinearPolicy,
    init_state: jnp.ndarray,
    goal_state: jnp.ndarray,
    horizon: int,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def episode(carry, time):
        key, state = carry
        action = policy(state, time)
        cost = transient_cost(state, action, time, goal_state)

        key, sub_key = jr.split(key, 2)
        rv = 1e-1 * jax.random.normal(sub_key, shape=state_space.shape)
        next_state = dynamics(state, action, time) + rv
        return (key, next_state), (next_state, action, cost)

    next_state, action, cost = scan(
        f=episode,
        init=(key, init_state),
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    total_cost = jnp.sum(cost) + final_cost(state[-1], goal_state)
    return state, action, total_cost
