from typing import NamedTuple, Callable
from functools import partial

import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsc

from jax import jit
from jax.lax import scan

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
    Trajectory,
    Box,
)

from tox.helpers import (
    quadratize_final_cost,
    quadratize_transient_cost,
    linearize_dynamics,
)

from tox.utils import symmetrize


class StochasticLinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray
    cov: jnp.ndarray

    def __call__(
        self, state: jnp.ndarray, time: int, key: jr.PRNGKey
    ) -> jnp.ndarray:
        return jr.multivariate_normal(
            key, mean=self.mean(state, time), cov=self.cov[time]
        )

    def mean(
        self,
        state: jnp.ndarray,
        time: int,
    ) -> jnp.ndarray:
        return self.K[time] @ state + self.kff[time]


def _backward_pass(
    quadratic_final_cost: QuadraticFinalCost,
    quadratic_transient_cost: QuadraticTransientCost,
    linear_dynamics: LinearDynamics,
) -> StochasticLinearPolicy:

    fCxx, fcx, _ = quadratic_final_cost
    Cxx, Cuu, Cxu, cx, cu, _ = quadratic_transient_cost
    A, B, c = linear_dynamics

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B, c = params

        Vxx, vx = carry

        Qxx = Cxx + A.T @ Vxx @ A
        Quu = Cuu + B.T @ Vxx @ B
        Qux = (Cxu + A.T @ Vxx @ B).T

        qx = cx + A.T @ Vxx @ c + A.T @ vx
        qu = cu + B.T @ Vxx @ c + B.T @ vx

        K = -jsc.linalg.solve(Quu, Qux, assume_a='pos')
        kff = -jsc.linalg.solve(Quu, qu, assume_a='pos')
        cov = jsc.linalg.inv(Quu)

        Vxx = symmetrize(Qxx + Qux.T @ K)
        vx = qx + Qux.T @ kff

        return (Vxx, vx), (K, kff, cov)

    K, kff, cov = scan(
        f=_backwards,
        init=(fCxx, fcx),
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B, c),
        reverse=True,
    )[1]

    return StochasticLinearPolicy(K, kff, cov)


@partial(jit, static_argnums=(0, 1, 3, 4, 5, -1))
def solver(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    state_space: Box,
    action_space: Box,
    horizon: int,
) -> StochasticLinearPolicy:

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

    return _backward_pass(
        quadratic_final_cost, quadratic_transient_cost, linear_dynamics
    )


@partial(jit, static_argnums=(0, 1, 3, 5, 7, 8))
def rollout(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    policy: StochasticLinearPolicy,
    action_space: Box,
    horizon: int,
    key: jr.PRNGKey,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def episode(args, time):
        state, key = args
        policy_key, key = jr.split(key, 2)
        action = action_space.clip(policy(state, time, policy_key))
        cost = transient_cost(state, action, time, goal_state)
        next_state = state_space.clip(dynamics(state, action, time))
        return (next_state, key), (next_state, action, cost)

    next_state, action, cost = scan(
        f=episode,
        init=(init_state, key),
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    total_cost = jnp.sum(cost) + final_cost(state[-1], goal_state)
    return state, action, total_cost
