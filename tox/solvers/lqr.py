from typing import NamedTuple, Callable
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsc

from jax import jit
from jax.lax import scan

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
    Trajectory
)

from tox.helpers import (
    quadratize_final_cost,
    quadratize_transient_cost,
    linearize_dynamics,
)

from tox.utils import symmetrize


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(
        self, state: jnp.ndarray, time: int, reference: Trajectory
    ) -> jnp.ndarray:
        return (
            reference.action[time]
            + self.K[time] @ (state - reference.state[time])
            + self.kff[time]
        )


def _backward_pass(
    quadratic_final_cost: QuadraticFinalCost,
    quadratic_transient_cost: QuadraticTransientCost,
    linear_dynamics: LinearDynamics,
) -> LinearPolicy:

    fCxx, fcx = quadratic_final_cost.Cxx, quadratic_final_cost.cx

    Cxx, Cuu, Cxu, cx, cu = (
        quadratic_transient_cost.Cxx,
        quadratic_transient_cost.Cuu,
        quadratic_transient_cost.Cxu,
        quadratic_transient_cost.cx,
        quadratic_transient_cost.cu,
    )
    A, B = linear_dynamics.A, linear_dynamics.B

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B = params

        Vxx, vx = carry

        Qxx = Cxx + A.T @ Vxx @ A
        Quu = Cuu + B.T @ Vxx @ B
        Qux = (Cxu + A.T @ Vxx @ B).T

        # qx = cx + A.T @ Vxx @ c + A.T @ vx
        # qu = cu + B.T @ Vxx @ c + B.T @ vx

        qx = cx + A.T @ vx
        qu = cu + B.T @ vx

        K = -jsc.linalg.solve(Quu, Qux, sym_pos=True)
        kff = -jsc.linalg.solve(Quu, qu, sym_pos=True)

        Vxx = symmetrize(Qxx + Qux.T @ K)
        vx = qx + Qux.T @ kff

        return [Vxx, vx], [K, kff]

    K, kff = scan(
        f=_backwards,
        init=[fCxx, fcx],
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B),
        reverse=True,
    )[1]

    return LinearPolicy(K, kff)


@partial(jit, static_argnums=(0, 1, 2))
def solver(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    reference: Trajectory,
) -> LinearPolicy:

    horizon = reference.horizon
    time = jnp.linspace(0, horizon, horizon + 1)

    quadratic_final_cost = quadratize_final_cost(
        final_cost, reference.final,
    )
    quadratic_transient_cost = quadratize_transient_cost(
        transient_cost, reference.transient, time[:-1]
    )
    linear_dynamics = linearize_dynamics(
        dynamics, reference.transient, time[:-1]
    )

    return _backward_pass(
        quadratic_final_cost, quadratic_transient_cost, linear_dynamics
    )


@partial(jit, static_argnums=(0, 1, 2))
def rollout(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    init_state: jnp.ndarray,
    policy: LinearPolicy,
    reference: Trajectory,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):

    def episode(state, time):
        action = policy(state, time, reference)
        cost = transient_cost(state, action, time)
        next_state = dynamics(state, action, time)
        return next_state, [next_state, action, cost]

    horizon = reference.horizon
    next_state, action, cost = scan(
        f=episode,
        init=init_state,
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    total_cost = jnp.sum(cost) + final_cost(state[-1])
    return state, action, total_cost
