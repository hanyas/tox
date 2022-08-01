from typing import Callable

from functools import partial

from jax import jacobian as jac
from jax import hessian as hess

from jax import vmap
import jax.numpy as jnp

from tox.objects import (
    Trajectory,
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics, Box,
)


def quadratize_final_cost(
    final_cost: Callable, state: jnp.ndarray
) -> QuadraticFinalCost:
    # f(x, u) = 0.5 * (x - xr).T * Cxx(xr, ur) * (x - xr)
    #           + (x - xr).T * cx + f(xr, ur)
    Cxx = hess(final_cost, 0)(state)
    cx = jac(final_cost, 0)(state)
    c0 = final_cost(state)
    return QuadraticFinalCost(Cxx, cx, c0)


@partial(vmap, in_axes=(None, 0, 0))
def quadratize_transient_cost(
    transient_cost: Callable, reference: Trajectory, time: jnp.ndarray
) -> QuadraticTransientCost:
    # f(x, u) = 0.5 * (x - xr).T * Cxx(xr, ur) * (x - xr)
    #           + 0.5 * (u - ur).T * Cuu(xr, ur) * (u - ur)
    #           + 0.5 * (x - xr).T * Cxu(xr, ur) * (u - ur)
    #           + 0.5 * (u - ur).T * Cux(xr, ur) * (x - xr)
    #           + (x - xr).T * cx + (u - ur).T * cu + f(xr, ur)
    Cxx = hess(transient_cost, 0)(reference.state, reference.action, time)
    Cuu = hess(transient_cost, 1)(reference.state, reference.action, time)
    Cxu = jac(jac(transient_cost, 0), 1)(
        reference.state, reference.action, time
    )

    cx = jac(transient_cost, 0)(reference.state, reference.action, time)
    cu = jac(transient_cost, 1)(reference.state, reference.action, time)
    c0 = transient_cost(reference.state, reference.action, time)
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def linearize_dynamics(
    dynamics: Callable, state_space: Box, reference: Trajectory, time: jnp.ndarray,
) -> LinearDynamics:
    # f(x, u) = f(xr, ur) + A(xr, ur) (x - xr) + B(xr, ur) (u - ur)
    A = jac(state_space(dynamics), 0)(reference.state, reference.action, time)
    B = jac(state_space(dynamics), 1)(reference.state, reference.action, time)
    f0 = state_space(dynamics)(reference.state, reference.action, time)
    return LinearDynamics(A, B, f0)
