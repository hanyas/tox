from typing import Callable

from functools import partial

from jax import jacobian as jac
from jax import hessian as hess

from jax import vmap
import jax.numpy as jnp

from tox.utils import symmetrize

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
    Trajectory,
    BeliefTrajectory,
    Box,
)


def quadratize_delta_final_cost(
    final_cost: Callable,
    goal_state: jnp.ndarray,
    state: jnp.ndarray,
) -> QuadraticFinalCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x, u) = 0.5 * (x - xr).T * H(c)(xr) * (x - xr)
              + (x - xr).T * J(c)(xr) + c(xr)
    """
    Cxx = hess(final_cost, 0)(state, goal_state)
    cx = jac(final_cost, 0)(state, goal_state)
    c0 = final_cost(state, goal_state)
    return QuadraticFinalCost(Cxx, cx, c0)


def quadratize_final_cost(
    final_cost: Callable,
    goal_state: jnp.ndarray,
    state: jnp.ndarray,
) -> QuadraticFinalCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x) = 0.5 * x.T * H(c)(xr) * x
           + x.T * (-H(c)(xr) * xr + J(c)(xr))
           + 0.5 * xr.T * H(c)(xr) * xr + - xr.T @ J(c)(xr) + c(xr)
    """
    Cxx = hess(final_cost, 0)(state, goal_state)
    cx = -Cxx @ state + jac(final_cost, 0)(state, goal_state)
    c0 = (
        0.5 * state.T @ Cxx @ state
        - state.T @ jac(final_cost, 0)(state, goal_state)
        + final_cost(state, goal_state)
    )
    return QuadraticFinalCost(Cxx, cx, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_delta_transient_cost(
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    reference: Trajectory,
    time: jnp.ndarray,
) -> QuadraticTransientCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x, u) = 0.5 * (x - xr).T * Hxx(c)(xr, ur) * (x - xr)
              + 0.5 * (u - ur).T * Huu(c)(xr, ur) * (u - ur)
              + 0.5 * (x - xr).T * Jx(Ju(c)))(xr, ur) * (u - ur)
              + 0.5 * (u - ur).T * Ju(Jx(c)))(xr, ur) * (x - xr)
              + (x - xr).T * Jx(c)(xr, ur) + (u - ur).T * Ju(c)(xr, ur) + c(xr, ur)
    """
    state, action = reference.state, reference.action

    Cxx = hess(transient_cost, 0)(state, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(state, action, time, goal_state)
    Cxu = jac(jac(transient_cost, 0), 1)(state, action, time, goal_state)

    cx = jac(transient_cost, 0)(state, action, time, goal_state)
    cu = jac(transient_cost, 1)(state, action, time, goal_state)
    c0 = transient_cost(state, action, time, goal_state)
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_transient_cost(
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    reference: Trajectory,
    time: jnp.ndarray,
) -> QuadraticTransientCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x, u) = 0.5 * x.T * Hxx(c)(xr, ur) * x
              + 0.5 * u.T * Huu(c)(xr, ur) * u
              + 0.5 * x.T * Hxu(c)(xr, ur) * u
              + 0.5 * u.T * Hux(c)(xr, ur) * x
              + x.T * (-Hxx(c)(xr, ur) * xr - Hxu(c)(xr, ur) * ur + Jx(c)(xr, ur))
              + u.T * (-Huu(c)(xr, ur) * ur - Hux(c)(xr, ur) * xr + Ju(c)(xr, ur))
              + 0.5 * xr.T * Hxx(c)(xr, ur) * xr + 0.5 * ur.T * Huu(c)(xr, ur) * ur
              + 0.5 * xr.T * Hxu(c)(xr, ur) * ur + 0.5 * ur.T * Hux(c)(xr, ur) * xr
              - xr.T * Jx(c)(xr, ur) - ur.T * Ju(c)(xr, ur) + c(xr, ur)
    """
    state, action = reference.state, reference.action

    Cxx = hess(transient_cost, 0)(state, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(state, action, time, goal_state)
    Cxu = jac(jac(transient_cost, 0), 1)(state, action, time, goal_state)

    cx = (
        -Cxx @ state
        - Cxu @ action
        + jac(transient_cost, 0)(state, action, time, goal_state)
    )
    cu = (
        -Cuu @ action
        - Cxu.T @ state
        + jac(transient_cost, 1)(state, action, time, goal_state)
    )
    c0 = (
        0.5 * state.T @ Cxx @ state
        + 0.5 * action.T @ Cuu @ action
        + 0.5 * state.T * Cxu * action
        + 0.5 * action.T * Cxu.T * state
        - state.T @ jac(transient_cost, 0)(state, action, time, goal_state)
        - action.T @ jac(transient_cost, 1)(state, action, time, goal_state)
        + transient_cost(state, action, time, goal_state)
    )
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def linearize_delta_dynamics(
    dynamics: Callable,
    state_space: Box,
    reference: Trajectory,
    time: jnp.ndarray,
) -> LinearDynamics:
    """
    f(x, u) = A(xr, ur) * (x - xr) + B(xr, ur) * (u - ur) + c(xr, ur)
    """
    state, action = reference.state, reference.action

    A = jac(state_space(dynamics), 0)(state, action, time)
    B = jac(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time)
    return LinearDynamics(A, B, c)


@partial(vmap, in_axes=(None, None, 0, 0))
def linearize_dynamics(
    dynamics: Callable,
    state_space: Box,
    reference: Trajectory,
    time: jnp.ndarray,
) -> LinearDynamics:
    """
    f(x, u) = A(xr, ur) * x + B(xr, ur) * u
              + f(xr, ur) - A(xr, ur) * xr - B(xr, ur) * ur
    """
    state, action = reference.state, reference.action

    A = jac(state_space(dynamics), 0)(state, action, time)
    B = jac(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time) - A @ state - B @ action
    return LinearDynamics(A, B, c)
