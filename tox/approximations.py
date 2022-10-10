from typing import Callable, Tuple

from functools import partial

import jax
from jax import jacfwd
from jax import hessian as hess

from jax import vmap
import jax.numpy as jnp

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
    QuadraticDynamics,
    Trajectory,
    QuadraticFinalBeliefCost,
    QuadraticTransientBeliefCost,
    LinearBeliefDynamics,
    BeliefTrajectory,
    Box,
)


def quadratize_diff_final_cost(
    final_cost: Callable,
    goal_state: jnp.ndarray,
    state: jnp.ndarray,
) -> QuadraticFinalCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x) = 0.5 * (x - xr).T * H(c)(xr) * (x - xr)
           + (x - xr).T * J(c)(xr) + c(xr)
    """
    Cxx = hess(final_cost, 0)(state, goal_state)
    cx = jacfwd(final_cost, 0)(state, goal_state)
    c0 = final_cost(state, goal_state)
    return QuadraticFinalCost(Cxx, cx, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_diff_transient_cost(
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
    state, action = reference

    Cxx = hess(transient_cost, 0)(state, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(state, action, time, goal_state)
    Cxu = jacfwd(jacfwd(transient_cost, 0), 1)(state, action, time, goal_state)

    cx = jacfwd(transient_cost, 0)(state, action, time, goal_state)
    cu = jacfwd(transient_cost, 1)(state, action, time, goal_state)
    c0 = transient_cost(state, action, time, goal_state)
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def linearize_diff_dynamics(
    dynamics: Callable,
    state_space: Box,
    reference: Trajectory,
    time: jnp.ndarray,
) -> LinearDynamics:
    """
    f(x, u) = A(xr, ur) * (x - xr) + B(xr, ur) * (u - ur) + f(xr, ur)
    """
    state, action = reference

    A = jacfwd(state_space(dynamics), 0)(state, action, time)
    B = jacfwd(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time)
    return LinearDynamics(A, B, c)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_diff_dynamics(
    dynamics: Callable,
    state_space: Box,
    reference: Trajectory,
    time: jnp.ndarray,
) -> QuadraticDynamics:
    """
    f(x, u) = 0.5 * (x - xr).T * fxx * (x - xr)
              + 0.5 * (u - ur).T * fuu * (u - ur)
              + 0.5 * (x - xr).T * fxu * (u - ur)
              + 0.5 * (u - ur).T * fux * (x - xr)
              + fx.T * (x - xr) + fu.T * (u - ur) + f(xr, ur)
    """
    state, action = reference

    fxx = hess(state_space(dynamics), 0)(state, action, time)
    fuu = hess(state_space(dynamics), 1)(state, action, time)
    fxu = jacfwd(jacfwd(state_space(dynamics), 0), 1)(state, action, time)

    fx = jacfwd(state_space(dynamics), 0)(state, action, time)
    fu = jacfwd(state_space(dynamics), 1)(state, action, time)
    f0 = state_space(dynamics)(state, action, time)

    return QuadraticDynamics(fxx, fuu, fxu, fx, fu, f0)


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
    cx = -Cxx @ state + jacfwd(final_cost, 0)(state, goal_state)
    c0 = (
        0.5 * state.T @ Cxx @ state
        - state.T @ jacfwd(final_cost, 0)(state, goal_state)
        + final_cost(state, goal_state)
    )
    return QuadraticFinalCost(Cxx, cx, c0)


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
    state, action = reference

    Cxx = hess(transient_cost, 0)(state, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(state, action, time, goal_state)
    Cxu = jacfwd(jacfwd(transient_cost, 0), 1)(state, action, time, goal_state)

    cx = (
        -Cxx @ state
        - Cxu @ action
        + jacfwd(transient_cost, 0)(state, action, time, goal_state)
    )
    cu = (
        -Cuu @ action
        - Cxu.T @ state
        + jacfwd(transient_cost, 1)(state, action, time, goal_state)
    )
    c0 = (
        0.5 * state.T @ Cxx @ state
        + 0.5 * action.T @ Cuu @ action
        + 0.5 * state.T * Cxu * action
        + 0.5 * action.T * Cxu.T * state
        - state.T @ jacfwd(transient_cost, 0)(state, action, time, goal_state)
        - action.T @ jacfwd(transient_cost, 1)(state, action, time, goal_state)
        + transient_cost(state, action, time, goal_state)
    )
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


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
    state, action = reference

    A = jacfwd(state_space(dynamics), 0)(state, action, time)
    B = jacfwd(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time) - A @ state - B @ action
    return LinearDynamics(A, B, c)


def quadratize_diff_final_belief_cost(
    final_cost: Callable,
    goal_state: jnp.ndarray,
    belief: jnp.ndarray,
) -> QuadraticFinalBeliefCost:

    Cbb = hess(final_cost, 0)(belief, goal_state)
    cb = jacfwd(final_cost, 0)(belief, goal_state)
    c0 = final_cost(belief, goal_state)
    return QuadraticFinalBeliefCost(Cbb, cb, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_diff_transient_belief_cost(
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    reference: BeliefTrajectory,
    time: int,
) -> QuadraticTransientBeliefCost:

    belief, action = reference

    Cbb = hess(transient_cost, 0)(belief, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(belief, action, time, goal_state)
    Cbu = jacfwd(jacfwd(transient_cost, 0), 1)(belief, action, time, goal_state)

    cb = jacfwd(transient_cost, 0)(belief, action, time, goal_state)
    cu = jacfwd(transient_cost, 1)(belief, action, time, goal_state)
    c0 = transient_cost(belief, action, time, goal_state)

    return QuadraticTransientBeliefCost(Cbb, Cuu, Cbu, cb, cu, c0)


@partial(vmap, in_axes=(None, 0, 0))
def linearize_diff_belief_dynamics(
    belief_dynamics: Tuple,
    reference: BeliefTrajectory,
    time: jnp.ndarray,
):

    belief, action = reference
    belief_mean, belief_covar = belief_dynamics

    gb = jacfwd(belief_mean, 0)(belief, action, time)
    gu = jacfwd(belief_mean, 1)(belief, action, time)

    # column-wise derivatives of covariance
    Wb = jacfwd(belief_covar, 0)(belief, action, time)
    Wu = jacfwd(belief_covar, 1)(belief, action, time)
    W = belief_covar(belief, action, time)

    return LinearBeliefDynamics(gb, gu, Wb, Wu, W)
