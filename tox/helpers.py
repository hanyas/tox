from typing import Callable

from functools import partial

import jax.scipy.linalg
from jax.flatten_util import ravel_pytree
from jax import jacobian as jac
from jax import hessian as hess

from jax import vmap
import jax.numpy as jnp

from tox.utils import tria, symmetrize

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
    Belief,
    Box,
)


def quadratize_delta_final_cost(
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
    cx = jac(final_cost, 0)(state, goal_state)
    c0 = final_cost(state, goal_state)
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
    state, action = reference

    Cxx = hess(transient_cost, 0)(state, action, time, goal_state)
    Cuu = hess(transient_cost, 1)(state, action, time, goal_state)
    Cxu = jac(jac(transient_cost, 0), 1)(state, action, time, goal_state)

    cx = jac(transient_cost, 0)(state, action, time, goal_state)
    cu = jac(transient_cost, 1)(state, action, time, goal_state)
    c0 = transient_cost(state, action, time, goal_state)
    return QuadraticTransientCost(Cxx, Cuu, Cxu, cx, cu, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def linearize_delta_dynamics(
    dynamics: Callable,
    state_space: Box,
    reference: Trajectory,
    time: jnp.ndarray,
) -> LinearDynamics:
    """
    f(x, u) = A(xr, ur) * (x - xr) + B(xr, ur) * (u - ur) + f(xr, ur)
    """
    state, action = reference

    A = jac(state_space(dynamics), 0)(state, action, time)
    B = jac(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time)
    return LinearDynamics(A, B, c)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_delta_dynamics(
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
    fxu = jac(jac(state_space(dynamics), 0), 1)(state, action, time)

    fx = jac(state_space(dynamics), 0)(state, action, time)
    fu = jac(state_space(dynamics), 1)(state, action, time)
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
    cx = -Cxx @ state + jac(final_cost, 0)(state, goal_state)
    c0 = (
        0.5 * state.T @ Cxx @ state
        - state.T @ jac(final_cost, 0)(state, goal_state)
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

    A = jac(state_space(dynamics), 0)(state, action, time)
    B = jac(state_space(dynamics), 1)(state, action, time)
    c = state_space(dynamics)(state, action, time) - A @ state - B @ action
    return LinearDynamics(A, B, c)


def quadratize_delta_final_belief_cost(
    final_cost: Callable,
    goal_state: jnp.ndarray,
    belief: Belief,
) -> QuadraticFinalBeliefCost:
    """
    H(c) : Hessian, J(c): Jacobian
    c(x, s) = 0.5 * (x - xr).T * H(c)(xr) * (x - xr)
              + (x - xr).T * J(c)(xr) + c(xr)
              + vec(c - cr).T @ J(c)(vec(cr))
    """
    bel_mu, bel_cov = belief

    Cxx = hess(final_cost, 0)(bel_mu, bel_cov, goal_state)
    cx = jac(final_cost, 0)(bel_mu, bel_cov, goal_state)
    cs = jnp.ravel(jac(final_cost, 1)(bel_mu, bel_cov, goal_state))
    c0 = final_cost(bel_mu, bel_cov, goal_state)
    return QuadraticFinalBeliefCost(Cxx, cx, cs, c0)


@partial(vmap, in_axes=(None, None, 0, 0))
def quadratize_delta_transient_belief_cost(
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    reference: BeliefTrajectory,
    time: int,
) -> QuadraticTransientBeliefCost:

    bel_mu, bel_cov, action = reference

    Cxx = hess(transient_cost, 0)(bel_mu, bel_cov, action, time, goal_state)
    Cuu = hess(transient_cost, 2)(bel_mu, bel_cov, action, time, goal_state)
    Cxu = jac(jac(transient_cost, 0), 2)(bel_mu, bel_cov, action, time, goal_state)

    cx = jac(transient_cost, 0)(bel_mu, bel_cov, action, time, goal_state)
    cu = jac(transient_cost, 2)(bel_mu, bel_cov, action, time, goal_state)
    cs = jnp.ravel(jac(transient_cost, 1)(bel_mu, bel_cov, action, time, goal_state))
    c0 = transient_cost(bel_mu, bel_cov, action, time, goal_state)

    return QuadraticTransientBeliefCost(Cxx, Cuu, Cxu, cx, cu, cs, c0)


@partial(vmap, in_axes=(None, None, None, None, None, None, 0, 0))
def linearize_delta_belief_dynamics(
    dynamics_mean: Callable,
    dynamics_noise: Callable,
    state_space: Box,
    observation_mean: Callable,
    observation_noise: Callable,
    observation_space: Box,
    reference: BeliefTrajectory,
    time: jnp.ndarray,
):
    # def _extended_kalman(
    #     bel_mu: jnp.ndarray,
    #     bel_cov_chol: jnp.ndarray,
    #     action: jnp.ndarray,
    # ):
    #
    #     constrained_dynamics_mean = state_space(dynamics_mean)
    #     constrained_observation_mean = observation_space(observation_mean)
    #
    #     A = jac(constrained_dynamics_mean, 0)(bel_mu, action, time)
    #     B = jac(constrained_dynamics_mean, 1)(bel_mu, action, time)
    #     bel_mu_mu = constrained_dynamics_mean(bel_mu, action, time)
    #     dyn_cov_chol = dynamics_noise(bel_mu, action, time)
    #
    #     H = jac(constrained_observation_mean, 0)(bel_mu, time)
    #     h0 = constrained_observation_mean(bel_mu, time)
    #     noise_cov_chol = observation_noise(bel_mu, action, time)
    #
    #     pred_cov_chol = tria(jnp.concatenate([A @ bel_cov_chol, dyn_cov_chol], axis=1))
    #
    #     dy, dx = H.shape
    #     block_chol = tria(jnp.block([[H @ pred_cov_chol, noise_cov_chol],
    #                                  [pred_cov_chol, jnp.zeros_like(pred_cov_chol, shape=(dx, dy))]]))
    #
    #     bel_mu_cov_chol = block_chol[dy:, :dy]
    #     bel_cov_chol = block_chol[dy:, dy:]
    #
    #     return bel_mu_mu, bel_mu_cov_chol, bel_cov_chol

    def _extended_kalman(
        bel_mu: jnp.ndarray,
        bel_cov: jnp.ndarray,
        action: jnp.ndarray,
        time: jnp.ndarray
    ):
        constrained_dynamics_mean = state_space(dynamics_mean)
        constrained_observation_mean = observation_space(observation_mean)

        A = jac(constrained_dynamics_mean, 0)(bel_mu, action, time)
        bel_mu_mu = constrained_dynamics_mean(bel_mu, action, time)
        dyn_cov = dynamics_noise(bel_mu, action, time)

        H = jac(constrained_observation_mean, 0)(bel_mu, action, time)
        obs_cov = observation_noise(bel_mu, action, time)

        G = symmetrize(A @ bel_cov @ A.T + dyn_cov)
        K = jax.scipy.linalg.solve(H @ G @ H.T + obs_cov, H @ G.T).T

        bel_mu_cov = symmetrize(K @ H @ G)
        bel_cov = symmetrize(G - K @ H @ G)

        return bel_mu_mu, bel_mu_cov, bel_cov

    bel_mu, bel_cov, action = reference

    raveled_input, unravel = ravel_pytree((bel_mu, bel_cov, action))

    def raveled_extended_kalman(raveled_input):
        return ravel_pytree(_extended_kalman(*unravel(raveled_input), time))[0]

    _jacs = jac(raveled_extended_kalman)(raveled_input)

    belief_dim = bel_mu.shape[-1]
    action_dim = action.shape[-1]

    fx = _jacs[:belief_dim, :belief_dim]
    fu = _jacs[:belief_dim, -action_dim:]

    Wx = _jacs[belief_dim: belief_dim + belief_dim * belief_dim, :belief_dim]
    Ws = _jacs[
        belief_dim: belief_dim + belief_dim * belief_dim,
        belief_dim: belief_dim + belief_dim * belief_dim,
    ]
    Wu = _jacs[belief_dim: belief_dim + belief_dim * belief_dim, -action_dim:]

    Px = _jacs[belief_dim + belief_dim * belief_dim:, :belief_dim]
    Ps = _jacs[
        belief_dim + belief_dim * belief_dim:,
        belief_dim: belief_dim + belief_dim * belief_dim,
    ]
    Pu = _jacs[belief_dim + belief_dim * belief_dim:, -action_dim:]

    _, W, _ = _extended_kalman(bel_mu, bel_cov, action, time)

    return LinearBeliefDynamics(fx, fu, Wx, Ws, Wu, Px, Ps, Pu, jnp.ravel(W))
