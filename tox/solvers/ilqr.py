from typing import NamedTuple, Callable, Any

from functools import partial
from dataclasses import dataclass

import numpy as onp
import jax.numpy as jnp
import jax.scipy as jsc

from jax import jit
from jax.lax import scan, cond

from tox.objects import Trajectory
from tox.spaces import Box

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
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


@jit
def _differential_backward_pass(
    quadratic_final_cost: QuadraticFinalCost,
    quadratic_transient_cost: QuadraticTransientCost,
    linear_dynamics: LinearDynamics,
    lmbda: float,
) -> (LinearPolicy, Any, Any):

    fCxx, fcx = quadratic_final_cost.Cxx, quadratic_final_cost.cx

    Cxx, Cuu, Cxu, cx, cu = (
        quadratic_transient_cost.Cxx,
        quadratic_transient_cost.Cuu,
        quadratic_transient_cost.Cxu,
        quadratic_transient_cost.cx,
        quadratic_transient_cost.cu,
    )
    A, B = linear_dynamics.A, linear_dynamics.B

    dV = jnp.zeros((2,))

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B = params

        Vxx, vx, dV = carry

        Qxx = symmetrize(Cxx + A.T @ Vxx @ A)
        Quu = symmetrize(Cuu + B.T @ Vxx @ B)
        Qux = (Cxu + A.T @ Vxx @ B).T

        qx = cx + A.T @ vx
        qu = cu + B.T @ vx

        Quu_reg = symmetrize(
            Cuu + B.T @ (Vxx + lmbda * jnp.eye(Vxx.shape[0])) @ B
        )
        Qux_reg = (Cxu + A.T @ (Vxx + lmbda * jnp.eye(Vxx.shape[0])) @ B).T

        feasability = jnp.all(jnp.linalg.eigvals(Quu_reg) > 0.0)

        def _not_feasible(Vxx, vx, dV, Quu, Qux, Quu_reg, Qux_reg, qx, qu):
            K, kff = jnp.zeros_like(Qux), jnp.zeros_like(qu)
            return [Vxx, vx, dV], [K, kff, False]

        def _feasible(Vxx, vx, dV, Quu, Qux, Quu_reg, Qux_reg, qx, qu):
            K = -jsc.linalg.solve(Quu_reg, Qux_reg, sym_pos=True)
            kff = -jsc.linalg.solve(Quu_reg, qu, sym_pos=True)

            Vxx = symmetrize(Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K)
            vx = qx + K.T @ Quu @ kff + K.T @ qu + Qux.T @ kff

            dV += jnp.array([kff.T @ qu, 0.5 * kff.T @ Quu @ kff])
            return [Vxx, vx, dV], [K, kff, True]

        return cond(
            feasability,
            _feasible,
            _not_feasible,
            *[Vxx, vx, dV, Quu, Qux, Quu_reg, Qux_reg, qx, qu]
        )

    (_, _, dV), (K, kff, feasible) = scan(
        f=_backwards,
        init=[fCxx, fcx, dV],
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B),
        reverse=True,
    )

    return LinearPolicy(K, kff), dV, feasible


@dataclass
class Hyperparameters:
    alphas: onp.ndarray = onp.power(10.0, onp.linspace(0, -3, 11))
    lmbda: float = 1.0
    d_lmbda: float = 1.0
    min_lmbda: float = 1e-6
    max_lmbda: float = 1e6
    mult_lmbda: float = 1.6
    tol_fun: float = 1e-6
    tol_grad: float = 1e-4
    min_improv: float = 0.0
    max_iter: int = 100


def solver(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    options: Hyperparameters,
    init_state: jnp.ndarray,
) -> (LinearPolicy, Trajectory):

    horizon = reference.horizon
    time = jnp.linspace(0, horizon, horizon + 1)

    trace = []
    last_total_cost = None

    for alpha in options.alphas:
        _state, _action, _total_cost = rollout(
            final_cost,
            transient_cost,
            dynamics,
            state_space,
            policy,
            action_space,
            reference,
            init_state,
        )
        if jnp.all(_state < 1e8):
            reference = Trajectory(_state, _action)
            last_total_cost = _total_cost
            break
        else:
            print("Initial trajectory diverges")

    trace.append(last_total_cost)

    for _ in range(options.max_iter):
        # get quadratic cost around reference
        quadratic_final_cost = quadratize_final_cost(
            final_cost, reference.final,
        )
        quadratic_transient_cost = quadratize_transient_cost(
            transient_cost, reference.transient, time[:-1]
        )
        linear_dynamics = linearize_dynamics(
            dynamics, state_space, reference.transient, time[:-1]
        )

        backpass_feasible = False
        _next_policy, dV = None, None
        while not backpass_feasible:
            _next_policy, dV, feasible = _differential_backward_pass(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                options.lmbda,
            )
            if not jnp.all(feasible):
                # increase lmbda
                options.d_lmbda = jnp.maximum(
                    options.d_lmbda * options.mult_lmbda, options.mult_lmbda
                )
                options.lmbda = jnp.maximum(
                    options.lmbda * options.d_lmbda, options.min_lmbda
                )
                if options.lmbda > options.max_lmbda:
                    print("Maximum regularization reached")
                    break
                else:
                    continue
            else:
                backpass_feasible = True

        # terminate if gradient too small
        _grad_norm = jnp.mean(
            jnp.max(
                jnp.abs(_next_policy.kff) / (jnp.abs(reference.action) + 1.0),
                axis=1,
            )
        )
        if _grad_norm < options.tol_grad and options.lmbda < 1e-5:
            options.d_lmbda = jnp.minimum(
                options.d_lmbda / options.mult_lmbda, 1.0 / options.mult_lmbda
            )
            options.lmbda = (
                options.lmbda
                * options.d_lmbda
                * (options.lmbda > options.min_lmbda)
            )
            print("Gradient tolerance reached")
            break

        _next_reference = None
        _total_cost, _total_cost_diff = None, None

        # execute a forward pass
        fwdpass_feasible = False
        if backpass_feasible:
            for alpha in options.alphas:
                # apply on actual system
                _state, _action, _total_cost = rollout(
                    final_cost,
                    transient_cost,
                    dynamics,
                    state_space,
                    _next_policy,
                    action_space,
                    reference,
                    init_state,
                )

                _next_reference = Trajectory(_state, _action)

                # check total cost improvement
                _total_cost_diff = last_total_cost - _total_cost
                _expected_improvement = -1.0 * alpha * (dV[0] + alpha * dV[1])
                _improvment = _total_cost_diff / _expected_improvement
                if _improvment >= options.min_improv:
                    fwdpass_feasible = True
                    break

        # accept or reject
        if fwdpass_feasible:
            # decrease lmbda
            options.d_lmbda = jnp.minimum(
                options.d_lmbda / options.mult_lmbda, 1.0 / options.mult_lmbda
            )
            options.lmbda = (
                options.lmbda
                * options.d_lmbda
                * (options.lmbda > options.min_lmbda)
            )

            policy = _next_policy
            reference = _next_reference
            last_return = _total_cost

            trace.append(last_return)

            # terminate if reached objective tolerance
            if _total_cost_diff < options.tol_fun:
                print("Objective tolerance reached")
                break
        else:
            # increase lmbda
            options.d_lmbda = jnp.maximum(
                options.d_lmbda * options.mult_lmbda, options.mult_lmbda
            )
            options.lmbda = jnp.maximum(
                options.lmbda * options.d_lmbda, options.min_lmbda
            )
            if options.lmbda > options.max_lmbda:
                break
            else:
                continue

    return policy, reference, trace


@partial(jit, static_argnums=(0, 1, 2))
def rollout(
    final_cost: Callable,
    transient_cost: Callable,
    dynamics: Callable,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    init_state: jnp.ndarray,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def episode(state, time):
        action = action_space.clip(policy(state, time, reference))
        cost = transient_cost(state, action, time)
        next_state = state_space.clip(dynamics(state, action, time))
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
