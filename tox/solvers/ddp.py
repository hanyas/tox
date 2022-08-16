from typing import NamedTuple, Callable, Any

from functools import partial

import jax.numpy as jnp
import jax.scipy as jsc

from jax import jit
from jax.lax import scan, cond, while_loop

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    QuadraticDynamics,
    Trajectory,
    Box,
)

from tox.helpers import (
    quadratize_delta_final_cost,
    quadratize_delta_transient_cost,
    quadratize_delta_dynamics,
)

from tox.utils import symmetrize


class Hyperparameters(NamedTuple):
    alphas: jnp.ndarray = jnp.power(10.0, jnp.linspace(0, -3, 11))
    init_lmbda: float = 1.0
    init_d_lmbda: float = 1.0
    min_lmbda: float = 1e-6
    max_lmbda: float = 1e16
    mult_lmbda: float = 1.4
    tol_fun: float = 1e-2
    tol_grad: float = 1e-6
    min_improv: float = 0.0
    max_iter: int = 100


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(
        self,
        state: jnp.ndarray,
        time: int,
        reference: Trajectory,
        alpha: float,
    ) -> jnp.ndarray:
        return (
            reference.action[time]
            + self.K[time] @ (state - reference.state[time])
            + alpha * self.kff[time]
        )


def _delta_backward_pass(
    quadratic_final_cost: QuadraticFinalCost,
    quadratic_transient_cost: QuadraticTransientCost,
    quadratic_dynamics: QuadraticDynamics,
    lmbda: float,
) -> (LinearPolicy, Any, Any):

    fCxx, fcx, _ = quadratic_final_cost
    Cxx, Cuu, Cxu, cx, cu, _ = quadratic_transient_cost
    fxx, fuu, fxu, fx, fu, _ = quadratic_dynamics

    dV = jnp.zeros((2,))

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, fxx, fuu, fxu, fx, fu = params

        Vxx, vx, dV = carry

        Qxx = symmetrize(Cxx + fx.T @ Vxx @ fx + jnp.tensordot(vx, fxx, axes=1))
        Quu = symmetrize(Cuu + fu.T @ Vxx @ fu + jnp.tensordot(vx, fuu, axes=1))
        Qux = (Cxu + fx.T @ Vxx @ fu + jnp.tensordot(vx, fxu, axes=1)).T

        qx = cx + fx.T @ vx
        qu = cu + fu.T @ vx

        Quu_reg = symmetrize(Quu + lmbda * jnp.eye(Quu.shape[0]))

        feasability = jnp.all(jnp.linalg.eigvals(Quu_reg) > 0.0)

        def _not_feasible(args):
            Vxx, vx, dV, Qxx, Quu, Quu_reg, Qux, qx, qu = args

            K, kff = jnp.zeros_like(Qux), jnp.zeros_like(qu)
            return (Vxx, vx, dV), (K, kff, False)

        def _feasible(args):
            Vxx, vx, dV, Qxx, Quu, Quu_reg, Qux, qx, qu = args

            K = -jsc.linalg.solve(Quu_reg, Qux, sym_pos=True)
            kff = -jsc.linalg.solve(Quu_reg, qu, sym_pos=True)

            Vxx = symmetrize(Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K)
            vx = qx + K.T @ Quu @ kff + K.T @ qu + Qux.T @ kff

            dV += jnp.array([kff.T @ qu, 0.5 * kff.T @ Quu @ kff])
            return (Vxx, vx, dV), (K, kff, True)

        return cond(
            feasability,
            _feasible,
            _not_feasible,
            (Vxx, vx, dV, Qxx, Quu, Quu_reg, Qux, qx, qu),
        )

    (_, _, dV), (K, kff, feasible) = scan(
        f=_backwards,
        init=(fCxx, fcx, dV),
        xs=(Cxx, Cuu, Cxu, cx, cu, fxx, fuu, fxu, fx, fu),
        reverse=True,
    )

    return LinearPolicy(K, kff), dV, feasible


_jit_backward_pass = jit(_delta_backward_pass)


def _quadratize_quadratize(
    final_cost, transient_cost, goal_state, dynamics, state_space, reference
):

    horizon = reference.horizon
    time = jnp.linspace(0, horizon, horizon + 1)

    quadratic_final_cost = quadratize_delta_final_cost(
        final_cost,
        goal_state,
        reference.final,
    )
    quadratic_transient_cost = quadratize_delta_transient_cost(
        transient_cost, goal_state, reference.transient, time[:-1]
    )
    quadratic_dynamics = quadratize_delta_dynamics(
        dynamics, state_space, reference.transient, time[:-1]
    )

    return quadratic_final_cost, quadratic_transient_cost, quadratic_dynamics


_jit_quadratize_quadratize = jit(
    _quadratize_quadratize, static_argnums=(0, 1, 3, 4)
)


def py_solver(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    options: Hyperparameters,
    verbose: bool = False,
) -> (LinearPolicy, Trajectory):

    trace = []

    k = 0
    initialization_feasible = False
    state, action, total_cost = None, None, None
    while k < options.alphas.shape[0] and not initialization_feasible:
        _state, _action, _total_cost = _jit_rollout(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            init_state,
            state_space,
            policy,
            action_space,
            reference,
            options.alphas[k],
        )
        if jnp.all(jnp.abs(_state) < 1e8):
            initialization_feasible = True
            state = _state
            action = _action
            total_cost = _total_cost
        else:
            k = k + 1

    reference = Trajectory(state, action)
    trace.append(total_cost)

    if initialization_feasible:

        lmbda = options.init_lmbda
        d_lmbda = options.init_d_lmbda

        for _ in range(options.max_iter):
            (
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
            ) = _jit_quadratize_quadratize(
                final_cost,
                transient_cost,
                goal_state,
                dynamics,
                state_space,
                reference,
            )

            backpass_feasible = False
            next_policy, dV = None, None
            while lmbda < options.max_lmbda and not backpass_feasible:
                _next_policy, _dV, _feasible = _jit_backward_pass(
                    quadratic_final_cost,
                    quadratic_transient_cost,
                    linear_dynamics,
                    lmbda,
                )
                if jnp.all(_feasible):
                    backpass_feasible = True
                    next_policy, dV = _next_policy, _dV
                else:
                    # increase lmbda
                    d_lmbda = jnp.maximum(
                        d_lmbda * options.mult_lmbda, options.mult_lmbda
                    )
                    lmbda = jnp.maximum(lmbda * d_lmbda, options.min_lmbda)

            if not backpass_feasible:
                if verbose:
                    print(
                        "Backpass not feasible, maximum regularization reached"
                    )
                return policy, reference, trace

            # terminate if gradient too small
            grad_norm = jnp.mean(
                jnp.max(
                    jnp.abs(next_policy.kff)
                    / (jnp.abs(reference.action) + 1.0),
                    axis=0,
                )
            )
            if grad_norm < options.tol_grad and lmbda < 1e-5:
                if verbose:
                    print("Gradient tolerance reached")
                return policy, reference, trace

            # execute a forward pass
            m = 0
            improvement_feasible = False
            next_state, next_action, next_total_cost, total_cost_diff = (
                None,
                None,
                None,
                None,
            )
            while m < options.alphas.shape[0] and not improvement_feasible:
                # apply on actual system
                _next_state, _next_action, _next_total_cost = _jit_rollout(
                    final_cost,
                    transient_cost,
                    goal_state,
                    dynamics,
                    init_state,
                    state_space,
                    next_policy,
                    action_space,
                    reference,
                    options.alphas[m],
                )

                # check total cost improvement
                _total_cost_diff = total_cost - _next_total_cost
                _expected_improvement = (
                    -1.0
                    * options.alphas[m]
                    * (dV[0] + options.alphas[m] * dV[1])
                )
                _improvement = _total_cost_diff / _expected_improvement
                if _improvement > options.min_improv:
                    improvement_feasible = True
                    next_state = _next_state
                    next_action = _next_action
                    next_total_cost = _next_total_cost
                    total_cost_diff = _total_cost_diff
                else:
                    m = m + 1

            # accept or reject
            if improvement_feasible:
                # decrease lmbda
                d_lmbda = jnp.minimum(
                    d_lmbda / options.mult_lmbda, 1.0 / options.mult_lmbda
                )
                lmbda = lmbda * d_lmbda * (lmbda > options.min_lmbda)

                policy = next_policy
                reference = Trajectory(next_state, next_action)
                total_cost = next_total_cost

                trace.append(total_cost)

                # terminate if reached objective tolerance
                if total_cost_diff < options.tol_fun:
                    if verbose:
                        print("Objective tolerance reached")
                    return policy, reference, trace
            else:
                # increase lmbda
                d_lmbda = jnp.maximum(
                    d_lmbda * options.mult_lmbda, options.mult_lmbda
                )
                lmbda = jnp.maximum(lmbda * d_lmbda, options.min_lmbda)
                if lmbda > options.max_lmbda:
                    if verbose:
                        print(
                            "Improvment not feasible, maximum regularization reached"
                        )
                    return policy, reference, trace

        return policy, reference, trace
    else:
        if verbose:
            print("Initial trajectory diverges")
        return policy, reference, trace


@partial(jit, static_argnums=(0, 1, 3, 5, 7))
def jax_solver(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    options: Hyperparameters,
) -> (LinearPolicy, Trajectory):

    # check initial feasability
    def _initial_feasability_cond(carry):
        k, (state, _, _) = carry
        return jnp.logical_and(
            (k < options.alphas.shape[0]), (jnp.any(jnp.abs(state) > 1e8))
        )

    def _initial_feasability_body(carry):
        k, _ = carry

        state, action, total_cost = rollout(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            init_state,
            state_space,
            policy,
            action_space,
            reference,
            options.alphas[k],
        )

        return k + 1, (state, action, total_cost)

    def _start_main(args):
        (
            policy,
            reference,
            options,
            total_cost,
        ) = args

        def _main_loop_cond(carry):
            i, (_, _, _, lmbda, _) = carry
            return jnp.logical_and(
                (i < options.max_iter),
                (lmbda <= options.max_lmbda),
            )

        def _main_loop_body(carry):
            i, (policy, reference, total_cost, lmbda, d_lmbda) = carry

            (
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
            ) = _quadratize_quadratize(
                final_cost,
                transient_cost,
                goal_state,
                dynamics,
                state_space,
                reference,
            )

            # backward pass feasability
            def _backward_pass_feasability_cond(carry):
                _, _, backpass_feasible, lmbda, _ = carry

                return jnp.logical_and(
                    (lmbda < options.max_lmbda),
                    (jnp.logical_not(jnp.all(backpass_feasible))),
                )

            def _backward_pass_feasability_body(carry):
                _, _, _, lmbda, d_lmbda = carry

                d_lmbda = jnp.maximum(
                    d_lmbda * options.mult_lmbda, options.mult_lmbda
                )
                lmbda = jnp.maximum(lmbda * d_lmbda, options.min_lmbda)

                (
                    next_policy,
                    dV,
                    backpass_feasible,
                ) = _delta_backward_pass(
                    quadratic_final_cost,
                    quadratic_transient_cost,
                    linear_dynamics,
                    lmbda,
                )

                return next_policy, dV, backpass_feasible, lmbda, d_lmbda

            next_policy, dV, backpass_feasible = _delta_backward_pass(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                lmbda,
            )

            next_policy, dV, backpass_feasible, lmbda, d_lmbda = while_loop(
                _backward_pass_feasability_cond,
                _backward_pass_feasability_body,
                init_val=(next_policy, dV, backpass_feasible, lmbda, d_lmbda),
            )

            def _backpass_success(args):
                (
                    policy,
                    reference,
                    total_cost,
                    next_policy,
                    dV,
                    lmbda,
                    d_lmbda,
                ) = args

                def _gradient_converged(args):
                    (
                        policy,
                        reference,
                        total_cost,
                        next_policy,
                        dV,
                        lmbda,
                        d_lmbda,
                    ) = args

                    return policy, reference, total_cost, lmbda, d_lmbda

                def _gradient_not_converged(args):
                    (
                        policy,
                        reference,
                        total_cost,
                        next_policy,
                        dV,
                        lmbda,
                        d_lmbda,
                    ) = args

                    # check if improvement possible
                    def _improvement_feasability_cond(carry):
                        m, (_, _, _, _, _, _, _, improvement) = carry

                        return jnp.logical_and(
                            (m < options.alphas.shape[0]),
                            (improvement <= options.min_improv),
                        )

                    def _improvement_feasability_body(carry):
                        m, (
                            total_cost,
                            next_policy,
                            dV,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = carry

                        next_state, next_action, next_total_cost = rollout(
                            final_cost,
                            transient_cost,
                            goal_state,
                            dynamics,
                            init_state,
                            state_space,
                            next_policy,
                            action_space,
                            reference,
                            options.alphas[m],
                        )

                        total_cost_diff = total_cost - next_total_cost
                        expected_improvement = (
                            -1.0
                            * options.alphas[m]
                            * (dV[0] + options.alphas[m] * dV[1])
                        )
                        improvement = total_cost_diff / expected_improvement

                        return m + 1, (
                            total_cost,
                            next_policy,
                            dV,
                            next_state,
                            next_action,
                            next_total_cost,
                            total_cost_diff,
                            improvement,
                        )

                    next_state = jnp.zeros_like(reference.state)
                    next_action = jnp.zeros_like(reference.action)
                    next_total_cost = jnp.finfo(jnp.float64).max
                    total_cost_diff = jnp.finfo(jnp.float64).min
                    improvement = jnp.finfo(jnp.float64).min

                    _, (
                        _,
                        _,
                        _,
                        next_state,
                        next_action,
                        next_total_cost,
                        total_cost_diff,
                        improvement,
                    ) = while_loop(
                        _improvement_feasability_cond,
                        _improvement_feasability_body,
                        init_val=(
                            0,
                            (
                                total_cost,
                                next_policy,
                                dV,
                                next_state,
                                next_action,
                                next_total_cost,
                                total_cost_diff,
                                improvement,
                            ),
                        ),
                    )

                    def _accept_step(args):
                        (
                            policy,
                            next_policy,
                            reference,
                            next_reference,
                            total_cost,
                            next_total_cost,
                            lmbda,
                            d_lmbda,
                        ) = args

                        # decrease lmbda
                        d_lmbda = jnp.minimum(
                            d_lmbda / options.mult_lmbda,
                            1.0 / options.mult_lmbda,
                        )
                        lmbda = lmbda * d_lmbda * (lmbda > options.min_lmbda)

                        return (
                            next_policy,
                            next_reference,
                            next_total_cost,
                            lmbda,
                            d_lmbda,
                        )

                    def _reject_step(args):
                        (
                            policy,
                            next_policy,
                            reference,
                            next_reference,
                            total_cost,
                            next_total_cost,
                            lmbda,
                            d_lmbda,
                        ) = args

                        # increase lmbda
                        d_lmbda = jnp.maximum(
                            d_lmbda * options.mult_lmbda, options.mult_lmbda
                        )
                        lmbda = jnp.maximum(lmbda * d_lmbda, options.min_lmbda)

                        return (
                            policy,
                            reference,
                            total_cost,
                            lmbda,
                            d_lmbda,
                        )

                    next_reference = Trajectory(next_state, next_action)
                    policy, reference, total_cost, lmbda, d_lmbda = cond(
                        improvement > options.min_improv,
                        _accept_step,
                        _reject_step,
                        (
                            policy,
                            next_policy,
                            reference,
                            next_reference,
                            total_cost,
                            next_total_cost,
                            lmbda,
                            d_lmbda,
                        ),
                    )
                    return policy, reference, total_cost, lmbda, d_lmbda

                grad_norm = jnp.mean(
                    jnp.max(
                        jnp.abs(next_policy.kff)
                        / (jnp.abs(reference.action) + 1.0),
                        axis=0,
                    )
                )

                policy, reference, total_cost, lmbda, d_lmbda = cond(
                    jnp.logical_and(
                        (grad_norm < options.tol_grad), (lmbda < 1e-6)
                    ),
                    _gradient_converged,
                    _gradient_not_converged,
                    (
                        policy,
                        reference,
                        total_cost,
                        next_policy,
                        dV,
                        lmbda,
                        d_lmbda,
                    ),
                )
                return policy, reference, total_cost, lmbda, d_lmbda

            def _backpass_failure(args):
                (
                    policy,
                    reference,
                    total_cost,
                    next_policy,
                    dV,
                    lmbda,
                    d_lmbda,
                ) = args
                return policy, reference, total_cost, lmbda, d_lmbda

            policy, reference, total_cost, lmbda, d_lmbda = cond(
                jnp.all(backpass_feasible),
                _backpass_success,
                _backpass_failure,
                (
                    policy,
                    reference,
                    total_cost,
                    next_policy,
                    dV,
                    lmbda,
                    d_lmbda,
                ),
            )

            return i + 1, (policy, reference, total_cost, lmbda, d_lmbda)

        lmbda = options.init_lmbda
        d_lmbda = options.init_d_lmbda

        _, (policy, reference, _, _, _) = while_loop(
            _main_loop_cond,
            _main_loop_body,
            init_val=(0, (policy, reference, total_cost, lmbda, d_lmbda)),
        )
        return policy, reference

    def _exit_programme(args):
        (
            policy,
            reference,
            options,
            total_cost,
        ) = args

        return policy, reference

    state, action, total_cost = rollout(
        final_cost,
        transient_cost,
        goal_state,
        dynamics,
        init_state,
        state_space,
        policy,
        action_space,
        reference,
        options.alphas[0],
    )

    _, (state, action, total_cost) = while_loop(
        _initial_feasability_cond,
        _initial_feasability_body,
        init_val=(0, (state, action, total_cost)),
    )

    reference = Trajectory(state, action)

    return cond(
        jnp.all(jnp.abs(state) < 1e8),
        _start_main,
        _exit_programme,
        (
            policy,
            reference,
            options,
            total_cost,
        ),
    )


def rollout(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    alpha: float = 1.0,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def episode(state, time):
        action = action_space.clip(policy(state, time, reference, alpha))
        cost = transient_cost(state, action, time, goal_state)
        next_state = state_space.clip(dynamics(state, action, time))
        return next_state, [next_state, action, cost]

    horizon = reference.horizon
    next_state, action, cost = scan(
        f=episode,
        init=init_state,
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    total_cost = jnp.sum(cost) + final_cost(state[-1], goal_state)
    return state, action, total_cost


_jit_rollout = jit(rollout, static_argnums=(0, 1, 3, 5, 7))


@partial(jit, static_argnums=(0, 1, 3, 5, 7, -1))
def mpc_rollout(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    policy: LinearPolicy,
    action_space: Box,
    reference: Trajectory,
    options: Hyperparameters,
    nb_steps: int,
) -> (jnp.ndarray, jnp.ndarray):
    def mpc_step(carry, args):
        policy, reference, state = carry

        next_policy, next_reference = jax_solver(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            state,
            state_space,
            policy,
            action_space,
            reference,
            options,
        )

        action = next_reference.action[0]
        next_state = state_space.clip(dynamics(state, action, 0))

        return (next_policy, next_reference, next_state), (next_state, action)

    _, (state, action) = scan(
        mpc_step, init=(policy, reference, init_state), xs=jnp.arange(nb_steps)
    )

    state = jnp.vstack((init_state, state))
    return state, action
