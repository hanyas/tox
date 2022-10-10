from typing import NamedTuple, Callable, List

from functools import partial

import jax.numpy as jnp
import jax.scipy as jsc

import jax
from jax import jit
from jax.lax import scan, cond, while_loop

from tox.objects import (
    QuadraticFinalCost,
    QuadraticTransientCost,
    LinearDynamics,
    Trajectory,
    Box,
)

from tox.approximations import (
    quadratize_diff_final_cost,
    quadratize_diff_transient_cost,
    linearize_diff_dynamics,
)

from tox.utils import symmetrize


class Hyperparameters(NamedTuple):
    init_lmbda: float = 1e-6
    init_mult_lmbda: float = 2.0
    max_lmbda: float = 1e6
    tol_grad: float = 1e-8
    min_ratio: float = 0.0
    max_iter: int = 100


class LinearPolicy(NamedTuple):
    K: jnp.ndarray
    kff: jnp.ndarray

    def __call__(
        self,
        state: jnp.ndarray,
        time: int,
        trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.K[time] @ (state - trajectory[time]) + self.kff[time]


def _diff_backward_pass(
    quadratic_final_cost: QuadraticFinalCost,
    quadratic_transient_cost: QuadraticTransientCost,
    linear_dynamics: LinearDynamics,
    lmbda: float,
) -> (LinearPolicy, float, jnp.ndarray):

    fCxx, fcx, _ = quadratic_final_cost
    Cxx, Cuu, Cxu, cx, cu, _ = quadratic_transient_cost
    A, B, _ = linear_dynamics

    dV = 0.0

    def _backwards(carry, params):
        Cxx, Cuu, Cxu, cx, cu, A, B = params

        Vxx, vx, dV = carry

        Qxx = symmetrize(Cxx + A.T @ Vxx @ A)
        Quu = symmetrize(Cuu + B.T @ Vxx @ B)
        Qux = (Cxu + A.T @ Vxx @ B).T

        qx = cx + A.T @ vx
        qu = cu + B.T @ vx

        Quu_reg = symmetrize(Quu + lmbda * jnp.eye(Quu.shape[0]))

        def _not_feasible(args):
            Vxx, vx, dV, Quu, Quu_reg, Qux, qx, qu = args
            K, kff = jnp.zeros_like(Qux), jnp.zeros_like(qu)
            return (Vxx, vx, dV), (K, kff, False)

        def _feasible(args):
            Vxx, vx, dV, Quu, Quu_reg, Qux, qx, qu = args

            K = -jsc.linalg.solve(Quu_reg, Qux, sym_pos=True)
            kff = -jsc.linalg.solve(Quu_reg, qu, sym_pos=True)

            Vxx = symmetrize(Qxx + Qux.T @ K)
            vx = qx + Qux.T @ kff
            dV += 0.5 * kff.T @ qu
            return (Vxx, vx, dV), (K, kff, True)

        return cond(
            jnp.all(jnp.linalg.eigvals(Quu_reg) > 0.0),
            _feasible,
            _not_feasible,
            (Vxx, vx, dV, Quu, Quu_reg, Qux, qx, qu),
        )

    (_, _, dV), (K, kff, feasible) = scan(
        f=_backwards,
        init=(fCxx, fcx, dV),
        xs=(Cxx, Cuu, Cxu, cx, cu, A, B),
        reverse=True,
    )

    return LinearPolicy(K, kff), dV, jnp.all(feasible)


_jit_backward_pass = jit(_diff_backward_pass)


def _linearize_quadratize(
    final_cost, transient_cost, goal_state, dynamics, state_space, reference
):

    horizon = reference.horizon
    time = jnp.linspace(0, horizon, horizon + 1)

    quadratic_final_cost = quadratize_diff_final_cost(
        final_cost,
        goal_state,
        reference.final,
    )
    quadratic_transient_cost = quadratize_diff_transient_cost(
        transient_cost, goal_state, reference.transient, time[:-1]
    )
    linear_dynamics = linearize_diff_dynamics(
        dynamics, state_space, reference.transient, time[:-1]
    )

    return quadratic_final_cost, quadratic_transient_cost, linear_dynamics


_jit_linearize_quadratize = jit(
    _linearize_quadratize, static_argnums=(0, 1, 3, 4)
)


def exact_open_loop_rollout(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def step(state, time):
        action = action_space.clip(control[time])
        cost = transient_cost(state, action, time, goal_state)
        next_state = state_space.clip(dynamics(state, action, time))
        return next_state, (next_state, action, cost)

    next_state, action, cost = scan(
        f=step,
        init=init_state,
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    total_cost = jnp.sum(cost) + final_cost(state[-1], goal_state)
    return state, action, total_cost


_jit_exact_open_loop_rollout = jit(
    exact_open_loop_rollout, static_argnums=(0, 1, 3, 5, 7, 8)
)


def exact_feedback_loop_rollout(
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    trajectory: jnp.ndarray,
    control: jnp.ndarray,
    policy: LinearPolicy,
    action_space: Box,
    horizon: int,
) -> jnp.ndarray:
    def step(state, time):
        diff_control = policy(state, time, trajectory)
        action = action_space.clip(control[time] + diff_control)
        next_state = state_space.clip(dynamics(state, action, time))
        return next_state, diff_control

    _, diff_control = scan(
        f=step,
        init=init_state,
        xs=jnp.arange(horizon),
    )

    return diff_control


_jit_exact_feedback_loop_rollout = jit(
    exact_feedback_loop_rollout, static_argnums=(0, 2, 6, 7)
)


@partial(jit, static_argnums=(0, 1, 3, 5, 7, 8, 9))
def exact_mpc_rollout(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    nb_steps: int,
    options,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def mpc_step(carry, args):
        state, control = carry

        _, control, cost = jax_solver(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            state,
            state_space,
            control,
            action_space,
            horizon,
            options,
        )

        action = action_space.clip(control[0])
        next_state = state_space.clip(dynamics(state, action, 0))

        return (next_state, control), (next_state, action, cost)

    _, (state, action, cost) = scan(
        mpc_step, init=(init_state, control), xs=jnp.arange(nb_steps)
    )

    state = jnp.vstack((init_state, state))
    return state, action, cost


def py_solver(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    options: Hyperparameters,
    verbose: bool = False,
) -> (jnp.ndarray, jnp.ndarray, List):
    def objective(control: jnp.ndarray):
        _, _, total_cost = _jit_exact_open_loop_rollout(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            init_state,
            state_space,
            control,
            action_space,
            horizon,
        )
        return total_cost

    def solver(
        quadratic_final_cost: QuadraticFinalCost,
        quadratic_transient_cost: QuadraticTransientCost,
        linear_dynamics: LinearDynamics,
        control: jnp.ndarray,
        trajectory: jnp.ndarray,
        lmbda: float,
    ):
        def oracle(lmbda):
            diff_policy, diff_cost, feasible = _jit_backward_pass(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                lmbda,
            )

            diff_control = _jit_exact_feedback_loop_rollout(
                dynamics,
                init_state,
                state_space,
                trajectory,
                control,
                diff_policy,
                action_space,
                horizon,
            )
            return diff_control, diff_cost, jnp.all(feasible)

        return oracle

    trace = []

    initialization_feasible = False
    trajectory, control, total_cost = _jit_exact_open_loop_rollout(
        final_cost,
        transient_cost,
        goal_state,
        dynamics,
        init_state,
        state_space,
        control,
        action_space,
        horizon,
    )
    trace.append(total_cost)

    if jnp.all(jnp.abs(trajectory) < 1e8):
        initialization_feasible = True

    if initialization_feasible:
        lmbda = options.init_lmbda
        mult_lmbda = options.init_mult_lmbda

        for _ in range(options.max_iter):
            (
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
            ) = _jit_linearize_quadratize(
                final_cost,
                transient_cost,
                goal_state,
                dynamics,
                state_space,
                Trajectory(trajectory, control),
            )

            oracle = solver(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                control,
                trajectory,
                lmbda,
            )

            step_accepted = False
            while not step_accepted:
                backpass_feasible = False
                diff_control, diff_cost = None, None
                while lmbda < options.max_lmbda and not backpass_feasible:
                    _diff_control, _diff_cost, backpass_feasible = oracle(
                        lmbda
                    )

                    if backpass_feasible:
                        diff_control, diff_cost = _diff_control, _diff_cost
                    else:
                        # step not feasible, increase regularization
                        lmbda = jax.lax.min(
                            lmbda * mult_lmbda, options.max_lmbda
                        )
                        mult_lmbda = 2.0 * mult_lmbda

                if not backpass_feasible:
                    if verbose:
                        print("Backpass not feasible")
                    return trajectory, control, trace

                grad_norm = jnp.linalg.norm(diff_control)
                if grad_norm < options.tol_grad and lmbda < 1e-5:
                    if verbose:
                        print("Gradient tolerance reached")
                    return trajectory, control, trace

                next_control = control + diff_control
                next_total_cost = objective(next_control)
                gain_ratio = (next_total_cost - total_cost) / diff_cost

                if gain_ratio > options.min_ratio:
                    step_accepted = True
                    control = next_control

                    # step accepted, decrease regularization
                    lmbda = lmbda * jax.lax.max(
                        1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3
                    )
                    mult_lmbda = options.init_mult_lmbda
                else:
                    # step rejected, increase regularization
                    lmbda = lmbda * mult_lmbda
                    mult_lmbda = 2.0 * mult_lmbda

            trajectory, control, total_cost = _jit_exact_open_loop_rollout(
                final_cost,
                transient_cost,
                goal_state,
                dynamics,
                init_state,
                state_space,
                control,
                action_space,
                horizon,
            )
            trace.append(total_cost)

        return trajectory, control, trace
    else:
        if verbose:
            print("Initial solution diverges")
        return trajectory, control, trace


@partial(jit, static_argnums=(0, 1, 3, 5, 7, 8))
def jax_solver(
    final_cost: Callable,
    transient_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    options: Hyperparameters,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def objective(control: jnp.ndarray):
        _, _, total_cost = exact_open_loop_rollout(
            final_cost,
            transient_cost,
            goal_state,
            dynamics,
            init_state,
            state_space,
            control,
            action_space,
            horizon,
        )
        return total_cost

    def solver(
        quadratic_final_cost: QuadraticFinalCost,
        quadratic_transient_cost: QuadraticTransientCost,
        linear_dynamics: LinearDynamics,
        control: jnp.ndarray,
        trajectory: jnp.ndarray,
        lmbda: float,
    ):
        def oracle(lmbda):
            diff_policy, diff_cost, feasible = _diff_backward_pass(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                lmbda,
            )

            diff_control = exact_feedback_loop_rollout(
                dynamics,
                init_state,
                state_space,
                trajectory,
                control,
                diff_policy,
                action_space,
                horizon,
            )
            return diff_control, diff_cost, feasible

        return oracle

    def _start_main(args):
        (
            trajectory,
            control,
            total_cost,
        ) = args

        def _main_loop_cond(carry):
            i, (_, _, _, lmbda, mult_lmbda) = carry
            return i < options.max_iter

        def _main_loop_body(carry):
            i, (trajectory, control, total_cost, lmbda, mult_lmbda) = carry

            (
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
            ) = _linearize_quadratize(
                final_cost,
                transient_cost,
                goal_state,
                dynamics,
                state_space,
                Trajectory(trajectory, control),
            )

            oracle = solver(
                quadratic_final_cost,
                quadratic_transient_cost,
                linear_dynamics,
                control,
                trajectory,
                lmbda,
            )

            # backward pass feasability
            def _backward_pass_feasability_cond(carry):
                _, _, backpass_feasible, lmbda, mult_lmbda = carry

                return jnp.logical_and(
                    (lmbda < options.max_lmbda),
                    (jnp.logical_not(backpass_feasible)),
                )

            def _backward_pass_feasability_body(carry):
                _, _, _, lmbda, mult_lmbda = carry

                lmbda = jax.lax.min(lmbda * mult_lmbda, options.max_lmbda)
                mult_lmbda = 2.0 * mult_lmbda

                (diff_control, diff_cost, backpass_feasible,) = oracle(
                    lmbda,
                )

                return (
                    diff_control,
                    diff_cost,
                    backpass_feasible,
                    lmbda,
                    mult_lmbda,
                )

            diff_control, diff_cost, backpass_feasible = oracle(lmbda)

            (
                diff_control,
                diff_cost,
                backpass_feasible,
                lmbda,
                mult_lmbda,
            ) = while_loop(
                _backward_pass_feasability_cond,
                _backward_pass_feasability_body,
                init_val=(
                    diff_control,
                    diff_cost,
                    backpass_feasible,
                    lmbda,
                    mult_lmbda,
                ),
            )

            def _backpass_success(args):
                (
                    trajectory,
                    control,
                    total_cost,
                    diff_control,
                    diff_cost,
                    lmbda,
                    mult_lmbda,
                ) = args

                def _gradient_converged(args):
                    (
                        trajectory,
                        control,
                        total_cost,
                        diff_control,
                        lmbda,
                        mult_lmbda,
                    ) = args
                    return trajectory, control, total_cost, lmbda, mult_lmbda

                def _gradient_not_converged(args):
                    (
                        trajectory,
                        control,
                        total_cost,
                        diff_control,
                        lmbda,
                        mult_lmbda,
                    ) = args

                    def _accept(args):
                        (
                            trajectory,
                            control,
                            total_cost,
                            next_control,
                            lmbda,
                            mult_lmbda,
                        ) = args

                        lmbda = lmbda * jax.lax.max(
                            1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3
                        )
                        mult_lmbda = options.init_mult_lmbda

                        (
                            trajectory,
                            control,
                            total_cost,
                        ) = exact_open_loop_rollout(
                            final_cost,
                            transient_cost,
                            goal_state,
                            dynamics,
                            init_state,
                            state_space,
                            next_control,
                            action_space,
                            horizon,
                        )

                        return (
                            trajectory,
                            control,
                            total_cost,
                            lmbda,
                            mult_lmbda,
                        )

                    def _reject(args):
                        (
                            trajectory,
                            control,
                            total_cost,
                            next_control,
                            lmbda,
                            mult_lmbda,
                        ) = args

                        lmbda = lmbda * mult_lmbda
                        mult_lmbda = 2.0 * mult_lmbda

                        return (
                            trajectory,
                            control,
                            total_cost,
                            lmbda,
                            mult_lmbda,
                        )

                    next_control = control + diff_control
                    next_total_cost = objective(next_control)
                    gain_ratio = (next_total_cost - total_cost) / diff_cost

                    return cond(
                        gain_ratio > options.min_ratio,
                        _accept,
                        _reject,
                        (
                            trajectory,
                            control,
                            total_cost,
                            next_control,
                            lmbda,
                            mult_lmbda,
                        ),
                    )

                grad_norm = jnp.linalg.norm(diff_control)

                return cond(
                    jnp.logical_and(
                        (grad_norm < options.tol_grad), (lmbda < 1e-5)
                    ),
                    _gradient_converged,
                    _gradient_not_converged,
                    (
                        trajectory,
                        control,
                        total_cost,
                        diff_control,
                        lmbda,
                        mult_lmbda,
                    ),
                )

            def _backpass_failure(args):
                (
                    trajectory,
                    control,
                    total_cost,
                    diff_control,
                    diff_cost,
                    lmbda,
                    mult_lmbda,
                ) = args
                return trajectory, control, total_cost, lmbda, mult_lmbda

            trajectory, control, total_cost, lmbda, mult_lmbda = cond(
                backpass_feasible,
                _backpass_success,
                _backpass_failure,
                (
                    trajectory,
                    control,
                    total_cost,
                    diff_control,
                    diff_cost,
                    lmbda,
                    mult_lmbda,
                ),
            )

            return i + 1, (trajectory, control, total_cost, lmbda, mult_lmbda)

        lmbda = options.init_lmbda
        mult_lmbda = options.init_mult_lmbda

        _, (trajectory, control, total_cost, _, _) = while_loop(
            _main_loop_cond,
            _main_loop_body,
            init_val=(0, (trajectory, control, total_cost, lmbda, mult_lmbda)),
        )
        return trajectory, control, total_cost

    def _exit_main(args):
        (
            trajectory,
            control,
            total_cost,
        ) = args

        return trajectory, control, total_cost

    trajectory, control, total_cost = exact_open_loop_rollout(
        final_cost,
        transient_cost,
        goal_state,
        dynamics,
        init_state,
        state_space,
        control,
        action_space,
        horizon,
    )

    return cond(
        jnp.all(jnp.abs(trajectory) < 1e8),
        _start_main,
        _exit_main,
        (
            trajectory,
            control,
            total_cost,
        ),
    )
