from typing import NamedTuple, Callable, List, Tuple
from functools import partial

import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsc

import jax
from jax import jit
from jax.lax import scan, cond, while_loop

from tox.objects import (
    QuadraticFinalBeliefCost,
    QuadraticTransientBeliefCost,
    LinearBeliefDynamics,
    BeliefTrajectory,
    Box,
)

from tox.approximations import (
    quadratize_diff_final_belief_cost,
    quadratize_diff_transient_belief_cost,
    linearize_diff_belief_dynamics,
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
        belief: jnp.ndarray,
        time: int,
        trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.K[time] @ (belief - trajectory[time]) + self.kff[time]


def _diff_backward_pass(
    quadratic_final_belief_cost: QuadraticFinalBeliefCost,
    quadratic_transient_belief_cost: QuadraticTransientBeliefCost,
    linear_belief_dynamics: LinearBeliefDynamics,
    lmbda: float,
) -> (LinearPolicy, jnp.ndarray, List):

    fCbb, fcb, _ = quadratic_final_belief_cost
    Cbb, Cuu, Cbu, cb, cu, _ = quadratic_transient_belief_cost
    gb, gu, Wb, Wu, W = linear_belief_dynamics

    dV = 0.0

    def _backwards(carry, params):
        (
            Cbb,
            Cuu,
            Cbu,
            cb,
            cu,
            gb,
            gu,
            Wb,
            Wu,
            W,
        ) = params

        Vbb, vb, dV = carry

        Qbb = symmetrize(Cbb + gb.T @ Vbb @ gb + jnp.einsum("nkh,kd,ndm->hm", Wb, Vbb, Wb))
        Quu = symmetrize(Cuu + gu.T @ Vbb @ gu + jnp.einsum("nkh,kd,ndm->hm", Wu, Vbb, Wu))
        Qub = (Cbu + gb.T @ Vbb @ gu + jnp.einsum("nkh,kd,ndm->hm", Wb, Vbb, Wu)).T

        qb = cb + gb.T @ vb + jnp.einsum("nkh,kd,nd->h", Wb, Vbb, W)
        qu = cu + gu.T @ vb + jnp.einsum("nkh,kd,nd->h", Wu, Vbb, W)

        Quu_reg = symmetrize(Quu + lmbda * jnp.eye(Quu.shape[0]))

        def _not_feasible(args):
            (
                Vbb,
                vb,
                dV,
                Quu,
                Quu_reg,
                Qub,
                qb,
                qu,
            ) = args

            K, kff = jnp.zeros_like(Qub), jnp.zeros_like(qu)
            return (Vbb, vb, dV), (K, kff, False)

        def _feasible(args):
            (
                Vbb,
                vb,
                dV,
                Quu,
                Quu_reg,
                Qub,
                qb,
                qu,
            ) = args

            K = -jsc.linalg.solve(Quu_reg, Qub, sym_pos=True)
            kff = -jsc.linalg.solve(Quu_reg, qu, sym_pos=True)

            Vbb = symmetrize(Qbb + Qub.T @ K)
            vb = qb + Qub.T @ kff
            dV += 0.5 * kff.T @ qu
            return (Vbb, vb, dV), (K, kff, True)

        return cond(
            jnp.all(jnp.linalg.eigvals(Quu_reg) > 0.0),
            _feasible,
            _not_feasible,
            (
                Vbb,
                vb,
                dV,
                Quu,
                Quu_reg,
                Qub,
                qb,
                qu,
            ),
        )

    (_, _, dV), (K, kff, feasible) = scan(
        f=_backwards,
        init=(fCbb, fcb, dV),
        xs=(Cbb, Cuu, Cbu, cb, cu, gb, gu, Wb, Wu, W),
        reverse=True,
    )

    return LinearPolicy(K, kff), dV, jnp.all(feasible)


_jit_backward_pass = jit(_diff_backward_pass)


def _linearize_quadratize(
    final_belief_cost,
    transient_belief_cost,
    goal_state,
    belief_dynamics,
    reference,
):

    horizon = reference.horizon
    time = jnp.linspace(0, horizon, horizon + 1)

    quadratic_final_belief_cost = quadratize_diff_final_belief_cost(
        final_belief_cost,
        goal_state,
        reference.final,
    )
    quadratic_transient_belief_cost = (
        quadratize_diff_transient_belief_cost(
            transient_belief_cost,
            goal_state,
            reference.transient,
            time[:-1],
        )
    )
    linear_belief_dynamics = linearize_diff_belief_dynamics(
        belief_dynamics,
        reference.transient,
        time[:-1],
    )

    return (
        quadratic_final_belief_cost,
        quadratic_transient_belief_cost,
        linear_belief_dynamics,
    )


_jit_linearize_quadratize = jit(
    _linearize_quadratize, static_argnums=(0, 1, 3)
)


def approximate_open_loop_rollout(
    final_belief_cost: Callable,
    transient_belief_cost: Callable,
    goal_state: jnp.ndarray,
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    belief_dynamics_mean, belief_dynamics_cov = belief_dynamics

    def step(belief, time):
        action = action_space.clip(control[time])
        cost = transient_belief_cost(belief, action, time, goal_state)
        next_belief = belief_dynamics_mean(belief, action, time)
        return next_belief, (next_belief, action, cost)

    next_belief, action, cost = scan(
        f=step,
        init=init_belief,
        xs=jnp.arange(horizon),
    )[1]

    belief = jnp.vstack((init_belief, next_belief))
    total_cost = jnp.sum(cost) + final_belief_cost(belief[-1], goal_state)
    return belief, action, total_cost


_jit_approximate_open_loop_rollout = jit(
    approximate_open_loop_rollout, static_argnums=(0, 1, 3, 6, 7)
)


def approximate_feedback_loop_rollout(
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    trajectory: jnp.ndarray,
    control: jnp.ndarray,
    policy: LinearPolicy,
    action_space: Box,
    horizon: int,
) -> jnp.ndarray:

    belief_dynamics_mean, belief_dynamics_cov = belief_dynamics

    def step(belief, time):
        diff_control = policy(belief, time, trajectory)
        action = action_space.clip(control[time] + diff_control)
        next_belief = belief_dynamics_mean(belief, action, time)
        return next_belief, diff_control

    _, diff_control = scan(
        f=step,
        init=init_belief,
        xs=jnp.arange(horizon),
    )

    return diff_control


_jit_approximate_feedback_loop_rollout = jit(
    approximate_feedback_loop_rollout, static_argnums=(0, 5, 6)
)


@partial(jit, static_argnums=(0, 1, 3, 5, 6, 7, 8, 10, 12, 13))
def approximate_closed_loop_rollout(
    final_belief_cost: Callable,
    transient_belief_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    observation: Callable,
    observation_space: Box,
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    bayes_filter: Callable,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    options: Hyperparameters,
    key: jr.PRNGKey,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):

    dx = state_space.shape[0]
    dy = observation_space.shape[0]

    def step(carry, time):
        state, belief, key = carry

        action = action_space.clip(control[time])
        cost = transient_belief_cost(belief, action, time, goal_state)

        key, state_key = jr.split(key, 2)
        delta = jr.multivariate_normal(
            state_key, mean=jnp.zeros((dx,)), cov=jnp.eye(dx)
        )
        next_state = dynamics(state, action, delta, time)

        key, measurement_key = jr.split(key, 2)
        eta = jr.multivariate_normal(
            measurement_key, mean=jnp.zeros((dy,)), cov=jnp.eye(dy)
        )
        next_measurement = observation(next_state, eta, time)

        next_belief = bayes_filter(
            belief,
            action,
            next_measurement,
            time,
        )

        return (next_state, next_belief, key), (
            next_state,
            next_belief,
            action,
            cost,
        )

    next_state, next_belief, action, cost = scan(
        f=step,
        init=(init_state, init_belief, key),
        xs=jnp.arange(horizon),
    )[1]

    state = jnp.vstack((init_state, next_state))
    belief = jnp.vstack((init_belief, next_belief))

    total_cost = jnp.sum(cost) + final_belief_cost(belief[-1], goal_state)
    return state, belief, action, total_cost


@partial(jit, static_argnums=(0, 1, 3, 5, 6, 7, 8, 10, 12, 13, 14))
def approximate_mpc_rollout(
    final_belief_cost: Callable,
    transient_belief_cost: Callable,
    goal_state: jnp.ndarray,
    dynamics: Callable,
    init_state: jnp.ndarray,
    state_space: Box,
    observation: Callable,
    observation_space: Box,
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    bayes_filter: Callable,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    nb_steps: int,
    options: Hyperparameters,
    key: jr.PRNGKey,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):

    dx = state_space.shape[0]
    dy = observation_space.shape[0]

    def mpc_step(carry, args):
        state, belief, control, key = carry

        _, control, cost = jax_solver(
            final_belief_cost,
            transient_belief_cost,
            goal_state,
            belief_dynamics,
            belief,
            control,
            action_space,
            horizon,
            options,
        )

        action = action_space.clip(control[0])

        key, state_key = jr.split(key, 2)
        delta = jr.multivariate_normal(
            state_key, mean=jnp.zeros((dx,)), cov=jnp.eye(dx)
        )
        next_state = dynamics(state, action, delta, 0)

        key, measurement_key = jr.split(key, 2)
        eta = jr.multivariate_normal(
            measurement_key, mean=jnp.zeros((dy,)), cov=jnp.eye(dy)
        )
        next_measurement = observation(next_state, eta, 0)

        next_belief = bayes_filter(
            belief,
            action,
            next_measurement,
            0,
        )
        return (next_state, next_belief, control, key), (
            next_state,
            next_belief,
            action,
            cost,
        )

    state, belief, action, cost = scan(
        mpc_step,
        init=(init_state, init_belief, control, key),
        xs=jnp.arange(nb_steps),
    )[1]

    state = jnp.vstack((init_state, state))
    belief = jnp.vstack((init_belief, belief))

    return state, belief, action, cost


def py_solver(
    final_belief_cost: Callable,
    transient_belief_cost: Callable,
    goal_state: jnp.ndarray,
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    options: Hyperparameters,
    verbose: bool = False,
) -> (jnp.ndarray, jnp.ndarray, List):

    def objective(control: jnp.ndarray):
        _, _, total_cost = _jit_approximate_open_loop_rollout(
            final_belief_cost,
            transient_belief_cost,
            goal_state,
            belief_dynamics,
            init_belief,
            control,
            action_space,
            horizon,
        )
        return total_cost

    def solver(
        quadratic_final_belief_cost: QuadraticFinalBeliefCost,
        quadratic_transient_belief_cost: QuadraticTransientBeliefCost,
        linear_belief_dynamics: LinearBeliefDynamics,
        control: jnp.ndarray,
        trajectory: jnp.ndarray,
        lmbda: float,
    ):
        def oracle(lmbda):
            diff_policy, diff_cost, feasible = _diff_backward_pass(
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
                lmbda,
            )

            diff_control = _jit_approximate_feedback_loop_rollout(
                belief_dynamics,
                init_belief,
                trajectory,
                control,
                diff_policy,
                action_space,
                horizon,
            )
            return diff_control, diff_cost, feasible

        return oracle

    trace = []

    initialization_feasible = False
    trajectory, control, total_cost = _jit_approximate_open_loop_rollout(
        final_belief_cost,
        transient_belief_cost,
        goal_state,
        belief_dynamics,
        init_belief,
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
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
            ) = _jit_linearize_quadratize(
                final_belief_cost,
                transient_belief_cost,
                goal_state,
                belief_dynamics,
                BeliefTrajectory(trajectory, control),
            )

            oracle = solver(
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
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

            (
                trajectory,
                control,
                total_cost,
            ) = _jit_approximate_open_loop_rollout(
                final_belief_cost,
                transient_belief_cost,
                goal_state,
                belief_dynamics,
                init_belief,
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


@partial(jit, static_argnums=(0, 1, 3, 6, 7))
def jax_solver(
    final_belief_cost: Callable,
    transient_belief_cost: Callable,
    goal_state: jnp.ndarray,
    belief_dynamics: Tuple,
    init_belief: jnp.ndarray,
    control: jnp.ndarray,
    action_space: Box,
    horizon: int,
    options: Hyperparameters,
) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
    def objective(control: jnp.ndarray):
        _, _, total_cost = approximate_open_loop_rollout(
            final_belief_cost,
            transient_belief_cost,
            goal_state,
            belief_dynamics,
            init_belief,
            control,
            action_space,
            horizon,
        )
        return total_cost

    def solver(
        quadratic_final_belief_cost: QuadraticFinalBeliefCost,
        quadratic_transient_belief_cost: QuadraticTransientBeliefCost,
        linear_belief_dynamics: LinearBeliefDynamics,
        control: jnp.ndarray,
        trajectory: jnp.ndarray,
        lmbda: float,
    ):
        def oracle(lmbda):
            diff_policy, diff_cost, feasible = _diff_backward_pass(
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
                lmbda,
            )

            diff_control = approximate_feedback_loop_rollout(
                belief_dynamics,
                init_belief,
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
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
            ) = _linearize_quadratize(
                final_belief_cost,
                transient_belief_cost,
                goal_state,
                belief_dynamics,
                BeliefTrajectory(trajectory, control),
            )

            oracle = solver(
                quadratic_final_belief_cost,
                quadratic_transient_belief_cost,
                linear_belief_dynamics,
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
                        ) = approximate_open_loop_rollout(
                            final_belief_cost,
                            transient_belief_cost,
                            goal_state,
                            belief_dynamics,
                            init_belief,
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

    trajectory, control, total_cost = approximate_open_loop_rollout(
        final_belief_cost,
        transient_belief_cost,
        goal_state,
        belief_dynamics,
        init_belief,
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
